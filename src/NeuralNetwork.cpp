
#include "NeuralNetwork.hpp"
#include "NetworkDataManager.hpp"
#include "Optimizers.hpp"
#include "Timer.hpp"
#include "NetworkDebugger.hpp"

#include <random>
#include <chrono>
#include <fstream>


NeuralNetwork::NeuralNetwork(uint32_t _input_dim)
{
    m_layerDims.push_back(_input_dim);
    m_layerCount = 1;
}
//---------------------------------------------------------------------------------------
NeuralNetwork::NeuralNetwork(const std::vector<uint32_t>& _layer_dims)
{
    m_layerDims = _layer_dims;
    m_layerCount = _layer_dims.size();
    m_activationFuncs.resize(m_layerCount - 1);
}
//---------------------------------------------------------------------------------------
NeuralNetwork::NeuralNetwork(const std::vector<uint32_t>& _layer_dims,
                             const std::vector<ActivationFunction*>& _activation_functions)
{
    m_layerDims = _layer_dims;
    m_layerCount = _layer_dims.size();
    m_activationFuncs = _activation_functions;
}
//---------------------------------------------------------------------------------------
NeuralNetwork::~NeuralNetwork()
{
    if (m_dataManager != nullptr)
    {
        delete m_dataManager;
        m_dataManager = nullptr;
    }

    for (ActivationFunction* fnc : m_activationFuncs)
    {
        if (fnc != nullptr)
        {
            delete fnc;
            fnc = nullptr;
        }
    }

    if (m_optimizer != nullptr)
    {
        delete m_optimizer;
        m_optimizer = nullptr;
    }

    if (m_lossFunction != nullptr)
    {
        delete m_lossFunction;
        m_lossFunction = nullptr;
    }
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::setOptimizer(Optimizer* _optimizer)
{
    m_optimizer = _optimizer;
    m_optimizer->initialize(this);
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::setLossFunction(LossFunction* _loss_function)
{
    m_lossFunction = _loss_function;
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::initializeParameters()
{
    if (m_optimizer != nullptr)
    {
        std::cout << __PRETTY_FUNCTION__ << ": Optimizer set before network initialization.\n";
        return;
    }

    // clear weights and biases (in case this is a retraining of the same network object)
    m_W.resize(m_layerCount-1);
    m_b.resize(m_layerCount-1);

    for (size_t i = 1; i < m_layerDims.size(); i++)
    {
        uint32_t rows = m_layerDims[i];
        uint32_t cols = m_layerDims[i-1];
        
        uint32_t seed = std::chrono::system_clock::now().time_since_epoch().count();
        std::default_random_engine rng(seed);
        std::normal_distribution<double> N(0.0f, 1.0f);
        
        Matrix W(rows, cols);
        //W = Matrix::Random(rows, cols);
        
        for (size_t j = 0; j < W.rows(); j++)
            for (size_t k = 0; k < W.cols(); k++)
                W(j, k) = N(rng) / sqrtf64((double)m_layerDims[i-1]);
        m_W[i-1] = W;
        m_b[i-1] = ColVector::Constant(rows, 0);

        //std::cout << "W[" << i << "]:\n" << m_W.back().format(fmt) << "\n";
        //std::cout << "b[" << i << "]:\n" << m_b.back().format(fmt) << "\n";
    }

    m_A.resize(m_layerCount);
    m_Z.resize(m_layerCount-1);

    m_dA.resize(m_layerCount-1);
    m_dW.resize(m_layerCount-1);
    m_db.resize(m_layerCount-1);
    //for (size_t i = 0; i < m_layerCount; i++)
    //    m_A.push_back(Matrix());

    m_epochLosses.resize(m_epochLossSize);

}
//---------------------------------------------------------------------------------------
size_t NeuralNetwork::train(NetworkDataManager* _data_manager_ptr,
                            size_t _epochs,
                            bool _verbose)
{
    /*
     * All training input is of shape (m, 1), where m is the number of inputs in the network,
     * i.e. m_layerDims[0]. A batch is then constructed using a m x n matrix, where n is the
     * batch size.
     * 
     * For labels, the matrix is of dimensions Y_m by n, where Y_m is the number of network
     * output nodes, i.e. m_layerDims.back().
     * 
     * The NetworkDataManger instance is responsible for preparing and handing shuffled 
     * batches to the network.
     */
    m_dataManager = _data_manager_ptr;
    m_dataManager->resetBatches();

    // start the training
    return trainNetwork(_epochs, _verbose);
}
//---------------------------------------------------------------------------------------
size_t NeuralNetwork::train(const NetworkDataManager& _data_manager_ref,
                            size_t _epochs,
                            bool _verbose)
{
    // copy reference
    m_dataManager = new NetworkDataManager(_data_manager_ref);
    m_dataManager->resetBatches();

    // start the training
    return trainNetwork(_epochs, _verbose);
}
//---------------------------------------------------------------------------------------
size_t NeuralNetwork::train(const std::vector<Sample>& _data,
                            size_t _epochs,
                            size_t _batch_size,
                            bool _shuffle,
                            bool _verbose)
{
    m_dataManager = new NetworkDataManager(*this, _batch_size, _shuffle);

    for (size_t i = 0; i < _data.size(); i++)
        m_dataManager->addSample(_data[i]);
    m_dataManager->resetBatches();

    // start the training
    return trainNetwork(_epochs, _verbose);

}
//---------------------------------------------------------------------------------------
Matrix NeuralNetwork::predict(const Matrix& _X)
{
    if (_X.rows() != m_layerDims[0])
    {
        std::cout << "error : input dimensions of network (" << m_layerDims[0];
        std::cout << ") and data dimensions (" << _X.rows() << ") don't agree." << std::endl;
    }
    
    Matrix output;
    
    forwardPass(_X);

    output = m_A.back();
    return output;

}
//---------------------------------------------------------------------------------------
double NeuralNetwork::computeCost(Matrix& _Y)//, const std::string& _cost_function/*="binary_cross_entropy*/)
                                /*CostFunction* _cost_function)*/
{
    //m_loss = 0.0;
    //double div = 1.0 / (double)_Y.cols();
    // convert expected and observed output to RowVectors
    //Matrix M_AL = m_A.back();

    double loss = m_lossFunction->compute(m_A.back(), _Y);

    //if (m_useRegularization)
    //{
    //    double div = m_lambda / (2.0 * (double)_Y.cols());
    //    double L2_loss = 0.0;
    //    // add L2 norm of weights
    //    for (size_t l = 0; l < m_W.size(); l++)
    //        L2_loss += m_W[l].array().square().sum();
    //
    //    L2_loss *= m_lambda / (2.0 * (double)_Y.cols());
    //    
    //    loss += L2_loss;
    //}

    return loss;

    //RowVector AL(Eigen::Map<RowVector>(M_AL.data(), M_AL.cols()));
    //RowVector Y (Eigen::Map<RowVector>(_Y.data(), _Y.cols()));
    //#ifdef DEBUG_NEURAL_NET
    //    std::cout << "div = " << div << "\n";
    //    std::cout << "Y shape  = " << shape<RowVector>(_Y) << "\n";
    //    std::cout << "AL shape = " << shape<RowVector>(AL) << "\n";
    //#endif

    /*if (_cost_function == "binary_cross_entropy")
    {
        //cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        RowVector AL_log = AL.array().log();
        RowVector AL_log_m_1 = (1.0 - AL.array()).log();
        RowVector Y_m_1 = 1.0 - Y.array();
        #ifdef DEBUG_NEURAL_NET
            std::cout << "AL.array().log()          = " << AL_log << "\n";
            std::cout << "(1.0 - AL.array()).log() = " << AL_log_m_1 << "\n";
            std::cout << "1.0 - Y.array()          = " << Y_m_1 << "\n";
            std::cout << "-dot(Y, log(AL).T)         = " << (-_Y.dot(AL_log.transpose())) << "\n";
            std::cout << "-dot(1-Y, log(1-AL).T)     = " << -Y_m_1.dot(AL_log_m_1.transpose()) << "\n";
        #endif
        m_loss = div * (-Y.dot(AL_log.transpose()) - Y_m_1.dot(AL_log_m_1.transpose()));
        std::cout << "m_loss = " << m_loss << ", loss = " << e << "\n";
    }
    else
        std::cout << "unknown cost function '" << _cost_function << "'.\n";
     */
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::reset()
{
    if (m_optimizer != nullptr)
    {
        delete m_optimizer;
        m_optimizer = nullptr;
    }

    if (m_lossFunction != nullptr)
    {
        delete m_lossFunction;
        m_lossFunction = nullptr;
    }

    for (ActivationFunction* fnc : m_activationFuncs)
    {
        delete fnc;
        fnc = nullptr;
    }
    
    m_activationFuncs.resize(0);

}
//---------------------------------------------------------------------------------------
size_t NeuralNetwork::trainNetwork(size_t _epochs, bool _verbose)
{
    Timer timer("", false);

    if (m_optimizer == nullptr)
    {
        std::cout << __PRETTY_FUNCTION__ << ": no Optimizer specified, defaulting to GradientDescent(0.05).\n";
        m_optimizer = new GradientDescent(0.05f);
    }

    if (m_lossFunction == nullptr)
    {
        std::cout << __PRETTY_FUNCTION__ << ": no LossFunction specified, defaulting to CrossEntropy.\n";
        m_lossFunction = new BinaryCrossEntropy;
    }

    double batch_count_inv = 1.0 / (double)m_dataManager->getBatchCount();
    static NetworkDebugger debug;

    size_t epoch;
    for (epoch = 0; epoch < _epochs; epoch++)
    {
        double loss = 0.0;
        size_t n_data = 1;
        Matrix X, Y;
    
        // run through all batches
        while (n_data != 0)
        {
            n_data = m_dataManager->getNextBatch(&X, &Y);
            
            // run the network forward
            forwardPass(X);

            // compute the erro between observed and expected
            loss += computeCost(Y);
            
            // compute the deltas
            backwardPass(Y);
            
            // update parameters using set Optimizer
            m_optimizer->updateParameters(this, epoch);

        }

        // account for number of batches when calculating epoch total loss
        m_loss = loss * batch_count_inv;
        m_lossHistory.push_back(m_loss);
        
        // store losses and check for lack of improvement
        m_epochLosses.pop_front();
        m_epochLosses.push_back(m_loss);

        if ((epoch && epoch % m_epochLossSize == 0) && 
            (abs(m_epochLosses[0] - m_epochLosses[m_epochLossSize]) <= 0.01))
        {
            perturbWeights();
            m_perturbationCount++;
        }

        // abort early if error is below epsilon
        if (m_loss < m_epsilon) // TODO : || prediction accuracy > threshold
            break;

        // reset batches for next pass
        m_dataManager->resetBatches();
    }

    // number of completed epochs before stopping (possibly early breakout)
    m_completedEpochs = epoch;

    float trainingTime = timer.getDeltaTimeMs();

    //for (size_t i = 0; i < m_lossHistory.size(); i++)
    //{
    //    if (i % 250 == 0 || i == m_lossHistory.size() -1)
    //        printf("epoch %10zu : error = %f\n", i, m_lossHistory[i]);
    //}

    if (_verbose)
    {
        std::cout << __PRETTY_FUNCTION__ << ": training network for " << _epochs << " epochs in ";
        std::cout << trainingTime << " ms.\n";
    }

    if (m_loss < m_epsilon)
    {
        if (_verbose)
        {
            std::cout << "[finished training after " << epoch << " epochs: error=";
            std::cout << m_loss << ", epsilon=" << m_epsilon << "]\n";
        }
    }

    if (_verbose)
    {
        std::cout << "perturbed weights in " << m_perturbationCount << " epochs (";
        std::cout << 100.0f*(float)m_perturbationCount/(float)epoch << "%).\n";
    }

    return m_completedEpochs;
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::forwardPass(const Matrix& _X)
{
    // set the input of the network to _X
    m_A[0] = _X;
    
    //std::cout << "m_A[0] shape = " << shape<Matrix>(m_A[0]) << "\n";
    
    // compute Z and A of the current layer from the input from the previous layer
    for (int i = 1; i < m_layerCount; i++)
        linearActivationForward(i);

    //std::cout << "AL shape = " << shape<Matrix>(m_A.back()) << "\n";

}
//---------------------------------------------------------------------------------------
void NeuralNetwork::linearActivationForward(size_t _layer)
{
    assert(_layer > 0);

    //std::cout << "activating layer " << _layer << " using W[" << _layer-1 << "], b[" << _layer-1 << "] and A[" << _layer-1 << "]\n";
    //std::cout << "W[" << _layer-1 << "] shape = " << shape<Matrix>(m_W[_layer-1]) << "\n";
    //std::cout << "A[" << _layer-1 << "] shape = " << shape<Matrix>(m_A[_layer-1]) << "\n";
    //std::cout << "b[" << _layer-1 << "] shape = " << shape<ColVector>(m_b[_layer-1]) << "\n\n";

    //size_t l = _layer;
    //Matrix m = m_W[l-1] * m_A[l-1];
    //std::cout << "W[" << l-1 << "] * A[" << l-1 << "] shape = " << shape<Matrix>(m) << "\n";

    m_Z[_layer-1] = (m_W[_layer-1] * m_A[_layer-1]).colwise() + m_b[_layer-1];

    //std::cout << "Z[" << _layer-1 << "] =\n" << m_Z[_layer-1] << "\n";

    // use this layers activation function to evaluate the inpute of teh previous layer
    m_A[_layer] = m_activationFuncs[_layer-1]->eval(m_Z[_layer-1]);

    //if (_activation_func == "sigmoid")
    //{
    //    // m_Z[_layer] now contains the product of m_A[prevLayer] and this layers' weights
    //    // now compute activated m_Z[_layer].
    //    m_A[_layer] = sigmoid(m_Z[_layer-1]);
    //    #ifdef DEBUG_NEURAL_NET
    //        std::cout << "A[" << _layer << "] =\n" << m_A[_layer] << "\n";
    //    #endif
    //}
    //else
    //    std::cout << "unknown activation function\n";
        
}
//---------------------------------------------------------------------------------------
void NeuralNetwork::backwardPass(Matrix& _Y)
{
    //std::cout << __PRETTY_FUNCTION__ << ":\n";

    /*    
    Matrix AL = m_A.back();
    assert(AL.rows() == _Y.rows() && AL.cols() == _Y.cols());

    Matrix R0 = elementWiseDivide(_Y, AL);
    Matrix R1 = elementWiseDivide(1.0f - _Y.array(), 1.0f - AL.array());

    Matrix dAL = -(R0.array() - R1.array());
    */

    Matrix dAL = m_lossFunction->compute_d(m_A.back(), _Y);
    linearActivationBackward(dAL, m_layerCount-1);

    for (size_t l = m_layerCount-2; l--; )
    {
        Matrix dA = m_dA[l+1];
        //std::cout << "got dA_prev from cache : shape = " << shape<Matrix>(dA) << "\n" << dA << "\n";
        linearActivationBackward(dA, l+1);
    }

}
//---------------------------------------------------------------------------------------
void NeuralNetwork::linearActivationBackward(const Matrix& _dA, size_t _layer)
{
    //std::cout << __PRETTY_FUNCTION__ << ":\n";
    
    size_t l = _layer - 1;

    // compute derivative of activation w.r.t. the previous layers activation
    Matrix dZ = m_activationFuncs[l]->eval_d(_dA, m_Z[l]);

    //Matrix dZ = sigmoid_d(_dA, m_Z[l]);

    Matrix A_prev = m_A[l];

    double one_over_m = 1.0 / (double)A_prev.cols();
    //double lambda_over_m = m_lambda * one_over_m;

    m_dW[l] = one_over_m * (dZ * A_prev.transpose()).array();
    m_db[l] = one_over_m * dZ.rowwise().sum();

    //if (m_useRegularization)
    //    m_dW[l] = m_dW[l].array() + lambda_over_m * m_W[l].array();

    m_dA[l] = m_W[l].transpose() * dZ;

    //std::cout << "A_prev : shape = " << shape<Matrix>(A_prev) << "\n" << A_prev << "\n";
    //std::cout << "dZ     : shape = " << shape<Matrix>(dZ) << "\n" << dZ <<"\n";
    //std::cout << "dW     : shape = " << shape<Matrix>(m_dW[_layer]) << "\n" << m_dW[_layer] << "\n";
    //std::cout << "db     : shape = " << shape<ColVector>(m_db[_layer]) << "\n" << m_db[_layer] << "\n";
    //std::cout << "dA_prev: shape = " << shape<Matrix>(m_dA[_layer]) << "\n" << m_dA[_layer] << "\n";
    //std::cout << "\n";

}
//---------------------------------------------------------------------------------------
void NeuralNetwork::perturbWeights()
{
    for (size_t l = 0; l < m_layerCount-1; l++)
    {
        Matrix W(m_W[l].rows(), m_W[l].cols());
        W = Matrix::Random(m_W[l].rows(), m_W[l].cols()) * m_perturbationFactor;
        m_W[l] = m_W[l].array() + W.array();

        ColVector b(m_b[l].rows());
        b = ColVector::Random(m_b[l].rows()) * m_perturbationFactor;
        m_b[l] = m_b[l] + b;
    }
}

