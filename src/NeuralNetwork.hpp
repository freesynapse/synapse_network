
#pragma once

#include "Utils.hpp"
#include "ActivationFunctions.hpp"
#include "LossFunctions.hpp"

#include "Timer.hpp"

#include <vector>
#include <deque>
#include <optional>


class NetworkDataManager;   // forward declaration for .train()
class NetworkDebugger;
class Optimizer;

class NeuralNetwork
{
public:
    friend class NetworkDataManager;
    friend class NetworkDebugger;
    friend class Optimizer;

public:
    
    // ctors / dtor
    //

    NeuralNetwork(uint32_t _input_dim);
    NeuralNetwork(const std::vector<uint32_t>& _layer_dims);
    NeuralNetwork(const std::vector<uint32_t>& _layer_dims,
                  const std::vector<ActivationFunction*>& _activation_functions);
    ~NeuralNetwork();


    // API functions
    //

    /*
     * Adds a layer to the network, with size _dim and activation
     * function (as pointer). The ownership of the ActivationFunction
     * object is transferred to the network.
     */
    void addLayer(uint32_t _dim, ActivationFunction* _activation_function)
    {
        if (m_optimizer != nullptr)
        {
            std::cout << __PRETTY_FUNCTION__ << ": Optimizer set before network layers set.\n";
            return;
        }

        m_layerDims.push_back(_dim);
        m_layerCount++;
        m_activationFuncs.push_back(_activation_function);
    }


    /*
     * Adds an activation function to specified layer.
     */
    void addActivationFunction(size_t _layer, ActivationFunction* _activation_function)
    {
        assert(_layer < m_activationFuncs.size());

        if (m_optimizer != nullptr)
        {
            std::cout << __PRETTY_FUNCTION__ << ": Optimizer set before network layers set.\n";
            return;
        }

        m_activationFuncs[_layer] = _activation_function;
    }

    /*
     * Adds an activation function to the last layer without an 
     * activation function (push_back()).
     */
    void addActivationFunction(ActivationFunction* _activation_function)
    {
        if (m_optimizer != nullptr)
        {
            std::cout << __PRETTY_FUNCTION__ << ": Optimizer set before network layers set.\n";
            return;
        }

        m_activationFuncs.push_back(_activation_function);
    }

    /*
     * Adds activation function objects for all layers, e.g.
     *  net.addActivationFunctions({ new ReLU, new ReLU, new Sigmoid });
     */
    void addActivationFunctions(const std::vector<ActivationFunction*>& _activation_functions)
    {
        if (m_optimizer != nullptr)
        {
            std::cout << __PRETTY_FUNCTION__ << ": Optimizer set before network layers set.\n";
            return;
        }

        m_activationFuncs = _activation_functions;
    }

    /*
     * Sets the derived optimizer for training. This class needs to
     * inherit the Optimizer class, specified in Optimizers.hpp.
     */
    void setOptimizer(Optimizer* _optimizer);


    /*
     * Set the loss function of the network. The passed object needs
     * to inherit the LossFunction class, specified in LossFunctions.hpp.
     */
    void setLossFunction(LossFunction* _loss_function);

    /*
     * Initializes weight and bias matrices based on the layers
     * dimensions. the W matrices are initialized normalized
     * normally distributed random numbers ~N(0, 1) and biases
     * are intialized to 1.
     */
    void initializeParameters();

    /*
     * Trains the network for specified number of epochs. Uses a ptr
     * to a network data manager class. Takes ownership of this ptr.
     * Requires an Optimizer ( setOptimizer() ).
     */
    size_t train(NetworkDataManager* _data_manager_ptr,
                 size_t _epochs=25000,
                 bool _verbose=false);
    /*
     * Trains the network for specified number of epochs. Uses a
     * reference to a network data manager class. 
     * Requires an Optimizer ( setOptimizer() ).
     */
    size_t train(const NetworkDataManager& _data_manager_ref,
                 size_t _epochs=25000,
                 bool _verbose=false);

    /*
     * Trains the network for specified number of epochs.
     * Requires an Optimizer ( setOptimizer() ).
     */
    size_t train(const std::vector<Sample>& _data,
                 size_t _epochs=2500,
                 size_t _batch_size=0,
                 bool _shuffle=true,
                 bool _verbose=false);

    /*
     * Run a forward pass with the input _X, returns the output
     * of the network.
     */
    Matrix predict(const Matrix& _X);
    
    /*
     * Evaluates chosen cost function for an example _Y and the
     * result of the network output, stored in m_A.back().
     */
    double computeCost(Matrix& _Y);//, const std::string& _cost_function="binary_cross_entropy");

    /*
     * Resets the network, in case we want to re-train the same network
     * object. Deletes pointers to Optimizer, LossFunction and ActivationFunction:s.
     */
    void reset();


public:

    /*
     * Trains the network using the NetworkDataManager for training data
     * and set hyperparameters. Called by API function train().
     */
    size_t trainNetwork(size_t _epochs, bool _verbose);
    
    /*
     * Performs a forward pass through the network, using input _X.
     * The input is of shape (input size, number of examples). The 
     * result of the forward pass is stored in m_A.back() and can 
     * be accessed through getAL().
     *
     */
    void forwardPass(const Matrix& _X/*, parameters) --> AL, caches */);
    
    /*
     * Computes linear function followed by activation of a layer, 
     * i.e. Z^(l-1) and A^(l) using acivation_function( Z^(l-1) ).
     */
    inline void linearActivationForward(size_t _layer);

    /*
     * Computes the derivatives of error w.r.t. the weights of the 
     * network, propagating errors backwards through the net. 
     * Gradients are stored in m_dW[], m_db[].
     */
    void backwardPass(Matrix& _Y);

    /*
     * Computes the derivatives of error w.r.t. the weights 
     * of the current layer l. Gradients are stored in m_dW[l], 
     * m_db[l].
     */
    inline void linearActivationBackward(const Matrix& _dA, size_t _layer);

    /*
     * Perturbation of the weights and biases in case of lack of
     * training error improvement.
     */
    inline void perturbWeights();
    

public:
    // Accessors
    //
    /* Output of layer L (i.e. network output). */
    const Matrix& getAL() const { return m_A.back(); }
    /* Last training error. */
    const double getLoss() const { return m_loss; }
    /* History of errors, with length equal to the number of epochs. */
    const std::vector<double>& getLossHistory() const { return m_lossHistory; }
    /* Layer count */
    const size_t getLayerCount() const { return m_layerCount; }
    /* Set minimum training error (default 0.001f). */
    void setEpsilon(double _eps) { m_epsilon = _eps; }
    /* Training minimum error. */
    const double getEpsilon() const { return m_epsilon; }
    /* Set the perturbation scaling factor (default = 1.0) */
    const void setPerturbationFactor(const double _f) { m_perturbationFactor = _f; }
    /* Set L2 regularization flag */
    //const void useRegularization(const bool _r=true) { m_useRegularization = _r; }
    /* Set L2 regularization lambda (default = 0.1) */
    //const void setRegularizationLambda(const double _l) { m_lambda = _l; }


public:
    std::vector<uint32_t> m_layerDims;
    size_t m_layerCount;
    std::vector<ActivationFunction*> m_activationFuncs;

    Optimizer* m_optimizer = nullptr;   // chosen optimizer, required
    
    LossFunction* m_lossFunction = nullptr; // chosen loss function, required

    // weights and biases matrices
    std::vector<Matrix> m_W;            // weights of layer l
    std::vector<ColVector> m_b;         // biases of layer l
    std::vector<Matrix> m_A;            // inputs of layer l
    std::vector<Matrix> m_Z;            // outputs of layer l
    
    std::vector<Matrix> m_dW;           // dW, w.r.t. back-propagated error
    std::vector<ColVector> m_db;        // db, w.r.t. back-propagated error
    std::vector<Matrix> m_dA;           // dA for previous layer l-1

    // training
    NetworkDataManager* m_dataManager = nullptr;    // training manager
    double m_loss;                      // training loss (diff between output and expected)
    std::vector<double> m_lossHistory;  // training loss history over all epochs
    double m_epsilon = 0.001;           // stop training if loss is below epsilon
    size_t m_completedEpochs;           // number of training epochs completed before (early?)
                                        // stopping

    //bool m_useRegularization = true;    // flag for using L2-regularization
    //double m_lambda = 0.1;              // regularization factor

    std::deque<double> m_epochLosses;   // storage of the n last losses, used for weight
                                        // perturbations
    size_t m_epochLossSize = 25;        // size of immediate training history recorded for
                                        // perturbation decision
    double m_perturbationFactor = 1.0;  // multiplier of Eigen::Matrix::Random used for 
                                        // weights and biases perturbations
    size_t m_perturbationCount = 0;

};

