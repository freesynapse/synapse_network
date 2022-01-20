
#include "NeuralNetwork.hpp"
#include "NetworkDataManager.hpp"
#include "NetworkDebugger.hpp"
#include "Optimizers.hpp"

#include "Eigen/Eigen"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>


NetworkDebugger debug;

//---------------------------------------------------------------------------------------
void test_linear_forward()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    // OK! same as W.dot(A)+b in numpy
    Matrix W(3, 3), A(3, 1), b(3, 1);
    W << 1, 1, 1, 2, 2, 2, 3, 3, 3;
    A << 1, 2, 3;
    b << 0.1f, 0.2f, 0.3f;

    Matrix Z = W * A + b;
    std::cout << Z << '\n';
}
//---------------------------------------------------------------------------------------
void test_sigmoid_d()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    Matrix dA(3, 1), Z(3, 1);
    dA << 1, 2, 3;
    Z << 4, 5, 6;

    std::cout << "sigmoid()\n";
    Matrix A = sigmoid(Z);
    std::cout << A << '\n';
    std::cout << "Z shape: " << shape<Matrix>(Z) << '\n';

    std::cout << "sigmoid_d()\n";
    Matrix dZ = sigmoid_d(dA, Z);
    std::cout << dZ << '\n';
    std::cout << "dZe shape: " << shape<Matrix>(dZ) << '\n';
}
//---------------------------------------------------------------------------------------
void test_element_wise_log()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    Matrix A(2, 2);
    A << 0.2f, 0.4f, 0.6f, 0.8f;
    std::cout << "A before .array().log():\n" << A << "\n" << shape<Matrix>(A) << "\n";
    Matrix B = A.array().log();
    std::cout << "A after .array().log():\n" << B << "\n" << shape<Matrix>(B) << "\n";
}
//---------------------------------------------------------------------------------------
void test_row_vector_dot()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    Eigen::Matrix<double, 1, 3> a, b;
    a << 1, 2, 3;
    b << 1, 2, 3;
    std::cout << "a.shape = " << shape<Eigen::Matrix<double, 1, 3>>(a) << "\n" <<
                 "b.shape = " << shape<Eigen::Matrix<double, 1, 3>>(b) << "\n";
    std::cout << "a.dot(b.T) = " << a.dot(b.transpose()) << "\n";

    RowVector A = Matrix(1, 3);
    RowVector B = Matrix(1, 3);
    A << 1, 2, 3;
    B << 1, 2, 3;
    std::cout << "A.shape = " << shape<RowVector>(A) << "\n" <<
                 "B.shape = " << shape<RowVector>(B) << "\n";
    std::cout << "A.dot(B.T) = " << A.dot(B.transpose()) << "\n";
}
//---------------------------------------------------------------------------------------
void test_matrix_dot()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    // input X : 2 inputs each, 4 examples
    Matrix X(2, 4), W(2, 2);
    ColVector b(2);
    X << 0.0f, 1.0f, 
         1.0f, 0.0f,
         0.2f, 0.8f,
         0.8f, 0.2f;
    W << 1.1f, 0.9f, 0.8f, 0.6f;
    b << 0.1f, 0.2f;
    std::cout << "X shape = " << shape<Matrix>(X) << "\n" << X << "\n";
    // In Eigen, appearantly the matrix dot product is calculated 
    // as W * X, not W.dot(X) as in numpy.

    // In addition, broadcasting is performed using .colwise()
    //std::cout << "W.dot(X) =\n" << W * X << "\n";
    Matrix M = W * X;
    std::cout << "W * X shape = " << shape<Matrix>(M) << ":\n" << M << "\n";
    std::cout << "b shape = " << shape<ColVector>(b) << ":\n" << b << "\n";

    Matrix M2 = (W * X).colwise() + b;
    std::cout << "(W * X).colwise() shape = " << shape<Matrix>(M2) << ":\n" << M2 << "\n";
}
//---------------------------------------------------------------------------------------
void test_scalar_minus_matrix()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    RowVector rv(3);
    Matrix A(1, 3);
    A << 0.1f, 0.2f, 0.3f;
    rv = A;
    std::cout << "rv.log() (has to be called as rv.array().log()):\n" << A.array().log() << "\n\n";
    std::cout << "1.0f - rv (has to be called as 1.0f - rv.array() ) = \n" << 1.0f - rv.array() << "\n\n";

}
//---------------------------------------------------------------------------------------
void test_compute_cost(NeuralNetwork& _net)
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";
    //RowVector Y(3), AL(3);
    //Y  << 1, 2, 3;
    //AL << 0.1, 0.2, 0.3;
    //std::cout << "Y  : " << Y << "\n";
    //std::cout << "AL : " << AL << "\n";
    //
    //_net.computeCost(Y, AL);
    //std::cout << "error = " << _net.getError() << "\n\n";
}
//---------------------------------------------------------------------------------------
void test_model_forward(NeuralNetwork& _net)
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";
    
    // manually set weights and biases
    Matrix W1(2, 2), W2(1, 2);
    W1 << 1.1f, 0.9f, 0.8f, 0.6f;
    W2 << 0.5f, 0.6f;
    debug.debugSetWeights(_net, 0, W1);
    debug.debugSetWeights(_net, 1, W2);

    ColVector b1(2), b2(1);
    b1 << 0.1f, 0.2f;
    b2 << 0.3f;
    debug.debugSetBiases(_net, 0, b1);
    debug.debugSetBiases(_net, 1, b2);

    debug.debugPrintWeights(_net);

    Matrix X(2, 4);
    X << 0.0f, 1.0f, 
         1.0f, 0.0f,
         0.2f, 0.8f,
         0.8f, 0.2f;
    //Matrix X(2, 1);
    //X << 0.0f, 1.0f;
    //std::cout << "_X     shape = " << shape<Matrix>(X) << "\n";
    _net.forwardPass(X);

    std::cout << "AL = " << _net.getAL() << "\n";

}
//---------------------------------------------------------------------------------------
void test_compute_cost_AL(NeuralNetwork& _net)
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";
    //Matrix AL = _net.getAL();
    //RowVector vAL(Eigen::Map<RowVector>(AL.data(), AL.cols()));
    //RowVector Y(4);
    //Y << 0.2f, 1.6f, 1.6f, 0.2f;
    //
    //std::cout << "AL = " << vAL << ", shape = " << shape<RowVector>(vAL) << "\n";
    //std::cout << "Y  = " << Y   << ", shape = " << shape<RowVector>(Y  ) << "\n";
    //_net.computeCost(Y, AL, "binary_cross_entropy");
    //std::cout << "binary cross entropy error = " << _net.getError() << "\n";
}
//---------------------------------------------------------------------------------------
void test_sum_vector()
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";
    Matrix m(2, 4);
    m << 0.25877769f, -0.4135453f, -0.4135453f, 0.25877769f, 0.31053323f, -0.49625436f, -0.49625436f, 0.31053323f;
    ColVector b(2);
    double div = 1.0 / 4.0f;
    b = div * m.rowwise().sum();
    std::cout << "dZ : " << shape<Matrix>(m) << "\n" << m << "\n";
    std::cout << "b  : " << shape<ColVector>(b) << "\n" << b << "\n";
}
//---------------------------------------------------------------------------------------
void test_np_divide(NeuralNetwork& _net)
{
    std::cout << "\n" << __PRETTY_FUNCTION__ << ":\n";

    Matrix AL = _net.getAL();
    Matrix Y(1, 4);
    
    assert(Y.cols() == AL.cols() && Y.rows() == AL.rows());

    Y << 0.2f, 1.6f, 1.6f, 0.2f;
    std::cout << "AL : shape = " << shape<Matrix>(AL) << "\n" << AL << "\n";
    std::cout << " Y : shape = " << shape<Matrix>( Y) << "\n" <<  Y << "\n";

    printf("Y, AL : (%ld %ld), (%ld %ld)\n", Y.rows(), Y.cols(), AL.rows(), AL.cols());

    // the following function does not exist in Eigen AFAIK (although not SIMD)
    Matrix R = elementWiseDivide(Y, AL);
    
    std::cout << "R : shape = " << shape<Matrix>(R) << "\n" << R << "\n";
}
//---------------------------------------------------------------------------------------
void test_compute_dAL(NeuralNetwork& _net)
{
    Matrix AL = _net.getAL();
    Matrix Y(1, 4);
    
    assert(Y.cols() == AL.cols() && Y.rows() == AL.rows());

    Y << 0.2f, 1.6f, 1.6f, 0.2f;

    // dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    Matrix R0 = elementWiseDivide(Y, AL);
    Matrix R1 = elementWiseDivide(1.0f - Y.array(), 1.0f - AL.array());

    std::cout << "R0 : shape = " << shape<Matrix>(R0) << "\n" << R0 << "\n";
    std::cout << "R1 : shape = " << shape<Matrix>(R1) << "\n" << R1 << "\n";

    Matrix dAL = -(R0.array() - R1.array());
    std::cout << "dAL : shape = " << shape<Matrix>(dAL) << "\n" << dAL << "\n";
}
//---------------------------------------------------------------------------------------
void test_different_size_batches()
{
    std::vector<uint32_t> layerDims = { 2, 2, 1 };
    NeuralNetwork net(layerDims);
    net.addActivationFunctions({ new Sigmoid, new Sigmoid });

    // TODO : explicitally called for now, move to constructor? -- no
    net.initializeParameters();

    Matrix W1(2, 2), W2(1, 2);
    W1 << 1.1f, 0.9f, 0.8f, 0.6f;
    W2 << 0.5f, 0.6f;
    debug.debugSetWeights(net, 0, W1);
    debug.debugSetWeights(net, 1, W2);

    ColVector b1(2), b2(1);
    b1 << 0.1f, 0.2f;
    b2 << 0.3f;
    debug.debugSetBiases(net,0, b1);
    debug.debugSetBiases(net,1, b2);

    debug.debugVectorSizes(net);
    debug.debugPrintWeights(net);

    Matrix X(2, 4);     // XOR input
    Matrix Y(1, 4);     // XOR output
    X << 0.0f, 1.0f, 0.0f, 1.0f,
         0.0f, 0.0f, 1.0f, 1.0f;
    Y << 0.0f, 1.0f, 1.0f, 0.0f;
    
    net.forwardPass(X);
    net.backwardPass(Y);

    Matrix X1(2, 3);     // XOR input
    Matrix Y1(1, 3);     // XOR output
    X1 << 0.0f, 1.0f, 0.0f,
          0.0f, 0.0f, 1.0f;
    Y1 << 0.0f, 1.0f, 1.0f;

    net.forwardPass(X1);
    net.backwardPass(Y1);

    //net.updateParameters(0.1f);
    //debug.debugPrintWeights(net);
}
//---------------------------------------------------------------------------------------
void test_matrix_blocks(NeuralNetwork& _net)
{
    //std::vector<Sample> trainingData;
    //
    //addSample(trainingData, { 0.0f,  0.0f }, {  0.0f });
    //addSample(trainingData, { 0.0f,  1.0f }, {  1.0f });
    //addSample(trainingData, { 1.0f,  0.0f }, {  1.0f });
    //addSample(trainingData, { 1.0f,  1.0f }, {  0.0f });
    //
    //_net.train(trainingData, 100);
}
//---------------------------------------------------------------------------------------
void test_network_data_manager(NeuralNetwork& _net)
{
    NetworkDataManager* data = new NetworkDataManager(_net, 4, true);
    data->addSample({ 0.0f, 0.0f }, { 0.0f });
    data->addSample({ 1.0f, 0.0f }, { 1.0f });
    data->addSample({ 0.0f, 1.0f }, { 1.0f });
    data->addSample({ 1.0f, 1.0f }, { 0.0f });
    
    _net.train(data, 1);

    delete data;    
}
//---------------------------------------------------------------------------------------
void test_relu()
{
    Matrix M(2, 4);
    M << -1.0f, -0.01f, 0.01f, 1.0f,
         -2.0f, -0.02f, 0.02f, 2.0f;
    std::cout << " M : " << M.format(fmt) << std::endl;
    Matrix Z = relu(M);
    std::cout << " Z : " << Z.format(fmt) << std::endl;

    Matrix dA(2, 4);
    dA << 0.05f, 0.10f, 0.15f, 0.20f,
          0.25f, 0.30f, 0.35f, 0.40f;
    Matrix dZ = relu_d(dA, Z);
    std::cout << " dZ : " << dZ.format(fmt) << std::endl;
    
}
//---------------------------------------------------------------------------------------
void test_matrix_slicing()
{
    Matrix M0(2, 4);
    M0 << -0.1f, 0.2f, -0.3f, 0.4f,
          0.5f, -0.6f, 0.7f, -0.8f;
    Matrix M1(2, 4);
    M1 << 1, 2, 3, 4,
          5, 6, 7, 8;
    
    Matrix M2(2, 4);
    for (size_t i = 0; i < M0.rows(); i++)
        for (size_t j = 0; j < M0.cols(); j++)
            if (M0(i, j) > 0.0f)
                M2(i, j) = M1(i, j);

    std::cout << " M0 : " << M0.format(fmt) << std::endl;
    std::cout << " M1 : " << M1.format(fmt) << std::endl;
    std::cout << " M2 : " << M2.format(fmt) << std::endl;

}
//---------------------------------------------------------------------------------------
void test_random_weights()
{
    std::vector<uint32_t> layers = { 100, 100, 1 };
    NeuralNetwork net(layers);

    net.initializeParameters();
    debug.debugPrintWeights(net);

}
//---------------------------------------------------------------------------------------
void test_activation_funcs_objects(NeuralNetwork& _net)
{
    Matrix M0(2, 4), M1(2, 4);
    M0 << 1.0f, 2.0f, 3.0f, 4.0f,
          5.0f, 6.0f, 7.0f, 8.0f;
    M1 << 0.1f, 0.2f, 0.3f, 0.4f,
          0.5f, 0.6f, 0.7f, 0.8f;

    std::cout << _net.m_activationFuncs[0] << "\n";
    std::cout << _net.m_activationFuncs[1] << "\n";

    Matrix R(2, 4);
    for (size_t i = 0; i < _net.m_activationFuncs.size(); i++)
    {
        std::cout << "type : " << _net.m_activationFuncs[i]->getTypeStr() << "\n";
        
        R = _net.m_activationFuncs[i]->eval(M0);
        std::cout << "eval()\n";
        std::cout << "R: " << R.format(fmt) << "\n";

        std::cout << "eval_d()\n";
        R = _net.m_activationFuncs[i]->eval_d(M0, M1);
        std::cout << "R: " << R.format(fmt) << "\n";
        std::cout << "\n";
    }

}
//---------------------------------------------------------------------------------------
void test_adam()
{
    std::cout << "\n\n" << __PRETTY_FUNCTION__ << ":\n\n";
    NeuralNetwork net({ 2, 2, 1 });
    net.initializeParameters();
    net.addActivationFunctions({ new Sigmoid, new Sigmoid });

    Matrix W1(2, 2), W2(1, 2);
    W1 << 1.1f, 0.9f, 0.8f, 0.6f;
    W2 << 0.5f, 0.6f;
    debug.debugSetWeights(net, 0, W1);
    debug.debugSetWeights(net, 1, W2);
    ColVector b1(2), b2(1);
    b1 << 0.1f, 0.2f;
    b2 << 0.3f;
    debug.debugSetBiases(net, 0, b1);
    debug.debugSetBiases(net, 1, b2);

    net.setOptimizer(new Adam(0.01f, 0.9f, 0.999f));


    // training data
    NetworkDataManager data(net, 4, true);
    data.addSample({ 0.0f, 0.0f }, { 0.0f });
    data.addSample({ 1.0f, 0.0f }, { 1.0f });
    data.addSample({ 0.0f, 1.0f }, { 1.0f });
    data.addSample({ 1.0f, 1.0f }, { 0.0f });
    
    net.train(data, 25000);

    
}
//---------------------------------------------------------------------------------------
void test_set_matrix()
{
    Matrix M(2, 4);
    std::cout << "M shape = " << shape<Matrix>(M) << "\n";
    for (size_t i = 0; i < M.rows(); i++)
        for (size_t j = 0; j < M.cols(); j++)
            M(i, j) = (double)(j * M.rows() + i);
    std::cout << "M:\n" << M.format(fmt) << "\n";
}
//---------------------------------------------------------------------------------------
void test_matrix_random()
{
    Matrix M(100, 100);
    M = Matrix::Random(100, 100).array() * 0.05;
    std::cout << "M shape = " << shape<Matrix>(M) << "\n";
    std::cout << "max = " << M.maxCoeff() << ", min = " << M.minCoeff() << "\n";
}
//---------------------------------------------------------------------------------------
void test_deque()
{
    std::deque<int> q;
    q.resize(5);
    std::cout << "before assignment: ";
    for (size_t i = 0; i < q.size(); i++)
        std::cout << q[i] << " ";
    std::cout << "\n\n";

    for (int i = 0; i < 20; i++)
    {
        q.pop_front();
        q.push_back(i);
        for (size_t j = 0; j < q.size(); j++)
            std::cout << q[j] << " ";
        std::cout << "\n";
    }
}
//---------------------------------------------------------------------------------------
void test_convergence(NeuralNetwork& _net, NetworkDataManager& _data)
{
    size_t n_episodes = 1000;
    size_t n_epochs = 200000;
    size_t failed_convergence = 0;
    std::vector<size_t> convergences;

    for (size_t i = 0; i < n_episodes; i++)
    {
        std::cout << "episode " << i+1 << ":\n";
        _net.reset();

        _net.addActivationFunctions({ new Sigmoid, new Sigmoid });
        _net.initializeParameters();
        _net.setOptimizer(new Momentum(0.05, 0.9));
        _net.setLossFunction(new BinaryCrossEntropy);

        size_t e = _net.train(_data, n_epochs, true);
        convergences.push_back(e);
        if (e == n_epochs)
            failed_convergence++;
    }

    std::cout << "\nFailed convergence: " << failed_convergence << " of ";
    std::cout << n_episodes << " (" << 100.0f*(float)failed_convergence/(float)n_episodes;
    std::cout << " %).\n\n";

}
//---------------------------------------------------------------------------------------
void test_L2_norm()
{
    Matrix M(8, 8);
    for (size_t i = 0; i < 8; i++)
        for (size_t j = 0; j < 8; j++)
            M(j, i) = 0.1 * (i*8+j);
    
    print_matrix<Matrix>(M, "M");

    double lambda = 0.1;
    double div = lambda / (double)M.cols();
    double L2_loss = div * M.array().square().sum();
    std::cout << "L2 loss = " << L2_loss << '\n';

    Matrix R(8, 8);
    R.setConstant(1.0);
    double one_over_m = 0.1;
    print_matrix<Matrix>(one_over_m * R, "0.1 * R");
    Matrix T = M.array() + one_over_m * R.array();
    print_matrix<Matrix>(T, "M + 0.1 * R");

}
//---------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    std::vector<uint32_t> layerDims = { 2, 8, 1 };
    NeuralNetwork net(layerDims);
    
    net.addActivationFunctions({ new Sigmoid, new Sigmoid });
    
    net.initializeParameters();
    
    //net.setOptimizer(new GradientDescent(0.05));
    net.setOptimizer(new Momentum(0.05, 0.9));
    //net.setOptimizer(new Adam());
    
    net.setLossFunction(new BinaryCrossEntropy);

    // OK : test_activation_funcs_objects(net);
    // OK : test_linear_forward();
    // OK : test_sigmoid_d();
    // OK : test_element_wise_log();
    // OK : test_row_vector_dot();
    // OK : test_scalar_minus_matrix();
    // OK : test_compute_cost(net);
    // OK : test_matrix_dot();
    // need m_b as ColVector to be able to compute the activation
    // --> m_Z[l-1] = (W[l-1] * m_A[l-1]).colwise() + m_b[l-1]
    // m_A[l] = activation_func(m_Z[l-1])
    // OK : test_model_forward(net);
    // m_A.back() contains the final network output:
    //  --> Has to be converted to a RowVector before use in computeCost()
    // OK : test_compute_cost_AL(net);
    // OK : test_model_forward(net);
    //net.debugPrintZ();
    //net.debugPrintA();
    //net.debugVectorSizes();
    // OK : test_sum_vector();
    // OK : test_np_divide(net);
    // OK : test_compute_dAL(net);
    // OK : test_different_size_batches();
    //Matrix W1(2, 2), W2(1, 2);
    //W1 << 1.1f, 0.9f, 0.8f, 0.6f;
    //W2 << 0.5f, 0.6f;
    //debug.debugSetWeights(net, 0, W1);
    //debug.debugSetWeights(net, 1, W2);
    //ColVector b1(2), b2(1);
    //b1 << 0.1f, 0.2f;
    //b2 << 0.3f;
    //debug.debugSetBiases(net, 0, b1);
    //debug.debugSetBiases(net, 1, b2);
    //net.debugVectorSizes();
    //net.debugPrintWeights();
    //Matrix X(2, 4);     // XOR input
    //Matrix Y(1, 4);     // XOR output
    //X << 0.0f, 1.0f, 0.0f, 1.0f,
    //     0.0f, 0.0f, 1.0f, 1.0f;
    //Y << 0.0f, 1.0f, 1.0f, 0.0f;
    //
    //net.forwardPass(X);
    //net.backwardPass(Y);
    //
    //net.computeCost(Y);
    //std::cout << "error : " << net.getError() << "\n";
    //
    //net.updateParameters(0.1f);
    //net.debugPrintWeights();
    // OK : test_matrix_blocks(net);
    // OK : test_network_data_manager(net);

    NetworkDataManager data(net, 4, true);
    data.addSample({ 0.0, 0.0 }, { 0.0 });
    data.addSample({ 1.0, 0.0 }, { 1.0 });
    data.addSample({ 0.0, 1.0 }, { 1.0 });
    data.addSample({ 1.0, 1.0 }, { 0.0 });

    // train the network
    size_t n_epochs = 50000;
    size_t trained_epochs = net.train(data, n_epochs, true);

    // check training results
    Matrix X = data.getInputMatrix();
    Matrix output = net.predict(X);
    std::cout << "input\n" << X.format(fmt) << "\n";
    std::cout << "output\n " << output.format(fmt) << "\n";

    
    //std::vector<double> history = net.getLossHistory();
    //for (size_t i = 0; i < history.size(); i++)
    //{
    //    if (i && i % 1000 == 0)
    //        printf("%5zu : %f\n", i, history[i]);
    //}

    // OK : works exactly like numpy : test_L2_norm();
    // OK : test_convergence(net, data);

    //debug.debugPrintWeights(net);
    // OK : test_adam();
    // OK : test_matrix_slicing();
    // OK : test_relu();
    // OK : test_random_weights();
    // OK : test_set_matrix();
    // OK : test_matrix_random();
    // OK : test_deque();

    return 0;

}




