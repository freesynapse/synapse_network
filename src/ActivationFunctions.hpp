
#pragma once

#include "Utils.hpp"

#include <iostream>
#include <string>


//---------------------------------------------------------------------------------------
class ActivationFunction
{
public:
    ActivationFunction() = default;
    virtual ~ActivationFunction() = default;

    virtual Matrix eval   (const Matrix& _Z) = 0;
    virtual Matrix eval_d (const Matrix& _dA, const Matrix& _Z) = 0;
    virtual std::string getTypeStr() = 0;
};
//---------------------------------------------------------------------------------------
class Sigmoid : public ActivationFunction
{
public:
     Sigmoid() = default;
    ~Sigmoid() = default;

    Matrix eval(const Matrix& _Z) override
    { 
        Matrix A = 1.0f / (1.0f + (-_Z).array().exp());
        return A;
    }

    Matrix eval_d(const Matrix& _dA, const Matrix& _Z) override
    {
        Matrix s = eval(_Z);
        Matrix dZ = _dA.array() * s.array() * (1.0f - s.array());
        return dZ;
    }

    std::string getTypeStr() override { return "sigmoid"; }
};
//---------------------------------------------------------------------------------------
class ReLU : public ActivationFunction
{
public:
     ReLU() = default;
    ~ReLU() = default;

    Matrix eval(const Matrix& _Z) override
    {
        Matrix A = _Z.array().max(0.0f);
        return A;
    }

    Matrix eval_d(const Matrix& _dA, const Matrix& _Z) override
    {
        assert(_dA.rows() == _Z.rows() && _dA.cols() == _Z.cols());
        
        Matrix dZ = _dA;

        for (size_t i = 0; i < _Z.rows(); i++)
            for (size_t j = 0; j < _Z.cols(); j++)
                if (_Z(i, j) <= 0.0f)
                    dZ(i, j) = 0.0f;
        return dZ;
    }

    std::string getTypeStr() override { return "ReLU"; }
};
//---------------------------------------------------------------------------------------
class Tanh : public ActivationFunction
{
public:
     Tanh() = default;
    ~Tanh() = default;

    Matrix eval(const Matrix& _Z) override
    {
        Matrix Z_exp = _Z.array().exp();
        Matrix Z_neg_exp = (-_Z).array().exp();
        Matrix A = (Z_exp.array() - Z_neg_exp.array()) / (Z_exp.array() + Z_neg_exp.array());
        return A;
    }

    Matrix eval_d(const Matrix& _dA, const Matrix& _Z) override
    {
        Matrix tanh = eval(_Z);
        Matrix dZ = _dA.array() * (1.0f - tanh.array().square());
        return dZ;
    }

    std::string getTypeStr() override { return "tanh"; }
};
//---------------------------------------------------------------------------------------
class SoftMax : public ActivationFunction
{
public:
     SoftMax() = default;
    ~SoftMax() = default;

    Matrix eval(const Matrix& _Z) override
    {
        return Matrix;
    }

    Matrix eval_d(const Matrix& _dA, const Matrix& _Z) override
    {
        return Matrix;
    }

    std::string getTypeStr() override { return "softmax"; }
};





//---------------------------------------------------------------------------------------
inline Matrix sigmoid(const Matrix& _Z)
{
    Matrix A = 1.0f / (1.0f + (-_Z).array().exp());
    return A;
}

inline Matrix sigmoid_d(const Matrix& _dA, const Matrix& _Z)
{
    Matrix s = sigmoid(_Z);
    Matrix dZ = _dA.array() * s.array() * (1.0f - s.array());
    return dZ;
}

inline Matrix relu(const Matrix& _Z)
{
    Matrix A = _Z.array().max(0.0f);
    return A;
}

inline Matrix relu_d(const Matrix& _dA, const Matrix& _Z)
{
    assert(_dA.rows() == _Z.rows() && _dA.cols() == _Z.cols());

    //Matrix dZ = Matrix(_dA.rows(), _dA.cols());
    Matrix dZ = _dA;

    for (size_t i = 0; i < _Z.rows(); i++)
        for (size_t j = 0; j < _Z.cols(); j++)
            if (_Z(i, j) <= 0.0f)
                dZ(i, j) = 0.0f;
    return dZ;
}

