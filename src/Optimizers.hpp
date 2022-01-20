
#pragma once

#include "NeuralNetwork.hpp"

#include <math.h>


class Optimizer
{
public:
    Optimizer() = default;
    virtual ~Optimizer() = default;

    virtual void updateParameters(NeuralNetwork* _net, uint32_t _epoch) = 0;
    virtual void initialize(NeuralNetwork* _net) = 0;

protected:
    double m_eta;
    uint32_t m_t;
};
//---------------------------------------------------------------------------------------
class GradientDescent : public Optimizer
{
public:
    GradientDescent(double _eta=0.05)
    {
        m_eta = _eta;
    }

    void updateParameters(NeuralNetwork* _net, uint32_t _epoch) override
    {
        // gradients for W and b are stored in m_dW[] and m_db[] per layer
        for (size_t i = 0; i < _net->m_W.size(); i++)
        {
            _net->m_W[i] = _net->m_W[i].array() - (m_eta * _net->m_dW[i].array());
            _net->m_b[i] = _net->m_b[i].array() - (m_eta * _net->m_db[i].array());
        }
    }

    void initialize(NeuralNetwork* _net) override {}

};
//---------------------------------------------------------------------------------------
class Momentum : public Optimizer
{
public:
    /*
     * Eta is the learning rate and gamma is the update weight:
     *  v_t = gamma * v_{t-1} + eta * dW
     *  W = W - dW
     */
    Momentum(double _eta=0.05, double _gamma=0.9)
    {
        m_eta = _eta;
        m_gamma = _gamma;
    }

    void updateParameters(NeuralNetwork* _net, uint32_t _epoch) override
    {
        // gradients for W and b are stored in m_dW[] and m_db[] per layer
        for (size_t i = 0; i < _net->m_W.size(); i++)
        {
            m_vW[i] = m_gamma * m_vW[i].array() + m_eta * _net->m_dW[i].array();
            m_vb[i] = m_gamma * m_vb[i].array() + m_eta * _net->m_db[i].array();

            _net->m_W[i] = _net->m_W[i].array() - m_vW[i].array();
            _net->m_b[i] = _net->m_b[i].array() - m_vb[i].array();
        }
    }

    void initialize(NeuralNetwork* _net) override
    {
        for (size_t i = 0; i < _net->m_W.size(); i++)
        {
            // initialize previous to zeros.
            m_vW.push_back(Matrix::Zero(_net->m_W[i].rows(), _net->m_W[i].cols()));
            m_vb.push_back(ColVector::Zero(_net->m_b[i].cols()));
        }
    }

private:
    double m_gamma;
    std::vector<Matrix> m_vW;
    std::vector<ColVector> m_vb;
};
//---------------------------------------------------------------------------------------
class RMSprop : public Optimizer
{
public:
    void updateParameters(NeuralNetwork* _net, uint32_t _epoch) override {}
    void initialize(NeuralNetwork* _net) override {}

};
//---------------------------------------------------------------------------------------
class Adam : public Optimizer
{
public:
    Adam(double _eta=0.01f, double _beta1=0.9f, double _beta2=0.999f)
    {
        m_eta   = _eta;
        m_beta1 = _beta1;
        m_beta2 = _beta2;
    }

    void updateParameters(NeuralNetwork* _net, uint32_t _epoch) override
    {
        static double epsilon = (double)(1e-8);

        double div_beta1 = (_epoch < m_beta1Limit ? 1.0 / (1.0 - pow(m_beta1, _epoch+1)) : 1.0);
        double div_beta2 = (_epoch < m_beta2Limit ? 1.0 / (1.0 - pow(m_beta2, _epoch+1)) : 1.0);

        for (size_t i = 0; i < _net->m_dW.size(); i++)
        {
            // compute 1st moment moving averages of gradients
            m_vW[i] = m_beta1 * m_vW[i].array() + (1.0 - m_beta1) * _net->m_dW[i].array();
            m_vb[i] = m_beta1 * m_vb[i].array() + (1.0 - m_beta1) * _net->m_db[i].array();

            // bias-corrected 1st moment estimates
            m_vW_c[i] = m_vW[i].array() * div_beta1;
            m_vb_c[i] = m_vb[i].array() * div_beta1;

            // compute 2nd order moving averages of gradients
            m_sW[i] = m_beta2 * m_sW[i].array() + (1.0 - m_beta2) * _net->m_dW[i].array().square();
            m_sb[i] = m_beta2 * m_sb[i].array() + (1.0 - m_beta2) * _net->m_db[i].array().square();

            // bias-corrected 2nd moment estimates
            m_sW_c[i] = m_sW[i].array() * div_beta2;
            m_sb_c[i] = m_sb[i].array() * div_beta2;

            // update parameters
            _net->m_W[i] = _net->m_W[i].array() - m_eta * (m_vW_c[i].array() / ( m_sW_c[i].array().sqrt() + epsilon));
            _net->m_b[i] = _net->m_b[i].array() - m_eta * (m_vb_c[i].array() / ( m_sb_c[i].array().sqrt() + epsilon));
        }

    }

    void initialize(NeuralNetwork* _net) override
    {
        for (size_t i = 0; i < _net->m_W.size(); i++)
        {
            // initialize previous to zeros.
            m_vW.push_back(Matrix::Zero(_net->m_W[i].rows(), _net->m_W[i].cols()));
            m_sW.push_back(Matrix::Zero(_net->m_W[i].rows(), _net->m_W[i].cols()));
            m_vW_c.push_back(Matrix::Zero(_net->m_W[i].rows(), _net->m_W[i].cols()));
            m_sW_c.push_back(Matrix::Zero(_net->m_W[i].rows(), _net->m_W[i].cols()));
            m_vb.push_back(ColVector::Zero(_net->m_b[i].cols()));
            m_sb.push_back(ColVector::Zero(_net->m_b[i].cols()));
            m_vb_c.push_back(ColVector::Zero(_net->m_b[i].cols()));
            m_sb_c.push_back(ColVector::Zero(_net->m_b[i].cols()));
        }

        // find the epochs where the power(beta, epoch) approaches 1.0f,
        // thereby permitting us to skip the computation of 
        //      1.0f / (1.0f - pow(beta, epoch))
        //
        // This is basically solving the equation
        //      1.0 - (1.0 - beta^x) < epsilon -->
        //      beta^x < epsilon  -->
        //      log(beta) * x < log(epsilon) -->
        //      x < log(epsilon) / log(beta)
        //
        double epsilon = 0.001;
        m_beta1Limit = (int)(logf(epsilon) / logf(m_beta1)); // for 0.9   -->   65
        m_beta2Limit = (int)(logf(epsilon) / logf(m_beta2)); // for 0.999 --> 6904
    }

private:
    double m_beta1;  // exponential decay for the 1st moment estimates
    double m_beta2;  // exponential decay for the 2nd moment estimates

    uint32_t m_beta1Limit;  // epoch limit after which pow(beta1, epoch+1) is very close to 1.0f
    uint32_t m_beta2Limit;  // epoch limit after which pow(beta2, epoch+1) is very close to 1.0f

    std::vector<Matrix>     m_vW;   // moving averages of 1st moment gradients
    std::vector<ColVector>  m_vb;   //
    std::vector<Matrix>     m_vW_c; // bias-corrected 1st moment estimates
    std::vector<ColVector>  m_vb_c; //
    
    std::vector<Matrix>     m_sW;   // moving averages of 2nd moment (squared) gradients
    std::vector<ColVector>  m_sb;   //
    std::vector<Matrix>     m_sW_c; // bias-corrected 2nd moment (squared) estimates
    std::vector<ColVector>  m_sb_c; //
};
