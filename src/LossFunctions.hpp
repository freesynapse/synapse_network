
#pragma once

#include "Utils.hpp"


class LossFunction
{
public:
    friend class NeuralNetwork;

public:
    LossFunction() = default;
    virtual ~LossFunction() = default;

    const double getLoss() const { return m_loss; }

private:
    virtual inline double compute(Matrix& _AL, Matrix& _Y) = 0;
    virtual inline const Matrix& compute_d(Matrix& _AL, Matrix& _Y) = 0;

protected:
    double m_loss = 0;
    Matrix m_dAL;

};
//---------------------------------------------------------------------------------------
class BinaryCrossEntropy : public LossFunction
{
private:    
    inline double compute(Matrix& _AL, Matrix& _Y) override
    {
        double div = 1.0 / (double)_Y.cols();
    
        RowVector AL(Eigen::Map<RowVector>(_AL.data(), _AL.cols()));
        RowVector Y (Eigen::Map<RowVector>(_Y.data(), _Y.cols()));

        RowVector AL_log = AL.array().log();
        RowVector AL_log_m_1 = (1.0 - AL.array()).log();
        RowVector Y_m_1 = 1.0 - Y.array();

        m_loss = div * (-Y.dot(AL_log.transpose()) - Y_m_1.dot(AL_log_m_1.transpose()));

        return m_loss;
    }
    //-----------------------------------------------------------------------------------
    inline const Matrix& compute_d(Matrix& _AL, Matrix& _Y) override
    {
        Matrix R0 = elementWiseDivide(_Y, _AL);
        Matrix R1 = elementWiseDivide(1.0 - _Y.array(), 1.0 - _AL.array());

        m_dAL = -(R0.array() - R1.array());

        return m_dAL;
    }
};
//---------------------------------------------------------------------------------------
class CategoricalCrossEntropy : public LossFunction
{
private:    
    inline double compute(Matrix& _AL, Matrix& _Y) override
    {
        m_loss = 0.0;
        return 0.0;
    }
    //-----------------------------------------------------------------------------------
    inline const Matrix& compute_d(Matrix& _AL, Matrix& _Y) override
    {
        m_dAL = _AL - _Y;
        return m_dAL;
    }
};
//---------------------------------------------------------------------------------------
class MeanSquaredError : public LossFunction
{
private:
    inline double compute(Matrix& _AL, Matrix& _Y) override
    {
        m_loss = 0.0;
        return 0.0;
    }
    //-----------------------------------------------------------------------------------
    inline const Matrix& compute_d(Matrix& _AL, Matrix& _Y) override
    {
        return m_dAL;
    }
};


