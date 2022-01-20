
#pragma once

#include "Eigen/Eigen"

#include <sstream>
#include <random>
#include <initializer_list>
#include <iostream>
#include <string>


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrix;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector;
typedef Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor> RowVector;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor> ColVector;
typedef Eigen::Array<double, Eigen::Dynamic, 1> Array1d;
typedef std::pair<Array1d, Array1d> Sample;


//#define DEBUG_NEURAL_NET

/* Formatting of matrix output. */
static Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, " ", "\n", "\t");


/*
 * Adds an input and a target to a vector of Samples
 */
static void addSample(std::vector<Sample>& _data, 
                      std::initializer_list<double>_input, 
                      std::initializer_list<double>_output)
{
    Array1d input  = Eigen::Map<const Array1d>( _input.begin(),  _input.size());
    Array1d output = Eigen::Map<const Array1d>(_output.begin(), _output.size());
    Sample in_out(input, output);
    _data.push_back(in_out);
}

/*
 * Shuffles a vector based on the Fisher-Yates algorithm.
 */
inline void shuffle_in_place(std::vector<Sample>& _data)
{
    // different seed every time, but same permutation for X and Y.
    srand(time(NULL));

    int size = (int)_data.size();
    for (int i = 0; i < size; i++)
    {
        int j = i + rand() % (size - i);
        std::swap(_data[i], _data[j]);
    }
}

/*
 * Divides element (r, c) of matrix _A with element (r, c) in 
 * matrix _B.
 */
inline Matrix elementWiseDivide(const Matrix& _A, const Matrix& _B)
{
    Matrix R(_A.rows(), _A.cols());
 
    for (size_t r = 0; r < _A.rows(); r++)
        for (size_t c = 0; c < _A.cols(); c++)
            R(r, c) = _A(r, c) / _B(r, c);

    return R;    
}

/*
 * Returns the shape of a Eigen matrix or vector.
 */
using Eigen::EigenBase;
template<typename T>
inline std::string shape(const EigenBase<T>& _x)
{
    std::ostringstream os;
    os << '(' << _x.rows() << ", " << _x.cols() << ')';
    return os.str();
}

/*
 * Prints the shape of a Eigen matrix or vector along with its shape.
 */
template<typename T>
inline void print_matrix(const T& _m, const std::string& _name="")
{
    std::string name = _name == "" ? "matrix" : _name;

    std::cout << name << " shape = " << shape<T>(_m) << '\n';
    std::cout << _m.format(fmt) << '\n';
}

