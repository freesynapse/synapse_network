
#pragma once

#include "NeuralNetwork.hpp"

#include <iostream>



class NetworkDebugger
{
public:
    //
    void debugSetWeights(NeuralNetwork& _net, uint32_t _l, const Matrix& _W) 
    { 
        _net.m_W[_l] = _W; 
    }
    //-----------------------------------------------------------------------------------
    void debugSetBiases(NeuralNetwork& _net, uint32_t _l, const ColVector& _b)
    { 
        _net.m_b[_l] = _b; 
    }
    //-----------------------------------------------------------------------------------
    void debugPrintWeights(NeuralNetwork& _net)
    {
        for (size_t l = 0; l < _net.m_W.size(); l++)
        {
            std::cout << "layer " << l << "\n\n";
            std::cout << "W" << l << " shape = " << shape<Matrix>(_net.m_W[l]) << "\n";
            std::cout << _net.m_W[l] << "\n";
            std::cout << "b" << l << " shape = " << shape<ColVector>(_net.m_b[l]) << "\n";
            std::cout << _net.m_b[l] << "\n\n\n";
        }
    }
    //-----------------------------------------------------------------------------------
    void debugPrintA(NeuralNetwork& _net)
    {
        for (size_t l = 0; l < _net.m_A.size(); l++)
        {
            std::cout << "layer " << l << "\n\n";
            std::cout << "A" << l << " shape = " << shape<Matrix>(_net.m_A[l]);
            std::cout << "\n" << _net.m_A[l] << "\n\n\n";
        }        
    }
    //-----------------------------------------------------------------------------------
    void debugPrintZ(NeuralNetwork& _net)
    {
        for (size_t l = 0; l < _net.m_Z.size(); l++)
        {
            std::cout << "layer " << l << "\n\n";
            std::cout << "Z" << l << " shape = " << shape<Matrix>(_net.m_Z[l]);
            std::cout << "\n" << _net.m_Z[l] << "\n\n\n";
        }        
    }
    //-----------------------------------------------------------------------------------
    void debugVectorSizes(NeuralNetwork& _net)
    {
        std::cout << "layer count : " << _net.m_layerCount << "\n";
        std::cout << "m_W  : " << _net.m_W.size()  << "\n";
        std::cout << "m_b  : " << _net.m_b.size()  << "\n";
        std::cout << "m_Z  : " << _net.m_Z.size()  << "\n";
        std::cout << "m_A  : " << _net.m_A.size()  << "\n";
        std::cout << "m_dW : " << _net.m_dW.size() << "\n";
        std::cout << "m_db : " << _net.m_dW.size() << "\n";
        std::cout << "m_dA : " << _net.m_dA.size() << "\n";
    }

};
