
#pragma once

#include "NeuralNetwork.hpp"

#include <initializer_list>
#include <random>


class NetworkDataManager
{
public:
    friend class NeuralNetwork;
    
public:   
    // ctor / dtor
    //
    NetworkDataManager(NeuralNetwork& _net, size_t _batch_size=0, bool _shuffle=true)
    {
        m_batchSize = _batch_size;
        m_shuffle = _shuffle;
        // network input and output dimensions
        m_Xm = _net.m_layerDims[0];
        m_Ym = _net.m_layerDims.back();
        // seed the random generator
        std::random_device rd;
        m_rng = std::mt19937(rd());
    }
    ~NetworkDataManager() = default;


    // API functions
    //

    /* Adds a training example. */
    void addSample(std::initializer_list<double>_x, 
                   std::initializer_list<double>_y)
    {
        Array1d x = Eigen::Map<const Array1d>(_x.begin(), _x.size());
        Array1d y = Eigen::Map<const Array1d>(_y.begin(), _y.size());
        Sample example(x, y);
        m_data.push_back(example);
    }

    /* Adds a training example. */
    void addSample(const Array1d& _x, const Array1d& _y)
    {
        Sample example(_x, _y);
        m_data.push_back(example);
    }

    /* Adds a training example. */
    void addSample(const Sample& _sample)
    {
        m_data.push_back(_sample);
    }

    /* Adds a batch of samples. */
    void addSamples(const std::vector<Array1d>& _X,
                    const std::vector<Array1d>& _Y)
    {
        if (_X.size() != _Y.size())
        {
            std::cout << "dimensions of X and Y unequal.\n";
            return;
        }

        for (size_t i = 0; i < _X.size(); i++)
            addSample(_X[i], _Y[i]);
    }

    /* Fills the matrices _X and _Y with the next batch. */
    size_t getNextBatch(Matrix* _X_out, Matrix* _Y_out)
    {
        if (!m_finalized)
            finalizeData();

        if (m_samplesLeft > m_batchSize)
        {
            // Construct a matrix of batch_size examples (columns), with rows as 
            // the number of network input/output nodes.
            Matrix X(m_Xm, m_batchSize);
            Matrix Y(m_Ym, m_batchSize);

            m_sampleIndex = m_currentBatch * m_batchSize;
            for (size_t i = 0; i < m_batchSize; i++)
            {
                // insert to batch matrices
                X.block(0, i, m_Xm, 1) = m_data[m_sampleIndex + i].first;
                Y.block(0, i, m_Ym, 1) = m_data[m_sampleIndex + i].second;
            }
            // set up for getting the next batch on next call
            m_currentBatch++;

            // return batch
            *_X_out = X;
            *_Y_out = Y;

            m_samplesLeft -= m_batchSize;
        }
        // is there remaining examples?
        else
        {
            m_sampleIndex = m_data.size() - m_samplesLeft;
            size_t remaining = m_samplesLeft;//m_data.size() - m_sampleIndex;
            if (remaining > 0)
            {
                Matrix X(m_Xm, remaining);
                Matrix Y(m_Ym, remaining);

                for (size_t i = 0; i < remaining; i++)
                {
                    X.block(0, i, m_Xm, 1) = m_data[m_sampleIndex + i].first;
                    Y.block(0, i, m_Ym, 1) = m_data[m_sampleIndex + i].second;
                }

                // return remaining samples
                *_X_out = X;
                *_Y_out = Y;

                m_samplesLeft -= remaining;
            }
        }

        return m_samplesLeft;
    }

    /* Resets batches for next epoch */
    void resetBatches()
    {
        if (m_finalized != true)
            finalizeData();

        if (m_shuffle)
            shuffleData();

        m_currentBatch = 0;
        m_samplesLeft = m_data.size();
    }


    // accessors
    void setBatchSize(const size_t _sz)  { m_batchSize = _sz; finalizeData(); }
    const size_t getSampleSize()   const { return m_data.size();  }
    const size_t getBatchSize()    const { return m_batchSize;    }
    const size_t getBatchCount()   const { return m_batchCount;   }
    const size_t getCurrentBatch() const { return m_currentBatch; }
    const std::vector<Sample>& getData() const { return m_originalData; }
    Matrix getInputMatrix()
    {
        Matrix X(m_Xm, m_data.size());
        for (size_t i = 0; i < m_data.size(); i++)
            X.block(0, i, m_Xm, 1) = m_data[i].first;
        return X;
    }
    Matrix getOutputMatrix()
    {
        Matrix Y(m_Ym, m_data.size());
        for (size_t i = 0; i < m_data.size(); i++)
            Y.block(0, i, m_Ym, 1) = m_data[i].second;
        return Y;
    }


private:
    /* 
     * Shuffles the training data in-place. 
     */
    void shuffleData()
    {
        std::shuffle(m_data.begin(), m_data.end(), m_rng);
    }

    /* 
     * Called after all data been entered. Calculates batch sizes,
     * number of batches and resets the batch counter.
     */
    void finalizeData()
    {
        // sanity checks
        if (!m_data.size())
            std::cout << "warning : no training data\n";

        m_batchSize = m_batchSize == 0 ? m_data.size() : m_batchSize;
        if (m_batchSize > m_data.size())
            m_batchSize = m_data.size();
        
        // the remainder will be handled separately
        m_batchCount = m_data.size() / m_batchSize;

        // set the current batch to 0, setting up for getNextBatch().
        m_currentBatch = 0;

        m_samplesLeft = m_data.size();

        // store a copy of the original data
        m_originalData = m_data;

        if (m_shuffle)
            shuffleData();

        m_finalized = true;
    }


private:
    std::vector<Sample> m_data;
    std::vector<Sample> m_originalData; // copy of the original, unshuffled data
    size_t m_batchSize    = 0;
    size_t m_batchCount   = 0;
    size_t m_currentBatch = 0;
    size_t m_sampleIndex  = 0;
    size_t m_samplesLeft  = 0;
    size_t m_Xm           = 0;  // network input and output dimensions (rows)
    size_t m_Ym           = 0;
    bool   m_finalized    = false;
    bool   m_shuffle      = false;
    
    std::mt19937 m_rng;

};


