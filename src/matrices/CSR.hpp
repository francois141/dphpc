//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_CSR_H
#define DPHPC_CSR_H

#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "triplet.h"
#include "COO.hpp"

template<class T>
class COO;

template<class T>
class CSR {
public:
    
    CSR()
    : rows(0), cols(0), rowPositions(std::vector<int>(0)), colPositions(std::vector<int>(0)), values(std::vector<T>(0))
    {}

    CSR(int rows, int cols, std::vector<Triplet<T>> triplets)
    : rows(rows), cols(cols) {
        init_csr(triplets);
    }

    CSR(const CSR &other)
    : rows(other.rows), cols(other.cols), rowPositions(other.rowPositions), colPositions(other.colPositions), values(other.values), startIdx(other.startIdx)
    {}

    CSR(COO<T> &coo) {
        this->rows = coo.getRows();
        this->cols = coo.getCols();

        std::vector<Triplet<T>> triplets(coo.getValues().size());
        
        for (uint32_t i = 0; i < coo.getValues().size(); i++) {
            triplets[i].row = coo.getRowPositions()[i];
            triplets[i].col = coo.getColPositions()[i];
            triplets[i].value = coo.getValues()[i];
        }
        init_csr(triplets);
    }

    friend std::ostream &operator<<(std::ostream &os, const CSR &csr) {
        for(const T& value: csr.values) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: csr.colPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: csr.rowPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        return os;
    }

    friend bool operator==(const CSR &lhs, const CSR &rhs) {
        bool posMatch = lhs.colPositions == rhs.colPositions &&
               lhs.rowPositions == rhs.rowPositions &&
               lhs.rows == rhs.rows &&
               lhs.cols == rhs.cols;

        if(!posMatch) {
            std::cout << "Cols and Rows don't match" << std::endl;
        }

        bool valMatch = true;
        T largestDiff = 0;

        if(std::is_integral<T>::value) {
            valMatch = (lhs.values == rhs.values);
        } else {
            if(lhs.values.size() != rhs.values.size()) {
                return false;
            }

            for(uint32_t i = 0; i < lhs.values.size();i++) {
                T diff = (std::abs(rhs.values[i] - lhs.values[i]) / rhs.values[i]);
                largestDiff = std::max(largestDiff, diff);
                valMatch &= diff <= 1e-3;
            }
        }

        if(!valMatch) {
            std::cout << "Largest relative difference is : " << largestDiff << std::endl;
        }

        return posMatch && valMatch;
    }

    int getRows() const {
        return this->rows;
    }

    int getCols() const {
        return this->cols;
    }

    void clearValues() {
        std::fill(this->values.begin(), this->values.end(), 0);
    }

    std::vector<T> &getValues() {
        return this->values;
    }

    void setValues(const std::vector<T> &values) {
        this->values = values;
    }

    const std::vector<int> &getColPositions() {
        return this->colPositions;
    }

    void setColPositions(const std::vector<int> &colPositions) {
        this->colPositions = colPositions;
    }

    const std::vector<int> &getRowPositions() {
        return this->rowPositions;
    }

    void setRowPositions(const std::vector<int> &rowPositions) {
        this->rowPositions = rowPositions;
    }

    T *getValue(int j) {
        return &values[j];
    }

    const std::vector<int> &getStartIdx(){
        return this->startIdx;
    }

    void computeDispatcher(int numThreads) {
        this->_dispatch(numThreads,1);
    }

    void computeDispatcherTensorCores(int nbThreads) {
        this->_dispatch(nbThreads, 16);
    }

private:
    int rows;
    int cols;
    std::vector<int> rowPositions;
    std::vector<int> colPositions;
    std::vector<T> values;
    std::vector<int> startIdx;

    void init_csr(std::vector<Triplet<T>> triplets) {
        assert(triplets.size() > 0);

        std::sort(triplets.begin(), triplets.end());

        for (auto e: triplets) {
            assert(e.row >= 0 && e.row < this->rows);
            assert(e.col >= 0 && e.col < this->cols);
        }

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);

        this->colPositions.reserve(triplets.size());
        this->values.reserve(triplets.size());

        unsigned int idx = 0;
        for (int i = 0; i < this->rows; i++) {
            this->rowPositions.emplace_back(idx);

            while(idx < triplets.size() && triplets[idx].row == i) {
                this->values.push_back(triplets[idx].value);
                this->colPositions.push_back(triplets[idx].col);
                idx++;
            }
        }

        this->rowPositions.emplace_back(idx);

        this->_dispatch(32*32,1);
    }

    bool testValue(const std::vector<int> &sizes, int val, int nbThreads) {
        if(*std::max_element(sizes.begin(), sizes.end()) > val) {
            return false;
        }

        int cnt = 1;
        int sum = 0;
        for(const int e: sizes) {
            sum += e;
            if(sum > val) {
                cnt++;
                sum = e;
            }
        }

        return cnt <= nbThreads;
    }

    void _dispatch(int nbThreads, int stride) {
        // Prepare the sizes
        std::vector<int> sizes;
        for(int i = 0; i < this->rows; i+=stride) {
            sizes.push_back(this->rowPositions[i+stride] - this->rowPositions[i]);
        }

        // Step 1)
        // Split the given array into K sub-arrays such that maximum sum of all sub arrays is minimum
        // https://www.geeksforgeeks.org/split-the-given-array-into-k-sub-arrays-such-that-maximum-sum-of-all-sub-arrays-is-minimum/
        int start = 1;
        int end = this->values.size();
        int segSize = 0;

        while(start <= end) {
            int middle = (start + end) / 2;
            if(testValue(sizes, middle, nbThreads)) {
                segSize = middle;
                end = middle-1;
            } else {
                start = middle+1;
            }
        }

        // Step 2)
        // Compute each thread's range of responsibility
        int currSizeThread = 0;

        this->startIdx.reserve(this->rowPositions.size());

        // First thread starts at 0
        this->startIdx.clear();
        this->startIdx.push_back(0);

        // While currSizeThread <= segSize ==> give it to the same thread
        for(size_t i = 0; i < sizes.size(); i++) {
            currSizeThread += sizes[i];

            if(currSizeThread > segSize) {
                // Give it to the new thread;
                this->startIdx.push_back((int)i*stride);
                currSizeThread = sizes[i];
            }
        }

        // We need to make sure the size is similar
        while((int)this->startIdx.size() <= nbThreads) {
            // The threads here don't do anything
            this->startIdx.push_back(this->rows);
        }
    }
};


#endif //DPHPC_CSR_H
