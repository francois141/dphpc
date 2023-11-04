//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_CSR_H
#define DPHPC_CSR_H

#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>

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
    : rows(other.rows), cols(other.cols), rowPositions(other.rowPositions), colPositions(other.colPositions), values(other.values)
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
    
        bool valMatch = true;

        if(std::is_integral<T>::value) {
            valMatch = (lhs.values == rhs.values);
        } else {
            if(lhs.values.size() != rhs.values.size()) {
                return false;
            }

            for(uint32_t i = 0; i < lhs.values.size();i++) {
                valMatch &= abs(rhs.values[i] - lhs.values[i]) <= 1e-6;
            }
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

    const std::vector<T> &getValues() {
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

private:
    int rows;
    int cols;
    std::vector<int> rowPositions;
    std::vector<int> colPositions;
    std::vector<T> values;

    void init_csr(std::vector<Triplet<T>> triplets) {
        assert(triplets.size() > 0);

        for(auto e: triplets) {
            assert(e.row >= 0 && e.row < this->rows);
            assert(e.col >= 0 && e.col < this->cols);
        }

        auto comp = [](const Triplet<T> t1, const Triplet<T> t2) -> bool {
            return t1.row < t2.row || (t1.row == t2.row && t1.col < t2.col);
        };

        std::sort(triplets.begin(), triplets.end(), comp);

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);

        this->colPositions.reserve(triplets.size());
        this->values.reserve(triplets.size());

        unsigned int idx = 0;
        for (size_t i = 0; i < this->rows; i++) {
            this->rowPositions.emplace_back(idx);

            while(idx < triplets.size() && triplets[idx].row == i) {
                this->values.push_back(triplets[idx].value);
                this->colPositions.push_back(triplets[idx].col);
                idx++;
            }
        }

        this->rowPositions.emplace_back(idx);
    }
};


#endif //DPHPC_CSR_H
