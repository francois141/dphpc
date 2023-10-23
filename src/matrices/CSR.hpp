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
    
    CSR() {
        this->rows = 0;
        this->cols = 0;

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);
    }

    CSR(int rows, int cols, std::vector<Triplet<T>> triplets) {
        this->rows = rows;
        this->cols = cols;

        init_csr(triplets);
    }

    CSR(const CSR &other) : rows(other.rows), cols(other.cols), values(other.values), colPositions(other.colPositions), rowPositions(other.rowPositions) {}

    CSR(COO<T> &coo) {
        this->rows = coo.getRows();
        this->cols = coo.getCols();

        std::vector<Triplet<T>> triplets(coo.getValues().size());
        
        for (int i = 0; i < coo.getValues().size(); i++) {
            triplets[i].row = coo.getRowPositions()[i];
            triplets[i].col = coo.getColPositions()[i];
            triplets[i].value = coo.getValues()[i];
        }
        init_csr(triplets);
    }

    friend std::ostream &operator<<(std::ostream &os, const CSR &csr) {
        for(const T& value: csr.rowPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: csr.colPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: csr.values) {
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

            for(int i = 0; i < lhs.values.size();i++) {
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
    std::vector<int> colPositions;
    std::vector<int> rowPositions;
    std::vector<T> values;
    int rows;
    int cols;

    void init_csr(std::vector<Triplet<T>> triplets) {
        assert(triplets.size() > 0);

        auto comp = [](const Triplet<T> t1, const Triplet<T> t2) -> bool {
            return t1.row < t2.row || (t1.row == t2.row && t1.col < t2.col);
        };

        std::sort(triplets.begin(), triplets.end(), comp);

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);

        this->colPositions.reserve(triplets.size());
        this->values.reserve(triplets.size());

        this->rowPositions.emplace_back(0);
        this->colPositions.push_back(triplets[0].col);
        this->values.push_back(triplets[0].value);

        for (size_t i = 1; i < triplets.size(); i++) {
            this->colPositions.emplace_back(triplets[i].col);
            if (triplets[i].row != triplets[i-1].row) {
                this->rowPositions.emplace_back(static_cast<int>(i));
            }

            this->values.push_back(triplets[i].value);
        }

        this->rowPositions.emplace_back(static_cast<int>(this->values.size()));
    }
};


#endif //DPHPC_CSR_H
