//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_COO_H
#define DPHPC_COO_H

#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>

#include "triplet.h"
#include "CSR.hpp"

template<class T>
class CSR;

template<class T>
class COO {
public:

    COO() {
        this->rows = 0;
        this->cols = 0;

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);
    }
    
    COO(int rows, int cols, std::vector<Triplet<T>> values) {
        assert(values.size() > 0);

        this->rows = rows;
        this->cols = cols;

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);

        this->colPositions.reserve(values.size());
        this->rowPositions.reserve(values.size());
        this->values.reserve(values.size());

        for (const Triplet<T> &triplet: values) {
            this->colPositions.push_back(triplet.x);
            this->rowPositions.push_back(triplet.y);
            this->values.push_back(triplet.value);
        }
    }

    COO(const COO &other) : rows(other.rows), cols(other.cols), values(other.values), colPositions(other.colPositions), rowPositions(other.rowPositions) {}

    COO(CSR<T> &csr) {
        this->rows = csr.getRows();
        this->cols = csr.getCols();

        this->rowPositions = std::vector<int>(0);
        this->colPositions = std::vector<int>(0);
        this->values       = std::vector<T>(0);

        this->colPositions.reserve(csr.getValues().size());
        this->rowPositions.reserve(csr.getValues().size());
        this->values.reserve(csr.getValues().size());

        for (int i = 0; i < csr.getRows(); i++) {
            for (int j = csr.getRowPositions()[i]; j < csr.getRowPositions()[i+1];j++) {
                this->rowPositions.push_back(i);
                this->colPositions.push_back(csr.getColPositions()[j]);
                this->values.push_back(csr.getValues()[j]);
            }
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const COO &COO) {
        for(const T& value: COO.rowPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: COO.colPositions) {
            std::cout << value << " ";
        }
        std::cout << std::endl;

        for(const T& value: COO.values) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
        
        return os;
    }

    friend bool operator==(const COO &lhs, const COO &rhs) {
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

private:
    std::vector<int> colPositions;
    std::vector<int> rowPositions;
    std::vector<T> values;
    int rows;
    int cols;
};


#endif //DPHPC_COO_H
