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

template<class T>
class COO {
public:
    COO(int rows, int cols, std::vector<Triplet<T>> values);

    COO(const COO &other) : rows(other.rows), cols(other.cols), values(other.values), colPositions(other.colPositions), rowPositions(other.rowPositions) {}

    friend std::ostream &operator<<(std::ostream &os, const COO &COO) {
        for(const T& value: COO.values) {
            std::cout << value << " ";
        }
        std::cout << "\n";

        for(const T& value: COO.colPositions) {
            std::cout << value << " ";
        }
        std::cout << "\n";

        for(const T& value: COO.rowPositions) {
            std::cout << value << " ";
        }
        return os;
    }

    friend bool operator==(const COO &lhs, const COO &rhs) {
        return lhs.colPositions == rhs.colPositions &&
               lhs.rowPositions == rhs.rowPositions &&
               lhs.values == rhs.values &&
               lhs.rows == rhs.rows &&
               lhs.cols == rhs.cols;
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

    const std::vector<T> &getValues();

    void setValues(const std::vector<T> &values);

    const std::vector<int> &getColPositions();

    void setColPositions(const std::vector<int> &colPositions);

    const std::vector<int> &getRowPositions();

    void setRowPositions(const std::vector<int> &rowPositions);

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
