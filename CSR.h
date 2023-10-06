//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_CSR_H
#define DPHPC_CSR_H

#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>

template<class T>
class CSR {
public:
    CSR(int rows, int cols, std::vector<std::pair<int,int>> positions, std::vector<T> values);

    CSR(const CSR &other) : rows(other.rows), cols(other.cols), values(other.values), colPositions(other.colPositions), rowPositions(other.rowPositions) {}

    friend std::ostream &operator<<(std::ostream &os, const CSR &csr) {
        for(const T& value: csr.values) {
            std::cout << value << " ";
        }
        std::cout << "\n";

        for(const T& value: csr.colPositions) {
            std::cout << value << " ";
        }
        std::cout << "\n";

        for(const T& value: csr.rowPositions) {
            std::cout << value << " ";
        }
        return os;
    }

    friend bool operator==(const CSR &lhs, const CSR &rhs) {
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

#endif //DPHPC_CSR_H
