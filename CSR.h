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
    CSR(std::vector<std::pair<int,int>> &positions, std::vector<T> values);

    CSR(const CSR &other) {
        this->values = other.values;
        this->colPositions = other.colPositions;
        this->rowPositions = other.rowPositions;
    }

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

    void clearValues() {
        std::fill(this->values.begin(), this->values.end(), 0);
    }
private:
    std::vector<T> values;
public:
    const std::vector<T> &getValues();

    void setValues(const std::vector<T> &values);

    const std::vector<int> &getColPositions();

    void setColPositions(const std::vector<int> &colPositions);

    const std::vector<int> &getRowPositions();

    void setRowPositions(const std::vector<int> &rowPositions);

private:
    std::vector<int> colPositions;
    std::vector<int> rowPositions;
};

#endif //DPHPC_CSR_H
