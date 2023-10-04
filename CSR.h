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
private:
    std::vector<T> values;
    std::vector<int> colPositions;
    std::vector<int> rowPositions;
};

#endif //DPHPC_CSR_H
