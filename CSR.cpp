//
// Created by francois on 04.10.23.
//

#include "CSR.h"

// TODO: Check bounds and size > 1 or don't execute everything in the code
template<class T>
CSR<T>::CSR(int rows, int cols, std::vector<std::pair<int,int>> positions, std::vector<T> values) {
    this->values = values;
    this->rows = rows;
    this->cols = cols;

    auto comp = [](const std::pair<int,int> pos1, const std::pair<int,int> pos2) -> bool {
        return pos1.first < pos2.first || (pos1.first == pos2.first && pos1.second < pos2.second);
    };

    // TODO: Not optimal, we sort input positions and this is a pass by reference
    sort(positions.begin(), positions.end(), comp);

    this->rowPositions = std::vector<T>(0);
    this->colPositions = std::vector<T>(0);
    this->colPositions.reserve(this->values.size());

    this->rowPositions.emplace_back(0);
    this->colPositions.push_back(positions[0].second);

    for(int i = 1; i < positions.size();i++) {
        const std::pair<int,int> &entry = positions[i];
        this->colPositions.emplace_back(entry.second);
        if(entry.first != positions[i-1].first) {
            this->rowPositions.emplace_back(i);
        }
    }

    this->rowPositions.emplace_back(this->values.size());
}


template<class T>
void CSR<T>::setValues(const std::vector<T> &values) {
    CSR::values = values;
}

template<class T>
const std::vector<int> &CSR<T>::getColPositions() {
    return colPositions;
}

template<class T>
void CSR<T>::setColPositions(const std::vector<int> &colPositions) {
    CSR::colPositions = colPositions;
}

template<class T>
const std::vector<int> &CSR<T>::getRowPositions() {
    return rowPositions;
}

template<class T>
void CSR<T>::setRowPositions(const std::vector<int> &rowPositions) {
    CSR::rowPositions = rowPositions;
}

template<class T>
const std::vector<T> &CSR<T>::getValues() {
    return values;
}

