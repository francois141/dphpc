//
// Created by francois on 04.10.23.
//

#include "CSR.h"

template<class T>
CSR<T>::CSR(int rows, int cols, std::vector<Triplet<T>> values) {
    assert(values.size() > 0);

    this->rows = rows;
    this->cols = cols;

    auto comp = [](const Triplet<T> t1, const Triplet<T> t2) -> bool {
        return t1.y < t2.y || (t1.y == t2.y && t1.x < t2.x);
    };

    sort(values.begin(), values.end(), comp);

    this->rowPositions = std::vector<int>(0);
    this->colPositions = std::vector<int>(0);
    this->values       = std::vector<T>(0);

    this->colPositions.reserve(values.size());
    this->values.reserve(values.size());

    this->rowPositions.emplace_back(0);
    this->colPositions.push_back(values[0].x);
    this->values.push_back(values[0].value);

    for(int i = 1; i < values.size();i++) {
        this->colPositions.emplace_back(values[i].x);
        if(values[i].y != values[i-1].y) {
            this->rowPositions.emplace_back(i);
        }

        this->values.push_back(values[i].value);
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

