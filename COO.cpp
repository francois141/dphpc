//
// Created by francois on 04.10.23.
//

#include "COO.h"

template<class T>
COO<T>::COO(int rows, int cols, std::vector<Triplet<T>> values) {
    assert(values.size() > 0);

    this->rows = rows;
    this->cols = cols;

    this->rowPositions = std::vector<int>(0);
    this->colPositions = std::vector<int>(0);
    this->values       = std::vector<T>(0);

    this->colPositions.reserve(values.size());
    this->rowPositions.reserve(values.size());
    this->values.reserve(values.size());

    for(const Triplet<T> &triplet: values) {
        this->colPositions.push_back(triplet.x);
        this->rowPositions.push_back(triplet.y);
        this->values.push_back(triplet.value);
    }
}


template<class T>
void COO<T>::setValues(const std::vector<T> &values) {
    COO::values = values;
}

template<class T>
const std::vector<int> &COO<T>::getColPositions() {
    return colPositions;
}

template<class T>
void COO<T>::setColPositions(const std::vector<int> &colPositions) {
    COO::colPositions = colPositions;
}

template<class T>
const std::vector<int> &COO<T>::getRowPositions() {
    return rowPositions;
}

template<class T>
void COO<T>::setRowPositions(const std::vector<int> &rowPositions) {
    COO::rowPositions = rowPositions;
}

template<class T>
const std::vector<T> &COO<T>::getValues() {
    return values;
}

