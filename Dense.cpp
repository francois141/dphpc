//
// Created by francois on 04.10.23.
//

#include "Dense.h"

template<typename T>
Dense<T>::Dense(std::vector<std::vector<T>> &matrix) {
    this->matrix = matrix;
}
