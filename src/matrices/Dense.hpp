//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_DENSE_H
#define DPHPC_DENSE_H

#include <iostream>
#include <ostream>
#include <vector>
#include <assert.h>

template<class T>
class Dense {
public:
    
    Dense()
    : matrix(std::vector<T>())
    {}

    Dense(std::vector<std::vector<T>> &matrix_in) {
        assert(matrix_in.size() > 0);

        this->rows = matrix_in.size();
        this->cols = matrix_in[0].size();
        this->matrix.reserve(matrix_in.size() * matrix_in[0].size());

        for(const std::vector<T> &row : matrix_in) {
            std::copy(row.begin(), row.end(), std::back_inserter(this->matrix));
        }
    }

    friend std::ostream &operator<<(std::ostream &os, const Dense &dense) {
        for(int i = 0; i < dense.rows;i++) {
            for(int j = 0; j < dense.cols;j++) {
                std::cout << dense.matrix[i*dense.cols + j] << " ";
            }
            std::cout << "\n";
        }
        return os;
    }

    T& getValue(int x, int y) {
        return this->matrix[y * cols + x];
    }

    T* get_pointer() {
        return this->matrix.data();
    }

    unsigned int getRows() const {
        return this->rows;
    }

    unsigned int getCols() const {
        return this->cols;
    };


private:
    unsigned int rows;
    unsigned int cols;
    std::vector<T> matrix;
};



#endif //DPHPC_DENSE_H
