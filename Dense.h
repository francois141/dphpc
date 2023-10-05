//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_DENSE_H
#define DPHPC_DENSE_H

#include <iostream>
#include <ostream>
#include <vector>

#define BOUNDS_CHECK

template<class T>
class Dense {
public:
    Dense(std::vector<std::vector<T>> &matrix);

    friend std::ostream &operator<<(std::ostream &os, const Dense &dense) {
        for(const std::vector<T> &v : dense.matrix) {
            for(const T &value: v) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }
        return os;
    }

    T getValue(int x, int y) {
#ifdef BOUNDS_CHECK
        if(x < 0 || x >= matrix.size() || y < 0 || y >= matrix.size()) {
            std::cerr << "Invalid index access " << x << " " << y << std::endl;
            exit(0);
        }
#endif
        return matrix[x][y];
    }

    unsigned int getRows() const {
        return this->matrix.size();
    }

    unsigned int getCols() const {
#ifdef BOUNDS_CHECK
        if(matrix.empty()) {
            return 0;
        }
#endif
        return this->matrix[0].size();
    };


private:
    std::vector<std::vector<T>> matrix;
};



#endif //DPHPC_DENSE_H
