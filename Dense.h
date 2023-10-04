//
// Created by francois on 04.10.23.
//

#ifndef DPHPC_DENSE_H
#define DPHPC_DENSE_H

#include <iostream>
#include <ostream>
#include <vector>

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

private:
    std::vector<std::vector<T>> matrix;
};



#endif //DPHPC_DENSE_H
