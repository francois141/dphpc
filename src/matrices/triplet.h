#ifndef DPHPC_TRIPLET_H
#define DPHPC_TRIPLET_H

template <typename T>
struct Triplet {
    int row;
    int col;
    T value;
};


template <typename S>
std::ostream& operator<<(std::ostream& os, const Triplet<S>& triplet) {
    os << triplet.row << " " << triplet.col << " " << triplet.value;
    return os;
}
 

#endif