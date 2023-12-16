#ifndef DPHPC_TRIPLET_H
#define DPHPC_TRIPLET_H

template <typename T>
struct Triplet {
    int row;
    int col;
    T value;
};

template<typename T>
bool operator==(const Triplet<T> &triplet1, const Triplet<T> triplet2) {
    return triplet1.row == triplet2.row && triplet1.col == triplet2.col && triplet1.value && triplet2.value;
}

template<typename T>
bool operator<(const Triplet<T> &triplet1, const Triplet<T> triplet2) {
    return triplet1.row < triplet2.row || (triplet1.row == triplet2.row && triplet1.col < triplet2.col);
}

template <typename S>
std::ostream& operator<<(std::ostream& os, const Triplet<S>& triplet) {
    os << triplet.row << " " << triplet.col << " " << triplet.value;
    return os;
}
 

#endif