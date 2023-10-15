#ifndef DPHPC_TRIPLET_H
#define DPHPC_TRIPLET_H

template <typename T>
struct Triplet {
    int x;
    int y;
    T value;
};


template <typename S>
std::ostream& operator<<(std::ostream& os, const Triplet<S>& triplet) {
    os << triplet.x << " " << triplet.y << " " << triplet.value;
    return os;
}
 

#endif