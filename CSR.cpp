//
// Created by francois on 04.10.23.
//

#include "CSR.h"

// TODO: Check bounds and size > 1 or don't execute everything in the code
template<class T>
CSR<T>::CSR(std::vector<std::pair<int,int>> &positions, std::vector<T> values) {
    this->values = values;

    auto comp = [](const std::pair<int,int> pos1, const std::pair<int,int> pos2) -> bool {
        return pos1.second < pos2.second || (pos1.second == pos2.second && pos1.first < pos2.second);
    };

    // TODO: Not optimal, we sort input positions and this is a pass by reference
    sort(positions.begin(), positions.end(), comp);

    this->rowPositions = std::vector<T>(0);
    this->colPositions = std::vector<T>(0);
    this->colPositions.reserve(this->values.size());

    this->rowPositions.emplace_back(positions[0].first);
    this->colPositions.push_back(positions[0].second);

    for(int i = 1; i < positions.size();i++) {
        const std::pair<int,int> &entry = positions[i];
        this->colPositions.emplace_back(entry.second);
        if(entry.first != this->rowPositions.back()) {
            this->rowPositions.emplace_back(entry.first);
        }
    }
}
