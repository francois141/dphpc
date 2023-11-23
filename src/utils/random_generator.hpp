#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <type_traits>
#include <map>


// https://stackoverflow.com/a/69646192/8803949 & https://stackoverflow.com/a/32887614/8803949
template<typename T>
void generate_data(std::vector<T> &data, T min, T max) {
    static std::random_device rd;
    static std::default_random_engine generator(rd());

    using dist_t = std::conditional_t<std::is_integral<T>::value, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;
    static dist_t distribution(min, max);

    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
}

template<typename T>
std::vector<T> generate_data(size_t size, T min, T max) {
    std::vector<T> data(size);
    generate_data(data, min, max);
    return data;
}

template<typename T>
std::vector<T> generate_data(size_t size) {
    return generate_data(size, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

template<typename T>
using distributor = std::conditional_t<std::is_integral<T>::value, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;

template<typename T>
inline distributor<T> getDistributor(T range_from, T range_to) {
    return distributor<T>(range_from, range_to);
}

template<typename T>
inline T randomValue(distributor<T>& distribution, std::mt19937 &generator) {
    return distribution(generator);
}

// TODO: Francois Costa - if sampling is too slow, make a faster algorithm
template<typename T>
std::vector<Triplet<T>> sampleTriplets(const int M, const int N, const int nbSamples) {
    assert(M > 0 && N > 0);

    static std::random_device                  rand_dev;
    static std::mt19937                        generator(rand_dev());

    distributor<T> distributorValues = getDistributor<T>(0, 1);
    distributor<int> distributorRow = getDistributor<int>(0, M-1);
    distributor<int> distributorCol = getDistributor<int>(0, N-1);

    std::map<std::pair<int,int>, T> mp;
    while(mp.size() < (size_t)nbSamples) {
        const int row = randomValue<int>(distributorRow, generator);
        const int col = randomValue<int>(distributorCol, generator);

        assert(row <= M-1 && col <= N-1);

        mp[ std::make_pair(row, col)]= randomValue<T>(distributorValues, generator);
    }

    std::vector<Triplet<T>> triplets;
    triplets.reserve(nbSamples);

    for(auto &entry: mp) {
        triplets.push_back(Triplet<T>{entry.first.first, entry.first.second, entry.second});
    }

    return triplets;
}


template <typename S>
std::ostream& operator<<(std::ostream& os, const std::vector<S>& vector) {
    for (auto element : vector) {
        os << element << "; ";
    }
    return os;
}