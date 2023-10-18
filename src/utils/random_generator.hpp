#pragma once

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <type_traits>

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

template <typename S>
std::ostream& operator<<(std::ostream& os, const std::vector<S>& vector) {
    for (auto element : vector) {
        os << element << "; ";
    }
    return os;
}