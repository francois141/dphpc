#pragma once

#define SECOND(x) ((float)x / 1000000000)
#define MILLISECOND(x) ((float)x / 1000000)
#define MICROSECOND(x) ((float)x / 1000)

template <typename T>
std::vector<T> copy_pytorch_mat(const int64_t* data_ptr, int64_t count) {
    auto v = std::vector<uint64_t>(data_ptr, data_ptr+count);
    std::vector<T> result(count);
    std::transform(v.begin(), v.end(), result.begin(), [](uint64_t x) {return static_cast<T>(x); });
    return result;
}