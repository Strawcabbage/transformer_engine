#pragma once

#include <cstdint>

namespace te {

constexpr int kMaxRank = 4;

struct Tensor {
    float* data = nullptr;

    int32_t shape[kMaxRank] = {0, 0, 0, 0};
    int32_t stride[kMaxRank] = {0, 0, 0, 0};
    int32_t rank = 0;

    int64_t numel() const {
        int64_t n = 1;
        for (int d = 0; d < rank; ++d) n *= shape[d];
        return n;
    }

    bool is_contiguous() const;

};

void compute_strides(const int32_t* shape, int rank, int32_t* out_stride);

void flat_to_coords(int64_t flat, const int32_t* shape, int rank,
                    int32_t* out_coords);

int64_t offset_of(const int32_t* coords, const int32_t* stride, int rank);

void print(const Tensor& t);

}