#pragma once

#include <cstdint>
#include <cassert>

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
    };

    bool is_contiguous() const {
        int acc = 1;
        for (int d = rank-1; d >= 0; d--) {
            if (acc != stride[d]) {
                return false;
            }
            acc *= shape[d];
        }

        return true;
    };

    Tensor transpose(int32_t a, int32_t b) const{

        assert(a >= 0 && a < rank && b >= 0 && b < rank);

        Tensor o = *this;

        int32_t temp = o.shape[a];
        o.shape[a] = o.shape[b];
        o.shape[b] = temp;

        temp = o.stride[a];
        o.stride[a] = o.stride[b];
        o.stride[b] = temp;

        return o;
    }

};

inline void compute_strides(const int32_t* shape, int rank, int32_t* out_stride) {assert(rank <= kMaxRank); int64_t acc = 1; for (int d = rank-1; d >= 0; d--) {out_stride[d] = acc; acc *= shape[d];}};

inline void flat_to_coords(int64_t flat, const int32_t* shape, int rank, int32_t* out_coords) {for (int d = rank-1; d >= 0; d--) {out_coords[d] = flat % shape[d]; flat /= shape[d];}};

inline int64_t offset_of(const int32_t* coords, const int32_t* stride, int rank) {int64_t res = 0; for (int d = rank-1; d >= 0; d--) { res += static_cast<int64_t>(coords[d]) * stride[d];} return res;};

void print(const Tensor& t);

}