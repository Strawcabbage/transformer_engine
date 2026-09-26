#include "tensor.h"

#include <cstdio>
#include <cassert>

namespace te {

namespace {
    
}

bool Tensor::is_contiguous() const {
    
    int acc = 1;
    for (int d = rank-1; d >= 0; d--) {
        if (acc != stride[d]) {
            return false;
        }
        acc *= shape[d];
    }

    return true;
}

void compute_strides(const int32_t* shape, int rank, int32_t* out_stride) {
    assert(rank <= kMaxRank);

    int acc = 1;
    for (int d = rank-1; d >= 0; d--) {
        out_stride[d] = acc;
        acc *= shape[d];
    }
}

void flat_to_coords(int64_t flat, const int32_t* shape, int rank,
                    int32_t* out_coords) {
    
    for (int d = rank-1; d >= 0; d--) {
        out_coords[d] = flat % shape[d];
        flat /= shape[d];
    }
}

int64_t offset_of(const int32_t* coords, const int32_t* stride, int rank) {
    
    int64_t res = 0;
    for (int d = rank-1; d >= 0; d--) {
        res += static_cast<int64_t>(coords[d]) * stride[d];
    }

    return res;

}

void print(const Tensor& t) {
    std::printf("Tensor(rank=%d, numel=%lld)\n", t.rank,
                static_cast<long long>(t.numel()));
}

}

int main() {

}