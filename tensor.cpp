#include "tensor.h"

#include <cstdio>

namespace te {

namespace {
    
}

bool Tensor::is_contiguous() const {
    return false;
}

void compute_strides(const int32_t* shape, int rank, int32_t* out_stride) {
    (void)shape; (void)rank; (void)out_stride;
}

void flat_to_coords(int64_t flat, const int32_t* shape, int rank,
                    int32_t* out_coords) {
    (void)flat; (void)shape; (void)rank; (void)out_coords;
}

int64_t offset_of(const int32_t* coords, const int32_t* stride, int rank) {
    (void)coords; (void)stride; (void) rank;
    return 0;
}

void print(const Tensor& t) {
    std::printf("Tensor(rank=%d, numel=%lld)\n", t.rank,
                static_cast<long long>(t.numel()))
}

}