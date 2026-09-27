#include "tensor.h"

#include <cstdio>
#include <cassert>
#include <vector>

namespace te {

namespace {
    
    void print_dims(const char* name, const int32_t* out, int32_t rank) {
        printf("%s=(", name);
        for (int d = 0; d < rank; d++) {
            printf("%d", out[d]);
            if (d == rank-1) {
                break;
            }
            printf(",");
        }
        printf(") ");
    }

}

void print(const Tensor& t) {

    print_dims("shape", t.shape, t.rank);
    print_dims("stride", t.stride, t.rank);
    printf("contiguous=%d\n", t.is_contiguous());
    
    
    for (int f = 0; f < t.numel(); f++) {
        int32_t coords[kMaxRank];
        flat_to_coords(f, t.shape, t.rank, coords);
        int64_t off = offset_of(coords, t.stride, t.rank);
        float v = t.data[off];
        std::printf("%g ", v);
    }
    printf("\n");

}

}

int main() {

    // one shared buffer holding 0..23
      std::vector<float> buf(24);
      for (int i = 0; i < 24; ++i) buf[i] = static_cast<float>(i);

      // --- case 1: packed rank-3 (2,3,4) ---
      te::Tensor t;
      t.data = buf.data();
      t.rank = 3;
      t.shape[0] = 2; t.shape[1] = 3; t.shape[2] = 4;
      te::compute_strides(t.shape, t.rank, t.stride);
      std::printf("### case 1: packed (2,3,4)\n");
      te::print(t);

      // --- case 2: transposed view, SAME buffer, no data moved ---
      std::printf("### case 2: t.transpose(1,2)\n");
      te::Tensor tr = t.transpose(1, 2);
      te::print(tr);

      // --- case 3: rank-1 vector ---
      std::printf("### case 3: rank-1 (6,)\n");
      te::Tensor v;
      v.data = buf.data();
      v.rank = 1;
      v.shape[0] = 6;
      te::compute_strides(v.shape, v.rank, v.stride);
      te::print(v);

      // --- case 4: non-contiguous slice -- row 1 of each block.
      // shape (2,4), strides (12,1), data advanced by 1*4 = 4
      std::printf("### case 4: slice t[:,1] -> (2,4)\n");
      te::Tensor s;
      s.data = buf.data() + 4;
      s.rank = 2;
      s.shape[0] = 2;   s.shape[1] = 4;
      s.stride[0] = 12; s.stride[1] = 1;
      te::print(s);

      // --- case 5: rank-4, all kMaxRank slots in use ---
      std::printf("### case 5: packed rank-4 (1,2,3,4)\n");
      te::Tensor q;
      q.data = buf.data();
      q.rank = 4;
      q.shape[0]=1; q.shape[1]=2; q.shape[2]=3; q.shape[3]=4;
      te::compute_strides(q.shape, q.rank, q.stride);
      te::print(q);
      return 0;

}