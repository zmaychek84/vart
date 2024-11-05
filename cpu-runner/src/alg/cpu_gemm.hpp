/*
 * Copyright (C) 2022 Xilinx, Inc.
 * Copyright (C) 2023 – 2024 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// #include <immintrin.h>
#include "cpu_std_inc.hpp"
#include "cpu_types.hpp"

namespace vart {
namespace cpu {

template <typename T>
inline void inner_product(const T* A, const T* B, T* C, int64_t K) {
  *C = 0;
  for (auto k = 0; k < K; k++) {
    *C += A[k] * B[k];
  }
}
template <typename T, typename D>
inline void inner_product(const T* A, const D* B, T* C, int64_t K) {
  *C = 0;
  for (auto k = 0; k < K; k++) {
    *C += A[k] * B[k];
  }
}

template <typename T>
inline void eltwise_product(const T* A, const T* B, T* C, int64_t K) {
  for (auto k = 0; k < K; k++) {
    C[k] = A[k] * B[k];
  }
}

template <typename T>
inline void group_product(const T* A, const T* B, T* C, int64_t K, int64_t group) {
  int64_t group_size = K / group;

  int64_t k = 0;
  for (auto i = 0; i < group; i++) {
    C[i] = 0;
    for (auto j = 0; j < group_size; j++) {
      C[i] += A[k] * B[k];
      k++;
    }
  }
}

template <typename T>
inline void inner_product_with_kernel_stride(const T* A, const T* B, T* C,
                                             int64_t fmap_slice, int64_t kernel_slice,
                                             Kernel_t kernel) {
  *C = 0;
  for (auto i = 0; i < kernel.h; i++) {
    auto* addrA = A + i * fmap_slice;
    auto* addrB = B + i * kernel_slice;
    for (auto j = 0; j < kernel_slice; j++) {
      *C += addrA[j] * addrB[j];
    }
  }
}

template <typename T>
inline void inner_product_with_kernel_stride(const T* A, const T* B, T* C,
                                             FMap_t fmap_i, FMap_t fmap_w,
                                             Kernel_t kernel) {
  inner_product_with_kernel_stride<T, T>(A, B, C, fmap_i, fmap_w, kernel);
}

template <typename T, typename D>
inline void inner_product_with_kernel_stride(const T* A, const D* B, T* C,
                                             const FMap_t& fmap_i,
                                             const FMap_t& fmap_w,
                                             Kernel_t kernel) {
  *C = 0;
  for (auto h = 0; h < kernel.h; h++) {
    for (auto w = 0; w < kernel.w; w++) {
      for (auto d = 0; d < kernel.d; d++) {
        auto* addrA =
            A + h * fmap_i.hcod() + w * fmap_i.wcod() + d * fmap_i.dcod();
        auto* addrB =
            B + h * fmap_w.hcod() + w * fmap_w.wcod() + d * fmap_w.dcod();
        for (auto c = 0; c < fmap_w.c; c++) {
          *C += addrA[c] * addrB[c];
        }
      }
    }
  }
}

static uint32_t __float_as_uint(float x) {
  uint32_t uint_x;
  memcpy(&uint_x, &x, sizeof(float));
  return uint_x;
}

static uint32_t GetExponentCPU(float v) {
  // Get the biased exponent.
  uint32_t uint_v = __float_as_uint(v);
  // Shift away mantissa bits.
  return (uint_v & 0x7f800000) >> 23;
}

// Get a unsinged value of the max biased exponent.
static uint32_t GetMaxExponentCPU(const float* input, int n) {
  uint32_t max_exp = 0;
  for (int i = 0; i < n; i++) {
    max_exp = std::max(max_exp, GetExponentCPU(input[i]));
  }
  return max_exp;
}

static float py3_round(float x) {
  float x_floor = std::floor(x);
  float diff = x - x_floor;
  if (diff > 0.5)
    return x_floor + 1;
  else if (diff == 0.5)
    return (int)x_floor % 2 == 1 || (int)x_floor % 2 == -1 ? x_floor + 1
                                                           : x_floor;
  else
    return x_floor;
}

template <typename T, typename D>
inline void transform_fp(const T* ifm, const D* wgt, T* C, float *res) {
  int ifm_exp = GetMaxExponentCPU((float *)ifm, 8);
  int wgt_exp = GetMaxExponentCPU((float *)wgt, 8);
  int add_exp = ifm_exp + wgt_exp - 127;
  int acc_exp = GetExponentCPU(*(float *)C);
  int m_bits = 23;
  if (acc_exp < add_exp && (add_exp - 127 - m_bits) > -128) {
    float scale = std::pow(2.0, add_exp - 127 - m_bits);
    *(float *)C = py3_round(*(float *)C / scale) * scale;
  } else if (acc_exp > add_exp && (acc_exp - 127 - m_bits) > -128){
    float scale = std::pow(2.0, acc_exp - 127 - m_bits);
    *res = py3_round(*res / scale) * scale;
  }
}

template <>
inline void inner_product_with_kernel_stride<float, float>(const float* A, const float* B, float* C,
                                             const FMap_t& fmap_i,
                                             const FMap_t& fmap_w,
                                             Kernel_t kernel) {
  *C = 0;
  int icp = 8;
  int icg =  fmap_w.c / icp;
  if (!icg) {
      icg = 1;
      icp = fmap_w.c;
  }
  for (auto cg = 0; cg < icg; cg++) {
  for (auto h = 0; h < kernel.h; h++) {
    for (auto w = 0; w < kernel.w; w++) {
      for (auto d = 0; d < kernel.d; d++) {
        auto* addrA = A + h * fmap_i.hcod() + w * fmap_i.wcod() + d * fmap_i.dcod();;
        auto* addrB = B + h * fmap_w.hcod() + w * fmap_w.wcod() + d * fmap_w.dcod();
        float res = .0f;
        for (auto c = 0; c < icp; c++) {
          res += addrA[cg*8+c] * addrB[cg*8+c];
        } 
        transform_fp(addrA + cg * 8, addrB + cg * 8, C, &res);
        *C = *C + res;
        }
      }
    }
  }
}

/*
template<>
inline void inner_product<int32_t>(const int32_t* A, const int32_t* B, int32_t*
C, int K) { auto ALIGNMENT = 64; auto parallel = 16; auto ngroups = K /
parallel; auto left = K % parallel; auto pad = 0; if(left != 0) { ngroups += 1;
    pad = parallel - left;
  }
  auto DATA_SIZE = ngroups * parallel * sizeof(int32_t);

  // A's tmp buffer
  auto sizeA = DATA_SIZE + ALIGNMENT;
  char* bufA = new char[sizeA];
  void* ptrA = bufA;
  size_t spaceA = sizeA;
  std::align(ALIGNMENT, DATA_SIZE, ptrA, spaceA);
  std::copy_n(A, K, (int32_t*)ptrA);
  if(pad != 0) {
    std::fill_n((int32_t*)ptrA + K, pad, 0);
  }

  // B's tmp buffer
  auto sizeB = DATA_SIZE + ALIGNMENT;
  char* bufB = new char[sizeB];
  void* ptrB = bufB;
  size_t spaceB = sizeB;
  std::align(ALIGNMENT, DATA_SIZE, ptrB, spaceB);
  std::copy_n(B, K, (int32_t*)ptrB);
  if(pad != 0) {
    std::fill_n((int32_t*)ptrB + K, pad, 0);
  }

  // C's tmp buffer
  alignas(64) int32_t bufC[16];

  __m512i acc = _mm512_setzero_epi32();

  for(auto i=0; i<ngroups; i++) {
    const auto* tmpA = (int32_t*)ptrA + i*parallel;
    const auto* tmpB = (int32_t*)ptrB + i*parallel;

    __m512i a = _mm512_load_epi32(tmpA);
    __m512i b = _mm512_load_epi32(tmpB);

    __m512i c = _mm512_mullo_epi32(a, b);

    acc = _mm512_add_epi32(c, acc);
  }

  _mm512_store_epi32(bufC, acc);

  *C = 0;
  for(auto i=0; i<parallel; i++) {
    *C += bufC[i];
  }

  delete []bufA;
  delete []bufB;
}
//*/

// K is the axis of matrix multiply
// A dimention: [X, K]
// B dimention: [Y, K]
// C dimention: [X, Y]
template <typename T>
void matmul(const T* A, const T* B, T* C, int64_t X, int64_t Y, int64_t K) {
  matmul<T, T>(A, B, C, X, Y, K);
}

template <typename T, typename D>
void matmul(const T* A, const D* B, T* C, int64_t X, int64_t Y, int64_t K) {
  for (auto x = 0; x < X; x++) {
    for (auto y = 0; y < Y; y++) {
      const auto* addrA = A + x * K;
      const auto* addrB = B + y * K;

      auto* addrC = C + x * Y + y;

      inner_product(addrA, addrB, addrC, K);
    }
  }
}

template <typename T>
void matmul_thread(const T* A, const T* B, T* C, int64_t X, int64_t Y, int64_t K) {
  matmul_thread<T, T>(A, B, C, X, Y, K);
}

template <typename T, typename D>
void matmul_thread(const T* A, const D* B, T* C, int64_t X, int64_t Y, int64_t K) {
  int THREAD_NUM = CPU_NUM;
  int64_t SIZE = X * Y;
  int64_t THREAD_WORKLOAD = ceil((float)SIZE / THREAD_NUM);

  vector<std::future<int>> fut(THREAD_NUM);

  for (auto i = 0; i < THREAD_NUM; i++) {
    fut[i] =
        std::async(std::launch::async,
                   [A, B, C, X, Y, K, SIZE, THREAD_WORKLOAD](decltype(i) i) {
                     int64_t BASE_POS = i * THREAD_WORKLOAD;
                     for (auto j = 0; j < THREAD_WORKLOAD; j++) {
                       int64_t pos = BASE_POS + j;
                       if (pos >= SIZE) break;

                       int64_t x = pos / Y;
                       int64_t y = pos % Y;
                       auto* addrA = A + x * K;
                       auto* addrB = B + y * K;
                       auto* addrC = C + pos;
                       inner_product(addrA, addrB, addrC, K);
                     }

                     return 0;
                   },
                   i);
  }

  for (auto i = 0; i < THREAD_NUM; i++) {
    fut[i].wait();
  }
}

}  // namespace cpu
}  // namespace vart
