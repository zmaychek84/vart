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

#include "normalization.hpp"

#include <algorithm>

#include "vitis/ai/env_config.hpp"
#include "vitis/ai/vart_plugin.hpp"
DEF_ENV_PARAM(USE_AIE_API, "0");

namespace vart {
namespace cpu {
typedef union value_convert {
  std::uint32_t u;
  std::int32_t i;
  float f;
} value_convert_t;

static inline std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

static inline float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}
static inline std::int32_t f_to_i(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.i;
}

static inline float i_to_f(std::int32_t data) {
  value_convert_t vc{};
  vc.i = data;
  return vc.f;
}

static inline int expo(float v) { return (f_to_i(v) >> 23) & 0xFF; }

static inline float rnd(float v, int m) {
  int sh = m - expo(v);
  if (sh <= 0) return v;
  float x = i_to_f((f_to_i(v) & (0x1FF << 23)) + (sh << 23));
  return (v + x) - x;
}

static inline float add32(float a, float b) {
  int m = std::max(expo(a), expo(b));
  return rnd(a, m) + rnd(b, m);
}

static inline float mac32(float a, float b, float c) {
  float d = (float)a * (float)b;
  int m = std::max(expo(a) + expo(b) - 127, expo(c));
  return rnd(c, m) + rnd(d, m);
}

static inline float addmac32(float a, float b, float c, float d) {
  float e = (float)a * (float)b;
  int m = std::max(std::max(expo(a) + expo(b) - 127, expo(c)), expo(d));
  return rnd(c, m) + rnd(d, m) + rnd(e, m);
}
//

Normalization::Normalization(const xir::Subgraph* subg, const xir::Op* op,
                             IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {}

float Normalization::aie_invsqrt(float val) {
  float ret = 0;
  if (ENV_PARAM(USE_AIE_API) == 1) {
    // don't use this api in windows,see #578
    auto plugin = vitis::ai::Plugin::get("invsqrt");
    if (plugin != nullptr) {
      ret = plugin->invoke<float, float>("model_invsqrt", val);
    }
  } else {
    ret = powf(val, -0.5);
  }
  return ret;
}

float Normalization::float2bfloat_cpu(float x) {
  std::uint32_t itmp = f_to_u(x);           // float32 bitwise to int32
  if ((itmp & 0x00008000) == 0x00008000) {  // half even
    if ((itmp & 0xFFFF) > 0x00008000 ||
        (((itmp & 0xFFFF) == 0x00008000) && (itmp & 0x10000) == 0x10000)) {
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return u_to_f(itmp);  // int32 bitwise to float32
}

float Normalization::msc32(float a, float b, float c) {
  float a0 = float2bfloat_cpu(a);
  float a1 = float2bfloat_cpu(a - (float)a0);
  float a2 = float2bfloat_cpu(a - (float)a0 - (float)a1);
  float b0 = float2bfloat_cpu(-b);
  float b1 = float2bfloat_cpu(-b - (float)b0);
  float b2 = float2bfloat_cpu(-b - (float)b0 - (float)b1);
  float ab_terms = mac32(
      a0, b1,
      mac32(
          a1, b0,
          mac32(a2, b0,
                mac32(a1, b1,
                      mac32(a0, b2, mac32(a1, b2, mac32(a2, b1, a2 * b2)))))));
  float result_tmp = mac32(a0, b0, ab_terms);
  float result = add32(result_tmp, c);
  return result;
}
static float uint16_to_bf(uint16_t u16, int fix_point) {
  float ret = static_cast<float>(u16);  // round to even
  return ret * pow(2, fix_point);
}
static float int16_to_bf(int16_t i16, int fix_point) {
  float ret = static_cast<float>(i16);  // round to even
  return ret * pow(2, fix_point);
}
float Normalization::aie_int64_to_float(int64_t i64) {
  float ret = 0;
  bool negtive = i64 < 0;
  if (i64 < 0) {
    i64 = i64 * -1;
  }
  ret += int16_to_bf((i64 >> 48) & 0xffff, 48);
  ret += uint16_to_bf((i64 >> 32) & 0xffff, 32);
  ret += uint16_to_bf((i64 >> 16) & 0xffff, 16);
  ret += uint16_to_bf((i64 >> 0) & 0xffff, 0);
  if (negtive) {
    ret *= -1;
  }
  return ret;
}
}  // namespace cpu
}  // namespace vart