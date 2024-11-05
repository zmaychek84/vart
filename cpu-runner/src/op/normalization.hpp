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

#include "cpu_op_base.hpp"

namespace vart {
namespace cpu {
/**
 * base class of qlinear-instnorm instnorm-fix qgroupnorm
 *
 */
enum class NonLinearType {
  None = 0,
  Relu,
  Relu6,
  Silu,
};
class Normalization : public CPUOPBase {
 public:
  Normalization(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output);
  ~Normalization() = default;

  virtual void run() = 0;

  virtual void read() = 0;

 protected:
  float aie_invsqrt(float);
  float float2bfloat_cpu(float);
  float msc32(float a, float b, float c);
  float aie_int64_to_float(int64_t i64);
};

}  // namespace cpu
}  // namespace vart