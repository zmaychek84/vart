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
template <typename DType, typename FixType>
class DequantizeLinear : public CPUOPBase {
 public:
  enum InputTensorType {
    INPUT,
    SCALE,
    ZERO_POINT,
  };
  const static std::vector<std::string> ITName;

 public:
  DequantizeLinear(const xir::Subgraph* subg, const xir::Op* op,
                   IMapTBs_t inputs, CPUTBPtr_t output);
  ~DequantizeLinear() = default;

  virtual void run() override final;

  virtual void read() override final;

 private:
  inline void float2fix_dpu_round(std::int32_t i) {
    output_output_ptr[i] =
        (input_input_ptr[i] - (zero_point.size() == 0 ? 0 : zero_point[0])) *
        scale[0];
  }

 private:
  bool has_zero_point;
  DType* input_input_ptr;
  FixType* output_output_ptr;
  size_t input_len;

 private:
  // attr
  std::int32_t axis;
  std::string data_type;
  std::vector<float> scale;
  std::vector<std::int32_t> zero_point;
};

}  // namespace cpu
}  // namespace vart
