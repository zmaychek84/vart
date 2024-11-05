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
class QuantizeLinear : public CPUOPBase {
 public:
  enum InputTensorType {
    INPUT,
    SCALE,
    ZERO_POINT,
  };
  const static std::vector<std::string> ITName;

 public:
  QuantizeLinear(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                 CPUTBPtr_t output);
  virtual ~QuantizeLinear();

  virtual void run() override final;

  virtual void read() override final;

 private:
  inline void float2fix_dpu_round(std::int32_t i) {
    auto div_result = input_input_ptr[i] / scale[0];
    auto zp = (zero_point.size() == 0 ? 0 : zero_point[0]);


    float lower_bound_f = float(data_min_)-zp;
    float upper_bound_f = float(data_max_)-zp;

    div_result = std::max(div_result,lower_bound_f);
    div_result = std::min(div_result,upper_bound_f);

    if ((div_result - floor(div_result)) == 0.5){
      div_result = std::round(div_result * 0.5f) * 2.0f;//round to even
    } else {
      div_result = round(div_result);
    }

    auto tmp = div_result + zp;
    output_output_ptr[i] = static_cast<FixType>(tmp);

  }

 private:
  using CPUOPBase::data_max_;
  using CPUOPBase::data_min_;
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
