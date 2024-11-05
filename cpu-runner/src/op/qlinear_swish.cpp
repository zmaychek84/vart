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

#include "qlinear_swish.hpp"

#include <cmath>

namespace vart {
namespace cpu {

namespace {
float truncate_to_bfloat16(float inp) {
  int32_t temp;
  std::memcpy(&temp, &inp, sizeof(float));
  int32_t out = temp & 0xFFFF0000;
  float temp1;
  std::memcpy(&temp1, &out, sizeof(float));
  return temp1;
}

typedef union value_convert {
  std::uint32_t u;
  float f;
} value_convert_t;

std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

float f_to_bf(float data) {
  std::uint32_t u = f_to_u(data);
  std::uint32_t flag = (u & 0x00010000) >> 16;
  u = (u + 0x7fff + flag) & 0xFFFF0000;
  return u_to_f(u);
}

float approx_tanh(float inp1) {
  float inp = truncate_to_bfloat16(inp1);
  // cout<< inp1 <<endl;
  float out;
  if (inp >= 3.0)
    out = 1.0;
  else if (inp >= 2.0)
    out = 0.03125 * inp + 0.90625;
  else if (inp >= 1.75)
    out = 0.125 * inp + 0.71875;
  else if (inp >= 1.5)
    out = 0.125 * inp + 0.72265625;
  else if (inp >= 1.25)
    out = 0.25 * inp + 0.53515625;
  else if (inp >= 1.0)
    out = 0.25 * inp + 0.52734375;
  else if (inp >= 0.75)
    out = 0.5 * inp + 0.265625;
  else if (inp >= 0.625)
    out = 0.5 * inp + 0.25390625;
  else if (inp >= 0.5)
    out = 0.5 * inp + 0.25;
  else if (inp > -0.5)
    out = inp;
  else if (inp > -0.625)
    out = 0.5 * inp - 0.25;
  else if (inp > -0.75)
    out = 0.5 * inp - 0.25390625;
  else if (inp > -1.0)
    out = 0.5 * inp - 0.265625;
  else if (inp > -1.25)
    out = 0.25 * inp - 0.52734375;
  else if (inp > -1.5)
    out = 0.25 * inp - 0.53515625;
  else if (inp > -1.75)
    out = 0.125 * inp - 0.72265625;
  else if (inp > -2.0)
    out = 0.125 * inp - 0.71875;
  else if (inp > -3.0)
    out = 0.03125 * inp - 0.90625;
  else
    out = -1.0;
  return truncate_to_bfloat16(out);
}
}  // namespace

template <typename DType>
QlinearSwish<DType>::QlinearSwish(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : Eltwise<DType>(subg, op, inputs, output) {}

template <typename DType>
void QlinearSwish<DType>::eltwise(std::uint32_t start_index,
                                  std::uint32_t end_index) {
  float x_s = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  float y_s = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];
  std::int32_t x_zp =
      xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
  std::int32_t y_zp =
      xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
  float s_s = 1.0;
  std::int32_t s_zp = 0;
  auto pattern_type = xir_op_->template get_attr<std::string>("pattern_type");
  auto beta = xir_op_->template get_attr<float>("beta");
  if (pattern_type == "qsigmoid_mul") {
    s_s = xir_op_->template get_attr<std::vector<float>>("sigmoid_q_scale")[0];
    s_zp = xir_op_->template get_attr<std::vector<int32_t>>(
        "sigmoid_q_zero_point")[0];
  }
  auto input_tensor_ = CPUOPBase::xir_op_->get_input_tensor("input");
  // auto input_data_type_ = input_tensor_->get_data_type().type;
  auto input_data_bit_width_ = input_tensor_->get_data_type().bit_width;

  float sigmoid_lower = float(data_min_) - s_zp;
  float sigmoid_upper = float(data_max_) - s_zp;
  float y_lower = float(data_min_) - y_zp;
  float y_upper = float(data_max_) - y_zp;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    float d_in = float(data_in_[0][pos_iter] - x_zp) / x_s;
    float d_in_a16 = (data_in_[0][pos_iter] - x_zp) * f_to_bf(1 / x_s);
    // sigmoid
    float sigmoid = 0.0;
    if (input_data_bit_width_ == 16) {
      // sigmoid = (approx_tanh(d_in_a16 * 0.5f) + 1.0f) * 0.5f;
      sigmoid = (approx_tanh(d_in_a16 * beta * 0.5) + 1) * 0.5;

    } else if (input_data_bit_width_ == 8) {
      sigmoid = 1.0f / (exp(-d_in * beta) + 1.0f);
    }
    if (pattern_type == "qsigmoid_mul" && input_data_bit_width_ == 8) {
      sigmoid = sigmoid * (s_s) + s_zp;
      sigmoid = std::min(std::max(sigmoid, sigmoid_lower), sigmoid_upper);
      if ((sigmoid - floor(sigmoid)) == 0.5) {
        sigmoid = std::round(sigmoid * 0.5f) * 2.0f;  // round to even
      } else {
        sigmoid = round(sigmoid);
      }
      sigmoid = (sigmoid - s_zp) / (s_s);
    }
    // mul
    float res = sigmoid;
    if (pattern_type == "qsigmoid_mul" || pattern_type == "sigmoid_mul") {
      if (input_data_bit_width_ == 8)
        res = res * d_in;
      else
        res = res * d_in_a16;
    }
    if (input_data_bit_width_ == 8)
      res = res * y_s;
    else
      res = res * f_to_bf(y_s);
    // res = DPURound<DType>(res, y_lower, y_upper); //////////////////
    res = std::min(std::max(res, y_lower), y_upper);
    if ((res - floor(res)) == 0.5) {
      res = std::round(res * 0.5f) * 2.0f;  // round to even
    } else {
      res = round(res);
    }
    res = res + y_zp;

    // if ((res - floor(res)) == 0.5) {
    //   res = std::round(res * 0.5f) * 2.0f;  // round to even
    // } else {
    //   res = round(res);
    // }
    // data_out_[pos_iter] = DPURound<DType>(res, CPUOPBase::data_min_,
    // CPUOPBase::data_max_);
    data_out_[pos_iter] = static_cast<DType>(res);
  }
}

INSTANTIATE_TPCLASS(QlinearSwish);
REG_OP_INSTANCE_FUNC("qlinear-swish", QlinearSwish);

}  // namespace cpu
}  // namespace vart
