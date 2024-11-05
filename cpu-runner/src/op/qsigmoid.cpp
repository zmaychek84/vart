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

#include <cmath>

#include "qsigmoid.hpp"

namespace vart {
namespace cpu {

template <typename DType>
Qsigmoid<DType>::Qsigmoid(const xir::Subgraph* subg, const xir::Op* op,
                          IMapTBs_t inputs, CPUTBPtr_t output)
    : Eltwise<DType>(subg, op, inputs, output) {
  auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);

  x_s = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  y_s = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];
  x_zp = xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
  y_zp = xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
}

template <typename DType>
void Qsigmoid<DType>::print_param() {
  fmap_i_[0].print_param("fmap_i");
  fmap_o_.print_param("fmap_o");
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
    UNI_LOG_DEBUG_INFO << "x_scale = " << x_s << endl;
    UNI_LOG_DEBUG_INFO << "y_scale = " << y_s << endl;
    UNI_LOG_DEBUG_INFO << "x_zero_point = " << x_zp << endl;
    UNI_LOG_DEBUG_INFO << "y_zero_point = " << y_zp << endl;
  }
}

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
void Qsigmoid<DType>::eltwise(std::uint32_t start_index,
                              std::uint32_t end_index) {
  auto input_tensor_ = CPUOPBase::xir_op_->get_input_tensor("input");
  auto input_data_bit_width_ = input_tensor_->get_data_type().bit_width;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    float tmp;
    if (input_data_bit_width_ == 8) {
      float y_low = float(CPUOPBase::data_min_) - y_zp;
      float y_up = float(CPUOPBase::data_max_) - y_zp;
      tmp = y_s * 1 / (exp((x_zp - (float)data_in_[0][pos_iter]) / x_s) + 1);
      tmp = round_to_even<float>(tmp, y_low, y_up);
      tmp += y_zp;
    } else {
      // sigmoid(x)=tanh(x/2)/2
      // inv_scale_in
      float sxx = f_to_bf(0.5f / x_s);
      // scale_out
      float syy = f_to_bf(0.5f * y_s);
      // DQ
      float xf = (float)(data_in_[0][pos_iter] - x_zp);
      xf = f_to_bf(xf) * (sxx);
      // Tanh(x)+1
      float out = approx_tanh(xf) + f_to_bf(1.0f);
      // Q
      tmp = round_to_even<int32_t>(f_to_bf((f_to_bf(out) * syy))) + y_zp;
    }
    // rnd and saturate
    data_out_[pos_iter] =
        round_to_even<DType>(tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

INSTANTIATE_TPCLASS(Qsigmoid);
REG_OP_INSTANCE_FUNC("qlinear-sigmoid", Qsigmoid);

}  // namespace cpu
}  // namespace vart
