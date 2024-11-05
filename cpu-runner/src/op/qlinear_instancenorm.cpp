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

#include "qlinear_instancenorm.hpp"
#include <algorithm>
#include "vitis/ai/vart_plugin.hpp"
#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(USE_AIE_API, "0");

static inline float aie_invsqrt(float val){
  float ret = 0;
  if (ENV_PARAM(USE_AIE_API) ==1){
    // don't use this api in windows,see #578
    auto plugin = vitis::ai::Plugin::get("invsqrt");
    if (plugin!=nullptr){
        ret = plugin->invoke<float,float>("model_invsqrt",val);
    }
  }else{
    ret = powf(val,-0.5);
  }
  return ret;
}

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
static inline float float2bfloat_cpu(const float x) {
  std::uint32_t itmp = f_to_u(x);          // float32 bitwise to int32
  if ((itmp & 0x00008000) == 0x00008000) { // half even
    if ((itmp & 0xFFFF) > 0x00008000 ||
        (((itmp & 0xFFFF) == 0x00008000) && (itmp & 0x10000) == 0x10000)) {
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return u_to_f(itmp); // int32 bitwise to float32
}

static inline int expo(float v) { return (f_to_i(v) >> 23) & 0xFF; }

static inline float rnd(float v, int m) {
  int sh = m - expo(v);
  if ( sh <= 0 ) return v;
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

// msc32 = c - a * b
static float msc32(float a, float b, float c) {
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
          mac32(
              a2, b0,
              mac32(
                  a1, b1,
                  mac32(
                      a0, b2, mac32(a1, b2, mac32(a2, b1, a2 * b2))
                  )
              )
          )
      )
  );
  float result_tmp = mac32(a0, b0, ab_terms);
  float result = add32(result_tmp, c);
  return result;
}

static int64_t calculate_mean_square_sum_shift(const int& a_bit_width, int64_t K) {
  return std::max(a_bit_width + int(std::ceil(std::log2(K))) - 32, 0);
}

int32_t round_even(int64_t n, int shift, int32_t data_max, int32_t data_min);
template <typename DType>
QinstanceNorm<DType>::QinstanceNorm(const xir::Subgraph *subg,
                                          const xir::Op *op, IMapTBs_t inputs,
                                          CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  // Ensure
  auto input_bit_width_ = xir_op_->get_input_tensor("input",0)->get_data_type().bit_width;
  UNI_LOG_CHECK(input_bit_width_ != 32, VART_NOT_SUPPORT)
        << "qlinear-instnorm not support int32 output now.";
      
  UNI_LOG_CHECK(xir_op_->has_attr("eps"), VART_NOT_FOUND)
      << "attr `eps` is required";
  epsilon_ = xir_op_->get_attr<float>("eps");
  input_zero_point =
      xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
  output_zero_point =
      xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
  weight_zero_point =
      xir_op_->template get_attr<std::vector<int32_t>>("w_zero_point")[0];
  bias_zero_point =
      xir_op_->template get_attr<std::vector<int32_t>>("b_zero_point")[0];
  input_scale = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  output_scale = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];
  weight_scale = xir_op_->template get_attr<std::vector<float>>("w_scale")[0];
  bias_scale = xir_op_->template get_attr<std::vector<float>>("b_scale")[0];

  GET_INPUT_DIMX_FMAP(input_shape, input, 4);
  UNI_LOG_CHECK(input_shape.size() == 4, VART_SIZE_ERROR)
      << "input shape must be NHWC";
  UNI_LOG_CHECK(input_shape[0] == 1, VART_SIZE_ERROR)
      << "input_shape[0] must be 0";
  epsilon_ = float2bfloat_cpu(epsilon_ / input_scale / input_scale);
  nonlinear_type = NonLinearType::None;
  if (op->has_attr("nonlinear")) {
    auto act_type_str = op->get_attr<std::string>("nonlinear");
    if (act_type_str == "NONE") {
      nonlinear_type = NonLinearType::None;
    } else if (act_type_str == "RELU") {
      nonlinear_type = NonLinearType::Relu;
    } else if (act_type_str == "RELU6") {
      nonlinear_type = NonLinearType::Relu6;
    } else {
      UNI_LOG_CHECK(false, VART_NOT_SUPPORT)
          << " instancenorm nonlinear attr"
          << " only support NONE/RELU/RELU6."
          << " But the " << get_name() << " nonlinear type is " << act_type_str;
    }
  }

  auto input_tensor = this->xir_op_->get_input_tensor("input",0);
  input_bit_width = input_tensor->get_data_type().bit_width;
}

template <typename DType> void QinstanceNorm<DType>::run() {
  // do necessary check
  // print_param();
  // this->check_param();

  // // read data
  // this->read();
  // process
  calculate_mean_var();
  inst_norm();

  // // do save, debug...
  // this->save();
}

template <typename DType> void QinstanceNorm<DType>::calculate_mean_var() {
  means_.clear();
  variances_.clear();
  std::vector<int32_t> mean_sum;
  std::vector<int32_t> mean_square_sum;
  int shift_mean_sum = calculate_mean_square_sum_shift(
      input_bit_width, input_shape[1] * input_shape[2]);
  int shift_mean_square_sum = calculate_mean_square_sum_shift(
      input_bit_width * 2, input_shape[1] * input_shape[2]);
  for (auto c_dim = 0; c_dim < input_shape[3]; c_dim++) {
    int64_t sum = 0;
    int64_t sum2 = 0;
    for (auto h_dim = 0; h_dim < input_shape[1]; h_dim++) {
      for (auto w_dim = 0; w_dim < input_shape[2]; w_dim++) {
        auto offset = h_dim * input_shape[2] * input_shape[3] +
                      w_dim * input_shape[3] + c_dim;
        int64_t input = *(input_input_ptr_ + offset) - input_zero_point;
        sum += input;
        sum2 += input * input;
      }
    }
    mean_sum.push_back(round_even(sum, shift_mean_sum, INT32_MAX, INT32_MIN));
    mean_square_sum.push_back(round_even(sum2, shift_mean_square_sum, INT32_MAX, INT32_MIN));
  }
  float inv_size_sum = (float)(pow(2.0f, shift_mean_sum) /
                  (input_shape[1] * input_shape[2]));
  float inv_size_square_sum = (float)(pow(2.0f, shift_mean_square_sum) /
                  (input_shape[1] * input_shape[2]));
  for (auto c_dim = 0; c_dim < input_shape[3]; c_dim++) {
    float mean = mean_sum[c_dim] * inv_size_sum;
    means_.push_back(float2bfloat_cpu(mean));

    // compute var
    float Ex2 = mean_square_sum[c_dim] * inv_size_square_sum;
    variances_.push_back(msc32(means_[c_dim], means_[c_dim], Ex2));
  }
}

template <typename DType> void QinstanceNorm<DType>::inst_norm() {
  for (int c_dim = 0; c_dim < input_shape[3]; c_dim++) {
    float scale = *(input_weights_ptr_ + c_dim) - weight_zero_point;
    scale = float2bfloat_cpu(scale / weight_scale);
    float bias = *(input_bias_ptr_ + c_dim) - bias_zero_point;
    bias = float2bfloat_cpu(bias / bias_scale);
    float mean = means_[c_dim];
    float variance = variances_[c_dim];
    float inv_stdev = float2bfloat_cpu(aie_invsqrt(variance + float2bfloat_cpu(epsilon_)));  // invsqrt
    const float channel_scale = float2bfloat_cpu(inv_stdev * scale);
    const float channel_shift =
        float2bfloat_cpu(bias - float2bfloat_cpu(mean * channel_scale));
    for (int h_dim = 0; h_dim < input_shape[1]; h_dim++) {
      for (int w_dim = 0; w_dim < input_shape[2]; w_dim++) {
        auto offset = h_dim * input_shape[2] * input_shape[3] +
                      w_dim * input_shape[3] + c_dim;
        float input = (*(input_input_ptr_ + offset)) - input_zero_point;
        input = float2bfloat_cpu(input);

        float Y_float = msc32(channel_scale,-1*input,channel_shift);

        float Y_scaled = float2bfloat_cpu(float2bfloat_cpu(Y_float) * float2bfloat_cpu(output_scale));

        int32_t Y =
            round_to_even<int32_t>(Y_scaled, CPUOPBase::data_min_- output_zero_point,
                                   CPUOPBase::data_max_- output_zero_point);
        int32_t result = static_cast<int32_t>(Y + output_zero_point);

        switch (nonlinear_type) {
        case NonLinearType::None:
          break;
        case NonLinearType::Relu:
          result = std::max(int32_t(0), result);
          break;
        case NonLinearType::Relu6:
          result = std::max(int32_t(0), result);
          result = std::min(result, int32_t(96));
          break;
        default:
          throw "not support now";
          break;
        }

        *(output_output_ptr_ + offset) = result;
      }
    }
  }
}

template <typename DType> void QinstanceNorm<DType>::read() {
  input_input_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(0));
  input_weights_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("weights").at(0));
  input_bias_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("bias").at(0));
  output_output_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

INSTANTIATE_TPCLASS(QinstanceNorm);
REG_OP_INSTANCE_FUNC("qlinear-instancenorm", QinstanceNorm);

} // namespace cpu
} // namespace vart