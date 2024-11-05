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

#include "qlinear_groupnorm.hpp"

#include <algorithm>
namespace vart {
namespace cpu {

template <typename DType>
QGroupNorm<DType>::QGroupNorm(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : Normalization(subg, op, inputs, output) {
  // Ensure
  auto input_bit_width_ =
      xir_op_->get_input_tensor("input", 0)->get_data_type().bit_width;
  UNI_LOG_CHECK(input_bit_width_ != 32, VART_NOT_SUPPORT)
      << "qlinear-instnorm not support int32 output now.";

  UNI_LOG_CHECK(xir_op_->has_attr("eps"), VART_NOT_FOUND)
      << "attr `eps` is required";
  epsilon_ = xir_op_->get_attr<float>("eps");
  group_size = xir_op_->get_attr<int>("num_groups");
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
  UNI_LOG_CHECK(input_shape[3] % group_size == 0, VART_SIZE_ERROR)
      << "group should be divisible by the number of channels so that there "
         "are an equal number of channels per group";
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
    } else if (act_type_str == "SILU") {
      nonlinear_type = NonLinearType::Silu;
      output_scale = float2bfloat_cpu(output_scale/2);
    } else {
      UNI_LOG_CHECK(false, VART_NOT_SUPPORT)
          << " instancenorm nonlinear attr" << " only support NONE/RELU/RELU6."
          << " But the " << get_name() << " nonlinear type is " << act_type_str;
    }
  }

  auto input_tensor = this->xir_op_->get_input_tensor("input",0);
  input_bit_width = input_tensor->get_data_type().bit_width;
}

template <typename DType>
void QGroupNorm<DType>::run() {
  // do necessary check
  // print_param();
  // this->check_param();

  // // read data
  // this->read();
  // process
  calculate_mean_var();
  do_norm();
  // // do save, debug...
  // this->save();
}
static int64_t calculate_mean_square_sum_shift(const int& a_bit_width, int64_t K) {
  return std::max(a_bit_width + int(std::ceil(std::log2(K))) - 32, 0);
}

int32_t round_even(int64_t n, int shift, int32_t data_max, int32_t data_min);

template <typename DType>
void QGroupNorm<DType>::calculate_mean_var() {
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
  std::vector<float> group_mean_sum;
  std::vector<float> group_mean_square_sum;
  for (int i = 0; i < input_shape[3]; i += group_size) {
    float sum = 0;
    float sum2 = 0;
    for (int j = 0; j < group_size; j++) {
      sum += float(mean_sum[i + j]);
      sum2 += float(mean_square_sum[i + j]);
    }
    for (int j = 0; j < group_size; j++) {
      group_mean_sum.push_back(float(sum));
      group_mean_square_sum.push_back(float(sum2));
    }
  }

  float inv_size_sum = (float)(pow(2.0f, shift_mean_sum) /
                  (input_shape[1] * input_shape[2] * group_size));
  float inv_size_square_sum = (float)(pow(2.0f, shift_mean_square_sum) /
                  (input_shape[1] * input_shape[2] * group_size));
  for (auto c_dim = 0; c_dim < input_shape[3]; c_dim++) {
    float mean = group_mean_sum[c_dim] * inv_size_sum;
    means_.push_back(float2bfloat_cpu(mean));

    // compute var
    float Ex2 = group_mean_square_sum[c_dim] * inv_size_square_sum;
    variances_.push_back(msc32(means_[c_dim], means_[c_dim], Ex2));
  }
}

template <typename DType>
float QGroupNorm<DType>::approx_tanh(float inp1) {
  float inp = float2bfloat_cpu(inp1);
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
  return float2bfloat_cpu(out);
}
template <typename DType>
void QGroupNorm<DType>::do_norm() {
  for (int c_dim = 0; c_dim < input_shape[3]; c_dim++) {
    float scale = *(input_weights_ptr_ + c_dim) - weight_zero_point;
    scale = float2bfloat_cpu(scale / weight_scale);
    float bias = *(input_bias_ptr_ + c_dim) - bias_zero_point;
    bias = float2bfloat_cpu(bias / bias_scale);
    float mean = means_[c_dim];
    float variance = variances_[c_dim];
    float inv_stdev = float2bfloat_cpu(
        aie_invsqrt(variance + float2bfloat_cpu(epsilon_)));  // invsqrt
    const float channel_scale = float2bfloat_cpu(inv_stdev * scale);
    const float channel_shift =
        float2bfloat_cpu(bias - float2bfloat_cpu(mean * channel_scale));

    for (int h_dim = 0; h_dim < input_shape[1]; h_dim++) {
      for (int w_dim = 0; w_dim < input_shape[2]; w_dim++) {
        auto offset = h_dim * input_shape[2] * input_shape[3] +
                      w_dim * input_shape[3] + c_dim;
        float input = (*(input_input_ptr_ + offset)) - input_zero_point;
        input = float2bfloat_cpu(input);
        float Y_float = msc32(channel_scale, -1 * input, channel_shift);
        if (nonlinear_type == NonLinearType::Silu){
          float tanh_rst = approx_tanh(float2bfloat_cpu(Y_float / 2));
          Y_float = msc32(Y_float, -1 * tanh_rst, Y_float);
        }
        float Y_scaled = float2bfloat_cpu(float2bfloat_cpu(Y_float) *
                                          float2bfloat_cpu(output_scale));

        int32_t Y = round_to_even<int32_t>(
            Y_scaled, CPUOPBase::data_min_ - output_zero_point,
            CPUOPBase::data_max_ - output_zero_point);
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
          case NonLinearType::Silu:
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
template <typename DType>
void QGroupNorm<DType>::read() {
  input_input_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(0));
  input_weights_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("weights").at(0));
  input_bias_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("bias").at(0));
  output_output_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

INSTANTIATE_TPCLASS(QGroupNorm);
REG_OP_INSTANCE_FUNC("qlinear-groupnorm", QGroupNorm);

}  // namespace cpu
}  // namespace vart
