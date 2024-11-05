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
#include "q_linear_tconv2d.hpp"

#include "fast_pad.hpp"

namespace vart {
namespace cpu {
template <typename T>
static std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
  int index = 0;
  s << "[";
  for (auto c : v) {
    if (index++ != 0) {
      s << ",";
    }
    s << c;
  }
  s << "]";
  return s;
}

int32_t float_to_int32(float value);
int32_t round_even(int64_t n, int shift, int32_t data_max, int32_t data_min);
std::int32_t get_shift_from_int32_rep(float in_f);

// constructor and deconstructor
template <typename DType, typename WType>
QLinearTConv2d<DType, WType>::QLinearTConv2d(const xir::Subgraph* subg,
                                             const xir::Op* op,
                                             IMapTBs_t inputs,
                                             CPUTBPtr_t output)
    : TConv2d<DType, WType>(subg, op, inputs, output) {
  sa_ = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  sw_ = xir_op_->template get_attr<std::vector<float>>("w_scale")[0];
  so_ = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];

  za_ = xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
  zw_ = xir_op_->template get_attr<std::vector<int32_t>>("w_zero_point")[0];
  zo_ = xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
}

template <typename DType, typename WType>
void QLinearTConv2d<DType, WType>::print_param() {
  TConv2d<DType, WType>::print_param();

  UNI_LOG_DEBUG_INFO << "fp_input: " << sa_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_weights: " << sw_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output: " << so_ << endl;
  UNI_LOG_DEBUG_INFO << "zo_: " << zo_ << endl;
}

template <typename DType, typename WType>
void QLinearTConv2d<DType, WType>::check_param() {
  TConv2d<DType, WType>::check_param();
  CHECK(zw_ == 0) << "only support 2term";
}

template <typename DType, typename WType>
void QLinearTConv2d<DType, WType>::run() {
  // // do necessary check
  print_param();
  check_param();

  // // read data
  auto* data_in_tb = inputs_.at(ITName[TConv2d<DType, WType>::INPUT]).at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);
  FastPad(raw_fmap_i_, fmap_i_, pad_, dilation_)
      .transform<DType>(tb_input_ptr, data_in_ptr_, za_);

  // calc conv
  this->conv();

  // do fix
  this->fix();

  // // do save, debug...
  // this->save();
}

template <typename DType, typename WType>
void QLinearTConv2d<DType, WType>::fix() {
  auto f_cf1 = so_ / (sa_ * sw_);
  int64_t cf1 = float_to_int32(f_cf1);
  auto shift_out = get_shift_from_int32_rep(f_cf1);
  auto ch = fmap_o_.c;
  auto K = fmap_w_.num() / ch;
  vector<int32_t> wgt(ch, 0);
  for (auto i = 0; i < fmap_w_.num(); i++) {
    wgt[i / K] += weights_ptr_[i];
  }
  std::vector<int64_t> cf0(ch);
  for (auto i = 0u; i < ch; i++) {
    cf0[i] = cf1 * (-za_ * wgt[i] + bias_ptr_[i]) + (int64_t(zo_) << shift_out);
  }
  UNI_LOG_DEBUG_INFO << "f_cf1 " << f_cf1 << " cf1 " << cf1;
  UNI_LOG_DEBUG_INFO << "cf0 " << cf0;
  UNI_LOG_DEBUG_INFO << "shift_out " << shift_out;
  for (auto i = 0; i < fmap_o_.num(); i++) {
    int64_t tmp = cf1 * data_out_ptr_[i] + cf0[i % ch];
    data_out_ptr_[i] = round_even(tmp, shift_out, data_max_, data_min_);
  }
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(QLinearTConv2d);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("qlinear-transposed-conv2d",
                                                QLinearTConv2d, "weights");

}  // namespace cpu
}  // namespace vart
