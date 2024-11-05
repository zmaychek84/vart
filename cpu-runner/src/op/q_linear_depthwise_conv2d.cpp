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
#include "q_linear_depthwise_conv2d.hpp"

#include <ostream>

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
std::vector<int64_t> floats_to_int64s(const std::vector<float>& data) {
  std::vector<int64_t> result;
  for (auto value : data) {
    result.emplace_back(float_to_int32(value));
  }
  return result;
}
// constructor and deconstructor
template <typename DType, typename WType>
QLinearDWConv2d<DType, WType>::QLinearDWConv2d(const xir::Subgraph* subg,
                                               const xir::Op* op,
                                               IMapTBs_t inputs,
                                               CPUTBPtr_t output)
    : DWConv<DType, WType>(subg, op, inputs, output) {
  sa_ = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  sw_ = xir_op_->template get_attr<std::vector<float>>("w_scale")[0];
  sb_ = xir_op_->template get_attr<std::vector<float>>("b_scale")[0];
  so_ = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];

  za_ = xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
  zw_ = xir_op_->template get_attr<std::vector<int32_t>>("w_zero_point")[0];
  zb_ = xir_op_->template get_attr<std::vector<int32_t>>("b_zero_point")[0];
  zo_ = xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
}

template <typename DType, typename WType>
void QLinearDWConv2d<DType, WType>::print_param() {
  DWConvBase<DType, WType>::print_param();

  UNI_LOG_DEBUG_INFO << "fp_input: " << sa_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_weights: " << sw_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output: " << so_ << endl;
  UNI_LOG_DEBUG_INFO << "zo_: " << zo_ << endl;
}

template <typename DType, typename WType>
void QLinearDWConv2d<DType, WType>::check_param() {
  DWConvBase<DType, WType>::check_param();
}

template <typename DType, typename WType>
void QLinearDWConv2d<DType, WType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  auto* data_in_tb = inputs_.at(ITName[DWConvBase<DType, WType>::INPUT]).at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);
  FastPad(raw_fmap_i_, fmap_i_, pad_, dilation_)
      .transform<DType>(tb_input_ptr, data_in_ptr_, za_);

  // calc conv
  this->dwconv();

  // do fix
  this->fix();

  // // do save, debug...
  // this->save();
}
template <typename DType, typename WType>
std::vector<int64_t> QLinearDWConv2d<DType, WType>::calculate_cf2(
    std::vector<float> cf_float) {
  auto int32_cf = floats_to_int64s(cf_float);
  UNI_LOG_DEBUG_INFO << "coeffs2 float to int " << int32_cf ;

  std::vector<int> coeffs0_shift_b;
  auto calc_shift_b = [](int v) {
    v = std::abs(v);
    int cont = 0;
    while (v > 0) {
      v = v >> 1;
      cont++;
    }
    return cont;
  };
  for (auto value : cf_float) {
    coeffs0_shift_b.push_back(calc_shift_b(value));
  }
  UNI_LOG_DEBUG_INFO << " coeffs0_shift_b " << coeffs0_shift_b;
  auto max_coeffs0 =
      *std::max_element(coeffs0_shift_b.begin(), coeffs0_shift_b.end());
  std::vector<int64_t> cf;
  for (size_t i = 0; i < int32_cf.size(); i++) {
    int shift = max_coeffs0 - coeffs0_shift_b[i];
    if (shift_b_ < 0)
      cf.push_back((int32_cf[i] >> shift) << (-shift_b_));
    else
      cf.push_back((int32_cf[i] >> shift) >> shift_b_);
  }
  return cf;
}

template <typename DType, typename WType>
std::vector<DType> QLinearDWConv2d<DType, WType>::calculate_ifm() {
  std::vector<DType> ifm(fmap_o_.num(), 0);
  for (auto idx_dst_n = 0; idx_dst_n < fmap_o_.n; idx_dst_n++) {
    for (auto idx_dst_h = 0; idx_dst_h < fmap_o_.h; idx_dst_h++) {
      for (auto idx_dst_w = 0; idx_dst_w < fmap_o_.w; idx_dst_w++) {
        vector<DType> result(fmap_o_.c, 0);
        int idx_src_n = idx_dst_n;
        int idx_src_h = idx_dst_h * stride_.h;
        int idx_src_w = idx_dst_w * stride_.w;

        for (int c = 0; c < fmap_w_.c; c++) {
          for (auto n = 0; n < fmap_w_.n; n++) {
            auto rlt_pos = c * fmap_w_.n + n;

            for (int h = 0; h < fmap_w_.h; h++) {
              for (int w = 0; w < fmap_w_.w; w++) {
                auto img_addr = idx_src_n * fmap_i_.h * fmap_i_.w * fmap_i_.c +
                                (idx_src_h + h) * fmap_i_.w * fmap_i_.c +
                                (idx_src_w + w) * fmap_i_.c + c;

                result[rlt_pos] += data_in_ptr_[img_addr];
              }
            }
          }
        }

        auto rlt_addr = idx_dst_n * fmap_o_.h * fmap_o_.w * fmap_o_.c +
                        idx_dst_h * fmap_o_.w * fmap_o_.c +
                        idx_dst_w * fmap_o_.c;
        for (auto i = 0; i < fmap_o_.c; i++) {
          ifm[rlt_addr + i] = result[i];
        }
      }
    }
  }
  return ifm;
}

template <typename DType, typename WType>
std::vector<WType> QLinearDWConv2d<DType, WType>::calculate_wgt() {
  auto c = fmap_w_.c;
  vector<WType> wgt(c, 0);
  for (auto i = 0; i < fmap_w_.num(); i++) {
    wgt[i % c] += weights_ptr_[i];
  }
  return wgt;
}

template <typename DType, typename WType>
void QLinearDWConv2d<DType, WType>::fix() {
  auto f_cf0 = so_ / (sa_ * sw_);
  int64_t cf0 = float_to_int32(f_cf0);
  shift_out_ = get_shift_from_int32_rep(f_cf0);

  auto wgt = calculate_wgt();
  float cfw = -1.0f * za_ * so_ / sw_ / sa_;
  float cfb = so_ / sb_;
  float cfo = /*za_ * zw_ * so_ / sa_ / sw_ - zb_ * so_ / sb_ +*/ zo_;
  std::vector<float> f_cf2(fmap_b_.num());
  for (auto i = 0u; i < f_cf2.size(); i++) {
    f_cf2[i] = cfw * wgt[i] + cfb * bias_ptr_[i] + cfo;
  }
  auto max_f_cf2 =
      std::max(*std::max_element(f_cf2.begin(), f_cf2.end()),
               std::abs(*std::min_element(f_cf2.begin(), f_cf2.end())));
  auto c2_shift = get_shift_from_int32_rep(max_f_cf2);
  shift_b_ = c2_shift - shift_out_;
  auto cf2 = calculate_cf2(f_cf2);

  UNI_LOG_DEBUG_INFO << "coeffs0 " << f_cf0 << "\n" << cf0;
  UNI_LOG_DEBUG_INFO << "coeffs2 " << f_cf2 << "\n" << cf2;
  UNI_LOG_DEBUG_INFO << "shift_b_" << shift_b_ << " shift_out_ " << shift_out_;
  for (auto i = 0; i < fmap_o_.num(); i++) {
    int64_t new_value =
        cf0 * data_out_ptr_[i] + /*cf1 * ifm[i] +*/ cf2[i % cf2.size()];
    data_out_ptr_[i] = round_even(new_value, shift_out_, data_max_, data_min_);
  }
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(QLinearDWConv2d);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("qlinear-depthwise-conv2d",
                                                QLinearDWConv2d, "weights");

}  // namespace cpu
}  // namespace vart
