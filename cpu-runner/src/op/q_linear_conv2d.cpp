/*
 * Copyright 2022 Xilinx Inc.
 * Copyright 2023-2024 Advanced Micro Devices Inc.
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
#include "q_linear_conv2d.hpp"

#include <iostream>
#include <ostream>
#include <thread>
#include <vector>

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

std::int32_t get_shift_from_int32_rep(float in_f) {
  std::int32_t rep{0};
  std::memcpy(&rep, &in_f, sizeof in_f);
  rep &= 0x7fffffff;  // Remove sign bit
  return 127 - (((rep >> 23) & 255) + 1) + (8 * sizeof(std::int32_t) - 1);
}

int32_t float_to_int32(float value) {
  int32_t floatBits;
  std::memcpy(&floatBits, &value, sizeof(value));
  int32_t result = 0;
  result |= (floatBits & 0x80000000);
  if (result < 0) {
    result |= (~((1 << 30) | ((floatBits & 0x007FFFFF) << 7)) + 1);

  } else {
    result |= ((1 << 30) | ((floatBits & 0x007FFFFF) << 7));
  }
  return result;
}

// constructor and deconstructor
template <typename DType, typename WType>
QLinearConv2d<DType, WType>::QLinearConv2d(const xir::Subgraph* subg,
                                           const xir::Op* op, IMapTBs_t inputs,
                                           CPUTBPtr_t output)
    : Conv2d<DType, WType>(subg, op, inputs, output) {
  channel_ = fmap_o_.c;

  scale_input_ = xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
  zp_input_ =
      xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];

  scale_weights_ = xir_op_->template get_attr<std::vector<float>>("w_scale")[0];
  zp_weights_ =
      xir_op_->template get_attr<std::vector<int32_t>>("w_zero_point")[0];

  scale_output_ = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];
  zp_output_ =
      xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];

  auto nonlinear_type_str = xir_op_->template get_attr<string>("nonlinear");
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "")
    nonlinear_type_ = 0;
  else if (nonlinear_type_str == "RELU")
    nonlinear_type_ = 1;
  else if (nonlinear_type_str == "LEAKYRELU") {
    prelu_alpha_ = (double)xir_op_->template get_attr<float>("LEAKYRELU_alpha");
    nonlinear_type_ = 2;
  } else if (nonlinear_type_str == "PRELU") {
    nonlinear_type_ = 3;
    prelu_in_ = CPUOPBase::xir_op_->get_attr<int>("prelu_in");
    prelu_shift_ = CPUOPBase::xir_op_->get_attr<int>("prelu_shift");
    prelu_alpha_ = ((double)prelu_in_) / pow(2.0, prelu_shift_);
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT)
        << "Unsupported nonlinear type: " << nonlinear_type_str << ".";
  }
  tmp_data_out_ = std::vector<int64_t>(fmap_o_.num());

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::print_param() {
  ConvBase<DType, WType>::print_param();

  UNI_LOG_DEBUG_INFO << "fp_input: " << scale_input_ << endl;
  UNI_LOG_DEBUG_INFO << "zp_input: " << zp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_weights: " << scale_weights_ << endl;
  UNI_LOG_DEBUG_INFO << "zp_weights: " << zp_weights_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output: " << scale_output_ << endl;
  UNI_LOG_DEBUG_INFO << "zp_output_: " << zp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "data_min_: " << data_min_ << endl;
  UNI_LOG_DEBUG_INFO << "data_max_: " << data_max_ << endl;
  if (nonlinear_type_ == 2) {
    UNI_LOG_DEBUG_INFO << "leakyrelu_alpha: " << prelu_alpha_ << endl;
  }
  if (nonlinear_type_ == 3) {
    UNI_LOG_DEBUG_INFO << "prelu_in: " << prelu_in_ << endl;
    UNI_LOG_DEBUG_INFO << "prelu_shift: " << prelu_shift_ << endl;
  }
}

template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::check_param() {
  ConvBase<DType, WType>::check_param();
  UNI_LOG_CHECK(nonlinear_type_ == 0 || nonlinear_type_ == 1 ||
                    nonlinear_type_ == 2 || nonlinear_type_ == 3,
                VART_INVALID_VALUE)
      << " not supported. ";
}

template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::run() {
  // // do necessary check
  print_param();
  // check_param();

  // // read data
  auto* data_in_tb = inputs_.at(ITName[ConvBase<DType, WType>::INPUT]).at(0);
  auto* tb_input_ptr = GET_CPUTB_DType_PTR(DType, data_in_tb);
  FastPad(raw_fmap_i_, fmap_i_, pad_, dilation_)
      .transform<DType>(tb_input_ptr, data_in_ptr_, zp_input_);

  // calc conv
  this->qlinearconv2d_conv();

  // do fix
  this->fix();

  // // do save, debug...
  // this->save();
}

template <typename DType, typename WType>
std::vector<int32_t> QLinearConv2d<DType, WType>::calculate_ifm() {
  std::vector<int32_t> ifm(fmap_o_.num() / fmap_o_.c, 0);
  for (auto n = 0; n < fmap_o_.n; n++) {
    for (auto h = 0; h < fmap_o_.h; h++) {
      for (auto w = 0; w < fmap_o_.w; w++) {
        auto rlt_addr = n * fmap_o_.h * fmap_o_.w + h * fmap_o_.w + w;
        auto addr_base_i = n * fmap_i_.ncod() + h * stride_.h * fmap_i_.hcod() +
                           w * stride_.w * fmap_i_.wcod();
        for (auto hk = 0; hk < kernel_.h; ++hk) {
          for (auto wk = 0; wk < kernel_.w; ++wk) {
            auto* tmp_ptr_i = data_in_ptr_ + addr_base_i + hk * fmap_i_.hcod() +
                              wk * fmap_i_.wcod();
            for (auto ck = 0; ck < fmap_i_.c; ++ck) {
              ifm[rlt_addr] += tmp_ptr_i[ck];
            }
          }
        }
      }
    }
  }
  return ifm;
}

int32_t round_even(int64_t n, int shift, int32_t data_max, int32_t data_min) {
  int64_t mask = 1LL << (shift - 1);
  int32_t integer_part = static_cast<int32_t>(n >> shift);
  // int i = 0;
  // for (; std::abs(integer_part) >> i; i++)
  //   ;
  // auto int_shift = shift - (23 - i);
  if (n & mask) {  // >=0.5
    // int64_t fractional_mask = ((mask - 1) >> int_shift) << int_shift;
    int64_t fractional_mask = mask - 1;
    if (!(n & fractional_mask)) {  //==0.5
      if (integer_part % 2) {
        integer_part++;
      }
    } else {
      integer_part++;
    }
  }
  if (integer_part > data_max) integer_part = data_max;
  if (integer_part < data_min) integer_part = data_min;
  return integer_part;
}

int64_t calculate_matmul_shift(const int& a_bit_width, const int& w_bit_width,
                               int64_t K) {
  int64_t shift = 0;
  if (a_bit_width + w_bit_width == 24) {
    // 0 < K < 32768
    shift = std::min(std::max(24 + int(std::ceil(std::log2(K))) - 32, 0), 7);
  } else if (a_bit_width + w_bit_width > 24) {
    UNI_LOG_FATAL(VART_NOT_SUPPORT)
        << "Unsupported bit_width " << a_bit_width << " " << w_bit_width;
  }
  return shift;
}

template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::qlinearconv2d_conv() {
  vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          auto FMAP_SIZE = fmap_o_.num();
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            int pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;
            auto fmap = fmap_o_.pos2coord(pos);
            qlinearconv2d_conv_one(data_in_ptr_, weights_ptr_, fmap.n, fmap.h,
                                   fmap.w, fmap.c, 0, fmap_i_.c);
          }

          return 0;
        },
        i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
  // for (auto n = 0; n < fmap_o_.n; n++) {
  //   for (auto h = 0; h < fmap_o_.h; h++) {
  //     for (auto w = 0; w < fmap_o_.w; w++) {
  //       for (auto c = 0; c < fmap_o_.c; c++) {
  //         qlinearconv2d_conv_one(data_in_ptr_, weights_ptr_, n, h, w, c, 0,
  //                                fmap_i_.c);
  //       }
  //     }
  //   }
  // }
}
template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::qlinearconv2d_conv_one(
    DType* src, WType* wts, int idx_dst_n, int idx_dst_h, int idx_dst_w,
    int idx_oc, int32_t ic_begin, int32_t ic_end) {
  // assign conv value to data_out_ptr_ var
  auto rlt_addr = idx_dst_n * fmap_o_.ncod() + idx_dst_h * fmap_o_.hcod() +
                  idx_dst_w * fmap_o_.wcod() + idx_oc;

  // NOTE: special handling for conv-fix
  tmp_data_out_[rlt_addr] = 0;

  auto idx_src_h = idx_dst_h * stride_.h;
  auto idx_src_w = idx_dst_w * stride_.w;

  auto addr_base_i = idx_dst_n * fmap_i_.ncod() + idx_src_h * fmap_i_.hcod() +
                     idx_src_w * fmap_i_.wcod();
  auto addr_base_w = idx_oc * fmap_w_.ncod();

  // calculate conv value
  for (auto h = 0; h < kernel_.h; ++h) {
    for (auto w = 0; w < kernel_.w; ++w) {
      auto* tmp_ptr_i =
          src + addr_base_i + h * fmap_i_.hcod() + w * fmap_i_.wcod();
      auto* tmp_ptr_w =
          wts + addr_base_w + h * fmap_w_.hcod() + w * fmap_w_.wcod();
      for (auto c = ic_begin; c < ic_end; ++c) {
        tmp_data_out_[rlt_addr] += tmp_ptr_i[c] * tmp_ptr_w[c];
      }
    }
  }
}
template <typename DType, typename WType>
void QLinearConv2d<DType, WType>::fix() {
  // xcompiler
  int64_t za = zp_input_;
  int64_t zw = zp_weights_;
  auto& ch = channel_;
  int64_t K = fmap_w_.num() / channel_;
  std::vector<int32_t> wn(channel_, 0);
  for (auto i = 0; i < fmap_w_.num(); i++) {
    wn[i / K] += weights_ptr_[i];
  }

  auto f_cf2 = scale_output_ / (scale_input_ * scale_weights_);
  // calc cf2
  int64_t cf2 = float_to_int32(f_cf2);
  auto shift_out = get_shift_from_int32_rep(f_cf2);
  cf2 = cf2 >> 7;
  shift_out = shift_out - 7;
  if (-cf2 * zw < INT32_MIN || -cf2 * zw > INT32_MAX) {
    cf2 = cf2 >> 1;
    shift_out = shift_out - 1;
  }
  auto a_bit_width =
      xir_op_->get_input_tensor(ITName[ConvBase<DType, WType>::INPUT])
          ->get_data_type()
          .bit_width;
  auto w_bit_width =
      xir_op_->get_input_tensor(ITName[ConvBase<DType, WType>::WEIGHTS])
          ->get_data_type()
          .bit_width;
  auto shift_matmul = calculate_matmul_shift(a_bit_width, w_bit_width, K);
  cf2 = cf2 << shift_matmul;

  // use cf2
  int64_t cf4 = (cf2 * prelu_in_) * pow(2, -prelu_shift_);
  auto cf1 = -cf2 * zw;
  auto zo = int64_t(zp_output_) << shift_out;

  std::vector<int64_t> cf0(ch);
  std::vector<int64_t> cf3(ch);
  for (auto i = 0; i < ch; i++) {
    cf0[i] = cf2 * (-za * wn[i] + K * za * zw + bias_ptr_[i]) + zo;
    cf3[i] = cf2 * (-za * wn[i] + K * za * zw + bias_ptr_[i]) * prelu_in_ *
                 pow(2, -prelu_shift_) +
             zo;
  }

  UNI_LOG_DEBUG_INFO << "f_cf2 " << f_cf2 << " cf2 " << cf2;
  UNI_LOG_DEBUG_INFO << "cf1 " << cf1;
  UNI_LOG_DEBUG_INFO << "cf0 " << cf0;
  UNI_LOG_DEBUG_INFO << "shift_matmul " << shift_matmul;
  UNI_LOG_DEBUG_INFO << "shift_out " << shift_out;
  // std::ofstream("int64.bin",
  // std::ios::binary).write(reinterpret_cast<char*>(tmp_data_out_),
  // fmap_o_.num() * sizeof(int64_t));

  // kernel
  auto ifm = calculate_ifm();
  for (auto i = 0; i < fmap_o_.num(); i++) {
    int32_t tmp_data = tmp_data_out_[i];
    if (shift_matmul) {
      tmp_data =
          round_even(tmp_data_out_[i], shift_matmul, INT32_MAX, INT32_MIN);
    }
    int64_t tmp = cf2 * tmp_data + cf1 * ifm[i / ch] + cf0[i % ch] - zo;
    if (nonlinear_type_ == 1) {
      if (tmp < 0) tmp = 0;
      tmp = tmp + zo;
    } else if (nonlinear_type_ == 2 || nonlinear_type_ == 3) {
      int64_t tmp1 = cf4 * (tmp_data - zw * ifm[i / ch]) + cf3[i % ch];
      tmp = std::max(tmp + zo, tmp1);
    } else {
      tmp = tmp + zo;
    }
    data_out_ptr_[i] = round_even(tmp, shift_out, data_max_, data_min_);
  }
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(QLinearConv2d);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("qlinear-conv2d", QLinearConv2d,
                                                "weights");

}  // namespace cpu
}  // namespace vart
