/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
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

#include "qlinear_expand.hpp"

#include <iostream>
#include <vector>

namespace vart {
namespace cpu {

int32_t round_even(int64_t n, int shift, int32_t data_max, int32_t data_min);

template <typename DType>
const vector<string> QlinearExpand<DType>::ITName = {"input", "shape"};

// constructor and deconstructor
template <typename DType>
QlinearExpand<DType>::QlinearExpand(const xir::Subgraph* subg,
                                    const xir::Op* op, IMapTBs_t inputs,
                                    CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  // expand_size = xir_op_->template get_attr<vector<std::int32_t>>("size");
  // output_f_.resize(fmap_o_.num());
}

template <typename DType>
void QlinearExpand<DType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  read();

  // do reshape
  q_expand();

  // // do save, debug...
  // save();
}

template <typename DType>
void QlinearExpand<DType>::print_param() {
  fmap_i_.print_param("fmap_i");
  fmap_o_.print_param("fmap_o");
}

// read data from DDR
template <typename DType>
void QlinearExpand<DType>::read() {
  // read img data
  auto* cputb = inputs_.at(ITName[INPUT]).at(0);

  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t QlinearExpand<DType>::get_workload() {
  // TODO
  return 0;
}

template <typename DType>
void QlinearExpand<DType>::q_expand() {
  if (xir_op_->has_attr("c0_int") && xir_op_->has_attr("c1_int") &&
      xir_op_->has_attr("shift_c0") && xir_op_->has_attr("shift_c1")) {
    C_0 = xir_op_->template get_attr<int32_t>("c0_int");
    C_1 = xir_op_->template get_attr<int32_t>("c1_int");

    FP_0 = xir_op_->template get_attr<int32_t>("shift_c0");
    FP_1 = xir_op_->template get_attr<int32_t>("shift_c1");
    LOG(INFO) << "QDQ info : " << FP_0 << " " << C_0 << " " << FP_1 << " "
              << C_1;
  } else if (xir_op_->has_attr("x_zero_point") &&
             xir_op_->has_attr("y_zero_point") &&
             xir_op_->has_attr("x_scale") && xir_op_->has_attr("y_scale")) {
    std::int32_t x_zp =
        xir_op_->template get_attr<std::vector<int32_t>>("x_zero_point")[0];
    float x_s =
        xir_op_->template get_attr<std::vector<float>>("x_scale").front();

    std::int32_t y_zp =
        xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
    float y_s = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];

    float c0_f = y_s / x_s;
    std::int32_t c0_qint = convert_float_to_qint(c0_f);
    FP_0 = get_shift_from_int32_rep(c0_qint);
    C_0 = get_int32_representation_from_float(c0_f);

    float c1_f = y_zp - x_zp * y_s / x_s;
    std::int32_t c1_qint = convert_float_to_qint(c1_f);
    FP_1 = get_shift_from_int32_rep(c1_qint);
    C_1 = get_int32_representation_from_float(c1_f);
    LOG(INFO) << "QDQ info : " << FP_0 << " " << C_0 << " " << FP_1 << " "
              << C_1;
  }

  for (int n = 0; n < fmap_o_.n; ++n) {
    for (int c = 0; c < fmap_o_.c; ++c) {
      for (int h = 0; h < fmap_o_.h; ++h) {
        for (int w = 0; w < fmap_o_.w; ++w) {
          int input_idx = (n % fmap_i_.n) * fmap_i_.ncod() +
                          (c % fmap_i_.c) * fmap_i_.ccod() +
                          (h % fmap_i_.h) * fmap_i_.hcod() +
                          (w % fmap_i_.w) * fmap_i_.wcod();
          int output_idx = n * fmap_o_.ncod() + c * fmap_o_.ccod() +
                           h * fmap_o_.hcod() + w * fmap_o_.wcod();
          // output_f_[output_idx] =
          //     (data_in_ptr_[input_idx] - x_zp) / x_s * y_s + y_zp;
          // data_out_ptr_[output_idx] = output_f_[output_idx];

          // output_f_[output_idx] =
          std::int64_t tmp_o;
          if (FP_0 > FP_1)
            tmp_o = (static_cast<std::int64_t>(data_in_ptr_[input_idx]) *
                     static_cast<std::int64_t>(C_0)) +
                    (static_cast<std::int64_t>(C_1) << (FP_0 - FP_1));
          else
            tmp_o = (static_cast<std::int64_t>(data_in_ptr_[input_idx]) *
                     static_cast<std::int64_t>(C_0)) +
                    (static_cast<std::int64_t>(C_1) >> (FP_0 - FP_1));
          data_out_ptr_[output_idx] = round_even(
              static_cast<std::int64_t>(tmp_o), FP_0, data_max_, data_min_);
        }
      }
    }
  }
}

INSTANTIATE_TPCLASS(QlinearExpand);
REG_OP_INSTANCE_FUNC("qlinear-expand", QlinearExpand);
// REG_OP_INSTANCE_FUNC("qlinear-expand-fix", QlinearExpand);

}  // namespace cpu
}  // namespace vart
