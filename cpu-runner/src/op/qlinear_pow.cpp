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

#include "qlinear_pow.hpp"

#include <cmath>
#include <vector>

namespace vart {
namespace cpu {
template <typename DType>
const vector<string> QlinearPow<DType>::ITName = {"input", "exponent"};

template <typename DType>
QlinearPow<DType>::QlinearPow(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAPS(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);
}

template <typename DType>
void QlinearPow<DType>::read() {
  // handle input buffer

  auto* cputb = inputs_.at(ITName[INPUT]).at(0);
  data_in_ = GET_CPUTB_DType_PTR(DType, cputb);

  auto* exponent_tb = inputs_.at(ITName[EXPONENT]).at(0);
  exponent_ = GET_CPUTB_DType_PTR(DType, exponent_tb);

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
void QlinearPow<DType>::run() {
  eltwise(0, fmap_o_.num());
}

template <typename DType>
void QlinearPow<DType>::eltwise(std::uint32_t start_index,
                                std::uint32_t end_index) {
  read();
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;

  float x0_s =
      xir_op_->template get_attr<std::vector<float>>("a_scale").front();

  std::int32_t x0_zp =
      xir_op_->template get_attr<std::vector<int32_t>>("a_zero_point")[0];

  float x1_s =
      xir_op_->template get_attr<std::vector<float>>("b_scale").front();
  std::int32_t x1_zp =
      xir_op_->template get_attr<std::vector<int32_t>>("b_zero_point")[0];

  std::int32_t y_zp =
      xir_op_->template get_attr<std::vector<int32_t>>("y_zero_point")[0];
  float y_s = xir_op_->template get_attr<std::vector<float>>("y_scale")[0];

  for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
    auto ret = std::pow((data_in_[pos_iter] - x0_zp) / x0_s,
                        float((exponent_[0] - x1_zp) / x1_s)) *
                   y_s +
               y_zp;
    if (ret - floor(ret) == 0.5) {
      ret = std::round(ret * 0.5f) * 2.0f;
    } else {
      ret = round(ret);
    }
    if (int(ret) > data_max_) {
      data_out_[pos_iter] = data_max_;
    } else if (int(ret) < data_min_) {
      data_out_[pos_iter] = data_min_;
    } else {
      data_out_[pos_iter] = int(ret);
    }
  }
}

INSTANTIATE_TPCLASS(QlinearPow);
REG_OP_INSTANCE_FUNC("qlinear-pow", QlinearPow);

}  // namespace cpu
}  // namespace vart
