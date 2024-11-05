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

#include "qreduce_mean.hpp"
// #include "reduce_mean_fix.cpp"

#include "pair_hash.hpp"

namespace vart {
namespace cpu {

static std::int32_t get_shift_from_float(float in_f) {
  std::int32_t rep{0};
  std::memcpy(&rep, &in_f, sizeof in_f);
  rep &= 0x7fffffff;  // Remove sign bit
  return 127 - (((rep >> 23) & 255) + 1) + (8 * sizeof(std::int32_t) - 2);
}

template <typename DType>
QReduceMean<DType>::QReduceMean(const xir::Subgraph* subg, const xir::Op* op,
                                IMapTBs_t inputs, CPUTBPtr_t output)

    : ReduceMean<DType>(subg, op, inputs, output) {
  const xir::Op* reduce_op = CPUOPBase::xir_op_;

  auto input_tensor = reduce_op->get_input_tensor("input");
  auto i_shape = input_tensor->get_shape();

  std::uint32_t mean_divide = std::transform_reduce(
      reduce_dims_.begin(), reduce_dims_.end(), 1, std::multiplies<>(),
      [&i_shape](int dim) { return i_shape.at(dim); });
  std::int32_t x_zp =
      reduce_op->get_attr<std::vector<std::int32_t>>("x_zero_point").front();
  std::int32_t y_zp =
      reduce_op->get_attr<std::vector<std::int32_t>>("y_zero_point").front();
  float x_s = reduce_op->get_attr<std::vector<float>>("x_scale").front();
  float y_s = reduce_op->get_attr<std::vector<float>>("y_scale").front();
  float c0_f = y_s / x_s / mean_divide;
  float c1_f = -(x_zp * y_s) / x_s + y_zp;

  std::int32_t c0_int = get_int32_representation_from_float(c0_f);
  auto c0_shift = get_shift_from_float(c0_f);
  std::int32_t c1_int = get_int32_representation_from_float(c1_f);
  auto c1_shift = get_shift_from_float(c1_f);

  // set coefficients and shifts info
  C_0 = c0_int;
  C_1 = c1_int;

  FP_0 = c0_shift;
  FP_1 = c1_shift;
}

std::pair<int32_t, int32_t> qdq_get_mean_dpu_fix_factors(
    const std::int32_t& input_div) {
  auto rec = input_div;
  auto multi_factor = 0;
  auto shift_factor = 0;
  auto diff = 1.f;
  auto max_factor = std::ceil(std::log2(rec * 128));
  for (auto shift_factor_ = 0; shift_factor_ < max_factor; shift_factor_++) {
    auto factor = std::round(std::exp2(shift_factor_) / rec);
    auto diff_ = std::abs(factor / std::exp2(shift_factor_) - 1.f / rec);
    if (diff_ < diff) {
      multi_factor = factor;
      diff = diff_;
      shift_factor = shift_factor_;
    }
  }
  return {multi_factor, shift_factor};
}

template <typename DType>
void QReduceMean<DType>::print_param() {
  ReduceBase<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "c0_int: " << C_0 << endl;
  UNI_LOG_DEBUG_INFO << "c1_int: " << C_1 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c0: " << FP_0 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c1: " << FP_1 << endl;
}

template <typename DType>
void QReduceMean<DType>::calculate() {
  // ReduceMean<DType>::calculate();
  // caculate
  // double factor = 1.0;
  // float pow_shift = 1.0;
  for (auto idx : reduce_dims_) {
    auto cur_dim = fmap_i_.dim(idx);
    // factor = factor / cur_dim;
    // auto factors = qdq_get_mean_dpu_fix_factors(cur_dim);
    // factor *= (double)(factors.first);
    // pow_shift *= std::exp2(factors.second);
    auto cur_cod = fmap_i_.cod(idx);
    auto n_cur_chunk = fmap_i_.num() / (cur_dim * cur_cod);

    for (auto i = 0; i < n_cur_chunk; i++) {
      DType* prlt = data_in_ptr_ + i * cur_dim * cur_cod;

      for (auto j = 1; j < cur_dim; j++) {
        DType* pcur = prlt + j * cur_cod;

        // accumulate
        for (auto k = 0; k < cur_cod; k++) {
          prlt[k] += pcur[k];
        }
      }
    }
  }

  int32_t c1_shift_pre = ((FP_1 - FP_0) >= 32) ? 0 : (FP_0 - FP_1);
  int64_t c1_int_pre = ((FP_1 - FP_0) >= 32) ? 0 : C_1;
  for (auto i = 0; i < fmap_i_.num(); i++) {
    // simulate aie shift-round-saturate
    int64_t tmp = (double)(static_cast<std::int64_t>(data_in_ptr_[i]) * C_0 +
                           (c1_int_pre << c1_shift_pre));
    if (FP_0 > 0) {
      // shift
      int8_t truncated_bit = (tmp >> (FP_0 - 1)) & 0x1;
      // round
      tmp = (tmp >> FP_0) + truncated_bit;
    }
    // saturate
    if (tmp > CPUOPBase::data_max_) {
      data_in_ptr_[i] = CPUOPBase::data_max_;
    } else if (tmp < CPUOPBase::data_min_) {
      data_in_ptr_[i] = CPUOPBase::data_min_;
    } else {
      data_in_ptr_[i] = tmp;
    }
  }
}

template <>
void QReduceMean<float>::calculate() {
  double factor = 1.0;
  for (auto idx : reduce_dims_) {
    auto cur_dim = fmap_i_.dim(idx);
    factor = factor / cur_dim;
    auto cur_cod = fmap_i_.cod(idx);
    auto n_cur_chunk = fmap_i_.num() / (cur_dim * cur_cod);

    for (auto i = 0; i < n_cur_chunk; i++) {
      float* prlt = data_in_ptr_ + i * cur_dim * cur_cod;

      for (auto j = 1; j < cur_dim; j++) {
        float* pcur = prlt + j * cur_cod;

        // accumulate
        for (auto k = 0; k < cur_cod; k++) {
          prlt[k] += pcur[k];
        }
      }
    }
  }
  for (auto i = 0; i < fmap_i_.num(); i++) {
    data_in_ptr_[i] *= factor;
  }
}

INSTANTIATE_TPCLASS(QReduceMean);
REG_OP_INSTANCE_FUNC("qlinear-reduction-mean", QReduceMean);
}  // namespace cpu
}  // namespace vart
