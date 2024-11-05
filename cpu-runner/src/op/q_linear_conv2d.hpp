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

#pragma once

#include <cstdint>

#include "conv2d.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
class QLinearConv2d : public Conv2d<DType, WType> {
 public:
  QLinearConv2d(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output);
  ~QLinearConv2d() = default;

  virtual void run() override final;

  virtual void print_param() override;
  virtual void check_param() override;

 private:
  void fix();
  std::vector<int32_t> calculate_ifm();

  void qlinearconv2d_conv();
  void qlinearconv2d_conv_one(DType* src, WType* wts, int idx_dst_n,
                              int idx_dst_h, int idx_dst_w, int idx_oc,
                              int32_t ic_begin, int32_t ic_end);

 private:
  int64_t channel_;
  float scale_input_;
  float scale_b_;
  int32_t zp_input_;

  float scale_weights_;
  int32_t zp_weights_;

  float scale_output_;
  int32_t zp_output_;
  std::vector<int64_t> tmp_data_out_;

  double prelu_alpha_{0.1015625};
  int prelu_in_{0};
  int prelu_shift_{0};

  int32_t nonlinear_type_;
  uint32_t THREAD_NUM;
  uint32_t THREAD_WORKLOAD;

  using ConvBase<DType, WType>::xir_op_;
  using ConvBase<DType, WType>::has_bias_;
  using ConvBase<DType, WType>::fmap_o_;
  using ConvBase<DType, WType>::raw_fmap_i_;
  using ConvBase<DType, WType>::fmap_i_;
  using ConvBase<DType, WType>::fmap_w_;
  using ConvBase<DType, WType>::fmap_b_;
  using ConvBase<DType, WType>::pad_;
  using ConvBase<DType, WType>::dilation_;
  using ConvBase<DType, WType>::ITName;
  using CPUOPBase::inputs_;

  using CPUOPBase::data_max_;
  using CPUOPBase::data_min_;
  using ConvBase<DType, WType>::bias_ptr_;
  using ConvBase<DType, WType>::data_out_ptr_;
  using ConvBase<DType, WType>::weights_ptr_;
  using ConvBase<DType, WType>::data_in_ptr_;

  using ConvBase<DType, WType>::stride_;
  using ConvBase<DType, WType>::kernel_;
};

}  // namespace cpu
}  // namespace vart
