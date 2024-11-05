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

#include "dwconv.hpp"

namespace vart {
namespace cpu {

template <typename DType, typename WType>
class QLinearDWConv2d : public DWConv<DType, WType> {
 public:
  QLinearDWConv2d(const xir::Subgraph* subg, const xir::Op* op,
                  IMapTBs_t inputs, CPUTBPtr_t output);
  ~QLinearDWConv2d() = default;

  virtual void run() override final;

  virtual void print_param() override;
  virtual void check_param() override;

 private:
  void fix();
  std::vector<DType> calculate_ifm();
  std::vector<WType> calculate_wgt();
  std::vector<int64_t> calculate_cf2(std::vector<float> cf_float);

 private:
  float sa_;
  int32_t za_;

  float sw_;
  int32_t zw_;

  float so_;
  int32_t zo_;

  float sb_;
  int32_t zb_;

  int32_t shift_b_;
  int32_t shift_out_;
  
  using DWConvBase<DType, WType>::xir_op_;
  using DWConvBase<DType, WType>::has_bias_;
  using DWConvBase<DType, WType>::fmap_o_;
  using DWConvBase<DType, WType>::raw_fmap_i_;
  using DWConvBase<DType, WType>::fmap_i_;
  using DWConvBase<DType, WType>::fmap_w_;
  using DWConvBase<DType, WType>::fmap_b_;
  using DWConvBase<DType, WType>::pad_;
  using DWConvBase<DType, WType>::dilation_;
  using DWConvBase<DType, WType>::ITName;
  using CPUOPBase::data_max_;
  using CPUOPBase::data_min_;
  using CPUOPBase::inputs_;
  using DWConvBase<DType, WType>::bias_ptr_;
  using DWConvBase<DType, WType>::data_out_ptr_;
  using DWConvBase<DType, WType>::weights_ptr_;
  using DWConvBase<DType, WType>::data_in_ptr_;

  using DWConvBase<DType, WType>::stride_;
  using DWConvBase<DType, WType>::kernel_;
};

}  // namespace cpu
}  // namespace vart
