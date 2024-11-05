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

#include "reduce_base.hpp"
#include "reduce_sum.hpp"

namespace vart {
namespace cpu {

template<typename DType>
class QReduceSum : public ReduceSum<DType> {
public:
  QReduceSum(const xir::Subgraph* subg, const xir::Op* op,
               IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~QReduceSum() = default;

protected:
  virtual void calculate() override final;
  virtual void print_param() override final;

protected:

  std::int32_t C_0;
  std::int32_t C_1;
  std::int8_t FP_0;
  std::int8_t FP_1;

  using ReduceBase<DType>::fmap_i_;
  using ReduceBase<DType>::fmap_o_;
  using ReduceBase<DType>::reduce_dims_;
  using ReduceBase<DType>::data_in_ptr_;
  using ReduceBase<DType>::data_out_ptr_;
};

} // namespace cpu
} // namespace vart


