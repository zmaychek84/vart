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

#include "cpu_op_base.hpp"

namespace vart {
namespace cpu {

template<typename DType>
class MacFix : virtual public CPUOPBase {
public:
  MacFix(const xir::Subgraph* subg, const xir::Op* op,
      IMapTBs_t inputs, CPUTBPtr_t output);
  virtual ~MacFix() = default;

  virtual void run() override;
  virtual void read() override;
  virtual void print_param() override;

protected:
  uint32_t broadcast_pos(uint32_t pos_iter, const Dimension& dim);
  std::vector<Dimension> fmap_i_;
  Dimension fmap_o_;
  int fp_0_;
  int fp_1_;
  int fp_2_;
  int fp_max_; // fp_b - fp_a *fp_c
  int fp_out_;
  DType* ifm0_ptr_; // Out = ifm0 * ifm2 + ifm1 // Out = ifm0 * ifm1 + ifm2
  DType* ifm1_ptr_;
  DType* ifm2_ptr_;
  DType* out_ptr_;
};

} // namespace cpu
} // namespace vart


