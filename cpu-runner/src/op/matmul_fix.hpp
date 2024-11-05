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

#include "matmul.hpp"

namespace vart {
namespace cpu {

template <typename DType1, typename DType2, typename BType>
class MatmulFix : public Matmul<DType1, DType2, BType> {
 public:
  MatmulFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs, CPUTBPtr_t output);
  ~MatmulFix() = default;

  virtual void run() override final;

 private:
  int fp_input_a_;
  int fp_input_b_;
  int fp_bias_;
  int fp_output_;
  std::string round_mode_;
};

}  // namespace cpu
}  // namespace vart
