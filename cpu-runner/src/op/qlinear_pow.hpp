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

#pragma once

#include "eltwise.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class QlinearPow : public CPUOPBase {
 public:
  enum InputTensorType { INPUT, EXPONENT };
  // input tensor name
  const static vector<string> ITName;

 public:
  QlinearPow(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
             CPUTBPtr_t output);
  ~QlinearPow() = default;
  virtual void run() override;
  virtual void eltwise(std::uint32_t start_index, std::uint32_t end_index);
  virtual void read() override final;

 private:
  vector<Dimension> fmap_i_;
  Dimension fmap_e_;
  Dimension fmap_o_;
  uint32_t broadcast_pos(uint32_t);
  using CPUOPBase::bit_width_;
  using CPUOPBase::data_max_;
  using CPUOPBase::data_min_;

  DType* data_in_{nullptr};
  DType* data_out_{nullptr};
  DType* exponent_{nullptr};

  using CPUOPBase::inputs_;
  using CPUOPBase::output_;
};

}  // namespace cpu
}  // namespace vart
