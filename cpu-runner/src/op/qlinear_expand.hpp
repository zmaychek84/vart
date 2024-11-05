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

#include "cpu_op_base.hpp"

namespace vart {
namespace cpu {

template <typename DType>
class QlinearExpand : public CPUOPBase {
 public:
  enum InputTensorType { INPUT, SHAPE };
  // input tensor name
  const static vector<string> ITName;

 public:
  QlinearExpand(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                CPUTBPtr_t output);
  ~QlinearExpand() = default;

  virtual void run() override final;

  virtual void print_param() override final;

  virtual void read() override final;

  virtual uint64_t get_workload() override final;

 private:
  void q_expand();

 private:
  FMap_t fmap_i_;
  FMap_t fmap_o_;

  vector<DType> shape_;
  // vector<std::int32_t> expand_size;

  vector<std::int32_t> shift_factor_;
  int32_t C_0, C_1;
  int32_t FP_0, FP_1;
  using CPUOPBase::bit_width_;
  using CPUOPBase::data_max_;
  using CPUOPBase::data_min_;
  // caculate buffer
  DType* data_in_ptr_{nullptr};
  // std::vector<float> output_f_;
  DType* data_out_ptr_{nullptr};
};

}  // namespace cpu
}  // namespace vart
