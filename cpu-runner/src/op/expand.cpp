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

#include "expand.hpp"

#include <iostream>
#include <vector>

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> Expand<DType>::ITName = {"input", "shape"};

// constructor and deconstructor
template <typename DType>
Expand<DType>::Expand(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  // expand_size = xir_op_->template get_attr<vector<std::int32_t>>("size");
  output_f_.resize(fmap_o_.num());
}

template <typename DType>
void Expand<DType>::run() {
  // // do necessary check
  // print_param();
  // check_param();

  // // read data
  read();

  // do reshape
  expand();

  // // do save, debug...
  // save();
}

template <typename DType>
void Expand<DType>::print_param() {
  fmap_i_.print_param("fmap_i");
  fmap_o_.print_param("fmap_o");
}

// read data from DDR
template <typename DType>
void Expand<DType>::read() {
  // read img data
  auto* cputb = inputs_.at(ITName[INPUT]).at(0);

  data_in_ptr_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t Expand<DType>::get_workload() {
  // TODO
  return 0;
}

template <typename DType>
void Expand<DType>::expand() {
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
          output_f_[output_idx] = data_in_ptr_[input_idx];
          data_out_ptr_[output_idx] = output_f_[output_idx];
        }
      }
    }
  }
}

INSTANTIATE_TPCLASS(Expand);
REG_OP_INSTANCE_FUNC("expand", Expand);
// REG_OP_INSTANCE_FUNC("expand-fix", Expand);

}  // namespace cpu
}  // namespace vart
