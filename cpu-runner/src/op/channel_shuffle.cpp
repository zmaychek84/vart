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

#include "channel_shuffle.hpp"

namespace vart {
namespace cpu {

template <typename DType>
ChannelShuffle<DType>::ChannelShuffle(const xir::Subgraph* subg, const xir::Op* op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  input_shape_ =
      op->get_input_ops("input")[0]->get_output_tensor()->get_shape();
  output_shape_ = op->get_output_tensor()->get_shape();
  //upscale_ = xir_op_->get_attr<bool>("upscale");
  group_ = xir_op_->get_attr<int>("group");

  print_param();
}

template <typename DType>
void ChannelShuffle<DType>::run() {
  read();
  shuffle();
}

template <typename DType>
void ChannelShuffle<DType>::print_param() {
  //UNI_LOG_DEBUG_INFO << "upscale: " << Bool2Str(upscale_) << endl;
  UNI_LOG_DEBUG_INFO << "group: " << group_ << endl;
}

template <typename DType>
void ChannelShuffle<DType>::check_param() {}

template <typename DType>
void ChannelShuffle<DType>::read() {
  auto* cputb = inputs_.at("input").at(0);
  data_in_ = GET_CPUTB_DType_PTR(DType, cputb);
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);
}

template <typename DType>
uint64_t ChannelShuffle<DType>::get_workload() {
  return 0;
}

template <typename DType>
void ChannelShuffle<DType>::shuffle() {
  
  int idx = 0;
  for (auto b = 0; b < output_shape_[0]; b++)
    for (auto h = 0; h < output_shape_[1]; h++)
      for (auto w = 0; w < output_shape_[2]; w++)
        for (auto c = 0; c < output_shape_[3]; c++) {
          auto each_tile = output_shape_[3] / group_;
          auto i_index =
              b * input_shape_[1] * input_shape_[2] * input_shape_[3] +
              h * input_shape_[2] * input_shape_[3] +
              w * input_shape_[3] + c;
          idx = (i_index % each_tile) * group_ + std::floor(i_index / each_tile);
          data_out_[idx] = data_in_[i_index];
        }
  
}

INSTANTIATE_TPCLASS(ChannelShuffle);

REG_OP_INSTANCE_FUNC("channel-shuffle", ChannelShuffle);

}  // namespace cpu
}  // namespace vart
