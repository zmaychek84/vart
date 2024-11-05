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

#include "instancenorm.hpp"

namespace vart {
namespace cpu {

template <typename DType>
InstanceNorm<DType>::InstanceNorm(const xir::Subgraph *subg, const xir::Op *op,
                                  IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  UNI_LOG_CHECK(xir_op_->has_attr("eps"), VART_NOT_FOUND)
      << "attr `eps` is required";
  epsilon_ = xir_op_->get_attr<float>("eps");
  GET_INPUT_DIMX_FMAP(input_shape, input, 4);
  UNI_LOG_CHECK(input_shape.size() == 4, VART_SIZE_ERROR)
      << "input shape must be NHWC";
  UNI_LOG_CHECK(input_shape[0] == 1, VART_SIZE_ERROR)
      << "input_shape[0] must be 0";
}

template <typename DType> void InstanceNorm<DType>::run() {
  // // do necessary check
  // print_param();
  // this->check_param();
  for (auto i = 0; i < input_shape[input_shape.size() - 1]; i++) {
    inst_norm(i);
  }

  // // read data
  // this->read();
  // Output = (scale * (Input - mean) / sqrt(variance + epsilon)) + bias
  // process

  // // do save, debug...
  // this->save();
}
template <typename DType> void InstanceNorm<DType>::inst_norm(int32_t k) {
  float mean = 0;
  float variance = 0;

  DType scale = *(input_weights_ptr_ + k);
  DType bias = *(input_bias_ptr_ + k);
  for (auto i = 0; i < input_shape[1]; i++) {
    for (auto j = 0; j < input_shape[2]; j++) {
      auto offset =
          i * input_shape[2] * input_shape[3] + j * input_shape[3] + k;
      mean += *(input_input_ptr_ + offset);
    }
  }
  mean /= input_shape[1] * input_shape[2];

  float var_sum = 0;
  for (auto i = 0; i < input_shape[1]; i++) {
    for (auto j = 0; j < input_shape[2]; j++) {
      auto offset =
          i * input_shape[2] * input_shape[3] + j * input_shape[3] + k;
      var_sum += (*(input_input_ptr_ + offset) - mean) *
                 (*(input_input_ptr_ + offset) - mean);
    }
  }
  variance = var_sum / input_shape[1] / input_shape[2];
  float inv_stdev = 1.0f / sqrtf(variance + epsilon_);
  float channel_scale = scale * inv_stdev;
  float channel_shift = bias - channel_scale * mean;
  for (auto i = 0; i < input_shape[1]; i++) {
    for (auto j = 0; j < input_shape[2]; j++) {
      auto offset =
          i * input_shape[2] * input_shape[3] + j * input_shape[3] + k;
      *(output_output_ptr_ + offset) =
          channel_scale * (*(input_input_ptr_ + offset)) + channel_shift;
    }
  }
}

template <typename DType> void InstanceNorm<DType>::read() {
  input_input_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(0));
  input_weights_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("weights").at(0));
  input_bias_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("bias").at(0));
  output_output_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}

INSTANTIATE_TPCLASS(InstanceNorm);
REG_OP_INSTANCE_FUNC("instancenorm", InstanceNorm);

} // namespace cpu
} // namespace vart