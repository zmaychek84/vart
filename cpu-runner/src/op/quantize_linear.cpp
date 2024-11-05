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

#include "quantize_linear.hpp"

#include "cpu_util.hpp"
namespace vart {
namespace cpu {

template <typename DType, typename FixType>
const std::vector<string> QuantizeLinear<DType, FixType>::ITName = {
    "input",
    "scale",
    "zero_point",
};

template <typename DType, typename FixType>
QuantizeLinear<DType, FixType>::QuantizeLinear(const xir::Subgraph* subg,
                                               const xir::Op* op,
                                               IMapTBs_t inputs,
                                               CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  UNI_LOG_CHECK(xir_op_->has_attr("data_type"), VART_FOUND)
      << "attr `data_type` is required";
  data_type = xir_op_->get_attr<std::string>("data_type");
  UNI_LOG_CHECK(xir_op_->has_attr("scale"), VART_FOUND)
      << "attr `scale` is required";
  scale = xir_op_->get_attr<std::vector<float>>("scale");
  for (float elem : scale) {
    UNI_LOG_CHECK(elem != 0, VART_INVALID_VALUE) << "scale must non-zero";
  }
  UNI_LOG_CHECK(xir_op_->has_attr("zero_point"), VART_FOUND)
      << "attr `zero_point` is required";
  zero_point = xir_op_->get_attr<std::vector<std::int32_t>>("zero_point");
  UNI_LOG_CHECK(xir_op_->has_attr("axis"), VART_FOUND)
      << "attr `axis` is required";
  axis = xir_op_->get_attr<std::int32_t>("axis");
  UNI_LOG_CHECK(zero_point.size() == scale.size() && zero_point.size() == 1,
                VART_SIZE_ERROR)
      << "zero_point's shape must match scale";
  bool if_signed;
  std::int32_t bit_width_;
  if(data_type == "INT8"){
    if_signed = true;
    bit_width_ = 8;
  }else if(data_type == "UINT8"){
    if_signed = false;
    bit_width_ = 8;
  }else if(data_type == "INT16"){
    if_signed = true;
    bit_width_ = 16;
  }else if(data_type == "UINT16"){
    if_signed = false;
    bit_width_ = 16;
  }else{
    throw "qdq only support INT8,UINT8,INT16,UINT16, data_type = "+data_type;
  }
  data_min_ = if_signed ? -pow(2, bit_width_ - 1) : 0;
  data_max_ =
      if_signed ? pow(2, bit_width_ - 1) - 1 : pow(2, bit_width_) - 1;
  // get input_len
  std::vector<std::int32_t> input_shape;
  GET_INPUT_DIMX_FMAP(
      input_shape, input,
      6);  // input_shape is output , input is input tensor's name
  input_len = get_vec_mul(input_shape);
}

template <typename DType, typename FixType>
QuantizeLinear<DType, FixType>::~QuantizeLinear() {
  UNI_LOG_DEBUG_INFO << "destroy this " << (void*)this << endl;
}

template <typename DType, typename FixType>
void QuantizeLinear<DType, FixType>::run() {
  // // do necessary check
  // this->print_param();
  // this->check_param();

  // // read data
  // read();

  // do qlinear
  for (auto i = 0u; i < input_len; i++) {
    this->float2fix_dpu_round(i);
  }
  // do save, debug...
  // save();
}

template <typename DType, typename FixType>
void QuantizeLinear<DType, FixType>::read() {
  // read input
  input_input_ptr = GET_CPUTB_DType_PTR(DType, inputs_.at(ITName[INPUT]).at(0));
  // handle output buffer
  output_output_ptr = GET_CPUTB_DType_PTR(FixType, output_);

  // If there is an input, we will prioritize the input scale and zp
  if (xir_op_->get_input_num(ITName[SCALE]) > 0) {
    auto input_scale_ptr =
        GET_CPUTB_DType_PTR(DType, inputs_.at(ITName[SCALE]).at(0));
    scale.clear();
    scale.push_back(input_scale_ptr[0]);
    if (xir_op_->get_input_num(ITName[ZERO_POINT]) > 0) {
      auto input_zero_point_ptr =
          GET_CPUTB_DType_PTR(FixType, inputs_.at(ITName[ZERO_POINT]).at(0));
      zero_point.clear();
      zero_point.push_back(input_zero_point_ptr[0]);
    } else {
      zero_point.clear();
    }
  }
}

class QuantizeLinear_impl : public QuantizeLinear<float, std::int32_t> {
 public:
  QuantizeLinear_impl(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
      : QuantizeLinear<float, std::int32_t>(subg, op, inputs, output) {}
  ~QuantizeLinear_impl() = default;
};

REG_NONTP_OP_INSTANCE_FUNC("quantize-linear", QuantizeLinear_impl);

}  // namespace cpu
}  // namespace vart
