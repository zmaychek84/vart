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

#include "matmul_fix.hpp"

namespace vart {
namespace cpu {

float py3_round_float(const float& input) {
  float ret;
  auto default_round_mode = std::fegetround();
  if (!(FE_TONEAREST == default_round_mode)) {
    std::fesetround(FE_TONEAREST);
    ret = std::nearbyint(input);
    std::fesetround(default_round_mode);
  } else {
    ret = std::nearbyint(input);
  }
  return ret;
}

float dpu_round_float(const float& input) {
  float ret = input;
  if (ret >= 0) {
    ret = std::round(ret);
  } else {
    float delta = ret - std::floor(ret);
    if (delta == 0.5f) {
      ret = std::ceil(ret);
    } else {
      ret = std::round(ret);
    }
  }
  return ret;
}
/**
 * @brief round method borrowed from XIR's internal implementation
 */
float xround(const float& data, const std::string& round_mode) {
  float ret;
  if ("STD_ROUND" == round_mode) {
    ret = std::round(data);
  } else if ("DPU_ROUND" == round_mode) {
    ret = dpu_round_float(data);
  } else if ("PY3_ROUND" == round_mode) {
    ret = py3_round_float(data);
  } else {
    UNI_LOG_FATAL(XIR_UNSUPPORTED_ROUND_MODE) << round_mode << " is not supported.";
  }
  return ret;
}
// constructor and deconstructor
template <typename DType1, typename DType2, typename BType>
MatmulFix<DType1, DType2, BType>::MatmulFix(const xir::Subgraph* subg, const xir::Op* op, IMapTBs_t inputs,
                                            CPUTBPtr_t output)
    : Matmul<DType1, DType2, BType>(subg, op, inputs, output) {
  auto tensor_a = op->get_input_tensor("input", 0);
  fp_input_a_ = tensor_a->get_attr<int>("fix_point");

  if (this->has_wts_) {
    fp_input_b_ = op->get_input_tensor("weights")->get_attr<int>("fix_point");
  } else {
    fp_input_b_ = op->get_input_tensor("input", 1)->get_attr<int>("fix_point");
  }

  if (this->has_bias_) {
    auto tensor_bias = op->get_input_tensor("bias");
    fp_bias_ = tensor_bias->get_attr<int>("fix_point");
  }
  auto tensor_output = op->get_output_tensor();
  fp_output_ = tensor_output->get_attr<int>("fix_point");
  round_mode_ = tensor_output->get_attr<std::string>("round_mode");

  UNI_LOG_DEBUG_INFO << "input fix_point:" << fp_input_a_ << " " << fp_input_b_ << std::endl;
  if (this->has_bias_) {
    UNI_LOG_DEBUG_INFO << "bias fix_point:" << fp_bias_ << std::endl;
  }
  UNI_LOG_DEBUG_INFO << "output fix_point:" << fp_output_ << std::endl;
  UNI_LOG_DEBUG_INFO << "round mode:" << round_mode_ << std::endl;
}

/**
 * @brief implementation of Matmul-fix OP in CPURunner
 * @note Follow the basic flow of Matmul OP and perform the fix calculation for inputs and outputs
 */
template <typename DType1, typename DType2, typename BType>
void MatmulFix<DType1, DType2, BType>::run() {
  UNI_LOG_DEBUG_INFO << "[CPURunner] Run Matmul-fix..." << std::endl;
  for (auto i = 0; i < this->fmap_o_.num(); i++) {
    auto coord = this->fmap_o_.pos2coord(i);

    auto coord_a = coord;
    auto coord_b = coord;
    for (auto j = 0; j < this->fmap_o_.ndims(); j++) {
      if (this->fmap_ia_[j] == 1 || j == this->fmap_o_.ndims() - 1) {
        coord_a[j] = 0;
      }

      if (this->fmap_ib_[j] == 1 || j == this->fmap_o_.ndims() - 2) {
        coord_b[j] = 0;
      }
    }

    auto pos_a = this->fmap_ia_.coord2pos(coord_a);
    auto pos_b = this->fmap_ib_.coord2pos(coord_b);

    float tmp = 0;
    for (auto j = 0; j < this->K_; j++) {
      tmp += this->data_ina_[pos_a + j] * this->data_inb_[pos_b + j * this->N_];
    }
    tmp /= 1.0 * pow(2, fp_input_a_ + fp_input_b_);
    if (this->has_bias_) {
      tmp += this->data_bias_[i % this->N_] * 1.0 / pow(2, fp_bias_);
    }
    tmp = xround(tmp * pow(2, fp_output_), round_mode_);
    this->data_out_[i] = static_cast<DType1>(tmp);
  }
}

INSTANTIATE_TPCLASS_MATMUL(MatmulFix);

REG_SPECIFIED_MATMUL_DATATYPE_OP_INSTANCE_FUNC("matmul-fix", MatmulFix);

}  // namespace cpu
}  // namespace vart
