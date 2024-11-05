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

#include "mac_fix.hpp"
#define READ_FP(tname, idx, out)                                               \
  auto tensor_##out = op->get_input_tensor(tname, idx);                        \
  UNI_LOG_CHECK(tensor_##out->has_attr("fix_point"), VART_NOT_SUPPORT)         \
      << std::string("can't read fix_point of ") + tname + "_" +               \
             std::to_string(idx);                                              \
  out = tensor_##out->get_attr<int>("fix_point");

namespace vart {
namespace cpu {

template <typename DType>
MacFix<DType>::MacFix(const xir::Subgraph* subg, const xir::Op* op,
                      IMapTBs_t inputs, CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAPS(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);
  READ_FP("input", 0, fp_0_);
  READ_FP("input", 1, fp_1_);
  READ_FP("input", 2, fp_2_);
  fp_max_ = std::max(fp_0_ + fp_2_, fp_1_);
  auto tensor_output = op->get_output_tensor();
  UNI_LOG_CHECK(tensor_output->has_attr("fix_point"), VART_NOT_SUPPORT)
      << "can't read fix_point of tensor_output";
  fp_out_ = tensor_output->get_attr<int>("fix_point");
}

template <typename DType>
uint32_t MacFix<DType>::broadcast_pos(uint32_t pos_iter, const Dimension& dim) {
  auto get_sub_coord = [](coord_t coord, int start) {
    return coord_t(coord.begin() + start, coord.end());
  };
  auto pos = pos_iter;
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  fmap_o_.pos2coord(pos_iter, dst_coord);
  int delta = 0;
  for (auto dim_iter = 0U; dim_iter < dst_coord.size(); dim_iter++) {
    if (dim.ndims() < int(dst_coord.size())) {
      delta = dst_coord.size() - dim.ndims();
    }
    if (int(dim_iter) >= delta)
      src_coord[dim_iter] = dst_coord[dim_iter] % dim[dim_iter - delta];
  }
  pos = dim.coord2pos(get_sub_coord(src_coord, delta));
  return pos;
}

template <typename DType>
void MacFix<DType>::run() {
  // ofm = ifm0 * ifm2 + ifm1
  for (auto pos_iter = 0; pos_iter < fmap_o_.num(); ++pos_iter) {
    auto pos_0 = broadcast_pos(pos_iter, fmap_i_[0]);
    auto pos_1 = broadcast_pos(pos_iter, fmap_i_[1]);
    auto pos_2 = broadcast_pos(pos_iter, fmap_i_[2]);

    DType out_tmp;
    out_tmp =
        ifm0_ptr_[pos_0] * ifm2_ptr_[pos_2] * pow(2, fp_max_ - fp_0_ - fp_2_) +
        ifm1_ptr_[pos_1] * pow(2, fp_max_ - fp_1_);
    out_ptr_[pos_iter] = round_normal(
        CPUOPBase::round_mode_, float(out_tmp) * pow(2, fp_out_ - fp_max_),
        CPUOPBase::data_min_, CPUOPBase::data_max_);
  }
}

template <typename DType>
void MacFix<DType>::read() {
  ifm0_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(0));
  ifm1_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(1));
  ifm2_ptr_ = GET_CPUTB_DType_PTR(DType, inputs_.at("input").at(2));
  out_ptr_ = GET_CPUTB_DType_PTR(DType, output_);
}
template <typename DType>
void MacFix<DType>::print_param() {
  fmap_o_.print_param("fmap_o_");
  fmap_i_[0].print_param("fmap_i_[0]");
  fmap_i_[1].print_param("fmap_i_[1]");
  fmap_i_[2].print_param("fmap_i_[2]");
  UNI_LOG_DEBUG_INFO << "fp_0_ = " << fp_0_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_1_ = " << fp_1_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_2_ = " << fp_2_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_out_ = " << fp_out_ << endl;
}
INSTANTIATE_TPCLASS(MacFix);
REG_OP_INSTANCE_FUNC("mac-fix", MacFix);

}  // namespace cpu
}  // namespace vart
