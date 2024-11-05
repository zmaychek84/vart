/*
 * Copyright 2022 Xilinx Inc.
 * Copyright 2023-2024 Advanced Micro Devices Inc.
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

#include "conv2d_fix.hpp"
#include "fast_pad.hpp"

#include "cpu_tensor_utils.hpp"
namespace vart {
namespace cpu {

// constructor and deconstructor
template <typename DType, typename WType>
Conv2dFix<DType, WType>::Conv2dFix(const xir::Subgraph *subg, const xir::Op *op,
                                   IMapTBs_t inputs, CPUTBPtr_t output)
    : Conv2d<DType, WType>(subg, op, inputs, output) {
  auto *xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");

  auto *xir_tensor_w = CPUOPBase::xir_op_->get_input_tensor("weights");
  fp_weights_ = xir_tensor_w->get_attr<int>("fix_point");

  if (has_bias_) {
    auto *xir_tensor_b = CPUOPBase::xir_op_->get_input_tensor("bias");
    fp_bias_ = xir_tensor_b->get_attr<int>("fix_point");
    shift_bias_ = fp_input_ + fp_weights_ - fp_bias_;
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  fp_output_ = xir_tensor_o->has_attr("fix_point") ? xir_tensor_o->get_attr<int>("fix_point") : 0;
  output_data_type_ = xir_tensor_o->get_data_type();
  shift_cut_ = fp_input_ + fp_weights_ - fp_output_;

  auto nonlinear_type_str = CPUOPBase::xir_op_->get_attr<string>("nonlinear");
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "")
    nonlinear_type_ = 0;
  else if (nonlinear_type_str == "RELU")
    nonlinear_type_ = 1;
  else if (nonlinear_type_str == "PRELU") {
    nonlinear_type_ = 2;
    prelu_in_ = CPUOPBase::xir_op_->get_attr<int>("prelu_in");
    prelu_shift_ = CPUOPBase::xir_op_->get_attr<int>("prelu_shift");
    prelu_alpha_ = ((double)prelu_in_) / pow(2.0, prelu_shift_);
  } else if (nonlinear_type_str == "LEAKYRELU") {
    prelu_alpha_ =
        (double)(CPUOPBase::xir_op_->get_attr<float>("LEAKYRELU_alpha"));
    nonlinear_type_ = 3;
  } else if (nonlinear_type_str == "RELU6")
    nonlinear_type_ = 4;
  else if (nonlinear_type_str == "HSIGMOID") {
    nonlinear_type_ = 5;
    shift_bias_ = CPUOPBase::xir_op_->get_attr<int>("shift_bias");
    shift_cut_ = CPUOPBase::xir_op_->get_attr<int>("shift_cut");
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
  } else if (nonlinear_type_str == "HSWISH") {
    nonlinear_type_ = 6;
    shift_bias_ = CPUOPBase::xir_op_->get_attr<int>("shift_bias");
    shift_cut_ = CPUOPBase::xir_op_->get_attr<int>("shift_cut");
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = CPUOPBase::xir_op_->get_attr<int>("shift_hswish");
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT) << "Unsupported nonlinear type: "
                                    << nonlinear_type_str << ".";
  }
  memcpy(&data_min_, &(CPUOPBase::data_min_), sizeof(CPUOPBase::data_min_));
  memcpy(&data_max_, &(CPUOPBase::data_max_), sizeof(CPUOPBase::data_max_));
  enable_bfp_ = false;
  if (CPUOPBase::xir_op_->has_attr("enable_bfp")) {
    enable_bfp_ = CPUOPBase::xir_op_->get_attr<bool>("enable_bfp");
  }
}

template <typename DType, typename WType>
void Conv2dFix<DType, WType>::print_param() {
  ConvBase<DType, WType>::print_param();

  UNI_LOG_DEBUG_INFO << "fp_input: " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_weights: " << fp_weights_ << endl;
  if (has_bias_) {
    UNI_LOG_DEBUG_INFO << "fp_bias: " << fp_bias_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output: " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "output rounding: " << round_mode_ << endl;
  UNI_LOG_DEBUG_INFO << "output data type: " << output_data_type_.to_string()
                     << endl;
  if (has_shift_psum_) {
    UNI_LOG_DEBUG_INFO << "shift_psum: " << shift_psum_ << endl;
  }
  if (has_bias_) {
    UNI_LOG_DEBUG_INFO << "shift_bias: " << shift_bias_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "shift_cut: " << shift_cut_ << endl;
  UNI_LOG_DEBUG_INFO << "nonlinear_type: " << nonlinear_type_ << endl;
  if (nonlinear_type_ == 3) {
    UNI_LOG_DEBUG_INFO << "leakyrelu_alpha: " << prelu_alpha_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "enable_conv_dirty: " << enable_conv_dirty_ << endl;
}

template <typename DType, typename WType>
void Conv2dFix<DType, WType>::check_param() {
  ConvBase<DType, WType>::check_param();

  UNI_LOG_CHECK(nonlinear_type_ == 0 || nonlinear_type_ == 1 ||
                    nonlinear_type_ == 3 || nonlinear_type_ == 4 ||
                    nonlinear_type_ == 5 || nonlinear_type_ == 6,
                VART_INVALID_VALUE)
      << " not supported. ";
}

std::vector<float>
calc_align_data(const std::vector<float> &ori_data,
                const std::vector<std::int64_t> &ori_shape,
                const std::vector<std::int64_t> &align_shape) {
  if (ori_shape == align_shape) {
    return ori_data;
  }
  auto align_data_num = accumulate(align_shape.begin(), align_shape.end(), 1,
                                   std::multiplies<std::int64_t>());
  std::vector<float> align_data(align_data_num);
  for (std::int32_t n = 0; n < ori_shape[0]; ++n) {
    for (std::int32_t h = 0; h < ori_shape[1]; ++h) {
      for (std::int32_t w = 0; w < ori_shape[2]; ++w) {
        auto ori_addr = n * ori_shape[1] * ori_shape[2] * ori_shape[3] +
                        h * ori_shape[2] * ori_shape[3] + w * ori_shape[3];
        auto new_addr = n * align_shape[1] * align_shape[2] * align_shape[3] +
                        h * align_shape[2] * align_shape[3] +
                        w * align_shape[3];
        std::copy_n(ori_data.begin() + ori_addr, ori_shape[3],
                    align_data.begin() + new_addr);
      }
    }
  }
  return align_data;
}
int32_t align_upper(int32_t a_, int32_t b_) {
  UNI_LOG_CHECK(b_ > 0, VART_INVALID_VALUE)
      << "Division by 0 or minus in align_upper, " << a_ << " align to " << b_;
  return a_ % b_ == 0 ? a_ - (a_ % b_) : a_ - (a_ % b_) + b_;
}

template <typename T>
std::vector<T>
strided_slice_data(const std::vector<T> &data,
                   const std::vector<std::int64_t> &input_shape,
                   const std::vector<std::int64_t> &output_shape) {
  UNI_LOG_CHECK(input_shape.size() == output_shape.size(), VART_INVALID_VALUE);
  auto shape_size = accumulate(input_shape.begin(), input_shape.end(), 1,
                               std::multiplies<std::int64_t>());
  auto input_data_num = static_cast<std::int32_t>(data.size());
  UNI_LOG_CHECK(input_data_num == shape_size, VART_INVALID_VALUE);
  auto data_num = accumulate(output_shape.begin(), output_shape.end(), 1,
                             std::multiplies<std::int64_t>());
  auto input_strides =
      cpu_tensor_utils::calc_tensor_stride_based_on_dims(input_shape);
  std::vector<T> ret(data_num);
  auto dst_coord = cpu_tensor_utils::pos2coord(0, output_shape);
  for (auto i = 0; i < data_num;) {
    dst_coord = cpu_tensor_utils::pos2coord(i, output_shape);
    auto src_pos = 0;
    for (auto j = 0U; j < output_shape.size() - 1; ++j) {
      src_pos += dst_coord[j] * input_strides[j];
    }
    src_pos += (dst_coord.back()) * input_strides.back();
    std::copy_n(data.begin() + src_pos, output_shape.back(), ret.begin() + i);
    i += output_shape.back();
  }
  return ret;
}

void LaunchBFPCPUKernel(const float *input, float *output, int n, int bit_width,
                        int block_size, int rounding_mode);
#if 0
float fp2bf16(float x) {
  uint32_t int_rep;
  std::memcpy(&int_rep, &x, sizeof(x));
  uint32_t lsb = (int_rep >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  int_rep += rounding_bias;
  int_rep = int_rep & 0xFFFF0000;
  float out = 0.f;
  std::memcpy(&out, &int_rep, sizeof(x));
  return out;
}
#endif
void LaunchBFPPrimeCPUKernel(const float *input, float *output, const int n,
                             const int bit_width, const int block_size,
                             const int sub_block_size,
                             const int sub_block_shift_bits,
                             const int rounding_mode);
// to bfp16, first convert to bf16, to bfp32 then bfp fix, last return to bf16
void to_bfp16(float *T_ptr, const std::uint32_t data_num,
              const std::vector<int64_t> shape, const std::string &bfp_method,
              const std::int64_t &block_size, const std::int64_t &axis,
              const int64_t &sub_block_size,
              const int64_t &sub_block_shift_bits, const int rounding_mode,
              bool convert_to_bfloat_before_bfp) {
  std::vector<float> ret_fp32(data_num);
  if (convert_to_bfloat_before_bfp) {
    for (uint32_t idx = 0u; idx < data_num; idx++) {
      xir::bfloat16_t ret_bf16 = (xir::bfloat16_t)(T_ptr[idx]);
      ret_fp32[idx] = (float)ret_bf16;
    }
    int64_t abs_axis = axis < 0 ? axis + shape.size() : axis;
    auto align_shape = shape;
    align_shape[abs_axis] = align_upper(shape[abs_axis], block_size);
    std::vector<float> align_ret = calc_align_data(ret_fp32, shape, align_shape);
    std::vector<float> ret_bfp(align_ret.size(), 0);
    if (bfp_method == "to_bfp") {
      LaunchBFPCPUKernel(align_ret.data(), ret_bfp.data(), align_ret.size(), 16,
                         block_size, rounding_mode);
    } else if (bfp_method == "to_bfp_prime_shared" ||
               bfp_method == "to_bfp_prime") {
      LaunchBFPPrimeCPUKernel(align_ret.data(), ret_bfp.data(), align_ret.size(),
                              16, block_size, sub_block_size,
                              sub_block_shift_bits, rounding_mode);
    } else if (bfp_method == "none") {
      // pass
    } else {
      UNI_LOG_FATAL(VART_NOT_SUPPORT)
          << "Unsupported bfp_method: " << bfp_method << ".";
    }
    std::vector<int32_t> strides(shape.size(), 1);
    auto ret_slice_bfp = strided_slice_data(ret_bfp, align_shape, shape);
    for (uint32_t idx = 0u; idx < data_num; idx++) {
      T_ptr[idx] = (xir::bfloat16_t)(ret_slice_bfp[idx]);
    }
  } else {
    for (uint32_t idx = 0u; idx < data_num; idx++) {
      ret_fp32[idx] = f_to_bf_aie2p(T_ptr[idx]);
      T_ptr[idx] = (xir::bfloat16_t)(ret_fp32[idx]);
    }
  }
}

void to_bfp16(void *ret_fp32, const FMap_t fmap, const xir::Tensor *tensor,
              int fp32_to_bfp16 = 1) {
  std::vector<int64_t> shape{ fmap.n, fmap.h, fmap.w, fmap.c };
  auto bfp_method = tensor->has_attr("bfp_method")
                        ? tensor->get_attr<std::string>("bfp_method")
                        : "none";
  auto block_size = tensor->has_attr("block_size")
                        ? tensor->get_attr<int64_t>("block_size")
                        : 1;
  auto axis = tensor->has_attr("axis") ? tensor->get_attr<int64_t>("axis") : 0;
  auto sub_block_size = tensor->has_attr("sub_block_size")
                            ? tensor->get_attr<int64_t>("sub_block_size")
                            : 1;
  auto sub_block_shift_bits =
      tensor->has_attr("sub_block_shift_bits")
          ? tensor->get_attr<int64_t>("sub_block_shift_bits")
          : 0;

  auto round_mode = tensor->has_attr("round_mode")
                        ? tensor->get_attr<std::string>("round_mode")
                        : "STD_ROUND";
  std::map<std::string, int> round_mode_to_int{ { "STD_ROUND", 0 },
                                                { "DPU_ROUND", 1 },
                                                { "PY3_ROUND", 2 } };
  UNI_LOG_CHECK(round_mode_to_int.count(round_mode), VART_INVALID_VALUE)
      << "invalid round mode: " << round_mode;

  //auto convert_to_bfloat_before_bfp =
  //    tensor->has_attr("convert_to_bfloat_before_bfp")
  //        ? tensor->get_attr<int64_t>("convert_to_bfloat_before_bfp") == 1
  //        : false;

  to_bfp16(reinterpret_cast<float *>(ret_fp32), fmap.num(), shape, bfp_method,
           block_size, axis, sub_block_size, sub_block_shift_bits,
           round_mode_to_int[round_mode], fp32_to_bfp16);
}

template <typename DType, typename WType> void Conv2dFix<DType, WType>::run() {
  // // do necessary check
  print_param();
  // check_param();

  // // read data
  // this->read();

  if (enable_bfp_) {
    if (CPUOPBase::xir_op_->has_attr("channel_augmentation") &&
        CPUOPBase::xir_op_->get_attr<int32_t>("channel_augmentation") == 3) {
      FMap_t fmap_i_align(fmap_i_.n, fmap_i_.h,
                          align_upper(fmap_i_.w, stride_.w), fmap_i_.c);
      Pad_t pad_align_channel_aug(
          0, 0, align_upper(fmap_i_.w, stride_.w) - fmap_i_.w, 0);
      std::vector<DType> fmap_i_channel_aug_data(fmap_i_align.num());
      FastPad(fmap_i_, fmap_i_align, pad_align_channel_aug, dilation_)
          .transform<DType>(data_in_ptr_, fmap_i_channel_aug_data.data());
      FMap_t fmap_i_channel_aug(fmap_i_.n, fmap_i_.h,
                                align_upper(fmap_i_.w, stride_.w) / stride_.w,
                                (fmap_i_.c * stride_.w));
      UNI_LOG_CHECK(dilation_.w == 1, VART_SIZE_ERROR)
          << "channel augmentation requires dilation w is 1, now is "
          << dilation_.w;
      UNI_LOG_CHECK(fmap_i_align.num() == fmap_i_channel_aug.num(), VART_SIZE_ERROR)
          << "internal tensor size mismatch.";
      to_bfp16(fmap_i_channel_aug_data.data(), fmap_i_channel_aug,
                CPUOPBase::xir_op_->get_input_tensor("input"));
      auto req_channel_aug_data = strided_slice_data(
          fmap_i_channel_aug_data,
          {fmap_i_align.n, fmap_i_align.h, fmap_i_align.w, fmap_i_align.c},
          {fmap_i_.n, fmap_i_.h, fmap_i_.w, fmap_i_.c});
      std::copy_n(req_channel_aug_data.data(), fmap_i_.num(), data_in_ptr_);
    } else {
      to_bfp16(data_in_ptr_, fmap_i_,
              CPUOPBase::xir_op_->get_input_tensor("input"));
    }
  }
  // calc conv
  this->conv();

  // do fix
  this->fix();
  // // do save, debug...
  // this->save();
}

double dr(double x) {
  if (x - std::floor(x) >= 0.5)
    return double(std::min(127.0, std::max(-128.0, double(std::ceil(x)))));
  else
    return double(std::min(127.0, std::max(-128.0, double(std::floor(x)))));
}

template <typename DType, typename WType> void Conv2dFix<DType, WType>::fix() {
  //  if (!enable_conv_dirty_) {
  //    for (auto i = 0; i < fmap_o_.num(); i++) {
  //      data_out_ptr_[i] *= 2;
  //      if (has_bias_) {
  //        // do fix for bias
  //        auto pos = i % fmap_o_.c;
  //        data_out_ptr_[i] += floor(2.0 * bias_ptr_[pos] * pow(2.0,
  //        shift_bias_));
  //      }
  //    }
  //  }
  // do fix for output
  if (enable_bfp_) {
    for (auto i = 0; i < fmap_o_.num(); i++) {
      data_out_ptr_[i] += bias_ptr_[i % fmap_o_.c];
    }
    to_bfp16(data_out_ptr_, fmap_o_, CPUOPBase::xir_op_->get_output_tensor(), 0);
    for (auto i = 0; i < fmap_o_.num(); i++) {
      double tmp = data_out_ptr_[i];
      if (nonlinear_type_ == 1) {
        if (tmp < 0)
          tmp = 0;
      } else if (nonlinear_type_ == 2) {
        if (tmp < 0)
          tmp = tmp * prelu_alpha_;
      } else if (nonlinear_type_ == 3) {
        if (tmp < 0)
          tmp = tmp * prelu_alpha_;
      } else if (nonlinear_type_ == 4) {
        if (tmp < 0)
          tmp = 0;
        if (fp_output_ <= 4) {
          auto thr6 = 6 << 4;
          if (tmp >= thr6)
            tmp = thr6;
        }
      } else if (nonlinear_type_ == 5) {
        tmp = std::min(pow(2, 32),
                       std::max(0.0, (tmp * 2731 +
                                      3 * 2731 * pow(2, hsigmoid_in_)))) *
              pow(2, -shift_hsigmoid_);
      } else if (nonlinear_type_ == 6) {
        auto x = tmp;
        double hsigmoid_x =
            std::min(
                pow(2, 32),
                std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
        hsigmoid_x = double(
            round_normal<DType>(round_mode_, hsigmoid_x, data_min_, data_max_));
        tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
      }
      data_out_ptr_[i] = tmp;
    }
    to_bfp16(data_out_ptr_, fmap_o_, CPUOPBase::xir_op_->get_output_tensor(), 0);
  } else {
    auto factor = pow(2, shift_cut_ + (!enable_conv_dirty_ ? 1 : 0));
    for (auto i = 0; i < fmap_o_.num(); i++) {
      double tmp = data_out_ptr_[i];
      if (!enable_conv_dirty_) {
        tmp *= 2;
        if (has_bias_) {
          // do fix for bias
          auto pos = i % fmap_o_.c;
          tmp += 2.0 * bias_ptr_[pos] * pow(2.0, shift_bias_);
        }
      }
      tmp /= factor;
      if (nonlinear_type_ == 1) {
        if (tmp < 0)
          tmp = 0;
      } else if (nonlinear_type_ == 2) {
        if (tmp < 0)
          tmp = tmp * prelu_alpha_;
      } else if (nonlinear_type_ == 3) {
        if (tmp < 0)
          tmp = tmp * prelu_alpha_;
      } else if (nonlinear_type_ == 4) {
        if (tmp < 0)
          tmp = 0;
        if (fp_output_ <= 4) {
          auto thr6 = 6 << 4;
          if (tmp >= thr6)
            tmp = thr6;
        }
      } else if (nonlinear_type_ == 5) {
        tmp =
            double(round_normal<DType>(round_mode_, tmp, data_min_, data_max_));
        tmp = std::min(pow(2, 32),
                       std::max(0.0, (tmp * 2731 +
                                      3 * 2731 * pow(2, hsigmoid_in_)))) *
              pow(2, -shift_hsigmoid_);
      } else if (nonlinear_type_ == 6) {
        auto x =
            double(round_normal<DType>(round_mode_, tmp, data_min_, data_max_));
        double hsigmoid_x =
            std::min(
                pow(2, 32),
                std::max(0.0, (x * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
        hsigmoid_x = double(
            round_normal<DType>(round_mode_, hsigmoid_x, data_min_, data_max_));
        tmp = x * hsigmoid_x * pow(2, -shift_hswish_);
      }
      data_out_ptr_[i] =
          round_normal<DType>(round_mode_, tmp, data_min_, data_max_);
    }
  }
}

INSTANTIATE_TPCLASS_SPECIFIED_WEIGHTS(Conv2dFix);
REG_SPECIFIED_WEIGHTS_DATATYPE_OP_INSTANCE_FUNC("conv2d-fix", Conv2dFix,
                                                "weights");

} // namespace cpu
} // namespace vart
