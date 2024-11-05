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

#include "resize_bilinear_fix.hpp"

namespace vart {
namespace cpu {

template <typename DType>
ResizeBilinearFix<DType>::ResizeBilinearFix(const xir::Subgraph* subg, const xir::Op* op,
                            IMapTBs_t inputs, CPUTBPtr_t output)
    : Resize<DType>(subg, op, inputs, output) {
  auto xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input");
  auto xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  fp_input_ = xir_tensor_i->get_attr<int>("fix_point");
  fp_output_ = xir_tensor_o->get_attr<int>("fix_point");
  shift_ = fp_output_ - fp_input_;
}

struct CachedInterpolation {
  int lower;
  int upper;
  double lerp;
};

template <typename DType>
void ResizeBilinearFix<DType>::resize_bilinear_fix() {
  auto cal_scale = [](std::int64_t in, std::int64_t out,
                      bool align_corners) -> double {
    return (align_corners && out > 1) ? (in - 1) / static_cast<double>(out - 1)
                                      : in / static_cast<double>(out);
  };
  // The calculation of scaler is slightly different bewteen nearest neighbor
  // and linear/bilinear/trilinear when half_pixel_corner is True. In the
  // nearest neighbor mode, tf uses floor((x + 0.5) * s) instead of round((x +
  // 0.5) * s - 0.5). In the linear mode, coordinates are used to derive the
  // interpolation coefficients. tf uses ((x + 0.5) * s - 0.5) directly.
  auto scaler = [](std::int32_t out, double scale, bool half_pixel_centers,
                   double half_pixel_bias) -> double {
    return (half_pixel_centers)
               ? (static_cast<double>(out) + 0.5f) * scale - half_pixel_bias
               : static_cast<double>(out) * scale;
  };
  auto compute_interpolation_weights =
      [&](const int out_size, const int in_size, const double scale,
          CachedInterpolation* interpolation) {
        interpolation[out_size].lower = 0;
        interpolation[out_size].upper = 0;
        for (int i = out_size - 1; i >= 0; --i) {
          const double in = scaler(i, scale, half_pixel_centers_, 0.5);
          interpolation[i].lower =
              std::max(static_cast<int>(std::floor(in)), static_cast<int>(0));
          interpolation[i].upper = std::min(static_cast<int>(std::ceil(in)),
                                            static_cast<int>(in_size - 1));
          interpolation[i].lerp = in - std::floor(in);
        }
      };
  std::vector<CachedInterpolation> xs(fmap_o_.w + 1);
  std::vector<CachedInterpolation> ys(fmap_o_.h + 1);
  auto h_scale = cal_scale(fmap_i_.h, fmap_o_.h, align_corners_);
  auto w_scale = cal_scale(fmap_i_.w, fmap_o_.w, align_corners_);
  compute_interpolation_weights(fmap_o_.h, fmap_i_.h, h_scale, ys.data());
  compute_interpolation_weights(fmap_o_.w, fmap_i_.w, w_scale, xs.data());
  for (auto& x : xs) {
    x.lower *= fmap_i_.c;
    x.upper *= fmap_i_.c;
  }
  const int in_row_size = fmap_i_.w * fmap_i_.c;
  const int in_batch_num_values = fmap_i_.h * in_row_size;
  const int out_row_size = fmap_o_.w * fmap_o_.c;

  auto compute_lerp = [&](const int top_left, const int top_right,
                          const int bottom_left, const int bottom_right,
                          const double x_lerp, const double y_lerp) -> int {
    auto wgt_left =
        (uint16_t)std::round(std::max(0.0, 1.0 - x_lerp) * upshift_factor);
    auto wgt_right = (uint16_t)std::round(
        std::max(0.0, 1.0 - std::abs(x_lerp - 1.0)) * upshift_factor);
    auto wgt_top =
        (uint16_t)std::round(std::max(0.0, 1.0 - y_lerp) * upshift_factor);
    auto wgt_bottom = (uint16_t)std::round(
        std::max(0.0, 1.0 - std::abs(y_lerp - 1.0)) * upshift_factor);
    auto left = round_normal<int16_t>(
        round_mode_, (double)(top_left * wgt_top + bottom_left * wgt_bottom) /
                         pow(2., shift_0));
    auto right = round_normal<int16_t>(
        round_mode_, (double)(top_right * wgt_top + bottom_right * wgt_bottom) /
                         pow(2., shift_0));
    return round_normal<DType>(round_mode_,
                               (double)(left * wgt_left + right * wgt_right) /
                                   pow(2., shift_1 - shift_));
  };
  DType* in_ptr = data_in_ptr_;
  int idx_start = 0;
  for (int n = 0; n < fmap_o_.n; ++n) {
    for (int h = 0; h < fmap_o_.h; ++h) {
      const DType* ys_input_lower_ptr = in_ptr + ys[h].lower * in_row_size;
      const DType* ys_input_upper_ptr = in_ptr + ys[h].upper * in_row_size;
      const double ys_lerp = ys[h].lerp;
      for (int w = 0; w < fmap_o_.w; ++w) {
        auto xs_lower = xs[w].lower;
        auto xs_upper = xs[w].upper;
        auto xs_lerp = xs[w].lerp;
        for (int c = 0; c < fmap_o_.c; ++c) {
          const int top_left(ys_input_lower_ptr[xs_lower + c]);
          const int top_right(ys_input_lower_ptr[xs_upper + c]);
          const int bottom_left(ys_input_upper_ptr[xs_lower + c]);
          const int bottom_right(ys_input_upper_ptr[xs_upper + c]);
          auto idx = w * fmap_i_.wcod() + c + idx_start;
          data_out_ptr_[idx] = compute_lerp(top_left, top_right, bottom_left,
                                            bottom_right, xs_lerp, ys_lerp);
        }
      }
      idx_start += out_row_size;
    }
    in_ptr += in_batch_num_values;
  }
}  // bilinear resize-fix

template <typename DType>
void ResizeBilinearFix<DType>::run() {
  this->resize_bilinear_fix();
}

template <typename DType>
void ResizeBilinearFix<DType>::print_param() {
  Resize<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "fix_width = " << fix_width_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_input = " << fp_input_ << endl;
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  UNI_LOG_DEBUG_INFO << "shift = " << shift_ << endl;
}

INSTANTIATE_TPCLASS(ResizeBilinearFix);
REG_OP_INSTANCE_FUNC("resize-fix", ResizeBilinearFix);

}  // namespace cpu
}  // namespace vart
