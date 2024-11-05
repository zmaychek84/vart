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

#include "qlinear_eltwise.hpp"

namespace vart {
namespace cpu {
namespace {
int32_t round_even(int64_t n, int shift, int32_t data_max = 127,
                   int32_t data_min = -128) {
  int64_t mask = 1LL << (shift - 1);
  int64_t integer_part = n >> shift;
  // int i = 0;
  // for (; std::abs(integer_part) >> i; i++)
  //   ;
  // auto int_shift = shift - (23 - i);
  if ((n & mask)) {  // >=0.5
    // int64_t fractional_mask = ((mask - 1) >> int_shift) << int_shift;
    int64_t fractional_mask = mask - 1;
    if (!(std::abs(n) & fractional_mask)) {  //==0.5
      if (integer_part % 2) {
        integer_part++;
      }
    } else {
      integer_part++;
    }
  }
  int64_t data_max_int64 = static_cast<std::int64_t>(data_max);
  int64_t data_min_int64 = static_cast<std::int64_t>(data_min);
  if (integer_part > data_max_int64) integer_part = data_max_int64;
  if (integer_part < data_min_int64) integer_part = data_min_int64;
  return static_cast<int32_t>(integer_part);
}

float truncate_to_bfloat16(float inp) {
  int32_t temp;
  std::memcpy(&temp, &inp, sizeof(float));
  int32_t out = temp & 0xFFFF0000;
  float temp1;
  std::memcpy(&temp1, &out, sizeof(float));
  return temp1;
}

float approx_tanh(float inp1) {
  float inp = truncate_to_bfloat16(inp1);
  // cout<< inp1 <<endl;
  float out;
  if (inp >= 3.0)
    out = 1.0;
  else if (inp >= 2.0)
    out = 0.03125 * inp + 0.90625;
  else if (inp >= 1.75)
    out = 0.125 * inp + 0.71875;
  else if (inp >= 1.5)
    out = 0.125 * inp + 0.72265625;
  else if (inp >= 1.25)
    out = 0.25 * inp + 0.53515625;
  else if (inp >= 1.0)
    out = 0.25 * inp + 0.52734375;
  else if (inp >= 0.75)
    out = 0.5 * inp + 0.265625;
  else if (inp >= 0.625)
    out = 0.5 * inp + 0.25390625;
  else if (inp >= 0.5)
    out = 0.5 * inp + 0.25;
  else if (inp > -0.5)
    out = inp;
  else if (inp > -0.625)
    out = 0.5 * inp - 0.25;
  else if (inp > -0.75)
    out = 0.5 * inp - 0.25390625;
  else if (inp > -1.0)
    out = 0.5 * inp - 0.265625;
  else if (inp > -1.25)
    out = 0.25 * inp - 0.52734375;
  else if (inp > -1.5)
    out = 0.25 * inp - 0.53515625;
  else if (inp > -1.75)
    out = 0.125 * inp - 0.72265625;
  else if (inp > -2.0)
    out = 0.125 * inp - 0.71875;
  else if (inp > -3.0)
    out = 0.03125 * inp - 0.90625;
  else
    out = -1.0;
  return truncate_to_bfloat16(out);
}

typedef union value_convert {
  std::uint32_t u;
  float f;
} value_convert_t;

std::uint32_t f_to_u(float data) {
  value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

float u_to_f(std::uint32_t data) {
  value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

float f_to_bf(float data) {
  std::uint32_t u = f_to_u(data);
  std::uint32_t flag = (u & 0x00010000) >> 16;
  u = (u + 0x7fff + flag) & 0xFFFF0000;
  return u_to_f(u);
}

float fast_div(float z0, float z1) {
  if (z0 == 0 && z1 == 0) return 0;
  std::uint32_t magic_n = 0x7eb50000;
  float k1 = f_to_bf(1.9395974);
  float k2 = f_to_bf(1.436142);
  float ones = f_to_bf(1.0f);
  float x = f_to_bf(z1);
  float y = u_to_f(magic_n - f_to_u(x));
  float acc = f_to_bf(-x * y + k2);
  acc = f_to_bf(acc * y);
  acc = f_to_bf(acc * k1);
  y = acc;
  acc = f_to_bf(acc * -x + ones);
  acc = f_to_bf(acc * y + y);
  y = f_to_bf(z0) * acc;
  return f_to_bf(y);
}
}  // namespace

template <typename DType>
const vector<string> QLinearEltwise<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
QLinearEltwise<DType>::QLinearEltwise(const xir::Subgraph* subg,
                                      const xir::Op* op, IMapTBs_t inputs,
                                      CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAPS(fmap_i_, input, 5);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 5);

  input_num_ = fmap_i_.size();
  if (op->has_attr("type")) {
    elt_type_ = op->get_attr<std::string>("type");
    std::for_each(elt_type_.begin(), elt_type_.end(),
                  [](char& c) { c = std::toupper(c); });
  }

  broadcast_ =
      std::any_of(fmap_i_.begin(), fmap_i_.end(),
                  [this](const Dimension map_i) { return fmap_o_ != map_i; });

  auto input_ops = xir_op_->get_input_ops("input");
  if (xir_op_->has_attr("c0_int") && xir_op_->has_attr("c1_int") &&
      xir_op_->has_attr("c2_int") && xir_op_->has_attr("c3_int") &&
      xir_op_->has_attr("shift_c0") && xir_op_->has_attr("shift_c1") &&
      xir_op_->has_attr("shift_c2") && xir_op_->has_attr("shift_c3")) {
    C_0 = xir_op_->template get_attr<int32_t>("c0_int");
    C_1 = xir_op_->template get_attr<int32_t>("c1_int");
    C_2 = xir_op_->template get_attr<int32_t>("c2_int");
    C_3 = xir_op_->template get_attr<int32_t>("c3_int");

    FP_0 = xir_op_->template get_attr<int32_t>("shift_c0");
    FP_1 = xir_op_->template get_attr<int32_t>("shift_c1");
    FP_2 = xir_op_->template get_attr<int32_t>("shift_c2");
    FP_3 = xir_op_->template get_attr<int32_t>("shift_c3");
  } else if (xir_op_->has_attr("a_zero_point") &&
             xir_op_->has_attr("b_zero_point") &&
             xir_op_->has_attr("y_zero_point") &&
             xir_op_->has_attr("a_scale") && xir_op_->has_attr("b_scale") &&
             xir_op_->has_attr("y_scale")) {
    std::int32_t a_zp =
        xir_op_->get_attr<std::vector<std::int32_t>>("a_zero_point").front();
    std::int32_t b_zp =
        xir_op_->get_attr<std::vector<std::int32_t>>("b_zero_point").front();
    std::int32_t y_zp =
        xir_op_->get_attr<std::vector<std::int32_t>>("y_zero_point").front();
    float a_s = xir_op_->get_attr<std::vector<float>>("a_scale").front();
    float b_s = xir_op_->get_attr<std::vector<float>>("b_scale").front();
    float y_s = xir_op_->get_attr<std::vector<float>>("y_scale").front();
    float c0_f = 0;
    float c1_f = 0;
    float c2_f = 0;
    float c3_f = 0;
    // 0, 1, 2, 5, 28, refer to KernelParamGenPass
    if ("ADD" == elt_type_) {
      c0_f = y_s / a_s;
      c1_f = y_s / b_s;
      c2_f = -(a_zp / a_s + b_zp / b_s) * y_s + y_zp;
    } else if ("MUL" == elt_type_) {
      c0_f = y_s / (a_s * b_s);
      c1_f = -c0_f * b_zp;
      c2_f = -c0_f * a_zp;
      c3_f = c0_f * (b_zp * a_zp) + y_zp;
    } else if ("SUB" == elt_type_) {
      c0_f = y_s / a_s;
      c2_f = y_zp - a_zp * y_s / a_s;
      c1_f = y_zp;
    } else if ("DIV" == elt_type_) {
      c0_f = y_s * b_s / a_s;
    } else if ("REQUANTIZE" == elt_type_) {
      c0_f = y_s / a_s;
      c2_f = y_zp - a_zp * y_s / a_s;
    } else if ("RELU" == elt_type_) {
      c0_f = y_s / a_s;
      c2_f = y_zp - a_zp * y_s / a_s;
      c1_f = y_zp;
    } else if ("LEAKY-RELU" == elt_type_) {
      if ("BFLOAT16" == xir_op_->get_output_tensor()->get_data_type().to_string() ) {
        leaky_relu_alpha = xir_op_->get_attr<float>("LEAKYRELU_alpha");
      } else { 
        c0_f = y_s / a_s;
        c1_f = xir_op_->get_attr<float>("alpha") * c0_f;
      }  
    }
    {
      // Compute C0 shift
      std::int32_t c0_qint = convert_float_to_qint(c0_f);
      FP_0 = get_shift_from_int32_rep(c0_qint);
      // convert float to int based on shift
      C_0 = get_int32_representation_from_float(c0_f);
      // Compute C1 shift
      std::int32_t c1_qint = convert_float_to_qint(c1_f);
      FP_1 = get_shift_from_int32_rep(c1_qint);
      // convert float to int based on shift
      C_1 = get_int32_representation_from_float(c1_f);
      // Compute C2 shift
      std::int32_t c2_qint = convert_float_to_qint(c2_f);
      FP_2 = get_shift_from_int32_rep(c2_qint);
      // convert float to int based on shift
      C_2 = get_int32_representation_from_float(c2_f);

      std::int32_t c3_qint = convert_float_to_qint(c3_f);
      FP_3 = get_shift_from_int32_rep(c3_qint);
      // convert float to int based on shift
      C_3 = get_int32_representation_from_float(c3_f);
    }
    // do not affect calculation of minimum shift
    if ("REQUANTIZE" == elt_type_) {
      FP_1 = FP_0;
    }
  } else {
    UNI_LOG_CHECK(0, VART_INVALIDE_PARAM)
        << "missing c0_int, c1_int, c2_int, shift_c0, shift_c1, shift_c2 or "
           "a_zero_point, b_zero_point, y_zero_point, a_scale, b_scale, "
           "y_scale";
  }

  THREAD_NUM = CPU_NUM;
  THREAD_WORKLOAD = ceil((float)fmap_o_.num() / THREAD_NUM);
}

template <typename DType>
void QLinearEltwise<DType>::align_dim() {
  for (auto i = 0; i < input_num_; ++i) {
    while (fmap_i_[i].ndims() != fmap_o_.ndims()) {
      fmap_i_[i].insert_dim(0, 1);
    }
  }
}

template <typename DType>
void QLinearEltwise<DType>::run() {
  // if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
  //     CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
  //    eltwise_thread();
  //    todo
  // } else {
  //   eltwise_normal();
  // }
  eltwise_normal();
}

template <typename DType>
void QLinearEltwise<DType>::print_param() {
  for (auto i = 0; i < input_num_; i++) {
    fmap_i_[i].print_param("fmap_i" + to_string(i));
  }
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "elt_type = " << elt_type_ << endl;
  UNI_LOG_DEBUG_INFO << "broadcast = " << broadcast_ << endl;
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    UNI_LOG_DEBUG_INFO << "THREAD_NUM = " << THREAD_NUM << endl;
    UNI_LOG_DEBUG_INFO << "THREAD_WORKLOAD = " << THREAD_WORKLOAD << endl;
  }
  UNI_LOG_DEBUG_INFO << "c0_int: " << C_0 << endl;
  UNI_LOG_DEBUG_INFO << "c1_int: " << C_1 << endl;
  UNI_LOG_DEBUG_INFO << "c2_int: " << C_2 << endl;
  UNI_LOG_DEBUG_INFO << "c3_int: " << C_3 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c0: " << FP_0 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c1: " << FP_1 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c2: " << FP_2 << endl;
  UNI_LOG_DEBUG_INFO << "shift_c3: " << FP_3 << endl;
  UNI_LOG_DEBUG_INFO << "data_min_: " << data_min_ << endl;
  UNI_LOG_DEBUG_INFO << "data_max_: " << data_max_ << endl;
}

template <typename DType>
void QLinearEltwise<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() >= 1, VART_SIZE_ERROR)
      << ", Err: inputs size is " << inputs_.size();
  for (auto dim_iter = 0; dim_iter < fmap_o_.ndims(); dim_iter++) {
    for (auto fp_iter = 0U; fp_iter <= fmap_i_.size(); fp_iter++) {
      if ((fmap_o_[dim_iter] != fmap_i_[fp_iter][dim_iter])) {
        UNI_LOG_CHECK(fmap_i_[fp_iter][dim_iter] == 1, VART_INVALID_VALUE);
      }
    }
  }
}

// read data from DDR
template <typename DType>
void QLinearEltwise<DType>::read() {
  // handle input buffer
  data_in_.resize(input_num_);
  for (auto idx = 0; idx < input_num_; idx++) {
    auto* cputb = inputs_.at(ITName[INPUT]).at(idx);
    data_in_[idx] = GET_CPUTB_DType_PTR(DType, cputb);
  }

  // handle output buffer
  data_out_ = GET_CPUTB_DType_PTR(DType, output_);

  if ("MUL" == elt_type_) {
    std::fill_n(data_out_,  fmap_o_.num(), 1);
    // todo
  } else {
    std::fill_n(data_out_, fmap_o_.num(), 0);
  }
}

template <typename DType>
uint64_t QLinearEltwise<DType>::get_workload() {
  return fmap_o_.num();
}

std::int64_t right_shift(std::int64_t val, std::int32_t shift_val) {
  if (shift_val >= 0) {
    return val >> shift_val;
  }
  return val << (-1 * shift_val);
}

template <typename DType>
void QLinearEltwise<DType>::eltwise(std::uint32_t start_index,
                                    std::uint32_t end_index) {
  read();
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  std::vector<std::int32_t> FP{FP_0 - FP_2, FP_1 - FP_2};
  std::vector<std::int32_t> C{C_0, C_1, C_2, C_3};
  int fpmax = 0;
  std::vector<std::int32_t> fpmax_candidate{FP_0, FP_1, FP_2, FP_3};
  std::vector<std::int32_t> FP_minus_max{FP_0, FP_1, FP_2, FP_3};
  for (int i = 0; i < 3; i++) {
    if (fpmax_candidate[i] > fpmax) {
      if (fpmax_candidate[i] > 64 && C[i] == 0) {
        continue;
      } else {
        fpmax = fpmax_candidate[i];
      }
    }
  }
  for (int i = 0; i < 3; i++) {
    if (fpmax_candidate[i] > 64 && C[i] == 0) {
      continue;
    } else {
      FP_minus_max[i] = fpmax - fpmax_candidate[i];
    }
  }

  int cmin = std::min({FP_0, FP_1, FP_2, FP_3});
  std::vector<std::int32_t> FP1{FP_0 - cmin, FP_1 - cmin, FP_2 - cmin};
  for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
    std::int64_t temp_add = 0;
    // float temp_add_bf16 = 0.0;
    std::int64_t temp_add1 = 1;
    std::int64_t temp_relu = 0;
    std::int64_t final_sum = 0;
    for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
      auto pos = broadcast_pos(pos_iter,input_iter);
      if ("MUL" == elt_type_) {
        int64_t zp1 = xir_op_->template get_attr<int32_t>("a_zero_point");
        int64_t zp2 = xir_op_->template get_attr<int32_t>("b_zero_point");
        int final_shift = (FP_0 > FP_1) ? (FP_0 - FP_1) : 0;
        int c1_sft = (FP_0 > FP_1) ? 0 : (FP_1 - FP_0);
        if (FP1[input_iter] >= 0) {
          if (input_iter == 0) {
            temp_add1 *= static_cast<std::int64_t>(data_in_[input_iter][pos]);
            temp_add = zp2 * data_in_[input_iter][pos];
          }
          if (input_iter == 1) {
            temp_add1 *= static_cast<std::int64_t>(data_in_[input_iter][pos]);
            temp_add += zp1 * data_in_[input_iter][pos];
            int64_t final_sum1 = temp_add1 - temp_add;
            int64_t co_mul =
                int64_t(round_even(final_sum1, 0, INT32_MAX, INT32_MIN)) * C_0;
            if (bit_width_ == 8) {
              auto c1 = static_cast<std::int64_t>(C_1) << final_shift;
              final_sum = co_mul + c1;
            } else {
              auto c1 = round_even(C_1, c1_sft, INT32_MAX, INT32_MIN);
              final_sum = int64_t(round_even(co_mul, final_shift, INT32_MAX,
                                             INT32_MIN)) +
                          c1;
            }
          }
        }
      } else if ("RELU" == elt_type_) {
        int FP_min = std::min(FP_0, FP_2);
        temp_relu = (static_cast<std::int64_t>(data_in_[input_iter][pos]) *
                     (static_cast<std::int64_t>(C_0) >> (FP_0 - FP_min))) +
                    (static_cast<std::int64_t>(C_2) >> (FP_2 - FP_min));
        int tmp1 = round_even(static_cast<std::int64_t>(temp_relu), FP_min,
                              data_max_, data_min_);
        int tmp2 = round_even(static_cast<std::int64_t>(C_1), FP_1, data_max_,
                              data_min_);
        if (tmp1 < tmp2) {
          data_out_[pos_iter] = tmp2;
        } else {
          data_out_[pos_iter] = tmp1;
        }
      } else if ("LEAKY-RELU" == elt_type_) {
        int a_zp = xir_op_->get_attr<std::vector<int>>("a_zero_point").front();
        int y_zp = xir_op_->get_attr<std::vector<int>>("y_zero_point").front();
        auto data_max = data_max_ - y_zp;
        auto data_min = data_min_ - y_zp;

        int64_t tmp_0 = data_in_[input_iter][pos] - a_zp;
        auto v1 = round_even(C_0 * tmp_0, FP_0, data_max, data_min) + y_zp;
        auto v2 = round_even(C_1 * tmp_0, FP_1, data_max, data_min) + y_zp;
        data_out_[pos_iter] = tmp_0 > 0 ? v1 : v2;
      } else if ("ADD" == elt_type_) {
        if (FP_minus_max[input_iter] >= 0) {
          temp_add += ((static_cast<std::int64_t>(C[input_iter])
                        << FP_minus_max[input_iter]) *
                       static_cast<std::int64_t>(data_in_[input_iter][pos]));
        } else {
          temp_add += ((static_cast<std::int64_t>(C[input_iter]) >>
                        (-1 * FP_minus_max[input_iter])) *
                       static_cast<std::int64_t>(data_in_[input_iter][pos]));
        }
      } else if ("SUB" == elt_type_) {
        if (0 == input_iter) {
          temp_add = (right_shift(static_cast<std::int64_t>(C[input_iter]),
                                  FP1[input_iter]) *
                      static_cast<std::int64_t>(data_in_[input_iter][pos]));
        } else {
          temp_add -= (right_shift(static_cast<std::int64_t>(C[input_iter]),
                                   FP1[input_iter]) *
                       static_cast<std::int64_t>(data_in_[input_iter][pos]));
          temp_add += right_shift(static_cast<std::int64_t>(C_2), FP1[2]);
        }
      } else if ("REQUANTIZE" == elt_type_) {
        if (FP_2 - FP_0 >= 0) {
          temp_add = (static_cast<std::int64_t>(C_0) *
                      static_cast<std::int64_t>(data_in_[input_iter][pos])) +
                     (static_cast<std::int64_t>(C_2) >> (FP_2 - FP_0));
        } else {
          temp_add = (static_cast<std::int64_t>(C_0) *
                      static_cast<std::int64_t>(data_in_[input_iter][pos])) +
                     (static_cast<std::int64_t>(C_2) << (FP_0 - FP_2));
        }
      } else if ("TANH" == elt_type_) {
        // get attrs
        int a_zp = xir_op_->get_attr<std::vector<int>>("a_zero_point").front();
        int y_zp = xir_op_->get_attr<std::vector<int>>("y_zero_point").front();
        float a_s = xir_op_->get_attr<std::vector<float>>("a_scale").front();
        float y_s = xir_op_->get_attr<std::vector<float>>("y_scale").front();
        if (bit_width_ == 8) {
          float y_lower = float(data_min_) - y_zp;
          float y_upper = float(data_max_) - y_zp;
          float res = tanh(float(data_in_[input_iter][pos] - a_zp) / a_s) * y_s;
          data_out_[pos_iter] =
              round_to_even<DType>(res, y_lower, y_upper) + y_zp;
        } else {
          // inv_scale_in
          float sxx = f_to_bf(1.0f / a_s);
          // scale_out
          float syy = f_to_bf(y_s);
          // DQ
          float xf = (float)(data_in_[input_iter][pos] - a_zp);
          xf = f_to_bf(xf) * (sxx);
          // Tanh
          float out = approx_tanh(xf);
          // Q
          out = round_to_even<int32_t>(f_to_bf(out * syy)) + y_zp;
          // rnd and saturate
          data_out_[pos_iter] = round_to_even<DType>(out, data_min_, data_max_);
        }
      } else if ("DIV" == elt_type_) {
        int32_t zp1 = xir_op_->template get_attr<int32_t>("a_zero_point");
        int32_t zp2 = xir_op_->template get_attr<int32_t>("b_zero_point");
        int32_t zp3 = xir_op_->template get_attr<int32_t>("y_zero_point");
        float coeff = xir_op_->template get_attr<float>("coeff0_f");

        if (0 == input_iter) {
          temp_add = static_cast<std::int64_t>(data_in_[input_iter][pos]);
        } else {
          auto temp2 = static_cast<std::int64_t>(data_in_[input_iter][pos]);
          float tmp2 = f_to_bf(coeff) *
                       fast_div(float(temp_add - zp1), float(temp2 - zp2));
          data_out_[pos_iter] = round_to_even<DType>(
              round_to_even<int32_t>(tmp2) + zp3, data_min_, data_max_);
        }
      }
    }
    if ("REQUANTIZE" == elt_type_) {
      data_out_[pos_iter] = round_even(temp_add, FP_0, data_max_, data_min_);
    } else if ("ADD" == elt_type_) {
      temp_add += (static_cast<std::int64_t>(C_2) << FP_minus_max[2]);
      data_out_[pos_iter] = round_even(temp_add, fpmax, data_max_, data_min_);
    } else if ("SUB" == elt_type_) {
      data_out_[pos_iter] = round_even(temp_add, cmin, data_max_, data_min_);
    } else if ("MUL" == elt_type_) {
      if (8 == bit_width_)
        data_out_[pos_iter] = round_even(final_sum, FP_0, data_max_, data_min_);
      else
        data_out_[pos_iter] = round_even(final_sum, cmin, data_max_, data_min_);
    }
  }
}

template <typename DType>
uint32_t QLinearEltwise<DType>::broadcast_pos(uint32_t pos_iter,
                                              int input_iter) {
  auto get_sub_coord = [](coord_t coord, int start) {
    return coord_t(coord.begin() + start, coord.end());
  };
  auto pos = pos_iter;
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  if (broadcast_) {
    fmap_o_.pos2coord(pos_iter, dst_coord);
    int delta = 0;
    for (auto dim_iter = 0U; dim_iter < dst_coord.size(); dim_iter++) {
      if (fmap_i_[input_iter].ndims() < int(dst_coord.size())) {
        delta = dst_coord.size() - fmap_i_[input_iter].ndims();
      }
      if (int(dim_iter) >= delta)
        src_coord[dim_iter] =
            dst_coord[dim_iter] % fmap_i_[input_iter][dim_iter - delta];
    }
    pos = fmap_i_[input_iter].coord2pos(get_sub_coord(src_coord, delta));
  }
  return pos;
}

template <>
void QLinearEltwise<float>::eltwise(std::uint32_t start_index,
                                    std::uint32_t end_index) {
  read();
  if ("ADD" == elt_type_) {
    auto dst_coord = fmap_o_.pos2coord(0);
    auto src_coord = dst_coord;
    for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
      for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
        auto pos = broadcast_pos(pos_iter, input_iter);
        data_out_[pos_iter] += data_in_[input_iter][pos];
      }
      data_out_[pos_iter] = f_to_bf_aie2p(data_out_[pos_iter]);
      if (data_out_[pos_iter] == -0) {
        data_out_[pos_iter]  = 0.0f;
      }
      if (xir_op_->has_attr("nonlinear") &&
          xir_op_->get_attr<std::string>("nonlinear") == "LEAKYRELU" &&
          xir_op_->has_attr("LEAKYRELU_alpha")) {
            auto alpha = xir_op_->get_attr<float>("LEAKYRELU_alpha");
            if (data_out_[pos_iter] < 0) {
              data_out_[pos_iter] = f_to_bf_aie2p(data_out_[pos_iter] * alpha);
            }
      }
    }
  } 
  else if ("LEAKY-RELU" == elt_type_) {
    for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
      auto pos = pos_iter;
      for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
        if ("BFLOAT16" == xir_op_->get_output_tensor()->get_data_type().to_string() ) {
          float tmp_0 = data_in_[input_iter][pos];
          data_out_[pos_iter] = tmp_0 > 0 ? tmp_0 : f_to_bf_aie2p(tmp_0*leaky_relu_alpha);
        } else {
          int a_zp = xir_op_->get_attr<std::vector<int>>("a_zero_point").front();
          int y_zp = xir_op_->get_attr<std::vector<int>>("y_zero_point").front();
          auto data_max = data_max_ - y_zp;
          auto data_min = data_min_ - y_zp;
          int64_t tmp_0 = data_in_[input_iter][pos] - a_zp;
          auto v1 = round_even(C_0 * tmp_0, FP_0, data_max, data_min) + y_zp;
          auto v2 = round_even(C_1 * tmp_0, FP_1, data_max, data_min) + y_zp;
          data_out_[pos_iter] = tmp_0 > 0 ? v1 : v2;
        }
      } 
    }
  }
  else if ("L2NORM" == elt_type_) {
    UNI_LOG_CHECK(input_num_==1,VART_NOT_SUPPORT) << this->get_name()<<" len(input) should be 1" << endl;
    UNI_LOG_CHECK(this->get_xir_op()->has_attr("epsilon"),VART_NOT_SUPPORT) << this->get_name()<<" need attr:epsilon" << endl;
    UNI_LOG_CHECK(this->get_xir_op()->has_attr("axis"),VART_NOT_SUPPORT) << this->get_name()<<" need attr:axis" << endl;
    float eps = static_cast<float>(this->get_xir_op()->get_attr<double>("epsilon"));
    std::vector<int> axis_ = CPUOPBase::xir_op_->get_attr<std::vector<int>>("axis");

    std::vector<float> data_in_buf_;
    data_in_buf_.resize(fmap_i_[0].num());
    float* data_in_buf_ptr_ = data_in_buf_.data();
    std::copy_n(data_in_[0], fmap_i_[0].num(), data_in_buf_ptr_);
    for (auto k = 0U; k < axis_.size(); k++) {
      auto dim = axis_[k];
      auto cur_dim = fmap_i_[0].dim(dim);
      auto cur_cod = fmap_i_[0].cod(dim);
      auto chunk_size = cur_dim * cur_cod;
      auto n_chunk = fmap_i_[0].num() / chunk_size;
      std::vector<std::vector<float>> acc_buf(cur_cod, std::vector<float>(16, 0));    

      for (auto i = 0; i < n_chunk; i++) {
        float* prlt = data_in_buf_ptr_ + i * cur_dim * cur_cod;
        // init: first element's square
        for (auto k = 0; k < cur_cod; k++) {
          for (auto l = 0; l < 16; l++) {
            acc_buf[k][l] = 0;
          }
          // prlt[k] *= prlt[k];
          acc_buf[k][0] = f_to_bf(prlt[k]) * f_to_bf(prlt[k]);
        }

        // accumulate
        for (auto j = 1; j < cur_dim; j++) {
          float* pcur = prlt + j * cur_cod;
          for (auto k = 0; k < cur_cod; k++) {
            acc_buf[k][j % 16] += f_to_bf(pcur[k]) * f_to_bf(pcur[k]);
          }
        }

        for (auto k = 0; k < cur_cod; k++) {
          prlt[k] = 0;
          for (auto l = 0; l < 16; l++) {
            prlt[k] += f_to_bf(acc_buf[k][l]);
          }
          prlt[k] = f_to_bf(prlt[k]);
        }
      }
    }

    // copy calculation result from data_in_buf_ptr_ into data_out_ptr_
    for (auto pos = 0; pos < fmap_o_.num(); pos++) {
      auto coord = fmap_o_.pos2coord(pos);
      for (auto k = 0U; k < axis_.size(); k++) {
        auto dim = axis_[k];
        coord[dim] = 0;
      }
      auto src_pos = fmap_i_[0].coord2pos(coord);
      float tmp = f_to_bf((float)data_in_[0][pos]);
      float res;
      if (data_in_buf_ptr_[src_pos] < eps) {
        res = f_to_bf_aie2p(f_to_bf(tmp) * f_to_bf(1 / sqrt(eps)));
      } else {
        res = f_to_bf_aie2p(f_to_bf(tmp) * f_to_bf(1 / sqrt(data_in_buf_ptr_[src_pos])));
      }
      data_out_[pos] = tmp == -0.0f ? 0 : res;
    }
  }
  else if ("MAX" == elt_type_) {
    for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
      data_out_[pos_iter] = data_in_[0][pos_iter];
      for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
         data_out_[pos_iter] = max_aie2p(data_out_[pos_iter],data_in_[input_iter][pos_iter]);
      }
    }
  }
  else if ("MUL" == elt_type_) {
    bool skip_convert = false;

    std::array<std::uint32_t, 2> u32_in;
    std::uint32_t u32_out = 0;

    for (auto pos_iter = start_index; pos_iter < end_index; ++pos_iter) {
      skip_convert = false;
      for (auto input_iter = 0; input_iter < input_num_; input_iter++) {
        auto pos = broadcast_pos(pos_iter, input_iter);
        u32_in[input_iter] = bit_cast<uint32_t>(data_in_[input_iter][pos]);
        data_out_[pos_iter] *= data_in_[input_iter][pos];
      }

      u32_out = bit_cast<uint32_t>(data_out_[pos_iter]);

      if (((u32_out & 0x7FFFFFFF) == 0x00000000) &&
          ((u32_in[0] & 0x7FFFFFFF) != 0x00000000) &&
          ((u32_in[1] & 0x7FFFFFFF) != 0x00000000)) {
          skip_convert = true;
      } else {
          skip_convert = false;
      }

      if (skip_convert == false) {
        data_out_[pos_iter] = f_to_bf_aie2p(data_out_[pos_iter]);
      }
    }
  } 
}

template <typename DType>
void QLinearEltwise<DType>::eltwise_normal() {
  eltwise(0, fmap_o_.num());
}

INSTANTIATE_TPCLASS(QLinearEltwise);
REG_OP_INSTANCE_FUNC("qlinear-eltwise", QLinearEltwise);

}  // namespace cpu
}  // namespace vart
