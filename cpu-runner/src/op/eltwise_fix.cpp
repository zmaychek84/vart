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

#include "eltwise_fix.hpp"
#include <array>

namespace vart {
namespace cpu {
#define SUBSTITUTE_FOR_0 0.000000000001
namespace {
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

float fast_sqrt(float z) {
  std::uint32_t magic_n = 0x5f370000;
  z = std::max(z, 0.0f);  // return sqrt of negative number as 0
  float x = f_to_bf(z);
  float x2 = f_to_bf(x * f_to_bf(0.5f));
  float y = u_to_f(magic_n - (f_to_u(x) >> 1));
  x2 = f_to_bf(x2 * y);
  x2 = f_to_bf(x2 * y);
  float t = f_to_bf(f_to_bf(1.5f) - x2);
  x2 = f_to_bf(t * y);
  y = f_to_bf(x2 * x);
  return f_to_bf(y);
}

float fast_elu(float z, float alpha) {
  float static_z = z;
  z = std::min(std::max(z, -88.f), 88.f);
  float rcp_ln2 = f_to_bf(1.4453125);
  float x_bf16 = f_to_bf(z);
  float q = x_bf16 * rcp_ln2;
  int q_int = floor(q);
  float ln2 = 0.69140625;
  float approx = f_to_bf(float(q_int) * ln2);
  float m = x_bf16 - approx;

  float alpha_0 = f_to_bf(1.0);
  float alpha_1 = f_to_bf(1.0);
  float alpha_2 = f_to_bf(0.486328125);
  float alpha_3 = f_to_bf(0.21875);

  float exp_m = alpha_3 * m + alpha_2;
  exp_m = f_to_bf(exp_m) * m + alpha_1;
  exp_m = f_to_bf(f_to_bf(exp_m) * m + alpha_0);

  float pow2_q = pow(2.0, q_int);
  exp_m = f_to_bf(exp_m * pow2_q);

  float result = (static_z > 0.f) ? f_to_bf(static_z)
                                  : f_to_bf(f_to_bf(alpha) * (exp_m - 1.f));

  return result;
}

template <typename T>
T DPURoundEven(double data, T data_min = std::numeric_limits<T>::lowest(),
               T data_max = std::numeric_limits<T>::max()) {
  T rlt = 0;

  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else {
    if (data - floor(data) == 0.5) {
      if ((int)floor(data) % 2 == 0)
        data = floor(data);
      else
        data = ceil(data);
    }
    rlt = static_cast<T>(round(data));
  }

  return rlt;
}

int8_t gelu_lut(int8_t x) {
  // Generated from PyTorch. Ref:
  // https://gist.gitenterprise.xilinx.com/ZIJIANGY/691ec391e0fb9a3726057dc6308cdd96
  std::array<int8_t, 256> LUT{
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   1,   2,   3,   4,   6,
      7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20,  21,
      22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,
      37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
      52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,
      67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,
      82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
      97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
      112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126,
      127};
  int const i = x + 128;
  return LUT[i];
}

}  // namespace

template <typename DType>
EltwiseFix<DType>::EltwiseFix(const xir::Subgraph* subg, const xir::Op* op,
                              IMapTBs_t inputs, CPUTBPtr_t output)
    : Eltwise<DType>(subg, op, inputs, output) {
  auto nonlinear_type_str = op->get_attr<string>("nonlinear");
  if (nonlinear_type_str == "NONE" || nonlinear_type_str == "")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_NONE;
  else if (nonlinear_type_str == "RELU")
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_RELU;
  else if (nonlinear_type_str == "PRELU") {
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_PRELU;
    prelu_in_ = op->get_attr<int>("prelu_in");
    prelu_shift_ = op->get_attr<int>("prelu_shift");
    prelu_alpha_ = ((double)prelu_in_) / pow(2.0, prelu_shift_);
  } else if (nonlinear_type_str == "LEAKYRELU") {
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_LEAKY_RELU;
    prelu_alpha_ = (double)(op->get_attr<float>("LEAKYRELU_alpha"));
  } else if (nonlinear_type_str == "RELU6") {
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_RELU6;
  } else if (nonlinear_type_str == "HSIGMOID") {
    nonlinear_type_ = EltwiseNonlinearType::NONLINEAR_HSIGMOID;
  } else {
    UNI_LOG_FATAL(VART_NOT_SUPPORT)
        << "Unsupported nonlinear type: " << nonlinear_type_str << ".";
  }

  for (auto i = 0; i < input_num_; i++) {
    auto* xir_tensor_i = CPUOPBase::xir_op_->get_input_tensor("input", i);
    UNI_LOG_CHECK(xir_tensor_i != nullptr, VART_NULL_PTR);
    auto fp = xir_tensor_i->has_attr("fix_point")
                  ? xir_tensor_i->get_attr<int>("fix_point")
                  : 0;
    fp_inputs_.push_back(fp);
  }

  auto* xir_tensor_o = CPUOPBase::xir_op_->get_output_tensor();
  UNI_LOG_CHECK(xir_tensor_o != nullptr, VART_NULL_PTR);
  fp_output_ = xir_tensor_o->has_attr("fix_point")
                   ? xir_tensor_o->get_attr<int>("fix_point")
                   : 0;
  output_round_ = CPUOPBase::round_mode_;
  output_data_type_ = xir_tensor_o->get_data_type();

  if (elt_type_ == "CLAMP") {
    minval_ = op->get_attr<std::int32_t>("min");
    maxval_ = op->get_attr<std::int32_t>("max");
  }
  if (elt_type_ == "ELU") {
    alpha_val_ = op->get_attr<float>("alpha");
  }
  if (elt_type_ == "PRELU") {
    prelu_in_ = (int8_t)(uint8_t)op->get_attr<int>("prelu_in");
    prelu_shift_ = op->get_attr<int>("prelu_shift");
    prelu_alpha_ = ((double)prelu_in_) / pow(2.0, prelu_shift_);
  }
  if (elt_type_ == "LEAKY-RELU") {
    prelu_in_ = (uint8_t)op->get_attr<int>("prelu_in");
    prelu_shift_ = op->get_attr<int>("prelu_shift");
    prelu_alpha_ = ((double)prelu_in_) / pow(2.0, prelu_shift_);
  }
  if (elt_type_ == "HSIGMOID") {
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = -hsigmoid_in_;
  }
  if (elt_type_ == "HSWISH") {
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<int>("hsigmoid_in");
    shift_hsigmoid_ = CPUOPBase::xir_op_->get_attr<int>("shift_hsigmoid");
    shift_hswish_ = CPUOPBase::xir_op_->get_attr<int>("shift_hswish");
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = -hsigmoid_in_;
  }
  if (elt_type_ == "ADD" || elt_type_ == "SUB" || elt_type_ == "MAX" ||
      elt_type_ == "MIN" || elt_type_ == "RELU" || elt_type_ == "RELU6" ||
      elt_type_ == "PRELU" || elt_type_ == "LEAKY-RELU") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = fp_min - fp_output_;
  } else if (elt_type_ == "ABS" || elt_type_ == "NEG") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = fp_inputs_[0] - fp_output_;
  } else if (elt_type_ == "MUL") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    auto fp_sum = std::accumulate(fp_inputs_.begin(), fp_inputs_.end(), 0);
    shift_write_ = fp_sum - fp_output_;
  } else if (elt_type_ == "DIV") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(0);
    }
    shift_write_ = fp_inputs_[0] - fp_inputs_[1] - fp_output_;
  } else if (elt_type_ == "EQUAL" || elt_type_ == "LESS" ||
             elt_type_ == "LESS-EQUAL" || elt_type_ == "GREATER" ||
             elt_type_ == "GREATER-EQUAL" || elt_type_ == "NOT" ||
             elt_type_ == "AND" || elt_type_ == "OR") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    // fp_min = std::min(0, fp_min);
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = 0;
  } else if (elt_type_ == "SQRT" || elt_type_ == "CLAMP") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i]);
    }
    shift_write_ = -fp_output_;
  } else if (elt_type_ == "ELU") {
    auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
    fp_min = std::min(0, fp_min);
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i] - fp_min);
    }
    shift_write_ = -fp_output_;
  } else if (elt_type_ == "TANH" || elt_type_ == "SIGMOID") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i]);
    }
    shift_write_ = -fp_output_;
  } else if (elt_type_ == "QLINEARSIGMOID") {
    x_scale_ =
        CPUOPBase::xir_op_->template get_attr<std::vector<float>>("x_scale")[0];
    y_scale_ =
        CPUOPBase::xir_op_->template get_attr<std::vector<float>>("y_scale")[0];
    x_zero_point_ = CPUOPBase::xir_op_->template get_attr<std::vector<int32_t>>(
        "x_zero_point")[0];
    y_zero_point_ = CPUOPBase::xir_op_->template get_attr<std::vector<int32_t>>(
        "y_zero_point")[0];
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i]);
    }
    shift_write_ = -fp_output_;
  } else if (elt_type_ == "L2NORM") {
    for (auto i = 0; i < input_num_; i++) {
      shift_read_.push_back(fp_inputs_[i]);
    }
    shift_write_ = -fp_output_;
    // get dims info
    axis_ = CPUOPBase::xir_op_->get_attr<vector<int>>("axis");
    for (auto i = 0U; i < axis_.size(); i++) {
      if (axis_[i] < 0) axis_[i] += fmap_i_[0].ndims();
    }
  }

  if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
    hsigmoid_in_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("hsigmoid_in");
    shift_hsigmoid_ =
        CPUOPBase::xir_op_->get_attr<std::int32_t>("shift_hsigmoid");
    if (elt_type_ == "ADD" || elt_type_ == "SUB" || elt_type_ == "MAX" ||
        elt_type_ == "MIN") {
      auto fp_min = *(std::min_element(fp_inputs_.begin(), fp_inputs_.end()));
      shift_write_ = fp_min - hsigmoid_in_;
    }
  }
}

template <typename DType>
void EltwiseFix<DType>::print_param() {
  Eltwise<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "nonlinear_type = " << nonlinear_type_ << endl;
  if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
    UNI_LOG_DEBUG_INFO << "shift_hsigmoid = " << shift_hsigmoid_ << endl;
    UNI_LOG_DEBUG_INFO << "hsigmoid_in = " << hsigmoid_in_ << endl;
  } else if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_LEAKY_RELU) {
    UNI_LOG_DEBUG_INFO << "leakyrelu_alpha = " << prelu_alpha_ << endl;
  }
  if (elt_type_ == "QLINEARSIGMOID") {
    UNI_LOG_DEBUG_INFO << "x_scale = " << x_scale_ << endl;
    UNI_LOG_DEBUG_INFO << "y_scale = " << y_scale_ << endl;
    UNI_LOG_DEBUG_INFO << "x_zero_point = " << x_zero_point_ << endl;
    UNI_LOG_DEBUG_INFO << "y_zero_point = " << y_zero_point_ << endl;
  }
  UNI_LOG_DEBUG_INFO << "broadcast = " << broadcast_ << endl;
  UNI_LOG_DEBUG_INFO << "bit_width = " << CPUOPBase::bit_width_ << endl;
  if (elt_type_ != "GELU") {
  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "fp_input" << i << " = " << fp_inputs_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "fp_output = " << fp_output_ << endl;
  for (auto i = 0; i < input_num_; i++) {
    UNI_LOG_DEBUG_INFO << "shift_read" << i << " = " << shift_read_[i] << endl;
  }
  UNI_LOG_DEBUG_INFO << "shift_write = " << shift_write_ << endl;
  }
  if (nonlinear_type_ == 3) {
    UNI_LOG_DEBUG_INFO << "leakyrelu_alpha: " << prelu_alpha_ << endl;
  }
}

template <typename DType>
void EltwiseFix<DType>::run() {
  if (elt_type_ == "L2NORM") {
    calculate_pow();
  }
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    eltwise_thread();
  } else {
    eltwise_normal();
  }
}

template <typename DType>
void EltwiseFix<DType>::calculate_pow() {
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
        prlt[k] = f_to_bf(prlt[k] * pow(0.5, shift_read_[0]));
        acc_buf[k][0] = f_to_bf(prlt[k]) * f_to_bf(prlt[k]);
      }

      // accumulate
      for (auto j = 1; j < cur_dim; j++) {
        float* pcur = prlt + j * cur_cod;
        for (auto k = 0; k < cur_cod; k++) {
          pcur[k] = f_to_bf(pcur[k] * pow(0.5, shift_read_[0]));
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
}

template <typename DType>
uint32_t EltwiseFix<DType>::broadcast_pos(uint32_t pos_iter, int input_iter) {
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

template <typename DType>
void EltwiseFix<DType>::eltwise(std::uint32_t start_index,
                                std::uint32_t end_index) {
  // for (auto fp_iter = 0U; fp_iter < fmap_i_.size(); fp_iter++) {
  //   std::string name = "add" + std::to_string(fp_iter) + ".bin";
  //   FILE* fp_add = fopen(name.data(), "wb");
  //   for (auto i = 0; i < fmap_o_.num(); i++) {
  //     fwrite(data_in_[fp_iter] + i, sizeof(DType), 1, fp_add);
  //   }
  //   fclose(fp_add);
  // }
  auto dst_coord = fmap_o_.pos2coord(0);
  auto src_coord = dst_coord;
  for (auto pos_iter = start_index; pos_iter < end_index; pos_iter++) {
    double tmp = (elt_type_ == "MUL" || elt_type_ == "DIV") ? 1 : 0;
    double tmp_float = 0.0;
    float tmp_float_norm = 0.0;
    // int tmp_int = 0;
    for (auto fp_iter = 0U; fp_iter < fmap_i_.size(); fp_iter++) {
      // dim_t pos;
      auto pos = broadcast_pos(pos_iter, fp_iter);
      if (elt_type_ == "ADD" || elt_type_ == "RELU" || elt_type_ == "RELU6" ||
          (elt_type_ == "PRELU" && input_num_ == 1) ||
          elt_type_ == "LEAKY-RELU") {
        tmp += floor((double)data_in_[fp_iter][pos] *
                     pow(2.0, 7 - shift_read_[fp_iter]));
      } else if (elt_type_ == "SUB") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          tmp -= floor((double)data_in_[fp_iter][pos] *
                       pow(2.0, 7 - shift_read_[fp_iter]));
        }
      } else if (elt_type_ == "MUL") {
        tmp *= floor((double)data_in_[fp_iter][pos] * 4 /
                     pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "DIV") {
        if (fp_iter == 0U) {
          tmp = data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]);
        } else {
          tmp = fast_div(
              tmp, data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]));
        }
      } else if (elt_type_ == "CLAMP") {
        tmp = std::min(
            std::max(
                floor((double)data_in_[fp_iter][pos]),
                static_cast<double>(minval_ * pow(2.0, shift_read_[fp_iter]))),
            static_cast<double>(maxval_ * pow(2.0, shift_read_[fp_iter])));
      } else if (elt_type_ == "MAX") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 > tmp1 ? tmp0 : tmp1;
        }
      } else if (elt_type_ == "MIN") {
        if (fp_iter == 0U) {
          tmp = floor((double)data_in_[fp_iter][pos] *
                      pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 > tmp1 ? tmp1 : tmp0;
        }
      } else if (elt_type_ == "EQUAL") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 == tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "LESS") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 < tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "LESS-EQUAL") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 <= tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "GREATER") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 > tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "GREATER-EQUAL") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 >= tmp1 ? 1 : 0;
        }
      } else if (elt_type_ == "NOT") {
        auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                          pow(2.0, 7 - shift_read_[fp_iter]));
        tmp = !tmp1;
      } else if (elt_type_ == "AND") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 && tmp1;
        }
      } else if (elt_type_ == "OR") {
        if (fp_iter == 0U) {
          tmp_float = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
        } else {
          auto tmp0 = tmp_float;
          auto tmp1 = floor((double)data_in_[fp_iter][pos] *
                            pow(2.0, 7 - shift_read_[fp_iter]));
          tmp = tmp0 || tmp1;
        }
      } else if (elt_type_ == "ABS") {
        tmp = floor((double)data_in_[fp_iter][pos] * 4 /
                    pow(2.0, shift_read_[fp_iter]));
        tmp = tmp > 0 ? tmp : (-tmp);
      } else if (elt_type_ == "NEG") {
        tmp = floor((double)data_in_[fp_iter][pos] * 4 /
                    pow(2.0, shift_read_[fp_iter]));
        tmp = -tmp;
      } else if (elt_type_ == "SQRT") {
        tmp =
            fast_sqrt(data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "ELU") {
        tmp = fast_elu(data_in_[fp_iter][pos] / pow(2.0, shift_read_[fp_iter]),
                       alpha_val_);
      } else if (elt_type_ == "PRELU" && input_num_ == 2) {
        if (fp_iter == 0U) {
          tmp = (double)data_in_[fp_iter][pos] *
                pow(2.0, 7 - shift_read_[fp_iter]);
        } else {
          if (tmp < 0)
            tmp *= ((double)data_in_[fp_iter][pos]) / pow(2.0, prelu_shift_);
        }
      } else if (elt_type_ == "TANH") {
        tmp = tanh((double)data_in_[fp_iter][pos] /
                   pow(2.0, shift_read_[fp_iter]));
      } else if (elt_type_ == "SIGMOID") {
        tmp = 1.0 / (1.0 + std::exp(-data_in_[fp_iter][pos] /
                                    pow(2.0, shift_read_[fp_iter])));
      } else if (elt_type_ == "QLINEARSIGMOID") {
        tmp = y_scale_ * 1 /
                  (exp((x_zero_point_ - (float)data_in_[fp_iter][pos]) /
                       x_scale_) +
                   1) +
              y_zero_point_;
      } else if (elt_type_ == "HSIGMOID" || elt_type_ == "HSWISH") {
        tmp = min(pow(2, 32), max(0.0, (data_in_[fp_iter][pos] * 2731.0f +
                                        3.0f * 2731 * pow(2, hsigmoid_in_)))) /
              pow(2, shift_hsigmoid_);
      } else if (elt_type_ == "L2NORM") {
        auto coord = fmap_o_.pos2coord(pos);
        for (auto k = 0U; k < axis_.size(); k++) {
          auto dim = axis_[k];
          coord[dim] = 0;
        }
        auto src_pos = fmap_i_[0].coord2pos(coord);
        tmp_float_norm =
            f_to_bf((float)data_in_[fp_iter][pos] * pow(0.5, shift_read_[0]));
        if (data_in_buf_ptr_[src_pos] < SUBSTITUTE_FOR_0) {
          tmp_float_norm = f_to_bf(f_to_bf(tmp_float_norm) * f_to_bf(1 / sqrt(SUBSTITUTE_FOR_0)));
        } else {
          tmp_float_norm = f_to_bf(f_to_bf(tmp_float_norm) *
                        f_to_bf(1 / sqrt(data_in_buf_ptr_[src_pos])));
        }
      } else if (elt_type_ == "GELU") {
        int8_t const x = static_cast<int8_t>(data_in_[fp_iter][pos]);
        data_out_[pos] = static_cast<double>(gelu_lut(x));
      }
    }
    if (elt_type_ == "ABS" || elt_type_ == "NEG") {
      tmp = tmp / pow(2.0, shift_write_);
      tmp /= 4;
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }

    if (elt_type_ == "EQUAL" || elt_type_ == "LESS" ||
        elt_type_ == "LESS-EQUAL" || elt_type_ == "GREATER" ||
        elt_type_ == "GREATER-EQUAL" || elt_type_ == "AND" ||
        elt_type_ == "OR" || elt_type_ == "NOT") {
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if (elt_type_ == "ELU") {
      tmp /= pow(2.0, shift_write_);
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if (elt_type_ == "CLAMP") {
      tmp /= pow(2.0, (shift_write_ + shift_read_[0]));
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if (elt_type_ == "HSIGMOID") {
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if (elt_type_ == "HSWISH") {
      tmp = round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      tmp = (double)data_in_[0][pos_iter] * tmp * pow(2, -shift_hswish_);
      data_out_[pos_iter] =
          round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    if (elt_type_ == "L2NORM") {
      tmp_float_norm /= pow(2.0, shift_write_);
      data_out_[pos_iter] = DPURoundEven<DType>(
          f_to_bf(tmp_float_norm), CPUOPBase::data_min_, CPUOPBase::data_max_);
      continue;
    }
    tmp /= pow(2.0, shift_write_);
    if (elt_type_ == "DIV" || elt_type_ == "SQRT" || elt_type_ == "TANH" ||
        elt_type_ == "SIGMOID" || elt_type_ == "QLINEARSIGMOID") {
    } else if (elt_type_ == "ADD" || elt_type_ == "SUB" || elt_type_ == "MAX" ||
               elt_type_ == "MIN" || elt_type_ == "RELU" ||
               elt_type_ == "RELU6" || elt_type_ == "PRELU" ||
               elt_type_ == "LEAKY-RELU") {
      tmp /= pow(2, 7);
    } else if (elt_type_ == "L2NORM") {
      tmp = f_to_bf(tmp);
    } else {
      tmp /= pow(4, fmap_i_.size());
    }
    if (elt_type_ == "RELU" ||
        nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_RELU) {
      if (tmp < 0) tmp = 0;
    } else if ((elt_type_ == "PRELU" && input_num_ == 1) ||
               nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_PRELU ||
               elt_type_ == "LEAKY-RELU" ||
               nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_LEAKY_RELU) {
      if (tmp < 0) tmp *= prelu_alpha_;
    } else if (elt_type_ == "RELU6" ||
               nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_RELU6) {
      if (tmp < 0) tmp = 0;
      if (fp_output_ <= 4) {
        auto thr6 = 6 << 4;
        if (tmp >= thr6) tmp = thr6;
      }
    } else if (nonlinear_type_ == EltwiseNonlinearType::NONLINEAR_HSIGMOID) {
      tmp =
        double(round_normal<DType>(CPUOPBase::round_mode_, tmp, CPUOPBase::data_min_, CPUOPBase::data_max_));
      tmp = std::min(
                pow(2, 32),
                std::max(0.0, (tmp * 2731 + 3 * 2731 * pow(2, hsigmoid_in_)))) *
            pow(2, -shift_hsigmoid_);
    }

    int32_t data_min = CPUOPBase::data_min_;
    int32_t data_max = CPUOPBase::data_max_;
    data_out_[pos_iter] = round_normal<DType>(output_round_,tmp,data_min,data_max);
  }
}

INSTANTIATE_TPCLASS(EltwiseFix);
REG_OP_INSTANCE_FUNC("eltwise-fix", EltwiseFix);

}  // namespace cpu
}  // namespace vart
