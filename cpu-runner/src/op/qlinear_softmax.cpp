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

#include "qlinear_softmax.hpp"

namespace vart {
namespace cpu {

template <typename DType>
const vector<string> QlinearSoftmax<DType>::ITName = {
    "input",
};

// constructor and deconstructor
template <typename DType>
QlinearSoftmax<DType>::QlinearSoftmax(const xir::Subgraph *subg,
                                      const xir::Op *op, IMapTBs_t inputs,
                                      CPUTBPtr_t output)
    : CPUOPBase(subg, op, inputs, output) {
  GET_INPUT_DIMX_FMAP(fmap_i_, input, 6);
  GET_OUTPUT_DIMX_FMAP(fmap_o_, 6);

  // get pad info
  axis_ = xir_op_->get_attr<int>("axis");
  axis_ = axis_ == -1
              ? (xir_op_->get_input_tensor("input")->get_shape().size() - 1)
              : axis_;
              
  auto input_dtype = xir_op_->get_input_tensor("input")->get_data_type();
  bit_width_ = input_dtype.bit_width;
  if_signed_ = input_dtype.type == xir::DataType::INT;

  x_scale = xir_op_->get_attr<std::vector<float>>("x_scale").front();
  y_scale = xir_op_->get_attr<std::vector<float>>("y_scale").front();

  y_zero_point = xir_op_->get_attr<std::vector<int32_t>>("y_zero_point").front();
}

template <typename DType> void QlinearSoftmax<DType>::run() {
  // // do necessary check
  print_param();
  check_param();

  // // read data
  read();

  // do softmax
  softmax();

  // // do save, debug...
  // save();
}

template <typename DType> void QlinearSoftmax<DType>::print_param() {
  fmap_i_.print_param("fmap_i");
  fmap_o_.print_param("fmap_o");

  UNI_LOG_DEBUG_INFO << "axis = " << axis_ << endl;
  UNI_LOG_DEBUG_INFO << "x_scale = " << x_scale << endl;
  UNI_LOG_DEBUG_INFO << "y_scale = " << y_scale << endl;
  UNI_LOG_DEBUG_INFO << "y_zp = " << y_zero_point << endl;
}

template <typename DType> void QlinearSoftmax<DType>::check_param() {
  UNI_LOG_CHECK(inputs_.size() == 1, VART_SIZE_ERROR);

  // check fmap_out height
  UNI_LOG_CHECK(fmap_i_.ndims() == fmap_o_.ndims(), VART_SIZE_ERROR)
      << fmap_i_.ndims() << " != " << fmap_o_.ndims();
  for (auto i = 0; i < fmap_i_.ndims(); i++) {
    UNI_LOG_CHECK(fmap_i_[i] == fmap_o_[i], VART_INVALID_VALUE)
        << ", " << fmap_i_[i] << " != " << fmap_o_[i];
  }
}

// read data from DDR
template <typename DType> void QlinearSoftmax<DType>::read() {
  // read img data
  auto *cputb = inputs_.at(ITName[INPUT]).at(0);
  img_ = GET_CPUTB_DType_PTR(DType, cputb);

  // handle output buffer
  rlt_ = GET_CPUTB_DType_PTR(DType, output_);
  // rlt_ = GET_CPUTB_FLOAT_PTR(output_);
}

template <typename DType> uint64_t QlinearSoftmax<DType>::get_workload() {
  // TODO
  return 0;
}

typedef union ql_value_convert {
  std::uint32_t u;
  float f;
} ql_value_convert_t;

std::uint32_t ql_f_to_u(float data) {
  ql_value_convert_t vc{};
  vc.f = data;
  return vc.u;
}

float ql_u_to_f(std::uint32_t data) {
  ql_value_convert_t vc{};
  vc.u = data;
  return vc.f;
}

float ql_f_to_bf(float data) {
  std::uint32_t u = ql_f_to_u(data);
  std::uint32_t flag = (u & 0x00010000) >> 16;
  u = (u + 0x7fff + flag) & 0xFFFF0000;
  return ql_u_to_f(u);
}
float ql_f_to_bf_output(float data) {
  std::uint32_t u = ql_f_to_u(data);
  if ((u & 0x7f800000) == 0x00000000) {
    u = u & 0xFFFF0000;
  } else {
    std::uint32_t flag = (u & 0x00010000) >> 16;
    u = (u + 0x7fff + flag) & 0xFFFF0000;
  }
  return ql_u_to_f(u);
}

float ql_mult_add(float a_bf16, float b_bf16, float c_fp32) {
  float out = ql_f_to_bf(a_bf16) * ql_f_to_bf(b_bf16) + ql_f_to_bf(c_fp32);
  return ql_f_to_bf(out);
}

float ql_ploy_HORNER_SCHEME(float r, std::vector<float> cs, int degree) {
  float out = ql_mult_add(r, cs[degree], cs[degree - 1]);
  auto i = degree - 1;
  while (i > 0) {
    out = ql_mult_add(r, out, cs[i - 1]);
    i--;
  }
  return ql_f_to_bf(out);
}

float ql_exp_approx_poly(float data) {
  // floor 1 + x * (1 + x * (0.486328125 + x * 0.21875))
  std::vector<float> cs = {1, 1, 0.486328125, 0.21875};
  float LN2 = ql_f_to_bf(0.6931471805599453);
  std::uint32_t EXP_TABLE_SIZE = 1;
  std::uint32_t degree = 3;
  float _MAX_BFLOAT16 = ql_u_to_f(0x7f7f0000);
  float _MIN_BFLOAT16 = ql_u_to_f(0xff7f0000);
  float x = ql_f_to_bf(data);
  float scaling_factor = ql_f_to_bf(x) * ql_f_to_bf(EXP_TABLE_SIZE / LN2);
  float n = ql_f_to_bf(floor(ql_f_to_bf(scaling_factor)));
  float m = ql_f_to_bf(floor(n / EXP_TABLE_SIZE));
  float t = ql_f_to_bf(n * ql_f_to_bf(LN2 / EXP_TABLE_SIZE));
  float r = ql_f_to_bf(x - t);
  float exp_x = ql_ploy_HORNER_SCHEME(r, cs, degree);
  exp_x = pow(2, m) * exp_x;
  return ql_f_to_bf(std::max(std::min(exp_x, _MAX_BFLOAT16), _MIN_BFLOAT16));
}

// float reciprocal_approx_moroz(float z, bool output_subnormals = false) {
//  std::uint32_t magic_n = 0x7eb53567;
//  float k1 = f_to_bf(1.9395974);
//  float k2 = f_to_bf(1.436142);
//  float ones = f_to_bf(1.0f);
//  float x = f_to_bf(z);
//  float y = u_to_f(magic_n - f_to_u(x));
//  float acc = f_to_bf(-x * y + k2);
//  acc = f_to_bf(acc * y);
//  acc = f_to_bf(acc * k1);
//  y = acc;
//  acc = f_to_bf(acc * -x + ones);
//  acc = f_to_bf(acc * y + y);
//  y = acc;
//  return (y);
//}

float ql_float_inv_func(float x, bool output_subnormals = false) {
  static const std::uint8_t m_inv_lut[256] = {
      128, 126, 124, 122, 120, 118, 117, 115, 113, 111, 109, 108, 106, 104, 103,
      101, 100, 98,  96,  95,  93,  92,  90,  89,  88,  86,  85,  83,  82,  81,
      79,  78,  77,  76,  74,  73,  72,  71,  69,  68,  67,  66,  65,  64,  63,
      61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,  49,  48,  47,
      46,  45,  44,  44,  43,  42,  41,  40,  39,  38,  37,  37,  36,  35,  34,
      33,  33,  32,  31,  30,  30,  29,  28,  27,  27,  26,  25,  24,  24,  23,
      22,  22,  21,  20,  20,  19,  18,  18,  17,  16,  16,  15,  14,  14,  13,
      13,  12,  11,  11,  10,  10,  9,   9,   8,   7,   7,   6,   6,   5,   5,
      4,   4,   3,   3,   2,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,
      0,   4,   4,   3,   3,   2,   2,   1,   1,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   4,   4,   3,   3,   2,   2,   1,   1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   4,   3,   3,
      2,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   4,   4,   3,   3,   2,   2,   1,   1,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   4,   4,   3,   3,   2,   2,   1,   1,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   4,   4,   3,   3,
      2,   2,   1,   1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0};
  std::uint32_t exp_mask = 0x7F800000;
  std::uint32_t mantissa_mask = 0x007FFFFF;
  std::uint32_t mantissa_Q = 0x00008000;
  std::uint32_t B_x = ql_f_to_u(x);
  std::uint8_t exponent = (B_x & exp_mask) >> 23;
  std::uint8_t mantissa = ((B_x & mantissa_Q) == 0)
                              ? ((B_x & mantissa_mask) >> 16)
                              : ((B_x & mantissa_mask) >> 16) + 1;
  std::uint8_t inv_exponent = 253 - exponent;
  std::uint32_t inv_x_val = (inv_exponent << 23) + (m_inv_lut[mantissa] << 16);
  return ql_u_to_f(inv_x_val);
}

float ql_parallel_sum(std::vector<float> data) {
  std::vector<float> psum;
  std::vector<float> psum0;
  std::vector<float> psum1;
  std::vector<float> psum2;
  float sum = 0;
  for (std::uint32_t i = 0; i < 16; i++) {
    psum.push_back(0);
    psum0.push_back(0);
    psum1.push_back(0);
    psum2.push_back(0);
  }
  for (std::uint32_t i = 0; i < data.size(); i++) {
    psum[i % 16] += data[i];
  }
  for (std::uint32_t i = 0; i < 8; i++) {
    psum0[i] += psum[i] + psum[i + 8];
  }
  for (std::uint32_t i = 0; i < 4; i++) {
    psum1[i] += psum0[i] + psum0[i + 4];
  }
  for (std::uint32_t i = 0; i < 2; i++) {
    psum2[i] += psum1[i] + psum1[i + 2];
  }
  sum = psum2[0] + psum2[1];
  return sum;
}

// float32 bitwise to bfloat16 with 32 bitwidth from quantizer
static inline float ql_float2bfloat_cpu(const float x) {
  std::uint32_t itmp = ql_f_to_u(x);       // float32 bitwise to int32
  if ((itmp & 0x00008000) == 0x00008000) { // half even
    if ((itmp & 0xFFFF) > 0x00008000 ||
        (((itmp & 0xFFFF) == 0x00008000) && (itmp & 0x10000) == 0x10000)) {
      itmp += 0x10000;
    }
  }
  itmp &= 0xFFFF0000;
  return ql_u_to_f(itmp); // int32 bitwise to float32
}

template <typename DType>
void poly_softmax(DType *src, int num, DType *rlt, float x_s,
                  float y_s, int y_zp, int bit_width_, bool if_signed_) {
  std::vector<float> exp_res;
  std::vector<float> softmax_res;
  float sum = 0;

  int data_size = bit_width_;
  float _min_in_dtype_ = 0.0;
  float _max_in_dtype_ = (pow(2, data_size) - 1);
  if
    (if_signed_) {
      _min_in_dtype_ = -pow(2, (data_size - 1));
      _max_in_dtype_ = (pow(2, (data_size - 1)) - 1);
    }

  if
    (bit_width_ == 8) {
      for (auto i = 0; i < num; i++) {
        float input_data;
        if (if_signed_)
          input_data = ((float)((char)src[i]) / x_s);
        else
          input_data = ((float)((unsigned char)src[i]) / x_s);
        // lut exp
        float exp_x = ql_f_to_bf(exp(input_data));
        exp_res.push_back(exp_x);
      }
      sum = ql_parallel_sum(exp_res);
      float div = ql_f_to_bf(ql_float_inv_func(sum));

      for (auto i : exp_res) {
        float res_ = ql_f_to_bf(i * div);
        res_ = ((ql_f_to_bf(res_ * ql_f_to_bf(y_s))) + (float)y_zp);
        // printf ("%f\n",ql_f_to_bf(res_));
        softmax_res.push_back(res_);
      }
      for (auto i = 0; i < num; i++) {
        DType tmp_out;
        if (softmax_res[i] > 0)
          tmp_out = round(std::min(float(_max_in_dtype_), softmax_res[i]));
        else
          tmp_out = round(std::max(float(_min_in_dtype_), softmax_res[i]));

        rlt[i] = tmp_out;
      }
    }
  else { // for int16 / uint16
    std::int64_t max_val = (std::int64_t)_min_in_dtype_;
    std::vector<std::int64_t> de_q_ip;

    for (auto i = 0; i < num; i++) {
      max_val = (src[i] > max_val) ? src[i] : max_val;
    }

    int min_valid_val = -32768;
    for (auto i = 0; i < num; i++) {
      int input_data = (src[i] - max_val);
      input_data = input_data < min_valid_val ? min_valid_val : input_data;
      const int mask = 0x000000FF;
      float tmp1 = (float)(unsigned char)(input_data & mask);
      float tmp2 = (float)(char)(unsigned char)((input_data>>8) & mask);
      float exp1 = exp(tmp1/x_s);
      float exp2 = exp(tmp2/x_s*pow(2,8));

      float exp_x = exp1 * exp2;
      exp_res.push_back(exp_x);
    }
    sum = ql_parallel_sum(exp_res);
    float div = y_s / sum;
    
    int iter = 0;
    for (auto i : exp_res) {
      float res_ = (i * div);
      DType out_val;
      int tmp_out;
      if (res_ > 0)
        tmp_out = round(std::min(float(_max_in_dtype_), res_));
      else
        tmp_out = round(std::max(float(_min_in_dtype_), res_));

      out_val = tmp_out + y_zp;
      rlt[iter] = out_val;
      iter++;
    }
  }
}

template <typename DType> void QlinearSoftmax<DType>::softmax() {
  if (fmap_i_.ndims() == 1) {
    UNI_LOG_CHECK(axis_ == 0, VART_INVALID_VALUE) << ", " << axis_ << " != 0";
    poly_softmax(img_, (int)(fmap_i_[0]), rlt_, x_scale,
                 y_scale, y_zero_point, bit_width_, if_signed_);
  } else if (fmap_i_.ndims() == 2) {
    UNI_LOG_CHECK(axis_ == 1, VART_INVALID_VALUE) << ", " << axis_ << " != 1";
    for (auto i = 0; i < fmap_i_[0]; i++) {
      poly_softmax(img_ + i * fmap_i_[1], (int)fmap_i_[1],
                   rlt_ + i * fmap_i_[1], x_scale, y_scale,
                   y_zero_point, bit_width_, if_signed_);
    }
  } else if (fmap_i_.ndims() == 3) {
    UNI_LOG_CHECK(axis_ == 2, VART_INVALID_VALUE) << ", " << axis_ << " != 2";
    for (auto i = 0; i < fmap_i_[0]; i++) {
      for (auto j = 0; j < fmap_i_[1]; j++) {
        poly_softmax(img_ + i * fmap_i_[1] * fmap_i_[2] + j * fmap_i_[2],
                     (int)fmap_i_[2],
                     rlt_ + i * fmap_i_[1] * fmap_i_[2] + j * fmap_i_[2],
                     x_scale, y_scale, y_zero_point, bit_width_, if_signed_);
      }
    }
  } else if (fmap_i_.ndims() == 4) {
    UNI_LOG_CHECK(axis_ == 3, VART_INVALID_VALUE) << ", " << axis_ << " != 3";

    for (auto i = 0; i < fmap_i_[0]; i++) {
      for (auto j = 0; j < fmap_i_[1]; j++) {
        for (auto k = 0; k < fmap_i_[2]; k++) {
          poly_softmax(img_ + i * fmap_i_[1] * fmap_i_[2] * fmap_i_[3] +
                           j * fmap_i_[2] * fmap_i_[3] + k * fmap_i_[3],
                       (int)fmap_i_[3],
                       rlt_ + i * fmap_i_[1] * fmap_i_[2] * fmap_i_[3] +
                           j * fmap_i_[2] * fmap_i_[3] + k * fmap_i_[3],
                       x_scale, y_scale, y_zero_point, bit_width_, if_signed_);
        }
      }
    }
  }
}

static inline float poly_horner(float x) {
  // y = c0+x(c1 + x(c2 + x(c3)))
  static const float coeff[4] = {1.0f, 1.0f, f_to_bf_aie2p(0.486328125f),
                                 f_to_bf_aie2p(0.21875f)};
  float ret = coeff[3];
  ret = f_to_bf_aie2p(coeff[2] + x * ret);
  ret = f_to_bf_aie2p(coeff[1] + x * ret);
  ret = f_to_bf_aie2p(coeff[0] + x * ret);
  return ret;
}

static float poly_exp(float x) {
  x = x < -88 ? -88 : x;
  float output = 0;
  x = f_to_bf_aie2p(x);
  // step1
  float k1 = f_to_bf_aie2p(1.4453125f);
  float k2 = 0.69140625f;
  float sf_n_m = f_to_bf_aie2p(k1 * x);
  int floor_rst = floor(sf_n_m);
  float floor_tof = static_cast<float>(floor_rst);
  float temp2 = f_to_bf_aie2p(floor_tof * k2);
  float r = x - temp2;
  // step2
  float exp_r = poly_horner(r);
  output = exp_r;
  uint32_t a = floor_rst << 23;
  uint32_t c = ql_f_to_u(exp_r);
  uint32_t b = a + c;
  output = ql_u_to_f(b);
  return f_to_bf_aie2p(output);
}

template <>
void QlinearSoftmax<float>::softmax() {
  auto aie_exp = [](float x){
    float ret = poly_exp(x);
    // float ret = ql_float2bfloat_cpu(pow(2,ql_float2bfloat_cpu(1.44269504089) *ql_float2bfloat_cpu(x)));
    return ret;
  };
  std::cout <<"axis_ = "<<axis_<<std::endl;
  // reduce_sum
  int size = 1;
  for(auto i=0; i<fmap_o_.ndims(); i++) {
    size *= fmap_o_[i];
  }
  size /= fmap_o_[axis_];
  std::vector<float> reduce_sum_rlt(size, 0);
  std::vector<float> reduce_max_rlt(size, CPUOPBase::data_min_);
  for (int pos = 0; pos < fmap_o_.num(); pos++) {
    auto coord = fmap_o_.pos2coord(pos);
    coord[axis_] = 0;
    auto pos2 = fmap_o_.coord2pos(coord);
    if (*(img_ + pos)>reduce_max_rlt[pos2]){
      reduce_max_rlt[pos2] = *(img_ + pos);
    }
  }
 
  for (int pos = 0; pos < fmap_o_.num(); pos++) {
    auto coord = fmap_o_.pos2coord(pos);
    coord[axis_] = 0;
    auto pos2 = fmap_o_.coord2pos(coord);
    reduce_sum_rlt[pos2] += f_to_bf_aie2p(aie_exp(*(img_ + pos) - reduce_max_rlt[pos2]));
  }

  // softmax
  for (int pos = 0; pos < fmap_o_.num(); pos++) {
    auto coord = fmap_o_.pos2coord(pos);
    coord[axis_] = 0;
    auto pos2 = fmap_o_.coord2pos(coord);
    *(rlt_ + pos) = f_to_bf_aie2p(f_to_bf_aie2p(aie_exp(*(img_ + pos) - reduce_max_rlt[pos2])) * f_to_bf_aie2p(1/ reduce_sum_rlt[pos2]));
  }
}

INSTANTIATE_TPCLASS(QlinearSoftmax);
REG_OP_INSTANCE_FUNC("qlinear-hard-softmax", QlinearSoftmax);

} // namespace cpu
} // namespace vart
