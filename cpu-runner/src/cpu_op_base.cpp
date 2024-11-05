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

#include "cpu_op_base.hpp"

#include "vart/xir_helper.hpp"
#include "xir/op/op.hpp"

namespace vart {
namespace cpu {

std::atomic<uint64_t> CPUOPBase::subg_ops;
const string CPUOPBase::SUBG_DIFF_SCRIPT = "diff.sh";
const string CPUOPBase::SUBG_DIFF_SCRIPT_HEADER = R"code(
#!/bin/bash

if [ $# -eq 0 ];then
  if [ $GOLDEN_PATH ];then
    GOLDEN_PATH=${GOLDEN_PATH}
  else
    echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
    echo "./diff.sh {golden_path}"
    exit 1
  fi
elif [ $# -eq 1 ];then
    GOLDEN_PATH=$1
else
  echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
  exit 1
fi

#set -x

if_same=true
has_match_file=false
for file in ./*
do
    filename=$(basename $file )
    golden_file=${GOLDEN_PATH}/${filename}
    if [ ! -f $gloden_file ]
    then
      file_size="$(wc -c <"$file")"
      gloden_file_size="$(wc -c <"$gloden_file")"
      if [ $file_size -ne $gloden_file_size ]
      then
        echo "The size of $filename not match!"
      else
        diff ${file} ${golden_file} 1>diff.txt 2>&1 && result=0 || result=1

        if [ "$result" == 1 ];then
            echo "$filename is diff"
            if_same=false
        else
            has_match_file=true    
        fi
      fi
    #else
     # echo "$golden_file does not exist, please check filename."
    fi
done

if [ "$if_same" = true -a "$has_match_file" = true ]; then
  echo "All same!"
fi

)code";

void CPUOPBase::StaticInit() { subg_ops.store(0); }

CPUOPBase::CPUOPBase(const xir::Subgraph* subg, const xir::Op* op,
                     IMapTBs_t inputs, CPUTBPtr_t output)
    : xir_subg_(subg),
      xir_op_(op),
      output_tensor_(xir_op_->get_output_tensor()),
      inputs_(inputs),
      output_(output),
      data_type_(output_tensor_->get_data_type().type),
      bit_width_(output_tensor_->get_data_type().bit_width),
      round_mode_(output_tensor_->has_attr("round_mode")
                      ? output_tensor_->get_attr<std::string>("round_mode")
                      : "DPU_ROUND") {
  if_signed_ = get_if_signed(data_type_);
  if (data_type_ == xir::DataType::Type::XINT ||
      data_type_ == xir::DataType::Type::XUINT ||
      data_type_ == xir::DataType::Type::INT ||
      data_type_ == xir::DataType::Type::UINT) {
    data_min_ = if_signed_ ? -pow(2, bit_width_ - 1) : 0;
    data_max_ =
        if_signed_ ? pow(2, bit_width_ - 1) - 1 : pow(2, bit_width_) - 1;
  } else if (output_tensor_->get_data_type() == xir::DataType("BFLOAT16")) {
    if (op->get_type() != "const-fix" && op->get_type() != "const") {
      if (!(op->has_attr("compute_type") &&
            op->get_attr<std::string>("compute_type") == "float32")) {
        UNI_LOG_DEBUG_WARNING
            << op->to_string()
            << " bfloat16 IO type should has compute_type float32";
      }
    }
    data_min_ = 0xff7f0000;
    data_max_ = 0x7f7f0000;
  } else if (data_type_ == xir::DataType::Type::FLOAT) {
    if (bit_width_ == 16) {
      data_min_ = 0xfbff0000;
      data_max_ = 0x7bff0000;
    } else if (bit_width_ == 32) {
      data_min_ = 0xff7fffff;
      data_max_ = 0x7f7fffff;
    }
  }
}

void CPUOPBase::save() {
  if (CPUCfg::Instance().debug()) {
    output_->save_bin();
  }
}

string CPUOPBase::get_input_list() const {
  string s;
  auto v = vec_input_ops(xir_op_->get_input_ops());

  for (auto i = 0U; i < v.size(); i++) {
    s += v[i]->get_name();
    s += "(" + v[i]->get_type() + ")";
    if (i != v.size() - 1) {
      s += ", ";
    }
  }

  return s;
}

float f_to_bf_aie2p(float data) {
  std::array<uint16_t, 2> iraw = bit_cast<std::array<uint16_t, 2>>(data);
  std::uint16_t u16 = 0;
  switch (std::fpclassify(data)) {
    case FP_SUBNORMAL: {
      u16 = iraw[1] & 0x8000;
      break;
    }
    case FP_ZERO: {
      u16 = iraw[1] & 0x0000;
      break;
    }
    case FP_INFINITE: {
      u16 = iraw[1];
      break;
    }
    case FP_NAN: {
      // truncate and set MSB of the mantissa force QNAN
      u16 = iraw[1];
      u16 |= (1 << 6);
      break;
    }
    case FP_NORMAL: {
      // round to nearest even and truncate
      const uint32_t rounding_bias = 0x00007FFF + (iraw[1] & 0x1);
      const uint32_t int_raw = bit_cast<uint32_t>(data) + rounding_bias;
      iraw = bit_cast<std::array<uint16_t, 2>>(int_raw);
      u16 = iraw[1];
      break;
    }
  }

  std::uint32_t u16_in_u32 = bit_cast<std::uint32_t>(u16);
  return bit_cast<float>(u16_in_u32 << 16);
}

bool check_signed_zero(const float a, bool sign) {
  std::uint32_t u32 = bit_cast<std::uint32_t>(a);

  if (sign) {
    return (u32 == 0x80000000);
  } else {
    return (u32 == 0x00000000);
  }
}

float max_aie2p(const float a, const float b ) {
  if ((check_signed_zero(a, true) && check_signed_zero(b, false)) ||
      (check_signed_zero(a, false) && check_signed_zero(b, true))) {
    return bit_cast<float, std::uint32_t>(0x00000000);
  }

  if ((check_signed_zero(a, true) && std::signbit(b)) ||
      (check_signed_zero(b, true) && std::signbit(a))) {
    return bit_cast<float, std::uint32_t>(0x80000000);
  }

  return std::max(a, b);
}
}  // namespace cpu
}  // namespace vart
