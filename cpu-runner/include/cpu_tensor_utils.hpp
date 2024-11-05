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

#pragma once

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <numeric>


namespace vart {
namespace cpu_tensor_utils {

// calc the tensor stride based on the dims
template <typename T>
std::vector<T> calc_tensor_stride_based_on_dims(const std::vector<T>& dims) {
  std::vector<T> ret(dims.size(), 1);
  T stride{1};
  auto stride_idx = dims.size() - 1;
  for (auto it = dims.rbegin(); it < dims.rend(); it++) {
    ret[stride_idx--] = stride;
    stride *= (*it);
  }
  return ret;
}

template <typename T>
std::int64_t coord2pos(const std::vector<T>& coord,
                       const std::vector<T>& shape) {
  // UNI_LOG_CHECK(coord.size() == shape.size(), VART_SIZE_MISMATCH);
  auto strides = calc_tensor_stride_based_on_dims(shape);
  std::int64_t ret = 0;
  for (auto i = 0U; i < shape.size(); ++i) {
    ret += (int64_t)coord[i] * (int64_t)strides[i];
  }
  return ret;
}

template <typename T>
std::vector<T> pos2coord(std::int64_t pos, const std::vector<T>& shape) {
  // auto data_size =
  //     accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  auto strides = calc_tensor_stride_based_on_dims(shape);
  auto ret = shape;
  // UNI_LOG_CHECK(pos >= 0 && pos < data_size, VART_SIZE_MISMATCH);
  for (auto i = 0U; i < shape.size(); i++) {
    ret[i] = pos / strides[i];
    pos %= strides[i];
  }
  return ret;
}

template <typename T>
std::vector<T> transpose_data(const std::vector<T>& data,
                              const std::vector<std::int32_t>& shape,
                              const std::vector<std::int32_t>& order) {
  auto data_size =
      accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
  // UNI_LOG_CHECK(data_size == (int)data.size(), VART_SIZE_MISMATCH)
  //     << "Data transposition error, the size of the data does not match the "
  //        "shape.";
  // UNI_LOG_CHECK(shape.size() == order.size(), VART_SIZE_MISMATCH)
  //     << "Data transposition error, the size of shape not equal to the size of "
  //        "order.";

  auto output_shape = shape;
  for (auto i = 0U; i < order.size(); ++i) {
    output_shape[i] = shape[order[i]];
  }

  auto strides = calc_tensor_stride_based_on_dims(shape);
  auto ret = data;
  for (auto i = 0; i < data_size; ++i) {
    auto coord = pos2coord(i, shape);
    auto new_coord = coord;
    for (auto j = 0U; j < shape.size(); ++j) {
      new_coord[j] = coord[order[j]];
    }
    auto out_addr = coord2pos(new_coord, output_shape);
    ret[out_addr] = data[i];
  }
  return ret;
}
} // cpu_tensor_utils
} // vart
