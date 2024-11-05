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
#include "DM.hpp"
#include <sstream>
#include "SimCfg.hpp"
#include "conf/ArchCfg.hpp"
#include "util/Util.hpp"

template <typename DType>
std::shared_ptr<DM<DType>> DM<DType>::get_instance() {
  static std::mutex mutex_;
  if (instance_ == nullptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance_ == nullptr) {
      instance_ = std::make_shared<DM<DType>>();
    }
  }
  return instance_;
}

template <typename DType>
DM<DType>::DM() {
  UNI_LOG_INFO << "DM initializing...";
  init_bank();
}

template <typename DType>
shared_ptr<Bank<DType>> DM<DType>::get_dm_bank() {
  return bank_;
}

template <typename DType>
int32_t DM<DType>::get_dm_height() {
  return bank_height_;
}
template <typename DType>
int32_t DM<DType>::get_dm_width() {
  return bank_width_;
}
template <typename DType>
int32_t DM<DType>::get_dm_size() {
  return bank_size_;
}

template <typename DType>
int32_t DM<DType>::apply(int32_t size) {
  bank_addr_ += size;
  return bank_addr_;
}

template <typename DType>
void DM<DType>::init_bank() {
  auto bank_per_size = 64 * 1024;  // 64kB
  auto row_num =
      ArchCfg::Instance().get_param().overlay_info().aienum_row_num();
  auto col_num =
      ArchCfg::Instance().get_param().overlay_info().aienum_col_num();
  auto core_num = row_num * col_num;
  bank_size_ = bank_per_size * core_num;
  bank_width_ = 8;  // bytes
  bank_height_ = bank_size_ / bank_width_;
  bank_ = std::make_shared<Bank<DType>>(0, bank_height_, bank_width_);
  vector<DPU_DATA_TYPE> data(bank_size_, 0);
  bank_->Write(0, bank_size_, data.data());
  bank_addr_ = 0;
}

template class DM<DPU_DATA_TYPE>;
