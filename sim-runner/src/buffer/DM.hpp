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

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "buffer/Bank.hpp"
#include "inst/InstTable.hpp"

/**
 * @brief on-chip buffer data structure
 * @details on-chip buffer is arranged as multi-bank structure,
 * for each type of data, the bank configuration is different,
 * which is indicated in the SimCfg.txt file
 */
template <typename DType>
class DM {
 public:
  DM();
  DM(const DM&) = delete;
  DM& operator=(const DM&) = delete;
  static std::shared_ptr<DM<DType>> get_instance();

  int32_t get_dm_height();
  int32_t get_dm_width();
  int32_t get_dm_size();
  int32_t apply(int32_t size);
  std::shared_ptr<Bank<DType>> get_dm_bank();

 private:
  void init_bank();

 private:
  int32_t bank_size_;
  int32_t bank_height_;
  int32_t bank_width_;
  int32_t bank_addr_;
  std::shared_ptr<Bank<DType>> bank_;
  static std::shared_ptr<DM<DType>> instance_;
};

template <typename DType>
std::shared_ptr<DM<DType>> DM<DType>::instance_;
