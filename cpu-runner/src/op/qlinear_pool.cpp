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

#include "qlinear_pool.hpp"

namespace vart {
namespace cpu {

template <typename DType>
QlinearPool<DType>::QlinearPool(const xir::Subgraph* subg, const xir::Op* op,
                                IMapTBs_t inputs, CPUTBPtr_t output)
    : PoolBase<DType>(subg, op, inputs, output),
      AvgPool<DType>(subg, op, inputs, output),
      MaxPool<DType>(subg, op, inputs, output) {
  auto type_str = op->get_attr<std::string>("type");
  if (type_str == "MAX") {
    pool_type_ = 0;
  } else if (type_str == "AVG"){
    pool_type_ = 1;
  }else{
    UNI_LOG_FATAL(VART_NOT_SUPPORT) << "Unknown avgpool type: " << type_str;
  }

  c_0_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("c0_int");
  c_1_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("c1_int");
  shift_c0_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("shift_c0");
  shift_c1_ = CPUOPBase::xir_op_->get_attr<std::int32_t>("shift_c1");

}

template <typename DType>
void QlinearPool<DType>::run() {
  // // do necessary check
  // print_param();
  // this->check_param();

  // // read data
  // this->read();

  // calc avgpool
  if (pool_type_ == PoolBase<DType>::MAX_POOL) {
    this->max_pool();
    avg_pool_qdq();
  } else {
    this->acc_pool();
    avg_pool_qdq();
  }
  // // do save, debug...
  // this->save();
}

template <typename DType>
void QlinearPool<DType>::print_param() {
  PoolBase<DType>::print_param();

  UNI_LOG_DEBUG_INFO << "pooltype = " << pool_type_ << endl;
  UNI_LOG_DEBUG_INFO << "fix_width = " << fix_width_ << endl;
}

template <typename DType>
void QlinearPool<DType>::avg_pool_qdq() {
  if (CPU_RUN_MODE == CPURunMode::NORMAL_THREAD ||
      CPU_RUN_MODE == CPURunMode::GEMM_THREAD) {
    avg_pool_qdq_thread();
  } else {
    avg_pool_qdq_normal();
  }
}

template <typename DType>
void QlinearPool<DType>::avg_pool_qdq_normal() {
  for (auto i = 0; i < fmap_o_.num(); i++) {
    avg_pool_qdq_one(i);
  }
}

template <typename DType>
void QlinearPool<DType>::avg_pool_qdq_thread() {
  std::vector<std::future<int>> thr_fut(THREAD_NUM);

  for (auto i = 0U; i < THREAD_NUM; i++) {
    // thr_fut[i] = std::async(std::launch::async | std::launch::deferred,
    thr_fut[i] = std::async(
        std::launch::async,
        [this](decltype(i) i) {
          auto BASE_POS = i * THREAD_WORKLOAD;
          for (auto j = 0U; j < THREAD_WORKLOAD; j++) {
            auto pos = BASE_POS + j;
            if (pos >= FMAP_SIZE) break;

            avg_pool_qdq_one(pos);
          }

          return 0;
        },
        i);
  }

  for (auto i = 0U; i < THREAD_NUM; i++) {
    thr_fut[i].wait();
  }
}

static int32_t round_even(int64_t n, int shift, int32_t min, int32_t max) {
  int64_t mask = 1LL << (shift - 1);
  int32_t integer_part = static_cast<int32_t>(n >> shift);
  if ((n & mask)) {
    if (integer_part & 1) {
      integer_part++;
    } else {
      int64_t fractional_mask = (mask - 1);
      if (n & fractional_mask) {
        integer_part++;
      }
    }
  }
  if (integer_part > max) integer_part = max;
  if (integer_part < min) integer_part = min;
  return integer_part;
}

template <typename DType>
void QlinearPool<DType>::avg_pool_qdq_one(int i) {
  std::int64_t tmp = 0;
  if (shift_c0_ > shift_c1_) {
    tmp = (static_cast<std::int64_t>(c_0_) * data_out_ptr_[i]) + (static_cast<std::int64_t>(c_1_) << (shift_c0_ - shift_c1_));
    data_out_ptr_[i] = round_even(tmp, shift_c0_,data_min_,data_max_);
  } else {
    tmp = (static_cast<std::int64_t>(c_0_) * data_out_ptr_[i]) + (static_cast<std::int64_t>(c_1_) >> (shift_c1_ - shift_c0_));
    data_out_ptr_[i] = round_even(tmp, shift_c0_,data_min_,data_max_);
  }
}

INSTANTIATE_TPCLASS(QlinearPool);
REG_OP_INSTANCE_FUNC("qlinear-pool", QlinearPool);

}  // namespace cpu
}  // namespace vart
