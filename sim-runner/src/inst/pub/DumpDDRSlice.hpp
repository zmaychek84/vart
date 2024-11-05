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
#ifndef __DUMPDDRSLICE_HPP__
#define __DUMPDDRSLICE_HPP__

#include "InstBase.hpp"

template <DPUVersion dv> class DumpDDRSlice : public InstBase {
public:
  // real constructor
  explicit DumpDDRSlice(int inst_type, int instid, vector<string> &inst_str,
                        vector<uint32_t> &inst_val);
  ~DumpDDRSlice();

public:
  // excute load instruction
  void Exec() override final;
  void GenInstStr(int inst_fmt) override final;

public:
  void dump_ddr_range();

private:
  string save_name_;
  int save_fmt_;
  int ddr_reg_id_;
  uint32_t ddr_start_;
  uint32_t fmap_h_;
  uint32_t fmap_hs_;
  uint32_t fmap_w_;
  uint32_t fmap_ws_;
  uint32_t fmap_c_;
  uint32_t fmap_cs_;

  uint32_t ddr_size_;

  // uint32_t tile_h_;
  // uint32_t tile_w_;
  // uint32_t right_width_;
  // uint32_t mid_width_;
  // uint32_t ddr_right_offset_w_;
  // uint32_t ddr_mid_offset_w_;
  // uint32_t ddr_bottom_offset_h_;
  // uint32_t ddr_mid_offset_h_;
  // uint32_t bottom_height_;
  // uint32_t mid_height_;
  uint32_t b0_loc_w_;
  uint32_t b0_loc_h_;
  uint32_t b1_loc_w_;
  uint32_t b1_loc_h_;
  uint32_t b2_loc_w_;
  uint32_t b2_loc_h_;
  uint32_t b3_loc_w_;
  uint32_t b3_loc_h_;
  uint32_t b4_loc_w_;
  uint32_t b4_loc_h_;
  uint32_t b5_loc_w_;
  uint32_t b5_loc_h_;
  uint32_t b6_loc_w_;
  uint32_t b6_loc_h_;
  uint32_t b7_loc_w_;
  uint32_t b7_loc_h_;
  uint32_t b8_loc_w_;
  uint32_t b8_loc_h_;
  uint32_t b9_loc_w_;
  uint32_t b9_loc_h_;
  uint32_t b10_loc_w_;
  uint32_t b10_loc_h_;
  uint32_t b11_loc_w_;
  uint32_t b11_loc_h_;
  uint32_t b12_loc_w_;
  uint32_t b12_loc_h_;
  uint32_t b13_loc_w_;
  uint32_t b13_loc_h_;
  uint32_t b14_loc_w_;
  uint32_t b14_loc_h_;
  uint32_t b15_loc_w_;
  uint32_t b15_loc_h_;
};

#endif /* __DUMPDDRSLICE_HPP__ */
