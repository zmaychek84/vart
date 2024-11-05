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
#include "Save.hpp"

#include <memory.h>

#include "SimCfg.hpp"
#include "pub/DumpBank.hpp"
#include "xv2dpu/simCfg.hpp"

template <>
Save<DPUVersion::DPUV2>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV2>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV2>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];
}

template <>
Save<DPUVersion::DPUV3E>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV3E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3E>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::DPUV4E>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV4E>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV4E>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::DPUV3ME>::Save(int inst_type, int instid,
                                vector<string>& inst_str,
                                vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPUV3ME>::InstInit(inst_type_, root_debug_path_, debug_,
                                        opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPUV3ME>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <>
Save<DPUVersion::XVDPU>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::XVDPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XVDPU>::inst_table;
  // get save instruction field value
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];

  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];

  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
}

template <>
Save<DPUVersion::XV2DPU>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::XV2DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV2DPU>::inst_table;
  // get save instruction field value
  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR];

  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];

  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];

  // length_mid_ = field_val_[TABLE::SAVE_FIELD_LENGTH_MID];
  // length_right_ = field_val_[TABLE::SAVE_FIELD_LENGTH_RIGHT];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
  // tile_w_ = field_val_[TABLE::SAVE_FIELD_TILE_W];
  // tile_h_ = field_val_[TABLE::SAVE_FIELD_TILE_H];
  // mt_addr_h_skip_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR_H_SKIP];
  // mt_addr_offset_w_right_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR_OFFSET_W_RIGHT];
  // mt_addr_offset_w_mid_   = field_val_[TABLE::SAVE_FIELD_MT_ADDR_OFFSET_W_MID];
  // ddr_addr_offset_right_  = field_val_[TABLE::SAVE_FIELD_DDR_ADDR_OFFSET_RIGHT];
  // ddr_addr_offset_mid_    = field_val_[TABLE::SAVE_FIELD_DDR_ADDR_OFFSET_MID];
  // b0_loc_w_ = field_val_[TABLE::SAVE_FIELD_B0_LOC_W];
  // b0_loc_h_ = field_val_[TABLE::SAVE_FIELD_B0_LOC_H];
  // b1_loc_w_ = field_val_[TABLE::SAVE_FIELD_B1_LOC_W];
  // b1_loc_h_ = field_val_[TABLE::SAVE_FIELD_B1_LOC_H];
  // b2_loc_w_ = field_val_[TABLE::SAVE_FIELD_B2_LOC_W];
  // b2_loc_h_ = field_val_[TABLE::SAVE_FIELD_B2_LOC_H];
  // b3_loc_w_ = field_val_[TABLE::SAVE_FIELD_B3_LOC_W];
  // b3_loc_h_ = field_val_[TABLE::SAVE_FIELD_B3_LOC_H];
  // b4_loc_w_ = field_val_[TABLE::SAVE_FIELD_B4_LOC_W];
  // b4_loc_h_ = field_val_[TABLE::SAVE_FIELD_B4_LOC_H];
  // b5_loc_w_ = field_val_[TABLE::SAVE_FIELD_B5_LOC_W];
  // b5_loc_h_ = field_val_[TABLE::SAVE_FIELD_B5_LOC_H];
  // b6_loc_w_ = field_val_[TABLE::SAVE_FIELD_B6_LOC_W];
  // b6_loc_h_ = field_val_[TABLE::SAVE_FIELD_B6_LOC_H];
  // b7_loc_w_ = field_val_[TABLE::SAVE_FIELD_B7_LOC_W];
  // b7_loc_h_ = field_val_[TABLE::SAVE_FIELD_B7_LOC_H];
  // b8_loc_w_ = field_val_[TABLE::SAVE_FIELD_B8_LOC_W];
  // b8_loc_h_ = field_val_[TABLE::SAVE_FIELD_B8_LOC_H];
  // b9_loc_w_ = field_val_[TABLE::SAVE_FIELD_B9_LOC_W];
  // b9_loc_h_ = field_val_[TABLE::SAVE_FIELD_B9_LOC_H];
  // b10_loc_w_ = field_val_[TABLE::SAVE_FIELD_B10_LOC_W];
  // b10_loc_h_ = field_val_[TABLE::SAVE_FIELD_B10_LOC_H];
  // b11_loc_w_ = field_val_[TABLE::SAVE_FIELD_B11_LOC_W];
  // b11_loc_h_ = field_val_[TABLE::SAVE_FIELD_B11_LOC_H];
  // b12_loc_w_ = field_val_[TABLE::SAVE_FIELD_B12_LOC_W];
  // b12_loc_h_ = field_val_[TABLE::SAVE_FIELD_B12_LOC_H];
  // b13_loc_w_ = field_val_[TABLE::SAVE_FIELD_B13_LOC_W];
  // b13_loc_h_ = field_val_[TABLE::SAVE_FIELD_B13_LOC_H];
  // b14_loc_w_ = field_val_[TABLE::SAVE_FIELD_B14_LOC_W];
  // b14_loc_h_ = field_val_[TABLE::SAVE_FIELD_B14_LOC_H];
  // b15_loc_w_ = field_val_[TABLE::SAVE_FIELD_B15_LOC_W];
  // b15_loc_h_ = field_val_[TABLE::SAVE_FIELD_B15_LOC_H];

  bank_id_ = 0;
  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];

    // auto engine_id = SimCfg::Instance().get_tiling_engine_id();
    // // engine_id => batch_loc
    // auto field_idx_w = TABLE::SAVE_FIELD_B0_LOC_W - 2 * (engine_id % 4) +
    //                           (engine_id / 4) * 8;
    // auto field_idx_h = TABLE::SAVE_FIELD_B0_LOC_H - 2 * (engine_id % 4) + 
    //                           (engine_id / 4) * 8;
    // int  b_loc_w = field_val_[field_idx_w];
    // int  b_loc_h = field_val_[field_idx_h];

    // // use b_loc_w/b_loc_h and tile_w/tile_h to calculate position
    // if (tile_w_ == 1 && tile_h_ == 1) {
    //   // not tiling subgraph pattern 
    // } else {
    //   if (b_loc_h == 0) {
    //     // top
    //     mt_addr_skip_valid_ = ((mt_addr_h_skip_ >> 2) & 0x1);
    //   } else if (b_loc_h == tile_h_ - 1) {
    //     // bottom
    //     mt_addr_skip_valid_ = mt_addr_h_skip_ & 0x1;
    //   } else {
    //     // mid
    //     mt_addr_skip_valid_ = ((mt_addr_h_skip_ >> 1) & 0x1);
    //   }
    //   // udpate bank_addr
    //   int32_t bank_offset_val = 0;
    //   int32_t ddr_offset_val  = 0;
    //   if (b_loc_w == 0) {
    //     // do nothing
    //   } else if (b_loc_w == tile_w_ - 1) {
    //     length_ = length_right_;
    //     bank_offset_val = mt_addr_offset_w_right_;  
    //     ddr_offset_val  = ddr_addr_offset_right_;
    //   } else {
    //     length_ = length_mid_;
    //     bank_offset_val = mt_addr_offset_w_mid_;
    //     ddr_offset_val  = ddr_addr_offset_mid_;
    //   }
    //   bank_addr_ += bank_offset_val;
    //   ddr_addr_ += ddr_offset_val;
    //   mt_addr_skip_valid_ = (length_ == 0) ? 1 : mt_addr_skip_valid_;
    // }
    // // update ddr_addr_
    // auto ddr_offset_map = SimCfg::Instance().get_ddr_addr_offset_map();
    // auto fake_rid_to_real_rid_map = SimCfg::Instance().get_fake_rid_to_real_rid_map();
    // auto combine_num_to_str = [] (int32_t v1, int32_t v2) -> 
    //     std::string { return std::to_string(v1) + "_" + std::to_string(v2); };
    // auto key_str = combine_num_to_str(reg_id_, engine_id);
    // if (ddr_offset_map.count(key_str)) {
    //   ddr_addr_ += ddr_offset_map.at(key_str);
    //   reg_id_ = fake_rid_to_real_rid_map.at(key_str);
    // }
}

template <>
Save<DPUVersion::XV3DPU>::Save(int inst_type, int instid,
                               vector<string>& inst_str,
                               vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::XV3DPU>::InstInit(inst_type_, root_debug_path_, debug_,
                                       opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::XV3DPU>::inst_table;
  // get save instruction field value
  const_en_ = field_val_[TABLE::SAVE_FIELD_CONST_EN];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR];

  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  const_value_ = field_val_[TABLE::SAVE_FIELD_CONST_VALUE];
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_write_endl_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE_ENDL];

  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  jump_read_endl_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ_ENDL];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];

  block_num_ = field_val_[TABLE::SAVE_FIELD_BLOCK_NUM];

  // length_mid_ = field_val_[TABLE::SAVE_FIELD_LENGTH_MID];
  // length_right_ = field_val_[TABLE::SAVE_FIELD_LENGTH_RIGHT];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
  // tile_w_ = field_val_[TABLE::SAVE_FIELD_TILE_W];
  // tile_h_ = field_val_[TABLE::SAVE_FIELD_TILE_H];
  // mt_addr_h_skip_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR_H_SKIP];
  // mt_addr_offset_w_right_ = field_val_[TABLE::SAVE_FIELD_MT_ADDR_OFFSET_W_RIGHT];
  // mt_addr_offset_w_mid_   = field_val_[TABLE::SAVE_FIELD_MT_ADDR_OFFSET_W_MID];
  // ddr_addr_offset_right_  = field_val_[TABLE::SAVE_FIELD_DDR_ADDR_OFFSET_RIGHT];
  // ddr_addr_offset_mid_    = field_val_[TABLE::SAVE_FIELD_DDR_ADDR_OFFSET_MID];
  // b0_loc_w_ = field_val_[TABLE::SAVE_FIELD_B0_LOC_W];
  // b0_loc_h_ = field_val_[TABLE::SAVE_FIELD_B0_LOC_H];
  // b1_loc_w_ = field_val_[TABLE::SAVE_FIELD_B1_LOC_W];
  // b1_loc_h_ = field_val_[TABLE::SAVE_FIELD_B1_LOC_H];
  // b2_loc_w_ = field_val_[TABLE::SAVE_FIELD_B2_LOC_W];
  // b2_loc_h_ = field_val_[TABLE::SAVE_FIELD_B2_LOC_H];
  // b3_loc_w_ = field_val_[TABLE::SAVE_FIELD_B3_LOC_W];
  // b3_loc_h_ = field_val_[TABLE::SAVE_FIELD_B3_LOC_H];
  // b4_loc_w_ = field_val_[TABLE::SAVE_FIELD_B4_LOC_W];
  // b4_loc_h_ = field_val_[TABLE::SAVE_FIELD_B4_LOC_H];
  // b5_loc_w_ = field_val_[TABLE::SAVE_FIELD_B5_LOC_W];
  // b5_loc_h_ = field_val_[TABLE::SAVE_FIELD_B5_LOC_H];
  // b6_loc_w_ = field_val_[TABLE::SAVE_FIELD_B6_LOC_W];
  // b6_loc_h_ = field_val_[TABLE::SAVE_FIELD_B6_LOC_H];
  // b7_loc_w_ = field_val_[TABLE::SAVE_FIELD_B7_LOC_W];
  // b7_loc_h_ = field_val_[TABLE::SAVE_FIELD_B7_LOC_H];
  // b8_loc_w_ = field_val_[TABLE::SAVE_FIELD_B8_LOC_W];
  // b8_loc_h_ = field_val_[TABLE::SAVE_FIELD_B8_LOC_H];
  // b9_loc_w_ = field_val_[TABLE::SAVE_FIELD_B9_LOC_W];
  // b9_loc_h_ = field_val_[TABLE::SAVE_FIELD_B9_LOC_H];
  // b10_loc_w_ = field_val_[TABLE::SAVE_FIELD_B10_LOC_W];
  // b10_loc_h_ = field_val_[TABLE::SAVE_FIELD_B10_LOC_H];
  // b11_loc_w_ = field_val_[TABLE::SAVE_FIELD_B11_LOC_W];
  // b11_loc_h_ = field_val_[TABLE::SAVE_FIELD_B11_LOC_H];
  // b12_loc_w_ = field_val_[TABLE::SAVE_FIELD_B12_LOC_W];
  // b12_loc_h_ = field_val_[TABLE::SAVE_FIELD_B12_LOC_H];
  // b13_loc_w_ = field_val_[TABLE::SAVE_FIELD_B13_LOC_W];
  // b13_loc_h_ = field_val_[TABLE::SAVE_FIELD_B13_LOC_H];
  // b14_loc_w_ = field_val_[TABLE::SAVE_FIELD_B14_LOC_W];
  // b14_loc_h_ = field_val_[TABLE::SAVE_FIELD_B14_LOC_H];
  // b15_loc_w_ = field_val_[TABLE::SAVE_FIELD_B15_LOC_W];
  // b15_loc_h_ = field_val_[TABLE::SAVE_FIELD_B15_LOC_H];

  bank_id_ = 0;
  argmax_ = field_val_[TABLE::SAVE_FIELD_ARGMAX];

    // auto engine_id = SimCfg::Instance().get_tiling_engine_id();
    // // engine_id => batch_loc
    // auto field_idx_w = TABLE::SAVE_FIELD_B0_LOC_W - 2 * (engine_id % 4) +
    //                           (engine_id / 4) * 8;
    // auto field_idx_h = TABLE::SAVE_FIELD_B0_LOC_H - 2 * (engine_id % 4) + 
    //                           (engine_id / 4) * 8;
    // int  b_loc_w = field_val_[field_idx_w];
    // int  b_loc_h = field_val_[field_idx_h];

    // // use b_loc_w/b_loc_h and tile_w/tile_h to calculate position
    // if (tile_w_ == 1 && tile_h_ == 1) {
    //   // not tiling subgraph pattern 
    // } else {
    //   if (b_loc_h == 0) {
    //     // top
    //     mt_addr_skip_valid_ = ((mt_addr_h_skip_ >> 2) & 0x1);
    //   } else if (b_loc_h == tile_h_ - 1) {
    //     // bottom
    //     mt_addr_skip_valid_ = mt_addr_h_skip_ & 0x1;
    //   } else {
    //     // mid
    //     mt_addr_skip_valid_ = ((mt_addr_h_skip_ >> 1) & 0x1);
    //   }
    //   // udpate bank_addr
    //   int32_t bank_offset_val = 0;
    //   int32_t ddr_offset_val  = 0;
    //   if (b_loc_w == 0) {
    //     // do nothing
    //   } else if (b_loc_w == tile_w_ - 1) {
    //     length_ = length_right_;
    //     bank_offset_val = mt_addr_offset_w_right_;  
    //     ddr_offset_val  = ddr_addr_offset_right_;
    //   } else {
    //     length_ = length_mid_;
    //     bank_offset_val = mt_addr_offset_w_mid_;
    //     ddr_offset_val  = ddr_addr_offset_mid_;
    //   }
    //   bank_addr_ += bank_offset_val;
    //   ddr_addr_ += ddr_offset_val;
    //   mt_addr_skip_valid_ = (length_ == 0) ? 1 : mt_addr_skip_valid_;
    // }
    // // update ddr_addr_
    // auto ddr_offset_map = SimCfg::Instance().get_ddr_addr_offset_map();
    // auto fake_rid_to_real_rid_map = SimCfg::Instance().get_fake_rid_to_real_rid_map();
    // auto combine_num_to_str = [] (int32_t v1, int32_t v2) -> 
    //     std::string { return std::to_string(v1) + "_" + std::to_string(v2); };
    // auto key_str = combine_num_to_str(reg_id_, engine_id);
    // if (ddr_offset_map.count(key_str)) {
    //   ddr_addr_ += ddr_offset_map.at(key_str);
    //   reg_id_ = fake_rid_to_real_rid_map.at(key_str);
    // }
}  

template <>
Save<DPUVersion::DPU4F>::Save(int inst_type, int instid,
                              vector<string>& inst_str,
                              vector<uint32_t>& inst_val)
    : InstBase(inst_type, instid, inst_str, inst_val) {
  // debug
  debug_ = debug_ && SimCfg::Instance().get_debug_instr(SimCfg::DBG_INSTR_SAVE);
  Helper<DPUVersion::DPU4F>::InstInit(inst_type_, root_debug_path_, debug_,
                                      opcode_, debug_path_);
  using TABLE = TableInterface<DPUVersion::DPU4F>::inst_table;
  // get save instruction field value
  hp_id_ = field_val_[TABLE::SAVE_FIELD_HP_ID];
  bank_id_ = field_val_[TABLE::SAVE_FIELD_BANK_ID];
  bank_addr_ = field_val_[TABLE::SAVE_FIELD_BANK_ADDR];
  quant_lth_ = field_val_[TABLE::SAVE_FIELD_QUANT_LTH];  // 0: 4 bit 1: 8 bit
  jump_write_ = field_val_[TABLE::SAVE_FIELD_JUMP_WRITE];
  jump_read_ = field_val_[TABLE::SAVE_FIELD_JUMP_READ];
  length_ = field_val_[TABLE::SAVE_FIELD_LENGTH];
  channel_ = field_val_[TABLE::SAVE_FIELD_CHANNEL];
  reg_id_ = field_val_[TABLE::SAVE_FIELD_REG_ID];
  ddr_addr_ = field_val_[TABLE::SAVE_FIELD_DDR_ADDR];
}

template <DPUVersion T>
Save<T>::~Save() {}

// public funcs
template <DPUVersion T>
void Save<T>::Exec() {
  if (mt_addr_skip_valid_) {
    return;
  }
  target_check();
  size_t ddr_size = 0;
  auto* ddr_img = DDR::Instance().GetAddr(reg_id_, 0, &ddr_size);
  UNI_LOG_CHECK((int32_t)(ddr_addr_ + (length_ - 1) * jump_write_ + channel_) <=
                    static_cast<int32_t>(DDR::Instance().GetSize(reg_id_)),
                SIM_OUT_OF_RANGE)
                << "ddr_addr " << ddr_addr_ << " reg_id" << reg_id_ << endl;
  UNI_LOG_CHECK(argmax_ >= 0 && argmax_ <= 3, SIM_OUT_OF_RANGE);
  // decode argmax
  auto if_argmax = (this->argmax_ >> 1);
  std::string argmax_rlt =
      (this->argmax_ & 0x1) ? std::string{"data"} : std::string{"index"};

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  for (auto idx_block = 0; idx_block < block_num_; ++idx_block) {
    for (auto pixel = 0; pixel < length_; pixel++) {
      auto ddr_addr =
          ddr_addr_ + pixel * jump_write_ + idx_block * jump_write_endl_;

      if (const_en_) {  // write const
        memset(ddr_img + ddr_addr, const_value_, channel_);
      } else if (if_argmax) {  // enable argmax
        auto bank_addr =
            (bank_addr_ + pixel * jump_read_ + idx_block * jump_read_endl_) %
            bank_depth;
        auto data = vector<DPU_DATA_TYPE>(channel_);
        bank->Read(bank_addr, channel_, data.data());
        auto max_index = argmax(data);
        ddr_img[ddr_addr] =
            argmax_rlt == std::string{"index"} ? max_index : data.at(max_index);
      } else {  // normal save
        auto bank_addr =
            (bank_addr_ + pixel * jump_read_ + idx_block * jump_read_endl_) %
            bank_depth;
        UNI_LOG_CHECK_LT(ddr_addr, ddr_size, SIM_ADDR_OVERFLOW);
        bank->Read(bank_addr, channel_,
                   reinterpret_cast<DPU_DATA_TYPE*>(ddr_img + ddr_addr));
        if (SimCfg::Instance().get_memory_doubleWrite_check()) {
          DDRblock ddrBlock(reg_id_, ddr_addr, channel_);
          DDRblock::insert(debug_path_, ddrBlock);
        }
      }
    }
  }

  // record ddr used
  int hp_width = SimCfg::Instance().get_hp_width();
  for (auto idx_block = 0; idx_block < block_num_; ++idx_block) {
    for (auto pixel = 0; pixel < length_; pixel++) {
      auto addr_base =
          ddr_addr_ + pixel * jump_write_ + idx_block * jump_write_endl_;
      int32_t num_bytes = hp_width - (addr_base % hp_width);
      int num_entry = ceil((double)channel_ / hp_width);
      if (num_bytes != 0) {
        if (channel_ <= num_bytes)
          num_entry = 1;
        else
          num_entry = ceil((double)(channel_ - num_bytes) / hp_width) + 1;
      }

      for (auto i = 0; i < num_entry; i++) {
        auto addr = addr_base + i * hp_width;
        DDR::Instance().set_addr_used(reg_id_, addr);
      }
    }
  }

  debug_tick();

  save_ofm_PL();
  save_gen_ctrlpkg();
}

// public funcs
template <>
void Save<DPUVersion::DPU4F>::Exec() {
  target_check();
  auto* ddr_img = DDR::Instance().GetAddr(reg_id_, 0);
  UNI_LOG_CHECK(length_ * jump_write_ <=
                    static_cast<int32_t>(DDR::Instance().GetSize(reg_id_)),
                SIM_OUT_OF_RANGE);
  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();

  for (auto pixel = 0; pixel < length_; pixel++) {
    auto ddr_addr = ddr_addr_ + pixel * jump_write_;
    auto bank_addr = (bank_addr_ + pixel * jump_read_) % bank_depth;
    vector<DPU_DATA_TYPE> data_buf(channel_);
    BankShell::read(quant_lth_, bank_id_, channel_, bank_addr, data_buf.data());
    combine(quant_lth_, data_buf,
            reinterpret_cast<DPU_DATA_TYPE*>(ddr_img + ddr_addr));
  }
  debug_tick();
}

template <DPUVersion T>
void Save<T>::combine(bool is_8_bit, vector<DPU_DATA_TYPE> data_in,
                      DPU_DATA_TYPE* data_out) {
  auto num =
      is_8_bit ? data_in.size() : (data_in.size() / 2 + data_in.size() % 2);
  for (auto idx_data = 0U; idx_data < num; idx_data++) {
    if (is_8_bit)
      data_out[idx_data] = data_in.at(idx_data);
    else {
      auto lo = data_in.at(2 * idx_data);
      auto hi = static_cast<DPU_DATA_TYPE>(
          ((idx_data < num - 1) || (data_in.size() % 2 == 0))
              ? data_in.at(2 * idx_data + 1)
              : 0);
      data_out[idx_data] = static_cast<DPU_DATA_TYPE>((hi << 4) | (0x0F & lo));
    }
  }
}

template <DPUVersion T>
DPU_DATA_TYPE Save<T>::argmax(const vector<DPU_DATA_TYPE>& data_in) {
  auto num = data_in.size();

  DPU_DATA_TYPE pos = 0;
  auto max = data_in[pos];
  for (auto idx_data = 1U; idx_data < num; ++idx_data) {
    if (data_in[idx_data] > max) {
      max = data_in[idx_data];
      pos = idx_data;
    }
  }
  return pos;
}

template <DPUVersion T>
void Save<T>::debug_tick() {
  if (!debug_) return;

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  for (int i = 0; i < length_; i++) {
    int bank_addr = bank_addr_ + i * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContBigEndBankAddr(
        save_name, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }
}

template <>
void Save<DPUVersion::DPUV4E>::debug_tick() {
  if (!debug_) return;

  string save_name_input = debug_path_ + "input_inst_" + to_string(instid_) +
                           "tick." + to_string(instid_classify_);
  string save_name_output = debug_path_ + "output_inst_" + to_string(instid_) +
                            "tick." + to_string(instid_classify_);

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  auto* ddr_buf = reinterpret_cast<DPU_DATA_TYPE*>(
      DDR::Instance().GetAddr(reg_id_, ddr_addr_));

  for (auto idx_length = 0; idx_length < length_; idx_length++) {
    auto bank_addr = idx_length * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContSmallEndBankAddr(
        save_name_input, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);

    auto ddr_offset = ddr_addr_ + idx_length * jump_write_;
    Util::SaveHexContSmallEndDDRAddr(save_name_output, &ddr_buf[ddr_offset],
                                     channel_ * sizeof(DPU_DATA_TYPE),
                                     hp_width_, ddr_addr_ + ddr_offset, reg_id_,
                                     SM_APPEND);
  }
}
template <>
void Save<DPUVersion::DPU4F>::debug_tick() {
  if (!debug_) return;

  string save_name = debug_path_ + "inst_" + to_string(instid_) + "_" +
                     to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  // how many DPU_DATA_TYPE element save engine can write to DDR at once
  auto cp = ArchCfg::Instance().get_param().save_engine().channel_parallel();
  auto bank_w = quant_lth_ ? bank_width : bank_width / 2;
  cp = quant_lth_ ? cp : 2 * cp;
  int cg = ceil((double)channel_ / cp);
  int pixel_channel = cg * cp;
  vector<DPU_DATA_TYPE> inter_buf(pixel_channel);
  auto data_size = quant_lth_ ? pixel_channel : pixel_channel / 2;
  vector<DPU_DATA_TYPE> buf(data_size);

  for (int i = 0; i < length_; i++) {
    int bank_addr = (bank_addr_ + i * jump_read_) % bank_depth;

    BankShell::read(quant_lth_, bank_id_, pixel_channel, bank_addr,
                    inter_buf.data());
    combine(quant_lth_, inter_buf, buf.data());
    Util::SaveHexContSmallEndBankAddr(
        save_name, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_w * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }
}

template <>
void Save<DPUVersion::XV2DPU>::debug_tick() {
  if (!debug_) return;

  string save_name_input = debug_path_ + "input_inst_" + to_string(instid_) +
                           "_" + to_string(instid_classify_) + ".tick";
  string save_name_output = debug_path_ + "output_inst_" + to_string(instid_) +
                            "_" + to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  for (int i = 0; i < length_; i++) {
    int bank_addr = bank_addr_ + i * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContBigEndBankAddr(
        save_name_input, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }

  // for compare reasons, use bank_width to show the ddr data
  auto* ddr_bufs = reinterpret_cast<DPU_DATA_TYPE*>
                    (DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  Util::SaveHexContBigEndDDRAddr(save_name_output, &ddr_bufs[0], 
  channel_ * sizeof(DPU_DATA_TYPE) * length_, bank_width, ddr_addr_, reg_id_,SM_TRUNC);
}

template <>
void Save<DPUVersion::XV3DPU>::debug_tick() {
  if (!debug_) return;

  string save_name_input = debug_path_ + "input_inst_" + to_string(instid_) +
                           "_" + to_string(instid_classify_) + ".tick";
  string save_name_output = debug_path_ + "output_inst_" + to_string(instid_) +
                            "_" + to_string(instid_classify_) + ".tick";

  auto bank = Buffer<DPU_DATA_TYPE>::Instance().GetBank(bank_id_);
  UNI_LOG_CHECK(bank != nullptr, SIM_OUT_OF_RANGE)
      << "bank_id " << bank_id_ << " out of range!" << endl;
  auto bank_depth = bank->GetH();
  auto bank_width = bank->GetW();

  int cg = ceil((double)channel_ / bank_width);
  int pixel_channel = cg * bank_width;
  vector<DPU_DATA_TYPE> buf(pixel_channel);

  for (int i = 0; i < length_; i++) {
    int bank_addr = bank_addr_ + i * jump_read_;

    bank->Read(bank_addr, pixel_channel, buf.data());
    Util::SaveHexContBigEndBankAddr(
        save_name_input, reinterpret_cast<const char*>(buf.data()),
        buf.size() * sizeof(DPU_DATA_TYPE), bank_width * sizeof(DPU_DATA_TYPE),
        bank_depth, bank_id_, bank_addr, SM_APPEND);
  }

  // for compare reasons, use bank_width to show the ddr data
  auto* ddr_bufs = reinterpret_cast<DPU_DATA_TYPE*>
                    (DDR::Instance().GetAddr(reg_id_, ddr_addr_));
  Util::SaveHexContBigEndDDRAddr(save_name_output, &ddr_bufs[0], 
  channel_ * sizeof(DPU_DATA_TYPE) * length_, bank_width, ddr_addr_, reg_id_,SM_TRUNC);
}

template <DPUVersion T>
void Save<T>::target_check() {
  // bank id check
  auto white_list = ArchCfg::Instance().get_bank_access_white_list("save-in");
  UNI_LOG_CHECK((white_list.count(bank_id_) > 0), SIM_OUT_OF_RANGE)
      << "bank_id (= " << bank_id_ << ") does not match target!" << endl;
}

template <DPUVersion T>
void Save<T>::GenInstStr(int inst_fmt) {
  ac_ = Helper<T>::GetInstStr(inst_type_, inst_fmt, dpdon_, dpdby_, field_str_);
}
