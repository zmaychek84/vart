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

#include <array>
#include <set>
#include <string>
#include "ArchCfg.hpp"
#include "UniLog/UniLog.hpp"

using namespace std;

enum InstFmt {
  INST_FMT_MIN = 0,
  INST_FMT_BIN = INST_FMT_MIN,
  INST_FMT_AC_DESP_NO,
  INST_FMT_AC_DESP_YES,
  INST_FMT_MAX,
};

/**
 * @brief configuration interface for simulator
 */
class SimCfg {
 public:
  enum DbgInstType {
    DBG_INSTR_LOAD,
    DBG_INSTR_SAVE,
    DBG_INSTR_CONV,
    DBG_INSTR_POOL,
    DBG_INSTR_DWCONV,
    DBG_INSTR_ELEW,
    DBG_INSTR_THD,
    DBG_INSTR_DUMP,
    DBG_INSTR_ALU,
    DBG_INSTR_MAX
  };

 public:
  /**
   * @brief Static function to create a singleton SimCfg object
   * @param config_name Name of the configuration file
   * @return a global SimCfg instance object
   */
  static SimCfg& Instance(
      const std::string& config_fname = "./config/SIMCfg.txt") {
    static SimCfg sc(config_fname);
    return sc;
  }

 public:
  // access interface
  int get_hp_width() const;
  void set_fmap_bank_group();
  int get_fmap_bank_group() const;
  std::string get_run_mode() const;
  void set_run_mode(std::string);
  int get_isa_version() const;
  int get_inst_type_max() const;
  std::string get_inst_type_name(int idx) const;
  void set_isa_version(const std::string isa);

  bool get_bank_init() const;
  int get_bank_init_type() const;
  std::string get_bank_init_file() const;

  void disable_debug();
  void enable_debug();
  bool get_debug_enable() const;
  int get_batch_index() const;
  bool get_debug_layer() const;
  bool is_layer_name_in_list(const std::string& layer_name) const;
  std::string get_debug_path() const;
  bool get_ddr_dump_net() const;
  bool get_ddr_dump_layer() const;
  bool get_ddr_dump_init() const;
  bool get_ddr_dump_end() const;
  void set_ddr_dump_end_fast(int val);
  int get_ddr_dump_end_fast() const;
  bool get_ddr_dump_split() const;
  int get_ddr_dump_format() const;
  int get_layer_dump_format() const;
  bool get_dump_instr() const;
  bool get_debug_instr(int type) const;
  void set_batch_index(int idx);
  void set_debug_path(const std::string path);
  void set_debug_layer(bool flag);
  bool get_gen_aie_data() const;
  int get_gen_aie_data_format() const;
  int get_xvdpu_conv_remain() const;
  void set_xvdpu_conv_remain(int conv_remain);
  int get_xvdpu_conv_num() const;
  void set_xvdpu_conv_num(int conv_num);
  bool get_co_sim_on();
  bool get_memory_doubleWrite_check();
  int get_save_parallel() const;
  int get_load_img_parallel() const;
  int get_load_wgt_parallel() const;
  int get_MT_IMG_CHN_LOAD() const;   
  int get_MT_IMG_CHN_SAVE() const;   
  int get_MT_WGT_CHN_LOAD_L1() const;
  int get_MT_WGT_CHN_LOAD_L2() const;
  int get_MT_IMG_CHN_CONV_IFM_H0() const; 
  int get_MT_IMG_CHN_CONV_IFM_H1() const; 
  int get_MT_IMG_CHN_CONV_OFM_H0() const; 
  int get_MT_IMG_CHN_CONV_OFM_H1() const; 
  int get_MT_WGT_CHN_CONV() const;
  int get_MT_IMG_CHN_ALU_IFM() const;
  int get_MT_IMG_CHN_ALU_OFM() const;
  int get_MT_WGT_CHN_ALU() const;
  bool get_dump_ddr_all() const;

  void set_tiling_engine_id(int engine_id);
  int  get_tiling_engine_id() const;
  void clear_ddr_addr_offset_map();
  void set_ddr_addr_offset_map(std::string key_str, int32_t offset_value);
  std::unordered_map<std::string, int32_t> get_ddr_addr_offset_map() const;
  void set_reg_id_engine_id_offset_value(std::vector<uint32_t> input_reg_engine_info);
  std::vector<std::vector<uint32_t>> get_rid_eid_offset() const;
  void set_fake_rid_to_real_rid_map(std::string key_str, int32_t real_rid);
  std::unordered_map<std::string, int32_t> get_fake_rid_to_real_rid_map() const;

 private:
  SimCfg(const std::string& config_fname);
  SimCfg(const SimCfg&) = delete;
  SimCfg& operator=(const SimCfg&) = delete;

 private:
  // arch
  int hp_width_;
  int fmap_bank_group_;
  std::string run_mode_;
  std::string isa_version_;

  // bank
  bool bank_init_;
  int bank_init_type_;
  std::string bank_init_file_;

  // debug
  bool debug_;
  std::string debug_path_;
  int batch_index_;
  bool debug_layer_;
  std::set<string> debug_layer_name_list_;
  int layer_dump_format_;
  bool ddr_dump_net_;
  bool ddr_dump_layer_;
  bool ddr_dump_init_;
  bool ddr_dump_end_;
  int ddr_dump_end_fast_;  // 0x10: <ori, now>=<ture, false>
  bool ddr_dump_split_;
  int ddr_dump_format_;
  bool dump_inst_;
  std::array<bool, SimCfg::DBG_INSTR_MAX> debug_inst_;
  bool gen_aie_data_{false};
  int gen_aie_data_format_{2};  // 1-txt, 2-hex, 3-both
  int xvdpu_conv_remain_{0};
  int xvdpu_conv_num_{0};

  bool memory_doubleWrite_check_{false};
  // co-sim
  bool co_sim_on_{false};

  int save_parallel_;
  int load_img_parallel_;
  int load_wgt_parallel_;

  int MT_IMG_CHN_LOAD_;
  int MT_IMG_CHN_SAVE_;
  int MT_WGT_CHN_LOAD_L1_;
  int MT_WGT_CHN_LOAD_L2_;

  int MT_IMG_CHN_CONV_IFM_H0_; 
  int MT_IMG_CHN_CONV_IFM_H1_; 
  int MT_IMG_CHN_CONV_OFM_H0_; 
  int MT_IMG_CHN_CONV_OFM_H1_; 
  int MT_WGT_CHN_CONV_;

  int MT_IMG_CHN_ALU_IFM_;
  int MT_IMG_CHN_ALU_OFM_;
  int MT_WGT_CHN_ALU_;

  bool dump_ddr_all_;

  int tiling_engine_id;
  std::vector<std::vector<uint32_t>> rid_eid_offset_value;
  std::unordered_map<std::string, int32_t> ddr_addr_offset_map_;
  std::unordered_map<std::string, int32_t> fake_rid_to_real_rid_map_;
};
