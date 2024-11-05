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

#include <fstream>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <string>
#include <vart/trace/traceclass.hpp>
#include <xir/graph/graph.hpp>
#if _WIN32
#  include <windows.h>
#endif
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::ios;
using std::make_pair;
using std::map;
using std::ofstream;
using std::ostringstream;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;
using trace_entry_t = map<string, string>;

#define _j(x) (j_format(make_pair(#x, x)))
#define _jkv(k, v) (j_format(make_pair(k, v)))
#define PDI_ID_BIT_OFFSET (10)
#define PDI_ID_MASK ((1 << PDI_ID_BIT_OFFSET) - 1)
#define LONG_PDI_ID_BIT_OFFSET (20)
#define LONG_PDI_ID_MASK ((1 << LONG_PDI_ID_BIT_OFFSET) - 1)
#define MAX_LAYER_ID ((1 << PDI_ID_BIT_OFFSET) - 1)
#define GRAPH_TYPE_BIT_OFFSET (30)
#define GRAPH_TYPE_PDI (1 << GRAPH_TYPE_BIT_OFFSET)
#define GRAPH_TYPE_SUPERLAYER (0 << GRAPH_TYPE_BIT_OFFSET)

string _q(string s) {
  s.insert(0, "\"").append("\"");
  return s;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i != vec.size() - 1) os << ',';
  };
  os << "]";

  return os;
};

template <typename T>
string j_format(pair<const char*, T> x) {
  stringstream ret;
  ret << _q(x.first) << ":" << x.second;
  return ret.str();
}

template <>
string j_format(pair<const char*, string> x) {
  stringstream ret;
  ret << _q(x.first) << ":" << _q(x.second);
  return ret.str();
}

class op_info {
 public:
  op_info() = delete;
  op_info(const xir::Op* op);
  std::string to_json(void);
  std::string to_str(void);

 private:
  string op_name;
  string op_type;

  uint64_t workload;
  uint64_t workload_on_arch;
  int32_t stride_scalar;
  vector<int> i_tensors_shape;
  vector<int> o_tensors_shape;
  vector<int> i_tensors_stride;
  vector<int> o_tensors_stride;
  vector<int> shape;
  vector<int> kernel;
  vector<int> dilation;
  vector<int> pad;
  vector<int> stride;
  string nonlinear;
  string type;
};

op_info::op_info(const xir::Op* op)
    : op_name(op->get_name()),
      op_type(op->get_type()),
      workload(0),
      workload_on_arch(0),
      stride_scalar(0),
      i_tensors_shape({}),
      o_tensors_shape({}),
      i_tensors_stride({}),
      o_tensors_stride({}),
      shape({}),
      kernel({}),
      dilation({}),
      pad({}),
      stride({}),
      nonlinear(""),
      type("") {
  if (op->has_attr("workload"))
    workload = op->get_attr<decltype(workload)>("workload");
  if (op->has_attr("workload_on_arch"))
    workload_on_arch = op->get_attr<decltype(workload)>("workload_on_arch");
  if (op->has_attr("kernel")) kernel = op->get_attr<decltype(kernel)>("kernel");
  if (op->has_attr("shape")) shape = op->get_attr<decltype(shape)>("shape");
  if (op->has_attr("pad")) pad = op->get_attr<decltype(pad)>("pad");
  if (op->has_attr("nonlinear"))
    nonlinear = op->get_attr<decltype(nonlinear)>("nonlinear");
  if (op->has_attr("type")) type = op->get_attr<decltype(type)>("type");

  if ((op_type == "gstiling") || (op_type == "tile-fix")) {
    if (op->has_attr("stride")) {
      stride_scalar = op->get_attr<decltype(stride_scalar)>("stride");
    }
  } else {
    if (op->has_attr("stride")) {
      stride = op->get_attr<decltype(stride)>("stride");
    }
  }
};

std::string op_info::to_json() {
  std::string ret_string = "";

  vector<string> rec;
  rec.push_back(_j(op_name));
  rec.push_back(_j(op_type));
  // rec.push_back(_j(workload));
  // rec.push_back(_j(workload_on_arch));
  if (!kernel.empty()) rec.push_back(_j(kernel));
  if (!shape.empty()) rec.push_back(_j(shape));
  if (!type.empty()) rec.push_back(_j(type));

  ret_string += "{";
  for (size_t i = 0; i < rec.size(); i++) {
    ret_string += rec[i];
    if (i != rec.size() - 1) ret_string += ",";
  }

  ret_string += "}";

  return ret_string;
}

std::string op_info::to_str() {
  stringstream ret;
  ret << "(";

  ret << "op_name:";
  ret << op_name;
  ret << ", ";

  ret << "op_type:";
  ret << op_type;
  ret << ", ";

  if (!kernel.empty()) {
    ret << "kernel:";
    ret << kernel;
    ret << ", ";
  }
  if (!shape.empty()) {
    ret << "shape:";
    ret << shape;
    ret << ", ";
  }
  if (!type.empty()) {
    ret << "type:";
    ret << type;
    ret << ", ";
  }
  if (!stride.empty()) {
    ret << "stride:";
    ret << stride;
    ret << ", ";
  }
  if (stride_scalar != 0) {
    ret << "stride:";
    ret << stride_scalar;
    ret << ", ";
  }

  return ret.str();
}

class subgraph_info {
 public:
  subgraph_info() = delete;
  subgraph_info(const xir::Subgraph* subg, uint32_t& global_uid);
  void to_json(ofstream& o);

 private:
  string subgraph_name;
  string device;
  string dpu_name;
  string pdi_name;
  string type;
  uint32_t depth;
  uint32_t uid;
  uint64_t workload;
  uint64_t workload_on_arch;
  uint32_t op_num;
  string op_list;
  vector<vector<int>> i_tensors_shape;
  vector<vector<int>> o_tensors_shape;
  vector<vector<int>> i_tensors_stride;
  vector<vector<int>> o_tensors_stride;
  ostringstream mc_code_sstr;
  ostringstream mc_code_IPUMapRegisterPass_sstr;
  ostringstream op_info_sstr;
  ostringstream profile_info_sstr;
};

static std::vector<uint32_t> pdi_id_list;
subgraph_info::subgraph_info(const xir::Subgraph* subg, uint32_t& global_uid)
    : subgraph_name(""),
      device(""),
      dpu_name(""),
      pdi_name(""),
      type(""),
      depth(0),
      uid(0),
      workload(0),
      workload_on_arch(0),
      op_num(0),
      op_list(""),
      mc_code_sstr(""),
      op_info_sstr(""),
      profile_info_sstr("") {
  vector<char> mc_code;
  vector<char> mc_code_IPUMapRegisterPass;

  subgraph_name = subg->get_name();
  depth = subg->get_depth();
  op_num = subg->get_op_num();

  auto ops = subg->topological_sort();
  for (auto* op : ops) {
    string op_desc;
    op_desc = op->get_name() + "@" + op->get_type() + "|";
    op_list += op_desc;
  }

  if (subg->has_attr("device"))
    device = subg->get_attr<decltype(device)>("device");
  if (subg->has_attr("dpu_name"))
    dpu_name = subg->get_attr<decltype(dpu_name)>("dpu_name");
  if (subg->has_attr("workload"))
    workload = subg->get_attr<decltype(workload)>("workload");
  if (subg->has_attr("type")) type = subg->get_attr<decltype(type)>("type");
  if (subg->has_attr("workload_on_arch"))
    workload_on_arch = subg->get_attr<decltype(workload)>("workload_on_arch");
  if (subg->has_attr("mc_code")) {
    mc_code = subg->get_attr<decltype(mc_code)>("mc_code");
  }
  if (subg->has_attr("mc_code_IPUMapRegisterPass")) {
    mc_code_IPUMapRegisterPass =
        subg->get_attr<decltype(mc_code_IPUMapRegisterPass)>(
            "mc_code_IPUMapRegisterPass");
  } else if (subg->has_attr("mc_code")) {
    mc_code_IPUMapRegisterPass =
        subg->get_attr<decltype(mc_code_IPUMapRegisterPass)>("mc_code");
  }

  for (unsigned char m : mc_code) {
    static char buf[4] = {0};
    snprintf(buf, sizeof(buf), "%02x", m);
    mc_code_sstr << buf;
  }

  for (unsigned char m : mc_code_IPUMapRegisterPass) {
    static char buf[4] = {0};
    snprintf(buf, sizeof(buf), "%02x", m);
    mc_code_IPUMapRegisterPass_sstr << buf;
  }

  if (depth == 2 && subg->has_attr("name")) {
    pdi_name = subg->get_attr<decltype(pdi_name)>("name");
  }
  auto i_tensors = subg->get_sorted_input_tensors();
  for (auto tensor : i_tensors) {
    i_tensors_shape.push_back(tensor->get_shape());
    std::vector<int> stride = {};
    if (tensor->has_attr("stride"))
      stride = tensor->get_attr<decltype(stride)>("stride");
    i_tensors_stride.push_back(stride);
  };
  auto o_tensors = subg->get_output_tensors();
  for (auto tensor : o_tensors) {
    o_tensors_shape.push_back(tensor->get_shape());
    std::vector<int> stride = {};
    if (tensor->has_attr("stride"))
      stride = tensor->get_attr<decltype(stride)>("stride");
    o_tensors_stride.push_back(stride);
  };

  auto simple_hash = [](const std::string& str) {
    uint32_t hash_value = 0;
    for (char ch : str) {
      hash_value = hash_value * 11 + ch;
    }
    return hash_value;
  };

  uint32_t layer_id = 0;

  // Workaround for xv3
  static string dpu_name_static("");
  if (depth == 1 && subg->has_attr("dpu_name")) {
    dpu_name_static = subg->get_attr<decltype(device)>("dpu_name");
  }
  string xv3_name("DPUCV3DX8G");
  if (dpu_name_static.compare(0, xv3_name.size(), xv3_name) == 0) {
    bool skip_code_gen = false;

    if (subg->has_attr("skip_code_gen"))
      skip_code_gen = subg->get_attr<decltype(skip_code_gen)>("skip_code_gen");

    if ((skip_code_gen == false) && (depth == 2)) {
      layer_id = global_uid & MAX_LAYER_ID;
      global_uid = (global_uid & 0xfc000) | (layer_id + 1);

      uid = (((global_uid >> 14) & 0x3f) << 14) | layer_id |
            GRAPH_TYPE_SUPERLAYER;
    } else {
      uid = MAX_LAYER_ID;
    }
  } else {
    if (type == "PDI") {
      uint32_t pdi_uid = 0u;
      for (auto& o_tensor : subg->get_output_tensors()) {
        auto o_tensor_name = ((o_tensor))->get_name();
        pdi_uid = pdi_uid ^ simple_hash(o_tensor_name);
      }

      layer_id = 0;

      // uint16_t pdi_uid_short = (pdi_uid & PDI_ID_MASK);
      for (auto _id : pdi_id_list) {
        if (pdi_uid == _id) {
          std::cout << "!!!!!ERROR: PDI ID CONFLICT" << std::endl;
          break;
        }
      }
      pdi_id_list.push_back(pdi_uid);

      /* Clear Layer_ID */
      global_uid = global_uid & ~MAX_LAYER_ID;

      /* Update PDI_UID, use long pdi_uid for super-layers */
      global_uid &= ~(LONG_PDI_ID_MASK << PDI_ID_BIT_OFFSET);
      global_uid |= (pdi_uid << PDI_ID_BIT_OFFSET);

      /* Use long pdi_uid for PDI layers*/
      uid =
          ((pdi_uid & LONG_PDI_ID_MASK) << PDI_ID_BIT_OFFSET) | GRAPH_TYPE_PDI;
    } else {
      /* Update Layer_ID */
      layer_id = global_uid & MAX_LAYER_ID;
      if (mc_code_IPUMapRegisterPass.size() > 4) {
        if (mc_code_IPUMapRegisterPass[3] != 10)
          global_uid = (global_uid & (LONG_PDI_ID_MASK << PDI_ID_BIT_OFFSET)) |
                       (layer_id + 1);
        else
          layer_id = MAX_LAYER_ID;
      } else {
        layer_id = MAX_LAYER_ID;
      }
      uid = (((global_uid >> PDI_ID_BIT_OFFSET) & LONG_PDI_ID_MASK)
             << PDI_ID_BIT_OFFSET) |
            layer_id | GRAPH_TYPE_SUPERLAYER;
    }
  }

  /*
  if (layer_id != 0x3ff) {
    if (uid == 0) {
      std::cout << "-----------------------------------\n";
    }
    std::cout << "@@@@ " << std::hex
              << "PDI_UID:" << ((global_uid >> PDI_ID_BIT_OFFSET) &
  LONG_PDI_ID_MASK)
              << ", Layer_ID:" << layer_id << ", UID: " << uid << std::dec
              << ", t: " << type << std::endl;
  }
  */

  /* Additional profiling information for IPU */
  if (subg->has_attr("profile_info_map")) {
    std::map<std::string, std::string> profile_info;
    profile_info = subg->get_attr<decltype(profile_info)>("profile_info_map");
    for (auto item : profile_info) {
      auto k = item.first.c_str();
      auto v = item.second.c_str();

      profile_info_sstr << k << "@" << v << "|";
    }
  }

  string op_str("");

  for (auto* op : subg->topological_sort()) {
    if (op->get_type().find("const-fix") == 0) continue;
    if (op->get_type().find("upload") == 0) continue;
    if (op->get_type().find("download") == 0) continue;

    op_str += op_info(op).to_str();

    stringstream op_i_tensor_str;
    stringstream op_o_tensor_str;
    stringstream op_i_tensor_dtype_str;
    stringstream op_o_tensor_dtype_str;

    op_i_tensor_str << "{";

    {
      auto op_tensor = op->get_input_tensors("input");
      if (op_tensor.size() > 0) {
        op_i_tensor_str << "input:";
        op_i_tensor_str << op_tensor.at(0)->get_shape();
      }
    }

    {
      auto op_tensor = op->get_input_tensors("weights");
      if (op_tensor.size() > 0) {
        op_i_tensor_str << ",weights:";
        op_i_tensor_str << op_tensor.at(0)->get_shape();
      }
    }

    {
      auto op_tensor = op->get_input_tensors("bias");
      if (op_tensor.size() > 0) {
        op_i_tensor_str << ",bias:";
        op_i_tensor_str << op_tensor.at(0)->get_shape();
      }
    }
    op_i_tensor_str << "}";

    op_str += "op_i_tensor_shape: ";
    op_str += op_i_tensor_str.str();

    /* gets and adds the dtype field for input tensors */
    op_i_tensor_dtype_str << "{";

    {
      auto op_tensor = op->get_input_tensors("input");
      if (op_tensor.size() > 0) {
        op_i_tensor_dtype_str << "input:";
        op_i_tensor_dtype_str << op_tensor.at(0)->get_data_type().to_string();
      }
    }

    {
      auto op_tensor = op->get_input_tensors("weights");
      if (op_tensor.size() > 0) {
        op_i_tensor_dtype_str << ",weights:";
        op_i_tensor_dtype_str << op_tensor.at(0)->get_data_type().to_string();
      }
    }

    {
      auto op_tensor = op->get_input_tensors("bias");
      if (op_tensor.size() > 0) {
        op_i_tensor_dtype_str << ",bias:";
        op_i_tensor_dtype_str << op_tensor.at(0)->get_data_type().to_string();
      }
    }
    op_i_tensor_dtype_str << "}";

    op_str += ", op_i_tensor_dtype: ";
    op_str += op_i_tensor_dtype_str.str();

    op_str += ", op_o_tensor_shape: ";
    auto op_o_tensors = op->get_output_tensor();
    op_o_tensor_str << op_o_tensors->get_shape();
    op_str += op_o_tensor_str.str();

    /* gets and adds the dtype field for output tensors */
    op_str += ", op_o_tensor_dtype: ";
    op_o_tensor_dtype_str << op_o_tensors->get_data_type().to_string();
    op_str += op_o_tensor_dtype_str.str();

    op_str += ")#";
  }

  op_info_sstr << op_str;
};

void subgraph_info::to_json(ofstream& o) {
  vector<string> rec;
  rec.push_back(_j(subgraph_name));
  rec.push_back(_j(dpu_name));
  rec.push_back(_j(pdi_name));
  rec.push_back(_j(device));
  rec.push_back(_j(type));
  rec.push_back(_j(depth));
  rec.push_back(_j(workload));
  rec.push_back(_j(workload_on_arch));
  rec.push_back(_j(op_num));
  rec.push_back(_j(op_list));
  rec.push_back(_j(i_tensors_shape));
  rec.push_back(_j(o_tensors_shape));
  rec.push_back(_j(i_tensors_stride));
  rec.push_back(_j(o_tensors_stride));
  rec.push_back(_j(mc_code_sstr.str()));
  rec.push_back(_j(mc_code_IPUMapRegisterPass_sstr.str()));
  rec.push_back(_j(profile_info_sstr.str()));
  rec.push_back(_j(op_info_sstr.str()));
  rec.push_back(_j(uid));

  /* Debug Usage */
  vector<string> dbg_msg;
  dbg_msg.push_back(_j(subgraph_name));
  dbg_msg.push_back(_j(uid));
  // std::cout << "{";
  // for (size_t i = 0; i < dbg_msg.size(); i++) {
  //     std::cout << dbg_msg[i];
  //   if (i != dbg_msg.size() - 1) std::cout << ",";
  // }
  // std::cout << "}\n";
  /**************************************************/

  o << "{";
  for (size_t i = 0; i < rec.size(); i++) {
    o << rec[i];
    if (i != rec.size() - 1) o << ",";
  }
  o << "}\n";
}

namespace vitis::ai::trace {
void add_subgraph_recursive(const xir::Subgraph* subg, ofstream& of,
                            uint32_t& global_uid) {
  subgraph_info(subg, global_uid).to_json(of);
  of.flush();

  auto depth = subg->get_depth();
  auto children_topo_size = 0;
  if (subg->has_attr("children_topological_sort")) {
    children_topo_size =
        subg->get_attr<std::vector<std::string>>("children_topological_sort")
            .size();
  }
  if (children_topo_size == 0 || depth == 1) {
    for (auto child : subg->children_topological_sort()) {
      add_subgraph_recursive(child, of, global_uid);
    }
  } else {
    auto children = subg->get_children();
    auto child_order =
        subg->get_attr<std::vector<std::string>>("children_topological_sort");
    for (const auto& child_name : child_order) {
      auto child = std::find_if(
          children.begin(), children.end(),
          [&child_name](auto g) { return g->get_name() == child_name; });

      if (child == children.end()) {
        throw std::runtime_error("Error: missing child layer info in xmodel");
      } else {
        add_subgraph_recursive((*child), of, global_uid);
      }
    }
  }
}

string add_subgraph_raw(const xir::Subgraph* subg) {
  static uint32_t subgraph_id = 0;
  uint32_t global_uid = 0;
  auto dir = vitis::ai::my_getenv_s("VAI_TRACE_DIR", "/tmp/");
  auto pid =
#if _WIN32
      GetCurrentProcessId();
#else
      getpid();
#endif
  string path = dir + "vaitrace_subgraph_info_" + to_string(pid) + "_" +
                to_string(subgraph_id++);

  ofstream output_f;
  output_f.open(path, ios::out | ios::trunc);

  add_subgraph_recursive(subg, output_f, global_uid);

  output_f.close();

  return path;
};

}  // namespace vitis::ai::trace
