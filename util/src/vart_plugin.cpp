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
// include glog/logging.h to define CHECK before include vaip_plugin.hpp
#include "vitis/ai/plugin.hpp"
#include "vitis/ai/vart_plugin.hpp"
#include "vitis/ai/weak.hpp"

namespace vitis {
namespace ai {
std::string Plugin::guess_name(const char* name) {
#ifdef _WIN32
  return std::string("") + name + ".dll";
#else
  return std::string("lib") + name + ".so";
#endif
}
struct Tag_Plugin_Func_Set {
  vitis::ai::plugin_t (*open_plugin)(const std::string& name,
                                     vitis::ai::scope_t scope);
  void* (*plugin_sym)(vitis::ai::plugin_t plugin, const std::string& name);
  void (*close_plugin)(vitis::ai::plugin_t plugin);
};
Plugin_Func_Set g_static_plugin_func_set = {
    vitis::ai::open_plugin, vitis::ai::plugin_sym, vitis::ai::close_plugin};

Plugin_Func_Set g_dynamic_plugin_func_set = {
    vitis::ai::open_plugin, vitis::ai::plugin_sym, vitis::ai::close_plugin};

Plugin_Func_Set* g_static_plugin_func_set_ptr = &g_static_plugin_func_set;
Plugin_Func_Set* g_dynamic_plugin_func_set_ptr = &g_dynamic_plugin_func_set;

Plugin::Plugin(const char* name, Plugin_Func_Set* func_set)
    : name_{name}, so_name_{guess_name(name)}, func_set_{func_set},
      plugin_{func_set->open_plugin(so_name_, vitis::ai::scope_t::PUBLIC)} {
      if(plugin_ == nullptr){
        UndefinedSymbolException e;
        e.m = std::string("cannot open plugin name_ ") + name_ + "; so_name_" + so_name_ + " ";
        throw e;
      }
}

Plugin::~Plugin() { func_set_->close_plugin((vitis::ai::plugin_t)plugin_); }

std::unordered_map<std::string, std::shared_ptr<Plugin>> Plugin::store_;

Plugin* Plugin::get(const std::string& plugin_name, Plugin_Func_Set* func_set) {
  auto it = store_.find(plugin_name);
  if (it == store_.end()) {
    store_[plugin_name] = vitis::ai::WeakStore<std::string, Plugin>::create(
        plugin_name, plugin_name.c_str(), func_set);
  }
  it = store_.find(plugin_name);
  if(it == store_.end()){
    std::cout << "cannot load plugin. plugin_name=" << plugin_name<<std::endl;
    return nullptr;
  }
  return it->second.get();
}
void* Plugin::my_plugin_sym(void* handle, const char* name) {
  return func_set_->plugin_sym((vitis::ai::plugin_t)handle, name);
}
} // namespace ai
} // namespace vitsi
