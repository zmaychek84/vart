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
#include <memory>
#include <string>
#include <unordered_map>
#include <iostream>
#define VAIP_DLL_SPEC
namespace vitis {
namespace ai {
struct Tag_Plugin_Func_Set;
typedef Tag_Plugin_Func_Set Plugin_Func_Set;
extern Plugin_Func_Set* g_static_plugin_func_set_ptr;
extern Plugin_Func_Set* g_dynamic_plugin_func_set_ptr;

struct UndefinedSymbolException : public std::exception {
public:
  virtual const char* what() const throw() { return m.c_str(); }
  std::string m;
};

struct Plugin {
  VAIP_DLL_SPEC
  Plugin(const char* name,
         Plugin_Func_Set* func_set = g_static_plugin_func_set_ptr);
  VAIP_DLL_SPEC ~Plugin();
  template <typename R, typename... Args>
  R invoke(const char* name, Args... args) {
    typedef R (*fun_type_t)(Args...);
    auto sym = my_plugin_sym(plugin_, name);
    if(sym == nullptr){
      UndefinedSymbolException e;
      e.m = std::string("no such function: ")+name+"; libname " + name_ + " so_name " + so_name_ +" ";
      throw e;
    }
    fun_type_t f = reinterpret_cast<fun_type_t>(sym);
    return f(std::forward<Args>(args)...);
  }
  static Plugin* get(const std::string& name,
                     Plugin_Func_Set* func_set = g_static_plugin_func_set_ptr);

private:
  std::string name_;
  std::string so_name_;
  Plugin_Func_Set* func_set_;
  void* plugin_;

private:
  static std::string guess_name(const char* name);
  static std::unordered_map<std::string, std::shared_ptr<Plugin>> store_;
  void* my_plugin_sym(void*, const char*);
};

template <typename T, typename... Args> class WithPlugin {
public:
  static std::unique_ptr<T> create(const std::string& plugin_name,
                                   Plugin_Func_Set* func_set, Args... args) {
    auto plugin = Plugin::get(plugin_name, func_set);
    auto ret = plugin->invoke<T*>(T::entry_point, args...);
    return std::unique_ptr<T>(ret);
  }
};
} // namespace ai
} // namespace vitis
