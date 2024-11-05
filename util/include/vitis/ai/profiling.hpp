/*
 * Copyright 2022-2024 Advanced Micro Devices Inc.
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
#ifndef DEEPHI_PROFILING_HPP_
#define DEEPHI_PROFILING_HPP_

#include <glog/logging.h>
#include <chrono>
#include "./env_config.hpp"
DEF_ENV_PARAM(DEEPHI_PROFILING, "0");

namespace vitis {
namespace ai {

namespace profiling {

enum class Level
{
  L_INFO,
  L_WARNING,
  L_ERROR,
  L_FATAL
};

void add_duration(
  Level level, const char *tag,
  const long long duration,
  const std::chrono::time_point<std::chrono::steady_clock>& timestamp,
  const char *timescale);

void dump();

} // profiling

using Clock = std::chrono::steady_clock;

#define __TIC__(tag)                                                           \
  auto __##tag##_start_time =                                                  \
      ENV_PARAM(DEEPHI_PROFILING)                                              \
          ? vitis::ai::Clock::now()                                            \
          : std::chrono::time_point<vitis::ai::Clock>();

#define __TOC__(tag)                                                           \
  do {                                                                         \
    if (!ENV_PARAM(DEEPHI_PROFILING)) break;                                   \
    auto __##tag##_end_time = vitis::ai::Clock::now();                         \
    vitis::ai::profiling::add_duration(vitis::ai::profiling::Level::L_INFO  ,  \
      #tag,                                                                    \
      std::chrono::duration_cast<std::chrono::microseconds>(                   \
        __##tag##_end_time - __##tag##_start_time)                             \
      .count(),  __##tag##_start_time, "us");                                  \
  } while(0);

#define __TOC_FLEX__(tag, level, timescale)                                    \
  do {                                                                         \
    if (!ENV_PARAM(DEEPHI_PROFILING)) break;                                   \
    auto __##tag##_end_time = vitis::ai::Clock::now();                         \
    vitis::ai::profiling::add_duration(vitis::ai::profiling::Level::L_##level, \
      #tag,                                                                    \
      std::chrono::duration_cast<std::chrono::timescale>(                      \
        __##tag##_end_time - __##tag##_start_time)                             \
      .count(), __##tag##_start_time, timescale);                              \
  } while(0);

#define __TIC_SUM__(tag)                                                       \
  static auto __##tag##_total_time = 0U;                                       \
  auto __##tag##_start_time =                                                  \
      ENV_PARAM(DEEPHI_PROFILING)                                              \
          ? vitis::ai::Clock::now()                                            \
          : std::chrono::time_point<vitis::ai::Clock>();

#define __TOC_SUM__(tag)                                                       \
  do {                                                                         \
    if (!ENV_PARAM(DEEPHI_PROFILING)) break;                                   \
    auto __##tag##_end_time = vitis::ai::Clock::now();                         \
    vitis::ai::profiling::add_duration(vitis::ai::profiling::Level::L_INFO,    \
      #tag, __##tag##_total_time +                       \
      std::chrono::duration_cast<std::chrono::microseconds>(                   \
        __##tag##_end_time - __##tag##_start_time)                             \
      .count(), __##tag##_start_time, timescale);                              \
  } while(0);

}  // namespace ai
}  // namespace vitis

#endif
