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

#include "vitis/ai/profiling.hpp"

#ifdef _WIN32
#  include <Windows.h>
#else
#  include <sys/syscall.h>
#endif

#include <glog/logging.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

DEF_ENV_PARAM(VAIPROFILING_DETAILS, "0");

namespace vitis {
namespace ai {

namespace profiling {

inline unsigned int get_tid() {
#ifdef _WIN32
  return static_cast<unsigned int>(GetCurrentThreadId());
#else
  return static_cast<unsigned int>(syscall(SYS_gettid));
#endif
}

inline unsigned int get_pid() {
#ifdef _WIN32
  return static_cast<unsigned int>(GetCurrentProcessId());
#else
  return static_cast<unsigned int>(syscall(SYS_getpid));
#endif
}

struct DataHolder {
  struct Entry {
    Level level;
    const char* tag;
    const long long duration;
    long long timestamp;
    const char* timescale;
    unsigned int pid;
    unsigned int tid;
  };

  std::mutex dataMutex;
  std::vector<Entry> data;

  DataHolder() {
    if (ENV_PARAM(DEEPHI_PROFILING)) {
      data.reserve(2048);
    }
  }

  void dump() {
    if (!ENV_PARAM(DEEPHI_PROFILING)) {
      return;
    }

    decltype(data) localData;
    {
      std::lock_guard<std::mutex> lock(dataMutex);
      data.swap(localData);
    }

    if (localData.empty()) {
      return;
    }

#define LOG_LINE                                                               \
      entry.tag << " : " << entry.duration << " " << entry.timescale

#define LOG_LINE_DUMP                                                               \
  "[" << entry.pid << ":" << entry.tid << "]"                                  \
      << " " << entry.timestamp << " " << entry.tag << " : " << entry.duration \
      << " " << entry.timescale

    auto vaiprofiling_dump_path = my_getenv_s("VAIPROFILING_DUMP_PATH", "");
    if (!vaiprofiling_dump_path.empty()) {
      std::ofstream dump_file;
      dump_file.open(vaiprofiling_dump_path, std::ios::out);

      if (dump_file.is_open()) {
        for (const auto& entry : localData) dump_file << LOG_LINE_DUMP << std::endl;
        dump_file.close();
      }
    } else {
      for (const auto& entry : localData) {
        switch (entry.level) {
          case Level::L_WARNING:
            LOG(WARNING) << LOG_LINE;
            break;
          case Level::L_ERROR:
            LOG(ERROR) << LOG_LINE;
            break;
          case Level::L_FATAL:
            LOG(FATAL) << LOG_LINE;
            break;
          case Level::L_INFO:
          default:
            LOG(INFO) << LOG_LINE;
            break;
        }
      }
    }
#undef LOG_LINE
  }

  ~DataHolder() {
    dump();
  }
};

static DataHolder* dataholder_instance() {
  static DataHolder dh;
  return &dh;
}

void add_duration(
    Level level, const char* tag,
    const long long duration,
    const std::chrono::time_point<std::chrono::steady_clock>& timestamp,
    const char* timescale) {
  DataHolder* dh = dataholder_instance();

  unsigned int pid;
  unsigned int tid;
  long long ts;

  static auto enable_detailed = ENV_PARAM(VAIPROFILING_DETAILS);

  if (enable_detailed) {
    pid = get_pid();
    tid = get_tid();
    ts = std::chrono::duration_cast<std::chrono::microseconds>(
             timestamp.time_since_epoch())
             .count();
  } else {
    pid = 0;
    tid = 0;
    ts = 0;
  }

  std::lock_guard<std::mutex> lock(dh->dataMutex);
  dh->data.push_back(
      DataHolder::Entry{level, tag, duration, ts, timescale, pid, tid});
}

void dump() {
  dataholder_instance()->dump();
}

}  // namespace profiling

}  // namespace ai
}  // namespace vitis
