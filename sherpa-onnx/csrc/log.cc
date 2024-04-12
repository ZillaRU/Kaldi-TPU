// sherpa-onnx/csrc/log.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/log.h"

#ifdef SHERPA_ONNX_HAVE_EXECINFO_H
#include <execinfo.h>  // To get stack trace in error messages.
#ifdef SHERPA_ONNX_HAVE_CXXABI_H
#include <cxxabi.h>  // For name demangling.
// Useful to decode the stack trace, but only used if we have execinfo.h
#endif  // SHERPA_ONNX_HAVE_CXXABI_H
#endif  // SHERPA_ONNX_HAVE_EXECINFO_H

#include <stdlib.h>

#include <ctime>
#include <iomanip>
#include <string>

namespace sherpa_onnx {

std::string GetDateTimeStr() {
  std::ostringstream os;
  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  os << std::put_time(&tm, "%F %T");  // yyyy-mm-dd hh:mm:ss
  return os.str();
}

static bool LocateSymbolRange(const std::string &trace_name, std::size_t *begin,
                              std::size_t *end) {
  // Find the first '_' with leading ' ' or '('.
  *begin = std::string::npos;
  for (std::size_t i = 1; i < trace_name.size(); ++i) {
    if (trace_name[i] != '_') {
      continue;
    }
    if (trace_name[i - 1] == ' ' || trace_name[i - 1] == '(') {
      *begin = i;
      break;
    }
  }
  if (*begin == std::string::npos) {
    return false;
  }
  *end = trace_name.find_first_of(" +", *begin);
  return *end != std::string::npos;
}

std::string GetStackTrace() {
  std::string ans;
  return ans;
}

}  // namespace sherpa_onnx
