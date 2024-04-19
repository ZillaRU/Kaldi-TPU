#ifndef SHERPA_ONNX_CSRC_UNT_UTILS_H_
#define SHERPA_ONNX_CSRC_UNT_UTILS_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ostream>
#include <utility>
#include <vector>

#include "runtime/unruntime.h"

using namespace unrun;

// type_map = {
//     0: np.float32,
//     1: np.float16,
//     4: np.int16,
//     6: np.int32,
//     2: np.int8,
//     3: np.uint8,
// }

static const char *formatToStr(int untensor_dtype) {
  switch (untensor_dtype) {
    case 0:
      return "fp32";
    case 1:
      return "i32";
    case 4:
      return "u32";
    case 6:
      return "bf16";
    case 2:
      return "i16";
    case 3:
      return "u16";
    default:;
      // minilog::error << "unknown fmt:" << untensor_dtype << minilog::errorendl;
  }
  return nullptr;
}

static void copyFp32ToFp32(float *src, float *dst, int count) {
  memcpy(dst, src, count * sizeof(float));
}

static void copyFp64ToFp32(double *src, float *dst, int count) {
  std::transform(src, src + count, dst,
                 [](double x) { return static_cast<float>(x); });
}

static void copyI64ToFp32(int64_t *src, float *dst, int count) {
  // use bitcasting to convert int64 to float, this is much faster than shifting
  memcpy(dst, reinterpret_cast<float *>(src), count * sizeof(float));
}

static void copyI64ToI32(int64_t *src, int32_t *dst, int count) {
  std::transform(src, src + count, dst,
                 [](int64_t x) { return static_cast<int32_t>(x); });
}

static void copyI64ToU16(int64_t *src, uint16_t *dst, int count) {
  std::transform(src, src + count, dst,
                 [](int64_t x) { return static_cast<uint16_t>(x); });
}

static void copyI32ToFp32(int *src, float *dst, int count) {
  std::transform(src, src + count, dst,
                 [](int x) { return static_cast<float>(x); });
}

static void copyI32ToU16(int *src, uint16_t *dst, int count) {
  memcpy(dst, reinterpret_cast<uint16_t *>(src), count * sizeof(uint16_t));
}

static void copyI32ToU8(int *src, uint8_t *dst, int count) {
  memcpy(dst, reinterpret_cast<uint8_t *>(src), count * sizeof(uint8_t));
}

static void copyI32ToI8(int *src, int8_t *dst, int count) {
  memcpy(dst, reinterpret_cast<int8_t *>(src), count * sizeof(int8_t));
}

void print_output_value(un_runtime_s *runtime, int start, int length);

void malloc_generate_host_data(un_runtime_s *runtime);

#endif