#ifndef SHERPA_CVIRTL_UTILS_H_
#define SHERPA_CVIRTL_UTILS_H_

#include <cassert>
#include <ostream>
#include <cstring>
#include <utility>
#include <vector>
#include <algorithm>
#include "cviruntime.h"  // NOLINT


static const char* formatToStr(CVI_FMT fmt) {
  switch(fmt) {
    case CVI_FMT_FP32:  return "fp32";
    case CVI_FMT_INT32:  return "i32";
    case CVI_FMT_UINT32: return "u32";
    case CVI_FMT_BF16:   return "bf16";
    case CVI_FMT_INT16:  return "i16";
    case CVI_FMT_UINT16: return "u16";
    case CVI_FMT_INT8:   return "i8";
    case CVI_FMT_UINT8:  return "u8";
    default:
      TPU_LOG_FATAL("unknown fmt:%d\n", fmt);
  }
  return nullptr;
}

static void copyFp32ToFp32(float *src, float *dst, int count) {
  memcpy(dst, src, count * sizeof(float));
}

static void copyFp64ToFp32(double *src, float *dst, int count) {
  std::transform(src, src + count, dst, [](double x) { return static_cast<float>(x); });
}

static void copyI64ToFp32(int64_t *src, float *dst, int count) {
  // use bitcasting to convert int64 to float, this is much faster than shifting
  memcpy(dst, reinterpret_cast<float*>(src), count * sizeof(float));
}

static void copyI64ToU16(int64_t *src, uint16_t *dst, int count) {
  std::transform(src, src + count, dst, [](int64_t x) { return static_cast<uint16_t>(x); });
}

static void copyI32ToFp32(int *src, float *dst, int count) {
  std::transform(src, src + count, dst, [](int x) { return static_cast<float>(x); });
}

static void copyI32ToU16(int *src, uint16_t *dst, int count) {
  memcpy(dst, reinterpret_cast<uint16_t*>(src), count * sizeof(uint16_t));
}

static void copyI32ToU8(int *src, uint8_t *dst, int count) {
  memcpy(dst, reinterpret_cast<uint8_t*>(src), count * sizeof(uint8_t));
}

static void copyI32ToI8(int *src, int8_t *dst, int count) {
  memcpy(dst, reinterpret_cast<int8_t*>(src), count * sizeof(int8_t));
}



/**
 * Get the input and output names and shapes of a model.
 * @param model A cviruntime model handle.
*/
void GetInputOutPutInfo(CVI_MODEL_HANDLE model);

#endif  // SHERPA_CVIRTL_UTILS_H_