#ifndef ONNX_TO_UNT_H
#define ONNX_TO_UNT_H

#include "onnxruntime_cxx_api.h" // NOLINT
#include "runtime/unruntime.h"

using namespace un_tensor;
using namespace unrun;

static const std::unordered_map<int, int> dtype_size_map = {
    {0, 4}, // fp32
    {1, 4}, // i32
    {4, 4}, // u32
    {6, 2}, // bf16
    {2, 2}, // i16
    {3, 2} // u16
};

static const std::unordered_map<ONNXTensorElementDataType, int> ortvalue_size_map = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 4},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, 4},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32, 4},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16, 2},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, 2},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16, 2},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, 1},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8, 1},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, 8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, 8},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, 8},
};

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
      minilog::T_error << "unknown fmt:" << untensor_dtype << minilog::errorendl;
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

Ort::Value GetOrtValueFromUnTensor(untensor_s& un_tensor);

void ConvertOrtValueToUnTensor(Ort::Value& ort_value, untensor_s& un_tensor);

void LoadOrtValuesToUnTensors(std::vector<Ort::Value> &ort_value_list, std::vector<untensor_s> &untensor_list, const int num);

std::vector<Ort::Value> GetOrtValuesFromUnTensors(std::vector<untensor_s>::iterator untensor_start_p, const int num);

#endif
