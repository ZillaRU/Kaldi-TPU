#ifndef ONNX_TO_UNT_H
#define ONNX_TO_UNT_H

// include untool
#include "runtime/unruntime.h"
#include "onnxruntime_cxx_api.h" // NOLINT

using namespace un_tensor;

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


Ort::Value GetOrtValueFromUnTensor(untensor_s& un_tensor);

void ConvertOrtValueToUnTensor(Ort::Value& ort_value, untensor_s& un_tensor);

void LoadOrtValuesToUnTensors(std::vector<Ort::Value> &ort_value_list, std::vector<untensor_s> &untensor_list, const int num);

std::vector<Ort::Value> GetOrtValuesFromUnTensors(std::vector<untensor_s>::iterator untensor_start_p, const int num);

#endif
