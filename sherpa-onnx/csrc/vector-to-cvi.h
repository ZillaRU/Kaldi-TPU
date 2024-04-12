#ifndef ONNX_TO_CVI_H
#define ONNX_TO_CVI_H

#include "cviruntime.h"
#include "onnxruntime_cxx_api.h" // NOLINT


static const std::unordered_map<CVI_FMT, int> fmt_size_map = {
    {CVI_FMT_FP32, 4},
    {CVI_FMT_INT32, 4},
    {CVI_FMT_UINT32, 4},
    {CVI_FMT_BF16, 2},
    {CVI_FMT_INT16, 2},
    {CVI_FMT_UINT16, 2},
    {CVI_FMT_INT8, 1},
    {CVI_FMT_UINT8, 1},
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


Ort::Value GetOrtValueFromCviTensor(CVI_TENSOR& cvi_tensor);

void ConvertOrtValueToCviTensor(Ort::Value& ort_value, CVI_TENSOR& cvi_tensor);

void LoadOrtValuesToCviTensors(std::vector<Ort::Value> &ort_value_list, CVI_TENSOR* target_cvi_tensors, const int &num);

std::vector<Ort::Value> GetOrtValuesFromCviTensors(CVI_TENSOR* cvi_tensors, const int num);

#endif
