#include "sherpa-onnx/csrc/onnx-to-bm.h"
#include <iostream>
#include <cassert>
using namespace bmruntime;

Ort::Value GetOrtValueFromBMTensor(Tensor* const bmr_tensor) {
  // 根据CVI_TENSOR的数据类型确定Ort的数据类型
  ONNXTensorElementDataType type;
  switch (bmr_tensor->tensor()->dtype) {
    case 0:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    default:
      throw std::runtime_error("Unsupported data type for conversion.");
  }
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  bm_shape_t tensor_shape = bmr_tensor->tensor()->shape;
  int64_t *dim64 = new int64_t[tensor_shape.num_dims];
  for (int i = 0; i <  tensor_shape.num_dims; ++i) {
    dim64[i] = static_cast<int64_t>(tensor_shape.dims[i]);
  }
  void* temp_data_p = calloc(bmr_tensor->num_elements(), sizeof(float));
  bm_status_t status = bmr_tensor->CopyTo(temp_data_p);
  assert(BM_SUCCESS == status);
  auto ort_value = Ort::Value::CreateTensor<float>(memory_info, (float*)temp_data_p, static_cast<size_t>(bmr_tensor->num_elements()), dim64, tensor_shape.num_dims);
  std::free(temp_data_p);
  return ort_value;
}

std::vector<Ort::Value> GetOrtValuesFromBMTensors(std::vector<Tensor*>::const_iterator tensor_start_p, const int num) {
  std::vector<Ort::Value> target_ort_value_list;
  target_ort_value_list.reserve(num);
  for (int i = 0; i < num; i++) {
    auto &bmr_tensor = *(tensor_start_p++);
    target_ort_value_list.push_back(GetOrtValueFromBMTensor(bmr_tensor));
  }
  return target_ort_value_list;
}

void ConvertOrtValueToBMTensor(Ort::Value &ort_value, Tensor* const bmr_tensor) {
  // assert ort_value中数据的数量和cvi_tensor的数据的数量一致
  size_t element_count = static_cast<size_t>(bmr_tensor->num_elements()); // uint64_t ->size_t
  assert(element_count == ort_value.GetTensorTypeAndShapeInfo().GetElementCount()); // size_t
  auto type = ort_value.GetTensorTypeAndShapeInfo().GetElementType();
  const char* bm_tensor_type = formatToStr(bmr_tensor->tensor()->dtype);

  bm_status_t status;
  void *temp_data_p = nullptr;
  
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && bm_tensor_type == "i32") {
    temp_data_p = calloc(bmr_tensor->num_elements(), ByteSize(bmr_tensor->tensor()->dtype));
    copyI64ToI32((int64_t *)(ort_value.GetTensorMutableRawData()), (int32_t*) temp_data_p, element_count);
    status = bmr_tensor->CopyFrom(temp_data_p);
    std::free(temp_data_p);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && bm_tensor_type == "fp32") {
    temp_data_p = calloc(bmr_tensor->num_elements(), ByteSize(bmr_tensor->tensor()->dtype));
    copyI64ToFp32((int64_t *)(ort_value.GetTensorMutableRawData()), (float *)(temp_data_p), element_count);
    status = bmr_tensor->CopyFrom(temp_data_p);
    std::free(temp_data_p);
  } else if (ByteSize(bmr_tensor->tensor()->dtype) == ortvalue_size_map.at(type)) {
    status = bmr_tensor->CopyFrom(ort_value.GetTensorMutableRawData());
  } else {
    std::cout << "ort_value element size: " << ortvalue_size_map.at(type)
          << "bm_tensor element size: " << ByteSize(bmr_tensor->tensor()->dtype) << std::endl;
    throw std::runtime_error("Unsupported data type for conversion.");
  }
  if(BM_SUCCESS != status) {
    throw std::runtime_error("Fail to copy host memory data to device.");
  }
}

void LoadOrtValuesToBMTensors(std::vector<Ort::Value> &ort_value_list,
                              const std::vector<Tensor*> &tensor_list,
                              const int num) {
  assert(num == (int)(ort_value_list.size()));
  int idx = 0;
  for (auto &ort_value : ort_value_list) {
    auto &_tensor = tensor_list[idx];
    ConvertOrtValueToBMTensor(ort_value, _tensor);
    idx += 1;
  }
}