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

Ort::Value GetOrtValueFromCviTensor(const CVI_TENSOR& cvi_tensor, Ort::Allocator* allocator) {
    // 根据CVI_TENSOR的数据类型确定Ort的数据类型
  ONNXTensorElementDataType type;
  switch (cvi_tensor.fmt) {
      case CVI_FMT_FP32: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; break;
      case CVI_FMT_INT32: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; break;
      case CVI_FMT_UINT32: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32; break;
      case CVI_FMT_BF16: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16; break;
      case CVI_FMT_INT16: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16; break;
      case CVI_FMT_UINT16: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16; break;
      case CVI_FMT_INT8: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8; break;
      case CVI_FMT_UINT8: type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8; break;
      default: throw std::runtime_error("Unsupported data type for conversion.");
  }
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  //   const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const int64_t* shape, size_t shape_len, ONNXTensorElementDataType type
  int64_t* dim64 = new int64_t[CVI_DIM_MAX];
  for (int i = 0; i < cvi_tensor.shape.dim_size; ++i) {
      dim64[i] = static_cast<int64_t>(cvi_tensor.shape.dim[i]);
  }
  // todo: check data type cast
  auto ort_value = Ort::Value::CreateTensor(
      memory_info, (void*)(cvi_tensor.sys_mem), cvi_tensor.count, dim64, cvi_tensor.shape.dim_size, type);
  return ort_value;
}

void ConvertOrtValueToCviTensor(const Ort::Value& ort_value, CVI_TENSOR& cvi_tensor) {
    // 设置数据类型
    cvi_tensor.fmt = static_cast<CVI_FMT>(ort_value.GetTensorTypeAndShapeInfo().GetElementType());

    // 获取形状信息并设置
    const std::vector<int64_t>& shape = ort_value.GetTensorTypeAndShapeInfo().GetShape();
    cvi_tensor.shape.dim_size = shape.size();
    for (size_t i = 0; i < cvi_tensor.shape.dim_size; ++i) {
        cvi_tensor.shape.dim[i] = static_cast<int32_t>(shape[i]);
    }

    // 获取数据并设置
    cvi_tensor.count = 1;
    for (size_t i = 0; i < cvi_tensor.shape.dim_size; ++i) {
        cvi_tensor.count *= cvi_tensor.shape.dim[i];
    }
    cvi_tensor.mem_size = cvi_tensor.count * fmt_size_map.at(cvi_tensor.fmt);

    cvi_tensor.sys_mem = static_cast<uint8_t*>(malloc(cvi_tensor.mem_size));
    
    // 从OrtValue中获取数据并拷贝( todo ：数据类型转换)
    memcpy(cvi_tensor.sys_mem, ort_value.GetTensorRawData(), cvi_tensor.mem_size);

    // 设置其他必要的CVI_TENSOR字段
    // ...
}

void LoadOrtValuesToCviTensors(const std::vector<Ort::Value> &ort_value_list, CVI_TENSOR* target_cvi_tensors, const int &num) {
    assert(num == (int)(ort_value_list.size()));
    int idx = 0;
    for(auto &ort_value : ort_value_list) {
        auto &cvi_tensor = target_cvi_tensors[idx];
        ConvertOrtValueToCviTensor(ort_value, cvi_tensor);
        idx += 1;
    }
}

std::vector<Ort::Value> GetOrtValuesFromCviTensors(const CVI_TENSOR* cvi_tensors, const int num, Ort::Allocator *allocator) {
    std::vector<Ort::Value> target_ort_value_list;
    target_ort_value_list.reserve(num);
    for(int i = 0; i < num; i++) {
        auto &cvi_tensor = cvi_tensors[i];
        GetOrtValueFromCviTensor(cvi_tensor, allocator);
    }
}
