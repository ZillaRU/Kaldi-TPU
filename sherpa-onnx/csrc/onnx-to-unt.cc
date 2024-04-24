#include "sherpa-onnx/csrc/onnx-to-unt.h"

#undef LOG


using namespace un_tensor;
using namespace unrun;

void print_output_value(un_runtime_s* runtime, int start, int length){
    int output_num = runtime->output_tensors.size();
    for (int i = 0; i < output_num; i++) {
        print_data_by_fp32(runtime->output_tensors[i].data, runtime->output_tensors[i].size, runtime->output_tensors[i].dtype, start, length);
    }
}

void malloc_generate_host_data(un_runtime_s* runtime){
    int input_num = runtime->input_tensors.size();
    for (int i = 0; i < input_num; i++) {
        malloc_host_data(&runtime->input_tensors[i]);
    }
}

Ort::Value GetOrtValueFromUnTensor(untensor_s &un_tensor) {
  // 根据CVI_TENSOR的数据类型确定Ort的数据类型
  ONNXTensorElementDataType type;
  switch (un_tensor.dtype) {
    case 0:
      type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      break;
    // case "i32":
    // case "u32":
    // case "bf16":
    // case "i16":
    // case "u16":
    default:
      throw std::runtime_error("Unsupported data type for conversion.");
  }
  Ort::MemoryInfo memory_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  //   const OrtMemoryInfo* info, void* p_data, size_t p_data_byte_count, const
  //   int64_t* shape, size_t shape_len, ONNXTensorElementDataType type
  int64_t *dim64 = new int64_t[MAX_DIMS];
  for (int i = 0; i < un_tensor.dims; ++i) {
    dim64[i] = static_cast<int64_t>(un_tensor.shape[i]);
  }
  // todo: check data type cast
  auto ort_value = Ort::Value::CreateTensor(memory_info, un_tensor.data, un_tensor.size, dim64, un_tensor.dims, type);
  return ort_value;
}

std::vector<Ort::Value> GetOrtValuesFromUnTensors(std::vector<untensor_s>::iterator untensor_start_p, const int num) {
  std::vector<Ort::Value> target_ort_value_list;
  target_ort_value_list.reserve(num);
  for (int i = 0; i < num; i++) {
    auto &_untensor = *(untensor_start_p++);
    target_ort_value_list.push_back(GetOrtValueFromUnTensor(_untensor));
  }
  return target_ort_value_list;
}

void ConvertOrtValueToUnTensor(Ort::Value &ort_value, untensor_s &untensor) {
  // assert ort_value中数据的数量和cvi_tensor的数据的数量一致
  int element_count = untensor.size / dtype_size_map.at(untensor.dtype);
  assert(element_count == int(ort_value.GetTensorTypeAndShapeInfo().GetElementCount()));
  auto type = ort_value.GetTensorTypeAndShapeInfo().GetElementType();
  auto untensor_type = formatToStr(untensor.dtype);
  
  if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && untensor_type == "i32") {
    copyI64ToI32((int64_t *)(ort_value.GetTensorMutableRawData()),
                 (int32_t *)(untensor.data), element_count);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && untensor_type == "fp32") {
    copyI64ToFp32((int64_t *)(ort_value.GetTensorMutableRawData()),
                  (float *)(untensor.data), element_count);
  } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && untensor_type == "u16") {
    copyI64ToU16((int64_t *)(ort_value.GetTensorMutableRawData()),
                 (uint16_t *)(untensor.data), element_count);
  } else {
    if (dtype_size_map.at(untensor.dtype) == ortvalue_size_map.at(type)) {
      memcpy(untensor.data, ort_value.GetTensorMutableRawData(), untensor.size);
    } else {
      minilog::T_warning << "ort_value element size: " << ortvalue_size_map.at(type)
            << "un_tensor element size: " << untensor.size << minilog::warningendl;
      throw std::runtime_error("Unsupported data type for conversion.");
    }
  }
}

void LoadOrtValuesToUnTensors(std::vector<Ort::Value> &ort_value_list,
                              std::vector<untensor_s> &untensor_list,
                              const int num) {
  assert(num == (int)(ort_value_list.size()));
  int idx = 0;
  for (auto &ort_value : ort_value_list) {
    auto &un_tensor = untensor_list[idx];
    ConvertOrtValueToUnTensor(ort_value, un_tensor);
    idx += 1;
  }
}