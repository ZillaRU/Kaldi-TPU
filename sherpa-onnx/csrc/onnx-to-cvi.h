#include "cviruntime.h"
#include "onnxruntime_cxx_api.h" // NOLINT

Ort::Value GetOrtValueFromCviTensor(const CVI_TENSOR& cvi_tensor, Ort::Allocator& allocator) {
    // 根据CVI_TENSOR的数据类型确定Ort的数据类型
    OrtDataType type;
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

    // 创建Ort的内存信息
    Ort::MemoryInfo memory_info(allocator);

    // 创建张量并填充数据
    std::vector<int64_t> shape(cvi_tensor.shape.dim, cvi_tensor.shape.dim + cvi_tensor.shape.dim_size);
    Ort::Value ort_value = Ort::Value::CreateTensor(memory_info, type, shape);

    // 假设cvi_tensor的数据已经准备好，直接复制到Ort::Value中
    Ort::Value::TensorType& tensor = ort_value.GetTensorMutable();
    tensor.MutableDataRaw().CopyFrom(cvi_tensor.sys_mem, cvi_tensor.mem_size);

    // 返回创建的Ort::Value
    return ort_value;
}

void ConvertOrtValueToCviTensor(const Ort::Value& ort_value, CVI_TENSOR& cvi_tensor) {
    // 设置数据类型
    cvi_tensor.fmt = static_cast<CVI_FMT>(ort_value.GetTensorTypeAndShapeInfo().GetElementType());

    // 获取形状信息并设置
    const std::vector<int64_t>& shape = ort_value.GetTensorTypeAndShapeInfo().GetShape();
    cvi_tensor.shape.dim_size = shape.size();
    for (size_t i = 0; i < cvi_tensor.shape.dim_size; ++i) {
        cvi_tensor.shape.dim[i] = shape[i];
    }

    // 获取数据并设置
    cvi_tensor.count = 1;
    for (size_t i = 0; i < cvi_tensor.shape.dim_size; ++i) {
        cvi_tensor.count *= cvi_tensor.shape.dim[i];
    }
    cvi_tensor.mem_size = cvi_tensor.count * OrtDataTypeSize(cvi_tensor.fmt);
    cvi_tensor.sys_mem = malloc(cvi_tensor.mem_size);
    memcpy(cvi_tensor.sys_mem, ort_value.GetTensorData<float>(), cvi_tensor.mem_size);

    // 设置其他必要的CVI_TENSOR字段
    // ...

    return cvi_tensor;
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

std::vector<Ort::Value> GetOrtValuesFromCviTensors(const CVI_TENSOR* cvi_tensors, const int num, OrtAllocator *allocator) {
    std::vector<Ort::Value> target_ort_value_list;
    target_ort_value_list.reserve(num);
    for(int i = 0; i < num; i++) {
        auto &cvi_tensor = target_cvi_tensors[i];
        GetOrtValueFromCviTensor(cvi_tensor, allocator);
    }
}
