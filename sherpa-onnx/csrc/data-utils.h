// sherpa-onnx/csrc/onnx-utils.h
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#ifndef SHERPA_ONNX_CSRC_ONNX_UTILS_H_
#define SHERPA_ONNX_CSRC_ONNX_UTILS_H_

#ifdef _MSC_VER
#endif

#include <cassert>
#include <ostream>
#include <string>
#include <utility>
#include <vector>


namespace sherpa_onnx {

Ort::Value GetEncoderOutFrame(OrtAllocator *allocator, Ort::Value *encoder_out,
                              int32_t t);

void PrintModelMetadata(std::ostream &os,
                        const Ort::ModelMetadata &meta_data);  // NOLINT

// Return a deep copy of v
Ort::Value Clone(OrtAllocator *allocator, const Ort::Value *v);

// Return a shallow copy
Ort::Value View(Ort::Value *v);

template <typename T = float>
void Fill(Ort::Value *tensor, T value) {
  auto n = tensor->GetTypeInfo().GetTensorTypeAndShapeInfo().GetElementCount();
  auto p = tensor->GetTensorMutableData<T>();
  std::fill(p, p + n, value);
}

std::vector<char> ReadFile(const std::string &filename);

// TODO(fangjun): Document it
Ort::Value Repeat(OrtAllocator *allocator, Ort::Value *cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split);
                  
}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONNX_UTILS_H_
