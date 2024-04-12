// sherpa-onnx/csrc/cat.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_CAT_H_
#define SHERPA_ONNX_CSRC_CAT_H_

#include <vector>

namespace sherpa_onnx {

template <typename T>
std::vector<std::vector<std::vector<T>>> Concatenate3DTensorsDim1(
    const std::vector<std::vector<std::vector<T>>>& tensors);

template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> Concatenate4DTensorsDim1(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& tensors);

template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> Concatenate4DTensorsDim2(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& tensors);

}

#endif  // SHERPA_ONNX_CSRC_CAT_H_
