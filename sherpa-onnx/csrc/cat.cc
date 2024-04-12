// sherpa-onnx/csrc/cat.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/cat.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

template <typename T>
std::vector<std::vector<std::vector<T>>> Concatenate3DTensorsDim1(
    const std::vector<std::vector<std::vector<T>>>& tensors) {
    assert(!tensors.empty());

    size_t dim0 = tensors.front().size();
    size_t dim2 = tensors.front().front().size();

    for (const auto& tensor : tensors) {
        assert(tensor.size() == dim0);
        for (const auto& slice : tensor) {
            assert(slice.size() == dim2);
        }
    }

    // 计算拼接维度的总大小
    size_t total_dim1 = 0;
    for (const auto& tensor : tensors) {
        total_dim1 += tensor.front().size();
    }

    std::vector<std::vector<std::vector<T>>> result(dim0, std::vector<std::vector<T>>(total_dim1));

    for (size_t i = 0; i < dim0; ++i) {
        size_t current_dim1 = 0;
        for (const auto& tensor : tensors) {
            auto& result_slice = result[i];
            for (const auto& row : tensor[i]) {
                result_slice[current_dim1].reserve(dim2);
                std::copy(row.begin(), row.end(), std::back_inserter(result_slice[current_dim1]));
                ++current_dim1;
            }
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> Concatenate4DTensorsDim1(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& tensors) {
    assert(!tensors.empty());

    size_t dim0 = tensors.front().size();
    size_t dim2 = tensors.front().front().front().size();
    size_t dim3 = tensors.front().front().front().front().size();

    for (const auto& tensor : tensors) {
        assert(tensor.size() == dim0);
        for (const auto& mat : tensor) {
            assert(mat.front().size() == dim2);
            for (const auto& vec : mat) {
                assert(vec.size() == dim3);
            }
        }
    }

    // 计算拼接维度的总大小
    size_t total_dim1 = 0;
    for (const auto& tensor : tensors) {
        total_dim1 += tensor.front().size();
    }

    std::vector<std::vector<std::vector<std::vector<T>>>> result(dim0, std::vector<std::vector<std::vector<T>>>(total_dim1));

    for (size_t i = 0; i < dim0; ++i) {
        size_t current_dim1 = 0;
        for (const auto& tensor : tensors) {
            auto& result_mat = result[i];
            for (const auto& mat : tensor[i]) {
                result_mat[current_dim1].reserve(dim2);
                for (const auto& vec : mat) {
                    result_mat[current_dim1].push_back(vec); // Copy the 3D vector
                }
                ++current_dim1;
            }
        }
    }

    return result;
}

template <typename T>
std::vector<std::vector<std::vector<std::vector<T>>>> Concatenate4DTensorsDim2(
    const std::vector<std::vector<std::vector<std::vector<T>>>>& tensors) {
    assert(!tensors.empty());

    size_t dim0 = tensors.front().size();
    size_t dim1 = tensors.front().front().size();
    size_t dim3 = tensors.front().front().front().front().size();

    for (const auto& tensor : tensors) {
        assert(tensor.size() == dim0);
        for (const auto& mat : tensor) {
            assert(mat.size() == dim1);
            for (const auto& vec : mat) {
                assert(vec.size() == dim3);
            }
        }
    }

    // 计算拼接维度的总大小
    size_t total_dim2 = 0;
    for (const auto& tensor : tensors) {
        total_dim2 += tensor.front().front().size();
    }

    std::vector<std::vector<std::vector<std::vector<T>>>> result(dim0, std::vector<std::vector<std::vector<T>>>(dim1, std::vector<std::vector<T>>(total_dim2)));

    for (size_t i = 0; i < dim0; ++i) {
        for (size_t j = 0; j < dim1; ++j) {
            size_t current_dim2 = 0;
            for (const auto& tensor : tensors) {
                auto& result_slice = result[i][j];
                for (const auto& row : tensor[i][j]) {
                    result_slice[current_dim2].reserve(dim3);
                    std::copy(row.begin(), row.end(), std::back_inserter(result_slice[current_dim2]));
                    ++current_dim2;
                }
            }
        }
    }

    return result;
}

template std::vector<std::vector<std::vector<float>>> Concatenate3DTensorsDim1(
    const std::vector<std::vector<std::vector<float>>>& tensors);

template std::vector<std::vector<std::vector<int64_t>>> Concatenate3DTensorsDim1(
    const std::vector<std::vector<std::vector<int64_t>>>& tensors);

template std::vector<std::vector<std::vector<std::vector<float>>>> Concatenate4DTensorsDim1(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensors);

template std::vector<std::vector<std::vector<std::vector<int64_t>>>> Concatenate4DTensorsDim1(
    const std::vector<std::vector<std::vector<std::vector<int64_t>>>>& tensors);

template std::vector<std::vector<std::vector<std::vector<float>>>> Concatenate4DTensorsDim2(
    const std::vector<std::vector<std::vector<std::vector<float>>>>& tensors);

template std::vector<std::vector<std::vector<std::vector<int64_t>>>> Concatenate4DTensorsDim2(
    const std::vector<std::vector<std::vector<std::vector<int64_t>>>>& tensors);

}  // namespace sherpa_onnx
