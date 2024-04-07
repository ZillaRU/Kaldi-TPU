// sherpa-onnx/csrc/online-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#include "sherpa-onnx/csrc/online-transducer-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"
#include "sherpa-onnx/csrc/cvi-utils.h"

namespace {

enum class ModelType {
  kZipformer,
  kUnknown,
};

}  // namespace

namespace sherpa_onnx {

std::unique_ptr<OnlineTransducerModel> OnlineTransducerModel::Create(
    const OnlineModelConfig &config) {
  if (!config.model_type.empty()) {
    const auto &model_type = config.model_type;
    if (model_type == "zipformer") {
      return std::make_unique<OnlineZipformerTransducerModel>(config);
    } else {
      SHERPA_ONNX_LOGE(
          "Invalid model_type: %s. Trying to load the model to get its type",
          model_type.c_str());
          return nullptr;
    }
  }
  return std::make_unique<OnlineZipformerTransducerModel>(config);
}

CVI_TENSOR* OnlineTransducerModel::BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  int32_t context_size = ContextSize();
  std::array<int64_t, 2> shape{batch_size, context_size};
  CVI_TENSOR *decoder_input = new CVI_TENSOR;
  if (!decoder_input) {
    SHERPA_ONNX_LOGE("Failed to create decoder input tensor");
    return nullptr;
  }
  decoder_input->shape = shape.data();
  decoder_input-
      Allocator(), shape.data(), shape.size());
  int64_t *p = decoder_input.GetTensorMutableData<int64_t>();

  for (const auto &r : results) {
    const int64_t *begin = r.tokens.data() + r.tokens.size() - context_size;
    const int64_t *end = r.tokens.data() + r.tokens.size();
    std::copy(begin, end, p);
    p += context_size;
  }
  return decoder_input;
}

CVI_TENSOR OnlineTransducerModel::BuildDecoderInput(
    const std::vector<Hypothesis> &hyps) {
  int32_t batch_size = static_cast<int32_t>(hyps.size());
  int32_t context_size = ContextSize();
  std::array<int64_t, 2> shape{batch_size, context_size};
  CVI_TENSOR decoder_input = CVI_TENSOR::CreateTensor<int64_t>(
      Allocator(), shape.data(), shape.size());
  int64_t *p = decoder_input.GetTensorMutableData<int64_t>();

  for (const auto &h : hyps) {
    std::copy(h.ys.end() - context_size, h.ys.end(), p);
    p += context_size;
  }
  return decoder_input;
}

}  // namespace sherpa_onnx
