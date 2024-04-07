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
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace {

enum class ModelType {
  kZipformer,
  kUnknown,
};

}  // namespace

namespace sherpa_onnx {

static ModelType GetModelType(char *model_data, size_t model_data_length,
                              bool debug) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  Ort::SessionOptions sess_opts;
  sess_opts.SetIntraOpNumThreads(1);
  sess_opts.SetInterOpNumThreads(1);

  auto sess = std::make_unique<Ort::Session>(env, model_data, model_data_length,
                                             sess_opts);

  Ort::ModelMetadata meta_data = sess->GetModelMetadata();
  if (debug) {
    std::ostringstream os;
    PrintModelMetadata(os, meta_data);
    SHERPA_ONNX_LOGE("%s", os.str().c_str());
  }

  Ort::AllocatorWithDefaultOptions allocator;
  auto model_type =
      meta_data.LookupCustomMetadataMapAllocated("model_type", allocator);
  if (!model_type) {
    SHERPA_ONNX_LOGE(
        "No model_type in the metadata!\n"
        "Please make sure you are using the latest export-onnx.py from icefall "
        "to export your transducer models");
    return ModelType::kUnknown;
  }

  if (model_type.get() == std::string("zipformer")) {
    return ModelType::kZipformer;
  } else {
    SHERPA_ONNX_LOGE("Unsupported model_type: %s", model_type.get());
    return ModelType::kUnknown;
  }
}

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
    }
  }
  ModelType model_type = ModelType::kUnknown;

  {
    auto buffer = ReadFile(config.transducer.encoder);

    model_type = GetModelType(buffer.data(), buffer.size(), config.debug);
  }

  switch (model_type) {
    case ModelType::kZipformer:
      return std::make_unique<OnlineZipformerTransducerModel>(config);
    case ModelType::kUnknown:
      SHERPA_ONNX_LOGE("Unknown model type in online transducer!");
      return nullptr;
  }

  // unreachable code
  return nullptr;
}

Ort::Value OnlineTransducerModel::BuildDecoderInput(
    const std::vector<OnlineTransducerDecoderResult> &results) {
  int32_t batch_size = static_cast<int32_t>(results.size());
  int32_t context_size = ContextSize();
  std::array<int64_t, 2> shape{batch_size, context_size};
  Ort::Value decoder_input = Ort::Value::CreateTensor<int64_t>(
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

Ort::Value OnlineTransducerModel::BuildDecoderInput(
    const std::vector<Hypothesis> &hyps) {
  int32_t batch_size = static_cast<int32_t>(hyps.size());
  int32_t context_size = ContextSize();
  std::array<int64_t, 2> shape{batch_size, context_size};
  Ort::Value decoder_input = Ort::Value::CreateTensor<int64_t>(
      Allocator(), shape.data(), shape.size());
  int64_t *p = decoder_input.GetTensorMutableData<int64_t>();

  for (const auto &h : hyps) {
    std::copy(h.ys.end() - context_size, h.ys.end(), p);
    p += context_size;
  }
  return decoder_input;
}

}  // namespace sherpa_onnx
