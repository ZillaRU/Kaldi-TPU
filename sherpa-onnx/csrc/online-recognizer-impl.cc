// sherpa-onnx/csrc/online-recognizer-impl.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-recognizer-impl.h"

#include "sherpa-onnx/csrc/online-recognizer-ctc-impl.h"
#include "sherpa-onnx/csrc/online-recognizer-transducer-impl.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineRecognizerImpl> OnlineRecognizerImpl::Create(
    const OnlineRecognizerConfig &config) {
  if (!config.model_config.transducer.encoder.empty()) {
    return std::make_unique<OnlineRecognizerTransducerImpl>(config);
  }

  if (!config.model_config.wenet_ctc.model.empty() ||
      !config.model_config.zipformer2_ctc.model.empty()) {
    return std::make_unique<OnlineRecognizerCtcImpl>(config);
  }

  SHERPA_ONNX_LOGE("Please specify a model");
  exit(-1);
}

}  // namespace sherpa_onnx