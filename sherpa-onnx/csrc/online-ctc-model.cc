// sherpa-onnx/csrc/online-ctc-model.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-ctc-model.h"

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-zipformer2-ctc-model.h"
#include "sherpa-onnx/csrc/onnx-utils.h"

namespace sherpa_onnx {

std::unique_ptr<OnlineCtcModel> OnlineCtcModel::Create(
    const OnlineModelConfig &config) {
if (!config.zipformer2_ctc.model.empty()) {
    return std::make_unique<OnlineZipformer2CtcModel>(config);
  } else {
    SHERPA_ONNX_LOGE("Please specify a CTC model");
    exit(-1);
  }
}

}  // namespace sherpa_onnx