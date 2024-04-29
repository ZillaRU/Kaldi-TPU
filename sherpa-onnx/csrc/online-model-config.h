// sherpa-onnx/csrc/online-model-config.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/online-zipformer2-ctc-model-config.h"

namespace sherpa_onnx {

struct OnlineModelConfig {
  OnlineTransducerModelConfig transducer;
  OnlineZipformer2CtcModelConfig zipformer2_ctc;
  std::string tokens;
  int32_t num_threads = 1;
  int32_t warm_up = 0;
  bool is_debug = false;
  std::string provider = "cpu";

  // Valid values:
  //  - zipformer, zipformer transducer from icefall
  //
  // All other values are invalid and lead to loading the model twice.
  std::string model_type;

  OnlineModelConfig() = default;
  OnlineModelConfig(const OnlineTransducerModelConfig &transducer, const OnlineZipformer2CtcModelConfig &zipformer2_ctc,
                    const std::string &tokens, int32_t num_threads, int32_t warm_up, bool is_debug,
                    const std::string &provider, const std::string &model_type)
      : transducer(transducer),
        zipformer2_ctc(zipformer2_ctc),
        tokens(tokens),
        num_threads(num_threads),
        warm_up(warm_up),
        is_debug(is_debug),
        provider(provider),
        model_type(model_type) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_MODEL_CONFIG_H_
