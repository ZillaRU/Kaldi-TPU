// sherpa-onnx/csrc/online-zipformer-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cviruntime.h"  // NOLINT
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineZipformerTransducerModel(const OnlineModelConfig &config);

  std::vector<CVI_TENSOR> StackStates(
      const std::vector<std::vector<CVI_TENSOR>> &states) const override;

  std::vector<std::vector<CVI_TENSOR>> UnStackStates(
      const std::vector<CVI_TENSOR> &states) const override;

  std::vector<CVI_TENSOR> GetEncoderInitStates() override;

  std::pair<CVI_TENSOR, std::vector<CVI_TENSOR>> RunEncoder(
      CVI_TENSOR features, std::vector<CVI_TENSOR> states,
      CVI_TENSOR processed_frames) override;

  CVI_TENSOR RunDecoder(CVI_TENSOR decoder_input) override;

  CVI_TENSOR RunJoiner(CVI_TENSOR encoder_out, CVI_TENSOR decoder_out) override;

  int32_t ContextSize() const override { return context_size_; }

  int32_t ChunkSize() const override { return T_; }

  int32_t ChunkShift() const override { return decode_chunk_len_; }

 private:
  void InitEncoder(const std::string &model_path);
  void InitDecoder(const std::string &model_path);
  void InitJoiner(const std::string &model_path);

 private:
  std::unique_ptr<CVI_MODEL_HANDLE> encoder_sess_;
  std::unique_ptr<CVI_MODEL_HANDLE> decoder_sess_;
  std::unique_ptr<CVI_MODEL_HANDLE> joiner_sess_;

  std::vector<std::string> encoder_input_names_;
  std::vector<const char *> encoder_input_names_ptr_;

  std::vector<std::string> encoder_output_names_;
  std::vector<const char *> encoder_output_names_ptr_;

  std::vector<std::string> decoder_input_names_;
  std::vector<const char *> decoder_input_names_ptr_;

  std::vector<std::string> decoder_output_names_;
  std::vector<const char *> decoder_output_names_ptr_;

  std::vector<std::string> joiner_input_names_;
  std::vector<const char *> joiner_input_names_ptr_;

  std::vector<std::string> joiner_output_names_;
  std::vector<const char *> joiner_output_names_ptr_;

  OnlineModelConfig config_;

  std::vector<int32_t> encoder_dims_;
  std::vector<int32_t> attention_dims_;
  std::vector<int32_t> num_encoder_layers_;
  std::vector<int32_t> cnn_module_kernels_;
  std::vector<int32_t> left_context_len_;

  int32_t T_ = 0;
  int32_t decode_chunk_len_ = 0;

  int32_t context_size_ = 0;
  int32_t vocab_size_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
