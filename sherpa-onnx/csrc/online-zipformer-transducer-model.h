// sherpa-onnx/csrc/online-zipformer-transducer-model.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_
#define SHERPA_ONNX_CSRC_ONLINE_ZIPFORMER_TRANSDUCER_MODEL_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "onnx-to-bm.h"

#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-transducer-model.h"
#include "bmruntime_cpp.h"

namespace sherpa_onnx {

class OnlineZipformerTransducerModel : public OnlineTransducerModel {
 public:
  explicit OnlineZipformerTransducerModel(const OnlineModelConfig &config);

  virtual ~OnlineZipformerTransducerModel();
  
  std::vector<Ort::Value> StackStates(
      const std::vector<std::vector<Ort::Value>> &states) const override;

  std::vector<std::vector<Ort::Value>> UnStackStates(
      const std::vector<Ort::Value> &states) const override;

  std::vector<Ort::Value> GetEncoderInitStates() override;

  std::pair<Ort::Value, std::vector<Ort::Value>> RunEncoder(
      Ort::Value features, std::vector<Ort::Value> states,
      Ort::Value processed_frames) override;

  Ort::Value RunDecoder(Ort::Value decoder_input) override;

  Ort::Value RunJoiner(Ort::Value encoder_out, Ort::Value decoder_out) override;

  int32_t ContextSize() const override { return context_size_; }

  int32_t ChunkSize() const override { return T_; }

  int32_t ChunkShift() const override { return decode_chunk_len_; }

  int32_t VocabSize() const override { return vocab_size_; }

  OrtAllocator *Allocator() override { return allocator_; }

 private:
  void InitEncoder(const std::string &model_path);
  void InitDecoder(const std::string &model_path);
  void InitJoiner(const std::string &model_path);
  void ReleaseModels();

 private:
  int dev_id = 0;

  std::shared_ptr<bmruntime::Context> encoder_ctx;
  std::shared_ptr<bmruntime::Network> encoder_net;
  int enc_in_num=36;
  int enc_out_num=36;

  std::shared_ptr<bmruntime::Context> decoder_ctx;
  std::shared_ptr<bmruntime::Network> decoder_net;
  int dec_in_num=1;
  int dec_out_num=1;

  std::shared_ptr<bmruntime::Context> joiner_ctx;
  std::shared_ptr<bmruntime::Network> joiner_net;
  int jo_in_num=2;
  int jo_out_num=1;

  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

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
