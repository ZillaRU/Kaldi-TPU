// sherpa-onnx/csrc/online-zipformer-transducer-model.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/online-zipformer-transducer-model.h"

#include <assert.h>

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "cvi-utils.h"
#include "state_struct.h"
#include "cviruntime.h"
#include "vector-to-cvi.h"
#include "sherpa-onnx/csrc/cat.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/online-transducer-decoder.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/unbind.h"

namespace sherpa_onnx {

OnlineZipformerTransducerModel::OnlineZipformerTransducerModel(
    const OnlineModelConfig &config)
    : env_(ORT_LOGGING_LEVEL_WARNING),
      config_(config),
      sess_opts_(GetSessionOptions(config)),
      allocator_{} {
  { InitEncoder(config.transducer.encoder); }

  { InitDecoder(config.transducer.decoder); }

  { InitJoiner(config.transducer.joiner); }
}

void OnlineZipformerTransducerModel::InitEncoder(
    const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &encoder_sess_);
  GetInputOutPutInfo(encoder_sess_);

  // set meta data
  encoder_dims_ = {384, 384, 384, 384, 384};
  attention_dims_ = {192, 192, 192, 192, 192};
  num_encoder_layers_ = {2, 4, 3, 2, 4};
  cnn_module_kernels_ = {31, 31, 31, 31, 31};
  left_context_len_ = {64, 32, 16, 8, 32};
  T_ = 39;
  decode_chunk_len_ = 32;
}

void OnlineZipformerTransducerModel::InitDecoder(
    const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &decoder_sess_);
  GetInputOutPutInfo(decoder_sess_);

  // set meta data
  vocab_size_ = 6254;
  context_size_ = 2;
}

void OnlineZipformerTransducerModel::InitJoiner(const std::string &model_path) {
  CVI_NN_RegisterModel(model_path.c_str(), &joiner_sess_);
  GetInputOutPutInfo(joiner_sess_);
  // set meta data
  // joiner_dim = 512
}

void OnlineZipformerTransducerModel::ReleaseModels() {
  CVI_NN_CleanupModel(encoder_sess_);
  CVI_NN_CleanupModel(decoder_sess_);
  CVI_NN_CleanupModel(joiner_sess_);
  TPU_LOG_INFO("release CVI models");
}

OnlineZipformerTransducerModel::~OnlineZipformerTransducerModel() {
  ReleaseModels();
}

std::vector<Ort::Value> OnlineZipformerTransducerModel::StackStates(
    const std::vector<std::vector<Ort::Value>> &states) const {
  int32_t batch_size = static_cast<int32_t>(states.size());
  int32_t num_encoders = static_cast<int32_t>(num_encoder_layers_.size());

  std::vector<const Ort::Value *> buf(batch_size);

  std::vector<Ort::Value> ans;
  ans.reserve(states[0].size());

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][i];
    }
    auto v = Cat<int64_t>(allocator_, buf, 1);  // (num_layers, 1)
    ans.push_back(std::move(v));
  }

  // cached_avg
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders + i];
    }
    auto v = Cat(allocator_, buf, 1);  // (num_layers, 1, encoder_dims)
    ans.push_back(std::move(v));
  }

  // cached_key
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 2 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 3 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_val2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 4 + i];
    }
    // (num_layers, left_context_len, 1, attention_dims/2)
    auto v = Cat(allocator_, buf, 2);
    ans.push_back(std::move(v));
  }

  // cached_conv1
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 5 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  // cached_conv2
  for (int32_t i = 0; i != num_encoders; ++i) {
    for (int32_t n = 0; n != batch_size; ++n) {
      buf[n] = &states[n][num_encoders * 6 + i];
    }
    // (num_layers, 1, encoder_dims, cnn_module_kernels-1)
    auto v = Cat(allocator_, buf, 1);
    ans.push_back(std::move(v));
  }

  return ans;
}

std::vector<std::vector<Ort::Value>>
OnlineZipformerTransducerModel::UnStackStates(
    const std::vector<Ort::Value> &states) const {
  assert(states.size() == num_encoder_layers_.size() * 7);

  int32_t batch_size = states[0].GetTensorTypeAndShapeInfo().GetShape()[1];
  int32_t num_encoders = num_encoder_layers_.size();

  std::vector<std::vector<Ort::Value>> ans;
  ans.resize(batch_size);

  // cached_len
  for (int32_t i = 0; i != num_encoders; ++i) {
    auto v = Unbind<int64_t>(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_avg
  for (int32_t i = num_encoders; i != 2 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_key
  for (int32_t i = 2 * num_encoders; i != 3 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val
  for (int32_t i = 3 * num_encoders; i != 4 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_val2
  for (int32_t i = 4 * num_encoders; i != 5 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 2);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv1
  for (int32_t i = 5 * num_encoders; i != 6 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  // cached_conv2
  for (int32_t i = 6 * num_encoders; i != 7 * num_encoders; ++i) {
    auto v = Unbind(allocator_, &states[i], 1);
    assert(v.size() == batch_size);

    for (int32_t n = 0; n != batch_size; ++n) {
      ans[n].push_back(std::move(v[n]));
    }
  }

  return ans;
}

CachedTensors OnlineZipformerTransducerModel::GetEncoderInitStates() {
  // Please see
  // https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/pruned_transducer_stateless7_streaming/zipformer.py#L673
  // for details

  int32_t n = static_cast<int32_t>(encoder_dims_.size());
  std::vector<std::vector<int64_t> > cached_len_vec;
  std::vector<std::vector<std::vector<std::vector<float> > > cached_avg_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > cached_key_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > cached_val_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > cached_val2_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > cached_conv1_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float> > > > > cached_conv2_vec;

  cached_len_vec.reserve(n);
  cached_avg_vec.reserve(n);
  cached_key_vec.reserve(n);
  cached_val_vec.reserve(n);
  cached_val2_vec.reserve(n);
  cached_conv1_vec.reserve(n);
  cached_conv2_vec.reserve(n);

  for (int32_t i = 0; i != n; ++i) {
    {
      std::vector<int64_t> v(num_encoder_layers_[i], 0); 
      cached_len_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<float> > > v(
          num_encoder_layers_[i], 
          std::vector<std::vector<float>>(
              1, 
              std::vector<float>(encoder_dims_[i], 0.0f)
          )
      );
      cached_avg_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<std::vector<float> > > > v(
          num_encoder_layers_[i], 
          std::vector<std::vector<std::vector<float> > >(
              left_context_len_[i], 
              std::vector<std::vector<float>>(
                  1, 
                  std::vector<float>(attention_dims_[i], 0.0f)
              )
          )
      );
      cached_key_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<std::vector<float> > > > v(
          num_encoder_layers_[i],
          std::vector<std::vector<std::vector<float>>>(
              left_context_len_[i],
              std::vector<std::vector<float>>(
                  1,
                  std::vector<float>(attention_dims_[i] / 2, 0.0f)
              )
          )
      );
      cached_val_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<std::vector<float> > > > v(
          num_encoder_layers_[i],
          std::vector<std::vector<std::vector<float>>>(
              left_context_len_[i],
              std::vector<std::vector<float>>(
                  1,
                  std::vector<float>(attention_dims_[i] / 2, 0.0f)
              )
          )
      );
      cached_val2_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<std::vector<float> > > > v(
          num_encoder_layers_[i],
          std::vector<std::vector<std::vector<float>>>(
              1,
              std::vector<std::vector<float>>(
                  encoder_dims_[i],
                  std::vector<float>(cnn_module_kernels_[i] - 1, 0.0f)
              )
          )
      );
      cached_conv1_vec.push_back(std::move(v));
    }

    {
      std::vector<std::vector<std::vector<std::vector<float> > > > v(
          num_encoder_layers_[i],
          std::vector<std::vector<std::vector<float>>>(
              1,
              std::vector<std::vector<float>>(
                  encoder_dims_[i],
                  std::vector<float>(cnn_module_kernels_[i] - 1, 0.0f)
              )
          )
      );
      cached_conv2_vec.push_back(std::move(v));
    }

  }

  CachedTensors tensors;

  tensors.cached_len_vec = std::move(cached_len_vec);
  tensors.cached_avg_vec = std::move(cached_avg_vec);
  tensors.cached_key_vec = std::move(cached_key_vec);
  tensors.cached_val_vec = std::move(cached_val_vec);
  tensors.cached_val2_vec = std::move(cached_val2_vec);
  tensors.cached_conv1_vec = std::move(cached_conv1_vec);
  tensors.cached_conv2_vec = std::move(cached_conv2_vec);

  return tensors;
}

std::pair<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>, EncoderOutNextStates> >
OnlineZipformerTransducerModel::RunEncoder(Ort::Value features,
                                           std::vector<Ort::Value> states,
                                           Ort::Value /* processed_frames */) {
  std::vector<Ort::Value> encoder_inputs;
  encoder_inputs.reserve(1 + states.size());

  encoder_inputs.push_back(std::move(features));
  for (auto &v : states) {
    encoder_inputs.push_back(std::move(v));
  }

  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(encoder_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[encoder] input num: %d, output num: %d\n", input_num, output_num);

  LoadOrtValuesToCviTensors(encoder_inputs, input_tensors, input_num);

  CVI_NN_Forward(encoder_sess_, input_tensors, input_num, output_tensors, output_num);

  std::vector<Ort::Value> next_states =
      GetVectorsFromCviTensors(output_tensors + 1, output_num - 1);

  auto res = EncoderOut();
  res.encoder_out_Add_f32_ = GetOrtValueFromCviTensor(output_tensors[0]);
  return res;
}

Ort::Value OnlineZipformerTransducerModel::RunDecoder(
    Ort::Value decoder_input) {
  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(decoder_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[Decoder] input num: %d, output num: %d\n", input_num, output_num);
  std::vector<Ort::Value> temp = {};
  temp.push_back(std::move(decoder_input));
  LoadOrtValuesToCviTensors(temp, input_tensors, input_num);
  CVI_NN_Forward(decoder_sess_, input_tensors, input_num, output_tensors,
                 output_num);
  return std::move(GetOrtValueFromCviTensor(output_tensors[0]));
}

Ort::Value OnlineZipformerTransducerModel::RunJoiner(Ort::Value encoder_out,
                                                     Ort::Value decoder_out) {
  // get input / output tensors
  CVI_TENSOR *input_tensors, *output_tensors;
  int32_t input_num, output_num;
  CVI_NN_GetInputOutputTensors(joiner_sess_, &input_tensors, &input_num,
                               &output_tensors, &output_num);
  printf("[Decoder] input num: %d, output num: %d\n", input_num, output_num);
  std::vector<Ort::Value> temp = {};
  temp.push_back(std::move(encoder_out));
  temp.push_back(std::move(decoder_out));
  LoadOrtValuesToCviTensors(temp, input_tensors, input_num);
  CVI_NN_Forward(joiner_sess_, input_tensors, input_num, output_tensors,
                 output_num);
  return std::move(GetOrtValueFromCviTensor(output_tensors[0]));
}

}  // namespace sherpa_onnx
