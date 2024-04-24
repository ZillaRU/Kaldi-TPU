// sherpa-onnx/csrc/online-recognizer.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_

#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/endpoint.h"
#include "sherpa-onnx/csrc/features.h"
#include "sherpa-onnx/csrc/online-model-config.h"
#include "sherpa-onnx/csrc/online-stream.h"
#include "sherpa-onnx/csrc/online-transducer-model-config.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OnlineRecognizerResult {
  /// Recognition results.
  /// For English, it consists of space separated words.
  /// For Chinese, it consists of Chinese words without spaces.
  /// Example 1: "hello world"
  /// Example 2: "你好世界"
  std::string text;

  /// Decoded results at the token level.
  /// For instance, for BPE-based models it consists of a list of BPE tokens.
  std::vector<std::string> tokens;

  /// timestamps.size() == tokens.size()
  /// timestamps[i] records the time in seconds when tokens[i] is decoded.
  std::vector<float> timestamps;

  std::vector<float> ys_probs;  //< log-prob scores from ASR model
                                //
  /// log-domain scores from "hot-phrase" contextual boosting
  std::vector<float> context_scores;

  /// ID of this segment
  /// When an endpoint is detected, it is incremented
  int32_t segment = 0;

  /// Starting time of this segment.
  /// When an endpoint is detected, it will change
  float start_time = 0;

  /// True if the end of this segment is reached
  bool is_final = false;

  /** Return a json string.
   *
   * The returned string contains:
   *   {
   *     "text": "The recognition result",
   *     "tokens": [x, x, x],
   *     "timestamps": [x, x, x],
   *     "ys_probs": [x, x, x],
   *     "lm_probs": [x, x, x],
   *     "context_scores": [x, x, x],
   *     "segment": x,
   *     "start_time": x,
   *     "is_final": true|false
   *   }
   */
  std::string AsJsonString() const;
};

struct OnlineRecognizerConfig {
  FeatureExtractorConfig feat_config;
  OnlineModelConfig model_config;
  EndpointConfig endpoint_config;
  bool enable_endpoint = true;

  std::string decoding_method = "greedy_search";
  // now support modified_beam_search and greedy_search

  // used only for modified_beam_search
  int32_t max_active_paths = 4;

  /// used only for modified_beam_search
  float hotwords_score = 1.5;
  std::string hotwords_file;

  float blank_penalty = 0.0;

  OnlineRecognizerConfig() = default;

  OnlineRecognizerConfig(const FeatureExtractorConfig &feat_config,
                         const OnlineModelConfig &model_config,
                         const EndpointConfig &endpoint_config,
                         bool enable_endpoint,
                         const std::string &decoding_method,
                         int32_t max_active_paths,
                         const std::string &hotwords_file, float hotwords_score,
                         float blank_penalty)
      : feat_config(feat_config),
        model_config(model_config),
        endpoint_config(endpoint_config),
        enable_endpoint(enable_endpoint),
        decoding_method(decoding_method),
        max_active_paths(max_active_paths),
        hotwords_score(hotwords_score),
        hotwords_file(hotwords_file),
        blank_penalty(blank_penalty) {}

  void Register(ParseOptions *po);
  bool Validate() const;

  std::string ToString() const;
};

class OnlineRecognizerImpl;

class OnlineRecognizer {
 public:
  explicit OnlineRecognizer(const OnlineRecognizerConfig &config);

#if __ANDROID_API__ >= 9
  OnlineRecognizer(AAssetManager *mgr, const OnlineRecognizerConfig &config);
#endif

  ~OnlineRecognizer();

  /// Create a stream for decoding.
  std::unique_ptr<OnlineStream> CreateStream() const;

  /** Create a stream for decoding.
   *
   *  @param The hotwords for this string, it might contain several hotwords,
   *         the hotwords are separated by "/". In each of the hotwords, there
   *         are cjkchars or bpes, the bpe/cjkchar are separated by space (" ").
   *         For example, hotwords I LOVE YOU and HELLO WORLD, looks like:
   *
   *         "▁I ▁LOVE ▁YOU/▁HE LL O ▁WORLD"
   */
  std::unique_ptr<OnlineStream> CreateStream(const std::string &hotwords) const;

  /**
   * Return true if the given stream has enough frames for decoding.
   * Return false otherwise
   */
  bool IsReady(OnlineStream *s) const;

  /** Decode a single stream. */
  void DecodeStream(OnlineStream *s) const {
    OnlineStream *ss[1] = {s};
    DecodeStreams(ss, 1);
  }

  /** Decode multiple streams in parallel
   *
   * @param ss Pointer array containing streams to be decoded.
   * @param n Number of streams in `ss`.
   */
  void DecodeStreams(OnlineStream **ss, int32_t n) const;

  OnlineRecognizerResult GetResult(OnlineStream *s) const;

  // Return true if we detect an endpoint for this stream.
  // Note: If this function returns true, you usually want to
  // invoke Reset(s).
  bool IsEndpoint(OnlineStream *s) const;

  // Clear the state of this stream. If IsEndpoint(s) returns true,
  // after calling this function, IsEndpoint(s) will return false
  void Reset(OnlineStream *s) const;

 private:
  std::unique_ptr<OnlineRecognizerImpl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_ONLINE_RECOGNIZER_H_
