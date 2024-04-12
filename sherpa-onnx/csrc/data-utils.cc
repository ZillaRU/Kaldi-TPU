// sherpa-onnx/csrc/onnx-utils.cc
//
// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023  Pingfeng Luo
#include "sherpa-onnx/csrc/data-utils.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <numeric>

namespace sherpa_onnx {

std::vector<float> GetEncoderOutFrame(const std::vector<float> &encoder_out,
                                      int32_t t) {
  assert(t < static_cast<int32_t>(encoder_out.size()) / 3);

  auto batch_size = encoder_out[1];
  auto encoder_out_dim = encoder_out[2];

  std::vector<float> ans(batch_size * encoder_out_dim);

  for (int32_t i = 0; i != batch_size; ++i) {
    std::copy(encoder_out.begin() + t * encoder_out_dim + i * encoder_out.size() / 3,
              encoder_out.begin() + (t + 1) * encoder_out_dim + i * encoder_out.size() / 3,
              ans.begin() + i * encoder_out_dim);
  }
  return ans;
}

std::vector<float> Repeat(const std::vector<float> &cur_encoder_out,
                  const std::vector<int32_t> &hyps_num_split) {
  int32_t encoder_out_dim = cur_encoder_out.size() / hyps_num_split.front();
  int32_t new_size = std::accumulate(hyps_num_split.begin(), hyps_num_split.end(), 0) * encoder_out_dim;
  std::vector<float> ans(new_size);
  auto it_ans = ans.begin();
  auto it_src = cur_encoder_out.begin();
  for (int32_t b = 0; b != hyps_num_split.size() - 1; ++b) {
    int32_t cur_stream_hyps_num = hyps_num_split[b + 1] - hyps_num_split[b];
    for (int32_t i = 0; i != cur_stream_hyps_num; ++i) {
      std::copy_n(it_src, encoder_out_dim, it_ans);
      it_ans += encoder_out_dim;
    }
    it_src += encoder_out_dim;
  }
  return ans;
}

