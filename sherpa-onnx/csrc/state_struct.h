#ifndef SHERPA_ONNX_CSRC_STATE_STRUCT_H_
#define SHERPA_ONNX_CSRC_STATE_STRUCT_H_

#include <vector>
#include <cstdint>

struct CachedTensors {
  std::vector<std::vector<int64_t>> cached_len_vec;
  std::vector<std::vector<std::vector<std::vector<float>>>> cached_avg_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> cached_key_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> cached_val_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> cached_val2_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> cached_conv1_vec;
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> cached_conv2_vec;
};


struct EncoderOutNextStates {
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_len_0_Concat_f32; // <2,1,1,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_len_1_Concat_f32; //  <4,1,1,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_len_2_Concat_f32; //  <3,1,1,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_len_3_Concat_f32; //  <2,1,1,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_len_4_Concat_f32; // <4,1,1,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_avg_0_Concat_f32; // <2,1,384,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_avg_1_Concat_f32; // <4,1,384,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_avg_2_Concat_f32; // <3,1,384,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_avg_3_Concat_f32; // <2,1,384,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_avg_4_Concat_f32; // <4,1,384,1>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_key_0_Concat_f32; // <2,64,1,192>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_key_1_Concat_f32; // <4,32,1,192>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_key_2_Concat_f32; // <3,16,1,192>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_key_3_Concat_f32; // <2,8,1,192>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_key_4_Concat_f32; // <4,32,1,192>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val_0_Concat_f32; // <2,64,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val_1_Concat_f32; // <4,32,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val_2_Concat_f32; // <3,16,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val_3_Concat_f32; // <2,8,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val_4_Concat_f32; // <4,32,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val2_0_Concat_f32; // <2,64,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val2_1_Concat_f32; // <4,32,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val2_2_Concat_f32; // <3,16,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val2_3_Concat_f32; // <2,8,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_val2_4_Concat_f32; // <4,32,1,96>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv1_0_Concat_f32; // <2,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv1_1_Concat_f32; // <4,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv1_2_Concat_f32; // <3,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv1_3_Concat_f32; // <2,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv1_4_Concat_f32; // <4,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv2_0_Concat_f32; // <2,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv2_1_Concat_f32; // <4,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv2_2_Concat_f32; // <3,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv2_3_Concat_f32; // <2,1,384,30>,fp32
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> new_cached_conv2_4_Concat_f32; // <4,1,384,30>,fp32
};

struct DecoderOut {
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> decoder_out_Gemm_f32; // <1,512,1,1>,fp32
};

struct JoinerOut{
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> logit_Gemm_f32; // <1,6254,1,1>,fp32
};

#endif  // SHERPA_ONNX_CSRC_STATE_STRUCT_H_