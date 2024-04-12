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

struct EncoderOut {
  std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>> encoder_out_Add_f32;
  [1] new_cached_len_0_Concat_f32 <2,1,1,1>,fp32
  [2] new_cached_len_1_Concat_f32 <4,1,1,1>,fp32
  [3] new_cached_len_2_Concat_f32 <3,1,1,1>,fp32
  [4] new_cached_len_3_Concat_f32 <2,1,1,1>,fp32
  [5] new_cached_len_4_Concat_f32 <4,1,1,1>,fp32
  [6] new_cached_avg_0_Concat_f32 <2,1,384,1>,fp32
  [7] new_cached_avg_1_Concat_f32 <4,1,384,1>,fp32
  [8] new_cached_avg_2_Concat_f32 <3,1,384,1>,fp32
  [9] new_cached_avg_3_Concat_f32 <2,1,384,1>,fp32
  [10] new_cached_avg_4_Concat_f32 <4,1,384,1>,fp32
  [11] new_cached_key_0_Concat_f32 <2,64,1,192>,fp32
  [12] new_cached_key_1_Concat_f32 <4,32,1,192>,fp32
  [13] new_cached_key_2_Concat_f32 <3,16,1,192>,fp32
  [14] new_cached_key_3_Concat_f32 <2,8,1,192>,fp32
  [15] new_cached_key_4_Concat_f32 <4,32,1,192>,fp32
  [16] new_cached_val_0_Concat_f32 <2,64,1,96>,fp32
  [17] new_cached_val_1_Concat_f32 <4,32,1,96>,fp32
  [18] new_cached_val_2_Concat_f32 <3,16,1,96>,fp32
  [19] new_cached_val_3_Concat_f32 <2,8,1,96>,fp32
  [20] new_cached_val_4_Concat_f32 <4,32,1,96>,fp32
  [21] new_cached_val2_0_Concat_f32 <2,64,1,96>,fp32
  [22] new_cached_val2_1_Concat_f32 <4,32,1,96>,fp32
  [23] new_cached_val2_2_Concat_f32 <3,16,1,96>,fp32
  [24] new_cached_val2_3_Concat_f32 <2,8,1,96>,fp32
  [25] new_cached_val2_4_Concat_f32 <4,32,1,96>,fp32
  [26] new_cached_conv1_0_Concat_f32 <2,1,384,30>,fp32
  [27] new_cached_conv1_1_Concat_f32 <4,1,384,30>,fp32
  [28] new_cached_conv1_2_Concat_f32 <3,1,384,30>,fp32
  [29] new_cached_conv1_3_Concat_f32 <2,1,384,30>,fp32
  [30] new_cached_conv1_4_Concat_f32 <4,1,384,30>,fp32
  [31] new_cached_conv2_0_Concat_f32 <2,1,384,30>,fp32
  [32] new_cached_conv2_1_Concat_f32 <4,1,384,30>,fp32
  [33] new_cached_conv2_2_Concat_f32 <3,1,384,30>,fp32
  [34] new_cached_conv2_3_Concat_f32 <2,1,384,30>,fp32
  [35] new_cached_conv2_4_Concat_f32 <4,1,384,30>,fp32

}

#endif  // SHERPA_ONNX_CSRC_STATE_STRUCT_H_