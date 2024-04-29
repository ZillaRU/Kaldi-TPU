import os
import numpy as np

def generate_encoder_input():
  npz_file = "/home/shenglinli/code/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder.npz"
  N = 1
  x = np.random.randn(N, 39, 80).astype(np.float32)
  cached_len_0 = np.random.randint(0, 10, dtype = np.int32, size = (2, N))
  cached_len_1 = np.random.randint(0, 10, dtype = np.int32, size = (4, N))
  cached_len_2 = np.random.randint(0, 10, dtype = np.int32, size = (3, N))
  cached_len_3 = np.random.randint(0, 10, dtype = np.int32, size = (2, N))
  cached_len_4 = np.random.randint(0, 10, dtype = np.int32, size = (4, N))
  cached_avg_0 = np.random.randn(2, N, 384).astype(np.float32)
  cached_avg_1 = np.random.randn(4, N, 384).astype(np.float32)
  cached_avg_2 = np.random.randn(3, N, 384).astype(np.float32)
  cached_avg_3 = np.random.randn(2, N, 384).astype(np.float32)
  cached_avg_4 = np.random.randn(4, N, 384).astype(np.float32)
  cached_key_0 = np.random.randn(2, 64, N, 192).astype(np.float32)
  cached_key_1 = np.random.randn(4, 32, N, 192).astype(np.float32)
  cached_key_2 = np.random.randn(3, 16, N, 192).astype(np.float32)
  cached_key_3 = np.random.randn(2, 8, N, 192).astype(np.float32)
  cached_key_4 = np.random.randn(4, 32, N, 192).astype(np.float32)
  cached_val_0 = np.random.randn(2, 64, N, 96).astype(np.float32)
  cached_val_1 = np.random.randn(4, 32, N, 96).astype(np.float32)
  cached_val_2 = np.random.randn(3, 16, N, 96).astype(np.float32)
  cached_val_3 = np.random.randn(2, 8, N, 96).astype(np.float32)
  cached_val_4 = np.random.randn(4, 32, N, 96).astype(np.float32)
  cached_val2_0 = np.random.randn(2, 64, N, 96).astype(np.float32)
  cached_val2_1 = np.random.randn(4, 32, N, 96).astype(np.float32)
  cached_val2_2 = np.random.randn(3, 16, N, 96).astype(np.float32)
  cached_val2_3 = np.random.randn(2, 8, N, 96).astype(np.float32)
  cached_val2_4 = np.random.randn(4, 32, N, 96).astype(np.float32)
  cached_conv1_0 = np.random.randn(2, N, 384, 30).astype(np.float32)
  cached_conv1_1 = np.random.randn(4, N, 384, 30).astype(np.float32)
  cached_conv1_2 = np.random.randn(3, N, 384, 30).astype(np.float32)
  cached_conv1_3 = np.random.randn(2, N, 384, 30).astype(np.float32)
  cached_conv1_4 = np.random.randn(4, N, 384, 30).astype(np.float32)
  cached_conv2_0 = np.random.randn(2, N, 384, 30).astype(np.float32)
  cached_conv2_1 = np.random.randn(4, N, 384, 30).astype(np.float32)
  cached_conv2_2 = np.random.randn(3, N, 384, 30).astype(np.float32)
  cached_conv2_3 = np.random.randn(2, N, 384, 30).astype(np.float32)
  cached_conv2_4 = np.random.randn(4, N, 384, 30).astype(np.float32)
  np.savez(npz_file, x = x, 
           cached_len_0 = cached_len_0, cached_len_1 = cached_len_1, cached_len_2 = cached_len_2, cached_len_3 = cached_len_3, cached_len_4 = cached_len_4,
           cached_avg_0 = cached_avg_0, cached_avg_1 = cached_avg_1, cached_avg_2 = cached_avg_2, cached_avg_3 = cached_avg_3, cached_avg_4 = cached_avg_4,
           cached_key_0 = cached_key_0, cached_key_1 = cached_key_1, cached_key_2 = cached_key_2, cached_key_3 = cached_key_3, cached_key_4 = cached_key_4,
           cached_val_0 = cached_val_0, cached_val_1 = cached_val_1, cached_val_2 = cached_val_2, cached_val_3 = cached_val_3, cached_val_4 = cached_val_4,
           cached_val2_0 = cached_val2_0, cached_val2_1 = cached_val2_1, cached_val2_2 = cached_val2_2, cached_val2_3 = cached_val2_3, cached_val2_4 = cached_val2_4,
           cached_conv1_0 = cached_conv1_0, cached_conv1_1 = cached_conv1_1, cached_conv1_2 = cached_conv1_2, cached_conv1_3 = cached_conv1_3, cached_conv1_4 = cached_conv1_4,
           cached_conv2_0 = cached_conv2_0, cached_conv2_1 = cached_conv2_1, cached_conv2_2 = cached_conv2_2, cached_conv2_3 = cached_conv2_3, cached_conv2_4 = cached_conv2_4)
  print("generate_encoder_input finish.")

def generate_decoder_input():
  npz_file = "/home/shenglinli/code/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder.npz"
  N = 1
  y = np.random.randint(0, 10, dtype = np.int32, size = (N, 2))
  np.savez(npz_file, y = y)
  print("generate_decoder_input finish.")

def generate_decoder_cvimodel_input():
  npz_file = "/home/shenglinli/code/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder_cvimodel_input.npz"
  N = 1
  y = np.random.randint(0, 10, dtype = np.uint16, size = (N, 2))
  np.savez(npz_file, y = y)
  print("generate_decoder_cvimodel_input finish.")

def generate_joiner_input():
  npz_file = "/home/shenglinli/code/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner.npz"
  N = 1
  encoder_out = np.random.randn(N, 512).astype(np.float32)
  decoder_out = np.random.randn(N, 512).astype(np.float32)
  np.savez(npz_file, encoder_out = encoder_out, decoder_out = decoder_out)
  print("generate_joiner_input finish.")
generate_joiner_input()
#generate_decoder_cvimodel_input()
# generate_decoder_input()