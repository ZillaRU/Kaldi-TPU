// sherpa-onnx/csrc/sherpa-onnx-alsa.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cctype>  // std::tolower
#include <cstdint>

#include "sherpa-onnx/csrc/alsa.h"
#include "sherpa-onnx/csrc/display.h"
#include "sherpa-onnx/csrc/online-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"

bool stop = false;

static void Handler(int sig) {
  stop = true;
  fprintf(stderr, "\nCaught Ctrl + C. Exiting...\n");
}

int main(int32_t argc, char *argv[]) {
  signal(SIGINT, Handler);

  const char *kUsageMessage = R"usage(
Usage:
  ./bin/sherpa-onnx-alsa \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --provider=cpu \
    --num-threads=2 \
    --decoding-method=greedy_search \
    device_name

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
设备名用来指定系统中可用的麦克风中具体使用哪一个。

可以使用命令 arecord -l 查看你系统中所有可用的麦克风。例如，如果输出如下：

**** 列出所有捕获硬件设备 ****
card 3: UACDemoV10 [UACDemoV1.0], device 0: USB 音频 [USB 音频]
  Subdevices: 1/1
  Subdevice #0: subdevice #0

那么，如果你想选择 card 3 和 device 0 这个设备，请使用：
  plughw:3,0
作为 device_name。
)usage";
  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OnlineRecognizerConfig config;

  config.Register(&po);

  po.Read(argc, argv);
  if (po.NumArgs() != 1) {
    fprintf(stderr, "Please provide only 1 argument: the device name\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }
  sherpa_onnx::OnlineRecognizer recognizer(config);

  int32_t expected_sample_rate = config.feat_config.sampling_rate;

  std::string device_name = po.GetArg(1);
  sherpa_onnx::Alsa alsa(device_name.c_str());
  fprintf(stderr, "Use recording device: %s\n", device_name.c_str());

  if (alsa.GetExpectedSampleRate() != expected_sample_rate) {
    fprintf(stderr, "sample rate: %d != %d\n", alsa.GetExpectedSampleRate(),
            expected_sample_rate);
    exit(-1);
  }

  int32_t chunk = 0.1 * alsa.GetActualSampleRate();

  std::string last_text;

  auto stream = recognizer.CreateStream();

  sherpa_onnx::Display display;

  int32_t segment_index = 0;
  while (!stop) {
    const std::vector<float> &samples = alsa.Read(chunk);

    stream->AcceptWaveform(expected_sample_rate, samples.data(),
                           samples.size());

    while (recognizer.IsReady(stream.get())) {
      recognizer.DecodeStream(stream.get());
    }

    auto text = recognizer.GetResult(stream.get()).text;

    bool is_endpoint = recognizer.IsEndpoint(stream.get());

    if (!text.empty() && last_text != text) {
      last_text = text;

      std::transform(text.begin(), text.end(), text.begin(),
                     [](auto c) { return std::tolower(c); });

      display.Print(segment_index, text);
      fflush(stderr);
    }

    if (is_endpoint) {
      if (!text.empty()) {
        ++segment_index;
      }

      recognizer.Reset(stream.get());
    }
  }

  return 0;
}
