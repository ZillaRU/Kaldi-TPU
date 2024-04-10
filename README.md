# 实时语音识别 Demo （using zipformer）
## 1. 获取代码
执行`git clone https://github.com/ZillaRU/Kaldi-TPU.git`。

## 2. 环境准备
- 获取docker镜像，创建容器并进入：`docker run --privileged --name mytpudev -v $PWD:/workspace -it sophgo/tpuc_dev:latest`。执行完这一命令后，会进入docker，后续步骤都在`mytpudev`容器中执行。

- 若是在x86机器上模拟运行，需要在docker内执行。按以下步骤操作：
 1. 获取docker镜像，
 2. 获取`cvitek_tpu_sdk_x86`，然后执行`source cvitek_tpu_sdk_x86/envs_tpu_sdk.sh`。
 3. 设置TPU SDK路径，`export TPU_SDK_PATH=/workspace/cvitek_tpu_sdk_x86`。

- 若是在板子上实测，需要在x86机器上完成交叉编译，按以下步骤操作：
 1. 获取cvitek_tpu_sdk_rv64，然后执行`source cvitek_tpu_sdk_rv64/envs_tpu_sdk.sh`。
 2. 获取`riscv64-linux-musl-x86_64`，然后执行`export PATH=/workspace/riscv64-linux-musl-x86_64/bin:$PATH`添加到环境变量。
 3. 设置TPU SDK路径，`export TPU_SDK_PATH=/workspace/cvitek_tpu_sdk_rv64`。

- 安装ALSA音频框架
```sh
sudo apt-get update -y
sudo apt-get install -y alsa-utils libasound2-dev
```

## 3. 编译在docker的sample
docker内执行以下命令。
```sh
cd Kaldi-TPU
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DTPU_SDK_PATH=$TPU_SDK_PATH ..
make clean && make -j6
```
编译完成后，会在`build/bin`目录得到2个可执行文件。

## 4. 运行
- [下载cvimodel模型和token文件]
- 流式音频文件文字识别：命令行执行下列命令，
  ```sh
  ./bin/sherpa-onnx \
    --tokens=path-to-tokens.txt \
    --encoder=path-to-sherpa_encoder_bf16.cvimodel \
    --decoder=path-to-sherpa_decoder_bf16.cvimodel \
    --joiner=.path-to-sherpa_joiner_bf16.cvimodel \
    --provider=cpu \
    --num-threads=1 \
    --decoding-method=greedy_search \
    path-to-your-audio-file
  ```
- 实时语音识别（需要有麦克风等音频输入设备）
  ```sh
  ./bin/sherpa-onnx-alsa \
    --tokens=path-to-tokens.txt \
    --encoder=path-to-sherpa_encoder_bf16.cvimodel \
    --decoder=path-to-sherpa_decoder_bf16.cvimodel \
    --joiner=.path-to-sherpa_joiner_bf16.cvimodel \
    --provider=cpu \
    --num-threads=1 \
    --decoding-method=greedy_search \
    device_name
  ```
  device_name用来指定系统中可用的麦克风中具体使用哪一个。
  - 使用命令`arecord -l`来查看你系统中所有可用的麦克风。例如，如果输出如下：
  ```sh
  card 3: UACDemoV10 [UACDemoV1.0], device 0: USB 音频 [USB 音频]
    Subdevices: 1/1
    Subdevice #0: subdevice #0
  ```
  那么，如果你想选择 card 3 和 device 0 这个设备，请使用：`plughw:3,0`作为 device_name。

# Introduction

This repository supports running the following functions **locally**

  - Speech-to-text (i.e., ASR)
  - Text-to-speech (i.e., TTS)
  - Speaker identification

on the following platforms and operating systems:

  - Linux, macOS, Windows
  - Android
  - iOS
  - Raspberry Pi
  - etc

# Useful links

- Documentation: https://k2-fsa.github.io/sherpa/onnx/
- APK for the text-to-speech engine: https://k2-fsa.github.io/sherpa/onnx/tts/apk-engine.html
- APK for speaker identification: https://k2-fsa.github.io/sherpa/onnx/speaker-identification/apk.html

# How to reach us

Please see
https://k2-fsa.github.io/sherpa/social-groups.html
for 新一代 Kaldi **微信交流群** and **QQ 交流群**.
