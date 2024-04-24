cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sherpa-onnx)

set(SHERPA_ONNX_VERSION "1.9.15")

# Disable warning about
#
# "The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
#  not set.
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

option(SHERPA_ONNX_ENABLE_PYTHON "Whether to build Python" OFF)
option(SHERPA_ONNX_ENABLE_TESTS "Whether to build tests" OFF)
option(SHERPA_ONNX_ENABLE_CHECK "Whether to build with assert" OFF)
option(BUILD_SHARED_LIBS "Whether to build shared libraries" OFF)
option(SHERPA_ONNX_ENABLE_PORTAUDIO "Whether to build with portaudio" ON)
option(SHERPA_ONNX_ENABLE_JNI "Whether to build JNI internface" OFF)
option(SHERPA_ONNX_ENABLE_C_API "Whether to build C API" ON)
option(SHERPA_ONNX_ENABLE_WEBSOCKET "Whether to build webscoket server/client" ON)
option(SHERPA_ONNX_ENABLE_GPU "Enable ONNX Runtime GPU support" OFF)
option(SHERPA_ONNX_ENABLE_WASM "Whether to enable WASM" OFF)
option(SHERPA_ONNX_ENABLE_WASM_TTS "Whether to enable WASM for TTS" OFF)
option(SHERPA_ONNX_ENABLE_WASM_ASR "Whether to enable WASM for ASR" OFF)
option(SHERPA_ONNX_ENABLE_WASM_KWS "Whether to enable WASM for KWS" OFF)
option(SHERPA_ONNX_ENABLE_WASM_NODEJS "Whether to enable WASM for NodeJS" OFF)
option(SHERPA_ONNX_ENABLE_BINARY "Whether to build binaries" ON)
option(SHERPA_ONNX_ENABLE_TTS "Whether to build TTS related code" OFF)
option(SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY "True to link libstdc++ statically. Used only when BUILD_SHARED_LIBS is OFF on Linux" ON)
option(SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE "True to use pre-installed onnxruntime if available" ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if(NOT APPLE)
  set(SHERPA_ONNX_RPATH_ORIGIN "$ORIGIN")
else()
  set(SHERPA_ONNX_RPATH_ORIGIN "@loader_path")
endif()

set(CMAKE_INSTALL_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})
set(CMAKE_BUILD_RPATH ${SHERPA_ONNX_RPATH_ORIGIN})

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Release")
  set(CMAKE_BUILD_TYPE Release)
endif()


message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS}")
message(STATUS "SHERPA_ONNX_ENABLE_PYTHON ${SHERPA_ONNX_ENABLE_PYTHON}")
message(STATUS "SHERPA_ONNX_ENABLE_TESTS ${SHERPA_ONNX_ENABLE_TESTS}")
message(STATUS "SHERPA_ONNX_ENABLE_CHECK ${SHERPA_ONNX_ENABLE_CHECK}")
message(STATUS "SHERPA_ONNX_ENABLE_PORTAUDIO ${SHERPA_ONNX_ENABLE_PORTAUDIO}")
message(STATUS "SHERPA_ONNX_ENABLE_JNI ${SHERPA_ONNX_ENABLE_JNI}")
message(STATUS "SHERPA_ONNX_ENABLE_C_API ${SHERPA_ONNX_ENABLE_C_API}")
message(STATUS "SHERPA_ONNX_ENABLE_WEBSOCKET ${SHERPA_ONNX_ENABLE_WEBSOCKET}")
message(STATUS "SHERPA_ONNX_ENABLE_GPU ${SHERPA_ONNX_ENABLE_GPU}")
message(STATUS "SHERPA_ONNX_ENABLE_WASM ${SHERPA_ONNX_ENABLE_WASM}")
message(STATUS "SHERPA_ONNX_ENABLE_WASM_TTS ${SHERPA_ONNX_ENABLE_WASM_TTS}")
message(STATUS "SHERPA_ONNX_ENABLE_WASM_ASR ${SHERPA_ONNX_ENABLE_WASM_ASR}")
message(STATUS "SHERPA_ONNX_ENABLE_WASM_KWS ${SHERPA_ONNX_ENABLE_WASM_KWS}")
message(STATUS "SHERPA_ONNX_ENABLE_WASM_NODEJS ${SHERPA_ONNX_ENABLE_WASM_NODEJS}")
message(STATUS "SHERPA_ONNX_ENABLE_BINARY ${SHERPA_ONNX_ENABLE_BINARY}")
message(STATUS "SHERPA_ONNX_ENABLE_TTS ${SHERPA_ONNX_ENABLE_TTS}")
message(STATUS "SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY ${SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY}")
message(STATUS "SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE ${SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE}")

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")


include(CheckIncludeFileCXX)


if(UNIX AND NOT APPLE AND NOT SHERPA_ONNX_ENABLE_WASM AND NOT CMAKE_SYSTEM_NAME STREQUAL Android)
  check_include_file_cxx(alsa/asoundlib.h SHERPA_ONNX_HAS_ALSA)
  if(SHERPA_ONNX_HAS_ALSA)
    message(STATUS "With Alsa")
    add_definitions(-DSHERPA_ONNX_ENABLE_ALSA=1)
  else()
    message(WARNING "\
Could not find alsa/asoundlib.h !
We won't build sherpa-onnx-alsa
To fix that, please do:
  (1) sudo apt-get update -y && sudo apt-get install -y alsa-utils libasound2-dev
  (2) rm -rf build
  (3) re-try
  ")
  endif()
endif()

check_include_file_cxx(cxxabi.h SHERPA_ONNX_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h SHERPA_ONNX_HAVE_EXECINFO_H)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  if(SHERPA_ONNX_LINK_LIBSTDCPP_STATICALLY)
    message(STATUS "Link libstdc++ statically")
    set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc ")
  else()
    message(STATUS "Link libstdc++ dynamically")
  endif()
endif()

include(kaldi-native-fbank)
include(kaldi-decoder)
include(onnxruntime)
set(ONNXRUNTIME_DIR ${onnxruntime_SOURCE_DIR})
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")

set(LIBSOPHON /opt/sophon/libsophon-current)

#set(UNTPU /data/untool/install)
#include_directories(${UNTPU}/include)
#link_directories(${UNTPU}/lib)
#message(STATUS "Using UNTPU: ${UNTPU}")

if(SHERPA_ONNX_ENABLE_TESTS)
  enable_testing()
  include(googletest)
endif()

add_subdirectory(sherpa-onnx)

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

if(NOT BUILD_SHARED_LIBS)
  if(APPLE)
    set(SHERPA_ONNX_PKG_CONFIG_EXTRA_LIBS "-lc++ -framework Foundation")
  endif()

  if(UNIX AND NOT APPLE)
    set(SHERPA_ONNX_PKG_CONFIG_EXTRA_LIBS "-lstdc++ -lm -pthread -ldl")
  endif()
endif()

# See https://people.freedesktop.org/~dbn/pkg-config-guide.html
configure_file(cmake/sherpa-onnx-no-tts.pc.in ${PROJECT_BINARY_DIR}/sherpa-onnx.pc @ONLY)

install(
  FILES
    ${PROJECT_BINARY_DIR}/sherpa-onnx.pc
  DESTINATION
    .
)
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
