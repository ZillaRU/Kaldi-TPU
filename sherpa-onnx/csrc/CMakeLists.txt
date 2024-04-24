include_directories(${CMAKE_SOURCE_DIR})

if(SHERPA_ONNX_ENABLE_PYTHON)
  message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print('.'.join(sys.version.split('.')[:2]))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_VERSION
  )
  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")
endif()

set(sources
  base64-decode.cc
  cat.cc
  circular-buffer.cc
  context-graph.cc
  endpoint.cc
  features.cc
  file-utils.cc
  hypothesis.cc
  online-model-config.cc
  online-recognizer-impl.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  onnx-utils.cc
  onnx-to-unt.cc
  pad-sequence.cc
  parse-options.cc
  provider.cc
  resample.cc
  session.cc
  symbol-table.cc
  text-utils.cc
  transpose.cc
  unbind.cc
  utils.cc
  wave-reader.cc
)

if(SHERPA_ONNX_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()
list(APPEND sources alsa.cc)
add_executable(sherpa-onnx-alsa sherpa-onnx-alsa.cc ${sources})


target_link_libraries(sherpa-onnx-alsa
  kaldi-native-fbank-core
  kaldi-decoder-core
)


if(BUILD_SHARED_LIBS)
  target_link_libraries(sherpa-onnx-alsa onnxruntime)
else()
  target_link_libraries(sherpa-onnx-alsa ${onnxruntime_lib_files})
endif()

# target_include_directories(sherpa-onnx-alsa PUBLIC ${CMAKE_SOURCE_DIR})

target_link_libraries(sherpa-onnx-alsa -lm)

target_include_directories(sherpa-onnx-alsa PRIVATE ${LIBSOPHON}/include)
link_directories(${LIBSOPHON}/lib)
target_link_libraries(sherpa-onnx-alsa bmlib libbmodel.a bmrt)
target_include_directories(sherpa-onnx-alsa PRIVATE /data/untool/include)



if(SHERPA_ONNX_ENABLE_CHECK)
  target_compile_definitions(sherpa-onnx-alsa PUBLIC SHERPA_ONNX_ENABLE_CHECK=1)

  if(SHERPA_ONNX_HAVE_EXECINFO_H)
    target_compile_definitions(sherpa-onnx-alsa PRIVATE SHERPA_ONNX_HAVE_EXECINFO_H=1)
  endif()

  if(SHERPA_ONNX_HAVE_CXXABI_H)
    target_compile_definitions(sherpa-onnx-alsa PRIVATE SHERPA_ONNX_HAVE_CXXABI_H=1)
  endif()
endif()

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-alsa -ldl)
endif()

if(NOT WIN32 AND NOT SHERPA_ONNX_ENABLE_WASM AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-alsa -pthread)
endif()

target_link_libraries(sherpa-onnx-alsa "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
target_link_libraries(sherpa-onnx-alsa "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")


if(SHERPA_ONNX_ENABLE_BINARY)
  install(
    TARGETS
      sherpa-onnx-alsa
    DESTINATION
      bin
  )
endif()
