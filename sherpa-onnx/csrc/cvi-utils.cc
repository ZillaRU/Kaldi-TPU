#include "sherpa-onnx/csrc/cvi-utils.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

#include "cviruntime.h"  // NOLINT
#include "cvitpu_debug.h"

namespace sherpa_onnx {

#define EXIT_IF_ERROR(cond, statement)                                                   \
  if ((cond)) {                                                                          \
    TPU_LOG_ERROR("%s\n", statement);                                                         \
    exit(1);                                                                             \
  }

static const char* formatToStr(CVI_FMT fmt) {
  switch(fmt) {
    case CVI_FMT_FP32:  return "fp32";
    case CVI_FMT_INT32:  return "i32";
    case CVI_FMT_UINT32: return "u32";
    case CVI_FMT_BF16:   return "bf16";
    case CVI_FMT_INT16:  return "i16";
    case CVI_FMT_UINT16: return "u16";
    case CVI_FMT_INT8:   return "i8";
    case CVI_FMT_UINT8:  return "u8";
    default:
      TPU_LOG_FATAL("unknown fmt:%d\n", fmt);
  }
  return nullptr;
}

void GetInputOutPutInfo(CVI_MODEL_HANDLE model) {
    CVI_TENSOR *input_tensors, *output_tensors;
    int32_t input_num, output_num;
    CVI_RC ret = CVI_NN_GetInputOutputTensors(model, &input_tensors, &input_num, &output_tensors, &output_num);

    EXIT_IF_ERROR(ret != CVI_RC_SUCCESS, "failed to get inputs & outputs from model");
    
    // print the inputs & outputs's information
    if (1) {
        TPU_LOG_INFO("Inputs:\n");
        for (int i = 0; i < input_num; ++i) {
        auto &tensor = input_tensors[i];
        TPU_LOG_INFO("  [%d] %s <%d,%d,%d,%d>,%s\n",
                    i, tensor.name, tensor.shape.dim[0], tensor.shape.dim[1], tensor.shape.dim[2],
                    tensor.shape.dim[3], formatToStr(tensor.fmt));
        }
        TPU_LOG_INFO("Outputs:\n");
        for (int i = 0; i < output_num; ++i) {
        auto &tensor = output_tensors[i];
        TPU_LOG_INFO("  [%d] %s <%d,%d,%d,%d>,%s\n",
                    i, tensor.name, tensor.shape.dim[0], tensor.shape.dim[1], tensor.shape.dim[2],
                    tensor.shape.dim[3], formatToStr(tensor.fmt));
        }
    }
}

}