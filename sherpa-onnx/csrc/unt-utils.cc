#include "sherpa-onnx/csrc/unt-utils.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include "runtime/unruntime.h"


void print_output_value(un_runtime_s* runtime, int start, int length){
    int output_num = runtime->output_tensors.size();
    for (int i = 0; i < output_num; i++) {
        print_data_by_fp32(runtime->output_tensors[i].data, runtime->output_tensors[i].size, runtime->output_tensors[i].dtype, start, length);
    }
}

void malloc_generate_host_data(un_runtime_s* runtime){
    int input_num = runtime->input_tensors.size();
    for (int i = 0; i < input_num; i++) {
        malloc_host_data(&runtime->input_tensors[i]);
    }
}