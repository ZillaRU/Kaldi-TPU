#include "bmruntime_cpp.h"
#include "string"
#include <memory>
#include <cassert>
#include <iostream>
using namespace bmruntime;

int main(int argc, char** argv) {
    std::string bmodel_path = "/data/Kaldi-TPU/zipformer-bmodels/zipformer_3in1.bmodel";
    int dev_id = 0;

    std::shared_ptr<Context> model_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = model_ctx->load_bmodel(bmodel_path.c_str());
    assert(BM_SUCCESS == status);
    // create Network
    std::vector<const char *> network_names;
    model_ctx->get_network_names(&network_names);
    std::cout << network_names[0] << network_names[1] << network_names[2] << std::endl;
    return 0;
}