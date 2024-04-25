#include "bmruntime_cpp.h"
#include "string"
#include <memory>
#include <cassert>
#include <iostream>
using namespace bmruntime;

int main(int argc, char** argv) {
    std::string encoder_bmodel_path = "/data/Kaldi-TPU/zipformer-bmodels/zipformer_encoder_N1_F16_1684x.bmodel";
    std::string decoder_bmodel_path = "/data/Kaldi-TPU/zipformer-bmodels/zipformer_decoder_N1_F16_1684x.bmodel";
    std::string joiner_bmodel_joiner = "/data/Kaldi-TPU/zipformer-bmodels/zipformer_joiner_N1_F16_1684x.bmodel";
    int dev_id = 0;

    std::shared_ptr<Context> encoder_ctx = std::make_shared<Context>(dev_id);
    bm_status_t status = encoder_ctx->load_bmodel(encoder_bmodel_path.c_str());
    assert(BM_SUCCESS == status);

    std::shared_ptr<Context> decoder_ctx = std::make_shared<Context>(dev_id);
    status = decoder_ctx->load_bmodel(decoder_bmodel_path.c_str());
    assert(BM_SUCCESS == status);

    std::shared_ptr<Context> joiner_ctx = std::make_shared<Context>(dev_id);
    status = joiner_ctx->load_bmodel(joiner_bmodel_joiner.c_str());
    assert(BM_SUCCESS == status);

    // create Network
    std::vector<const char *> network_names;
    encoder_ctx->get_network_names(&network_names);
    auto encoder_net = std::make_shared<Network>(*encoder_ctx, network_names[0], 0); // use stage[0]
    assert(encoder_net->info()->input_num == 36 && encoder_net->info()->output_num == 36);

    // Initialize the memory space required for the input and output tensors
    auto encoder_inputs = encoder_net->Inputs();
    auto encoder_outputs = encoder_net->Outputs();

    for(auto itr = encoder_outputs.begin(); itr != encoder_outputs.end(); itr++) {
        std::cout << (*itr)->num_elements() << " " << (*itr)->tensor()->dtype << std::endl;
    }

    decoder_ctx->get_network_names(&network_names);
    auto decoder_net = std::make_shared<Network>(*decoder_ctx, network_names[0], 0); // use stage[0]
    assert(decoder_net->info()->input_num == 1 && decoder_net->info()->output_num == 1);

    joiner_ctx->get_network_names(&network_names);
    auto joiner_net = std::make_shared<Network>(*joiner_ctx, network_names[0], 0); // use stage[0]
    assert(joiner_net->info()->input_num == 2 && joiner_net->info()->output_num == 1);

    return 0;
}