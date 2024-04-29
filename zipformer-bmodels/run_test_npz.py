import numpy as np 
import time 
import torch
import os
import sophon.sail as sail 


class EngineOV:
    
    def __init__(self, model_path="",output_names="", device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model_path = model_path
        self.device_id = device_id
        try:
            self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
        except Exception as e:
            print("load model error; please check model path and device status;")
            print(">>>> model_path: ",model_path)
            print(">>>> device_id: ",device_id)
            print(">>>> sail.Engine error: ",e)
            raise e
        sail.set_print_flag(False)
        self.graph_name = self.model.get_graph_names()[0]
        self.input_name = self.model.get_input_names(self.graph_name)
        self.output_name= self.model.get_output_names(self.graph_name)

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
    def __call__(self, args):
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
        args = {}
        for i in range(len(values)):
            args[self.input_name[i]] = values[i]
        output = self.model.process(self.graph_name, args)
        res = []

        for name in self.output_name:
            res.append(output[name])
        return res

# net = EngineOV('zipformer_decoder_FP32.bmodel', device_id=0)
# dec_inputs = list(dict(np.load("test_npz/decoder_cvimodel_input.npz")).values())
# # dec_inputs = torch.from_numpy(dec_inputs)
# net_out = net(dec_inputs)[0]
# np.save('net_out.npy', net_out)
# gt = list(dict(np.load("test_npz/de_test.npz")).values())[0]
# print(np.mean(np.abs(net_out - gt)))

# net2 = EngineOV('zipformer_joiner_FP32.bmodel', device_id=0)
# joiner_inputs = list(dict(np.load("test_npz/joiner.npz")).values())
# # dec_inputs = torch.from_numpy(dec_inputs)
# net_out = net2(joiner_inputs)[0]
# np.save('join_out.npy', net_out)
# gt = list(dict(np.load("test_npz/jo_test.npz")).values())[0]
# print(np.mean(np.abs(net_out - gt)))

# net3 = EngineOV('zipformer_encoder_FP32.bmodel', device_id=0)
# _inputs = list(dict(np.load("test_npz/encoder.npz")).values())
# # dec_inputs = torch.from_numpy(dec_inputs)
# net_out = net3(_inputs)
# np.savez('en_out.npz', net_out)
# print('==================================')
# gt = list(dict(np.load("test_npz/en_test.npz", allow_pickle=True)).values())
# pred = list(dict(np.load("en_out.npz", allow_pickle=True)).values())
# for i in range(36):
#     gti=np.squeeze(gt[i])
#     pri=np.squeeze(pred[0][i])
#     np.save(f"temp-gt/{i}_"+str(gti.shape)+"_gt.npy", gti)
#     np.save(f"temp-pred/{i}_"+str(pri.shape)+".npy", pri)

import os

for i in os.listdir("temp-pred"):
    gti = np.load(f"temp-gt/{i.replace('.npy','_gt.npy')}")
    predi = np.load(f"temp-pred/{i}")
    print(np.mean(np.abs(gti-predi)))