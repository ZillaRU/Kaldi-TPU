import onnx
import onnx.checker

def get_fixed_shape():
  model = onnx.load("encoder-epoch-99-avg-1.onnx")
  for input in model.graph.input:
    # print(input.name)
    for dim in input.type.tensor_type.shape.dim:
      dim_proto = dim
      if dim_proto.dim_param == "N":
        dim_proto.dim_value = 1
  for output in model.graph.output:
    for dim in output.type.tensor_type.shape.dim:
      dim_proto = dim
      if dim_proto.dim_param == "N":
        dim_proto.dim_value = 1
  onnx.save(model, "encoder_fixed_shape.onnx")
    # print(input.type.tensor_type.shape.dim)
  # print(model.graph.input)
get_fixed_shape()