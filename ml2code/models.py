import onnx
import torch
import numpy as np
from onnx.helper import tensor_dtype_to_np_dtype
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit
from extra.onnx import get_run_onnx
from extra.export_model import export_model, compile_net, jit_model
import onnx2torch
import onnxruntime as ort

#{'x.1': <Tensor <LB RUST (1, 32) contig:True (<LoadOps.COPY: 3>, <buf real:True device:RUST size:32 dtype:dtypes.float>)> on RUST with grad None>}
#{'x.1': <Tensor <LB RUST (1, 32) contig:True (<LoadOps.COPY: 3>, None)> on RUST with grad None>}

def doo_foo():
  print("foo")


class ModelInputs:
  def __init__(self, model):
    np.random.seed(123)
    excluded = {inp.name for inp in model.graph.initializer}
    # first make input_shapes and input_types by walking the graph
    input_shapes = {}
    input_types = {}
    for inp in model.graph.input:
      if inp.name in excluded: continue
      t = []
      for x in inp.type.tensor_type.shape.dim:
        y = x.dim_value if x.dim_value != 0 else 1
        t.append(y)
      input_shapes[inp.name] = t
      input_types[inp.name] = tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
    # now we can make the inputs formated as required
    self.np_inputs = {}
    for k,shp in input_shapes.items():
      #self.np_inputs[k] = torch.randn(shp).numpy().astype(input_types[k])
      print(k, shp, input_types[k])
      self.np_inputs[k] = np.random.randn(*shp).astype(input_types[k])

  def torch(self):
    return [torch.tensor(x) for x in self.np_inputs.values()]
  
  def tiny(self):
    return {k:Tensor(inp) for k,inp in self.np_inputs.items()}
  
  def onnx(self):
    return self.np_inputs


class OnnxModel:

  def __init__(self, model_path):
    self.model_path = model_path
    self.model = onnx.load(model_path)
    self.modelinputs = ModelInputs(self.model)
    self.inputs = self.modelinputs.onnx()
    self.output_names = [out.name for out in self.model.graph.output]
    self.input_names = [inp.name for inp in self.model.graph.input]

    print(self.inputs)

  def tiny(self):
    return TinyModel(self.model, self.modelinputs.tiny())

  def torch(self):
    return TorchModel(self.model, self.modelinputs.torch())

  def run(self, input, backend="CPU"):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3
    provider = backend.upper()+"ExecutionProvider"
    if provider not in ort.get_available_providers(): raise Exception(f"Provider {provider} not found")
    session = ort.InferenceSession(self.model.SerializeToString(), opts, [provider])
    output = session.run(self.output_names, self.inputs)
    return output


# converts ONNX model to Tinygrad compatible model
class TinyOnnx:
  def __init__(self, onnxmodel):
    self.model = onnxmodel
    self.output_names = [out.name for out in self.model.graph.output]
    self.input_names = [inp.name for inp in self.model.graph.input]
    self.run_onnx = get_run_onnx(onnxmodel)

  def forward(self, xin):
    print("forward")
    print(xin)
    x = {self.input_names[i]:xa for i,xa in enumerate(xin)}
    print(x)
    print(type(x))
    raise
    a = self.run_onnx(x, debug=3)
    return list(a.values())


class TinyModel:

  def __init__(self, model, inputs):
    if 'onnx' in str(type(model)):
      #runner = get_run_onnx(model)
      self.model = TinyOnnx(model)
      #self.model = TinyJit(lambda args,**kwargs: (args, {k:v.realize() for k,v in runner(kwargs).items())})
    else:
      raise Exception(f"Unknown model type - {type(model)}")
    self.inputs = [v for _,v in inputs.items()]
    model_code, ainputs, outputs, _ = export_model(self.model, "rust", self.inputs, encoded_weights=True)
    print(model_code)
    print(f"ainputs: {ainputs}")
    print(outputs)
    self.runner, self.special_names = jit_model(self.model, self.inputs)

  def run(self, input):
    print("special names: ", self.special_names)
    output = self.runner(input)
    #return {k:v.numpy() for k,v in output.items()}
    return [o.numpy() for o in output]

  def export(self):
    mc, i, o, d = export_model(self.model, "clang", self.inputs)
    print(mc)
    print(i)
    print(o)
    print(d)



class TorchModel:

  def __init__(self, model, inputs):
    if 'onnx' in str(type(model)):
      self.model = onnx2torch.convert(model)
    elif 'torch' in str(type(model)):
      self.model = model
    else:
      raise Exception(f"Unknown model type - {type(model)}")
    self.inputs = inputs

  def run(self, inputs, device="cpu"):
    model = self.model.to(device)
    torch_inputs = [x.to(device) for x in inputs]
    with torch.no_grad():
      output = model(*torch_inputs)
    return output.cpu().numpy()
