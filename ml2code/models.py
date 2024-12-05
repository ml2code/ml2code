import onnx
import torch
import time
import numpy as np
import json
import struct
from pandas.core.common import flatten
from onnx.helper import tensor_dtype_to_np_dtype
import onnx2torch
import onnxruntime as ort

# from tinygrad
from tinygrad import Tensor, Device
from tinygrad.engine.jit import TinyJit
from extra.onnx import get_run_onnx
from extra.export_model import export_model, compile_net, jit_model


def set_tinygrad_device(device):
  Device.DEFAULT = device.upper()


def compare_lists(n1, l1, n2, l2):
  if len(l1)!= len(l2):
    print(f"len({n1})={len(l1)} != len({n2})={len(l2)}")
    return False
  for i in range(len(l1)):
    c1 = l1[i]
    c2 = l2[i]
    if c1 == 0:
      a = c2-c1
    else:
      a = abs((c2-c1)/c1)
    if a > 0.0001:
      print(f"{n1}[{i}]={c1} != {n2}[{i}]={c2}")
      return False
  return True


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def make_np_inputs(onnx_model, seed=123):
  np.random.seed(seed)
  excluded = {inp.name for inp in onnx_model.graph.initializer}
  # first make input_shapes and input_types by walking the graph
  input_shapes = {}
  input_types = {}
  for inp in onnx_model.graph.input:
    if inp.name in excluded: continue
    t = []
    for x in inp.type.tensor_type.shape.dim:
      y = x.dim_value if x.dim_value != 0 else 1
      t.append(y)
    input_shapes[inp.name] = t
    input_types[inp.name] = tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
  # now we can make the inputs formated as required
  np_inputs = []
  for k,shp in input_shapes.items():
    np_inputs.append(np.random.randn(*shp).astype(input_types[k]))
  if len(np_inputs) != 1:
    raise Exception(f"Only one input is supported {np_inputs}")
  return np_inputs[0]


class ModelData:

  def __init__(self, onnx=None, binary=None, json=None, model=None, seed=123):
    if onnx is not None:
      self.np_inputs = onnx
    elif binary is not None:
      self.from_binary(binary)
    elif model is not None:
      self.from_model(model, seed)
    elif json is not None:
      self.from_json(json)
    else:
      raise Exception("Must provide either onnx, binary or model")

  def torch(self):
    return torch.tensor(self.np_inputs)

  def tiny(self):
    return Tensor(self.np_inputs)

  def onnx(self):
    return self.np_inputs

  def json(self):
    a = torch.tensor(np.array(self.np_inputs)).numpy()
    return json.dumps(a, cls=NumpyEncoder)

  def binary(self):
    # dumps LE binary
    return self.np_inputs.tobytes()

  def from_binary(self, filename):
    with open(filename, "rb") as f:
      a = f.read()
    b = struct.unpack('f'*int(len(a)/4), a)
    self.np_inputs = np.array(b)

  def from_model(self, model, seed):
    self.np_inputs = make_np_inputs(model, seed=seed)

  def from_json(self, model):
    a = json.loads(model)
    self.np_inputs = np.array(a)

  def python(self):
    return list(flatten(json.loads(self.json())))


class BaseModel:

  def benchmark(self, input, count=1, threads=0, device="cpu"):
    st = time.time()
    self.run(input, count=count, threads=threads, device=device)
    return time.time() - st


class OnnxModel(BaseModel):

  def __init__(self, model_path):
    self.model_path = model_path
    self.model = onnx.load(model_path)
    self.inputs = ModelData(onnx=make_np_inputs(self.model))
    self.output_names = [out.name for out in self.model.graph.output]
    self.input_names = [inp.name for inp in self.model.graph.input]
    self.session = None

  def tiny(self, device=None):
    return TinyModel(self.model, device=device)

  def torch(self):
    return TorchModel(self.model)

  def run(self, input, count=1, threads=0, device="cpu"):
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    opts.log_severity_level = 3
    # device can be "CPU", "CUDA"
    provider = device.upper()+"ExecutionProvider"
    if provider not in ort.get_available_providers(): raise Exception(f"Provider {provider} not found")
    session = ort.InferenceSession(self.model.SerializeToString(), opts, [provider])
    input = {self.input_names[0]: input}
    for _ in range(count):
      output = session.run(self.output_names, input)
    return output


# converts ONNX model to Tinygrad compatible model
class TinyOnnx:

  def __init__(self, onnx_model):
    self.xname = onnx_model.graph.input[0].name
    self.yname = onnx_model.graph.output[0].name
    self.run_onnx = get_run_onnx(onnx_model)

  def forward(self, x):
    # b = {self.xname: x}
    # print(f"forward input: {b}")
    o = self.run_onnx({self.xname: x}, debug=False)[self.yname]
    # print(o)
    return o


class TinyModel(BaseModel):

  def __init__(self, model, device=None):
    #print(f"Device: {device} Default: {Device.DEFAULT}")
    if device is not None:
      Device.DEFAULT = device.upper()
    if 'onnx' in str(type(model)):
      self.model = TinyOnnx(model)
      self.inputs = ModelData(onnx=make_np_inputs(model))
    else:
      raise Exception(f"Unknown model type - {type(model)}")
    self.runner, self.special_names = jit_model(self.model, self.inputs.tiny())

  def run(self, input, count=1, threads=None, device=None):
    #print("special names: ", self.special_names)
    for _ in range(count):
      output = self.runner(input)
    #return {k:v.numpy() for k,v in output.items()}
    return [o.numpy() for o in output]


class TorchModel(BaseModel):

  def __init__(self, model):
    if 'onnx' in str(type(model)):
      self.model = onnx2torch.convert(model)
    elif 'torch' in str(type(model)):
      self.model = model
    else:
      raise Exception(f"Unknown model type - {type(model)}")

  def run(self, inputs, count=1, threads=0, device="cpu"):
    model = self.model.to(device)
    torch_inputs = inputs.to(device)
    with torch.no_grad():
      # Our compiled model is going to be single threaded, so we need to set it to single thread
      if threads > 0:
        torch.set_num_threads(threads)
      for _ in range(count):
        output = model(torch_inputs)
    return output.cpu().numpy()
