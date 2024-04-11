from extra.export_model import compile_net
#from tinygrad.nn.state import get_state_dict
from .rust import rust_generate

class SrcGenerator:
  def __init__(self, tinymodel):
    self.model = tinymodel.model
    self.runner = tinymodel.runner
    self.special_names = tinymodel.special_names

  def generate_functions(self):
    functions, statements, bufs, bufs_to_save = compile_net(self.runner, self.special_names)
    #state = get_state_dict(self.model)
    input_names = []
    output_names = []
    print("special names: ", self.special_names)
    for _,name in self.special_names.items():
      if "input" in name: input_names.append(name)
      if "output" in name: output_names.append(name)
    inputs = {input:bufs[input][0] for input in input_names}
    outputs = {output:bufs[output][0] for output in output_names}
    return functions, statements, bufs, bufs_to_save, inputs, outputs


class RustSrc(SrcGenerator):
  def generate(self, encoded_weights=True, model_name="model", export_dir="export"):
    functions, statements, bufs, bufs_to_save, inputs, outputs = self.generate_functions()
    print(f"input: {inputs}")
    print(f"types: {type(inputs)}")
    print(f"output: {outputs}")
    print(f"types: {type(outputs)}")
    print(f"bufs: {bufs.keys()}")
    print(f"bufs_to_save: {bufs_to_save.keys()}")
    return rust_generate(functions, statements, bufs, bufs_to_save, inputs, outputs, encoded_weights=encoded_weights, model_name=model_name, export_dir=export_dir)

