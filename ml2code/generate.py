from collections import namedtuple
import jinja2
import os
import time

# Tinygrad includes
from extra.export_model import compile_net
#from tinygrad.nn.state import get_state_dict

GeneratorFunctions = namedtuple("GeneratorFunctions", "functions statements bufs bufs_to_save inputs outputs")

class SrcGenerator:
  def __init__(self, tinymodel, settings):
    self.model = tinymodel.model
    self.settings = settings
    self.runner = tinymodel.runner
    self.special_names = tinymodel.special_names

  def generate_functions(self):
    functions, statements, bufs, bufs_to_save = compile_net(self.runner, self.special_names)
    #state = get_state_dict(self.model)
    input_names = []
    output_names = []
    for _,name in self.special_names.items():
      if "input" in name: input_names.append(name)
      if "output" in name: output_names.append(name)
    inputs = {input:bufs[input][0] for input in input_names}
    outputs = {output:bufs[output][0] for output in output_names}
    return GeneratorFunctions(functions, statements, bufs, bufs_to_save, inputs, outputs)

  def render_jinja(self, template, metadata):
    if os.path.exists(template):
      with open(template, 'r') as f:
        j2 = f.read()
    else:
      j2 = template
    template = jinja2.Environment(loader=jinja2.BaseLoader).from_string(j2)
    output = template.render(metadata)
    return output

  def benchmark(self, input, count=1):
    input_path = os.path.join(os.getcwd(),"output.bin")
    with open(input_path, "wb") as f:
      f.write(input.binary())
    st = time.time()
    self.run(input_path, count=count)
    return time.time() - st
