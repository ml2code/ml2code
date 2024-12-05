from collections import namedtuple
import jinja2
import os
import time
import yaml
import subprocess

# Tinygrad includes
from extra.export_model import compile_net
#from tinygrad.nn.state import get_state_dict


GeneratorFunctions = namedtuple("GeneratorFunctions", "functions statements bufs bufs_to_save inputs outputs")
DTypeTuple = namedtuple("DTypeTuple", "name size")
IOTuple = namedtuple("IOTuple", "name size type")


class SrcGenerator:

  def __init__(self, tinymodel, settings):
    self.model = tinymodel.model
    self.settings = settings
    self.runner = tinymodel.runner
    self.special_names = tinymodel.special_names

  def get_variable_tuples(self, g):
    input_name = list(g.inputs.keys())[0]
    output_name = list(g.outputs.keys())[0]
    input_type = self.type_map[g.bufs[input_name][1]]
    output_type = self.type_map[g.bufs[output_name][1]]
    input_len = int(g.inputs[input_name]//input_type[1])
    output_len = int(g.outputs[output_name]//output_type[1])
    weight_len = 0
    for _,cl in g.bufs_to_save.items():
      weight_len += cl.size
    input = IOTuple(input_name, input_len, DTypeTuple(input_type[0],input_type[1]))
    output = IOTuple(output_name, output_len, DTypeTuple(output_type[0],output_type[1]))
    weight = IOTuple('weight', weight_len, DTypeTuple(input_type[0],input_type[1])) # HACK assume input type for weights
    return input, output, weight

  def metadata(self,g):
    input, output, weight = self.get_variable_tuples(g)
    rendered = self.render_code(g, input, output, weight)
    metadata = {'input': input, 'output': output, 'weight': weight, 'rendered': rendered, 'settings': self.settings}
    return metadata

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

  def generate_file(self, dir_name, module_name, file, g, metadata):
    # Check if we should skip this file
    if file.get("options", None) is not None:
      for k,v in file["options"].items():
        s = self.settings.get(k, None)
        if s is None: continue
        if s != v: return
    metadata['dir_name'] = dir_name
    metadata['crate_name'] = dir_name # for Rust
    template = os.path.join(self.settings["template_dir"], module_name, file["template"])
    output_path = os.path.join(self.settings["export_dir"], dir_name, file["path"])
    # create the directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    # Render the code
    rendered_file = self.render_jinja(template, metadata)
    # Write the code to the file
    with open(output_path, "w") as f:
      f.write(rendered_file)

  def generate(self):
    g = self.generate_functions()
    metadata = self.metadata(g)
    self.metadata = metadata

    mapfilepath = os.path.join(self.settings["template_dir"],"map.yml")
    if not os.path.exists(mapfilepath):
      raise Exception(f"{mapfilepath} not found in template directory")
    with open(mapfilepath, "r") as f:
      filemap = yaml.load(f, Loader=yaml.FullLoader)

    for module_name, module in filemap.get("modules", {}).items():
      dir_name = self.render_jinja(module['name'], metadata)
      if module_name == "test":
        self.test_path = os.path.join(self.settings["export_dir"], dir_name)
      for file in module.get("files", []):
        template_path = os.path.join(self.settings["template_dir"], module_name, file['template'])
        if not os.path.exists(template_path):
          raise Exception(f"{template_path} from {mapfilepath} not found in template directory")
        self.generate_file(dir_name, module_name, file, g, metadata)

  def benchmark(self, input, count=1):
    if not os.path.exists("tmp"):
      os.mkdir("tmp")
    input_path = os.path.join(os.getcwd(),"tmp","input.bin")
    with open(input_path, "wb") as f:
      f.write(input.binary())
    st = time.time()
    self.run(input_path, count=count)
    return time.time() - st

  def run(self, input_path, count=1):
    cmd = [self.test_bin_path, input_path]
    if self.weights_filename is not None: cmd.append(self.weights_filename)
    if not os.path.exists("tmp"):
      os.mkdir("tmp")
    output_path = os.path.join(os.getcwd(),"tmp","output.bin")
    cmd.append(output_path)
    cmd.append(str(count))
    subprocess.run(cmd)
    return output_path
