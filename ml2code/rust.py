import os
import struct
import yaml
import time
import subprocess
from collections import namedtuple
from .generate import SrcGenerator


# Tinygrad includes
from tinygrad.runtime.ops_rust import RUST_TYPE_MAP


RustRenderedCode = namedtuple("RustRenderedCode", "net_struct_members net_struct_initializers net_weights_initialization net_run_args net_run_body weights_bytes_conversion input_bytes_conversion weights_code functions")
DTypeTuple = namedtuple("DTypeTuple", "name size")
IOTuple = namedtuple("IOTuple", "name size type")


class RustSrc(SrcGenerator):

  def render_code(self, g, input, output, weight):
    input_names = list(g.inputs.keys())
    output_names = list(g.outputs.keys())
    # Iterate through the buffers to render out various chunks
    net_struct_members = []
    net_struct_initializers = []
    net_run_args = []
    for name,(length,dtype,_) in g.bufs.items():
      dtype_name = RUST_TYPE_MAP[dtype][0]
      count = int(length/RUST_TYPE_MAP[dtype][1])
      # Handle the commandline arg for the run() function
      if name in input_names:
        net_run_args.append(f"{name}: &[{dtype_name}; {count}]")
        continue
      if name in output_names:
        net_run_args.append(f"{name}: &mut [{dtype_name}; {count}]")
        continue
      # Construct a list out the buffers for the rust struct
      net_struct_members.append(f"{' '*2}{name}: Box<[{dtype_name}; {count}]>,")
      # Construct a list of initializers for the rust struct,
      #  either zero or consts with encoded weights
      if not self.settings['noweights'] and name in list(g.bufs_to_save.keys()):
        line = f"{' '*6}{name}: Box::new({name.upper()}_DATA),"
      else:
        line = f"{' '*6}{name}: Box::new([0.0; {count}]),"
      net_struct_initializers.append(line)
    net_struct_members = "\n".join(net_struct_members)
    net_struct_initializers = "\n".join(net_struct_initializers)
    net_run_args = ", ".join(net_run_args)
    # Iterate through the bufs_to_save to render the weight initialization code, and the consts for the weights
    net_weights_initialization = []
    weights_code = []
    weights = bytes()
    for name,cl in g.bufs_to_save.items():
      dtype_size = RUST_TYPE_MAP[cl.dtype][1]
      dtype_name = RUST_TYPE_MAP[cl.dtype][0]
      start = int(len(weights)/dtype_size)
      # Construct the code to initialize the weights
      net_weights_initialization.append(f"{' '*4}self.{name}.copy_from_slice(&weights[{start}..{start+cl.size}]);")
      weight_buf = bytes(cl._buf)
      # Encode the weights
      wbytes = [str(struct.unpack('f', weight_buf[i:i+4])[0]) for i in range(0, len(weight_buf), dtype_size)]
      weights_code.append(f"pub const {name.upper()}_DATA: [{dtype_name}; {cl.size}] = [{','.join(wbytes)}];")
      weights += weight_buf
    # Writes the weights to disk if they aren't encoded
    if self.settings['noweights']:
      self.weights_filename = os.path.join(self.settings['export_dir'], "weights.bin")
      with open(self.weights_filename, "wb") as f:
        f.write(weights)
    else:
      self.weights_filename = None
    net_weights_initialization = "\n".join(net_weights_initialization)
    weights_code = "\n".join(weights_code)
    # Construct the body of the run function
    net_run_body = []
    statement_names = [name for (name, args, _, _) in g.statements]
    for (name, args, _, _) in g.statements:
      fixed_name = name.lower()
      if name != fixed_name and fixed_name in statement_names:
        raise Exception(f"Fixed version of {name} '{fixed_name}' is already used")
      params = ['&self.'+arg if arg not in input_names+output_names else arg for arg in args]
      params[0] = params[0].replace('&self.', '&mut self.') # first arg is mutable
      net_run_body.append(f"{' '*4}{fixed_name}({', '.join(params)});")
    net_run_body = "\n".join(net_run_body)
    # Construct a little bit of code to convert the weights from bytes
    weights_bytes_conversion = []
    for i in range(weight.type.size):
      weights_bytes_conversion.append(f"weights_bytes[i*4+{str(i)}]")
    weights_bytes_conversion = ", ".join(weights_bytes_conversion)
    # Construct a little bit of code to convert the input from bytes
    input_bytes_conversion = []
    for i in range(input.type.size):
      input_bytes_conversion.append(f"input_bytes[i*4+{str(i)}]")
    input_bytes_conversion = ", ".join(input_bytes_conversion)
    # Clean up the functions
    functions = []
    for k,fn in g.functions.items():
      # clean out the CDLL stuff
      fn = fn.replace("#[no_mangle]\n", "")
      fn = fn.replace("extern \"C\" ", "")
      fn = fn.replace(k, k.lower())
      functions.append(fn)
    functions = "\n\n".join(functions)
    return RustRenderedCode(net_struct_members, net_struct_initializers, net_weights_initialization, net_run_args, net_run_body, weights_bytes_conversion, input_bytes_conversion, weights_code, functions)

  def get_variable_tuples(self, g):
    input_name = list(g.inputs.keys())[0]
    output_name = list(g.outputs.keys())[0]
    input_type = RUST_TYPE_MAP[g.bufs[input_name][1]]
    output_type = RUST_TYPE_MAP[g.bufs[output_name][1]]
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

  def generate_file(self, crate_name, module_name, file, g, metadata):
    # Check if we should skip this file
    if file.get("options", None) is not None:
      for k,v in file["options"].items():
        s = self.settings.get(k, None)
        if s is None: continue
        if s != v: return
    metadata['crate_name'] = crate_name
    template = os.path.join(self.settings["template_dir"], module_name, file["template"])
    output_path = os.path.join(self.settings["export_dir"], crate_name, file["path"])
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
      crate_name = self.render_jinja(module['name'], metadata)
      if module_name == "test":
        self.test_crate_path = os.path.join(self.settings["export_dir"], crate_name)
      for file in module.get("files", []):
        template_path = os.path.join(self.settings["template_dir"], module_name, file['template'])
        if not os.path.exists(template_path):
          raise Exception(f"{template_path} from {mapfilepath} not found in template directory")
        self.generate_file(crate_name, module_name, file, g, metadata)

  def build(self):
    if self.test_crate_path is None:
      raise Exception("No test crate path set")
    # Run cargo build
    basedir = os.getcwd()
    os.chdir(self.test_crate_path)
    subprocess.run(["cargo", "build", "--release"])
    os.chdir(basedir)
    self.test_bin_path = os.path.join(self.test_crate_path, "target", "release", os.path.basename(self.test_crate_path))

  def run(self, input_path, count=1):
    cmd = [self.test_bin_path, input_path]
    if self.weights_filename is not None: cmd.append(self.weights_filename)
    output_path = os.path.join(os.getcwd(),"output.bin")
    cmd.append(output_path)
    cmd.append(str(count))
    subprocess.run(cmd)
    return output_path
