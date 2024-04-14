import os
import struct
import subprocess
from collections import namedtuple
from .generate import SrcGenerator


# Tinygrad includes
from tinygrad.runtime.ops_rust import RUST_TYPE_MAP


RustRenderedCode = namedtuple("RustRenderedCode", "net_struct_members net_struct_initializers net_weights_initialization net_run_args net_run_body weights_bytes_conversion input_bytes_conversion weights_code functions")


class RustSrc(SrcGenerator):

  def __init__(self, tinymodel, settings):
    super().__init__(tinymodel, settings)
    self.type_map = RUST_TYPE_MAP

  def render_code(self, g, input, output, weight):
    input_names = list(g.inputs.keys())
    output_names = list(g.outputs.keys())
    # Iterate through the buffers to render out various chunks
    net_struct_members = []
    net_struct_initializers = []
    net_run_args = []
    for name,(length,dtype,_) in g.bufs.items():
      dtype_name = self.type_map[dtype][0]
      count = int(length/self.type_map[dtype][1])
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
      dtype_size = self.type_map[cl.dtype][1]
      dtype_name = self.type_map[cl.dtype][0]
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

  def metadata(self,g):
    input, output, weight = self.get_variable_tuples(g)
    rendered = self.render_code(g, input, output, weight)
    metadata = {'input': input, 'output': output, 'weight': weight, 'rendered': rendered, 'settings': self.settings}
    return metadata

  def build(self):
    if self.test_path is None:
      raise Exception("No test crate path set")
    # Run cargo build
    basedir = os.getcwd()
    os.chdir(self.test_path)
    subprocess.run(["cargo", "build", "--release"])
    os.chdir(basedir)
    self.test_bin_path = os.path.join(self.test_path, "target", "release", os.path.basename(self.test_path))

