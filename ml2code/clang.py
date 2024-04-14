import os
import struct
import yaml
import time
import subprocess
from collections import namedtuple
from .generate import SrcGenerator
from tinygrad.dtype import dtypes
from tinygrad.runtime.ops_clang import CLANG_PROG_HEADER

CLANG_TYPE_MAP = {dtypes.long:["long",8], dtypes.ulong:["unsigned long",8], dtypes.float64:["double",8], dtypes.double:["double",8],
                 dtypes.int:["int",4], dtypes.uint32:["unsigned int",4], dtypes.int32:["int",4], dtypes.float:["float",4],
                 dtypes.int16:["short",2], dtypes.uint16:["unsigned short",2], dtypes.short:["short",2], dtypes.ushort:["unsigned short",2],
                 dtypes.int8:["char",1], dtypes.uint8:["unsigned char",1], dtypes.char:["char",1], dtypes.uchar:["unsigned char",1], dtypes.bool:["bool",1]}


ClangRenderedCode = namedtuple("ClangRenderedCode", "buffer_initializers, weights_initialization, net_run_args, net_run_body, weights_code, functions")

class ClangSrc(SrcGenerator):

  def __init__(self, tinymodel, settings):
    super().__init__(tinymodel, settings)
    self.type_map = CLANG_TYPE_MAP

  def render_code(self, g, input, output, weight):
    input_names = list(g.inputs.keys())
    output_names = list(g.outputs.keys())
    # Iterate through the buffers to render out various chunks
    buffer_initializers = []
    net_run_args = []
    for name,(length,dtype,_) in g.bufs.items():
      dtype_name = self.type_map[dtype][0]
      count = int(length/self.type_map[dtype][1])
      # Handle the commandline arg for the run() function
      if name in input_names+output_names:
        net_run_args.append(f"{dtype_name}* {name}")
        continue
      # Construct a list of initializers for the rust struct,
      #  either zero or consts with encoded weights
      if not self.settings['noweights'] and name in list(g.bufs_to_save.keys()):
        line = f"{dtype_name} *{name} = (float *){name.upper()}_DATA;"
      else:
        line = f"{dtype_name} {name}[{count}];"
      buffer_initializers.append(line)
    buffer_initializers = "\n".join(buffer_initializers)
    net_run_args = ", ".join(net_run_args)
    # Iterate through the bufs_to_save to render the weight initialization code, and the consts for the weights
    weights_initialization = []
    weights_code = []
    weights = bytes()
    for name,cl in g.bufs_to_save.items():
      dtype_size = self.type_map[cl.dtype][1]
      dtype_name = self.type_map[cl.dtype][0]
      start = int(len(weights)/dtype_size)
      # Construct the code to initialize the weights
      weights_initialization.append(f"{' '*2}memcpy({name}, weights + {start}, {cl.size*dtype_size});")
      weight_buf = bytes(cl._buf)
      # Encode the weights
      wbytes = ''.join(["\\x%02X"%x for x in bytes(cl._buf)])
      weights_code.append(f"const unsigned char {name.upper()}_DATA[] = \"{wbytes}\";")
      weights += weight_buf
    # Writes the weights to disk if they aren't encoded
    if self.settings['noweights']:
      self.weights_filename = os.path.join(self.settings['export_dir'], "weights.bin")
      if not os.path.exists(os.path.basename(self.weights_filename)):
        os.makedirs(os.path.dirname(self.weights_filename))
      with open(self.weights_filename, "wb") as f:
        f.write(weights)
    else:
      self.weights_filename = None
    weights_initialization = "\n".join(weights_initialization)
    weights_code = "\n".join(weights_code)
    # Construct the body of the run function
    net_run_body = []
    statement_names = [name for (name, args, _, _) in g.statements]
    for (name, args, _, _) in g.statements:
      fixed_name = name.lower()
      if name != fixed_name and fixed_name in statement_names:
        raise Exception(f"Fixed version of {name} '{fixed_name}' is already used")
      params = [arg for arg in args]
      net_run_body.append(f"{' '*2}{fixed_name}({', '.join(params)});")
    net_run_body = "\n".join(net_run_body)
    # Clean up the functions
    functions = []
    for k,fn in g.functions.items():
      fn = fn.replace(CLANG_PROG_HEADER, "static ")
      fn = fn.replace(k, k.lower())
      functions.append(fn)
    functions = "\n\n".join(functions)
    return ClangRenderedCode(buffer_initializers, weights_initialization, net_run_args, net_run_body, weights_code, functions)

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

  def build(self):
    if self.test_path is None:
      raise Exception("No test path set")
    # Run cargo build
    basedir = os.getcwd()
    os.chdir(self.test_path)
    cmd = ["clang", "-march=native", "-lm", "-O2", "-Wall", "-Werror", "-x", "c", "-o", os.path.basename(self.test_path), "main.c"]
    subprocess.run(cmd)
    os.chdir(basedir)
    self.test_bin_path = os.path.join(self.test_path, os.path.basename(self.test_path))
