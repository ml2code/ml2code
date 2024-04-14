import os
import struct
import yaml
import time
import subprocess
from collections import namedtuple
from .generate import SrcGenerator

class ClangSrc(SrcGenerator):


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


  def metadata(self,g):
    input, output, weight = self.get_variable_tuples(g)
    rendered = self.render_code(g, input, output, weight)
    metadata = {'input': input, 'output': output, 'weight': weight, 'rendered': rendered, 'settings': self.settings}
    return metadata

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



def export_model_clang(functions, statements, bufs, bufs_to_save, input_names, output_names, encoded_weights):
  cprog = ["#include <stdbool.h>\n#include <tgmath.h>\n#define max(x,y) ((x>y)?x:y)\n#define half __fp16\n"]

  if encoded_weights: cprog.append("// Encoded Weights")
  for name,cl in bufs_to_save.items():
    weight = ''.join(["\\x%02X"%x for x in bytes(cl._buf)])
    if encoded_weights: cprog.append(f"unsigned char {name}_data[] = \"{weight}\";")
    else: cprog.append(f"unsigned char {name}_data[{len(bytes(cl._buf))}];")

  inputs = ", ".join([f'float* {input}' for input in input_names])
  outputs = ", ".join([f'float* {output}' for output in output_names])
  cprog += [f"float {name}[{len}];" if name not in bufs_to_save else f"float *{name} = (float *){name}_data;" for name,(len,dtype,_key) in bufs.items() if name not in ['input', 'outputs']]
  cprog += list(functions.values())
  cprog += [f"void net({inputs}, {outputs}) {{"] + [f"{name}({', '.join(args)});" for (name, args, _global_size, _local_size) in statements] + ["}"]
  return '\n'.join(cprog)



def clang_generate(functions, statements, bufs, bufs_to_save, inputs, outputs, encoded_weights):
  dtype_map = {dtypes.float: ("float",4)}
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = dtype_map[bufs[input_name][1]]
  output_type = dtype_map[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  wtype = input_type

  c_code = export_model_clang(functions, statements, bufs, bufs_to_save, list(inputs.keys()), list(outputs.keys()), encoded_weights)
  cprog = ["#include <string.h>", "#include <stdio.h>", "#include <stdlib.h>"]
  cprog += [c_code, ""]

  # weights
  if not encoded_weights:
    cprog += [f"void initialize({wtype[0]} *weights) {{"]
    weights = bytes()
    for name,cl in bufs_to_save.items():
      cprog.append(f"  memcpy({name}, weights + {len(weights)//wtype[1]}, {len(cl._buf)});")
      weights += bytes(cl._buf)
    cprog += ["}", ""]
    # write the weights to disk
    with open("/tmp/clang_weights", "wb") as f:
      f.write(weights)

  output_print = ["printf(\""]
  for _ in range(output_len-1):
    output_print.append("%f ")
  output_print.append("%f\\n\", ")
  for i in range(output_len-1):
    output_print.append(f"outputs[{i}], ")
  output_print.append(f"outputs[{output_len-1}]);")
  output_print = ''.join(output_print)

  # test program
  m = []
  m += [f"int main(int argc, char *argv[]) {{"]
  cpro

  if not encoded_weights:
    cprog += ["  // read in the weights from disk","  FILE *f = fopen(\"/tmp/clang_weights\", \"rb\");"]
    cprog += [f"  {wtype[0]} *weights = ({wtype[0]} *)malloc({len(weights)});",f"  fread(weights, 1, {len(weights)}, f);"]
    cprog += ["  fclose(f);", "","  // init the net","  initialize(weights);",""]

  cprog += ["  // test run",f"  {input_type[0]} input[{input_len}];"]
  cprog += [f"  {output_type[0]} outputs[{output_len}];"]
  cprog += [f"  for (int i = 0; i < {input_len}; i++) scanf(\"%f\", &input[i]);"]
  cprog += [f"  net(input, outputs);","",f"  {output_print}", "}"]

  # ready the program
  prg = '\n'.join(cprog)
  return prg


# #include <stdio.h>
# #include <stdlib.h>

# int main(int argc, char *argv[]) {
#     FILE *file;
#     float *array;
#     int numFloats;

#     // Check if the filename is provided as a command-line argument
#     if (argc != 2) {
#         fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
#         return EXIT_FAILURE;
#     }

#     const char *filename = argv[1]; // Take the filename from command line arguments

#     // Open the file in binary read mode
#     file = fopen(filename, "rb");
#     if (file == NULL) {
#         fprintf(stderr, "Failed to open file: %s\n", filename);
#         return EXIT_FAILURE;
#     }

#     // Move to the end of the file to determine its size
#     fseek(file, 0, SEEK_END);
#     long fileSize = ftell(file);
#     rewind(file); // Go back to the start of the file

#     // Calculate the number of float elements in the file
#     numFloats = fileSize / sizeof(float);

#     // Allocate memory for the array of floats
#     array = (float *) malloc(numFloats * sizeof(float));
#     if (array == NULL) {
#         fprintf(stderr, "Memory allocation failed\n");
#         fclose(file);
#         return EXIT_FAILURE;
#     }

#     // Read the file into the array
#     size_t readCount = fread(array, sizeof(float), numFloats, file);
#     if (readCount != numFloats) {
#         fprintf(stderr, "Failed to read the complete file\n");
#         free(array);
#         fclose(file);
#         return EXIT_FAILURE;
#     }

#     // Close the file
#     fclose(file);

#     // Example usage: print the array's contents
#     for (int i = 0; i < numFloats; i++) {
#         printf("%f\n", array[i]);
#     }

#     // Free the allocated memory
#     free(array);

#     return EXIT_SUCCESS;
# }
