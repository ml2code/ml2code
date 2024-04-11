from tinygrad.runtime.ops_rust import RUST_TYPE_MAP
import os


def export_model_rust(functions, statements, bufs, bufs_to_save, input_names, output_names, encoded_weights=True):
  type_map = RUST_TYPE_MAP
  def for_bufs(lda, name_filter=[]):
    codeblock = []
    for name,(len,dtype,_key) in bufs.items():
      if name not in input_names+output_names+name_filter: codeblock.append(lda(name,len,dtype))
    return codeblock

  # main struct definition
  rs_struct = ["pub struct Net {"] + for_bufs(lambda name,len,dtype: f"  {name}: [{type_map[dtype][0]}; {int(len/type_map[dtype][1])}],") +  ["}"]
  #   new() init fn
  if encoded_weights: l = lambda name,len,dtype: f"      {name}: {name.upper()}_DATA,"
  else: l = lambda name,len,dtype: f"      {name}: [0.0; {int(len/type_map[dtype][1])}],"
  rs_new = ["  pub fn new() -> Self {","    Net {",] + for_bufs(l) + ["    }","  }"]
  #   weights both a fn to load and optional static arrays
  rs_weights = []
  if encoded_weights:
    rs_weights += ["// Encoded Weights"]
    l = lambda name,len,dtype: f"const {name.upper()}_DATA: [{type_map[dtype][0]}; {int(len/type_map[dtype][1])}] = [0.0; {int(len/type_map[dtype][1])}];"
    rs_weights += for_bufs(l,list(bufs_to_save.keys()))
  rs_initw = ["  #[allow(dead_code)]",f"  pub fn initialize_weights(&mut self, weights: &[{type_map[bufs_to_save[list(bufs_to_save.keys())[0]].dtype][0]}]) {{"]
  weights = bytes()
  for name,cl in bufs_to_save.items():
    rs_initw += [f"    self.{name}.copy_from_slice(&weights[{len(weights)//type_map[cl.dtype][1]}..{len(weights)//type_map[cl.dtype][1]+cl.size}]);"]
    weight = bytes(cl._buf)
    if encoded_weights:
      rs_weights += [f"const {name.upper()}_DATA: [{type_map[cl.dtype][0]}; {cl.size}] = [{','.join([str(struct.unpack('f', weight[i:i+4])[0]) for i in range(0, len(weight), type_map[cl.dtype][1])])}];"]
    weights += weight
  rs_initw += ["  }"]
  #   fn to run the network
  inputs = ", ".join([f"{name}: &[{type_map[dtype][0]}; {int(len/type_map[dtype][1])}]" for name, (len, dtype, _key) in bufs.items() if name in input_names])
  outputs = ", ".join([f"{name}: &mut [{type_map[dtype][0]}; {int(len/type_map[dtype][1])}]" for name, (len, dtype, _key) in bufs.items() if name in output_names])
  rs_run = [f"  pub fn run(&mut self, {inputs}, {outputs}) {{"]
  for (name, args, _, _) in statements:
    params = ['&self.'+arg if arg not in input_names+output_names else arg for arg in args]
    params[0] = params[0].replace('&self.', '&mut self.') # first arg is mutable
    rs_run += [f"    {name}({', '.join(params)});"]
  rs_run += ["  }"]
  rs_impl = ["impl Net {"] + rs_new + [""] + rs_initw + [""] + rs_run + ["}"]
  fns = []
  for f in functions.values(): fns += [f, ""]
  rsprog = fns + rs_weights + [""] + rs_struct + [""] + rs_impl + [""]
  return '\n'.join(rsprog)


def rust_generate(functions, statements, bufs, bufs_to_save, inputs, outputs, encoded_weights=True, crate=True, model_name="model", export_dir=""):
  input_name = list(inputs.keys())[0]
  output_name = list(outputs.keys())[0]
  input_type = RUST_TYPE_MAP[bufs[input_name][1]]
  output_type = RUST_TYPE_MAP[bufs[output_name][1]]
  input_len = int(inputs[input_name]//input_type[1])
  output_len = int(outputs[output_name]//output_type[1])
  wtype = input_type

  rs_code = export_model_rust(functions, statements, bufs, bufs_to_save,  list(inputs.keys()), list(outputs.keys()), encoded_weights=True)

  # test 
  rs_main = ["use std::fs::File;","use std::io::{self, Read};",""] if not encoded_weights else ["use std::io::{self};",""]
  if not crate: rs_main += [rs_code,""]
  rs_main += ["// Simple testing setup using stdin/stdout"]
  rs_main += ["fn main() -> io::Result<()> {"]
  rs_main += ["  // Initialize network","  let mut net = Net::new();",""]
  rs_main += [f"  // Create an input buffer of {input_len} {input_type[0]}s"]
  rs_main += [f"  let mut input = [0.0; {input_len}];",f"  let mut output = [0.0; {output_len}];","  let mut line = String::new();",""]
  if not encoded_weights:
    # write the weights to disk
    weights = bytes()
    for name,cl in bufs_to_save.items(): weights += bytes(cl._buf)
    with open("/tmp/rust_weights", "wb") as f:
      f.write(weights)
    rs_main += ["  // Read weights from a file","  let mut f = File::open(\"/tmp/rust_weights\")?;","  let mut weights_bytes = Vec::new();"]
    rs_main += ["  f.read_to_end(&mut weights_bytes)?;","",f"  // Convert bytes to {wtype[0]}"]
    rs_main += [f"  let mut weights: Vec<{wtype[0]}> = Vec::with_capacity(weights_bytes.len() / {wtype[1]});"]
    rs_main += ["  // Now map the weights_bytes into weights",f"  for i in 0..(weights_bytes.len()/{wtype[1]}) {{"]
    rs_main += [f"    weights.push({wtype[0]}::from_le_bytes([{','.join(['weights_bytes[i*4+'+str(i)+']' for i in range(wtype[1])])}]));","  }",""]
    rs_main += ["  // Initialize the network with weights","  net.initialize_weights(&weights);",""]
  rs_main += ["  // Get inputs","  for i in 0..input.len() {","    io::stdin().read_line(&mut line).unwrap();"]
  rs_main += ["    input[i] = line.trim().parse::<f32>().unwrap();","    line.clear();","  }",""]
  rs_main += ["  // Run the network","  net.run(&input, &mut output);","","  // Print the output"]
  rs_main += ["  let outputstr = output.iter().map(|item| item.to_string()).collect::<Vec<_>>().join(\" \");","  print!(\"{}\", outputstr);",""]
  rs_main += ["  Ok(())","}"]

  # export the code if not a crate, just as a string
  if not crate:
    prg = '\n'.join(rs_main)
    return prg

  # Isolate weights, if encoded, so we can put them in a separate file
  weights = []
  if encoded_weights:
    rs_code_new = [[],[]]
    for line in rs_code.split("\n"):
      if len(weights) == 0 and line != "// Encoded Weights":
        rs_code_new[0].append(line)
        continue
      if line == "": rs_code_new[1].append(line)
      if len(rs_code_new[1]) == 0: weights.append(f"pub {line}" if len(weights) != 0 else line)
      else: rs_code_new[1].append(line)
    rs_code = "\n".join(["use crate::weights::{*};",""] + rs_code_new[0] + rs_code_new[1])

  ## Make the main Rust crate
  crate_path = os.path.join(export_dir,model_name)
  os.makedirs(crate_path, exist_ok=True)
  crate_src_path = os.path.join(crate_path, "src")
  os.makedirs(crate_src_path, exist_ok=True)

  # Make main crate Cargo.toml file
  cargo_toml = ["[package]",f"name = \"{model_name}\"","version = \"0.1.0\"","authors = [\"<NAME> <<EMAIL>>\"]","edition = \"2021\"",""]
  with open(os.path.join(crate_path, "Cargo.toml"), "w") as f:
    f.write('\n'.join(cargo_toml))

  # Make the src/model.rs file
  with open(os.path.join(crate_src_path, "model.rs"), "w") as f:
      f.write("#![allow(unused_mut,unused_parens)]\n"+rs_code)

  if encoded_weights:
    # Make the src/weights.rs file
    with open(os.path.join(crate_src_path, "weights.rs"), "w") as f:
      f.write('\n'.join(weights))

  # Make the src/lib.rs file
  with open(os.path.join(crate_src_path, "lib.rs"), "w") as f:
    incs = ["mod weights;","mod model;","pub use model::Net;"] if encoded_weights else ["mod model;","pub use model::Net;"]
    f.write('\n'.join(incs))

  # Make the src/main.rs file
  with open(os.path.join(crate_src_path, "main.rs"), "w") as f:
    incs = ["mod weights;","mod model;","use model::Net;"] if encoded_weights else ["mod model;","use model::Net;"]
    f.write('\n'.join(incs+rs_main))

  ## Make a second crate to test the main crate as a library
  crate2_name = f"rlib_test_{model_name}"
  crate2_path = os.path.join(export_dir,crate2_name)
  os.makedirs(crate2_path, exist_ok=True)
  crate2_src_path = os.path.join(crate2_path, "src")
  os.makedirs(crate2_src_path, exist_ok=True)

  # Make main crate Cargo.toml file
  cargo_toml = ["[package]",f"name = \"{crate2_name}\"","version = \"0.1.0\"","authors = [\"<NAME> <<EMAIL>>\"]","edition = \"2021\"",""]
  cargo_toml += ["[dependencies]",f"{model_name} = {{ path = \"../{model_name}\" }}",""]
  with open(os.path.join(crate2_path, "Cargo.toml"), "w") as f:
    f.write('\n'.join(cargo_toml))

  # Make the src/main.rs file
  with open(os.path.join(crate2_src_path, "main.rs"), "w") as f:
    f.write('\n'.join(["use model::Net;"]+rs_main))

  return (crate_path, crate2_path)

