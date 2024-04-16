#!/usr/bin/env python3
#from tinygrad.extras import export_model
import os
import argparse
from ml2code.models import OnnxModel, ModelData, compare_lists, set_tinygrad_device
from ml2code.rust import RustSrc
from ml2code.clang import ClangSrc


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to model file")
parser.add_argument("--language", type=str, default="rust", choices=["rust", "clang"], help="Language to generate")
parser.add_argument("--test", action="store_true", help="Test the tinygrad conversions")
parser.add_argument("--benchmark", action="store_true", help="Benchmark the models")
parser.add_argument("--nobuild", action="store_true", help="Don't build the generated code to a binary")
parser.add_argument("--count", type=int, default=1, help="How many times to run the test")
parser.add_argument("--version", type=str, default="0.1.0", help="Version number for the code")
parser.add_argument("--name", type=str, default="model", help="Name of the model")
parser.add_argument("--author", type=str, default="<NAME> <<EMAIL>>", help="Author of the model")
parser.add_argument("--export_dir", type=str, default="export", help="Where do we put the generated src code")
parser.add_argument("--template_dir", type=str, default="templates", help="Where do we put the templates")
parser.add_argument("--noweights", action="store_true", help="Don't Encode weights in the generated src code")

args = parser.parse_args()

settings = {
  "template_dir":os.path.join(args.template_dir, args.language),
  "export_dir":args.export_dir, "noweights":args.noweights, "test":args.test,
  "author":args.author, "version":args.version, "model_name":args.name,
  "language":args.language, "model_file":args.model
}

set_tinygrad_device(args.language.upper())

outputs = {}
#om = OnnxModel("tmp/efficientnet-lite4-11.onnx")
om = OnnxModel(settings["model_file"])
i = om.inputs
t = om.tiny()
tc = om.torch()

print("----generate----------------")
if args.language == "rust":
  c = RustSrc(t, settings)
elif args.language == "clang":
  c = ClangSrc(t, settings)
else:
  raise Exception("Unknown language: " + args.language)
c.generate()
if (args.test or args.benchmark) and not args.nobuild:
  print("----build----------------")
  c.build()
if args.test:
  print("----test----------------")
  outputs['onnx'] = om.run(i.onnx())
  outputs['torch'] = tc.run(i.torch())
  outputs['tiny'] = t.run(i.tiny())
  with open("input.bin", "wb") as f:
    f.write(i.binary())
  of = c.run("input.bin")
  outputs['compiled'] = ModelData(binary=of).onnx()

  failed = False
  e = ModelData(onnx=outputs['onnx']).python()
  for k,v in outputs.items():
      if k == "onnx": continue
      a = ModelData(onnx=v).python()
      if not compare_lists('onnx', e, k, a): failed = True
  print("Passed: ", not failed)

if args.benchmark:
  outputs = {}
  print("----benchmark----------------")
  outputs['onnx cpu'] = om.benchmark(i.onnx(), count=args.count, device="cpu")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  # Installing the CUDAExecutionProvider wasn't trivial. This install worked for me:
  #  pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
  outputs['onnx gpu'] = om.benchmark(i.onnx(), count=args.count, device="cuda")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  outputs['onnx cpu singlecore'] = om.benchmark(i.onnx(), count=args.count, threads=1, device="cpu")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  outputs['torch cpu'] = tc.benchmark(i.torch(), count=args.count, device="cpu")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  outputs['torch gpu'] = tc.benchmark(i.torch(), count=args.count, device="cuda")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  outputs['torch cpu singlecore'] = tc.benchmark(i.torch(), count=args.count, threads=1, device="cpu")
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  outputs[f"tiny {args.language} jit"] = t.benchmark(i.tiny(), count=args.count)
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  tg = om.tiny(device="CUDA")
  outputs[f"tiny CUDA jit"] = tg.benchmark(i.tiny(), count=args.count)
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")
  del(tg) # oddly needs this to be deleted
  set_tinygrad_device(args.language.upper())
  outputs[f"tiny {args.language} compiled"] = c.benchmark(i, count=args.count)
  print(f"{list(outputs.keys())[-1]}: {outputs[list(outputs.keys())[-1]]:.2f}s")

if not args.nobuild:
  print(f"binary at: {c.test_bin_path}")