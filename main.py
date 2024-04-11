#!/usr/bin/env python3
#from tinygrad.extras import export_model
import ml2code
from ml2code.models import OnnxModel
from ml2code.generate import RustSrc
import argparse


print("Hello world")
#om = OnnxModel("tmp/efficientnet-lite4-11.onnx")
om = OnnxModel("tmp/model.onnx")
i = om.modelinputs.torch()
# o = om.run(i)
# print(o)
print("----tiny----------------")
t = om.tiny()
i = t.inputs
o = t.run(i)
print(o)

print("----generate----------------")
c = RustSrc(t)
print(c.generate())
#t.export()
# print("----torch----------------")
# c = om.torch()
# i = c.inputs
# o = c.run(i)
# print(o)