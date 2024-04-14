#!/usr/bin/env python3
import os
import numpy as np
import onnxruntime as rt
import cv2
import json
import onnxruntime as rt
from pandas.core.common import flatten



# set image file dimensions to 224x224 by resizing and cropping image from center
def pre_process_edgetpu(img, dims):
    output_height, output_width, _ = dims
    img = resize_with_aspectratio(img, output_height, output_width, inter_pol=cv2.INTER_LINEAR)
    img = center_crop(img, output_height, output_width)
    img = np.asarray(img, dtype='float32')
    # converts jpg pixel value from [0 - 255] to float array [-1.0 - 1.0]
    img -= [127.0, 127.0, 127.0]
    img /= [128.0, 128.0, 128.0]
    return img

# resize the image with a proportional scale
def resize_with_aspectratio(img, out_height, out_width, scale=87.5, inter_pol=cv2.INTER_LINEAR):
    height, width, _ = img.shape
    new_height = int(100. * out_height / scale)
    new_width = int(100. * out_width / scale)
    if height > width:
        w = new_width
        h = int(new_height * height / width)
    else:
        h = new_height
        w = int(new_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)
    return img

# crop the image around the center based on given height and width
def center_crop(img, out_height, out_width):
    height, width, _ = img.shape
    left = int((width - out_width) / 2)
    right = int((width + out_width) / 2)
    top = int((height - out_height) / 2)
    bottom = int((height + out_height) / 2)
    img = img[top:bottom, left:right]
    return img


sess = rt.InferenceSession("tmp/efficientnet-lite4-11.onnx")

def inference(img_batch):
  results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
  result = reversed(results[0].argsort()[-5:])
  resultdic = {}
  for r in result:
      resultdic[labels[str(r)]] = float(results[0][r])
  return resultdic


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# load the labels text file
labels = json.load(open("tmp/labels_map.txt", "r"))

img = cv2.imread("tmp/catonnx.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = pre_process_edgetpu(img, (224, 224, 3))

img_batch = np.expand_dims(img, axis=0)

results = sess.run(["Softmax:0"], {"images:0": img_batch})[0]
result = reversed(results[0].argsort()[-5:])
resultdic = {}
for r in result:
  resultdic[labels[str(r)]] = float(results[0][r])
print(resultdic)

with open("catonnx.json", "w") as f:
  a = json.loads(json.dumps(img_batch, cls=NumpyEncoder))
  c = json.dumps(list(flatten(a)))
  f.write(c)

import subprocess

subprocess.run(["export/model_test/target/debug/model_test", "catonnx.json"]) 

with open("output.json", "r") as f:
  results = json.load(f)

result = reversed(results[0].argsort()[-5:])
resultdic = {}
for r in result:
  resultdic[labels[str(r)]] = float(results[0][r])
print(resultdic)
