#!/usr/bin/env python3
import os
import numpy as np
import onnxruntime as rt
import cv2
import json
import onnxruntime as rt
import struct
import subprocess


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

def decode_predictions(results, labels, top=1):
  result = reversed(results.argsort()[-top:])
  resultdic = {}
  for r in result:
    resultdic[labels[str(r)]] = float(results[r])
  return resultdic


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--language", type=str, default="rust", choices=["rust", "clang"], help="Language to generate")
  parser.add_argument("--image", type=str, help="jpg file to test with")

  args = parser.parse_args()

  dataset = {
    "tmp/efficientnet-lite4-11.onnx": "https://github.com/onnx/models/raw/main/validated/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
    "tmp/labels_map.txt": "https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/efficientnet-lite4/dependencies/labels_map.txt",
    "tmp/cat.jpg": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Siam_lilacpoint.jpg/236px-Siam_lilacpoint.jpg"
  }

  # download the dataset
  if not os.path.exists("tmp"):
    os.mkdir("tmp")
  for k, v in dataset.items():
    if not os.path.exists(k):
      os.system(f"wget {v} -O {k}")

  # load the labels text file
  labels = json.load(open("tmp/labels_map.txt", "r"))

  # load and prep the image
  if args.image is not None:
    imgfile = args.image
  else:
    imgfile = "tmp/cat.jpg"
  imgorig = cv2.imread(imgfile)
  img = cv2.cvtColor(imgorig, cv2.COLOR_BGR2RGB)
  img = pre_process_edgetpu(img, (224, 224, 3))
  img_batch = np.expand_dims(img, axis=0)
  # convert the image to binary format for the compiled model to read
  with open("tmp/image.bin", "wb") as f:
    f.write(img_batch.tobytes())

  # show the image
  print("Categorizing this image: ", imgfile)
  cv2.imshow(f"Sample Picture",imgorig)
  cv2.waitKey(3000)
  cv2.destroyAllWindows()

  # generate the model code
  print("Generating model code and compiling it")
  out = subprocess.run(["uv", "run", "ml2code", "--model", "tmp/efficientnet-lite4-11.onnx", "--language", args.language], capture_output=True, text=True)
  if out.returncode != 0:
    print("Error generating model code")
    print(out.stderr)
    #exit(1)
  binfile = out.stdout.split("\n")[-2]
  binfile = binfile.split(": ")[1]
  print(f"Built model as: {binfile}")

  # run the model with onnxruntime
  sess = rt.InferenceSession("tmp/efficientnet-lite4-11.onnx")
  results = sess.run(["Softmax:0"], {"images:0": img_batch})[0][0]
  pred = decode_predictions(results, labels)
  print(f"The onnx model is {(list(pred.values())[0]*100):.2f}% confident the image is a '{list(pred.keys())[0]}'")

  subprocess.run([binfile, "tmp/image.bin", "tmp/output.bin"]) 

  with open("tmp/output.bin", "rb") as f:
    a = f.read()
    results = np.array(list(struct.unpack('f'*int(len(a)/4), a)))

  pred = decode_predictions(results, labels)
  print(f"The compiled model is {(list(pred.values())[0]*100):.2f}% confident the image is a '{list(pred.keys())[0]}'")
