#!/bin/bash

# NOTE: tinygrad is compatible with python 3.8 as a minimum.
#  You can use python 3.8 as shown here if you have problems
# # Create a virtual environment with 3.8
# virtualenv -p /usr/bin/python3.8 venv
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
