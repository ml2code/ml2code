# DEPRECATED, moved to bitflux-ai org


# ml2code
Converts onnx ML models to standalone Rust and C libraries

# Scope
We use this in production, but note we only do it for a class of very tiny models (KByte range).  And we do test it with a few in the MByte range.  I don't think this makes sense for anything in the GByte range.  It is not intended to compete with dedicated CPU LLM projects.

We are not doing any highlevel optimization like threading because we don't, but are planning on doing the best lowlevel optimizations we can without going to intrinsics or assembly (idea it to have this handled by tinygrad).

# Install

Download the code from github and pull the submodules.

```bash
git clone --recurse-submodules https://github.com/ml2code/ml2code.git
```

You can install using uv or some other python package manager.
```bash
uv sync
uv pip install -e .
source .venv/bin/activate
```
At this point you can run the locally installed ml2code command.
```bash
ml2code --help
```
Alternatively you can just use the `uv run` command as a no install option.

```bash
uv run ml2code --help
```


# Usage

## Creating a Rust library
```bash
ml2code --model ./tmp/efficientnet-lite4-11.onnx --language rust --nobuild
```
Will create the following output.  ./export/model is the model library and the ./export/model_test is a test program that uses the library.
```
.
└── export
   ├── model
   │  ├── Cargo.toml
   │  ├── model.h
   │  ├── src
   │  │  ├── lib.rs
   │  │  ├── model.rs
   │  │  └── weights.rs
   │  └── weights.h
   └── model_test
      ├── Cargo.toml
      ├── main.c
      └── src
         └── main.rs
```

## Creating a C library
```bash
ml2code --model ./tmp/efficientnet-lite4-11.onnx --language clang
```

Note we left off the --nobuild flag, the .h library in ./export/model/ builds against a simple test program in ./export/model_test/.  The result is an executable ./export/model_test/model_test that can be run to test or benchmark the library.

Mostly this build is to make sure the library builds cleanly which is why the build is the default behavior.

```
.
├── export
   ├── model
   │  ├── model.h
   │  └── weights.h
   └── model_test
      ├── main.c
      └── model_test
```
## Benchmarking
We have a very simple benchmark built into the tool.  It require CUDA support and isn't very good.  Yet it can show some interesting patterns.

To run the benchmark you need to have a CUDA enabled GPU and the CUDA toolkit installed.  Then you can run the benchmark with the following command.
```bash
ml2code --model ./tmp/efficientnet-lite4-11.onnx --benchmark --count 100
```
Adjust --count until you get decent results.

## test
This test is intended to be a sanity check that the code generates the same results as the original model.

NOTE: Running this on efficientnet-lite4-11.onnx gives an error.  I think this is signal that there is some problem in underlying libraries here.  It passes for the simple models I use in production.  Not sure what to make of this yet.  Concerning that torch is the outlier.
```bash
ml2code --model ./tmp/efficientnet-lite4-11.onnx --test
```

