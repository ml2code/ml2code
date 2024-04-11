
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


#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    FILE *file;
    float *array;
    int numFloats;

    // Check if the filename is provided as a command-line argument
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <filename>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *filename = argv[1]; // Take the filename from command line arguments

    // Open the file in binary read mode
    file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return EXIT_FAILURE;
    }

    // Move to the end of the file to determine its size
    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    rewind(file); // Go back to the start of the file

    // Calculate the number of float elements in the file
    numFloats = fileSize / sizeof(float);

    // Allocate memory for the array of floats
    array = (float *) malloc(numFloats * sizeof(float));
    if (array == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Read the file into the array
    size_t readCount = fread(array, sizeof(float), numFloats, file);
    if (readCount != numFloats) {
        fprintf(stderr, "Failed to read the complete file\n");
        free(array);
        fclose(file);
        return EXIT_FAILURE;
    }

    // Close the file
    fclose(file);

    // Example usage: print the array's contents
    for (int i = 0; i < numFloats; i++) {
        printf("%f\n", array[i]);
    }

    // Free the allocated memory
    free(array);

    return EXIT_SUCCESS;
}
