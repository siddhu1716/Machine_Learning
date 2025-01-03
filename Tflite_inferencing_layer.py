import tensorflow as tf
import numpy as np
import json

# Load the TFLite model
model_path = "/home/mcw/Desktop/Perfalign_research/conv_add_relu_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path,experimental_preserve_all_tensors=True)

# Allocate tensors (this must be done before accessing tensor data)
interpreter.allocate_tensors()

# Get tensor details for all layers
tensor_details = interpreter.get_tensor_details()

# Get input details and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print input and output details
# print(f"Input Details: {input_details}")
# print(f"Output Details: {output_details}")

# Get the input shape and generate random input data
input_index = input_details[0]['index']
input_shape = input_details[0]['shape']
a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float32)
print(a_INPUT)

# Set the input tensor
interpreter.set_tensor(input_index, a_INPUT)

# Run inference
interpreter.invoke()

# Create a dictionary to store layer-wise outputs
layer_outputs = {}
key_value={}
# Loop through all tensors (including intermediate layers)
print(tensor_details[0])
print(tensor_details[1])
print(tensor_details[2])
for tensor in tensor_details:
    tensor_name = tensor['name']
    tensor_index = tensor['index']
    # Access tensor data
    tensor_data = interpreter.get_tensor(tensor_index)
    # Store the output data (convert numpy array to list for JSON serialization)
    layer_outputs[tensor_name] = tensor_data.tolist()
    key_value[tensor_index]=tensor_name


# Save layer outputs to a JSON file
with open('tflite_layer_outputs.json', 'w') as json_file:
    json.dump(layer_outputs, json_file)

# Print the shapes of all layers' outputs
ls=[]
for layer_name, layer_output in layer_outputs.items():
    ls.append(layer_name)
    print(f"Shape of output for layer '{layer_name}': {np.array(layer_output).shape}")