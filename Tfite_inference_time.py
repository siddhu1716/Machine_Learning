import tensorflow as tf
import time

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="/home/mcw/Desktop/Perfalign_research/models/mobilenet_v2_1.0_224_INT8.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference and capture execution time for each layer
profiling_data = []

# Run the model once to ensure everything is loaded
interpreter.invoke()

# Enable profiling (if available in your TFLite runtime)
try:
    interpreter._start_profiling()
except AttributeError:
    print("Profiling is not available in your version of the TFLite interpreter.")

# Run inference and track time for each operation
try:
    start_time = time.time()
    interpreter.invoke()
    end_time = time.time()
except Exception as e:
    print(f"Error during inference: {e}")

# Stop profiling
try:
    profile_data = interpreter._end_profiling()
except AttributeError:
    print("Profiling is not available in your version of the TFLite interpreter.")
    profile_data = []

# Extract tensor details for each layer
for i, op in enumerate(interpreter._get_ops_details()):
    # Get operator type and tensor details
    op_type = op['op_name']  # E.g., Conv2D, Relu, Add, etc.
    input_tensors = [interpreter._get_tensor_details(tensor_id)['name'] for tensor_id in op['inputs']]
    output_tensors = [interpreter._get_tensor_details(tensor_id)['name'] for tensor_id in op['outputs']]

    # Extract execution time (if profiling is enabled)
    if profile_data:
        op_exec_time = next((entry['end_time'] - entry['start_time'] 
                             for entry in profile_data if entry['index'] == i), None)
    else:
        op_exec_time = None

    # Save profiling data
    layer_info = {
        'layer_index': i,
        'layer_type': op_type,
        'input_tensors': input_tensors,
        'output_tensors': output_tensors,
        'execution_time_ms': op_exec_time if op_exec_time else "Profiling Not Available"
    }
    profiling_data.append(layer_info)

# Print the layer details
for layer in profiling_data:
    print(f"Layer Index: {layer['layer_index']}")
    print(f"Layer Type: {layer['layer_type']}")
    print(f"Input Tensors: {layer['input_tensors']}")
    print(f"Output Tensors: {layer['output_tensors']}")
    print(f"Execution Time (ms): {layer['execution_time_ms']}")
    print("-" * 50)
