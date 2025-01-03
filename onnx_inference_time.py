import onnxruntime as ort
import numpy as np
import json

# Load ONNX model
model_path = "/home/mcw/Desktop/6.Models-Test-Perfalign/onnx_models/mobilenetv2-7.onnx"

# Create an ONNX Runtime session with profiling enabled
session_options = ort.SessionOptions()
session_options.enable_profiling = True
session = ort.InferenceSession(model_path, session_options)

# Get input details
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
input_type = session.get_inputs()[0].type

# Generate random input data (adjust as needed)
input_data = np.random.rand(*[dim if dim else 1 for dim in input_shape]).astype(np.float32)

# Run inference
outputs = session.run(None, {input_name: input_data})

# Get the profiling file path
profiling_file_path = session.end_profiling()
print(f"Profiling data saved to: {profiling_file_path}")

# Parse the profiling data
with open(profiling_file_path, "r") as f:
    profiling_data = json.load(f)

# Display layer-wise profiling information
print("Layer-wise profiling information:")
for event in profiling_data:
    if "args" in event and "op_name" in event["args"]:
        op_name = event["args"]["op_name"]
        duration = event["dur"]  # Duration in microseconds
        print(f"Operator: {op_name}, Time: {duration} µs")