import onnxruntime as ort
import os

# Define model path and desired optimized model path
model_path = "/home/mcw/Desktop/6.Models-Test-Perfalign/onnx_models/mobilenet_v3.onnx"

# Define the directory where you want to store the optimized model
output_dir = "/home/mcw/Desktop/Perfalign_research/optimized_models"
os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

# Generate the optimized model filename (optional: you can customize it)
optimized_model_name = "mobilenetv3_optimized.onnx"
optimized_model_path = os.path.join(output_dir, optimized_model_name)

# Set ONNX runtime options
options = ort.SessionOptions()
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
options.optimized_model_filepath = optimized_model_path

# Create the inference session
ort_session = ort.InferenceSession(model_path, options)

# Check if the file was created and print the path
if os.path.exists(optimized_model_path):
    print(f"Optimized model was created at: {optimized_model_path}")
else:
    print(f"Failed to create optimized model at: {optimized_model_path}")
