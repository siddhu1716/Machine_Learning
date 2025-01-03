import onnx
import onnxruntime as ort
import numpy as np

def load_model(model_path):
    """Load and check an ONNX model."""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    return model

def infer_layer_by_layer(model_path):
    """Perform layer-by-layer inference with outputs fed as inputs to the next layer."""
    model = load_model(model_path)
    session = ort.InferenceSession(model_path)
    layer_outputs = [x.name for x in session.get_outputs()]
    
    # Initial input setup
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_data = np.random.randn(*[dim if dim is not None else 1 for dim in input_shape]).astype(np.float32)
    layer_outputs[input_name] = input_data
    
    print("Running layer-by-layer inference with dynamic input feeding...")

    for node in model.graph.node:
        try:
            # Get inputs for the current node, retrieving them from layer_outputs
            node_inputs = [layer_outputs[input_name] for input_name in node.input if input_name in layer_outputs]
            
            if len(node_inputs) < len(node.input):  # Skip layer if inputs are missing
                print(f"Skipping layer {node.name} ({node.op_type}) due to missing inputs.")
                continue
            
            # Run the current node with available inputs
            ort_inputs = {input_name: node_inputs[i] for i, input_name in enumerate(node.input)}
            node_outputs = session.run(node.output, ort_inputs)
            
            # Store outputs to use as inputs for the next layer
            for i, output_name in enumerate(node.output):
                layer_outputs[output_name] = node_outputs[i]
                
            print(f"Processed layer {node.name}, op_type: {node.op_type}, output shape: {node_outputs[0].shape}")
        
        except Exception as e:
            print(f"Failed to run layer {node.name} ({node.op_type}): {e}")

    print("Inference completed for all layers.")
    return layer_outputs

# Example usage
model_path = "/home/mcw/Desktop/6.Models-Test-Perfalign/onnx_models/yolov5.onnx"
layer_outputs = infer_layer_by_layer(model_path)
