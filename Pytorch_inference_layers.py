import torch
import numpy as np
import json
import torchgen
import torchvision
# Load the PyTorch model
model = torchvision.models.resnet50(pretrained = True)

model.eval()  # Set the model to evaluation mode

input_shape = (1, 3, 64, 64) 
input_data = torch.from_numpy(np.random.uniform(0.0, 0.1, input_shape).astype(np.float32))

# Dictionary to store layer-wise outputs
layer_outputs = {}

# Hook function to capture the output of each layer
def hook_fn(module, input, output):
    layer_name = str(module)  # Use layer name as the key
    layer_outputs[layer_name] = output.detach().cpu().numpy().tolist()  # Convert to list for JSON compatibility

# Register hooks to all layers
for name, layer in model.named_modules():
    if not list(layer.children()):  # Only hook leaf layers
        layer.register_forward_hook(hook_fn)

# Run inference
with torch.inference_mode():
    _ = model(input_data)

# Save layer outputs to JSON file
output_json_path = "pytorch_layer_outputs.json"
with open(output_json_path, 'w') as json_file:
    json.dump(layer_outputs, json_file)

# Print shapes of all layers' outputs
for layer_name, output_data in layer_outputs.items():
    print(f"Shape of output for layer '{layer_name}': {np.array(output_data).shape}")
