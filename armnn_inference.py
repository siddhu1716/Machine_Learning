import os
import json
import numpy as np

# Path to the folder containing intermediate layer outputs
folder_path = "/tmp/ArmNNIntermediateLayerOutputs"

# Get all .numpy files in the folder
numpy_files = [f for f in os.listdir(folder_path) if f.endswith('.numpy')]

loaded_data = {}

for file in numpy_files:
    file_path = os.path.join(folder_path, file)
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        layer_name = data.get("layerName", None)
        shape = data.get("shape", None)
        array_data = data.get("data", None)

        if array_data is not None and shape is not None:
            array = np.array(array_data)
            array = array.reshape(shape)  
        else:
            array = None
            print(f"Warning: No valid data or shape in {file}")
        print(layer_name)
        loaded_data[layer_name] = array.tolist()

    except Exception as e:
        print(f"Error loading {file}: {e}")

# Function to extract the numeric suffix for sorting
def extract_numeric_suffix(name):
    try:
        return int(name.split(':')[-1])
    except ValueError:
        # Handle non-numeric or invalid suffixes by assigning a default value
        return float('inf')  

# Sorting the layers by the numeric suffix after the last ':'
sorted_layer_names = sorted(loaded_data.keys(), key=extract_numeric_suffix)

# Create a new dictionary with sorted order
sorted_loaded_data = {name: loaded_data[name] for name in sorted_layer_names}

# Save the sorted outputs to a JSON file
with open("armnn_layer_outputs_sorted.json", "w") as outfile: 
    json.dump(sorted_loaded_data, outfile)

# Print the shape of each layer output
for layer_name, layer_output in sorted_loaded_data.items():
    print(f"Shape of output for layer '{layer_name}': {np.array(layer_output).shape}")
