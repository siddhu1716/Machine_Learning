import json

# Load the JSON file
with open('/home/mcw/Desktop/Perfalign_research/arm_yolov5.tflite_Model_resp.json', 'r') as f:
    armnn_model = json.load(f)

def preprocess_graph(graph):
    """
    Preprocess the graph:
    - Remove nodes with the label 'pseudo_const'.
    - Preserve the order of the first (input) and last (output) nodes.
    - Sort the remaining nodes by their 'id'.

    Args:
        graph (dict): The input graph in dictionary format.

    Returns:
        dict: The preprocessed graph.
    """
    # Extract the nodes
    nodes = graph["server_address"]["graphs"][0]["nodes"]
    
    # Separate the first and last nodes
    first_node = nodes[0]  # Input node
    last_node = nodes[-1]  # Output node
    
    # Filter out pseudo_const nodes and exclude the first and last nodes
    middle_nodes = [node for node in nodes[1:-1] if node["label"] != "pseudo_const"]
    
    # Sort the middle nodes by their 'id'
    sorted_middle_nodes = sorted(middle_nodes, key=lambda x: int(x["id"]))
    
    # Reconstruct the node list: input node + sorted middle nodes + output node
    processed_nodes = [first_node] + sorted_middle_nodes + [last_node]
    
    # Update the nodes back in the original graph
    graph["server_address"]["graphs"][0]["nodes"] = processed_nodes
    
    return graph

# Apply the preprocessing function
processed_model = preprocess_graph(armnn_model)

# Print or save the result
print(json.dumps(processed_model, indent=4))  # Pretty print the updated model

# Optionally, save the processed model back to a file
with open('/home/mcw/Desktop/Perfalign_research/processed_arm_yolov5.tflite_Model_resp.json', 'w') as f:
    json.dump(processed_model, f, indent=4)
