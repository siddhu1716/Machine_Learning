import json

def preprocess_graph_v2(graph):
    """
    Preprocess the graph:
    - Remove nodes with the label 'pseudo_const'.
    - Preserve the order of input and output nodes.
    - Sort the remaining nodes by their 'id'.

    Args:
        graph (dict): The input graph in dictionary format.

    Returns:
        dict: The preprocessed graph.
    """
    # Access the nodes
    nodes = graph["server_address"]["graphCollections"][0]["graphs"][0]["nodes"]
    
    # Separate the first (input) and last (output) nodes
    first_node = nodes[0]  # Input node
    last_node = nodes[-1]  # Output node
    
    # Filter out pseudo_const nodes and exclude the first and last nodes
    middle_nodes = [node for node in nodes[1:-1] if node["label"] != "pseudo_const"]
    
    # Sort the middle nodes by their 'id'
    sorted_middle_nodes = sorted(middle_nodes, key=lambda x: int(x["id"]))
    
    # Reconstruct the node list: input node + sorted middle nodes + output node
    processed_nodes = [first_node] + sorted_middle_nodes + [last_node]
    
    # Update the graph with processed nodes
    graph["server_address"]["graphCollections"][0]["graphs"][0]["nodes"] = processed_nodes
    
    return graph

# Example usage
with open('/home/mcw/Desktop/Perfalign_research/example_graph.json', 'r') as f:
    graph_data = json.load(f)

processed_graph = preprocess_graph_v2(graph_data)

# Print or save the processed graph
print(json.dumps(processed_graph, indent=4))

# Optionally save the updated graph to a new file
with open('/home/mcw/Desktop/Perfalign_research/processed_graph.json', 'w') as f:
    json.dump(processed_graph, f, indent=4)
 