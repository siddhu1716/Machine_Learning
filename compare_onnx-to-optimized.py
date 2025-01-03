

# import json

# with open('/home/mcw/Desktop/Perfalign_research/gpt2_Opset17.onnx_Model_resp.json', 'r') as f:
#     normal_model = json.load(f)

# with open('/home/mcw/Desktop/Perfalign_research/gpt2_Opset17.onnx_optimized.onnx_Comp_Model_resp.json', 'r') as f:
#     optimized_model = json.load(f)



# def extract_mappings(normal, opti):
#     nodes1 = normal["server_address"]["graphs"][0]["nodes"]  # Normal model nodes
#     nodes2 = opti["server_address"]["graphs"][0]["nodes"]  # Optimized model nodes

#     i, j = 0, 0  
#     mappings = {}  
#     fusion_group = []  

#     while i < len(nodes1) and j < len(nodes2):
#         node1 = nodes1[i]
#         node2 = nodes2[j]

#         # Skip pseudo_const nodes
#         if node1.get("label", "") == "pseudo_const":
#             i += 1
#             continue

#         if node2.get("label", "") == "pseudo_const":
#             j += 1
#             continue

#         # Extract __tensor_tag from the first position of inputsMetadata (only if it exists)
#         Input_tensor_tag1 = None
#         if node1.get("inputsMetadata"):
#             first_input_metadata = node1.get("inputsMetadata", [])[0] if node1["inputsMetadata"] else {}
#             Input_tensor_tag1 = next(
#                 (attr["value"] for attr in first_input_metadata.get("attrs", []) if attr["key"] == "__tensor_tag"), 
#                 None
#             )

#         Input_tensor_tag2 = None
#         if node2.get("inputsMetadata"):
#             first_input_metadata = node2.get("inputsMetadata", [])[0] if node2["inputsMetadata"] else {}
#             Input_tensor_tag2 = next(
#                 (attr["value"] for attr in first_input_metadata.get("attrs", []) if attr["key"] == "__tensor_tag"), 
#                 None
#             )

#         print(f"\ni pointer: {i}, j pointer: {j}")
#         print(f"Normal node: {node1.get('label')}")
#         print(f"Optimized node: {node2.get('label')}")

#         if Input_tensor_tag1 == Input_tensor_tag2:
#             print("\nInput tensors are matched\n")

#             output_tensor_name_1 = next(
#                 (attr["value"] for output in node1.get("outputsMetadata", []) 
#                 for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
#                 None
#             )

#             output_tensor_name_2 = next(
#                 (attr["value"] for output in node2.get("outputsMetadata", []) 
#                 for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
#                 None
#             )

#             if output_tensor_name_1 == output_tensor_name_2:
#                 print("\nOutput tensors are matched\n")

#                 if fusion_group:
#                     # Store fusion group for this key
#                     mappings[output_tensor_name_1] = fusion_group[:]  # Copy the fusion group
#                     fusion_group.clear()  # Reset fusion group
                
#                 # Store the single layer mapping
#                 mappings[output_tensor_name_1] = [node1.get('label')]
                
#                 # Move both pointers
#                 i += 1
#                 j += 1
#             else:
#                 print("\nOutput tensors are not matched\n")
#                 print("Only moving i pointer and adding node to fusion group\n")
                
#                 # This indicates that the normal node is being fused
#                 fusion_group.append(node1.get('label'))
                
#                 # Only increment i since j stays on the same fused node
#                 i += 1  
#         else:
#             print(f"Input tensors do not match: {Input_tensor_tag1} vs {Input_tensor_tag2}")
#             print("\nMoving both i and j pointers\n")
            
#             # If input tensors don't match, no fusion is possible
#             i += 1
#             j += 1

#     return mappings


# # Extract mappings and save to JSON
# mappings = extract_mappings(normal_model, optimized_model)
# print("Final Mappings of Fused Layers to Tensor Names:")

# with open("/home/mcw/Desktop/Perfalign_research/mapping1.json", "w") as file:
#     json.dump(mappings, file, indent=2)


# import json

# with open('/home/mcw/Desktop/Perfalign_research/yolov5.onnx_Model_resp.json', 'r') as f:
#     normal_model = json.load(f)

# with open('/home/mcw/Desktop/Perfalign_research/yolo_optimized.onnx_Model_resp.json', 'r') as f:
#     optimized_model = json.load(f)


# def extract_mappings(normal, opti):
#     nodes1 = normal["server_address"]["graphs"][0]["nodes"]  # Normal model nodes
#     nodes2 = opti["server_address"]["graphs"][0]["nodes"]  # Optimized model nodes

#     i, j = 0, 0  
#     mappings = {}  
#     fusion_group = []  

#     while i < len(nodes1) and j < len(nodes2):
#         node1 = nodes1[i]
#         node2 = nodes2[j]

#         # Skip pseudo_const nodes
#         if node1.get("label", "") in ["pseudo_const","DequantizeLinear"]:
#             i += 1
#             continue

#         if node2.get("label", "") in ["pseudo_const","DequantizeLinear"]:
#             j += 1
#             continue

#         # Extract __tensor_tag from inputsMetadata of the first input (if it exists)
#         Input_tensor_tag1 = None
#         if node1.get("inputsMetadata"):
#             first_input_metadata = node1.get("inputsMetadata", [])[0] if node1["inputsMetadata"] else {}
#             Input_tensor_tag1 = next(
#                 (attr["value"] for attr in first_input_metadata.get("attrs", []) if attr["key"] == "__tensor_tag"), 
#                 None
#             )

#         Input_tensor_tag2 = None
#         if node2.get("inputsMetadata"):
#             first_input_metadata = node2.get("inputsMetadata", [])[0] if node2["inputsMetadata"] else {}
#             Input_tensor_tag2 = next(
#                 (attr["value"] for attr in first_input_metadata.get("attrs", []) if attr["key"] == "__tensor_tag"), 
#                 None
#             )

#         print(f"\ni pointer: {i}, j pointer: {j}")
#         print(f"Normal node: {node1.get('label')}")
#         print(f"Optimized node: {node2.get('label')}")

#         # Step 1: If input tensors match, fusion starts
#         if Input_tensor_tag1 == Input_tensor_tag2:
#             print("\nInput tensors are matched\n")
#             result_output = next(
#                 (attr["value"] for output in node1.get("outputsMetadata", []) 
#                 for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
#                 None
#             )
            
#             res=result_output

#             # Begin fusion group, keep collecting normal nodes
#             fusion_group = []  # Reset the fusion group
#             while i < len(nodes1):  # Process all possible fused layers
#                 node1 = nodes1[i]  # Current normal node
#                 fusion_group.append(node1.get('label'))

#                 # Extract the output tensor name from node1
#                 output_tensor_name_1 = next(
#                     (attr["value"] for output in node1.get("outputsMetadata", []) 
#                     for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
#                     None
#                 )
                
#                 # Extract the output tensor name from node2
#                 output_tensor_name_2 = next(
#                     (attr["value"] for output in node2.get("outputsMetadata", []) 
#                     for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
#                     None
#                 )

#                 print(f"Fusion in progress. Normal node: {node1.get('label')}")
#                 print(f"Output tensor (normal): {output_tensor_name_1}")
#                 print(f"Output tensor (optimized): {output_tensor_name_2}")

#                 if output_tensor_name_1 == output_tensor_name_2:
#                     print("\nFusion completed. Output tensors match\n")
                    
#                     # Store the fusion group for this key
#                     mappings[res] = fusion_group[:]  # Copy the fusion group
#                     fusion_group.clear()  # Reset the fusion group

#                     # Move both pointers as fusion is complete
#                     i += 1
#                     j += 1
#                     break  # End the internal while loop
#                 else:
#                     print("\nOutput tensors are not matched, continue fusing...\n")
#                     i += 1  # Move i to the next normal layer (since fusion is ongoing)
                    
#             if output_tensor_name_1 != output_tensor_name_2:
#                 print("Warning: output tensor did not match even after processing all possible layers.")
#         else:
#             print(f"Input tensors do not match: {Input_tensor_tag1} vs {Input_tensor_tag2}")
#             print("\nMoving both i and j pointers\n")
            
#             # No fusion, move both pointers
#             i += 1
#             j += 1

#     return mappings


# # Extract mappings and save to JSON
# mappings = extract_mappings(normal_model, optimized_model)
# print("Final Mappings of Fused Layers to Tensor Names:")

# with open("/home/mcw/Desktop/Perfalign_research/mapping2.json", "w") as file:
#     json.dump(mappings, file, indent=2)





import json
import re
import os

with open('gpt2_Opset17.onnx_Model_resp.json', 'r') as f:
    normal_model = json.load(f)

with open('/home/mcw/Desktop/Perfalign_research/gpt2_Opset17.onnx_optimized.onnx_Comp_Model_resp.json', 'r') as f:
    optimized_model = json.load(f)


# =====================================================================================================================

def check_fusion(label):
    match = re.search(r'com\.microsoft::(\w+)', label)
    if match:
        layer_name = match.group(1)
        return True, layer_name
    return False, ""

def get_tensor_name(text):
    pattern = r"(/h\.\d+/attn/MatMul)" 

    match = re.match(pattern, text)
    if match:
        result = match.group(1)
    return result+"_output_0"

def function(layer,node_id,activation,nodes1):
    process_functions = [
        (['FusedMatMul', "Fusedmatmul"], process_fusedmatmul),
        (['FusedConv', 'FusedConv2d'], process_fusedconv),
        (['FastGelu'], process_fastgelu)
    ]
    
    for layer_names, process_func in process_functions:
        if layer in layer_names:
            return process_func(node_id,activation,nodes1)
    return None

def process_fusedmatmul(node_id,activation,nodes1):
    try:
        print("Processing FusedMatMul...")

        fusions = [] 

        pattern = r"(/h\.\d+/attn/MatMul)" 
        match = re.match(pattern, node_id)
        if match:
            node_id = match.group(1)

        i = next((index for index, node in enumerate(nodes1) if node.get("id", "") == node_id), -1)
        
        if i == -1: 
            print(f"Error: Node with ID '{node_id}' not found.")
            return []
        
        if activation== None:
            activation="Div"

        while i < len(nodes1):
            node = nodes1[i]
            layer_name = node.get("label", "")
            
            if layer_name == activation:
                fusions.append(node.get("label", ""))
                break  
            
            if node.get("label", "") in ["pseudo_const", "DequantizeLinear", "Constant","ConstantOfShape","Add","Pow"]:
                i += 1
                continue

            fusions.append(node.get("label", ""))
            i += 1 

        return fusions[:] 
    except Exception as e:
        print("error in fused matmul",e)

def process_fusedconv(node_id, activation, nodes1):
    try:
        print("Processing FusedConv...")
        
        fusions = [] 

        i = next((index for index, node in enumerate(nodes1) if node.get("id", "") == node_id), -1)
        
        print(i)

        if i == -1: 
            print(f"Error: Node with ID '{node_id}' not found.")
            return []

        while i < len(nodes1):
            node = nodes1[i]
            layer_name = node.get("label", "")
            
            if layer_name == activation:
                fusions.append(node.get("label", ""))
                break  
            
            if node.get("label", "") in ["pseudo_const", "DequantizeLinear", "Constant"]:
                i += 1
                continue

            fusions.append(node.get("label", ""))
            i += 1 

        return fusions[:] 
    except Exception as e:
        print("error in fused conv peocessing",e)

def process_fastgelu(node_ids, activation, nodes1):
    try:
        print("Processing FastGelu...")

        fusions = [] 
        node_id = re.sub(r'_output.*$', '', node_ids)  # Clean node_id

        i = next((index for index, node in enumerate(nodes1) if node.get("id", "") == node_id), -1)
        if i == -1: 
            print(f"Error: Node with ID '{node_id}' not found.")
            return []

        if activation is None:
            activation = "Tanh"

        fond_activation = None
        mul_count = 0

        while i < len(nodes1):
            node = nodes1[i]
            layer_name = node.get("label", "")

            if layer_name in ["pseudo_const", "DequantizeLinear", "Constant"]:
                i += 1
                continue

            print(f"Current Layer: {layer_name}")

            output_tensor = next(
                (attr["value"] for output in node.get("outputsMetadata", []) 
                for attr in output.get("attrs", []) if attr["key"] == "tensor_name"), 
                None
            )

            print(f"Output Tensor: {output_tensor}")

            input_tensor = None
            j = i + 1
            if j < len(nodes1):
                next_node = nodes1[j]

                next_layer_name = next_node.get('label', "")
                print(f"Next Layer: {next_layer_name}")
                
                # Skip consecutive constant nodes
                while next_layer_name in ["pseudo_const", "DequantizeLinear", "Constant"] and j + 1 < len(nodes1):
                    j += 1 
                    next_node = nodes1[j]
                    next_layer_name = next_node.get('label', "")
                
                if next_layer_name not in ["pseudo_const", "DequantizeLinear", "Constant"]:
                    inputs_metadata = next_node.get("inputsMetadata", [])
                    if inputs_metadata and len(inputs_metadata) > 0:
                        input_tensor = next(
                            (attr["value"] for attr in inputs_metadata[0].get("attrs", []) 
                            if attr["key"] == "__tensor_tag"), 
                            None
                        )

            if output_tensor and output_tensor == input_tensor:
                fusions.append(layer_name)
                
                if layer_name == "Tanh":
                    fond_activation = "Tanh"
                    
                if layer_name == "Mul":
                    mul_count += 1
            else:
                break 

            i += 1  # Increment i at the end to continue with the next node
            if fond_activation == activation and mul_count >= 3:
                break

        return fusions
    except Exception as e:
        print("Exception in fast gelu:", e)


def extract_mappings(normal, opti):
    try:
        print("Entering extract mappings")

        # Extract nodes from the normal and optimized graphs
        nodes1 = normal["server_address"]["graphs"][0]["nodes"]
        nodes2 = opti["server_address"]["graphs"][0]["nodes"]

        j = 0 
        mappings = {}  
        
        while j < len(nodes2):
            node2 = nodes2[j]
            
            # Skip nodes that are not relevant
            if node2.get("label", "") in ["pseudo_const", "DequantizeLinear", "Constant"]:
                j += 1
                continue
            
            # Check if the node is part of a fusion
            status, label = check_fusion(node2.get("label", ""))
            if status:
                initial_node_id = node2.get("id")
                activation = next((attr.get("value") for attr in node2.get("attrs", []) if attr.get("key") == "activation"), None)

                if activation == None and label=="FastGelu":
                    input_tensor_id = next(
                        (attr.get("value") for attr in node2.get("inputsMetadata", [{}])[0].get("attrs", []) 
                         if attr.get("key") == "__tensor_tag"), 
                        None
                    )
                    if input_tensor_id:
                        initial_node_id = input_tensor_id  
                # Call the processing function for the fusion
                to_map = function(label, initial_node_id, activation, nodes1)

                if activation is None:
                    if label == "FusedMatMul":
                        initial_node_id = get_tensor_name(initial_node_id)
                    elif label == "FastGelu":
                        output_tensor_name = next(
                            (attr.get("value") for output in node2.get("outputsMetadata", []) 
                             for attr in output.get("attrs", []) if attr.get("key") == "tensor_name"), 
                            None
                        )
                        if output_tensor_name:
                            initial_node_id = output_tensor_name

                if initial_node_id is not None: 
                    mappings[initial_node_id] = to_map 
                else:
                    print(f"Warning: Node ID not found for node {node2.get('label')} at index {j}")

            j += 1  # Increment loop counter

        # Process mapping results for visualization
        main_graph_results = {
            node_name: ndb.NodeDataResult(value=time)
            for node_name, time in mappings.items()
        }

        gradient = [
            ndb.GradientItem(stop=0, bgColor='green'),
            ndb.GradientItem(stop=0.5, bgColor='yellow'),
            ndb.GradientItem(stop=1, bgColor='red'),
        ]

        fusion_graph_data = ndb.GraphNodeData(results=main_graph_results, gradient=gradient)
        
        fusions_json_path = os.path.join(COMPARATOR_MODEL_RESP, f"{model_name}_graph_fusions.json")
        fusion_graph_data.save_to_file(fusions_json_path)

        return fusions_json_path

    except Exception as e:
        print(f"Error in extract_mappings: {e}")


        
# =====================================================================================================================


# Extract mappings and save to JSON
mappings = extract_mappings(normal_model, optimized_model)
print("Final Mappings of Fused Layers to Tensor Names:")

with open("/home/mcw/Desktop/Perfalign_research/mapping2.json", "w") as file:
    json.dump(mappings, file, indent=2)