import json
import numpy as np

# we get 2 jsons 

# Load generated outputs from layer_outputs.json
with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/normal_mobilenet_v2.json', 'r') as f:
    normal_model = json.load(f)

# Load expected outputs from expected_outputs.json (modify the filename as needed)
with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/armnn_mobilenet_v2.json', 'r') as f:
    armnn_model = json.load(f)


# Example dictionary to map label names
label_mapping = {
    "conv_2d": ['conv_2d', 'conv2d', 'Conv2D', 'Conv2d', 'CONV2D', 'CONV2d', 'Conv','-Conv2D'],
    "relu6": "RELU6",
    "GraphInputs":"tfl.quantize",
    "GraphOutputs":"Output" ,#for layer type,
    "softmax":"MobilenetV2/Predictions/Reshape_11",
    "depthwise_conv_2d":["DepthwiseConv2D","DepthwiseConvolution2d"]
}

def extract_mappings(model1, model2):
    list1 = []
    list2 = []

    nodes1 = model1["server_address"]["graphCollections"][0]["graphs"][0]["nodes"]
    nodes2 = model2["server_address"]["graphs"][0]["nodes"]

    i, j = 0, 0  # Two pointers for the two model nodes

    while i < len(nodes1) and j < len(nodes2):
        node1 = nodes1[i]
        node2 = nodes2[j]

        # Skip pseudo_const layers in the first model
        if node1.get("label", "") == "pseudo_const":
            i += 1
            continue

        # Skip pseudo_const layers in the second model
        if node2.get("label", "") == "pseudo_const":
            j += 1
            continue
        print("i pointer",i)
        print("j pointer",j)
        # Extract fused activation function
        fused_activation = next(
            (attr["value"] for attr in node1.get("attrs", []) if attr["key"] == "fused_activation_function"),
            None,
        )
        print("fused act",fused_activation)

        if fused_activation and fused_activation.lower() != "none":
            # Handle the fused activation layer
            j+=1
            if j < len(nodes2):
                next_node2 = nodes2[j]
                print(next_node2)
                next_layer_type = next(
                    (attr["value"] for attr in next_node2.get("attrs", []) if attr["key"] == "LayerType"),
                    None,
                )
                next_layer_name = next(
                    (attr["value"] for attr in next_node2.get("attrs", []) if attr["key"] == "LayerName"),
                    None,
                )
                print("layer name",next_layer_name)

                print("layer type",next_layer_type)
                if (
                    next_layer_name in label_mapping.get(node1["label"])
                    and next_layer_type == "Activation"
                ):
                    # Record the mapping
                    list1.append(node1["namespace"])
                    list2.append(next_layer_name)
                    i+=1
                    j+=1
                    continue

        # Compare label and LayerName for non-activation layers
        label1 = node1.get("label", "")
        print("label1",label1)
        label2 = next((attr["value"] for attr in node2.get("attrs", []) if attr["key"] == "LayerName"), None)
        print("label2",label2)

        if label_mapping.get(label1, label1) == label_mapping.get(label2, label2):
            # Record the mapping
            list1.append(node1["namespace"])
            list2.append(label2)

        # Move both pointers forward
        i += 1
        j += 1

    return list1, list2

l1,l2=extract_mappings(normal_model,armnn_model)
print(l1)
print(l2)
print(len(l1))
print(len(l2))