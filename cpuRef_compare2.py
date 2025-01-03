import json
import numpy as np

with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/mobilenet_v2_1.0_224_INT8.tflite_Model_resp.json', 'r') as f:
    normal_model = json.load(f)

with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/armnn_mobilenet_v2.json', 'r') as f:
    armnn_model = json.load(f)

label_mapping = {
    "conv_2d": ['conv_2d', 'conv2d', 'Conv2D', 'Conv2d', 'CONV2D', 'CONV2d', 'Conv','-Conv2D',"Convolution2d"],
    "relu6": ["RELU6"],
    "GraphInputs":["tfl.quantize","input_1"],
    "GraphOutputs":["Output" ],#for layer type,
    "softmax":["MobilenetV2/Predictions/Reshape_11"],
    "depthwise_conv_2d":["DepthwiseConv2D","DepthwiseConvolution2d"],
    "Add":["ADD","add"],
    "average_pool_2d":["AveragePool"],
    "strided_slice":["StridedSlice"],
    "logistic":["Activation"],
    "Mul":["mul"],
    "reshape":["Reshape_for","Reshape"]
}

def extract_mappings(model1, model2):
    list1 = []
    list2 = []

    nodes1 = model1["server_address"]["graphCollections"][0]["graphs"][0]["nodes"]
    nodes2 = model2["server_address"]["graphs"][0]["nodes"]

    i, j = 0, 0  

    while i < len(nodes1) and j < len(nodes2):
        node1 = nodes1[i]
        node2 = nodes2[j]

        if node1.get("label", "") == "pseudo_const":
            i += 1
            continue

        if node2.get("label", "") == "pseudo_const":
            j += 1
            continue

        print("i pointer:", i)
        print("j pointer:", j)

        print(node1.get("label"))
        print(node1.get("namespace"))
        print(node2.get("label"))


        fused_activation = next(
            (attr["value"] for attr in node1.get("attrs", []) if attr["key"] == "fused_activation_function"),
            None,
        )
        print("fused_activation:", fused_activation)

        if fused_activation and fused_activation.lower() != "none":
            j = j+1
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
                print("next_layer_name:", next_layer_name)
                print("next_layer_type:", next_layer_type)

                if (
                    next_layer_name in label_mapping.get(node1["label"], [])
                    and next_layer_type == "Activation"
                ):
                    list1.append(node1["namespace"])
                    print(list1)
                    list2.append(next_node2['label'])
                    print(list2)
                    i+=1
                    j+=1
                    continue

        label1 = node1.get("label", "")
        print("label1",label1)
        layer_name2 = next(
            (attr["value"] for attr in node2.get("attrs", []) if attr["key"] == "LayerName"),
            None,
        )
        print("label 2",layer_name2)
        if layer_name2 in label_mapping.get(label1, []):
            print("present")
            # Record the mapping
            list1.append(node1["namespace"])
            list2.append(node2['label'])
            print(list1)
            print(list2)
        else:
            print(f"Unmapped label: {label1} for LayerName: {layer_name2}")
            if layer_name2  in ["folded-pad"]:
                i+=1
                continue

            # label_mapping[label1] = [layer_name2]
            # j+=1
            # continue

        i += 1
        j += 1

    return list1, list2


l1, l2 = extract_mappings(normal_model, armnn_model)

print("Model 1 Namespaces:", l1)
print("Model 2 Layer Names:", l2)
print(len(l1))
print(len(l2))
