import json
import numpy as np

with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/mobilenet_v2_1.0_224_INT8.tflite_Model_resp.json', 'r') as f:
    normal_model = json.load(f)

with open('/home/mcw/Desktop/Perfalign_research/ModelResponses/CpuAcc_mobilenet.dot_Model_resp.json', 'r') as f:
    armnn_model = json.load(f)


label_mapping = {
    "conv_2d": ['conv_2d', 'conv2d', 'Conv2D', 'Conv2d', 'CONV2D', 'CONV2d', 'Conv','-Conv2D',"Convolution2d"],
    "relu6": ["RELU6"],
    "GraphInputs":["tfl.quantize","input_1","input_ids"],
    "GraphOutputs":["Output" ],#for layer type,
    "softmax":["MobilenetV2/Predictions/Reshape_11","Softmax"],
    "depthwise_conv_2d":["DepthwiseConv2D","DepthwiseConvolution2d"],
    "Add":["ADD","add"],
    "add":["ADD","Add"],
    "average_pool_2d":["AveragePool","AveragePool2D"],
    "strided_slice":["StridedSlice"],
    "logistic":["Activation"],
    "Mul":["mul"],
    "reshape":["Reshape_for","Reshape"],
    "gather":["gather"],
    "Sub":["Sub"],
    "FullyConnected":["FullyConnected"],
    "input_mask":["input_mask"]
}

fuse_into_single_layers=["fused-Conv2D","fused-DepthwiseConv2D","fused-FullyConnected","fused-Add"]

def extract_mappings(model1, model2):
    list1 = []
    list2 = []

    nodes1 = model1["server_address"]["graphCollections"][0]["graphs"][0]["nodes"]
    nodes2 = model2["server_address"]["graphs"][0]["nodes"]

    i, j = 0, 0  

    def get_attr(node, key):
        return next((attr["value"] for attr in node.get("attrs", []) if attr["key"] == key), None)

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
        print(node2.get("label"))
        fused_activation = get_attr(node1, "fused_activation_function")
        print("fused_activation:", fused_activation)

        if fused_activation and fused_activation.lower() != "none":
            function_name = get_attr(node2, "Function")
            layer_name = get_attr(node2, "LayerName")
            print(function_name)
            print(layer_name)
            if function_name in ["BoundedReLu","ReLu"] and layer_name in fuse_into_single_layers:
                list1.append(node1["namespace"])
                list2.append(node2["label"])
                i += 1
                j += 1
                print(list1)
                print(list2)
                continue

        label1 = node1.get("label", "")
        layer_name2 = get_attr(node2, "LayerName")

        if layer_name2 in label_mapping.get(label1, []):
            list1.append(node1["namespace"])
            list2.append(node2["label"])
        else:
            print(f"Unmapped label: {label1} for LayerName: {layer_name2}")
            label_mapping.setdefault(label1, []).append(layer_name2)

        i += 1
        j += 1

    return list1, list2



l1, l2 = extract_mappings(normal_model, armnn_model)

print("Model 1 Namespaces:", l1)
print("Model 2 Layer Names:", l2)
print(len(l1))
print(len(l2))
