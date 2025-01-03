# import onnx
# import onnxruntime as ort
# import numpy as np
# import json
# file_name = 'lenet (1).onnx'

# # Load the ONNX model
# ort_session_1 = ort.InferenceSession(file_name)

# # Get the original output names
# org_outputs = [x.name for x in ort_session_1.get_outputs()]
# # print(org_outputs)
# # Load the model
# model = onnx.load(file_name)

# # Add all layers as output
# for node in model.graph.node:
#     # print(node)
#     for output in node.output:
#         if output not in org_outputs:
#             model.graph.output.extend([onnx.ValueInfoProto(name=output)])

# # Serialize the modified model and create a new session
# ort_session = ort.InferenceSession(model.SerializeToString())

# # Get input information
# input_info = ort_session.get_inputs()[0]
# # print(input_info)
# input_name = input_info.name
# input_shape = input_info.shape

# # Generate random input data
# a_INPUT = np.random.uniform(low=0.0, high=0.1, size=input_shape).astype(np.float32)

# # Get the output names for the modified model
# outputs = [x.name for x in ort_session.get_outputs()]

# # Run inference
# ort_outs = ort_session.run(outputs, {input_name: a_INPUT})

# # Map outputs to their names
# ort_outs_dict = dict(zip(outputs, ort_outs))
# # Create a dictionary to store layer names and their contents
# output_content = {layer_name: ort_outs_dict[layer_name].tolist() for layer_name in ort_outs_dict}

# # Save content to a JSON file
# with open('Onnx_layer_outputs.json', 'w') as json_file:
#     json.dump(output_content, json_file)
# # Print shapes of the intermediate layer outputs
# for layer_name, lay_wise_output in ort_outs_dict.items():
#     print(f"Shape of output '{layer_name}': {lay_wise_output.shape}")

import onnx
import onnxruntime as ort
import numpy as np
import json
import os

def infer_onnx_model_with_tensor_details(model_path):
    print("Running inference on ONNX model...")

    # Load the original ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Get input tensor details
    input_info = ort_session.get_inputs()
    print("Input Tensor Details:")
    input_details = []
    for idx, input_tensor in enumerate(input_info):
        input_details.append({
            'tensor_name': input_tensor.name,
            'tensor_shape': input_tensor.shape,
            'tensor_type': input_tensor.type,
            'tensor_index': idx
        })
    print(input_details)

    # Get output tensor details
    output_info = ort_session.get_outputs()
    print("Output Tensor Details:")
    output_details = []
    for idx, output_tensor in enumerate(output_info):
        output_details.append({
            'tensor_name': output_tensor.name,
            'tensor_shape': output_tensor.shape,
            'tensor_type': output_tensor.type,
            'tensor_index': idx
        })
    print(output_details)

    # Load the ONNX model to modify outputs
    model = onnx.load(model_path)

    # Add all layers as outputs
    for node in model.graph.node:
        for output in node.output:
            if output not in [out.name for out in output_info]:
                model.graph.output.append(onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None))

    # Serialize the modified model to string
    modified_model = model.SerializeToString()
    ort_session_1 = ort.InferenceSession(modified_model)

    # Get input information from modified session
    input_info = ort_session_1.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Retrieve the data type dynamically from the type string
    input_dtype = get_np_dtype_from_onnx_type(input_info.type)

    # Generate random input data with the correct data type
    input_data = np.random.uniform(0.0, 0.1, input_shape).astype(input_dtype)

    # Get all layer outputs
    outputs = [x.name for x in ort_session_1.get_outputs()]
    ort_outs = ort_session_1.run(outputs, {input_name: input_data})

    # Save the layer outputs to JSON
    layer_outputs = dict(zip(outputs, [o.tolist() for o in ort_outs]))
    output_file = "Onnx_layer_outputs.json"
    with open(output_file, 'w') as json_file:
        json.dump(layer_outputs, json_file)

    print(f"ONNX inference completed. Outputs saved to {output_file}")

    return ort_session_1  # Returning the ort_session for further use


def get_np_dtype_from_onnx_type(type_str):
    type_map = {
        "tensor(float)": np.float32,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(int8)": np.int8,
        "tensor(float16)": np.float16,
        "tensor(bool)": np.bool_,
    }
    return type_map.get(type_str, np.float32) 


# Test the function
model_path = 'your_model.onnx'
ort_session = infer_onnx_model_with_tensor_details(model_path)
