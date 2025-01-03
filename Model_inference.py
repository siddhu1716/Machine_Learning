import os
import sys
import json
import numpy as np
import onnx
# Framework imports
import onnxruntime as ort
import tensorflow as tf
import torch
import torchvision
from onnx import helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE

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


def infer_tflite_model(model_path):
    print("Running inference on TFLite model...")
    
    interpreter = tf.lite.Interpreter(model_path=model_path, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    tensor_details = interpreter.get_tensor_details()

    input_index = input_details[0]['index']
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']  

    input_data = np.random.uniform(0.0, 0.1, input_shape).astype(input_dtype)
    # print(input_data)
    np.savetxt('input_data1.txt', input_data.flatten(), delimiter=' ')
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    layer_outputs = {}
    print(tensor_details[0])
    print(tensor_details[1])
    print(tensor_details[5])
    for tensor in tensor_details:
        tensor_name = tensor['name']
        tensor_index = tensor['index']
        quantization_params = tensor.get('quantization_parameters', None)
        
        if quantization_params and quantization_params['scales'].size > 0 and quantization_params['zero_points'].size > 0:
            # Model is quantized
            print(tensor_index)
            print(" ")
            print(f"{tensor_name} is quantized")
            print("-------------------------")
            # print("Quantized tensor found!")
            is_quantized = True
            # break
        else:
            # Model is not quantized
            print(f"{tensor_name} NOT QUANTIZED")
            is_quantized = False

        # quantization_params = tensor.get('quantization', None)
        # if quantization_params:
        #     print("Quantized tensor found!")
        tensor_data = interpreter.get_tensor(tensor_index)
        layer_outputs[tensor_name] = tensor_data.tolist()

    with open("Tflite_layer_outputs.json", 'w') as json_file:
        json.dump(layer_outputs, json_file)
    print("TFLite inference completed. Outputs saved.")

    print(is_quantized)


def infer_onnx_model(model_path):
    print("Running inference on ONNX model...")

    # Load the original ONNX model
    ort_session = ort.InferenceSession(model_path)

    # Get original model outputs
    org_outputs = [x.name for x in ort_session.get_outputs()]

    # Load the ONNX model to modify outputs
    model = onnx.load(model_path)




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


    # Add all layers as outputs
    for node in model.graph.node:
        for output in node.output:
            if output not in org_outputs:
                model.graph.output.append(helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None))

    # Serialize the modified model to string
    modified_model = model.SerializeToString()
    ort_session_1 = ort.InferenceSession(modified_model)

    # Get input information
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
    layer_outputs = dict(zip(outputs, [o.tolist() for o in ort_outs]))

    # Save to JSON
    output_file = "Onnx_layer_outputs.json"
    with open(output_file, 'w') as json_file:
        json.dump(layer_outputs, json_file)

    print(f"ONNX inference completed. Outputs saved to {output_file}")


def infer_pytorch_model(model_path):
    print("Running inference on PyTorch model...")

    # Load a PyTorch model (example: ResNet50)
    model = torchvision.models.resnet50(pretrained=True)  # You can modify this to load your custom model
    model.eval()

    # Generate random input data
    input_shape = (1, 3, 64, 64)  # Modify input shape based on your model requirements
    input_data = torch.from_numpy(np.random.uniform(0.0, 0.1, input_shape).astype(np.float32))

    # Hook function to capture layer outputs
    layer_outputs = {}

    def hook_fn(module, input, output):
        layer_name = str(module)
        layer_outputs[layer_name] = output.detach().cpu().numpy().tolist()

    # Register hooks
    for name, layer in model.named_modules():
        if not list(layer.children()):  # Only hook leaf layers
            layer.register_forward_hook(hook_fn)

    # Run inference
    with torch.inference_mode():
        _ = model(input_data)

    # Save to JSON
    output_file = "pytorch_layer_outputs.json"
    with open(output_file, 'w') as json_file:
        json.dump(layer_outputs, json_file)

    print(f"PyTorch inference completed. Outputs saved to {output_file}")


def main(model_path):
    if model_path.endswith(".tflite"):
        infer_tflite_model(model_path)
    elif model_path.endswith(".onnx"):
        infer_onnx_model(model_path)
    elif model_path.endswith(".pt") or model_path.endswith(".pth"):
        infer_pytorch_model(model_path)
    else:
        print("Unsupported model format. Please provide a TFLite (.tflite), ONNX (.onnx), or PyTorch (.pt or .pth) model.")


def upload_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
    main(model_path)

# upload_model("/home/mcw/Desktop/Perfalign_research/models/mobilenet_v2_1.0_224_INT8.tflite")
upload_model("/home/mcw/Desktop/6.Models-Test-Perfalign/tflite_models/yolov5.tflite")
# upload_model("/home/mcw/Desktop/6.Models-Test-Perfalign/onnx_models/mobilenetv2-7.onnx")

# c1 , c2  -- non quantized (MSE)

# "-l" -- final output or the intermediate layers 

# we have to loop throug every layer and find weather is quantized or not and then dequantize it (sale and zero point)

# coun nq , qu (nq > qu) not pass -l argumrnt to armnn 

