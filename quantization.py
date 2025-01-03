import onnx
import onnxmltools

# Load the model
model = onnx.load("/home/mcw/Desktop/Perfalign_research/mobilenetv2-7.onnx")

# Convert to FP16
fp16_model = onnxmltools.utils.float16_converter.convert_float_to_float16(model)

# Save the model
onnx.save(fp16_model, "/home/mcw/Desktop/Perfalign_research/mobilenet_fp16.onnx")
