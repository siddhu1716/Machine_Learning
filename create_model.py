import onnx
import onnx.helper as helper
import onnx.checker as checker
import numpy as np

# Create the inputs and outputs for the computational graph
input_tensor = helper.make_tensor_value_info('Input', onnx.TensorProto.FLOAT, [1, 3, 224, 224])
output_tensor = helper.make_tensor_value_info('Output', onnx.TensorProto.FLOAT, [1, 1000])

# Initialize weights for the Conv layers
conv1_weights = helper.make_tensor(
    name='Conv1_W',
    data_type=onnx.TensorProto.FLOAT,
    dims=[16, 3, 3, 3],  # [out_channels, in_channels, kernel_height, kernel_width]
    vals=np.random.randn(16 * 3 * 3 * 3).astype(np.float32).tolist()
)
conv2_weights = helper.make_tensor(
    name='Conv2_W',
    data_type=onnx.TensorProto.FLOAT,
    dims=[32, 16, 3, 3],  # [out_channels, in_channels, kernel_height, kernel_width]
    vals=np.random.randn(32 * 16 * 3 * 3).astype(np.float32).tolist()
)
conv3_weights = helper.make_tensor(
    name='Conv3_W',
    data_type=onnx.TensorProto.FLOAT,
    dims=[64, 32, 3, 3],  # [out_channels, in_channels, kernel_height, kernel_width]
    vals=np.random.randn(64 * 32 * 3 * 3).astype(np.float32).tolist()
)

# BatchNorm Initializers for BN1, BN2
def create_batchnorm_initializers(prefix, channels):
    """Helper to create the BatchNormalization initializers."""
    scale = helper.make_tensor(
        name=f'{prefix}_Scale',
        data_type=onnx.TensorProto.FLOAT,
        dims=[channels],
        vals=np.ones(channels).astype(np.float32).tolist()
    )
    bias = helper.make_tensor(
        name=f'{prefix}_Bias',
        data_type=onnx.TensorProto.FLOAT,
        dims=[channels],
        vals=np.zeros(channels).astype(np.float32).tolist()
    )
    mean = helper.make_tensor(
        name=f'{prefix}_Mean',
        data_type=onnx.TensorProto.FLOAT,
        dims=[channels],
        vals=np.zeros(channels).astype(np.float32).tolist()
    )
    variance = helper.make_tensor(
        name=f'{prefix}_Variance',
        data_type=onnx.TensorProto.FLOAT,
        dims=[channels],
        vals=np.ones(channels).astype(np.float32).tolist()
    )
    return [scale, bias, mean, variance]

# BatchNorm initializers
bn1_initializers = create_batchnorm_initializers('BN1', 16)
bn2_initializers = create_batchnorm_initializers('BN2', 32)

# Define the layers (Conv, BatchNorm, ReLU, etc.)
conv1 = helper.make_node(
    'Conv', 
    inputs=['Input', 'Conv1_W'], 
    outputs=['Conv1'], 
    kernel_shape=[3, 3], 
    pads=[1, 1, 1, 1], 
    strides=[1, 1]
)
bn1 = helper.make_node(
    'BatchNormalization', 
    inputs=['Conv1', 'BN1_Scale', 'BN1_Bias', 'BN1_Mean', 'BN1_Variance'], 
    outputs=['BN1'], 
    epsilon=1e-5
)
relu1 = helper.make_node('Relu', inputs=['BN1'], outputs=['ReLU1'])

conv2 = helper.make_node(
    'Conv', 
    inputs=['ReLU1', 'Conv2_W'], 
    outputs=['Conv2'], 
    kernel_shape=[3, 3], 
    pads=[1, 1, 1, 1], 
    strides=[1, 1]
)
bn2 = helper.make_node(
    'BatchNormalization', 
    inputs=['Conv2', 'BN2_Scale', 'BN2_Bias', 'BN2_Mean', 'BN2_Variance'], 
    outputs=['BN2'], 
    epsilon=1e-5
)
relu2 = helper.make_node('Relu', inputs=['BN2'], outputs=['ReLU2'])

conv3 = helper.make_node(
    'Conv', 
    inputs=['ReLU2', 'Conv3_W'], 
    outputs=['Conv3'], 
    kernel_shape=[3, 3], 
    pads=[1, 1, 1, 1], 
    strides=[1, 1]
)

add = helper.make_node('Add', inputs=['BN2', 'Conv3'], outputs=['Add'])
relu3 = helper.make_node('Relu', inputs=['Add'], outputs=['ReLU3'])

fc = helper.make_node('Gemm', inputs=['ReLU3'], outputs=['FC'], alpha=1.0, beta=1.0)
output = helper.make_node('Identity', inputs=['FC'], outputs=['Output'])

# Create the graph
graph = helper.make_graph(
    nodes=[conv1, bn1, relu1, conv2, bn2, relu2, conv3, add, relu3, fc, output],
    name='ParallelizableModel',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[conv1_weights, conv2_weights, conv3_weights] + bn1_initializers + bn2_initializers
)

# Create the model
model = helper.make_model(graph, producer_name='custom_demo_model')

# Check model correctness
checker.check_model(model)

# Save the model to a file
onnx_file_path = 'parallel_demo_model.onnx'
onnx.save(model, onnx_file_path)

print(f"ONNX model has been saved to {onnx_file_path}")
