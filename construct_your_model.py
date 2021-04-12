import os

from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt


# You can set the logger severity higher to suppress messages (or lower to display more messages).
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def populate_network(network, weights_path):
    params = np.load(weights_path)
    print(params['conv1.weight'])

    input_tensor = network.add_input(name='data', dtype=trt.float32, shape=(-1, 1, 28, 28))
    
    # construct the first convolutional layer
    conv1_w = params['conv1.weight']
    conv1_b = params['conv1.bias']
    conv1 = network.add_convolution(input=input_tensor, num_output_maps=20, kernel_shape=(5, 5), kernel=conv1_w, bias=conv1_b)
    conv1.stride = (1, 1)
    print('===================conv1 output shape is: ', conv1.get_output(0).shape)

    # construct the first pooling layer
    pool1 = network.add_pooling(input=conv1.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool1.stride = (2, 2)

    # construct the second convolutional layer
    conv2_w = params['conv2.weight']
    conv2_b = params['conv2.bias']
    conv2 = network.add_convolution(input=pool1.get_output(0), num_output_maps=50, kernel_shape=(5, 5), kernel=conv2_w, bias=conv2_b)
    conv2.stride = (1, 1)
    print('===================conv2 output shape is: ', conv2.get_output(0).shape)

    # construct the second pooling layer
    pool2 = network.add_pooling(input=conv2.get_output(0), type=trt.PoolingType.MAX, window_size=(2, 2))
    pool2.stride = (2, 2)
    
    # construct the first full connected layer
    fc1_w = params['fc1.weight']
    fc1_b = params['fc1.bias']
    fc1 = network.add_fully_connected(input=pool2.get_output(0), num_outputs=500, kernel=fc1_w, bias=fc1_b)

    # construct the first relu layer
    relu1 = network.add_activation(input=fc1.get_output(0), type=trt.ActivationType.RELU)
    
    # construct the second full connected layer
    fc2_w = params['fc2.weight']
    fc2_b = params['fc2.bias']
    fc2 = network.add_fully_connected(input=relu1.get_output(0), num_outputs=10, kernel=fc2_w, bias=fc2_b)

    fc2.get_output(0).name = 'prob'
    network.mark_output(tensor=fc2.get_output(0))

    return network


if __name__ == '__main__':

    weights_path = 'torchPara.npz'
    if os.path.isfile('./mynet.engine'):
        print('engine found!')
        with open('./mynet.engine', 'rb') as f:
            engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine( f.read() )
    else:
        print('engine not found')
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:
            # builder.max_workspace_size = 1 << 20
            # Populate the network using weights from the PyTorch model.
            network = populate_network(network, weights_path)
            profile = builder.create_optimization_profile()  # 需要profile
            config = builder.create_builder_config()         # 需要config
            config.max_workspace_size = 1 << 30              # workspace等需要在config中调整, 不再在builder中调整
            profile.set_shape(network.get_input(0).name, (1, 1, 28, 28), (4, 1, 28, 28), (8, 1, 28, 28))
            config.add_optimization_profile(profile)
            engine = builder.build_engine(network, config)
            with open('./mynet.engine', 'wb') as f:
                f.write( engine.serialize() )

