import os
import sys

from PIL import Image
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

import model


class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (1, 28, 28)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32


def main():
    mnist_model = model.MnistModel()
    mnist_model.learn()
    weights = mnist_model.get_weights()
    torchPara = {}
    for name, para in mnist_model.network.named_parameters():
        torchPara[name] = para.detach().numpy()
    np.savez('./torchPara.npz', **torchPara)
    print(weights)
    mnist_model.export_to_onnx()


if __name__ == '__main__':
    main()

