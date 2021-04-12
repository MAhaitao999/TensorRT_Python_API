# TensorRT_Python_API
用TensorRT的Python API去搭建trt网络

执行脚本`python3 sample.py`, 将会对网络进行训练, 训练完成之后, 将会在当前目录下将模型的所有参数保存成名为`torchPara.npz`的文件, 同时还会将PyTorch模型导出成名为`test.onnx`的onnx模型.

再执行`python3 construct_your_model.py`就可以将采用TensorRT的原生API搭建模型了, 并将最后的模型序列化成`mynet.engine`文件. 该TRT模型是支持动态BatchSize的.
