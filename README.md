# TensorRT_Python_API
用TensorRT的Python API去搭建trt网络

执行脚本`python3 sample.py`, 将会对网络进行训练, 训练完成之后, 将会在当前目录下将模型的所有参数保存成名为`torchPara.npz`的文件, 同时还会将PyTorch模型导出成名为`test.onnx`的onnx模型.

再执行`python3 construct_your_model.py`就可以采用TensorRT的原生API搭建模型了, 并将最后的模型序列化成`mynet.engine`文件. 该TRT模型是支持动态BatchSize的.

通过trtexec命令将生成的onnx文件也转成trt文件.

```bash_script
trtexec --onnx=test.onnx \
        --workspace=1600 \
        --explicitBatch \
        --optShapes=data:4x1x28x28 \
        --maxShapes=data:8x4x28x28 \
        --minShapes=data:1x4x28x28 \
        --shapes=data:4x3x28x28 \
        --saveEngine=test.engine
```
