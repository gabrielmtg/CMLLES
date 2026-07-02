# pnnx model stat
# model inputshape = [1,3,32,32]f32
# FLOPS = 20.804M
# memory OPS = 350.788K

import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.net_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.net_1 = nn.ReLU()
        self.net_2 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.net_3 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.net_4 = nn.ReLU()
        self.net_5 = nn.MaxPool2d(ceil_mode=False, dilation=(1,1), kernel_size=(2,2), padding=(0,0), return_indices=False, stride=(2,2))
        self.net_6 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.net_7 = nn.ReLU()
        self.net_8 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.net_10 = nn.Linear(bias=True, in_features=128, out_features=2)

        archive = zipfile.ZipFile('model/cnn_intermediate_f32.pnnx.bin', 'r')
        self.net_0.bias = self.load_pnnx_bin_as_parameter(archive, 'net.0.bias', (32), 'float32')
        self.net_0.weight = self.load_pnnx_bin_as_parameter(archive, 'net.0.weight', (32,3,3,3), 'float32')
        self.net_3.bias = self.load_pnnx_bin_as_parameter(archive, 'net.3.bias', (64), 'float32')
        self.net_3.weight = self.load_pnnx_bin_as_parameter(archive, 'net.3.weight', (64,32,3,3), 'float32')
        self.net_6.bias = self.load_pnnx_bin_as_parameter(archive, 'net.6.bias', (128), 'float32')
        self.net_6.weight = self.load_pnnx_bin_as_parameter(archive, 'net.6.weight', (128,64,3,3), 'float32')
        self.net_10.bias = self.load_pnnx_bin_as_parameter(archive, 'net.10.bias', (2), 'float32')
        self.net_10.weight = self.load_pnnx_bin_as_parameter(archive, 'net.10.weight', (2,128), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.net_0(v_0)
        v_2 = self.net_1(v_1)
        v_3 = self.net_2(v_2)
        v_4 = self.net_3(v_3)
        v_5 = self.net_4(v_4)
        v_6 = self.net_5(v_5)
        v_7 = self.net_6(v_6)
        v_8 = self.net_7(v_7)
        v_9 = self.net_8(v_8)
        v_10 = torch.flatten(v_9, end_dim=-1, start_dim=1)
        v_11 = self.net_10(v_10)
        return v_11

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("model/cnn_intermediate_f32_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    torch.onnx.export(net, v_0, "model/cnn_intermediate_f32_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    import pnnx
    pnnx.export(net, "model/cnn_intermediate_f32_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 32, 32, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
