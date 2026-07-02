# pnnx model stat
# model inputshape = [1,3,96,96]f32
# FLOPS = 46.228M
# memory OPS = 2.27M

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

        self.net_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=3, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.net_1 = nn.ReLU()
        self.net_2_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=32, in_channels=32, kernel_size=(3,3), out_channels=32, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.net_2_1 = nn.ReLU()
        self.net_2_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=32, kernel_size=(1,1), out_channels=64, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.net_2_3 = nn.ReLU()
        self.net_3_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=64, in_channels=64, kernel_size=(3,3), out_channels=64, padding=(1,1), padding_mode='zeros', stride=(2,2))
        self.net_3_1 = nn.ReLU()
        self.net_3_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=64, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.net_3_3 = nn.ReLU()
        self.net_4_0 = nn.Conv2d(bias=True, dilation=(1,1), groups=128, in_channels=128, kernel_size=(3,3), out_channels=128, padding=(1,1), padding_mode='zeros', stride=(1,1))
        self.net_4_1 = nn.ReLU()
        self.net_4_2 = nn.Conv2d(bias=True, dilation=(1,1), groups=1, in_channels=128, kernel_size=(1,1), out_channels=128, padding=(0,0), padding_mode='zeros', stride=(1,1))
        self.net_4_3 = nn.ReLU()
        self.net_5 = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.net_7 = nn.Linear(bias=True, in_features=128, out_features=2)

        archive = zipfile.ZipFile('model/cnn_mobilenet_f32.pnnx.bin', 'r')
        self.net_0.bias = self.load_pnnx_bin_as_parameter(archive, 'net.0.bias', (32), 'float32')
        self.net_0.weight = self.load_pnnx_bin_as_parameter(archive, 'net.0.weight', (32,3,3,3), 'float32')
        self.net_2_0.bias = self.load_pnnx_bin_as_parameter(archive, 'net.2.0.bias', (32), 'float32')
        self.net_2_0.weight = self.load_pnnx_bin_as_parameter(archive, 'net.2.0.weight', (32,1,3,3), 'float32')
        self.net_2_2.bias = self.load_pnnx_bin_as_parameter(archive, 'net.2.2.bias', (64), 'float32')
        self.net_2_2.weight = self.load_pnnx_bin_as_parameter(archive, 'net.2.2.weight', (64,32,1,1), 'float32')
        self.net_3_0.bias = self.load_pnnx_bin_as_parameter(archive, 'net.3.0.bias', (64), 'float32')
        self.net_3_0.weight = self.load_pnnx_bin_as_parameter(archive, 'net.3.0.weight', (64,1,3,3), 'float32')
        self.net_3_2.bias = self.load_pnnx_bin_as_parameter(archive, 'net.3.2.bias', (128), 'float32')
        self.net_3_2.weight = self.load_pnnx_bin_as_parameter(archive, 'net.3.2.weight', (128,64,1,1), 'float32')
        self.net_4_0.bias = self.load_pnnx_bin_as_parameter(archive, 'net.4.0.bias', (128), 'float32')
        self.net_4_0.weight = self.load_pnnx_bin_as_parameter(archive, 'net.4.0.weight', (128,1,3,3), 'float32')
        self.net_4_2.bias = self.load_pnnx_bin_as_parameter(archive, 'net.4.2.bias', (128), 'float32')
        self.net_4_2.weight = self.load_pnnx_bin_as_parameter(archive, 'net.4.2.weight', (128,128,1,1), 'float32')
        self.net_7.bias = self.load_pnnx_bin_as_parameter(archive, 'net.7.bias', (2), 'float32')
        self.net_7.weight = self.load_pnnx_bin_as_parameter(archive, 'net.7.weight', (2,128), 'float32')
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
        v_3 = self.net_2_0(v_2)
        v_4 = self.net_2_1(v_3)
        v_5 = self.net_2_2(v_4)
        v_6 = self.net_2_3(v_5)
        v_7 = self.net_3_0(v_6)
        v_8 = self.net_3_1(v_7)
        v_9 = self.net_3_2(v_8)
        v_10 = self.net_3_3(v_9)
        v_11 = self.net_4_0(v_10)
        v_12 = self.net_4_1(v_11)
        v_13 = self.net_4_2(v_12)
        v_14 = self.net_4_3(v_13)
        v_15 = self.net_5(v_14)
        v_16 = torch.flatten(v_15, end_dim=-1, start_dim=1)
        v_17 = self.net_7(v_16)
        return v_17

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 96, 96, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("model/cnn_mobilenet_f32_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 96, 96, dtype=torch.float)

    torch.onnx.export(net, v_0, "model/cnn_mobilenet_f32_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 96, 96, dtype=torch.float)

    import pnnx
    pnnx.export(net, "model/cnn_mobilenet_f32_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 3, 96, 96, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
