# pnnx model stat
# model inputshape = [1,122]f32
# FLOPS = 105.526K
# memory OPS = 54.402K

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

        self.0 = nn.Linear(bias=True, in_features=122, out_features=128)
        self.1 = nn.LeakyReLU(negative_slope=0.1)
        self.2 = nn.Linear(bias=True, in_features=128, out_features=64)
        self.3 = nn.LeakyReLU(negative_slope=0.1)
        self.4 = nn.Linear(bias=True, in_features=64, out_features=32)
        self.5 = nn.LeakyReLU(negative_slope=0.1)
        self.6 = nn.Linear(bias=True, in_features=32, out_features=64)
        self.7 = nn.LeakyReLU(negative_slope=0.1)
        self.8 = nn.Linear(bias=True, in_features=64, out_features=128)
        self.9 = nn.LeakyReLU(negative_slope=0.1)
        self.10 = nn.Linear(bias=True, in_features=128, out_features=122)
        self.11 = nn.Sigmoid()

        archive = zipfile.ZipFile('model/ae_large_f32.pnnx.bin', 'r')
        self.0.bias = self.load_pnnx_bin_as_parameter(archive, '0.bias', (128), 'float32')
        self.0.weight = self.load_pnnx_bin_as_parameter(archive, '0.weight', (128,122), 'float32')
        self.2.bias = self.load_pnnx_bin_as_parameter(archive, '2.bias', (64), 'float32')
        self.2.weight = self.load_pnnx_bin_as_parameter(archive, '2.weight', (64,128), 'float32')
        self.4.bias = self.load_pnnx_bin_as_parameter(archive, '4.bias', (32), 'float32')
        self.4.weight = self.load_pnnx_bin_as_parameter(archive, '4.weight', (32,64), 'float32')
        self.6.bias = self.load_pnnx_bin_as_parameter(archive, '6.bias', (64), 'float32')
        self.6.weight = self.load_pnnx_bin_as_parameter(archive, '6.weight', (64,32), 'float32')
        self.8.bias = self.load_pnnx_bin_as_parameter(archive, '8.bias', (128), 'float32')
        self.8.weight = self.load_pnnx_bin_as_parameter(archive, '8.weight', (128,64), 'float32')
        self.10.bias = self.load_pnnx_bin_as_parameter(archive, '10.bias', (122), 'float32')
        self.10.weight = self.load_pnnx_bin_as_parameter(archive, '10.weight', (122,128), 'float32')
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
        v_1 = self.0(v_0)
        v_2 = self.1(v_1)
        v_3 = self.2(v_2)
        v_4 = self.3(v_3)
        v_5 = self.4(v_4)
        v_6 = self.5(v_5)
        v_7 = self.6(v_6)
        v_8 = self.7(v_7)
        v_9 = self.8(v_8)
        v_10 = self.9(v_9)
        v_11 = self.10(v_10)
        v_12 = self.11(v_11)
        return v_12

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 122, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("model/ae_large_f32_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 122, dtype=torch.float)

    torch.onnx.export(net, v_0, "model/ae_large_f32_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 122, dtype=torch.float)

    import pnnx
    pnnx.export(net, "model/ae_large_f32_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 122, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
