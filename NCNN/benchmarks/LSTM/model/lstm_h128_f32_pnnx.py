# pnnx model stat
# model inputshape = [1,30,14]f32
# FLOPS = 4.432M
# memory OPS = 78.246K

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

        self.lstm = nn.LSTM(batch_first=True, bias=True, bidirectional=False, hidden_size=128, input_size=14, num_layers=1, proj_size=0)
        self.fc = nn.Linear(bias=True, in_features=128, out_features=1)

        archive = zipfile.ZipFile('model/lstm_h128_f32.pnnx.bin', 'r')
        self.lstm.bias_hh_l0 = self.load_pnnx_bin_as_parameter(archive, 'lstm.bias_hh_l0', (512), 'float32')
        self.lstm.bias_ih_l0 = self.load_pnnx_bin_as_parameter(archive, 'lstm.bias_ih_l0', (512), 'float32')
        self.lstm.weight_hh_l0 = self.load_pnnx_bin_as_parameter(archive, 'lstm.weight_hh_l0', (512,128), 'float32')
        self.lstm.weight_ih_l0 = self.load_pnnx_bin_as_parameter(archive, 'lstm.weight_ih_l0', (512,14), 'float32')
        self.fc.bias = self.load_pnnx_bin_as_parameter(archive, 'fc.bias', (1), 'float32')
        self.fc.weight = self.load_pnnx_bin_as_parameter(archive, 'fc.weight', (1,128), 'float32')
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
        v_1, _ = self.lstm(v_0)
        v_2 = v_1.select(dim=1, index=-1)
        v_3 = self.fc(v_2)
        return v_3

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 30, 14, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("model/lstm_h128_f32_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 30, 14, dtype=torch.float)

    torch.onnx.export(net, v_0, "model/lstm_h128_f32_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

def export_pnnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 30, 14, dtype=torch.float)

    import pnnx
    pnnx.export(net, "model/lstm_h128_f32_pnnx.py.pt", v_0)

def export_ncnn():
    export_pnnx()

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 30, 14, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
