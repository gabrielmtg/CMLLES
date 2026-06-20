import tvm
from tvm import relax
import torch
from tvm.relax.frontend.torch import from_fx
import numpy as np

# Carrega o modelo PyTorch treinado
# ex: model = torch.load("../model/cmapss_gru.pth")
class DummyGRU(torch.nn.Module):
    def forward(self, x): return x
model = DummyGRU().eval()

# Define o formato de entrada (Batch=1, Sequência/Ciclos=30, Features/Sensores=14)
input_shape = [1, 30, 14]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Conversão para o compilador TVM
shape_list = [("input0", input_shape)]
mod, params = relax.frontend.from_pytorch(scripted_model, shape_list)

# Alvo de Hardware (Cortex-A72 / Raspberry Pi 4)
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
print(f"Compilando GRU CMAPSS para: {target.kind.name}...")

# Otimização e Exportação
with tvm.transform.PassContext(opt_level=3):
    lib = relax.build(mod, target=target) #params=params

lib.export_library("tvm_cmapss.so")
print("Sucesso! Arquivo 'tvm_cmapss.so' gerado.")
