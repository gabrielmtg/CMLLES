import tvm
from tvm import relay
import torch
import numpy as np

# Carrega o modelo PyTorch treinado
# ex: model = torch.load("../model/vww_cnn.pth")
class DummyCNN(torch.nn.Module):
    def forward(self, x): return x
model = DummyCNN().eval()

# Define o formato de entrada (Batch=1, Canais=3, Alt=32, Larg=32)
# Nota: Se o seu VWW em C usou tons de cinza, mude os canais para 1.
input_shape = [1, 3, 32, 32]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Conversão para o compilador TVM
shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Alvo de Hardware (Cortex-A72 / Raspberry Pi 4)
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
print(f"Compilando CNN VWW para: {target.kind.name}...")

# Otimização e Exportação
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

lib.export_library("tvm_vww.so")
print("Sucesso! Arquivo 'tvm_vww.so' gerado.")