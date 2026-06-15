import tvm
from tvm import relay
import torch
import numpy as np

# Carrega o modelo PyTorch treinado
# Substitua o DummyModel pelo carregamento real do seu modelo KDD
# ex: model = torch.load("../model/nslkdd_autoencoder.pth")
class DummyKDD(torch.nn.Module):
    def forward(self, x): return x
model = DummyKDD().eval()

# Define o formato de entrada (Batch=1, Features=122)
input_shape = [1, 122]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Conversão para o compilador TVM
shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Alvo de Hardware (Cortex-A72 / Raspberry Pi 4)
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
print(f"Compilando Autoencoder NSL-KDD para: {target.kind.name}...")

# Otimização e Exportação
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

lib.export_library("tvm_nslkdd.so")
print("Sucesso! Arquivo 'tvm_nslkdd.so' gerado.")