import tvm
from tvm import relay
import torch

# Carrega o modelo PyTorch treinado
# ex: model = torch.load("../model/iaes_autoencoder.pth")
class DummyIAES(torch.nn.Module):
    def forward(self, x): return x
model = DummyIAES().eval()

# Define o formato de entrada (Batch=1, Features=?)
# Substitua o 'X' abaixo pelo número exato de features do seu IAES
NUM_FEATURES = 10 # <-- Altere aqui para o tamanho real do seu vetor do IAES
input_shape = [1, NUM_FEATURES]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Conversão para o compilador TVM
shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Alvo de Hardware (Cortex-A72 / BCM2711)
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
print(f"Compilando Modelo IAES para: {target.kind.name}...")

# Otimização e Exportação
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

lib.export_library("tvm_iaes.so")
print("Sucesso! Arquivo 'tvm_iaes.so' gerado.")