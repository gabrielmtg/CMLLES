import tvm
from tvm import relay
import torch

# Carrega o modelo PyTorch treinado
# ex: model = torch.load("../model/iris_mlp.pth")
class DummyIris(torch.nn.Module):
    def forward(self, x): return x
model = DummyIris().eval()

# Define o formato de entrada (Batch=1, Features=4)
input_shape = [1, 4]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

# Conversão para o compilador TVM
shape_list = [("input0", input_shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

# Alvo de Hardware (Cortex-A72 / BCM2711)
target = tvm.target.Target("llvm -mtriple=aarch64-linux-gnu -mcpu=cortex-a72")
print(f"Compilando MLP Iris para: {target.kind.name}...")

# Otimização e Exportação
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

lib.export_library("tvm_iris.so")
print("Sucesso! Arquivo 'tvm_iris.so' gerado.")