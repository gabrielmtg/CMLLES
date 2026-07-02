import os
import torch
import torch.nn as nn
import pnnx

SEED = 42
torch.manual_seed(SEED)

MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

AE_CONFIGS = {
    "small":  [64, 32, 16],
    "medium": [128, 64, 32],
    "large":  [256, 128, 64],
}


def build_autoencoder(dims):
    enc = list(dims)
    dec = list(reversed(dims[:-1]))
    all_dims = enc + dec
    layers = []
    for i in range(len(all_dims) - 1):
        layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))
        if i < len(all_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


for size, dims in AE_CONFIGS.items():
    model = build_autoencoder(dims).eval()
    dummy = torch.randn(1, dims[0])
    pt_path = os.path.join(MODEL_DIR, f"ae_{size}_f32.pt")
    torch.jit.save(torch.jit.trace(model, dummy), pt_path)
    pnnx.export(model, pt_path, dummy)
    param_src = pt_path.replace(".pt", ".ncnn.param")
    if os.path.exists(param_src):
        print(f"Exported {param_src}")
    else:
        print(f"WARNING: pnnx did not produce {param_src}")
