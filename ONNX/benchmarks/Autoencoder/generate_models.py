import os
import torch
import torch.nn as nn

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
    path = os.path.join(MODEL_DIR, f"ae_{size}_f32.onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
    )
    print(f"Exported {path} ({os.path.getsize(path)} bytes)")
