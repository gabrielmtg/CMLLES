import os
import subprocess
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATASET_PATH = "../../../datasets/iris/iris.data"
MODEL_DIR    = "model"
HIDDEN_SIZE  = 16
EPOCHS       = 500
LR           = 0.01
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


def load_iris(path):
    features = []
    labels = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            feat = [float(x) for x in parts[:4]]
            label = CLASS_NAMES.index(parts[4])
            features.append(feat)
            labels.append(label)
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    # Normalização min-max [0, 1]
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    rng = xmax - xmin
    rng[rng == 0] = 1.0
    X = (X - xmin) / rng

    return X, y


class IrisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, 3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 50)
    print("  Iris MLP — Treinamento PyTorch → NCNN")
    print("=" * 50)

    X, y = load_iris(DATASET_PATH)
    print(f"\nDataset: {len(y)} amostras carregadas")

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = IrisMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print(f"Treinando por {EPOCHS} épocas...")
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            preds = outputs.argmax(dim=1)
            acc = (preds == y_t).float().mean().item() * 100
            print(f"  Época {epoch+1:4d}: loss={loss.item():.4f}, acurácia={acc:.1f}%")

    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1)
        acc = (preds == y_t).float().mean().item() * 100
    print(f"\nAcurácia final: {acc:.1f}%")

    os.makedirs(MODEL_DIR, exist_ok=True)
    onnx_path = os.path.join(MODEL_DIR, "iris_mlp.onnx")
    dummy = torch.randn(1, 4)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )
    print(f"\nONNX exportado: {onnx_path}")

    param_path = os.path.join(MODEL_DIR, "iris_mlp.param")
    bin_path   = os.path.join(MODEL_DIR, "iris_mlp.bin")

    for tool in ["onnx2ncnn", "pnnx"]:
        try:
            if tool == "onnx2ncnn":
                subprocess.run(
                    [tool, onnx_path, param_path, bin_path],
                    check=True, capture_output=True
                )
            else:
                subprocess.run(
                    [tool, onnx_path, f"inputshape=[1,4]"],
                    check=True, capture_output=True, cwd=MODEL_DIR
                )
            print(f"NCNN convertido com {tool}: {param_path}, {bin_path}")
            return
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue

    print("\nAVISO: onnx2ncnn/pnnx não encontrados. Gerando .param/.bin manualmente...")
    generate_ncnn_manual(model, param_path, bin_path)


def generate_ncnn_manual(model, param_path, bin_path):
    import struct

    fc1_w = model.fc1.weight.data.numpy()   # [16, 4]
    fc1_b = model.fc1.bias.data.numpy()     # [16]
    fc2_w = model.fc2.weight.data.numpy()   # [3, 16]
    fc2_b = model.fc2.bias.data.numpy()     # [3]

    with open(param_path, "w") as f:
        f.write("7767517\n")
        f.write("5 5\n")
        f.write("Input            input    0 1 input\n")
        f.write("InnerProduct     fc1      1 1 input fc1_out 0=16 1=1 2=64\n")
        f.write("ReLU             relu1    1 1 fc1_out relu1_out\n")
        f.write("InnerProduct     fc2      1 1 relu1_out fc2_out 0=3 1=1 2=48\n")
        f.write("Softmax          softmax  1 1 fc2_out output 0=0\n")

    with open(bin_path, "wb") as f:
        f.write(struct.pack('I', 0))
        for row in fc1_w:
            for v in row:
                f.write(struct.pack('f', float(v)))
        for v in fc1_b:
            f.write(struct.pack('f', float(v)))

        f.write(struct.pack('I', 0))
        for row in fc2_w:
            for v in row:
                f.write(struct.pack('f', float(v)))
        for v in fc2_b:
            f.write(struct.pack('f', float(v)))

    print(f"  {param_path} ({os.path.getsize(param_path)} bytes)")
    print(f"  {bin_path} ({os.path.getsize(bin_path)} bytes)")


if __name__ == "__main__":
    main()
