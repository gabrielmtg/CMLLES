import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DATASET_PATH = "../../../datasets/iris/iris.data"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "iris_mlp.onnx")
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

    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    rng = xmax - xmin
    rng[rng == 0] = 1.0
    X = (X - xmin) / rng

    return X, y


class IrisMLP(nn.Module):
    def __init__(self, input_size=4, hidden_size=HIDDEN_SIZE, output_size=3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("=" * 50)
    print("  Iris MLP — Treinamento PyTorch → ONNX")
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
    dummy = torch.randn(1, 4)
    torch.onnx.export(
        model, dummy, MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=11
    )
    print(f"Modelo exportado: {MODEL_PATH}")
    print(f"Tamanho: {os.path.getsize(MODEL_PATH)} bytes")


if __name__ == "__main__":
    main()
