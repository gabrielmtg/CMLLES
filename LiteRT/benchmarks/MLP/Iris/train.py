import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import litert_torch as ai_edge_torch

DATASET_PATH = "../../../../datasets/MLP/iris/iris.data"
MODEL_DIR    = "model"
EPOCHS       = 500
LR           = 0.01
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

MLP_CONFIGS = {
    "small":  [4, 32, 64, 3],
    "medium": [4, 64, 128, 64, 3],
    "large":  [4, 128, 256, 128, 64, 3],
}
ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}


def load_iris(path):
    features, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(",")
            if len(parts) != 5: continue
            features.append([float(x) for x in parts[:4]])
            labels.append(CLASS_NAMES.index(parts[4]))
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    xmin, xmax = X.min(axis=0), X.max(axis=0)
    rng = np.where(xmax - xmin == 0, 1.0, xmax - xmin)
    return (X - xmin) / rng, y


def build_mlp(dims, act_cls):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act_cls())
    return nn.Sequential(*layers)


def save_test_data(path, X, y):
    n, nf = X.shape
    with open(path, "w") as fout:
        fout.write(f"{n} {nf} 1\n")
        for xi, yi in zip(X, y):
            fout.write(" ".join(f"{v:.6f}" for v in xi) + f" {int(yi)}\n")


os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

X, y = load_iris(DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
save_test_data(os.path.join(MODEL_DIR, "test_data.txt"), X_test, y_test)
print(f"Test split saved: {len(y_test)} samples")

metrics = {}

for size, dims in MLP_CONFIGS.items():
    for act_name, act_cls in ACTS.items():
        key = f"mlp_{size}_{act_name}"
        model = build_mlp(dims, act_cls)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)
        X_t = torch.from_numpy(X_train)
        y_t = torch.from_numpy(y_train)

        t0 = time.time()
        final_loss = 0.0
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
        elapsed = time.time() - t0

        metrics[key] = {
            "training_time_s": round(elapsed, 3),
            "epochs": EPOCHS,
            "final_train_loss": round(final_loss, 6),
        }
        print(f"{key}: loss={final_loss:.4f} time={elapsed:.1f}s")

        dummy = (torch.randn(1, dims[0]),)
        path = os.path.join(MODEL_DIR, f"mlp_{size}_{act_name}_f32.tflite")
        edge_model = ai_edge_torch.convert(model.eval(), dummy)
        edge_model.export(path)
        print(f"  Exported {path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
