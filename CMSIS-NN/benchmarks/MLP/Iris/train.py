import os
import json
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

DATASET_PATH = "../../../../datasets/MLP/iris/iris.data"
MODEL_DIR    = "model"
EPOCHS       = 500
LR           = 0.01
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
INPUT_SCALE  = 1.0 / 127.0

MLP_CONFIGS = {
    "small":  [4, 32, 64, 3],
    "medium": [4, 64, 128, 64, 3],
    "large":  [4, 128, 256, 128, 64, 3],
}
ACTS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid}

os.makedirs(MODEL_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_iris(path):
    features, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 5:
                continue
            features.append([float(x) for x in parts[:4]])
            labels.append(CLASS_NAMES.index(parts[4]))
    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int64)
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
    with open(path, "w") as f:
        f.write(f"{n} {nf} 1\n")
        for xi, yi in zip(X, y):
            f.write(" ".join(f"{v:.6f}" for v in xi) + f" {int(yi)}\n")


def quantize_weights(W):
    abs_max = np.max(np.abs(W))
    if abs_max == 0:
        return np.zeros_like(W, dtype=np.int8), 1.0
    scale = abs_max / 127.0
    return np.clip(np.round(W / scale), -128, 127).astype(np.int8), scale


def quantize_bias(b, in_scale, w_scale):
    bias_scale = in_scale * w_scale
    if bias_scale == 0:
        return np.zeros_like(b, dtype=np.int32)
    return np.clip(np.round(b / bias_scale), -(2**31), 2**31 - 1).astype(np.int32)


def array_to_c(name, arr, dtype):
    lines = [f"static const {dtype} {name}[] = {{"]
    flat = arr.flatten()
    row = "    "
    for i, v in enumerate(flat):
        row += f"{int(v)}, "
        if (i + 1) % 12 == 0:
            lines.append(row.rstrip())
            row = "    "
    if row.strip():
        lines.append(row.rstrip())
    lines.append("};")
    return "\n".join(lines)


X, y = load_iris(DATASET_PATH)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
save_test_data(os.path.join(MODEL_DIR, "test_data.txt"), X_test, y_test)
print(f"Test split saved: {len(y_test)} samples")

metrics = {}

for size, dims in MLP_CONFIGS.items():
    for act_name, act_cls in ACTS.items():
        key      = f"mlp_{size}_{act_name}"
        variant  = f"mlp_{size}_{act_name}"
        model    = build_mlp(dims, act_cls)
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

        n_fc = len(dims) - 1
        weights, biases, w_scales = [], [], []
        in_scale = INPUT_SCALE
        fc_layers = [(m.weight.detach().numpy(), m.bias.detach().numpy())
                     for m in model if isinstance(m, nn.Linear)]
        for W_f, b_f in fc_layers:
            Wq, ws = quantize_weights(W_f)
            bq = quantize_bias(b_f, in_scale, ws)
            weights.append(Wq)
            biases.append(bq)
            w_scales.append(ws)
            in_scale = ws

        weights_path = os.path.join(MODEL_DIR, f"{variant}_weights.h")
        params_path  = os.path.join(MODEL_DIR, f"{variant}_params.h")

        with open(weights_path, "w") as f:
            guard = variant.upper() + "_WEIGHTS_H"
            f.write(f"#ifndef {guard}\n#define {guard}\n#include <stdint.h>\n\n")
            for i, (Wq, bq) in enumerate(zip(weights, biases)):
                f.write(array_to_c(f"fc{i}_weights", Wq, "int8_t") + "\n\n")
                f.write(array_to_c(f"fc{i}_bias", bq, "int32_t") + "\n\n")
            f.write("static const int8_t * const FC_WEIGHTS[] = {")
            f.write(", ".join(f"fc{i}_weights" for i in range(n_fc)))
            f.write("};\n")
            f.write("static const int32_t * const FC_BIAS[] = {")
            f.write(", ".join(f"fc{i}_bias" for i in range(n_fc)))
            f.write("};\n\n#endif\n")

        with open(params_path, "w") as f:
            guard = variant.upper() + "_PARAMS_H"
            f.write(f"#ifndef {guard}\n#define {guard}\n\n")
            f.write(f"#define NUM_FC_LAYERS {n_fc}\n")
            f.write(f"#define INPUT_SCALE_F32 {INPUT_SCALE:.10f}f\n")
            f.write("static const int LAYER_SIZES[] = {")
            f.write(", ".join(str(d) for d in dims))
            f.write("};\n")
            mults, shifts = [], []
            for ws in w_scales:
                m = int(round(ws * (1 << 20)))
                mults.append(m)
                shifts.append(-20)
            f.write("static const int32_t FC_MULTIPLIERS[] = {")
            f.write(", ".join(str(m) for m in mults))
            f.write("};\n")
            f.write("static const int FC_SHIFTS[] = {")
            f.write(", ".join(str(s) for s in shifts))
            f.write("};\n")
            f.write(f"#define USE_SIGMOID_OUTPUT {'1' if act_name == 'sigmoid' else '0'}\n")
            f.write("\n#endif\n")

        print(f"  Generated {weights_path}, {params_path}")

with open(os.path.join(MODEL_DIR, "training_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved training_metrics.json")
