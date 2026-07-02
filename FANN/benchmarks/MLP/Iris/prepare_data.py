import os
import numpy as np
from sklearn.model_selection import train_test_split

DATASET_PATH = "../../../../datasets/MLP/iris/iris.data"
MODEL_DIR    = "model"
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

os.makedirs(MODEL_DIR, exist_ok=True)
np.random.seed(SEED)

features, labels = [], []
with open(DATASET_PATH) as f:
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
X = (X - xmin) / rng

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

Y_train = np.eye(3, dtype=np.float32)[y_train]
with open(os.path.join(MODEL_DIR, "iris_fann_train.data"), "w") as f:
    f.write(f"{len(X_train)} 4 3\n")
    for xi, yi in zip(X_train, Y_train):
        f.write(" ".join(f"{v:.6f}" for v in xi) + "\n")
        f.write(" ".join(f"{v:.6f}" for v in yi) + "\n")

with open(os.path.join(MODEL_DIR, "test_data.txt"), "w") as f:
    f.write(f"{len(X_test)} 4 1\n")
    for xi, yi in zip(X_test, y_test):
        f.write(" ".join(f"{v:.6f}" for v in xi) + f" {int(yi)}\n")

print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
print(f"Saved: {MODEL_DIR}/iris_fann_train.data, {MODEL_DIR}/test_data.txt")
