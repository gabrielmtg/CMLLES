import os
import numpy as np
from sklearn.model_selection import train_test_split
import ai_edge_quantizer as aq

DATASET_PATH = "../../../../datasets/MLP/iris/iris.data"
MODEL_DIR    = "model"
CLASS_NAMES  = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

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
y = np.array(labels, dtype=np.int64)
xmin, xmax = X.min(axis=0), X.max(axis=0)
rng = np.where(xmax - xmin == 0, 1.0, xmax - xmin)
X = (X - xmin) / rng
X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

for size in ["small", "medium", "large"]:
    for act in ["relu", "sigmoid"]:
        f32 = os.path.join(MODEL_DIR, f"mlp_{size}_{act}_f32.tflite")
        i8  = os.path.join(MODEL_DIR, f"mlp_{size}_{act}_int8.tflite")
        if not os.path.exists(f32):
            print(f"SKIP {f32} (not found)")
            continue
        quantizer = aq.Quantizer(bytearray(open(f32, "rb").read()))
        quantizer.load_quantization_recipe(aq.recipe.dynamic_wi8_afp32())
        calib = {"serving_default": [{"args_0": X_train[i:i+1]} for i in range(len(X_train))]}
        i8_dir, i8_name = os.path.split(i8)
        quantizer.quantize(quantizer.calibrate(calib)).save(i8_dir, i8_name.replace(".tflite", ""), overwrite=True)
        print(f"Quantized {i8}")
