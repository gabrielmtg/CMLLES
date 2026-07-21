import os
import numpy as np
from sklearn.model_selection import train_test_split
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat, QuantType

DATASET_PATH = "../../../../datasets/MLP/iris/iris.data"
MODEL_DIR    = "model"
CLASS_NAMES  = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


class _CalibReader(CalibrationDataReader):
    def __init__(self, X):
        self.data = [{"input": X[i:i+1]} for i in range(len(X))]
        self.idx  = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        d = self.data[self.idx]
        self.idx += 1
        return d


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
        f32 = os.path.join(MODEL_DIR, f"mlp_{size}_{act}_f32.onnx")
        i8  = os.path.join(MODEL_DIR, f"mlp_{size}_{act}_int8.onnx")
        if not os.path.exists(f32):
            print(f"SKIP {f32} (not found)")
            continue
        quantize_static(f32, i8, _CalibReader(X_train),
                        quant_format=QuantFormat.QDQ, weight_type=QuantType.QInt8)
        print(f"Quantized {i8}")
