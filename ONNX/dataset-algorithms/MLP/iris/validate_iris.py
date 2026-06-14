import numpy as np
import onnxruntime as ort
import time

DATASET_PATH = "../../../../datasets/iris/iris.data"
ONNX_MODEL_PATH = "model/iris_mlp.onnx"
CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
NUM_CLASSES = 3


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


def main():
    print("=" * 55)
    print("       VALIDAÇÃO — Iris MLP (ONNX)")
    print("=" * 55)

    X, y = load_iris(DATASET_PATH)
    print(f"  Dataset: {len(y)} amostras carregadas")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    active_provider = session.get_providers()[0]
    print(f"  Modelo: {ONNX_MODEL_PATH}")
    print(f"  Provider: {active_provider}")

    all_outputs = []
    t0 = time.perf_counter()
    for i in range(len(X)):
        sample = X[i:i+1]
        out = session.run([output_name], {input_name: sample})[0]
        all_outputs.append(out)
    t1 = time.perf_counter()
    total_inference_time = t1 - t0

    outputs = np.concatenate(all_outputs, axis=0)
    preds = np.argmax(outputs, axis=1)
    total_samples = len(y)

    correct = (preds == y).sum()
    accuracy = correct / total_samples

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for true_label, pred_label in zip(y, preds):
        confusion[true_label][pred_label] += 1

    avg_latency_ms = (total_inference_time / total_samples) * 1000
    throughput = total_samples / total_inference_time

    print("-" * 55)
    print(f"  Acurácia:         {accuracy:.4f}  ({correct}/{total_samples})")
    print("-" * 55)
    print(f"  Latência média:   {avg_latency_ms:.3f} ms/amostra")
    print(f"  Throughput:       {throughput:.1f} amostras/s")
    print("-" * 55)

    print("  Métricas por classe:")
    print()
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0
    for i in range(NUM_CLASSES):
        tp = confusion[i][i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        print(f"    {CLASS_NAMES[i]:20s} — Precisão: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")

    macro_precision /= NUM_CLASSES
    macro_recall /= NUM_CLASSES
    macro_f1 /= NUM_CLASSES
    print()
    print(f"  Macro Precisão:   {macro_precision:.4f}")
    print(f"  Macro Recall:     {macro_recall:.4f}")
    print(f"  Macro F1-Score:   {macro_f1:.4f}")

    print("-" * 55)
    print("  Matriz de Confusão:")
    header = "                    " + "  ".join(f"{CLASS_NAMES[i]:>12s}" for i in range(NUM_CLASSES))
    print(header)
    for i in range(NUM_CLASSES):
        row = f"  {CLASS_NAMES[i]:18s}" + "  ".join(f"{confusion[i][j]:>12d}" for j in range(NUM_CLASSES))
        print(row)
    print("=" * 55)


if __name__ == "__main__":
    main()
