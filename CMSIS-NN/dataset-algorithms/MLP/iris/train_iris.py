import os
import numpy as np

DATASET_PATH = "../../../datasets/iris/iris.data"
MODEL_DIR    = "model"
HIDDEN_SIZE  = 16
EPOCHS       = 300
LR           = 0.01
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

np.random.seed(SEED)


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
    return X, y


def normalize_minmax(X):
    xmin = X.min(axis=0)
    xmax = X.max(axis=0)
    rng = xmax - xmin
    rng[rng == 0] = 1.0
    return (X - xmin) / rng, xmin, xmax


def quantize_array(arr, num_bits=8):
    abs_max = np.max(np.abs(arr))
    if abs_max == 0:
        return np.zeros_like(arr, dtype=np.int8), 1.0, 0
    scale = abs_max / 127.0
    quantized = np.clip(np.round(arr / scale), -128, 127).astype(np.int8)
    return quantized, scale, 0


def quantize_bias(bias, input_scale, weight_scale):
    bias_scale = input_scale * weight_scale
    if bias_scale == 0:
        return np.zeros_like(bias, dtype=np.int32), 1.0
    quantized = np.clip(np.round(bias / bias_scale),
                        -2147483648, 2147483647).astype(np.int32)
    return quantized, bias_scale


def array_to_c(name, arr, dtype="int8_t"):
    lines = []
    lines.append(f"static const {dtype} {name}[] = {{")
    flat = arr.flatten()
    row = "    "
    for i, v in enumerate(flat):
        row += f"{v}, "
        if (i + 1) % 12 == 0:
            lines.append(row)
            row = "    "
    if row.strip():
        lines.append(row)
    lines.append("};")
    return "\n".join(lines)


def main():
    try:
        import tensorflow as tf
        USE_TF = True
    except ImportError:
        USE_TF = False
        print("TensorFlow não disponível, usando treinamento NumPy manual")

    print("=" * 50)
    print("  Iris MLP — Treinamento + Quantização INT8")
    print("  (para CMSIS-NN arm_fully_connected_s8)")
    print("=" * 50)

    X, y = load_iris(DATASET_PATH)
    X_norm, x_min, x_max = normalize_minmax(X)
    print(f"\nDataset: {len(y)} amostras carregadas")

    if USE_TF:
        tf.random.set_seed(SEED)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu',
                                  input_shape=(4,)),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X_norm, y, epochs=EPOCHS, batch_size=32, verbose=0)

        _, acc = model.evaluate(X_norm, y, verbose=0)
        print(f"Acurácia (float32): {acc*100:.1f}%")

        w1, b1 = model.layers[0].get_weights()
        w2, b2 = model.layers[1].get_weights()
    else:
        n_input, n_hidden, n_output = 4, HIDDEN_SIZE, 3

        w1 = np.random.randn(n_input, n_hidden).astype(np.float32) * 0.1
        b1 = np.zeros(n_hidden, dtype=np.float32)
        w2 = np.random.randn(n_hidden, n_output).astype(np.float32) * 0.1
        b2 = np.zeros(n_output, dtype=np.float32)

        def relu(x):
            return np.maximum(0, x)

        def softmax(x):
            e = np.exp(x - x.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)

        lr = LR
        for epoch in range(EPOCHS):
            z1 = X_norm @ w1 + b1
            a1 = relu(z1)
            z2 = a1 @ w2 + b2
            a2 = softmax(z2)

            n = len(y)
            y_onehot = np.eye(n_output)[y]

            dz2 = (a2 - y_onehot) / n
            dw2 = a1.T @ dz2
            db2 = dz2.sum(axis=0)

            da1 = dz2 @ w2.T
            dz1 = da1 * (z1 > 0).astype(np.float32)
            dw1 = X_norm.T @ dz1
            db1 = dz1.sum(axis=0)

            w2 -= lr * dw2
            b2 -= lr * db2
            w1 -= lr * dw1
            b1 -= lr * db1

        preds = np.argmax(softmax(relu(X_norm @ w1 + b1) @ w2 + b2), axis=1)
        acc = (preds == y).mean()
        print(f"Acurácia (float32): {acc*100:.1f}%")

    input_scale = 1.0 / 127.0
    input_zero_point = 0

    w1_q, w1_scale, w1_zp = quantize_array(w1.T)
    w2_q, w2_scale, w2_zp = quantize_array(w2.T)

    b1_q, b1_scale = quantize_bias(b1, input_scale, w1_scale)
    b2_q, b2_scale = quantize_bias(b2, w1_scale, w2_scale)


    print(f"\nQuantização INT8:")
    print(f"  W1: scale={w1_scale:.6f}, shape={w1_q.shape}")
    print(f"  W2: scale={w2_scale:.6f}, shape={w2_q.shape}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "iris_weights.h"), "w") as f:
        f.write("/* Auto-generated by train_iris.py — NÃO EDITAR */\n")
        f.write("#ifndef IRIS_WEIGHTS_H\n#define IRIS_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"/* FC1: [{w1_q.shape[0]}x{w1_q.shape[1]}] */\n")
        f.write(array_to_c("fc1_weights", w1_q) + "\n\n")
        f.write(f"/* FC1 bias: [{len(b1_q)}] */\n")
        f.write(array_to_c("fc1_bias", b1_q, "int32_t") + "\n\n")
        f.write(f"/* FC2: [{w2_q.shape[0]}x{w2_q.shape[1]}] */\n")
        f.write(array_to_c("fc2_weights", w2_q) + "\n\n")
        f.write(f"/* FC2 bias: [{len(b2_q)}] */\n")
        f.write(array_to_c("fc2_bias", b2_q, "int32_t") + "\n\n")
        f.write("#endif /* IRIS_WEIGHTS_H */\n")

    with open(os.path.join(MODEL_DIR, "iris_parameters.h"), "w") as f:
        f.write("/* Auto-generated by train_iris.py — NÃO EDITAR */\n")
        f.write("#ifndef IRIS_PARAMETERS_H\n#define IRIS_PARAMETERS_H\n\n")
        f.write(f"#define INPUT_SIZE    {4}\n")
        f.write(f"#define HIDDEN_SIZE   {HIDDEN_SIZE}\n")
        f.write(f"#define OUTPUT_SIZE   {3}\n\n")
        f.write(f"/* Escalas de quantização */\n")
        f.write(f"#define INPUT_SCALE      {input_scale:.10f}f\n")
        f.write(f"#define INPUT_ZERO_POINT {input_zero_point}\n\n")
        f.write(f"#define FC1_WEIGHT_SCALE {w1_scale:.10f}f\n")
        f.write(f"#define FC1_BIAS_SCALE   {b1_scale:.10f}f\n")
        f.write(f"#define FC2_WEIGHT_SCALE {w2_scale:.10f}f\n")
        f.write(f"#define FC2_BIAS_SCALE   {b2_scale:.10f}f\n\n")

        f.write("/* Normalização min-max do dataset */\n")
        f.write("static const float input_min[] = {")
        f.write(", ".join(f"{v:.4f}f" for v in x_min))
        f.write("};\n")
        f.write("static const float input_max[] = {")
        f.write(", ".join(f"{v:.4f}f" for v in x_max))
        f.write("};\n\n")

        f.write("#endif /* IRIS_PARAMETERS_H */\n")

    print(f"\nArquivos gerados:")
    print(f"  {MODEL_DIR}/iris_weights.h")
    print(f"  {MODEL_DIR}/iris_parameters.h")


if __name__ == "__main__":
    main()
