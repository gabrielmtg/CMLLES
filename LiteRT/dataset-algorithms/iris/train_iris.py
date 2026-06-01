#!/usr/bin/env python3
"""
Iris MLP — Treinamento e exportação para TFLite.

Treina um MLP com Keras e exporta:
1. model/iris_mlp.tflite — Modelo TFLite
2. model/iris_model_data.h — Modelo como C array para embedar em firmware

Uso: python3 train_iris.py
"""

import os
import subprocess
import numpy as np

# ─── Configurações ────────────────────────────────────────────
DATASET_PATH = "../../../datasets/iris/iris.data"
MODEL_DIR    = "model"
HIDDEN_SIZE  = 16
EPOCHS       = 300
LR           = 0.01
SEED         = 42

CLASS_NAMES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]


# ─── Carregar dataset ─────────────────────────────────────────
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


# ─── Main ─────────────────────────────────────────────────────
def main():
    import tensorflow as tf

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("=" * 50)
    print("  Iris MLP — Treinamento Keras → TFLite")
    print("=" * 50)

    X, y = load_iris(DATASET_PATH)
    print(f"\nDataset: {len(y)} amostras carregadas")

    # Criar modelo
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(HIDDEN_SIZE, activation='relu',
                              input_shape=(4,), name='fc1'),
        tf.keras.layers.Dense(3, activation='softmax', name='fc2')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Treinar
    print(f"\nTreinando por {EPOCHS} épocas...")
    model.fit(X, y, epochs=EPOCHS, batch_size=32, verbose=0)

    _, acc = model.evaluate(X, y, verbose=0)
    print(f"Acurácia: {acc*100:.1f}%")

    # Converter para TFLite
    os.makedirs(MODEL_DIR, exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    tflite_path = os.path.join(MODEL_DIR, "iris_mlp.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"\nModelo TFLite: {tflite_path} ({len(tflite_model)} bytes)")

    # Converter para C array usando xxd
    header_path = os.path.join(MODEL_DIR, "iris_model_data.h")
    try:
        result = subprocess.run(
            ["xxd", "-i", tflite_path],
            capture_output=True, text=True, check=True
        )
        with open(header_path, "w") as f:
            f.write("/* Auto-generated — NÃO EDITAR */\n")
            f.write("#ifndef IRIS_MODEL_DATA_H\n#define IRIS_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n\n")
            # Renomear o array para um nome limpo
            content = result.stdout
            content = content.replace("model_iris_mlp_tflite", "iris_model_data")
            f.write("alignas(16) ")
            f.write(content)
            f.write("\n#endif /* IRIS_MODEL_DATA_H */\n")
        print(f"Header C:      {header_path}")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback: gerar manualmente
        with open(header_path, "w") as f:
            f.write("/* Auto-generated — NÃO EDITAR */\n")
            f.write("#ifndef IRIS_MODEL_DATA_H\n#define IRIS_MODEL_DATA_H\n\n")
            f.write("#include <stdint.h>\n\n")
            f.write(f"alignas(16) const unsigned char iris_model_data[] = {{\n")
            for i, b in enumerate(tflite_model):
                if i % 12 == 0:
                    f.write("  ")
                f.write(f"0x{b:02x}, ")
                if (i + 1) % 12 == 0:
                    f.write("\n")
            f.write("\n};\n")
            f.write(f"const unsigned int iris_model_data_len = {len(tflite_model)};\n")
            f.write("\n#endif /* IRIS_MODEL_DATA_H */\n")
        print(f"Header C:      {header_path} (gerado manualmente)")


if __name__ == "__main__":
    main()
