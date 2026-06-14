import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib

FEATURES   = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13',
               's14', 's15', 's17', 's20', 's21']
SEQ_LENGTH = 30
MAX_RUL    = 125


def validate():
    session = ort.InferenceSession("model/cmapss_lstm.onnx")

    # Carrega o scaler ajustado no conjunto de TREINO
    scaler = joblib.load("model/scaler.pkl")

    cols = ['id', 'cycle', 'set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]
    df_test  = pd.read_csv(
        "../../../../datasets/LSTM/CMAPSSData/test_FD001.txt",
        sep=r'\s+', header=None, names=cols
    )
    rul_test = pd.read_csv(
        "../../../../datasets/LSTM/CMAPSSData/RUL_FD001.txt",
        sep=r'\s+', header=None, names=['RUL']
    )

    # Aplica a mesma normalização do treino (apenas transform, sem fit!)
    df_test[FEATURES] = scaler.transform(df_test[FEATURES])

    # Para cada motor, usa os últimos SEQ_LENGTH ciclos disponíveis
    X_test      = []
    valid_ids   = []
    for i, engine_id in enumerate(df_test['id'].unique()):
        engine_data = df_test[df_test['id'] == engine_id]
        if len(engine_data) >= SEQ_LENGTH:
            X_test.append(engine_data[FEATURES].values[-SEQ_LENGTH:])
            valid_ids.append(i)
        else:
            print(f"  [AVISO] Motor {engine_id} tem apenas {len(engine_data)} ciclos "
                  f"(< {SEQ_LENGTH}) — ignorado.")

    X_test = np.array(X_test, dtype=np.float32)
    y_true = rul_test['RUL'].values[valid_ids]

    # Inferência
    ort_inputs    = {session.get_inputs()[0].name: X_test}
    preds         = session.run(None, ort_inputs)[0]
    predicted_rul = np.clip(preds.flatten() * MAX_RUL, 0, None)

    errors = predicted_rul - y_true
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    mse    = np.mean(errors ** 2)

    print("=" * 45)
    print(f"  Motores avaliados : {len(valid_ids)}")
    print(f"  MSE               : {mse:.2f}")
    print(f"  RMSE              : {rmse:.2f} ciclos")
    print(f"  MAE               : {mae:.2f} ciclos")
    print("=" * 45)
    print(f"\n  Exemplo Motor 1  →  Previsto: {predicted_rul[0]:.1f}  |  Real: {y_true[0]}")


if __name__ == "__main__":
    validate()
