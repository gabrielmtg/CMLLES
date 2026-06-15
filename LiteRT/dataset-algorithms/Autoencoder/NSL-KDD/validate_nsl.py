import pandas as pd
import numpy as np
import tensorflow as tf
import time

def process_nsl_files(train_path, test_path):
    col_names = ["duration","protocol_type","service","flag","src_bytes",
                 "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
                 "logged_in","num_compromised","root_shell","su_attempted","num_root",
                 "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
                 "is_host_login","is_guest_login","count","srv_count","serror_rate",
                 "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
                 "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
                 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
                 "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
    df_train = pd.read_csv(train_path, names=col_names)
    df_test = pd.read_csv(test_path, names=col_names)
    df_train.drop(['difficulty'], axis=1, inplace=True)
    df_test.drop(['difficulty'], axis=1, inplace=True)
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_all = pd.get_dummies(df_all, columns=categorical_cols, dtype=float)
    df_all['target'] = df_all['label'].apply(lambda x: 0.0 if x == 'normal' else 1.0)
    df_all.drop(['label'], axis=1, inplace=True)
    df_train = df_all[df_all['is_train'] == 1].drop(['is_train'], axis=1).copy()
    df_test = df_all[df_all['is_train'] == 0].drop(['is_train'], axis=1).copy()
    feature_cols = [c for c in df_train.columns if c != 'target']
    return df_train, df_test, feature_cols

def main():
    print("=" * 60)
    print("       VALIDAÇÃO — NSL-KDD Autoencoder (LiteRT)")
    print("=" * 60)
    train_csv = "../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
    test_csv = "../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
    tflite_model_path = "model/nsl_model.tflite"

    print(f"  Lendo arquivos: {train_csv} e {test_csv}")
    df_train, df_test, feature_cols = process_nsl_files(train_csv, test_csv)
    input_dim = len(feature_cols)
    print(f"  Dimensão de entrada: {input_dim} atributos")

    print("  Aplicando normalização Min-Max baseada no Treinamento...")
    norm_params = {}
    for col in feature_cols:
        min_val = df_train[col].min()
        max_val = df_train[col].max()
        if max_val - min_val == 0:
            max_val = min_val + 1e-9
        norm_params[col] = (min_val, max_val)
        df_train[col] = (df_train[col] - min_val) / (max_val - min_val)
        df_test[col] = ((df_test[col] - min_val) / (max_val - min_val)).clip(0, 1)

    df_train_normal = df_train[df_train['target'] == 0.0]
    X_train_normal = df_train_normal[feature_cols].values.astype(np.float32)
    X_test = df_test[feature_cols].values.astype(np.float32)
    y_test = df_test['target'].values.astype(np.float32)

    print(f"  Amostras de treino (normal): {len(X_train_normal)}")
    print(f"  Amostras de teste: {len(X_test)}")
    print(f"    Normal: {(y_test == 0).sum()}  |  Ataque: {(y_test == 1).sum()}")

    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"  Modelo: {tflite_model_path}")

    def run_inference(data):
        preds = []
        for row in data:
            interpreter.set_tensor(input_details[0]['index'], [row])
            interpreter.invoke()
            preds.append(interpreter.get_tensor(output_details[0]['index'])[0])
        return np.array(preds)

    print("\n  Calculando threshold com dados normais do treino...")
    train_out = run_inference(X_train_normal)
    train_mse = np.mean((X_train_normal - train_out) ** 2, axis=1)
    threshold = np.mean(train_mse) + 2 * np.std(train_mse)
    print(f"  MSE médio (treino normal): {np.mean(train_mse):.6f}")
    print(f"  MSE std (treino normal):   {np.std(train_mse):.6f}")
    print(f"  Threshold (mean + 2*std):  {threshold:.6f}")

    print("\n  Executando inferência no teste...")
    t0 = time.perf_counter()
    test_out = run_inference(X_test)
    t1 = time.perf_counter()
    total_inference_time = t1 - t0

    test_mse = np.mean((X_test - test_out) ** 2, axis=1)
    preds = (test_mse > threshold).astype(np.float32)

    total_samples = len(y_test)
    correct = (preds == y_test).sum()
    accuracy = correct / total_samples

    tp = ((preds == 1) & (y_test == 1)).sum()
    fp = ((preds == 1) & (y_test == 0)).sum()
    fn = ((preds == 0) & (y_test == 1)).sum()
    tn = ((preds == 0) & (y_test == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_latency_ms = (total_inference_time / total_samples) * 1000
    throughput = total_samples / total_inference_time

    print("\n" + "-" * 60)
    print(f"  Acurácia:             {accuracy:.4f}  ({correct}/{total_samples})")
    print(f"  Precisão (Ataque):    {precision:.4f}")
    print(f"  Recall (Ataque):      {recall:.4f}")
    print(f"  F1-Score (Ataque):    {f1:.4f}")
    print("-" * 60)
    print(f"  Latência média:       {avg_latency_ms:.6f} ms/amostra")
    print(f"  Throughput:           {throughput:.1f} amostras/s")
    print("-" * 60)

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    print("  Por classe:")
    print(f"    Normal (0)  — Precisão: {prec_0:.4f}  Recall: {rec_0:.4f}  F1: {f1_0:.4f}")
    print(f"    Ataque (1)  — Precisão: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print("-" * 60)
    print("  Matriz de Confusão:")
    print(f"                    Pred=Normal   Pred=Ataque")
    print(f"  Real=Normal       {tn:>10}     {fp:>10}")
    print(f"  Real=Ataque       {fn:>10}     {tp:>10}")
    print("-" * 60)

    print("\n  Distribuição do erro de reconstrução (MSE):")
    normal_mse = test_mse[y_test == 0]
    attack_mse = test_mse[y_test == 1]
    print(f"    Normal  — Média: {np.mean(normal_mse):.6f}  Std: {np.std(normal_mse):.6f}  Max: {np.max(normal_mse):.6f}")
    print(f"    Ataque  — Média: {np.mean(attack_mse):.6f}  Std: {np.std(attack_mse):.6f}  Max: {np.max(attack_mse):.6f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
