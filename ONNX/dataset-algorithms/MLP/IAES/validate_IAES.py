import pandas as pd
import numpy as np
import glob
import os
import onnxruntime as ort
import time


def process_single_file(csv_path, window_size=10):
    df = pd.read_csv(csv_path)
    eps = 1e-9
    df['sig_ipc'] = df['INSTRUCTIONS'] / (df['CPU_CYCLES'] + eps)
    df['sig_mpki'] = (df['CACHE_MISSES'] * 1000.0) / (df['INSTRUCTIONS'] + eps)
    df['sig_l2p'] = df['L2_CACHE_ACCESS'] / (df['CPU_CYCLES'] + eps)
    df['sig_branch'] = df['BRANCH_MISSES'] / (df['INSTRUCTIONS'] + eps)

    signals = ['sig_ipc', 'sig_mpki', 'sig_l2p', 'sig_branch']
    feature_cols = []

    for sig in signals:
        mean_col = f'{sig}_mean'
        df[mean_col] = df[sig].rolling(window=window_size).mean()
        std_col = f'{sig}_std'
        df[std_col] = df[sig].rolling(window=window_size).std(ddof=1)
        delta_col = f'{sig}_delta'
        df[delta_col] = df[sig].diff(periods=window_size)
        feature_cols.extend([mean_col, std_col, delta_col])

    df_clean = df.dropna(subset=feature_cols).copy()
    df_clean['target'] = df_clean['LABEL'].apply(lambda x: 1.0 if x in [1, 2, 3] else 0.0)
    return df_clean[feature_cols + ['target']], feature_cols


def main():
    print("=" * 55)
    print("       VALIDAÇÃO — IAES MLP (ONNX)")
    print("=" * 55)

    folders = [
        "../../../../datasets/MLP/IAES-dataset/2-cores",
        "../../../../datasets/MLP/IAES-dataset/3-cores"
    ]

    all_processed_dfs = []
    feature_cols = None

    print("  Processando CSVs brutos...")
    for folder in folders:
        for file in glob.glob(os.path.join(folder, "*.csv")):
            df_clean, cols = process_single_file(file)
            all_processed_dfs.append(df_clean)
            feature_cols = cols

    final_df = pd.concat(all_processed_dfs, ignore_index=True)

    print("  Aplicando normalização Z-Score...")
    for col in feature_cols:
        mean_val = final_df[col].mean()
        std_val = final_df[col].std(ddof=1)
        if std_val == 0:
            std_val = 1e-9
        final_df[col] = (final_df[col] - mean_val) / std_val

    X = final_df[feature_cols].values.astype(np.float32)
    y = final_df['target'].values.astype(np.float32)
    total_samples = len(y)
    print(f"  Dataset: {total_samples} amostras carregadas")

    onnx_model_path = "model/iaes_model.onnx"
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    active_provider = session.get_providers()[0]
    print(f"  Modelo: {onnx_model_path}")
    print(f"  Provider: {active_provider}")

    batch_size = 100000
    all_preds_prob = []
    total_inference_time = 0.0

    print("  Executando inferência...")
    for i in range(0, total_samples, batch_size):
        batch = X[i:i+batch_size]
        t0 = time.perf_counter()
        out = session.run([output_name], {input_name: batch})[0]
        t1 = time.perf_counter()
        total_inference_time += (t1 - t0)
        all_preds_prob.append(out.flatten())

    preds_prob = np.concatenate(all_preds_prob)
    preds = (preds_prob >= 0.5).astype(np.float32)

    correct = (preds == y).sum()
    accuracy = correct / total_samples

    tp = ((preds == 1) & (y == 1)).sum()
    fp = ((preds == 1) & (y == 0)).sum()
    fn = ((preds == 0) & (y == 1)).sum()
    tn = ((preds == 0) & (y == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    avg_latency_ms = (total_inference_time / total_samples) * 1000
    throughput = total_samples / total_inference_time

    print("-" * 55)
    print(f"  Acurácia:             {accuracy:.4f}  ({correct}/{total_samples})")
    print(f"  Precisão (Ataque):    {precision:.4f}")
    print(f"  Recall (Ataque):      {recall:.4f}")
    print(f"  F1-Score (Ataque):    {f1:.4f}")
    print("-" * 55)
    print(f"  Latência média:       {avg_latency_ms:.6f} ms/amostra")
    print(f"  Throughput:           {throughput:.1f} amostras/s")
    print("-" * 55)

    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0

    print("  Por classe:")
    print(f"    Normal (0)  — Precisão: {prec_0:.4f}  Recall: {rec_0:.4f}  F1: {f1_0:.4f}")
    print(f"    Ataque (1)  — Precisão: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print("-" * 55)
    print("  Matriz de Confusão:")
    print(f"                    Pred=Normal   Pred=Ataque")
    print(f"  Real=Normal       {tn:>10}     {fp:>10}")
    print(f"  Real=Ataque       {fn:>10}     {tp:>10}")
    print("=" * 55)


if __name__ == "__main__":
    main()
