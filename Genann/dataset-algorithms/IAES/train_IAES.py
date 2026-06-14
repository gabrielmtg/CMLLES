import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim

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

def export_genann_weights(model, filepath):
    print(f"Exportando pesos para o Genann em {filepath}...")
    with open(filepath, 'w') as f:
        # Acessando as camadas lineares da nn.Sequential (índices 0, 2 e 4)
        for layer_idx in [0, 2, 4]:
            layer = model.net[layer_idx]
            w = layer.weight.data.numpy()
            b = layer.bias.data.numpy()
            
            for i in range(w.shape[0]):
                # CORREÇÃO 1: O Genann exige o Bias PRIMEIRO e com o sinal INVERTIDO
                f.write(f"{-b[i]:.8f}\n") 
                for j in range(w.shape[1]):
                    f.write(f"{w[i,j]:.8f}\n")

def train_and_export(folders, window_size=10):
    all_processed_dfs = []
    feature_cols = None
    
    print("processando CSVs brutos")
    for folder in folders:
        print(f"Procurando em: {os.path.abspath(folder)}")
        for file in glob.glob(os.path.join(folder, "*.csv")):
            df_clean, cols = process_single_file(file, window_size)
            all_processed_dfs.append(df_clean)
            feature_cols = cols
            
    final_df = pd.concat(all_processed_dfs, ignore_index=True)
    
    print("aplicando normalizacao Z-Score")
    for col in feature_cols:
        mean_val = final_df[col].mean()
        std_val = final_df[col].std(ddof=1)
        if std_val == 0: std_val = 1e-9
        final_df[col] = (final_df[col] - mean_val) / std_val

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    test_data_path = os.path.join(model_dir, "iaes_test_data.txt")
    onnx_path = os.path.join(model_dir, "iaes_model.onnx")
    genann_weights_path = os.path.join(model_dir, "iaes_genann_weights.txt")

    print(f"Salvando arquivos no diretorio exato: {model_dir}")
    
    with open(test_data_path, 'w') as f:
        f.write(f"{len(final_df)} 12 1\n")
    
    with open(test_data_path, 'a') as f:
        np.savetxt(f, final_df.values, fmt="%.6f", delimiter=" ")

    X = torch.tensor(final_df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(final_df['target'].values, dtype=torch.float32).view(-1, 1)

    # CORREÇÃO 2: Substituindo ReLU por Sigmoid para parear com o backend C puro
    class IAES_Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(12, 16),
                nn.Sigmoid(),
                nn.Linear(16, 16),
                nn.Sigmoid(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    model = IAES_Model()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("treinando o modelo PyTorch por 200 epocas")
    epochs = 200
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 20 == 0:
            print(f"epoca [{epoch+1}/{epochs}] - Erro (Loss): {loss.item():.4f}")

    dummy_input = torch.randn(1, 12)
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])
    print(f"sucesso!!! modelo salvo como {onnx_path}")
    
    export_genann_weights(model, genann_weights_path)

if __name__ == "__main__":
    pastas = [
        "../../../datasets/IAES-dataset/2-cores", 
        "../../../datasets/IAES-dataset/3-cores"
    ]
    train_and_export(pastas)