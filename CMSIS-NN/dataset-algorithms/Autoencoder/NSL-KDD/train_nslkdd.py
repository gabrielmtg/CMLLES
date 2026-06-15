import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

INPUT_DIM = 38
HIDDEN_DIM = 16

def export_genann_weights(model, filepath):
    print(f"Exportando pesos para o Genann em {filepath}...")
    with open(filepath, 'w') as f:
        for layer_idx in [0, 2]:
            layer = model.net[layer_idx]
            w = layer.weight.data.numpy()
            b = layer.bias.data.numpy()
            
            for i in range(w.shape[0]):
                f.write(f"{-b[i]:.8f}\n") # Bias invertido para o Genann
                for j in range(w.shape[1]):
                    f.write(f"{w[i,j]:.8f}\n")

def export_cmsis_weights(model, filepath):
    print(f"Exportando pesos para CMSIS-DSP em {filepath}...")
    with open(filepath, 'w') as f:
        for layer_idx in [0, 2]:
            layer = model.net[layer_idx]
            w = layer.weight.data.numpy()
            b = layer.bias.data.numpy()
            
            # Matriz achatada (row-major) padrão da ARM
            for val in w.flatten():
                f.write(f"{val:.8f}\n")
            # Bias sem inversão de sinal
            for val in b.flatten():
                f.write(f"{val:.8f}\n")

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    
    test_data_path = os.path.join(model_dir, "../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt")
    onnx_path = os.path.join(model_dir, "nsl_autoencoder.onnx")
    genann_weights_path = os.path.join(model_dir, "nsl_genann_weights.txt")
    cmsis_weights_path = os.path.join(model_dir, "nsl_cmsis_weights.txt")

    print("Gerando dataset de telemetria NSL-KDD (Mock Normalizado)...")
    num_samples = 10000
    X_data = np.random.rand(num_samples, INPUT_DIM).astype(np.float32)
    
    with open(test_data_path, 'w') as f:
        f.write(f"{num_samples} {INPUT_DIM} {INPUT_DIM}\n")
        np.savetxt(f, X_data, fmt="%.6f", delimiter=" ")

    X_tensor = torch.tensor(X_data)

    class KDD_Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(INPUT_DIM, HIDDEN_DIM),
                nn.Sigmoid(),
                nn.Linear(HIDDEN_DIM, INPUT_DIM),
                nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    model = KDD_Autoencoder()
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Treinando Autoencoder PyTorch por 150 epocas...")
    epochs = 150
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, X_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 30 == 0:
            print(f"epoca [{epoch+1}/{epochs}] - MSE Loss: {loss.item():.6f}")

    dummy_input = torch.randn(1, INPUT_DIM)
    
    export_genann_weights(model, genann_weights_path)
    export_cmsis_weights(model, cmsis_weights_path)

    print("\nTentando exportar modelo ONNX...")
    try:
        torch.onnx.export(
            model, dummy_input, onnx_path, 
            input_names=['input'], output_names=['output'],
            do_constant_folding=False, opset_version=11
        )
        print(f"Modelo salvo em {onnx_path}")
    except Exception as e:
        print("[Aviso] Bug de exportação ONNX ignorado. Pesos do C gerados com sucesso.")

if __name__ == "__main__":
    main()
