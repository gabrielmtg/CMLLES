import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

IMG_SIZE = 32
CHANNELS = 1
FILTERS = 4
KERNEL = 3
POOL = 2

# Saída após Conv2d (sem padding): 32 - 3 + 1 = 30
# Saída após MaxPool2d: 30 / 2 = 15
FLATTEN_SIZE = FILTERS * 15 * 15 

class VWW_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(CHANNELS, FILTERS, kernel_size=KERNEL, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(POOL)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(FLATTEN_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.conv(x)))
        return self.sigmoid(self.fc(self.flatten(x)))

def export_cnn_weights(model, filepath):
    print(f"Exportando pesos da CNN para CMSIS-DSP em {filepath}...")
    with open(filepath, 'w') as f:
        # Extrai pesos da Convolução: [Out_Channels, In_Channels, H, W]
        w_conv = model.conv.weight.data.numpy()
        b_conv = model.conv.bias.data.numpy()
        for val in w_conv.flatten(): f.write(f"{val:.8f}\n")
        for val in b_conv.flatten(): f.write(f"{val:.8f}\n")
        
        # Extrai pesos da Camada Densa
        w_fc = model.fc.weight.data.numpy()
        b_fc = model.fc.bias.data.numpy()
        for val in w_fc.flatten(): f.write(f"{val:.8f}\n")
        for val in b_fc.flatten(): f.write(f"{val:.8f}\n")

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    
    # Mock VWW Dataset: 100 amostras, 1 canal, 32x32
    num_samples = 100
    X_data = np.random.rand(num_samples, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y_data = np.random.randint(0, 2, (num_samples, 1)).astype(np.float32)
    
    # Salva o dataset achatado para o C
    with open("model/vww_test_data.txt", 'w') as f:
        f.write(f"{num_samples} {IMG_SIZE * IMG_SIZE} 1\n")
        np.savetxt(f, X_data.reshape(num_samples, -1), fmt="%.6f", delimiter=" ")

    model = VWW_CNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Treinando VWW CNN...")
    for epoch in range(50):
        optimizer.zero_grad()
        loss = criterion(model(torch.tensor(X_data)), torch.tensor(y_data))
        loss.backward()
        optimizer.step()
        
    export_cnn_weights(model, "model/vww_cmsis_weights.txt")