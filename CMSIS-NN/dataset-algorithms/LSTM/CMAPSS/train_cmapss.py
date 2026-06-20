import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

SEQ_LEN = 30
FEATURES = 14
HIDDEN = 16

# Caminho para os dados extraídos da NASA
DATA_DIR = "../../../../datasets/LSTM/CMAPSSData"

class CMAPSS_GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(FEATURES, HIDDEN, batch_first=True)
        self.fc = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def export_gru_weights(model, filepath):
    print(f"Fatiando pesos da GRU para o formato CMSIS-DSP em {filepath}...")
    with open(filepath, 'w') as f:
        w_ih = model.gru.weight_ih_l0.data.numpy()
        w_hh = model.gru.weight_hh_l0.data.numpy()
        b_ih = model.gru.bias_ih_l0.data.numpy()
        b_hh = model.gru.bias_hh_l0.data.numpy()

        for i in range(3):
            for val in w_ih[i*HIDDEN:(i+1)*HIDDEN, :].flatten(): f.write(f"{val:.8f}\n")
        for i in range(3):
            for val in w_hh[i*HIDDEN:(i+1)*HIDDEN, :].flatten(): f.write(f"{val:.8f}\n")
        for i in range(3):
            sub_b = b_ih[i*HIDDEN:(i+1)*HIDDEN] + b_hh[i*HIDDEN:(i+1)*HIDDEN]
            for val in sub_b.flatten(): f.write(f"{val:.8f}\n")

        for val in model.fc.weight.data.numpy().flatten(): f.write(f"{val:.8f}\n")
        for val in model.fc.bias.data.numpy().flatten(): f.write(f"{val:.8f}\n")

def process_cmapss_data():
    print("Carregando e processando dataset real da NASA CMAPSS (FD001)...")
    
    # Nomes das colunas padrão do CMAPSS
    cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    
    train_df = pd.read_csv(os.path.join(DATA_DIR, 'train_FD001.txt'), sep='\s+', header=None, names=cols)
    test_df = pd.read_csv(os.path.join(DATA_DIR, 'test_FD001.txt'), sep='\s+', header=None, names=cols)
    truth_df = pd.read_csv(os.path.join(DATA_DIR, 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # O CMAPSS possui 21 sensores, mas alguns ficam constantes durante o voo. 
    # A literatura isola estes 14 sensores úteis que mostram a degradação térmica/física:
    useful_sensors = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
    
    # Normalização Min-Max manual (para não exigir o scikit-learn)
    min_vals = train_df[useful_sensors].min()
    max_vals = train_df[useful_sensors].max()
    
    # Evita divisão por zero
    denominador = max_vals - min_vals
    denominador[denominador == 0] = 1.0 
    
    train_df[useful_sensors] = (train_df[useful_sensors] - min_vals) / denominador
    test_df[useful_sensors] = (test_df[useful_sensors] - min_vals) / denominador

    # Calcula o RUL real para o treino (Ciclo Máximo do motor - Ciclo Atual)
    rul_train = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul_train.columns = ['id', 'max']
    train_df = train_df.merge(rul_train, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']

    # Gerador de Sequências Temporais (Janelas Deslizantes)
    def gen_sequence(id_df, seq_length, seq_cols):
        data_matrix = id_df[seq_cols].values
        num_elements = data_matrix.shape[0]
        for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
            yield data_matrix[start:stop, :]

    def gen_labels(id_df, seq_length, label):
        data_matrix = id_df[label].values
        num_elements = data_matrix.shape[0]
        return data_matrix[seq_length:num_elements, :]

    seq_gen = (list(gen_sequence(train_df[train_df['id']==id], SEQ_LEN, useful_sensors)) for id in train_df['id'].unique())
    X_train = np.concatenate(list(seq_gen)).astype(np.float32)
    
    label_gen = [gen_labels(train_df[train_df['id']==id], SEQ_LEN, ['RUL']) for id in train_df['id'].unique()]
    y_train = np.concatenate(label_gen).astype(np.float32)

    # Para o teste, a NASA avalia apenas a *última* sequência de cada motor
    X_test = []
    for id in test_df['id'].unique():
        id_data = test_df[test_df['id']==id][useful_sensors].values
        if len(id_data) >= SEQ_LEN:
            X_test.append(id_data[-SEQ_LEN:, :])
        else:
            # Se o motor quebrou antes de 30 ciclos, preenchemos com zeros no começo (Padding)
            pad = np.zeros((SEQ_LEN - len(id_data), FEATURES))
            X_test.append(np.concatenate((pad, id_data)))
            
    X_test = np.array(X_test).astype(np.float32)
    y_test = truth_df['RUL'].values.astype(np.float32).reshape(-1, 1)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)
    
    X_train, y_train, X_test, y_test = process_cmapss_data()

    print(f"Dataset formatado! Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras (motores)")

    # Salva o dataset de teste real para o C avaliar
    with open("model/cmapss_test_data.txt", 'w') as f:
        f.write(f"{X_test.shape[0]} {SEQ_LEN * FEATURES} 1\n")
        dataset_combinado = np.hstack((X_test.reshape(X_test.shape[0], -1), y_test))
        np.savetxt(f, dataset_combinado, fmt="%.6f", delimiter=" ")

    model = CMAPSS_GRU()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Convertendo para tensores PyTorch
    X_tensor = torch.tensor(X_train)
    y_tensor = torch.tensor(y_train)

    print("Treinando CMAPSS GRU (100 Épocas)...")
    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tensor), y_tensor)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Época [{epoch+1}/{epochs}] - MSE Loss: {loss.item():.4f}")
        
    export_gru_weights(model, "model/cmapss_cmsis_weights.txt")
