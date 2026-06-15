import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
import pnnx

SEQ_LENGTH  = 30
MAX_RUL     = 125
FEATURES    = ['s2', 's3', 's4', 's7', 's8', 's9', 's11', 's12', 's13',
                's14', 's15', 's17', 's20', 's21']
INPUT_SIZE  = len(FEATURES)
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.2
BATCH_SIZE  = 256
EPOCHS      = 150
LR          = 1e-3
PATIENCE    = 15

def load_and_prep_data(filepath, scaler=None, fit_scaler=True):
    cols = ['id', 'cycle', 'set1', 'set2', 'set3'] + [f's{i}' for i in range(1, 22)]
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=cols)
    rul = df.groupby('id')['cycle'].max().reset_index()
    rul.columns = ['id', 'max']
    df = df.merge(rul, on='id', how='left')
    df['RUL'] = (df['max'] - df['cycle']).clip(upper=MAX_RUL)
    df.drop('max', axis=1, inplace=True)
    df['RUL'] = df['RUL'] / MAX_RUL
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[FEATURES] = scaler.fit_transform(df[FEATURES])
    else:
        df[FEATURES] = scaler.transform(df[FEATURES])
    X, y = [], []
    for engine_id in df['id'].unique():
        engine_data = df[df['id'] == engine_id]
        feats = engine_data[FEATURES].values
        rul_vals = engine_data['RUL'].values
        for i in range(len(engine_data) - SEQ_LENGTH):
            X.append(feats[i:i + SEQ_LENGTH])
            y.append(rul_vals[i + SEQ_LENGTH - 1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler

class CmapssLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1    = nn.Linear(hidden_size, 64)
        self.relu   = nn.ReLU()
        self.fc2    = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out

if __name__ == "__main__":
    X_train, y_train, scaler = load_and_prep_data(
        "../../../../datasets/LSTM/CMAPSSData/train_FD001.txt",
        fit_scaler=True
    )
    os.makedirs("model", exist_ok=True)
    joblib.dump(scaler, "model/scaler.pkl")
    dataset  = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    val_size = int(0.2 * len(dataset))
    trn_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [trn_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    model     = CmapssLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=7
    )
    best_val_loss   = float('inf')
    epochs_no_impro = 0
    best_state      = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= trn_size
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= val_size
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            epochs_no_impro = 0
            best_state      = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impro += 1
        if epochs_no_impro >= PATIENCE:
            break
    model.load_state_dict(best_state)
    
    model.eval()
    dummy_input = torch.randn(1, SEQ_LENGTH, INPUT_SIZE, dtype=torch.float32)
    pnnx.export(model, "model/cmapss_lstm.pt", dummy_input)
