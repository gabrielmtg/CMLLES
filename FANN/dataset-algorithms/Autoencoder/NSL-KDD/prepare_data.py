import pandas as pd
import numpy as np
import os

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

def generate_fann_data(train_csv, test_csv):
    df_train, df_test, feature_cols = process_nsl_files(train_csv, test_csv)
    
    input_dim = len(feature_cols)
    
    for col in feature_cols:
        min_val = df_train[col].min()
        max_val = df_train[col].max()
        if max_val - min_val == 0: 
            max_val = min_val + 1e-9
            
        df_train[col] = (df_train[col] - min_val) / (max_val - min_val)
        df_test[col] = ((df_test[col] - min_val) / (max_val - min_val)).clip(0, 1)

    os.makedirs("model", exist_ok=True)

    df_train_normal = df_train[df_train['target'] == 0.0]
    X_train_normal = df_train_normal[feature_cols].values

    num_train = len(X_train_normal)
    with open("model/nsl_train.data", 'w') as f:
        f.write(f"{num_train} {input_dim} {input_dim}\n")
        for row in X_train_normal:
            row_str = " ".join(f"{val:.6f}" for val in row)
            f.write(f"{row_str}\n")
            f.write(f"{row_str}\n")

    num_test = len(df_test)
    with open("model/nsl_test.txt", 'w') as f:
        f.write(f"{num_test} {input_dim} 1\n")
        chunk_size = 100000
        for i in range(0, num_test, chunk_size):
            chunk = df_test[feature_cols + ['target']].iloc[i:i+chunk_size].values
            np.savetxt(f, chunk, fmt="%.6f", delimiter=" ")

if __name__ == "__main__":
    treino = "../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
    teste = "../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
    generate_fann_data(treino, teste)
