import pandas as pd
import numpy as np
import ncnn
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
    train_csv = "../../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTrain+.txt"
    test_csv = "../../../../../datasets/Autoencoder/NSL_KDD_Dataset/KDDTest+.txt"
    param_path = "model/nsl_model.ncnn.param"
    bin_path = "model/nsl_model.ncnn.bin"

    df_train, df_test, feature_cols = process_nsl_files(train_csv, test_csv)
    input_dim = len(feature_cols)

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

    net = ncnn.Net()
    net.load_param(param_path)
    net.load_model(bin_path)

    def run_inference(data):
        preds = []
        for row in data:
            ex = net.create_extractor()
            in_mat = ncnn.Mat(row)
            ex.input("in0", in_mat)
            _, out_mat = ex.extract("out0")
            preds.append(np.array(out_mat))
        return np.array(preds)

    train_out = run_inference(X_train_normal)
    train_mse = np.mean((X_train_normal - train_out) ** 2, axis=1)
    threshold = np.mean(train_mse) + 2 * np.std(train_mse)

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

if __name__ == "__main__":
    main()
