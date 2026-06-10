import pandas
import numpy
import glob
import os

def process_single_file(csv_path, window_size=10):
    df = pandas.read_csv(csv_path)

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

def generate_fann_from_folders(folders, output_path, window_size=10):
    all_processed_dfs = []
    feature_cols = None
    
    for folder in folders:
        search_pattern = os.path.join(folder, "*.csv")
        csv_files = glob.glob(search_pattern)
        
        if not csv_files:
            print(f"nenhum arquivo .csv encontrado na pasta: {folder}")
            continue
            
        for file in csv_files:
            try:
                df_clean, cols = process_single_file(file, window_size)
                all_processed_dfs.append(df_clean)
                feature_cols = cols
            except Exception as e:
                print(f"erro ao processar {file}: {e}")
                
    if not all_processed_dfs:
        return

    final_df = pandas.concat(all_processed_dfs, ignore_index=True)
    
    means = []
    stds = []
    
    for col in feature_cols:
        mean_val = final_df[col].mean()
        std_val = final_df[col].std(ddof=1)
        
        if std_val == 0:
            std_val = 1e-9
            
        means.append(mean_val)
        stds.append(std_val)
        
        final_df[col] = (final_df[col] - mean_val) / std_val

    num_samples = len(final_df)
    num_features = len(feature_cols)
    num_outputs = 1
    
    with open(output_path, 'w') as f:
        f.write(f"{num_samples} {num_features} {num_outputs}\n")
        
        for _, row in final_df.iterrows():
            inputs = [f"{row[col]:.6f}" for col in feature_cols]
            f.write(" ".join(inputs) + "\n")
            
            f.write(f"{row['target']:.1f}\n")

if __name__ == "__main__":
    pastas_dataset = [
            "../../../../datasets/IAES-dataset/2-cores",
            "../../../../datasets/IAES-dataset/3-cores"
            ]

    generate_fann_from_folders(folders=pastas_dataset, output_path="model/IAES_fann.train", window_size=10)
