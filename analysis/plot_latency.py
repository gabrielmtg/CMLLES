import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_latencies, setup_style, save,
    FRAMEWORK_ORDER, COLORS, HATCHES,
    SIZE_ORDER, SIZE_LABEL, HIDDEN_ORDER, HIDDEN_LABEL, CNN_ORDER, CNN_LABEL,
)

setup_style()


def _parse_variant(label, algorithm):
    parts = label.split("_")
    if algorithm == "MLP-Iris":
        size = parts[1] if len(parts) > 1 else ""
        act  = parts[2] if len(parts) > 2 else ""
        return size, act
    if algorithm == "MLP-IAES":
        size = parts[1] if len(parts) > 1 else ""
        act  = parts[2] if len(parts) > 2 else ""
        return size, act
    if algorithm == "Autoencoder":
        return parts[1] if len(parts) > 1 else "", None
    if algorithm == "CNN2D":
        return parts[1] if len(parts) > 1 else "", None
    if algorithm == "LSTM":
        return parts[1] if len(parts) > 1 else "", None
    return "", None


def _latency_stats(df, algorithm):
    rows = []
    for _, grp in df[df["algorithm"] == algorithm].groupby(["framework", "label"]):
        fw    = grp["framework"].iloc[0]
        label = grp["label"].iloc[0]
        size, act = _parse_variant(label, algorithm)
        stats = grp["latency_us"].agg(["mean", "std", "median"])
        rows.append({
            "framework": fw,
            "size":      size,
            "activation": act,
            "mean":      stats["mean"],
            "std":       stats["std"],
            "median":    stats["median"],
        })
    return pd.DataFrame(rows)


def _bar_group(ax, stats, x_order, x_labels, activation=None):
    fw_list   = [f for f in FRAMEWORK_ORDER if f in stats["framework"].values]
    n_groups  = len(x_order)
    n_bars    = len(fw_list)
    width     = 0.8 / n_bars
    x_pos     = np.arange(n_groups)

    sub = stats
    if activation is not None:
        sub = stats[stats["activation"] == activation]

    for i, fw in enumerate(fw_list):
        fw_data = sub[sub["framework"] == fw]
        means = []
        stds  = []
        for x in x_order:
            row = fw_data[fw_data["size"] == x]
            if len(row):
                means.append(row["mean"].values[0])
                stds.append(row["std"].values[0])
            else:
                means.append(np.nan)
                stds.append(0)
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width, yerr=stds,
               label=fw, color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5,
               error_kw={"elinewidth": 0.8, "capsize": 2})

    ax.set_xticks(x_pos)
    ax.set_xticklabels([x_labels[x] for x in x_order])
    ax.set_ylabel("Latência média (µs)")
    ax.set_yscale("log")


def plot_mlp(df, dataset_key):
    stats = _latency_stats(df, f"MLP-{dataset_key}")
    if stats.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharey=True)
    fig.suptitle(f"MLP — {dataset_key}", fontsize=10)

    for ax, act, title in zip(axes, ["relu", "sigmoid"], ["ReLU", "Sigmoid"]):
        _bar_group(ax, stats, SIZE_ORDER, SIZE_LABEL, activation=act)
        ax.set_title(title)
        ax.set_xlabel("Tamanho")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in stats["framework"].values]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, f"latency_mlp_{dataset_key.lower()}.pdf")
    plt.close(fig)


def plot_algo(df, algorithm, x_order, x_labels, title):
    stats = _latency_stats(df, algorithm)
    if stats.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_title(title)
    _bar_group(ax, stats, x_order, x_labels)
    ax.set_xlabel("Configuração")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in stats["framework"].values]
    ax.legend(handles=handles, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, f"latency_{algorithm.lower().replace('-', '_')}.pdf")
    plt.close(fig)


def plot_overview(df):
    algos = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
    algo_labels = {
        "MLP-Iris":    "MLP\n(Iris)",
        "MLP-IAES":   "MLP\n(IAES)",
        "Autoencoder": "AE",
        "CNN2D":       "CNN",
        "LSTM":        "LSTM",
    }

    rows = []
    for algo in algos:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        for fw in FRAMEWORK_ORDER:
            fwsub = sub[sub["framework"] == fw]
            if fwsub.empty:
                continue
            rows.append({
                "algorithm": algo,
                "framework": fw,
                "mean":   fwsub["latency_us"].mean(),
                "median": fwsub["latency_us"].median(),
            })
    ov = pd.DataFrame(rows)
    if ov.empty:
        return

    present_algos = [a for a in algos if a in ov["algorithm"].values]
    n_algos  = len(present_algos)
    n_fw     = len(FRAMEWORK_ORDER)
    width    = 0.8 / n_fw
    x_pos    = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, fw in enumerate(FRAMEWORK_ORDER):
        sub = ov[ov["framework"] == fw]
        means = [sub[sub["algorithm"] == a]["mean"].values[0]
                 if a in sub["algorithm"].values else np.nan
                 for a in present_algos]
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, means, width, label=fw,
               color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Latência média de inferência (µs)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([algo_labels[a] for a in present_algos])
    ax.set_xlabel("Algoritmo")
    ax.legend(loc="upper left", ncol=2)
    ax.set_title("Comparação de latência de inferência — todos os algoritmos")
    fig.tight_layout()
    save(fig, "latency_overview.pdf")
    plt.close(fig)


def main():
    df = load_latencies()
    if df.empty:
        print("Nenhum CSV de latência encontrado em results/latencies_rpi/")
        return

    plot_overview(df)
    plot_mlp(df, "Iris")
    plot_mlp(df, "IAES")
    plot_algo(df, "Autoencoder", SIZE_ORDER,   SIZE_LABEL,   "Autoencoder")
    plot_algo(df, "CNN2D",       CNN_ORDER,    CNN_LABEL,    "CNN-2D")
    plot_algo(df, "LSTM",        HIDDEN_ORDER, HIDDEN_LABEL, "LSTM")


if __name__ == "__main__":
    main()
