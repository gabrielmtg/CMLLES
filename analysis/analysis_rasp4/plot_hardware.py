import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_perf, setup_style, save,
    FRAMEWORK_ORDER, COLORS, HATCHES,
    SIZE_ORDER, SIZE_LABEL, HIDDEN_ORDER, HIDDEN_LABEL, CNN_ORDER, CNN_LABEL,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP\n(Iris)",
    "MLP-IAES":   "MLP\n(IAES)",
    "Autoencoder": "AE",
    "CNN2D":       "CNN",
    "LSTM":        "LSTM",
}


def _mean_per_algo_fw(df, metric):
    rows = []
    for algo in ALGO_ORDER:
        for fw in FRAMEWORK_ORDER:
            sub = df[(df["algorithm"] == algo) & (df["framework"] == fw)]
            if sub.empty:
                continue
            rows.append({
                "algorithm": algo,
                "framework": fw,
                "value":     sub[metric].mean(),
            })
    return pd.DataFrame(rows)


def _grouped_bars(ax, data, title, ylabel, ylim=None):
    present = [a for a in ALGO_ORDER if a in data["algorithm"].values]
    n_algos = len(present)
    n_fw    = len(FRAMEWORK_ORDER)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    for i, fw in enumerate(FRAMEWORK_ORDER):
        sub = data[data["framework"] == fw]
        vals = [
            sub[sub["algorithm"] == a]["value"].values[0]
            if a in sub["algorithm"].values else np.nan
            for a in present
        ]
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width, label=fw,
               color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)


def plot_ipc_and_l1(df):
    ipc_data  = _mean_per_algo_fw(df, "ipc")
    l1_data   = _mean_per_algo_fw(df, "l1_miss_rate_pct")

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))

    _grouped_bars(axes[0], ipc_data,
                  "IPC — Instructions per Cycle",
                  "IPC (mean across variants)")
    _grouped_bars(axes[1], l1_data,
                  "L1-D Miss Rate",
                  "L1-D miss rate (%)")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in ipc_data["framework"].values]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss.pdf")
    plt.close(fig)


def plot_branch_misses(df):
    bm_data = _mean_per_algo_fw(df, "branch_misses")

    fig, ax = plt.subplots(figsize=(5, 3))
    _grouped_bars(ax, bm_data,
                  "Branch Misses (total per 1000 inferences)",
                  "Branch misses")
    ax.set_yscale("log")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in bm_data["framework"].values]
    ax.legend(handles=handles, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "hardware_branch_misses.pdf")
    plt.close(fig)


def plot_ipc_detailed(df, algorithm, x_order, x_labels, title):
    sub = df[df["algorithm"] == algorithm].copy()
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(5, 3))
    n_variants = len(x_order)
    n_fw       = len(FRAMEWORK_ORDER)
    width      = 0.8 / n_fw
    x_pos      = np.arange(n_variants)

    for i, fw in enumerate(FRAMEWORK_ORDER):
        fw_sub = sub[sub["framework"] == fw]
        vals = []
        for v in x_order:
            row = fw_sub[fw_sub["variant"].str.startswith(v)]
            vals.append(row["ipc"].mean() if not row.empty else np.nan)
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width, label=fw,
               color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([x_labels[v] for v in x_order])
    ax.set_ylabel("IPC")
    ax.set_title(f"IPC — {title}")
    ax.set_xlabel("Configuration")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fname = f"hardware_ipc_{algorithm.lower().replace('-', '_').replace('/', '_')}.pdf"
    save(fig, fname)
    plt.close(fig)


def main():
    df = load_perf()
    if df.empty:
        print("perf_metrics_all.csv not found or empty")
        return

    plot_ipc_and_l1(df)
    plot_branch_misses(df)
    plot_ipc_detailed(df, "LSTM",        HIDDEN_ORDER, HIDDEN_LABEL, "LSTM")
    plot_ipc_detailed(df, "CNN2D",       CNN_ORDER,    CNN_LABEL,    "CNN-2D")
    plot_ipc_detailed(df, "Autoencoder", SIZE_ORDER,   SIZE_LABEL,   "Autoencoder")


if __name__ == "__main__":
    main()
