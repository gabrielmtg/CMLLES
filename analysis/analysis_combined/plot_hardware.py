import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_perf, setup_style, save,
    FRAMEWORK_ORDER, COLORS, HATCHES, PLATFORM_ORDER, PLATFORM_LABEL,
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
            rows.append({"algorithm": algo, "framework": fw, "value": sub[metric].mean()})
    return pd.DataFrame(rows)


def _grouped_bars(ax, data, title, ylabel, fw_seen):
    present = [a for a in ALGO_ORDER if a in data["algorithm"].values]
    n_algos = len(present)
    n_fw    = len(FRAMEWORK_ORDER)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    for i, fw in enumerate(FRAMEWORK_ORDER):
        sub = data[data["framework"] == fw]
        if sub.empty:
            continue
        fw_seen.add(fw)
        vals = [
            sub[sub["algorithm"] == a]["value"].values[0]
            if a in sub["algorithm"].values else np.nan
            for a in present
        ]
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, vals, width, color=COLORS[fw], hatch=HATCHES[fw],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot_ipc_and_l1(df):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.6))
    fw_seen = set()

    for col, platform in enumerate(PLATFORM_ORDER):
        sub_platform = df[df["platform"] == platform]
        ipc_data = _mean_per_algo_fw(sub_platform, "ipc")
        l1_data  = _mean_per_algo_fw(sub_platform, "l1_miss_rate_pct")

        _grouped_bars(axes[0, col], ipc_data, f"IPC — {PLATFORM_LABEL[platform]}",
                      "IPC (mean across variants)" if col == 0 else "", fw_seen)
        _grouped_bars(axes[1, col], l1_data, f"L1-D Miss Rate — {PLATFORM_LABEL[platform]}",
                      "L1-D miss rate (%)" if col == 0 else "", fw_seen)

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in fw_seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss.pdf")
    plt.close(fig)


def plot_branch_misses(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    fw_seen = set()

    for i, (ax, platform) in enumerate(zip(axes, PLATFORM_ORDER)):
        sub_platform = df[df["platform"] == platform]
        bm_data = _mean_per_algo_fw(sub_platform, "branch_misses")
        _grouped_bars(ax, bm_data, PLATFORM_LABEL[platform],
                      "Branch misses (per 1000 inferences)" if i == 0 else "", fw_seen)
        ax.set_yscale("log")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in fw_seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Total branch misses per 1000 inferences")
    fig.tight_layout()
    save(fig, "hardware_branch_misses.pdf")
    plt.close(fig)


def main():
    df = load_perf()
    if df.empty:
        print("No hardware/perf data found for RPi or RISC-V")
        return
    plot_ipc_and_l1(df)
    plot_branch_misses(df)


if __name__ == "__main__":
    main()
