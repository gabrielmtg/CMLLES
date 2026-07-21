import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_latencies, load_perf, setup_style, save,
    INT8_FRAMEWORKS, COLORS, HATCHES,
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

PREC_ALPHA = {"f32": 0.45, "int8": 1.0}
PREC_HATCH = {"f32": "///", "int8": ""}


def _mean_lat(df, algo, fw, prec):
    sub = df[(df["algorithm"] == algo) & (df["framework"] == fw) & (df["precision"] == prec)]
    if sub.empty:
        return np.nan
    return sub["latency_us"].mean()


def plot_int8_latency_comparison(df):
    present_algos = [a for a in ALGO_ORDER if a in df["algorithm"].values]
    if not present_algos:
        return

    n_algos = len(present_algos)
    fw_list  = [f for f in INT8_FRAMEWORKS if f in df["framework"].values]
    n_fw     = len(fw_list)

    bars_per_group = n_fw * 2 - 1
    width = 0.8 / bars_per_group
    x_pos = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    slot = 0
    handles = []
    for fw in fw_list:
        if fw == "CMSIS-NN":
            vals = [_mean_lat(df, a, fw, "int8") for a in present_algos]
            offset = (slot - bars_per_group / 2 + 0.5) * width
            ax.bar(x_pos + offset, vals, width,
                   color=COLORS[fw], edgecolor="white", linewidth=0.5)
            handles.append(mpatches.Patch(facecolor=COLORS[fw], label=f"{fw} (INT8)"))
            slot += 1
        else:
            for prec in ["f32", "int8"]:
                vals = [_mean_lat(df, a, fw, prec) for a in present_algos]
                offset = (slot - bars_per_group / 2 + 0.5) * width
                ax.bar(x_pos + offset, vals, width,
                       color=COLORS[fw], alpha=PREC_ALPHA[prec],
                       hatch=PREC_HATCH[prec],
                       edgecolor="white", linewidth=0.5)
                slot += 1
            handles.append(mpatches.Patch(facecolor=COLORS[fw], alpha=PREC_ALPHA["f32"],
                                          hatch=PREC_HATCH["f32"], label=f"{fw} F32"))
            handles.append(mpatches.Patch(facecolor=COLORS[fw], alpha=PREC_ALPHA["int8"],
                                          label=f"{fw} INT8"))

    ax.set_yscale("log")
    ax.set_ylabel("Average inference latency (µs)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    ax.set_xlabel("Algorithm")
    ax.set_title("INT8 vs F32 inference latency comparison")
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "int8_latency_comparison.pdf")
    plt.close(fig)


def plot_int8_speedup(df):
    fw_list = [f for f in INT8_FRAMEWORKS if f != "CMSIS-NN" and f in df["framework"].values]
    if not fw_list:
        return
    present_algos = [a for a in ALGO_ORDER if a in df["algorithm"].values]
    if not present_algos:
        return

    n_algos = len(present_algos)
    n_fw    = len(fw_list)
    width   = 0.8 / n_fw
    x_pos   = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, fw in enumerate(fw_list):
        speedups = []
        for algo in present_algos:
            m_f32  = _mean_lat(df, algo, fw, "f32")
            m_int8 = _mean_lat(df, algo, fw, "int8")
            speedups.append(m_f32 / m_int8 if (not np.isnan(m_f32) and not np.isnan(m_int8) and m_int8 > 0) else np.nan)
        offset = (i - n_fw / 2 + 0.5) * width
        ax.bar(x_pos + offset, speedups, width, label=fw,
               color=COLORS[fw], edgecolor="white", linewidth=0.5)

    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    ax.set_ylabel("Speedup (F32 latency / INT8 latency)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    ax.set_xlabel("Algorithm")
    ax.set_title("INT8 speedup over F32 per framework")
    handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list]
    ax.legend(handles=handles, loc="upper right", ncol=2)
    fig.tight_layout()
    save(fig, "int8_speedup.pdf")
    plt.close(fig)


def plot_int8_ipc(df_perf):
    if df_perf.empty or "precision" not in df_perf.columns:
        return

    fw_list = [f for f in INT8_FRAMEWORKS if f in df_perf["framework"].values]
    present_algos = [a for a in ALGO_ORDER if a in df_perf["algorithm"].values]
    if not fw_list or not present_algos:
        return

    n_algos     = len(present_algos)
    bars_per_fw = 2
    n_bars      = len(fw_list) * bars_per_fw
    width       = 0.8 / n_bars
    x_pos       = np.arange(n_algos)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    handles = []
    slot = 0
    for fw in fw_list:
        for prec in ["f32", "int8"]:
            sub = df_perf[(df_perf["framework"] == fw) & (df_perf["precision"] == prec)]
            vals = []
            for algo in present_algos:
                asub = sub[sub["algorithm"] == algo]
                vals.append(asub["ipc"].mean() if not asub.empty else np.nan)
            offset = (slot - n_bars / 2 + 0.5) * width
            ax.bar(x_pos + offset, vals, width,
                   color=COLORS.get(fw, "#999999"),
                   alpha=PREC_ALPHA[prec],
                   hatch=PREC_HATCH[prec],
                   edgecolor="white", linewidth=0.5)
            slot += 1
        if fw == "CMSIS-NN":
            handles.append(mpatches.Patch(facecolor=COLORS.get(fw, "#999999"), label=f"{fw} (INT8)"))
        else:
            handles.append(mpatches.Patch(facecolor=COLORS.get(fw, "#999999"),
                                          alpha=PREC_ALPHA["f32"], hatch=PREC_HATCH["f32"],
                                          label=f"{fw} F32"))
            handles.append(mpatches.Patch(facecolor=COLORS.get(fw, "#999999"),
                                          alpha=PREC_ALPHA["int8"], label=f"{fw} INT8"))

    ax.set_ylabel("IPC (mean across variants)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    ax.set_xlabel("Algorithm")
    ax.set_title("IPC comparison — F32 vs INT8")
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=7)
    fig.tight_layout()
    save(fig, "int8_ipc_comparison.pdf")
    plt.close(fig)


def main():
    df = load_latencies()
    if df.empty:
        print("No latency CSV found in results/latencies_rpi/")
        return

    plot_int8_latency_comparison(df)
    plot_int8_speedup(df)

    try:
        df_perf = load_perf()
        if "precision" not in df_perf.columns:
            def _prec_from_bench(name):
                n = name.lower()
                if n.endswith("_int8"):
                    return "int8"
                if n.endswith("_f32"):
                    return "f32"
                return "int8" if "cmsisnn" in n else "f32"
            df_perf["precision"] = df_perf["benchmark"].apply(_prec_from_bench)
        plot_int8_ipc(df_perf)
    except Exception:
        pass


if __name__ == "__main__":
    main()
