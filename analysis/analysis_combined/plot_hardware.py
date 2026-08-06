import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
from utils import (
    load_perf, load_latencies, setup_style, save, overlay_bar, platform_legend_handles,
    FRAMEWORK_ORDER, COLORS, HATCHES, PLATFORM_ORDER, PLATFORM_LABEL,
    INT8_FRAMEWORKS_BY_PLATFORM, PRECISION_ORDER,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
ALGO_LABEL = {
    "MLP-Iris":    "MLP (Iris)",
    "MLP-IAES":   "MLP (IAES)",
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


def _grouped_bars(ax, data, ylabel, fw_seen):
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


def plot_ipc_and_l1(df):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    fw_seen = set()

    for col, platform in enumerate(PLATFORM_ORDER):
        sub_platform = df[df["platform"] == platform]
        ipc_data = _mean_per_algo_fw(sub_platform, "ipc")
        l1_data  = _mean_per_algo_fw(sub_platform, "l1_miss_rate_pct")

        _grouped_bars(axes[0, col], ipc_data,
                      "IPC (mean across variants)" if col == 0 else "", fw_seen)
        _grouped_bars(axes[1, col], l1_data,
                      "L1-D miss rate (%)" if col == 0 else "", fw_seen)
        axes[1, col].set_xlabel("Algorithm")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in fw_seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss.pdf")
    plt.close(fig)


def plot_branch_misses(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), sharey=True)
    fw_seen = set()

    for i, (ax, platform) in enumerate(zip(axes, PLATFORM_ORDER)):
        sub_platform = df[df["platform"] == platform]
        bm_data = _mean_per_algo_fw(sub_platform, "branch_misses")
        _grouped_bars(ax, bm_data,
                      "Branch misses (per 1000 inferences)" if i == 0 else "", fw_seen)
        ax.set_yscale("log")
        ax.set_xlabel("Algorithm")

    handles = [mpatches.Patch(facecolor=COLORS[f], label=f, hatch=HATCHES[f])
               for f in FRAMEWORK_ORDER if f in fw_seen]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    save(fig, "hardware_branch_misses.pdf")
    plt.close(fig)


def _grouped_bars_overlay(ax, df, metric, fw_seen, fw_list=None):
    """Same grouping as _grouped_bars, but the two platforms are overlaid
    at the same x position instead of living in separate subplots."""
    fw_list = fw_list if fw_list is not None else FRAMEWORK_ORDER
    present = [a for a in ALGO_ORDER if a in df["algorithm"].values]
    x_pos = np.arange(len(present))
    width = 0.8 / len(fw_list)

    for xi, algo in enumerate(present):
        algo_df = df[df["algorithm"] == algo]
        algo_fws = [fw for fw in fw_list if not algo_df[algo_df["framework"] == fw].empty]
        n_present = len(algo_fws)
        for i, fw in enumerate(algo_fws):
            sub = algo_df[algo_df["framework"] == fw]
            fw_seen.add(fw)
            offset = (i - n_present / 2 + 0.5) * width
            rpi_sub   = sub[sub["platform"] == "RPi"][metric]
            riscv_sub = sub[sub["platform"] == "RISC-V"][metric]
            overlay_bar(ax, x_pos[xi] + offset, rpi_sub.mean(), riscv_sub.mean(), COLORS[fw],
                        size=width * 0.95, err_rpi=rpi_sub.std(), err_riscv=riscv_sub.std())

    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present])


def plot_ipc_and_l1_overlay(df):
    fig, axes = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True)
    fw_seen = set()

    _grouped_bars_overlay(axes[0], df, "ipc", fw_seen)
    axes[0].set_ylabel("IPC (mean across variants)", fontsize=14)
    axes[0].set_ylim(top=df["ipc"].max() * 1.15)
    axes[0].tick_params(axis="both", labelsize=13)

    _grouped_bars_overlay(axes[1], df, "l1_miss_rate_pct", fw_seen)
    axes[1].set_ylabel("L1-D miss rate (%)", fontsize=14)
    axes[1].set_ylim(top=10)
    axes[1].tick_params(axis="both", labelsize=13)
    axes[1].set_xlabel("Algorithm")

    # No fig.suptitle here (redundant with the LaTeX caption) and the
    # framework legend moved into the emptied headroom at the top of each
    # panel instead of a separate row below the figure - keeps the whole
    # image noticeably shorter.
    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in FRAMEWORK_ORDER if f in fw_seen]
    axes[0].legend(handles=fw_handles, loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    axes[1].legend(handles=platform_legend_handles(), loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss_overlay.pdf")
    plt.close(fig)


def plot_ipc_and_l1_overlay_cols(df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.2))
    fw_seen = set()

    _grouped_bars_overlay(axes[0], df, "ipc", fw_seen)
    axes[0].set_ylabel("IPC (mean across variants)", fontsize=14)
    axes[0].set_ylim(top=df["ipc"].max() * 1.15)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].set_xlabel("Algorithm")

    _grouped_bars_overlay(axes[1], df, "l1_miss_rate_pct", fw_seen)
    axes[1].set_ylabel("L1-D miss rate (%)", fontsize=14)
    axes[1].set_ylim(top=10)
    axes[1].tick_params(axis="both", labelsize=13)
    axes[1].set_xlabel("Algorithm")

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in FRAMEWORK_ORDER if f in fw_seen]
    axes[0].legend(handles=fw_handles, loc="upper left", ncol=3, fontsize=8, framealpha=0.9)
    axes[1].legend(handles=platform_legend_handles(), loc="upper right", fontsize=8)
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss_overlay_cols.pdf")
    plt.close(fig)


def plot_branch_misses_overlay(df):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    fw_seen = set()

    _grouped_bars_overlay(ax, df, "branch_misses", fw_seen)
    ax.set_yscale("log")
    ax.set_ylabel("Branch misses")
    ax.set_xlabel("Algorithm")
    ax.set_ylim(top=df["branch_misses"].max() * 3.5)

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in FRAMEWORK_ORDER if f in fw_seen]
    leg1 = ax.legend(handles=fw_handles, loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=platform_legend_handles(), loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    save(fig, "hardware_branch_misses_overlay.pdf")
    plt.close(fig)


def plot_ipc_l1miss_branch_overlay(df):
    fig, axes = plt.subplots(1, 3, figsize=(21, 4.2))
    fw_seen = set()

    _grouped_bars_overlay(axes[0], df, "ipc", fw_seen)
    axes[0].set_ylabel("IPC (mean across variants)", fontsize=14)
    axes[0].set_ylim(top=df["ipc"].max() * 1.30)
    axes[0].tick_params(axis="both", labelsize=13)
    axes[0].set_xlabel("Algorithm")
    axes[0].set_title("(a)", loc="center", fontsize=14)

    _grouped_bars_overlay(axes[1], df, "l1_miss_rate_pct", fw_seen)
    axes[1].set_ylabel("L1-D miss rate (%)", fontsize=14)
    axes[1].set_ylim(bottom=0, top=12)
    axes[1].tick_params(axis="both", labelsize=13)
    axes[1].set_xlabel("Algorithm")
    axes[1].set_title("(b)", loc="center", fontsize=14)

    _grouped_bars_overlay(axes[2], df, "branch_misses", fw_seen)
    axes[2].set_yscale("log")
    axes[2].set_ylabel("Branch misses", fontsize=14)
    axes[2].tick_params(axis="both", labelsize=13)
    axes[2].set_xlabel("Algorithm")
    axes[2].set_ylim(top=df["branch_misses"].max() * 3.5)
    axes[2].set_title("(c)", loc="center", fontsize=14)

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in FRAMEWORK_ORDER if f in fw_seen]
    axes[0].legend(handles=fw_handles, loc="upper left", ncol=3, fontsize=12, framealpha=0.9)
    axes[2].legend(handles=platform_legend_handles(), loc="upper left", fontsize=15)
    fig.tight_layout()
    save(fig, "hardware_ipc_l1miss_branch_overlay.pdf")
    plt.close(fig)


def plot_int8_hardware_overlay(df):
    fw_list = INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]  # ["LiteRT", "ONNX"], both precisions on both platforms
    sub = df[df["framework"].isin(fw_list)]
    present_algos = [a for a in ALGO_ORDER if a in sub["algorithm"].values]
    if sub.empty or not present_algos:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    fw_seen = set()
    metrics = [("ipc", "IPC (mean across variants)", "IPC"),
               ("l1_miss_rate_pct", "L1-D miss rate (%)", "L1-D Miss Rate")]

    for row, (metric, ylabel, title) in enumerate(metrics):
        row_max = sub[metric].max()
        for col, prec in enumerate(PRECISION_ORDER):
            sub_prec = sub[sub["precision"] == prec]
            _grouped_bars_overlay(axes[row, col], sub_prec, metric, fw_seen, fw_list=fw_list)
            if col == 0:
                axes[row, col].set_ylabel(ylabel)
            if metric == "l1_miss_rate_pct":
                axes[row, col].set_ylim(bottom=0, top=10)
            else:
                axes[row, col].set_ylim(top=row_max * 1.15)
            axes[row, col].set_title(f"{title} — {prec.upper()}")
            if row == 1:
                axes[row, col].set_xlabel("Algorithm")

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list if f in fw_seen]
    axes[0, 0].legend(handles=fw_handles, loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
    axes[0, 1].legend(handles=platform_legend_handles(), loc="upper right", fontsize=8)

    fig.tight_layout()
    save(fig, "hardware_int8_precision_overlay.pdf")
    plt.close(fig)


def plot_int8_hardware_overlay_row(df):
    fw_list = INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]
    sub = df[df["framework"].isin(fw_list)]
    present_algos = [a for a in ALGO_ORDER if a in sub["algorithm"].values]
    if sub.empty or not present_algos:
        return

    fig, axes = plt.subplots(1, 4, figsize=(21, 4.2))
    fw_seen = set()
    specs = [
        ("ipc", "f32", "IPC (mean across variants)"),
        ("ipc", "int8", "IPC (mean across variants)"),
        ("l1_miss_rate_pct", "f32", "L1-D miss rate (%)"),
        ("l1_miss_rate_pct", "int8", "L1-D miss rate (%)"),
    ]
    letters = "abcd"
    for ax, letter, (metric, prec, ylabel) in zip(axes, letters, specs):
        _draw_precision_panel(ax, sub, metric, prec, fw_list, fw_seen, ylabel)
        ax.set_xticks(np.arange(len(present_algos)))
        ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
        ax.set_xlabel("Algorithm")
        ax.set_title(f"({letter})", loc="center", fontsize=14)
        offset = mtransforms.ScaledTranslation(-0.12, 0, fig.dpi_scale_trans)
        ax.get_xticklabels()[0].set_transform(ax.get_xticklabels()[0].get_transform() + offset)

    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list if f in fw_seen]
    axes[0].legend(handles=fw_handles, loc="upper left", ncol=2, fontsize=15, framealpha=0.9)
    axes[3].legend(handles=platform_legend_handles(), loc="upper left", fontsize=15)

    fig.tight_layout()
    save(fig, "hardware_int8_precision_overlay_row.pdf")
    plt.close(fig)


def _speedup_fw_list():
    return [f for f in INT8_FRAMEWORKS_BY_PLATFORM["RPi"]
            if f != "CMSIS-NN" and f in INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]]


def _draw_speedup_panel(ax, df_lat, present_algos, x_pos):
    fw_list = _speedup_fw_list()
    n_fw = len(fw_list)
    width = 0.8 / n_fw
    ax.axhline(1.0, color="#888888", linewidth=0.8, linestyle="--")
    for i, fw in enumerate(fw_list):
        offset = (i - n_fw / 2 + 0.5) * width
        for xi, algo in enumerate(present_algos):
            speedups = {}
            for platform in PLATFORM_ORDER:
                sub = df_lat[(df_lat["platform"] == platform) & (df_lat["framework"] == fw) & (df_lat["algorithm"] == algo)]
                m_f32  = sub[sub["precision"] == "f32"]["latency_us"].mean()
                m_int8 = sub[sub["precision"] == "int8"]["latency_us"].mean()
                speedups[platform] = m_f32 / m_int8 if (not np.isnan(m_f32) and not np.isnan(m_int8) and m_int8 > 0) else np.nan
            overlay_bar(ax, x_pos[xi] + offset, speedups["RPi"], speedups["RISC-V"], COLORS[fw], size=width * 0.95)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
    ax.set_ylabel("Speedup (F32 / INT8)")
    fw_handles = [mpatches.Patch(facecolor=COLORS[f], label=f) for f in fw_list]
    leg1 = ax.legend(handles=fw_handles, loc="upper left", fontsize=8, framealpha=0.9)
    ax.add_artist(leg1)
    ax.legend(handles=platform_legend_handles(), loc="upper right", fontsize=8, framealpha=0.9)


def _draw_precision_panel(ax, df_perf, metric, prec, fw_list, fw_seen, ylabel):
    sub_prec = df_perf[df_perf["precision"] == prec]
    _grouped_bars_overlay(ax, sub_prec, metric, fw_seen, fw_list=fw_list)
    ax.set_ylabel(ylabel)
    if metric == "l1_miss_rate_pct":
        ax.set_ylim(bottom=0, top=12)
    else:
        ax.set_ylim(top=df_perf[metric].max() * 1.15)


def plot_int8_precision_speedup_row(df_perf, df_lat):
    fw_list = INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]
    sub_perf = df_perf[df_perf["framework"].isin(fw_list)]
    present_algos = [a for a in ALGO_ORDER if a in sub_perf["algorithm"].values]
    if sub_perf.empty or not present_algos:
        return
    x_pos = np.arange(len(present_algos))
    fw_seen = set()

    fig, axes = plt.subplots(1, 5, figsize=(26, 4.2))
    specs = [
        ("speedup", None, None, "Speedup (F32/INT8)"),
        ("ipc", "ipc", "f32", "IPC — F32"),
        ("ipc", "ipc", "int8", "IPC — INT8"),
        ("l1", "l1_miss_rate_pct", "f32", "L1-D Miss Rate — F32"),
        ("l1", "l1_miss_rate_pct", "int8", "L1-D Miss Rate — INT8"),
    ]
    letters = "abcde"
    for ax, letter, (kind, metric, prec, title) in zip(axes, letters, specs):
        if kind == "speedup":
            _draw_speedup_panel(ax, df_lat, present_algos, x_pos)
        else:
            _draw_precision_panel(ax, sub_perf, metric, prec, fw_list, fw_seen, title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
        ax.set_xlabel("Algorithm")
        ax.set_title(f"({letter})", loc="left", fontsize=14)

    fig.tight_layout()
    save(fig, "hardware_int8_precision_speedup_row.pdf")
    plt.close(fig)


def plot_int8_precision_speedup_grid(df_perf, df_lat):
    fw_list = INT8_FRAMEWORKS_BY_PLATFORM["RISC-V"]
    sub_perf = df_perf[df_perf["framework"].isin(fw_list)]
    present_algos = [a for a in ALGO_ORDER if a in sub_perf["algorithm"].values]
    if sub_perf.empty or not present_algos:
        return
    x_pos = np.arange(len(present_algos))
    fw_seen = set()

    fig = plt.figure(figsize=(19, 8.2))
    gs = fig.add_gridspec(2, 6)
    top = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])]
    bottom = [fig.add_subplot(gs[1, 1:3]), fig.add_subplot(gs[1, 3:5])]

    top_specs = [
        ("speedup", None, None, "Speedup (F32/INT8)"),
        ("ipc", "ipc", "f32", "IPC — F32"),
        ("ipc", "ipc", "int8", "IPC — INT8"),
    ]
    bottom_specs = [
        ("l1", "l1_miss_rate_pct", "f32", "L1-D Miss Rate — F32"),
        ("l1", "l1_miss_rate_pct", "int8", "L1-D Miss Rate — INT8"),
    ]

    letters = iter("abcde")
    for ax, (kind, metric, prec, title) in zip(top + bottom, top_specs + bottom_specs):
        if kind == "speedup":
            _draw_speedup_panel(ax, df_lat, present_algos, x_pos)
        else:
            _draw_precision_panel(ax, sub_perf, metric, prec, fw_list, fw_seen, title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([ALGO_LABEL[a] for a in present_algos])
        ax.set_xlabel("Algorithm")
        ax.set_title(f"({next(letters)})", loc="left", fontsize=14)

    fig.tight_layout()
    save(fig, "hardware_int8_precision_speedup_grid.pdf")
    plt.close(fig)


def main():
    df = load_perf()
    if df.empty:
        print("No hardware/perf data found for RPi or RISC-V")
        return
    plot_ipc_and_l1(df)
    plot_branch_misses(df)
    plot_ipc_and_l1_overlay(df)
    plot_ipc_and_l1_overlay_cols(df)
    plot_branch_misses_overlay(df)
    plot_ipc_l1miss_branch_overlay(df)
    plot_int8_hardware_overlay(df)
    plot_int8_hardware_overlay_row(df)

    df_lat = load_latencies()
    if not df_lat.empty:
        plot_int8_precision_speedup_row(df, df_lat)
        plot_int8_precision_speedup_grid(df, df_lat)


if __name__ == "__main__":
    main()
