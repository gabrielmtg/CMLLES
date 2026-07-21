import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import (
    load_binary, setup_style, save, overlay_bar,
    FRAMEWORK_ORDER, COLORS, PLATFORM_ORDER, PLATFORM_LABEL, PLATFORM_HATCH,
)

setup_style()

ALGO_ORDER = ["MLP-Iris", "MLP-IAES", "Autoencoder", "CNN2D", "LSTM"]
HEADER_COLOR = "#1A2F55"


def bar_footprint_summary(df):
    """One row per framework, one bar per platform (RPi solid, RISC-V
    hatched) - mean static footprint across algorithms, log scale since
    NCNN sits well above the rest. Each bar is annotated with its
    total/.text/.data/.bss breakdown, same style as analysis/analysis_riscv."""
    sub = df[df["algorithm"].isin(ALGO_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)]
    if sub.empty:
        return

    agg = sub.groupby(["framework", "platform"], as_index=False)[
        ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]
    ].mean()
    for col in ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]:
        agg[col] = agg[col] / 1024
    agg = agg.rename(columns={
        "text_bytes": "text_kb", "data_bytes": "data_kb",
        "bss_bytes": "bss_kb", "total_bytes": "total_kb",
    })

    rpi_order = agg[agg["platform"] == "RPi"].set_index("framework")["total_kb"]
    frameworks = sorted(
        agg["framework"].unique(),
        key=lambda f: rpi_order.get(f, agg[agg["framework"] == f]["total_kb"].mean()),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(12, 1.0 * len(frameworks) + 1.1))
    y_pos = np.arange(len(frameworks))
    bar_h = 0.38

    max_total = agg["total_kb"].max()
    for j, platform in enumerate(PLATFORM_ORDER):
        offset = (j - (len(PLATFORM_ORDER) - 1) / 2) * bar_h
        vals = []
        for fw in frameworks:
            row = agg[(agg["framework"] == fw) & (agg["platform"] == platform)]
            vals.append(row["total_kb"].values[0] if not row.empty else np.nan)
        ax.barh(y_pos + offset, vals, height=bar_h * 0.9,
                color=[COLORS[f] for f in frameworks],
                hatch=PLATFORM_HATCH[platform], edgecolor="white", linewidth=0.5)

        for i, fw in enumerate(frameworks):
            row = agg[(agg["framework"] == fw) & (agg["platform"] == platform)]
            if row.empty:
                continue
            row = row.iloc[0]
            label = (f"{row.total_kb:,.0f} KB  "
                      f"(.text {row.text_kb:,.0f} / .data {row.data_kb:,.0f} / .bss {row.bss_kb:,.0f})")
            ax.text(row.total_kb * 1.08, y_pos[i] + offset, label,
                    va="center", fontsize=10, color=HEADER_COLOR)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frameworks, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Static memory footprint, mean across algorithms (KB, log scale)", fontsize=12)
    ax.set_title("Executable memory footprint by framework and platform", fontsize=13)
    ax.set_xlim(right=max_total * 40)

    handles = [mpatches.Patch(facecolor="white", edgecolor="#444444",
                               hatch=PLATFORM_HATCH[p], label=PLATFORM_LABEL[p])
               for p in PLATFORM_ORDER]
    ax.legend(handles=handles, loc="upper right", fontsize=11)

    fig.tight_layout()
    save(fig, "memory_footprint.pdf")
    plt.close(fig)


def bar_footprint_summary_overlay(df):
    """Same content as bar_footprint_summary, but platforms overlaid at the
    same y position instead of paired offset rows (see utils.overlay_bar).
    Each bar keeps its total/.text/.data/.bss breakdown annotation, same
    as bar_footprint_summary, stacked as two lines (RPi above, RISC-V
    below) since both bars now share one row."""
    sub = df[df["algorithm"].isin(ALGO_ORDER) & df["framework"].isin(FRAMEWORK_ORDER)]
    if sub.empty:
        return

    agg = sub.groupby(["framework", "platform"], as_index=False)[
        ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]
    ].mean()
    for col in ["text_bytes", "data_bytes", "bss_bytes", "total_bytes"]:
        agg[col] = agg[col] / 1024
    agg = agg.rename(columns={
        "text_bytes": "text_kb", "data_bytes": "data_kb",
        "bss_bytes": "bss_kb", "total_bytes": "total_kb",
    })

    rpi_order = agg[agg["platform"] == "RPi"].set_index("framework")["total_kb"]
    frameworks = sorted(
        agg["framework"].unique(),
        key=lambda f: rpi_order.get(f, agg[agg["framework"] == f]["total_kb"].mean()),
        reverse=True,
    )

    fig, ax = plt.subplots(figsize=(12, 0.9 * len(frameworks) + 1.1))
    y_pos = np.arange(len(frameworks))
    text_offset = 0.16

    max_total = agg["total_kb"].max()
    for i, fw in enumerate(frameworks):
        rpi_row = agg[(agg["framework"] == fw) & (agg["platform"] == "RPi")]
        rv_row = agg[(agg["framework"] == fw) & (agg["platform"] == "RISC-V")]
        rpi_v = rpi_row["total_kb"].values[0] if not rpi_row.empty else np.nan
        rv_v = rv_row["total_kb"].values[0] if not rv_row.empty else np.nan
        overlay_bar(ax, y_pos[i], rpi_v, rv_v, COLORS[fw], size=0.7, horizontal=True)

        for row, plat, dy in ((rpi_row, "RPi", text_offset), (rv_row, "RISC-V", -text_offset)):
            if row.empty:
                continue
            row = row.iloc[0]
            label = (f"{plat}: {row.total_kb:,.0f} KB  "
                      f"(.text {row.text_kb:,.0f} / .data {row.data_kb:,.0f} / .bss {row.bss_kb:,.0f})")
            ax.text(row.total_kb * 1.08, y_pos[i] + dy, label,
                    va="center", fontsize=9, color=HEADER_COLOR)

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(frameworks, fontsize=12)
    ax.tick_params(axis="x", labelsize=10)
    ax.set_xlabel("Static memory footprint, mean across algorithms (KB, log scale)", fontsize=12)
    ax.set_title("Memory footprint, RPi vs RISC-V overlaid", fontsize=13)
    ax.set_xlim(right=max_total * 40)

    style_handles = [
        mpatches.Patch(facecolor="#888888", alpha=0.35, label="Wide/translucent = larger value"),
        mpatches.Patch(facecolor="#888888", alpha=1.0, hatch="///", label="Narrow/solid, hatched = RISC-V"),
    ]
    ax.legend(handles=style_handles, loc="upper right", fontsize=10)

    fig.tight_layout()
    save(fig, "memory_footprint_overlay.pdf")
    plt.close(fig)


def main():
    df = load_binary()
    if df.empty:
        print("binary_metrics.csv / binary_metrics_riscv.csv not found or empty")
        return
    bar_footprint_summary(df)
    bar_footprint_summary_overlay(df)


if __name__ == "__main__":
    main()
