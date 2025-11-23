from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# CONFIG: set these paths
CSV_PATH = Path("prompts_averages.csv")
OUTPUT_PATH = Path("metrics_grouped_bars.png")

# Configuration of figure size, spacing, style
FIG_WIDTH  = 8
FIG_HEIGHT = 10

BAR_WIDTH   = 0.10
BAR_PAD     = 0.02
GROUP_GAP   = 0.10
TICK_ROT    = 45
TICK_HA     = "right"
TICK_ROT_MODE = "anchor"

GROUPS = OrderedDict([
    ("xl-auto", ["xl", "auto"]),
    ("short-long", ["short", "long"]),
    ("zero-one-three", ["zero", "one", "three"]),
    ("non_RCI-RCI", ["non_RCI", "RCI"]),
])

COND_ORDER = ["xl", "auto", "short", "long", "zero", "one", "three", "non_RCI", "RCI"]


def read_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "Test name" not in df.columns:
        raise ValueError("CSV must contain a 'Test name' column.")

    df = df[~df["Test name"].str.contains("difference", case=False, na=False)].copy()

    # Ensure numeric columns
    for col in df.columns:
        if col != "Test name":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.set_index("Test name", inplace=True)
    return df


def compute_layout_positions(groups, bar_width=BAR_WIDTH, bar_pad=BAR_PAD, group_gap=GROUP_GAP):
    cond_to_xs = {c: [] for c in COND_ORDER}
    current_x = 0.0

    for _, conds in groups.items():
        positions_in_group = []
        for cond in conds:
            x = current_x
            positions_in_group.append(x)
            cond_to_xs[cond].append(x)
            current_x += bar_width + bar_pad

        current_x += group_gap

    return cond_to_xs


def plot_grouped_bars(df: pd.DataFrame, output_path: Path):
    metrics = df.columns.tolist()

    cond_to_xs = compute_layout_positions(GROUPS)

    cond_to_vals_by_metric = {}
    for metric in metrics:
        per_cond_vals = {c: [] for c in COND_ORDER}
        for _, conds in GROUPS.items():
            for cond in conds:
                val = np.nan
                if cond in df.index:
                    val = df.at[cond, metric]
                per_cond_vals[cond].append(val)

        cond_to_vals_by_metric[metric] = {
            cond: vals for cond, vals in per_cond_vals.items()
            if len(vals) > 0 and not all(np.isnan(vals))
        }

    ncols, nrows = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(FIG_WIDTH, FIG_HEIGHT), squeeze=False)
    axes_flat = axes.flatten()

    # Per-bar ticks: one tick per condition
    bar_xs = [cond_to_xs[c][0] for c in COND_ORDER if cond_to_xs.get(c)]
    bar_labels = [c for c in COND_ORDER if cond_to_xs.get(c)]

    for ax, metric in zip(axes_flat, metrics):
        for cond in COND_ORDER:
            xs = cond_to_xs.get(cond, [])
            vals = cond_to_vals_by_metric[metric].get(cond, [])
            if xs and vals:
                ax.bar(xs, vals, width=BAR_WIDTH, label=cond)

        ax.set_title(metric)
        ax.set_xticks(bar_xs)
        ax.set_xticklabels(bar_labels, rotation=TICK_ROT, ha=TICK_HA, rotation_mode=TICK_ROT_MODE)
        ax.set_ylabel("Score")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    for j in range(len(metrics), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])

    # Save and show
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    plt.show()


def main():
    df = read_data(CSV_PATH)

    expected = set([c for conds in GROUPS.values() for c in conds])
    missing = [c for c in expected if c not in df.index]
    if missing:
        print(f"Warning: Missing rows for conditions: {', '.join(missing)}")

    plot_grouped_bars(df, OUTPUT_PATH)


if __name__ == "__main__":
    main()
