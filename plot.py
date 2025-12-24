#!/usr/bin/env python3
import json
import argparse
import os

import numpy as np

# --- NO TYPE 3 FONTS IN PDF OUTPUT ---
import matplotlib as mpl
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"
mpl.rcParams["mathtext.fontset"] = "dejavusans"
mpl.rcParams["text.usetex"] = False

# Font sizes
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 14
mpl.rcParams["xtick.labelsize"] = 14
mpl.rcParams["ytick.labelsize"] = 14
mpl.rcParams["legend.fontsize"] = 12
# -------------------------------------

import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def prepare_data(data):
    runs = data["runs"]
    methods = sorted({r["method"] for r in runs})
    variance_ratios = sorted({float(r["variance_ratio"]) for r in runs})

    lookup = {(r["method"], float(r["variance_ratio"])): r for r in runs}

    per_method = {
        m: {
            "variance_ratio": [],
            "max_posterior_var": [],
            "num_placements": [],
            "mse": [],
            "smse": [],
            "runtime_sec": [],
            "distance_m": [],
        }
        for m in methods
    }

    target_var_by_ratio = {}
    for vr in variance_ratios:
        for m in methods:
            run = lookup.get((m, vr))
            if run is not None:
                target_var_by_ratio[vr] = run["target_post_var_threshold"]
                break

    for m in methods:
        for vr in variance_ratios:
            run = lookup.get((m, vr))
            per_method[m]["variance_ratio"].append(vr)
            if run is None:
                per_method[m]["max_posterior_var"].append(np.nan)
                per_method[m]["num_placements"].append(np.nan)
                per_method[m]["mse"].append(np.nan)
                per_method[m]["smse"].append(np.nan)
                per_method[m]["runtime_sec"].append(np.nan)
                per_method[m]["distance_m"].append(np.nan)
            else:
                per_method[m]["max_posterior_var"].append(run["max_posterior_var"])
                per_method[m]["num_placements"].append(run["num_placements"])
                per_method[m]["mse"].append(run["mse"])
                per_method[m]["smse"].append(run["smse"])
                per_method[m]["runtime_sec"].append(run["runtime_sec"])
                per_method[m]["distance_m"].append(run["distance_m"])

    return methods, variance_ratios, per_method, target_var_by_ratio


def _sanitize(s: str) -> str:
    """Make a safe filename token."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in s)


def plot_metrics(methods, variance_ratios, per_method, target_var_by_ratio,
                 output_base="metrics_vs_ratio.pdf"):
    """
    Create one figure per metric, save each to its own PDF, then show.
    output_base can be:
      - a directory (existing or not): outputs saved inside it
      - or a filename like 'metrics.pdf': base name used for per-metric PDFs
    """
    metrics = [
        ("max_posterior_var", "Max posterior variance"),
        ("num_placements", "Number of placements"),
        ("mse", "MSE"),
        ("smse", "SMSE"),
        ("runtime_sec", "Runtime (s)"),
        ("distance_m", "Distance (m)"),
    ]

    # Decide output directory + base stem
    output_base = output_base or "metrics_vs_ratio.pdf"
    if output_base.lower().endswith(".pdf"):
        out_dir = os.path.dirname(output_base) or "."
        stem = os.path.splitext(os.path.basename(output_base))[0]
    else:
        # treat as directory
        out_dir = output_base
        stem = "metrics_vs_ratio"

    os.makedirs(out_dir, exist_ok=True)

    figures = []

    for key, label in metrics:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))

        for method in methods:
            d = per_method[method]
            if "Dist" in method:
                method_ = "GCBCover with distance budget"
            else:
                method_ = method
            ax.plot(
                d["variance_ratio"],
                d[key],
                marker="o",
                linestyle="-",
                label=method_,
            )

        if key == "max_posterior_var":
            target_vals = [target_var_by_ratio[vr] for vr in variance_ratios]
            ax.plot(
                variance_ratios,
                target_vals,
                linestyle="--",
                linewidth=1.5,
                label="Target variance",
            )

        ax.set_title(f"Coverage benchmark: {label} vs variance ratio")
        ax.set_xlabel("Variance ratio")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        if "Max" in label:
            ax.legend()

        # x-axis from largest to smallest
        ax.invert_xaxis()

        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{stem}_{_sanitize(key)}.pdf")
        fig.savefig(out_path, bbox_inches="tight")  # rcParams ensure no Type 3 fonts

        figures.append(fig)

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage benchmark metrics vs variance ratio."
    )
    parser.add_argument("json_path", help="Path to the results JSON file (e.g., results.json)")
    parser.add_argument(
        "-o", "--output",
        default="metrics_vs_ratio.pdf",
        help=(
            "Output base for PDFs. If ends with .pdf, saves per-metric PDFs like "
            "'<stem>_<metric>.pdf'. If a directory path, saves into that directory."
        ),
    )
    args = parser.parse_args()

    data = load_results(args.json_path)
    methods, variance_ratios, per_method, target_var_by_ratio = prepare_data(data)
    plot_metrics(methods, variance_ratios, per_method, target_var_by_ratio, args.output)


if __name__ == "__main__":
    main()
