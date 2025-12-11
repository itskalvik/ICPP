#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    """Load benchmark results JSON."""
    with open(path, "r") as f:
        return json.load(f)


def prepare_data(data):
    """
    Organize runs by method and variance ratio and extract metrics.

    Expects a structure like:
      {
        "variance_ratios": [...],
        "methods": [...],
        "runs": [
           {
             "method": ...,
             "variance_ratio": ...,
             "num_placements": ...,
             "max_posterior_var": ...,
             "target_var_threshold": ...,
             "mse": ...,
             "smse": ...,
             "runtime_sec": ...,
             "distance_m": ...
           },
           ...
        ]
      }
    """
    runs = data["runs"]

    methods = sorted({r["method"] for r in runs})
    # Use all variance ratios present in the runs to be robust
    variance_ratios = sorted({float(r["variance_ratio"]) for r in runs})

    # Map (method, variance_ratio) -> run dict
    lookup = {(r["method"], float(r["variance_ratio"])): r for r in runs}

    # Per-method metric arrays aligned with variance_ratios
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

    # Target variance threshold is the same for all methods at a given ratio;
    # grab it once per ratio.
    target_var_by_ratio = {}
    for vr in variance_ratios:
        for m in methods:
            run = lookup.get((m, vr))
            if run is not None:
                target_var_by_ratio[vr] = run["target_var_threshold"]
                break

    # Fill metric arrays; if a method is missing a ratio, use NaN
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


def plot_metrics(methods, variance_ratios, per_method, target_var_by_ratio,
                 output_path=None):
    """Create 6 subplots of metrics vs variance ratio for each method."""
    metrics = [
        ("max_posterior_var", "Max posterior variance"),
        ("num_placements", "Number of placements"),
        ("mse", "MSE"),
        ("smse", "SMSE"),
        ("runtime_sec", "Runtime (s)"),
        ("distance_m", "Distance (m)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes = axes.ravel()

    for ax, (key, label) in zip(axes, metrics):
        for method in methods:
            d = per_method[method]
            ax.plot(
                d["variance_ratio"],
                d[key],
                marker="o",
                linestyle="-",
                label=method if key == "max_posterior_var" else None,
            )

        # Add target variance line on the max posterior variance plot
        if key == "max_posterior_var":
            target_vals = [target_var_by_ratio[vr] for vr in variance_ratios]
            ax.plot(
                variance_ratios,
                target_vals,
                linestyle="--",
                linewidth=1.5,
                label="Target variance",
            )
            ax.legend()

        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)

    # Shared x-label for bottom row
    axes[-2].set_xlabel("Variance ratio")
    axes[-1].set_xlabel("Variance ratio")

    fig.suptitle("Coverage benchmark metrics vs variance ratio", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot coverage benchmark metrics vs variance ratio."
    )
    parser.add_argument(
        "json_path",
        help="Path to the results JSON file (e.g., results.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Optional path to save the figure (e.g., metrics_vs_ratio.png)",
    )
    args = parser.parse_args()

    data = load_results(args.json_path)
    methods, variance_ratios, per_method, target_var_by_ratio = prepare_data(data)
    plot_metrics(methods, variance_ratios, per_method, target_var_by_ratio, args.output)


if __name__ == "__main__":
    main()
