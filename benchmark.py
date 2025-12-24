#!/usr/bin/env python3
import os
import json
import argparse
from time import time

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
mpl.rcParams["legend.fontsize"] = 14
# -------------------------------------

from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(1234)

import tensorflow as tf
tf.random.set_seed(1234)

import gpflow
gpflow.config.set_default_float(np.float32)
gpflow.config.set_default_jitter(1e-2)

from sgptools.methods import *
from sgptools.kernels import get_kernel
from sgptools.utils.tsp import *
from sgptools.utils.misc import *
from sgptools.utils.metrics import *
from sgptools.utils.data import Dataset
from sgptools.utils.gpflow import get_model_params


def get_grid(X_data, num_x, num_y):
    grid_x, grid_y = np.mgrid[
        min(X_data[:, 0]):max(X_data[:, 0]):complex(num_x),
        min(X_data[:, 1]):max(X_data[:, 1]):complex(num_y)
    ]
    X_grid = np.stack([grid_x, grid_y], axis=-1)
    return X_grid.reshape(-1, 2).astype(X_data.dtype)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GP coverage benchmark and save plots + metrics to JSON."
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Path to .npy dataset (e.g. N17E073.npy)."
    )
    parser.add_argument(
        "--kernel",
        type=str,
        choices=["Attentive", "RBF"],
        default="Attentive",
        help="Kernel type for the GP model."
    )
    parser.add_argument(
        "--variance-ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="List of target variance ratios in (0,1], e.g. --variance-ratios 0.3 0.5 0.7",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["HexCover", "GreedyCover", "GCBCover", "GCBCover-Dist"],
        help="List of coverage methods to benchmark.",
    )
    parser.add_argument(
        "--num-initial",
        type=int,
        default=350,
        help="Number of initial points along the TSP path."
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=5000,
        help="Number of training samples for Dataset."
    )
    parser.add_argument(
        "--num-inducing",
        type=int,
        default=15,
        help="Number of inducing points for initial path."
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        nargs=2,
        default=[100, 100],
        help="Grid resolution for evaluation, as two ints: NX NY (default: 100 100)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./",
        help=(
            "Base directory where plots and JSON results will be saved. "
            "Final output directory will be <output-dir>/<dataset_name>/<kernel>."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_stem = os.path.splitext(os.path.basename(args.dataset))[0]
    output_dir = os.path.join(args.output_dir, dataset_stem, args.kernel)
    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------------------------------
    # Load data and create Dataset
    # -----------------------------------------------------
    print(f"Loading dataset from {args.dataset}...")
    data = np.load(args.dataset)

    dataset = Dataset(
        data=data,
        dtype=np.float32,
        num_train=args.num_train
    )
    del data

    X_train, y_train = dataset.get_train()

    # -----------------------------------------------------
    # Generate initial path X_init
    # -----------------------------------------------------
    print("Generating initial path...")
    X_init = get_inducing_pts(X_train, num_inducing=args.num_inducing)
    X_init, _ = run_tsp(X_init)
    X_init = X_init[0]
    X_init = resample_path(X_init, args.num_initial)
    X_init = X_init.astype(X_train.dtype)
    X_init, y_init = dataset.get_sensor_data(X_init, max_samples=len(X_init))
    print("Init Set Dims:", X_init.shape)

    # -----------------------------------------------------
    # Generate test grid
    # -----------------------------------------------------
    x_dim, y_dim = args.grid_size
    X_grid = get_grid(X_train, x_dim, y_dim)
    X_grid, y_grid = dataset.get_sensor_data(X_grid, max_samples=len(X_grid))
    print("Grid Set Dims:", X_grid.shape)

    extent = [
        float(np.min(X_train[:, 0])),
        float(np.max(X_train[:, 0])),
        float(np.min(X_train[:, 1])),
        float(np.max(X_train[:, 1])),
    ]

    # -----------------------------------------------------
    # Optional: lengthscale map for Attentive kernel
    # -----------------------------------------------------
    lengthscale_fig_path = None
    if args.kernel == "Attentive":
        print("Fitting Attentive kernel to full training set for lengthscale plot...")
        _, _, length_kernel = get_model_params(
            X_train=X_train,
            y_train=y_train,
            kernel=get_kernel("Attentive")(np.linspace(1, 10, 10)),
            optimizer='tf.Nadam',
            learning_rate=1e-2,
            max_steps=1000,
            verbose=True,
        )
        ls_grid = length_kernel.get_lengthscales(X_grid)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

        # Ground truth
        sc1 = axes[0].imshow(
            y_grid.reshape(x_dim, y_dim).T,
            extent=extent,
            origin="lower"
        )
        axes[0].set_title("Ground Truth")
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # Lengthscale predictions
        sc2 = axes[1].imshow(
            ls_grid.reshape(x_dim, y_dim).T,
            extent=extent,
            origin="lower"
        )
        axes[1].set_title("Lengthscale Predictions")
        axes[1].set_aspect("equal")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        fig.colorbar(
            sc2,
            ax=axes,
            orientation="vertical",
            fraction=0.05,
            pad=0.04,
            label="Lengthscale",
        )

        lengthscale_fig_path = os.path.join(
            output_dir,
            "lengthscale_groundtruth.png"
        )
        fig.savefig(lengthscale_fig_path, dpi=300)
        plt.close(fig)
        print(f"Saved lengthscale figure to {lengthscale_fig_path}")

    # -----------------------------------------------------
    # Initial GP model on X_init to get max prior variance
    # -----------------------------------------------------
    print(f"Fitting initial GP model with kernel={args.kernel} on X_init...")

    if args.kernel == "Attentive":
        base_kernel = get_kernel("Attentive")(np.linspace(1, 10, 10))
        _, noise_variance, kernel, init_model = get_model_params(
            X_train=X_init,
            y_train=y_init,
            kernel=base_kernel,
            optimizer='tf.Nadam',
            learning_rate=1e-2,
            max_steps=1000,
            return_model=True,
            verbose=True,
        )
    else:  # RBF
        base_kernel = get_kernel("RBF")()
        _, noise_variance, kernel, init_model = get_model_params(
            X_train=X_init,
            y_train=y_init,
            kernel=base_kernel,
            return_model=True,
            verbose=True,
        )

    prior_mean, prior_var = init_model.predict_f(X_grid)
    max_prior_var = float(prior_var.numpy().max())
    print(f"Max prior variance on grid: {max_prior_var:.4f}")

    # -----------------------------------------------------
    # Benchmark loop
    # -----------------------------------------------------
    results = {
        "dataset": args.dataset,
        "kernel": args.kernel,
        "num_initial": args.num_initial,
        "num_train": args.num_train,
        "num_inducing": args.num_inducing,
        "grid_size": [x_dim, y_dim],
        "variance_ratios": args.variance_ratios,
        "methods": args.methods,
        "lengthscale_figure": lengthscale_fig_path,
        "runs": [],
    }

    for target_var_ratio in args.variance_ratios:
        post_var_threshold = max_prior_var * float(target_var_ratio)
        print(
            f"\n=== Target variance ratio: {target_var_ratio:.2f} "
            f"(post_var_threshold={post_var_threshold:.4f}) ==="
        )

        for method in args.methods:
            print(f"Running method: {method} ...")
            
            kwargs = {}
            if 'Dist' in method:
                kwargs['distance_budget'] = distance - 20

            cmodel = get_method(method.split('-')[0])(
                num_sensing=len(X_train),
                X_objective=X_train,
                kernel=kernel,
                noise_variance=noise_variance,
            )

            s_time = time()
            X_sol, fovs = cmodel.optimize(
                post_var_threshold=post_var_threshold,
                return_fovs=True,
                start_nodes=X_init[None, -1],
                **kwargs
            )
            X_sol = X_sol[0]
            run_time = time() - s_time

            # Evaluate solution
            X_pred, y_pred = dataset.get_sensor_data(
                X_sol,
                max_samples=len(X_sol)
            )

            _, _, _, model_sol = get_model_params(
                X_train=np.vstack([X_init, X_pred]),
                y_train=np.vstack([y_init, y_pred]),
                kernel=kernel,
                noise_variance=noise_variance,
                max_steps=0,
                return_model=True,
                verbose=False,
                force_gp=True
            )

            mean, var = model_sol.predict_f(X_grid)
            mean_np = mean.numpy()
            var_np = var.numpy()

            distance = float(get_distance(X_sol))
            mse_e = float(get_mse(mean_np, y_grid))
            smse_e = float(get_smse(mean_np, y_grid, var_np))
            max_post_var = float(model_sol.predict_f(X_train)[1].numpy().max())

            # -------------------------------------------------
            # Plotting
            # -------------------------------------------------
            fig, axes = plt.subplots(
                1, 3,
                figsize=(13, 4.5),
                constrained_layout=True
            )

            # GP mean predictions
            sc1 = axes[0].imshow(
                mean_np.reshape(x_dim, y_dim).T,
                extent=extent,
                origin="lower"
            )
            axes[0].set_title("Solution GP Predictions")
            axes[0].set_aspect("equal")
            axes[0].set_xlabel("X")
            axes[0].set_ylabel("Y")

            # GP variance + path
            sc2 = axes[1].imshow(
                var_np.reshape(x_dim, y_dim).T,
                extent=extent,
                origin="lower"
            )
            axes[1].scatter(X_sol[:, 0], X_sol[:, 1], c="r", s=25)
            axes[1].plot(X_init[:, 0], X_init[:, 1], c="tab:orange")
            axes[1].plot(X_sol[:, 0], X_sol[:, 1], c="r")
            axes[1].set_title("Solution GP Variance")
            axes[1].set_aspect("equal")
            axes[1].set_xlabel("X")
            axes[1].set_ylabel("Y")

            fig.colorbar(
                sc2,
                ax=axes[1],
                orientation="vertical",
                fraction=0.05,
                pad=0.04,
                label="Variance",
            )

            # FoVs + path
            axes[2].scatter(X_sol[:, 0], X_sol[:, 1], c="r", s=5)
            for fov in fovs:
                patch = MplPolygon(
                    list(fov.exterior.coords),
                    closed=True,
                    facecolor="k",
                    edgecolor="k",
                    alpha=0.3,
                )
                axes[2].add_patch(patch)
            axes[2].set_title("Solution FoVs")
            axes[2].set_aspect("equal")
            axes[2].set_xlabel("X")
            axes[2].set_ylabel("Y")
            axes[2].set_xlim(axes[1].get_xlim())
            axes[2].set_ylim(axes[1].get_ylim())

            ratio_str = str(target_var_ratio).replace(".", "p")
            fig_filename = os.path.join(
                output_dir,
                f"{method}_ratio{ratio_str}.png"
            )
            fig.savefig(fig_filename, dpi=300)
            plt.close(fig)

            print(
                f"Saved figure for {method}, ratio={target_var_ratio:.2f} "
                f"to {fig_filename}"
            )

            # -------------------------------------------------
            # Store structured results (from suptitle + extras)
            # -------------------------------------------------
            run_result = {
                "method": method,
                "variance_ratio": float(target_var_ratio),
                "num_placements": int(len(fovs)),
                "max_prior_var": max_prior_var,
                "target_post_var_threshold": post_var_threshold,
                "max_posterior_var": max_post_var,
                "mse": mse_e,
                "smse": smse_e,
                "runtime_sec": run_time,
                "distance_m": distance,
                "figure": fig_filename,
                "Budget": kwargs.get('distance_budget')
            }
            results["runs"].append(run_result)

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    json_path = os.path.join(output_dir, "results.json")
    # Ensure everything is JSON-serializable
    def _convert(o):
        if isinstance(o, (np.generic,)):
            return o.item()
        return o

    results_serializable = json.loads(json.dumps(results, default=_convert))

    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nBenchmark complete. Results written to {json_path}")


if __name__ == "__main__":
    main()
