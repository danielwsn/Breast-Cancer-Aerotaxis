"""Recompute, verify and plot the radial simulations reported in Figures 2--5 and C.1."""

import argparse
import os
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from full_reduced import experiment as comparison, plot as comparison_plot
from reduced_controls import experiment as controls, plot as controls_plot


ROOT = Path(__file__).resolve().parent
PARTS = {
    "full-reduced": ("figure_2", comparison.compute, comparison_plot.plot),
    "reduced-controls": ("reduced_controls", controls.compute, controls_plot.plot),
}


def verify(data, reference):
    if set(data) != set(reference.files):
        raise ValueError("Computed and reference archives have different field names.")
    largest = 0.0
    for key, actual in data.items():
        expected = reference[key]
        if actual.shape != expected.shape or not np.all(np.isfinite(actual)):
            raise ValueError(f"Invalid shape or non-finite values in {key}.")
        if key in ("time", "radius", "lambda_values"):
            np.testing.assert_array_equal(actual, expected, err_msg=key)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9, err_msg=key)
            largest = max(largest, float(np.max(np.abs(actual - expected))))
    return largest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=("all", *PARTS), default="all", help="Experiment to run.")
    parser.add_argument("--workers", type=int, default=min(5, os.cpu_count() or 1),
                        help="Maximum number of simultaneous independent cases (default: up to 5).")
    parser.add_argument("--reference-only", action="store_true",
                        help="Plot the supplied reference profiles without running the solvers.")
    parser.add_argument("--output", type=Path, help="Output directory (default: outputs or reference_preview).")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be a positive integer")
    output = args.output or ROOT / ("reference_preview" if args.reference_only else "outputs")
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    parts = PARTS if args.part == "all" else (args.part,)
    for part in parts:
        start = perf_counter()
        stem, compute, plot = PARTS[part]
        with np.load(ROOT / "reference" / (stem + ".npz"), allow_pickle=False) as reference:
            if args.reference_only:
                print(f"Rendering reference profiles: {part}", flush=True)
                data = {key: reference[key] for key in reference.files}
            else:
                print(f"Computing {part} (up to {args.workers} workers)", flush=True)
                data = compute(args.workers)
                error = verify(data, reference)
                profiles = output / "profiles"
                profiles.mkdir(exist_ok=True)
                np.savez_compressed(profiles / (stem + ".npz"), **data)
                print(f"All profiles verified. Maximum absolute field difference: {error:.3e}", flush=True)
        for name, figure in plot(data).items():
            for ext in ("pdf", "png"):
                figure.savefig(figures / f"{name}.{ext}", dpi=250, facecolor="white")
            plt.close(figure)
        print(f"Finished {part} in {(perf_counter() - start) / 60:.1f} min.", flush=True)
    print(f"Figures written to {figures.resolve()}", flush=True)


if __name__ == "__main__":
    main()
