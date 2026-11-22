"""Compute the five diagonal relaxation cases and their reduced reference (Section 3.2)."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from numerics.parameters import FullModelParameters, MAIN_LAMBDA_VALUES, PHENOTYPE_GRID_VALUES
from numerics.radial import simulate_case
from .full_model import FullModelNumerics, PhenotypeGridSpec, simulate_full_model


TIMES = (1, 5, 10, 20, 30)
FIELDS = ("density", "oxygen", "egf")
SETTINGS = FullModelNumerics(
    final_time=30.0,
    snapshot_interval=1.0,
    diagnostic_interval=0.5,
    max_dt=0.02,
    phenotype_grid_spec=PhenotypeGridSpec(**PHENOTYPE_GRID_VALUES),
)


def at_figure_times(snapshots):
    indices = []
    for target in TIMES:
        matches = np.flatnonzero(np.isclose(snapshots["time"], target, rtol=0, atol=2e-10))
        if len(matches) != 1:
            raise ValueError(f"Expected one snapshot at t={target}.")
        indices.append(matches[0])
    return {field: snapshots[field][indices] for field in FIELDS}


def solve_full(job):
    relaxation, initial = job
    print(f"  Full model: lambda = {relaxation:g}", flush=True)
    settings = replace(SETTINGS, tau_environment=relaxation, epsilon_phenotype=relaxation)
    result = simulate_full_model(
        FullModelParameters.calibrated(), settings,
        initial_density_profile=initial[0], initial_reduced_fields=initial[1:],
    )
    print(f"  Finished lambda = {relaxation:g} ({result.run_stats['step_count']} steps)", flush=True)
    return at_figure_times(result.snapshots)


def compute(workers):
    params = FullModelParameters.calibrated()
    reference = simulate_case(
        params.sigma_max, params, SETTINGS.reduced_numerics(), params.cell_boundary_condition
    )
    snapshots = reference["snapshots"]
    initial = tuple(snapshots[field][0] for field in FIELDS)
    jobs = [(value, initial) for value in MAIN_LAMBDA_VALUES]
    if workers == 1:
        cases = list(map(solve_full, jobs))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            cases = list(pool.map(solve_full, jobs))
    data = {
        "time": np.asarray(TIMES, dtype=float), "radius": reference["grid"].centers,
        "lambda_values": np.asarray(MAIN_LAMBDA_VALUES),
    }
    for field, reduced in at_figure_times(snapshots).items():
        data["reduced_" + field] = reduced
        data["full_" + field] = np.stack([case[field] for case in cases])
    return data
