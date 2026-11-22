"""Radial mechanism tests (Section 3.3) and initial-size comparison (Appendix C.4)."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from numerics import radial
from numerics.parameters import FullModelParameters


TIMES = (0.0, 2.5, 5.0)
ACTIVITY = (0.0, 0.1, 0.25, 0.5, 1.0)
ACTIVATION = (0.0, 0.25, 0.5, 0.75, 1.0)
RADII = (1.5, 2.0, 3.0, 4.0, 5.0)
SETTINGS = radial.NumericalParameters(final_time=5.0, max_dt=0.0025)
BASE = replace(
    FullModelParameters.calibrated(640), xi1=0.0757, eta2=1.2, f3_half=0.45,
    sigma_max=1.15, sigma_hill_power=1,
)


@dataclass(frozen=True)
class Case:
    name: str
    radius: float = 5.0
    activity: float = 1.0
    activation: float = 1.0
    reaction: bool = True
    taxis: bool = True


def case_name(prefix, value):
    default = BASE.initial_radius if prefix == "radius" else 1.0
    return "reference" if value == default else prefix + "_" + str(value).replace(".", "p")


def cases():
    return [
        Case("reference"), Case("reaction_off", reaction=False),
        Case("diffusion_only", reaction=False, taxis=False),
        *(Case(case_name("theta", x), activity=x) for x in ACTIVITY[:-1]),
        *(Case(case_name("activation", x), activation=x) for x in ACTIVATION[:-1]),
        *(Case(case_name("radius", x), radius=x) for x in RADII[:-1]),
    ]


def solve_case(case):
    params = replace(
        BASE, initial_radius=case.radius, hbar_max=BASE.hbar_max * case.activity,
        k3=BASE.k3 * case.activation,
    )
    # Reaction ablation applies only to the density step. Oxygen consumption is retained.
    reaction_params = params if case.reaction else replace(params, phi=0.0, hbar_max=0.0, kappa=0.0)
    sigma = params.sigma_max if case.taxis else 0.0
    grid = radial.make_grid(params.radius, params.n_cells)
    env = radial.EnvironmentSolver(grid, params)
    density = radial.initial_density(grid, params)
    # The plotted theta=0.1 run starts on the invariant zero-EGF branch (Appendix C.2).
    seed = 0.0 if case.activity == 0.1 else params.egf_initial
    oxygen, egf, _ = env.solve(
        density, np.full_like(density, params.oxygen_boundary), np.full_like(density, seed), SETTINGS
    )
    fields = {key: [] for key in ("density", "oxygen", "egf")}
    time = 0.0
    for target in TIMES:
        while target - time > 1e-12:
            velocity = radial.face_velocity(oxygen, egf, sigma, grid, params)
            dt = min(target - time, radial.stable_timestep(
                velocity, grid, params, SETTINGS, params.cell_boundary_condition
            ))
            density = radial.advance_density(
                density, oxygen, egf, sigma, dt, grid, reaction_params, params.cell_boundary_condition
            )[0]
            oxygen, egf, _ = env.solve(density, oxygen, egf, SETTINGS)
            time += dt
        for key, array in zip(fields, (density, oxygen, egf)):
            fields[key].append(array.copy())
    print(f"  Finished reduced case: {case.name}", flush=True)
    return {case.name + "_" + key: np.stack(values) for key, values in fields.items()}


def compute(workers):
    runs = cases()
    data = {"time": np.asarray(TIMES), "radius": radial.make_grid(BASE.radius, BASE.n_cells).centers}
    if workers == 1:
        for result in map(solve_case, runs):
            data.update(result)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(runs))) as pool:
            for result in pool.map(solve_case, runs):
                data.update(result)
    return data
