#!/usr/bin/env python3
r"""Asymptotic-preserving radial solver for the phenotype-structured model.

The full density is ``rho[x, y, z, r]`` and is a density with respect to
phenotype volume ``dx dy dz``.  The solver implements the two-parameter model

.. math::

   \partial_t\rho = -\varepsilon_p^{-1}\nabla_p\cdot(U\rho)
       + \alpha_1\Delta_r\rho
       - r^{-1}\partial_r(r\sigma(z)\rho u_r)
       + (h_1(x)+h_2(x,u)-\kappa N)\rho,

with ``tau * u_t`` and ``tau * v_t`` in the environmental equations.  The
environmental update is backward Euler and remains well-defined at ``tau=0``.
Phenotype relaxation is advanced by a positive conservative characteristic
remap, so its stability does not require ``dt = O(epsilon_p)``.

"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from numerics.parameters import FullModelParameters
from numerics.radial import (
    EnvironmentSolver as ReducedEnvironmentSolver,
    ModelParameters,
    NumericalParameters,
    RadialGrid,
    aerotaxis_sensitivity,
    damped_newton,
    hill,
    hill_derivative,
    initial_density,
    make_grid,
    make_record_times,
    negative_laplacian_coefficients,
    outer_face_value,
    profile_diagnostics,
    radial_mass,
    selected_phenotype,
    solve_tridiagonal,
    tridiagonal_matvec,
)


@dataclass(frozen=True)
class PhenotypeGridSpec:
    """Uniform phenotype box and the Gaussian width scales.

    The initial covariance is
    ``variance_parameter * diag(initial_std_scales**2)``.  Thus the standard
    deviation is proportional to ``sqrt(variance_parameter)``.
    """

    nx: int = 12  # Number of finite-volume cells in phenotype x.
    ny: int = 12  # Number of finite-volume cells in phenotype y.
    nz: int = 24  # Number of finite-volume cells in phenotype z.
    x_min: float = 0.0  # Lower bound of the x phenotype box.
    x_max: float = 5.0  # Upper bound of the x phenotype box.
    y_min: float = 0.0  # Lower bound of the y phenotype box.
    y_max: float = 5.0  # Upper bound of the y phenotype box.
    z_min: float = 0.0  # Lower bound of the z phenotype box.
    z_max: float = 30.0  # Upper bound of the z phenotype box.
    # If the Gaussian variance parameter is lambda, the initial standard
    # deviations are sqrt(lambda) times these three direction-dependent scales.
    initial_std_scales: tuple[float, float, float] = (1.0, 1.0, 5.0)

    def __post_init__(self) -> None:
        if min(self.nx, self.ny, self.nz) < 2:
            raise ValueError("Every phenotype direction needs at least two cells")
        if not (self.x_min < self.x_max and self.y_min < self.y_max):
            raise ValueError("Invalid x/y phenotype bounds")
        if not self.z_min < self.z_max:
            raise ValueError("Invalid z phenotype bounds")
        if min(self.initial_std_scales) <= 0.0:
            raise ValueError("Gaussian width scales must be positive")


@dataclass(frozen=True)
class PhenotypeGrid:
    spec: PhenotypeGridSpec
    x_faces: np.ndarray
    y_faces: np.ndarray
    z_faces: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    dx: float
    dy: float
    dz: float
    cell_volume: float

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.spec.nx, self.spec.ny, self.spec.nz)


def make_phenotype_grid(
    spec: PhenotypeGridSpec | None = None,
) -> PhenotypeGrid:
    """Construct a uniform cell-centered phenotype grid."""
    if spec is None:
        spec = PhenotypeGridSpec()

    def axis(lo: float, hi: float, count: int) -> tuple[np.ndarray, np.ndarray, float]:
        faces = np.linspace(lo, hi, count + 1, dtype=float)
        centers = 0.5 * (faces[:-1] + faces[1:])
        return faces, centers, float((hi - lo) / count)

    x_faces, x, dx = axis(spec.x_min, spec.x_max, spec.nx)
    y_faces, y, dy = axis(spec.y_min, spec.y_max, spec.ny)
    z_faces, z, dz = axis(spec.z_min, spec.z_max, spec.nz)
    return PhenotypeGrid(
        spec=spec,
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        x=x,
        y=y,
        z=z,
        dx=dx,
        dy=dy,
        dz=dz,
        cell_volume=dx * dy * dz,
    )


@dataclass(frozen=True)
class FullModelNumerics:
    """Time integration, nonlinear-solve and output controls."""

    # tau in front of u_t and v_t; tau=0 selects the elliptic limit.
    tau_environment: float = 0.1
    # epsilon_p in -(1/epsilon_p) div_p(U rho); zero is a discrete delta limit.
    epsilon_phenotype: float = 0.1
    final_time: float = 0.1  # End time T.
    snapshot_interval: float = 0.05  # Saved-profile interval.
    diagnostic_interval: float = 0.01  # Morphology/statistics interval.
    max_dt: float = 2.0e-3  # Upper time-step bound before the radial CFL limit.
    cfl_safety: float = 0.72  # Safety factor for explicit radial transport.
    environment_tolerance: float = 2.0e-10  # Relative Newton tolerance.
    environment_max_iterations: int = 30  # Newton iteration cap.
    phenotype_grid_spec: PhenotypeGridSpec = field(
        default_factory=PhenotypeGridSpec
    )
    # Variance parameter of the initial phenotype Gaussian.  ``None`` means
    # use epsilon_phenotype; the reported diagonal scan therefore uses lambda.
    initial_gaussian_variance: float | None = None
    well_prepared_full_iterations: int = 0  # Optional full-moment IC refinements.
    density_floor_for_moments: float = 1.0e-14  # Safe division threshold.
    store_full_rho: bool = False  # False saves moments instead of huge rho arrays.
    phenotype_strang_splitting: bool = True  # Half/full/half phenotype updates.
    boundary_exclusion: float = 0.25  # Excluded strip in profile diagnostics.
    slope_tolerance: float = 1.0e-4  # Zero-slope tolerance for extrema.
    amplitude_tolerance: float = 2.0e-4  # Minimum trough/peak prominence.

    def __post_init__(self) -> None:
        if self.tau_environment < 0.0 or self.epsilon_phenotype < 0.0:
            raise ValueError("tau_environment and epsilon_phenotype must be >= 0")
        if self.final_time < 0.0 or self.max_dt <= 0.0:
            raise ValueError("Invalid time interval or max_dt")
        if self.snapshot_interval <= 0.0 or self.diagnostic_interval <= 0.0:
            raise ValueError("Record intervals must be positive")
        if self.initial_gaussian_variance is not None:
            if self.initial_gaussian_variance < 0.0:
                raise ValueError("initial_gaussian_variance must be nonnegative")

    def reduced_numerics(self) -> NumericalParameters:
        """Subset needed by the reduced elliptic initializer/diagnostics."""
        return NumericalParameters(
            final_time=self.final_time,
            snapshot_interval=self.snapshot_interval,
            diagnostic_interval=self.diagnostic_interval,
            max_dt=self.max_dt,
            cfl_safety=self.cfl_safety,
            elliptic_tolerance=self.environment_tolerance,
            elliptic_max_iterations=self.environment_max_iterations,
            boundary_exclusion=self.boundary_exclusion,
            slope_tolerance=self.slope_tolerance,
            amplitude_tolerance=self.amplitude_tolerance,
        )


@dataclass
class FullState:
    """Full phenotype-spatial state; rho shape is (nx, ny, nz, nr)."""

    rho: np.ndarray
    oxygen: np.ndarray
    egf: np.ndarray
    time: float = 0.0

    def copy(self) -> "FullState":
        return FullState(
            self.rho.copy(), self.oxygen.copy(), self.egf.copy(), self.time
        )


@dataclass(frozen=True)
class PhenotypeMoments:
    density: np.ndarray
    mean_x: np.ndarray
    mean_y: np.ndarray
    mean_z: np.ndarray
    var_x: np.ndarray
    var_y: np.ndarray
    var_z: np.ndarray
    h2_factor_mass: np.ndarray
    oxygen_consumption: np.ndarray
    egf_production: np.ndarray
    sensitivity_mass: np.ndarray
    h1_mass: np.ndarray


@dataclass(frozen=True)
class FullSimulationResult:
    radial_grid: RadialGrid
    phenotype_grid: PhenotypeGrid
    snapshots: dict[str, np.ndarray]
    diagnostics: list[dict[str, float | bool]]
    final_state: FullState
    run_stats: dict[str, float | int | str]


def _discrete_gaussian_weights(
    centers: np.ndarray,
    center: np.ndarray,
    standard_deviation: float,
) -> np.ndarray:
    """Return positive columns normalized to one on the discrete grid."""
    nr = center.size
    if standard_deviation <= 1.0e-14:
        # A two-cell cloud-in-cell delta is the zero-variance endpoint.
        q = (center - centers[0]) / (centers[1] - centers[0])
        q = np.clip(q, 0.0, centers.size - 1.0)
        lower = np.floor(q).astype(int)
        upper = np.minimum(lower + 1, centers.size - 1)
        fraction = q - lower
        weights = np.zeros((centers.size, nr), dtype=float)
        columns = np.arange(nr)
        np.add.at(weights, (lower, columns), 1.0 - fraction)
        np.add.at(weights, (upper, columns), fraction)
        return weights

    log_weight = -0.5 * (
        (centers[:, None] - center[None, :]) / standard_deviation
    ) ** 2
    log_weight -= np.max(log_weight, axis=0, keepdims=True)
    # Clipping prevents an exact zero without introducing a noticeable floor.
    weights = np.exp(np.maximum(log_weight, -745.0))
    weights /= np.sum(weights, axis=0, keepdims=True)
    return weights


def gaussian_phenotype_density(
    density: np.ndarray,
    oxygen: np.ndarray,
    egf: np.ndarray,
    phenotype_grid: PhenotypeGrid,
    params: FullModelParameters,
    variance_parameter: float,
    std_scales: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Deposit ``density`` as normalized Gaussian phenotype distributions.

    The discrete integral over phenotype is exactly ``density`` to rounding.
    ``variance_parameter=0`` uses a conservative cloud-in-cell delta.
    """
    if variance_parameter < 0.0:
        raise ValueError("variance_parameter must be nonnegative")
    density = np.asarray(density, dtype=float)
    oxygen = np.asarray(oxygen, dtype=float)
    egf = np.asarray(egf, dtype=float)
    if not (density.shape == oxygen.shape == egf.shape):
        raise ValueError("density, oxygen and egf must have the same radial shape")
    x_star, y_star, z_star = selected_phenotype(oxygen, egf, params)
    scales = std_scales or phenotype_grid.spec.initial_std_scales
    root_variance = math.sqrt(variance_parameter)
    wx = _discrete_gaussian_weights(
        phenotype_grid.x, x_star, root_variance * scales[0]
    )
    wy = _discrete_gaussian_weights(
        phenotype_grid.y, y_star, root_variance * scales[1]
    )
    wz = _discrete_gaussian_weights(
        phenotype_grid.z, z_star, root_variance * scales[2]
    )
    rho = np.einsum("ir,jr,kr,r->ijkr", wx, wy, wz, density, optimize=True)
    rho /= phenotype_grid.cell_volume
    return rho


def phenotype_moments(
    rho: np.ndarray,
    phenotype_grid: PhenotypeGrid,
    params: FullModelParameters,
    oxygen: np.ndarray | None = None,
    density_floor: float = 1.0e-14,
) -> PhenotypeMoments:
    """Compute marginal density, sources and first two phenotype moments."""
    expected = phenotype_grid.shape
    if rho.ndim != 4 or rho.shape[:3] != expected:
        raise ValueError(
            f"rho must have shape {expected}+(nr,), received {rho.shape}"
        )
    weight = phenotype_grid.cell_volume
    density = weight * np.sum(rho, axis=(0, 1, 2))
    safe = np.where(density > density_floor, density, 1.0)

    def first(axis_values: np.ndarray, axis: int) -> np.ndarray:
        shape = [1, 1, 1, 1]
        shape[axis] = axis_values.size
        return weight * np.sum(
            rho * axis_values.reshape(shape), axis=(0, 1, 2)
        )

    mx = first(phenotype_grid.x, 0)
    my = first(phenotype_grid.y, 1)
    mz = first(phenotype_grid.z, 2)
    mx2 = first(phenotype_grid.x**2, 0)
    my2 = first(phenotype_grid.y**2, 1)
    mz2 = first(phenotype_grid.z**2, 2)
    mean_x = np.where(density > density_floor, mx / safe, 0.0)
    mean_y = np.where(density > density_floor, my / safe, 0.0)
    mean_z = np.where(density > density_floor, mz / safe, 0.0)
    var_x = np.where(
        density > density_floor, np.maximum(mx2 / safe - mean_x**2, 0.0), 0.0
    )
    var_y = np.where(
        density > density_floor, np.maximum(my2 / safe - mean_y**2, 0.0), 0.0
    )
    var_z = np.where(
        density > density_floor, np.maximum(mz2 / safe - mean_z**2, 0.0), 0.0
    )

    k_star = params.k1 / params.k2
    h2_factor = np.maximum(1.0 - (phenotype_grid.x / k_star) ** 2, 0.0)
    h1_values = params.phi * (
        1.0 - (1.0 - phenotype_grid.x / k_star) ** 2
    )
    f3_values = hill(
        phenotype_grid.y,
        params.f3_max,
        params.f3_half,
        params.signalling_hill_power,
    )
    sigma_values = aerotaxis_sensitivity(
        phenotype_grid.z, params.sigma_max, params
    )
    h2_factor_mass = weight * np.sum(
        rho * h2_factor[:, None, None, None], axis=(0, 1, 2)
    )
    egf_production = weight * np.sum(
        rho * f3_values[None, :, None, None], axis=(0, 1, 2)
    )
    sensitivity_mass = weight * np.sum(
        rho * sigma_values[None, None, :, None], axis=(0, 1, 2)
    )
    h1_mass = weight * np.sum(
        rho * h1_values[:, None, None, None], axis=(0, 1, 2)
    )
    if oxygen is None:
        oxygen_consumption = np.zeros_like(density)
    else:
        hbar = hill(
            np.asarray(oxygen),
            params.hbar_max,
            params.hbar_half,
            params.hbar_hill_power,
        )
        oxygen_consumption = hbar * h2_factor_mass
    return PhenotypeMoments(
        density=density,
        mean_x=mean_x,
        mean_y=mean_y,
        mean_z=mean_z,
        var_x=var_x,
        var_y=var_y,
        var_z=var_z,
        h2_factor_mass=h2_factor_mass,
        oxygen_consumption=oxygen_consumption,
        egf_production=egf_production,
        sensitivity_mass=sensitivity_mass,
        h1_mass=h1_mass,
    )


class ImplicitEnvironmentSolver:
    """Unified backward-Euler environmental solver, including ``tau=0``."""

    def __init__(
        self,
        radial_grid: RadialGrid,
        phenotype_grid: PhenotypeGrid,
        params: FullModelParameters,
    ):
        self.radial_grid = radial_grid
        self.phenotype_grid = phenotype_grid
        self.params = params
        (
            self.u_lower,
            self.u_diffusion_diagonal,
            self.u_upper,
            self.u_boundary_rhs,
        ) = negative_laplacian_coefficients(
            radial_grid,
            params.alpha2,
            params.oxygen_boundary,
            params.oxygen_boundary_type,
            params.oxygen_robin_transfer,
        )
        (
            self.v_lower,
            self.v_diffusion_diagonal,
            self.v_upper,
            self.v_boundary_rhs,
        ) = negative_laplacian_coefficients(
            radial_grid,
            params.alpha3,
            params.egf_boundary,
            params.egf_boundary_type,
            params.egf_robin_transfer,
        )

    def step(
        self,
        rho: np.ndarray,
        oxygen_old: np.ndarray,
        egf_old: np.ndarray,
        dt: float,
        tau_environment: float,
        numerical: FullModelNumerics,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Advance the environment with frozen ``rho``.

        Setting ``tau_environment=0`` gives the full elliptic solve without a
        separate code path.  Uptake/production use phenotype integrals rather
        than equilibrium substitutions.
        """
        if dt <= 0.0:
            raise ValueError("The environment step requires dt > 0")
        if tau_environment < 0.0:
            raise ValueError("tau_environment must be nonnegative")
        params = self.params
        moments = phenotype_moments(
            rho,
            self.phenotype_grid,
            params,
            density_floor=numerical.density_floor_for_moments,
        )
        tau_over_dt = tau_environment / dt
        u_base_diagonal = (
            self.u_diffusion_diagonal + params.xi1 + tau_over_dt
        )
        u_rhs = (
            self.u_boundary_rhs
            + params.oxygen_source
            + tau_over_dt * oxygen_old
        )

        def oxygen_residual_and_diagonal(oxygen: np.ndarray):
            hbar = hill(
                oxygen,
                params.hbar_max,
                params.hbar_half,
                params.hbar_hill_power,
            )
            hbar_prime = hill_derivative(
                oxygen,
                params.hbar_max,
                params.hbar_half,
                params.hbar_hill_power,
            )
            residual = (
                tridiagonal_matvec(
                    self.u_lower,
                    u_base_diagonal,
                    self.u_upper,
                    oxygen,
                )
                + params.eta1 * moments.h2_factor_mass * hbar
                - u_rhs
            )
            jacobian_diagonal = (
                u_base_diagonal
                + params.eta1 * moments.h2_factor_mass * hbar_prime
            )
            return residual, jacobian_diagonal

        oxygen, u_iterations, u_residual = damped_newton(
            oxygen_old,
            oxygen_residual_and_diagonal,
            self.u_lower,
            self.u_upper,
            numerical.environment_tolerance,
            numerical.environment_max_iterations,
            "full oxygen",
        )

        v_diagonal = (
            self.v_diffusion_diagonal
            + params.d1
            + params.eta3 * params.k3 * moments.density
            + tau_over_dt
        )
        v_rhs = (
            self.v_boundary_rhs
            + params.eta2 * moments.egf_production
            + tau_over_dt * egf_old
        )
        egf = solve_tridiagonal(
            self.v_lower, v_diagonal, self.v_upper, v_rhs
        )
        if float(np.min(egf)) < -1.0e-11:
            raise RuntimeError(
                f"Implicit EGF update lost positivity: min={np.min(egf):.3e}"
            )
        egf = np.maximum(egf, 0.0)
        v_residual_vector = (
            tridiagonal_matvec(
                self.v_lower, v_diagonal, self.v_upper, egf
            )
            - v_rhs
        )
        v_scale = 1.0 + float(np.max(np.abs(egf)))
        v_residual = float(np.max(np.abs(v_residual_vector))) / v_scale
        return oxygen, egf, {
            "tau_over_dt": float(tau_over_dt),
            "oxygen_newton_iterations": float(u_iterations),
            "oxygen_residual": float(u_residual),
            "egf_residual": float(v_residual),
            "minimum_oxygen": float(np.min(oxygen)),
            "minimum_egf": float(np.min(egf)),
        }


def _conservative_remap_axis(
    rho: np.ndarray,
    mapped_positions: np.ndarray,
    axis: int,
    centers: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Cloud-in-cell deposition of characteristic endpoints along one axis.

    ``mapped_positions`` may have singleton dimensions and is broadcast to the
    shape of ``rho``.  Mass mapped outside the phenotype *faces* is discarded
    and returned; endpoints between a boundary face and its first/last center
    are deposited completely into that boundary cell.
    """
    source = np.moveaxis(rho, axis, 0)
    mapped = np.moveaxis(np.broadcast_to(mapped_positions, rho.shape), axis, 0)
    source_flat = source.reshape(source.shape[0], -1)
    mapped_flat = mapped.reshape(mapped.shape[0], -1)
    output_flat = np.zeros_like(source_flat)
    columns = np.arange(source_flat.shape[1])
    spacing = centers[1] - centers[0]
    lost = 0.0
    for source_index in range(source_flat.shape[0]):
        position = mapped_flat[source_index]
        values = source_flat[source_index]
        valid = (position >= faces[0] - 1.0e-13) & (
            position <= faces[-1] + 1.0e-13
        )
        if not np.all(valid):
            lost += float(np.sum(values[~valid]))
        if not np.any(valid):
            continue
        q = (position[valid] - centers[0]) / spacing
        q = np.clip(q, 0.0, centers.size - 1.0)
        lower = np.floor(q).astype(np.intp)
        upper = np.minimum(lower + 1, centers.size - 1)
        fraction = q - lower
        valid_columns = columns[valid]
        valid_values = values[valid]
        np.add.at(
            output_flat,
            (lower, valid_columns),
            (1.0 - fraction) * valid_values,
        )
        np.add.at(
            output_flat,
            (upper, valid_columns),
            fraction * valid_values,
        )
    output = output_flat.reshape(source.shape)
    return np.moveaxis(output, 0, axis), lost


def advance_phenotype_characteristics(
    rho: np.ndarray,
    oxygen: np.ndarray,
    egf: np.ndarray,
    dt: float,
    epsilon_phenotype: float,
    phenotype_grid: PhenotypeGrid,
    params: FullModelParameters,
) -> tuple[np.ndarray, dict[str, float]]:
    """Advance fast intracellular drift by a conservative characteristic map.

    A symmetric x/z--y--x/z characteristic splitting is used.  Each scalar
    relaxation is integrated exactly with frozen environmental fields.  For
    ``epsilon_phenotype=0`` the exponential factors are set directly to zero,
    giving the discrete instantaneous-relaxation limit.
    """
    if dt < 0.0 or epsilon_phenotype < 0.0:
        raise ValueError("dt and epsilon_phenotype must be nonnegative")
    if dt == 0.0:
        return rho.copy(), {
            "phenotype_mass_relative_error": 0.0,
            "phenotype_boundary_loss_fraction": 0.0,
        }
    pg = phenotype_grid
    nr = oxygen.size
    if rho.shape != pg.shape + (nr,):
        raise ValueError("rho/phenotype/radial shapes are inconsistent")
    f1_value = hill(
        oxygen,
        params.f1_max,
        params.f1_half,
        params.signalling_hill_power,
    )
    rate_x = params.k2 + f1_value
    x_equilibrium = params.k1 / rate_x
    z_equilibrium = params.k3 * np.maximum(egf, 0.0) / params.d2

    def decay(rate: np.ndarray | float, duration: float) -> np.ndarray:
        if epsilon_phenotype == 0.0:
            return np.zeros_like(np.asarray(rate, dtype=float))
        exponent = -np.asarray(rate, dtype=float) * duration / epsilon_phenotype
        return np.exp(np.maximum(exponent, -745.0))

    total_before = float(np.sum(rho))
    lost_total = 0.0
    updated = rho
    half = 0.5 * dt
    for _ in range(2):
        x_decay = decay(rate_x, half)
        mapped_x = x_equilibrium[None, :] + (
            pg.x[:, None] - x_equilibrium[None, :]
        ) * x_decay[None, :]
        updated, lost = _conservative_remap_axis(
            updated,
            mapped_x[:, None, None, :],
            axis=0,
            centers=pg.x,
            faces=pg.x_faces,
        )
        lost_total += lost

        z_decay = decay(params.d2, half)
        mapped_z = z_equilibrium[None, :] + (
            pg.z[:, None] - z_equilibrium[None, :]
        ) * z_decay
        updated, lost = _conservative_remap_axis(
            updated,
            mapped_z[None, None, :, :],
            axis=2,
            centers=pg.z,
            faces=pg.z_faces,
        )
        lost_total += lost

        if _ == 0:
            # Midpoint x,z bins provide a second-order characteristic
            # approximation for the nonlinear y equation.
            f2_z = hill(
                pg.z,
                params.f2_max,
                params.f2_half,
                params.signalling_hill_power,
            )
            y_rate = (
                rate_x[None, :] + f2_z[:, None]
            )  # (nz,nr)
            y_equilibrium = (
                pg.x[:, None, None]
                * f2_z[None, :, None]
                / y_rate[None, :, :]
            )  # (nx,nz,nr)
            y_decay = decay(y_rate, dt)
            mapped_y = y_equilibrium[:, None, :, :] + (
                pg.y[None, :, None, None] - y_equilibrium[:, None, :, :]
            ) * y_decay[None, None, :, :]
            updated, lost = _conservative_remap_axis(
                updated,
                mapped_y,
                axis=1,
                centers=pg.y,
                faces=pg.y_faces,
            )
            lost_total += lost

    minimum = float(np.min(updated))
    if minimum < -1.0e-13:
        raise RuntimeError(
            f"Characteristic remap lost positivity: min rho={minimum:.3e}"
        )
    updated = np.maximum(updated, 0.0)
    total_after = float(np.sum(updated))
    relative_error = abs(total_after + lost_total - total_before) / (
        1.0 + abs(total_before)
    )
    return updated, {
        "phenotype_mass_relative_error": float(relative_error),
        "phenotype_boundary_loss_fraction": float(
            lost_total / max(total_before, 1.0e-300)
        ),
    }


def radial_face_velocity(
    oxygen: np.ndarray,
    phenotype_grid: PhenotypeGrid,
    radial_grid: RadialGrid,
    params: FullModelParameters,
) -> np.ndarray:
    """Return taxis velocities with shape ``(nz, nr+1)``."""
    nr = radial_grid.n_cells
    gradient = np.zeros(nr + 1, dtype=float)
    gradient[1:nr] = (oxygen[1:] - oxygen[:-1]) / radial_grid.dr
    outer_oxygen = outer_face_value(
        oxygen[-1],
        params.alpha2,
        radial_grid,
        params.oxygen_boundary,
        params.oxygen_boundary_type,
        params.oxygen_robin_transfer,
    )
    gradient[-1] = (outer_oxygen - oxygen[-1]) / (0.5 * radial_grid.dr)
    sigma = aerotaxis_sensitivity(
        phenotype_grid.z, params.sigma_max, params
    )
    velocity = sigma[:, None] * gradient[None, :]
    velocity[:, 0] = 0.0
    return velocity


def stable_radial_timestep(
    oxygen: np.ndarray,
    phenotype_grid: PhenotypeGrid,
    radial_grid: RadialGrid,
    params: FullModelParameters,
    numerical: FullModelNumerics,
) -> float:
    """Positive explicit radial FV step; independent of epsilon_phenotype."""
    velocity = radial_face_velocity(
        oxygen, phenotype_grid, radial_grid, params
    )
    nz = phenotype_grid.spec.nz
    nr = radial_grid.n_cells
    rate = np.zeros((nz, nr), dtype=float)
    for face in range(1, nr):
        face_radius = radial_grid.faces[face]
        speed = velocity[:, face]
        left = face - 1
        right = face
        rate[:, left] += face_radius * (
            np.maximum(speed, 0.0) + params.alpha1 / radial_grid.dr
        ) / radial_grid.volumes[left]
        rate[:, right] += face_radius * (
            np.maximum(-speed, 0.0) + params.alpha1 / radial_grid.dr
        ) / radial_grid.volumes[right]
    boundary = params.cell_boundary_condition
    if boundary == "outflow":
        rate[:, -1] += (
            radial_grid.faces[-1]
            * np.maximum(velocity[:, -1], 0.0)
            / radial_grid.volumes[-1]
        )
    elif boundary == "absorbing":
        rate[:, -1] += radial_grid.faces[-1] * (
            np.maximum(velocity[:, -1], 0.0)
            + 2.0 * params.alpha1 / radial_grid.dr
        ) / radial_grid.volumes[-1]
    elif boundary != "no_flux":
        raise ValueError(f"Unknown cell boundary: {boundary}")
    maximum_rate = float(np.max(rate))
    if maximum_rate <= 0.0:
        return numerical.max_dt
    return min(
        numerical.max_dt, numerical.cfl_safety / maximum_rate
    )


def advance_radial_transport(
    rho: np.ndarray,
    oxygen: np.ndarray,
    dt: float,
    phenotype_grid: PhenotypeGrid,
    radial_grid: RadialGrid,
    params: FullModelParameters,
) -> tuple[np.ndarray, dict[str, float]]:
    """Conservative radial diffusion/aerotaxis update of every phenotype."""
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    if dt == 0.0:
        return rho.copy(), {
            "radial_transport_mass_error": 0.0,
            "radial_outward_mass": 0.0,
        }
    nr = radial_grid.n_cells
    velocity = radial_face_velocity(
        oxygen, phenotype_grid, radial_grid, params
    )
    flux = np.zeros(rho.shape[:-1] + (nr + 1,), dtype=float)
    internal_velocity = velocity[:, 1:nr][None, None, :, :]
    upwind = np.where(
        internal_velocity >= 0.0, rho[..., :-1], rho[..., 1:]
    )
    flux[..., 1:nr] = (
        internal_velocity * upwind
        - params.alpha1 * (rho[..., 1:] - rho[..., :-1]) / radial_grid.dr
    )
    outer_velocity = velocity[:, -1][None, None, :]
    boundary = params.cell_boundary_condition
    if boundary == "outflow":
        flux[..., -1] = np.maximum(outer_velocity, 0.0) * rho[..., -1]
    elif boundary == "absorbing":
        flux[..., -1] = (
            np.maximum(outer_velocity, 0.0) * rho[..., -1]
            + 2.0 * params.alpha1 * rho[..., -1] / radial_grid.dr
        )
    elif boundary == "no_flux":
        flux[..., -1] = 0.0
    else:
        raise ValueError(f"Unknown cell boundary: {boundary}")

    divergence = (
        radial_grid.faces[1:] * flux[..., 1:]
        - radial_grid.faces[:-1] * flux[..., :-1]
    ) / radial_grid.volumes
    updated = rho - dt * divergence
    minimum = float(np.min(updated))
    if minimum < -2.0e-11:
        raise RuntimeError(
            f"Radial transport lost positivity: min rho={minimum:.3e}"
        )
    updated = np.maximum(updated, 0.0)

    weight = phenotype_grid.cell_volume
    mass_before = weight * float(
        np.dot(radial_grid.volumes, np.sum(rho, axis=(0, 1, 2)))
    )
    mass_after = weight * float(
        np.dot(radial_grid.volumes, np.sum(updated, axis=(0, 1, 2)))
    )
    outward = (
        dt
        * radial_grid.faces[-1]
        * weight
        * float(np.sum(flux[..., -1]))
    )
    error = abs(mass_after - mass_before + outward) / (
        1.0 + abs(mass_before)
    )
    return updated, {
        "radial_transport_mass_error": float(error),
        "radial_outward_mass": float(outward),
        "maximum_radial_velocity": float(np.max(np.abs(velocity))),
    }


def advance_reaction(
    rho: np.ndarray,
    oxygen: np.ndarray,
    dt: float,
    phenotype_grid: PhenotypeGrid,
    params: FullModelParameters,
) -> np.ndarray:
    """Positive phenotype growth followed by exact common logistic death."""
    if dt < 0.0:
        raise ValueError("dt must be nonnegative")
    if dt == 0.0:
        return rho.copy()
    k_star = params.k1 / params.k2
    x = phenotype_grid.x
    h1_value = params.phi * (1.0 - (1.0 - x / k_star) ** 2)
    h2_factor = np.maximum(1.0 - (x / k_star) ** 2, 0.0)
    hbar = hill(
        oxygen,
        params.hbar_max,
        params.hbar_half,
        params.hbar_hill_power,
    )
    linear_growth = (
        h1_value[:, None] + h2_factor[:, None] * hbar[None, :]
    )  # (nx,nr)
    grown = rho * np.exp(
        np.clip(linear_growth[:, None, None, :] * dt, -50.0, 50.0)
    )
    grown_density = phenotype_grid.cell_volume * np.sum(
        grown, axis=(0, 1, 2)
    )
    return grown / (
        1.0 + params.kappa * grown_density[None, None, None, :] * dt
    )


class FullModelSolver:
    """High-level full-model initializer, stepper and recorder."""

    def __init__(
        self,
        params: FullModelParameters | None = None,
        numerical: FullModelNumerics | None = None,
        phenotype_grid: PhenotypeGrid | None = None,
    ):
        self.params = params or FullModelParameters.calibrated()
        self.numerical = numerical or FullModelNumerics()
        self.radial_grid = make_grid(self.params.radius, self.params.n_cells)
        self.phenotype_grid = phenotype_grid or make_phenotype_grid(
            self.numerical.phenotype_grid_spec
        )
        self.environment_solver = ImplicitEnvironmentSolver(
            self.radial_grid, self.phenotype_grid, self.params
        )

    def initial_variance_parameter(self) -> float:
        """Gaussian variance control; fixed explicitly or tied to epsilon_p."""
        specified = self.numerical.initial_gaussian_variance
        if specified is not None:
            return specified
        return self.numerical.epsilon_phenotype

    def initialize_well_prepared(
        self,
        density: np.ndarray | None = None,
        oxygen: np.ndarray | None = None,
        egf: np.ndarray | None = None,
        *,
        refine_full_environment: bool | None = None,
    ) -> FullState:
        """Create positive Gaussian data centered at the selected phenotype.

        If environmental fields are not supplied, the positive reduced
        elliptic branch is computed using an EGF initial guess of 5.  Optional
        fixed-point refinements replace the reduced sources by Gaussian full
        moments while retaining the same prescribed marginal density.
        """
        grid = self.radial_grid
        params = self.params
        numerical = self.numerical
        if density is None:
            density = initial_density(grid, params)
        density = np.asarray(density, dtype=float)
        if density.shape != (grid.n_cells,):
            raise ValueError("Initial density has the wrong radial shape")
        if float(np.min(density)) < 0.0:
            raise ValueError("Initial density must be nonnegative")

        if (oxygen is None) != (egf is None):
            raise ValueError("Supply both oxygen and EGF, or neither")
        if oxygen is None:
            reduced_environment = ReducedEnvironmentSolver(grid, params)
            oxygen_guess = np.full(
                grid.n_cells, params.oxygen_boundary, dtype=float
            )
            # The O(1) positive seed deliberately selects the activated
            # no-flux EGF branch instead of its invariant zero solution.
            egf_guess = np.full(grid.n_cells, params.egf_initial, dtype=float)
            oxygen, egf, _ = reduced_environment.solve(
                density,
                oxygen_guess,
                egf_guess,
                numerical.reduced_numerics(),
            )
        else:
            oxygen = np.asarray(oxygen, dtype=float).copy()
            egf = np.asarray(egf, dtype=float).copy()
            if oxygen.shape != density.shape or egf.shape != density.shape:
                raise ValueError("Initial environmental fields have wrong shape")

        variance = self.initial_variance_parameter()
        rho = gaussian_phenotype_density(
            density,
            oxygen,
            egf,
            self.phenotype_grid,
            params,
            variance,
        )
        if refine_full_environment is None:
            refinement_count = numerical.well_prepared_full_iterations
        else:
            refinement_count = (
                numerical.well_prepared_full_iterations
                if refine_full_environment
                else 0
            )
        for _ in range(refinement_count):
            oxygen, egf, _ = self.environment_solver.step(
                rho,
                oxygen,
                egf,
                dt=1.0,
                tau_environment=0.0,
                numerical=numerical,
            )
            rho = gaussian_phenotype_density(
                density,
                oxygen,
                egf,
                self.phenotype_grid,
                params,
                variance,
            )
        return FullState(rho=rho, oxygen=oxygen, egf=egf, time=0.0)

    def _validate_state(self, state: FullState) -> None:
        expected_rho = self.phenotype_grid.shape + (
            self.radial_grid.n_cells,
        )
        if state.rho.shape != expected_rho:
            raise ValueError(
                f"rho shape {state.rho.shape}, expected {expected_rho}"
            )
        expected_field = (self.radial_grid.n_cells,)
        if state.oxygen.shape != expected_field or state.egf.shape != expected_field:
            raise ValueError("Environmental field shape mismatch")
        if not (
            np.all(np.isfinite(state.rho))
            and np.all(np.isfinite(state.oxygen))
            and np.all(np.isfinite(state.egf))
        ):
            raise ValueError("State contains a non-finite value")
        if min(
            float(np.min(state.rho)),
            float(np.min(state.oxygen)),
            float(np.min(state.egf)),
        ) < -1.0e-12:
            raise ValueError("State contains a negative value")

    def step(
        self, state: FullState, dt: float
    ) -> tuple[FullState, dict[str, float]]:
        """Advance one positive split step."""
        self._validate_state(state)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        p = self.params
        num = self.numerical
        rho = state.rho
        phenotype_stats: list[dict[str, float]] = []
        phenotype_dt = 0.5 * dt if num.phenotype_strang_splitting else dt
        rho, ph_stats = advance_phenotype_characteristics(
            rho,
            state.oxygen,
            state.egf,
            phenotype_dt,
            num.epsilon_phenotype,
            self.phenotype_grid,
            p,
        )
        phenotype_stats.append(ph_stats)
        rho, radial_stats = advance_radial_transport(
            rho,
            state.oxygen,
            dt,
            self.phenotype_grid,
            self.radial_grid,
            p,
        )
        rho = advance_reaction(
            rho, state.oxygen, dt, self.phenotype_grid, p
        )
        oxygen, egf, environment_stats = self.environment_solver.step(
            rho,
            state.oxygen,
            state.egf,
            dt,
            num.tau_environment,
            num,
        )
        if num.phenotype_strang_splitting:
            rho, ph_stats = advance_phenotype_characteristics(
                rho,
                oxygen,
                egf,
                phenotype_dt,
                num.epsilon_phenotype,
                self.phenotype_grid,
                p,
            )
            phenotype_stats.append(ph_stats)
        new_state = FullState(rho, oxygen, egf, state.time + dt)
        self._validate_state(new_state)
        stats = {
            **radial_stats,
            **environment_stats,
            "phenotype_mass_relative_error": max(
                item["phenotype_mass_relative_error"]
                for item in phenotype_stats
            ),
            "phenotype_boundary_loss_fraction": sum(
                item["phenotype_boundary_loss_fraction"]
                for item in phenotype_stats
            ),
        }
        return new_state, stats

    def simulate(
        self,
        initial_state: FullState | None = None,
    ) -> FullSimulationResult:
        """Run a tiny or production simulation and return moment snapshots."""
        state = initial_state.copy() if initial_state is not None else (
            self.initialize_well_prepared()
        )
        self._validate_state(state)
        if abs(state.time) > 1.0e-14:
            raise ValueError("simulate currently expects initial_state.time == 0")
        num = self.numerical
        grid = self.radial_grid
        pg = self.phenotype_grid
        snapshot_times = make_record_times(
            num.final_time, num.snapshot_interval
        )
        diagnostic_times = make_record_times(
            num.final_time, num.diagnostic_interval
        )
        snapshot_lists: dict[str, list[np.ndarray] | list[float]] = {
            "time": [],
            "density": [],
            "oxygen": [],
            "egf": [],
            "mean_x": [],
            "mean_y": [],
            "mean_z": [],
            "var_x": [],
            "var_y": [],
            "var_z": [],
            "equilibrium_x": [],
            "equilibrium_y": [],
            "equilibrium_z": [],
            "mean_sensitivity": [],
        }
        if num.store_full_rho:
            snapshot_lists["rho"] = []
        diagnostics: list[dict[str, float | bool]] = []
        snapshot_index = 0
        diagnostic_index = 0
        steps = 0
        minimum_dt = math.inf
        maximum_dt = 0.0
        maximum_radial_mass_error = 0.0
        maximum_phenotype_mass_error = 0.0
        cumulative_phenotype_boundary_loss = 0.0
        maximum_environment_residual = 0.0
        latest_step_stats: dict[str, float] = {}

        def current_moments() -> PhenotypeMoments:
            return phenotype_moments(
                state.rho,
                pg,
                self.params,
                state.oxygen,
                num.density_floor_for_moments,
            )

        def weighted_mean(value: np.ndarray, density: np.ndarray) -> float:
            denominator = float(np.dot(grid.volumes, density))
            if denominator <= 1.0e-300:
                return 0.0
            return float(np.dot(grid.volumes, density * value)) / denominator

        def record() -> None:
            nonlocal snapshot_index, diagnostic_index
            moments = current_moments()
            x_eq, y_eq, z_eq = selected_phenotype(
                state.oxygen, state.egf, self.params
            )
            while (
                diagnostic_index < diagnostic_times.size
                and abs(state.time - diagnostic_times[diagnostic_index])
                <= 3.0e-10
            ):
                entry = profile_diagnostics(
                    moments.density, grid, num.reduced_numerics()
                )
                closure_sq = (
                    (moments.mean_x - x_eq) ** 2
                    + (moments.mean_y - y_eq) ** 2
                    + (moments.mean_z - z_eq) ** 2
                )
                entry.update(
                    {
                        "time": float(state.time),
                        "mass": radial_mass(moments.density, grid),
                        "minimum_rho": float(np.min(state.rho)),
                        "maximum_density": float(np.max(moments.density)),
                        "minimum_oxygen": float(np.min(state.oxygen)),
                        "minimum_egf": float(np.min(state.egf)),
                        "mean_var_x": weighted_mean(
                            moments.var_x, moments.density
                        ),
                        "mean_var_y": weighted_mean(
                            moments.var_y, moments.density
                        ),
                        "mean_var_z": weighted_mean(
                            moments.var_z, moments.density
                        ),
                        "mean_equilibrium_distance": math.sqrt(
                            max(weighted_mean(closure_sq, moments.density), 0.0)
                        ),
                        **latest_step_stats,
                    }
                )
                diagnostics.append(entry)
                diagnostic_index += 1
            while (
                snapshot_index < snapshot_times.size
                and abs(state.time - snapshot_times[snapshot_index])
                <= 3.0e-10
            ):
                snapshot_lists["time"].append(float(state.time))
                for key, value in (
                    ("density", moments.density),
                    ("oxygen", state.oxygen),
                    ("egf", state.egf),
                    ("mean_x", moments.mean_x),
                    ("mean_y", moments.mean_y),
                    ("mean_z", moments.mean_z),
                    ("var_x", moments.var_x),
                    ("var_y", moments.var_y),
                    ("var_z", moments.var_z),
                    ("equilibrium_x", x_eq),
                    ("equilibrium_y", y_eq),
                    ("equilibrium_z", z_eq),
                    (
                        "mean_sensitivity",
                        np.divide(
                            moments.sensitivity_mass,
                            moments.density,
                            out=np.zeros_like(moments.density),
                            where=moments.density
                            > num.density_floor_for_moments,
                        ),
                    ),
                ):
                    snapshot_lists[key].append(value.copy())
                if num.store_full_rho:
                    snapshot_lists["rho"].append(state.rho.copy())
                snapshot_index += 1

        record()
        while state.time < num.final_time - 1.0e-12:
            dt = stable_radial_timestep(
                state.oxygen, pg, grid, self.params, num
            )
            next_record = num.final_time
            if snapshot_index < snapshot_times.size:
                next_record = min(next_record, snapshot_times[snapshot_index])
            if diagnostic_index < diagnostic_times.size:
                next_record = min(next_record, diagnostic_times[diagnostic_index])
            dt = min(dt, next_record - state.time, num.final_time - state.time)
            if dt <= 1.0e-14:
                state.time = next_record
                record()
                continue
            state, latest_step_stats = self.step(state, dt)
            if abs(state.time - next_record) <= 3.0e-10:
                state.time = next_record
            steps += 1
            minimum_dt = min(minimum_dt, dt)
            maximum_dt = max(maximum_dt, dt)
            maximum_radial_mass_error = max(
                maximum_radial_mass_error,
                latest_step_stats["radial_transport_mass_error"],
            )
            maximum_phenotype_mass_error = max(
                maximum_phenotype_mass_error,
                latest_step_stats["phenotype_mass_relative_error"],
            )
            cumulative_phenotype_boundary_loss += latest_step_stats[
                "phenotype_boundary_loss_fraction"
            ]
            maximum_environment_residual = max(
                maximum_environment_residual,
                latest_step_stats["oxygen_residual"],
                latest_step_stats["egf_residual"],
            )
            record()

        snapshots = {
            key: np.asarray(value, dtype=float)
            for key, value in snapshot_lists.items()
        }
        run_stats: dict[str, float | int | str] = {
            "tau_environment": num.tau_environment,
            "epsilon_phenotype": num.epsilon_phenotype,
            "initial_gaussian_variance": self.initial_variance_parameter(),
            "step_count": steps,
            "minimum_dt": float(0.0 if steps == 0 else minimum_dt),
            "maximum_dt": float(maximum_dt),
            "maximum_radial_transport_mass_error": float(
                maximum_radial_mass_error
            ),
            "maximum_phenotype_mass_error": float(
                maximum_phenotype_mass_error
            ),
            "cumulative_phenotype_boundary_loss_fraction": (
                float(cumulative_phenotype_boundary_loss)
            ),
            "maximum_environment_residual": float(
                maximum_environment_residual
            ),
            "cell_boundary_condition": self.params.cell_boundary_condition,
        }
        return FullSimulationResult(
            radial_grid=grid,
            phenotype_grid=pg,
            snapshots=snapshots,
            diagnostics=diagnostics,
            final_state=state,
            run_stats=run_stats,
        )


def simulate_full_model(
    params: FullModelParameters | None = None,
    numerical: FullModelNumerics | None = None,
    initial_state: FullState | None = None,
    *,
    initial_density_profile: np.ndarray | None = None,
    initial_reduced_fields: tuple[np.ndarray, np.ndarray] | None = None,
) -> FullSimulationResult:
    """Convenience wrapper used by tau/epsilon scan drivers."""
    solver = FullModelSolver(params=params, numerical=numerical)
    if initial_state is None and (
        initial_density_profile is not None or initial_reduced_fields is not None
    ):
        oxygen = egf = None
        if initial_reduced_fields is not None:
            oxygen, egf = initial_reduced_fields
        initial_state = solver.initialize_well_prepared(
            density=initial_density_profile, oxygen=oxygen, egf=egf
        )
    return solver.simulate(initial_state)


__all__ = [
    "FullModelParameters",
    "PhenotypeGridSpec",
    "PhenotypeGrid",
    "FullModelNumerics",
    "FullState",
    "PhenotypeMoments",
    "FullSimulationResult",
    "make_phenotype_grid",
    "gaussian_phenotype_density",
    "phenotype_moments",
    "ImplicitEnvironmentSolver",
    "advance_phenotype_characteristics",
    "radial_face_velocity",
    "stable_radial_timestep",
    "advance_radial_transport",
    "advance_reaction",
    "FullModelSolver",
    "simulate_full_model",
]
