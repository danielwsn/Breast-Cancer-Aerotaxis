"""Radial finite-volume operators for the phenotype-selected model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ModelParameters:
    """Biological, radial, initial and boundary parameters.

    These are generic solver defaults. Use
    ``FullModelParameters.calibrated`` for the reported convergence experiment.
    """

    # Radial geometry and environmental boundary values.
    radius: float = 5.0  # R: outer radius of 0 <= r <= R.
    n_cells: int = 320  # Number of cell-centered radial finite volumes.
    oxygen_boundary: float = 10.0  # External oxygen value used by Dirichlet/Robin.
    egf_boundary: float = 6.0  # External EGF value used by Dirichlet/Robin.
    oxygen_boundary_type: str = "dirichlet"  # dirichlet, no_flux or robin.
    egf_boundary_type: str = "dirichlet"  # dirichlet, no_flux or robin.
    oxygen_robin_transfer: float = 1.0  # Oxygen Robin transfer coefficient.
    egf_robin_transfer: float = 1.0  # EGF Robin transfer coefficient.
    egf_initial: float = 6.0  # Newton seed used to select an EGF solution branch.

    # Intracellular phenotype dynamics and equilibrium selection.
    k1: float = 5.0  # Production in U_x=k1-(k2+f1(u))*x.
    k2: float = 1.0  # Basal phenotype decay/inactivation rate.
    k3: float = 1.0  # EGF activation coefficient in U_z=k3*v-d2*z.
    d1: float = 1.0  # Linear extracellular EGF degradation coefficient.
    d2: float = 0.2  # Activated-EGFR phenotype decay coefficient.

    f1_max: float = 25.0  # Maximum oxygen response f1(u).
    f1_half: float = 5.0  # Half-saturation of f1(u).
    f2_max: float = 4.0  # Maximum activated-EGFR response f2(z).
    f2_half: float = 4.0  # Half-saturation of f2(z).
    f3_max: float = 8.0  # Maximum phenotype-dependent EGF production f3(y).
    f3_half: float = 2.0  # Half-saturation of f3(y).
    signalling_hill_power: int = 2  # Common Hill exponent of f1, f2 and f3.

    # Growth and density saturation.
    phi: float = 0.8  # Maximum coefficient in anaerobic growth h1(x).
    hbar_max: float = 3.0  # Maximum oxygen-dependent aerobic rate hbar(u).
    hbar_half: float = 5.0  # Half-saturation oxygen level of hbar(u).
    hbar_hill_power: int = 2  # Hill exponent of hbar(u).
    kappa: float = 1.5  # Logistic crowding coefficient in -kappa*N.

    # Radial diffusion and environmental reaction coefficients.
    alpha1: float = 3.0e-3  # Cell random-motility diffusivity.
    alpha2: float = 1.0  # Oxygen diffusivity.
    alpha3: float = 1.0  # EGF diffusivity.
    xi1: float = 2.0  # Linear oxygen loss coefficient.
    eta1: float = 1.0  # Cell-mediated oxygen-consumption strength.
    eta2: float = 1.0  # Phenotype-dependent EGF-production strength.
    eta3: float = 1.0  # Cell-mediated EGF-uptake strength.
    oxygen_source: float = 0.0  # Volumetric oxygen source q.

    # Smooth phenotype-dependent aerotactic switch sigma(z).
    sigma_half: float = 6.0  # Half-saturation phenotype level.
    sigma_hill_power: int = 8  # Hill exponent controlling switch steepness.

    # Initial radial front N0=0.5*plateau*(1-tanh((r-r0)/width)).
    initial_plateau: float = 0.25  # Central density plateau.
    initial_radius: float = 0.6  # Initial interface location r0.
    initial_width: float = 0.08  # Initial interface thickness.


@dataclass(frozen=True)
class NumericalParameters:
    """Time stepping, elliptic solve, output and morphology controls."""

    final_time: float = 50.0  # Final time T.
    snapshot_interval: float = 5.0  # Saved-profile interval.
    diagnostic_interval: float = 0.1  # Morphology-diagnostic interval.
    max_dt: float = 5.0e-3  # Maximum step before applying CFL constraints.
    cfl_safety: float = 0.72  # Explicit finite-volume safety factor.
    elliptic_tolerance: float = 2.0e-10  # Relative Newton residual tolerance.
    elliptic_max_iterations: int = 30  # Newton iteration cap.
    boundary_exclusion: float = 0.25  # Boundary strip ignored in peak detection.
    slope_tolerance: float = 1.0e-4  # Numerical zero for radial slope.
    amplitude_tolerance: float = 2.0e-4  # Minimum extrema prominence.


@dataclass(frozen=True)
class RadialGrid:
    radius: float
    n_cells: int
    dr: float
    centers: np.ndarray
    faces: np.ndarray
    volumes: np.ndarray


def make_grid(radius: float, n_cells: int) -> RadialGrid:
    dr = radius / n_cells
    faces = np.arange(n_cells + 1, dtype=float) * dr
    centers = (np.arange(n_cells, dtype=float) + 0.5) * dr
    # Cell area divided by 2*pi.  This is exactly integral r dr over the cell.
    volumes = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    return RadialGrid(radius, n_cells, dr, centers, faces, volumes)


def hill(value: np.ndarray, maximum: float, half: float, power: int) -> np.ndarray:
    value_nonnegative = np.maximum(value, 0.0)
    scaled_power = (value_nonnegative / half) ** power
    return maximum * scaled_power / (1.0 + scaled_power)


def hill_derivative(
    value: np.ndarray, maximum: float, half: float, power: int
) -> np.ndarray:
    value_nonnegative = np.maximum(value, 0.0)
    scaled = value_nonnegative / half
    scaled_power = scaled**power
    derivative = np.zeros_like(value_nonnegative)
    positive = value_nonnegative > 0.0
    derivative[positive] = (
        maximum
        * power
        * scaled[positive] ** (power - 1)
        / (half * (1.0 + scaled_power[positive]) ** 2)
    )
    return derivative


def selected_phenotype(
    oxygen: np.ndarray, egf: np.ndarray, p: ModelParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f1_value = hill(
        oxygen, p.f1_max, p.f1_half, p.signalling_hill_power
    )
    x_star = p.k1 / (p.k2 + f1_value)
    z_star = p.k3 * np.maximum(egf, 0.0) / p.d2
    f2_value = hill(z_star, p.f2_max, p.f2_half, p.signalling_hill_power)
    y_star = x_star * f2_value / (f2_value + p.k2 + f1_value)
    return x_star, y_star, z_star


def growth_components(
    oxygen: np.ndarray, p: ModelParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x*, h1, h2 and d(h2)/du."""
    f1_value = hill(
        oxygen, p.f1_max, p.f1_half, p.signalling_hill_power
    )
    f1_prime = hill_derivative(
        oxygen, p.f1_max, p.f1_half, p.signalling_hill_power
    )
    denominator = p.k2 + f1_value
    x_star = p.k1 / denominator
    x_prime = -p.k1 * f1_prime / denominator**2
    k_star = p.k1 / p.k2

    h1_value = p.phi * (1.0 - (1.0 - x_star / k_star) ** 2)
    hbar_value = hill(
        oxygen, p.hbar_max, p.hbar_half, p.hbar_hill_power
    )
    hbar_prime = hill_derivative(
        oxygen, p.hbar_max, p.hbar_half, p.hbar_hill_power
    )
    aerobic_factor = 1.0 - (x_star / k_star) ** 2
    h2_value = hbar_value * aerobic_factor
    h2_prime = (
        hbar_prime * aerobic_factor
        - 2.0 * hbar_value * x_star * x_prime / k_star**2
    )
    return x_star, h1_value, h2_value, h2_prime


def aerotaxis_sensitivity(
    z_star: np.ndarray, sigma_max: float, p: ModelParameters
) -> np.ndarray:
    return hill(z_star, sigma_max, p.sigma_half, p.sigma_hill_power)


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Thomas algorithm without modifying the supplied arrays."""
    n = diagonal.size
    c_prime = np.empty(max(n - 1, 0), dtype=float)
    d_prime = np.empty(n, dtype=float)
    pivot = diagonal[0]
    if abs(pivot) < 1.0e-14:
        raise RuntimeError("Near-zero pivot in tridiagonal solve")
    if n > 1:
        c_prime[0] = upper[0] / pivot
    d_prime[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diagonal[i] - lower[i - 1] * c_prime[i - 1]
        if abs(pivot) < 1.0e-14:
            raise RuntimeError("Near-zero pivot in tridiagonal solve")
        if i < n - 1:
            c_prime[i] = upper[i] / pivot
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / pivot

    solution = np.empty(n, dtype=float)
    solution[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


def tridiagonal_matvec(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    product = diagonal * value
    product[1:] += lower * value[:-1]
    product[:-1] += upper * value[1:]
    return product


def negative_laplacian_coefficients(
    grid: RadialGrid,
    diffusivity: float,
    outer_value: float,
    boundary_type: str = "dirichlet",
    robin_transfer: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Finite-volume coefficients for -diffusivity*Delta_r.

    The center face has zero flux.  At r=R, the Dirichlet value is separated
    into the returned boundary right-hand side and uses the half-cell distance.
    """
    n = grid.n_cells
    lower = np.zeros(n - 1, dtype=float)
    diagonal = np.zeros(n, dtype=float)
    upper = np.zeros(n - 1, dtype=float)
    boundary_rhs = np.zeros(n, dtype=float)

    for i in range(n):
        volume = grid.volumes[i]
        if i > 0:
            coefficient = diffusivity * grid.faces[i] / (volume * grid.dr)
            lower[i - 1] = -coefficient
            diagonal[i] += coefficient
        if i < n - 1:
            coefficient = diffusivity * grid.faces[i + 1] / (volume * grid.dr)
            upper[i] = -coefficient
            diagonal[i] += coefficient
        else:
            half_cell = 0.5 * grid.dr
            if boundary_type == "dirichlet":
                conductance = diffusivity / half_cell
            elif boundary_type == "no_flux":
                conductance = 0.0
            elif boundary_type == "robin":
                if robin_transfer < 0.0:
                    raise ValueError("Robin transfer coefficient must be nonnegative")
                conductance = (
                    diffusivity * robin_transfer
                    / (diffusivity + robin_transfer * half_cell)
                )
            else:
                raise ValueError(
                    f"Unknown environmental boundary type: {boundary_type}"
                )
            coefficient = conductance * grid.faces[-1] / volume
            diagonal[i] += coefficient
            boundary_rhs[i] += coefficient * outer_value
    return lower, diagonal, upper, boundary_rhs


def outer_face_value(
    cell_value: float,
    diffusivity: float,
    grid: RadialGrid,
    outer_value: float,
    boundary_type: str,
    robin_transfer: float,
) -> float:
    """Recover the environmental value at the outer face."""
    if boundary_type == "dirichlet":
        return outer_value
    if boundary_type == "no_flux":
        return cell_value
    if boundary_type == "robin":
        half_cell = 0.5 * grid.dr
        return (
            diffusivity * cell_value / half_cell
            + robin_transfer * outer_value
        ) / (diffusivity / half_cell + robin_transfer)
    raise ValueError(f"Unknown environmental boundary type: {boundary_type}")


def damped_newton(
    initial: np.ndarray,
    residual_and_diagonal,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
    max_iterations: int,
    name: str,
) -> tuple[np.ndarray, int, float]:
    value = np.maximum(initial.copy(), 0.0)
    residual, jacobian_diagonal = residual_and_diagonal(value)
    scale = 1.0 + float(np.max(np.abs(value)))
    residual_norm = float(np.max(np.abs(residual))) / scale

    for iteration in range(1, max_iterations + 1):
        if residual_norm <= tolerance:
            return value, iteration - 1, residual_norm
        update = solve_tridiagonal(
            lower, jacobian_diagonal, upper, -residual
        )
        step = 1.0
        accepted = False
        while step >= 2.0**-14:
            candidate = value + step * update
            if np.min(candidate) < -1.0e-12:
                step *= 0.5
                continue
            candidate = np.maximum(candidate, 0.0)
            new_residual, new_diagonal = residual_and_diagonal(candidate)
            new_scale = 1.0 + float(np.max(np.abs(candidate)))
            new_norm = float(np.max(np.abs(new_residual))) / new_scale
            if new_norm <= (1.0 - 1.0e-4 * step) * residual_norm:
                value = candidate
                residual = new_residual
                jacobian_diagonal = new_diagonal
                residual_norm = new_norm
                accepted = True
                break
            step *= 0.5
        if not accepted:
            raise RuntimeError(
                f"{name} Newton line search failed at residual {residual_norm:.3e}"
            )
    raise RuntimeError(
        f"{name} Newton solver did not converge; residual={residual_norm:.3e}"
    )


class EnvironmentSolver:
    def __init__(self, grid: RadialGrid, p: ModelParameters):
        self.grid = grid
        self.p = p
        (
            self.u_lower,
            self.u_diffusion_diagonal,
            self.u_upper,
            self.u_boundary_rhs,
        ) = negative_laplacian_coefficients(
            grid,
            p.alpha2,
            p.oxygen_boundary,
            p.oxygen_boundary_type,
            p.oxygen_robin_transfer,
        )
        (
            self.v_lower,
            self.v_diffusion_diagonal,
            self.v_upper,
            self.v_boundary_rhs,
        ) = negative_laplacian_coefficients(
            grid,
            p.alpha3,
            p.egf_boundary,
            p.egf_boundary_type,
            p.egf_robin_transfer,
        )

    def solve_oxygen(
        self,
        density: np.ndarray,
        initial: np.ndarray,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, int, float]:
        base_diagonal = self.u_diffusion_diagonal + self.p.xi1
        rhs = self.u_boundary_rhs + self.p.oxygen_source

        def residual_and_diagonal(oxygen: np.ndarray):
            _, _, h2_value, h2_prime = growth_components(oxygen, self.p)
            residual = (
                tridiagonal_matvec(
                    self.u_lower, base_diagonal, self.u_upper, oxygen
                )
                + self.p.eta1 * density * h2_value
                - rhs
            )
            jacobian_diagonal = (
                base_diagonal + self.p.eta1 * density * h2_prime
            )
            return residual, jacobian_diagonal

        return damped_newton(
            initial,
            residual_and_diagonal,
            self.u_lower,
            self.u_upper,
            tolerance,
            max_iterations,
            "oxygen",
        )

    def solve_egf(
        self,
        density: np.ndarray,
        oxygen: np.ndarray,
        initial: np.ndarray,
        tolerance: float,
        max_iterations: int,
    ) -> tuple[np.ndarray, int, float]:
        p = self.p
        f1_value = hill(
            oxygen, p.f1_max, p.f1_half, p.signalling_hill_power
        )
        x_star = p.k1 / (p.k2 + f1_value)
        base_diagonal = (
            self.v_diffusion_diagonal + p.d1 + p.eta3 * p.k3 * density
        )
        rhs_boundary = self.v_boundary_rhs

        def residual_and_diagonal(egf: np.ndarray):
            z_star = p.k3 * np.maximum(egf, 0.0) / p.d2
            f2_value = hill(
                z_star, p.f2_max, p.f2_half, p.signalling_hill_power
            )
            f2_prime = hill_derivative(
                z_star, p.f2_max, p.f2_half, p.signalling_hill_power
            )
            denominator = f2_value + p.k2 + f1_value
            y_star = x_star * f2_value / denominator
            dy_dv = (
                x_star
                * (p.k2 + f1_value)
                / denominator**2
                * f2_prime
                * p.k3
                / p.d2
            )
            f3_value = hill(
                y_star, p.f3_max, p.f3_half, p.signalling_hill_power
            )
            f3_prime = hill_derivative(
                y_star, p.f3_max, p.f3_half, p.signalling_hill_power
            )
            production = p.eta2 * density * f3_value
            residual = (
                tridiagonal_matvec(
                    self.v_lower, base_diagonal, self.v_upper, egf
                )
                - rhs_boundary
                - production
            )
            jacobian_diagonal = (
                base_diagonal - p.eta2 * density * f3_prime * dy_dv
            )
            return residual, jacobian_diagonal

        return damped_newton(
            initial,
            residual_and_diagonal,
            self.v_lower,
            self.v_upper,
            tolerance,
            max_iterations,
            "EGF",
        )

    def solve(
        self,
        density: np.ndarray,
        oxygen_initial: np.ndarray,
        egf_initial: np.ndarray,
        numerical: NumericalParameters,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        oxygen, u_iterations, u_residual = self.solve_oxygen(
            density,
            oxygen_initial,
            numerical.elliptic_tolerance,
            numerical.elliptic_max_iterations,
        )
        egf, v_iterations, v_residual = self.solve_egf(
            density,
            oxygen,
            egf_initial,
            numerical.elliptic_tolerance,
            numerical.elliptic_max_iterations,
        )
        stats = {
            "oxygen_newton_iterations": float(u_iterations),
            "egf_newton_iterations": float(v_iterations),
            "oxygen_residual": u_residual,
            "egf_residual": v_residual,
        }
        return oxygen, egf, stats


def face_velocity(
    oxygen: np.ndarray,
    egf: np.ndarray,
    sigma_max: float,
    grid: RadialGrid,
    p: ModelParameters,
) -> np.ndarray:
    n = grid.n_cells
    velocity = np.zeros(n + 1, dtype=float)
    oxygen_gradient = (oxygen[1:] - oxygen[:-1]) / grid.dr
    egf_face = 0.5 * (egf[:-1] + egf[1:])
    z_face = p.k3 * egf_face / p.d2
    velocity[1:n] = (
        aerotaxis_sensitivity(z_face, sigma_max, p) * oxygen_gradient
    )

    outer_oxygen = outer_face_value(
        oxygen[-1],
        p.alpha2,
        grid,
        p.oxygen_boundary,
        p.oxygen_boundary_type,
        p.oxygen_robin_transfer,
    )
    outer_egf = outer_face_value(
        egf[-1],
        p.alpha3,
        grid,
        p.egf_boundary,
        p.egf_boundary_type,
        p.egf_robin_transfer,
    )
    outer_oxygen_gradient = (outer_oxygen - oxygen[-1]) / (0.5 * grid.dr)
    outer_z = np.asarray([p.k3 * outer_egf / p.d2])
    velocity[-1] = (
        aerotaxis_sensitivity(outer_z, sigma_max, p)[0]
        * outer_oxygen_gradient
    )
    # velocity[0] is exactly zero by radial symmetry.
    return velocity


def cell_flux(
    density: np.ndarray,
    velocity: np.ndarray,
    grid: RadialGrid,
    p: ModelParameters,
    boundary_condition: str,
) -> np.ndarray:
    n = grid.n_cells
    flux = np.zeros(n + 1, dtype=float)
    internal_velocity = velocity[1:n]
    upwind_density = np.where(
        internal_velocity >= 0.0, density[:-1], density[1:]
    )
    flux[1:n] = (
        internal_velocity * upwind_density
        - p.alpha1 * (density[1:] - density[:-1]) / grid.dr
    )

    if boundary_condition == "outflow":
        # Homogeneous Neumann condition for diffusion and zero exterior density
        # for any (unexpected) inward advective characteristic.
        flux[-1] = max(velocity[-1], 0.0) * density[-1]
    elif boundary_condition == "absorbing":
        # A genuinely open boundary: N(R)=0 for diffusion (half-cell
        # distance) and no advective inflow from the exterior.
        flux[-1] = (
            max(velocity[-1], 0.0) * density[-1]
            + 2.0 * p.alpha1 * density[-1] / grid.dr
        )
    elif boundary_condition == "no_flux":
        flux[-1] = 0.0
    else:
        raise ValueError(f"Unknown cell boundary condition: {boundary_condition}")
    return flux


def stable_timestep(
    velocity: np.ndarray,
    grid: RadialGrid,
    p: ModelParameters,
    numerical: NumericalParameters,
    boundary_condition: str,
) -> float:
    outgoing_rate = np.zeros(grid.n_cells, dtype=float)
    for face in range(1, grid.n_cells):
        face_radius = grid.faces[face]
        speed = velocity[face]
        left = face - 1
        right = face
        outgoing_rate[left] += face_radius * (
            max(speed, 0.0) + p.alpha1 / grid.dr
        ) / grid.volumes[left]
        outgoing_rate[right] += face_radius * (
            max(-speed, 0.0) + p.alpha1 / grid.dr
        ) / grid.volumes[right]
    if boundary_condition == "outflow":
        outgoing_rate[-1] += (
            grid.faces[-1] * max(velocity[-1], 0.0) / grid.volumes[-1]
        )
    elif boundary_condition == "absorbing":
        outgoing_rate[-1] += grid.faces[-1] * (
            max(velocity[-1], 0.0) + 2.0 * p.alpha1 / grid.dr
        ) / grid.volumes[-1]
    maximum_rate = float(np.max(outgoing_rate))
    if maximum_rate <= 0.0:
        return numerical.max_dt
    return min(numerical.max_dt, numerical.cfl_safety / maximum_rate)


def exact_logistic_step(
    density: np.ndarray,
    linear_growth: np.ndarray,
    kappa: float,
    dt: float,
) -> np.ndarray:
    exponent = np.exp(np.minimum(linear_growth * dt, 50.0))
    denominator = (
        linear_growth + kappa * density * np.expm1(linear_growth * dt)
    )
    result = np.empty_like(density)
    regular = np.abs(linear_growth) > 1.0e-12
    result[regular] = (
        linear_growth[regular]
        * density[regular]
        * exponent[regular]
        / denominator[regular]
    )
    result[~regular] = density[~regular] / (
        1.0 + kappa * density[~regular] * dt
    )
    return result


def advance_density(
    density: np.ndarray,
    oxygen: np.ndarray,
    egf: np.ndarray,
    sigma_max: float,
    dt: float,
    grid: RadialGrid,
    p: ModelParameters,
    boundary_condition: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    velocity = face_velocity(oxygen, egf, sigma_max, grid, p)
    flux = cell_flux(density, velocity, grid, p, boundary_condition)
    divergence = (
        grid.faces[1:] * flux[1:] - grid.faces[:-1] * flux[:-1]
    ) / grid.volumes
    transported = density - dt * divergence
    mass_change = float(np.dot(grid.volumes, transported - density))
    expected_mass_change = -dt * grid.faces[-1] * flux[-1]
    transport_mass_error = abs(mass_change - expected_mass_change) / (
        1.0 + abs(float(np.dot(grid.volumes, density)))
    )
    minimum = float(np.min(transported))
    if minimum < -2.0e-11:
        raise RuntimeError(
            f"Transport step lost positivity: minimum density={minimum:.3e}"
        )
    transported = np.maximum(transported, 0.0)
    _, h1_value, h2_value, _ = growth_components(oxygen, p)
    updated = exact_logistic_step(
        transported, h1_value + h2_value, p.kappa, dt
    )
    return updated, velocity, flux, transport_mass_error


def initial_density(grid: RadialGrid, p: ModelParameters) -> np.ndarray:
    return 0.5 * p.initial_plateau * (
        1.0 - np.tanh((grid.centers - p.initial_radius) / p.initial_width)
    )


def profile_diagnostics(
    density: np.ndarray,
    grid: RadialGrid,
    numerical: NumericalParameters,
) -> dict[str, float | bool]:
    slopes = np.diff(density) / grid.dr
    face_positions = grid.faces[1:-1]
    admissible = (
        (face_positions >= 2.0 * grid.dr)
        & (face_positions <= grid.radius - numerical.boundary_exclusion)
    )
    admissible_indices = np.nonzero(admissible)[0]
    maximum_positive_slope = (
        float(np.max(slopes[admissible])) if np.any(admissible) else 0.0
    )

    positive_variation = (
        float(np.sum(np.maximum(np.diff(density)[admissible], 0.0)))
        / max(float(np.max(density)), 1.0e-14)
        if np.any(admissible)
        else 0.0
    )
    best_ring_amplitude = 0.0
    ring_trough_index = -1
    ring_peak_index = -1
    best_strict_amplitude = 0.0
    strict_trough_index = -1
    strict_peak_index = -1
    if admissible_indices.size:
        start = int(admissible_indices[0])
        stop = int(admissible_indices[-1])
        i = start
        while i <= stop:
            # A trough is selected when the slope changes from negative to
            # a resolvably positive run (flat cells are skipped).
            if slopes[i] <= numerical.slope_tolerance:
                i += 1
                continue
            positive_start = i
            while i <= stop and slopes[i] > numerical.slope_tolerance:
                i += 1
            positive_end = i - 1
            positive_length = positive_end - positive_start + 1
            left_negative = slopes[
                max(start, positive_start - 3) : positive_start
            ]
            right_negative = slopes[
                positive_end + 1 : min(stop + 1, positive_end + 4)
            ]
            has_left_descent = (
                left_negative.size == 3
                and np.all(left_negative < -numerical.slope_tolerance)
            )
            if (
                positive_length >= 3
                and right_negative.size == 3
                and np.all(right_negative < -numerical.slope_tolerance)
            ):
                if has_left_descent:
                    candidate_trough = positive_start
                else:
                    # A radial ring may grow directly out of a depressed
                    # center, in which case the center itself is the trough.
                    candidate_trough = int(
                        np.argmin(density[: positive_start + 1])
                    )
                candidate_peak = positive_end + 1
                amplitude = float(
                    density[candidate_peak] - density[candidate_trough]
                )
                if amplitude > best_ring_amplitude:
                    best_ring_amplitude = amplitude
                    ring_trough_index = candidate_trough
                    ring_peak_index = candidate_peak
                if has_left_descent:
                    strict_amplitude = float(
                        density[candidate_peak] - density[positive_start]
                    )
                    if strict_amplitude > best_strict_amplitude:
                        best_strict_amplitude = strict_amplitude
                        strict_trough_index = positive_start
                        strict_peak_index = candidate_peak
            i = max(i, positive_end + 2)

    required_amplitude = max(
        numerical.amplitude_tolerance, 0.005 * float(np.max(density))
    )
    nonmonotone = best_ring_amplitude > required_amplitude
    strict_trough_peak_pair = best_strict_amplitude > required_amplitude
    interior_limit = grid.radius - numerical.boundary_exclusion
    interior_cells = grid.centers <= interior_limit
    interior_peak_index = int(np.argmax(np.where(interior_cells, density, -np.inf)))
    ring_index = float(density[interior_peak_index] - density[0])
    whole_peak_index = int(np.argmax(density))
    return {
        "nonmonotone": bool(nonmonotone),
        "strict_trough_peak_pair": bool(strict_trough_peak_pair),
        "maximum_positive_slope": maximum_positive_slope,
        "normalized_positive_variation": positive_variation,
        "ring_amplitude": best_ring_amplitude,
        "trough_peak_amplitude": best_strict_amplitude,
        "required_amplitude": required_amplitude,
        "trough_radius": (
            float(grid.centers[strict_trough_index])
            if strict_trough_index >= 0
            else math.nan
        ),
        "peak_radius": (
            float(grid.centers[strict_peak_index])
            if strict_peak_index >= 0
            else math.nan
        ),
        "ring_trough_radius": (
            float(grid.centers[ring_trough_index])
            if ring_trough_index >= 0
            else math.nan
        ),
        "ring_peak_radius": (
            float(grid.centers[ring_peak_index])
            if ring_peak_index >= 0
            else math.nan
        ),
        "interior_peak_radius": float(grid.centers[interior_peak_index]),
        "interior_peak_density": float(density[interior_peak_index]),
        "center_density": float(density[0]),
        "ring_index": ring_index,
        "whole_peak_radius": float(grid.centers[whole_peak_index]),
        "whole_peak_density": float(density[whole_peak_index]),
        "whole_ring_index": float(density[whole_peak_index] - density[0]),
        "wall_cell_density": float(density[-1]),
    }


def radial_mass(density: np.ndarray, grid: RadialGrid) -> float:
    return 2.0 * math.pi * float(np.dot(grid.volumes, density))


def make_record_times(final_time: float, interval: float) -> np.ndarray:
    count = int(math.floor(final_time / interval + 1.0e-12))
    values = np.arange(count + 1, dtype=float) * interval
    if values[-1] < final_time - 1.0e-12:
        values = np.append(values, final_time)
    else:
        values[-1] = final_time
    return values


def simulate_case(
    sigma_max: float,
    p: ModelParameters,
    numerical: NumericalParameters,
    boundary_condition: str,
) -> dict[str, object]:
    grid = make_grid(p.radius, p.n_cells)
    density = initial_density(grid, p)
    environment_solver = EnvironmentSolver(grid, p)
    oxygen_guess = np.full(grid.n_cells, p.oxygen_boundary, dtype=float)
    egf_guess = np.full(grid.n_cells, p.egf_initial, dtype=float)
    oxygen, egf, environment_stats = environment_solver.solve(
        density, oxygen_guess, egf_guess, numerical
    )

    snapshot_times = make_record_times(
        numerical.final_time, numerical.snapshot_interval
    )
    diagnostic_times = make_record_times(
        numerical.final_time, numerical.diagnostic_interval
    )
    snapshots: dict[str, list[np.ndarray] | list[float]] = {
        "time": [],
        "density": [],
        "oxygen": [],
        "egf": [],
        "sigma": [],
        "velocity": [],
    }
    diagnostics: list[dict[str, float | bool]] = []

    snapshot_index = 0
    diagnostic_index = 0
    time = 0.0
    step_count = 0
    minimum_dt = math.inf
    maximum_dt_used = 0.0
    maximum_transport_mass_error = 0.0
    maximum_environment_iterations = max(
        environment_stats["oxygen_newton_iterations"],
        environment_stats["egf_newton_iterations"],
    )
    transition_time = math.nan
    strict_transition_time = math.nan
    consecutive_nonmonotone_records = 0
    consecutive_strict_records = 0

    def record_current_state() -> None:
        nonlocal snapshot_index, diagnostic_index, transition_time
        nonlocal consecutive_nonmonotone_records
        nonlocal strict_transition_time, consecutive_strict_records
        velocity_now = face_velocity(oxygen, egf, sigma_max, grid, p)
        while (
            diagnostic_index < diagnostic_times.size
            and abs(time - diagnostic_times[diagnostic_index]) <= 2.0e-10
        ):
            entry = profile_diagnostics(density, grid, numerical)
            entry.update(
                {
                    "time": float(time),
                    "mass": radial_mass(density, grid),
                    "front_radius": float(
                        grid.centers[
                            np.max(
                                np.nonzero(
                                    density
                                    >= 0.05 * max(float(np.max(density)), 1.0e-14)
                                )[0]
                            )
                        ]
                    ),
                    "maximum_speed": float(np.max(np.abs(velocity_now))),
                    "minimum_density": float(np.min(density)),
                    "maximum_density": float(np.max(density)),
                }
            )
            diagnostics.append(entry)
            if bool(entry["nonmonotone"]):
                consecutive_nonmonotone_records += 1
                if (
                    consecutive_nonmonotone_records >= 3
                    and math.isnan(transition_time)
                ):
                    transition_time = float(
                        diagnostics[-consecutive_nonmonotone_records]["time"]
                    )
            else:
                consecutive_nonmonotone_records = 0
            if bool(entry["strict_trough_peak_pair"]):
                consecutive_strict_records += 1
                if consecutive_strict_records >= 3 and math.isnan(strict_transition_time):
                    strict_transition_time = float(
                        diagnostics[-consecutive_strict_records]["time"]
                    )
            else:
                consecutive_strict_records = 0
            diagnostic_index += 1

        while (
            snapshot_index < snapshot_times.size
            and abs(time - snapshot_times[snapshot_index]) <= 2.0e-10
        ):
            _, _, z_star = selected_phenotype(oxygen, egf, p)
            snapshots["time"].append(float(time))
            snapshots["density"].append(density.copy())
            snapshots["oxygen"].append(oxygen.copy())
            snapshots["egf"].append(egf.copy())
            snapshots["sigma"].append(
                aerotaxis_sensitivity(z_star, sigma_max, p)
            )
            snapshots["velocity"].append(velocity_now.copy())
            snapshot_index += 1

    record_current_state()
    next_progress_time = 5.0
    while time < numerical.final_time - 1.0e-12:
        velocity = face_velocity(oxygen, egf, sigma_max, grid, p)
        dt = stable_timestep(
            velocity, grid, p, numerical, boundary_condition
        )
        next_record_time = numerical.final_time
        if diagnostic_index < diagnostic_times.size:
            next_record_time = min(next_record_time, diagnostic_times[diagnostic_index])
        if snapshot_index < snapshot_times.size:
            next_record_time = min(next_record_time, snapshot_times[snapshot_index])
        dt = min(dt, next_record_time - time, numerical.final_time - time)
        if dt <= 1.0e-14:
            time = next_record_time
            record_current_state()
            continue

        density, _, _, transport_mass_error = advance_density(
            density,
            oxygen,
            egf,
            sigma_max,
            dt,
            grid,
            p,
            boundary_condition,
        )
        maximum_transport_mass_error = max(
            maximum_transport_mass_error, transport_mass_error
        )
        oxygen, egf, environment_stats = environment_solver.solve(
            density, oxygen, egf, numerical
        )
        maximum_environment_iterations = max(
            maximum_environment_iterations,
            environment_stats["oxygen_newton_iterations"],
            environment_stats["egf_newton_iterations"],
        )
        time += dt
        if abs(time - next_record_time) <= 2.0e-10:
            time = next_record_time
        step_count += 1
        minimum_dt = min(minimum_dt, dt)
        maximum_dt_used = max(maximum_dt_used, dt)
        record_current_state()

        if time + 1.0e-10 >= next_progress_time:
            latest = diagnostics[-1] if diagnostics else {}
            print(
                f"sigma={sigma_max:.3g} t={time:6.2f} steps={step_count:6d} "
                f"maxN={float(np.max(density)):.4f} "
                f"A_tp={float(latest.get('trough_peak_amplitude', 0.0)):.3e}",
                flush=True,
            )
            next_progress_time += 5.0

    snapshot_arrays = {
        "time": np.asarray(snapshots["time"], dtype=float),
        "density": np.asarray(snapshots["density"], dtype=float),
        "oxygen": np.asarray(snapshots["oxygen"], dtype=float),
        "egf": np.asarray(snapshots["egf"], dtype=float),
        "sigma": np.asarray(snapshots["sigma"], dtype=float),
        "velocity": np.asarray(snapshots["velocity"], dtype=float),
    }
    run_stats = {
        "sigma_max": sigma_max,
        "boundary_condition": boundary_condition,
        "step_count": step_count,
        "minimum_dt": minimum_dt,
        "maximum_dt": maximum_dt_used,
        "maximum_environment_newton_iterations": maximum_environment_iterations,
        "maximum_transport_mass_error": maximum_transport_mass_error,
        "transition_time": transition_time,
        "strict_transition_time": strict_transition_time,
    }
    return {
        "grid": grid,
        "snapshots": snapshot_arrays,
        "diagnostics": diagnostics,
        "run_stats": run_stats,
    }
