"""Parameters for the simulations reported in Sections 3.2 and 3.3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .radial import ModelParameters

# Main one-parameter path used in the convergence figures:
# lambda = tau_environment = epsilon_phenotype.
MAIN_LAMBDA_VALUES: Final[tuple[float, ...]] = (1.0, 0.2, 0.05, 0.01, 0.001)


# Biological, spatial and boundary parameters shared by both models.
#
# The radial resolution is set separately for each experiment.
MODEL_PARAMETER_VALUES: Final[dict[str, float | int | str]] = {
    # Radial geometry and environmental boundary data.
    "radius": 10.0,  # R: outer radius of the radial domain 0 <= r <= R.
    "oxygen_boundary": 10.0,  # Dirichlet value u(t,R).
    "egf_boundary": 6.0,  # Inactive for the chosen no-flux EGF condition.
    "oxygen_boundary_type": "dirichlet",  # u(t,R)=oxygen_boundary.
    "egf_boundary_type": "no_flux",  # v_r(t,R)=0.
    "oxygen_robin_transfer": 1.0,  # Inactive unless Robin oxygen BC is used.
    "egf_robin_transfer": 1.0,  # Inactive unless Robin EGF BC is used.
    "egf_initial": 5.0,  # Newton seed selecting the positive EGF branch.

    # Intracellular HIF--EGF--EGFR relaxation dynamics.
    "k1": 5.0,  # Basal production in U_x = k1-(k2+f1(u))*x.
    "k2": 1.0,  # Basal phenotype decay/inactivation rate.
    "k3": 1.0,  # EGF-driven activation in U_z = k3*v-d2*z.
    "d1": 1.0,  # Extracellular EGF linear degradation coefficient.
    "d2": 0.2,  # Activated-EGFR phenotype decay coefficient.

    # Hill response f1(u): oxygen-dependent HIF inactivation.
    "f1_max": 25.0,  # Maximum of f1.
    "f1_half": 5.0,  # Half-saturation oxygen concentration of f1.
    # Hill response f2(z): activated-EGFR feedback on phenotype y.
    "f2_max": 4.0,  # Maximum of f2.
    "f2_half": 4.0,  # Half-saturation phenotype level of f2.
    # Hill response f3(y): phenotype-dependent EGF production.
    "f3_max": 8.0,  # Maximum of f3.
    "f3_half": 2.0,  # Half-saturation phenotype level of f3.
    "signalling_hill_power": 2,  # Common Hill exponent for f1, f2 and f3.

    # Cell proliferation and crowding.
    "phi": 0.8,  # Maximum coefficient in anaerobic growth h1(x).
    "hbar_max": 3.0,  # Maximum oxygen-dependent aerobic rate hbar(u).
    "hbar_half": 5.0,  # Half-saturation oxygen concentration of hbar.
    "hbar_hill_power": 2,  # Hill exponent used in hbar(u).
    "kappa": 1.5,  # Logistic crowding/death coefficient in -kappa*N.

    # Radial diffusion and environmental reaction coefficients.
    "alpha1": 1.5e-3,  # Cell random-motility diffusivity.
    "alpha2": 1.0,  # Oxygen diffusivity.
    "alpha3": 1.0,  # EGF diffusivity.
    "xi1": 0.25,  # Linear oxygen loss coefficient.
    "eta1": 4.0,  # Strength of cell-mediated oxygen consumption.
    "eta2": 2.0,  # Strength of phenotype-dependent EGF production.
    "eta3": 1.0,  # Strength of cell-mediated EGF uptake.
    "oxygen_source": 0.0,  # q=0: oxygen is supplied only at r=R.

    # Phenotype-regulated aerotactic sensitivity sigma(z).
    "sigma_half": 6.0,  # Half-saturation activated-EGFR level.
    "sigma_hill_power": 8,  # Steep Hill exponent (smooth switch).
    "sigma_max": 0.75,  # Saturated aerotactic sensitivity.

    "initial_plateau": 8.0 / 15.0,  # Central plateau of N_0.
    "initial_radius": 5.0,  # Initial transition location r_0.
    "initial_width": 0.2,  # Width in tanh((r-r_0)/width).

    # Cell boundary condition at r=R.
    "cell_boundary_condition": "outflow",  # Outflow allowed; inflow blocked.
}


# Phenotype box and quadrature resolution used in the reported run.
PHENOTYPE_GRID_VALUES: Final[dict[str, object]] = {
    "nx": 12,  # Number of x cells.
    "ny": 12,  # Number of y cells.
    "nz": 40,  # Number of z cells.
    "x_min": 0.0,  # Lower x boundary.
    "x_max": 5.0,  # Upper x boundary.
    "y_min": 0.0,  # Lower y boundary.
    "y_max": 5.0,  # Upper y boundary.
    "z_min": 0.0,  # Lower z boundary.
    "z_max": 30.0,  # Upper z boundary.
    # Nominal initial covariance is lambda*diag(1,1,25), hence these scales.
    "initial_std_scales": (1.0, 1.0, 5.0),
}


RADIAL_CELLS: Final[int] = 64


def model_parameter_values(n_radial: int = RADIAL_CELLS) -> dict[str, object]:
    """Return a fresh parameter mapping for ``FullModelParameters``."""
    if n_radial < 2:
        raise ValueError("The radial finite-volume grid needs at least two cells")
    return {**MODEL_PARAMETER_VALUES, "n_cells": int(n_radial)}


@dataclass(frozen=True)
class FullModelParameters(ModelParameters):
    """Biological and radial parameters for the full model.

    The inherited fields (``k1``, ``alpha1``, ``eta1``, etc.) are documented
    with their numerical values in ``MODEL_PARAMETER_VALUES``.  The two
    fields below are specific to the phenotype-resolved solver.
    """

    sigma_max: float = 0.75  # Saturated value of sigma(z).
    cell_boundary_condition: str = "outflow"  # Outer radial cell-flux rule.

    @classmethod
    def calibrated(cls, n_cells: int = RADIAL_CELLS) -> "FullModelParameters":
        """Return exactly the parameter set used in the reported experiment."""
        return cls(**model_parameter_values(n_cells))
