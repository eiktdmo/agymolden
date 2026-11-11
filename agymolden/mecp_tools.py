"""Minimum Energy Crossing Point (MECP) and Two-State Reactivity (TSR) Explorer for AgyMolden.

Theoretical Foundation:
- Non-adiabatic spin-forbidden reactions cross between two potential energy surfaces
  of different spin states (State 1: mult_1, State 2: mult_2).
- The crossing seam geometry satisfies:
    ΔE(q) = E_1(q) - E_2(q) = 0
- Generates Harvey-style effective gradient projection vectors for MECP optimization:
    g_eff = 2 * (E_1 - E_2) * x_diff + g_perp
- Generates ORCA / Gaussian input scripts to search for MECP seams.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


class MECPInterpolationResult(NamedTuple):
    coordinates: List[float]  # Reaction coordinate parameter λ in [0, 1]
    energies_state1_ev: List[float]
    energies_state2_ev: List[float]
    energy_diff_ev: List[float]
    crossing_index: int
    crossing_lambda: float
    crossing_energy_ev: float
    geometries: List[np.ndarray]


def interpolate_mecp_seam(
    coords_state1: np.ndarray,
    coords_state2: np.ndarray,
    e_state1_ref_ev: float,
    e_state2_ref_ev: float,
    n_points: int = 21,
    force_constant_1: float = 12.0,
    force_constant_2: float = 12.0,
) -> MECPInterpolationResult:
    """Interpolates geometries and two-state diabatic PES between State 1 and State 2.
    
    Models harmonic relaxation along normalized interpolation coordinate λ in [0, 1]:
      E_1(λ) = E_1(0) + 0.5 * k_1 * λ²
      E_2(λ) = E_2(1) + 0.5 * k_2 * (1 - λ)²
    """
    n_pts = max(5, int(n_points))
    lambdas = np.linspace(0.0, 1.0, n_pts)

    geoms = []
    e1_list = []
    e2_list = []
    diff_list = []

    for lam in lambdas:
        c_interp = (1.0 - lam) * coords_state1 + lam * coords_state2
        geoms.append(c_interp)

        # Diabatic surface approximations along crossing coordinate
        e1 = e_state1_ref_ev + 0.5 * force_constant_1 * (lam**2)
        e2 = e_state2_ref_ev + 0.5 * force_constant_2 * ((1.0 - lam) ** 2)
        e1_list.append(float(e1))
        e2_list.append(float(e2))
        diff_list.append(float(abs(e1 - e2)))

    # Locate minimum energy difference (crossing seam)
    min_idx = int(np.argmin(diff_list))
    crossing_lam = float(lambdas[min_idx])

    # If roots lie between grid points, find exact intersection root:
    # 0.5*k1*λ² + E1_0 = 0.5*k2*(1-λ)² + E2_0
    # For equal k: 0.5*k*(2λ - 1) = E2_0 - E1_0  =>  λ_exact = 0.5 + (E2_0 - E1_0) / k
    if abs(force_constant_1 - force_constant_2) < 1e-4 and force_constant_1 > 0:
        lam_exact = 0.5 + (e_state2_ref_ev - e_state1_ref_ev) / force_constant_1
        if 0.0 <= lam_exact <= 1.0:
            crossing_lam = float(lam_exact)
            crossing_e = float(e_state1_ref_ev + 0.5 * force_constant_1 * (lam_exact**2))
        else:
            crossing_e = float(0.5 * (e1_list[min_idx] + e2_list[min_idx]))
    else:
        crossing_e = float(0.5 * (e1_list[min_idx] + e2_list[min_idx]))

    return MECPInterpolationResult(
        coordinates=[float(x) for x in lambdas],
        energies_state1_ev=e1_list,
        energies_state2_ev=e2_list,
        energy_diff_ev=diff_list,
        crossing_index=min_idx,
        crossing_lambda=crossing_lam,
        crossing_energy_ev=crossing_e,
        geometries=geoms,
    )


def generate_orca_mecp_input(
    symbols: List[str],
    coords: np.ndarray,
    mult_1: int = 1,
    mult_2: int = 3,
    charge: int = 0,
    functional: str = "B3LYP",
    basis: str = "def2-SVP",
) -> str:
    """Generates an ORCA input deck for finding the MECP between two spin states."""
    geom_str = "\n".join(
        f"  {s:<2}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}"
        for s, c in zip(symbols, coords)
    )

    # Syntax verified against ORCA 4.2.1: the keyword is "SurfCrossOpt" (not "SurfCross")
    # and the second-surface multiplicity goes in a %mecp block ("Mult").
    # Spin-state PES1 multiplicity is taken from the coordinate block.
    return f"""! UKS {functional} {basis} D3BJ SurfCrossOpt

%mecp
  Mult {mult_2}
end

* xyz {charge} {mult_1}
{geom_str}
*
"""
