"""Continuum Solvation Models & Thermodynamic Free Energy Estimator.

Evaluates solvent effects using the SMD / CPCM / PCM framework:
- Dielectric constant (ε), refractive index (n), surface tension (γ)
- Electrostatic Born-Kirkwood-Onsager solvation energy: ΔG_el
- Non-electrostatic cavitation, dispersion, and solvent structure (CDS) term: ΔG_cds
- Total free energy of solvation: ΔG_solv = ΔG_el + ΔG_cds
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .elements import get_cov_radius


class SolventProperties(NamedTuple):
    name: str
    formula: str
    dielectric_constant: float  # ε
    refractive_index: float     # n
    surface_tension: float      # dyne/cm (cal/mol/Å²)
    molar_volume: float         # cm³/mol
    probe_radius: float         # Å (solvent molecular radius)


SOLVENT_DATABASE: Dict[str, SolventProperties] = {
    "water": SolventProperties(
        name="Water", formula="H2O", dielectric_constant=78.36, refractive_index=1.333,
        surface_tension=71.99, molar_volume=18.07, probe_radius=1.40
    ),
    "acetonitrile": SolventProperties(
        name="Acetonitrile", formula="CH3CN", dielectric_constant=35.69, refractive_index=1.344,
        surface_tension=29.29, molar_volume=52.87, probe_radius=2.18
    ),
    "dmso": SolventProperties(
        name="Dimethylsulfoxide (DMSO)", formula="(CH3)2SO", dielectric_constant=46.83, refractive_index=1.478,
        surface_tension=43.54, molar_volume=71.27, probe_radius=2.45
    ),
    "methanol": SolventProperties(
        name="Methanol", formula="CH3OH", dielectric_constant=32.61, refractive_index=1.328,
        surface_tension=22.60, molar_volume=40.73, probe_radius=2.00
    ),
    "ethanol": SolventProperties(
        name="Ethanol", formula="CH3CH2OH", dielectric_constant=24.85, refractive_index=1.361,
        surface_tension=22.10, molar_volume=58.68, probe_radius=2.25
    ),
    "dichloromethane": SolventProperties(
        name="Dichloromethane (DCM)", formula="CH2Cl2", dielectric_constant=8.93, refractive_index=1.424,
        surface_tension=28.12, molar_volume=64.50, probe_radius=2.34
    ),
    "chloroform": SolventProperties(
        name="Chloroform", formula="CHCl3", dielectric_constant=4.71, refractive_index=1.446,
        surface_tension=27.16, molar_volume=80.67, probe_radius=2.52
    ),
    "thf": SolventProperties(
        name="Tetrahydrofuran (THF)", formula="C4H8O", dielectric_constant=7.43, refractive_index=1.407,
        surface_tension=26.40, molar_volume=81.90, probe_radius=2.52
    ),
    "toluene": SolventProperties(
        name="Toluene", formula="C7H8", dielectric_constant=2.37, refractive_index=1.496,
        surface_tension=28.52, molar_volume=106.85, probe_radius=2.82
    ),
    "benzene": SolventProperties(
        name="Benzene", formula="C6H6", dielectric_constant=2.27, refractive_index=1.501,
        surface_tension=28.88, molar_volume=89.41, probe_radius=2.60
    ),
    "hexane": SolventProperties(
        name="n-Hexane", formula="C6H14", dielectric_constant=1.88, refractive_index=1.375,
        surface_tension=18.43, molar_volume=131.61, probe_radius=2.99
    ),
}


class SolvationResult(NamedTuple):
    solvent_name: str
    epsilon: float
    sasa_angstrom2: float
    volume_angstrom3: float
    delta_g_el_kcal: float
    delta_g_cds_kcal: float
    delta_g_solv_kcal: float
    delta_g_solv_kj: float


def compute_approximate_sasa(
    symbols: List[str],
    coords: np.ndarray,
    probe_radius: float = 1.40,
    n_points: int = 64,
) -> Tuple[float, float, np.ndarray]:
    """
    Computes Solvent Accessible Surface Area (SASA) and Molecular Cavity Volume
    using the Shrake-Rupley numerical sphere integration algorithm.
    Returns: (total_sasa, total_volume, atomic_sasas)
    """
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return 0.0, 0.0, np.array([])

    radii = np.array([get_cov_radius(s) + probe_radius for s in symbols], dtype=np.float64)

    # Golden spiral distribution on sphere surface
    indices = np.arange(0, n_points, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n_points)
    theta = math.pi * (1.0 + 5.0**0.5) * indices
    sphere_pts = np.vstack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi)
    ]).T

    atomic_sasas = np.zeros(n, dtype=np.float64)

    for i in range(n):
        r_i = radii[i]
        c_i = coords[i]
        test_pts = c_i + r_i * sphere_pts
        n_accessible = 0

        # Check occlusion against neighbor atoms
        for pt in test_pts:
            dists_sq = np.sum((coords - pt)**2, axis=1)
            # Pt is accessible if it's not inside any other atom's expanded radius
            # (ignoring self)
            dists_sq[i] = 1e9
            if np.all(dists_sq >= radii**2):
                n_accessible += 1

        fraction = n_accessible / float(n_points)
        atomic_sasas[i] = 4.0 * math.pi * (r_i**2) * fraction

    total_sasa = float(np.sum(atomic_sasas))
    # Approximate molecular cavity volume: 4/3 π R_eff^3
    r_eff = (total_sasa / (4.0 * math.pi))**0.5 if total_sasa > 0 else 1.0
    cavity_vol = (4.0 / 3.0) * math.pi * (r_eff**3)

    return total_sasa, cavity_vol, atomic_sasas


def estimate_solvation_energy(
    symbols: List[str],
    coords: np.ndarray,
    solvent_key: str = "water",
    partial_charges: Optional[np.ndarray] = None,
    dipole_debye: Optional[float] = None,
) -> SolvationResult:
    """
    Estimates free energy of solvation ΔG_solv = ΔG_el + ΔG_cds.
    - ΔG_el: Born-Kirkwood-Onsager reaction field dielectric response.
    - ΔG_cds: Cavitation, dispersion, and solvent structural rearrangement (SMD style).
    """
    solv = SOLVENT_DATABASE.get(solvent_key.lower(), SOLVENT_DATABASE["water"])
    eps = solv.dielectric_constant

    sasa, vol, at_sasa = compute_approximate_sasa(symbols, coords, probe_radius=solv.probe_radius)

    # 1. Non-electrostatic CDS term (Cavitation + Dispersion):
    # Proportional to solvent surface tension and accessible area
    # SMD empirical scaling factor ~ 0.0055 kcal/(mol * Å^2 * dyne/cm)
    gamma_eff = solv.surface_tension * 0.0055
    delta_g_cds = (sasa * gamma_eff) * 0.038

    # 2. Electrostatic reaction field term (Born / Onsager model)
    # ΔG_el = - (1/2) * (1 - 1/ε) * (sum(q_i^2 / r_i) + interaction terms)
    # Using dipole moment if provided or partial charges
    r_cavity = max(1.2, (3.0 * vol / (4.0 * math.pi))**(1.0 / 3.0))

    if dipole_debye is not None and dipole_debye > 0.01:
        # Onsager dipole reaction field:
        # ΔG_el = - [ (ε - 1) / (2ε + 1) ] * (μ² / a³)
        # μ in Debye -> convert to kcal/mol: 1 D² / Å³ ≈ 14.39 kcal/mol
        f_eps = (eps - 1.0) / (2.0 * eps + 1.0)
        delta_g_el = - f_eps * ((dipole_debye**2) / (r_cavity**3)) * 14.39
    elif partial_charges is not None and len(partial_charges) == len(symbols):
        # Generalized Born summation:
        f_eps = 1.0 - (1.0 / eps)
        # Born self-energy sum
        q_arr = np.array(partial_charges, dtype=np.float64)
        c_radii = np.array([get_cov_radius(s) for s in symbols], dtype=np.float64)
        born_sum = np.sum((q_arr**2) / c_radii)
        # 332.0637 kcal*Å/(mol*e^2)
        delta_g_el = - 0.5 * 332.0637 * f_eps * (born_sum / max(1.0, float(len(symbols))**0.33)) * 0.25
    else:
        # Neutral organic default estimation from cavity size
        f_eps = (eps - 1.0) / (2.0 * eps + 1.0)
        delta_g_el = - f_eps * 4.5

    delta_g_solv = delta_g_el + delta_g_cds
    delta_g_solv_kj = delta_g_solv * 4.184

    return SolvationResult(
        solvent_name=solv.name,
        epsilon=eps,
        sasa_angstrom2=sasa,
        volume_angstrom3=vol,
        delta_g_el_kcal=delta_g_el,
        delta_g_cds_kcal=delta_g_cds,
        delta_g_solv_kcal=delta_g_solv,
        delta_g_solv_kj=delta_g_solv_kj,
    )
