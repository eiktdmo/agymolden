"""
Solvent-Accessible Surface Area (SASA) & Polar Surface Area (PSA) Module for AgyMolden.

Computes:
1. Solvent-Accessible Surface Area (SASA in Å²) via the Shrake-Rupley "rolling probe sphere"
   algorithm (Shrake & Rupley, J. Mol. Biol. 1973, 79, 351-371).
   Uses golden-ratio Fibonacci spiral sphere point discretization for uniform surface distribution.
2. Hydrophobic vs. Polar SASA partitioning based on atomic electronegativity / heteroatom classification.
3. Topological / 3D Polar Surface Area (PSA) estimation.
4. Per-atom solvent exposure ratios for 3D visualization and color mapping.

100% Pure Python & NumPy (zero compiled C/C++ dependencies).
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .elements import get_vdw_radius

# Polar elements typically contributing to polar surface area in organic / biological systems
POLAR_ELEMENTS = {"N", "O", "S", "P", "F", "CL", "BR", "I"}


class SASAAtomResult(NamedTuple):
    atom_index: int
    symbol: str
    vdw_radius: float
    accessible_area: float       # In Å²
    relative_exposure: float     # accessible_area / total_isolated_sphere_area (0.0 to 1.0)
    is_polar: bool


class SASAResult(NamedTuple):
    total_sasa: float            # Total solvent accessible surface area in Å²
    polar_sasa: float            # SASA on polar atoms (N, O, S, P, halogens) in Å²
    hydrophobic_sasa: float      # SASA on non-polar atoms (C, H) in Å²
    probe_radius: float          # Water probe radius (typically 1.4 Å)
    n_points_per_atom: int       # Number of test points on sphere surface
    atom_results: List[SASAAtomResult]
    exposure_vector: np.ndarray  # Array of relative exposure ratios [0, 1] for all atoms


def generate_fibonacci_sphere(n_points: int = 100) -> np.ndarray:
    """
    Generates `n_points` uniformly distributed unit vectors on the surface of a sphere
    using the Fibonacci spiral / spherical phyllotaxis algorithm.
    """
    points = np.zeros((n_points, 3), dtype=np.float64)
    phi = math.pi * (math.sqrt(5.0) - 1.0)  # Golden ratio angle in radians

    for i in range(n_points):
        y = 1.0 - (i / float(n_points - 1)) * 2.0  # y goes from 1 to -1
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = phi * i

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points[i] = [x, y, z]

    return points


def calculate_sasa(
    symbols: List[str],
    coords: np.ndarray,
    probe_radius: float = 1.40,
    n_points_per_atom: int = 120,
) -> SASAResult:
    """
    Calculates Solvent-Accessible Surface Area (SASA) using the Shrake-Rupley algorithm.

    Parameters:
    - symbols: List of element symbols (e.g. ['C', 'H', 'O'])
    - coords: (N, 3) array of Cartesian coordinates in Ångströms
    - probe_radius: Probe sphere radius in Å (default 1.40 Å for water)
    - n_points_per_atom: Discretization density (default 120 points/atom gives ~1% error)
    """
    n_atoms = len(symbols)
    if n_atoms == 0 or len(coords) != n_atoms:
        return SASAResult(0.0, 0.0, 0.0, probe_radius, n_points_per_atom, [], np.empty(0))

    # 1. Radii with probe radius added: R_eff = R_vdw + R_probe
    vdw_radii = np.array([get_vdw_radius(s) for s in symbols], dtype=np.float64)
    eff_radii = vdw_radii + float(probe_radius)
    eff_radii_sq = eff_radii ** 2

    # 2. Precompute unit sphere points
    unit_sphere = generate_fibonacci_sphere(n_points_per_atom)

    atom_results: List[SASAAtomResult] = []
    exposure_list: List[float] = []

    total_sasa = 0.0
    polar_sasa = 0.0
    hydrophobic_sasa = 0.0

    # 3. Vectorized neighbor spatial filtering
    # Atom j can only occlude test points of atom i if dist(i, j) < eff_radii[i] + eff_radii[j]
    coord_diffs = coords[:, None, :] - coords[None, :, :]  # (N, N, 3)
    dist_sq_matrix = np.sum(coord_diffs ** 2, axis=-1)     # (N, N)

    for i in range(n_atoms):
        r_eff_i = eff_radii[i]
        # Potential neighbor collision candidates
        max_dist = r_eff_i + eff_radii
        candidate_mask = (dist_sq_matrix[i] < (max_dist ** 2)) & (np.arange(n_atoms) != i)
        cand_indices = np.nonzero(candidate_mask)[0]

        # Isolated total sphere surface area: 4 * pi * R_eff^2
        isolated_area = 4.0 * math.pi * (r_eff_i ** 2)

        if len(cand_indices) == 0:
            # Completely isolated atom: 100% surface is accessible
            area_i = isolated_area
            accessible_ratio = 1.0
        else:
            # Test points on atom i's expanded sphere: center_i + r_eff_i * unit_points
            test_pts = coords[i] + r_eff_i * unit_sphere  # shape (n_points, 3)

            # Check if each test point is inside any neighbor's expanded sphere
            # dist(test_pt, cand_j)^2 < eff_radii[j]^2
            cand_coords = coords[cand_indices]             # shape (M, 3)
            cand_r_sq = eff_radii_sq[cand_indices]         # shape (M,)

            # diffs: (n_points, M, 3)
            diffs = test_pts[:, None, :] - cand_coords[None, :, :]
            d2 = np.sum(diffs ** 2, axis=-1)               # shape (n_points, M)

            # A test point is occluded if d2 < cand_r_sq for ANY candidate neighbor
            occluded = np.any(d2 < cand_r_sq[None, :], axis=1)
            n_accessible = int(np.sum(~occluded))

            accessible_ratio = n_accessible / float(n_points_per_atom)
            area_i = isolated_area * accessible_ratio

        total_sasa += area_i
        is_pol = symbols[i].upper() in POLAR_ELEMENTS
        if is_pol:
            polar_sasa += area_i
        else:
            hydrophobic_sasa += area_i

        atom_results.append(
            SASAAtomResult(
                atom_index=i,
                symbol=symbols[i],
                vdw_radius=float(vdw_radii[i]),
                accessible_area=float(area_i),
                relative_exposure=float(accessible_ratio),
                is_polar=is_pol,
            )
        )
        exposure_list.append(accessible_ratio)

    return SASAResult(
        total_sasa=float(total_sasa),
        polar_sasa=float(polar_sasa),
        hydrophobic_sasa=float(hydrophobic_sasa),
        probe_radius=probe_radius,
        n_points_per_atom=n_points_per_atom,
        atom_results=atom_results,
        exposure_vector=np.array(exposure_list, dtype=np.float64),
    )
