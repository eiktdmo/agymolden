"""
1D Dihedral Rotor Conformer Scanner & Frozen-Atom Constraint Utilities.

Supports both Rigid Rotor Scans (0 relaxation steps per torsion angle) and
Relaxed Potential Energy Surface (PES) Scans (optimizing all other degrees of freedom
at each step while holding the 4 dihedral atoms fixed at the target torsion angle).
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
import numpy as np

from .zmatrix import calculate_dihedral
from .smartslides import modify_dihedral
from .forcefield import optimize_geometry_force_field, detect_connectivity


@dataclass
class DihedralScanPoint:
    step_index: int
    dihedral_deg: float
    energy_kcal: float
    relative_energy_kcal: float
    coords: np.ndarray


@dataclass
class DihedralScanResult:
    atom_indices: Tuple[int, int, int, int]
    points: List[DihedralScanPoint]
    min_index: int
    max_index: int
    barrier_kcal: float


def run_dihedral_rotor_scan(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]],
    atom_indices: Tuple[int, int, int, int],
    n_steps: int = 24,
    step_deg: Optional[float] = None,
    relax_steps: int = 25,
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    frozen_indices: Optional[Set[int]] = None,
    method: str = "uff_mmff",
) -> DihedralScanResult:
    """
    Scan the dihedral angle defined by `atom_indices = (i1, i2, i3, i4)` over `n_steps`
    increments of `step_deg` (defaults to `360.0 / n_steps`).

    At each step:
      1. Rotates the substituent subgraph around `i2 -> i3` to the target dihedral angle.
      2. Evaluates the force-field or semiempirical energy (with optional constrained relaxation
         of `relax_steps` iterations where `(i1, i2, i3, i4)` and any user `frozen_indices` are pinned).
    """
    n = len(symbols)
    i1, i2, i3, i4 = atom_indices
    if n < 4 or len({i1, i2, i3, i4}) < 4 or not all(0 <= k < n for k in (i1, i2, i3, i4)):
        raise ValueError("Four distinct valid atom indices are required for a dihedral scan.")

    explicit_bonds = (
        list(bonds)
        if bonds is not None
        else detect_connectivity(symbols, coords, strict=False)
    )
    steps = max(2, int(n_steps))
    d_step = float(step_deg) if step_deg is not None else (360.0 / steps)

    base_frozen = set(frozen_indices or set()) | {i1, i2, i3, i4}
    curr_coords = np.array(coords, dtype=np.float64, copy=True)
    start_dih = calculate_dihedral(curr_coords[i1], curr_coords[i2], curr_coords[i3], curr_coords[i4])

    raw_points: List[Tuple[float, float, np.ndarray]] = []
    for s in range(steps):
        target_angle = ((start_dih + s * d_step + 180.0) % 360.0) - 180.0
        step_coords = modify_dihedral(
            symbols, curr_coords, explicit_bonds, i1, i2, i3, i4, target_angle, mode="group"
        )
        # Evaluate energy (and optionally relax unconstrained atoms)
        iters = max(0, int(relax_steps))
        opt_coords, e_val = optimize_geometry_force_field(
            symbols,
            step_coords,
            user_bonds=explicit_bonds,
            max_iter=iters,
            strict_bonds=True,
            user_bond_orders=user_bond_orders,
            method=method,
            frozen_indices=base_frozen,
        )
        curr_coords = opt_coords
        actual_dih = calculate_dihedral(
            curr_coords[i1], curr_coords[i2], curr_coords[i3], curr_coords[i4]
        )
        raw_points.append((actual_dih, float(e_val), np.copy(curr_coords)))

    energies = [p[1] for p in raw_points]
    min_e = min(energies) if energies else 0.0
    max_e = max(energies) if energies else 0.0
    min_idx = int(np.argmin(energies)) if energies else 0
    max_idx = int(np.argmax(energies)) if energies else 0

    points: List[DihedralScanPoint] = []
    for idx, (dih_val, e_val, pt_coords) in enumerate(raw_points):
        points.append(
            DihedralScanPoint(
                step_index=idx,
                dihedral_deg=dih_val,
                energy_kcal=e_val,
                relative_energy_kcal=e_val - min_e,
                coords=pt_coords,
            )
        )

    return DihedralScanResult(
        atom_indices=(i1, i2, i3, i4),
        points=points,
        min_index=min_idx,
        max_index=max_idx,
        barrier_kcal=max(0.0, max_e - min_e),
    )
