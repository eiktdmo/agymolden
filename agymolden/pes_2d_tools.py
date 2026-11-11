"""2D Relaxed Potential Energy Surface (PES) Scan Grid Builder for AgyMolden.

Generates 2D coordinate scan grids across two independent internal degrees of freedom:
- Coordinate 1: Bond distance (Å), Bond angle (deg), or Dihedral (deg)
- Coordinate 2: Bond distance (Å), Bond angle (deg), or Dihedral (deg)

Exports:
- ORCA 2D scan input decks (%geom Scan ... end end)
- Gaussian 2D ModRedundant scan input decks (B/A/D ... S N Step)
- PySCF 2D scan Python runner scripts
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple
import numpy as np


class ScanCoordinateDef(NamedTuple):
    coord_type: str  # "distance", "angle", "dihedral"
    atom_indices: Tuple[int, ...]  # 2, 3, or 4 atom indices (0-based)
    start_val: float
    end_val: float
    num_steps: int


def generate_orca_2d_scan_input(
    symbols: List[str],
    coords: np.ndarray,
    scan1: ScanCoordinateDef,
    scan2: ScanCoordinateDef,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "B3LYP",
    basis: str = "def2-SVP",
) -> str:
    """Generates an ORCA input deck for a 2D relaxed surface scan."""
    geom_str = "\n".join(
        f"  {s:<2}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}"
        for s, c in zip(symbols, coords)
    )

    def _orca_scan_line(s_def: ScanCoordinateDef) -> str:
        idx_str = " ".join(str(i) for i in s_def.atom_indices)
        if s_def.coord_type == "distance":
            prefix = "B"
        elif s_def.coord_type == "angle":
            prefix = "A"
        else:
            prefix = "D"
        return f"  {prefix} {idx_str} = {s_def.start_val:.3f}, {s_def.end_val:.3f}, {s_def.num_steps}"

    line1 = _orca_scan_line(scan1)
    line2 = _orca_scan_line(scan2)

    return f"""! {functional} {basis} Opt

%geom
Scan
{line1}
{line2}
  end
end

* xyz {charge} {multiplicity}
{geom_str}
*
"""


def generate_gaussian_2d_scan_input(
    symbols: List[str],
    coords: np.ndarray,
    scan1: ScanCoordinateDef,
    scan2: ScanCoordinateDef,
    charge: int = 0,
    multiplicity: int = 1,
    functional: str = "B3LYP",
    basis: str = "6-31G(d)",
) -> str:
    """Generates a Gaussian ModRedundant input deck for a 2D relaxed surface scan."""
    geom_str = "\n".join(
        f" {s:<2}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}"
        for s, c in zip(symbols, coords)
    )

    def _gau_line(s_def: ScanCoordinateDef) -> str:
        # 1-based indexing for Gaussian
        idx_str = " ".join(str(i + 1) for i in s_def.atom_indices)
        step_sz = (s_def.end_val - s_def.start_val) / max(1, s_def.num_steps)
        if s_def.coord_type == "distance":
            prefix = "B"
        elif s_def.coord_type == "angle":
            prefix = "A"
        else:
            prefix = "D"
        return f"{prefix} {idx_str} S {s_def.num_steps} {step_sz:.4f}"

    l1 = _gau_line(scan1)
    l2 = _gau_line(scan2)

    return f"""%nprocshared=4
%mem=8GB
#p {functional}/{basis} Opt=ModRedundant

AgyMolden 2D PES Scan Setup

{charge} {multiplicity}
{geom_str}

{l1}
{l2}

"""
