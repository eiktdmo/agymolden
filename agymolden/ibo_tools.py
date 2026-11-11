"""
Intrinsic Bond Orbitals (IBO) and Intrinsic Atomic Orbitals (IAO) Localization Engine for AgyMolden.

Provides:
1. PySCF IAO/IBO orbital localization (Knizia, J. Chem. Theory Comput. 2013, 9, 4834-4843).
2. Automated classification of localized orbitals into:
   - Core pairs (1-center localized on heavy atoms)
   - Sigma / Pi covalent bonds (2-center or 3-center shared electrons)
   - Lone pairs (1-center localized valence electron pairs)
3. Direct conversion of localized orbital coefficients into Molden volumetric grid representation
   for real-time 3D isosurface rendering in AgyMolden's OpenGL / Vispy viewer.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


class LocalizedOrbital(NamedTuple):
    index: int
    orbital_type: str            # 'core', 'bond', 'lone_pair', 'delocalized'
    label: str                   # e.g. "σ(O1-H2)", "Lone Pair (O1)", "Core 1s (O1)"
    major_atoms: List[Tuple[str, int, float]]  # [(element, atom_idx, percentage_population)]
    coeff: np.ndarray            # Coefficient vector in AO basis


class IBOResult(NamedTuple):
    orbitals: List[LocalizedOrbital]
    ibo_coeff_matrix: np.ndarray  # Shape (n_ao, n_localized_mo)
    atom_labels: List[str]
    atom_coords_angstrom: np.ndarray


def compute_ibos(molden_path: str, exponent: int = 4) -> IBOResult:
    """Localizes occupied molecular orbitals from a Molden file using PySCF IAO/IBO."""
    try:
        from pyscf.lo import iao, ibo
        from .volumetric import _load_pyscf_molden, BOHR_TO_ANGSTROM
    except ImportError as exc:
        raise RuntimeError(
            "PySCF is required for Intrinsic Bond Orbital (IBO) calculations. "
            "Please ensure pyscf is installed."
        ) from exc

    res = _load_pyscf_molden(molden_path)
    if not res:
        raise ValueError(f"Could not parse wavefunction from {molden_path}")

    mol = res[0]
    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    # Handle unrestricted (spin-polarized) or restricted
    if mo_coeff.ndim == 3:
        # Take alpha spin for localization
        mo_coeff = mo_coeff[0]
        mo_occ = mo_occ[0]

    # Filter occupied orbitals (occupation > 0.5)
    occ_idx = np.where(mo_occ > 0.5)[0]
    if len(occ_idx) == 0:
        # Fallback: assume half the orbitals are occupied
        n_occ = max(1, mol.nelectron // 2)
        occ_idx = np.arange(n_occ)

    c_occ = mo_coeff[:, occ_idx]
    s = mol.intor("int1e_ovlp")

    # 1. Compute Intrinsic Atomic Orbitals (IAO)
    a = iao.iao(mol, c_occ)
    a = iao.vec_lowdin(a, s)

    # 2. Localize to Intrinsic Bond Orbitals (IBO) using exponent P (typically 4)
    ibos = ibo.ibo(mol, c_occ, iaos=a, exponent=exponent)

    # 3. Analyze Mulliken populations of each IBO on each atom
    atom_labels = [mol.atom_symbol(i) for i in range(mol.natm)]
    atom_coords = mol.atom_coords() * BOHR_TO_ANGSTROM
    ao_slices = mol.aoslice_by_atom()

    localized_list: List[LocalizedOrbital] = []
    for i in range(ibos.shape[1]):
        c_i = ibos[:, i]
        pop_bf = c_i * (s @ c_i)  # AO population vector

        atom_pops: List[Tuple[str, int, float]] = []
        for at_i in range(mol.natm):
            sym = atom_labels[at_i]
            p_val = float(np.sum(pop_bf[ao_slices[at_i][2] : ao_slices[at_i][3]]))
            atom_pops.append((sym, at_i, p_val))

        # Sort descending by population share
        atom_pops.sort(key=lambda x: x[2], reverse=True)
        # Select atoms with > 12% population share
        active_atoms = [(sym, idx, p) for sym, idx, p in atom_pops if p >= 0.12]

        if not active_atoms:
            active_atoms = [(atom_pops[0][0], atom_pops[0][1], atom_pops[0][2])]

        # Lewis classification
        if len(active_atoms) == 1:
            sym, idx, pop = active_atoms[0]
            # Core vs Lone Pair heuristic: check if atom has core and orbital is tightly localized
            # For 1st row (C, N, O, F), 1s core has ~100% pop and very small spatial extent
            # The earliest single-center orbitals on heavy atoms are core
            if sym not in ("H", "He") and pop > 0.95 and i < mol.natm:
                orb_type = "core"
                label = f"Core 1s ({sym}{idx + 1})"
            else:
                orb_type = "lone_pair"
                label = f"Lone Pair ({sym}{idx + 1})"
        elif len(active_atoms) == 2:
            orb_type = "bond"
            a1, a2 = active_atoms[0], active_atoms[1]
            label = f"Bond ({a1[0]}{a1[1] + 1} - {a2[0]}{a2[1] + 1})"
        else:
            orb_type = "delocalized"
            bonded_str = "-".join(f"{sym}{idx+1}" for sym, idx, _ in active_atoms[:3])
            label = f"Multi-center ({bonded_str})"

        localized_list.append(
            LocalizedOrbital(
                index=i,
                orbital_type=orb_type,
                label=label,
                major_atoms=active_atoms,
                coeff=c_i,
            )
        )

    return IBOResult(
        orbitals=localized_list,
        ibo_coeff_matrix=ibos,
        atom_labels=atom_labels,
        atom_coords_angstrom=atom_coords,
    )
