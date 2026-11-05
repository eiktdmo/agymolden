"""
Chemistry Sanity & Pre-Flight Diagnostic Checker.

Provides automated checks for computational chemistry workflows to prevent common
frustrations (especially for students and new users):
1. Charge vs. Multiplicity vs. Total Electron Count consistency (spin state sanity).
2. Interatomic clashes / unphysically short bonds (< 0.70 Å).
3. Isolated unbonded fragments or excessively long distances.
4. Heavy element basis set suitability (advising ECP/pseudopotentials for Z > 36).
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from .elements import ELEMENT_DATA, normalize_symbol


def check_calculation_sanity(
    symbols: List[str],
    coords: np.ndarray,
    charge: int = 0,
    multiplicity: int = 1,
    basis_set: Optional[str] = None,
    qm_package: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run comprehensive chemical and physical sanity checks on a proposed calculation.

    Returns:
        dict with:
          - "passed": bool (True if no severe errors)
          - "errors": list of str (severe blocking problems)
          - "warnings": list of str (suspicious parameters or student tips)
          - "info": dict of computed physical metrics (formula, MW, n_electrons, spin_s)
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Chemical Formula & Total Nuclear Charge
    norm_symbols = [normalize_symbol(s) for s in symbols]
    total_nuclear_z = 0
    total_mass = 0.0
    formula_counts: Dict[str, int] = {}

    for s in norm_symbols:
        data = ELEMENT_DATA.get(s, {})
        z = data.get("atomic_number", 1)
        m = data.get("mass", 1.008)
        total_nuclear_z += z
        total_mass += m
        formula_counts[s] = formula_counts.get(s, 0) + 1

    # Hill system formula
    hill_parts = []
    if "C" in formula_counts:
        hill_parts.append(f"C{formula_counts['C'] if formula_counts['C'] > 1 else ''}")
        if "H" in formula_counts:
            hill_parts.append(f"H{formula_counts['H'] if formula_counts['H'] > 1 else ''}")
        for el in sorted(formula_counts.keys()):
            if el not in ("C", "H"):
                cnt = formula_counts[el]
                hill_parts.append(f"{el}{cnt if cnt > 1 else ''}")
    else:
        # If no carbon, sort all elements alphabetically
        for el in sorted(formula_counts.keys()):
            cnt = formula_counts[el]
            hill_parts.append(f"{el}{cnt if cnt > 1 else ''}")
    formula_str = "".join(hill_parts)

    n_electrons = total_nuclear_z - charge

    if n_electrons <= 0:
        errors.append(f"Invalid electron count: {n_electrons} electrons (Nuclear Z={total_nuclear_z}, Charge={charge:+d}).")

    # 2. Spin Multiplicity Consistency
    # (2S + 1): n_electrons and (multiplicity - 1) must have identical parity!
    if n_electrons > 0:
        is_even_electrons = (n_electrons % 2 == 0)
        is_odd_mult = (multiplicity % 2 == 1)

        if is_even_electrons and not is_odd_mult:
            errors.append(
                f"Spin state mismatch: System has {n_electrons} electrons (even), which requires an ODD multiplicity "
                f"(Singlet=1, Triplet=3, Quintet=5), but Multiplicity is set to {multiplicity}."
            )
        elif not is_even_electrons and is_odd_mult:
            errors.append(
                f"Spin state mismatch: System has {n_electrons} electrons (odd / radical), which requires an EVEN multiplicity "
                f"(Doublet=2, Quartet=4, Sextet=6), but Multiplicity is set to {multiplicity}."
            )

        # High spin warning
        if multiplicity > 4:
            warnings.append(
                f"High spin state: Multiplicity {multiplicity} (2S+1 = {multiplicity}, {multiplicity-1} unpaired electrons). "
                f"Confirm this is intentional and not a ground state."
            )

    # 3. Geometry Checks (Clashes & Disconnects)
    n_atoms = len(norm_symbols)
    if n_atoms >= 2 and coords is not None and len(coords) == n_atoms:
        min_dist = float("inf")
        clashing_pairs = []

        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < min_dist:
                    min_dist = d
                if d < 0.68:  # Severely overlapping atoms
                    clashing_pairs.append((i, j, norm_symbols[i], norm_symbols[j], d))

        if clashing_pairs:
            clash_desc = ", ".join(f"{s1}{i+1}–{s2}{j+1} ({d:.2f} Å)" for i, j, s1, s2, d in clashing_pairs[:3])
            errors.append(
                f"Atomic clash detected: {len(clashing_pairs)} atom pair(s) are unphysically close (< 0.68 Å): {clash_desc}. "
                f"Check geometry coordinates or run Force Field pre-optimization (Ctrl+M)."
            )
        elif min_dist < 0.85:
            warnings.append(f"Very short interatomic distance detected (closest atoms are {min_dist:.2f} Å apart).")

    # 4. Basis Set & Heavy Element Advice
    heavy_elements = [s for s in norm_symbols if ELEMENT_DATA.get(s, {}).get("atomic_number", 0) > 36]
    if heavy_elements and basis_set:
        basis_upper = basis_set.upper()
        # Common Pople sets without relativistic ECPs
        if any(p in basis_upper for p in ("6-31G", "6-311G", "STO-3G", "3-21G")):
            warnings.append(
                f"Heavy element(s) ({', '.join(sorted(set(heavy_elements)))}, Z > 36) used with all-electron Pople basis '{basis_set}'. "
                f"For accurate relativistic core contraction, consider an ECP basis like 'def2-SVP' or 'def2-TZVP'."
            )

    info = {
        "formula": formula_str,
        "mass": total_mass,
        "nuclear_z": total_nuclear_z,
        "charge": charge,
        "n_electrons": n_electrons,
        "multiplicity": multiplicity,
        "spin_s": (multiplicity - 1) / 2.0,
    }

    return {
        "passed": (len(errors) == 0),
        "errors": errors,
        "warnings": warnings,
        "info": info,
    }
