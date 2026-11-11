"""Open-Shell Spin Contamination, Yamaguchi Approximate Spin Projection,
and Heisenberg Exchange Coupling (J) Module for AgyMolden.

Theoretical Foundation:
- Unrestricted DFT/HF (UKS/UHF) yields broken-symmetry states contaminated by higher multiplets:
    <S^2>_exact = S*(S+1)
    % Spin Contamination = ((<S^2>_calc - <S^2>_exact) / <S^2>_exact) * 100%
- Yamaguchi Approximate Spin Projection (AP) corrects the singlet broken-symmetry energy:
    E_AP(singlet) = (E_BS * <S^2>_HS - E_HS * <S^2>_BS) / (<S^2>_HS - <S^2>_BS)
- Heisenberg Exchange Coupling Constant J (H = -2J * S_A * S_B):
    Yamaguchi formal:
        J_Yamaguchi = (E_BS - E_HS) / (<S^2>_HS - <S^2>_BS)
    Noodleman weak-coupling limit:
        J_Noodleman = (E_BS - E_HS) / (S_max^2)
"""

from __future__ import annotations

import math
from typing import Dict, NamedTuple, Optional
import numpy as np


class SpinDiagnosticsResult(NamedTuple):
    spin_s: float
    multiplicity: int
    s2_exact: float
    s2_calc: float
    contamination_percent: float
    is_contaminated: bool
    unpaired_electrons: int
    status_summary: str


class YamaguchiProjectionResult(NamedTuple):
    e_bs_ha: float
    e_hs_ha: float
    s2_bs: float
    s2_hs: float
    e_ap_ha: float
    e_ap_delta_kcal: float
    j_yamaguchi_cm1: float
    j_noodleman_cm1: float
    coupling_type: str  # Antiferromagnetic (J < 0) or Ferromagnetic (J > 0)


def evaluate_spin_contamination(
    multiplicity: int,
    s2_calc: float,
    threshold_percent: float = 10.0,
) -> SpinDiagnosticsResult:
    """Evaluates spin contamination of an unrestricted calculation from <S^2>."""
    mult = max(1, int(multiplicity))
    s = (mult - 1) / 2.0
    s2_exact = s * (s + 1.0)

    if s2_exact < 1e-6:
        # Pure singlet S=0: any deviation from 0 is pure contamination
        diff = max(0.0, float(s2_calc))
        contam_pct = diff * 100.0  # e.g., 0.10 -> 10%
    else:
        diff = max(0.0, float(s2_calc) - s2_exact)
        contam_pct = (diff / s2_exact) * 100.0

    is_contam = contam_pct > threshold_percent

    if not is_contam:
        status = f"Acceptable: <S²> = {s2_calc:.4f} (Ideal {s2_exact:.4f}, Contamination: {contam_pct:.1f}%)"
    else:
        status = f"Severe Contamination: <S²> = {s2_calc:.4f} exceeds ideal {s2_exact:.4f} by {contam_pct:.1f}%!"

    return SpinDiagnosticsResult(
        spin_s=s,
        multiplicity=mult,
        s2_exact=s2_exact,
        s2_calc=float(s2_calc),
        contamination_percent=float(contam_pct),
        is_contaminated=is_contam,
        unpaired_electrons=mult - 1,
        status_summary=status,
    )


def compute_yamaguchi_projection(
    e_bs_ha: float,
    e_hs_ha: float,
    s2_bs: float,
    s2_hs: float,
) -> YamaguchiProjectionResult:
    """Computes Yamaguchi approximate spin projection (AP) and Heisenberg J coupling.
    
    1 Hartree = 219474.63 cm^-1 = 627.509474 kcal/mol
    """
    HA_TO_KCAL = 627.509474
    HA_TO_CM1 = 219474.63

    ds2 = float(s2_hs - s2_bs)
    if abs(ds2) < 1e-4:
        raise ValueError("Denominator <S²>_HS - <S²>_BS is too close to zero for Yamaguchi projection.")

    # E_AP = (E_BS * <S^2>_HS - E_HS * <S^2>_BS) / (<S^2>_HS - <S^2>_BS)
    e_ap = (e_bs_ha * s2_hs - e_hs_ha * s2_bs) / ds2
    e_ap_delta_kcal = (e_ap - e_bs_ha) * HA_TO_KCAL

    # J_Yamaguchi = (E_BS - E_HS) / (<S^2>_HS - <S^2>_BS)
    # in H = -2J * S_A * S_B convention
    j_ha = (e_bs_ha - e_hs_ha) / ds2
    j_cm1 = j_ha * HA_TO_CM1

    # Noodleman limit: assumes weak overlap, S_max = 1 for diradicals
    j_noodleman_cm1 = (e_bs_ha - e_hs_ha) * HA_TO_CM1

    coupling = "Antiferromagnetic (J < 0)" if j_cm1 < 0 else "Ferromagnetic (J > 0)"

    return YamaguchiProjectionResult(
        e_bs_ha=float(e_bs_ha),
        e_hs_ha=float(e_hs_ha),
        s2_bs=float(s2_bs),
        s2_hs=float(s2_hs),
        e_ap_ha=float(e_ap),
        e_ap_delta_kcal=float(e_ap_delta_kcal),
        j_yamaguchi_cm1=float(j_cm1),
        j_noodleman_cm1=float(j_noodleman_cm1),
        coupling_type=coupling,
    )
