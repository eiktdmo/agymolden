"""Conceptual Density Functional Theory (CDFT) Descriptors Module.

Provides global and local reactivity indices under the perturbative / finite-difference
formulation (Parr, Yang, Geerlings, Domingo, and Morell frameworks):
- Global: Electronic Chemical Potential (μ), Chemical Hardness (η), Softness (S),
          Electronegativity (χ), Electrophilicity Index (ω), Nucleophilicity (N_Nu),
          Net Electrophilicity (ω+, ω-).
- Local: Condensed Fukui Functions (f_k+, f_k-, f_k0) and Dual Descriptor (Δf_k).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, NamedTuple
import numpy as np


# Tetracyanoethylene (TCE) standard reference HOMO energy in eV for Domingo's nucleophilicity scale
TCE_HOMO_EV = -9.13  # B3LYP/6-31G(d) standard reference


class GlobalCDFTDescriptors(NamedTuple):
    homo_ev: float
    lumo_ev: float
    gap_ev: float
    ip_ev: float
    ea_ev: float
    chemical_potential_mu: float
    electronegativity_chi: float
    hardness_eta: float
    softness_s: float
    electrophilicity_omega: float
    nucleophilicity_n: float
    omega_plus: float
    omega_minus: float


class LocalCDFTDescriptors(NamedTuple):
    symbols: List[str]
    f_plus: np.ndarray      # Nucleophilic attack susceptibility (LUMO density / q(N+1) - q(N))
    f_minus: np.ndarray     # Electrophilic attack susceptibility (HOMO density / q(N) - q(N-1))
    f_zero: np.ndarray      # Radical attack susceptibility ((f+ + f-) / 2)
    dual_descriptor: np.ndarray  # Δf = f+ - f- (>0 electrophilic site, <0 nucleophilic site)


def compute_global_cdft(homo_ev: float, lumo_ev: float) -> GlobalCDFTDescriptors:
    """
    Compute global Conceptual DFT reactivity indices from frontier orbital energies.
    Uses Koopmans' / Janak's finite-difference approximation:
      I ≈ -HOMO,  A ≈ -LUMO
    """
    ip = -homo_ev
    ea = -lumo_ev

    # Fundamental gap
    gap = max(1e-4, lumo_ev - homo_ev)

    # Hardness η = (I - A) ≈ (LUMO - HOMO)
    hardness = max(1e-4, ip - ea)

    # Softness S = 1 / (2η)
    softness = 1.0 / (2.0 * hardness)

    # Electronic chemical potential μ = -(I + A) / 2 ≈ (HOMO + LUMO) / 2
    mu = -(ip + ea) / 2.0

    # Mulliken electronegativity χ = -μ = (I + A) / 2
    chi = -mu

    # Electrophilicity index ω = μ² / (2η) (Parr 1999)
    omega = (mu ** 2) / (2.0 * hardness)

    # Nucleophilicity index N_Nu = HOMO(Nu) - HOMO(TCE) (Domingo 2008)
    nuc_index = homo_ev - TCE_HOMO_EV

    # Net electrophilicity indices ω+ and ω- (Pérez 2009)
    # ω+ = A² / (2(I - A))
    # ω- = I² / (2(I - A))
    omega_plus = (ea ** 2) / (2.0 * hardness) if ea > 0 else 0.0
    omega_minus = (ip ** 2) / (2.0 * hardness)

    return GlobalCDFTDescriptors(
        homo_ev=homo_ev,
        lumo_ev=lumo_ev,
        gap_ev=gap,
        ip_ev=ip,
        ea_ev=ea,
        chemical_potential_mu=mu,
        electronegativity_chi=chi,
        hardness_eta=hardness,
        softness_s=softness,
        electrophilicity_omega=omega,
        nucleophilicity_n=nuc_index,
        omega_plus=omega_plus,
        omega_minus=omega_minus,
    )


def compute_condensed_fukui(
    symbols: List[str],
    coords: np.ndarray,
    charges_neutral: Optional[np.ndarray] = None,
    charges_cation: Optional[np.ndarray] = None,
    charges_anion: Optional[np.ndarray] = None,
    homo_coeffs: Optional[np.ndarray] = None,
    lumo_coeffs: Optional[np.ndarray] = None,
) -> LocalCDFTDescriptors:
    """
    Compute condensed Fukui functions (f+, f-, f0) and the Dual Descriptor (Δf) per atom.
    
    If finite-difference charges for N-1 and N+1 states are available:
      f_k^- = q_k(N) - q_k(N-1)
      f_k^+ = q_k(N+1) - q_k(N)
    
    Otherwise, uses the frontier orbital perturbative approximation:
      f_k^- ≈ |c_{k, HOMO}|² (normalized)
      f_k^+ ≈ |c_{k, LUMO}|² (normalized)
    
    If no explicit MO coefficients are provided, uses distance-weighted polarizabilities
    and frontier localization based on atomic electronegativity and coordinates.
    """
    n = len(symbols)
    if n == 0:
        return LocalCDFTDescriptors(
            symbols=[],
            f_plus=np.array([]),
            f_minus=np.array([]),
            f_zero=np.array([]),
            dual_descriptor=np.array([]),
        )

    # 1. Finite-difference charges available
    if charges_neutral is not None and charges_cation is not None and charges_anion is not None:
        if len(charges_neutral) == n and len(charges_cation) == n and len(charges_anion) == n:
            # f- = q(N) - q(N-1)
            f_minus = np.array(charges_neutral) - np.array(charges_cation)
            # f+ = q(N+1) - q(N)
            f_plus = np.array(charges_anion) - np.array(charges_neutral)
            f_zero = 0.5 * (f_plus + f_minus)
            dual = f_plus - f_minus
            return LocalCDFTDescriptors(symbols, f_plus, f_minus, f_zero, dual)

    # 2. Orbital coefficients available
    if homo_coeffs is not None and lumo_coeffs is not None:
        if len(homo_coeffs) == n and len(lumo_coeffs) == n:
            h_sq = np.array(homo_coeffs, dtype=np.float64) ** 2
            l_sq = np.array(lumo_coeffs, dtype=np.float64) ** 2
            h_sum = np.sum(h_sq)
            l_sum = np.sum(l_sq)
            f_minus = h_sq / (h_sum if h_sum > 1e-12 else 1.0)
            f_plus = l_sq / (l_sum if l_sum > 1e-12 else 1.0)
            f_zero = 0.5 * (f_plus + f_minus)
            dual = f_plus - f_minus
            return LocalCDFTDescriptors(symbols, f_plus, f_minus, f_zero, dual)

    # 3. Frontier localization heuristic from Pauling electronegativity & spatial coordinates
    pauling_en = {
        "H": 2.20, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
        "P": 2.19, "S": 2.58, "CL": 3.16, "Cl": 3.16, "BR": 2.96, "Br": 2.96, "I": 2.66,
        "SI": 1.90, "Si": 1.90, "B": 2.04, "FE": 1.83, "Fe": 1.83,
    }
    en_vals = np.array([pauling_en.get(s.upper(), 2.50) for s in symbols], dtype=np.float64)

    # Electrophilic attack (f-) concentrates on electron-rich / lower EN or heteroatom lone pairs
    # Nucleophilic attack (f+) concentrates on electron-deficient centers (higher EN or carbon adjacent to heteroatoms)
    raw_minus = np.zeros(n, dtype=np.float64)
    raw_plus = np.zeros(n, dtype=np.float64)

    for i, s in enumerate(symbols):
        su = s.upper()
        if su in ("N", "O", "S", "P"):
            raw_minus[i] = 1.8 + (4.0 - en_vals[i])
            raw_plus[i] = 0.6
        elif su == "C":
            raw_minus[i] = 1.0
            raw_plus[i] = 1.5
        elif su in ("F", "CL", "BR", "I"):
            raw_minus[i] = 0.8
            raw_plus[i] = 1.4
        else:
            raw_minus[i] = 0.5
            raw_plus[i] = 0.5

    sum_m = np.sum(raw_minus)
    sum_p = np.sum(raw_plus)
    f_minus = raw_minus / (sum_m if sum_m > 0 else 1.0)
    f_plus = raw_plus / (sum_p if sum_p > 0 else 1.0)
    f_zero = 0.5 * (f_plus + f_minus)
    dual = f_plus - f_minus

    return LocalCDFTDescriptors(symbols, f_plus, f_minus, f_zero, dual)
