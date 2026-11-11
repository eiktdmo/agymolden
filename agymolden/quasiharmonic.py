"""
Quasi-Harmonic Thermochemistry Module for AgyMolden (Grimme mRRHO & Truhlar Models).

Solves the low-frequency breakdown of standard Rigid-Rotor Harmonic Oscillator (RRHO)
entropy calculations (which diverges as omega -> 0) by smoothly interpolating
between harmonic oscillator entropy and free-rotor entropy.

Models implemented:
1. Grimme Modified RRHO (mRRHO) [Grimme, Chem. Eur. J. 2012, 18, 9955-9964]:
   - Interpolation using damping function w(omega) = 1 / (1 + (omega_0 / omega)^4)
   - Free-rotor entropy S_rotor evaluated using average molecular moment of inertia B_av
2. Truhlar Quasi-Harmonic [Ribeiro, Marenich, Cramer, Truhlar, J. Phys. Chem. B 2011, 115, 14556]:
   - All frequencies below cutoff omega_0 (default 100 cm^-1) are raised to omega_0
3. Automated Truhlar / NIST Vibrational Frequency & ZPE Scaling Factor database
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

# Physical constants (CODATA 2018)
H_PLANCK = 6.62607015e-34       # J s
K_BOLTZMANN = 1.380649e-23      # J / K
C_LIGHT = 2.99792458e10         # cm / s
R_GAS = 8.314462618             # J / (mol K)
R_GAS_KCAL = 1.98720425864e-3   # kcal / (mol K)
AVOGADRO = 6.02214076e23        # mol^-1
HARTREE_TO_KCAL = 627.5094740631

# Standard database of vibrational frequency scaling factors (Truhlar group / CCCBDB)
FREQUENCY_SCALING_FACTORS: Dict[str, float] = {
    "B3LYP/6-31G(d)": 0.9614,
    "B3LYP/6-31+G(d,p)": 0.9640,
    "B3LYP/def2-SVP": 0.9650,
    "B3LYP/def2-TZVP": 0.9670,
    "wB97X-D/6-31G(d)": 0.9570,
    "wB97X-D/def2-SVP": 0.9560,
    "wB97X-D/def2-TZVP": 0.9580,
    "M06-2X/def2-TZVP": 0.9700,
    "PBE0/def2-TZVP": 0.9600,
    "HF/6-31G(d)": 0.8953,
    "MP2/def2-TZVP": 0.9740,
}


class QuasiHarmonicResult(NamedTuple):
    temperature_k: float
    cutoff_cm1: float
    frequencies_raw: List[float]
    frequencies_scaled: List[float]
    scaling_factor: float
    # Standard RRHO
    s_vib_standard_cal_mol_k: float
    g_vib_standard_kcal_mol: float
    # Grimme mRRHO
    s_vib_grimme_cal_mol_k: float
    g_vib_grimme_kcal_mol: float
    # Truhlar Quasi-harmonic
    s_vib_truhlar_cal_mol_k: float
    g_vib_truhlar_kcal_mol: float
    # Quasi-harmonic Delta G corrections relative to standard RRHO (positive = free energy stabilization)
    delta_g_grimme_kcal_mol: float
    delta_g_truhlar_kcal_mol: float
    zpe_kcal_mol: float


def vibrational_entropy_harmonic(freq_cm1: float, temp_k: float) -> float:
    """Computes standard harmonic oscillator vibrational entropy S_vib in J / (mol K) for a single mode."""
    if freq_cm1 <= 1e-4 or temp_k <= 0.0:
        return 0.0
    nu = freq_cm1 * C_LIGHT  # s^-1
    x = (H_PLANCK * nu) / (K_BOLTZMANN * temp_k)
    if x > 150.0:
        return 0.0
    exp_x = math.exp(x)
    # S = R * [ x / (exp(x) - 1) - ln(1 - exp(-x)) ]
    return float(R_GAS * (x / (exp_x - 1.0) - math.log(1.0 - math.exp(-x))))


def vibrational_entropy_free_rotor(freq_cm1: float, temp_k: float, b_av: float = 1e-44) -> float:
    """Computes free rotor entropy S_rot in J / (mol K) using effective moment of inertia."""
    if freq_cm1 <= 1e-4 or temp_k <= 0.0:
        return 0.0
    nu = freq_cm1 * C_LIGHT
    # Effective moment of inertia mu = h / (8 * pi^2 * nu)
    mu = H_PLANCK / (8.0 * (math.pi ** 2) * nu)
    # Combined effective moment of inertia: mu' = (mu * B_av) / (mu + B_av)
    mu_prime = (mu * b_av) / (mu + b_av)
    val = (8.0 * (math.pi ** 3) * mu_prime * K_BOLTZMANN * temp_k) / (H_PLANCK ** 2)
    if val <= 0:
        return 0.0
    return float(R_GAS * (0.5 + 0.5 * math.log(val)))


def calculate_quasiharmonic_thermo(
    frequencies_cm1: List[float],
    temperature_k: float = 298.15,
    cutoff_cm1: float = 100.0,
    scaling_factor: float = 1.0,
    molecular_mass_amu: float = 100.0,
) -> QuasiHarmonicResult:
    """
    Computes standard RRHO, Grimme modified RRHO (mRRHO), and Truhlar quasi-harmonic vibrational
    entropy and Gibbs free energy corrections.
    """
    if temperature_k <= 0:
        raise ValueError("Temperature must be positive.")

    # Filter real positive vibrational frequencies (ignore imaginary or near-zero rotational/translational modes)
    real_freqs = [f for f in frequencies_cm1 if f > 5.0]
    scaled_freqs = [f * scaling_factor for f in real_freqs]

    if not scaled_freqs:
        return QuasiHarmonicResult(
            temperature_k=temperature_k,
            cutoff_cm1=cutoff_cm1,
            frequencies_raw=real_freqs,
            frequencies_scaled=[],
            scaling_factor=scaling_factor,
            s_vib_standard_cal_mol_k=0.0,
            g_vib_standard_kcal_mol=0.0,
            s_vib_grimme_cal_mol_k=0.0,
            g_vib_grimme_kcal_mol=0.0,
            s_vib_truhlar_cal_mol_k=0.0,
            g_vib_truhlar_kcal_mol=0.0,
            delta_g_grimme_kcal_mol=0.0,
            delta_g_truhlar_kcal_mol=0.0,
            zpe_kcal_mol=0.0,
        )

    # Estimate average molecular moment of inertia B_av (kg m^2) from molecular mass
    # Typical small-to-medium organic molecule has B_av ~ 1e-44 kg m^2
    b_av = max(1e-45, 1e-44 * (molecular_mass_amu / 100.0))

    # Zero-Point Energy: ZPE = 0.5 * sum(h * nu)
    zpe_j = sum(0.5 * H_PLANCK * (f * C_LIGHT) * AVOGADRO for f in scaled_freqs)
    zpe_kcal = (zpe_j / 4184.0)

    s_std_j = 0.0
    s_grimme_j = 0.0
    s_truhlar_j = 0.0

    cutoff_4 = cutoff_cm1 ** 4

    for f in scaled_freqs:
        # Standard harmonic entropy
        s_h = vibrational_entropy_harmonic(f, temperature_k)
        s_std_j += s_h

        # Grimme mRRHO: damping function w(omega) = 1 / (1 + (omega_0 / omega)^4)
        w = 1.0 / (1.0 + (cutoff_4 / (f ** 4))) if f > 0 else 0.0
        s_rot = vibrational_entropy_free_rotor(f, temperature_k, b_av)
        s_grimme_j += (w * s_h + (1.0 - w) * s_rot)

        # Truhlar model: frequencies below cutoff are raised to cutoff
        f_truhlar = max(f, cutoff_cm1)
        s_truhlar_j += vibrational_entropy_harmonic(f_truhlar, temperature_k)

    # Convert J/(mol K) to cal/(mol K)
    j_to_cal = 1.0 / 4.184
    s_std_cal = s_std_j * j_to_cal
    s_grimme_cal = s_grimme_j * j_to_cal
    s_truhlar_cal = s_truhlar_j * j_to_cal

    # Vibrational contribution to G: G_vib = ZPE - T * S_vib (in kcal/mol)
    t_s_std = (temperature_k * s_std_cal) / 1000.0
    t_s_grimme = (temperature_k * s_grimme_cal) / 1000.0
    t_s_truhlar = (temperature_k * s_truhlar_cal) / 1000.0

    g_std_kcal = zpe_kcal - t_s_std
    g_grimme_kcal = zpe_kcal - t_s_grimme
    g_truhlar_kcal = zpe_kcal - t_s_truhlar

    # Delta G corrections = G_quasiharmonic - G_standard (usually negative or positive depending on entropy increase)
    d_g_grimme = g_grimme_kcal - g_std_kcal
    d_g_truhlar = g_truhlar_kcal - g_std_kcal

    return QuasiHarmonicResult(
        temperature_k=temperature_k,
        cutoff_cm1=cutoff_cm1,
        frequencies_raw=real_freqs,
        frequencies_scaled=scaled_freqs,
        scaling_factor=scaling_factor,
        s_vib_standard_cal_mol_k=s_std_cal,
        g_vib_standard_kcal_mol=g_std_kcal,
        s_vib_grimme_cal_mol_k=s_grimme_cal,
        g_vib_grimme_kcal_mol=g_grimme_kcal,
        s_vib_truhlar_cal_mol_k=s_truhlar_cal,
        g_vib_truhlar_kcal_mol=g_truhlar_kcal,
        delta_g_grimme_kcal_mol=d_g_grimme,
        delta_g_truhlar_kcal_mol=d_g_truhlar,
        zpe_kcal_mol=zpe_kcal,
    )
