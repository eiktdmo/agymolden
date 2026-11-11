"""Electrochemical Redox Potentials and Born-Haber Solvation Cycles Module for AgyMolden.

Computes:
- Standard reduction potential E° (vs SHE, SCE, Fc/Fc+) from QM gas-phase energies and solvation free energies:
    ΔG°_red(solv) = ΔG°_red(gas) + ΔG_solv(Red) - ΔG_solv(Ox)
    E°(vs SHE) = -ΔG°_red(solv) / (n * F) - V_ref(SHE)

Reference Electrodes:
- Standard Hydrogen Electrode (SHE): Absolute potential 4.281 V (IUPAC recommended) or 4.44 V
- Saturated Calomel Electrode (SCE): +0.241 V vs SHE
- Ferrocene / Ferrocenium (Fc/Fc+): +0.400 V in MeCN vs SCE (+0.641 V vs SHE)
- Ag/AgCl (sat. KCl): +0.197 V vs SHE
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional
import numpy as np

# Faraday constant in kcal/(mol * V)
# 1 eV = 23.060548 kcal/mol => 1 F = 23.060548 kcal / (mol * V)
FARADAY_KCAL_MOL_V = 23.060548

REFERENCE_ELECTRODES: Dict[str, Dict[str, float]] = {
    "SHE (IUPAC 4.28 V)": {
        "abs_potential_v": 4.281,
        "offset_from_she_v": 0.0,
        "description": "Standard Hydrogen Electrode (IUPAC 4.281 V)",
    },
    "SHE (Trasatti 4.44 V)": {
        "abs_potential_v": 4.440,
        "offset_from_she_v": 0.0,
        "description": "Standard Hydrogen Electrode (Trasatti 4.44 V)",
    },
    "SCE (Sat. Calomel, +0.241 V vs SHE)": {
        "abs_potential_v": 4.522,
        "offset_from_she_v": 0.241,
        "description": "Saturated Calomel Electrode (SCE)",
    },
    "Ag/AgCl (Sat. KCl, +0.197 V vs SHE)": {
        "abs_potential_v": 4.478,
        "offset_from_she_v": 0.197,
        "description": "Silver/Silver Chloride (Ag/AgCl, sat. KCl)",
    },
    "Fc/Fc+ (Ferrocene in MeCN, +0.641 V vs SHE)": {
        "abs_potential_v": 4.922,
        "offset_from_she_v": 0.641,
        "description": "Ferrocene/Ferrocenium couple in Acetonitrile",
    },
}


class RedoxPotentialResult(NamedTuple):
    g_ox_gas_ha: float
    g_red_gas_ha: float
    dg_gas_kcal: float
    dg_solv_ox_kcal: float
    dg_solv_red_kcal: float
    ddg_solv_kcal: float
    dg_total_solv_kcal: float
    electrons_transferred: int
    e0_she_v: float
    potentials_vs_ref: Dict[str, float]
    redox_type: str  # Reduction or Oxidation


def compute_redox_potential(
    g_ox_gas_ha: float,
    g_red_gas_ha: float,
    dg_solv_ox_kcal: float,
    dg_solv_red_kcal: float,
    electrons_n: int = 1,
    she_absolute_v: float = 4.281,
) -> RedoxPotentialResult:
    """Evaluates the thermodynamic Born-Haber cycle for reduction: Ox + n e- -> Red."""
    HA_TO_KCAL = 627.509474
    n = max(1, int(electrons_n))

    # Gas phase reduction free energy: G(Red) - G(Ox)
    dg_gas_ha = g_red_gas_ha - g_ox_gas_ha
    dg_gas_kcal = dg_gas_ha * HA_TO_KCAL

    # Differential solvation: ΔG_solv(Red) - ΔG_solv(Ox)
    ddg_solv_kcal = dg_solv_red_kcal - dg_solv_ox_kcal

    # Total solution-phase reduction free energy
    dg_tot_kcal = dg_gas_kcal + ddg_solv_kcal

    # Absolute reduction potential: E°_abs = -ΔG°_tot / (n * F)
    e_abs_v = -dg_tot_kcal / (n * FARADAY_KCAL_MOL_V)

    # Standard reduction potential vs SHE
    e0_she = e_abs_v - she_absolute_v

    # Map across standard reference electrodes
    v_refs = {}
    for name, ref in REFERENCE_ELECTRODES.items():
        v_refs[name] = e0_she - ref["offset_from_she_v"]

    return RedoxPotentialResult(
        g_ox_gas_ha=float(g_ox_gas_ha),
        g_red_gas_ha=float(g_red_gas_ha),
        dg_gas_kcal=float(dg_gas_kcal),
        dg_solv_ox_kcal=float(dg_solv_ox_kcal),
        dg_solv_red_kcal=float(dg_solv_red_kcal),
        ddg_solv_kcal=float(ddg_solv_kcal),
        dg_total_solv_kcal=float(dg_tot_kcal),
        electrons_transferred=n,
        e0_she_v=float(e0_she),
        potentials_vs_ref=v_refs,
        redox_type="Reduction: Ox + n e⁻ → Red",
    )
