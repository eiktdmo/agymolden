"""
Reaction Thermochemistry Workbench for AgyMolden.
Computes activation barriers (ΔG‡, ΔH‡), reaction free energies (ΔGrxn, ΔHrxn, ΔSrxn),
equilibrium constants (Keq), and Eyring reaction rate constants (k_T).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Physical Constants (CODATA standard)
HARTREE_TO_KCAL = 627.5094740631  # 1 Ha = 627.5095 kcal/mol
HARTREE_TO_KJ = 2625.4996395      # 1 Ha = 2625.5 kJ/mol
R_GAS_KCAL = 0.00198720425864083  # Gas constant R in kcal/(mol*K)
R_GAS_J = 8.31446261815324        # Gas constant R in J/(mol*K)
BOLTZMANN_K = 1.380649e-23        # J/K
PLANCK_H = 6.62607015e-34         # J*s


@dataclass
class SpeciesThermo:
    """Represents a chemical species in a reaction with thermochemical data."""
    name: str
    role: str  # "reactant", "ts", "product"
    coefficient: float = 1.0
    electronic_energy_ha: Optional[float] = None
    zpe_ha: Optional[float] = None
    enthalpy_ha: Optional[float] = None
    gibbs_ha: Optional[float] = None
    file_path: Optional[str] = None

    @property
    def enthalpy_val(self) -> Optional[float]:
        if self.enthalpy_ha is not None:
            return self.enthalpy_ha
        if self.electronic_energy_ha is not None and self.zpe_ha is not None:
            return self.electronic_energy_ha + self.zpe_ha
        return self.electronic_energy_ha

    @property
    def free_energy_val(self) -> Optional[float]:
        if self.gibbs_ha is not None:
            return self.gibbs_ha
        return self.enthalpy_val


@dataclass
class ReactionThermoResult:
    """Calculated reaction thermochemical and kinetic profile."""
    temperature_k: float = 298.15
    # Reactants total
    g_reactants_ha: float = 0.0
    h_reactants_ha: float = 0.0
    # Products total
    g_products_ha: float = 0.0
    h_products_ha: float = 0.0
    # Reaction thermodynamics
    delta_g_rxn_kcal: float = 0.0
    delta_h_rxn_kcal: float = 0.0
    delta_s_rxn_cal_mol_k: float = 0.0
    k_eq: Optional[float] = None
    is_exergonic: bool = False
    is_exothermic: bool = False
    # Activation barrier (if TS present)
    has_ts: bool = False
    g_ts_ha: Optional[float] = None
    delta_g_act_kcal: Optional[float] = None  # ΔG‡ fwd
    delta_h_act_kcal: Optional[float] = None  # ΔH‡ fwd
    delta_g_rev_kcal: Optional[float] = None  # ΔG‡ rev
    rate_constant_k_fwd: Optional[float] = None  # Eyring rate constant (s^-1)
    rate_constant_k_rev: Optional[float] = None  # Eyring rate constant (s^-1)


def calculate_reaction_thermochemistry(
    species_list: List[SpeciesThermo],
    temperature_k: float = 298.15,
) -> ReactionThermoResult:
    """
    Computes complete reaction thermochemical properties and activation kinetics.
    Safeguarded against division-by-zero, unphysical negative Kelvin, and missing data.
    """
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be greater than 0 Kelvin.")

    reactants = [s for s in species_list if s.role == "reactant"]
    products = [s for s in species_list if s.role == "product"]
    ts_list = [s for s in species_list if s.role == "ts"]

    if not reactants:
        raise ValueError("At least one reactant must be specified.")
    if not products:
        raise ValueError("At least one product must be specified.")

    # Calculate sum of reactants
    g_r = sum(s.coefficient * (s.free_energy_val or 0.0) for s in reactants)
    h_r = sum(s.coefficient * (s.enthalpy_val or 0.0) for s in reactants)

    # Calculate sum of products
    g_p = sum(s.coefficient * (s.free_energy_val or 0.0) for s in products)
    h_p = sum(s.coefficient * (s.enthalpy_val or 0.0) for s in products)

    delta_g_ha = g_p - g_r
    delta_h_ha = h_p - h_r

    delta_g_kcal = delta_g_ha * HARTREE_TO_KCAL
    delta_h_kcal = delta_h_ha * HARTREE_TO_KCAL
    delta_s_cal = ((delta_h_kcal - delta_g_kcal) * 1000.0) / temperature_k

    # Equilibrium constant: K_eq = exp(-ΔG / RT)
    rt = R_GAS_KCAL * temperature_k
    exponent = -delta_g_kcal / rt
    if exponent > 600.0:
        k_eq = float("inf")
    elif exponent < -600.0:
        k_eq = 0.0
    else:
        k_eq = math.exp(exponent)

    res = ReactionThermoResult(
        temperature_k=temperature_k,
        g_reactants_ha=g_r,
        h_reactants_ha=h_r,
        g_products_ha=g_p,
        h_products_ha=h_p,
        delta_g_rxn_kcal=delta_g_kcal,
        delta_h_rxn_kcal=delta_h_kcal,
        delta_s_rxn_cal_mol_k=delta_s_cal,
        k_eq=k_eq,
        is_exergonic=(delta_g_kcal < 0.0),
        is_exothermic=(delta_h_kcal < 0.0),
        has_ts=bool(ts_list),
    )

    # Eyring pre-exponential factor: (kB * T) / h
    eyring_prefactor = (BOLTZMANN_K * temperature_k) / PLANCK_H

    if ts_list:
        ts = ts_list[0]
        g_ts = ts.free_energy_val
        h_ts = ts.enthalpy_val
        if g_ts is not None:
            res.g_ts_ha = g_ts
            act_g_ha = g_ts - g_r
            res.delta_g_act_kcal = act_g_ha * HARTREE_TO_KCAL
            res.delta_g_rev_kcal = (g_ts - g_p) * HARTREE_TO_KCAL

            # Rate constants: k = (kB*T / h) * exp(-ΔG‡ / RT)
            exp_act = -res.delta_g_act_kcal / rt
            if exp_act < -600.0:
                res.rate_constant_k_fwd = 0.0
            elif exp_act > 600.0:
                res.rate_constant_k_fwd = float("inf")
            else:
                res.rate_constant_k_fwd = eyring_prefactor * math.exp(exp_act)

            if res.delta_g_rev_kcal is not None:
                exp_rev = -res.delta_g_rev_kcal / rt
                if exp_rev < -600.0:
                    res.rate_constant_k_rev = 0.0
                elif exp_rev > 600.0:
                    res.rate_constant_k_rev = float("inf")
                else:
                    res.rate_constant_k_rev = eyring_prefactor * math.exp(exp_rev)

        if h_ts is not None:
            res.delta_h_act_kcal = (h_ts - h_r) * HARTREE_TO_KCAL

    return res


def extract_species_from_log_file(file_path: str, role: str = "reactant", name: Optional[str] = None) -> SpeciesThermo:
    """Parses a calculation log file (ORCA, Gaussian, PySCF) to extract Gibbs, Enthalpy, ZPE, and E_el."""
    from .qm_log_parser import parse_qm_log

    base_name = name or os.path.splitext(os.path.basename(file_path))[0]
    result = parse_qm_log(file_path)

    # Determine electronic energy from trajectory or default
    e_el = None
    if result.trajectory_energies:
        # Check if last step energy was extracted
        e_el = result.trajectory_energies[-1] / 27.211386245988  # eV to Ha

    return SpeciesThermo(
        name=base_name,
        role=role,
        electronic_energy_ha=e_el,
        zpe_ha=result.zpe_ha,
        enthalpy_ha=result.enthalpy_ha,
        gibbs_ha=result.gibbs_ha,
        file_path=file_path,
    )
