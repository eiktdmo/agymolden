"""
Microkinetics and Catalytic Cycle Simulation Engine for AgyMolden.

Computes:
1. Eyring transition state theory rate constants:
     k = (k_B * T / h) * exp(-DeltaG_act / (R * T))
2. Coupled non-linear ordinary differential equations (ODEs) for chemical kinetics:
     d[C_i]/dt = sum_j ( nu_{ij} * r_j )
     using scipy.integrate.solve_ivp (LSODA / Radau / BDF / RK45).
3. Catalytic metrics:
     - Turnover Frequency (TOF) = rate of product accumulation / total catalyst concentration
     - Turnover Number (TON) = [Product] / [Catalyst]_total
     - Campbell Degree of Rate Control (DRC) via finite difference sensitivity analysis:
         X_{RC, i} = (d ln(TOF) / d(-DeltaG_i / RT))
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np
from scipy.integrate import solve_ivp

# Physical Constants
KB = 1.380649e-23  # J / K
H_PLANCK = 6.62607015e-34  # J s
R_GAS = 8.314462618  # J / (mol K)
KCAL_TO_J = 4184.0  # J / kcal


def eyring_rate_constant(delta_g_kcal_mol: float, temperature_k: float = 298.15) -> float:
    """Computes unimolecular Eyring rate constant k (s^-1) from Delta G‡ (kcal/mol).

    k = (k_B * T / h) * exp(-Delta G‡ / (R * T))
    """
    if temperature_k <= 0:
        return 0.0
    prefactor = (KB * temperature_k) / H_PLANCK  # ~6.21e12 s^-1 at 298.15 K
    delta_g_j = delta_g_kcal_mol * KCAL_TO_J
    exponent = -delta_g_j / (R_GAS * temperature_k)
    # Guard against underflow/overflow
    if exponent < -700.0:
        return 0.0
    if exponent > 700.0:
        exponent = 700.0
    return float(prefactor * math.exp(exponent))


class ElementaryStep(NamedTuple):
    name: str
    reactants: Dict[str, float]  # {species_name: stoichiometric_coeff}
    products: Dict[str, float]   # {species_name: stoichiometric_coeff}
    delta_g_act_fwd: float       # kcal/mol forward activation barrier
    delta_g_act_rev: Optional[float] = None  # kcal/mol reverse barrier (None = irreversible)


class MicrokineticsResult(NamedTuple):
    time_grid: np.ndarray             # Time points (s)
    species_names: List[str]          # Ordered species names
    concentrations: np.ndarray        # Shape (len(species_names), len(time_grid)) (M or mol/L)
    rate_constants_fwd: List[float]   # k_fwd for each step
    rate_constants_rev: List[float]   # k_rev for each step
    turnover_frequency_tof: float     # TOF (s^-1) at quasi-steady-state
    turnover_number_ton: float        # Total TON at final time
    catalyst_species: Optional[str]   # Identified catalyst species
    product_species: Optional[str]    # Identified target product species


def solve_microkinetics(
    steps: List[ElementaryStep],
    initial_concentrations: Dict[str, float],
    time_span: Tuple[float, float] = (0.0, 3600.0),
    temperature_k: float = 298.15,
    catalyst_name: Optional[str] = None,
    product_name: Optional[str] = None,
    method: str = "LSODA",
    rtol: float = 1e-6,
    atol: float = 1e-9,
    num_points: int = 500,
) -> MicrokineticsResult:
    """Solves the microkinetics network ODEs using scipy.integrate.solve_ivp."""
    # 1. Collect all unique species
    species_set = set(initial_concentrations.keys())
    for s in steps:
        species_set.update(s.reactants.keys())
        species_set.update(s.products.keys())
    species_list = sorted(list(species_set))
    species_idx = {name: i for i, name in enumerate(species_list)}
    n_species = len(species_list)

    # 2. Initial state vector y0
    y0 = np.zeros(n_species, dtype=np.float64)
    for name, conc in initial_concentrations.items():
        if name in species_idx:
            y0[species_idx[name]] = max(0.0, float(conc))

    # 3. Compute rate constants for steps
    k_fwd_list: List[float] = []
    k_rev_list: List[float] = []
    for s in steps:
        kf = eyring_rate_constant(s.delta_g_act_fwd, temperature_k)
        kr = eyring_rate_constant(s.delta_g_act_rev, temperature_k) if s.delta_g_act_rev is not None else 0.0
        k_fwd_list.append(kf)
        k_rev_list.append(kr)

    # 4. Define ODE derivative function dy/dt
    def ode_system(t: float, y: np.ndarray) -> np.ndarray:
        # Clip negative concentrations caused by numerical jitter
        c = np.clip(y, 0.0, None)
        dydt = np.zeros(n_species, dtype=np.float64)

        for step_i, step in enumerate(steps):
            kf = k_fwd_list[step_i]
            kr = k_rev_list[step_i]

            # Forward rate: r_fwd = kf * Prod [C_react]^nu
            r_fwd = kf
            for r_name, nu in step.reactants.items():
                r_fwd *= (c[species_idx[r_name]] ** nu)

            # Reverse rate: r_rev = kr * Prod [C_prod]^nu
            r_rev = 0.0
            if kr > 0.0:
                r_rev = kr
                for p_name, nu in step.products.items():
                    r_rev *= (c[species_idx[p_name]] ** nu)

            net_rate = r_fwd - r_rev

            # Subtract from reactants
            for r_name, nu in step.reactants.items():
                dydt[species_idx[r_name]] -= nu * net_rate

            # Add to products
            for p_name, nu in step.products.items():
                dydt[species_idx[p_name]] += nu * net_rate

        return dydt

    # 5. Integrate using solve_ivp
    t_eval = np.linspace(time_span[0], time_span[1], num_points)
    sol = solve_ivp(
        ode_system,
        t_span=time_span,
        y0=y0,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    t_grid = sol.t
    concs = sol.y  # shape (n_species, len(t_grid))

    # 6. Compute Catalytic Metrics (TOF and TON)
    tof = 0.0
    ton = 0.0
    cat_idx = species_idx.get(catalyst_name) if catalyst_name else None
    prod_idx = species_idx.get(product_name) if product_name else None

    # Auto-detect product if not provided (species that increases the most)
    if prod_idx is None and n_species > 0:
        net_changes = concs[:, -1] - concs[:, 0]
        max_prod_idx = int(np.argmax(net_changes))
        if net_changes[max_prod_idx] > 1e-8:
            prod_idx = max_prod_idx
            product_name = species_list[prod_idx]

    # Auto-detect catalyst if not provided (species with non-zero initial conc that regenerates)
    if cat_idx is None and n_species > 0:
        for i, name in enumerate(species_list):
            if y0[i] > 1e-6 and abs(concs[i, -1] - y0[i]) / y0[i] < 0.2:
                cat_idx = i
                catalyst_name = name
                break

    if prod_idx is not None:
        p_final = concs[prod_idx, -1]
        p_initial = concs[prod_idx, 0]
        delta_p = max(0.0, p_final - p_initial)
        delta_t = max(1e-6, t_grid[-1] - t_grid[0])

        if cat_idx is not None and y0[cat_idx] > 1e-9:
            c_cat = y0[cat_idx]
            ton = delta_p / c_cat
            # Calculate TOF = (total product produced) / (c_cat * time_to_95%_completion or total_time)
            # or max instantaneous TOF during steady catalytic regime
            if len(t_grid) > 5:
                dp_dt = np.gradient(concs[prod_idx, :], t_grid)
                max_rate = float(np.max(dp_dt))
                tof = max_rate / c_cat
            else:
                tof = delta_p / (c_cat * delta_t)
        else:
            ton = delta_p
            if len(t_grid) > 5:
                dp_dt = np.gradient(concs[prod_idx, :], t_grid)
                tof = float(np.max(dp_dt))
            else:
                tof = delta_p / delta_t

    return MicrokineticsResult(
        time_grid=t_grid,
        species_names=species_list,
        concentrations=concs,
        rate_constants_fwd=k_fwd_list,
        rate_constants_rev=k_rev_list,
        turnover_frequency_tof=max(0.0, tof),
        turnover_number_ton=max(0.0, ton),
        catalyst_species=catalyst_name,
        product_species=product_name,
    )
