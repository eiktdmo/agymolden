"""
Boltzmann Conformer Ensemble & Property Averager for AgyMolden.

Computes statistically rigorous conformational ensemble properties:
- Conformational entropy S_conf = -R * sum(w_i * ln(w_i))
- Free energy conformational correction -T * S_conf (kcal/mol)
- Effective number of accessible conformers N_eff = exp(S_conf / R)
- Boltzmann-weighted Radius of Gyration <R_g> (Å)
- Population cutoffs (e.g. 95% and 99% cumulative ensemble thresholds)
- Publication-ready summary formatting for Supporting Information
"""

from typing import List, Dict, Any, Tuple
import math
import numpy as np

from .crest_explorer import ConformerRecord, KB_KCAL_MOL_K
from .elements import get_mass


# Gas constant R in cal / (mol * K)
R_CAL_MOL_K = 1.98720425864083


def calculate_radius_of_gyration(symbols: List[str], coords: np.ndarray) -> float:
    """
    Calculate mass-weighted radius of gyration (Å) for a 3D molecular structure.
    R_g = sqrt( sum(m_i * ||r_i - r_com||^2) / sum(m_i) )
    """
    if len(symbols) == 0 or len(coords) != len(symbols):
        return 0.0

    masses = np.array([get_mass(s) for s in symbols], dtype=np.float64)
    total_mass = float(np.sum(masses))
    if total_mass <= 1e-6:
        return 0.0

    # Center of mass
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    diff = coords - com
    sq_dist = np.sum(diff ** 2, axis=1)
    rg_sq = float(np.sum(masses * sq_dist) / total_mass)
    return float(math.sqrt(max(0.0, rg_sq)))


def calculate_conformational_entropy(
    weights: np.ndarray, temperature_k: float = 298.15
) -> Tuple[float, float, float]:
    """
    Calculate conformational entropy and free energy correction from Boltzmann weights.

    Parameters:
        weights: 1D array of fractional weights summing to 1.0 (w_i in [0, 1])
        temperature_k: Temperature in Kelvin

    Returns:
        (s_conf_cal, minus_t_s_kcal, n_eff):
          - s_conf_cal: S_conf in cal / (mol * K)
          - minus_t_s_kcal: -T * S_conf in kcal / mol
          - n_eff: Effective accessible conformers N_eff = exp(S_conf / R)
    """
    w = np.asarray(weights, dtype=np.float64)
    valid_w = w[w > 1e-12]
    if len(valid_w) <= 1:
        return 0.0, 0.0, 1.0

    # Normalize if needed
    sum_w = float(np.sum(valid_w))
    if abs(sum_w - 1.0) > 1e-5:
        valid_w = valid_w / sum_w

    # S_conf = -R * sum(w * ln(w))
    entropy_term = -float(np.sum(valid_w * np.log(valid_w)))
    s_conf_cal = R_CAL_MOL_K * entropy_term
    s_conf_kcal = KB_KCAL_MOL_K * entropy_term
    minus_t_s_kcal = -temperature_k * s_conf_kcal
    n_eff = float(math.exp(entropy_term))

    return s_conf_cal, minus_t_s_kcal, n_eff


def calculate_ensemble_statistics(
    conformers: List[ConformerRecord], temperature_k: float = 298.15
) -> Dict[str, Any]:
    """
    Compute comprehensive Boltzmann ensemble statistics for a set of conformers.
    """
    if not conformers:
        return {
            "n_total": 0,
            "n_active": 0,
            "n_eff": 0.0,
            "s_conf_cal": 0.0,
            "minus_t_s_kcal": 0.0,
            "avg_rg": 0.0,
            "delta_e_range": (0.0, 0.0),
            "pop_95_count": 0,
            "summary_text": "No conformers loaded.",
        }

    n_total = len(conformers)
    # Extract populations (0-100%) and convert to fractional weights
    pops = np.array([max(0.0, c.boltzmann_pop) for c in conformers], dtype=np.float64)
    total_pop = float(np.sum(pops))
    if total_pop > 1e-6:
        weights = pops / total_pop
    else:
        weights = np.ones(n_total, dtype=np.float64) / float(n_total)

    s_conf_cal, minus_t_s_kcal, n_eff = calculate_conformational_entropy(weights, temperature_k)

    # Compute weighted radius of gyration
    r_gyrs = []
    for c in conformers:
        rg = calculate_radius_of_gyration(c.symbols, c.coords)
        r_gyrs.append(rg)
    r_gyrs_arr = np.array(r_gyrs, dtype=np.float64)
    avg_rg = float(np.sum(weights * r_gyrs_arr))

    # Conformers accounting for 95% of cumulative population
    cum_pop = 0.0
    pop_95_count = 0
    for idx, w in enumerate(weights):
        cum_pop += w
        pop_95_count += 1
        if cum_pop >= 0.95:
            break

    # Energy spread
    delta_es = [c.delta_e_kcal for c in conformers]
    min_de = float(min(delta_es))
    max_de = float(max(delta_es))

    # Count within chemical significance window (<= 3.0 kcal/mol)
    within_3kcal = sum(1 for de in delta_es if de <= 3.0)

    summary_text = (
        f"Ensemble: {n_total} conformers ({within_3kcal} within 3.0 kcal/mol) | "
        f"T = {temperature_k:.1f} K\n"
        f"Effective Accessible States (N_eff): {n_eff:.2f}\n"
        f"Conformational Entropy: S_conf = {s_conf_cal:.2f} cal/(mol·K)\n"
        f"Free Energy Correction: -T·ΔS_conf = {minus_t_s_kcal:.3f} kcal/mol\n"
        f"Ensemble ⟨R_g⟩: {avg_rg:.3f} Å (spread: {float(np.min(r_gyrs_arr)):.3f} - {float(np.max(r_gyrs_arr)):.3f} Å)\n"
        f"95% Population Threshold: Top {pop_95_count} conformer(s)"
    )

    return {
        "n_total": n_total,
        "n_within_3kcal": within_3kcal,
        "n_eff": n_eff,
        "s_conf_cal": s_conf_cal,
        "minus_t_s_kcal": minus_t_s_kcal,
        "avg_rg": avg_rg,
        "rg_min": float(np.min(r_gyrs_arr)),
        "rg_max": float(np.max(r_gyrs_arr)),
        "delta_e_range": (min_de, max_de),
        "pop_95_count": pop_95_count,
        "temperature_k": temperature_k,
        "summary_text": summary_text,
    }
