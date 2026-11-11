"""Energy Decomposition Analysis (EDA) and SAPT Interaction Module for AgyMolden.

Decomposes non-covalent or donor-acceptor interaction energies into:
1. Electrostatic Energy (ΔE_elstat): Classical Coulombic interaction between frozen fragment densities.
2. Pauli Repulsion / Exchange (ΔE_Pauli): Orthogonalization & anti-symmetrization kinetic penalty.
3. Orbital / Polarization / Charge-Transfer (ΔE_orb): Relaxation of MOs upon complex formation.
4. Dispersion (ΔE_disp): Dynamic London dispersion (D3, D4, or SAPT dispersion).

Total Interaction Energy:
    ΔE_int = ΔE_elstat + ΔE_Pauli + ΔE_orb + ΔE_disp
"""

from __future__ import annotations

import re
from typing import Dict, NamedTuple, Optional


class EDAResult(NamedTuple):
    method: str
    elstat_kcal: float
    pauli_kcal: float
    orb_kcal: float
    disp_kcal: float
    total_int_kcal: float
    summary: str


def parse_eda_text(text: str) -> Optional[EDAResult]:
    """Parses Energy Decomposition Analysis (EDA) from Q-Chem ALMO-EDA, Psi4 SAPT, or ORCA output."""
    if not text:
        return None

    # 1. Psi4 SAPT (SAPT0 / SAPT2+)
    # Electrostatics:   -12.34 kcal/mol
    # Exchange:          18.56 kcal/mol
    # Induction:         -4.12 kcal/mol
    # Dispersion:        -7.89 kcal/mol
    # Total SAPT0:       -5.79 kcal/mol
    sapt_el = re.search(r'Electrostatics\s*[:=]\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    sapt_ex = re.search(r'Exchange\s*[:=]\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    sapt_ind = re.search(r'Induction\s*[:=]\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    sapt_disp = re.search(r'Dispersion\s*[:=]\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    sapt_tot = re.search(r'Total\s+SAPT\w*\s*[:=]\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)

    if sapt_el and sapt_ex:
        el = float(sapt_el.group(1))
        ex = float(sapt_ex.group(1))
        ind = float(sapt_ind.group(1)) if sapt_ind else 0.0
        disp = float(sapt_disp.group(1)) if sapt_disp else 0.0
        tot = float(sapt_tot.group(1)) if sapt_tot else (el + ex + ind + disp)
        return EDAResult(
            method="Psi4 SAPT",
            elstat_kcal=el,
            pauli_kcal=ex,
            orb_kcal=ind,
            disp_kcal=disp,
            total_int_kcal=tot,
            summary=f"SAPT Total: {tot:.2f} kcal/mol (Elstat: {el:.2f}, Exch: {ex:.2f}, Ind: {ind:.2f}, Disp: {disp:.2f})",
        )

    # 2. Q-Chem ALMO-EDA / Morokuma
    # DELTA E ELSTAT = ...
    # DELTA E PAULI  = ...
    # DELTA E POL    = ...
    # DELTA E CT     = ...
    # DELTA E DISP   = ...
    m_el = re.search(r'(?:E_ELSTAT|DELTA E ELSTAT|Electrostatic energy)\s*[:=]?\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    m_pauli = re.search(r'(?:E_PAULI|DELTA E PAULI|Pauli repulsion energy)\s*[:=]?\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    m_orb = re.search(r'(?:E_ORB|DELTA E ORB|DELTA E POL|Orbital interaction energy)\s*[:=]?\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    m_disp = re.search(r'(?:E_DISP|DELTA E DISP|Dispersion energy)\s*[:=]?\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)
    m_tot = re.search(r'(?:DELTA E INT|TOTAL INTERACTION ENERGY)\s*[:=]?\s*([+-]?\d+\.\d+)', text, re.IGNORECASE)

    if m_el and m_pauli:
        el = float(m_el.group(1))
        pauli = float(m_pauli.group(1))
        orb = float(m_orb.group(1)) if m_orb else 0.0
        disp = float(m_disp.group(1)) if m_disp else 0.0
        tot = float(m_tot.group(1)) if m_tot else (el + pauli + orb + disp)
        return EDAResult(
            method="ALMO-EDA / Morokuma",
            elstat_kcal=el,
            pauli_kcal=pauli,
            orb_kcal=orb,
            disp_kcal=disp,
            total_int_kcal=tot,
            summary=f"EDA Total: {tot:.2f} kcal/mol (Elstat: {el:.2f}, Pauli: {pauli:.2f}, Orb: {orb:.2f}, Disp: {disp:.2f})",
        )

    return None
