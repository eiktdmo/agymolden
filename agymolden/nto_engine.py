"""
Natural Transition Orbitals (NTO) Engine for Excited States & Photochemistry in AgyMolden.

Computes:
1. Transition density matrix T from CIS / TD-DFT CI excitation coefficients:
     T_{ia} = c_{ia}
2. Singular Value Decomposition (SVD) of the transition matrix:
     T = U * Sigma * V^T
   where:
     - Columns of U define the "Hole" Natural Transition Orbitals (donor / particle origin)
     - Columns of V define the "Particle" Natural Transition Orbitals (acceptor / destination)
     - Diagonal values Sigma_k^2 give the transition weights (exciton fraction, sum_k Sigma_k^2 = 1.0)
3. Hole and Particle NTO expansion coefficients in the Atomic Orbital (AO) basis for 3D visualization.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np


class NTOPair(NamedTuple):
    state_index: int
    pair_index: int              # 1 for dominant pair, 2 for secondary, etc.
    eigenvalue_weight: float     # lambda_k = sigma_k^2 (typically 0.70 - 0.99 for dominant pair)
    hole_mo_weights: Dict[int, float]      # Occupied MO index -> weight in hole
    particle_mo_weights: Dict[int, float]  # Virtual MO index -> weight in particle
    hole_ao_coeff: Optional[np.ndarray] = None      # Shape (n_ao,)
    particle_ao_coeff: Optional[np.ndarray] = None  # Shape (n_ao,)


class ExcitedStateInfo(NamedTuple):
    state_index: int             # 1 for S1, 2 for S2, etc.
    multiplicity: str            # "Singlet" or "Triplet"
    energy_ev: float
    energy_nm: float
    oscillator_strength: float   # f
    nto_pairs: List[NTOPair]


class NTOResult(NamedTuple):
    excited_states: List[ExcitedStateInfo]
    n_occupied: int
    n_virtual: int


def compute_ntos_from_amplitudes(
    amplitudes: List[Tuple[int, int, float]],
    n_occ: int,
    n_virt: int,
    state_idx: int = 1,
    multiplicity: str = "Singlet",
    energy_ev: float = 3.0,
    oscillator_strength: float = 0.1,
    mo_coeff: Optional[np.ndarray] = None,
) -> ExcitedStateInfo:
    """
    Computes Natural Transition Orbitals via SVD of the transition matrix T:
        T_{i, a} = amplitude(i -> a)
    """
    if n_occ <= 0 or n_virt <= 0:
        raise ValueError("Must have at least 1 occupied and 1 virtual orbital.")

    # 1. Assemble transition matrix T of shape (n_occ, n_virt)
    T = np.zeros((n_occ, n_virt), dtype=np.float64)
    for i_occ, a_virt, amp in amplitudes:
        if 0 <= i_occ < n_occ and 0 <= a_virt < n_virt:
            T[i_occ, a_virt] = amp

    # Normalize T if sum of squares != 1.0
    norm_t = np.linalg.norm(T)
    if norm_t > 1e-6:
        T = T / norm_t

    # 2. Singular Value Decomposition: T = U @ diag(s) @ Vt
    U, s, Vt = np.linalg.svd(T, full_matrices=False)
    V = Vt.T  # Columns of V are particle orbitals

    weights = s ** 2  # Eigenvalue fractions

    nto_pairs: List[NTOPair] = []
    n_pairs = min(len(weights), 3)  # Keep top 3 NTO pairs

    for k in range(n_pairs):
        w_k = float(weights[k])
        if w_k < 0.01 and k > 0:
            continue

        u_k = U[:, k]  # Hole vector in occupied MO basis
        v_k = V[:, k]  # Particle vector in virtual MO basis

        hole_dict = {i: float(u_k[i]) for i in range(n_occ) if abs(u_k[i]) > 0.05}
        part_dict = {a: float(v_k[a]) for a in range(n_virt) if abs(v_k[a]) > 0.05}

        # AO projection if full MO coefficient matrix is provided
        hole_ao = None
        part_ao = None
        if mo_coeff is not None and mo_coeff.shape[1] >= (n_occ + n_virt):
            c_occ = mo_coeff[:, :n_occ]
            c_virt = mo_coeff[:, n_occ : n_occ + n_virt]
            hole_ao = c_occ @ u_k
            part_ao = c_virt @ v_k

        nto_pairs.append(
            NTOPair(
                state_index=state_idx,
                pair_index=k + 1,
                eigenvalue_weight=w_k,
                hole_mo_weights=hole_dict,
                particle_mo_weights=part_dict,
                hole_ao_coeff=hole_ao,
                particle_ao_coeff=part_ao,
            )
        )

    energy_nm = 1239.84193 / energy_ev if energy_ev > 0 else 0.0

    return ExcitedStateInfo(
        state_index=state_idx,
        multiplicity=multiplicity,
        energy_ev=energy_ev,
        energy_nm=energy_nm,
        oscillator_strength=oscillator_strength,
        nto_pairs=nto_pairs,
    )


def parse_td_dft_and_compute_ntos(
    log_text: str,
    mo_coeff: Optional[np.ndarray] = None,
    n_occ: Optional[int] = None,
    n_virt: Optional[int] = None,
) -> NTOResult:
    """
    Parses TD-DFT / CIS excitation energies, oscillator strengths, and transition amplitudes
    from ORCA or Gaussian output text and computes NTOs for each excited state.
    """
    states: List[ExcitedStateInfo] = []

    # 1. ORCA Parser: "STATE  1:  E=   0.142340 au      3.873 eV    312.4 nm  f=0.0423"
    # Transition lines: " 15a ->  16a  :     0.94212"
    orca_pattern = re.compile(
        r'STATE\s+(\d+):\s+E=\s+([+-]?\d+\.\d+)\s+au\s+([+-]?\d+\.\d+)\s+eV\s+([+-]?\d+\.\d+)\s+nm.*?f=\s*([+-]?\d+\.\d+)',
        re.DOTALL
    )

    # Simple heuristic to discover n_occ if not provided: find max occupied MO index mentioned
    detected_occ = 1
    detected_virt = 1

    matches = list(orca_pattern.finditer(log_text))
    if matches:
        # First pass: collect raw transitions across states to determine global n_occ and n_virt
        raw_states = []
        for idx, m in enumerate(matches[:20]):
            st_num = int(m.group(1))
            e_ev = float(m.group(3))
            f_osc = float(m.group(5))

            start_pos = m.end()
            end_pos = min(len(log_text), start_pos + 2000)
            block = log_text[start_pos:end_pos]

            raw_amps: List[Tuple[int, int, float]] = []
            for line in block.splitlines():
                if "STATE" in line:
                    break
                line_m = re.search(r'(\d+)[a-zA-Z]?\s*->\s*(\d+)[a-zA-Z]?\s*:\s*([+-]?\d+\.\d+)', line)
                if line_m:
                    # ORCA uses 0-based orbital numbering (e.g. 25a -> 26a)
                    i_from = int(line_m.group(1))
                    a_to = int(line_m.group(2))
                    coeff_val = float(line_m.group(3))
                    # Check if explicit (c= ...) amplitude is given on the line
                    c_m = re.search(r'\(c=\s*([+-]?\d+\.\d+)\)', line)
                    amp = float(c_m.group(1)) if c_m else math.sqrt(abs(coeff_val))
                    detected_occ = max(detected_occ, i_from + 1)
                    raw_amps.append((i_from, a_to, amp))

            if raw_amps:
                raw_states.append((st_num, "Singlet", e_ev, f_osc, raw_amps))

        occ_cnt = n_occ or detected_occ
        max_virt_idx = max((a_to - occ_cnt for _, _, _, _, r_amps in raw_states for _, a_to, _ in r_amps), default=0)
        virt_cnt = n_virt or max(10, max_virt_idx + 1)

        for st_num, mult_str, e_ev, f_osc, raw_amps in raw_states:
            clean_amps = [(i_from, max(0, a_to - occ_cnt), amp) for i_from, a_to, amp in raw_amps]
            info = compute_ntos_from_amplitudes(
                clean_amps,
                n_occ=occ_cnt,
                n_virt=virt_cnt,
                state_idx=st_num,
                multiplicity=mult_str,
                energy_ev=e_ev,
                oscillator_strength=f_osc,
                mo_coeff=mo_coeff,
            )
            states.append(info)

    # 2. Gaussian Parser: "Excited State   1:      Singlet-A      3.5421 eV  350.03 nm  f=0.1245 <S**2>=0.000"
    # Transition lines: "     18 -> 19         0.68412"
    if not states:
        gauss_pattern = re.compile(
            r'Excited State\s+(\d+):\s+([A-Za-z0-9\-\?]+)\s+([+-]?\d+\.\d+)\s+eV\s+([+-]?\d+\.\d+)\s+nm\s+f=([+-]?\d+\.\d+)'
        )
        g_matches = list(gauss_pattern.finditer(log_text))
        if g_matches:
            raw_states = []
            for idx, m in enumerate(g_matches[:20]):
                st_num = int(m.group(1))
                mult_label = m.group(2)
                mult_str = "Triplet" if "Triplet" in mult_label else "Singlet"
                e_ev = float(m.group(3))
                f_osc = float(m.group(5))

                start_pos = m.end()
                end_pos = g_matches[idx + 1].start() if (idx + 1) < len(g_matches) else min(len(log_text), start_pos + 2000)
                block = log_text[start_pos:end_pos]

                raw_amps = []
                for line in block.splitlines():
                    line_m = re.search(r'^\s*(\d+)[a-zA-Z]?\s*(?:->|<-)\s*(\d+)[a-zA-Z]?\s+([+-]?\d+\.\d+)', line)
                    if line_m:
                        # Gaussian uses 1-based orbital numbering
                        i_from = int(line_m.group(1)) - 1
                        a_to = int(line_m.group(2)) - 1
                        amp = float(line_m.group(3))
                        detected_occ = max(detected_occ, i_from + 1)
                        raw_amps.append((i_from, a_to, amp))

                if raw_amps:
                    raw_states.append((st_num, mult_str, e_ev, f_osc, raw_amps))

            occ_cnt = n_occ or detected_occ
            max_virt_idx = max((a_to - occ_cnt for _, _, _, _, r_amps in raw_states for _, a_to, _ in r_amps), default=0)
            virt_cnt = n_virt or max(10, max_virt_idx + 1)

            for st_num, mult_str, e_ev, f_osc, raw_amps in raw_states:
                clean_amps = [(i_from, max(0, a_to - occ_cnt), amp) for i_from, a_to, amp in raw_amps]
                info = compute_ntos_from_amplitudes(
                    clean_amps,
                    n_occ=occ_cnt,
                    n_virt=virt_cnt,
                    state_idx=st_num,
                    multiplicity=mult_str,
                    energy_ev=e_ev,
                    oscillator_strength=f_osc,
                    mo_coeff=mo_coeff,
                )
                states.append(info)
                detected_virt = virt_cnt

    return NTOResult(
        excited_states=states,
        n_occupied=n_occ or detected_occ,
        n_virtual=n_virt or detected_virt,
    )
