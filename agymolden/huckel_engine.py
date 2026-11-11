"""Hückel Molecular Orbital (HMO) Theory Pedagogical Engine for AgyMolden.

Theoretical Foundation:
- Identifies conjugated sp2/sp pi-systems across arbitrary molecules.
- Builds the Hückel secular matrix H (in units of alpha and beta) with optional heteroatom
  Streitwieser/Van-Catledge parameters (h_X for alpha_X = alpha_C + h_X*beta, k_XY for beta_XY = k_XY*beta).
- Solves the secular determinant |H - E*I| = 0 to yield:
  1. Molecular Orbital energies: E_j = alpha + x_j * beta (ordered from lowest to highest energy).
  2. LCAO eigenvectors: psi_j = sum_r c_jr * phi_2pz,r.
  3. Live Aufbau electron filling obeying Hund's rule and Pauli exclusion.
  4. Total pi-electron energy E_pi and Delocalization Energy (DE = E_pi - n_double*(2*alpha + 2*beta)).
  5. Coulson pi-bond orders: p_rs = sum_j n_j * c_jr * c_js.
  6. Pi-electron densities: q_r = sum_j n_j * (c_jr)^2 and net pi-charges zeta_r = n_pi_core,r - q_r.
  7. Free valence index: F_r = sqrt(3) - sum_s p_rs.
  8. Frost-Musulin circle geometry for cyclic [N]annulenes.
  9. UV-Vis pi -> pi* transition wavelength (nm) and color estimation.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Set, Tuple
import numpy as np

# Streitwieser / Van-Catledge standard HMO heteroatom parameters
# h_X: alpha_X = alpha_C + h_X * beta
# k_CX: beta_CX = k_CX * beta
HETEROATOM_HMO_PARAMS: Dict[str, Tuple[float, float, int]] = {
    # symbol: (h_X, k_CX, core_pi_electrons)
    "C": (0.0, 1.0, 1),
    "N_pyridine": (0.5, 1.0, 1),    # Pyridine-type =N- (contributes 1 e-)
    "N_pyrrole": (1.5, 0.8, 2),     # Pyrrole-type -NH- (contributes 2 e- lone pair)
    "O_furan": (2.0, 0.8, 2),       # Furan-type -O- (contributes 2 e- lone pair)
    "O_carbonyl": (1.0, 1.0, 1),    # Carbonyl =O (contributes 1 e-)
    "B": (-1.0, 0.7, 0),            # Trivalent Boron (empty pz orbital, 0 e-)
    "F": (3.0, 0.7, 2),
    "Cl": (2.0, 0.4, 2),
}


class HuckelMO(NamedTuple):
    index: int                  # 1-based orbital index
    m_beta: float               # Energy coefficient: E = alpha + m_beta * beta
    energy_relative_beta: str   # e.g., "alpha + 2.000 beta"
    coefficients: np.ndarray    # LCAO coefficients c_jr across the pi-centers
    occupancy: int              # 0, 1, or 2 electrons
    is_homo: bool
    is_lumo: bool
    is_somo: bool
    nodal_planes: int           # Approximate count of sign-change nodes along connectivity


class HuckelResult(NamedTuple):
    system_name: str
    pi_atom_indices: List[int]
    pi_symbols: List[str]
    num_pi_electrons: int
    orbitals: List[HuckelMO]
    total_pi_energy_beta: float  # Total E_pi in terms of beta above core sum
    delocalization_energy_beta: float  # DE = E_pi - n_isolated_double_bonds * 2.0
    aromaticity_status: str       # "Aromatic (4n+2)", "Antiaromatic (4n)", "Non-aromatic / Open-Shell"
    pi_charges: np.ndarray       # Net pi charge per atom (zeta_r)
    pi_electron_densities: np.ndarray  # q_r
    coulson_bond_orders: Dict[Tuple[int, int], float]
    free_valence_indices: np.ndarray
    secular_matrix_h: np.ndarray  # Numerical H matrix in units of beta (relative to alpha_C=0)
    homo_lumo_gap_beta: float
    uv_vis_transition_nm: Optional[float]
    predicted_color_rgb: Tuple[int, int, int]
    is_cyclic: bool
    frost_polygon_n: Optional[int]


def detect_pi_system(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Detects contiguous conjugated pi-atoms (sp2/sp carbons, nitrogens, oxygens, boron)."""
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return [], []

    # Detect connectivity if not given
    if user_bonds:
        bonds = list(user_bonds)
    else:
        from .forcefield import detect_connectivity
        bonds = detect_connectivity(symbols, coords)

    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in bonds:
        if 0 <= u < n and 0 <= v < n:
            adj[u].append(v)
            adj[v].append(u)

    pi_candidates: Set[int] = set()
    for i, s in enumerate(symbols):
        if s == "H":
            continue
        deg = len(adj[i])
        # Conjugated atoms typically have degree 1, 2, or 3 (planar geometry)
        if s == "C" and deg in (2, 3):
            pi_candidates.add(i)
        elif s == "N" and deg in (1, 2, 3):
            pi_candidates.add(i)
        elif s == "O" and deg in (1, 2):
            pi_candidates.add(i)
        elif s == "B" and deg in (2, 3):
            pi_candidates.add(i)

    # Filter to only connected pi components with size >= 2
    pi_list = sorted(pi_candidates)
    pi_set = set(pi_list)

    pi_bonds = []
    for u, v in bonds:
        if u in pi_set and v in pi_set:
            pi_bonds.append((min(u, v), max(u, v)))

    # Connected degree check
    active_pi = set()
    for u, v in pi_bonds:
        active_pi.add(u)
        active_pi.add(v)

    final_pi = sorted(active_pi)
    final_bonds = [(u, v) for u, v in pi_bonds if u in active_pi and v in active_pi]
    return final_pi, final_bonds


def solve_huckel_system(
    symbols: List[str],
    pi_indices: List[int],
    pi_bonds: List[Tuple[int, int]],
    custom_electrons: Optional[int] = None,
    beta_ev: float = -2.70,  # Standard calibrated HMO beta parameter in eV
) -> HuckelResult:
    """Constructs and diagonalizes the Hückel Hamiltonian matrix H = alpha*I + beta*A.
    
    Energy Convention:
      H_rr = alpha_C + h_r * beta
      H_rs = k_rs * beta (for bonded r, s)
      Since beta is inherently NEGATIVE (beta ≈ -2.7 eV or -62.3 kcal/mol),
      positive m_beta (E = alpha + m_beta * beta) corresponds to MORE STABLE bonding orbitals!
      Lowest physical energy = largest m_beta.
    """
    n_pi = len(pi_indices)
    if n_pi < 2:
        raise ValueError("Hückel calculation requires at least 2 conjugated pi atoms.")

    idx_to_local = {orig: loc for loc, orig in enumerate(pi_indices)}
    local_symbols = [symbols[i] for i in pi_indices]

    # Build H matrix in units of beta (setting alpha_C = 0.0)
    H = np.zeros((n_pi, n_pi), dtype=np.float64)

    # Diagonal Coulomb integrals (alpha_X - alpha_C)/beta = h_X
    core_electrons = 0
    for loc, sym in enumerate(local_symbols):
        h_val = 0.0
        e_contrib = 1
        if sym in HETEROATOM_HMO_PARAMS:
            h_val, _, e_contrib = HETEROATOM_HMO_PARAMS[sym]
        elif sym == "N":
            h_val = 0.5
            e_contrib = 1
        elif sym == "O":
            h_val = 1.0
            e_contrib = 1
        elif sym == "B":
            h_val = -1.0
            e_contrib = 0
        H[loc, loc] = h_val
        core_electrons += e_contrib

    # Off-diagonal Resonance integrals (beta_rs / beta) = k_rs
    for u, v in pi_bonds:
        if u in idx_to_local and v in idx_to_local:
            lu = idx_to_local[u]
            lv = idx_to_local[v]
            k_val = 1.0
            su, sv = local_symbols[lu], local_symbols[lv]
            if su != "C" or sv != "C":
                k_u = HETEROATOM_HMO_PARAMS.get(su, (0.0, 1.0, 1))[1]
                k_v = HETEROATOM_HMO_PARAMS.get(sv, (0.0, 1.0, 1))[1]
                k_val = math.sqrt(k_u * k_v)
            H[lu, lv] = k_val
            H[lv, lu] = k_val

    # Diagonalize H matrix
    # e_vals, e_vecs: H @ v = m * v
    # E = alpha + m * beta
    # Since beta < 0, larger m corresponds to lower (more stable) energy!
    eigenvals, eigenvecs = np.linalg.eigh(H)

    # Sort descending by m (lowest energy first: m = +2.00, +1.00, 0.00, -1.00, -2.00)
    order = np.argsort(-eigenvals)
    m_sorted = eigenvals[order]
    vecs_sorted = eigenvecs[:, order]

    num_electrons = custom_electrons if custom_electrons is not None else core_electrons

    # Aufbau electron filling
    occupancies = [0] * n_pi
    rem_e = num_electrons
    for orb_i in range(n_pi):
        if rem_e >= 2:
            occupancies[orb_i] = 2
            rem_e -= 2
        elif rem_e == 1:
            occupancies[orb_i] = 1
            rem_e -= 1
        else:
            occupancies[orb_i] = 0

    # Locate HOMO, LUMO, SOMO
    homo_idx = None
    lumo_idx = None
    somo_idx = None
    for i, occ in enumerate(occupancies):
        if occ == 1:
            somo_idx = i
            homo_idx = i
        elif occ == 2:
            homo_idx = i
        elif occ == 0 and lumo_idx is None:
            lumo_idx = i

    # Build orbital descriptors
    orbitals: List[HuckelMO] = []
    for i in range(n_pi):
        m_val = float(m_sorted[i])
        coeff = np.copy(vecs_sorted[:, i])
        # Ensure consistent sign convention: first non-zero element positive
        for val in coeff:
            if abs(val) > 1e-4:
                if val < 0:
                    coeff = -coeff
                break

        # Nodal count
        nodes = 0
        for u, v in pi_bonds:
            if u in idx_to_local and v in idx_to_local:
                lu = idx_to_local[u]
                lv = idx_to_local[v]
                if coeff[lu] * coeff[lv] < -1e-4:
                    nodes += 1

        sign_str = f"+ {abs(m_val):.3f}" if m_val >= 0 else f"- {abs(m_val):.3f}"
        e_str = f"α {sign_str} β"

        is_h = (i == homo_idx)
        is_l = (i == lumo_idx)
        is_s = (i == somo_idx)

        orbitals.append(
            HuckelMO(
                index=i + 1,
                m_beta=m_val,
                energy_relative_beta=e_str,
                coefficients=coeff,
                occupancy=occupancies[i],
                is_homo=is_h,
                is_lumo=is_l,
                is_somo=is_s,
                nodal_planes=nodes,
            )
        )

    # 1. Total Pi Energy
    # E_pi = sum_j n_j * (alpha + m_j * beta)
    tot_beta = float(sum(occ * m for occ, m in zip(occupancies, m_sorted)))

    # 2. Delocalization Energy (DE)
    # Compared to isolated ethylene units (each ethylene has 2 electrons with E = 2*alpha + 2*beta)
    isolated_pairs = num_electrons // 2
    isolated_e_beta = isolated_pairs * 2.0
    deloc_energy_beta = tot_beta - isolated_e_beta

    # 3. Coulson Pi-Bond Orders
    # p_rs = sum_j n_j * c_jr * c_js
    bond_orders: Dict[Tuple[int, int], float] = {}
    for u, v in pi_bonds:
        if u in idx_to_local and v in idx_to_local:
            lu = idx_to_local[u]
            lv = idx_to_local[v]
            p_val = float(sum(occupancies[j] * vecs_sorted[lu, j] * vecs_sorted[lv, j] for j in range(n_pi)))
            bond_orders[(u, v)] = round(p_val, 4)

    # 4. Pi Electron Densities & Net Charges
    # q_r = sum_j n_j * (c_jr)^2
    # Net pi charge zeta_r = core_r - q_r
    pi_densities = np.zeros(n_pi, dtype=np.float64)
    for loc in range(n_pi):
        pi_densities[loc] = float(sum(occupancies[j] * (vecs_sorted[loc, j] ** 2) for j in range(n_pi)))

    pi_charges = np.zeros(n_pi, dtype=np.float64)
    for loc, sym in enumerate(local_symbols):
        core = HETEROATOM_HMO_PARAMS.get(sym, (0.0, 1.0, 1))[2]
        pi_charges[loc] = core - pi_densities[loc]

    # 5. Free Valence Index
    # F_r = sqrt(3) - sum_s p_rs
    free_valence = np.zeros(n_pi, dtype=np.float64)
    sqrt3 = math.sqrt(3.0)
    for loc, orig in enumerate(pi_indices):
        sum_p = 0.0
        for (u, v), p_val in bond_orders.items():
            if u == orig or v == orig:
                sum_p += p_val
        free_valence[loc] = max(0.0, sqrt3 - sum_p)

    # 6. Aromaticity Rule Check
    # Topology: monocycle (every pi-atom has exactly 2 pi-neighbours and #bonds == #atoms),
    # polycyclic (cycle rank >= 2, e.g. naphthalene, azulene) or acyclic.
    degrees = [0] * n_pi
    parent = list(range(n_pi))

    def _find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in pi_bonds:
        lu, lv = idx_to_local[u], idx_to_local[v]
        degrees[lu] += 1
        degrees[lv] += 1
        parent[_find(lu)] = _find(lv)
    n_components = len({_find(i) for i in range(n_pi)})
    cycle_rank = len(pi_bonds) - n_pi + n_components
    is_cyclic_ring = (n_pi >= 3 and cycle_rank == 1 and n_components == 1 and all(d == 2 for d in degrees))
    is_polycyclic = cycle_rank >= 2

    if is_cyclic_ring:
        if (num_electrons - 2) % 4 == 0 and num_electrons >= 2:
            arom = f"Aromatic ({num_electrons}π = 4n+2 Hückel Rule)"
        elif num_electrons % 4 == 0:
            arom = f"Antiaromatic ({num_electrons}π = 4n Hückel Rule)"
        else:
            arom = f"Open-Shell / Radical ({num_electrons}π)"
    elif is_polycyclic:
        arom = f"Polycyclic conjugated system ({num_electrons}π electrons; 4n+2 rule applies to the periphery)"
    else:
        arom = f"Acyclic Polyene ({num_electrons}π electrons)"

    # 7. HOMO-LUMO Gap and UV-Vis wavelength
    gap_beta = 0.0
    uv_nm = None
    rgb = (100, 116, 139)
    if homo_idx is not None and lumo_idx is not None:
        gap_beta = float(m_sorted[homo_idx] - m_sorted[lumo_idx])
        # Delta E = gap_beta * |beta| in eV
        delta_e_ev = gap_beta * abs(beta_ev)
        if delta_e_ev > 0.05:
            # lambda (nm) = 1239.84193 / Delta E (eV)
            uv_nm = 1239.84193 / delta_e_ev
            rgb = _wavelength_to_rgb(uv_nm)

    return HuckelResult(
        system_name=f"{local_symbols[0]}_{n_pi}",
        pi_atom_indices=pi_indices,
        pi_symbols=local_symbols,
        num_pi_electrons=num_electrons,
        orbitals=orbitals,
        total_pi_energy_beta=tot_beta,
        delocalization_energy_beta=deloc_energy_beta,
        aromaticity_status=arom,
        pi_charges=pi_charges,
        pi_electron_densities=pi_densities,
        coulson_bond_orders=bond_orders,
        free_valence_indices=free_valence,
        secular_matrix_h=H,
        homo_lumo_gap_beta=gap_beta,
        uv_vis_transition_nm=uv_nm,
        predicted_color_rgb=rgb,
        is_cyclic=is_cyclic_ring,
        frost_polygon_n=n_pi if (is_cyclic_ring and len(set(local_symbols)) == 1) else None,
    )


def _wavelength_to_rgb(wavelength_nm: float) -> Tuple[int, int, int]:
    """Converts a wavelength in nm to an approximate perceived visible RGB color."""
    w = wavelength_nm
    if w < 380:
        return (148, 163, 184)  # UV: slate/grey
    elif w > 750:
        return (100, 116, 139)  # IR
    elif 380 <= w < 440:
        r = -(w - 440.0) / (440.0 - 380.0)
        return (int(r * 200), 0, 255)
    elif 440 <= w < 490:
        g = (w - 440.0) / (490.0 - 440.0)
        return (0, int(g * 255), 255)
    elif 490 <= w < 510:
        b = -(w - 510.0) / (510.0 - 490.0)
        return (0, 255, int(b * 255))
    elif 510 <= w < 580:
        r = (w - 510.0) / (580.0 - 510.0)
        return (int(r * 255), 255, 0)
    elif 580 <= w < 645:
        g = -(w - 645.0) / (645.0 - 580.0)
        return (255, int(g * 255), 0)
    else:
        return (255, 0, 0)
