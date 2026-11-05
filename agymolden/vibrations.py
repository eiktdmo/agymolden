"""Vibrational frequency calculation, normal modes analysis, and quantum chemistry import support."""

import logging
import math
from typing import List, Tuple, Dict, Optional
import numpy as np

from .elements import get_cov_radius, estimate_bond_length, ELEMENT_DATA
from .forcefield import detect_connectivity, optimize_geometry_force_field

log = logging.getLogger(__name__)


# =============================================================================
# NIST CCCBDB & Literature Vibrational Scaling Factors
# (https://cccbdb.nist.gov/vibscalex.asp and Kesharwani, Brauer, Martin 2015)
# Harmonic vibrational frequencies systematically overestimate experimental
# fundamental frequencies due to anharmonicity and electronic model deficiencies.
# =============================================================================

NIST_VIBRATIONAL_SCALE_FACTORS: Dict[str, Dict[str, float]] = {
    # Format: method: {basis: factor}
    # Special key "_default" provides the method average
    "b3lyp": {
        "6-31g*": 0.9620,
        "6-31g(d)": 0.9620,
        "6-31g**": 0.9620,
        "6-31g(d,p)": 0.9620,
        "6-31+g**": 0.9603,
        "6-311g*": 0.9608,
        "6-311+g(d,p)": 0.9670,
        "6-311++g(d,p)": 0.9672,
        "6-31g": 0.9649,
        "def2-svp": 0.9650,
        "def2-tzvp": 0.9680,
        "def2-tzvpd": 0.9682,
        "cc-pvdz": 0.9610,
        "cc-pvtz": 0.9660,
        "sto-3g": 0.9850,
        "_default": 0.9620,
    },
    "pbe0": {
        "6-31g*": 0.9520,
        "6-31g(d)": 0.9520,
        "6-311+g(d,p)": 0.9580,
        "def2-svp": 0.9530,
        "def2-tzvp": 0.9580,
        "cc-pvdz": 0.9510,
        "cc-pvtz": 0.9570,
        "_default": 0.9540,
    },
    "pbe": {
        "6-31g*": 0.9870,
        "def2-svp": 0.9880,
        "def2-tzvp": 0.9900,
        "_default": 0.9880,
    },
    "wb97x-d": {
        "6-31g*": 0.9550,
        "6-31+g(d,p)": 0.9560,
        "def2-svp": 0.9540,
        "def2-tzvp": 0.9570,
        "cc-pvdz": 0.9540,
        "cc-pvtz": 0.9580,
        "_default": 0.9560,
    },
    "wb97x": {
        "6-31g*": 0.9560,
        "def2-svp": 0.9550,
        "def2-tzvp": 0.9580,
        "_default": 0.9560,
    },
    "m06-2x": {
        "6-31g*": 0.9520,
        "6-31+g(d,p)": 0.9540,
        "6-311+g(d,p)": 0.9560,
        "def2-svp": 0.9510,
        "def2-tzvp": 0.9560,
        "_default": 0.9530,
    },
    "m06": {
        "6-31g*": 0.9650,
        "def2-svp": 0.9640,
        "def2-tzvp": 0.9670,
        "_default": 0.9650,
    },
    "m06-l": {
        "6-31g*": 0.9700,
        "def2-svp": 0.9680,
        "def2-tzvp": 0.9710,
        "_default": 0.9700,
    },
    "cam-b3lyp": {
        "6-31g*": 0.9600,
        "6-311+g(d,p)": 0.9640,
        "def2-svp": 0.9590,
        "def2-tzvp": 0.9630,
        "_default": 0.9610,
    },
    "b97-d3": {
        "def2-svp": 0.9700,
        "def2-tzvp": 0.9740,
        "6-31g*": 0.9690,
        "_default": 0.9710,
    },
    "hf": {
        "6-31g": 0.9029,
        "6-31g*": 0.8985,
        "6-31g(d)": 0.8985,
        "6-31g**": 0.9026,
        "6-31g(d,p)": 0.9026,
        "6-31+g**": 0.9030,
        "6-311g*": 0.9050,
        "6-311+g(d,p)": 0.9051,
        "def2-svp": 0.9020,
        "def2-tzvp": 0.9060,
        "sto-3g": 0.8710,
        "_default": 0.8990,
    },
    "mp2": {
        "6-31g*": 0.9430,
        "6-31g(d)": 0.9430,
        "6-311+g(d,p)": 0.9500,
        "def2-svp": 0.9450,
        "def2-tzvp": 0.9520,
        "cc-pvdz": 0.9410,
        "cc-pvtz": 0.9500,
        "_default": 0.9450,
    },
    "am1": {
        "_default": 0.9540,
    },
    "pm3": {
        "_default": 0.9740,
    },
    "pm6": {
        "_default": 1.0620,
    },
    "xtb": {
        "_default": 1.0000,
    },
    "gfn2-xtb": {
        "_default": 1.0000,
    },
}

try:
    import json as _json
    import os as _os
    _json_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "nist_scale_factors.json")
    if _os.path.isfile(_json_path):
        with open(_json_path, "r", encoding="utf-8") as _f:
            _loaded_factors = _json.load(_f)
            if isinstance(_loaded_factors, dict):
                for _k, _v in _loaded_factors.items():
                    if isinstance(_v, dict):
                        NIST_VIBRATIONAL_SCALE_FACTORS.setdefault(_k, {}).update(_v)
except Exception as e:
    log.debug("Could not load nist_scale_factors.json: %s", e)


def get_nist_scale_factor(method: str, basis: str = "") -> Tuple[float, str]:
    """
    Look up the recommended vibrational frequency scale factor from NIST CCCBDB / literature.
    
    Returns:
        (scale_factor: float, citation: str)
    """
    m_clean = (method or "").strip().lower()
    b_clean = (basis or "").strip().lower()

    # Normalize method aliases
    if "b3lyp" in m_clean:
        m_key = "b3lyp"
    elif "pbe0" in m_clean or "pbe1pbe" in m_clean:
        m_key = "pbe0"
    elif "wb97x-d" in m_clean or "wb97xd" in m_clean:
        m_key = "wb97x-d"
    elif "wb97x" in m_clean:
        m_key = "wb97x"
    elif "m06-2x" in m_clean or "m062x" in m_clean:
        m_key = "m06-2x"
    elif "m06-l" in m_clean or "m06l" in m_clean:
        m_key = "m06-l"
    elif "m06" in m_clean:
        m_key = "m06"
    elif "cam-b3lyp" in m_clean or "camb3lyp" in m_clean:
        m_key = "cam-b3lyp"
    elif "b97-d3" in m_clean or "b97d3" in m_clean or "b97d" in m_clean:
        m_key = "b97-d3"
    elif "pbe" in m_clean:
        m_key = "pbe"
    elif "hf" in m_clean or "hartree-fock" in m_clean:
        m_key = "hf"
    elif "mp2" in m_clean:
        m_key = "mp2"
    elif "am1" in m_clean:
        m_key = "am1"
    elif "pm3" in m_clean:
        m_key = "pm3"
    elif "pm6" in m_clean:
        m_key = "pm6"
    elif "xtb" in m_clean:
        m_key = "xtb"
    else:
        m_key = m_clean

    # Normalize basis aliases
    b_norm = b_clean.replace(" ", "").replace("*", "*")
    if b_norm in ("6-31g(d)", "6-31gd"):
        b_norm = "6-31g*"
    elif b_norm in ("6-31g(d,p)", "6-31gdp"):
        b_norm = "6-31g**"

    m_table = NIST_VIBRATIONAL_SCALE_FACTORS.get(m_key)
    if not m_table:
        return 1.0000, "Unscaled (Method not in NIST database; default = 1.000)"

    if b_norm in m_table:
        val = m_table[b_norm]
        return val, f"NIST CCCBDB ({method.upper()}/{basis}): λ = {val:.4f}"
    
    # Try default for this method
    default_val = m_table.get("_default", 1.0000)
    return default_val, f"NIST CCCBDB ({method.upper()} average): λ = {default_val:.4f}"



class VibrationalAnalysis:
    """
    Encapsulates normal modes and vibrational frequencies for a molecule.
    Can be generated from force field Hessian or loaded from quantum chemistry outputs.
    """
    def __init__(
        self,
        frequencies: List[float],
        displacements: np.ndarray,  # shape: (n_modes, n_atoms, 3)
        symbols: List[str],
        base_coords: np.ndarray,    # shape: (n_atoms, 3)
        source: str = "Force Field",
        ir_intensities: Optional[List[float]] = None,
        raman_activities: Optional[List[float]] = None,
    ):
        self.frequencies = list(frequencies)
        self.displacements = np.array(displacements, dtype=np.float64)
        self.symbols = list(symbols)
        self.base_coords = np.array(base_coords, dtype=np.float64)
        self.source = source
        self.ir_intensities = list(ir_intensities) if ir_intensities is not None else None
        self.raman_activities = list(raman_activities) if raman_activities is not None else None

    @property
    def n_modes(self) -> int:
        return len(self.frequencies)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def compute_force_field_vibrations(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    optimize_first: bool = False,
    method: str = "uff_mmff",
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None
) -> Tuple[np.ndarray, VibrationalAnalysis]:
    """
    Calculate vibrational frequencies and normal mode displacements using the molecular mechanics
    (`"uff_mmff"`) or Semiempirical Tight-Binding (`"semiempirical"`) potential energy surface.

    Returns:
        (equilibrium_coords, VibrationalAnalysis)
    """
    source_label = "Semiempirical QM (GFN0-xTB)" if method == "semiempirical" else "Analytical UFF/MMFF94s"
    n = len(symbols)
    if n < 2 or len(coords) != n:
        empty_analysis = VibrationalAnalysis([], np.zeros((0, n, 3)), symbols, coords, source=source_label)
        return np.copy(coords), empty_analysis

    # Optionally optimize geometry first so force constants are evaluated at equilibrium minimum
    if optimize_first:
        eq_coords, _ = optimize_geometry_force_field(
            symbols, coords, user_bonds=user_bonds, max_iter=250,
            user_bond_orders=user_bond_orders, method=method
        )
    else:
        eq_coords = np.copy(coords)

    # 1. Build molecular mechanics / semiempirical energy topology
    bonds = detect_connectivity(symbols, eq_coords, user_bonds=user_bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    qm_wiberg_orders = None
    if method == "semiempirical":
        from .semiempirical import compute_semiempirical_properties
        qm_props = compute_semiempirical_properties(symbols, eq_coords, bonds=bonds)
        qm_wiberg_orders = qm_props["wiberg_bond_orders"]

    hybridization: Dict[int, str] = {}
    for i in range(n):
        s = symbols[i]
        deg = len(adj[i])
        if s == "C":
            if deg <= 2:
                hybridization[i] = "sp"
            elif deg == 3:
                hybridization[i] = "sp2"
            else:
                hybridization[i] = "sp3"
        elif s == "N":
            if deg <= 2:
                hybridization[i] = "sp2"
            else:
                nbr_c_sp2 = any(symbols[nb] == "C" and len(adj[nb]) <= 3 for nb in adj[i])
                nbr_o = any(symbols[nb] == "O" for nb in adj[i])
                hybridization[i] = "sp2" if (nbr_c_sp2 or nbr_o) else "sp3"
        elif s == "O":
            hybridization[i] = "sp2" if deg == 1 else "sp3"
        elif s == "S":
            if deg <= 2:
                hybridization[i] = "sp3"
            elif deg <= 4:
                hybridization[i] = "sp3d"
            else:
                hybridization[i] = "sp3d2"
        elif s == "P":
            hybridization[i] = "sp3" if deg <= 3 else "sp3d"
        elif s == "B":
            hybridization[i] = "sp2" if deg <= 3 else "sp3"
        else:
            hybridization[i] = "sp3"

    from .aromatic import find_delocalized_bonds
    deloc_bonds = set(find_delocalized_bonds(symbols, eq_coords, bonds, user_bond_orders).keys())

    # Bond stretching parameters
    bond_data = []
    for i, j in bonds:
        s_i, s_j = symbols[i], symbols[j]
        hyb_i, hyb_j = hybridization[i], hybridization[j]
        pair = tuple(sorted([s_i, s_j]))
        b_key = (min(i, j), max(i, j))
        order = user_bond_orders.get(b_key, 1) if user_bond_orders else 1

        is_aro_bond = b_key in deloc_bonds
        if pair == ("C", "C"):
            if order == 3 or (hyb_i == "sp" and hyb_j == "sp"):
                r0, k_b = 1.205, 820.0
            elif is_aro_bond:
                r0, k_b = 1.392, 550.0
            elif order == 2:
                r0, k_b = 1.335, 600.0
            elif (hyb_i == "sp" or hyb_j == "sp") and (hyb_i != "sp3" and hyb_j != "sp3"):
                r0, k_b = 1.280, 650.0
            elif hyb_i == "sp2" and hyb_j == "sp2":
                r0, k_b = 1.390, 550.0
            elif (hyb_i == "sp2" and hyb_j == "sp3") or (hyb_i == "sp3" and hyb_j == "sp2"):
                r0, k_b = 1.505, 400.0
            else:
                r0, k_b = 1.535, 350.0
        elif pair == ("C", "H"):
            c_idx = i if s_i == "C" else j
            c_hyb = hybridization[c_idx]
            if c_hyb == "sp":
                r0 = 1.060
            elif c_hyb == "sp2":
                r0 = 1.085
            else:
                r0 = 1.095
            k_b = 370.0
        elif pair == ("H", "O"):
            r0, k_b = 0.960, 450.0
        elif pair == ("H", "N"):
            r0, k_b = 1.010, 420.0
        elif pair == ("H", "S"):
            r0, k_b = 1.340, 350.0
        elif pair == ("H", "P"):
            r0, k_b = 1.420, 320.0
        elif pair == ("C", "O"):
            c_idx = i if s_i == "C" else j
            o_idx = j if s_i == "C" else i
            if order == 2 or (len(adj[o_idx]) == 1 and hybridization[c_idx] in ("sp", "sp2")):
                r0, k_b = 1.220, 720.0
            elif hybridization[c_idx] == "sp2":
                r0, k_b = 1.395, 420.0
            else:
                r0, k_b = 1.430, 380.0
        elif pair == ("C", "N"):
            c_idx = i if s_i == "C" else j
            n_idx = j if s_i == "C" else i
            if order == 3 or (len(adj[n_idx]) == 1 and hybridization[c_idx] == "sp"):
                r0, k_b = 1.157, 850.0
            elif order == 2 or (hybridization[c_idx] == "sp2" and hybridization[n_idx] == "sp2"):
                has_carbonyl = any(symbols[nb] == "O" and len(adj[nb]) == 1 for nb in adj[c_idx])
                if has_carbonyl:
                    r0, k_b = 1.340, 520.0
                else:
                    r0, k_b = 1.350, 550.0
            else:
                r0, k_b = 1.470, 380.0
        elif pair == ("N", "O"):
            o_idx = j if s_i == "N" else i
            r0, k_b = (1.220, 700.0) if len(adj[o_idx]) == 1 else (1.400, 380.0)
        elif pair == ("N", "N"):
            if hyb_i == "sp" or hyb_j == "sp":
                r0, k_b = 1.120, 850.0
            elif hyb_i == "sp2" and hyb_j == "sp2":
                r0, k_b = 1.250, 650.0
            else:
                r0, k_b = 1.420, 350.0
        elif pair == ("O", "S"):
            o_idx = j if s_i == "S" else i
            r0, k_b = (1.440, 650.0) if len(adj[o_idx]) == 1 else (1.580, 350.0)
        elif pair == ("O", "P"):
            o_idx = j if s_i == "P" else i
            r0, k_b = (1.480, 700.0) if len(adj[o_idx]) == 1 else (1.600, 350.0)
        elif pair == ("C", "S"):
            c_idx = i if s_i == "C" else j
            s_idx = j if s_i == "C" else i
            if len(adj[s_idx]) == 1 and hybridization[c_idx] == "sp2":
                r0, k_b = 1.620, 520.0
            else:
                r0, k_b = 1.815, 320.0
        elif pair == ("C", "P"):
            r0, k_b = 1.850, 300.0
        elif pair == ("C", "F"):
            r0, k_b = 1.350, 450.0
        elif pair == ("C", "Cl"):
            r0, k_b = 1.770, 350.0
        elif pair == ("C", "Br"):
            r0, k_b = 1.940, 300.0
        elif pair == ("C", "I"):
            r0, k_b = 2.140, 250.0
        elif pair == ("C", "Si"):
            r0, k_b = 1.870, 300.0
        elif pair == ("C", "B"):
            r0, k_b = 1.560, 350.0
        else:
            r0 = estimate_bond_length(s_i, s_j)
            k_b = 350.0

        if method == "semiempirical" and qm_wiberg_orders and s_i != "H" and s_j != "H":
            wbo = qm_wiberg_orders.get(b_key, 1.0)
            if 0.85 <= wbo <= 3.1:
                k_b = k_b * (0.85 + 0.15 * wbo)

        bond_data.append((i, j, r0, k_b))

    # Angle bending and out-of-plane parameters
    angle_data = []
    oop_data = []
    for apex in range(n):
        nbrs = adj[apex]
        deg = len(nbrs)
        if deg < 2:
            continue

        sym = symbols[apex]
        hyb = hybridization[apex]

        if sym == "C":
            if hyb == "sp":
                th0, k_a = 180.0, 90.0
            elif hyb == "sp2":
                th0, k_a = 120.0, 90.0
                if deg == 3:
                    oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 150.0))
            else:
                th0, k_a = 109.47, 60.0
        elif sym == "N":
            if hyb == "sp2":
                th0, k_a = 120.0, 75.0
                if deg == 3:
                    oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 120.0))
            else:
                th0, k_a = 107.0, 60.0
        elif sym == "O":
            th0, k_a = 104.5, 65.0
        elif sym == "S":
            th0, k_a = (98.0, 60.0) if deg == 2 else (109.5, 65.0)
        elif sym == "P":
            th0, k_a = (96.0, 55.0) if deg == 3 else (109.5, 65.0)
        elif sym == "B":
            th0, k_a = 120.0, 80.0
            if deg == 3:
                oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 120.0))
        elif sym == "Si":
            th0, k_a = 109.47, 55.0
        else:
            th0, k_a = 109.5, 50.0

        for a_idx in range(deg):
            for b_idx in range(a_idx + 1, deg):
                angle_data.append((nbrs[a_idx], apex, nbrs[b_idx], th0, k_a))

    # Torsional parameters
    torsion_data = []
    for b, c in bonds:
        hyb_b, hyb_c = hybridization[b], hybridization[c]
        if hyb_b == "sp" or hyb_c == "sp":
            continue
        nbrs_b = [x for x in adj[b] if x != c]
        nbrs_c = [y for y in adj[c] if y != b]
        if not nbrs_b or not nbrs_c:
            continue
        n_quads = max(1, len(nbrs_b) * len(nbrs_c))
        if hyb_b == "sp2" and hyb_c == "sp2":
            n_fold, cos_n_phi0, v_tot = 2, 1.0, 14.0
        elif hyb_b == "sp3" and hyb_c == "sp3":
            n_fold, cos_n_phi0, v_tot = 3, -1.0, 2.2
        else:
            n_fold, cos_n_phi0, v_tot = 3, -1.0, 1.0
        v_per_quad = v_tot / n_quads
        for a in nbrs_b:
            for d in nbrs_c:
                if a != d:
                    torsion_data.append((a, b, c, d, n_fold, cos_n_phi0, v_per_quad))

    # Non-bonded parameters
    bonded_set = set((min(i, j), max(i, j)) for i, j in bonds)
    one_three_set = set((min(i, k), max(i, k)) for i, apex, k, _, _ in angle_data)
    nonbonded_data = []
    for i in range(n):
        r_i = get_cov_radius(symbols[i])
        for j in range(i + 1, n):
            pair = (i, j)
            if pair in bonded_set or pair in one_three_set:
                continue
            r_j = get_cov_radius(symbols[j])
            sigma = (r_i + r_j) * 1.12
            eps = 0.15
            nonbonded_data.append((i, j, sigma, eps))

    # Term index/parameter arrays, built once, so the energy can be evaluated
    # for many displaced geometries at once with numpy (same potential as before).
    def _arr(rows, col, dtype=float):
        return np.array([r[col] for r in rows], dtype=dtype)

    b_i, b_j = _arr(bond_data, 0, int), _arr(bond_data, 1, int)
    b_r0, b_k = _arr(bond_data, 2), _arr(bond_data, 3)
    a_i, a_c, a_k = _arr(angle_data, 0, int), _arr(angle_data, 1, int), _arr(angle_data, 2, int)
    a_th0, a_ka = _arr(angle_data, 3), _arr(angle_data, 4)
    o_apex, o_i = _arr(oop_data, 0, int), _arr(oop_data, 1, int)
    o_j, o_k, o_kk = _arr(oop_data, 2, int), _arr(oop_data, 3, int), _arr(oop_data, 4)
    t_a, t_b, t_c, t_d = (_arr(torsion_data, c, int) for c in range(4))
    t_n, t_cos, t_v = _arr(torsion_data, 4), _arr(torsion_data, 5), _arr(torsion_data, 6)
    n_i, n_j = _arr(nonbonded_data, 0, int), _arr(nonbonded_data, 1, int)
    n_sigma, n_eps = _arr(nonbonded_data, 2), _arr(nonbonded_data, 3)

    def _norm_sq(v):
        return np.einsum("...i,...i->...", v, v)

    def _energy_chunk(P: np.ndarray) -> np.ndarray:
        """Potential energy for a stack of geometries P with shape (B, n, 3)."""
        B = P.shape[0]
        energy = np.zeros(B)

        if len(b_i):  # bond stretching
            d = np.sqrt(_norm_sq(P[:, b_i] - P[:, b_j]) + 1e-12)
            energy += (0.5 * b_k * (d - b_r0) ** 2).sum(axis=1)

        if len(a_i):  # angle bending
            v1 = P[:, a_i] - P[:, a_c]
            v2 = P[:, a_k] - P[:, a_c]
            n1 = np.sqrt(_norm_sq(v1) + 1e-12)
            n2 = np.sqrt(_norm_sq(v2) + 1e-12)
            cos_th = np.clip(np.einsum("...i,...i->...", v1, v2) / (n1 * n2), -1.0, 1.0)
            th = np.degrees(np.arccos(cos_th))
            d_rad = np.radians(th - a_th0)
            energy += (0.5 * a_ka * d_rad * d_rad).sum(axis=1)

        if len(o_apex):  # out-of-plane planar penalty
            v_ij = P[:, o_j] - P[:, o_i]
            v_ik = P[:, o_k] - P[:, o_i]
            nrm = np.cross(v_ij, v_ik)
            norm_n = np.sqrt(_norm_sq(nrm) + 1e-12)
            h_oop = np.einsum("...i,...i->...", P[:, o_apex] - P[:, o_i], nrm) / norm_n
            energy += (0.5 * o_kk * h_oop * h_oop).sum(axis=1)

        if len(t_a):  # torsions
            b1 = P[:, t_b] - P[:, t_a]
            b2 = P[:, t_c] - P[:, t_b]
            b3 = P[:, t_d] - P[:, t_c]
            norm_b2 = np.sqrt(_norm_sq(b2) + 1e-16)
            m = np.cross(b1, b2)
            n_pl = np.cross(b2, b3)
            valid = (_norm_sq(m) >= 1e-10) & (_norm_sq(n_pl) >= 1e-10)
            y = np.einsum("...i,...i->...", np.cross(m, n_pl), b2) / norm_b2
            x_dot = np.einsum("...i,...i->...", m, n_pl)
            phi = np.arctan2(y, x_dot)
            e_t = 0.5 * t_v * (1.0 - t_cos * np.cos(t_n * phi))
            energy += np.where(valid, e_t, 0.0).sum(axis=1)

        if len(n_i):  # steric non-bonded
            d = np.sqrt(_norm_sq(P[:, n_i] - P[:, n_j]) + 1e-12)
            ratio = n_sigma / np.maximum(0.4, d)
            r6 = ratio ** 6
            e_nb = n_eps * (r6 * r6 - r6)
            energy += np.where(d < n_sigma * 1.5, e_nb, 0.0).sum(axis=1)

        return energy

    # keep every temporary array under ~64 MB
    n_terms = max(1, len(bond_data) + len(angle_data) + len(oop_data) + len(torsion_data) + len(nonbonded_data))
    chunk = int(max(8, min(4096, 4_000_000 // (n_terms * 3))))

    def potential_energy_batch(P: np.ndarray) -> np.ndarray:
        out = np.empty(P.shape[0])
        for s0 in range(0, P.shape[0], chunk):
            out[s0:s0 + chunk] = _energy_chunk(P[s0:s0 + chunk])
        return out

    # 2. Numerical Hessian H_kl = d^2 E / (dx_k dx_l) from the same central-difference
    #    stencil as before: E at x +/- h e_k +/- h e_l.
    dim = 3 * n
    x0 = eq_coords.flatten()
    h = 1e-4
    ar = np.arange(dim)
    H = np.zeros((dim, dim))
    for k in range(dim):
        X = np.empty((4, dim, dim))  # [sign_k, sign_l] stacks, one row per l
        for q, (sk, sl) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            blk = np.tile(x0, (dim, 1))
            blk[:, k] += sk * h
            blk[ar, ar] += sl * h
            X[q] = blk
        E = potential_energy_batch(X.reshape(4 * dim, n, 3)).reshape(4, dim)
        gp = (E[0] - E[1]) / (2.0 * h)   # gradient at x + h e_k
        gm = (E[2] - E[3]) / (2.0 * h)   # gradient at x - h e_k
        H[k, :] = (gp - gm) / (2.0 * h)

    # Symmetrize Hessian
    H = 0.5 * (H + H.T)

    # 3. Mass weighting: H_mw = M^(-1/2) * H * M^(-1/2)
    masses = [ELEMENT_DATA.get(s, {}).get("mass", 12.0) for s in symbols]
    mass_vec = np.repeat(masses, 3)
    inv_sqrt_m = 1.0 / np.sqrt(np.maximum(1e-4, mass_vec))
    H_mw = H * np.outer(inv_sqrt_m, inv_sqrt_m)

    # 4. Diagonalize mass-weighted Hessian
    eigvals, eigvecs = np.linalg.eigh(H_mw)

    # Sort in ascending order
    idx_sorted = np.argsort(eigvals)
    eigvals = eigvals[idx_sorted]
    eigvecs = eigvecs[:, idx_sorted]

    freq_conv = 108.59135859237747

    displacements = np.zeros((dim, n, 3))
    frequencies = []

    for mode_idx in range(dim):
        val = float(eigvals[mode_idx])
        if val >= 0:
            freq = freq_conv * math.sqrt(val)
        else:
            freq = -freq_conv * math.sqrt(-val)
        frequencies.append(freq)

        mode_mw = eigvecs[:, mode_idx]
        mode_cart = mode_mw * inv_sqrt_m
        norm_cart = np.linalg.norm(mode_cart)
        if norm_cart > 1e-12:
            mode_cart /= norm_cart
        displacements[mode_idx] = mode_cart.reshape((n, 3))

    analysis = VibrationalAnalysis(
        frequencies=frequencies,
        displacements=displacements,
        symbols=symbols,
        base_coords=eq_coords,
        source=source_label
    )

    return eq_coords, analysis


def assign_normal_mode_motion(
    symbols: List[str],
    coords: np.ndarray,
    disp: np.ndarray,
    freq_cm1: float = 1000.0,
    bonds: Optional[List[Tuple[int, int]]] = None,
) -> str:
    """
    Assign a concise PED-like (Potential Energy Distribution) internal coordinate motion
    label to a normal mode displacement vector `disp` (shape (N, 3)):
      - e.g. 'ν(O1–H2) Stretch' or 'δ(H2–O1–H3) Bend'
    """
    if abs(freq_cm1) <= 15.0:
        return "Translation / Rotation"

    coords_arr = np.asarray(coords, dtype=np.float64)
    disp_arr = np.asarray(disp, dtype=np.float64)
    n = len(symbols)
    if n < 2 or coords_arr.shape != (n, 3) or disp_arr.shape != (n, 3):
        return "Collective Mode"

    if bonds is None:
        bonds = detect_connectivity(symbols, coords_arr, strict=False)
    if not bonds:
        return "Lattice / Non-bonded Mode"

    # 1. Evaluate bond stretch projections |u_ij . (d_j - d_i)|
    best_stretch_val = 0.0
    best_stretch_lbl = ""
    for u, v in bonds:
        if not (0 <= u < n and 0 <= v < n):
            continue
        vec = coords_arr[v] - coords_arr[u]
        r = float(np.linalg.norm(vec))
        if r < 1e-6:
            continue
        u_vec = vec / r
        dr = abs(float(np.dot(u_vec, disp_arr[v] - disp_arr[u])))
        if dr > best_stretch_val:
            best_stretch_val = dr
            best_stretch_lbl = f"ν({symbols[u]}{u + 1}–{symbols[v]}{v + 1}) Stretch"

    # 2. Evaluate bond angle bend changes |d_theta| (in radians * 0.8 Angstrom weight)
    nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in bonds:
        if 0 <= u < n and 0 <= v < n:
            nbrs[u].append(v)
            nbrs[v].append(u)

    best_bend_val = 0.0
    best_bend_lbl = ""
    eps = 0.02
    c_plus = coords_arr + eps * disp_arr
    for c_idx, nb_list in nbrs.items():
        if len(nb_list) < 2:
            continue
        for idx1 in range(len(nb_list)):
            for idx2 in range(idx1 + 1, len(nb_list)):
                a = nb_list[idx1]
                b = nb_list[idx2]
                v1 = coords_arr[a] - coords_arr[c_idx]
                v2 = coords_arr[b] - coords_arr[c_idx]
                n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
                if n1 < 1e-6 or n2 < 1e-6:
                    continue
                th0 = math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2)) / (n1 * n2))))

                vp1 = c_plus[a] - c_plus[c_idx]
                vp2 = c_plus[b] - c_plus[c_idx]
                np1, np2 = float(np.linalg.norm(vp1)), float(np.linalg.norm(vp2))
                if np1 < 1e-6 or np2 < 1e-6:
                    continue
                th1 = math.acos(max(-1.0, min(1.0, float(np.dot(vp1, vp2)) / (np1 * np2))))

                d_th = abs(th1 - th0) / eps
                weighted_bend = d_th * 0.75
                if weighted_bend > best_bend_val:
                    best_bend_val = weighted_bend
                    best_bend_lbl = f"δ({symbols[a]}{a + 1}–{symbols[c_idx]}{c_idx + 1}–{symbols[b]}{b + 1}) Bend"

    prefix = "TS " if freq_cm1 < -5.0 else ""
    if best_stretch_val >= best_bend_val and best_stretch_lbl:
        return prefix + best_stretch_lbl
    if best_bend_lbl:
        return prefix + best_bend_lbl
    return (prefix + best_stretch_lbl) if best_stretch_lbl else "Collective Mode"


# Common isotope masses (amu) for Isotopologue Vibrational Shift & KIE calculations
COMMON_ISOTOPE_MASSES: Dict[str, List[Tuple[str, float]]] = {
    "H": [("¹H (1.008)", 1.007825), ("²D Deuterium (2.014)", 2.014102), ("³T Tritium (3.016)", 3.016049)],
    "C": [("¹²C (12.000)", 12.000000), ("¹³C (13.003)", 13.003355), ("¹⁴C (14.003)", 14.003242)],
    "N": [("¹⁴N (14.003)", 14.003074), ("¹⁵N (15.000)", 15.000109)],
    "O": [("¹⁶O (15.995)", 15.994915), ("¹⁷O (16.999)", 16.999132), ("¹⁸O (17.999)", 17.999161)],
    "S": [("³²S (31.972)", 31.972071), ("³⁴S (33.968)", 33.967867)],
    "Cl": [("³⁵Cl (34.969)", 34.968853), ("³⁷Cl (36.966)", 36.965903)],
    "Br": [("⁷⁹Br (78.918)", 78.918338), ("⁸¹Br (80.916)", 80.916291)],
}


def compute_isotopologue_vibrational_shifts(
    analysis: VibrationalAnalysis,
    custom_masses: Dict[int, float],
    temperature_k: float = 298.15,
) -> Dict[str, object]:
    """
    Compute isotopologue vibrational frequencies (cm^-1), frequency shifts Delta_nu (cm^-1),
    Zero-Point Energy shift Delta_ZPE (kcal/mol), and reduced isotopic partition function ratio
    (Bigeleisen-Mayer RPFR / KIE estimate) by mass-reweighting each normal mode via the Rayleigh
    quotient of the Born-Oppenheimer Cartesian Hessian.
    """
    symbols = analysis.symbols
    n_atoms = len(symbols)
    std_masses = np.array([float(ELEMENT_DATA.get(s, {}).get("mass", 12.0)) for s in symbols], dtype=np.float64)
    iso_masses = std_masses.copy()
    for idx, m_val in (custom_masses or {}).items():
        if 0 <= idx < n_atoms and m_val > 0.1:
            iso_masses[idx] = float(m_val)

    new_freqs: List[float] = []
    shifts: List[float] = []
    for m_i, freq0 in enumerate(analysis.frequencies):
        if m_i < len(analysis.displacements):
            disp = np.asarray(analysis.displacements[m_i], dtype=np.float64)
            # Cartesian displacement norm weighted by standard vs isotopic mass:
            # Since q_mw = M^(1/2) dx, the effective reduced mass ratio is (sum m_std |dx_a|^2) / (sum m_iso |dx_a|^2)
            atom_d2 = np.sum(disp ** 2, axis=1)
            mu_std = float(np.sum(std_masses * atom_d2))
            mu_iso = float(np.sum(iso_masses * atom_d2))
            ratio = math.sqrt(mu_std / max(1e-12, mu_iso)) if mu_std > 1e-12 else 1.0
        else:
            ratio = 1.0
        f_iso = float(freq0) * ratio
        new_freqs.append(round(f_iso, 2))
        shifts.append(round(f_iso - float(freq0), 2))

    # ZPE in kcal/mol: ZPE = 0.5 * sum_{nu > 15} h c nu  (1 cm^-1 = 0.002859144 kcal/mol)
    cm1_to_kcal = 0.002859144
    zpe_std = 0.5 * sum(f for f in analysis.frequencies if f > 15.0) * cm1_to_kcal
    zpe_iso = 0.5 * sum(f for f in new_freqs if f > 15.0) * cm1_to_kcal
    delta_zpe = zpe_iso - zpe_std

    # If an imaginary TS mode (< -15 cm^-1) is present, compute semiclassical Eyring/Bell ZPE KIE k_light / k_heavy
    rt_kcal = 0.001987204 * max(10.0, float(temperature_k))
    kie_zpe = math.exp(-delta_zpe / rt_kcal)
    imag_std = [abs(f) for f in analysis.frequencies if f < -15.0]
    imag_iso = [abs(f) for f in new_freqs if f < -15.0]
    if imag_std and imag_iso:
        kie_tunnel_factor = imag_std[0] / max(1.0, imag_iso[0])
    else:
        kie_tunnel_factor = 1.0

    return {
        "std_frequencies": [round(float(f), 2) for f in analysis.frequencies],
        "iso_frequencies": new_freqs,
        "shifts_cm1": shifts,
        "zpe_std_kcal": round(zpe_std, 4),
        "zpe_iso_kcal": round(zpe_iso, 4),
        "delta_zpe_kcal": round(delta_zpe, 4),
        "kie_estimate": round(kie_tunnel_factor * kie_zpe, 3),
    }

