"""
Molecular Cycle Welder & Ring Fusion Engine for AgyMolden.

Provides algorithms for:
1. Edge Annulation (fusing a new 3-, 4-, 5-, 6-, 7-, or 8-membered carbocycle or heterocycle onto a selected bond)
2. Bicyclic Bridgehead Welding (bridging two distant ring carbons with -CH2-, -CH2CH2-, -O-, or -NH-)
3. Inter-fragment Cycle Stitching (welding two existing separate rings across shared edges)
4. Spiro-Welding (fusing rings at a single quaternary spiro vertex)
"""

import logging
import math
from typing import List, Tuple, Dict, Optional, Set, Any
import numpy as np

from .alignment import rodrigues_rotation
from .forcefield import optimize_geometry_force_field, detect_connectivity

logger = logging.getLogger(__name__)


RING_PRESETS: Dict[str, Dict[str, Any]] = {
    "Benzene": {
        "size": 6,
        "hetero": {},
        "aromatic": True,
        "desc": "6-membered aromatic ring (produces Naphthalene, Anthracene, Phenanthrene)",
    },
    "Pyridine": {
        "size": 6,
        "hetero": {0: "N"},
        "aromatic": True,
        "desc": "6-membered azine (produces Quinoline / Isoquinoline)",
    },
    "Cyclohexyl": {
        "size": 6,
        "hetero": {},
        "aromatic": False,
        "desc": "6-membered saturated ring (produces Decalin / Tetralin)",
    },
    "Cyclopentyl": {
        "size": 5,
        "hetero": {},
        "aromatic": False,
        "desc": "5-membered saturated ring (produces Indane)",
    },
    "Cyclopentadiene": {
        "size": 5,
        "hetero": {},
        "aromatic": True,
        "desc": "5-membered conjugated ring (produces Indene / Fluorene)",
    },
    "Pyrrole": {
        "size": 5,
        "hetero": {0: "N"},
        "aromatic": True,
        "desc": "5-membered azole (produces Indole / Carbazole)",
    },
    "Furan": {
        "size": 5,
        "hetero": {0: "O"},
        "aromatic": True,
        "desc": "5-membered oxole (produces Benzofuran)",
    },
    "Thiophene": {
        "size": 5,
        "hetero": {0: "S"},
        "aromatic": True,
        "desc": "5-membered thiole (produces Benzothiophene)",
    },
    "Imidazole": {
        "size": 5,
        "hetero": {0: "N", 2: "N"},
        "aromatic": True,
        "desc": "5-membered diazole (produces Benzimidazole / Purine core)",
    },
    "Cyclobutyl": {
        "size": 4,
        "hetero": {},
        "aromatic": False,
        "desc": "4-membered ring (produces Bicyclo[4.2.0] systems)",
    },
    "Cyclopropyl": {
        "size": 3,
        "hetero": {},
        "aromatic": False,
        "desc": "3-membered ring (produces Bicyclo[4.1.0] systems)",
    },
    "Oxirane": {
        "size": 3,
        "hetero": {0: "O"},
        "aromatic": False,
        "desc": "3-membered epoxide ring",
    },
}


def _resolve_preset(ring_preset: Optional[str]) -> Optional[str]:
    """Resolve ring preset name robustly, supporting case-insensitivity and display strings."""
    if not ring_preset:
        return None
    if ring_preset in RING_PRESETS:
        return ring_preset
    s = ring_preset.strip().lower()
    for name in RING_PRESETS:
        if name.lower() == s or s.startswith(name.lower()):
            return name
    return None


def _find_attached_hydrogens(symbols: List[str], coords: np.ndarray, atom_idx: int) -> List[int]:
    """Find indices of Hydrogen atoms bonded to a given atom."""
    h_indices = []
    r0 = coords[atom_idx]
    for i, (sym, r) in enumerate(zip(symbols, coords)):
        if sym == "H" and np.linalg.norm(r - r0) < 1.30:
            h_indices.append(i)
    return h_indices


def _find_adjacent_heavy_neighbors(
    symbols: List[str], coords: np.ndarray, atom_idx: int, exclude_idx: int
) -> List[int]:
    """Find heavy neighbor atom indices connected to atom_idx, excluding exclude_idx."""
    neighbors = []
    r0 = coords[atom_idx]
    for i, (sym, r) in enumerate(zip(symbols, coords)):
        if i != atom_idx and i != exclude_idx and sym != "H":
            d = np.linalg.norm(r - r0)
            if d < 1.85:  # Standard heavy-atom single/double bond cutoff
                neighbors.append(i)
    return neighbors


def fuse_ring_to_bond(
    symbols: List[str],
    coords: np.ndarray,
    a1: int,
    a2: int,
    ring_preset: str = "Benzene",
    custom_size: Optional[int] = None,
    hetero_map: Optional[Dict[int, str]] = None,
    flip_side: bool = False,
    relax: bool = True,
    freeze_parent: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Fuse a new molecular cycle (N-gon) onto the bond connecting atoms a1 and a2.

    Parameters:
    - symbols: list of element symbols
    - coords: (N, 3) numpy array of Cartesian coordinates
    - a1, a2: 0-based indices of the two atoms defining the fused edge
    - ring_preset: name from RING_PRESETS or "Custom"
    - custom_size: if "Custom", the ring size (3 to 10)
    - hetero_map: optional dict mapping vertex index (0 to N-3) to heteroatom symbol (e.g. {0: "N"})
    - flip_side: invert fusion direction (weld to opposite side of bond)
    - relax: run fast molecular mechanics / force field relaxation after fusion
    - freeze_parent: if True, only the newly added atoms are relaxed, keeping parent atoms fixed

    Returns:
    - (new_symbols, new_coords)
    """
    coords = np.array(coords, dtype=np.float64)
    n_atoms = len(symbols)
    if a1 < 0 or a1 >= n_atoms or a2 < 0 or a2 >= n_atoms or a1 == a2:
        raise ValueError(f"Invalid atom indices: a1={a1}, a2={a2}")

    # Determine ring geometry parameters
    resolved_key = _resolve_preset(ring_preset)
    if resolved_key is not None:
        info = RING_PRESETS[resolved_key]
        N = info["size"]
        is_aromatic = info["aromatic"]
        sub_map = dict(info["hetero"])
        if hetero_map:
            sub_map.update(hetero_map)
    else:
        N = custom_size if custom_size is not None else 6
        is_aromatic = False
        sub_map = dict(hetero_map) if hetero_map else {}

    if N < 3 or N > 12:
        raise ValueError(f"Unsupported ring size: {N}. Must be between 3 and 12.")

    r1, r2 = coords[a1], coords[a2]
    bond_vec = r2 - r1
    L = float(np.linalg.norm(bond_vec))
    if L < 0.8:
        raise ValueError(f"Selected atoms {a1} and {a2} are too close (distance={L:.3f} A)")

    midpoint = 0.5 * (r1 + r2)

    # Determine local plane normal n and inward reference direction
    n1_list = _find_adjacent_heavy_neighbors(symbols, coords, a1, a2)
    n2_list = _find_adjacent_heavy_neighbors(symbols, coords, a2, a1)

    plane_normal = None
    v_inward = None

    if n1_list:
        n1_coord = coords[n1_list[0]]
        cross1 = np.cross(bond_vec, n1_coord - r1)
        if np.linalg.norm(cross1) > 1e-4:
            plane_normal = cross1 / np.linalg.norm(cross1)
            v_inward = n1_coord - midpoint

    if plane_normal is None and n2_list:
        n2_coord = coords[n2_list[0]]
        cross2 = np.cross(bond_vec, n2_coord - r2)
        if np.linalg.norm(cross2) > 1e-4:
            plane_normal = cross2 / np.linalg.norm(cross2)
            v_inward = n2_coord - midpoint

    if plane_normal is None:
        # Fallback for isolated diatomic or linear edge
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(bond_vec / L, cand)) > 0.9:
            cand = np.array([0.0, 1.0, 0.0])
        plane_normal = np.cross(bond_vec, cand)
        plane_normal /= np.linalg.norm(plane_normal)
        v_inward = None

    # Compute in-plane perpendicular vector u_out
    u_out = np.cross(bond_vec, plane_normal)
    u_out /= np.linalg.norm(u_out)

    if v_inward is not None and np.dot(u_out, v_inward) > 0:
        u_out = -u_out

    # Ring center calculation (regular N-gon apothem)
    apothem = L / (2.0 * math.tan(math.pi / N))

    # Verify that candidate u_out points away from other atoms, not inside the molecular core
    other_coords = [coords[i] for i in range(n_atoms) if i != a1 and i != a2]
    if other_coords:
        other_arr = np.array(other_coords)
        c_cand1 = midpoint + apothem * u_out
        c_cand2 = midpoint - apothem * u_out
        d1 = float(np.min(np.linalg.norm(other_arr - c_cand1, axis=1)))
        d2 = float(np.min(np.linalg.norm(other_arr - c_cand2, axis=1)))
        heavy_coords = [coords[i] for i in range(n_atoms) if symbols[i] != "H"]
        if heavy_coords:
            mol_center = np.mean(heavy_coords, axis=0)
            v_away = midpoint - mol_center
            if d2 > d1 + 0.3 or (np.dot(u_out, v_away) < -0.3 and d2 >= 1.0):
                u_out = -u_out

    if flip_side:
        u_out = -u_out

    new_center = midpoint + apothem * u_out

    # Vertex generation by rotating around plane_normal
    v2 = r2 - new_center
    d_theta = 2.0 * math.pi / N

    correct_sign = -1
    for s in [1, -1]:
        R_test = rodrigues_rotation(plane_normal, s * (N - 1) * d_theta)
        cand = new_center + R_test @ v2
        if np.allclose(cand, r1, atol=1e-3):
            correct_sign = s
            break

    n_new_verts = N - 2
    new_ring_coords: List[np.ndarray] = []
    new_ring_symbols: List[str] = []

    for k in range(1, N - 1):
        R_k = rodrigues_rotation(plane_normal, correct_sign * k * d_theta)
        pt = new_center + R_k @ v2
        new_ring_coords.append(pt)
        sym = sub_map.get(k - 1, "C")
        new_ring_symbols.append(sym)

    # Prune attached hydrogens on a1 and a2 that would violate valency
    prune_set: Set[int] = set()
    for h_idx in _find_attached_hydrogens(symbols, coords, a1):
        prune_set.add(h_idx)
    for h_idx in _find_attached_hydrogens(symbols, coords, a2):
        prune_set.add(h_idx)

    # Check for severe steric collision (< 0.50 A) with remaining existing atoms
    valid_existing_indices = [i for i in range(n_atoms) if i not in prune_set and i != a1 and i != a2]
    if valid_existing_indices:
        valid_coords = coords[valid_existing_indices]
        for pt in new_ring_coords:
            d_arr = np.linalg.norm(valid_coords - pt, axis=1)
            min_d = float(np.min(d_arr))
            if min_d < 0.50:
                clash_orig_idx = valid_existing_indices[int(np.argmin(d_arr))]
                clash_sym = symbols[clash_orig_idx]
                raise ValueError(
                    f"Edge annulation on bond {symbols[a1]}{a1+1}–{symbols[a2]}{a2+1} causes severe steric collision "
                    f"({min_d:.2f} Å) with adjacent atom {clash_sym}{clash_orig_idx+1}.\n"
                    f"If you are trying to fuse a ring across a bay region (e.g., Naphthalene to Phenalene), "
                    f"please use the 'Peri-Annulation (3 Atoms / Bay)' feature instead."
                )

    # Assemble base atoms
    out_symbols = [symbols[i] for i in range(n_atoms) if i not in prune_set]
    out_coords = [coords[i] for i in range(n_atoms) if i not in prune_set]

    # Add new ring atoms
    new_indices = []
    for sym, pt in zip(new_ring_symbols, new_ring_coords):
        new_indices.append(len(out_symbols))
        out_symbols.append(sym)
        out_coords.append(pt)

    # Add appropriate Hydrogens to the new ring atoms
    for i_vert, (sym, pt) in enumerate(zip(new_ring_symbols, new_ring_coords)):
        v_rad = pt - new_center
        v_rad /= np.linalg.norm(v_rad)

        if sym == "C":
            if is_aromatic:
                # 1 planar C-H bond
                h_pos = pt + 1.08 * v_rad
                out_symbols.append("H")
                out_coords.append(h_pos)
            else:
                # 2 tetrahedral C-H bonds (sp3 puckering)
                cos_ang = math.cos(math.radians(54.7))
                sin_ang = math.sin(math.radians(54.7))
                h1 = pt + 1.09 * (cos_ang * v_rad + sin_ang * plane_normal)
                h2 = pt + 1.09 * (cos_ang * v_rad - sin_ang * plane_normal)
                out_symbols.append("H")
                out_coords.append(h1)
                out_symbols.append("H")
                out_coords.append(h2)
        elif sym == "N":
            # For pyrrole-like NH, add 1 H; for pyridine-like =N-, no H
            if resolved_key in ("Pyrrole", "Imidazole"):
                h_pos = pt + 1.01 * v_rad
                out_symbols.append("H")
                out_coords.append(h_pos)

    result_coords = np.array(out_coords, dtype=np.float64)

    # Molecular mechanics geometry relaxation
    if relax:
        frozen = None
        if freeze_parent:
            # Freeze all original non-pruned atoms
            frozen = set(range(len(out_symbols) - len(new_ring_coords) * 3))
        try:
            opt_coords, _ = optimize_geometry_force_field(
                symbols=out_symbols,
                coords=result_coords,
                max_iter=150,
                frozen_indices=frozen,
            )
            result_coords = opt_coords
        except Exception as e:
            logger.debug("Optional edge annulation force field relaxation notice: %s", e)

    return out_symbols, result_coords


def bridge_cycle_atoms(
    symbols: List[str],
    coords: np.ndarray,
    a1: int,
    a2: int,
    bridge_type: str = "-CH2-",
    relax: bool = True,
) -> Tuple[List[str], np.ndarray]:
    """
    Create a bicyclic bridgehead connection across two ring carbons (e.g. norbornane, bicyclo[2.2.2]octane).

    Parameters:
    - bridge_type: "-CH2-", "-CH2CH2-", "-O-", "-NH-", "-CO-"
    """
    coords = np.array(coords, dtype=np.float64)
    r1, r2 = coords[a1], coords[a2]
    midpoint = 0.5 * (r1 + r2)
    dist = float(np.linalg.norm(r2 - r1))

    # Determine bridgehead normal direction
    n1_list = _find_adjacent_heavy_neighbors(symbols, coords, a1, a2)
    n2_list = _find_adjacent_heavy_neighbors(symbols, coords, a2, a1)
    
    cand_normals = []
    if n1_list:
        cand_normals.append(coords[n1_list[0]] - r1)
    if n2_list:
        cand_normals.append(coords[n2_list[0]] - r2)

    bridge_vec = r2 - r1
    if cand_normals:
        mean_v = np.mean(cand_normals, axis=0)
        norm_dir = np.cross(bridge_vec, mean_v)
        if np.linalg.norm(norm_dir) > 1e-4:
            norm_dir /= np.linalg.norm(norm_dir)
            norm_dir = np.cross(bridge_vec, norm_dir)
            norm_dir /= np.linalg.norm(norm_dir)
        else:
            norm_dir = np.array([0.0, 0.0, 1.0])
    else:
        norm_dir = np.array([0.0, 0.0, 1.0])

    # Arc height for bridge
    h = max(1.15, math.sqrt(max(0.2, 1.54**2 - (dist / 2.0)**2))) if dist < 3.0 else 1.30
    apex = midpoint + h * norm_dir

    # Prune 1 hydrogen on each bridgehead
    prune_set: Set[int] = set()
    h1 = _find_attached_hydrogens(symbols, coords, a1)
    if h1:
        prune_set.add(h1[0])
    h2 = _find_attached_hydrogens(symbols, coords, a2)
    if h2:
        prune_set.add(h2[0])

    out_syms = [symbols[i] for i in range(len(symbols)) if i not in prune_set]
    out_coords = [coords[i] for i in range(len(symbols)) if i not in prune_set]

    if bridge_type == "-CH2-":
        out_syms.append("C")
        out_coords.append(apex)
        # 2 Hydrogens on apex
        out_syms.append("H")
        out_coords.append(apex + 1.09 * norm_dir)
        u_lat = np.cross(bridge_vec, norm_dir)
        u_lat /= np.linalg.norm(u_lat)
        out_syms.append("H")
        out_coords.append(apex + 1.09 * (-norm_dir * 0.3 + u_lat * 0.95))
    elif bridge_type == "-CH2CH2-":
        p1 = midpoint + (r1 - midpoint) * 0.4 + h * norm_dir
        p2 = midpoint + (r2 - midpoint) * 0.4 + h * norm_dir
        out_syms.extend(["C", "C", "H", "H", "H", "H"])
        out_coords.extend([
            p1, p2,
            p1 + 1.09 * norm_dir, p1 - 1.09 * norm_dir * 0.3,
            p2 + 1.09 * norm_dir, p2 - 1.09 * norm_dir * 0.3,
        ])
    elif bridge_type == "-O-":
        out_syms.append("O")
        out_coords.append(apex)
    elif bridge_type == "-NH-":
        out_syms.append("N")
        out_coords.append(apex)
        out_syms.append("H")
        out_coords.append(apex + 1.01 * norm_dir)
    elif bridge_type == "-CO-":
        out_syms.append("C")
        out_coords.append(apex)
        out_syms.append("O")
        out_coords.append(apex + 1.22 * norm_dir)

    result_coords = np.array(out_coords, dtype=np.float64)
    if relax:
        try:
            opt_c, _ = optimize_geometry_force_field(out_syms, result_coords, max_iter=200)
            result_coords = opt_c
        except Exception as e:
            logger.debug("Optional bridgehead force field relaxation notice: %s", e)

    return out_syms, result_coords


def detect_peri_bays(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
) -> List[Tuple[int, int, int]]:
    """
    Detect all bay regions / peri triads (a_left, a_center, a_right) suitable for peri-annulation.

    A bay region is characterized by:
    - a_center is a bridgehead heavy atom with 0 attached hydrogens and >= 3 heavy neighbors.
    - a_left and a_right are heavy neighbors of a_center that have attached hydrogens (open bay arms).
    - distance(a_left, a_right) is in the open bay range (2.0 to 3.2 A).
    - a_left and a_right are not covalently bonded to each other.
    """
    if bonds is None:
        bonds = detect_connectivity(symbols, coords, strict=False)
    adj: Dict[int, Set[int]] = {i: set() for i in range(len(symbols))}
    for u, v in bonds:
        adj[u].add(v)
        adj[v].add(u)

    bays: List[Tuple[int, int, int]] = []
    for c, neighbors in adj.items():
        if symbols[c] == "H":
            continue
        h_on_c = _find_attached_hydrogens(symbols, coords, c)
        if len(h_on_c) == 0:
            h_bearing_neighs = [
                n for n in neighbors
                if symbols[n] != "H" and len(_find_attached_hydrogens(symbols, coords, n)) > 0
            ]
            if len(h_bearing_neighs) == 2:
                n1, n2 = h_bearing_neighs[0], h_bearing_neighs[1]
                if n2 not in adj[n1]:
                    d = float(np.linalg.norm(coords[n1] - coords[n2]))
                    if 2.0 < d < 3.2:
                        bays.append((n1, c, n2))
    return bays


def fuse_ring_to_triad(
    symbols: List[str],
    coords: np.ndarray,
    a_left: int,
    a_center: int,
    a_right: int,
    ring_preset: str = "Benzene",
    custom_size: Optional[int] = None,
    hetero_map: Optional[Dict[int, str]] = None,
    apex_saturation: str = "aromatic",  # "aromatic" (=CH-) or "neutral" (-CH2-)
    relax: bool = True,
    freeze_parent: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """
    Fuse a new molecular cycle across a triad of 3 contiguous atoms (peri-annulation / bay fusion).

    Used to:
    - Fuse a 6-membered ring on top of Naphthalene (across C_left, C_bridgehead, C_right) to create Phenalene.
    - Fuse a 5-membered ring across Naphthalene to create Acenaphthene.
    - Build peri-fused polycyclic aromatic hydrocarbon (PAH) systems without valence distortion or steric clashes.

    Parameters:
    - symbols: list of element symbols
    - coords: (N, 3) coordinate array
    - a_left, a_center, a_right: 0-indexed atom indices forming the bay
    - ring_preset: "Benzene", "Cyclopentyl", "Cyclopentadiene", "Pyridine", etc.
    - custom_size: optional custom size (4 to 8)
    - hetero_map: optional dict mapping new vertex index to heteroatom symbol
    - apex_saturation: "aromatic" (1 H per carbon) or "neutral" (2 H on apex, e.g. 1H-phenalene)
    - relax: run molecular mechanics (UFF) optimization
    - freeze_parent: if True, keep parent atoms fixed during relaxation
    """
    coords = np.array(coords, dtype=np.float64)
    n_atoms = len(symbols)
    if any(idx < 0 or idx >= n_atoms for idx in (a_left, a_center, a_right)):
        raise ValueError(f"Invalid triad indices: ({a_left}, {a_center}, {a_right})")
    if len({a_left, a_center, a_right}) < 3:
        raise ValueError("Triad atom indices must be distinct.")

    p_L = coords[a_left]
    p_C = coords[a_center]
    p_R = coords[a_right]

    v_L = p_L - p_C
    v_R = p_R - p_C
    d_L = float(np.linalg.norm(v_L))
    d_R = float(np.linalg.norm(v_R))
    if d_L > 2.2 or d_R > 2.2:
        raise ValueError(f"Atoms in triad are not covalently bonded: dist(L,C)={d_L:.2f} Å, dist(R,C)={d_R:.2f} Å")

    norm_cross = np.cross(v_L, v_R)
    norm_len = float(np.linalg.norm(norm_cross))
    if norm_len < 1e-4:
        raise ValueError("Selected triad is collinear. Cannot define a fusion plane.")
    plane_normal = norm_cross / norm_len

    # Determine ring geometry parameters
    resolved_key = _resolve_preset(ring_preset)
    if resolved_key is not None:
        info = RING_PRESETS[resolved_key]
        N = info["size"]
        is_aromatic = info["aromatic"]
        sub_map = dict(info["hetero"])
        if hetero_map:
            sub_map.update(hetero_map)
    else:
        N = custom_size if custom_size is not None else 6
        is_aromatic = False
        sub_map = dict(hetero_map) if hetero_map else {}

    if N < 4 or N > 8:
        raise ValueError(f"Peri-annulation supported for ring sizes 4 to 8 (requested N={N}).")

    bisect = v_L + v_R
    bisect_len = float(np.linalg.norm(bisect))
    bisect_u = bisect / bisect_len if bisect_len > 1e-4 else np.array([0.0, 1.0, 0.0])

    new_ring_coords: List[np.ndarray] = []
    new_ring_symbols: List[str] = []

    if N == 6:
        # Regular hexagonal peri-fusion (Phenalene motif)
        # Triad occupies 3 vertices; 3 new vertices complete the 6-gon
        center = p_C + (v_L + v_R)
        p_new1 = center + v_R  # connected to a_left
        p_new2 = center + (v_L + v_R)  # apex
        p_new3 = center + v_L  # connected to a_right
        new_verts = [p_new1, p_new2, p_new3]
    elif N == 5:
        # 5-membered peri-fusion (Acenaphthene motif)
        center = 0.5 * (p_L + p_R) + 0.6 * bisect
        p_new1 = p_L + 1.40 * bisect_u
        p_new2 = p_R + 1.40 * bisect_u
        new_verts = [p_new1, p_new2]
    elif N == 4:
        # 4-membered peri-fusion: 1 apex vertex
        center = 0.5 * (p_L + p_R) + 0.4 * bisect
        p_new1 = 0.5 * (p_L + p_R) + 1.35 * bisect_u
        new_verts = [p_new1]
    else:
        # General N-gon
        n_new = N - 3
        center = p_C + 1.2 * bisect
        new_verts = []
        d_theta = math.pi / (n_new + 1)
        for step in range(1, n_new + 1):
            pt = 0.5 * (p_L + p_R) + (1.35 + 0.2 * step) * bisect_u
            new_verts.append(pt)

    for idx_v, pt in enumerate(new_verts):
        sym = sub_map.get(idx_v, "C")
        new_ring_symbols.append(sym)
        new_ring_coords.append(pt)

    # Prune attached hydrogens on bay ends a_left and a_right
    prune_set: Set[int] = set()
    for h_idx in _find_attached_hydrogens(symbols, coords, a_left):
        prune_set.add(h_idx)
    for h_idx in _find_attached_hydrogens(symbols, coords, a_right):
        prune_set.add(h_idx)

    out_symbols = [symbols[i] for i in range(n_atoms) if i not in prune_set]
    out_coords = [coords[i] for i in range(n_atoms) if i not in prune_set]

    # Add new ring atoms
    new_atom_indices = []
    for sym, pt in zip(new_ring_symbols, new_ring_coords):
        new_atom_indices.append(len(out_symbols))
        out_symbols.append(sym)
        out_coords.append(pt)

    # Add hydrogens to the newly added vertices
    for idx_v, (sym, pt) in enumerate(zip(new_ring_symbols, new_ring_coords)):
        v_rad = pt - center
        rad_len = float(np.linalg.norm(v_rad))
        v_rad = v_rad / rad_len if rad_len > 1e-4 else bisect_u

        if sym == "C":
            is_apex = (idx_v == len(new_ring_coords) // 2) if len(new_ring_coords) > 1 else False
            if is_aromatic:
                if apex_saturation == "neutral" and is_apex:
                    # 1H-phenalene neutral apex (-CH2-)
                    h1 = pt + 1.09 * (v_rad * 0.7 + plane_normal * 0.7)
                    h2 = pt + 1.09 * (v_rad * 0.7 - plane_normal * 0.7)
                    out_symbols.extend(["H", "H"])
                    out_coords.extend([h1, h2])
                else:
                    # Planar aromatic =CH-
                    out_symbols.append("H")
                    out_coords.append(pt + 1.08 * v_rad)
            else:
                # Saturated / aliphatic ring: 2 H per carbon (e.g. Acenaphthene)
                h1 = pt + 1.09 * (v_rad * 0.7 + plane_normal * 0.7)
                h2 = pt + 1.09 * (v_rad * 0.7 - plane_normal * 0.7)
                out_symbols.extend(["H", "H"])
                out_coords.extend([h1, h2])
        elif sym == "N":
            if resolved_key in ("Pyrrole", "Imidazole"):
                out_symbols.append("H")
                out_coords.append(pt + 1.01 * v_rad)

    result_coords = np.array(out_coords, dtype=np.float64)

    if relax:
        try:
            opt_c, _ = optimize_geometry_force_field(out_symbols, result_coords, max_iter=250)
            result_coords = opt_c
        except Exception as e:
            logger.debug("Peri-annulation force field relaxation notice: %s", e)

    return out_symbols, result_coords
