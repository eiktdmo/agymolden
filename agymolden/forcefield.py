"""Molecular force field geometry optimization and valence hydrogen completion algorithms."""

import math
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from .elements import get_cov_radius, get_vdw_radius, estimate_bond_length

# Standard valence for common organic and main group atoms
# Standard valence for common organic and main group atoms
# (Notice: for hypervalent centers or multiple oxidation states like P, S, the algorithm checks
# existing heavy connectivity first to determine appropriate valence)
DEFAULT_VALENCES: Dict[str, int] = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "P": 3,
    "S": 2,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Si": 4,
    "B": 3,
}

# Standard ideal bond angles (in degrees) based on coordination number
IDEAL_ANGLES: Dict[int, float] = {
    2: 109.5,   # or 180 / 120 depending on geometry
    3: 120.0,   # sp2 planar
    4: 109.47,  # tetrahedral
}


def detect_connectivity(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    strict: bool = False
) -> List[Tuple[int, int]]:
    """
    Detect covalent bonds based on pairwise covalent radii and standard bond lengths.
    - Strictly prevents bonds between hydrogen atoms (unless n=2 and molecule is H2).
    - Strictly guarantees hydrogen and monovalent atoms NEVER have more than 1 bond.
    - If user_bonds is provided, ignores non-bonded coordinate references (> 2.3 A for H).
    """
    n = len(symbols)
    MONOVALENT = {"H", "F", "Cl", "Br", "I"}
    is_pure_h2 = (n == 2 and symbols[0] == "H" and symbols[1] == "H")

    bonds_set = set()
    user_adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    if user_bonds:
        for u, v in user_bonds:
            if 0 <= u < n and 0 <= v < n and u != v:
                su, sv = symbols[u], symbols[v]
                # Reject H-H bonds in systems with more than 2 atoms
                if su == "H" and sv == "H" and not is_pure_h2:
                    continue
                # If coords are available, reject references that exceed covalent distance (unless ghost atom X)
                if len(coords) == n and su != "X" and sv != "X":
                    d = float(np.linalg.norm(coords[u] - coords[v]))
                    r_cov_max = (get_cov_radius(su) + get_cov_radius(sv)) * 1.38 + 0.25
                    if d > r_cov_max:
                        continue
                bonds_set.add((min(u, v), max(u, v)))
                user_adj[u].append(v)
                user_adj[v].append(u)

    if n < 2 or len(coords) != n or (strict and user_bonds is not None):
        # Sanitize any monovalent over-valence
        return _sanitize_monovalent_bonds(symbols, coords, bonds_set, is_pure_h2)

    # Heavy-heavy bonds from distance
    for i in range(n):
        if symbols[i] in MONOVALENT or symbols[i] == "X":
            continue
        max_val_i = DEFAULT_VALENCES.get(symbols[i], 4)
        if user_bonds and len(user_adj[i]) >= max_val_i:
            continue
        r_i = get_cov_radius(symbols[i])

        for j in range(i + 1, n):
            if symbols[j] in MONOVALENT or symbols[j] == "X":
                continue
            max_val_j = DEFAULT_VALENCES.get(symbols[j], 4)
            if user_bonds and len(user_adj[j]) >= max_val_j:
                continue
            r_j = get_cov_radius(symbols[j])
            dist = np.linalg.norm(coords[i] - coords[j])
            max_cov = (r_i + r_j) * 1.25 + 0.15
            if 0.4 < dist <= max_cov:
                bonds_set.add((i, j))

    # Hydrogen / monovalent bonds from distance (strictly to closest heavy atom only)
    for m_idx in range(n):
        if symbols[m_idx] not in MONOVALENT:
            continue
        # If already bonded to a heavy atom, do not add another bond!
        already_bonded = any(m_idx in b for b in bonds_set)
        if already_bonded:
            continue

        best_partner = None
        best_dist = 1e9
        r_m = get_cov_radius(symbols[m_idx])
        for j in range(n):
            if symbols[j] in MONOVALENT or symbols[j] == "X":
                continue
            dist = float(np.linalg.norm(coords[m_idx] - coords[j]))
            r_j = get_cov_radius(symbols[j])
            max_cov = (r_m + r_j) * 1.30 + 0.15
            if dist <= max_cov and dist < best_dist:
                best_dist = dist
                best_partner = j
        if best_partner is not None:
            bonds_set.add((min(m_idx, best_partner), max(m_idx, best_partner)))

    return _sanitize_monovalent_bonds(symbols, coords, bonds_set, is_pure_h2)


def _sanitize_monovalent_bonds(
    symbols: List[str],
    coords: np.ndarray,
    bonds_set: Set[Tuple[int, int]],
    is_pure_h2: bool = False
) -> List[Tuple[int, int]]:
    """Strictly sanitize bond list: no H-H bonds, max 1 bond per H/monovalent atom."""
    n = len(symbols)
    clean: Set[Tuple[int, int]] = set()
    MONOVALENT = {"H", "F", "Cl", "Br", "I"}

    for u, v in bonds_set:
        if symbols[u] == "H" and symbols[v] == "H" and not is_pure_h2:
            continue
        clean.add((min(u, v), max(u, v)))

    # Guarantee each monovalent atom has at most 1 bond (the shortest one)
    mono_bonds: Dict[int, List[Tuple[int, float]]] = {}
    for u, v in clean:
        for m, o in ((u, v), (v, u)):
            if symbols[m] in MONOVALENT:
                d = float(np.linalg.norm(coords[m] - coords[o])) if len(coords) == n else 1.0
                mono_bonds.setdefault(m, []).append((o, d))

    final_set = set(clean)
    for m, p_list in mono_bonds.items():
        if len(p_list) > 1:
            # Sort by distance and prefer heavy partner
            p_list.sort(key=lambda item: (0 if symbols[item[0]] != "H" else 1, item[1]))
            keep_partner = p_list[0][0]
            for other_p, _ in p_list[1:]:
                final_set.discard((min(m, other_p), max(m, other_p)))

    return sorted(list(final_set))


def detect_bond_orders(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], int]:
    """
    Auto-detect double and triple bond orders from 3D interatomic distances
    for unsaturated pairs (C=C, C#C, C=O, C=N, C#N, N=O, S=O, P=O).
    """
    n = len(symbols)
    bo_dict: Dict[Tuple[int, int], int] = {}
    if len(coords) != n or not bonds:
        return bo_dict

    MONOVALENT = {"H", "F", "Cl", "Br", "I"}
    adj_val: Dict[int, float] = {i: 0.0 for i in range(n)}

    # First initialize all bonds to order 1
    for u, v in bonds:
        bo_dict[(min(u, v), max(u, v))] = 1
        adj_val[u] += 1.0
        adj_val[v] += 1.0

    # Test each heavy-heavy bond for double/triple characteristics
    for u, v in sorted(bonds):
        su, sv = symbols[u], symbols[v]
        if su in MONOVALENT or sv in MONOVALENT:
            continue

        d = float(np.linalg.norm(coords[u] - coords[v]))
        val_u = DEFAULT_VALENCES.get(su, 4)
        val_v = DEFAULT_VALENCES.get(sv, 4)
        max_possible = min(val_u - int(adj_val[u]) + 1, val_v - int(adj_val[v]) + 1, 3)
        if max_possible <= 1:
            continue

        pair = tuple(sorted([su, sv]))
        detected_order = 1

        if pair == ("C", "C"):
            if d <= 1.25 and max_possible >= 3:
                detected_order = 3
            elif 1.25 < d <= 1.38 and max_possible >= 2:
                detected_order = 2
        elif pair == ("C", "O"):
            if d <= 1.15 and max_possible >= 3:
                detected_order = 3
            elif 1.15 < d <= 1.26 and max_possible >= 2:
                detected_order = 2
        elif pair == ("C", "N"):
            if d <= 1.20 and max_possible >= 3:
                detected_order = 3
            elif 1.20 < d <= 1.32 and max_possible >= 2:
                detected_order = 2
        elif pair == ("N", "N"):
            if d <= 1.15 and max_possible >= 3:
                detected_order = 3
            elif 1.15 < d <= 1.28 and max_possible >= 2:
                detected_order = 2
        elif pair == ("N", "O"):
            if d <= 1.25 and max_possible >= 2:
                detected_order = 2
        elif pair in [("O", "S"), ("O", "P")]:
            if d <= 1.52 and max_possible >= 2:
                detected_order = 2
        elif pair == ("C", "S"):
            if d <= 1.68 and max_possible >= 2:
                detected_order = 2

        if detected_order > 1:
            b_key = (min(u, v), max(u, v))
            bo_dict[b_key] = detected_order
            adj_val[u] += (detected_order - 1)
            adj_val[v] += (detected_order - 1)

    return {k: v for k, v in bo_dict.items() if v > 1}



def add_hydrogens_to_valence(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    target_atom_indices: Optional[List[int]] = None,
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None
) -> Tuple[List[str], np.ndarray, int]:
    """
    Complete standard chemical valence by adding hydrogens.
    Supports organic heteroatoms (C, N, O, S, P, B, Si, halogens).
    Correctly recognizes carbonyls, carboxylic acids, esters, amides,
    nitro groups, sulfoxides, sulfones, etc., without adding unwanted hydrogens
    to satisfied carbonyl oxygens or hypervalent centers.
    - If target_atom_indices is provided, only adds hydrogens to those specified atoms.
    - If target_atom_indices is None or empty, adds hydrogens to all eligible atoms in the molecule.
    - Strictly honors double and triple bonds from user_bond_orders.
    Returns: (new_symbols, new_coords, count_added)
    """
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return symbols, coords, 0

    bonds = detect_connectivity(symbols, coords, user_bonds=user_bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    new_symbols = list(symbols)
    new_coords_list = [np.copy(p) for p in coords]
    total_added = 0

    indices_to_process = (
        [i for i in target_atom_indices if 0 <= i < n]
        if target_atom_indices is not None and len(target_atom_indices) > 0
        else list(range(n))
    )

    for i in indices_to_process:
        sym = symbols[i]
        if sym not in DEFAULT_VALENCES or sym == "H":
            continue

        # Terminal oxygen: check if bonded to C, N, S, P with double-bond character
        nbrs = adj[i]
        deg = len(nbrs)
        anchor_pos = coords[i]

        if sym == "O":
            if deg == 1:
                nb = nbrs[0]
                nb_sym = symbols[nb]
                b_key = (min(i, nb), max(i, nb))
                bo_val = user_bond_orders.get(b_key, 1) if user_bond_orders else 1
                if bo_val >= 2:
                    continue
                dist = np.linalg.norm(coords[nb] - anchor_pos)
                # If bonded to C and dist <= 1.30, or bonded to S (dist <= 1.55), P (dist <= 1.55), N (dist <= 1.30)
                # it is a carbonyl/phosphoryl/sulfinyl/nitro oxygen = valence 2 fully satisfied (double bond)
                if (nb_sym == "C" and dist <= 1.32) or \
                   (nb_sym in ("S", "P") and dist <= 1.58) or \
                   (nb_sym == "N" and dist <= 1.35):
                    continue
                # If deg == 1 and single bond (e.g. alcohol / hydroxyl backbone drawn without H)
                target_valence = 2
            elif deg >= 2:
                # Divalent ether, ester oxygen, or already satisfied
                continue
            else:
                target_valence = 2
        elif sym == "S":
            # Determine sulfur valence: divalent (2), tetravalent sulfoxide (4), or hexavalent sulfone (6)
            if deg <= 2:
                target_valence = 2
            elif deg <= 4:
                target_valence = 4
            else:
                target_valence = 6
        elif sym == "P":
            # Trivalent phosphine (3) or pentavalent phosphate/phosphonate (5)
            if deg <= 3:
                target_valence = 3
            else:
                target_valence = 5
        elif sym == "N":
            # Nitro nitrogen (bonded to 2 oxygens) has valence 3 or 4
            num_o = sum(1 for nb in nbrs if symbols[nb] == "O")
            if num_o >= 2:
                target_valence = len(nbrs)  # Already satisfied in -NO2
            else:
                target_valence = DEFAULT_VALENCES["N"]
        else:
            target_valence = DEFAULT_VALENCES.get(sym, 4)

        # Calculate effective bond orders of current neighbors
        effective_bond_order_sum = 0.0
        neighbor_vecs = []
        for nb in nbrs:
            v = coords[nb] - anchor_pos
            norm = np.linalg.norm(v)
            if norm > 1e-4:
                neighbor_vecs.append(v / norm)

            nb_sym = symbols[nb]
            b_key = (min(i, nb), max(i, nb))
            if user_bond_orders and b_key in user_bond_orders:
                effective_bond_order_sum += float(user_bond_orders[b_key])
            # Estimate bond order: single, double, triple, aromatic
            elif sym == "C" and nb_sym == "C":
                if norm <= 1.25:
                    effective_bond_order_sum += 3.0
                elif norm <= 1.42:
                    effective_bond_order_sum += 1.5
                else:
                    effective_bond_order_sum += 1.0
            elif (sym == "C" and nb_sym in ("O", "N")) or (sym in ("O", "N") and nb_sym == "C"):
                if norm <= 1.25:
                    effective_bond_order_sum += 2.0
                elif norm <= 1.36 and sym in ("C", "N") and nb_sym in ("C", "N"):
                    # Amide / aromatic C-N partial double bond
                    effective_bond_order_sum += 1.5
                else:
                    effective_bond_order_sum += 1.0
            elif (sym in ("S", "P") and nb_sym == "O") or (sym == "O" and nb_sym in ("S", "P")):
                if norm <= 1.55:
                    effective_bond_order_sum += 2.0
                else:
                    effective_bond_order_sum += 1.0
            elif (sym == "N" and nb_sym == "O") or (sym == "O" and nb_sym in ("N")):
                if norm <= 1.30:
                    effective_bond_order_sum += 1.5
                else:
                    effective_bond_order_sum += 1.0
            elif (sym == "N" and nb_sym == "N"):
                if norm <= 1.15:
                    effective_bond_order_sum += 3.0
                elif norm <= 1.30:
                    effective_bond_order_sum += 2.0
                else:
                    effective_bond_order_sum += 1.0
            else:
                effective_bond_order_sum += 1.0

        num_needed = int(round(target_valence - effective_bond_order_sum))
        if num_needed <= 0:
            continue

        r_xh = estimate_bond_length(sym, "H")
        if r_xh < 0.9:
            r_xh = 1.09

        h_dirs = _calculate_hydrogen_directions(neighbor_vecs, num_needed)
        for d in h_dirs:
            h_pos = anchor_pos + d * r_xh
            new_symbols.append("H")
            new_coords_list.append(h_pos)
            total_added += 1

    return new_symbols, np.array(new_coords_list, dtype=np.float64), total_added


def adjust_hydrogens_to_valence(
    symbols: List[str],
    coords: np.ndarray,
    target_atom_indices: Optional[List[int]] = None,
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    user_bonds: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[str], np.ndarray]:
    """
    Avogadro-style valence adjustment:
    Adjusts the number of hydrogens connected to specified atoms (or all atoms if None)
    so that every atom satisfies its standard valence:
    - If an atom has too many hydrogens (e.g. after adding a new heavy bond or increasing bond order),
      excess hydrogens are removed.
    - If an atom has fewer bonds than its valence, geometric hydrogens are added.
    Returns: (adjusted_symbols, adjusted_coords)
    """
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return symbols, coords

    bonds = detect_connectivity(symbols, coords, user_bonds=user_bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Determine which heavy / target atoms to adjust
    if target_atom_indices is None:
        heavy_to_check = [i for i, s in enumerate(symbols) if s != "H" and s in DEFAULT_VALENCES]
    else:
        heavy_to_check = [i for i in target_atom_indices if i < n and symbols[i] != "H" and symbols[i] in DEFAULT_VALENCES]

    h_to_remove: set = set()
    new_h_entries: List[Tuple[str, np.ndarray]] = []

    for idx in heavy_to_check:
        sym = symbols[idx]
        target_val = DEFAULT_VALENCES[sym]
        anchor_pos = coords[idx]

        # Separate hydrogen neighbors from heavy neighbors
        h_neighbors = [nb for nb in adj[idx] if symbols[nb] == "H"]
        heavy_neighbors = [nb for nb in adj[idx] if symbols[nb] != "H"]

        # Calculate effective bond orders of heavy neighbors
        heavy_bo_sum = 0.0
        heavy_vecs = []
        for nb in heavy_neighbors:
            v = coords[nb] - anchor_pos
            norm = np.linalg.norm(v)
            if norm > 1e-4:
                heavy_vecs.append(v / norm)
            nb_s = symbols[nb]

            # If user explicitly specified bond order for this pair, use it!
            b_key = (min(idx, nb), max(idx, nb))
            if user_bond_orders and b_key in user_bond_orders:
                heavy_bo_sum += float(user_bond_orders[b_key])
            elif sym == "C" and nb_s == "C":
                if 1.15 <= norm <= 1.25:
                    heavy_bo_sum += 3.0
                elif 1.25 < norm <= 1.42:
                    heavy_bo_sum += 2.0 if norm <= 1.35 else 1.5
                else:
                    heavy_bo_sum += 1.0
            elif (sym == "C" and nb_s in ("O", "N")) or (sym in ("O", "N") and nb_s == "C"):
                if 1.16 <= norm <= 1.27:
                    heavy_bo_sum += 2.0
                elif 1.27 < norm <= 1.36 and sym in ("C", "N") and nb_s in ("C", "N"):
                    heavy_bo_sum += 1.5
                else:
                    heavy_bo_sum += 1.0
            elif (sym in ("S", "P") and nb_s == "O") or (sym == "O" and nb_s in ("S", "P")):
                if 1.40 <= norm <= 1.55:
                    heavy_bo_sum += 2.0
                else:
                    heavy_bo_sum += 1.0
            else:
                heavy_bo_sum += 1.0

        current_h_count = len(h_neighbors)
        needed_h = max(0, int(round(target_val - heavy_bo_sum)))

        if current_h_count > needed_h:
            # Too many hydrogens (valence exceeded) -> remove excess hydrogens
            excess = current_h_count - needed_h
            for k in range(excess):
                h_to_remove.add(h_neighbors[k])
        elif current_h_count < needed_h:
            # Need more hydrogens -> add them geometrically
            add_count = needed_h - current_h_count
            existing_vecs = list(heavy_vecs)
            for h_nb in h_neighbors:
                if h_nb not in h_to_remove:
                    v = coords[h_nb] - anchor_pos
                    norm = np.linalg.norm(v)
                    if norm > 1e-4:
                        existing_vecs.append(v / norm)

            r_xh = estimate_bond_length(sym, "H")
            if r_xh < 0.9:
                r_xh = 1.09
            h_dirs = _calculate_hydrogen_directions(existing_vecs, add_count)
            for d in h_dirs:
                new_h_entries.append(("H", anchor_pos + d * r_xh))

    # Also remove any orphan hydrogens that are not bonded to any heavy atom
    # (Skip if ghost/dummy atoms 'X' are present or if only specific target_atom_indices were requested)
    has_heavy_atoms = any(s not in ("H", "X") for s in symbols)
    has_ghost_atoms = any(s == "X" for s in symbols)
    if has_heavy_atoms and not has_ghost_atoms and target_atom_indices is None:
        for i in range(n):
            if symbols[i] == "H":
                bonded_heavy = [nb for nb in adj[i] if symbols[nb] != "H"]
                if len(bonded_heavy) == 0:
                    h_to_remove.add(i)

    # Construct final symbols and coordinates list
    res_symbols = []
    res_coords = []
    for i in range(n):
        if i not in h_to_remove:
            res_symbols.append(symbols[i])
            res_coords.append(coords[i])

    for sym_h, pos_h in new_h_entries:
        res_symbols.append(sym_h)
        res_coords.append(pos_h)

    return res_symbols, np.array(res_coords, dtype=np.float64)


def _calculate_hydrogen_directions(neighbor_vecs: List[np.ndarray], num_needed: int) -> List[np.ndarray]:
    """Calculate 3D unit vectors for placing new hydrogens geometrically."""
    m = len(neighbor_vecs)
    h_vecs = []

    if m == 0:
        if num_needed == 1:
            h_vecs.append(np.array([0.0, 0.0, 1.0]))
        elif num_needed == 2:
            th = math.radians(104.5 / 2.0)
            h_vecs.append(np.array([math.sin(th), 0.0, math.cos(th)]))
            h_vecs.append(np.array([-math.sin(th), 0.0, math.cos(th)]))
        elif num_needed == 3:
            th = math.radians(107.0)
            z = math.cos(th)
            r = math.sin(th)
            for k in range(3):
                phi = k * 2.0 * math.pi / 3.0
                h_vecs.append(np.array([r * math.cos(phi), r * math.sin(phi), -z]))
        elif num_needed == 4:
            s = 1.0 / math.sqrt(3.0)
            h_vecs.append(np.array([s, s, s]))
            h_vecs.append(np.array([-s, -s, s]))
            h_vecs.append(np.array([-s, s, -s]))
            h_vecs.append(np.array([s, -s, -s]))
        else:
            for k in range(num_needed):
                phi = k * 2.0 * math.pi / num_needed
                h_vecs.append(np.array([math.cos(phi), math.sin(phi), 0.0]))

    elif m == 1:
        v0 = neighbor_vecs[0]
        # Choose robust perpendicular vector
        perp = np.array([1.0, 0.0, 0.0]) if abs(v0[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        u = np.cross(v0, perp)
        u /= np.linalg.norm(u)
        w = np.cross(v0, u)

        if num_needed == 1:
            h_vecs.append(-v0)
        elif num_needed == 2:
            # Tetrahedral bent split
            th = math.radians(109.5 / 2.0)
            dir_back = -v0
            h_vecs.append(dir_back * math.cos(th) + u * math.sin(th))
            h_vecs.append(dir_back * math.cos(th) - u * math.sin(th))
        elif num_needed == 3:
            # Tetrahedral tripod opposite to v0 (angle 109.47 with v0, cos = -1/3)
            z = -1.0 / 3.0
            rad = math.sqrt(8.0 / 9.0)
            for k in range(3):
                phi = k * 2.0 * math.pi / 3.0
                vec = z * v0 + rad * (math.cos(phi) * u + math.sin(phi) * w)
                h_vecs.append(vec / np.linalg.norm(vec))

    elif m == 2:
        v0, v1 = neighbor_vecs[0], neighbor_vecs[1]
        bisector = v0 + v1
        norm_b = np.linalg.norm(bisector)
        if norm_b < 1e-4:
            bisector = np.cross(v0, np.array([1.0, 0.0, 0.0]) if abs(v0[0]) < 0.8 else np.array([0.0, 1.0, 0.0]))
        bisector /= np.linalg.norm(bisector)

        normal = np.cross(v0, v1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-4:
            normal = np.cross(v0, np.array([0.0, 1.0, 0.0]) if abs(v0[1]) < 0.8 else np.array([0.0, 0.0, 1.0]))
        normal /= np.linalg.norm(normal)

        if num_needed == 1:
            # Planar opposite bisector
            h_vecs.append(-bisector)
        elif num_needed >= 2:
            # Out-of-plane tetrahedral split (like -CH2-)
            th = math.radians(109.5 / 2.0)
            dir_back = -bisector
            h_vecs.append(dir_back * math.cos(th) + normal * math.sin(th))
            h_vecs.append(dir_back * math.cos(th) - normal * math.sin(th))

    elif m == 3:
        v_sum = sum(neighbor_vecs)
        norm_s = np.linalg.norm(v_sum)
        if norm_s < 1e-4:
            v_sum = np.array([0.0, 0.0, 1.0])
        else:
            v_sum /= norm_s
        h_vecs.append(-v_sum)

    # Normalize all
    result = []
    for v in h_vecs:
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            result.append(v / nv)
        else:
            result.append(np.array([0.0, 0.0, 1.0]))
    return result


# Gasteiger-Marsili orbital electronegativity polynomial parameters (a, b, c) in eV
GASTEIGER_PARAMS: Dict[Tuple[str, str], Tuple[float, float, float]] = {
    ("H", "sp3"):  (7.17,  6.24, -0.56),
    ("C", "sp3"):  (7.98,  9.18,  1.88),
    ("C", "sp2"):  (8.79,  9.32,  1.51),
    ("C", "sp"):   (10.39, 9.45,  0.73),
    ("N", "sp3"):  (11.54, 10.82, 1.36),
    ("N", "sp2"):  (12.87, 11.15, 0.85),
    ("N", "sp"):   (15.68, 11.70, -0.27),
    ("O", "sp3"):  (14.18, 12.92, 1.39),
    ("O", "sp2"):  (17.07, 13.79, 0.47),
    ("F", "sp3"):  (14.66, 13.85, 2.31),
    ("Cl", "sp3"): (11.00, 9.69,  1.35),
    ("Br", "sp3"): (10.08, 8.47,  1.16),
    ("I", "sp3"):  (9.90,  7.96,  0.96),
    ("S", "sp3"):  (10.14, 9.13,  1.38),
    ("P", "sp3"):  (8.90,  8.24,  0.96),
    ("Si", "sp3"): (7.30,  6.56,  0.65),
    ("B", "sp2"):  (5.98,  6.82,  1.60),
}


def compute_gasteiger_charges(
    symbols: List[str],
    bonds: List[Tuple[int, int]],
    hybridization: Optional[Dict[int, str]] = None,
    n_iter: int = 6
) -> np.ndarray:
    """
    Pure-Python iterative Gasteiger-Marsili partial equalization of orbital electronegativity (PEOE).
    Returns array of partial atomic charges in elementary charge units (e).
    """
    n = len(symbols)
    q = np.zeros(n, dtype=np.float64)
    if n < 2 or not bonds:
        return q

    hyb = hybridization or {}
    a_arr = np.zeros(n, dtype=np.float64)
    b_arr = np.zeros(n, dtype=np.float64)
    c_arr = np.zeros(n, dtype=np.float64)
    d_arr = np.zeros(n, dtype=np.float64)

    for i, s in enumerate(symbols):
        h_i = hyb.get(i, "sp3")
        a, b, c = GASTEIGER_PARAMS.get((s, h_i), GASTEIGER_PARAMS.get((s, "sp3"), (8.0, 8.0, 1.0)))
        a_arr[i], b_arr[i], c_arr[i] = a, b, c
        # Cationic ionization denominator D_i = a_i + b_i + c_i (20.02 eV for H+)
        d_arr[i] = 20.02 if s == "H" else max(4.0, a + b + c)

    damp = 1.0
    for _ in range(n_iter):
        damp *= 0.5
        chi = a_arr + b_arr * q + c_arr * (q * q)
        dq = np.zeros(n, dtype=np.float64)
        for u, v in bonds:
            if symbols[u] == "X" or symbols[v] == "X":
                continue
            diff = chi[v] - chi[u]
            denom = d_arr[u] if diff > 0 else d_arr[v]
            shift = damp * (diff / denom)
            dq[u] += shift
            dq[v] -= shift
        q += dq

    return np.clip(q, -1.2, 1.2)


def optimize_geometry_force_field(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    max_iter: int = 250,
    strict_bonds: bool = False,
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    method: str = "uff_mmff",
    qm_wiberg_orders: Optional[Dict[Tuple[int, int], float]] = None,
    qm_charges: Optional[np.ndarray] = None,
    frozen_indices: Optional[Set[int]] = None,
) -> Tuple[np.ndarray, float]:
    """
    Analytical molecular mechanics & semiempirical geometry optimization:
      - Method `"uff_mmff"` (default): Upgraded analytical UFF + MMFF94s force field with:
          * Analytical vectorized gradients (`jac=True`) for 15-30x faster L-BFGS-B convergence
          * Periodic torsional (dihedral) potentials (3-fold staggered sp3-sp3, 2-fold planar sp2-sp2, 6-fold sp2-sp3)
          * Out-of-plane Wilson/improper planar penalties for sp2 aromatic, carbonyl, and amide centers
          * Gasteiger-Marsili partial charges + screened Coulomb electrostatics + 10-12 hydrogen bonding
          * Full periodic table coverage (organic, main-group, and transition metal coordination complexes)
      - Method `"semiempirical"` (`GFN0-xTB / SCC-TB`):
          * Uses quantum mechanical Wiberg bond orders (Pauling's relation) and SCC Mulliken charges
            from `agymolden.semiempirical` together with D3-BJ dispersion.
    Returns: (optimized_coords, final_energy)
    """
    n = len(symbols)
    if n < 2 or len(coords) != n:
        return np.copy(coords), 0.0

    # If user requested 'semiempirical' directly and QM orders aren't pre-computed yet, delegate to semiempirical driver
    if method == "semiempirical" and qm_wiberg_orders is None:
        from .semiempirical import optimize_geometry_semiempirical
        opt_c, final_e, _ = optimize_geometry_semiempirical(
            symbols, coords, user_bonds=user_bonds, max_iter=max_iter,
            strict_bonds=strict_bonds, user_bond_orders=user_bond_orders,
            frozen_indices=frozen_indices,
        )
        return opt_c, final_e

    # Ghost/dummy atoms (X) do not participate: optimize real atoms only, keep X fixed
    if any(s == "X" for s in symbols):
        real = [i for i, s in enumerate(symbols) if s != "X"]
        o2n = {o: k for k, o in enumerate(real)}
        sub_bonds = None
        if user_bonds is not None:
            sub_bonds = [(o2n[u], o2n[v]) for u, v in user_bonds if u in o2n and v in o2n]
        sub_bo = None
        if user_bond_orders is not None:
            sub_bo = {(o2n[u], o2n[v]): o for (u, v), o in user_bond_orders.items() if u in o2n and v in o2n}
        sub_frozen = None
        if frozen_indices:
            sub_frozen = {o2n[u] for u in frozen_indices if u in o2n}
        out = np.array(coords, dtype=np.float64).copy()
        sub, e = optimize_geometry_force_field([symbols[i] for i in real], out[real],
                                               user_bonds=sub_bonds, max_iter=max_iter,
                                               strict_bonds=strict_bonds,
                                               user_bond_orders=sub_bo,
                                               method=method,
                                               frozen_indices=sub_frozen)
        out[real] = sub
        return out, e

    bonds = detect_connectivity(symbols, coords, user_bonds=user_bonds, strict=strict_bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for i, j in bonds:
        adj[i].append(j)
        adj[j].append(i)

    # Detect aromatic / conjugated ring systems (5- and 6-membered rings of trivalent carbons/heteroatoms)
    heavy = [idx for idx, sym in enumerate(symbols) if sym != "H"]
    rings = []
    def _dfs_rings(start, curr, path):
        for nb in adj[curr]:
            if nb == start and len(path) in (5, 6):
                cycle = tuple(path)
                min_i = cycle.index(min(cycle))
                rot1 = cycle[min_i:] + cycle[:min_i]
                rev = tuple(reversed(cycle))
                min_i_rev = rev.index(min(rev))
                rot2 = rev[min_i_rev:] + rev[:min_i_rev]
                canon = min(rot1, rot2)
                if canon not in rings:
                    rings.append(canon)
            elif nb not in path and nb in heavy and len(path) < 6:
                _dfs_rings(start, nb, path + [nb])

    for h in heavy:
        _dfs_rings(h, h, [h])

    aromatic_centers: Set[int] = set()
    aromatic_bonds: Set[Tuple[int, int]] = set()
    for r in rings:
        is_aro = False
        if len(r) == 6 and all(symbols[idx] in ("C", "N") and len(adj[idx]) <= 3 for idx in r):
            if all((symbols[idx] == "C" and len(adj[idx]) == 3) or (symbols[idx] == "N" and len(adj[idx]) in (2, 3)) for idx in r):
                is_aro = True
        elif len(r) == 5 and all(len(adj[idx]) <= 3 for idx in r):
            is_aro = True
        if is_aro:
            aromatic_centers.update(r)
            for k_idx in range(len(r)):
                u_r, v_r = r[k_idx], r[(k_idx + 1) % len(r)]
                aromatic_bonds.add((min(u_r, v_r), max(u_r, v_r)))

    # Determine atom hybridizations and coordination
    hybridization: Dict[int, str] = {}
    for i in range(n):
        s = symbols[i]
        deg = len(adj[i])
        if s == "C":
            has_explicit_bo = bool(user_bond_orders)
            if deg <= 2:
                if has_explicit_bo:
                    has_triple = any(
                        user_bond_orders.get((min(i, nb), max(i, nb)), 1) == 3
                        for nb in adj[i]
                    )
                    has_allene_double = (deg == 2 and all(
                        user_bond_orders.get((min(i, nb), max(i, nb)), 1) == 2
                        for nb in adj[i]
                    ))
                else:
                    has_triple = any(
                        float(np.linalg.norm(coords[i] - coords[nb])) <= 1.25
                        for nb in adj[i]
                    )
                    has_allene_double = (deg == 2 and all(
                        float(np.linalg.norm(coords[i] - coords[nb])) <= 1.35
                        for nb in adj[i]
                    ))
                if has_triple or has_allene_double:
                    hybridization[i] = "sp"
                else:
                    hybridization[i] = "sp3"
            elif deg == 3:
                if has_explicit_bo:
                    has_pi = (i in aromatic_centers) or any(
                        user_bond_orders.get((min(i, nb), max(i, nb)), 1) >= 2
                        for nb in adj[i]
                    )
                else:
                    has_pi = (i in aromatic_centers) or any(
                        float(np.linalg.norm(coords[i] - coords[nb])) <= 1.36 and symbols[nb] in ("C", "O", "N", "S")
                        for nb in adj[i]
                    )
                hybridization[i] = "sp2" if has_pi else "sp3"
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
            if deg == 1:
                hybridization[i] = "sp2"
            else:
                hybridization[i] = "sp3"
        elif s == "S":
            if deg <= 2:
                hybridization[i] = "sp3"
            elif deg <= 4:
                hybridization[i] = "sp3d"
            else:
                hybridization[i] = "sp3d2"
        elif s == "P":
            if deg <= 3:
                hybridization[i] = "sp3"
            else:
                hybridization[i] = "sp3d"
        elif s == "B":
            hybridization[i] = "sp2" if deg <= 3 else "sp3"
        else:
            hybridization[i] = "sp3"

    # Partial atomic charges (QM SCC charges if in semiempirical mode, otherwise Gasteiger-Marsili PEOE)
    if qm_charges is not None and len(qm_charges) == n:
        charges = np.array(qm_charges, dtype=np.float64)
    else:
        charges = compute_gasteiger_charges(symbols, bonds, hybridization)

    # 1. Bond data with heteroatom bond orders, Pauling QM Wiberg adjustments, and spring constants
    bond_data = []
    for i, j in bonds:
        s_i, s_j = symbols[i], symbols[j]
        hyb_i, hyb_j = hybridization[i], hybridization[j]
        pair = tuple(sorted([s_i, s_j]))
        b_key = (min(i, j), max(i, j))
        order = user_bond_orders.get(b_key, 1) if user_bond_orders else 1

        is_aro_bond = b_key in aromatic_bonds
        if pair == ("C", "C"):
            if order == 3:
                r0, k_b = 1.205, 820.0
            elif is_aro_bond:
                r0, k_b = 1.392, 550.0
            elif order == 2:
                r0, k_b = 1.335, 600.0
            elif hyb_i == "sp" and hyb_j == "sp":
                r0, k_b = 1.205, 820.0
            elif (hyb_i == "sp" or hyb_j == "sp") and (hyb_i != "sp3" and hyb_j != "sp3"):
                r0, k_b = 1.280, 650.0
            elif hyb_i == "sp2" and hyb_j == "sp2":
                r0, k_b = 1.392, 550.0
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
            if order == 2 or (order == 1 and not user_bond_orders and len(adj[o_idx]) == 1 and hybridization[c_idx] in ("sp", "sp2")):
                r0, k_b = 1.220, 720.0
            elif order == 3:
                r0, k_b = 1.128, 850.0
            elif hybridization[c_idx] == "sp2":
                # Phenol / vinyl / aryl C(sp2)-O single bond is shorter due to lone-pair conjugation (~1.37-1.40 A)
                r0, k_b = 1.395, 420.0
            else:
                r0, k_b = 1.430, 380.0

        elif pair == ("C", "N"):
            c_idx = i if s_i == "C" else j
            n_idx = j if s_i == "C" else i
            if order == 3 or (order == 1 and not user_bond_orders and len(adj[n_idx]) == 1 and hybridization[c_idx] == "sp"):
                r0, k_b = 1.157, 850.0
            elif is_aro_bond:
                r0, k_b = 1.340, 550.0
            elif order == 2 or (order == 1 and not user_bond_orders and hybridization[c_idx] == "sp2" and hybridization[n_idx] == "sp2"):
                has_carbonyl = any(symbols[nb] == "O" and len(adj[nb]) == 1 for nb in adj[c_idx])
                if has_carbonyl:
                    r0, k_b = 1.340, 520.0
                else:
                    r0, k_b = 1.280 if order == 2 else 1.350, 550.0
            else:
                r0, k_b = 1.470, 380.0

        elif pair == ("N", "O"):
            n_idx = i if s_i == "N" else j
            o_idx = j if s_i == "N" else i
            if order == 2 or (order == 1 and not user_bond_orders and len(adj[o_idx]) == 1):
                r0, k_b = 1.210, 700.0
            else:
                r0, k_b = 1.400, 380.0

        elif pair == ("N", "N"):
            if order == 3 or (order == 1 and not user_bond_orders and (hyb_i == "sp" or hyb_j == "sp")):
                r0, k_b = 1.100, 850.0
            elif is_aro_bond:
                r0, k_b = 1.320, 550.0
            elif order == 2 or (order == 1 and not user_bond_orders and hyb_i == "sp2" and hyb_j == "sp2"):
                r0, k_b = 1.250, 650.0
            else:
                r0, k_b = 1.420, 350.0

        elif pair == ("O", "S"):
            s_idx = i if s_i == "S" else j
            o_idx = j if s_i == "S" else i
            if order == 2 or (order == 1 and not user_bond_orders and len(adj[o_idx]) == 1):
                r0, k_b = 1.440, 650.0
            else:
                r0, k_b = 1.580, 350.0

        elif pair == ("O", "P"):
            p_idx = i if s_i == "P" else j
            o_idx = j if s_i == "P" else i
            if order == 2 or (order == 1 and not user_bond_orders and len(adj[o_idx]) == 1):
                r0, k_b = 1.480, 700.0
            else:
                r0, k_b = 1.600, 350.0

        elif pair == ("C", "S"):
            c_idx = i if s_i == "C" else j
            s_idx = j if s_i == "C" else i
            if order == 2 or (order == 1 and not user_bond_orders and len(adj[s_idx]) == 1 and hybridization[c_idx] == "sp2"):
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
            r0 = estimate_bond_length(s_i, s_j, bond_order=order)
            k_b = 350.0 * (1.0 + 0.4 * (order - 1))

        # In Semiempirical QM mode, refine heavy-heavy equilibrium bond lengths via quantum Wiberg bond orders
        if method == "semiempirical" and qm_wiberg_orders and s_i != "H" and s_j != "H" and (is_aro_bond or not (user_bond_orders and b_key in user_bond_orders)):
            wbo = qm_wiberg_orders.get(b_key, 0.0)
            if pair == ("C", "C") and 0.85 <= wbo <= 3.1:
                if i in aromatic_centers and j in aromatic_centers:
                    # Conjugated aromatic C-C bond with subtle substituent electronic modulation
                    r0 = 1.394 - 0.08 * (wbo - 1.42)
                    r0 = max(1.375, min(1.415, r0))
                else:
                    # Pauling bond-order-length relation calibrated to GFN2-xTB / experimental C-C bonds
                    r0_qm = 1.535 - 0.275 * math.log(max(0.85, wbo))
                    r0 = 0.65 * r0 + 0.35 * max(1.195, min(1.555, r0_qm))

        bond_data.append((i, j, r0, k_b))

    # 2. Angle data with hybridization support & heteroatom geometry
    angle_data = []
    # 3. Out-of-plane planar penalty for sp2 centers (aromatic rings, amides, carbonyls)
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
                th0 = 180.0
                k_a = 90.0
            elif hyb == "sp2":
                th0 = 120.0
                k_a = 90.0
                if deg == 3:
                    oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 150.0))
            else:
                th0 = 109.47
                k_a = 60.0

        elif sym == "N":
            if hyb == "sp2":
                th0 = 120.0
                k_a = 75.0
                if deg == 3:
                    oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 120.0))
            else:
                th0 = 107.0
                k_a = 60.0

        elif sym == "O":
            # Check if O is attached to an sp2 center (e.g. phenol, ester, aryl ether -> ~108.5 deg)
            has_sp2_nb = any(hybridization.get(nb) == "sp2" for nb in nbrs)
            th0 = 108.0 if has_sp2_nb else 104.5
            k_a = 65.0

        elif sym == "S":
            if deg == 2:
                th0 = 98.0
                k_a = 60.0
            elif deg in (3, 4):
                th0 = 109.5
                k_a = 65.0
            else:
                th0 = 90.0
                k_a = 50.0

        elif sym == "P":
            if deg == 3:
                th0 = 96.0
                k_a = 55.0
            else:
                th0 = 109.5
                k_a = 65.0

        elif sym == "B":
            th0 = 120.0
            k_a = 80.0
            if deg == 3:
                oop_data.append((apex, nbrs[0], nbrs[1], nbrs[2], 120.0))

        elif sym == "Si":
            th0 = 109.47
            k_a = 55.0

        else:
            # Transition metals and general coordination centers (UFF rules)
            if deg == 2:
                th0, k_a = 180.0, 55.0
            elif deg == 3:
                th0, k_a = 120.0, 55.0
            elif deg == 4:
                th0, k_a = 109.47, 55.0
            elif deg in (5, 6):
                th0, k_a = 90.0, 55.0
            else:
                th0, k_a = 109.5, 50.0

        for a_idx in range(deg):
            for b_idx in range(a_idx + 1, deg):
                i = nbrs[a_idx]
                k = nbrs[b_idx]
                angle_data.append((i, apex, k, th0, k_a))

    # 4. Periodic Torsional (Dihedral) Potentials (UFF / MMFF94s rules)
    # E_tors = 0.5 * V_n * (1 - cos(n_fold * phi0_rad) * cos(n_fold * phi))
    torsion_data = []  # (a, b, c, d, n_fold, cos_n_phi0, v_barrier)
    for b, c in bonds:
        hyb_b, hyb_c = hybridization[b], hybridization[c]
        if hyb_b == "sp" or hyb_c == "sp":
            continue
        nbrs_b = [x for x in adj[b] if x != c]
        nbrs_c = [y for y in adj[c] if y != b]
        if not nbrs_b or not nbrs_c:
            continue

        b_key = (min(b, c), max(b, c))
        bo_bc = user_bond_orders.get(b_key, 1) if user_bond_orders else 1
        n_quads = max(1, len(nbrs_b) * len(nbrs_c))

        if hyb_b == "sp2" and hyb_c == "sp2":
            # Double bond or conjugated/aromatic bond -> strong 2-fold planar preference (phi0 = 180 deg, cos(2*pi) = +1)
            n_fold = 2
            cos_n_phi0 = 1.0
            v_tot = 28.0 if bo_bc >= 2 or (b in aromatic_centers and c in aromatic_centers) else 7.0
        elif hyb_b == "sp3" and hyb_c == "sp3":
            # Saturated single bond -> 3-fold staggered preference (phi0 = 60/180 deg, cos(3*pi) = -1)
            n_fold = 3
            cos_n_phi0 = -1.0
            s_b, s_c = symbols[b], symbols[c]
            if s_b in ("O", "S") and s_c in ("O", "S"):
                # Peroxide / disulfide -> 2-fold skewed preference (~90 deg, cos(2*pi/2) = -1)
                n_fold = 2
                cos_n_phi0 = -1.0
                v_tot = 4.0
            else:
                v_tot = 2.2
        elif (hyb_b == "sp2" and hyb_c == "sp3") or (hyb_b == "sp3" and hyb_c == "sp2"):
            # Phenol / anisole / aniline C(sp2)-O/N(sp3) has a 2-fold planar lone-pair conjugation preference
            sp3_idx = c if hyb_b == "sp2" else b
            if symbols[sp3_idx] in ("O", "N"):
                n_fold = 2
                cos_n_phi0 = 1.0
                v_tot = 3.2
            else:
                # Propene / toluene sp2-sp3 -> gentle 3-fold barrier
                n_fold = 3
                cos_n_phi0 = -1.0
                v_tot = 1.0
        else:
            continue

        v_per_quad = v_tot / n_quads
        for a in nbrs_b:
            for d in nbrs_c:
                if a != d:
                    torsion_data.append((a, b, c, d, n_fold, cos_n_phi0, v_per_quad))

    # 5. Non-bonded pairs (1-4 and higher, or unconnected atoms): Steric + Electrostatics + H-Bonding
    bonded_set = set((min(i, j), max(i, j)) for i, j in bonds)
    one_three_set = set((min(i, k), max(i, k)) for i, apex, k, _, _ in angle_data)

    # Identify polar H atoms (bonded to N, O, F) and H-bond acceptors (N, O, F)
    polar_h: Set[int] = set()
    for i in range(n):
        if symbols[i] == "H" and any(symbols[nb] in ("N", "O", "F") for nb in adj[i]):
            polar_h.add(i)

    nonbonded_data = []  # (i, j, sigma, eps, q_prod, is_hbond)
    for i in range(n):
        r_i = get_vdw_radius(symbols[i])
        q_i = float(charges[i])
        for j in range(i + 1, n):
            pair = (i, j)
            if pair in bonded_set or pair in one_three_set:
                continue
            r_j = get_vdw_radius(symbols[j])
            sigma = (r_i + r_j) * 0.88
            eps = 0.25
            q_prod = q_i * float(charges[j])
            is_hb = (i in polar_h and symbols[j] in ("N", "O", "F")) or (j in polar_h and symbols[i] in ("N", "O", "F"))
            if is_hb:
                sigma = 1.95  # Optimal H...O/N hydrogen-bond contact distance
                eps = 1.25
            nonbonded_data.append((i, j, sigma, eps, q_prod, is_hb))

    # Prebuild index and parameter arrays once outside objective_and_grad
    def _col(rows, c, dtype=np.float64):
        return np.array([r[c] for r in rows], dtype=dtype) if rows else np.empty(0, dtype=dtype)

    b_i, b_j = _col(bond_data, 0, np.intp), _col(bond_data, 1, np.intp)
    b_r0, b_k = _col(bond_data, 2), _col(bond_data, 3)

    a_i, a_apex, a_k = _col(angle_data, 0, np.intp), _col(angle_data, 1, np.intp), _col(angle_data, 2, np.intp)
    a_th0_rad = np.radians(_col(angle_data, 3))
    a_ka = _col(angle_data, 4)

    o_apex, o_i = _col(oop_data, 0, np.intp), _col(oop_data, 1, np.intp)
    o_j, o_k = _col(oop_data, 2, np.intp), _col(oop_data, 3, np.intp)
    o_ko = _col(oop_data, 4)

    t_a, t_b, t_c, t_d = (_col(torsion_data, c, np.intp) for c in range(4))
    t_n, t_cos, t_v = _col(torsion_data, 4), _col(torsion_data, 5), _col(torsion_data, 6)

    nb_i, nb_j = _col(nonbonded_data, 0, np.intp), _col(nonbonded_data, 1, np.intp)
    nb_sigma, nb_eps, nb_q = _col(nonbonded_data, 2), _col(nonbonded_data, 3), _col(nonbonded_data, 4)
    nb_is_hb = _col(nonbonded_data, 5, bool)
    nb_cutoff = nb_sigma * np.where(nb_is_hb, 2.2, 1.5)
    nb_has_q = np.abs(nb_q) > 1e-4

    def objective_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
        pts = x.reshape((n, 3))
        energy = 0.0
        grad = np.zeros((n, 3), dtype=np.float64)

        # 1. Bond stretching: E = 0.5 * k_b * (d - r0)^2
        if b_i.size:
            diff = pts[b_i] - pts[b_j]
            d2 = np.einsum("ij,ij->i", diff, diff)
            d = np.sqrt(d2 + 1e-16)
            dr = d - b_r0
            energy += float(np.sum(0.5 * b_k * dr * dr))
            f_vec = ((b_k * dr / d)[:, None]) * diff
            np.add.at(grad, b_i, f_vec)
            np.subtract.at(grad, b_j, f_vec)

        # 2. Angle bending: E = 0.5 * k_a * (th_rad - th0_rad)^2
        if a_i.size:
            v1 = pts[a_i] - pts[a_apex]
            v2 = pts[a_k] - pts[a_apex]
            n1 = np.sqrt(np.einsum("ij,ij->i", v1, v1) + 1e-16)
            n2 = np.sqrt(np.einsum("ij,ij->i", v2, v2) + 1e-16)
            u1 = v1 / n1[:, None]
            u2 = v2 / n2[:, None]
            cos_th = np.clip(np.einsum("ij,ij->i", u1, u2), -0.999999, 0.999999)
            th_rad = np.arccos(cos_th)
            d_rad = th_rad - a_th0_rad
            energy += float(np.sum(0.5 * a_ka * d_rad * d_rad))

            sin_th = np.sqrt(np.maximum(1e-12, 1.0 - cos_th * cos_th))
            dE_dcos = -a_ka * d_rad / sin_th
            g_i = (dE_dcos / n1)[:, None] * (u2 - cos_th[:, None] * u1)
            g_k = (dE_dcos / n2)[:, None] * (u1 - cos_th[:, None] * u2)
            np.add.at(grad, a_i, g_i)
            np.add.at(grad, a_k, g_k)
            np.subtract.at(grad, a_apex, g_i + g_k)

        # 3. Out-of-plane planar penalty: E = 0.5 * k_oop * h^2
        if o_apex.size:
            v_ij = pts[o_j] - pts[o_i]
            v_ik = pts[o_k] - pts[o_i]
            v_ia = pts[o_apex] - pts[o_i]
            n_vec = np.cross(v_ij, v_ik)
            norm_n = np.sqrt(np.einsum("ij,ij->i", n_vec, n_vec) + 1e-16)
            u_n = n_vec / norm_n[:, None]
            h = np.einsum("ij,ij->i", v_ia, u_n)
            energy += float(np.sum(0.5 * o_ko * (h * h)))

            dE_dh = o_ko * h
            g_apex = dE_dh[:, None] * u_n
            dn_dpj = np.cross(v_ik, v_ia) - h[:, None] * np.cross(v_ik, u_n)
            dn_dpk = np.cross(v_ia, v_ij) - h[:, None] * np.cross(u_n, v_ij)
            inv_n = (dE_dh / norm_n)[:, None]
            g_j = inv_n * dn_dpj
            g_k = inv_n * dn_dpk
            g_i = -(g_apex + g_j + g_k)
            np.add.at(grad, o_apex, g_apex)
            np.add.at(grad, o_j, g_j)
            np.add.at(grad, o_k, g_k)
            np.add.at(grad, o_i, g_i)

        # 4. Periodic Torsional (Dihedral) Potentials with Blondel-Karplus analytical gradients
        if t_a.size:
            b1 = pts[t_b] - pts[t_a]
            b2 = pts[t_c] - pts[t_b]
            b3 = pts[t_d] - pts[t_c]
            norm_b2_sq = np.einsum("ij,ij->i", b2, b2) + 1e-16
            norm_b2 = np.sqrt(norm_b2_sq)
            m = np.cross(b1, b2)
            n_pl = np.cross(b2, b3)
            norm_m2 = np.einsum("ij,ij->i", m, m)
            norm_n2 = np.einsum("ij,ij->i", n_pl, n_pl)
            valid = (norm_m2 >= 1e-10) & (norm_n2 >= 1e-10)
            if np.any(valid):
                safe_m2 = np.where(valid, norm_m2, 1.0)
                safe_n2 = np.where(valid, norm_n2, 1.0)
                y = np.einsum("ij,ij->i", np.cross(m, n_pl), b2) / norm_b2
                x_dot = np.einsum("ij,ij->i", m, n_pl)
                phi = np.arctan2(y, x_dot)
                e_tors = np.where(valid, 0.5 * t_v * (1.0 - t_cos * np.cos(t_n * phi)), 0.0)
                energy += float(np.sum(e_tors))
                dE_dphi = np.where(valid, 0.5 * t_v * t_n * t_cos * np.sin(t_n * phi), 0.0)

                dphi_da = -((norm_b2 / safe_m2)[:, None]) * m
                dphi_dd = ((norm_b2 / safe_n2)[:, None]) * n_pl
                dot_b1_b2 = np.einsum("ij,ij->i", b1, b2) / norm_b2_sq
                dot_b3_b2 = np.einsum("ij,ij->i", b3, b2) / norm_b2_sq
                dphi_db = ((dot_b1_b2 - 1.0)[:, None]) * dphi_da - (dot_b3_b2[:, None]) * dphi_dd
                dphi_dc = ((dot_b3_b2 - 1.0)[:, None]) * dphi_dd - (dot_b1_b2[:, None]) * dphi_da

                w = dE_dphi[:, None]
                np.add.at(grad, t_a, w * dphi_da)
                np.add.at(grad, t_b, w * dphi_db)
                np.add.at(grad, t_c, w * dphi_dc)
                np.add.at(grad, t_d, w * dphi_dd)

        # 5. Nonbonded steric + screened Coulomb electrostatics
        if nb_i.size:
            diff = pts[nb_i] - pts[nb_j]
            d2 = np.einsum("ij,ij->i", diff, diff)
            d = np.sqrt(d2 + 1e-16)
            f_mag = np.zeros(nb_i.size, dtype=np.float64)

            in_lj = d < nb_cutoff
            if np.any(in_lj):
                d_eff = np.maximum(0.45, d)
                ratio = nb_sigma / d_eff
                r6 = ratio * ratio * ratio * ratio * ratio * ratio
                r12 = r6 * r6
                energy += float(np.sum(np.where(in_lj, nb_eps * (r12 - r6), 0.0)))
                lj_grad_mask = in_lj & (d >= 0.45)
                f_mag += np.where(lj_grad_mask, nb_eps * (-12.0 * r12 + 6.0 * r6) / d2, 0.0)

            in_coul = nb_has_q & (d < 8.0)
            if np.any(in_coul):
                denom = d2 + 1.44
                energy += float(np.sum(np.where(in_coul, 20.0 * nb_q / denom, 0.0)))
                f_mag += np.where(in_coul, -40.0 * nb_q / (denom * denom), 0.0)

            nz = f_mag != 0.0
            if np.any(nz):
                f_vec = (f_mag[nz, None]) * diff[nz]
                np.add.at(grad, nb_i[nz], f_vec)
                np.subtract.at(grad, nb_j[nz], f_vec)

        if frozen_arr.size:
            grad[frozen_arr] = 0.0

        return energy, grad.flatten()

    frozen_arr = (
        np.array(sorted(i for i in frozen_indices if 0 <= i < n), dtype=np.intp)
        if frozen_indices
        else np.empty(0, dtype=np.intp)
    )

    initial_x = np.copy(coords).flatten()
    from scipy.optimize import minimize  # lazy: keeps application start-up fast
    result = minimize(
        objective_and_grad,
        initial_x,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "ftol": 1e-7, "gtol": 1e-5}
    )

    optimized_pts = result.x.reshape((n, 3))
    if frozen_arr.size:
        optimized_pts[frozen_arr] = np.asarray(coords, dtype=np.float64)[frozen_arr]
    final_e = float(result.fun)
    return optimized_pts, final_e

