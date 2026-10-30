"""
GaussView-inspired "SmartSlides" (arbitrary 2/3/4-atom distance, angle, and dihedral manipulation)
and Local Stereocenter (R/S) Inversion module.

All functions in this module are pure geometric / graph transformations on
(symbols, coords, bonds) so they never interfere with existing Z-matrix parsing or table editing.
"""

import math
from collections import deque
from typing import List, Tuple, Set, Optional, Dict
import numpy as np

from .zmatrix import calculate_bond_distance, calculate_angle, calculate_dihedral
from .topology import build_adj, norm_bond
from .alignment import rodrigues_rotation, rotation_matrix_from_vectors


def get_connected_subgraph(
    n_atoms: int,
    bonds: List[Tuple[int, int]],
    start_idx: int,
    blocked_nodes: Set[int],
    blocked_edges: Optional[Set[Tuple[int, int]]] = None,
) -> List[int]:
    """
    Find all atoms reachable from `start_idx` in the bond graph without passing through
    `blocked_nodes` or traversing any edge in `blocked_edges`.
    Always includes `start_idx` itself (unless `start_idx` is out of range).
    If traversing the graph hits one of `blocked_nodes` via an alternate ring path,
    only `start_idx` (or the local branch) is moved so ring atoms on the anchor side stay safe.
    """
    if start_idx < 0 or start_idx >= n_atoms:
        return []
    adj = build_adj(n_atoms, bonds)
    blocked_e = blocked_edges or set()

    visited: Set[int] = {start_idx}
    queue = deque([start_idx])
    hit_blocked_via_cycle = False

    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            edge = norm_bond(u, v)
            if edge in blocked_e:
                continue
            if v in blocked_nodes:
                if u != start_idx:
                    hit_blocked_via_cycle = True
                continue
            if v not in visited:
                visited.add(v)
                queue.append(v)

    if hit_blocked_via_cycle:
        # In a closed ring where both sides connect back to the anchor, only move start_idx
        # and any non-ring terminal hydrogens directly attached to start_idx
        local_group = [start_idx]
        for v in adj.get(start_idx, []):
            if v not in blocked_nodes and len(adj.get(v, [])) == 1:
                local_group.append(v)
        return sorted(local_group)

    return sorted(visited)


def _get_rotatable_side_subgraph(
    n_atoms: int,
    bonds: List[Tuple[int, int]],
    axis_atom: int,
    other_axis_atom: int,
    target_atom: int,
) -> List[int]:
    """
    Identify the substituent subgraph on `axis_atom`'s side of the `other_axis_atom - axis_atom`
    central torsion axis (including any complete ring containing `axis_atom` and `target_atom`
    when `other_axis_atom - axis_atom` is a bridge bond between rings, such as in biphenyl / 2-phenylpyridine).
    If `other_axis_atom` and `axis_atom` are in the same closed ring, falls back to moving the local
    substituent branch on `target_atom`.
    """
    central_edge = {norm_bond(other_axis_atom, axis_atom)}
    adj = build_adj(n_atoms, bonds)

    # Traverse from axis_atom blocking only other_axis_atom and central_edge
    visited: Set[int] = {axis_atom}
    queue = deque([axis_atom])
    in_same_ring_with_other = False
    while queue:
        u = queue.popleft()
        for v in adj.get(u, []):
            if norm_bond(u, v) in central_edge:
                continue
            if v == other_axis_atom:
                in_same_ring_with_other = True
                continue
            if v not in visited:
                visited.add(v)
                queue.append(v)

    if not in_same_ring_with_other and (target_atom is None or target_atom in visited):
        moving = sorted(k for k in visited if k != axis_atom)
        if moving:
            return moving

    # Fallback: central bond is inside a closed ring or disconnected from target_atom
    if target_atom is not None:
        return get_connected_subgraph(
            n_atoms, bonds, target_atom, {axis_atom, other_axis_atom}, central_edge
        )
    return []


def modify_distance(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    idx1: int,
    idx2: int,
    target_distance: float,
    mode: str = "group",
) -> np.ndarray:
    """
    Set the distance between any two atoms `idx1` and `idx2` to `target_distance` (Å).

    Displacement modes (`mode`):
      - `"group"` (default): Keeps `idx1` (and its side of the molecule) fixed, and translates
        `idx2` along with its connected substituent subgraph.
      - `"symmetric"`: Translates both `idx1`'s subgraph and `idx2`'s subgraph symmetrically (50% / 50%).
      - `"atom"`: Moves ONLY atom `idx2` (leaving all other atoms, including its hydrogens, fixed).
      - `"atom_symmetric"`: Moves ONLY atoms `idx1` and `idx2` symmetrically (50% / 50%).
    """
    n = len(coords)
    out = np.array(coords, dtype=np.float64, copy=True)
    if not (0 <= idx1 < n and 0 <= idx2 < n) or idx1 == idx2:
        return out

    target_d = max(0.05, float(target_distance))
    vec = out[idx2] - out[idx1]
    curr_d = float(np.linalg.norm(vec))
    if curr_d < 1e-8:
        u_dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        u_dir = vec / curr_d

    delta = target_d - curr_d
    if abs(delta) < 1e-12:
        return out

    blocked_edge = {norm_bond(idx1, idx2)}

    if mode == "atom":
        out[idx2] += delta * u_dir
    elif mode == "atom_symmetric":
        out[idx1] -= 0.5 * delta * u_dir
        out[idx2] += 0.5 * delta * u_dir
    elif mode == "symmetric":
        grp1 = get_connected_subgraph(n, bonds, idx1, {idx2}, blocked_edge)
        grp2 = get_connected_subgraph(n, bonds, idx2, {idx1}, blocked_edge)
        out[grp1] -= 0.5 * delta * u_dir
        out[grp2] += 0.5 * delta * u_dir
    else:  # "group" (keep idx1 fixed, translate idx2's substituent group)
        grp2 = get_connected_subgraph(n, bonds, idx2, {idx1}, blocked_edge)
        out[grp2] += delta * u_dir

    return out


def modify_angle(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    idx1: int,
    idx2: int,
    idx3: int,
    target_angle_deg: float,
    mode: str = "group",
) -> np.ndarray:
    """
    Set the valence/interatomic angle `idx1 - idx2 - idx3` (with `idx2` as apex/vertex)
    to `target_angle_deg` (in degrees, [0.1, 179.9]).

    Displacement modes (`mode`):
      - `"group"` (default): Keeps `idx1` and apex `idx2` fixed, and rotates `idx3` + its
        connected substituent subgraph around apex `idx2` in the `idx1-idx2-idx3` plane.
      - `"symmetric"`: Rotates `idx1`'s subgraph by -Δθ/2 and `idx3`'s subgraph by +Δθ/2 around `idx2`.
      - `"atom"`: Rotates ONLY atom `idx3` around apex `idx2`.
    """
    n = len(coords)
    out = np.array(coords, dtype=np.float64, copy=True)
    if len({idx1, idx2, idx3}) < 3 or not all(0 <= k < n for k in (idx1, idx2, idx3)):
        return out

    target_deg = float(np.clip(target_angle_deg, 0.05, 179.95))
    curr_deg = calculate_angle(out[idx1], out[idx2], out[idx3])
    delta_rad = math.radians(target_deg - curr_deg)
    if abs(delta_rad) < 1e-10:
        return out

    apex = out[idx2].copy()
    v1 = out[idx1] - apex
    v3 = out[idx3] - apex
    normal = np.cross(v1, v3)
    norm_n = float(np.linalg.norm(normal))
    if norm_n < 1e-7:
        # Collinear fallback: pick any unit vector perpendicular to v1
        u1 = v1 / max(1e-8, float(np.linalg.norm(v1)))
        perp = np.array([1.0, 0.0, 0.0]) if abs(u1[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        normal = np.cross(u1, perp)
        normal /= float(np.linalg.norm(normal))
    else:
        normal /= norm_n

    if mode == "atom":
        R = rodrigues_rotation(normal, delta_rad)
        out[idx3] = apex + R @ (out[idx3] - apex)
    elif mode == "symmetric":
        R1 = rodrigues_rotation(normal, -0.5 * delta_rad)
        R3 = rodrigues_rotation(normal, 0.5 * delta_rad)
        grp1 = _get_rotatable_side_subgraph(n, bonds, idx2, idx3, idx1)
        grp3 = _get_rotatable_side_subgraph(n, bonds, idx2, idx1, idx3)
        out[grp1] = apex + (R1 @ (out[grp1] - apex).T).T
        out[grp3] = apex + (R3 @ (out[grp3] - apex).T).T
    else:  # "group"
        R = rodrigues_rotation(normal, delta_rad)
        grp3 = _get_rotatable_side_subgraph(n, bonds, idx2, idx1, idx3)
        out[grp3] = apex + (R @ (out[grp3] - apex).T).T

    return out


def modify_dihedral(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    idx1: int,
    idx2: int,
    idx3: int,
    idx4: int,
    target_dihedral_deg: float,
    mode: str = "group",
) -> np.ndarray:
    """
    Set the torsion/dihedral angle `idx1 - idx2 - idx3 - idx4` around the central axis `idx2 -> idx3`
    to `target_dihedral_deg` (in degrees, [-180, 180]).

    Displacement modes (`mode`):
      - `"group"` (default): Keeps the `idx1-idx2-idx3` side fixed and rotates `idx4` + the entire
        connected substituent subgraph on `idx3`'s side around the `idx2 -> idx3` bond axis.
      - `"symmetric"`: Rotates the `idx1/idx2` side by -Δφ/2 and the `idx3/idx4` side by +Δφ/2.
      - `"atom"`: Rotates ONLY atom `idx4` around the `idx2 -> idx3` axis.
    """
    n = len(coords)
    out = np.array(coords, dtype=np.float64, copy=True)
    if len({idx1, idx2, idx3, idx4}) < 4 or not all(0 <= k < n for k in (idx1, idx2, idx3, idx4)):
        return out

    curr_dih = calculate_dihedral(out[idx1], out[idx2], out[idx3], out[idx4])
    # Wrap difference into [-180, 180]
    diff_deg = (float(target_dihedral_deg) - curr_dih + 180.0) % 360.0 - 180.0
    if abs(diff_deg) < 1e-9:
        return out

    axis_vec = out[idx3] - out[idx2]
    axis_len = float(np.linalg.norm(axis_vec))
    if axis_len < 1e-8:
        return out
    axis_u = axis_vec / axis_len
    pivot = out[idx3].copy()

    # Check sign convention of calculate_dihedral: rotating +rad around (idx3 - idx2)
    # Test with a tiny rotation of out[idx4] to ensure sign matches calculate_dihedral convention
    test_R = rodrigues_rotation(axis_u, math.radians(1.0))
    test_p4 = pivot + test_R @ (out[idx4] - pivot)
    test_dih = calculate_dihedral(out[idx1], out[idx2], out[idx3], test_p4)
    step_check = (test_dih - curr_dih + 180.0) % 360.0 - 180.0
    sign = 1.0 if step_check >= 0.0 else -1.0
    rot_rad = sign * math.radians(diff_deg)

    central_edge = {norm_bond(idx2, idx3)}

    if mode == "atom":
        R = rodrigues_rotation(axis_u, rot_rad)
        out[idx4] = pivot + R @ (out[idx4] - pivot)
    elif mode == "symmetric":
        R_right = rodrigues_rotation(axis_u, 0.5 * rot_rad)
        R_left = rodrigues_rotation(axis_u, -0.5 * rot_rad)
        grp_right = _get_rotatable_side_subgraph(n, bonds, idx3, idx2, idx4)
        grp_left = _get_rotatable_side_subgraph(n, bonds, idx2, idx3, idx1)
        out[grp_right] = pivot + (R_right @ (out[grp_right] - pivot).T).T
        out[grp_left] = pivot + (R_left @ (out[grp_left] - pivot).T).T
    else:  # "group"
        R = rodrigues_rotation(axis_u, rot_rad)
        moving = _get_rotatable_side_subgraph(n, bonds, idx3, idx2, idx4)
        out[moving] = pivot + (R @ (out[moving] - pivot).T).T

    return out


def invert_chiral_center(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    selected_indices: List[int],
) -> Tuple[np.ndarray, str]:
    """
    Invert a local tetrahedral or trigonal-pyramidal stereocenter (R/S or umbrella inversion)
    without inverting the rest of the molecule!

    Selection patterns supported:
      1. 1 atom selected (`[center]`):
         - If `center` has >= 2 terminal or acyclic substituent branches, swaps the two smallest
           substituent subgraphs (e.g. H and OH/CH3) across their bisector plane, performing a true
           local R <-> S configuration inversion at `center` while keeping the main backbone fixed!
         - If `center` is a 3-coordinate pyramidal center (like NH3 or an amine R3N), reflects its
           substituents across the plane perpendicular to the pyramid axis (umbrella inversion).
      2. 3 atoms selected (`[center, subA, subB]` or `[subA, center, subB]`):
         - Swaps the two chosen substituent branches `subA` and `subB` attached to `center`.
      3. 0 atoms selected:
         - Falls back to global mirror inversion of the molecule (enantiomer flip).

    Returns `(new_coords, status_description)`.
    """
    n = len(coords)
    out = np.array(coords, dtype=np.float64, copy=True)
    if n == 0:
        return out, "No atoms to invert."

    valid = [i for i in selected_indices if 0 <= i < n]
    adj = build_adj(n, bonds)

    # Determine center and two substituent roots (sub1, sub2) if 3 atoms selected
    center: Optional[int] = None
    sub1: Optional[int] = None
    sub2: Optional[int] = None

    if len(valid) >= 3:
        a, b, c = valid[0], valid[1], valid[2]
        if b in adj.get(a, []) and c in adj.get(a, []):
            center, sub1, sub2 = a, b, c
        elif a in adj.get(b, []) and c in adj.get(b, []):
            center, sub1, sub2 = b, a, c
        elif a in adj.get(c, []) and b in adj.get(c, []):
            center, sub1, sub2 = c, a, b
        else:
            center, sub1, sub2 = a, b, c
    elif len(valid) == 1:
        center = valid[0]
        nbs = adj.get(center, [])
        if len(nbs) < 2:
            return out, f"Atom {symbols[center]}{center+1} needs at least 2 bonded neighbors to invert."

        # Find acyclic substituent branches attached to `center`
        branches: List[Tuple[int, List[int]]] = []
        for nb in nbs:
            edge = {norm_bond(center, nb)}
            grp = get_connected_subgraph(n, bonds, nb, {center}, edge)
            # Check if any other neighbor of center is in grp (which would mean nb is in a ring with center)
            in_ring = any(other_nb in grp for other_nb in nbs if other_nb != nb)
            if not in_ring:
                branches.append((nb, grp))

        # Sort acyclic branches by size (smallest first, e.g., H, F, OH, CH3) so backbone stays fixed
        branches.sort(key=lambda item: (len(item[1]), 0 if symbols[item[0]] == "H" else 1))
        if len(branches) >= 2:
            sub1, sub2 = branches[0][0], branches[1][0]
        elif len(nbs) == 3:
            # Pyramidal umbrella inversion of center along pyramid axis
            p_c = out[center]
            v_sum = sum((out[nb] - p_c) / max(1e-8, float(np.linalg.norm(out[nb] - p_c))) for nb in nbs)
            norm_v = float(np.linalg.norm(v_sum))
            if norm_v > 1e-5:
                axis_u = v_sum / norm_v
                # Reflect center across the plane of its 3 neighbors, or reflect terminal H neighbors
                plane_pt = np.mean(out[nbs], axis=0)
                h_dist = float(np.dot(p_c - plane_pt, axis_u))
                out[center] = p_c - 2.0 * h_dist * axis_u
                return out, f"Inverted pyramidal center {symbols[center]}{center+1}"
            return out, f"Center {symbols[center]}{center+1} is already planar."
        else:
            return out, f"Center {symbols[center]}{center+1} has fewer than 2 acyclic substituent groups to swap."

    if center is not None and sub1 is not None and sub2 is not None:
        p_c = out[center].copy()
        v1 = out[sub1] - p_c
        v2 = out[sub2] - p_c
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 < 1e-7 or d2 < 1e-7:
            return out, "Substituent bond vectors are zero-length."
        u1 = v1 / d1
        u2 = v2 / d2

        grp1 = get_connected_subgraph(n, bonds, sub1, {center, sub2}, {norm_bond(center, sub1)})
        grp2 = get_connected_subgraph(n, bonds, sub2, {center, sub1}, {norm_bond(center, sub2)})

        # Rotate grp1 so u1 -> u2 (preserving grp1's internal bond length d1 along u2),
        # and rotate grp2 so u2 -> u1 (preserving grp2's internal bond length d2 along u1)
        R_1to2 = rotation_matrix_from_vectors(u1, u2)
        R_2to1 = rotation_matrix_from_vectors(u2, u1)

        pts1 = out[grp1].copy()
        pts2 = out[grp2].copy()

        out[grp1] = p_c + (R_1to2 @ (pts1 - p_c).T).T
        out[grp2] = p_c + (R_2to1 @ (pts2 - p_c).T).T

        return (
            out,
            f"Inverted stereocenter {symbols[center]}{center+1} "
            f"(swapped substituents {symbols[sub1]}{sub1+1} ↔ {symbols[sub2]}{sub2+1})",
        )

    # Fallback: whole-molecule mirror reflection across XZ plane (y -> -y)
    out[:, 1] = -out[:, 1]
    return out, "Inverted molecular chirality (mirror reflection across XZ plane)"


def find_rotatable_bonds(
    symbols: List[str],
    bonds: List[Tuple[int, int]],
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
) -> List[Tuple[int, int, int, int]]:
    """
    Identify all acyclic rotatable single bonds `(idx1, u, v, idx4)` where `u - v` is a single bond
    between two non-terminal heavy atoms not in the same ring, and `idx1` / `idx4` are reference
    neighbors on `u` and `v` respectively.
    """
    n = len(symbols)
    adj = build_adj(n, bonds)
    bo = bond_orders or {}
    rotors: List[Tuple[int, int, int, int]] = []

    for u, v in sorted({norm_bond(a, b) for a, b in bonds}):
        if not (0 <= u < n and 0 <= v < n):
            continue
        if symbols[u] == "H" or symbols[v] == "H":
            continue
        if bo.get(norm_bond(u, v), 1) != 1:
            continue
        nbs_u = [k for k in adj.get(u, []) if k != v]
        nbs_v = [k for k in adj.get(v, []) if k != u]
        if not nbs_u or not nbs_v:
            continue
        # Check if u-v is inside a closed ring
        sub = _get_rotatable_side_subgraph(n, bonds, v, u, nbs_v[0])
        if not sub or u in sub:
            continue
        # Prefer heavy reference neighbors if available
        ref1 = next((k for k in nbs_u if symbols[k] != "H"), nbs_u[0])
        ref4 = next((k for k in nbs_v if symbols[k] != "H"), nbs_v[0])
        rotors.append((ref1, u, v, ref4))

    return rotors


def generate_conformers(
    symbols: List[str],
    coords: np.ndarray,
    bonds: List[Tuple[int, int]],
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    n_samples: int = 12,
    method: str = "uff_mmff",
    frozen_indices: Optional[Set[int]] = None,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Avogadro / Gabedit-style Conformational Search & Rotamer Ensemble Generator:
    1. Detects all rotatable single bonds `(ref1, u, v, ref4)`.
    2. Samples systematic 3-fold rotamers (±120°, 180°) for 1–2 rotors or low-discrepancy
       torsion combinations for >= 3 rotors (plus slight Cartesian thermal perturbation if 0 rotors).
    3. Minimizes each candidate conformer with UFF/MMFF94s or GFN0-xTB.
    4. Deduplicates conformers by heavy-atom RMSD / energy and returns `(conformer_coords_list, energies_eV)`
       sorted from lowest energy (most stable conformer at index 0) to highest energy!
    """
    from .forcefield import optimize_geometry_force_field

    base_c = np.array(coords, dtype=np.float64, copy=True)
    rotors = find_rotatable_bonds(symbols, bonds, bond_orders)
    bo = bond_orders or {}
    frozen = frozen_indices or set()

    trial_coords: List[np.ndarray] = [base_c.copy()]
    rng = np.random.default_rng(42)

    if len(rotors) == 1:
        r1, u, v, r4 = rotors[0]
        base_dih = calculate_dihedral(base_c[r1], base_c[u], base_c[v], base_c[r4])
        for step_deg in (60.0, 120.0, 180.0, -120.0, -60.0):
            trial_coords.append(
                modify_dihedral(symbols, base_c, bonds, r1, u, v, r4, base_dih + step_deg, mode="group")
            )
    elif len(rotors) >= 2:
        # Systematic + stochastic torsion combinations across all rotatable bonds
        shifts = (-120.0, 0.0, 120.0, 180.0)
        for s_idx in range(max(4, n_samples - 1)):
            c_work = base_c.copy()
            for r_i, (r1, u, v, r4) in enumerate(rotors):
                cur_d = calculate_dihedral(c_work[r1], c_work[u], c_work[v], c_work[r4])
                if len(rotors) == 2 and s_idx < 8:
                    delta = shifts[(s_idx + r_i) % len(shifts)] if r_i == 0 else shifts[(s_idx // 2) % len(shifts)]
                else:
                    delta = float(rng.choice([-120.0, -60.0, 60.0, 120.0, 180.0]))
                c_work = modify_dihedral(symbols, c_work, bonds, r1, u, v, r4, cur_d + delta, mode="group")
            trial_coords.append(c_work)
    else:
        # Ring/non-rotatable system: sample ring puckers / small out-of-plane kicks
        for _ in range(min(5, n_samples - 1)):
            kick = rng.normal(0.0, 0.22, size=base_c.shape)
            for fi in frozen:
                if 0 <= fi < len(kick):
                    kick[fi] = 0.0
            trial_coords.append(base_c + kick)

    results: List[Tuple[float, np.ndarray]] = []
    kcal_to_ev = 1.0 / 23.060548

    for cand in trial_coords[:n_samples]:
        try:
            if method == "semiempirical":
                from .semiempirical import optimize_geometry_semiempirical
                opt_c, e_kcal, _ = optimize_geometry_semiempirical(
                    symbols, cand, user_bonds=bonds, max_iter=80,
                    user_bond_orders=bo, frozen_indices=frozen
                )
            else:
                opt_c, e_kcal = optimize_geometry_force_field(
                    symbols, cand, user_bonds=bonds, max_iter=120,
                    user_bond_orders=bo, method="uff_mmff", frozen_indices=frozen
                )
            results.append((float(e_kcal) * kcal_to_ev, opt_c))
        except Exception:
            continue

    if not results:
        return [base_c], [0.0]

    # Sort from lowest energy to highest energy
    results.sort(key=lambda item: item[0])

    # Deduplicate conformers that converged to the same energy & interatomic distance matrix
    unique_coords: List[np.ndarray] = []
    unique_energies: List[float] = []

    def _dist_signature(c: np.ndarray) -> np.ndarray:
        diff = c[:, None, :] - c[None, :, :]
        return np.sort(np.linalg.norm(diff, axis=-1), axis=-1)

    sigs: List[np.ndarray] = []
    for e_ev, c in results:
        sig = _dist_signature(c)
        is_dup = False
        for prev_e, prev_sig in zip(unique_energies, sigs):
            if abs(e_ev - prev_e) < 0.002 and float(np.max(np.abs(sig - prev_sig))) < 0.12:
                is_dup = True
                break
        if not is_dup:
            unique_coords.append(c)
            unique_energies.append(e_ev)
            sigs.append(sig)

    return unique_coords, unique_energies
