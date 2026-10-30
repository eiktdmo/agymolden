"""
Explicit-topology helpers.

All structural edits (fragment replacement, bond formation) work on an explicit
bond list so that no spurious distance-based bonds are ever created, every
hydrogen has exactly one bond, and every heavy atom ends up with the correct
valence.
"""
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

from .elements import estimate_bond_length
from .forcefield import DEFAULT_VALENCES, _calculate_hydrogen_directions

Bond = Tuple[int, int]
MONOVALENT = {"H", "F", "Cl", "Br", "I"}


def norm_bond(u: int, v: int) -> Bond:
    return (u, v) if u < v else (v, u)


def build_adj(n: int, bonds) -> Dict[int, List[int]]:
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in bonds:
        if v not in adj[u]:
            adj[u].append(v)
            adj[v].append(u)
    return adj


def remove_atoms(symbols, coords, bonds, remove: Set[int], bond_orders=None):
    """Delete atoms and remap bonds / bond orders. Returns (symbols, coords, bonds, bond_orders, old_to_new)."""
    keep = [i for i in range(len(symbols)) if i not in remove]
    o2n = {o: k for k, o in enumerate(keep)}
    new_syms = [symbols[i] for i in keep]
    new_coords = np.array(coords, dtype=np.float64)[keep] if keep else np.empty((0, 3))
    new_bonds = sorted({norm_bond(o2n[u], o2n[v]) for u, v in bonds if u in o2n and v in o2n and u != v})
    new_bo = {}
    if bond_orders:
        for (u, v), o in bond_orders.items():
            if u in o2n and v in o2n:
                new_bo[norm_bond(o2n[u], o2n[v])] = o
    return new_syms, new_coords, new_bonds, new_bo, o2n


def fix_valence_explicit(symbols, coords, bonds, bond_orders=None, add_h: bool = True,
                         protect: Optional[Set[int]] = None,
                         target_indices: Optional[Set[int]] = None):
    """
    Enforce chemically valid valences on an explicit bond graph:
      - no bonds between two monovalent atoms, each monovalent atom keeps exactly one bond
        (to its nearest heavy partner); orphan hydrogens are deleted
      - heavy atoms over valence lose hydrogens (farthest from the protected direction first)
      - heavy atoms under valence get geometrically placed hydrogens (if add_h)
      - if target_indices is provided, only those atoms (plus any excess-valence atoms) have hydrogens added/removed,
        and aromatic 6-membered rings / distance-detected double/triple bonds are strictly respected.
    Returns (symbols, coords, bonds, bond_orders).
    """
    from .forcefield import detect_bond_orders
    symbols = list(symbols)
    coords = np.array(coords, dtype=np.float64)
    bo = {norm_bond(*k): v for k, v in (bond_orders or {}).items()}
    bset = {norm_bond(u, v) for u, v in bonds if u != v}
    # 1. Remove monovalent-monovalent bonds
    bset = {b for b in bset if not (symbols[b[0]] in MONOVALENT and symbols[b[1]] in MONOVALENT)}
    # 2. Each monovalent keeps only its nearest partner
    adj = build_adj(len(symbols), bset)
    for i, s in enumerate(symbols):
        if s in MONOVALENT and len(adj[i]) > 1:
            best = min(adj[i], key=lambda j: np.linalg.norm(coords[i] - coords[j]))
            for j in adj[i]:
                if j != best:
                    bset.discard(norm_bond(i, j))
    adj = build_adj(len(symbols), bset)

    # Merge auto-detected double/triple bonds for pairs not explicitly in bo
    auto_bo = detect_bond_orders(symbols, coords, sorted(bset))
    for k, v in auto_bo.items():
        if k not in bo:
            bo[k] = v

    # Detect 6-membered aromatic rings (e.g. benzene/phenyl/pyridyl where ring atoms have 3 bonds and ~1.39 A C-C)
    aromatic_atoms: Set[int] = set()
    heavy = [idx for idx, s in enumerate(symbols) if s not in MONOVALENT and s != "X"]
    def _find_6rings(start, curr, path):
        for nb in adj[curr]:
            if nb == start and len(path) == 6:
                if all(symbols[x] in ("C", "N") and len(adj[x]) <= 3 for x in path):
                    # Check average bond length in ring is in aromatic / sp2 range (< 1.45 A)
                    dists = [float(np.linalg.norm(coords[path[m]] - coords[path[(m + 1) % 6]])) for m in range(6)]
                    if max(dists) < 1.46:
                        aromatic_atoms.update(path)
            elif nb not in path and nb in heavy and len(path) < 6:
                _find_6rings(start, nb, path + [nb])
    for h_idx in heavy:
        _find_6rings(h_idx, h_idx, [h_idx])

    # 3. Orphan hydrogens and excess hydrogens
    has_ghost = any(s == "X" for s in symbols)
    remove: Set[int] = (
        set() if has_ghost else {i for i, s in enumerate(symbols) if s == "H" and not adj[i] and len(symbols) > 1}
    )
    for i, s in enumerate(symbols):
        if s in MONOVALENT or s == "X":
            continue
        val = DEFAULT_VALENCES.get(s, 4)
        total = sum(bo.get(norm_bond(i, j), 1) for j in adj[i] if symbols[j] != "X")
        if i in aromatic_atoms and total == 3 and val == 4:
            total = 4  # Aromatic carbon with 3 sigma bonds has fulfilled valence (4)
        excess = total - val
        if excess > 0:
            hs = [j for j in adj[i] if symbols[j] == "H" and j not in remove]
            # Remove H closest to other heavy neighbors (most crowded) first
            heavy_nbs = [j for j in adj[i] if symbols[j] != "H"]
            hs.sort(key=lambda h: min([np.linalg.norm(coords[h] - coords[k]) for k in heavy_nbs] or [0.0]))
            for h in hs[:excess]:
                remove.add(h)
    old_to_new_rem = {i: i for i in range(len(symbols))}
    if remove:
        symbols, coords, bset_l, bo, old_to_new_rem = remove_atoms(symbols, coords, bset, remove, bo)
        bset = set(bset_l)
        aromatic_atoms = {old_to_new_rem[a] for a in aromatic_atoms if a in old_to_new_rem}
        if target_indices is not None:
            target_indices = {old_to_new_rem[t] for t in target_indices if t in old_to_new_rem}
    # 4. Add missing hydrogens
    if add_h:
        adj = build_adj(len(symbols), bset)
        n0 = len(symbols)
        new_syms, new_pos = [], []
        for i in range(n0):
            if target_indices is not None and i not in target_indices:
                continue
            s = symbols[i]
            if s in MONOVALENT or s == "X" or s not in DEFAULT_VALENCES:
                continue
            val = DEFAULT_VALENCES.get(s, 4)
            total = sum(bo.get(norm_bond(i, j), 1) for j in adj[i] if symbols[j] != "X")
            if i in aromatic_atoms and total >= 3 and val == 4:
                continue
            need = val - total
            if need <= 0:
                continue
            vecs = []
            for j in adj[i]:
                v = coords[j] - coords[i]
                nv = np.linalg.norm(v)
                if nv > 1e-6:
                    vecs.append(v / nv)
            dirs = _calculate_hydrogen_directions(vecs, need)
            d = estimate_bond_length(s, "H")
            for k, dvec in enumerate(dirs[:need]):
                idx = n0 + len(new_syms)
                new_syms.append("H")
                new_pos.append(coords[i] + dvec * d)
                bset.add(norm_bond(i, idx))
        if new_syms:
            symbols = symbols + new_syms
            coords = np.vstack([coords, np.array(new_pos)])
    return symbols, coords, sorted(bset), bo


class BondFormationError(Exception):
    """Raised when a bond cannot be formed (e.g. no hydrogen available to free valence)."""


def form_single_bond(symbols, coords, bonds, bond_orders, i: int, j: int):
    """
    Form a new bond between atoms i and j following strict rules:
      1. The new bond is ALWAYS a single bond (order 1).
      2. If a center would exceed its valence, hydrogens bonded to it are removed
         (the ones pointing toward the partner first).
      3. If no hydrogen can be removed to satisfy valence, BondFormationError is raised
         with a message saying so.
      4. No other bond is created or changed, and no hydrogen is ever added.
    Selecting a hydrogen means "use its parent and remove this hydrogen".
    Ghost atoms 'X' can form bonds freely (no valence restrictions).
    Returns (symbols, coords, bonds, bond_orders, new_i, new_j, removed_old_indices).
    """
    symbols = list(symbols)
    coords = np.array(coords, dtype=np.float64)
    bonds = sorted({norm_bond(u, v) for u, v in bonds if u != v})
    bo = {norm_bond(*k): v for k, v in (bond_orders or {}).items() if v != 1}
    adj = build_adj(len(symbols), bonds)

    remove: Set[int] = set()
    centers = []
    for s_idx in (i, j):
        if symbols[s_idx] == "H" and adj[s_idx]:
            remove.add(s_idx)
            centers.append(adj[s_idx][0])
        else:
            centers.append(s_idx)
    a, b = centers
    if a == b:
        raise BondFormationError("Both selections belong to the same center.")
    if b in adj[a]:
        raise BondFormationError(f"{symbols[a]}{a+1} and {symbols[b]}{b+1} are already bonded.")

    for c, o in ((a, b), (b, a)):
        if symbols[c] == "X":
            continue  # Ghost atom has no valence restriction
        val = DEFAULT_VALENCES.get(symbols[c], 4)
        used = sum(bo.get(norm_bond(c, k), 1) for k in adj[c] if k not in remove and symbols[k] != "X")
        excess = used + 1 - val
        if excess > 0:
            hs = sorted([k for k in adj[c] if symbols[k] == "H" and k not in remove],
                        key=lambda h: np.linalg.norm(coords[h] - coords[o]))
            if len(hs) < excess:
                raise BondFormationError(
                    f"There are no hydrogen atoms on {symbols[c]}{c+1} that can be removed "
                    f"to fulfill its valence ({val}). Bond not formed."
                )
            remove.update(hs[:excess])

    bonds = bonds + [norm_bond(a, b)]
    symbols, coords, bonds, bo, o2n = remove_atoms(symbols, coords, bonds, remove, bo)
    bo.pop(norm_bond(o2n[a], o2n[b]), None)  # always single
    return symbols, coords, bonds, bo, o2n[a], o2n[b], sorted(remove)


def bfs_reorder(symbols, coords, bonds, bond_orders=None, root: int = 0):
    """Reorder atoms (heavy atoms first along a BFS, H right after their parent) so each atom's
    bonded parent precedes it. Returns (symbols, coords, bonds, bond_orders, old_to_new)."""
    n = len(symbols)
    if n == 0:
        return symbols, coords, list(bonds), dict(bond_orders or {}), {}
    adj = build_adj(n, bonds)
    order: List[int] = []
    seen = set()
    starts = [root] + [i for i in range(n) if symbols[i] != "H"] + list(range(n))
    for st in starts:
        if st in seen:
            continue
        seen.add(st)
        queue = [st]
        while queue:
            u = queue.pop(0)
            order.append(u)
            nbs = sorted(adj[u], key=lambda j: (symbols[j] == "H", j))
            for j in nbs:
                if j not in seen:
                    seen.add(j)
                    queue.append(j)
    o2n = {o: k for k, o in enumerate(order)}
    new_syms = [symbols[o] for o in order]
    new_coords = np.array(coords, dtype=np.float64)[order]
    new_bonds = sorted({norm_bond(o2n[u], o2n[v]) for u, v in bonds if u in o2n and v in o2n})
    new_bo = {norm_bond(o2n[u], o2n[v]): o for (u, v), o in (bond_orders or {}).items() if u in o2n and v in o2n and o != 1}
    return new_syms, new_coords, new_bonds, new_bo, o2n


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b."""
    a = a / (np.linalg.norm(a) or 1.0)
    b = b / (np.linalg.norm(b) or 1.0)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + 2.0 * (K @ K)
    axis /= s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1.0 - c) * (K @ K)


def fragment_outward_axis(frag_coords: np.ndarray, root_idx: int = 0) -> np.ndarray:
    """Axis from fragment root (default atom 0) toward the centroid of the rest of the fragment."""
    n = len(frag_coords)
    if n < 2:
        return np.array([0.0, 0.0, 1.0])
    root = max(0, min(n - 1, root_idx))
    other_indices = [k for k in range(n) if k != root]
    if not other_indices:
        return np.array([0.0, 0.0, 1.0])
    v = np.mean(frag_coords[other_indices], axis=0) - frag_coords[root]
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
