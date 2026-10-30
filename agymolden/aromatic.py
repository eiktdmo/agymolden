"""Aromaticity and resonance (delocalized bond) detection.

Bond orders elsewhere in AgyMolden are integers (a Kekulé structure). This module
derives, from topology + geometry, which bonds are *delocalized* so they can be
displayed and treated as bond order 1.5 instead of alternating single/double:

* aromatic rings (planar, 5-7 membered, Hückel 4n+2 pi electrons), including
  fused rings (each ring is tested separately);
* nitro groups  (N with two terminal O);
* carboxylate-like groups (C with two terminal O, no H on the singly bonded one).
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

Bond = Tuple[int, int]

MAX_RING = 7
PLANARITY_TOL = 0.30  # Angstrom, max deviation of ring atoms from the best plane


def _norm(i: int, j: int) -> Bond:
    return (i, j) if i < j else (j, i)


def find_rings(n_atoms: int, bonds: Iterable[Bond], max_size: int = MAX_RING) -> List[Tuple[int, ...]]:
    """Return unique simple cycles of length 3..max_size as ordered atom tuples."""
    adj: Dict[int, Set[int]] = {i: set() for i in range(n_atoms)}
    for u, v in bonds:
        if 0 <= u < n_atoms and 0 <= v < n_atoms and u != v:
            adj[u].add(v)
            adj[v].add(u)

    seen: Set[frozenset] = set()
    rings: List[Tuple[int, ...]] = []

    def dfs(start: int, node: int, path: List[int]):
        for nb in adj[node]:
            if nb == start and len(path) >= 3:
                key = frozenset(path)
                if key not in seen:
                    seen.add(key)
                    rings.append(tuple(path))
            elif nb > start and nb not in path and len(path) < max_size:
                path.append(nb)
                dfs(start, nb, path)
                path.pop()

    for s in range(n_atoms):
        if len(adj[s]) >= 2:
            dfs(s, s, [s])
    return rings


def _is_planar(coords: np.ndarray, ring: Tuple[int, ...], tol: float = PLANARITY_TOL) -> bool:
    pts = coords[list(ring)]
    centered = pts - pts.mean(axis=0)
    # smallest singular vector = plane normal
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dev = np.abs(centered @ vt[-1])
    return bool(dev.max() <= tol)


def _pi_electrons(
    atom: int,
    symbol: str,
    neighbors: Set[int],
    symbols: List[str],
    ring: Set[int],
    orders: Dict[Bond, int],
) -> Optional[int]:
    """Pi electrons an atom contributes to a ring, or None if it cannot be part of an aromatic ring."""
    nb = len(neighbors)
    # An exocyclic double bond (C=O, C=C ...) takes the pi electron out of the ring
    for other in neighbors:
        if other not in ring and orders.get(_norm(atom, other), 1) >= 2:
            return None
    if symbol == "C":
        return 1 if nb == 3 else None
    if symbol == "N":
        if nb == 2:
            return 1  # pyridine-like
        if nb == 3:
            return 2  # pyrrole-like lone pair
        return None
    if symbol in ("O", "S", "Se"):
        return 2 if nb == 2 else None
    if symbol in ("B", "P"):
        return 1 if nb == 3 else None
    return None


def find_aromatic_rings(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Iterable[Bond],
    bond_orders: Optional[Dict[Bond, int]] = None,
) -> List[Tuple[int, ...]]:
    """Return rings (5-7 atoms) that satisfy planarity and the Hückel 4n+2 rule."""
    n = len(symbols)
    if n == 0 or len(coords) != n:
        return []
    bonds = [_norm(u, v) for u, v in bonds]
    orders = {_norm(*k): v for k, v in (bond_orders or {}).items()}
    nbrs: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for u, v in bonds:
        nbrs[u].add(v)
        nbrs[v].add(u)

    result = []
    for ring in find_rings(n, bonds, MAX_RING):
        if len(ring) < 5:
            continue
        rset = set(ring)
        total = 0
        ok = True
        for a in ring:
            e = _pi_electrons(a, symbols[a], nbrs[a], symbols, rset, orders)
            if e is None:
                ok = False
                break
            total += e
        if not ok or total < 6 or (total - 2) % 4 != 0:
            continue
        if not _is_planar(np.asarray(coords, dtype=float), ring):
            continue
        result.append(ring)
    return result


def _terminal(atom: int, nbrs: Dict[int, Set[int]], symbols: List[str], sym: str) -> bool:
    return symbols[atom] == sym and len(nbrs[atom]) == 1


def find_delocalized_bonds(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Iterable[Bond],
    bond_orders: Optional[Dict[Bond, int]] = None,
) -> Dict[Bond, Optional[np.ndarray]]:
    """Map each delocalized bond to a unit vector pointing to the inside of its ring.

    The vector is None for non-ring resonance bonds (nitro, carboxylate).
    """
    n = len(symbols)
    out: Dict[Bond, Optional[np.ndarray]] = {}
    if n == 0 or len(coords) != n:
        return out
    coords = np.asarray(coords, dtype=float)
    bonds = [_norm(u, v) for u, v in bonds]
    nbrs: Dict[int, Set[int]] = {i: set() for i in range(n)}
    for u, v in bonds:
        nbrs[u].add(v)
        nbrs[v].add(u)

    # aromatic rings: remember the smallest ring for each bond (for the inward direction)
    best_ring: Dict[Bond, Tuple[int, ...]] = {}
    for ring in find_aromatic_rings(symbols, coords, bonds, bond_orders):
        for k in range(len(ring)):
            b = _norm(ring[k], ring[(k + 1) % len(ring)])
            if b not in best_ring or len(ring) < len(best_ring[b]):
                best_ring[b] = ring
    for b, ring in best_ring.items():
        centroid = coords[list(ring)].mean(axis=0)
        mid = 0.5 * (coords[b[0]] + coords[b[1]])
        axis = coords[b[1]] - coords[b[0]]
        ln = np.linalg.norm(axis)
        vec = None
        if ln > 1e-9:
            axis = axis / ln
            inward = centroid - mid
            inward = inward - np.dot(inward, axis) * axis
            m = np.linalg.norm(inward)
            if m > 1e-9:
                vec = inward / m
        out[b] = vec

    # resonance groups: nitro and carboxylate
    for c in range(n):
        sym = symbols[c]
        if sym not in ("N", "C"):
            continue
        t_oxy = [a for a in nbrs[c] if _terminal(a, nbrs, symbols, "O")]
        if len(t_oxy) != 2:
            continue
        if sym == "N" and len(nbrs[c]) == 3:
            for o in t_oxy:
                out.setdefault(_norm(c, o), None)
        elif sym == "C" and len(nbrs[c]) == 3:
            # a carboxylic acid has an O-H (the O would not be terminal), so two terminal O = carboxylate
            for o in t_oxy:
                out.setdefault(_norm(c, o), None)
    return out
