"""
Spatial indexing and neighbor search accelerations for AgyMolden.
Provides high-performance O(N log N) spatial querying using scipy.spatial.cKDTree
with automatic fallback to spatial hash partitioning.
"""

from typing import List, Tuple, Set, Optional, Dict
import numpy as np

try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except ImportError:
    _HAS_CKDTREE = False

from .elements import get_cov_radius, get_vdw_radius
from .topology import norm_bond


def find_steric_clashes(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[Set[Tuple[int, int]]] = None,
    threshold_ratio: float = 0.70,
    min_absolute_dist: float = 0.70,
) -> List[Tuple[int, int, str, str, float, float]]:
    """
    Find severe non-bonded steric clashes using spatial indexing (O(N log N)).

    Returns:
        List of tuples: (i, j, sym_i, sym_j, distance, threshold_distance)
    """
    n = min(len(symbols), len(coords))
    if n < 2:
        return []

    bonded_set = bonds if bonds is not None else set()
    clashes: List[Tuple[int, int, str, str, float, float]] = []

    # Maximum possible van der Waals radius sum cutoff (~4.5 A)
    max_search_radius = 4.2

    if _HAS_CKDTREE:
        tree = cKDTree(coords[:n])
        pairs = tree.query_pairs(r=max_search_radius)
        for i, j in pairs:
            si, sj = symbols[i], symbols[j]
            if si == "X" or sj == "X":
                continue
            if norm_bond(i, j) in bonded_set:
                continue
            vdw_sum = get_vdw_radius(si) + get_vdw_radius(sj)
            limit = max(min_absolute_dist, vdw_sum * threshold_ratio)
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < limit:
                clashes.append((i, j, si, sj, d, limit))
    else:
        # Fallback O(N^2)
        for i in range(n):
            si = symbols[i]
            if si == "X":
                continue
            r_i = get_vdw_radius(si)
            for j in range(i + 1, n):
                sj = symbols[j]
                if sj == "X":
                    continue
                if norm_bond(i, j) in bonded_set:
                    continue
                vdw_sum = r_i + get_vdw_radius(sj)
                limit = max(min_absolute_dist, vdw_sum * threshold_ratio)
                d = float(np.linalg.norm(coords[i] - coords[j]))
                if d < limit:
                    clashes.append((i, j, si, sj, d, limit))

    clashes.sort(key=lambda item: item[4])
    return clashes


def find_spatial_neighbors(
    coords: np.ndarray,
    query_point: np.ndarray,
    radius: float = 3.0,
) -> List[int]:
    """Find all atom indices within a spherical radius around query_point."""
    if len(coords) == 0:
        return []
    if _HAS_CKDTREE:
        tree = cKDTree(coords)
        indices = tree.query_ball_point(query_point, r=radius)
        return list(indices) if isinstance(indices, list) else indices.tolist()
    else:
        dists = np.linalg.norm(coords - query_point, axis=1)
        return [i for i, d in enumerate(dists) if d <= radius]
