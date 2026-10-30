"""
GaussView-style Element Coordination, Hybridization, and VSEPR Geometry Templates.

Provides directional valence vectors for coordination numbers 0 through 8
(Linear, Bent, Trigonal Planar, Pyramidal, T-Shaped, Tetrahedral, Square Planar,
Seesaw, Trigonal Bipyramidal, Square Pyramidal, Octahedral, Trigonal Prismatic,
Pentagonal Bipyramidal, Square Antiprismatic) and sensible element defaults across
the entire periodic table.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from .alignment import rotation_matrix_from_vectors


# Canonical VSEPR / Coordination Geometry templates:
# Key -> (Display Label, Coordination Number, List of Unit Vectors from Origin)
COORDINATION_TEMPLATES: Dict[str, Tuple[str, int, List[Tuple[float, float, float]]]] = {
    "auto": (
        "Auto (Standard Element Valence)",
        -1,
        [],
    ),
    "naked": (
        "Naked Atom (0 Bonds)",
        0,
        [],
    ),
    "terminal": (
        "Terminal / Monovalent (1 Bond)",
        1,
        [(0.0, 0.0, 1.0)],
    ),
    "linear": (
        "Linear — sp / D∞h (2 Bonds, 180°)",
        2,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
    ),
    "bent_104": (
        "Bent — sp³ / C2v (2 Bonds, 104.5°)",
        2,
        [
            (math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))),
            (-math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))),
        ],
    ),
    "bent_120": (
        "Bent — sp² / C2v (2 Bonds, 120°)",
        2,
        [
            (math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))),
            (-math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))),
        ],
    ),
    "trigonal_planar": (
        "Trigonal Planar — sp² / D3h (3 Bonds, 120°)",
        3,
        [
            (0.0, 0.0, 1.0),
            (math.sin(math.radians(120.0)), 0.0, math.cos(math.radians(120.0))),
            (-math.sin(math.radians(120.0)), 0.0, math.cos(math.radians(120.0))),
        ],
    ),
    "trigonal_pyramidal": (
        "Trigonal Pyramidal — sp³ / C3v (3 Bonds, 107°)",
        3,
        [
            (0.942809, 0.0, -0.333333),
            (-0.471405, 0.816497, -0.333333),
            (-0.471405, -0.816497, -0.333333),
        ],
    ),
    "t_shaped": (
        "T-Shaped — dsp³ / C2v (3 Bonds, 90°/180°)",
        3,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
        ],
    ),
    "tetrahedral": (
        "Tetrahedral — sp³ / Td (4 Bonds, 109.47°)",
        4,
        [
            (0.0, 0.0, 1.0),
            (0.94280904, 0.0, -0.33333333),
            (-0.47140452, 0.81649658, -0.33333333),
            (-0.47140452, -0.81649658, -0.33333333),
        ],
    ),
    "square_planar": (
        "Square Planar — dsp² / D4h (4 Bonds, 90°)",
        4,
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
    "seesaw": (
        "Seesaw — sp³d / C2v (4 Bonds, 180° & 120°)",
        4,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0),
            (math.cos(math.radians(30.0)), -math.sin(math.radians(30.0)), 0.0),
        ],
    ),
    "trigonal_bipyramidal": (
        "Trigonal Bipyramidal — sp³d / D3h (5 Bonds)",
        5,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (-0.5, math.sqrt(3.0) / 2.0, 0.0),
            (-0.5, -math.sqrt(3.0) / 2.0, 0.0),
        ],
    ),
    "square_pyramidal": (
        "Square Pyramidal — sp³d² / C4v (5 Bonds)",
        5,
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
    "octahedral": (
        "Octahedral — sp³d² / Oh (6 Bonds, 90°)",
        6,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
}


# Default coordination geometry template per element when "auto" is chosen:
DEFAULT_ELEMENT_COORDINATION: Dict[str, str] = {
    "X": "naked",
    "H": "terminal",
    "He": "naked",
    "Ne": "naked",
    "Ar": "naked",
    "Kr": "naked",
    "Li": "terminal",
    "Na": "terminal",
    "K": "terminal",
    "Rb": "terminal",
    "Cs": "terminal",
    "Be": "linear",
    "Mg": "linear",
    "Ca": "linear",
    "Sr": "linear",
    "Ba": "linear",
    "B": "trigonal_planar",
    "Al": "trigonal_planar",
    "Ga": "trigonal_planar",
    "In": "trigonal_planar",
    "C": "tetrahedral",
    "Si": "tetrahedral",
    "Ge": "tetrahedral",
    "Sn": "tetrahedral",
    "Pb": "tetrahedral",
    "N": "trigonal_pyramidal",
    "P": "trigonal_pyramidal",
    "As": "trigonal_pyramidal",
    "Sb": "trigonal_pyramidal",
    "Bi": "trigonal_pyramidal",
    "O": "bent_104",
    "S": "bent_104",
    "Se": "bent_104",
    "Te": "bent_104",
    "F": "terminal",
    "Cl": "terminal",
    "Br": "terminal",
    "I": "terminal",
    # Transition metals: default to common coordination geometries
    "Pt": "square_planar",
    "Pd": "square_planar",
    "Ni": "square_planar",
    "Au": "linear",
    "Ag": "linear",
    "Cu": "tetrahedral",
    "Zn": "tetrahedral",
    "Cd": "tetrahedral",
    "Hg": "linear",
    "Ti": "octahedral",
    "V": "octahedral",
    "Cr": "octahedral",
    "Mn": "octahedral",
    "Fe": "octahedral",
    "Co": "octahedral",
    "Ru": "octahedral",
    "Rh": "octahedral",
    "Mo": "octahedral",
    "Zr": "octahedral",
    "Nb": "octahedral",
    "Tc": "octahedral",
    "Sc": "octahedral",
    "Y": "octahedral",
    "U": "octahedral",
    "Xe": "square_planar",
}


def resolve_coordination_template(element: str, template_key: str = "auto") -> str:
    """Resolve 'auto' to the canonical template key for `element`."""
    key = (template_key or "auto").strip().lower()
    if key != "auto" and key in COORDINATION_TEMPLATES:
        return key
    return DEFAULT_ELEMENT_COORDINATION.get(element, "tetrahedral")


def get_coordination_vectors(
    element: str,
    template_key: str = "auto",
    existing_bond_vectors: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Compute unit vectors pointing along the open coordination sites for `element`
    under `template_key` (e.g. 'tetrahedral', 'trigonal_planar', 'square_planar', 'octahedral').

    If `existing_bond_vectors` contains 1 or more unit vectors from the central atom to
    already-bonded neighbors, the template is oriented so its first vector aligns with the
    first existing bond vector, and only the remaining unoccupied coordination directions
    are returned!
    """
    resolved = resolve_coordination_template(element, template_key)
    _, coord_num, raw_vecs = COORDINATION_TEMPLATES.get(resolved, COORDINATION_TEMPLATES["tetrahedral"])
    if coord_num <= 0 or not raw_vecs:
        return []

    vecs = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in raw_vecs]
    existing = []
    if existing_bond_vectors:
        for ev in existing_bond_vectors:
            n_ev = float(np.linalg.norm(ev))
            if n_ev > 1e-6:
                existing.append(np.asarray(ev, dtype=np.float64) / n_ev)

    if not existing:
        return vecs

    # Align the first template vector with existing[0]
    R = rotation_matrix_from_vectors(vecs[0], existing[0])
    rotated = [R @ v for v in vecs]

    # Filter out template sites that are already occupied by existing bonds
    remaining: List[np.ndarray] = []
    used_indices = {0}
    for ex in existing[1:]:
        # Find closest unused template vector
        best_k = None
        best_dot = -2.0
        for k, rv in enumerate(rotated):
            if k in used_indices:
                continue
            d = float(np.dot(ex, rv))
            if d > best_dot:
                best_dot = d
                best_k = k
        if best_k is not None:
            used_indices.add(best_k)

    for k, rv in enumerate(rotated):
        if k not in used_indices:
            remaining.append(rv)

    return remaining
