"""Builder Invariant Property Testing.

Verifies strict physical and chemical invariants across all molecular builder operations:
1. Clearance invariant: min_{i != j} ||r_i - r_j|| >= 0.80 Å (no severe steric overlap).
2. Valency invariant: maximum covalent degree matches standard valence rules (H: 1, C <= 4, N <= 4, O <= 2, Halogens: 1).
3. Exact formula preservation:
   - Benzene (C6H6) + Benzene edge fusion -> Naphthalene (C10H8).
   - Naphthalene (C10H8) + Benzene bay peri-annulation -> Phenalene (C13H10).
   - Naphthalene (C10H8) + Cyclopentyl bay peri-annulation -> Acenaphthene (C12H10).
4. Planarity preservation: aromatic rings remain planar within tight tolerance.
"""

import numpy as np
import pytest
from collections import Counter

from agymolden.cycle_welder import (
    fuse_ring_to_bond,
    fuse_ring_to_triad,
    detect_peri_bays,
)
from agymolden.forcefield import detect_connectivity


def _make_benzene():
    """Return standard planar Benzene (C6H6) coordinates and symbols."""
    r_cc = 1.40
    r_ch = 1.09
    symbols = ["C"] * 6 + ["H"] * 6
    coords = []
    # 6 Carbons
    for i in range(6):
        theta = i * np.pi / 3.0
        coords.append([r_cc * np.cos(theta), r_cc * np.sin(theta), 0.0])
    # 6 Hydrogens
    for i in range(6):
        theta = i * np.pi / 3.0
        coords.append([(r_cc + r_ch) * np.cos(theta), (r_cc + r_ch) * np.sin(theta), 0.0])
    return symbols, np.array(coords, dtype=np.float64)


def _check_clearance(coords: np.ndarray, min_dist: float = 0.80):
    """Assert no two atoms are closer than min_dist."""
    n = len(coords)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            assert d >= min_dist, f"Steric clash between atom {i} and {j}: distance = {d:.3f} Å < {min_dist} Å"


def _check_valencies(symbols: list, coords: np.ndarray):
    """Assert all atomic degrees obey chemical valence rules."""
    bonds = detect_connectivity(symbols, coords)
    degrees = Counter()
    for u, v in bonds:
        degrees[u] += 1
        degrees[v] += 1

    max_valences = {
        "H": 1,
        "C": 4,
        "N": 4,
        "O": 2,
        "F": 1,
        "Cl": 1,
        "Br": 1,
        "I": 1,
        "S": 6,
        "P": 5,
    }

    for idx, sym in enumerate(symbols):
        deg = degrees[idx]
        max_v = max_valences.get(sym, 6)
        assert deg <= max_v, f"Valence explosion on atom {idx} ({sym}): degree = {deg} > max allowed {max_v}"
        if sym == "H":
            assert deg == 1, f"Hydrogen {idx} has degree {deg} != 1"


def test_invariant_benzene_to_naphthalene_edge_fusion():
    """Fusing benzene onto benzene edge must yield exact C10H8 Naphthalene with proper clearance and valency."""
    syms, crds = _make_benzene()

    # Fuse on bond (C0, C1)
    res_syms, res_crds = fuse_ring_to_bond(
        syms, crds, 0, 1,
        ring_preset="Benzene (6-membered aromatic)",
        relax=True,
    )

    counts = Counter(res_syms)
    assert counts["C"] == 10, f"Expected 10 carbons, got {counts['C']}"
    assert counts["H"] == 8, f"Expected 8 hydrogens, got {counts['H']}"

    _check_clearance(res_crds, min_dist=0.85)
    _check_valencies(res_syms, res_crds)

    # Check planarity: all z coordinates should be close to 0
    z_deviations = np.abs(res_crds[:, 2])
    assert np.max(z_deviations) < 0.10, f"Planarity violated: max z deviation = {np.max(z_deviations):.3f} Å"


def test_invariant_naphthalene_to_phenalene_peri_annulation():
    """Peri-annulation on Naphthalene bay triad must yield C13H10 Phenalene without steric clash."""
    syms, crds = _make_benzene()
    # Step 1: Benzene -> Naphthalene (C10H8)
    naph_syms, naph_crds = fuse_ring_to_bond(
        syms, crds, 0, 1,
        ring_preset="Benzene (6-membered aromatic)",
        relax=True,
    )

    # Detect bay triads in Naphthalene
    bonds = detect_connectivity(naph_syms, naph_crds)
    bays = detect_peri_bays(naph_syms, naph_crds, bonds)
    assert len(bays) >= 2, f"Expected at least 2 peri-bays in naphthalene, found {len(bays)}"

    # Pick the first open bay triad
    a_left, a_center, a_right = bays[0]

    # Step 2: Peri-Annulate Benzene across bay triad -> 1H-Phenalene (C13H10)
    phen_syms, phen_crds = fuse_ring_to_triad(
        naph_syms, naph_crds, a_left, a_center, a_right,
        ring_preset="Benzene (6-membered aromatic)",
        apex_saturation="neutral",
        relax=True,
    )

    counts = Counter(phen_syms)
    assert counts["C"] == 13, f"Expected 13 carbons for Phenalene, got {counts['C']}"
    assert counts["H"] == 10, f"Expected 10 hydrogens for Phenalene, got {counts['H']}"

    _check_clearance(phen_crds, min_dist=0.85)
    _check_valencies(phen_syms, phen_crds)


def test_invariant_naphthalene_to_acenaphthene_peri_annulation():
    """Peri-annulation with cyclopentyl on Naphthalene bay must yield C12H10 Acenaphthene."""
    syms, crds = _make_benzene()
    naph_syms, naph_crds = fuse_ring_to_bond(
        syms, crds, 0, 1,
        ring_preset="Benzene (6-membered aromatic)",
        relax=True,
    )

    bonds = detect_connectivity(naph_syms, naph_crds)
    bays = detect_peri_bays(naph_syms, naph_crds, bonds)
    a_left, a_center, a_right = bays[0]

    # 5-membered ring peri-annulation -> Acenaphthene (C12H10)
    ace_syms, ace_crds = fuse_ring_to_triad(
        naph_syms, naph_crds, a_left, a_center, a_right,
        ring_preset="Cyclopentyl (5-membered saturated)",
        relax=True,
    )

    counts = Counter(ace_syms)
    assert counts["C"] == 12, f"Expected 12 carbons for Acenaphthene, got {counts['C']}"
    assert counts["H"] == 10, f"Expected 10 hydrogens for Acenaphthene, got {counts['H']}"

    _check_clearance(ace_crds, min_dist=0.85)
    _check_valencies(ace_syms, ace_crds)
