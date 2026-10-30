"""Aromaticity / resonance detection (Hückel 4n+2, nitro, carboxylate)."""
import math
import os

import numpy as np
import pytest

from agymolden.aromatic import find_aromatic_rings, find_delocalized_bonds, find_rings

EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")


def _ring(symbols_ring, r):
    n = len(symbols_ring)
    syms, xyz, bonds = [], [], []
    for k, s in enumerate(symbols_ring):
        a = 2 * math.pi * k / n
        syms.append(s)
        xyz.append([r * math.cos(a), r * math.sin(a), 0.0])
    for k in range(n):
        bonds.append((k, (k + 1) % n))
    return syms, xyz, bonds


def _add_h(syms, xyz, bonds, ring_atoms, r_h):
    for k in ring_atoms:
        a = math.atan2(xyz[k][1], xyz[k][0])
        syms.append("H")
        xyz.append([r_h * math.cos(a), r_h * math.sin(a), 0.0])
        bonds.append((k, len(syms) - 1))


def test_benzene_all_bonds_aromatic():
    s, x, b = _ring(["C"] * 6, 1.40)
    _add_h(s, x, b, range(6), 2.48)
    d = find_delocalized_bonds(s, np.array(x), b)
    assert {tuple(sorted(k)) for k in d} == {tuple(sorted(e)) for e in b[:6]}
    # inward vectors point to the ring centre (x*v < 0 for ring atoms at radius r)
    for (i, j), vec in d.items():
        mid = 0.5 * (np.array(x[i]) + np.array(x[j]))
        assert np.dot(vec, mid) < 0


def test_alternating_kekule_orders_do_not_matter():
    s, x, b = _ring(["C"] * 6, 1.40)
    _add_h(s, x, b, range(6), 2.48)
    orders = {(0, 1): 2, (2, 3): 2, (4, 5): 2}
    assert len(find_aromatic_rings(s, np.array(x), b, orders)) == 1


def test_pyridine_furan_pyrrole_are_aromatic():
    s, x, b = _ring(["N"] + ["C"] * 5, 1.40)
    _add_h(s, x, b, range(1, 6), 2.48)  # N has no H
    assert len(find_aromatic_rings(s, np.array(x), b)) == 1

    s, x, b = _ring(["O"] + ["C"] * 4, 1.2)
    _add_h(s, x, b, range(1, 5), 2.1)
    assert len(find_aromatic_rings(s, np.array(x), b)) == 1  # furan, 6 pi e-

    s, x, b = _ring(["N"] + ["C"] * 4, 1.2)
    _add_h(s, x, b, range(0, 5), 2.1)  # pyrrole N-H
    assert len(find_aromatic_rings(s, np.array(x), b)) == 1


def test_cyclopentadiene_and_cyclohexane_are_not_aromatic():
    s, x, b = _ring(["C"] * 5, 1.2)
    _add_h(s, x, b, range(5), 2.1)
    _add_h(s, x, b, [0], 2.4)  # CH2: sp3 carbon
    assert find_aromatic_rings(s, np.array(x), b) == []

    # puckered, all sp3 -> never aromatic
    s, x, b = _ring(["C"] * 6, 1.45)
    for k in range(6):
        x[k][2] = 0.25 if k % 2 else -0.25
    _add_h(s, x, b, range(6), 2.5)
    _add_h(s, x, b, range(6), 2.6)
    assert find_aromatic_rings(s, np.array(x), b) == []


def test_non_planar_six_ring_is_rejected():
    s, x, b = _ring(["C"] * 6, 1.40)
    for k in range(6):
        x[k][2] = 0.5 if k % 2 else -0.5
    _add_h(s, x, b, range(6), 2.48)
    assert find_aromatic_rings(s, np.array(x), b) == []


def test_cyclooctatetraene_like_ring_not_aromatic():
    s, x, b = _ring(["C"] * 8, 1.4)  # 8 > MAX_RING, would be 4n anyway
    _add_h(s, x, b, range(8), 2.4)
    assert find_aromatic_rings(s, np.array(x), b) == []


def test_exocyclic_carbonyl_breaks_aromaticity():
    s, x, b = _ring(["C"] * 6, 1.40)
    _add_h(s, x, b, range(1, 6), 2.48)
    s.append("O"); x.append([2.6, 0.0, 0.0]); b.append((0, 6 + 5))
    assert find_aromatic_rings(s, np.array(x), b, {(0, 11): 2}) == []


def test_naphthalene_both_rings():
    h = 1.40
    dx = h * math.sqrt(3)  # two fused hexagons sharing an edge
    left = [(h * math.cos(math.radians(60 * k + 30)), h * math.sin(math.radians(60 * k + 30))) for k in range(6)]
    right = [(p[0] + dx, p[1]) for p in left]
    xs, bonds, syms, index = [], [], [], {}

    def idx(p):
        key = (round(p[0], 3), round(p[1], 3))
        if key not in index:
            index[key] = len(xs)
            xs.append([p[0], p[1], 0.0])
            syms.append("C")
        return index[key]

    for ringpts in (left, right):
        ids = [idx(p) for p in ringpts]
        for k in range(6):
            e = tuple(sorted((ids[k], ids[(k + 1) % 6])))
            if e not in bonds:
                bonds.append(e)
    assert len(xs) == 10
    # hydrogens on the 8 non-fusion carbons (pointing away from the molecular centre)
    centre = np.array(xs).mean(axis=0)
    deg = {i: sum(i in e for e in bonds) for i in range(10)}
    for i in range(10):
        if deg[i] == 2:
            v = np.array(xs[i]) - centre
            v = v / np.linalg.norm(v)
            syms.append("H")
            xs.append(list(np.array(xs[i]) + 1.08 * v))
            bonds.append((i, len(syms) - 1))
    rings = find_aromatic_rings(syms, np.array(xs), bonds)
    assert len([r for r in rings if len(r) == 6]) == 2


def test_nitro_and_carboxylate_resonance():
    # nitromethane-like: C-N(=O)-O
    syms = ["C", "N", "O", "O"]
    x = np.array([[0, 0, 0], [1.5, 0, 0], [2.1, 1.1, 0], [2.1, -1.1, 0]], float)
    d = find_delocalized_bonds(syms, x, [(0, 1), (1, 2), (1, 3)], {(1, 2): 2})
    assert set(d) == {(1, 2), (1, 3)}
    # acetate: C-C(=O)-O(-)
    syms = ["C", "C", "O", "O"]
    d = find_delocalized_bonds(syms, x, [(0, 1), (1, 2), (1, 3)], {(1, 2): 2})
    assert set(d) == {(1, 2), (1, 3)}
    # acetic acid: the OH oxygen is not terminal -> no resonance
    syms = ["C", "C", "O", "O", "H"]
    x2 = np.vstack([x, [2.9, -1.1, 0]])
    d = find_delocalized_bonds(syms, x2, [(0, 1), (1, 2), (1, 3), (3, 4)], {(1, 2): 2})
    assert d == {}


def test_find_rings_simple_cycles():
    rings = find_rings(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    assert sorted(len(r) for r in rings) == [3, 3, 4]


def test_viewer_reports_aromatic_bond_name():
    pytest.importorskip("PyQt6")
    pytest.importorskip("OpenGL")
    from PyQt6 import QtWidgets
    from agymolden.zmatrix import parse_zmatrix_text, zmatrix_to_cartesian
    from agymolden.forcefield import detect_connectivity
    from agymolden.viewer import MoleculeViewer
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    with open(os.path.join(EXAMPLES, "toluene.zmat")) as f:
        zat, _ = parse_zmatrix_text(f.read())
    sy = [a.symbol for a in zat]
    co = zmatrix_to_cartesian(zat)
    v = MoleculeViewer()
    v.set_molecule(sy, co)
    assert "Aromatic" in v._bond_order_name(0, 1)
    assert v._bond_order_name(0, 11) in ("Single", "Double")  # ring C - CH3 bond stays single


def test_kekule_benzene_optimizes_to_symmetric_d6h():
    from agymolden.zmatrix import parse_zmatrix_text, zmatrix_to_cartesian
    from agymolden.forcefield import detect_connectivity, detect_bond_orders, optimize_geometry_force_field
    from agymolden.symmetry import find_symmetry_operations

    kekule_zmat = """C   
C       1    1.3956
C       2    1.3384    1   120.000
C       3    1.3956    2   120.000    1    -0.000
C       4    1.3384    3   120.000    2     0.000
C       1    1.3384    2   120.000    3     0.000
H       1    1.0845    2   119.993    3  -180.000
H       2    1.0845    1   119.993    6  -180.000
H       3    1.0845    2   120.007    1   180.000
H       4    1.0845    3   119.993    2  -180.000
H       5    1.0845    4   120.007    3  -180.000
H       6    1.0845    5   119.993    4   180.000
"""
    zat, _ = parse_zmatrix_text(kekule_zmat)
    syms = [a.symbol for a in zat]
    coords = zmatrix_to_cartesian(zat)
    bonds = detect_connectivity(syms, coords)
    bo = detect_bond_orders(syms, coords, bonds)
    # Even though detect_bond_orders assigns alternating 1,2,1,2,1,2 Kekulé orders,
    # aromatic delocalization ensures all 6 C-C bonds optimize to identical lengths.
    opt, _ = optimize_geometry_force_field(syms, coords, user_bonds=bonds, user_bond_orders=bo)
    ring = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    dists = [float(np.linalg.norm(opt[u] - opt[v])) for u, v in ring]
    assert max(dists) - min(dists) < 1e-4
    assert find_symmetry_operations(syms, opt, tolerance=0.05).symbol == "D6h"
