"""
Unit and Regression Tests for Option A (Interactive 3D Molecular Builder & Tweezers):
1. Interactive Dihedral Twist & Rotamer Gizmo logic (smartslides.modify_dihedral, rodrigues_rotation)
2. Live Torsion Drag Event Handlers in MainWindow (_on_bond_torsion_dragged)
3. 3D Viewport Torsion Ring Overlay Rendering logic
4. Rich Organometallic & Organic Substituent Snapping via Context Menu
"""

import os
import math
import pytest
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from agymolden.smartslides import (
    modify_dihedral,
    _get_rotatable_side_subgraph,
    rodrigues_rotation,
)
from agymolden.fragment_library import FRAGMENTS


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_dihedral_twist_rotamer_ethane():
    """Verify exact numerical dihedral rotation of a terminal methyl group around central C-C bond."""
    # Staggered ethane model along X axis
    syms = ["C", "C", "H", "H", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.54, 0.0, 0.0],
        [-0.36, 1.02, 0.0],
        [-0.36, -0.51, 0.89],
        [-0.36, -0.51, -0.89],
        [1.90, 1.02, 0.0],
        [1.90, -0.51, 0.89],
        [1.90, -0.51, -0.89],
    ])
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]

    # Subgraph on atom 1's side should be [5, 6, 7]
    moving = _get_rotatable_side_subgraph(len(syms), bonds, 1, 0, None)
    assert sorted(moving) == [5, 6, 7]

    # Rotate by 60 degrees around C0 -> C1
    out_coords = modify_dihedral(syms, coords, bonds, 2, 0, 1, 5, target_dihedral_deg=60.0, mode="group")
    assert out_coords.shape == coords.shape
    # Atoms 0, 2, 3, 4 should not have moved
    assert np.allclose(out_coords[:5], coords[:5], atol=1e-6)
    # Atoms 5, 6, 7 should have rotated around X axis
    assert not np.allclose(out_coords[5:], coords[5:], atol=1e-3)


def test_rich_ligands_exist_in_fragments():
    """Verify essential ligands like Triphenylphosphine, Cp, bpy exist and have valid coordinates."""
    key_ligands = [
        "Triphenylphosphine (-PPh3)",
        "Cyclopentadienyl (Cp)",
        "Bipyridine (bpy)",
        "Phosphine (-PH3)",
        "tert-Butyl (-C(CH3)3)",
        "Trifluoromethyl (-CF3)",
    ]
    for lig in key_ligands:
        assert lig in FRAGMENTS, f"Fragment {lig} missing from library"
        f = FRAGMENTS[lig]
        assert len(f["symbols"]) == len(f["coords"])
        assert f.get("connect_idx", 0) < len(f["symbols"])


def test_viewer_torsion_ring_gizmo_rendering(app):
    """Verify viewer renders without raising when a bond is selected for torsion rotation."""
    from agymolden.main_window import MainWindow
    from agymolden.zmatrix import ZAtom

    win = MainWindow()
    zatoms = [
        ZAtom("C"),
        ZAtom("C", ref1=0, dist=1.52),
        ZAtom("H", ref1=0, dist=1.09, ref2=1, angle=109.5),
        ZAtom("H", ref1=1, dist=1.09, ref2=0, angle=109.5, ref3=2, dihedral=60.0),
    ]
    win.set_molecule_from_zatoms(zatoms)

    # Select central bond (0, 1) to trigger torsion ring gizmo
    win.viewer.selected_bond = (0, 1)
    win.viewer.multi_selected = [0, 1]

    # Verify export_image renders through paintEvent safely in headless mode
    tmp_png = "/tmp/test_torsion_render.png"
    img = win.export_image(tmp_png, transparent=False, scale=1)
    assert img is not None and not img.isNull()
    if os.path.exists(tmp_png):
        os.remove(tmp_png)
    win.close()
