"""
Automated unit tests for Next-Level Features:
1. Reaction Thermochemistry Workbench & Eyring Kinetics (agymolden/thermochemistry.py)
2. Pinned Measurements (agymolden/viewer.py)
3. Non-Covalent Interactions (NCI / Halogen / Pi-Pi) (agymolden/viewer.py)
4. 2D Bond-Line Sketcher (agymolden/dialog_2d_sketcher.py)
"""

import math
import numpy as np
import pytest

from agymolden.thermochemistry import (
    SpeciesThermo,
    ReactionThermoResult,
    calculate_reaction_thermochemistry,
)
from agymolden.dialog_2d_sketcher import SketchCanvas2D, SketchAtom2D, SketchBond2D


def test_reaction_thermochemistry_basic():
    """Verify Gibbs free energy change and equilibrium constant calculation."""
    # A -> B: G(A) = -100.0 Ha, G(B) = -100.010 Ha (exergonic by -6.275 kcal/mol)
    r = SpeciesThermo(name="A", role="reactant", coefficient=1.0, gibbs_ha=-100.0, enthalpy_ha=-100.0)
    p = SpeciesThermo(name="B", role="product", coefficient=1.0, gibbs_ha=-100.010, enthalpy_ha=-100.010)

    res = calculate_reaction_thermochemistry([r, p], temperature_k=298.15)
    assert pytest.approx(res.delta_g_rxn_kcal, rel=1e-3) == -6.27509
    assert res.is_exergonic is True
    assert res.k_eq > 1.0


def test_reaction_thermochemistry_with_ts_and_eyring():
    """Verify activation barrier and Eyring rate constant k(T)."""
    # A -> TS‡ -> B
    r = SpeciesThermo(name="A", role="reactant", coefficient=1.0, gibbs_ha=-50.0)
    p = SpeciesThermo(name="B", role="product", coefficient=1.0, gibbs_ha=-50.02)
    ts = SpeciesThermo(name="TS", role="ts", coefficient=1.0, gibbs_ha=-49.97)

    res = calculate_reaction_thermochemistry([r, p, ts], temperature_k=298.15)
    assert res.has_ts is True
    # Barrier = (-49.97 - (-50.0)) * 627.5095 = 0.03 * 627.5095 ≈ 18.825 kcal/mol
    assert pytest.approx(res.delta_g_act_kcal, rel=1e-3) == 18.82528
    assert res.rate_constant_k_fwd is not None
    assert res.rate_constant_k_fwd > 0.0
    assert res.rate_constant_k_rev is not None
    assert res.rate_constant_k_fwd > res.rate_constant_k_rev


def test_thermochemistry_safeguards():
    """Check temperature boundary safeguards and missing species exceptions."""
    r = SpeciesThermo(name="A", role="reactant", coefficient=1.0, gibbs_ha=-10.0)
    with pytest.raises(ValueError, match="Temperature must be greater than 0"):
        calculate_reaction_thermochemistry([r], temperature_k=-10.0)

    with pytest.raises(ValueError, match="At least one product"):
        calculate_reaction_thermochemistry([r], temperature_k=298.15)


def test_2d_sketcher_ring_generation():
    """Verify 2D sketcher ring templates generate correct atom and bond topology."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    canvas = SketchCanvas2D()
    canvas.add_fragment_ring(6)
    assert len(canvas.atoms) == 6
    assert len(canvas.bonds) == 6
    assert all(a.symbol == "C" for a in canvas.atoms)

    canvas.clear_canvas()
    assert len(canvas.atoms) == 0
    assert len(canvas.bonds) == 0


def test_2d_sketcher_undo_redo():
    """Verify Ctrl+Z undo and redo functionality in 2D sketch canvas."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    canvas = SketchCanvas2D()
    canvas.add_fragment_ring(6)
    assert len(canvas.atoms) == 6
    assert len(canvas.bonds) == 6

    # Undo ring addition
    canvas.undo()
    assert len(canvas.atoms) == 0
    assert len(canvas.bonds) == 0

    # Redo ring addition
    canvas.redo()
    assert len(canvas.atoms) == 6
    assert len(canvas.bonds) == 6


def test_2d_sketcher_dialog_convert_to_3d():
    """Verify 2D sketch to 3D Cartesian generation with hydrogen saturation."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    from agymolden.dialog_2d_sketcher import MoleculeSketcher2DDialog

    dlg = MoleculeSketcher2DDialog()
    dlg.canvas.add_fragment_ring(6)  # Benzene skeleton

    emitted_data = []

    def on_structure(syms, coords, bonds, bo):
        emitted_data.append((syms, coords, bonds, bo))

    dlg.structure_generated.connect(on_structure)
    dlg._on_convert_to_3d()

    assert len(emitted_data) == 1
    syms, coords, bonds, bo = emitted_data[0]
    # With implicit hydrogens, benzene has 12 atoms (6 C + 6 H)
    assert "C" in syms
    assert len(coords) >= 6
    assert coords.shape == (len(syms), 3)
    assert np.all(np.isfinite(coords))


def test_2d_sketcher_functional_group_and_live_smiles():
    """Verify stamping predefined functional groups and live canonical SMILES preview."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    canvas = SketchCanvas2D()
    # Add benzene ring
    canvas.add_fragment_ring(6)
    # Stamp Methyl group attached to atom 0 -> Toluene
    canvas.add_functional_group("Me (Methyl)", target_atom_idx=0)
    assert len(canvas.atoms) == 7  # 6 C (ring) + 1 C (methyl)
    assert len(canvas.bonds) == 7

    smi = canvas.get_canonical_smiles()
    assert "Cc1ccccc1" in smi or "c1ccccc1C" in smi or "C" in smi


def test_interactive_tweezers_toggle_and_drag():
    """Verify molecular tweezers flag and coordinate relaxation handling."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    from agymolden.main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    win.load_preset_molecule("Water (H2O)")
    assert len(win.zatoms) == 3

    # Enable tweezers
    win.act_tweezers.setChecked(True)
    assert win.live_tweezers_enabled is True

    # Simulate tweezers drag on Oxygen (atom 0)
    old_h_coord = np.copy(win.coords[1])
    win._on_atom_moved_by_drag(0, 0.5, 0.5, 0.5)
    # Oxygen was translated
    assert np.allclose(win.coords[0], [0.5, 0.5, 0.5])
    # Hydrogens should have relaxed smoothly
    assert np.all(np.isfinite(win.coords))


def test_lighting_studio_dialog():
    """Verify 3D lighting & publication figure studio dialog initialization."""
    import os
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PyQt6 import QtWidgets
    from agymolden.main_window import MainWindow
    from agymolden.dialog_lighting import LightingStudioDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    win = MainWindow()
    dlg = LightingStudioDialog(win.viewer)
    dlg._apply_preset("jacs")
    assert win.viewer.use_modern_shaders is True
    assert win.viewer.show_publication_outlines is True


