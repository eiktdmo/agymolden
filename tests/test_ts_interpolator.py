"""
Unit tests for TS Interpolator (IDPP, Linear Transit, Kabsch alignment, and Input Generators),
Journal Presets, and Quick ORCA Solvent/Dispersion options.
"""

import pytest
import numpy as np
from PyQt6 import QtWidgets

from agymolden.ts_interpolator import (
    kabsch_align,
    interpolate_linear,
    interpolate_idpp,
    compute_pair_distances,
    format_path_xyz,
    generate_orca_neb_input,
    generate_gaussian_qst2_input,
    generate_gaussian_qst3_input,
)
from agymolden.dialog_reaction_path import ReactionPathDialog
from agymolden.dialog_quick_orca import QuickOrcaDialog
from agymolden.zmatrix import ZAtom


def test_kabsch_alignment():
    # Identity alignment
    ref = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    aligned = kabsch_align(ref, ref)
    assert np.allclose(ref, aligned, atol=1e-6)

    # Shifted and rotated
    mobile = ref + np.array([2.5, -1.0, 3.0])
    aligned2 = kabsch_align(ref, mobile)
    assert np.allclose(ref, aligned2, atol=1e-5)


def test_idpp_interpolation():
    # HCN -> CNH isomerization model
    r_coords = np.array([[0.0, 0.0, -1.06], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    p_coords = np.array([[0.0, 0.0, 2.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    symbols = ["H", "C", "N"]

    # Test linear interpolation
    lin_path = interpolate_linear(r_coords, p_coords, n_images=5)
    assert lin_path.shape == (5, 3, 3)

    # Test IDPP interpolation
    idpp_path = interpolate_idpp(r_coords, p_coords, n_images=7)
    assert idpp_path.shape == (7, 3, 3)

    # Verify end points match reactant and aligned product
    assert np.allclose(idpp_path[0], r_coords, atol=1e-5)
    p_aligned = kabsch_align(r_coords, p_coords)
    assert np.allclose(idpp_path[-1], p_aligned, atol=1e-5)

    # Verify pair distances do not collapse to zero (no atom clashing)
    for frame in idpp_path:
        d = compute_pair_distances(frame)
        # Off-diagonal distances should be >= 0.5 Angstrom
        np.fill_diagonal(d, 99.0)
        assert np.min(d) > 0.5


def test_reaction_path_formatters():
    symbols = ["H", "C", "N"]
    r_coords = np.array([[0.0, 0.0, -1.06], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    p_coords = np.array([[0.0, 0.0, 2.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    path = interpolate_idpp(r_coords, p_coords, n_images=5)

    # Multi-frame XYZ
    xyz_str = format_path_xyz(symbols, path)
    assert xyz_str.count("3\nReaction Frame") == 5

    # ORCA NEB-TS
    neb_inp = generate_orca_neb_input(symbols, path, method="wB97X-D3", basis="def2-TZVP")
    assert "! wB97X-D3 def2-TZVP" in neb_inp
    assert "NEB-TS" in neb_inp
    assert "NImages 5" in neb_inp

    # Gaussian QST2 & QST3
    qst2_inp = generate_gaussian_qst2_input(symbols, r_coords, p_coords)
    assert "opt=(qst2,calcfc)" in qst2_inp
    assert "Reactant Structure" in qst2_inp
    assert "Product Structure" in qst2_inp

    qst3_inp = generate_gaussian_qst3_input(symbols, r_coords, p_coords, path[2])
    assert "opt=(qst3,calcfc)" in qst3_inp
    assert "Transition State Guess" in qst3_inp


def test_reaction_path_dialog_headless(qapp):
    r_syms = ["H", "C", "N"]
    r_coords = np.array([[0.0, 0.0, -1.06], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    dlg = ReactionPathDialog(None, initial_symbols=r_syms, initial_coords=r_coords)

    # Set product manually
    p_coords = np.array([[0.0, 0.0, 2.15], [0.0, 0.0, 0.0], [0.0, 0.0, 1.15]])
    dlg.p_symbols = list(r_syms)
    dlg.p_coords = p_coords
    dlg._update_status()

    assert dlg.btn_interpolate.isEnabled()

    # Compute path
    dlg.spin_images.setValue(5)
    dlg._on_compute_path()

    assert dlg.path_images is not None
    assert len(dlg.path_images) == 5
    assert dlg.btn_adopt_ts.isEnabled()
    assert dlg.btn_preview.isEnabled()


def test_quick_orca_solvent_and_dispersion(qapp):
    zatoms = [
        ZAtom(symbol="O"),
        ZAtom(symbol="H", ref1=0, dist=0.96),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5),
    ]
    dlg = QuickOrcaDialog(None, zatoms=zatoms)

    dlg.combo_solvent.setCurrentText("Water (CPCM)")
    dlg.combo_dispersion.setCurrentText("D3BJ")

    # Trigger run logic in memory
    route_line = dlg.txt_route.text().strip()
    if not route_line.startswith("!"):
        route_line = "! " + route_line
    if dlg.combo_dispersion.currentText() in ("D3BJ", "D4"):
        route_line += f" {dlg.combo_dispersion.currentText()}"
    solv = dlg.combo_solvent.currentText()
    if solv != "None (Gas Phase)":
        route_line += f" CPCM({solv.split()[0]})"

    assert "CPCM(Water)" in route_line
    assert "D3BJ" in route_line


def test_journal_presets_in_viewer(qapp):
    from agymolden.viewer import MoleculeViewer
    viewer = MoleculeViewer()

    # Test JACS
    lbl = viewer.apply_journal_preset("jacs")
    assert lbl == "JACS Pastel"
    assert viewer.bg_color_mode == "white"
    assert viewer.show_publication_outlines is True

    # Test Nature
    lbl = viewer.apply_journal_preset("nature")
    assert lbl == "Nature Glossy"
    assert viewer.bg_color_mode == "white"
    assert viewer.show_publication_outlines is False

    # Test Dark
    lbl = viewer.apply_journal_preset("dark")
    assert lbl == "Dark Presentation"
    assert viewer.bg_color_mode == "dark"
    assert viewer.show_publication_outlines is False
