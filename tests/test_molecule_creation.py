import pytest
import numpy as np

from agymolden.main_window import MainWindow
from agymolden.zmatrix import ZAtom


def test_single_atom_rendering_no_crash(qapp):
    """Verify that a 1-atom molecule (0 bonds) renders in paintGL without UnboundLocalError."""
    win = MainWindow()
    # Add a single Carbon atom at origin
    win.set_molecule_from_zatoms([ZAtom(symbol="C")], exact_coords=np.array([[0.0, 0.0, 0.0]]))
    assert len(win.zatoms) == 1
    assert len(win.coords) == 1

    # Force a paintGL call
    win.viewer.paintGL()
    win.close()


def test_two_atoms_rendering_no_crash(qapp):
    """Verify that a 2-atom molecule (1 bond) renders properly."""
    win = MainWindow()
    win.set_molecule_from_zatoms(
        [ZAtom(symbol="C"), ZAtom(symbol="O", ref1=0, dist=1.43)],
        exact_coords=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.43]]),
    )
    assert len(win.zatoms) == 2
    assert len(win.viewer.bonds) == 1

    win.viewer.paintGL()
    win.close()


def test_build_molecule_from_smiles_or_name(qapp):
    """Verify that building a molecule from SMILES updates viewer and renders cleanly."""
    win = MainWindow()
    from agymolden.chem_identifiers import parse_smiles_or_inchi_to_3d

    # Resolve Benzene
    symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d("c1ccccc1")
    assert len(symbols) == 12

    from agymolden.zmatrix import cartesian_to_zmatrix
    zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds)

    win.set_molecule_from_zatoms(zatoms, exact_coords=coords)
    assert len(win.zatoms) == 12
    win.viewer.paintGL()
    win.close()


def test_empty_canvas_double_click_menu_actions(qapp):
    """Verify that canvas double-click menu actions are valid."""
    win = MainWindow()
    assert hasattr(win, "_on_canvas_double_clicked")
    assert hasattr(win, "_paste_element_to_free_space")
    assert hasattr(win, "_paste_fragment_to_free_space")
    win.close()
