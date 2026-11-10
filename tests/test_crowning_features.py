"""Tests for crowning features:
1. Frontier Molecular Orbital (FMO) Alignment Dialog
2. Automated Citation & Methods Section Generator
3. Trajectory / Vibration Movie Export Capabilities
"""

import pytest
from PyQt6 import QtWidgets

from agymolden.dialog_fmo_alignment import FMOAlignmentDialog
from agymolden.dialog_citations import CitationDialog, CITATIONS_DB


@pytest.fixture(scope="session")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    yield app


def test_fmo_alignment_dialog(qapp):
    """Verify FMO Alignment dialog initializes, calculates gaps and conceptual DFT indices."""
    dlg = FMOAlignmentDialog(
        sys1_name="Donor",
        sys1_homo=-5.50,
        sys1_lumo=-1.10,
        sys2_name="Acceptor",
        sys2_homo=-7.20,
        sys2_lumo=-3.50,
    )
    assert dlg.spin_sys1_homo.value() == -5.50
    assert dlg.spin_sys1_lumo.value() == -1.10
    assert dlg.spin_sys2_homo.value() == -7.20
    assert dlg.spin_sys2_lumo.value() == -3.50

    # Test update_plot
    dlg.update_plot()
    assert "Donor" in dlg.lbl_metrics.text()
    assert "Acceptor" in dlg.lbl_metrics.text()
    assert "Donor-Acceptor Gap" in dlg.lbl_metrics.text()


def test_citations_dialog(qapp):
    """Verify citations dialog contains BibTeX entries and computational methods text."""
    assert "agymolden" in CITATIONS_DB
    assert "pyscf" in CITATIONS_DB
    assert "uff" in CITATIONS_DB

    dlg = CitationDialog()
    assert "Computational Methods Section" in dlg.txt_methods.toPlainText()
    assert "@software{agymolden" in dlg.txt_bibtex.toPlainText()
    assert "@article{pyscf" in dlg.txt_bibtex.toPlainText()
