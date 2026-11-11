"""
Unit and Integration Tests for High-Impact Trio Features:
1. Quasi-Harmonic Thermochemistry (quasiharmonic.py, dialog_quasiharmonic.py)
2. Automated Supporting Information (SI) Generator (si_generator.py, dialog_si_generator.py)
3. QTAIM Topology & Bond Critical Points (qtaim_topology.py, dialog_qtaim.py)
"""

import os
import pytest
import numpy as np
from PyQt6 import QtWidgets

from agymolden.quasiharmonic import (
    calculate_quasiharmonic_thermo,
    vibrational_entropy_harmonic,
    vibrational_entropy_free_rotor,
)
from agymolden.si_generator import (
    SISpecies,
    generate_computational_methods_paragraph,
    generate_latex_si_document,
    generate_markdown_si_document,
    format_xyz_block,
)
from agymolden.qtaim_topology import (
    find_bond_critical_points,
    BondCriticalPoint,
    QTAIMTopologyResult,
)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_quasiharmonic_damping_and_entropy():
    """Verify that low-frequency modes (< 100 cm^-1) are dampened by Grimme mRRHO and Truhlar."""
    # Low frequency mode at 30 cm^-1 and high frequency mode at 1500 cm^-1
    freqs = [30.0, 1500.0]
    res = calculate_quasiharmonic_thermo(freqs, temperature_k=298.15, cutoff_cm1=100.0)

    assert res.s_vib_standard_cal_mol_k > 0.0
    assert res.s_vib_grimme_cal_mol_k > 0.0
    assert res.s_vib_truhlar_cal_mol_k > 0.0

    # In standard RRHO, a 30 cm^-1 mode gives an excessively large entropy.
    # Truhlar shifts 30 -> 100 cm^-1, so S_truhlar < S_standard
    assert res.s_vib_truhlar_cal_mol_k < res.s_vib_standard_cal_mol_k

    # Test scaling factor
    res_scaled = calculate_quasiharmonic_thermo(freqs, temperature_k=298.15, scaling_factor=0.96)
    assert res_scaled.frequencies_scaled[0] == pytest.approx(30.0 * 0.96, 0.001)
    assert res_scaled.frequencies_scaled[1] == pytest.approx(1500.0 * 0.96, 0.001)


def test_si_generator_synthesis():
    """Verify LaTeX and Markdown SI generation with formatted coordinates and energy tables."""
    syms = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.75, 0.58], [0.0, -0.75, 0.58]])
    sp = SISpecies(
        name="H2O_Intermediate",
        symbols=syms,
        coords=coords,
        electronic_energy_ha=-76.4321,
        zpe_ha=0.0215,
        enthalpy_ha=-76.4068,
        gibbs_ha=-76.4285,
        lowest_frequencies_cm1=[1620.0, 3750.0, 3850.0],
    )

    methods = generate_computational_methods_paragraph(
        program="ORCA 5.0",
        functional="ωB97X-D",
        basis_set="def2-TZVP",
        solvation="CPCM",
        solvent="dichloromethane",
    )
    assert "ORCA 5.0" in methods
    assert "ωB97X-D" in methods
    assert "def2-TZVP" in methods

    xyz = format_xyz_block(sp)
    assert "3\nH2O_Intermediate" in xyz
    assert "O" in xyz and "H" in xyz

    latex_doc = generate_latex_si_document([sp], methods)
    assert r"\documentclass" in latex_doc
    assert r"\begin{document}" in latex_doc
    assert "H2O_Intermediate" in latex_doc
    assert "-76.432100" in latex_doc

    md_doc = generate_markdown_si_document([sp], methods)
    assert "# Supporting Information" in md_doc
    assert "| **H2O_Intermediate** |" in md_doc


def test_qtaim_bcp_search_water():
    """Verify (3, -1) Bond Critical Point search on water Molden file."""
    molden_path = "examples/h2o_orca.molden"
    if not os.path.exists(molden_path):
        pytest.skip(f"{molden_path} not found")

    res = find_bond_critical_points(molden_path, max_bond_distance_angstrom=2.5)
    assert isinstance(res, QTAIMTopologyResult)
    # H2O has 2 O-H covalent bonds -> must find at least 2 BCPs
    assert len(res.bond_critical_points) >= 2

    for bcp in res.bond_critical_points:
        assert isinstance(bcp, BondCriticalPoint)
        # Density at covalent BCP is positive (~0.2 - 0.4 e/bohr^3 for O-H)
        assert bcp.rho_bcp > 0.05
        # (3, -1) critical point signature: lambda1 < 0, lambda2 < 0, lambda3 > 0
        l1, l2, l3 = bcp.eigenvalues
        assert l1 < 0
        assert l2 < 0
        assert l3 > 0


def test_high_impact_dialogs_smoke(app):
    """Verify GUI lifecycle for QuasiHarmonic, SI Generator, and QTAIM dialogs."""
    from agymolden.dialog_quasiharmonic import QuasiHarmonicDialog
    from agymolden.dialog_si_generator import SIGeneratorDialog
    from agymolden.dialog_qtaim import QTAIMDialog

    # 1. QuasiHarmonic Dialog
    q_dlg = QuasiHarmonicDialog(frequencies_cm1=[45.0, 120.0, 850.0, 1600.0])
    assert q_dlg.tbl_results.rowCount() == 4
    q_dlg.close()

    # 2. SI Generator Dialog
    si_dlg = SIGeneratorDialog()
    assert len(si_dlg.txt_methods.toPlainText()) > 20
    si_dlg.close()

    # 3. QTAIM Dialog
    qtaim_dlg = QTAIMDialog(molden_path="examples/h2o_orca.molden")
    assert qtaim_dlg.tbl_bcp.rowCount() >= 2
    qtaim_dlg.close()
