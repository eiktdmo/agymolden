"""
Unit tests for Multiwfn Integration and Wavefunction Analysis.
Tests:
1. Mayer & Wiberg bond order calculation (Multiwfn and PySCF fallback)
2. Conceptual DFT Fukui functions and dual descriptor
3. QTAIM critical points topology search
4. CM5 and Hirshfeld partial charges calculation
5. Dialog instantiation and integration
"""

import os
import pytest
import numpy as np
from PyQt6 import QtWidgets

from agymolden.multiwfn_analysis import (
    detect_multiwfn_executable,
    compute_mayer_and_wiberg_orders,
    compute_condensed_fukui,
    compute_qtaim_topology,
    compute_hirshfeld_and_cm5_charges,
    compute_grid_cube,
    compute_nci_analysis,
    _compute_bond_orders_pyscf,
)
from agymolden.dialog_multiwfn import MultiwfnAnalysisDialog


H2O_MOLDEN = os.path.join(os.path.dirname(__file__), "..", "examples", "h2o_orca.molden")


def test_detect_multiwfn_executable():
    bin_path = detect_multiwfn_executable()
    # If installed on this environment, it should be detected
    if bin_path:
        assert os.path.isfile(bin_path)
        assert os.access(bin_path, os.X_OK)


def test_compute_mayer_and_wiberg_orders():
    if not os.path.isfile(H2O_MOLDEN):
        pytest.skip("examples/h2o_orca.molden not found")

    try:
        mayer, wiberg = compute_mayer_and_wiberg_orders(H2O_MOLDEN)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Neither runnable Multiwfn nor pyscf available: {exc}")
    assert len(mayer.bonds) >= 2  # Two O-H bonds
    assert len(wiberg.bonds) >= 2

    # Check bond orders are around ~1.0 for O-H
    oh_bonds = [b for b in mayer.bonds if (b[0], b[1]) in {(1, 2), (2, 1), (1, 3), (3, 1)}]
    assert len(oh_bonds) == 2
    for b in oh_bonds:
        assert 0.8 <= b[4] <= 1.2

    # Check atomic valences
    assert 1 in mayer.valences
    assert 1.8 <= mayer.valences[1] <= 2.2  # Oxygen valence ~ 2.0


def test_pyscf_fallback_bond_orders():
    pytest.importorskip("pyscf")
    if not os.path.isfile(H2O_MOLDEN):
        pytest.skip("examples/h2o_orca.molden not found")

    mayer, wiberg = _compute_bond_orders_pyscf(H2O_MOLDEN)
    assert len(mayer.bonds) >= 2
    assert len(wiberg.bonds) >= 2
    assert "PySCF" in mayer.method


def test_compute_condensed_fukui():
    pytest.importorskip("pyscf")
    if not os.path.isfile(H2O_MOLDEN):
        pytest.skip("examples/h2o_orca.molden not found")

    res = compute_condensed_fukui(H2O_MOLDEN, population_scheme="Lowdin")
    assert len(res.atoms) == 3

    # Oxygen atom (atom 1): high f- (HOMO donor, nucleophilic lone pairs), negative dual descriptor
    o_atom = res.atoms[0]
    assert o_atom["index"] == 1
    assert o_atom["f_minus"] > 0.5
    assert o_atom["dual"] < 0  # Nucleophilic attack site

    # Hydrogen atoms (atoms 2 & 3): positive dual descriptor (electrophilic sites)
    for h_atom in res.atoms[1:]:
        assert h_atom["dual"] > 0


def test_compute_qtaim_topology():
    bin_path = detect_multiwfn_executable()
    if not bin_path or not os.path.isfile(H2O_MOLDEN):
        pytest.skip("Multiwfn not installed or h2o_orca.molden missing")

    try:
        res = compute_qtaim_topology(H2O_MOLDEN)
    except ModuleNotFoundError as exc:
        pytest.skip(f"Neither runnable Multiwfn nor pyscf available: {exc}")
    assert res.num_nuclear == 3  # 3 atoms
    assert res.num_bcp >= 2      # At least 2 bond critical points (O-H)
    assert len(res.critical_points) == res.num_nuclear + res.num_bcp + res.num_rcp + res.num_ccp


def test_compute_hirshfeld_and_cm5_charges():
    bin_path = detect_multiwfn_executable()
    if not bin_path or not os.path.isfile(H2O_MOLDEN):
        pytest.skip("Multiwfn not installed or h2o_orca.molden missing")

    try:
        res = compute_hirshfeld_and_cm5_charges(H2O_MOLDEN)
    except RuntimeError as exc:
        if "exit code 127" in str(exc):
            pytest.skip(f"Multiwfn binary is not runnable on this host: {exc}")
        raise
    assert len(res.atoms) == 3
    # Oxygen atom should be negatively charged
    o_atom = res.atoms[0]
    assert o_atom["cm5"] < -0.4
    assert o_atom["hirshfeld"] < -0.2

    # Hydrogen atoms should be positively charged
    assert res.atoms[1]["cm5"] > 0.2
    assert res.atoms[2]["cm5"] > 0.2

    # Total charge neutrality ~ 0.0
    assert abs(res.sum_cm5) < 0.01


def test_multiwfn_dialog_initialization(qapp):
    dlg = MultiwfnAnalysisDialog(None, molden_path=H2O_MOLDEN)
    assert dlg.tabs.count() == 5
    assert "Mayer & Wiberg" in dlg.tabs.tabText(0)
    assert "Conceptual DFT" in dlg.tabs.tabText(1)
    assert "QTAIM" in dlg.tabs.tabText(2)
    assert "CM5" in dlg.tabs.tabText(3)
    assert "3D Real-Space Grids" in dlg.tabs.tabText(4)


def test_compute_grid_cube_elf_and_nci(tmp_path):
    if not os.path.isfile(H2O_MOLDEN):
        pytest.skip("examples/h2o_orca.molden not found")

    out_elf = str(tmp_path / "water_elf.cube")
    res_elf = compute_grid_cube(H2O_MOLDEN, grid_type="ELF", quality="low", output_cube=out_elf)
    assert os.path.isfile(out_elf)
    assert res_elf.vol_data is not None
    assert res_elf.vol_data.voxels_x == 30
    assert float(np.min(res_elf.vol_data.grid_data)) >= 0.0

    # NCI RDG & density grids
    rdg_res, dens_res = compute_nci_analysis(H2O_MOLDEN, quality="low", output_dir=str(tmp_path))
    assert os.path.isfile(rdg_res.cube_path)
    assert os.path.isfile(dens_res.cube_path)
    assert rdg_res.vol_data.voxels_x == 30


def test_multiwfn_custom_path_and_missing_error(monkeypatch, tmp_path):
    import agymolden.multiwfn_analysis as mwa

    fake = tmp_path / "Multiwfn"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MULTIWFN_PATH", str(fake))
    monkeypatch.setattr(mwa, "get_multiwfn_custom_path", lambda: "")
    assert mwa.detect_multiwfn_executable() == str(fake)

    monkeypatch.delenv("MULTIWFN_PATH")
    monkeypatch.setattr(mwa, "detect_multiwfn_executable", lambda: None)
    with pytest.raises(mwa.MultiwfnNotFoundError):
        mwa.compute_qtaim_topology(H2O_MOLDEN, engine="multiwfn")
    with pytest.raises(mwa.MultiwfnNotFoundError):
        mwa.compute_hirshfeld_and_cm5_charges(H2O_MOLDEN)

    # auto/pyscf fallback works even if Multiwfn is missing
    pytest.importorskip("pyscf")
    qres = mwa.compute_qtaim_topology(H2O_MOLDEN, engine="auto")
    assert qres.method == "PySCF Analytical QTAIM"
    assert qres.num_bcp >= 2


def test_friendly_error_text_for_missing_pyscf():
    from agymolden.multiwfn_analysis import friendly_error_text, PYSCF_MISSING_MESSAGE

    assert friendly_error_text(ImportError("No module named 'pyscf'")) == PYSCF_MISSING_MESSAGE
    assert friendly_error_text(ValueError("boom")) == "boom"


def test_pyscf_qtaim_analytical_water_and_benzene():
    pytest.importorskip("pyscf")
    import agymolden.multiwfn_analysis as mwa

    # 1. Water QTAIM
    res_h2o = mwa.compute_qtaim_topology(H2O_MOLDEN, engine="pyscf")
    assert res_h2o.method == "PySCF Analytical QTAIM"
    assert res_h2o.num_nuclear == 3
    assert res_h2o.num_bcp == 2
    assert len(res_h2o.bond_paths) == 2
    for bcp in [cp for cp in res_h2o.critical_points if cp.cp_type == "(3,-1)"]:
        assert bcp.rho > 0.15
        assert bcp.grad_norm < 1e-3
        assert bcp.bond_character != ""
        assert bcp.bond_path is not None
        assert len(bcp.bond_path) >= 5

    # 2. Benzene QTAIM (checks ring critical point RCP)
    c6h6_molden = os.path.join(os.path.dirname(__file__), "..", "examples", "C6H6_OPT_HF_cc-pvdz_hf_electrons.molden")
    if os.path.isfile(c6h6_molden):
        res_c6h6 = mwa.compute_qtaim_topology(c6h6_molden, engine="pyscf")
        assert res_c6h6.num_nuclear == 12
        assert res_c6h6.num_bcp >= 12   # 6 C-C + 6 C-H
        assert res_c6h6.num_rcp >= 1    # Ring centroid
        rcps = [cp for cp in res_c6h6.critical_points if cp.cp_type == "(3,+1)"]
        assert len(rcps) >= 1
        assert rcps[0].rho > 0.01
        assert rcps[0].grad_norm < 1e-3
