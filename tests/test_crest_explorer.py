"""
Unit tests for CREST & Conformer Ensemble Explorer.
Tests:
- Comment line energy extraction (CREST, Hartree, kcal/mol, eV).
- Companion crest.energies parsing.
- Statistically accurate Boltzmann populations across temperatures.
- Kabsch all-atom and heavy-atom RMSDs.
- Duplicate conformer clustering.
- XYZ and CSV exports.
- QM batch generator (ORCA and Gaussian).
- Headless Qt ConformerEnsembleDialog operation.
"""

import math
import os
import tempfile
import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.crest_explorer import (
    ConformerRecord,
    extract_energy_from_comment,
    parse_crest_energies,
    recalculate_boltzmann,
    recalculate_rmsds,
    cluster_conformers,
    load_conformer_ensemble,
    export_ensemble_xyz,
    export_ensemble_csv,
    generate_qm_batch_scripts,
    HARTREE_TO_KCAL,
    KB_KCAL_MOL_K,
)
from agymolden.dialog_conformer_ensemble import ConformerEnsembleDialog, BoltzmannBarWidget


def test_energy_comment_extraction():
    # 1. Bare float standard in CREST
    res = extract_energy_from_comment("   -152.123456789012")
    assert res is not None
    assert pytest.approx(res[0], rel=1e-8) == -152.123456789012
    assert res[1] == "CREST"

    # 2. Scientific notation
    res = extract_energy_from_comment(" -1.521234e+02  0.0  1.0")
    assert res is not None
    assert pytest.approx(res[0], rel=1e-5) == -152.1234

    # 3. Explicit keywords and units
    res = extract_energy_from_comment("Energy: -40.500000 Hartree")
    assert res is not None
    assert pytest.approx(res[0], rel=1e-6) == -40.5
    assert res[1] == "Hartree"

    res = extract_energy_from_comment("SCF Done: E(RHF) = -75.123456 A.U.")
    assert res is not None
    assert pytest.approx(res[0], rel=1e-6) == -75.123456

    res = extract_energy_from_comment("deltaE = 1.25 kcal/mol")
    assert res is not None
    assert pytest.approx(res[0] * HARTREE_TO_KCAL, rel=1e-4) == 1.25


def test_companion_crest_energies_parser(tmp_path):
    energies_file = tmp_path / "crest.energies"
    content = (
        "# CREST energy output\n"
        "1    -152.5000000    0.00000    0.65000    1\n"
        "2    -152.4980000    1.25502    0.22000    1\n"
        "3    -152.4960000    2.51004    0.13000    1\n"
    )
    energies_file.write_text(content)

    parsed = parse_crest_energies(str(energies_file))
    assert len(parsed) == 3
    assert pytest.approx(parsed[1]["total_e"]) == -152.5
    assert pytest.approx(parsed[2]["rel_e_kcal"]) == 1.25502


def test_boltzmann_population_thermodynamics():
    syms = ["C", "H", "H", "H", "H"]
    c1 = np.zeros((5, 3))
    c2 = np.ones((5, 3))

    # Two conformers with known delta E of exactly 0.59248 kcal/mol (which is RT at 298.15 K)
    rt = KB_KCAL_MOL_K * 298.15
    de_hartree = rt / HARTREE_TO_KCAL

    confs = [
        ConformerRecord(index=1, symbols=syms, coords=c1, energy=-40.0),
        ConformerRecord(index=2, symbols=syms, coords=c2, energy=-40.0 + de_hartree),
    ]

    recalculate_boltzmann(confs, temperature_k=298.15)
    assert confs[0].delta_e_kcal == 0.0
    assert pytest.approx(confs[1].delta_e_kcal, rel=1e-4) == rt

    # At delta_E = RT, ratio of populations p1 / p2 = exp(1) ≈ 2.71828
    ratio = confs[0].boltzmann_pop / confs[1].boltzmann_pop
    assert pytest.approx(ratio, rel=1e-3) == math.e

    # Total population must sum to 100%
    assert pytest.approx(confs[0].boltzmann_pop + confs[1].boltzmann_pop, rel=1e-5) == 100.0

    # Test high temperature limit (T -> inf => 50/50 distribution)
    recalculate_boltzmann(confs, temperature_k=10000.0)
    assert pytest.approx(confs[0].boltzmann_pop, abs=1.0) == 50.0
    assert pytest.approx(confs[1].boltzmann_pop, abs=1.0) == 50.0


def test_kabsch_rmsd_and_clustering():
    # Setup water molecule
    syms = ["O", "H", "H"]
    c_ref = np.array([
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.469],
        [0.0, -0.757, -0.469],
    ])
    # Rotated duplicate
    rot_90 = np.array([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    c_dup = c_ref @ rot_90.T

    # Distinct conformation (bent further)
    c_distinct = np.array([
        [0.0, 0.0, 0.117],
        [0.0, 0.950, -0.300],
        [0.0, -0.950, -0.300],
    ])

    confs = [
        ConformerRecord(1, syms, c_ref, energy=-76.0),
        ConformerRecord(2, syms, c_dup, energy=-75.999),       # Higher energy duplicate
        ConformerRecord(3, syms, c_distinct, energy=-75.990),  # Distinct conformer
    ]

    recalculate_rmsds(confs, ref_idx=0)
    # The rotated duplicate should have RMSD ~ 0 after Kabsch alignment
    assert pytest.approx(confs[1].rmsd_all, abs=1e-5) == 0.0
    assert pytest.approx(confs[1].rmsd_heavy, abs=1e-5) == 0.0
    assert confs[2].rmsd_all > 0.1

    # Run clustering with threshold 0.05 Å
    cluster_conformers(confs, rmsd_threshold=0.05)
    assert confs[0].is_clustered is False
    assert confs[1].is_clustered is True
    assert confs[1].cluster_parent == 1
    assert confs[2].is_clustered is False


def test_export_and_qm_batch_generation(tmp_path):
    syms = ["C", "H", "H", "H", "H"]
    coords = np.zeros((5, 3))
    confs = [
        ConformerRecord(1, syms, coords, energy=-40.5, delta_e_kcal=0.0, boltzmann_pop=75.0),
        ConformerRecord(2, syms, coords, energy=-40.48, delta_e_kcal=1.2, boltzmann_pop=25.0),
    ]

    # Export XYZ
    xyz_path = tmp_path / "test_ensemble.xyz"
    n_exp = export_ensemble_xyz(confs, str(xyz_path))
    assert n_exp == 2
    assert os.path.isfile(xyz_path)
    content = xyz_path.read_text()
    assert "Conf_1" in content
    assert "Conf_2" in content

    # Export CSV
    csv_path = tmp_path / "test_metrics.csv"
    export_ensemble_csv(confs, str(csv_path))
    assert os.path.isfile(csv_path)
    csv_lines = csv_path.read_text().splitlines()
    assert len(csv_lines) == 3
    assert "Rank" in csv_lines[0]

    # Generate ORCA batch
    batch_dir = tmp_path / "orca_batch"
    gen_files = generate_qm_batch_scripts(confs, str(batch_dir), engine="orca", method="r2SCAN-3c", solvent="Water")
    assert len(gen_files) >= 4  # 2 inputs + run_batch.sh + run_batch.bat
    inp1 = (batch_dir / "conf_001.inp").read_text()
    assert "! r2SCAN-3c Opt CPCM(Water)" in inp1
    assert "* xyz 0 1" in inp1


def test_headless_dialog_operation(qapp):
    syms = ["C", "H", "H", "H", "H"]
    c1 = np.zeros((5, 3))
    c2 = np.ones((5, 3)) * 0.2

    dlg = ConformerEnsembleDialog(
        symbols=syms,
        trajectory_coords=[c1, c2],
        trajectory_energies=[-40.50, -40.48],
    )

    assert len(dlg.all_conformers) == 2
    assert dlg.table_conformers.rowCount() == 2
    assert dlg.chart_widget.minimumHeight() >= 100

    # Test dynamic temperature change
    dlg.spin_temp.setValue(100.0)
    # At lower temperature, lowest conformer population must increase
    p1_100k = dlg.all_conformers[0].boltzmann_pop
    dlg.spin_temp.setValue(600.0)
    p1_600k = dlg.all_conformers[0].boltzmann_pop
    assert p1_100k > p1_600k

    # Test selection and signals
    received_coords = []
    dlg.conformer_preview.connect(lambda s, c: received_coords.append(c))
    dlg.table_conformers.selectRow(1)
    assert len(received_coords) > 0
