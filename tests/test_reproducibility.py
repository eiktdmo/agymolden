"""Tests for Reproducibility Bundle Exporter and CLI bundle command."""

import os
import zipfile
import json
import numpy as np
import pytest

from agymolden.reproducibility import create_reproducibility_bundle, generate_si_markdown
from agymolden.qm_log_parser import QMLogParseResult
from agymolden.cli import cmd_bundle, build_cli_parser


def test_create_reproducibility_bundle(tmp_path):
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ])

    log_res = QMLogParseResult(
        program_name="ORCA 5.0.4",
        symbols=symbols,
        trajectory_coords=[coords],
        trajectory_energies=[-76.43215],
        vibrational_analysis=None,
        homo_ev=-12.5,
        lumo_ev=1.2,
        dipole_debye=1.85,
        zpe_ha=0.0215,
        enthalpy_ha=-76.406,
        gibbs_ha=-76.428,
        converged=True,
    )

    zip_out = str(tmp_path / "water_bundle.zip")
    res_path = create_reproducibility_bundle(
        output_archive_path=zip_out,
        symbols=symbols,
        coords=coords,
        log_result=log_res,
        job_title="Water Molecule B3LYP",
    )

    assert os.path.isfile(res_path)
    assert zipfile.is_zipfile(res_path)

    with zipfile.ZipFile(res_path, "r") as zf:
        namelist = zf.namelist()
        assert "GEOMETRY.xyz" in namelist
        assert "GEOMETRY.zmat" in namelist
        assert "CALCULATION_SUMMARY.json" in namelist
        assert "COMPUTATIONAL_DETAILS.md" in namelist

        # Verify summary JSON
        summary_raw = zf.read("CALCULATION_SUMMARY.json").decode("utf-8")
        summary = json.loads(summary_raw)
        assert summary["program"] == "ORCA 5.0.4"
        assert summary["final_energy_ha"] == -76.43215
        assert summary["converged"] is True
        assert summary["n_atoms"] == 3
        assert summary["dipole_debye"] == 1.85

        # Verify Markdown content
        md_text = zf.read("COMPUTATIONAL_DETAILS.md").decode("utf-8")
        assert "Supporting Information: Water Molecule B3LYP" in md_text
        assert "ORCA 5.0.4" in md_text
        assert "-76.43215" in md_text
        assert "Cartesian Coordinates" in md_text


def test_cli_bundle_command(tmp_path):
    # Create sample xyz
    xyz_path = str(tmp_path / "test.xyz")
    with open(xyz_path, "w") as f:
        f.write("2\nH2 molecule\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n")

    out_zip = str(tmp_path / "h2_bundle.zip")
    parser = build_cli_parser()
    args = parser.parse_args(["bundle", xyz_path, "-o", out_zip, "--title", "Dihydrogen"])

    code = cmd_bundle(args)
    assert code == 0
    assert os.path.isfile(out_zip)

    with zipfile.ZipFile(out_zip, "r") as zf:
        assert "GEOMETRY.xyz" in zf.namelist()
        assert "COMPUTATIONAL_DETAILS.md" in zf.namelist()
