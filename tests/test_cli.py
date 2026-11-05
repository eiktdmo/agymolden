"""Test headless CLI commands: parse, gen, check, convert."""

import json
import os
import subprocess
import sys
import numpy as np
import pytest

from agymolden.cli import build_cli_parser, cmd_parse, cmd_gen, cmd_check, cmd_convert


@pytest.fixture
def sample_xyz(tmp_path):
    p = tmp_path / "water.xyz"
    p.write_text("""3
Water molecule
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
""")
    return str(p)


def test_cli_convert_xyz_to_zmat(sample_xyz, tmp_path):
    out_zmat = str(tmp_path / "water.zmat")
    parser = build_cli_parser()
    args = parser.parse_args(["convert", sample_xyz, "-o", out_zmat])
    assert cmd_convert(args) == 0
    assert os.path.isfile(out_zmat)
    with open(out_zmat) as f:
        txt = f.read()
    assert "O" in txt and "H" in txt


def test_cli_check_valid_and_invalid(sample_xyz):
    parser = build_cli_parser()
    # Neutral singlet water (10 electrons) -> Valid
    args_ok = parser.parse_args(["check", sample_xyz, "--mult", "1", "--charge", "0"])
    assert cmd_check(args_ok) == 0

    # Neutral doublet water (10 electrons, doublet) -> Parity violation
    args_err = parser.parse_args(["check", sample_xyz, "--mult", "2", "--charge", "0"])
    assert cmd_check(args_err) != 0


def test_cli_gen_orca_and_gaussian(sample_xyz, tmp_path):
    parser = build_cli_parser()
    out_inp = str(tmp_path / "orca_test.inp")
    args = parser.parse_args([
        "gen", sample_xyz,
        "--code", "orca",
        "--method", "b3lyp",
        "--basis", "def2-svp",
        "--job-type", "Opt",
        "-o", out_inp,
    ])
    assert cmd_gen(args) == 0
    assert os.path.isfile(out_inp)
    with open(out_inp) as f:
        txt = f.read()
    assert "b3lyp def2-svp Opt" in txt or "! b3lyp def2-svp Opt" in txt


def test_cli_parse_summary_and_json(capsys):
    example_log = os.path.join(os.path.dirname(__file__), "..", "examples", "AgyMolden_generated_calculation.log")
    if not os.path.isfile(example_log):
        pytest.skip("example calculation log not found")

    parser = build_cli_parser()
    # Test human readable parse
    args = parser.parse_args(["parse", example_log])
    assert cmd_parse(args) == 0
    out = capsys.readouterr().out
    assert "Gaussian" in out
    assert "Final Energy:" in out

    # Test json parse
    args_json = parser.parse_args(["parse", example_log, "--json"])
    assert cmd_parse(args_json) == 0
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["program"] == "Gaussian"
    assert "final_energy_ha" in data
    assert data["normal_termination"] is True
