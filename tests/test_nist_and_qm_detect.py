"""Unit tests for NIST vibrational scaling factors and QM executable detection & route configuration."""

import os
import pytest
from agymolden.vibrations import (
    get_nist_scale_factor,
    NIST_VIBRATIONAL_SCALE_FACTORS,
)
from agymolden.qm_detect import (
    detect_qm_executable,
    save_qm_route,
    get_qm_route,
    search_qm_executable_anywhere,
    _probe_paths,
)


def test_nist_vibrational_scale_factors():
    # Test B3LYP lookups
    fac_b3lyp_631g, cite = get_nist_scale_factor("B3LYP", "6-31G*")
    assert fac_b3lyp_631g == pytest.approx(0.9620, rel=1e-3)
    assert "NIST CCCBDB" in cite

    # Test B3LYP def2-SVP
    fac_b3lyp_svp, _ = get_nist_scale_factor("B3LYP", "def2-SVP")
    assert fac_b3lyp_svp == pytest.approx(0.9650, rel=1e-3)

    # Test HF 6-31G(d) alias
    fac_hf, _ = get_nist_scale_factor("HF", "6-31G(d)")
    assert fac_hf == pytest.approx(0.8985, rel=1e-3)

    # Test PBE0
    fac_pbe0, _ = get_nist_scale_factor("PBE0", "def2-TZVP")
    assert fac_pbe0 == pytest.approx(0.9580, rel=1e-3)

    # Unknown method falls back to 1.000
    fac_unknown, _ = get_nist_scale_factor("MadeUpMethod123", "sto-3g")
    assert fac_unknown == 1.0000


def test_qm_detect_and_route_customization(tmp_path):
    # Test detection of ORCA and Gaussian on current system
    orca_info = detect_qm_executable("orca")
    assert orca_info.code_key == "orca"
    assert orca_info.display_name == "ORCA"
    # Either found on system or correctly reported not found with helpful message
    if orca_info.found:
        assert os.path.exists(orca_info.resolved_path)

    # Test custom route simulation
    fake_bin = tmp_path / "custom_orca_wrapper.sh"
    fake_bin.write_text("#!/bin/sh\necho 'Fake ORCA'\n")
    fake_bin.chmod(0o755)

    custom_info = detect_qm_executable("orca", custom_route=str(fake_bin))
    assert custom_info.found is True
    assert custom_info.resolved_path == str(fake_bin)
    assert "Configured route executable found" in custom_info.message

    # Test route persistence via save_qm_route / get_qm_route (restore the user's saved route afterwards)
    original = get_qm_route("orca")
    try:
        save_qm_route("orca", str(fake_bin))
        assert get_qm_route("orca") == str(fake_bin)
    finally:
        save_qm_route("orca", original)


def test_orca_directory_spaces_permissions_and_search(tmp_path, monkeypatch):
    # 1. Directory route resolution (both <dir>/orca and <dir>/bin/orca)
    orca_dir = tmp_path / "orca_6_0_1"
    orca_dir.mkdir()
    orca_bin = orca_dir / "orca"
    orca_bin.write_text("#!/bin/sh\necho 'ORCA'\n")
    orca_bin.chmod(0o755)

    dir_info = detect_qm_executable("orca", custom_route=str(orca_dir))
    assert dir_info.found is True
    assert dir_info.resolved_path == str(orca_bin)

    # 2. Path with spaces
    spaced_dir = tmp_path / "My Programs" / "ORCA Suite"
    spaced_dir.mkdir(parents=True)
    spaced_bin = spaced_dir / "orca"
    spaced_bin.write_text("#!/bin/sh\necho 'ORCA'\n")
    spaced_bin.chmod(0o755)

    spaced_info = detect_qm_executable("orca", custom_route=str(spaced_bin))
    assert spaced_info.found is True
    assert spaced_info.resolved_path == str(spaced_bin)

    # 3. Non-executable file gives actionable chmod hint (POSIX)
    if os.name != "nt":
        noexec_bin = tmp_path / "orca_noexec"
        noexec_bin.write_text("#!/bin/sh\necho 'ORCA'\n")
        noexec_bin.chmod(0o644)
        noexec_info = detect_qm_executable("orca", custom_route=str(noexec_bin))
        assert noexec_info.found is False
        assert "chmod +x" in noexec_info.message

    # 4. search_qm_executable_anywhere finds nested binary
    nested_dir = tmp_path / "custom_root" / "lab_tools" / "orca_release" / "bin"
    nested_dir.mkdir(parents=True)
    nested_bin = nested_dir / "orca"
    nested_bin.write_text("#!/bin/sh\necho 'ORCA'\n")
    nested_bin.chmod(0o755)

    matches = search_qm_executable_anywhere("orca", search_roots=[str(tmp_path / "custom_root")], timeout_sec=2.0)
    assert str(nested_bin) in matches

    # 5. ORCA_HOME env var probing
    monkeypatch.setenv("ORCA_HOME", str(orca_dir))
    cands = _probe_paths("orca")
    assert str(orca_bin) in cands


def test_all_qm_packages_directory_search_and_routes(tmp_path, monkeypatch):
    # Verify directory resolution, search, and route persistence across all QM packages
    cases = [
        ("gaussian", "g16", "g16root"),
        ("xtb", "xtb", "XTBHOME"),
        ("openlowdin", "openlowdin", "OPENLOWDIN_HOME"),
        ("qchem", "qchem", "QCHEM_HOME"),
        ("psi4", "psi4", "PSI4_HOME"),
    ]
    for code_key, bin_name, env_var in cases:
        pkg_dir = tmp_path / f"{code_key}_pkg"
        bin_dir = pkg_dir / "bin"
        bin_dir.mkdir(parents=True)
        exe = bin_dir / bin_name
        exe.write_text("#!/bin/sh\necho ok\n")
        exe.chmod(0o755)

        # Directory custom route resolves <dir>/bin/<bin_name>
        info = detect_qm_executable(code_key, custom_route=str(pkg_dir))
        assert info.found is True, f"{code_key} directory resolution failed: {info}"
        assert info.resolved_path == str(exe)

        # Environment variable probing finds it
        monkeypatch.setenv(env_var, str(pkg_dir))
        assert str(exe) in _probe_paths(code_key)

        # System search finds it under custom root
        matches = search_qm_executable_anywhere(code_key, search_roots=[str(pkg_dir)], timeout_sec=2.0)
        assert str(exe) in matches

        # Route persistence normalizes all code keys (including psi4)
        orig = get_qm_route(code_key)
        try:
            save_qm_route(code_key, str(exe))
            assert get_qm_route(code_key) == str(exe)
        finally:
            save_qm_route(code_key, orig)
