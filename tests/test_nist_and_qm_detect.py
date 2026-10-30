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
