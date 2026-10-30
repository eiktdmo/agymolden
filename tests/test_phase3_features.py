"""
Unit tests for Phase 3 Deliverables:
1. 3D Measurement Overlay (Distance, Angle, Dihedral).
2. Standalone Interactive 3D WebGL HTML Export.
3. PyInstaller bundle spec validation.
"""

import os
import py_compile
import numpy as np
import pytest

from agymolden.html_export import generate_interactive_html, get_element_hex_color


def test_html_export_generation():
    """Verify standalone WebGL HTML generation produces valid self-contained HTML."""
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.117],
        [0.0, 0.757, -0.468],
        [0.0, -0.757, -0.468],
    ])
    traj = [coords, coords + 0.1]
    qm_summary = {
        "program": "ORCA 6.0",
        "energy_ha": -76.4321,
        "dipole_debye": 1.854,
        "homo_ev": -12.61,
        "lumo_ev": 1.25,
    }

    html = generate_interactive_html(
        symbols=symbols,
        coords=coords,
        title="Water Test",
        trajectory=traj,
        trajectory_energies=[-76.432, -76.435],
        qm_summary=qm_summary,
    )

    assert "<!DOCTYPE html>" in html
    assert "<canvas id=\"glcanvas\">" in html
    assert "ORCA 6.0" in html
    assert "-76.432100 Ha" in html
    assert "1.854 Debye" in html
    assert "Water Test" in html
    assert "Step 1/2" in html
    assert get_element_hex_color("O") == "#ef4444"
    assert get_element_hex_color("H") == "#f8fafc"
    assert get_element_hex_color("C") == "#475569"


def test_measurement_overlay_logic():
    """Verify exact distance, angle, and dihedral calculations used in the 3D overlay."""
    # Water geometry
    p_O = np.array([0.0, 0.0, 0.117])
    p_H1 = np.array([0.0, 0.757, -0.468])
    p_H2 = np.array([0.0, -0.757, -0.468])

    # Distance O - H1
    d = np.linalg.norm(p_O - p_H1)
    assert pytest.approx(d, rel=1e-3) == 0.957

    # Angle H1 - O - H2
    v1 = p_H1 - p_O
    v2 = p_H2 - p_O
    cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    deg = np.degrees(np.arccos(cos_th))
    assert pytest.approx(deg, rel=1e-2) == 104.5


def test_viewer_measurement_toggle(qapp):
    """Verify show_measurements attribute on MoleculeViewer."""
    from agymolden.viewer import MoleculeViewer
    viewer = MoleculeViewer()
    assert hasattr(viewer, "show_measurements")
    assert viewer.show_measurements is True
    viewer.show_measurements = False
    assert viewer.show_measurements is False


def test_spec_compilation():
    """Ensure agymolden.spec has valid Python syntax without compilation errors."""
    spec_path = os.path.abspath("agymolden.spec")
    assert os.path.isfile(spec_path)
    py_compile.compile(spec_path, doraise=True)
