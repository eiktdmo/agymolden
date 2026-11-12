"""
Unit tests for Solvent-Accessible Surface Area (SASA),
Natural Transition Orbitals (NTO Photochemistry),
and Model Context Protocol (ChemMCP) AI assistant stdio server.
"""

import json
import numpy as np
import pytest
from PyQt6 import QtWidgets, QtCore

from agymolden.sasa import (
    generate_fibonacci_sphere,
    calculate_sasa,
    SASAResult,
    POLAR_ELEMENTS,
)
from agymolden.dialog_sasa import SASADialog
from agymolden.nto_engine import (
    compute_ntos_from_amplitudes,
    parse_td_dft_and_compute_ntos,
    NTOResult,
    ExcitedStateInfo,
)
from agymolden.dialog_nto import NTODialog
from agymolden.mcp_server import handle_mcp_request


# =========================================================================
# 1. SASA (Solvent-Accessible Surface Area) Tests
# =========================================================================

def test_fibonacci_sphere_points():
    pts = generate_fibonacci_sphere(120)
    assert pts.shape == (120, 3)
    # Each point must lie on the unit sphere (norm == 1.0)
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_sasa_water_molecule():
    # Water geometry (Å)
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.000000, 0.000000, 0.117300],
        [0.000000, 0.757200, -0.469200],
        [0.000000, -0.757200, -0.469200],
    ])
    res = calculate_sasa(symbols, coords, probe_radius=1.4, n_points_per_atom=120)

    assert isinstance(res, SASAResult)
    assert res.total_sasa > 50.0  # Typical water SASA ~80-120 Å²
    assert res.polar_sasa > 0.0
    assert len(res.atom_results) == 3
    assert len(res.exposure_vector) == 3
    assert np.all(res.exposure_vector >= 0.0)
    assert np.all(res.exposure_vector <= 1.0)


def test_sasa_methane_hydrophobic():
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.629118, 0.629118, 0.629118],
        [-0.629118, -0.629118, 0.629118],
        [-0.629118, 0.629118, -0.629118],
        [0.629118, -0.629118, -0.629118],
    ])
    res = calculate_sasa(symbols, coords, probe_radius=1.4, n_points_per_atom=100)
    # Methane is entirely non-polar (C, H)
    assert res.polar_sasa == 0.0
    assert res.hydrophobic_sasa == res.total_sasa


def test_sasa_dialog_smoke(qapp):
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1],
        [0.0, 0.7, -0.4],
        [0.0, -0.7, -0.4],
    ])
    dlg = SASADialog(symbols=symbols, coords=coords)
    dlg.show()

    assert dlg.tbl_atoms.rowCount() == 3
    dlg.spin_probe.setValue(1.2)
    dlg._recompute()
    assert dlg.result is not None
    dlg.close()


# =========================================================================
# 2. NTO (Natural Transition Orbitals) Tests
# =========================================================================

def test_nto_svd_single_transition():
    # Pure single configuration 4 -> 0 (HOMO -> LUMO) with amplitude 1.0
    transitions = [(4, 0, 1.0)]
    info = compute_ntos_from_amplitudes(transitions, n_occ=5, n_virt=5)

    assert len(info.nto_pairs) == 1
    p1 = info.nto_pairs[0]
    np.testing.assert_allclose(p1.eigenvalue_weight, 1.0, atol=1e-5)
    assert 4 in p1.hole_mo_weights
    assert 0 in p1.particle_mo_weights


def test_nto_svd_multiconfigurational():
    # Superposition: 0.8 * (3 -> 0) + 0.6 * (4 -> 1)
    # Sum of squares = 0.64 + 0.36 = 1.00
    transitions = [(3, 0, 0.8), (4, 1, 0.6)]
    info = compute_ntos_from_amplitudes(transitions, n_occ=5, n_virt=5)

    assert len(info.nto_pairs) == 2
    # Weights should sum to 1.0
    total_weight = sum(p.eigenvalue_weight for p in info.nto_pairs)
    np.testing.assert_allclose(total_weight, 1.0, atol=1e-5)
    assert info.nto_pairs[0].eigenvalue_weight >= info.nto_pairs[1].eigenvalue_weight
    np.testing.assert_allclose(info.nto_pairs[0].eigenvalue_weight, 0.64, atol=1e-4)
    np.testing.assert_allclose(info.nto_pairs[1].eigenvalue_weight, 0.36, atol=1e-4)


def test_parse_orca_tddft():
    mock_orca = """
-----------------------------------------------------------------------------
         TD-DFT EXCITED STATES (SINGLETS)
-----------------------------------------------------------------------------
STATE  1:  E=   0.142350 au      3.873 eV    312.4 nm  f=  0.0845
   25a -> 26a  :     0.945000 (c= 0.945)
   24a -> 26a  :     0.150000 (c= 0.150)

STATE  2:  E=   0.185200 au      5.040 eV    246.0 nm  f=  0.0020
   24a -> 27a  :     0.980000 (c= 0.980)
"""
    res = parse_td_dft_and_compute_ntos(mock_orca)
    assert len(res.excited_states) == 2
    st1 = res.excited_states[0]
    assert st1.state_index == 1
    assert st1.energy_ev == 3.873
    assert st1.oscillator_strength == 0.0845
    assert len(st1.nto_pairs) >= 1


def test_parse_gaussian_tddft():
    mock_gaussian = """
 Excited State   1:      Singlet-A      3.5421 eV  350.03 nm  f=0.1245 <S**2>=0.000
      18 -> 19         0.68412
      17 -> 20         0.15234
 Excited State   2:      Triplet-A      4.1200 eV  300.93 nm  f=0.0000 <S**2>=2.000
      18 -> 20         0.70110
"""
    res = parse_td_dft_and_compute_ntos(mock_gaussian)
    assert len(res.excited_states) == 2
    assert res.excited_states[0].state_index == 1
    assert res.excited_states[0].multiplicity == "Singlet"
    assert res.excited_states[0].energy_ev == 3.5421
    assert res.excited_states[1].multiplicity == "Triplet"
    assert len(res.excited_states[0].nto_pairs) >= 1


def test_nto_dialog_smoke(qapp):
    dlg = NTODialog()
    dlg.show()
    assert dlg.tbl_states.rowCount() >= 1
    dlg.tbl_states.selectRow(0)
    assert "Hole" in dlg.lbl_details.text()
    dlg.close()


# =========================================================================
# 3. ChemMCP (Model Context Protocol) Tests
# =========================================================================

def test_mcp_initialize():
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test-client", "version": "1.0"},
        },
    }
    resp = handle_mcp_request(req)
    assert resp["id"] == 1
    assert resp["result"]["serverInfo"]["name"] == "AgyMolden-ChemMCP"
    assert "tools" in resp["result"]["capabilities"]


def test_mcp_tools_list():
    req = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {},
    }
    resp = handle_mcp_request(req)
    tools = resp["result"]["tools"]
    tool_names = [t["name"] for t in tools]
    assert "get_molecule_info" in tool_names
    assert "calculate_sasa" in tool_names
    assert "calculate_sterics" in tool_names
    assert "generate_orca_input" in tool_names


def test_mcp_tool_call_sasa_and_orca():
    water_ctx = {
        "symbols": ["O", "H", "H"],
        "coords": [
            [0.0, 0.0, 0.117],
            [0.0, 0.757, -0.469],
            [0.0, -0.757, -0.469],
        ],
    }

    # Test SASA tool call
    call_sasa = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "calculate_sasa",
            "arguments": {"probe_radius": 1.4},
        },
    }
    resp_sasa = handle_mcp_request(call_sasa, context_molecule=water_ctx)
    assert "result" in resp_sasa
    content = resp_sasa["result"]["content"][0]["text"]
    data = json.loads(content)
    assert data["total_sasa_ang2"] > 50.0

    # Test ORCA input generation tool call
    call_orca = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "generate_orca_input",
            "arguments": {
                "functional": "r2SCAN-3c",
                "job_type": "Opt Freq",
                "charge": 0,
                "multiplicity": 1,
            },
        },
    }
    resp_orca = handle_mcp_request(call_orca, context_molecule=water_ctx)
    text = resp_orca["result"]["content"][0]["text"]
    assert "! r2SCAN-3c" in text
    assert "* xyz 0 1" in text

    # Test Quasi-Harmonic tool call
    call_qh = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call",
        "params": {
            "name": "calculate_quasiharmonic",
            "arguments": {"frequencies_cm1": [35.0, 90.0, 1500.0], "temperature_k": 298.15},
        },
    }
    resp_qh = handle_mcp_request(call_qh, context_molecule=water_ctx)
    qh_data = json.loads(resp_qh["result"]["content"][0]["text"])
    assert qh_data["zpe_kcal_mol"] > 0.0
    assert "s_vib_grimme_cal_mol_k" in qh_data

    # Test SI LaTeX tool call with inline xyz
    call_si = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "export_si_latex",
            "arguments": {
                "name": "Water_Min",
                "xyz": "3\nWater\nO 0.0 0.0 0.117\nH 0.0 0.757 -0.469\nH 0.0 -0.757 -0.469\n",
            },
        },
    }
    resp_si = handle_mcp_request(call_si)
    si_text = resp_si["result"]["content"][0]["text"]
    assert r"\documentclass" in si_text
    assert "Water_Min" in si_text
