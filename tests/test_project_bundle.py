"""
Tests for AgyMolden Project Bundle (.agyproj) format reader and writer.
Verifies lossless serialization and deserialization of coordinates, Z-matrix,
calculation history, and viewer metadata.
"""

import os
import numpy as np
import pytest

from agymolden.zmatrix import ZAtom
from agymolden.project_bundle import save_project_bundle, load_project_bundle


def test_project_bundle_save_and_load(tmp_path):
    bundle_file = str(tmp_path / "benzene_session.agyproj")

    zatoms = [
        ZAtom(symbol="C", ref1=None, dist=0.0, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="C", ref1=0, dist=1.40, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="C", ref1=1, dist=1.40, ref2=0, angle=120.0, ref3=None, dihedral=0.0),
    ]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.40, 0.0, 0.0],
        [2.10, 1.212, 0.0],
    ])
    variables = {"r_cc": 1.40}
    cross_bonds = [(0, 2)]
    calc_history = [
        {"engine": "orca", "method": "B3LYP", "basis": "def2-SVP", "energy_hartree": -232.145}
    ]
    viewer_state = {"theme": "dark", "style": "Ball & Stick"}

    success = save_project_bundle(
        filepath=bundle_file,
        zatoms=zatoms,
        coords=coords,
        variables=variables,
        cross_bonds=cross_bonds,
        calculation_history=calc_history,
        viewer_state=viewer_state,
        metadata={"title": "Benzene fragment study"},
    )
    assert success is True
    assert os.path.isfile(bundle_file)

    loaded = load_project_bundle(bundle_file)
    assert loaded is not None
    assert len(loaded["zatoms"]) == 3
    assert loaded["zatoms"][1].symbol == "C"
    assert loaded["zatoms"][1].ref1 == 0
    assert pytest.approx(loaded["zatoms"][1].dist) == 1.40

    np.testing.assert_allclose(loaded["coords"], coords, atol=1e-5)
    assert loaded["variables"] == {"r_cc": 1.40}
    assert loaded["cross_bonds"] == [(0, 2)]
    assert len(loaded["calculation_history"]) == 1
    assert loaded["calculation_history"][0]["engine"] == "orca"
    assert loaded["viewer_state"]["theme"] == "dark"
    assert loaded["metadata"]["title"] == "Benzene fragment study"
