"""
Tests for crystallographic CIF export and Gaussian Cube file write/read roundtrip.
Verifies format preservation, fractional-to-Cartesian conversion, and volumetric metadata.
"""

import os
import numpy as np
import pytest

from agymolden.zmatrix import format_cif_text, write_cif_file, parse_cif
from agymolden.volumetric import VolumetricData, write_cube_file, parse_cube_file


def test_cif_format_and_roundtrip(tmp_path):
    cif_file = str(tmp_path / "water_cell.cif")

    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.757, 0.586, 0.0],
        [-0.757, 0.586, 0.0],
    ])
    cell = {
        "a": 12.0,
        "b": 12.0,
        "c": 14.0,
        "alpha": 90.0,
        "beta": 90.0,
        "gamma": 90.0,
    }

    txt = format_cif_text(symbols, coords, cell=cell, data_name="H2O_Test")
    assert "data_H2O_Test" in txt
    assert "_cell_length_a" in txt
    assert "_atom_site_fract_x" in txt
    assert "O1" in txt

    write_cif_file(cif_file, symbols, coords, cell=cell)
    assert os.path.isfile(cif_file)

    parsed_syms, parsed_coords, parsed_cell = parse_cif(cif_file)
    assert parsed_syms == ["O", "H", "H"]
    assert pytest.approx(parsed_cell["a"]) == 12.0
    assert pytest.approx(parsed_cell["c"]) == 14.0
    np.testing.assert_allclose(parsed_coords, coords, atol=1e-4)


def test_cube_export_and_roundtrip(tmp_path):
    cube_file = str(tmp_path / "synthetic_density.cube")

    symbols = ["C", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.08, 0.0, 0.0],
    ])
    origin = np.array([-2.0, -2.0, -2.0])
    step_x = np.array([0.5, 0.0, 0.0])
    step_y = np.array([0.0, 0.5, 0.0])
    step_z = np.array([0.0, 0.0, 0.5])
    voxels = (8, 8, 8)
    grid_arr = np.linspace(0.001, 0.85, 8 * 8 * 8).reshape(voxels)

    vol = VolumetricData(
        title="Synthetic Electron Density",
        comment="AgyMolden Test Grid",
        symbols=symbols,
        atom_coords=coords,
        voxels_x=voxels[0],
        voxels_y=voxels[1],
        voxels_z=voxels[2],
        origin=origin,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_arr,
        is_orbital=False,
    )

    write_cube_file(cube_file, vol)
    assert os.path.isfile(cube_file)

    loaded_vol = parse_cube_file(cube_file)
    assert loaded_vol.symbols == ["C", "H"]
    assert loaded_vol.voxels_x == 8
    assert loaded_vol.voxels_y == 8
    assert loaded_vol.voxels_z == 8
    np.testing.assert_allclose(loaded_vol.origin, origin, atol=1e-4)
    np.testing.assert_allclose(loaded_vol.atom_coords, coords, atol=1e-4)
    np.testing.assert_allclose(loaded_vol.grid_data, grid_arr, atol=1e-4)
