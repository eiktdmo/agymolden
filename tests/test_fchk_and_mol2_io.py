import pytest
import os
import numpy as np
import tempfile
from agymolden.zmatrix import parse_mol2, write_mol2
from agymolden.qm_log_parser import parse_qm_logfile, _parse_fchk_logfile_native


def test_fchk_native_parser():
    fchk_sample = """# Test Water Formatted Checkpoint
SP          RHF                         STO-3G
Number of atoms                            I               3
Charge                                     I               0
Multiplicity                               I               1
Atomic numbers                             I   N=           3
           8           1           1
Nuclear charges                            R   N=           3
   8.00000000E+00  1.00000000E+00  1.00000000E+00
Current cartesian coordinates              R   N=           9
   0.00000000E+00  0.00000000E+00  2.24921650E-01  0.00000000E+00  1.45235805E+00 -8.99686600E-01
   0.00000000E+00 -1.45235805E+00 -8.99686600E-01
Cartesian Gradient                         R   N=           9
   0.00000000D+00  0.00000000D+00  1.50000000D-02  0.00000000D+00  1.00000000D-02 -7.50000000D-03
   0.00000000D+00 -1.00000000D-02 -7.50000000D-03
Total Energy                               R     -7.49653592414168D+01
"""
    with tempfile.NamedTemporaryFile("w", suffix=".fchk", delete=False) as f:
        f.write(fchk_sample)
        tmp_path = f.name

    try:
        # 1. Test native helper directly
        res_native = _parse_fchk_logfile_native(tmp_path)
        assert res_native is not None
        assert res_native.program_name == "Gaussian"
        assert res_native.symbols == ["O", "H", "H"]
        assert len(res_native.trajectory_coords) == 1
        coords = res_native.trajectory_coords[0]
        assert coords.shape == (3, 3)
        # Bohr to Angstrom check: 0.22492165 * 0.529177 ~= 0.11902
        assert np.isclose(coords[0, 2], 0.119024, atol=1e-3)
        assert len(res_native.trajectory_energies) == 1
        assert res_native.trajectory_energies[0] < -2000.0  # -74.965 Eh * 27.211 eV
        assert res_native.nuclear_forces is not None
        assert res_native.nuclear_forces.shape == (3, 3)

        # 2. Test universal parse_qm_logfile
        res_universal = parse_qm_logfile(tmp_path)
        assert res_universal.program_name == "Gaussian"
        assert res_universal.symbols == ["O", "H", "H"]
        assert len(res_universal.trajectory_coords) == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_mol2_roundtrip_io():
    symbols = ["C", "H", "H", "O"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.08, 0.0, 0.0],
        [-0.54, 0.93, 0.0],
        [-0.54, -0.93, 0.0],
    ], dtype=np.float64)
    bonds = [(0, 1), (0, 2), (0, 3)]
    bond_orders = {(0, 1): 1, (0, 2): 1, (0, 3): 2}

    with tempfile.NamedTemporaryFile("w", suffix=".mol2", delete=False) as f:
        tmp_path = f.name

    try:
        # Write Mol2
        write_mol2(tmp_path, symbols, coords, bonds=bonds, bond_orders=bond_orders, molecule_name="Formaldehyde")
        assert os.path.isfile(tmp_path)

        # Read Mol2 back
        read_syms, read_coords, read_bonds, read_bo = parse_mol2(tmp_path)
        assert read_syms == ["C", "H", "H", "O"]
        assert np.allclose(read_coords, coords, atol=1e-3)
        assert read_bonds == [(0, 1), (0, 2), (0, 3)]
        assert read_bo[(0, 3)] == 2
        assert read_bo[(0, 1)] == 1
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
