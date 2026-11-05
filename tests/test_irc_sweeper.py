"""
Unit tests for the Intrinsic Reaction Coordinate (IRC) Analysis & Sweeper Engine.
"""

import numpy as np
import pytest

from agymolden.irc_sweeper import (
    IRCPoint,
    IRCPathway,
    assemble_irc_pathway,
    parse_gaussian_irc_text,
    parse_orca_irc_text,
    parse_irc_xyz,
    HARTREE_TO_KCAL,
)


def test_assemble_irc_pathway_bidirectional():
    symbols = ["H", "C", "N"]
    
    # 5 points: TS at s=0, Forward at s=+0.5, +1.0, Reverse at s=-0.5, -1.0
    # Provide them in disjoint order (TS -> Forward -> Reverse) as often printed by QM packages
    raw_s = [0.0, 0.5, 1.0, -0.5, -1.0]
    raw_energies = [
        -93.00,  # TS (highest energy)
        -93.02,  # Forward 1
        -93.05,  # Product (s=+1.0)
        -93.01,  # Reverse 1
        -93.03,  # Reactant (s=-1.0)
    ]
    raw_coords = [
        np.array([[0, 0, 0], [0, 0, 1.1], [0, 0, 2.1]]),
        np.array([[0, 0, 0], [0, 0, 1.2], [0, 0, 2.2]]),
        np.array([[0, 0, 0], [0, 0, 1.3], [0, 0, 2.3]]),
        np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]]),
        np.array([[0, 0, 0], [0, 0, 0.9], [0, 0, 1.9]]),
    ]

    pathway = assemble_irc_pathway(
        symbols=symbols,
        coords_list=raw_coords,
        energies_ha=raw_energies,
        rxn_coords=raw_s,
    )

    assert pathway.n_points == 5
    # Verification that reaction coordinates are monotonically increasing
    s_vals = pathway.get_reaction_coords()
    assert s_vals == sorted(s_vals)
    assert s_vals[0] == -1.0
    assert s_vals[-1] == 1.0

    # Reactant should be index 0 (s=-1.0, E=-93.03)
    assert pathway.reactant_index == 0
    assert pathway.points[0].is_reactant is True

    # TS should be index 2 (s=0.0, E=-93.00)
    assert pathway.ts_index == 2
    assert pathway.points[2].is_ts is True
    assert pathway.points[2].energy_ha == -93.00

    # Product should be index 4 (s=+1.0, E=-93.05)
    assert pathway.product_index == 4
    assert pathway.points[4].is_product is True

    # Barrier calculations
    # Forward barrier: E(TS) - E(Reactant) = -93.00 - (-93.03) = 0.03 Ha
    expected_fwd = 0.03 * HARTREE_TO_KCAL
    assert np.isclose(pathway.delta_e_fwd_kcal, expected_fwd, atol=1e-3)

    # Reverse barrier: E(TS) - E(Product) = -93.00 - (-93.05) = 0.05 Ha
    expected_rev = 0.05 * HARTREE_TO_KCAL
    assert np.isclose(pathway.delta_e_rev_kcal, expected_rev, atol=1e-3)

    # Reaction energy: E(Product) - E(Reactant) = -93.05 - (-93.03) = -0.02 Ha (Exothermic)
    expected_rxn = -0.02 * HARTREE_TO_KCAL
    assert np.isclose(pathway.delta_e_rxn_kcal, expected_rxn, atol=1e-3)
    assert pathway.is_exothermic is True


def test_parse_irc_xyz_movie():
    xyz_content = """3
Reaction Frame 1: s = -0.50000 amu½·Bohr, E = -76.02000000 Ha | Reactant
O   0.000000   0.000000   0.117000
H   0.000000   0.757000  -0.469000
H   0.000000  -0.757000  -0.469000
3
Reaction Frame 2: s = 0.00000 amu½·Bohr, E = -76.00000000 Ha | TS ‡
O   0.000000   0.000000   0.150000
H   0.000000   0.850000  -0.469000
H   0.000000  -0.850000  -0.469000
3
Reaction Frame 3: s = +0.50000 amu½·Bohr, E = -76.01500000 Ha | Product
O   0.000000   0.000000   0.117000
H   0.000000   0.757000  -0.469000
H   0.000000  -0.757000  -0.469000
"""
    pathway = parse_irc_xyz(xyz_content)
    assert pathway is not None
    assert pathway.n_points == 3
    assert pathway.symbols == ["O", "H", "H"]
    assert pathway.ts_index == 1
    assert pathway.points[1].is_ts is True

    # Test export
    csv_str = pathway.export_csv()
    assert "Forward Barrier" in csv_str
    assert "Reaction_Coord" in csv_str

    xyz_str = pathway.export_xyz()
    assert "IRC Step 1:" in xyz_str
    assert "Delta E =" in xyz_str


def test_parse_orca_irc_snippet():
    sample_orca = """
------------------------------------------------------------------------------
                          ORCA INTRINSIC REACTION COORDINATE
------------------------------------------------------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  C      0.000000    0.000000   -0.550000
  H      0.000000    1.000000    0.000000
------------------------------------------------------------------------------
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  C      0.000000    0.000000   -0.500000
  H      0.000000    0.950000    0.000000
------------------------------------------------------------------------------
                                IRC PATH SUMMARY
------------------------------------------------------------------------------
  Step   Reaction Coord.       Energy (Eh)         dE (kcal/mol)
     0         -0.30000       -155.01300000               0.0000
     1          0.00000       -155.01000000               1.8825
------------------------------------------------------------------------------
"""
    path = parse_orca_irc_text(sample_orca)
    assert path is not None
    assert path.symbols == ["C", "H"]
    assert path.n_points == 2
    assert path.points[0].reaction_coord == -0.30
    assert path.points[1].reaction_coord == 0.00


def test_parse_gaussian_irc_snippet():

    sample_gaussian = """
 Entering Gaussian System, IRC-Calculation
 ...
 Point Number:    1   Path Number:    1
 NET REACTION COORDINATE UP TO THIS POINT =    0.25000
 Total Energy =      -155.01254321
                         Standard orientation:                         
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000   -0.550000
      2          1           0        0.000000    1.000000    0.000000
 ---------------------------------------------------------------------
 Point Number:    1   Path Number:    2
 NET REACTION COORDINATE UP TO THIS POINT =    0.25000
 Total Energy =      -155.01300000
                         Standard orientation:                         
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000   -0.500000
      2          1           0        0.000000    0.950000    0.000000
 ---------------------------------------------------------------------
"""
    path = parse_gaussian_irc_text(sample_gaussian)
    assert path is not None
    assert path.symbols == ["C", "H"]
    assert path.n_points == 2
    # Path 2 has s inverted to -0.25, Path 1 has +0.25
    assert path.points[0].reaction_coord == -0.25
    assert path.points[1].reaction_coord == 0.25


def test_irc_sweeper_dialog_gui(qapp):
    from agymolden.dialog_irc_sweeper import IRCReactionSweeperDialog

    symbols = ["H", "C", "N"]
    raw_s = [-1.0, -0.5, 0.0, 0.5, 1.0]
    raw_energies = [-93.03, -93.01, -93.00, -93.02, -93.05]
    raw_coords = [
        np.array([[0, 0, 0], [0, 0, 0.9], [0, 0, 1.9]]),
        np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]]),
        np.array([[0, 0, 0], [0, 0, 1.1], [0, 0, 2.1]]),
        np.array([[0, 0, 0], [0, 0, 1.2], [0, 0, 2.2]]),
        np.array([[0, 0, 0], [0, 0, 1.3], [0, 0, 2.3]]),
    ]

    pathway = assemble_irc_pathway(symbols, raw_coords, raw_energies, raw_s)
    dlg = IRCReactionSweeperDialog(pathway=pathway)

    # Initial state should be at TS (index 2)
    assert dlg.current_step == 2
    assert "Step 3/5" in dlg.lbl_step_info.text()
    assert "Forward Barrier" in dlg.lbl_thermo_summary.text()

    # Test landmark jumps
    dlg._on_jump_reactant()
    assert dlg.current_step == 0
    assert dlg.slider.value() == 0

    dlg._on_jump_product()
    assert dlg.current_step == 4
    assert dlg.slider.value() == 4

    dlg._on_jump_ts()
    assert dlg.current_step == 2
    assert dlg.slider.value() == 2

    # Test unit modes
    for u in ["kcal", "kj", "ev", "ha"]:
        dlg.combo_units.setCurrentIndex(dlg.combo_units.findData(u))
        assert dlg.unit_mode == u

    # Test playback toggle
    assert not dlg.is_playing
    dlg._on_toggle_play()
    assert dlg.is_playing
    dlg._on_timer_tick()
    dlg._on_toggle_play()
    assert not dlg.is_playing

