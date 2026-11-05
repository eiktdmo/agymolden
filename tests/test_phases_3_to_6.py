"""
Comprehensive unit tests for Phases 3, 4, 5, and 6:
  - Phase 3: GaussView "SmartSlides" (arbitrary 2/3/4-atom distance, angle, dihedral manipulation)
             & Local Stereocenter (R/S) Inversion (`agymolden.smartslides`)
  - Phase 4: GaussView Element Coordination & Hybridization Templates (`agymolden.coordination`)
  - Phase 5: Categorized Fragment Library & Custom User Fragments (`agymolden.fragment_library`)
  - Phase 6: Frozen-Atom Constraints & 1D Dihedral Conformer Scanner (`agymolden.conformer_scan`)
"""

import os
import numpy as np
import pytest
from PyQt6 import QtWidgets

from agymolden.zmatrix import (
    ZAtom,
    zmatrix_to_cartesian,
    parse_zmatrix_text,
    calculate_bond_distance,
    calculate_angle,
    calculate_dihedral,
)
from agymolden.fragments import FRAGMENTS, SAMPLE_MOLECULES
from agymolden.smartslides import (
    modify_distance,
    modify_angle,
    modify_dihedral,
    invert_chiral_center,
)
from agymolden.coordination import (
    COORDINATION_TEMPLATES,
    get_coordination_vectors,
    resolve_coordination_template,
)
from agymolden.fragment_library import (
    FRAGMENT_CATEGORIES,
    get_fragments_by_category,
    save_custom_fragment,
    load_custom_fragments_into_registry,
)
from agymolden.forcefield import optimize_geometry_force_field, detect_connectivity
from agymolden.conformer_scan import run_dihedral_rotor_scan
from agymolden.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ==============================================================================
# Phase 3: SmartSlides & Local Stereocenter Inversion Tests
# ==============================================================================

def test_smartslides_modify_distance_modes():
    """Verify arbitrary 2-atom distance modification in group, symmetric, and atom modes."""
    zatoms, _ = parse_zmatrix_text(SAMPLE_MOLECULES["Ethane (C2H6)"])
    symbols = [a.symbol for a in zatoms]
    coords = zmatrix_to_cartesian(zatoms)
    bonds = detect_connectivity(symbols, coords)

    # C1 is index 0, C2 is index 1
    orig_c1 = coords[0].copy()
    orig_c2_h_dists = [
        calculate_bond_distance(coords[1], coords[h])
        for h in (5, 6, 7)
    ]

    # Mode 1: "group" -> C1 stays fixed, C2 and its attached H's (5, 6, 7) translate together
    new_c_grp = modify_distance(symbols, coords, bonds, 0, 1, 2.10, mode="group")
    assert np.isclose(calculate_bond_distance(new_c_grp[0], new_c_grp[1]), 2.10, atol=1e-6)
    np.testing.assert_allclose(new_c_grp[0], orig_c1, atol=1e-7)
    for k, h in enumerate((5, 6, 7)):
        assert np.isclose(
            calculate_bond_distance(new_c_grp[1], new_c_grp[h]), orig_c2_h_dists[k], atol=1e-6
        )

    # Mode 2: "symmetric" -> both methyl groups move by equal and opposite amounts
    new_c_sym = modify_distance(symbols, coords, bonds, 0, 1, 2.10, mode="symmetric")
    assert np.isclose(calculate_bond_distance(new_c_sym[0], new_c_sym[1]), 2.10, atol=1e-6)
    disp0 = np.linalg.norm(new_c_sym[0] - coords[0])
    disp1 = np.linalg.norm(new_c_sym[1] - coords[1])
    assert np.isclose(disp0, disp1, atol=1e-6)

    # Mode 3: "atom" -> only atom 1 moves; its hydrogens (5, 6, 7) stay at their old positions
    new_c_atom = modify_distance(symbols, coords, bonds, 0, 1, 2.10, mode="atom")
    assert np.isclose(calculate_bond_distance(new_c_atom[0], new_c_atom[1]), 2.10, atol=1e-6)
    np.testing.assert_allclose(new_c_atom[5], coords[5], atol=1e-7)


def test_smartslides_modify_angle_and_dihedral():
    """Verify arbitrary 3-atom angle and 4-atom dihedral rotation on Ethanol."""
    zatoms, _ = parse_zmatrix_text(SAMPLE_MOLECULES["Ethanol (C2H5OH)"])
    symbols = [a.symbol for a in zatoms]
    coords = zmatrix_to_cartesian(zatoms)
    bonds = detect_connectivity(symbols, coords)

    # Angle C1(0) - C2(1) - O3(2): change to 125.0 deg while keeping O3-H4(3) bond length intact
    oh_dist_before = calculate_bond_distance(coords[2], coords[3])
    c_ang = modify_angle(symbols, coords, bonds, 0, 1, 2, 125.0, mode="group")
    assert np.isclose(calculate_angle(c_ang[0], c_ang[1], c_ang[2]), 125.0, atol=1e-5)
    assert np.isclose(calculate_bond_distance(c_ang[2], c_ang[3]), oh_dist_before, atol=1e-6)

    # Dihedral C1(0) - C2(1) - O3(2) - H4(3): set to +45.0 deg and -90.0 deg
    c_dih1 = modify_dihedral(symbols, c_ang, bonds, 0, 1, 2, 3, 45.0, mode="group")
    assert np.isclose(calculate_dihedral(c_dih1[0], c_dih1[1], c_dih1[2], c_dih1[3]), 45.0, atol=1e-4)

    c_dih2 = modify_dihedral(symbols, c_dih1, bonds, 0, 1, 2, 3, -90.0, mode="group")
    assert np.isclose(calculate_dihedral(c_dih2[0], c_dih2[1], c_dih2[2], c_dih2[3]), -90.0, atol=1e-4)


def test_invert_chiral_center_reverses_signed_tetrahedral_volume():
    """
    Build a chiral tetrahedral carbon C*(H)(F)(Cl)(CH3) and verify that `invert_chiral_center`
    swaps two substituents to reverse the R/S signed tetrahedral determinant while keeping
    all bond lengths unchanged.
    """
    symbols = ["C", "C", "H", "F", "Cl", "H", "H", "H"]
    # Central chiral carbon at index 0, bonded to C1(1), H(2), F(3), Cl(4)
    coords = np.array([
        [0.000,  0.000,  0.000],  # 0: C* (chiral center)
        [0.000,  0.000,  1.540],  # 1: C (methyl)
        [1.028,  0.000, -0.363],  # 2: H
        [-0.640, 1.108, -0.450],  # 3: F
        [-0.830, -1.438, -0.590], # 4: Cl
        [0.000,  1.028,  1.900],  # 5: H on methyl
        [-0.890, -0.514, 1.900],  # 6: H on methyl
        [0.890, -0.514,  1.900],  # 7: H on methyl
    ], dtype=np.float64)
    bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]

    def signed_volume(pts):
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        v3 = pts[3] - pts[0]
        return float(np.dot(v1, np.cross(v2, v3)))

    vol_before = signed_volume(coords)
    assert abs(vol_before) > 0.5

    inv_coords, msg = invert_chiral_center(symbols, coords, bonds, [0])
    vol_after = signed_volume(inv_coords)
    # R <-> S inversion reverses the sign of the tetrahedral volume!
    assert vol_before * vol_after < 0.0
    # And all C*-X bond distances remain preserved
    for nb in (1, 2, 3, 4):
        d_orig = calculate_bond_distance(coords[0], coords[nb])
        d_inv = calculate_bond_distance(inv_coords[0], inv_coords[nb])
        assert np.isclose(d_orig, d_inv, atol=1e-6)


# ==============================================================================
# Phase 4: Element Coordination & Hybridization Templates Tests
# ==============================================================================

def test_coordination_templates_angles_and_counts():
    """Verify VSEPR / coordination templates (linear, trigonal planar, tetrahedral, square planar, octahedral)."""
    # Linear: 2 vectors at 180 deg
    v_lin = get_coordination_vectors("Be", "linear")
    assert len(v_lin) == 2
    assert np.isclose(np.dot(v_lin[0], v_lin[1]), -1.0, atol=1e-6)

    # Trigonal planar: 3 coplanar vectors at 120 deg
    v_tp = get_coordination_vectors("B", "trigonal_planar")
    assert len(v_tp) == 3
    for i in range(3):
        for j in range(i + 1, 3):
            assert np.isclose(np.dot(v_tp[i], v_tp[j]), -0.5, atol=1e-5)

    # Tetrahedral: 4 vectors with cos(theta) = -1/3
    v_tet = get_coordination_vectors("C", "tetrahedral")
    assert len(v_tet) == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.isclose(np.dot(v_tet[i], v_tet[j]), -1.0 / 3.0, atol=1e-5)

    # Square planar: 4 vectors in XY plane with 90/180 deg angles
    v_sq = get_coordination_vectors("Pt", "square_planar")
    assert len(v_sq) == 4
    for v in v_sq:
        assert np.isclose(v[2], 0.0, atol=1e-6)

    # Octahedral: 6 vectors along +/-X, +/-Y, +/-Z
    v_oct = get_coordination_vectors("Fe", "octahedral")
    assert len(v_oct) == 6
    # If 1 existing bond vector is supplied, 5 remaining open sites are returned
    v_rem = get_coordination_vectors("Fe", "octahedral", existing_bond_vectors=[np.array([1.0, 0.0, 0.0])])
    assert len(v_rem) == 5


# ==============================================================================
# Phase 5: Categorized Fragment Library & Custom User Library Tests
# ==============================================================================

def test_categorized_fragment_library_and_custom_json_persistence(tmp_path):
    """Verify fragment categories and saving/reloading custom user fragments."""
    all_frags = get_fragments_by_category("All Fragments")
    assert len(all_frags) >= 35
    rings = get_fragments_by_category("Rings & Aromatics")
    assert "Phenyl (-C6H5)" in rings
    assert "Naphthyl (-C10H7)" in rings

    custom_json = str(tmp_path / "custom_fragments.json")
    custom_coords = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 4.2],
    ])
    save_custom_fragment(
        "My Test Carbonyl",
        ["C", "O"],
        custom_coords,
        bonds=[(0, 1)],
        bond_orders={(0, 1): 2},
        connect_idx=0,
        path=custom_json,
    )
    assert os.path.isfile(custom_json)
    loaded = load_custom_fragments_into_registry(path=custom_json)
    assert "My Test Carbonyl" in loaded
    assert "My Test Carbonyl" in FRAGMENTS
    assert FRAGMENTS["My Test Carbonyl"]["coords"][0] == (0.0, 0.0, 0.0)
    assert FRAGMENTS["My Test Carbonyl"]["bond_orders"][(0, 1)] == 2


# ==============================================================================
# Phase 6: Frozen-Atom Constraints & 1D Dihedral Conformer Scan Tests
# ==============================================================================

def test_frozen_atoms_remain_stationary_during_optimization():
    """Verify that freezing a subset of atoms locks their exact 3D coordinates during optimization."""
    zatoms, _ = parse_zmatrix_text(SAMPLE_MOLECULES["Ethane (C2H6)"])
    symbols = [a.symbol for a in zatoms]
    coords = zmatrix_to_cartesian(zatoms)
    bonds = detect_connectivity(symbols, coords)

    # Stretch C-C bond to 1.95 A and displace H3
    perturbed = coords.copy()
    perturbed[1] += np.array([0.0, 0.0, 0.42])
    perturbed[2] += np.array([0.15, -0.10, 0.08])

    # Freeze C1 (0) and C2 (1) at their stretched distance (1.95 A) while relaxing all hydrogens
    frozen_set = {0, 1}
    opt_coords, _ = optimize_geometry_force_field(
        symbols, perturbed, user_bonds=bonds, max_iter=120, frozen_indices=frozen_set
    )
    np.testing.assert_allclose(opt_coords[0], perturbed[0], atol=1e-10)
    np.testing.assert_allclose(opt_coords[1], perturbed[1], atol=1e-10)
    # Meanwhile the unfrozen hydrogens relaxed
    assert not np.allclose(opt_coords[2], perturbed[2], atol=1e-3)


def test_1d_dihedral_rotor_scan_finds_staggered_minima_and_eclipsed_barriers():
    """
    Run a 1D Dihedral Rotor Scan on Ethane (H-C-C-H torsion) and verify it identifies
    the staggered conformation as minimum and eclipsed conformation as barrier maximum.
    """
    zatoms, _ = parse_zmatrix_text(SAMPLE_MOLECULES["Ethane (C2H6)"])
    symbols = [a.symbol for a in zatoms]
    coords = zmatrix_to_cartesian(zatoms)
    bonds = detect_connectivity(symbols, coords)

    # Scan dihedral H3(2) - C1(0) - C2(1) - H6(5) over 12 steps (30 deg increments)
    res = run_dihedral_rotor_scan(
        symbols,
        coords,
        bonds,
        atom_indices=(2, 0, 1, 5),
        n_steps=12,
        relax_steps=0,
    )
    assert len(res.points) == 12
    assert res.barrier_kcal > 1.5  # Ethane 3-fold rotational barrier (~2.5-3.0 kcal/mol)
    min_pt = res.points[res.min_index]
    assert np.isclose(min_pt.relative_energy_kcal, 0.0, atol=1e-7)
    # Staggered dihedral angles in ethane are near +/-60 or +/-180 deg
    abs_min_dih = abs(min_pt.dihedral_deg)
    assert any(abs(abs_min_dih - target) < 10.0 for target in (60.0, 180.0))


def test_mainwindow_smartslides_ui_integration(qapp):
    """Verify that multi-selecting 2, 3, or 4 atoms in MainWindow drives SmartSlides live."""
    win = MainWindow()
    win.load_preset_molecule("Ethanol (C2H5OH)")

    # Select 2 non-Z-matrix-adjacent or bonded atoms [0, 2] (C1 and O3)
    win.viewer.highlighted_feature_atoms = {}
    win.viewer.multi_selected = [0, 2]
    win.viewer.selected_atom_idx = 0
    win._update_adjust_panel()

    d_init = calculate_bond_distance(win.coords[0], win.coords[2])
    assert np.isclose(win.spin_bond.value(), d_init, atol=1e-2)

    # Change distance via spin_bond to 2.85 A
    win._on_spin_bond_changed(2.85)
    d_after = calculate_bond_distance(win.coords[0], win.coords[2])
    assert np.isclose(d_after, 2.85, atol=1e-4)


def _find_sample(filename: str) -> str:
    for c in [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename),
        filename,
    ]:
        if os.path.isfile(c):
            return c
    return filename


def test_bp_zmat_inter_ring_dihedral_rotation_without_ring_distortion(qapp):
    """
    Verify that selecting atoms 1, 3, 7, 12 (0-indexed: 0, 2, 6, 11) in bp.zmat
    rotates the phenyl and pyridyl rings around the C3-C7 bridge bond without
    distorting either ring (GaussView SmartSlides behavior), both via 4-atom
    viewport selection and via Z-Matrix row/cell editing.
    """
    from PyQt6 import QtCore
    bp_path = _find_sample("bp.zmat")
    win = MainWindow()
    win.open_path(bp_path)

    c_init = win.coords.copy()
    ring1 = [0, 1, 2, 3, 4, 5, 7, 8, 9, 12]
    ring2 = [6, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20]

    def assert_rings_rigid(coords_after):
        r1_err = max(
            abs(np.linalg.norm(c_init[i] - c_init[j]) - np.linalg.norm(coords_after[i] - coords_after[j]))
            for i in ring1 for j in ring1
        )
        r2_err = max(
            abs(np.linalg.norm(c_init[i] - c_init[j]) - np.linalg.norm(coords_after[i] - coords_after[j]))
            for i in ring2 for j in ring2
        )
        assert r1_err < 1e-6, f"Ring 1 distorted by {r1_err:.2e} Å"
        assert r2_err < 1e-6, f"Ring 2 distorted by {r2_err:.2e} Å"

    # 1. Multi-select atoms 1, 3, 7, 12 in viewport (with selected_atom_idx on the 4th clicked atom, 11)
    win.viewer.selected_atom_idx = 11
    win.viewer.multi_selected = [0, 2, 6, 11]
    win.viewer.smartslide_selection_active = True
    win.viewer.highlighted_feature_atoms = {}
    win._on_viewport_atom_clicked(11, QtCore.Qt.MouseButton.LeftButton)

    assert win.spin_dihedral.isEnabled()
    assert np.isclose(win.spin_dihedral.value(), 16.200, atol=1e-2)

    win.spin_dihedral.setValue(90.0)
    assert np.isclose(calculate_dihedral(win.coords[0], win.coords[2], win.coords[6], win.coords[11]), 90.0, atol=1e-3)
    assert_rings_rigid(win.coords)

    # 2. Reverse selection order (12, 7, 3, 1) -> rotates Ring 1 without distorting either ring
    win.open_path(bp_path)
    win.viewer.selected_atom_idx = 0
    win.viewer.multi_selected = [11, 6, 2, 0]
    win.viewer.smartslide_selection_active = True
    win.viewer.highlighted_feature_atoms = {}
    win._on_viewport_atom_clicked(0, QtCore.Qt.MouseButton.LeftButton)

    win.spin_dihedral.setValue(-60.0)
    assert np.isclose(calculate_dihedral(win.coords[11], win.coords[6], win.coords[2], win.coords[0]), -60.0, atol=1e-3)
    assert_rings_rigid(win.coords)

    # 3. Symmetric mode (50/50 counter-rotation of both rings)
    win.open_path(bp_path)
    win.combo_smart_mode.setCurrentIndex(1)  # "symmetric"
    win.viewer.selected_atom_idx = 11
    win.viewer.multi_selected = [0, 2, 6, 11]
    win.viewer.smartslide_selection_active = True
    win.viewer.highlighted_feature_atoms = {}
    win._update_adjust_panel()
    win.spin_dihedral.setValue(45.0)
    assert np.isclose(calculate_dihedral(win.coords[0], win.coords[2], win.coords[6], win.coords[11]), 45.0, atol=1e-3)
    assert_rings_rigid(win.coords)

    # 4. Selecting Row 12 (atom 12) in the Z-Matrix table and editing its dihedral via slider or table cell
    win.open_path(bp_path)
    win.combo_smart_mode.setCurrentIndex(0)  # "group"
    win._on_table_atom_selected(11)
    win._on_table_cell_selected(11, win.table_widget.COL_DIH)
    win.spin_dihedral.setValue(120.0)
    assert np.isclose(calculate_dihedral(win.coords[0], win.coords[2], win.coords[6], win.coords[11]), 120.0, atol=1e-3)
    assert_rings_rigid(win.coords)

    win.table_widget.item(11, win.table_widget.COL_DIH).setText("-30.0")
    assert np.isclose(calculate_dihedral(win.coords[0], win.coords[2], win.coords[6], win.coords[11]), -30.0, atol=1e-3)
    assert_rings_rigid(win.coords)


def test_oh_zmat_octahedral_detection_and_impose_symmetry_co_replacement(qapp):
    """
    Verify that:
    1. oh.zmat (CrH6) is accurately detected as octahedral (Oh, order 48) with correct
       irreducible representations (Gamma_vib = A1g + Eg + 2T1u + T2g + T2u).
    2. Imposing Oh symmetry and replacing a single H atom by CO (via 'Attach at Selected Atom',
       double-clicking an H atom, or typing 'CO' in the Z-Matrix Sym cell) automatically
       replaces all 6 symmetry-equivalent H atoms with CO ligands, preserving Oh symmetry (Cr(CO)6).
    3. Element substitution and bond-length adjustments under imposed symmetry also replicate
       across all symmetry-equivalent atoms while preserving Oh symmetry.
    """
    from agymolden.symmetry import find_symmetry_operations, calculate_symmetry_representations, get_symmetry_orbits

    oh_path = _find_sample("oh.zmat")
    win = MainWindow()
    win.open_path(oh_path)

    symbols = [a.symbol for a in win.zatoms]
    res = find_symmetry_operations(symbols, win.coords)
    assert res.symbol == "Oh"
    assert res.order == 48

    reps = calculate_symmetry_representations(symbols, win.coords, res)
    assert reps["gamma_vib_dict"] == {"A1g": 1, "Eg": 1, "T1u": 2, "T2g": 1, "T2u": 1}

    orbits = get_symmetry_orbits(symbols, win.coords, res)
    assert orbits == [[0], [1, 2, 3, 4, 5, 6]]

    # 1. Impose Oh symmetry and attach 'Carbonyl Ligand (-CO)' to a single H atom (index 1)
    win._on_detect_symmetry()
    win.set_impose_symmetry(True)
    assert win.symmetry_imposed is True
    assert win.imposed_sym_res.symbol == "Oh"

    win.viewer.selected_atom_idx = 1
    win.combo_fragment.setCurrentText("Carbonyl Ligand (-CO)")
    win._on_attach_fragment()

    syms_co = [a.symbol for a in win.zatoms]
    assert syms_co.count("Cr") == 1
    assert syms_co.count("C") == 6
    assert syms_co.count("O") == 6
    assert syms_co.count("H") == 0
    res_co = find_symmetry_operations(syms_co, win.coords)
    assert res_co.symbol == "Oh"
    assert res_co.order == 48

    # 2. Double-clicking one H atom with Carbonyl Ligand (-CO) active under imposed Oh symmetry
    win.open_path(oh_path)
    win.set_impose_symmetry(True)
    win.combo_fragment.setCurrentText("Carbonyl Ligand (-CO)")
    win._on_fragment_combo_changed("Carbonyl Ligand (-CO)")
    win._on_atom_double_clicked(3)

    syms_dbl = [a.symbol for a in win.zatoms]
    assert syms_dbl.count("Cr") == 1
    assert syms_dbl.count("C") == 6
    assert syms_dbl.count("O") == 6
    assert find_symmetry_operations(syms_dbl, win.coords).symbol == "Oh"

    # 3. Typing 'CO' into a single H atom's Sym cell in the Z-Matrix table under imposed Oh symmetry
    win.open_path(oh_path)
    win.set_impose_symmetry(True)
    win.table_widget.item(1, win.table_widget.COL_SYM).setText("CO")

    syms_tbl = [a.symbol for a in win.zatoms]
    assert syms_tbl.count("Cr") == 1
    assert syms_tbl.count("C") == 6
    assert syms_tbl.count("O") == 6
    assert find_symmetry_operations(syms_tbl, win.coords).symbol == "Oh"

    # 4. Replacing one H atom with F and adjusting one Cr-F bond length under imposed Oh symmetry
    win.open_path(oh_path)
    win.set_impose_symmetry(True)
    win.table_widget.item(1, win.table_widget.COL_SYM).setText("F")
    syms_f = [a.symbol for a in win.zatoms]
    assert syms_f == ["Cr", "F", "F", "F", "F", "F", "F"]
    assert find_symmetry_operations(syms_f, win.coords).symbol == "Oh"

    win.viewer.selected_atom_idx = 1
    win._on_spin_bond_changed(1.95)
    cr_f_dists = [np.linalg.norm(win.coords[i] - win.coords[0]) for i in range(1, 7)]
    assert np.allclose(cr_f_dists, 1.95, atol=1e-4)
    assert find_symmetry_operations(syms_f, win.coords).symbol == "Oh"


def test_choose_lower_and_higher_symmetry_subgroups_and_supergroups(qapp):
    """
    Verify that AgyMolden discovers and lets the user select both:
    1. Lower symmetries (subgroups) of the detected point group:
       - For oh.zmat (Oh, order 48), lists Oh (detected) and lower subgroups including
         O, Td, Th, D4h, D3d, D2d, C4v, C3v, D2h, C2v, Cs, Ci, C1.
       - Selecting D4h (order 16) and imposing symmetry splits the 6 H atoms into 4 equatorial
         and 2 trans-axial H atoms: replacing one axial H (atom #4, index 3) with CO replaces
         ONLY the 2 trans-axial H atoms to form trans-CrH4(CO)2 with exact D4h symmetry!
       - Selecting C4v (order 8) and imposing symmetry makes each axial H a singleton orbit:
         replacing one axial H (index 3) with CO replaces ONLY that 1 H atom to form CrH5(CO)
         with exact C4v symmetry!
    2. Higher symmetries (approximate supergroups at looser tolerance):
       - When oh.zmat is slightly distorted (e.g. one Cr-H bond stretched by +0.25 Å so exact
         symmetry at tol=0.15 Å is C4v), find_available_symmetries lists C4v as 'detected',
         higher supergroups (Oh, Th, Td, O, D4h, D3d) as 'higher', and subgroups (C4, C2v, Cs, C1)
         as 'lower'.
       - Selecting the higher 'Oh' symmetry and enforcing/imposing it snaps the distorted
         geometry back into exact Oh symmetry (order 48)!
    """
    from agymolden.symmetry import (
        find_symmetry_operations,
        find_available_symmetries,
        get_symmetry_orbits,
    )

    oh_path = _find_sample("oh.zmat")
    win = MainWindow()
    win.open_path(oh_path)

    avail = find_available_symmetries([a.symbol for a in win.zatoms], win.coords, base_tolerance=0.15)
    avail_dict = {sym: (res, tag) for sym, res, tag in avail}

    assert avail_dict["Oh"][1] == "detected"
    assert avail_dict["Oh"][0].order == 48
    for sub in ("O", "Td", "Th", "D4h", "D3d", "D2d", "C4v", "C3v", "D2h", "C2v", "Cs", "Ci", "C1"):
        assert sub in avail_dict, f"Expected subgroup {sub} in available symmetries"
        assert avail_dict[sub][1] == "lower"

    # Verify clicking the toolbar '🔮 Symmetry' button and '🔮 Detect Point Group Symmetry' button
    # emits clicked(False) and properly switches to the Symmetry tab (index 4)
    win.tabs.setCurrentIndex(0)
    assert win.tabs.currentIndex() == 0
    win.btn_toolbar_sym.click()
    assert win.tabs.currentIndex() == 4
    assert win._current_sym_res.symbol == "Oh"

    win.tabs.setCurrentIndex(0)
    win.btn_detect_symmetry.click()
    assert win.tabs.currentIndex() == 4
    assert win._current_sym_res.symbol == "Oh"

    # Check combo box in UI is populated with all available groups
    assert win.combo_sym_group.count() == len(avail)
    assert win.combo_view_sym_group.count() == len(avail)

    # 1. Step lower button / select lower subgroup D4h and impose symmetry -> trans-CrH4(CO)2
    win.set_impose_symmetry(True)
    win.select_symmetry_group("D4h")
    assert win._current_sym_res.symbol == "D4h"
    assert win._current_sym_res.order == 16
    assert win.imposed_sym_res.symbol == "D4h"

    orbits_d4h = get_symmetry_orbits([a.symbol for a in win.zatoms], win.coords, win.imposed_sym_res)
    assert [0] in orbits_d4h
    assert [3, 4] in orbits_d4h
    assert [1, 2, 5, 6] in orbits_d4h

    # Replace axial H atom (index 3) with CO under imposed D4h symmetry -> only 2 trans H atoms replaced!
    win.table_widget.item(3, win.table_widget.COL_SYM).setText("CO")
    syms_trans = [a.symbol for a in win.zatoms]
    assert syms_trans.count("Cr") == 1
    assert syms_trans.count("H") == 4
    assert syms_trans.count("C") == 2
    assert syms_trans.count("O") == 2
    res_trans = find_symmetry_operations(syms_trans, win.coords)
    assert res_trans.symbol == "D4h"
    assert res_trans.order == 16

    # 2. Select lower subgroup C4v on oh.zmat and impose symmetry -> mono-substituted CrH5(CO)
    win.open_path(oh_path)
    win.set_impose_symmetry(True)
    win.select_symmetry_group("C4v")
    assert win.imposed_sym_res.symbol == "C4v"
    assert win.imposed_sym_res.order == 8

    # Under C4v, axial H atom index 3 is in its own singleton orbit [3]
    win.table_widget.item(3, win.table_widget.COL_SYM).setText("CO")
    syms_mono = [a.symbol for a in win.zatoms]
    assert syms_mono.count("Cr") == 1
    assert syms_mono.count("H") == 5
    assert syms_mono.count("C") == 1
    assert syms_mono.count("O") == 1
    res_mono = find_symmetry_operations(syms_mono, win.coords)
    assert res_mono.symbol == "C4v"
    assert res_mono.order == 8

    # 3. Distort oh.zmat (stretch atom #4 Cr-H bond by +0.25 Å so exact symmetry at 0.15 Å is C4v)
    win.open_path(oh_path)
    win.set_impose_symmetry(False)
    win.zatoms[3].dist += 0.25
    win.set_molecule_from_zatoms(win.zatoms)
    win._on_detect_symmetry(switch_tab=False)

    assert win._current_sym_res.symbol == "C4v"
    avail_dist = {sym: (res, tag) for sym, res, tag in win._available_symmetries}
    assert avail_dist["C4v"][1] == "detected"
    assert avail_dist["Oh"][1] == "higher"
    assert avail_dist["D4h"][1] == "higher"
    assert avail_dist["C2v"][1] == "lower"
    assert avail_dist["Cs"][1] == "lower"

    # Step up using _on_step_higher_symmetry() or select 'Oh' and click Enforce Symmetry
    win._on_step_higher_symmetry()
    assert win._current_sym_res.order > 8

    # Select higher 'Oh' group and enforce symmetry -> coordinates snap to exact Oh symmetry!
    win.select_symmetry_group("Oh", enforce_if_higher=False)
    assert win._current_sym_res.symbol == "Oh"
    win._on_enforce_symmetry()

    res_snapped = find_symmetry_operations([a.symbol for a in win.zatoms], win.coords, tolerance=0.05)
    assert res_snapped.symbol == "Oh"
    assert res_snapped.order == 48
    cr_h_dists = [np.linalg.norm(win.coords[i] - win.coords[0]) for i in range(1, 7)]
    assert np.max(cr_h_dists) - np.min(cr_h_dists) < 1e-6





def test_optimize_geometry_keeps_imposed_symmetry(qapp):
    """Optimize with Impose Symmetry active keeps the point group; falls back when impossible."""
    from agymolden.symmetry import (
        find_symmetry_operations,
        optimize_with_imposed_symmetry,
    )
    from agymolden.forcefield import optimize_geometry_force_field, detect_connectivity

    # 1. Direct unit test of optimize_with_imposed_symmetry on a perturbed benzene molecule
    s_benzene = ["C"] * 6 + ["H"] * 6
    theta = np.arange(6) * np.pi / 3.0
    c_benzene = np.array(
        [[1.40 * np.cos(a), 1.40 * np.sin(a), 0.0] for a in theta]
        + [[2.48 * np.cos(a), 2.48 * np.sin(a), 0.0] for a in theta]
    )
    sym_d6h = find_symmetry_operations(s_benzene, c_benzene, tolerance=0.10)
    assert sym_d6h.symbol == "D6h"

    # Perturb coordinates with small random noise
    rng = np.random.default_rng(42)
    c_noisy = c_benzene + rng.normal(0.0, 0.03, c_benzene.shape)
    # At tight tolerance, noise breaks symmetry
    assert find_symmetry_operations(s_benzene, c_noisy, tolerance=0.01).symbol != "D6h"

    bonds = detect_connectivity(s_benzene, c_benzene)
    opt_fn = lambda c: optimize_geometry_force_field(s_benzene, c, user_bonds=bonds, max_iter=60)
    c_opt, extra, msg = optimize_with_imposed_symmetry(opt_fn, s_benzene, c_noisy, sym_d6h, cycles=5)
    assert c_opt is not None
    assert msg.startswith("Optimized with D6h symmetry imposed")
    assert find_symmetry_operations(s_benzene, c_opt, tolerance=0.01).symbol == "D6h"

    # Perturb coordinates beyond tolerance (> 0.5 A) -> should cleanly reject and report reason
    c_excessive = c_benzene + rng.normal(0.0, 0.60, c_benzene.shape)
    c_opt_bad, _, msg_bad = optimize_with_imposed_symmetry(opt_fn, s_benzene, c_excessive, sym_d6h, cycles=5)
    assert c_opt_bad is None
    assert "too far from D6h symmetry" in msg_bad

    # 2. End-to-end integration test with MainWindow
    oh_path = _find_sample("oh.zmat")
    win = MainWindow()
    win.open_path(oh_path)
    win._on_detect_symmetry()
    win.set_impose_symmetry(True)
    win._on_optimize_geometry()
    # background worker: wait for completion
    for _ in range(300):
        QtWidgets_app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication
        QtWidgets_app.processEvents()
        import time; time.sleep(0.05)
        if "converged" in win.statusBar().currentMessage():
            break
    syms = [a.symbol for a in win.zatoms]
    assert find_symmetry_operations(syms, win.coords, tolerance=0.05).symbol == "Oh"
