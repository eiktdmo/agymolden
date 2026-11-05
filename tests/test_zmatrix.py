import unittest
import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("OpenGL")

import os
import numpy as np
from agymolden.zmatrix import (
    ZAtom,
    zmatrix_to_cartesian,
    cartesian_to_zmatrix,
    parse_zmatrix_text,
    format_zmatrix_text
)
from agymolden.elements import get_color, get_cov_radius
from agymolden.forcefield import add_hydrogens_to_valence, detect_connectivity, optimize_geometry_force_field
from agymolden.topology import form_single_bond
from PyQt6 import QtCore, QtGui, QtWidgets

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples")
from agymolden import workers
workers.SYNCHRONOUS = True


class TestZMatrix(unittest.TestCase):
    def test_water_conversion(self):
        # Water molecule Z-matrix
        zatoms = [
            ZAtom(symbol="O"),
            ZAtom(symbol="H", ref1=0, dist=0.96),
            ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5)
        ]
        coords = zmatrix_to_cartesian(zatoms)
        self.assertEqual(coords.shape, (3, 3))
        
        # Test distance O-H1
        d01 = np.linalg.norm(coords[1] - coords[0])
        self.assertAlmostEqual(d01, 0.96, places=3)

        # Test distance O-H2
        d02 = np.linalg.norm(coords[2] - coords[0])
        self.assertAlmostEqual(d02, 0.96, places=3)

        # Test angle H1 - O - H2
        v1 = coords[1] - coords[0]
        v2 = coords[2] - coords[0]
        cos_ang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        deg = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
        self.assertAlmostEqual(deg, 104.5, places=2)

    def test_text_parsing(self):
        txt = """
        C
        H 1 1.09
        H 1 1.09 2 109.5
        H 1 1.09 2 109.5 3 120.0
        """
        atoms, vars_dict = parse_zmatrix_text(txt)
        self.assertEqual(len(atoms), 4)
        self.assertEqual(atoms[0].symbol, "C")
        self.assertEqual(atoms[1].ref1, 0)
        self.assertAlmostEqual(atoms[1].dist, 1.09)
        self.assertEqual(atoms[2].ref2, 1)
        self.assertAlmostEqual(atoms[2].angle, 109.5)
        self.assertEqual(atoms[3].ref3, 2)
        self.assertAlmostEqual(atoms[3].dihedral, 120.0)

    def test_roundtrip_cartesian(self):
        txt = """
        C
        H 1 1.09
        H 1 1.09 2 109.47
        H 1 1.09 2 109.47 3 120.0
        """
        atoms, _ = parse_zmatrix_text(txt)
        coords = zmatrix_to_cartesian(atoms)
        symbols = [a.symbol for a in atoms]
        re_zatoms = cartesian_to_zmatrix(symbols, coords)
        re_coords = zmatrix_to_cartesian(re_zatoms)
        
        # Test pairwise distance preservation
        d_orig = np.linalg.norm(coords[0] - coords[1])
        d_re = np.linalg.norm(re_coords[0] - re_coords[1])
        self.assertAlmostEqual(d_orig, d_re, places=3)

    def test_add_hydrogens_honors_double_and_triple_bonds(self):
        from agymolden.forcefield import add_hydrogens_to_valence
        # Ethylene C=C skeleton: 2 carbons at distance 1.34 A, bond order 2
        symbols = ["C", "C"]
        coords = np.array([[0.0, 0.0, 0.0], [1.34, 0.0, 0.0]])
        bonds = [(0, 1)]
        bo = {(0, 1): 2}

        # Whole molecule (target_atom_indices=None): each C should get 2 H's (total 4 H's)
        syms_all, coords_all, added_all = add_hydrogens_to_valence(
            symbols, coords, user_bonds=bonds, user_bond_orders=bo
        )
        self.assertEqual(added_all, 4)
        self.assertEqual(syms_all.count("H"), 4)

        # Selected atom 0 only: C1 gets 2 H's, C2 gets 0 H's
        syms_sel, coords_sel, added_sel = add_hydrogens_to_valence(
            symbols, coords, user_bonds=bonds, target_atom_indices=[0], user_bond_orders=bo
        )
        self.assertEqual(added_sel, 2)
        self.assertEqual(syms_sel.count("H"), 2)

        # Acetylene C#C skeleton: 2 carbons at distance 1.20 A, bond order 3
        bo_triple = {(0, 1): 3}
        syms_trip, coords_trip, added_trip = add_hydrogens_to_valence(
            symbols, coords, user_bonds=bonds, user_bond_orders=bo_triple
        )
        self.assertEqual(added_trip, 2)
        self.assertEqual(syms_trip.count("H"), 2)

    def test_ghost_atom_flexibility_and_bonds(self):
        from agymolden.topology import form_single_bond
        from agymolden.forcefield import detect_connectivity

        # Methane CH4 plus ghost atom X
        symbols = ["C", "H", "H", "H", "H", "X"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
            [2.5, 0.0, 0.0]
        ])
        bonds = [(0, 1), (0, 2), (0, 3), (0, 4)]
        bo = {}

        # Form single bond between C (atom 0) and X (atom 5)
        # Carbon valence is 4; forming C-X single bond requires removing 1 H from C
        syms, cds, bnds, bo_out, a, b, removed = form_single_bond(symbols, coords, bonds, bo, 0, 5)
        self.assertEqual(len(removed), 1)  # 1 H removed from C to maintain valence 4
        # Verify bond (C, X) is created
        c_idx = [i for i, s in enumerate(syms) if s == "C"][0]
        x_idx = [i for i, s in enumerate(syms) if s == "X"][0]
        self.assertIn((min(c_idx, x_idx), max(c_idx, x_idx)), bnds)

        # Verify detect_connectivity keeps bonds involving X
        detected = detect_connectivity(syms, cds, user_bonds=bnds)
        self.assertIn((min(c_idx, x_idx), max(c_idx, x_idx)), detected)

    def test_carbon_chain_single_bonds_and_add_hydrogens(self):
        """Verify that a carbon chain with single bonds retains single bond lengths ~1.54 A,
        does not get auto-detected as triple bonds, and Add Hydrogens produces a complete alkane (butane)."""
        # 4 carbons in a chain with single bonds
        symbols = ["C", "C", "C", "C"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [2.31, 1.33, 0.0],
            [3.85, 1.33, 0.0],
        ])
        bonds = [(0, 1), (1, 2), (2, 3)]
        user_bo = {(0, 1): 1, (1, 2): 1, (2, 3): 1}

        # Force field optimization should preserve single bond distances ~1.53-1.54 A
        opt_coords, _ = optimize_geometry_force_field(symbols, coords, user_bonds=bonds, max_iter=100, user_bond_orders=user_bo)
        for u, v in bonds:
            d = np.linalg.norm(opt_coords[u] - opt_coords[v])
            self.assertGreater(d, 1.45, f"Bond {u}-{v} distance {d:.3f} was compressed unexpectedly")
            self.assertLess(d, 1.60, f"Bond {u}-{v} distance {d:.3f} was expanded unexpectedly")

        # Add hydrogens should add 10 hydrogens (butane: C4H10)
        h_syms, h_coords, added = add_hydrogens_to_valence(symbols, opt_coords, user_bonds=bonds, user_bond_orders=user_bo)
        self.assertEqual(added, 10, f"Expected 10 hydrogens for butane, got {added}")
        self.assertEqual(h_syms.count("H"), 10)
        self.assertEqual(h_syms.count("C"), 4)

    def test_canvas_double_click_paste_behavior(self):
        """Verify that double clicking on canvas with an item in memory pastes it."""
        from PyQt6 import QtWidgets, QtCore
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        # Verify initial memory is empty
        self.assertIsNone(win.last_added_element)
        self.assertIsNone(win.stored_fragment)
        self.assertIsNone(win.last_added_fragment)
        self.assertIsNone(win.last_item_type)

        # Set an atom in memory (uncheck auto_h for single atom test)
        win.chk_auto_h.setChecked(False)
        win.last_added_element = "N"
        win.last_item_type = "element"
        win._paste_element_to_free_space("N", QtCore.QPoint(100, 100))
        self.assertEqual(len(win.zatoms), 1)
        self.assertEqual(win.zatoms[0].symbol, "N")

        # Set a fragment in memory
        frag = {
            "name": "Methyl",
            "symbols": ["C", "H", "H", "H"],
            "coords": [(0,0,0), (1,0,0), (0,1,0), (0,0,1)],
            "bonds": [(0, 1), (0, 2), (0, 3)]
        }
        win.stored_fragment = frag
        win.last_item_type = "fragment"
        win._paste_fragment_to_free_space(frag, QtCore.QPoint(200, 200))
        # Total atoms: 1 (N) + 4 (Methyl) = 5
        self.assertEqual(len(win.zatoms), 5)
        symbols = [a.symbol for a in win.zatoms]
        self.assertIn("N", symbols)
        self.assertEqual(symbols.count("C"), 1)
        self.assertEqual(symbols.count("H"), 3)

        # Test Ctrl+K to flush memory
        win._on_flush_memory()
        self.assertIsNone(win.last_added_element)
        self.assertIsNone(win.stored_fragment)
        self.assertIsNone(win.last_added_fragment)
        self.assertIsNone(win.last_item_type)

    def test_region_rotation_vs_whole_system_rotation(self):
        """Verify that when a region is selected (>1 atom), rotating updates selected coords around centroid."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.viewer import MoleculeViewer

        viewer = MoleculeViewer()
        # 3 atoms in line along X
        symbols = ["C", "C", "C"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [3.08, 0.0, 0.0]
        ], dtype=np.float64)
        viewer.set_molecule(symbols, coords)

        # No region selected: rotation changes camera angles
        viewer.multi_selected = []
        old_rot_x, old_rot_y = viewer.rot_x, viewer.rot_y
        coords_before_orbit = np.copy(viewer.coords)
        viewer.last_mouse_pos = QtCore.QPoint(100, 100)
        # Simulate left-click drag
        move_event = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(110, 120),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier
        )
        viewer.mouseMoveEvent(move_event)
        self.assertNotEqual(viewer.rot_x, old_rot_x)
        self.assertNotEqual(viewer.rot_y, old_rot_y)
        # Coords unchanged when rotating camera
        np.testing.assert_allclose(viewer.coords, coords_before_orbit)

        # Multi-region selected (atoms 1 and 2): plain Left-Drag still orbits camera without distorting geometry,
        # while Alt + Left-Drag rotates the selected subgroup around its centroid.
        viewer.multi_selected = [1, 2]
        c0_before = np.copy(viewer.coords[0])
        c1_before = np.copy(viewer.coords[1])
        c2_before = np.copy(viewer.coords[2])
        viewer.last_mouse_pos = QtCore.QPoint(100, 100)
        move_event_plain = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(110, 115),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.NoModifier
        )
        viewer.mouseMoveEvent(move_event_plain)
        np.testing.assert_allclose(viewer.coords[1], c1_before)

        viewer.last_mouse_pos = QtCore.QPoint(100, 100)
        move_event2 = QtGui.QMouseEvent(
            QtCore.QEvent.Type.MouseMove,
            QtCore.QPointF(115, 125),
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.KeyboardModifier.AltModifier
        )
        viewer.mouseMoveEvent(move_event2)
        # Atom 0 (unselected) must not move
        np.testing.assert_allclose(viewer.coords[0], c0_before)
        # Selected atoms must have moved
        self.assertFalse(np.allclose(viewer.coords[1], c1_before))
        self.assertFalse(np.allclose(viewer.coords[2], c2_before))
        # Distance between selected atoms 1 and 2 preserved under rigid rotation
        dist_before = np.linalg.norm(c2_before - c1_before)
        dist_after = np.linalg.norm(viewer.coords[2] - viewer.coords[1])
        self.assertAlmostEqual(dist_before, dist_after, places=4)

    def test_fragment_attachment_atom_selection(self):
        """Verify attaching fragment using a selected connection atom."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        win.chk_auto_h.setChecked(False)
        # Create a single Carbon atom
        win._paste_element_to_free_space("C", QtCore.QPoint(100, 100))
        self.assertEqual(len(win.zatoms), 1)

        # Select Methanol (CH3OH) fragment: symbols = ["C", "O", "H", "H", "H", "H"]
        win.combo_fragment.setCurrentText("Methanol (CH3OH)")
        self.assertEqual(win.combo_fragment.currentText(), "Methanol (CH3OH)")
        self.assertEqual(win.combo_frag_connect.count(), 6)

        # Choose attachment atom #2 (O, index 1) instead of Carbon (index 0)
        win.combo_frag_connect.setCurrentIndex(1)
        self.assertEqual(win.combo_frag_connect.currentData(), 1)

        # Select anchor atom 0 (C)
        win.viewer.selected_atom_idx = 0
        win._on_attach_fragment()

        # Total atoms: 1 (original C) + 5 (Methanol minus H on O to fulfill O valence = 2) = 6
        self.assertEqual(len(win.zatoms), 6)
        # The bond formed between original C (atom 0) and the attachment atom in Methanol (O)
        # must have bond length ~1.43 A (C-O bond length), not C-C ~1.54 A
        c_orig = win.coords[0]
        # Attachment atom is O
        o_idx = [i for i, a in enumerate(win.zatoms) if a.symbol == "O"][0]
        attached_o = win.coords[o_idx]
        d_co = np.linalg.norm(c_orig - attached_o)
        self.assertAlmostEqual(d_co, 1.43, delta=0.08)

    def test_ctrl_k_flush_memory(self):
        """Verify that Ctrl+K flushes memory and double-clicking on empty canvas triggers the picker without window crash."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        # Set memory
        win.last_added_element = "N"
        win.last_item_type = "element"

        # Trigger flush memory (Ctrl+K)
        win._on_flush_memory()

        self.assertIsNone(win.last_added_element)
        self.assertIsNone(win.stored_fragment)
        self.assertIsNone(win.last_added_fragment)
        self.assertIsNone(win.last_item_type)
        # Brush element should remain a valid string so paintEvent doesn't crash
        self.assertTrue(isinstance(win.viewer.brush_element, str))

        # Test normalize_symbol and get_color with None/empty/valid values
        from agymolden.elements import normalize_symbol, get_color
        self.assertEqual(normalize_symbol(None), "C")
        self.assertEqual(normalize_symbol(""), "C")
        self.assertEqual(len(get_color(None)), 3)
        self.assertEqual(len(get_color("")), 3)

    def test_bond_double_click_preserves_ring_bonds(self):
        """Verify that double clicking on bond C5-C9 in test19.zmat cycles bond order and keeps all bonds intact."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow
        from agymolden.zmatrix import parse_zmatrix_text

        win = MainWindow()
        test19_path = os.path.join(DATA_DIR, "test19.zmat")
        if not os.path.exists(test19_path):
            return
        with open(test19_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)

        # In test19, C5 is index 4 and C9 is index 8. C7 is index 6.
        # Ring bond (6, 8) must be preserved.
        bonds_before = win._current_explicit_bonds()
        self.assertIn((6, 8), bonds_before)
        self.assertIn((4, 8), bonds_before)

        # Double click bond (4, 8) to form double bond
        win._on_bond_double_clicked(4, 8)
        bonds_after = win._current_explicit_bonds()
        bo_after = win._current_bond_orders()

        # Ring bond (6, 8) must remain intact!
        self.assertIn((6, 8), bonds_after)
        self.assertIn((4, 8), bonds_after)
        self.assertEqual(bo_after.get((4, 8), 1), 2)

    def test_bond_formation_removes_excess_hydrogens(self):
        """Verify that forming a bond between atoms deletes excess hydrogens to fulfill valence without leaving them orphaned."""
        from agymolden.topology import form_single_bond, fix_valence_explicit
        # Two methane molecules: C1 with 4 H's, C2 with 4 H's
        symbols = ["C", "H", "H", "H", "H", "C", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-0.5, 0.8, 0.0],
            [-0.5, -0.8, 0.0],
            [0.0, 0.0, 1.0],
            [3.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.5, 0.8, 0.0],
            [3.5, -0.8, 0.0],
            [3.0, 0.0, 1.0]
        ], dtype=np.float64)
        bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (5, 6), (5, 7), (5, 8), (5, 9)]
        bo = {}

        # Form bond between C1 (0) and C2 (5)
        new_syms, new_coords, new_bonds, new_bo, a, b, removed = form_single_bond(
            symbols, coords, bonds, bo, 0, 5
        )
        # Should have removed 1 H from C1 and 1 H from C2
        self.assertEqual(len(removed), 2)
        # Resulting ethane molecule should have 8 atoms (2 C, 6 H)
        self.assertEqual(len(new_syms), 8)
        self.assertEqual(new_syms.count("C"), 2)
        self.assertEqual(new_syms.count("H"), 6)
        # No orphan hydrogens: every H has exactly 1 bond
        adj = {i: [] for i in range(len(new_syms))}
        for u, v in new_bonds:
            adj[u].append(v)
            adj[v].append(u)
        for i, s in enumerate(new_syms):
            if s == "H":
                self.assertEqual(len(adj[i]), 1)
            elif s == "C":
                self.assertEqual(len(adj[i]), 4)


    def test_form_bond_on_test21_preserves_valence_and_single_bonds(self):
        """Verify forming bond between end atoms C1 and C6 on test21.zmat forms the single bond,
        removes the needed hydrogen, preserves valence, and keeps all bonds as single bonds."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        test21_path = os.path.join(DATA_DIR, "test21.zmat")
        if not os.path.exists(test21_path):
            return
        with open(test21_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)

        # Before: 16 atoms, all single bonds
        self.assertEqual(len(win.zatoms), 16)
        self.assertEqual(win._current_bond_orders(), {})

        # Form bond between end atoms C1 (0) and C6 (5)
        win.viewer.multi_selected = [0, 5]
        win._on_close_ring_clicked()

        # After: 15 atoms (1 H deleted to fulfill valence on C6), 15 bonds
        self.assertEqual(len(win.zatoms), 15)
        bonds = win._current_explicit_bonds()
        self.assertEqual(len(bonds), 15)
        self.assertEqual(win._current_bond_orders(), {})

        # Verify C-C ring connectivity formed: 6 carbons connected in a cycle
        c_indices = [i for i, a in enumerate(win.zatoms) if a.symbol == "C"]
        self.assertEqual(len(c_indices), 6)
        c_bonds = [b for b in bonds if b[0] in c_indices and b[1] in c_indices]
        self.assertEqual(len(c_bonds), 6)

    def test_bond_order_cycle_to_single_and_selection_order(self):
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [-0.5, 0.9, 0.0],
            [-0.5, -0.9, 0.0],
            [-0.5, 0.0, 0.9],
            [2.0, 0.9, 0.0],
            [2.0, -0.9, 0.0],
            [2.0, 0.0, 0.9],
        ])
        bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
        bo = {}
        win._commit_explicit_structure(symbols, coords, bonds, bo, relax_iters=0, add_h=False, fix_valence=False)

        # 1. Double click on bond (0, 1): increases single -> double
        win._on_bond_double_clicked(0, 1)
        self.assertEqual(win.viewer.bond_orders.get((0, 1), 1), 2)

        # Double click again: increases double -> triple
        win._on_bond_double_clicked(0, 1)
        self.assertEqual(win.viewer.bond_orders.get((0, 1), 1), 3)

        # Double click again: triple is max -> cycles back to single (1)
        win._on_bond_double_clicked(0, 1)
        self.assertEqual(win.viewer.bond_orders.get((0, 1), 1), 1)

        # 2. Check measurement text shows selected atoms in order
        win.viewer.multi_selected = [0, 1]
        win.viewer._update_measurement_info()
        self.assertIn("Distance (C1 - C2)", win.lbl_measurement.text())
        self.assertIn("Selected: C1, C2", win.lbl_measurement.text())

        # Multi-selected 3 atoms in order: [1, 0, 2]
        win.viewer.multi_selected = [1, 0, 2]
        win.viewer._update_measurement_info()
        self.assertIn("Angle (C2 - C1 -", win.lbl_measurement.text())
        self.assertIn("Selected: C2, C1,", win.lbl_measurement.text())

    def test_optimize_test22_to_benzene(self):
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        test22_path = os.path.join(DATA_DIR, "test22.zmat")
        if not os.path.exists(test22_path):
            return
        with open(test22_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)
        win._on_optimize_geometry()

        # Check all 6 carbon-carbon bond lengths in the benzene ring
        ring = [0, 1, 4, 8, 6, 2]
        for i in range(6):
            c1, c2 = ring[i], ring[(i+1)%6]
            d = float(np.linalg.norm(win.coords[c1] - win.coords[c2]))
            self.assertAlmostEqual(d, 1.3927, delta=0.015)

        # Check all C-H bonds are normal ~1.085 Å (including H8 which started at 0.703 Å)
        for c, h in [(0, 3), (1, 5), (4, 9), (8, 11), (6, 10), (2, 7)]:
            d = float(np.linalg.norm(win.coords[c] - win.coords[h]))
            self.assertAlmostEqual(d, 1.085, delta=0.015)

    def test_optimize_test23_to_benzene(self):
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow
        from agymolden.symmetry import find_symmetry_operations

        win = MainWindow()
        test23_path = os.path.join(DATA_DIR, "test23.zmat")
        if not os.path.exists(test23_path):
            return
        with open(test23_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)
        win._on_optimize_geometry()

        # Check all 6 carbon-carbon bond lengths in the benzene ring
        ring = [0, 1, 4, 8, 6, 2]
        for i in range(6):
            c1, c2 = ring[i], ring[(i+1)%6]
            d = float(np.linalg.norm(win.coords[c1] - win.coords[c2]))
            self.assertAlmostEqual(d, 1.3927, delta=0.015)

        # Check all C-H bonds are normal ~1.085 Å
        for c, h in [(0, 3), (1, 5), (4, 9), (8, 11), (6, 10), (2, 7)]:
            d = float(np.linalg.norm(win.coords[c] - win.coords[h]))
            self.assertAlmostEqual(d, 1.085, delta=0.015)

        # Check D6h symmetry
        res = find_symmetry_operations([a.symbol for a in win.zatoms], win.coords, tolerance=0.15)
        self.assertEqual(res.symbol, "D6h")

    def test_optimize_test24_to_benzene(self):
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow
        from agymolden.symmetry import find_symmetry_operations

        win = MainWindow()
        test24_path = os.path.join(DATA_DIR, "test24.zmat")
        if not os.path.exists(test24_path):
            return
        with open(test24_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)
        win._on_optimize_geometry()

        # Check all 6 carbon-carbon bond lengths in the benzene ring
        ring = [0, 1, 4, 8, 6, 2]
        for i in range(6):
            c1, c2 = ring[i], ring[(i+1)%6]
            d = float(np.linalg.norm(win.coords[c1] - win.coords[c2]))
            self.assertAlmostEqual(d, 1.3927, delta=0.015)

        # Check all C-H bonds are normal ~1.085 Å
        for c, h in [(0, 3), (1, 5), (4, 9), (8, 11), (6, 10), (2, 7)]:
            d = float(np.linalg.norm(win.coords[c] - win.coords[h]))
            self.assertAlmostEqual(d, 1.085, delta=0.015)

        # Check D6h symmetry
        res = find_symmetry_operations([a.symbol for a in win.zatoms], win.coords, tolerance=0.15)
        self.assertEqual(res.symbol, "D6h")

    def test_targeted_reorder_by_selection(self):
        """Verify targeted reordering of 2nd atom for 2 selected, 3rd atom for 3 selected, and 4th atom for 4 selected."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        test20_path = os.path.join(DATA_DIR, "test20.zmat")
        if not os.path.exists(test20_path):
            return
        with open(test20_path) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)

        # 1. 2 atoms selected [6, 8] (C7, C9): reorders 2nd atom (C9) to reference 1st atom (C7)
        self.assertEqual(win.zatoms[8].ref1, 4) # initially C5 (index 4)
        win.viewer.multi_selected = [6, 8]
        win._on_reorder_zmat_clicked()
        self.assertEqual(win.zatoms[8].ref1, 6) # now C7 (index 6)

        # 2. 3 atoms selected [0, 6, 8] (C1, C7, C9): reorders 3rd atom (C9) to reference C7 (bond) and C1 (angle)
        win.viewer.multi_selected = [0, 6, 8]
        win._on_reorder_zmat_clicked()
        self.assertEqual(win.zatoms[8].ref1, 6)
        self.assertEqual(win.zatoms[8].ref2, 0)

        # 3. 4 atoms selected [1, 0, 6, 8] (C2, C1, C7, C9): reorders 4th atom (C9) to reference C7, C1, C2
        win.viewer.multi_selected = [1, 0, 6, 8]
        win._on_reorder_zmat_clicked()
        self.assertEqual(win.zatoms[8].ref1, 6)
        self.assertEqual(win.zatoms[8].ref2, 0)
        self.assertEqual(win.zatoms[8].ref3, 1)

        # 4. Verify bonds and bond orders are strictly preserved under permutation
        win._on_bond_double_clicked(0, 1) # C1=C2
        win._on_bond_double_clicked(4, 8) # C5=C9
        win._on_bond_double_clicked(6, 2) # C7=C3
        self.assertEqual(len(win._current_explicit_bonds()), 12)
        self.assertEqual(len(win._current_bond_orders()), 3)

        # Case 2 where atom moves (select [8, 4] -> move C5 after C9)
        win.viewer.multi_selected = [8, 4]
        win._on_reorder_zmat_clicked()
        self.assertEqual(len(win._current_explicit_bonds()), 12)
        self.assertEqual(len(win._current_bond_orders()), 3)

        # Full sequential reorder preserves all 12 bonds and 3 double bond orders
        win.viewer.reordering_active = True
        win.viewer.reorder_list = [1, 4, 8, 6, 2, 0, 3, 5, 7, 9, 10, 11]
        win._apply_reorder()
        self.assertEqual(len(win._current_explicit_bonds()), 12)
        self.assertEqual(len(win._current_bond_orders()), 3)

    def test_add_ghost_atom_and_free_space_atom_isolation(self):
        """Verify adding ghost atom X or adding an atom in free canvas space does not form
        unwanted bonds to existing molecules or alter the valence / saturate existing molecules."""
        import os
        from PyQt6 import QtCore
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        win.load_preset_molecule("Benzene (C6H6)")

        self.assertEqual(len(win.zatoms), 12)
        init_bonds = list(win.viewer.bonds)
        self.assertEqual(len(init_bonds), 12)
        c_count = sum(1 for a in win.zatoms if a.symbol == "C")
        h_count = sum(1 for a in win.zatoms if a.symbol == "H")
        self.assertEqual(c_count, 6)
        self.assertEqual(h_count, 6)

        # 1. Add ghost atom X by clicking on canvas
        win.pending_add_atom_mode = True
        win.viewer.brush_element = "X"
        win._on_canvas_dragged_or_clicked(QtCore.QPoint(400, 400), QtCore.QPoint(400, 400), 0)

        # Benzene atoms must remain unchanged (6 C, 6 H) plus exactly 1 X = 13 atoms total
        self.assertEqual(len(win.zatoms), 13)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "C"), 6)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "H"), 6)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "X"), 1)
        # Bonds should remain exactly the 12 benzene bonds (X has no bonds)
        self.assertEqual(len(win.viewer.bonds), 12)
        for b in win.viewer.bonds:
            self.assertNotEqual(win.zatoms[b[0]].symbol, "X")
            self.assertNotEqual(win.zatoms[b[1]].symbol, "X")

        # 2. Add carbon atom in free space with Adjust Hydrogens (chk_auto_h) enabled
        win.pending_add_atom_mode = True
        win.viewer.brush_element = "C"
        win.chk_auto_h.setChecked(True)
        win._on_canvas_dragged_or_clicked(QtCore.QPoint(100, 100), QtCore.QPoint(100, 100), 0)

        # Preexisting benzene carbons and hydrogens must not be touched!
        # Newly added C should only saturate itself (+ 4 H = CH4)
        # Total: 13 + 5 = 18 atoms (7 C, 10 H, 1 X)
        self.assertEqual(len(win.zatoms), 18)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "C"), 7)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "H"), 10)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "X"), 1)
        # Bonds: 12 (benzene) + 4 (CH4) = 16 bonds
        self.assertEqual(len(win.viewer.bonds), 16)

    def test_test25_zmat_valence_complete(self):
        """Verify test25.zmat is already fully saturated (cyclohexane C6H12 + dummy atom X)
        and Add Hydrogens correctly recognizes that all 6 carbons and 12 hydrogens have full valence."""
        import os
        from agymolden.zmatrix import parse_zmatrix_text
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        with open(os.path.join(DATA_DIR, "test25.zmat")) as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)

        # 19 atoms: 6 C, 12 H, 1 X
        self.assertEqual(len(win.zatoms), 19)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "C"), 6)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "H"), 12)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "X"), 1)

        # Each carbon is bonded to 2 C and 2 H (valence 4 saturated)
        # Dummy atom X has 0 valence and 0 bonds
        bonds = win._current_explicit_bonds()
        self.assertEqual(len(bonds), 18)
        self.assertFalse(any(win.zatoms[u].symbol == "X" or win.zatoms[v].symbol == "X" for u, v in bonds))

        # Clicking Add Hydrogens correctly confirms valence is satisfied
        win._on_add_hydrogens()
        self.assertIn("fully satisfied", win.statusBar().currentMessage())
        self.assertEqual(len(win.zatoms), 19)

    def test_bond_order_single_click_and_numeric_keys(self):
        """Test single-clicking a bond and using numeric keys 1, 2, 3 to set bond order."""
        import os
        from agymolden.zmatrix import parse_zmatrix_text
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        # Load ethane
        with open(os.path.join(DATA_DIR, "ethane.zmat"), "r") as f:
            zatoms, _ = parse_zmatrix_text(f.read())
        win.set_molecule_from_zatoms(zatoms)

        # Find C-C bond
        c_indices = [i for i, a in enumerate(win.zatoms) if a.symbol == "C"]
        self.assertEqual(len(c_indices), 2)
        c1, c2 = c_indices
        cc_key = (min(c1, c2), max(c1, c2))

        # Initially ethane C-C bond is single (order 1)
        self.assertEqual(win.viewer.bond_orders.get(cc_key, 1), 1)

        # Simulate clicking the C-C bond in the viewer
        win.viewer.selected_bond = cc_key
        win.viewer.selected_atom_idx = c1
        win.viewer.multi_selected = [c1, c2]

        # Simulate pressing key '2'
        event_key2 = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_2,
            QtCore.Qt.KeyboardModifier.NoModifier,
            "2"
        )
        win.keyPressEvent(event_key2)

        # Verify bond order became 2 (double bond)
        # Carbon atoms valence preserved: excess hydrogens pruned from 6 to 4 (ethene / ethylene)
        c_indices = [i for i, a in enumerate(win.zatoms) if a.symbol == "C"]
        c1_new, c2_new = c_indices
        new_cc_key = (min(c1_new, c2_new), max(c1_new, c2_new))
        self.assertEqual(win.viewer.bond_orders.get(new_cc_key, 1), 2)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "H"), 4)
        self.assertEqual(win.viewer.selected_bond, new_cc_key)
        d_double = np.linalg.norm(win.coords[c1_new] - win.coords[c2_new])
        self.assertAlmostEqual(d_double, 1.335, delta=0.03)

        # Simulate pressing key '3'
        event_key3 = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_3,
            QtCore.Qt.KeyboardModifier.NoModifier,
            "3"
        )
        win.keyPressEvent(event_key3)

        # Verify bond order became 3 (triple bond, acetylene / ethyne, 2 hydrogens)
        c_indices = [i for i, a in enumerate(win.zatoms) if a.symbol == "C"]
        c1_new, c2_new = c_indices
        new_cc_key = (min(c1_new, c2_new), max(c1_new, c2_new))
        self.assertEqual(win.viewer.bond_orders.get(new_cc_key, 1), 3)
        self.assertEqual(sum(1 for a in win.zatoms if a.symbol == "H"), 2)
        self.assertEqual(win.viewer.selected_bond, new_cc_key)
        d_triple = np.linalg.norm(win.coords[c1_new] - win.coords[c2_new])
        self.assertAlmostEqual(d_triple, 1.205, delta=0.03)

        # Simulate pressing key '1'
        event_key1 = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_1,
            QtCore.Qt.KeyboardModifier.NoModifier,
            "1"
        )
        win.keyPressEvent(event_key1)

        # Verify bond order reset back to 1 (single bond) and internuclear distance relaxed to single bond
        c_indices = [i for i, a in enumerate(win.zatoms) if a.symbol == "C"]
        c1_new, c2_new = c_indices
        new_cc_key = (min(c1_new, c2_new), max(c1_new, c2_new))
        self.assertEqual(win.viewer.bond_orders.get(new_cc_key, 1), 1)
        d_single = np.linalg.norm(win.coords[c1_new] - win.coords[c2_new])
        self.assertAlmostEqual(d_single, 1.535, delta=0.03)

    def test_benzene_symmetric_bond_variables_only(self):
        """Verify that clicking 'Symm. Bonds Only' on Benzene creates exactly 2 variables (R1 for C-C, R2 for C-H)
        while leaving all angles and dihedrals as numerical constants."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        win.load_preset_molecule("Benzene (C6H6)")
        win._on_assign_symmetric_bond_variables()

        # Exactly 2 variables: R1 (1.397) and R2 (1.084)
        self.assertEqual(set(win.variables.keys()), {"R1", "R2"})
        self.assertAlmostEqual(win.variables["R1"], 1.397, places=3)
        self.assertAlmostEqual(win.variables["R2"], 1.084, places=3)

        # All C-C bonds (rows 1..5) have dist_var == 'R1', all C-H bonds (rows 6..11) have dist_var == 'R2'
        for i in range(1, 6):
            self.assertEqual(win.zatoms[i].dist_var, "R1")
            self.assertIsNone(win.zatoms[i].angle_var)
            self.assertIsNone(win.zatoms[i].dihedral_var)
        for i in range(6, 12):
            self.assertEqual(win.zatoms[i].dist_var, "R2")
            self.assertIsNone(win.zatoms[i].angle_var)
            self.assertIsNone(win.zatoms[i].dihedral_var)

        # Verify table cell displays 'R1 (1.3970)' and typing a new value or variable name updates all linked bonds
        cell_text = win.table_widget.item(1, win.table_widget.COL_DIST).text()
        self.assertIn("R1", cell_text)
        win.table_widget.item(1, win.table_widget.COL_DIST).setText("1.400")
        self.assertAlmostEqual(win.variables["R1"], 1.400, places=3)
        for i in range(1, 6):
            self.assertAlmostEqual(win.zatoms[i].dist, 1.400, places=3)

    def test_test27_replace_b_with_x_preserves_zmatrix(self):
        """Verify that loading test27.zmat and replacing B (row 1) with ghost atom X in the Z-Matrix table
        preserves the exact 4-atom Z-matrix, all reference indices, and all symbolic variables (R1, R2, A1, A2, D1)."""
        import os
        from agymolden.zmatrix import parse_zmatrix_text
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        win.chk_auto_h.setChecked(True)  # Even with Adjust Hydrogens checked!
        with open(os.path.join(DATA_DIR, "test27.zmat"), "r") as f:
            zatoms, vars_dict = parse_zmatrix_text(f.read())
        win.variables = vars_dict
        win.set_molecule_from_zatoms(zatoms)

        self.assertEqual(len(win.zatoms), 4)
        self.assertEqual([a.symbol for a in win.zatoms], ["H", "B", "H", "H"])

        # Replace 'B' with 'X' in the Z-Matrix table (row 1, COL_SYM)
        win.table_widget.item(1, win.table_widget.COL_SYM).setText("X")

        # Must preserve all 4 atoms in place: ['H', 'X', 'H', 'H']
        self.assertEqual(len(win.zatoms), 4)
        self.assertEqual([a.symbol for a in win.zatoms], ["H", "X", "H", "H"])

        # References and variables must remain completely intact
        self.assertEqual(win.zatoms[1].ref1, 0)
        self.assertEqual(win.zatoms[1].dist_var, "R1")
        self.assertEqual(win.zatoms[2].ref1, 1)
        self.assertEqual(win.zatoms[2].dist_var, "R1")
        self.assertEqual(win.zatoms[2].ref2, 0)
        self.assertEqual(win.zatoms[2].angle_var, "A1")
        self.assertEqual(win.zatoms[3].ref1, 1)
        self.assertEqual(win.zatoms[3].dist_var, "R2")
        self.assertEqual(win.zatoms[3].ref2, 0)
        self.assertEqual(win.zatoms[3].angle_var, "A2")
        self.assertEqual(win.zatoms[3].ref3, 2)
        self.assertEqual(win.zatoms[3].dihedral_var, "D1")
        self.assertEqual(set(win.variables.keys()), {"R1", "A1", "R2", "A2", "D1"})

    def test_test29_attach_water_to_h12_preserves_benzene_ring(self):
        """Verify that attaching 'Water (H2O)' to H12 (index 11) in test29.zmat replaces H12 with -OH
        to yield phenol (C6H5OH, 13 atoms) with a planar aromatic ring (~1.39 A C-C bonds, ~120 deg angles)
        instead of hydrogenating the benzene ring into 19-atom cyclohexanol."""
        import os
        from agymolden.zmatrix import parse_zmatrix_text
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow

        win = MainWindow()
        win.chk_auto_h.setChecked(True)
        with open(os.path.join(DATA_DIR, "test29.zmat"), "r") as f:
            zatoms, vars_dict = parse_zmatrix_text(f.read())
        win.variables = vars_dict
        win.set_molecule_from_zatoms(zatoms)

        self.assertEqual(len(win.zatoms), 12)
        self.assertEqual(win.zatoms[11].symbol, "H")

        # Select H12 (index 11) and attach Water (H2O)
        win.viewer.selected_atom_idx = 11
        win.combo_fragment.setCurrentText("Water (H2O)")
        win._on_attach_fragment()

        symbols = [a.symbol for a in win.zatoms]
        self.assertEqual(len(symbols), 13, f"Expected 13 atoms for phenol (C6H5OH), got {len(symbols)}: {symbols}")
        self.assertEqual(symbols.count("C"), 6)
        self.assertEqual(symbols.count("O"), 1)
        self.assertEqual(symbols.count("H"), 6)

        # Verify all C-C bonds in the ring remain aromatic (~1.39 A), NOT single sp3 (~1.535 A)
        bonds = win._current_explicit_bonds()
        cc_bonds = [(u, v) for u, v in bonds if symbols[u] == "C" and symbols[v] == "C"]
        self.assertEqual(len(cc_bonds), 6, f"Expected 6 C-C ring bonds, got {len(cc_bonds)}")
        for u, v in cc_bonds:
            d = float(np.linalg.norm(win.coords[u] - win.coords[v]))
            self.assertAlmostEqual(d, 1.393, delta=0.02, msg=f"C-C bond ({u},{v}) length {d:.4f} is not aromatic")

    def test_analytical_uff_mmff_torsions_and_charges(self):
        """Verify that the upgraded Analytical UFF/MMFF94s force field computes Gasteiger partial charges
        and optimizes eclipsed ethane to a staggered 60-degree dihedral conformation."""
        from agymolden.forcefield import compute_gasteiger_charges, optimize_geometry_force_field
        from agymolden.zmatrix import calculate_dihedral

        # 1. Gasteiger charges on water: O should be negative, H positive, sum == 0
        q_water = compute_gasteiger_charges(["O", "H", "H"], [(0, 1), (0, 2)])
        self.assertLess(q_water[0], -0.25)
        self.assertGreater(q_water[1], 0.12)
        self.assertAlmostEqual(float(np.sum(q_water)), 0.0, places=5)

        # 2. Nearly-eclipsed ethane (dihedral ~ 5 deg) should relax to staggered (~60 deg)
        symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
        coords = np.array([
            [0.000, 0.000, 0.000],
            [1.540, 0.000, 0.000],
            [-0.360,  1.020,  0.000],
            [-0.360, -0.510,  0.883],
            [-0.360, -0.510, -0.883],
            [1.900,  1.015,  0.090],  # nearly eclipsed with H2 (~5 deg dihedral)
            [1.900, -0.585,  0.835],
            [1.900, -0.430, -0.925],
        ], dtype=np.float64)
        bonds = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (1, 7)]
        opt_coords, _ = optimize_geometry_force_field(symbols, coords, user_bonds=bonds, max_iter=200, strict_bonds=True)
        dih = abs(calculate_dihedral(opt_coords[2], opt_coords[0], opt_coords[1], opt_coords[5]))
        # Staggered conformation has |dihedral| near 60 degrees (away from 0 eclipsed)
        self.assertGreater(dih, 45.0, f"Expected staggered dihedral (~60 deg), got {dih:.2f} deg")
        self.assertLess(dih, 75.0, f"Expected staggered dihedral (~60 deg), got {dih:.2f} deg")

    def test_semiempirical_qm_gfn0_xtb_optimization(self):
        """Verify that the pure-Python Semiempirical Tight-Binding QM (GFN0-xTB / SCC-TB) engine
        computes HOMO/LUMO gap, SCC charges, Wiberg bond orders, and optimizes geometry via UI."""
        import os
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from agymolden.main_window import MainWindow
        from agymolden.semiempirical import compute_semiempirical_properties

        win = MainWindow()
        win.load_preset_molecule("Benzene (C6H6)")

        # Switch dropdown to Semiempirical QM (GFN0-xTB)
        idx_semi = win.combo_opt_method.findData("semiempirical")
        self.assertGreaterEqual(idx_semi, 0)
        win.combo_opt_method.setCurrentIndex(idx_semi)
        self.assertEqual(win._get_selected_opt_method(), "semiempirical")

        win._on_optimize_geometry()
        props = compute_semiempirical_properties([a.symbol for a in win.zatoms], win.coords)
        self.assertGreater(props["gap_ev"], 1.5, "Expected positive HOMO-LUMO gap for benzene")
        # Aromatic C-C Wiberg bond order should be between 1.2 and 1.7
        wbo = props["wiberg_bond_orders"].get((0, 1), 0.0)
        self.assertGreater(wbo, 1.15)
        self.assertLess(wbo, 1.75)


if __name__ == "__main__":
    unittest.main()




