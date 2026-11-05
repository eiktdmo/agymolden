"""
Molecular Cycle Welder & Ring Fusion Studio Dialog for AgyMolden.

Provides an intuitive UI for:
1. Fusing aromatic or saturated carbo- and heterocycles onto selected bonds.
2. Creating bridged bicyclic systems (norbornane, bicyclo[2.2.2]octane, oxabicycles).
3. Live preview and 1-click force field relaxation.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from PyQt6 import QtWidgets, QtGui, QtCore

from .cycle_welder import (
    RING_PRESETS,
    fuse_ring_to_bond,
    bridge_cycle_atoms,
)
from .dialog_service import dialog_service


class CycleWelderDialog(QtWidgets.QDialog):
    """Interactive Studio for fusing, welding, and bridging molecular rings."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent if isinstance(parent, QtWidgets.QWidget) else None)
        self.window = parent
        self.setWindowTitle("💍 Molecular Cycle Welder & Ring Fusion Studio")
        self.resize(560, 520)

        self.target_bond: Optional[Tuple[int, int]] = None
        self._init_ui()
        self._populate_selection()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Tabs for Edge Annulation vs Bicyclic Bridging
        self.tabs = QtWidgets.QTabWidget()

        # Tab 1: Edge Annulation (Fuse Ring onto Bond)
        tab_fuse = QtWidgets.QWidget()
        fuse_layout = QtWidgets.QVBoxLayout(tab_fuse)
        fuse_layout.setSpacing(10)

        # Selection group
        grp_sel = QtWidgets.QGroupBox("1. Target Bond to Fuse Onto")
        sel_layout = QtWidgets.QVBoxLayout(grp_sel)
        sel_layout.setSpacing(6)

        row_bond = QtWidgets.QHBoxLayout()
        row_bond.addWidget(QtWidgets.QLabel("Select Bond:"))
        self.combo_bonds = QtWidgets.QComboBox()
        self.combo_bonds.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.combo_bonds.currentIndexChanged.connect(self._on_bond_combo_changed)
        row_bond.addWidget(self.combo_bonds, 1)

        btn_refresh = QtWidgets.QPushButton("🔄 Read 3D Viewport")
        btn_refresh.setToolTip("Re-read the bond or 2 atoms selected in the 3D viewport")
        btn_refresh.clicked.connect(self._populate_selection)
        row_bond.addWidget(btn_refresh)
        sel_layout.addLayout(row_bond)

        self.lbl_atoms = QtWidgets.QLabel("No bond selected.")
        self.lbl_atoms.setStyleSheet("font-weight: bold; color: #38bdf8;")
        sel_layout.addWidget(self.lbl_atoms)

        fuse_layout.addWidget(grp_sel)

        # Ring template group
        grp_ring = QtWidgets.QGroupBox("2. Ring Template & Annulation Preset")
        ring_layout = QtWidgets.QGridLayout(grp_ring)
        ring_layout.setVerticalSpacing(8)

        ring_layout.addWidget(QtWidgets.QLabel("Ring Type:"), 0, 0)
        self.combo_preset = QtWidgets.QComboBox()
        for name, data in RING_PRESETS.items():
            self.combo_preset.addItem(f"{name} ({data['size']}-membered)", name)
        self.combo_preset.addItem("Custom N-gon...", "Custom")
        self.combo_preset.currentIndexChanged.connect(self._on_preset_changed)
        ring_layout.addWidget(self.combo_preset, 0, 1)

        # Custom size spinner (hidden by default)
        self.lbl_custom = QtWidgets.QLabel("Ring Size (N):")
        self.spin_custom = QtWidgets.QSpinBox()
        self.spin_custom.setRange(3, 10)
        self.spin_custom.setValue(6)
        self.lbl_custom.setVisible(False)
        self.spin_custom.setVisible(False)
        ring_layout.addWidget(self.lbl_custom, 1, 0)
        ring_layout.addWidget(self.spin_custom, 1, 1)

        # Preset Description
        self.lbl_desc = QtWidgets.QLabel(RING_PRESETS["Benzene"]["desc"])
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setStyleSheet("color: #94a3b8; font-size: 11px; font-style: italic;")
        ring_layout.addWidget(self.lbl_desc, 2, 0, 1, 2)

        fuse_layout.addWidget(grp_ring)

        # Options group
        grp_opt = QtWidgets.QGroupBox("3. Fusion Geometry Options")
        opt_layout = QtWidgets.QVBoxLayout(grp_opt)
        opt_layout.setSpacing(6)

        self.chk_flip = QtWidgets.QCheckBox("Flip Fusion Side (weld to opposite side of bond)")
        opt_layout.addWidget(self.chk_flip)

        self.chk_relax = QtWidgets.QCheckBox("Auto-relax with Analytical Force Field (UFF / MMFF94)")
        self.chk_relax.setChecked(True)
        opt_layout.addWidget(self.chk_relax)

        self.chk_freeze = QtWidgets.QCheckBox("Freeze parent atoms (relax only newly fused ring)")
        self.chk_freeze.setChecked(False)
        opt_layout.addWidget(self.chk_freeze)

        fuse_layout.addWidget(grp_opt)

        # Fuse action button
        self.btn_fuse = QtWidgets.QPushButton("💍 Weld Ring onto Selected Bond")
        self.btn_fuse.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 6px;"
        )
        self.btn_fuse.clicked.connect(self._on_execute_fuse)
        fuse_layout.addWidget(self.btn_fuse)
        fuse_layout.addStretch()

        self.tabs.addTab(tab_fuse, "🔗 Edge Annulation (Fuse Ring)")

        # Tab 2: Bicyclic Bridgehead Welding
        tab_bridge = QtWidgets.QWidget()
        bridge_layout = QtWidgets.QVBoxLayout(tab_bridge)
        bridge_layout.setSpacing(10)

        lbl_br_info = QtWidgets.QLabel(
            "Select 2 bridgehead atoms across a ring (e.g. 1,4-positions of cyclohexane) to insert a bridge."
        )
        lbl_br_info.setWordWrap(True)
        lbl_br_info.setStyleSheet("color: #94a3b8;")
        bridge_layout.addWidget(lbl_br_info)

        grp_br_type = QtWidgets.QGroupBox("Bridge Motif")
        br_type_layout = QtWidgets.QHBoxLayout(grp_br_type)
        self.combo_bridge = QtWidgets.QComboBox()
        self.combo_bridge.addItem("-CH2- (Norbornane bridge)", "-CH2-")
        self.combo_bridge.addItem("-CH2-CH2- (Bicyclo[2.2.2]octane bridge)", "-CH2CH2-")
        self.combo_bridge.addItem("-O- (7-Oxanorbornane bridge)", "-O-")
        self.combo_bridge.addItem("-NH- (7-Azanorbornane bridge)", "-NH-")
        self.combo_bridge.addItem("-CO- (Carbonyl bridge / Camphor-like)", "-CO-")
        br_type_layout.addWidget(self.combo_bridge)
        bridge_layout.addWidget(grp_br_type)

        self.btn_bridge = QtWidgets.QPushButton("🌉 Form Bicyclic Bridge")
        self.btn_bridge.setStyleSheet(
            "background-color: #0d9488; color: white; font-weight: bold; font-size: 13px; padding: 10px; border-radius: 6px;"
        )
        self.btn_bridge.clicked.connect(self._on_execute_bridge)
        bridge_layout.addWidget(self.btn_bridge)
        bridge_layout.addStretch()

        self.tabs.addTab(tab_bridge, "🌉 Bicyclic Bridgehead")

        layout.addWidget(self.tabs, 1)

        # Bottom close button
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        layout.addWidget(btn_close, 0, QtCore.Qt.AlignmentFlag.AlignRight)

    def _populate_selection(self):
        """Read selected atoms from parent window, populate bond combo box, and update target bond."""
        self.combo_bonds.blockSignals(True)
        self.combo_bonds.clear()

        symbols = [a.symbol for a in getattr(self.window, "zatoms", [])] if self.window else []
        coords = getattr(self.window, "coords", None)
        viewer = getattr(self.window, "viewer", None) if self.window else None

        # 1. Determine active selection
        sel = []
        if viewer:
            # Did user click a bond directly?
            if getattr(viewer, "selected_bond", None) is not None:
                sel = list(viewer.selected_bond)
            # Or Shift-selected 2 atoms?
            elif getattr(viewer, "multi_selected", None) and len(viewer.multi_selected) == 2:
                sel = list(viewer.multi_selected)
            elif getattr(viewer, "selected_atom_idx", None) is not None and getattr(viewer, "multi_selected", None):
                combined = [viewer.selected_atom_idx] + [i for i in viewer.multi_selected if i != viewer.selected_atom_idx]
                if len(combined) == 2:
                    sel = combined[:2]

        if not sel and hasattr(self.window, "_get_ordered_selection"):
            ord_sel = self.window._get_ordered_selection()
            if len(ord_sel) == 2:
                sel = ord_sel

        if not sel and self.window:
            cand = list(getattr(self.window, "selected_atoms", []))
            if len(cand) == 2:
                sel = cand

        # 2. Get all molecular bonds
        bonds = []
        if viewer and getattr(viewer, "bonds", None):
            bonds = list(viewer.bonds)
        elif symbols and coords is not None and len(symbols) == len(coords) and len(symbols) > 1:
            from .forcefield import detect_connectivity
            bonds = detect_connectivity(symbols, coords, strict=False)

        # Ensure selected pair is in bonds list even if non-standard distance
        if len(sel) == 2:
            pair = tuple(sorted((sel[0], sel[1])))
            if pair not in [tuple(sorted(b)) for b in bonds]:
                bonds.insert(0, pair)

        selected_bond_idx = -1
        for idx, (u, v) in enumerate(bonds):
            s1 = symbols[u] if u < len(symbols) else f"Atom {u+1}"
            s2 = symbols[v] if v < len(symbols) else f"Atom {v+1}"
            dist = float(np.linalg.norm(coords[u] - coords[v])) if (coords is not None and u < len(coords) and v < len(coords)) else 0.0
            label = f"Bond: {s1}{u+1} — {s2}{v+1} ({dist:.3f} Å)"
            self.combo_bonds.addItem(label, (u, v))
            if len(sel) == 2 and {u, v} == {sel[0], sel[1]}:
                selected_bond_idx = idx

        if selected_bond_idx >= 0:
            self.combo_bonds.setCurrentIndex(selected_bond_idx)
        elif self.combo_bonds.count() > 0:
            self.combo_bonds.setCurrentIndex(0)

        self.combo_bonds.blockSignals(False)
        self._on_bond_combo_changed(self.combo_bonds.currentIndex())

    def _on_bond_combo_changed(self, idx: int):
        data = self.combo_bonds.currentData()
        if data and len(data) == 2:
            self.target_bond = (data[0], data[1])
            if self.window:
                self.window.selected_atoms = [data[0], data[1]]
                viewer = getattr(self.window, "viewer", None)
                if viewer:
                    viewer.selected_bond = (data[0], data[1])
                    viewer.multi_selected = [data[0], data[1]]
                    viewer.selected_atom_idx = data[0]
                    viewer.update()

            symbols = [a.symbol for a in getattr(self.window, "zatoms", [])] if self.window else []
            coords = getattr(self.window, "coords", None)
            u, v = data
            s1 = symbols[u] if u < len(symbols) else f"Atom {u+1}"
            s2 = symbols[v] if v < len(symbols) else f"Atom {v+1}"
            dist_str = ""
            if coords is not None and u < len(coords) and v < len(coords):
                d = float(np.linalg.norm(coords[u] - coords[v]))
                dist_str = f" (Distance: {d:.3f} Å)"
            self.lbl_atoms.setText(f"Target Bond: {s1}{u+1} — {s2}{v+1}{dist_str}")
            self.lbl_atoms.setStyleSheet("font-weight: bold; color: #4ade80;")
            self.btn_fuse.setEnabled(True)
            self.btn_bridge.setEnabled(True)
        else:
            self.target_bond = None
            self.lbl_atoms.setText("No bond selected in molecule.")
            self.lbl_atoms.setStyleSheet("font-weight: bold; color: #f87171;")
            self.btn_fuse.setEnabled(False)
            self.btn_bridge.setEnabled(False)

    def _on_preset_changed(self, idx: int):
        key = self.combo_preset.currentData()
        if key == "Custom":
            self.lbl_custom.setVisible(True)
            self.spin_custom.setVisible(True)
            self.lbl_desc.setText("Custom regular polygon ring fused onto the selected bond.")
        else:
            self.lbl_custom.setVisible(False)
            self.spin_custom.setVisible(False)
            if key in RING_PRESETS:
                self.lbl_desc.setText(RING_PRESETS[key]["desc"])

    def _on_execute_fuse(self):
        """Execute ring fusion."""
        bond = getattr(self, "target_bond", None)
        if not bond:
            sel = list(getattr(self.window, "selected_atoms", []))
            if len(sel) == 2:
                bond = (sel[0], sel[1])
        if not bond:
            dialog_service.warning(self, "Selection Required", "Please select a bond to weld the ring onto.")
            return

        a1, a2 = bond
        symbols = [a.symbol for a in self.window.zatoms]
        coords = self.window.coords

        preset_key = self.combo_preset.currentData()
        custom_sz = self.spin_custom.value() if preset_key == "Custom" else None
        flip = self.chk_flip.isChecked()
        relax = self.chk_relax.isChecked()
        freeze = self.chk_freeze.isChecked()

        try:
            new_syms, new_coords = fuse_ring_to_bond(
                symbols=symbols,
                coords=coords,
                a1=a1,
                a2=a2,
                ring_preset=preset_key,
                custom_size=custom_sz,
                flip_side=flip,
                relax=relax,
                freeze_parent=freeze,
            )

            # Push undo state and update window
            if hasattr(self.window, "_push_undo_state"):
                self.window._push_undo_state()

            from .zmatrix import cartesian_to_zmatrix
            from .forcefield import detect_connectivity
            detected_bonds = detect_connectivity(new_syms, new_coords, strict=False)
            zatoms = cartesian_to_zmatrix(new_syms, new_coords, user_bonds=detected_bonds)

            if hasattr(self.window, "set_molecule_from_zatoms"):
                self.window.set_molecule_from_zatoms(zatoms, exact_coords=new_coords)
            else:
                self.window.zatoms = zatoms
                self.window.coords = new_coords
                if hasattr(self.window, "viewer") and self.window.viewer:
                    self.window.viewer.set_molecule(new_syms, new_coords)

            if hasattr(self.window, "viewer") and self.window.viewer:
                self.window.viewer.selected_atom_idx = None
                self.window.viewer.multi_selected = []
                self.window.viewer.selected_bond = None
                self.window.viewer.auto_center()
                self.window.viewer.update()

            p_name = preset_key if preset_key != "Custom" else f"{custom_sz}-gon"
            dialog_service.info(
                self,
                "Ring Welded Successfully",
                f"Successfully welded {p_name} onto bond {symbols[a1]}{a1+1}–{symbols[a2]}{a2+1}!\n"
                f"Total atoms: {len(new_syms)} (hydrogens adjusted for valency)."
            )
            self.accept()
        except Exception as e:
            dialog_service.error(self, "Welding Error", f"Failed to weld ring:\n{e}")

    def _on_execute_bridge(self):
        """Execute bicyclic bridgehead insertion."""
        bond = getattr(self, "target_bond", None)
        if not bond:
            sel = list(getattr(self.window, "selected_atoms", []))
            if len(sel) == 2:
                bond = (sel[0], sel[1])
        if not bond:
            dialog_service.warning(self, "Selection Required", "Please select exactly 2 bridgehead atoms.")
            return

        a1, a2 = bond
        symbols = [a.symbol for a in self.window.zatoms]
        coords = self.window.coords
        b_type = self.combo_bridge.currentData()

        try:
            new_syms, new_coords = bridge_cycle_atoms(
                symbols=symbols,
                coords=coords,
                a1=a1,
                a2=a2,
                bridge_type=b_type,
                relax=True,
            )

            if hasattr(self.window, "_push_undo_state"):
                self.window._push_undo_state()

            from .zmatrix import cartesian_to_zmatrix
            from .forcefield import detect_connectivity
            detected_bonds = detect_connectivity(new_syms, new_coords, strict=False)
            zatoms = cartesian_to_zmatrix(new_syms, new_coords, user_bonds=detected_bonds)

            if hasattr(self.window, "set_molecule_from_zatoms"):
                self.window.set_molecule_from_zatoms(zatoms, exact_coords=new_coords)
            else:
                self.window.zatoms = zatoms
                self.window.coords = new_coords
                if hasattr(self.window, "viewer") and self.window.viewer:
                    self.window.viewer.set_molecule(new_syms, new_coords)

            if hasattr(self.window, "viewer") and self.window.viewer:
                self.window.viewer.selected_atom_idx = None
                self.window.viewer.multi_selected = []
                self.window.viewer.selected_bond = None
                self.window.viewer.auto_center()
                self.window.viewer.update()

            dialog_service.info(
                self,
                "Bridgehead Formed",
                f"Successfully formed {b_type} bridgehead across atoms {symbols[a1]}{a1+1} and {symbols[a2]}{a2+1}!\n"
                f"Total atoms: {len(new_syms)}."
            )
            self.accept()
        except Exception as e:
            dialog_service.error(self, "Bridgehead Error", f"Failed to create bridgehead:\n{e}")
