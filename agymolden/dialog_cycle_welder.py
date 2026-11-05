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
        self.resize(520, 480)

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
        grp_sel = QtWidgets.QGroupBox("1. Target Bond (Select 2 Adjacent Atoms)")
        sel_layout = QtWidgets.QHBoxLayout(grp_sel)
        self.lbl_atoms = QtWidgets.QLabel("No atoms selected. Please select 2 atoms in the 3D viewer.")
        self.lbl_atoms.setStyleSheet("font-weight: bold; color: #38bdf8;")
        sel_layout.addWidget(self.lbl_atoms, 1)

        btn_refresh = QtWidgets.QPushButton("🔄 Read 3D Selection")
        btn_refresh.setToolTip("Re-read the currently selected atoms from the 3D canvas")
        btn_refresh.clicked.connect(self._populate_selection)
        sel_layout.addWidget(btn_refresh)
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
        """Read selected atoms from parent window."""
        sel = []
        if self.window:
            sel = list(getattr(self.window, "selected_atoms", []))
            if not sel and hasattr(self.window, "viewer") and self.window.viewer:
                sel = list(getattr(self.window.viewer, "selected_atoms", []))

        symbols = [a.symbol for a in getattr(self.window, "zatoms", [])] if self.window else []
        coords = getattr(self.window, "coords", None)

        if len(sel) == 2:
            a1, a2 = sel[0], sel[1]
            s1 = symbols[a1] if a1 < len(symbols) else f"Atom {a1+1}"
            s2 = symbols[a2] if a2 < len(symbols) else f"Atom {a2+1}"
            dist_str = ""
            if coords is not None and a1 < len(coords) and a2 < len(coords):
                d = np.linalg.norm(coords[a1] - coords[a2])
                dist_str = f" (Distance: {d:.3f} Å)"
            self.lbl_atoms.setText(f"Target Bond: {s1} ({a1+1}) — {s2} ({a2+1}){dist_str}")
            self.lbl_atoms.setStyleSheet("font-weight: bold; color: #4ade80;")
            self.btn_fuse.setEnabled(True)
            self.btn_bridge.setEnabled(True)
        else:
            self.lbl_atoms.setText(
                f"Currently {len(sel)} atom(s) selected. Please select exactly 2 atoms."
            )
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
        sel = list(getattr(self.window, "selected_atoms", []))
        if len(sel) != 2:
            dialog_service.warning(self, "Selection Required", "Please select exactly 2 atoms defining the bond to weld.")
            return

        a1, a2 = sel[0], sel[1]
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
            zatoms = cartesian_to_zmatrix(new_syms, new_coords)
            self.window.zatoms = zatoms
            self.window.coords = new_coords

            if hasattr(self.window, "table_widget") and self.window.table_widget:
                self.window.table_widget.load_zatoms(zatoms)
            if hasattr(self.window, "viewer") and self.window.viewer:
                self.window.viewer.set_molecule(zatoms, new_coords)
                self.window.viewer.selected_atoms.clear()
            if hasattr(self.window, "_sync_raw_text"):
                self.window._sync_raw_text()

            p_name = preset_key if preset_key != "Custom" else f"{custom_sz}-gon"
            dialog_service.info(
                self,
                "Ring Welded Successfully",
                f"Successfully welded {p_name} onto bond {a1+1}–{a2+1}!\n"
                f"Total atoms: {len(new_syms)} (hydrogens adjusted for valency)."
            )
            self.accept()
        except Exception as e:
            dialog_service.error(self, "Welding Error", f"Failed to weld ring:\n{e}")

    def _on_execute_bridge(self):
        """Execute bicyclic bridgehead insertion."""
        sel = list(getattr(self.window, "selected_atoms", []))
        if len(sel) != 2:
            dialog_service.warning(self, "Selection Required", "Please select exactly 2 bridgehead atoms.")
            return

        a1, a2 = sel[0], sel[1]
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
            zatoms = cartesian_to_zmatrix(new_syms, new_coords)
            self.window.zatoms = zatoms
            self.window.coords = new_coords

            if hasattr(self.window, "table_widget") and self.window.table_widget:
                self.window.table_widget.load_zatoms(zatoms)
            if hasattr(self.window, "viewer") and self.window.viewer:
                self.window.viewer.set_molecule(zatoms, new_coords)
                self.window.viewer.selected_atoms.clear()
            if hasattr(self.window, "_sync_raw_text"):
                self.window._sync_raw_text()

            dialog_service.info(
                self,
                "Bridgehead Formed",
                f"Successfully formed {b_type} bridgehead across atoms {a1+1} and {a2+1}!\n"
                f"Total atoms: {len(new_syms)}."
            )
            self.accept()
        except Exception as e:
            dialog_service.error(self, "Bridgehead Error", f"Failed to create bridgehead:\n{e}")
