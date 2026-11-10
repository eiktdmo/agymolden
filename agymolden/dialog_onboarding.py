"""Interactive Onboarding Tour & First-Launch Walkthrough Dialog for AgyMolden."""

from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
from PyQt6 import QtCore, QtGui, QtWidgets


class OnboardingTourDialog(QtWidgets.QDialog):
    """Multi-step interactive onboarding tour introducing core features of AgyMolden."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to AgyMolden — Quick Tour & Onboarding")
        self.resize(720, 520)
        self.setMinimumSize(640, 460)

        self.current_step = 0
        self.settings = QtCore.QSettings("AgyMolden", "Preferences")

        self._init_ui()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 16)
        main_layout.setSpacing(14)

        # Header banner
        header = QtWidgets.QHBoxLayout()
        logo_lbl = QtWidgets.QLabel("⚛️")
        logo_lbl.setStyleSheet("font-size: 32px;")
        header.addWidget(logo_lbl)

        title_box = QtWidgets.QVBoxLayout()
        self.lbl_title = QtWidgets.QLabel("Welcome to AgyMolden")
        self.lbl_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #0284c7;")
        self.lbl_subtitle = QtWidgets.QLabel("Interactive Quantum Chemistry & Molecular Modeling Workbench")
        self.lbl_subtitle.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_subtitle)
        header.addLayout(title_box)
        header.addStretch()

        self.lbl_step_indicator = QtWidgets.QLabel("Step 1 of 4")
        self.lbl_step_indicator.setStyleSheet(
            "background: #1e293b; color: #38bdf8; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 11px;"
        )
        header.addWidget(self.lbl_step_indicator)
        main_layout.addLayout(header)

        # Stacked pages for the tour steps
        self.pages_stack = QtWidgets.QStackedWidget()
        self._create_pages()
        main_layout.addWidget(self.pages_stack, 1)

        # Bottom navigation controls
        bottom_row = QtWidgets.QHBoxLayout()
        self.chk_show_on_startup = QtWidgets.QCheckBox("Show this tour on startup")
        self.chk_show_on_startup.setChecked(self.settings.value("show_onboarding_on_startup", True, type=bool))
        self.chk_show_on_startup.toggled.connect(self._on_toggle_startup)
        bottom_row.addWidget(self.chk_show_on_startup)

        bottom_row.addStretch()

        self.btn_prev = QtWidgets.QPushButton("◀ Previous")
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._on_prev)
        bottom_row.addWidget(self.btn_prev)

        self.btn_next = QtWidgets.QPushButton("Next ▶")
        self.btn_next.setStyleSheet(
            "background-color: #0284c7; color: white; font-weight: bold; padding: 6px 16px; border-radius: 4px;"
        )
        self.btn_next.clicked.connect(self._on_next)
        bottom_row.addWidget(self.btn_next)

        main_layout.addLayout(bottom_row)

    def _create_pages(self):
        # Page 1: 3D Viewport & Navigation
        p1 = QtWidgets.QWidget()
        l1 = QtWidgets.QVBoxLayout(p1)
        l1.setSpacing(10)
        card1 = QtWidgets.QFrame()
        card1.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;")
        cl1 = QtWidgets.QVBoxLayout(card1)
        cl1.addWidget(QtWidgets.QLabel(
            "<h3 style='color:#38bdf8; margin:0;'>1. Interactive 3D Canvas Navigation</h3>"
            "<p style='color:#cbd5e1; line-height: 1.5; font-size: 13px;'>"
            "AgyMolden provides a high-performance OpenGL molecular viewport with hardware-accelerated instanced rendering:"
            "</p>"
            "<ul style='color:#cbd5e1; line-height: 1.6; font-size: 12px;'>"
            "<li><b>Left-Click Drag:</b> Rotate molecule smoothly in 3D arcball space.</li>"
            "<li><b>Right-Click Drag:</b> Pan and translate camera along viewport plane.</li>"
            "<li><b>Scroll Wheel:</b> Zoom in and out smoothly.</li>"
            "<li><b>Spacebar:</b> Center and reset camera view.</li>"
            "<li><b>Ctrl+1, Ctrl+2, Ctrl+3:</b> Instant orthogonal standard projections (XY, XZ, YZ).</li>"
            "</ul>"
        ))

        # Quick sample loading & workspace mode buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_water = QtWidgets.QPushButton("💧 Load Water (H₂O)")
        btn_water.clicked.connect(lambda: self._load_sample("h2o"))
        btn_benzene = QtWidgets.QPushButton("⬡ Load Benzene (C₆H₆)")
        btn_benzene.clicked.connect(lambda: self._load_sample("benzene"))
        btn_student = QtWidgets.QPushButton("🎓 Activate Student / Essentials Mode")
        btn_student.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;")
        btn_student.clicked.connect(lambda: self._activate_workspace_mode("student"))
        btn_row.addWidget(btn_water)
        btn_row.addWidget(btn_benzene)
        btn_row.addWidget(btn_student)
        btn_row.addStretch()
        cl1.addLayout(btn_row)

        l1.addWidget(card1)
        l1.addStretch()
        self.pages_stack.addWidget(p1)

        # Page 2: Z-Matrix & Molecule Builder
        p2 = QtWidgets.QWidget()
        l2 = QtWidgets.QVBoxLayout(p2)
        l2.setSpacing(10)
        card2 = QtWidgets.QFrame()
        card2.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;")
        cl2 = QtWidgets.QVBoxLayout(card2)
        cl2.addWidget(QtWidgets.QLabel(
            "<h3 style='color:#38bdf8; margin:0;'>2. Parametric Z-Matrix & Interactive Builder</h3>"
            "<p style='color:#cbd5e1; line-height: 1.5; font-size: 13px;'>"
            "Seamless bidirectional synchronization between 3D Cartesian coordinates and internal coordinates (Z-Matrix):"
            "</p>"
            "<ul style='color:#cbd5e1; line-height: 1.6; font-size: 12px;'>"
            "<li><b>Drag-to-Add:</b> Drag from any atom to instantly spawn a bonded substituent.</li>"
            "<li><b>Bond Orders:</b> Double-click any bond or press <b>1, 2, 3</b> to switch Single/Double/Triple order.</li>"
            "<li><b>Ring Welder (Ctrl+Shift+W):</b> 1-click fuse aromatic benzene, cyclohexane, or cyclopentane rings.</li>"
            "<li><b>Metric Sliders:</b> Adjust bond lengths (<b>Alt+B</b>), valence angles (<b>Alt+A</b>), and dihedrals (<b>Alt+D</b>) in real time.</li>"
            "<li><b>Mutate Elements:</b> Double-click any atom to open the Periodic Table for element replacement.</li>"
            "</ul>"
        ))
        l2.addWidget(card2)
        l2.addStretch()
        self.pages_stack.addWidget(p2)

        # Page 3: Multi-Engine Quantum Chemistry
        p3 = QtWidgets.QWidget()
        l3 = QtWidgets.QVBoxLayout(p3)
        l3.setSpacing(10)
        card3 = QtWidgets.QFrame()
        card3.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;")
        cl3 = QtWidgets.QVBoxLayout(card3)
        cl3.addWidget(QtWidgets.QLabel(
            "<h3 style='color:#38bdf8; margin:0;'>3. Quantum Chemistry Engine Matrix & Live Monitor</h3>"
            "<p style='color:#cbd5e1; line-height: 1.5; font-size: 13px;'>"
            "Execute and monitor quantum chemical calculations across top industry codes:"
            "</p>"
            "<ul style='color:#cbd5e1; line-height: 1.6; font-size: 12px;'>"
            "<li><b>Supported Engines:</b> Gaussian 16/09, ORCA, PySCF, Psi4, and newBRORSEN.</li>"
            "<li><b>Automated Input Generation (Ctrl+G):</b> Generates complete route cards, basis sets, and charge/multiplicity specs.</li>"
            "<li><b>Live Job Monitor (Ctrl+J):</b> Stream output logs with real-time electronic SCF convergence plots and 3D geometry step tracking.</li>"
            "<li><b>QC Doctor:</b> Automated diagnostics explaining SCF convergence failure, linear dependencies, and basis set deficiencies.</li>"
            "</ul>"
        ))
        l3.addWidget(card3)
        l3.addStretch()
        self.pages_stack.addWidget(p3)

        # Page 4: Wavefunction, Spectroscopy & Command Palette
        p4 = QtWidgets.QWidget()
        l4 = QtWidgets.QVBoxLayout(p4)
        l4.setSpacing(10)
        card4 = QtWidgets.QFrame()
        card4.setStyleSheet("background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 14px;")
        cl4 = QtWidgets.QVBoxLayout(card4)
        cl4.addWidget(QtWidgets.QLabel(
            "<h3 style='color:#38bdf8; margin:0;'>4. Wavefunction Surfaces, Spectroscopy & Command Palette</h3>"
            "<p style='color:#cbd5e1; line-height: 1.5; font-size: 13px;'>"
            "Publication-ready spectroscopic analysis and productivity accelerators:"
            "</p>"
            "<ul style='color:#cbd5e1; line-height: 1.6; font-size: 12px;'>"
            "<li><b>3D Molecular Orbitals & ESP:</b> Render HOMO/LUMO isosurfaces and electrostatic potential maps directly from Molden or cube files.</li>"
            "<li><b>Vibrational IR Animation:</b> Interactive harmonic normal mode player with animated vectors and GIF export.</li>"
            "<li><b>Command Palette (Ctrl+K):</b> Instant fuzzy search for all 60+ tools, exporters, and dialogs.</li>"
            "<li><b>Example Gallery (Ctrl+Shift+G):</b> Browse bundled benchmark wavefunctions and output logs.</li>"
            "</ul>"
        ))
        l4.addWidget(card4)
        l4.addStretch()
        self.pages_stack.addWidget(p4)

    def _update_step_ui(self):
        total = self.pages_stack.count()
        self.lbl_step_indicator.setText(f"Step {self.current_step + 1} of {total}")
        self.btn_prev.setEnabled(self.current_step > 0)
        if self.current_step == total - 1:
            self.btn_next.setText("Finish & Explore 🎉")
        else:
            self.btn_next.setText("Next ▶")

    def _on_next(self):
        if self.current_step < self.pages_stack.count() - 1:
            self.current_step += 1
            self.pages_stack.setCurrentIndex(self.current_step)
            self._update_step_ui()
        else:
            self.settings.setValue("has_seen_onboarding_tour", True)
            self.accept()

    def _on_prev(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.pages_stack.setCurrentIndex(self.current_step)
            self._update_step_ui()

    def _on_toggle_startup(self, checked: bool):
        self.settings.setValue("show_onboarding_on_startup", checked)

    def _activate_workspace_mode(self, mode_key: str):
        parent = self.parent()
        if parent and hasattr(parent, "combo_workspace_mode"):
            for i in range(parent.combo_workspace_mode.count()):
                if parent.combo_workspace_mode.itemData(i) == mode_key:
                    parent.combo_workspace_mode.setCurrentIndex(i)
                    break

    def _load_sample(self, sample_key: str):
        parent = self.parent()
        if not parent:
            return
        if sample_key == "h2o":
            symbols = ["O", "H", "H"]
            coords = [
                [0.0, 0.0, 0.117],
                [0.0, 0.757, -0.469],
                [0.0, -0.757, -0.469],
            ]
        elif sample_key == "benzene":
            symbols = ["C"] * 6 + ["H"] * 6
            coords = [
                [1.397, 0.0, 0.0], [0.699, 1.210, 0.0], [-0.699, 1.210, 0.0],
                [-1.397, 0.0, 0.0], [-0.699, -1.210, 0.0], [0.699, -1.210, 0.0],
                [2.481, 0.0, 0.0], [1.241, 2.149, 0.0], [-1.241, 2.149, 0.0],
                [-2.481, 0.0, 0.0], [-1.241, -2.149, 0.0], [1.241, -2.149, 0.0]
            ]
        else:
            return

        import numpy as np
        from agymolden.zmatrix import cartesian_to_zmatrix
        zatoms = cartesian_to_zmatrix(symbols, np.array(coords, dtype=float))
        if hasattr(parent, "set_molecule_from_zatoms"):
            parent.set_molecule_from_zatoms(zatoms, exact_coords=np.array(coords, dtype=float))
        elif hasattr(parent, "zatoms"):
            parent.zatoms = zatoms
            parent.coords = np.array(coords, dtype=float)
            if hasattr(parent, "viewer") and parent.viewer:
                parent.viewer.set_molecule(zatoms, parent.coords)
            if hasattr(parent, "table_widget") and parent.table_widget:
                parent.table_widget.load_zatoms(zatoms)
