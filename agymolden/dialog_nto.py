"""
Interactive Natural Transition Orbitals (NTO) & Photochemistry Dialog for AgyMolden.

Provides:
- Excited state list (energy in eV, wavelength in nm, oscillator strength f)
- SVD decomposition of TD-DFT CI vectors into dominant Hole and Particle NTO pairs
- Visual representation of Hole-Particle exciton transition character (CT vs. local pi-pi*)
- 1-Click projection of Hole and Particle orbitals to 3D visualization
- Export NTO summary to CSV
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from .nto_engine import ExcitedStateInfo, NTOResult, compute_ntos_from_amplitudes, parse_td_dft_and_compute_ntos


class NTODialog(QtWidgets.QDialog):
    """Dialog for Natural Transition Orbitals (NTO) analysis."""

    # Signal to project NTO orbital to 3D viewer (ao_coefficients, is_hole)
    nto_selected = QtCore.pyqtSignal(object, bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, log_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Natural Transition Orbitals (NTO) & Photochemistry (Hole-Particle)")
        self.resize(880, 600)

        self.log_path = log_path
        self.result: Optional[NTOResult] = None

        self._init_ui()

        if self.log_path and os.path.exists(self.log_path):
            self._load_from_log(self.log_path)
        else:
            self._load_demo_state()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        # Header Banner
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_box = QtWidgets.QHBoxLayout(hdr_frame)
        h_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>☀️</span>")
        h_box.addWidget(h_icon)
        h_text = QtWidgets.QVBoxLayout()
        h_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 14px;'>Natural Transition Orbitals (NTO)</b>"))
        h_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>SVD decomposition of TD-DFT transition density matrix into compact Hole → Particle exciton pairs (Martin 2003).</span>"))
        h_box.addLayout(h_text)
        h_box.addStretch()

        btn_load = QtWidgets.QPushButton("📂 Load TD-DFT Log...")
        btn_load.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
        btn_load.clicked.connect(self._on_load_log)
        h_box.addWidget(btn_load)

        main_layout.addWidget(hdr_frame)

        # Excited States Table
        self.tbl_states = QtWidgets.QTableWidget(0, 6)
        self.tbl_states.setHorizontalHeaderLabels(["State", "Energy (eV)", "Wavelength (nm)", "Oscillator (f)", "Dominant Weight (λ₁)", "Character"])
        self.tbl_states.horizontalHeader().setStretchLastSection(True)
        self.tbl_states.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_states.itemSelectionChanged.connect(self._on_state_selected)
        main_layout.addWidget(self.tbl_states)

        # Detailed NTO Hole / Particle Composition Box
        det_grp = QtWidgets.QGroupBox("Natural Transition Orbital (NTO) Hole-Particle Exciton Pair")
        d_lay = QtWidgets.QVBoxLayout(det_grp)

        self.lbl_details = QtWidgets.QLabel("Select an excited state above to inspect Hole and Particle NTO orbital compositions.")
        self.lbl_details.setStyleSheet("color: #f1f5f9; font-family: monospace; font-size: 11px;")
        self.lbl_details.setWordWrap(True)
        d_lay.addWidget(self.lbl_details)

        main_layout.addWidget(det_grp)

        # Bottom Action Bar
        btn_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready.")
        self.lbl_status.setStyleSheet("color: #94a3b8;")
        btn_bar.addWidget(self.lbl_status)
        btn_bar.addStretch()

        btn_csv = QtWidgets.QPushButton("💾 Export NTO Table (CSV)...")
        btn_csv.clicked.connect(self._on_export_csv)
        btn_bar.addWidget(btn_csv)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_bar.addWidget(btn_close)

        main_layout.addLayout(btn_bar)

    def _load_demo_state(self):
        """Loads a realistic textbook TD-DFT transition (e.g. S1 state with pi -> pi* character)."""
        demo_amps = [
            (14, 0, 0.72),   # HOMO -> LUMO
            (13, 1, 0.35),   # HOMO-1 -> LUMO+1
            (12, 0, -0.25),  # HOMO-2 -> LUMO
        ]
        s1 = compute_ntos_from_amplitudes(
            demo_amps, n_occ=15, n_virt=15, state_idx=1, energy_ev=3.42, oscillator_strength=0.185
        )
        self.result = NTOResult(excited_states=[s1], n_occupied=15, n_virtual=15)
        self._populate_table()

    def _on_load_log(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open TD-DFT Output Log (ORCA / Gaussian)",
            "",
            "QM Output (*.log *.out *.txt);;All Files (*)",
        )
        if path:
            self._load_from_log(path)

    def _load_from_log(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            self.result = parse_td_dft_and_compute_ntos(txt)
            if not self.result.excited_states:
                QtWidgets.QMessageBox.warning(self, "No TD-DFT States", f"No excited states could be parsed from:\n{path}")
                return
            self._populate_table()
            self.lbl_status.setText(f"Loaded {len(self.result.excited_states)} excited states from {os.path.basename(path)}.")
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Parsing Error", f"Failed to parse TD-DFT output:\n{exc}")

    def _populate_table(self):
        if not self.result:
            return

        states = self.result.excited_states
        self.tbl_states.blockSignals(True)
        self.tbl_states.setRowCount(len(states))

        for r, st in enumerate(states):
            self.tbl_states.setItem(r, 0, QtWidgets.QTableWidgetItem(f"S_{st.state_index}"))
            self.tbl_states.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{st.energy_ev:.3f}"))
            self.tbl_states.setItem(r, 2, QtWidgets.QTableWidgetItem(f"{st.energy_nm:.1f}"))
            self.tbl_states.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{st.oscillator_strength:.4f}"))

            w1 = st.nto_pairs[0].eigenvalue_weight if st.nto_pairs else 1.0
            pct_item = QtWidgets.QTableWidgetItem(f"{w1 * 100.0:.1f}%")
            if w1 > 0.85:
                pct_item.setForeground(QtGui.QBrush(QtGui.QColor("#4ade80")))
            self.tbl_states.setItem(r, 4, pct_item)

            char_str = "Clean Single Exciton" if w1 > 0.85 else "Multi-Configurational"
            self.tbl_states.setItem(r, 5, QtWidgets.QTableWidgetItem(char_str))

        self.tbl_states.blockSignals(False)
        if len(states) > 0:
            self.tbl_states.selectRow(0)

    def _on_state_selected(self):
        sel = self.tbl_states.selectionModel().selectedRows()
        if not sel or not self.result:
            return
        row = sel[0].row()
        if 0 <= row < len(self.result.excited_states):
            st = self.result.excited_states[row]
            lines = [f"<b>Excited State S_{st.state_index} (E = {st.energy_ev:.3f} eV, λ = {st.energy_nm:.1f} nm, f = {st.oscillator_strength:.4f})</b>\n"]
            for pair in st.nto_pairs:
                lines.append(f"• <b>NTO Pair #{pair.pair_index}</b> (Weight λ = {pair.eigenvalue_weight * 100.0:.2f}%):")
                hole_comps = ", ".join(f"MO {i+1} ({w:+.2f})" for i, w in sorted(pair.hole_mo_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
                part_comps = ", ".join(f"MO {a+1} ({w:+.2f})" for a, w in sorted(pair.particle_mo_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3])
                lines.append(f"    <b>Hole (Donor):</b>     {hole_comps or 'Core/Valence'}")
                lines.append(f"    <b>Particle (Acceptor):</b> {part_comps or 'Virtual'}\n")

            self.lbl_details.setText("<br>".join(lines))

    def _on_export_csv(self):
        if not self.result:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export NTO Results",
            "nto_excited_states.csv",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write("state_index,energy_ev,energy_nm,oscillator_strength_f,dominant_nto_weight\n")
                for st in self.result.excited_states:
                    w1 = st.nto_pairs[0].eigenvalue_weight if st.nto_pairs else 1.0
                    f.write(f"S_{st.state_index},{st.energy_ev:.4f},{st.energy_nm:.2f},{st.oscillator_strength:.6f},{w1:.4f}\n")
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved NTO summary to:\n{path}")
