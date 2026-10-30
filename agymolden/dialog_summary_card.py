"""
Interactive 1-Page Quantum Chemistry Calculation Summary Dialog.
Provides a clean, publication-ready summary of calculation results:
- Final Electronic Energy (Hartree, eV, kcal/mol)
- Zero-Point Energy, Enthalpy & Gibbs Free Energy (if freq job)
- Dipole Moment & Molecular Symmetry Point Group
- Frontier Orbitals: HOMO, LUMO, and Band Gap (Eg) with visual energy badge
- Quick Action buttons (View Orbitals, View IR Vibrations, View Full Log, Export Summary)
"""

import os
import html
from typing import Optional, Dict, Any, List
from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from .qm_log_parser import QMLogParseResult


class QMSummaryCardDialog(QtWidgets.QDialog):
    """Publication-ready 1-page summary card for completed QM jobs."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        parse_result: QMLogParseResult,
        file_path: str,
        point_group: str = "",
        companion_files: Optional[Dict[str, str]] = None,
    ):
        super().__init__(parent)
        self.parse_result = parse_result
        self.file_path = file_path
        self.point_group = point_group or getattr(parse_result, "point_group", "C1")
        self.companion_files = companion_files or {}

        filename = os.path.basename(file_path)
        self.setWindowTitle(f"Calculation Result Summary: {filename}")
        self.resize(680, 560)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        filename = os.path.basename(self.file_path)
        program = self.parse_result.program_name or "Quantum Chemistry"

        # 1. Header Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 10px;")
        h_layout = QtWidgets.QHBoxLayout(header)
        h_layout.setContentsMargins(8, 4, 8, 4)

        icon_lbl = QtWidgets.QLabel("⚛️")
        icon_lbl.setStyleSheet("font-size: 28px;")
        h_layout.addWidget(icon_lbl)

        title_col = QtWidgets.QVBoxLayout()
        title_lbl = QtWidgets.QLabel(f"<span style='font-size: 16px; font-weight: bold; color: #f8fafc;'>{filename}</span>")
        sub_lbl = QtWidgets.QLabel(f"<span style='color: #94a3b8; font-size: 11px;'>Engine: <b>{program}</b> | Status: <b style='color: #10b981;'>Completed Successfully</b></span>")
        title_col.addWidget(title_lbl)
        title_col.addWidget(sub_lbl)
        h_layout.addLayout(title_col, stretch=1)

        # Symmetry badge
        sym_box = QtWidgets.QLabel(f"Point Group<br><b style='font-size: 14px;'>{self.point_group}</b>")
        sym_box.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        sym_box.setStyleSheet("background-color: #334155; color: #38bdf8; border-radius: 4px; padding: 4px 10px;")
        h_layout.addWidget(sym_box)

        layout.addWidget(header)

        # 2. Main Content Splitter / Grid
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_w = QtWidgets.QWidget()
        c_layout = QtWidgets.QVBoxLayout(content_w)
        c_layout.setSpacing(10)

        # Section A: Energy & Thermodynamics
        grp_energy = QtWidgets.QGroupBox("⚡ Energy & Thermodynamics")
        form_e = QtWidgets.QFormLayout(grp_energy)
        form_e.setSpacing(8)

        # Final Energy
        final_e_ha = None
        if self.parse_result.trajectory_energies:
            final_e_ha = self.parse_result.trajectory_energies[-1]

        if final_e_ha is not None:
            e_ev = final_e_ha * 27.211386
            e_kcal = final_e_ha * 627.509474
            lbl_e = QtWidgets.QLabel(
                f"<b>{final_e_ha:.8f} Hartree</b> &nbsp;&nbsp;"
                f"<span style='color: #64748b;'>({e_ev:.3f} eV &nbsp;|&nbsp; {e_kcal:.2f} kcal/mol)</span>"
            )
            form_e.addRow("Final Total Energy (E):", lbl_e)

        # Optimization Steps Count
        n_steps = len(self.parse_result.trajectory_coords)
        if n_steps > 1:
            lbl_steps = QtWidgets.QLabel(f"<b>{n_steps} geometry cycles</b> (Converged: {self.parse_result.converged})")
            form_e.addRow("Optimization Trajectory:", lbl_steps)

        # Thermochemistry (ZPE, Enthalpy, Gibbs)
        if self.parse_result.zpe_ha is not None:
            zpe_kcal = self.parse_result.zpe_ha * 627.509474
            form_e.addRow("Zero-Point Energy (ZPE):", QtWidgets.QLabel(f"{self.parse_result.zpe_ha:.6f} Ha ({zpe_kcal:.2f} kcal/mol)"))
        if self.parse_result.enthalpy_ha is not None:
            form_e.addRow("Total Enthalpy (H):", QtWidgets.QLabel(f"{self.parse_result.enthalpy_ha:.8f} Ha"))
        if self.parse_result.gibbs_ha is not None:
            form_e.addRow("Gibbs Free Energy (G):", QtWidgets.QLabel(f"<b>{self.parse_result.gibbs_ha:.8f} Ha</b>"))

        c_layout.addWidget(grp_energy)

        # Section B: Electronic Structure & Frontier Orbitals (HOMO / LUMO)
        grp_frontier = QtWidgets.QGroupBox("🌐 Frontier Orbitals & Electronic Properties")
        form_f = QtWidgets.QFormLayout(grp_frontier)
        form_f.setSpacing(8)

        # Dipole moment
        if self.parse_result.dipole_debye is not None:
            lbl_dip = QtWidgets.QLabel(f"<b>{self.parse_result.dipole_debye:.4f} Debye</b>")
            form_f.addRow("Dipole Moment (μ):", lbl_dip)

        # HOMO / LUMO / Gap
        h_ev = self.parse_result.homo_ev
        l_ev = self.parse_result.lumo_ev
        if h_ev is not None and l_ev is not None:
            gap = abs(l_ev - h_ev)
            lbl_mo = QtWidgets.QLabel(
                f"<span style='color: #2563eb; font-weight: bold;'>HOMO: {h_ev:.3f} eV</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<span style='color: #dc2626; font-weight: bold;'>LUMO: {l_ev:.3f} eV</span> &nbsp;&nbsp;|&nbsp;&nbsp; "
                f"<span style='background-color: #0284c7; color: white; border-radius: 3px; padding: 2px 6px; font-weight: bold;'>ΔEg = {gap:.3f} eV</span>"
            )
            form_f.addRow("Frontier Orbitals:", lbl_mo)
        elif h_ev is not None:
            form_f.addRow("HOMO Energy:", QtWidgets.QLabel(f"{h_ev:.3f} eV"))

        # Vibrational Frequencies
        vib = self.parse_result.vibrational_analysis
        if vib and vib.frequencies:
            n_modes = len(vib.frequencies)
            min_freq = min(vib.frequencies)
            max_freq = max(vib.frequencies)
            has_imag = any(f < -0.1 for f in vib.frequencies)
            imag_txt = "<span style='color: #ef4444; font-weight: bold;'> (⚠️ Imaginary frequencies detected - Transition State)</span>" if has_imag else " (All real - Local Minimum)"
            lbl_vib = QtWidgets.QLabel(f"<b>{n_modes} normal modes</b> [{min_freq:.1f} to {max_freq:.1f} cm⁻¹]{imag_txt}")
            form_f.addRow("Vibrational Analysis:", lbl_vib)

        # Number of Atoms
        n_atoms = len(self.parse_result.symbols)
        atom_counts = {}
        for s in self.parse_result.symbols:
            atom_counts[s] = atom_counts.get(s, 0) + 1
        formula = "".join(f"{s}{atom_counts[s] if atom_counts[s] > 1 else ''}" for s in sorted(atom_counts.keys()))
        form_f.addRow("Molecular Formula:", QtWidgets.QLabel(f"<b>{formula}</b> ({n_atoms} atoms)"))

        c_layout.addWidget(grp_frontier)

        scroll.setWidget(content_w)
        layout.addWidget(scroll, stretch=1)

        # 3. Action Buttons Toolbar
        act_row = QtWidgets.QHBoxLayout()
        act_row.setSpacing(8)

        # Action: View Vibrations (if freq job)
        if vib and vib.frequencies:
            btn_view_vib = QtWidgets.QPushButton("🎵 View IR Spectrum & Modes")
            btn_view_vib.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 12px;")
            btn_view_vib.clicked.connect(self._on_view_vibrations)
            act_row.addWidget(btn_view_vib)

        # Action: View Molecular Orbitals
        btn_view_mo = QtWidgets.QPushButton("🌐 View 3D Orbitals")
        btn_view_mo.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 12px;")
        btn_view_mo.clicked.connect(self._on_view_orbitals)
        act_row.addWidget(btn_view_mo)

        # Action: Copy Summary to Clipboard
        btn_copy = QtWidgets.QPushButton("📋 Copy Summary")
        btn_copy.clicked.connect(self._on_copy_summary)
        act_row.addWidget(btn_copy)

        act_row.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        act_row.addWidget(btn_close)

        layout.addLayout(act_row)

    def _on_view_vibrations(self):
        """Trigger vibrational spectrum widget in parent MainWindow."""
        p = self.parent()
        if p and self.parse_result.vibrational_analysis:
            p.vibrational_analysis = self.parse_result.vibrational_analysis
            if hasattr(p, "tabs"):
                idx = p.tabs.indexOf(getattr(p, "tab_vibrations", None))
                if idx >= 0:
                    p.tabs.setCurrentIndex(idx)
            self.accept()

    def _on_view_orbitals(self):
        """Open Volumetric / Orbitals dialog in parent MainWindow."""
        p = self.parent()
        if p:
            self.accept()
            if hasattr(p, "tabs"):
                idx = p.tabs.indexOf(getattr(p, "tab_orbitals", None))
                if idx >= 0:
                    p.tabs.setCurrentIndex(idx)
            active_m = getattr(p, "_active_molden_path", None)
            if hasattr(p, "_on_open_volumetric_dialog"):
                p._on_open_volumetric_dialog()
                if active_m and os.path.isfile(active_m) and hasattr(p, "_volumetric_dlg") and p._volumetric_dlg:
                    p._volumetric_dlg.load_molden_path(active_m)

    def _on_copy_summary(self):
        """Copy formatted text summary to clipboard."""
        filename = os.path.basename(self.file_path)
        lines = [
            f"Calculation Summary: {filename}",
            f"Engine: {self.parse_result.program_name}",
            f"Symmetry: {self.point_group}",
        ]
        if self.parse_result.trajectory_energies:
            lines.append(f"Total Energy: {self.parse_result.trajectory_energies[-1]:.8f} Hartree")
        if self.parse_result.dipole_debye is not None:
            lines.append(f"Dipole Moment: {self.parse_result.dipole_debye:.4f} Debye")
        if self.parse_result.homo_ev is not None and self.parse_result.lumo_ev is not None:
            lines.append(f"HOMO: {self.parse_result.homo_ev:.3f} eV")
            lines.append(f"LUMO: {self.parse_result.lumo_ev:.3f} eV")
            lines.append(f"Gap (Eg): {abs(self.parse_result.lumo_ev - self.parse_result.homo_ev):.3f} eV")

        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
        QtWidgets.QMessageBox.information(self, "Copied", "Summary copied to clipboard!")
