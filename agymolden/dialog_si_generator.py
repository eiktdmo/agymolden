"""
Interactive 1-Click Automated Supporting Information (SI) & Computational Methods Dialog.

Provides:
- Batch collection of optimized geometries, energies (E, ZPE, H, G), and vibrational frequencies
- Real-time generated "Computational Methods" paragraph customized to user parameters
- Live preview of publication-ready LaTeX (.tex) and Markdown/Word tables
- 1-Click Export to LaTeX project or Markdown/HTML file
"""

from __future__ import annotations
import logging

import os
from typing import List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
import numpy as np

from .si_generator import (
    SISpecies,
    format_xyz_block,
    generate_computational_methods_paragraph,
    generate_latex_si_document,
    generate_markdown_si_document,
)


class SIGeneratorDialog(QtWidgets.QDialog):
    """Dialog for automated Supporting Information (SI) synthesis."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_species: Optional[List[SISpecies]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Automated Supporting Information (SI) & Methods Synthesizer")
        self.resize(1020, 720)

        self.species_list: List[SISpecies] = initial_species or []

        self._init_ui()
        self._update_methods_text()
        self._update_preview()

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Panel: Structure List & Method Parameters
        left_panel = QtWidgets.QWidget()
        left_panel.setFixedWidth(440)
        left_lay = QtWidgets.QVBoxLayout(left_panel)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(10)

        # Header Info Card
        hdr_frame = QtWidgets.QFrame()
        hdr_frame.setStyleSheet("background-color: #0f172a; border-radius: 6px; padding: 10px;")
        h_box = QtWidgets.QHBoxLayout(hdr_frame)
        h_icon = QtWidgets.QLabel("<span style='font-size: 26px;'>📄</span>")
        h_box.addWidget(h_icon)
        h_text = QtWidgets.QVBoxLayout()
        h_text.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 13px;'>Publication SI Synthesizer</b>"))
        h_text.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 11px;'>1-Click LaTeX & Markdown SI with XYZ blocks</span>"))
        h_box.addLayout(h_text)
        left_lay.addWidget(hdr_frame)

        # Method Parameters Group
        m_grp = QtWidgets.QGroupBox("Methodology & Software Citations")
        m_form = QtWidgets.QFormLayout(m_grp)
        m_form.setSpacing(6)

        self.cbo_program = QtWidgets.QComboBox()
        self.cbo_program.addItems(["ORCA 5.0", "Gaussian 16", "Q-Chem 6.0", "PySCF 2.7"])
        self.cbo_program.currentIndexChanged.connect(self._update_methods_text)
        m_form.addRow("Program:", self.cbo_program)

        self.edit_functional = QtWidgets.QLineEdit("ωB97X-D")
        self.edit_functional.textChanged.connect(self._update_methods_text)
        m_form.addRow("Functional:", self.edit_functional)

        self.edit_basis = QtWidgets.QLineEdit("def2-TZVP")
        self.edit_basis.textChanged.connect(self._update_methods_text)
        m_form.addRow("Basis Set:", self.edit_basis)

        self.edit_solvation = QtWidgets.QLineEdit("CPCM")
        self.edit_solvation.textChanged.connect(self._update_methods_text)
        m_form.addRow("Solvation Model:", self.edit_solvation)

        self.edit_solvent = QtWidgets.QLineEdit("dichloromethane")
        self.edit_solvent.textChanged.connect(self._update_methods_text)
        m_form.addRow("Solvent:", self.edit_solvent)

        left_lay.addWidget(m_grp)

        # Species / Structures Table
        s_grp = QtWidgets.QGroupBox("Included Structures & Stationary Points")
        s_lay = QtWidgets.QVBoxLayout(s_grp)

        self.tbl_species = QtWidgets.QTableWidget(0, 4)
        self.tbl_species.setHorizontalHeaderLabels(["Name", "Atoms", "E (Ha)", "G (Ha)"])
        self.tbl_species.horizontalHeader().setStretchLastSection(True)
        s_lay.addWidget(self.tbl_species)

        btn_row = QtWidgets.QHBoxLayout()
        btn_add_file = QtWidgets.QPushButton("📂 Add QM Log / Molden...")
        btn_add_file.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 8px;")
        btn_add_file.clicked.connect(self._on_add_file)
        btn_row.addWidget(btn_add_file)

        btn_rem = QtWidgets.QPushButton("➖ Remove")
        btn_rem.clicked.connect(self._on_remove_species)
        btn_row.addWidget(btn_rem)
        s_lay.addLayout(btn_row)

        left_lay.addWidget(s_grp)

        # Action Buttons
        btn_export_tex = QtWidgets.QPushButton("💾 Export Standalone LaTeX SI (.tex)...")
        btn_export_tex.clicked.connect(self._on_export_tex)
        left_lay.addWidget(btn_export_tex)

        btn_export_md = QtWidgets.QPushButton("💾 Export Markdown / Word SI (.md)...")
        btn_export_md.clicked.connect(self._on_export_md)
        left_lay.addWidget(btn_export_md)

        main_layout.addWidget(left_panel)

        # Right Panel: Live Document Preview
        right_panel = QtWidgets.QWidget()
        right_lay = QtWidgets.QVBoxLayout(right_panel)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(6)

        tab_preview = QtWidgets.QTabWidget()

        # Tab 1: Computational Methods Paragraph
        tab_methods = QtWidgets.QWidget()
        tm_lay = QtWidgets.QVBoxLayout(tab_methods)
        self.txt_methods = QtWidgets.QTextEdit()
        self.txt_methods.setStyleSheet("font-family: monospace; font-size: 11px;")
        tm_lay.addWidget(self.txt_methods)
        tab_preview.addTab(tab_methods, "Methods Paragraph")

        # Tab 2: LaTeX Preview
        tab_latex = QtWidgets.QWidget()
        tl_lay = QtWidgets.QVBoxLayout(tab_latex)
        self.txt_latex = QtWidgets.QTextEdit()
        self.txt_latex.setReadOnly(True)
        self.txt_latex.setStyleSheet("font-family: monospace; font-size: 11px;")
        tl_lay.addWidget(self.txt_latex)
        tab_preview.addTab(tab_latex, "LaTeX SI (.tex)")

        # Tab 3: Markdown Preview
        tab_md = QtWidgets.QWidget()
        tmd_lay = QtWidgets.QVBoxLayout(tab_md)
        self.txt_md = QtWidgets.QTextEdit()
        self.txt_md.setReadOnly(True)
        self.txt_md.setStyleSheet("font-family: monospace; font-size: 11px;")
        tmd_lay.addWidget(self.txt_md)
        tab_preview.addTab(tab_md, "Markdown (.md)")

        right_lay.addWidget(tab_preview)
        main_layout.addWidget(right_panel)

    def _update_methods_text(self):
        txt = generate_computational_methods_paragraph(
            program=self.cbo_program.currentText(),
            functional=self.edit_functional.text(),
            basis_set=self.edit_basis.text(),
            solvation=self.edit_solvation.text() if self.edit_solvation.text() else None,
            solvent=self.edit_solvent.text() if self.edit_solvent.text() else None,
        )
        self.txt_methods.setText(txt)
        self._update_preview()

    def _on_add_file(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select QM Calculation Files",
            "",
            "QM Files (*.log *.out *.txt *.molden);;All Files (*)",
        )
        if not paths:
            return

        from .qm_log_parser import parse_qm_log

        for p in paths:
            try:
                base = os.path.splitext(os.path.basename(p))[0]
                res = parse_qm_log(p)
                coords = res.trajectory_coords[-1] if res.trajectory_coords else np.empty((0, 3))
                e_ha = res.trajectory_energies[-1] / 627.5095 if res.trajectory_energies else None
                freqs = res.vibrational_analysis.frequencies if res.vibrational_analysis else None

                sp = SISpecies(
                    name=base,
                    symbols=res.symbols,
                    coords=coords,
                    electronic_energy_ha=e_ha,
                    zpe_ha=res.zpe_ha,
                    enthalpy_ha=res.enthalpy_ha,
                    gibbs_ha=res.gibbs_ha,
                    lowest_frequencies_cm1=freqs,
                    program_name=res.program_name or "ORCA",
                )
                self.species_list.append(sp)
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        self._refresh_table()
        self._update_preview()

    def _on_remove_species(self):
        sel = self.tbl_species.selectionModel().selectedRows()
        if sel:
            for s in sorted(sel, key=lambda x: x.row(), reverse=True):
                del self.species_list[s.row()]
            self._refresh_table()
            self._update_preview()

    def _refresh_table(self):
        self.tbl_species.blockSignals(True)
        self.tbl_species.setRowCount(len(self.species_list))
        for r, sp in enumerate(self.species_list):
            self.tbl_species.setItem(r, 0, QtWidgets.QTableWidgetItem(sp.name))
            self.tbl_species.setItem(r, 1, QtWidgets.QTableWidgetItem(str(len(sp.symbols))))
            e_str = f"{sp.electronic_energy_ha:.4f}" if sp.electronic_energy_ha is not None else "-"
            g_str = f"{sp.gibbs_ha:.4f}" if sp.gibbs_ha is not None else "-"
            self.tbl_species.setItem(r, 2, QtWidgets.QTableWidgetItem(e_str))
            self.tbl_species.setItem(r, 3, QtWidgets.QTableWidgetItem(g_str))
        self.tbl_species.blockSignals(False)

    def _update_preview(self):
        methods = self.txt_methods.toPlainText()
        tex_doc = generate_latex_si_document(self.species_list, methods)
        md_doc = generate_markdown_si_document(self.species_list, methods)

        self.txt_latex.setText(tex_doc)
        self.txt_md.setText(md_doc)

    def _on_export_tex(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export LaTeX Supporting Information",
            "supporting_information.tex",
            "LaTeX Files (*.tex);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt_latex.toPlainText())
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved LaTeX SI to:\n{path}")

    def _on_export_md(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Markdown Supporting Information",
            "supporting_information.md",
            "Markdown Files (*.md);;All Files (*)",
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt_md.toPlainText())
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Saved Markdown SI to:\n{path}")
