"""
Cross-Engine Calculation Benchmark & Result Comparison Dialog.

Provides multi-job side-by-side comparison across different QM codes (Gaussian, ORCA,
PySCF, openLOWDIN, newBRORSEN, xTB) or different levels of theory / basis sets:
- Energetics comparison: Final Energy (Ha), ΔE relative to minimum (kcal/mol and eV).
- Frontier orbitals: HOMO, LUMO, HOMO–LUMO gap (eV), and dipole moment (Debye).
- Optimization diagnostics: convergence status and number of geometry steps.
- Pairwise structural similarity: Kabsch SVD superposition all-atom & heavy-atom RMSD matrix.
- 1-Click export to publication-ready CSV and Markdown comparison tables.
- 1-Click structural superposition loaded directly into the 3D viewport.
"""

from dataclasses import dataclass, field
import os
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .qm_log_parser import parse_qm_logfile, QMLogParseResult
from .alignment import calculate_kabsch_alignment
from .elements import ELEMENT_DATA

HARTREE_TO_KCAL = 627.509474
HARTREE_TO_EV = 27.211386245988


@dataclass
class BenchmarkEntry:
    filepath: str
    label: str
    program: str
    symbols: List[str]
    coords: np.ndarray
    final_energy_ha: Optional[float] = None
    delta_e_kcal: float = 0.0
    delta_e_ev: float = 0.0
    converged: bool = True
    n_opt_steps: int = 1
    homo_ev: Optional[float] = None
    lumo_ev: Optional[float] = None
    gap_ev: Optional[float] = None
    dipole_debye: Optional[float] = None
    zpe_ha: Optional[float] = None
    gibbs_ha: Optional[float] = None
    frequencies: List[float] = field(default_factory=list)
    raw_result: Optional[QMLogParseResult] = None


class BenchmarkComparisonDialog(QtWidgets.QDialog):
    """Dialog for comparing multiple QM calculation outputs side-by-side."""

    structure_selected = QtCore.pyqtSignal(list, np.ndarray, str)

    def __init__(self, parent=None, initial_files: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("⚖️ Cross-Engine Calculation Benchmark & Result Comparison")
        self.resize(1000, 680)

        self.entries: List[BenchmarkEntry] = []
        self._init_ui()

        if initial_files:
            self.load_files(initial_files)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Top Action Bar
        top_bar = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("➕ Add Calculation Logs...")
        btn_add.setStyleSheet("font-weight: bold; padding: 6px 12px;")
        btn_add.clicked.connect(self._on_add_files_clicked)
        top_bar.addWidget(btn_add)

        btn_remove = QtWidgets.QPushButton("🗑️ Remove Selected")
        btn_remove.clicked.connect(self._on_remove_selected_clicked)
        top_bar.addWidget(btn_remove)

        btn_clear = QtWidgets.QPushButton("Clear All")
        btn_clear.clicked.connect(self._on_clear_clicked)
        top_bar.addWidget(btn_clear)

        top_bar.addStretch()

        btn_export_csv = QtWidgets.QPushButton("📊 Export CSV")
        btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        top_bar.addWidget(btn_export_csv)

        btn_export_md = QtWidgets.QPushButton("📝 Export Markdown Table")
        btn_export_md.clicked.connect(self._on_export_md_clicked)
        top_bar.addWidget(btn_export_md)

        layout.addLayout(top_bar)

        # Tab widget: Tab 1: Energetics & Properties, Tab 2: Pairwise RMSD Matrix, Tab 3: Vibrational Frequencies
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, stretch=1)

        # --- Tab 1: Energetics & Electronic Properties ---
        tab_energetics = QtWidgets.QWidget()
        t1_layout = QtWidgets.QVBoxLayout(tab_energetics)
        t1_layout.setContentsMargins(6, 6, 6, 6)

        self.table_energetics = QtWidgets.QTableWidget()
        self.table_energetics.setAlternatingRowColors(True)
        self.table_energetics.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_energetics.setColumnCount(11)
        self.table_energetics.setHorizontalHeaderLabels([
            "Calculation / Log",
            "Program",
            "Final Energy (Ha)",
            "ΔE (kcal/mol)",
            "ΔE (eV)",
            "Status",
            "Opt Steps",
            "HOMO (eV)",
            "LUMO (eV)",
            "ΔE_gap (eV)",
            "Dipole (D)",
        ])
        self.table_energetics.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table_energetics.itemSelectionChanged.connect(self._on_table_selection_changed)
        t1_layout.addWidget(self.table_energetics)

        # Button to view selected structure in 3D
        row_btn_layout = QtWidgets.QHBoxLayout()
        self.btn_load_viewport = QtWidgets.QPushButton("👁️ Load Selected Structure into 3D Viewport")
        self.btn_load_viewport.setEnabled(False)
        self.btn_load_viewport.clicked.connect(self._on_load_viewport_clicked)
        row_btn_layout.addWidget(self.btn_load_viewport)
        row_btn_layout.addStretch()
        t1_layout.addLayout(row_btn_layout)

        self.tabs.addTab(tab_energetics, "⚡ Energetics & Properties")

        # --- Tab 2: Pairwise SVD Kabsch RMSD Matrix ---
        tab_rmsd = QtWidgets.QWidget()
        t2_layout = QtWidgets.QVBoxLayout(tab_rmsd)
        t2_layout.setContentsMargins(6, 6, 6, 6)

        lbl_rmsd_info = QtWidgets.QLabel(
            "<b>Pairwise Kabsch SVD Structural RMSD Matrix (Å):</b> "
            "Values indicate geometric deviation after optimal 3D rigid-body superposition. "
            "Upper triangle = All-atom RMSD, Lower triangle = Heavy-atom RMSD."
        )
        lbl_rmsd_info.setWordWrap(True)
        t2_layout.addWidget(lbl_rmsd_info)

        self.table_rmsd = QtWidgets.QTableWidget()
        self.table_rmsd.setAlternatingRowColors(True)
        t2_layout.addWidget(self.table_rmsd)

        self.tabs.addTab(tab_rmsd, "📐 Structural RMSD Matrix")

        # Status Bar
        self.lbl_status = QtWidgets.QLabel("Ready. Add 2 or more calculation logs to benchmark.")
        layout.addWidget(self.lbl_status)

    def load_files(self, paths: List[str]):
        """Parse and append QM output logs."""
        new_count = 0
        for p in paths:
            if not os.path.isfile(p):
                continue
            res = parse_qm_logfile(p)
            if res and res.symbols and len(res.trajectory_coords) > 0:
                coords = res.trajectory_coords[-1]
                final_e = res.trajectory_energies[-1] if res.trajectory_energies else None
                gap = (res.lumo_ev - res.homo_ev) if (res.homo_ev is not None and res.lumo_ev is not None) else None
                freqs = list(res.vibrational_analysis.frequencies) if (res.vibrational_analysis and getattr(res.vibrational_analysis, "frequencies", None) is not None) else []
                entry = BenchmarkEntry(
                    filepath=os.path.abspath(p),
                    label=os.path.basename(p),
                    program=res.program_name or "Unknown",
                    symbols=res.symbols,
                    coords=coords,
                    final_energy_ha=final_e,
                    converged=res.converged,
                    n_opt_steps=len(res.trajectory_coords),
                    homo_ev=res.homo_ev,
                    lumo_ev=res.lumo_ev,
                    gap_ev=gap,
                    dipole_debye=res.dipole_debye,
                    zpe_ha=res.zpe_ha,
                    gibbs_ha=res.gibbs_ha,
                    frequencies=freqs,
                    raw_result=res,
                )
                self.entries.append(entry)
                new_count += 1

        self._recompute_and_refresh()
        self.lbl_status.setText(f"Loaded {len(self.entries)} calculations for benchmark.")

    def _recompute_and_refresh(self):
        """Compute relative energies and refresh all tables."""
        valid_energies = [e.final_energy_ha for e in self.entries if e.final_energy_ha is not None]
        min_e = min(valid_energies) if valid_energies else None

        for e in self.entries:
            if e.final_energy_ha is not None and min_e is not None:
                diff_ha = e.final_energy_ha - min_e
                e.delta_e_kcal = diff_ha * HARTREE_TO_KCAL
                e.delta_e_ev = diff_ha * HARTREE_TO_EV
            else:
                e.delta_e_kcal = 0.0
                e.delta_e_ev = 0.0

        self._refresh_energetics_table()
        self._refresh_rmsd_table()

    def _refresh_energetics_table(self):
        self.table_energetics.setRowCount(len(self.entries))
        for r, e in enumerate(self.entries):
            item_lbl = QtWidgets.QTableWidgetItem(e.label)
            item_lbl.setToolTip(e.filepath)
            self.table_energetics.setItem(r, 0, item_lbl)

            self.table_energetics.setItem(r, 1, QtWidgets.QTableWidgetItem(e.program))

            e_str = f"{e.final_energy_ha:.8f}" if e.final_energy_ha is not None else "N/A"
            item_e = QtWidgets.QTableWidgetItem(e_str)
            item_e.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table_energetics.setItem(r, 2, item_e)

            de_kcal_str = f"+{e.delta_e_kcal:.2f}" if e.delta_e_kcal > 1e-6 else ("0.00" if abs(e.delta_e_kcal) <= 1e-6 else f"{e.delta_e_kcal:.2f}")
            item_de_kcal = QtWidgets.QTableWidgetItem(de_kcal_str)
            item_de_kcal.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if abs(e.delta_e_kcal) <= 1e-6 and e.final_energy_ha is not None:
                item_de_kcal.setForeground(QtGui.QColor("#22c55e"))
                item_de_kcal.setText(f"{de_kcal_str} (min)")
            self.table_energetics.setItem(r, 3, item_de_kcal)

            de_ev_str = f"+{e.delta_e_ev:.3f}" if e.delta_e_ev > 1e-6 else ("0.000" if abs(e.delta_e_ev) <= 1e-6 else f"{e.delta_e_ev:.3f}")
            item_de_ev = QtWidgets.QTableWidgetItem(de_ev_str)
            item_de_ev.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table_energetics.setItem(r, 4, item_de_ev)

            status_str = "✔ Converged" if e.converged else "⚠ Incomplete"
            item_status = QtWidgets.QTableWidgetItem(status_str)
            item_status.setForeground(QtGui.QColor("#22c55e") if e.converged else QtGui.QColor("#ef4444"))
            self.table_energetics.setItem(r, 5, item_status)

            self.table_energetics.setItem(r, 6, QtWidgets.QTableWidgetItem(str(e.n_opt_steps)))

            homo_str = f"{e.homo_ev:+.2f}" if e.homo_ev is not None else "N/A"
            self.table_energetics.setItem(r, 7, QtWidgets.QTableWidgetItem(homo_str))

            lumo_str = f"{e.lumo_ev:+.2f}" if e.lumo_ev is not None else "N/A"
            self.table_energetics.setItem(r, 8, QtWidgets.QTableWidgetItem(lumo_str))

            gap_str = f"{e.gap_ev:.2f}" if e.gap_ev is not None else "N/A"
            item_gap = QtWidgets.QTableWidgetItem(gap_str)
            item_gap.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table_energetics.setItem(r, 9, item_gap)

            dipole_str = f"{e.dipole_debye:.3f}" if e.dipole_debye is not None else "N/A"
            self.table_energetics.setItem(r, 10, QtWidgets.QTableWidgetItem(dipole_str))

    def _refresh_rmsd_table(self):
        n = len(self.entries)
        self.table_rmsd.setRowCount(n)
        self.table_rmsd.setColumnCount(n)
        labels = [e.label for e in self.entries]
        self.table_rmsd.setHorizontalHeaderLabels(labels)
        self.table_rmsd.setVerticalHeaderLabels(labels)

        for i in range(n):
            for j in range(n):
                if i == j:
                    item = QtWidgets.QTableWidgetItem("0.0000")
                    item.setForeground(QtGui.QColor("#94a3b8"))
                else:
                    ei = self.entries[i]
                    ej = self.entries[j]
                    if len(ei.symbols) == len(ej.symbols) and ei.symbols == ej.symbols:
                        try:
                            if i < j:
                                # All-atom RMSD (upper triangle)
                                _, rmsd, _, _ = calculate_kabsch_alignment(ei.coords, ej.coords)
                            else:
                                # Heavy-atom RMSD (lower triangle)
                                heavy_mask = np.array([s != "H" for s in ei.symbols], dtype=bool)
                                if np.any(heavy_mask):
                                    _, rmsd, _, _ = calculate_kabsch_alignment(ei.coords[heavy_mask], ej.coords[heavy_mask])
                                else:
                                    _, rmsd, _, _ = calculate_kabsch_alignment(ei.coords, ej.coords)
                            item = QtWidgets.QTableWidgetItem(f"{rmsd:.4f}")
                            if rmsd < 0.05:
                                item.setBackground(QtGui.QColor(34, 197, 94, 45))  # Green: identical
                            elif rmsd > 0.50:
                                item.setBackground(QtGui.QColor(239, 68, 68, 45))  # Red: distinct conformer
                        except Exception:
                            item = QtWidgets.QTableWidgetItem("Err")
                    else:
                        item = QtWidgets.QTableWidgetItem("Mismatch")
                        item.setForeground(QtGui.QColor("#ef4444"))
                item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.table_rmsd.setItem(i, j, item)

    def _on_add_files_clicked(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Select Calculation Output Logs",
            "",
            "QM Logs (*.log *.out *.molden);;All Files (*)",
        )
        if paths:
            self.load_files(paths)

    def _on_remove_selected_clicked(self):
        selected_rows = sorted(set(idx.row() for idx in self.table_energetics.selectedIndexes()), reverse=True)
        for r in selected_rows:
            if 0 <= r < len(self.entries):
                del self.entries[r]
        self._recompute_and_refresh()

    def _on_clear_clicked(self):
        self.entries.clear()
        self._recompute_and_refresh()
        self.lbl_status.setText("Cleared benchmark list.")

    def _on_table_selection_changed(self):
        has_sel = bool(self.table_energetics.selectedIndexes())
        self.btn_load_viewport.setEnabled(has_sel)

    def _on_load_viewport_clicked(self):
        indexes = self.table_energetics.selectedIndexes()
        if not indexes:
            return
        row = indexes[0].row()
        if 0 <= row < len(self.entries):
            e = self.entries[row]
            self.structure_selected.emit(e.symbols, e.coords, e.label)
            self.lbl_status.setText(f"Loaded {e.label} into 3D Viewport.")

    def get_csv_string(self) -> str:
        """Export benchmark table as a standard CSV string."""
        lines = [
            "File,Program,Final_Energy_Ha,Delta_E_kcal_mol,Delta_E_eV,Converged,Opt_Steps,HOMO_eV,LUMO_eV,Gap_eV,Dipole_Debye"
        ]
        for e in self.entries:
            e_ha = f"{e.final_energy_ha:.8f}" if e.final_energy_ha is not None else ""
            homo = f"{e.homo_ev:.4f}" if e.homo_ev is not None else ""
            lumo = f"{e.lumo_ev:.4f}" if e.lumo_ev is not None else ""
            gap = f"{e.gap_ev:.4f}" if e.gap_ev is not None else ""
            dip = f"{e.dipole_debye:.4f}" if e.dipole_debye is not None else ""
            lines.append(
                f'"{e.label}",{e.program},{e_ha},{e.delta_e_kcal:.4f},{e.delta_e_ev:.4f},{e.converged},{e.n_opt_steps},{homo},{lumo},{gap},{dip}'
            )
        return "\n".join(lines) + "\n"

    def get_markdown_table(self) -> str:
        """Export benchmark table as GitHub-flavored Markdown."""
        lines = [
            "| Calculation | Program | Final Energy (Ha) | ΔE (kcal/mol) | ΔE (eV) | Status | Steps | HOMO (eV) | LUMO (eV) | Gap (eV) | Dipole (D) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for e in self.entries:
            e_ha = f"{e.final_energy_ha:.6f}" if e.final_energy_ha is not None else "N/A"
            de_kcal = f"+{e.delta_e_kcal:.2f}" if e.delta_e_kcal > 1e-4 else ("0.00 (min)" if abs(e.delta_e_kcal) <= 1e-4 else f"{e.delta_e_kcal:.2f}")
            de_ev = f"+{e.delta_e_ev:.3f}" if e.delta_e_ev > 1e-4 else ("0.000" if abs(e.delta_e_ev) <= 1e-4 else f"{e.delta_e_ev:.3f}")
            status = "Converged" if e.converged else "Incomplete"
            homo = f"{e.homo_ev:+.2f}" if e.homo_ev is not None else "N/A"
            lumo = f"{e.lumo_ev:+.2f}" if e.lumo_ev is not None else "N/A"
            gap = f"{e.gap_ev:.2f}" if e.gap_ev is not None else "N/A"
            dip = f"{e.dipole_debye:.2f}" if e.dipole_debye is not None else "N/A"
            lines.append(
                f"| **{e.label}** | {e.program} | {e_ha} | {de_kcal} | {de_ev} | {status} | {e.n_opt_steps} | {homo} | {lumo} | {gap} | {dip} |"
            )
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.entries:
            QtWidgets.QMessageBox.warning(self, "No Data", "No calculation entries to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Benchmark to CSV", "benchmark_comparison.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())
            QtWidgets.QMessageBox.information(self, "Exported", f"Successfully exported benchmark CSV to:\n{path}")

    def _on_export_md_clicked(self):
        if not self.entries:
            QtWidgets.QMessageBox.warning(self, "No Data", "No calculation entries to export.")
            return
        md = self.get_markdown_table()
        QtWidgets.QApplication.clipboard().setText(md)
        QtWidgets.QMessageBox.information(
            self, "Copied to Clipboard", "Benchmark Markdown table copied to clipboard for Supporting Information!"
        )
