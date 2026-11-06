"""
Interactive CREST & Conformer Ensemble Explorer Dialog.

Provides:
- Conformer table with Boltzmann populations, relative energies (kcal/mol & kJ/mol),
  and heavy-atom / all-atom Kabsch RMSD superpositions.
- Dynamic temperature slider/spinbox with real-time thermodynamic population updates.
- Interactive 2D Boltzmann population & energy spectrum bar chart.
- Conformer deduplication / greedy RMSD clustering.
- Real-time 3D viewport preview and 1-click adoption into Z-Matrix editor with undo.
- Superimposed ensemble visualization.
- Filtered XYZ and CSV table exports.
- 1-Click QM batch input generation (ORCA / Gaussian) for DFT conformer re-optimization.
"""

import os
from typing import List, Optional, Set
import numpy as np
from PyQt6 import QtWidgets, QtCore, QtGui

from .crest_explorer import (
    ConformerRecord,
    load_conformer_ensemble,
    recalculate_boltzmann,
    recalculate_rmsds,
    cluster_conformers,
    export_ensemble_xyz,
    export_ensemble_csv,
    generate_qm_batch_scripts,
    HARTREE_TO_KCAL,
)
from .alignment import calculate_kabsch_alignment


class BoltzmannBarWidget(QtWidgets.QWidget):
    """Interactive custom 2D bar chart for conformer Boltzmann populations and relative energies."""

    conformer_clicked = QtCore.pyqtSignal(int)  # Emits 1-based conformer index

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.conformers: List[ConformerRecord] = []
        self.selected_index: int = 1
        self.setMinimumHeight(140)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.setMouseTracking(True)
        self._bar_rects: List[Tuple[QtCore.QRectF, int]] = []

    def set_data(self, conformers: List[ConformerRecord], selected_index: int = 1):
        self.conformers = list(conformers)
        self.selected_index = selected_index
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        rect = self.rect()
        is_white = False
        p = self.parent()
        while p is not None:
            if hasattr(p, "styleSheet") and ("#ffffff" in (p.styleSheet() or "") or "#f8fafc" in (p.styleSheet() or "")):
                is_white = True
                break
            p = p.parent()

        bg_col = QtGui.QColor("#ffffff") if is_white else QtGui.QColor("#0f172a")
        border_col = QtGui.QColor("#cbd5e1") if is_white else QtGui.QColor("#334155")
        text_col = QtGui.QColor("#0f172a") if is_white else QtGui.QColor("#e2e8f0")
        subtext_col = QtGui.QColor("#64748b") if is_white else QtGui.QColor("#94a3b8")

        painter.fillRect(rect, bg_col)
        painter.setPen(QtGui.QPen(border_col, 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)

        if not self.conformers:
            painter.setPen(QtGui.QPen(subtext_col))
            painter.setFont(QtGui.QFont("DejaVu Sans", 10, QtGui.QFont.Weight.Medium))
            painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No conformers loaded")
            return

        # Title / axes banner
        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
        painter.setPen(QtGui.QPen(text_col))
        painter.drawText(12, 20, "Boltzmann Population Distribution (%)")

        self._bar_rects.clear()
        n = min(len(self.conformers), 25)  # Display up to top 25 conformers in chart
        plot_left = 16.0
        plot_right = rect.width() - 16.0
        plot_top = 32.0
        plot_bottom = rect.height() - 26.0
        avail_w = plot_right - plot_left
        avail_h = plot_bottom - plot_top

        bar_spacing = max(2.0, min(8.0, avail_w / (n * 4.0)))
        bar_w = max(4.0, (avail_w - (n - 1) * bar_spacing) / n)

        max_pop = max((c.boltzmann_pop for c in self.conformers[:n]), default=100.0)
        max_pop = max(5.0, max_pop)  # Scale ceiling

        for i, conf in enumerate(self.conformers[:n]):
            x = plot_left + i * (bar_w + bar_spacing)
            h = (conf.boltzmann_pop / max_pop) * avail_h
            h = max(3.0, h)
            y = plot_bottom - h

            bar_r = QtCore.QRectF(x, y, bar_w, h)
            self._bar_rects.append((bar_r, conf.index))

            # Color bar according to stability
            is_sel = (conf.index == self.selected_index)
            if is_sel:
                grad = QtGui.QLinearGradient(x, y, x, plot_bottom)
                grad.setColorAt(0.0, QtGui.QColor("#38bdf8"))
                grad.setColorAt(1.0, QtGui.QColor("#0284c7"))
                painter.setBrush(QtGui.QBrush(grad))
                painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 2))
            elif conf.delta_e_kcal < 1.0:
                grad = QtGui.QLinearGradient(x, y, x, plot_bottom)
                grad.setColorAt(0.0, QtGui.QColor("#34d399"))
                grad.setColorAt(1.0, QtGui.QColor("#059669"))
                painter.setBrush(QtGui.QBrush(grad))
                painter.setPen(QtGui.QPen(QtGui.QColor("#10b981"), 1))
            elif conf.delta_e_kcal < 3.0:
                grad = QtGui.QLinearGradient(x, y, x, plot_bottom)
                grad.setColorAt(0.0, QtGui.QColor("#fbbf24"))
                grad.setColorAt(1.0, QtGui.QColor("#d97706"))
                painter.setBrush(QtGui.QBrush(grad))
                painter.setPen(QtGui.QPen(QtGui.QColor("#f59e0b"), 1))
            else:
                grad = QtGui.QLinearGradient(x, y, x, plot_bottom)
                grad.setColorAt(0.0, QtGui.QColor("#94a3b8"))
                grad.setColorAt(1.0, QtGui.QColor("#64748b"))
                painter.setBrush(QtGui.QBrush(grad))
                painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1))

            painter.drawRoundedRect(bar_r, 3, 3)

            # Label top conformers
            if n <= 15 or is_sel:
                painter.setFont(QtGui.QFont("DejaVu Sans", 7, QtGui.QFont.Weight.Bold if is_sel else QtGui.QFont.Weight.Normal))
                painter.setPen(QtGui.QPen(text_col if is_sel else subtext_col))
                painter.drawText(
                    QtCore.QRectF(x - 4, plot_bottom + 4, bar_w + 8, 18),
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    f"#{conf.index}"
                )

        # Baseline axis line
        painter.setPen(QtGui.QPen(border_col, 1))
        painter.drawLine(QtCore.QPointF(plot_left, plot_bottom), QtCore.QPointF(plot_right, plot_bottom))

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            pos = event.position()
            for r, conf_idx in self._bar_rects:
                if r.contains(pos) or (r.left() <= pos.x() <= r.right()):
                    self.selected_index = conf_idx
                    self.update()
                    self.conformer_clicked.emit(conf_idx)
                    return
        super().mousePressEvent(event)


class ConformerEnsembleDialog(QtWidgets.QDialog):
    """Comprehensive CREST and Conformer Ensemble Explorer Dialog."""

    conformer_adopted = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords)
    conformer_preview = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords)
    ensemble_overlay_requested = QtCore.pyqtSignal(list, list)  # (symbols, list of coords)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        filepath: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        trajectory_coords: Optional[List[np.ndarray]] = None,
        trajectory_energies: Optional[List[float]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🧬 CREST & Conformer Ensemble Explorer")
        self.resize(1020, 680)

        self.all_conformers: List[ConformerRecord] = []
        self.displayed_conformers: List[ConformerRecord] = []
        self.temperature_k: float = 298.15
        self.max_delta_e: float = 5.0
        self.filter_by_energy: bool = False
        self.prune_duplicates: bool = False
        self.rmsd_dup_threshold: float = 0.15

        self._init_ui()

        # Load initial data if provided
        if filepath and os.path.isfile(filepath):
            self.load_file(filepath)
        elif symbols and trajectory_coords and len(trajectory_coords) > 0:
            self.load_from_trajectory(symbols, trajectory_coords, trajectory_energies)

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setSpacing(10)

        # Header description
        header = QtWidgets.QLabel(
            "<b>CREST & Conformer Ensemble Explorer</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Explore, rank, and analyze conformer-rotamer ensembles from CREST, xTB, or internal searches. "
            "Evaluates Boltzmann populations at thermal equilibrium, heavy-atom RMSD superpositions, and duplicate clustering.</span>"
        )
        header.setWordWrap(True)
        main_layout.addWidget(header)

        # Top Control Bar (Source, Temperature, Filters)
        top_bar = QtWidgets.QGroupBox("1. Ensemble Source & Thermodynamic Controls")
        tb_layout = QtWidgets.QHBoxLayout(top_bar)
        tb_layout.setSpacing(12)

        self.btn_open_file = QtWidgets.QPushButton("📂 Open File (CREST / XYZ)...")
        self.btn_open_file.setToolTip("Open crest_conformers.xyz, crest_rotamers.xyz, or any multi-frame XYZ ensemble")
        self.btn_open_file.clicked.connect(self._on_browse_file)
        tb_layout.addWidget(self.btn_open_file)

        self.btn_load_active = QtWidgets.QPushButton("📥 Load Active Trajectory")
        self.btn_load_active.setToolTip("Import conformational search or optimization steps from the active AgyMolden session")
        self.btn_load_active.clicked.connect(self._on_load_active_trajectory)
        tb_layout.addWidget(self.btn_load_active)

        tb_layout.addSpacing(12)

        # Temperature
        lbl_temp = QtWidgets.QLabel("<b>Temperature:</b>")
        tb_layout.addWidget(lbl_temp)
        self.spin_temp = QtWidgets.QDoubleSpinBox()
        self.spin_temp.setRange(10.0, 1500.0)
        self.spin_temp.setDecimals(1)
        self.spin_temp.setSingleStep(5.0)
        self.spin_temp.setValue(298.15)
        self.spin_temp.setSuffix(" K")
        self.spin_temp.setToolTip("Thermodynamic temperature for Boltzmann population calculation: p_i ∝ exp(-ΔE_i / RT)")
        self.spin_temp.valueChanged.connect(self._on_temp_changed)
        tb_layout.addWidget(self.spin_temp)

        tb_layout.addSpacing(12)

        # Filter by ΔE
        self.chk_filter_e = QtWidgets.QCheckBox("Filter ΔE ≤")
        self.chk_filter_e.setChecked(False)
        self.chk_filter_e.toggled.connect(self._on_filter_changed)
        tb_layout.addWidget(self.chk_filter_e)

        self.spin_filter_e = QtWidgets.QDoubleSpinBox()
        self.spin_filter_e.setRange(0.1, 50.0)
        self.spin_filter_e.setDecimals(2)
        self.spin_filter_e.setValue(5.0)
        self.spin_filter_e.setSuffix(" kcal/mol")
        self.spin_filter_e.setEnabled(False)
        self.spin_filter_e.valueChanged.connect(self._on_filter_changed)
        self.chk_filter_e.toggled.connect(self.spin_filter_e.setEnabled)
        tb_layout.addWidget(self.spin_filter_e)

        tb_layout.addSpacing(12)

        # Prune duplicates
        self.chk_prune = QtWidgets.QCheckBox("Prune Duplicates (RMSD <")
        self.chk_prune.setChecked(False)
        self.chk_prune.setToolTip("Greedy clustering: hide higher-energy conformers that are geometrically identical to a lower-energy centroid")
        self.chk_prune.toggled.connect(self._on_prune_changed)
        tb_layout.addWidget(self.chk_prune)

        self.spin_prune_rmsd = QtWidgets.QDoubleSpinBox()
        self.spin_prune_rmsd.setRange(0.02, 1.0)
        self.spin_prune_rmsd.setDecimals(2)
        self.spin_prune_rmsd.setSingleStep(0.05)
        self.spin_prune_rmsd.setValue(0.15)
        self.spin_prune_rmsd.setSuffix(" Å)")
        self.spin_prune_rmsd.setEnabled(False)
        self.spin_prune_rmsd.valueChanged.connect(self._on_prune_changed)
        self.chk_prune.toggled.connect(self.spin_prune_rmsd.setEnabled)
        tb_layout.addWidget(self.spin_prune_rmsd)

        tb_layout.addStretch()
        main_layout.addWidget(top_bar)

        # Splitter: Table on left, Chart + actions on right
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left Widget: Conformer Table
        table_container = QtWidgets.QWidget()
        tc_layout = QtWidgets.QVBoxLayout(table_container)
        tc_layout.setContentsMargins(0, 0, 0, 0)

        self.table_conformers = QtWidgets.QTableWidget(0, 8)
        self.table_conformers.setHorizontalHeaderLabels([
            "Rank #",
            "Energy (Eh)",
            "ΔE (kcal/mol)",
            "ΔE (kJ/mol)",
            "Pop (%)",
            "Cumul (%)",
            "RMSD_heavy (Å)",
            "Status",
        ])
        self.table_conformers.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table_conformers.horizontalHeader().setStretchLastSection(True)
        self.table_conformers.verticalHeader().setDefaultSectionSize(22)
        self.table_conformers.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_conformers.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table_conformers.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_conformers.itemSelectionChanged.connect(self._on_table_selection_changed)
        tc_layout.addWidget(self.table_conformers)

        splitter.addWidget(table_container)

        # Right Widget: Visualizer, summary card, actions
        right_container = QtWidgets.QWidget()
        rc_layout = QtWidgets.QVBoxLayout(right_container)
        rc_layout.setContentsMargins(0, 0, 0, 0)
        rc_layout.setSpacing(10)

        # Boltzmann population plot
        self.chart_widget = BoltzmannBarWidget()
        self.chart_widget.conformer_clicked.connect(self._on_chart_conformer_clicked)
        rc_layout.addWidget(self.chart_widget)

        # Summary Metrics Card
        self.grp_metrics = QtWidgets.QGroupBox("Ensemble Thermodynamic Metrics")
        m_layout = QtWidgets.QGridLayout(self.grp_metrics)
        m_layout.setSpacing(6)

        m_layout.addWidget(QtWidgets.QLabel("Total Loaded:"), 0, 0)
        self.lbl_stat_total = QtWidgets.QLabel("0 conformers")
        self.lbl_stat_total.setStyleSheet("font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_total, 0, 1)

        m_layout.addWidget(QtWidgets.QLabel("Lowest Energy:"), 0, 2)
        self.lbl_stat_emin = QtWidgets.QLabel("—")
        self.lbl_stat_emin.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_emin, 0, 3)

        m_layout.addWidget(QtWidgets.QLabel("Active / Shown:"), 1, 0)
        self.lbl_stat_active = QtWidgets.QLabel("0")
        self.lbl_stat_active.setStyleSheet("font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_active, 1, 1)

        m_layout.addWidget(QtWidgets.QLabel("Energy Span:"), 1, 2)
        self.lbl_stat_span = QtWidgets.QLabel("0.00 kcal/mol")
        self.lbl_stat_span.setStyleSheet("font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_span, 1, 3)

        m_layout.addWidget(QtWidgets.QLabel("90% Population:"), 2, 0)
        self.lbl_stat_pop90 = QtWidgets.QLabel("—")
        self.lbl_stat_pop90.setStyleSheet("color: #34d399; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_pop90, 2, 1)

        m_layout.addWidget(QtWidgets.QLabel("99% Population:"), 2, 2)
        self.lbl_stat_pop99 = QtWidgets.QLabel("—")
        self.lbl_stat_pop99.setStyleSheet("color: #a7f3d0; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_pop99, 2, 3)

        m_layout.addWidget(QtWidgets.QLabel("S_conf (Entropy):"), 3, 0)
        self.lbl_stat_sconf = QtWidgets.QLabel("—")
        self.lbl_stat_sconf.setStyleSheet("color: #f59e0b; font-family: monospace; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_sconf, 3, 1)

        m_layout.addWidget(QtWidgets.QLabel("-T·ΔS_conf:"), 3, 2)
        self.lbl_stat_tsconf = QtWidgets.QLabel("—")
        self.lbl_stat_tsconf.setStyleSheet("color: #ec4899; font-family: monospace; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_tsconf, 3, 3)

        m_layout.addWidget(QtWidgets.QLabel("States (N_eff):"), 4, 0)
        self.lbl_stat_neff = QtWidgets.QLabel("—")
        self.lbl_stat_neff.setStyleSheet("color: #8b5cf6; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_neff, 4, 1)

        m_layout.addWidget(QtWidgets.QLabel("Ensemble ⟨R_g⟩:"), 4, 2)
        self.lbl_stat_rg = QtWidgets.QLabel("—")
        self.lbl_stat_rg.setStyleSheet("color: #06b6d4; font-family: monospace; font-weight: bold;")
        m_layout.addWidget(self.lbl_stat_rg, 4, 3)

        self.btn_copy_ensemble_si = QtWidgets.QPushButton("📋 Copy SI Summary")
        self.btn_copy_ensemble_si.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; font-size: 11px; padding: 2px 6px;")
        self.btn_copy_ensemble_si.setToolTip("Copy publication-ready Boltzmann ensemble statistics for Supporting Information")
        self.btn_copy_ensemble_si.clicked.connect(self._on_copy_ensemble_si)
        m_layout.addWidget(self.btn_copy_ensemble_si, 5, 0, 1, 4)

        rc_layout.addWidget(self.grp_metrics)

        # 3D Viewport Controls & Adoption
        grp_adopt = QtWidgets.QGroupBox("3. Viewport & 3D Editor Integration")
        a_layout = QtWidgets.QVBoxLayout(grp_adopt)
        a_layout.setSpacing(8)

        self.chk_live_preview = QtWidgets.QCheckBox("👁️ Live Preview Selected Conformer in Viewport")
        self.chk_live_preview.setChecked(True)
        a_layout.addWidget(self.chk_live_preview)

        btn_row1 = QtWidgets.QHBoxLayout()
        self.btn_adopt = QtWidgets.QPushButton("🎯 Adopt Conformer into 3D Editor")
        self.btn_adopt.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 6px;")
        self.btn_adopt.setToolTip("Load the selected conformer directly into AgyMolden's editable Z-Matrix table (with Undo support)")
        self.btn_adopt.clicked.connect(self._on_adopt_conformer)
        btn_row1.addWidget(self.btn_adopt)

        self.btn_superimpose = QtWidgets.QPushButton("🌐 Superimpose Ensemble")
        self.btn_superimpose.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 6px;")
        self.btn_superimpose.setToolTip("Align and overlay the conformer ensemble in the active viewport")
        self.btn_superimpose.clicked.connect(self._on_superimpose_ensemble)
        btn_row1.addWidget(self.btn_superimpose)
        a_layout.addLayout(btn_row1)

        rc_layout.addWidget(grp_adopt)

        # Export & Batch QM
        grp_export = QtWidgets.QGroupBox("4. Export & DFT Re-Optimization")
        e_layout = QtWidgets.QHBoxLayout(grp_export)
        e_layout.setSpacing(8)

        self.btn_export_xyz = QtWidgets.QPushButton("💾 Export Filtered XYZ...")
        self.btn_export_xyz.clicked.connect(self._on_export_xyz)
        e_layout.addWidget(self.btn_export_xyz)

        self.btn_export_csv = QtWidgets.QPushButton("📊 Export CSV Table...")
        self.btn_export_csv.clicked.connect(self._on_export_csv)
        e_layout.addWidget(self.btn_export_csv)

        self.btn_batch_qm = QtWidgets.QPushButton("⚡ Batch QM Inputs (ORCA)...")
        self.btn_batch_qm.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_batch_qm.setToolTip("Generate ready-to-run ORCA / Gaussian input files for all active conformers")
        self.btn_batch_qm.clicked.connect(self._on_generate_batch_qm)
        e_layout.addWidget(self.btn_batch_qm)

        rc_layout.addWidget(grp_export)
        splitter.addWidget(right_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main_layout.addWidget(splitter, stretch=1)

    def load_file(self, filepath: str):
        """Load conformers from a CREST or XYZ file."""
        if not os.path.isfile(filepath):
            QtWidgets.QMessageBox.warning(self, "File Not Found", f"Cannot open file: {filepath}")
            return

        confs = load_conformer_ensemble(filepath, is_path=True, temperature_k=self.spin_temp.value())
        if not confs:
            QtWidgets.QMessageBox.warning(self, "Invalid File", "No molecular conformer frames found in file.")
            return

        self.all_conformers = confs
        self.setWindowTitle(f"🧬 CREST & Conformer Ensemble Explorer — {os.path.basename(filepath)} ({len(confs)} Conformers)")
        self._apply_filters_and_update()

    def load_from_trajectory(
        self,
        symbols: List[str],
        trajectory_coords: List[np.ndarray],
        trajectory_energies: Optional[List[float]] = None,
    ):
        """Import frames from MainWindow trajectory or conformer search."""
        if not trajectory_coords:
            return

        recs: List[ConformerRecord] = []
        n_frames = len(trajectory_coords)
        for i in range(n_frames):
            coords = np.array(trajectory_coords[i], dtype=np.float64, copy=True)
            e = None
            src = "Trajectory"
            if trajectory_energies and i < len(trajectory_energies):
                val = trajectory_energies[i]
                # If negative and < -0.1, likely Hartree; if small positive, could be eV or kcal/mol
                if val < -0.1:
                    e = val
                    src = "Hartree"
                else:
                    e = val / 27.211386245988  # assume eV
                    src = "eV"
            rec = ConformerRecord(
                index=i + 1,
                symbols=list(symbols),
                coords=coords,
                energy=e,
                energy_source=src,
                raw_comment=f"Step {i + 1}",
            )
            recs.append(rec)

        if any(r.energy is not None for r in recs):
            recs.sort(key=lambda r: (r.energy if r.energy is not None else 1e9))
            for new_idx, r in enumerate(recs, start=1):
                r.index = new_idx

        recalculate_boltzmann(recs, temperature_k=self.spin_temp.value())
        recalculate_rmsds(recs, ref_idx=0)
        self.all_conformers = recs
        self.setWindowTitle(f"🧬 CREST & Conformer Ensemble Explorer — Active Trajectory ({len(recs)} Conformers)")
        self._apply_filters_and_update()

    def _apply_filters_and_update(self):
        """Filter conformers by energy and duplicate status and populate UI."""
        if not self.all_conformers:
            self.table_conformers.setRowCount(0)
            self.chart_widget.set_data([])
            self.lbl_stat_total.setText("0")
            self.lbl_stat_active.setText("0")
            return

        recalculate_boltzmann(self.all_conformers, temperature_k=self.spin_temp.value())
        recalculate_rmsds(self.all_conformers, ref_idx=0)

        # Run clustering if requested
        if self.chk_prune.isChecked():
            cluster_conformers(self.all_conformers, rmsd_threshold=self.spin_prune_rmsd.value())
        else:
            for c in self.all_conformers:
                c.is_clustered = False
                c.cluster_parent = None

        max_e = self.spin_filter_e.value() if self.chk_filter_e.isChecked() else None

        filtered = []
        for c in self.all_conformers:
            if max_e is not None and c.delta_e_kcal > max_e:
                continue
            if self.chk_prune.isChecked() and c.is_clustered:
                continue
            filtered.append(c)

        self.displayed_conformers = filtered
        self._populate_table()
        self._update_metrics()
        self.chart_widget.set_data(self.displayed_conformers, selected_index=self.displayed_conformers[0].index if self.displayed_conformers else 1)

    def _populate_table(self):
        self.table_conformers.blockSignals(True)
        self.table_conformers.setRowCount(len(self.displayed_conformers))

        for row, conf in enumerate(self.displayed_conformers):
            # Rank #
            item_rank = QtWidgets.QTableWidgetItem(f"#{conf.index}")
            item_rank.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_rank.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))

            # Energy
            e_str = f"{conf.energy:.6f}" if conf.energy is not None else "—"
            item_e = QtWidgets.QTableWidgetItem(e_str)
            item_e.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_e.setFont(QtGui.QFont("DejaVu Sans Mono", 9))

            # Delta E kcal/mol
            item_de_kcal = QtWidgets.QTableWidgetItem(f"{conf.delta_e_kcal:.2f}")
            item_de_kcal.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_de_kcal.setFont(QtGui.QFont("DejaVu Sans Mono", 9, QtGui.QFont.Weight.Bold))
            if conf.delta_e_kcal < 0.01:
                item_de_kcal.setForeground(QtGui.QBrush(QtGui.QColor("#34d399")))  # Lowest
            elif conf.delta_e_kcal < 1.0:
                item_de_kcal.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            elif conf.delta_e_kcal < 3.0:
                item_de_kcal.setForeground(QtGui.QBrush(QtGui.QColor("#fbbf24")))
            else:
                item_de_kcal.setForeground(QtGui.QBrush(QtGui.QColor("#f87171")))

            # Delta E kJ/mol
            item_de_kj = QtWidgets.QTableWidgetItem(f"{conf.delta_e_kj:.2f}")
            item_de_kj.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_de_kj.setFont(QtGui.QFont("DejaVu Sans Mono", 9))

            # Boltzmann Pop (%)
            item_pop = QtWidgets.QTableWidgetItem(f"{conf.boltzmann_pop:5.2f}%")
            item_pop.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_pop.setFont(QtGui.QFont("DejaVu Sans Mono", 9, QtGui.QFont.Weight.Bold))
            if conf.boltzmann_pop > 20.0:
                item_pop.setForeground(QtGui.QBrush(QtGui.QColor("#34d399")))
            elif conf.boltzmann_pop > 5.0:
                item_pop.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))

            # Cumulative (%)
            item_cum = QtWidgets.QTableWidgetItem(f"{conf.cumulative_pop:5.2f}%")
            item_cum.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_cum.setFont(QtGui.QFont("DejaVu Sans Mono", 9))

            # RMSD Heavy
            item_rmsd_h = QtWidgets.QTableWidgetItem(f"{conf.rmsd_heavy:.3f}")
            item_rmsd_h.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_rmsd_h.setFont(QtGui.QFont("DejaVu Sans Mono", 9))

            # Status
            status_text = "Centroid"
            if conf.is_clustered:
                status_text = f"Dup of #{conf.cluster_parent}"
            item_status = QtWidgets.QTableWidgetItem(status_text)
            item_status.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            if conf.is_clustered:
                item_status.setForeground(QtGui.QBrush(QtGui.QColor("#94a3b8")))

            self.table_conformers.setItem(row, 0, item_rank)
            self.table_conformers.setItem(row, 1, item_e)
            self.table_conformers.setItem(row, 2, item_de_kcal)
            self.table_conformers.setItem(row, 3, item_de_kj)
            self.table_conformers.setItem(row, 4, item_pop)
            self.table_conformers.setItem(row, 5, item_cum)
            self.table_conformers.setItem(row, 6, item_rmsd_h)
            self.table_conformers.setItem(row, 7, item_status)

        self.table_conformers.blockSignals(False)
        if self.displayed_conformers:
            self.table_conformers.selectRow(0)

    def _update_metrics(self):
        n_total = len(self.all_conformers)
        n_active = len(self.displayed_conformers)
        self.lbl_stat_total.setText(f"{n_total} conformers")
        self.lbl_stat_active.setText(f"{n_active} conformers")

        if self.all_conformers:
            lowest_c = self.all_conformers[0]
            if lowest_c.energy is not None:
                self.lbl_stat_emin.setText(f"{lowest_c.energy:.6f} Eh")
            else:
                self.lbl_stat_emin.setText("—")

            span = self.all_conformers[-1].delta_e_kcal if len(self.all_conformers) > 1 else 0.0
            self.lbl_stat_span.setText(f"{span:.2f} kcal/mol")

            # 90% and 99% populations
            k90 = next((i + 1 for i, c in enumerate(self.all_conformers) if c.cumulative_pop >= 90.0), n_total)
            k99 = next((i + 1 for i, c in enumerate(self.all_conformers) if c.cumulative_pop >= 99.0), n_total)
            self.lbl_stat_pop90.setText(f"{k90} conformer(s)")
            self.lbl_stat_pop99.setText(f"{k99} conformer(s)")

            from .ensemble_averager import calculate_ensemble_statistics
            temp_k = float(self.spin_temp.value())
            self._last_ensemble_stats = calculate_ensemble_statistics(self.all_conformers, temp_k)

            self.lbl_stat_sconf.setText(f"{self._last_ensemble_stats['s_conf_cal']:.2f} cal/(mol·K)")
            self.lbl_stat_tsconf.setText(f"{self._last_ensemble_stats['minus_t_s_kcal']:.3f} kcal/mol")
            self.lbl_stat_neff.setText(f"{self._last_ensemble_stats['n_eff']:.2f}")
            self.lbl_stat_rg.setText(f"{self._last_ensemble_stats['avg_rg']:.3f} Å")
        else:
            self.lbl_stat_emin.setText("—")
            self.lbl_stat_span.setText("0.00 kcal/mol")
            self.lbl_stat_pop90.setText("—")
            self.lbl_stat_pop99.setText("—")
            self.lbl_stat_sconf.setText("—")
            self.lbl_stat_tsconf.setText("—")
            self.lbl_stat_neff.setText("—")
            self.lbl_stat_rg.setText("—")
            self._last_ensemble_stats = None

    def _on_copy_ensemble_si(self):
        """Copy formatted Boltzmann conformational statistics to system clipboard for journal SI."""
        if getattr(self, "_last_ensemble_stats", None):
            txt = self._last_ensemble_stats["summary_text"]
            QtWidgets.QApplication.clipboard().setText(txt)
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied Boltzmann Ensemble SI Summary to clipboard!", self)

    def _on_table_selection_changed(self):
        row = self.table_conformers.currentRow()
        if 0 <= row < len(self.displayed_conformers):
            conf = self.displayed_conformers[row]
            self.chart_widget.selected_index = conf.index
            self.chart_widget.update()
            if self.chk_live_preview.isChecked():
                self.conformer_preview.emit(conf.symbols, conf.coords)

    def _on_chart_conformer_clicked(self, conf_idx: int):
        for row, c in enumerate(self.displayed_conformers):
            if c.index == conf_idx:
                self.table_conformers.selectRow(row)
                break

    def _on_temp_changed(self):
        self._apply_filters_and_update()

    def _on_filter_changed(self):
        self._apply_filters_and_update()

    def _on_prune_changed(self):
        self._apply_filters_and_update()

    def _on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Conformer Ensemble",
            "",
            "Conformer Files (crest_conformers.xyz crest_rotamers.xyz *.xyz *.trj);;All Files (*)",
        )
        if path:
            self.load_file(path)

    def _on_load_active_trajectory(self):
        parent = self.parent()
        if parent and hasattr(parent, "trajectory_coords") and parent.trajectory_coords:
            syms = [a.symbol for a in parent.zatoms] if hasattr(parent, "zatoms") else []
            self.load_from_trajectory(
                symbols=syms,
                trajectory_coords=parent.trajectory_coords,
                trajectory_energies=getattr(parent, "trajectory_energies", None),
            )
        else:
            QtWidgets.QMessageBox.information(
                self, "No Trajectory", "No active trajectory or conformer ensemble found in the current session."
            )

    def _on_adopt_conformer(self):
        row = self.table_conformers.currentRow()
        if not (0 <= row < len(self.displayed_conformers)):
            return
        conf = self.displayed_conformers[row]

        parent = self.parent()
        if parent and hasattr(parent, "_push_undo_state") and hasattr(parent, "set_molecule_from_zatoms"):
            parent._push_undo_state()
            # If Z-matrix exists, reconstruct with exact Cartesian coordinates
            from .zmatrix import cartesian_to_zmatrix
            user_bonds = parent._current_explicit_bonds() if hasattr(parent, "_current_explicit_bonds") else None
            new_zatoms = cartesian_to_zmatrix(conf.symbols, conf.coords, existing_zatoms=getattr(parent, "zatoms", None), user_bonds=user_bonds)
            parent.set_molecule_from_zatoms(new_zatoms, exact_coords=conf.coords)
            if hasattr(parent, "statusBar") and parent.statusBar():
                parent.statusBar().showMessage(
                    f"Adopted Conformer #{conf.index} (ΔE = {conf.delta_e_kcal:.2f} kcal/mol, Pop = {conf.boltzmann_pop:.1f}%) into 3D Editor.",
                    5000,
                )
            QtWidgets.QMessageBox.information(
                self,
                "Conformer Adopted",
                f"Successfully adopted Conformer #{conf.index} into the active 3D editor.\n"
                f"ΔE: {conf.delta_e_kcal:.2f} kcal/mol | Boltzmann Pop: {conf.boltzmann_pop:.1f}%\n"
                f"(Press Ctrl+Z to undo anytime).",
            )
        self.conformer_adopted.emit(conf.symbols, conf.coords)

    def _on_superimpose_ensemble(self):
        """Align all active conformers to reference and update trajectory player."""
        if not self.displayed_conformers:
            return

        ref_coords = self.displayed_conformers[0].coords
        ref_symbols = self.displayed_conformers[0].symbols
        aligned_coords_list = []

        for conf in self.displayed_conformers:
            try:
                aln, _, _, _ = calculate_kabsch_alignment(ref_coords, conf.coords)
                aligned_coords_list.append(aln)
            except Exception:
                aligned_coords_list.append(conf.coords)

        parent = self.parent()
        if parent and hasattr(parent, "trajectory_coords") and hasattr(parent, "grp_trajectory"):
            parent.trajectory_coords = aligned_coords_list
            parent.trajectory_symbols = ref_symbols
            parent.trajectory_energies = [c.energy for c in self.displayed_conformers]
            n_c = len(aligned_coords_list)
            parent.grp_trajectory.setTitle(f"Conformer Ensemble ({n_c} Aligned Conformers)")
            parent.grp_trajectory.setVisible(True)
            parent.slider_trajectory.blockSignals(True)
            parent.slider_trajectory.setRange(0, n_c - 1)
            parent.slider_trajectory.setValue(0)
            parent.slider_trajectory.blockSignals(False)
            parent.lbl_trajectory_step.setText(f"Conf 1/{n_c}")
            parent.lbl_trajectory_energy.setText("ΔE = 0.00 kcal/mol")
            parent.statusBar().showMessage(f"Streamed {n_c} Kabsch-aligned conformers to Trajectory Player.", 4000)

        self.ensemble_overlay_requested.emit(ref_symbols, aligned_coords_list)

    def _on_export_xyz(self):
        if not self.displayed_conformers:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Filtered Conformer Ensemble", "conformer_ensemble.xyz", "XYZ Molecule (*.xyz);;All Files (*)"
        )
        if path:
            n_exp = export_ensemble_xyz(self.displayed_conformers, path)
            QtWidgets.QMessageBox.information(
                self, "Ensemble Exported", f"Successfully exported {n_exp} conformers to:\n{path}"
            )

    def _on_export_csv(self):
        if not self.displayed_conformers:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Conformer Table", "conformer_metrics.csv", "CSV Spreadsheet (*.csv);;All Files (*)"
        )
        if path:
            export_ensemble_csv(self.displayed_conformers, path)
            QtWidgets.QMessageBox.information(
                self, "CSV Exported", f"Successfully exported conformer metrics table to:\n{path}"
            )

    def _on_generate_batch_qm(self):
        """Prompt user for QM parameters and generate batch inputs."""
        if not self.displayed_conformers:
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("⚡ Generate DFT Conformer Batch Inputs")
        dialog.resize(440, 260)
        d_layout = QtWidgets.QVBoxLayout(dialog)

        form = QtWidgets.QFormLayout()
        combo_engine = QtWidgets.QComboBox()
        combo_engine.addItems(["ORCA", "Gaussian"])
        form.addRow("QM Package:", combo_engine)

        line_method = QtWidgets.QLineEdit("r2SCAN-3c")
        form.addRow("Method / Functional:", line_method)

        line_basis = QtWidgets.QLineEdit("")
        line_basis.setPlaceholderText("(Leave blank for composite methods like r2SCAN-3c)")
        form.addRow("Basis Set:", line_basis)

        combo_solv = QtWidgets.QComboBox()
        combo_solv.addItems(["Gas Phase", "Water", "Acetonitrile", "Dichloromethane", "Chloroform", "DMSO", "THF", "Toluene"])
        form.addRow("Solvent (CPCM / SMD):", combo_solv)

        spin_charge = QtWidgets.QSpinBox()
        spin_charge.setRange(-10, 10)
        spin_charge.setValue(0)
        form.addRow("Molecular Charge:", spin_charge)

        spin_mult = QtWidgets.QSpinBox()
        spin_mult.setRange(1, 10)
        spin_mult.setValue(1)
        form.addRow("Spin Multiplicity:", spin_mult)

        d_layout.addLayout(form)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        d_layout.addWidget(btn_box)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            out_dir = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory for Batch Inputs")
            if out_dir:
                eng = combo_engine.currentText()
                solv = "" if combo_solv.currentText() == "Gas Phase" else combo_solv.currentText()
                gen_files = generate_qm_batch_scripts(
                    self.displayed_conformers,
                    output_dir=out_dir,
                    engine=eng,
                    method=line_method.text().strip() or "r2SCAN-3c",
                    basis=line_basis.text().strip(),
                    charge=spin_charge.value(),
                    mult=spin_mult.value(),
                    solvent=solv,
                )
                QtWidgets.QMessageBox.information(
                    self,
                    "Batch Generation Complete",
                    f"Generated {len(gen_files)} files in:\n{out_dir}\n\n"
                    f"Includes automated 'run_batch.sh' and 'run_batch.bat' for running sequential DFT optimizations!",
                )
