"""
Visual Workflow DAG Pipeline Editor for AgyMolden.

Provides an interactive node/DAG visual editor for chaining multi-stage
quantum chemical calculations:
  [Opt] ──▶ [Freq (Check Min)] ──▶ [TD-DFT / TS / IRC / Single Point]
with per-stage parameter overrides, recipe save/load (JSON), and real-time
stage status updates during execution.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .calc_runner.pipeline import ChainedWorkflowPipeline, WorkflowStageConfig
from .qm_input_generator import QMCalculationParameters


@dataclass
class DAGStage:
    name: str
    job_type: str  # "Opt", "Freq", "SP", "TS", "TD-DFT", "IRC"
    description: str = ""
    method: str = ""
    basis: str = ""
    solvent: str = ""
    extra_keywords: str = ""
    status: str = "pending"  # "pending", "running", "success", "failed"
    status_msg: str = ""

    def to_config(self) -> WorkflowStageConfig:
        extra: Dict[str, Any] = {}
        if self.method:
            extra["method"] = self.method
        if self.basis:
            extra["basis"] = self.basis
        if self.solvent:
            extra["solvent"] = self.solvent
        if self.extra_keywords:
            extra["extra_keywords"] = self.extra_keywords
        return WorkflowStageConfig(
            name=self.name,
            job_type=self.job_type,
            description=self.description,
            extra_params=extra,
        )


DAG_TEMPLATES: Dict[str, List[DAGStage]] = {
    "Opt → Freq (Minimum Guarantee)": [
        DAGStage("Geometry Optimization", "Opt", "Optimize structure to local stationary point"),
        DAGStage("Harmonic Frequencies", "Freq", "Verify positive Hessian (0 imaginary frequencies)"),
    ],
    "Opt → Freq → TD-DFT (UV-Vis Spectrum)": [
        DAGStage("Ground State Optimization", "Opt", "Optimize geometry to equilibrium minimum"),
        DAGStage("Frequency Check", "Freq", "Verify stationary ground state"),
        DAGStage("TD-DFT Excitation", "TD-DFT", "Compute excited state energies & oscillator strengths"),
    ],
    "TS Opt → Freq (1 Imag) → IRC (Path)": [
        DAGStage("Transition State Search", "TS", "Optimize to 1st-order saddle point"),
        DAGStage("Frequency Verification", "Freq", "Confirm exactly 1 imaginary frequency"),
        DAGStage("Intrinsic Reaction Coordinate", "IRC", "Trace minimum energy path forward & backward"),
    ],
    "DFT Opt → Freq → DLPNO-CCSD(T) Single Point": [
        DAGStage("DFT Geometry Optimization", "Opt", "B3LYP-D4 / def2-TZVP structure optimization"),
        DAGStage("DFT Thermochemistry", "Freq", "Thermal corrections and zero-point energy"),
        DAGStage("High-Level Single Point", "SP", "DLPNO-CCSD(T) / def2-QZVPP benchmark electronic energy", method="DLPNO-CCSD(T)", basis="def2-QZVPP"),
    ],
}


class VisualDAGCanvas(QtWidgets.QWidget):
    """Custom widget rendering an interactive visual DAG flowchart of workflow stages."""

    stageSelected = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(170)
        self.stages: List[DAGStage] = []
        self.selected_idx: int = 0
        self.card_rects: List[QtCore.QRectF] = []
        self.setMouseTracking(True)
        self.hover_idx: int = -1

    def set_stages(self, stages: List[DAGStage], selected_idx: int = 0):
        self.stages = stages
        self.selected_idx = max(0, min(selected_idx, len(stages) - 1)) if stages else 0
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = event.position()
        for idx, rect in enumerate(self.card_rects):
            if rect.contains(pos):
                self.selected_idx = idx
                self.stageSelected.emit(idx)
                self.update()
                break

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        pos = event.position()
        new_hover = -1
        for idx, rect in enumerate(self.card_rects):
            if rect.contains(pos):
                new_hover = idx
                break
        if new_hover != self.hover_idx:
            self.hover_idx = new_hover
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Canvas background (subtle slate dark/light)
        is_dark = self.palette().color(QtGui.QPalette.ColorRole.Window).lightnessF() < 0.4
        bg_col = QtGui.QColor("#0f172a" if is_dark else "#f8fafc")
        painter.fillRect(0, 0, w, h, bg_col)

        if not self.stages:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.setFont(QtGui.QFont("DejaVu Sans", 11, QtGui.QFont.Weight.Medium))
            painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter, "No stages in workflow DAG. Click '➕ Add Stage' to begin.")
            return

        n_stages = len(self.stages)
        card_w = 175.0
        card_h = 100.0
        spacing = 45.0
        total_w = n_stages * card_w + (n_stages - 1) * spacing
        start_x = max(20.0, (w - total_w) / 2.0)
        y = (h - card_h) / 2.0

        self.card_rects = []
        centers: List[QtCore.QPointF] = []

        # First pass: calculate positions and store centers
        for i in range(n_stages):
            x = start_x + i * (card_w + spacing)
            r = QtCore.QRectF(x, y, card_w, card_h)
            self.card_rects.append(r)
            centers.append(QtCore.QPointF(x + card_w / 2.0, y + card_h / 2.0))

        # Second pass: Draw connecting arrows between stages
        arrow_pen = QtGui.QPen(QtGui.QColor("#38bdf8" if is_dark else "#0284c7"), 2.2)
        arrow_brush = QtGui.QBrush(QtGui.QColor("#38bdf8" if is_dark else "#0284c7"))
        painter.setPen(arrow_pen)
        painter.setBrush(arrow_brush)

        for i in range(n_stages - 1):
            p1 = QtCore.QPointF(self.card_rects[i].right(), y + card_h / 2.0)
            p2 = QtCore.QPointF(self.card_rects[i + 1].left(), y + card_h / 2.0)
            painter.drawLine(p1, p2)

            # Arrow head
            tip_x = p2.x()
            tip_y = p2.y()
            arrow_poly = QtGui.QPolygonF([
                QtCore.QPointF(tip_x, tip_y),
                QtCore.QPointF(tip_x - 8.0, tip_y - 5.0),
                QtCore.QPointF(tip_x - 8.0, tip_y + 5.0),
            ])
            painter.drawPolygon(arrow_poly)

        # Third pass: Draw stage cards
        status_colors = {
            "pending": (QtGui.QColor("#334155" if is_dark else "#cbd5e1"), QtGui.QColor("#94a3b8"), "⚪ Pending"),
            "running": (QtGui.QColor("#0369a1"), QtGui.QColor("#38bdf8"), "🔄 Running..."),
            "success": (QtGui.QColor("#15803d"), QtGui.QColor("#4ade80"), "✔ Success"),
            "failed":  (QtGui.QColor("#b91c1c"), QtGui.QColor("#f87171"), "✖ Failed"),
        }

        for i, (st, rect) in enumerate(zip(self.stages, self.card_rects)):
            is_sel = (i == self.selected_idx)
            is_hov = (i == self.hover_idx)

            border_col, badge_col, badge_text = status_colors.get(
                st.status, (QtGui.QColor("#475569"), QtGui.QColor("#94a3b8"), "⚪ Pending")
            )

            # Card fill
            card_fill = QtGui.QColor("#1e293b" if is_dark else "#ffffff")
            if is_sel:
                border_pen = QtGui.QPen(QtGui.QColor("#38bdf8" if is_dark else "#0284c7"), 3.0)
            elif is_hov:
                border_pen = QtGui.QPen(QtGui.QColor("#64748b" if is_dark else "#94a3b8"), 2.0)
            else:
                border_pen = QtGui.QPen(border_col, 1.5)

            painter.setPen(border_pen)
            painter.setBrush(QtGui.QBrush(card_fill))
            painter.drawRoundedRect(rect, 8.0, 8.0)

            # Top Header Bar of Card
            header_rect = QtCore.QRectF(rect.x(), rect.y(), rect.width(), 26.0)
            header_col = QtGui.QColor("#0f172a" if is_dark else "#e2e8f0")
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.setBrush(QtGui.QBrush(header_col))
            header_path = QtGui.QPainterPath()
            header_path.addRoundedRect(header_rect, 8.0, 8.0)
            painter.drawPath(header_path)

            # Header text: Stage # and Job Type badge
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            painter.setPen(QtGui.QColor("#f8fafc" if is_dark else "#0f172a"))
            painter.drawText(
                header_rect.adjusted(8, 0, -8, 0),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                f"#{i + 1} {st.job_type}",
            )

            # Status Badge text (top right)
            painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
            painter.setPen(badge_col)
            painter.drawText(
                header_rect.adjusted(8, 0, -8, 0),
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignRight,
                badge_text,
            )

            # Card Body text: Name
            body_rect = QtCore.QRectF(rect.x() + 8, rect.y() + 32, rect.width() - 16, 40)
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Medium))
            painter.setPen(QtGui.QColor("#e2e8f0" if is_dark else "#1e293b"))
            painter.drawText(
                body_rect,
                QtCore.Qt.AlignmentFlag.AlignTop | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.TextFlag.TextWordWrap,
                st.name,
            )

            # Bottom details: Method/Basis info
            detail_rect = QtCore.QRectF(rect.x() + 8, rect.bottom() - 22, rect.width() - 16, 18)
            info_str = f"{st.method or 'Default'} / {st.basis or 'Default'}"
            painter.setFont(QtGui.QFont("DejaVu Sans", 7, QtGui.QFont.Weight.Normal))
            painter.setPen(QtGui.QColor("#94a3b8" if is_dark else "#64748b"))
            painter.drawText(
                detail_rect,
                QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft,
                info_str,
            )


class VisualWorkflowDAGDialog(QtWidgets.QDialog):
    """
    Visual Workflow DAG Pipeline Editor and Runner.
    Allows composing multi-stage QM recipes with stage-by-stage parameter overrides,
    JSON recipe persistence, and real-time visual progress monitoring.
    """

    pipelineAdoptRequested = QtCore.pyqtSignal(list, object, str)  # (symbols, coords, title)

    def __init__(
        self,
        parent=None,
        code_key: str = "orca",
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        base_params: Optional[QMCalculationParameters] = None,
        work_dir: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("⚡ Visual Workflow DAG Pipeline Editor")
        self.resize(960, 680)

        self.code_key = code_key
        self.symbols = list(symbols) if symbols else ["O", "H", "H"]
        self.coords = np.array(coords, copy=True) if coords is not None else np.zeros((3, 3))
        self.base_params = base_params or QMCalculationParameters()
        self.work_dir = work_dir or os.path.join(os.getcwd(), "workflow_run")

        self.stages: List[DAGStage] = [
            DAGStage(st.name, st.job_type, st.description, st.method, st.basis, st.solvent, st.extra_keywords)
            for st in DAG_TEMPLATES["Opt → Freq (Minimum Guarantee)"]
        ]
        self.selected_idx: int = 0
        self.pipeline: Optional[ChainedWorkflowPipeline] = None

        self._init_ui()
        self._update_inspector()

    def _init_ui(self):
        main_lay = QtWidgets.QVBoxLayout(self)
        main_lay.setContentsMargins(12, 12, 12, 12)
        main_lay.setSpacing(10)

        # 1. Top Controls Bar: Preset Templates & Recipe IO
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("<b>Workflow Recipe:</b>"))

        self.combo_presets = QtWidgets.QComboBox()
        for name in DAG_TEMPLATES.keys():
            self.combo_presets.addItem(name)
        self.combo_presets.addItem("Custom User Workflow...")
        self.combo_presets.currentTextChanged.connect(self._on_preset_changed)
        top_bar.addWidget(self.combo_presets, stretch=2)

        btn_load = QtWidgets.QPushButton("📂 Load Recipe (.json)...")
        btn_load.clicked.connect(self._on_load_recipe)
        top_bar.addWidget(btn_load)

        btn_save = QtWidgets.QPushButton("💾 Save Recipe (.json)...")
        btn_save.clicked.connect(self._on_save_recipe)
        top_bar.addWidget(btn_save)

        top_bar.addStretch()

        # Engine selector
        top_bar.addWidget(QtWidgets.QLabel("<b>Target Engine:</b>"))
        self.combo_engine = QtWidgets.QComboBox()
        self.combo_engine.addItems(["ORCA", "PySCF", "Gaussian", "xTB", "openLOWDIN"])
        # Select current code_key if found
        for i in range(self.combo_engine.count()):
            if self.combo_engine.itemText(i).lower() == self.code_key.lower():
                self.combo_engine.setCurrentIndex(i)
                break
        top_bar.addWidget(self.combo_engine)

        main_lay.addLayout(top_bar)

        # 2. Visual DAG Canvas
        canvas_grp = QtWidgets.QGroupBox("Visual Workflow Directed Acyclic Graph (DAG)")
        canvas_lay = QtWidgets.QVBoxLayout(canvas_grp)
        canvas_lay.setContentsMargins(6, 6, 6, 6)

        self.canvas = VisualDAGCanvas()
        self.canvas.set_stages(self.stages, self.selected_idx)
        self.canvas.stageSelected.connect(self._on_stage_selected)
        canvas_lay.addWidget(self.canvas)

        main_lay.addWidget(canvas_grp)

        # 3. Middle Area: Stage Inspector (Left) + Console / Execution Status (Right)
        mid_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: Inspector Panel
        inspector_widget = QtWidgets.QWidget()
        insp_lay = QtWidgets.QVBoxLayout(inspector_widget)
        insp_lay.setContentsMargins(0, 0, 0, 0)

        insp_grp = QtWidgets.QGroupBox("Stage Inspector & Parameter Overrides")
        grid = QtWidgets.QGridLayout(insp_grp)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(6)

        grid.addWidget(QtWidgets.QLabel("Stage Name:"), 0, 0)
        self.txt_st_name = QtWidgets.QLineEdit()
        self.txt_st_name.textChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.txt_st_name, 0, 1)

        grid.addWidget(QtWidgets.QLabel("Job Type:"), 1, 0)
        self.combo_st_type = QtWidgets.QComboBox()
        self.combo_st_type.addItems(["Opt", "Freq", "SP", "TS", "TD-DFT", "IRC"])
        self.combo_st_type.currentTextChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.combo_st_type, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Method Override:"), 2, 0)
        self.txt_st_method = QtWidgets.QLineEdit()
        self.txt_st_method.setPlaceholderText("e.g. B3LYP-D4, wB97X-D3, DLPNO-CCSD(T)")
        self.txt_st_method.textChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.txt_st_method, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Basis Override:"), 3, 0)
        self.txt_st_basis = QtWidgets.QLineEdit()
        self.txt_st_basis.setPlaceholderText("e.g. def2-TZVP, 6-311+G(d,p)")
        self.txt_st_basis.textChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.txt_st_basis, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Solvent Override:"), 4, 0)
        self.txt_st_solvent = QtWidgets.QLineEdit()
        self.txt_st_solvent.setPlaceholderText("e.g. Water, Chloroform, Acetonitrile")
        self.txt_st_solvent.textChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.txt_st_solvent, 4, 1)

        grid.addWidget(QtWidgets.QLabel("Extra Keywords:"), 5, 0)
        self.txt_st_extra = QtWidgets.QLineEdit()
        self.txt_st_extra.setPlaceholderText("e.g. TightOpt, Grid4, Bohrs")
        self.txt_st_extra.textChanged.connect(self._on_stage_prop_changed)
        grid.addWidget(self.txt_st_extra, 5, 1)

        insp_lay.addWidget(insp_grp)

        # Stage list manipulation buttons
        btn_bar = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("➕ Add Stage")
        btn_add.clicked.connect(self._on_add_stage)
        btn_bar.addWidget(btn_add)

        btn_del = QtWidgets.QPushButton("❌ Remove")
        btn_del.clicked.connect(self._on_remove_stage)
        btn_bar.addWidget(btn_del)

        btn_up = QtWidgets.QPushButton("⬆️ Move Up")
        btn_up.clicked.connect(self._on_move_stage_up)
        btn_bar.addWidget(btn_up)

        btn_down = QtWidgets.QPushButton("⬇️ Move Down")
        btn_down.clicked.connect(self._on_move_stage_down)
        btn_bar.addWidget(btn_down)

        insp_lay.addLayout(btn_bar)
        insp_lay.addStretch()
        mid_splitter.addWidget(inspector_widget)

        # Right: Live Execution Console
        console_widget = QtWidgets.QWidget()
        con_lay = QtWidgets.QVBoxLayout(console_widget)
        con_lay.setContentsMargins(0, 0, 0, 0)

        con_grp = QtWidgets.QGroupBox("Live Execution Stream & Verification Banner")
        con_inner = QtWidgets.QVBoxLayout(con_grp)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Monospace", 9))
        self.txt_log.setStyleSheet("background-color: #0f172a; color: #f8fafc;")
        con_inner.addWidget(self.txt_log)

        self.lbl_status = QtWidgets.QLabel("Ready to launch workflow pipeline.")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #38bdf8;")
        con_inner.addWidget(self.lbl_status)

        con_lay.addWidget(con_grp)
        mid_splitter.addWidget(console_widget)

        mid_splitter.setSizes([380, 560])
        main_lay.addWidget(mid_splitter, stretch=1)

        # 4. Bottom Controls: Run / Abort / Adopt / Close
        bottom_bar = QtWidgets.QHBoxLayout()

        self.btn_run = QtWidgets.QPushButton("🚀 Execute Visual Workflow Pipeline")
        self.btn_run.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 7px 18px;")
        self.btn_run.clicked.connect(self._on_run_pipeline)
        bottom_bar.addWidget(self.btn_run)

        self.btn_abort = QtWidgets.QPushButton("⏹ Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort_pipeline)
        bottom_bar.addWidget(self.btn_abort)

        bottom_bar.addStretch()

        self.btn_adopt = QtWidgets.QPushButton("👁️ Adopt Final Geometry into 3D Editor")
        self.btn_adopt.setEnabled(False)
        self.btn_adopt.clicked.connect(self._on_adopt_geometry)
        bottom_bar.addWidget(self.btn_adopt)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bottom_bar.addWidget(btn_close)

        main_lay.addLayout(bottom_bar)

    # ---------------------------------------------------------------------
    # Inspector & Stage Management
    # ---------------------------------------------------------------------
    def _on_stage_selected(self, idx: int):
        self.selected_idx = idx
        self._update_inspector()

    def _update_inspector(self):
        if not self.stages or self.selected_idx >= len(self.stages):
            self.txt_st_name.setText("")
            self.txt_st_method.setText("")
            self.txt_st_basis.setText("")
            self.txt_st_solvent.setText("")
            self.txt_st_extra.setText("")
            return

        st = self.stages[self.selected_idx]
        self.txt_st_name.blockSignals(True)
        self.combo_st_type.blockSignals(True)
        self.txt_st_method.blockSignals(True)
        self.txt_st_basis.blockSignals(True)
        self.txt_st_solvent.blockSignals(True)
        self.txt_st_extra.blockSignals(True)

        self.txt_st_name.setText(st.name)
        idx_type = self.combo_st_type.findText(st.job_type)
        if idx_type >= 0:
            self.combo_st_type.setCurrentIndex(idx_type)
        self.txt_st_method.setText(st.method)
        self.txt_st_basis.setText(st.basis)
        self.txt_st_solvent.setText(st.solvent)
        self.txt_st_extra.setText(st.extra_keywords)

        self.txt_st_name.blockSignals(False)
        self.combo_st_type.blockSignals(False)
        self.txt_st_method.blockSignals(False)
        self.txt_st_basis.blockSignals(False)
        self.txt_st_solvent.blockSignals(False)
        self.txt_st_extra.blockSignals(False)

    def _on_stage_prop_changed(self):
        if not self.stages or self.selected_idx >= len(self.stages):
            return
        st = self.stages[self.selected_idx]
        st.name = self.txt_st_name.text().strip() or "Unnamed Stage"
        st.job_type = self.combo_st_type.currentText()
        st.method = self.txt_st_method.text().strip()
        st.basis = self.txt_st_basis.text().strip()
        st.solvent = self.txt_st_solvent.text().strip()
        st.extra_keywords = self.txt_st_extra.text().strip()
        self.canvas.update()

    def _on_add_stage(self):
        new_st = DAGStage(f"Stage {len(self.stages) + 1}", "SP", "Single point calculation")
        self.stages.append(new_st)
        self.selected_idx = len(self.stages) - 1
        self.canvas.set_stages(self.stages, self.selected_idx)
        self._update_inspector()

    def _on_remove_stage(self):
        if len(self.stages) <= 1:
            QtWidgets.QMessageBox.warning(self, "Cannot Remove", "Pipeline must contain at least 1 stage.")
            return
        self.stages.pop(self.selected_idx)
        self.selected_idx = max(0, min(self.selected_idx, len(self.stages) - 1))
        self.canvas.set_stages(self.stages, self.selected_idx)
        self._update_inspector()

    def _on_move_stage_up(self):
        if self.selected_idx > 0:
            self.stages[self.selected_idx - 1], self.stages[self.selected_idx] = (
                self.stages[self.selected_idx],
                self.stages[self.selected_idx - 1],
            )
            self.selected_idx -= 1
            self.canvas.set_stages(self.stages, self.selected_idx)
            self._update_inspector()

    def _on_move_stage_down(self):
        if self.selected_idx < len(self.stages) - 1:
            self.stages[self.selected_idx + 1], self.stages[self.selected_idx] = (
                self.stages[self.selected_idx],
                self.stages[self.selected_idx + 1],
            )
            self.selected_idx += 1
            self.canvas.set_stages(self.stages, self.selected_idx)
            self._update_inspector()

    def _on_preset_changed(self, name: str):
        if name in DAG_TEMPLATES:
            self.stages = [
                DAGStage(st.name, st.job_type, st.description, st.method, st.basis, st.solvent, st.extra_keywords)
                for st in DAG_TEMPLATES[name]
            ]
            self.selected_idx = 0
            self.canvas.set_stages(self.stages, self.selected_idx)
            self._update_inspector()

    def _on_save_recipe(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Workflow Recipe", "my_workflow.json", "JSON Recipe Files (*.json);;All Files (*)"
        )
        if not path:
            return
        data = {
            "version": "1.0",
            "recipe_name": self.combo_presets.currentText(),
            "engine": self.combo_engine.currentText(),
            "stages": [asdict(st) for st in self.stages],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            QtWidgets.QMessageBox.information(self, "Recipe Saved", f"Workflow recipe saved to {os.path.basename(path)}")

    def _on_load_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Workflow Recipe", "", "JSON Recipe Files (*.json);;All Files (*)"
        )
        if not path or not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        loaded_stages = []
        for s in data.get("stages", []):
            loaded_stages.append(DAGStage(
                name=s.get("name", "Stage"),
                job_type=s.get("job_type", "Opt"),
                description=s.get("description", ""),
                method=s.get("method", ""),
                basis=s.get("basis", ""),
                solvent=s.get("solvent", ""),
                extra_keywords=s.get("extra_keywords", ""),
            ))
        if loaded_stages:
            self.stages = loaded_stages
            self.selected_idx = 0
            self.canvas.set_stages(self.stages, self.selected_idx)
            self._update_inspector()

    # ---------------------------------------------------------------------
    # Execution & Pipeline Runner
    # ---------------------------------------------------------------------
    def _on_run_pipeline(self):
        if not self.stages:
            return

        for st in self.stages:
            st.status = "pending"
            st.status_msg = ""
        self.canvas.update()

        self.txt_log.clear()
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_adopt.setEnabled(False)

        configs = [st.to_config() for st in self.stages]
        selected_code = self.combo_engine.currentText().lower()

        self.pipeline = ChainedWorkflowPipeline(
            code_key=selected_code,
            symbols=self.symbols,
            initial_coords=self.coords,
            base_params=self.base_params,
            stages=configs,
            work_dir=self.work_dir,
            parent=self,
        )
        self.pipeline.stage_started.connect(self._on_stage_started)
        self.pipeline.stage_finished.connect(self._on_stage_finished)
        self.pipeline.pipeline_finished.connect(self._on_pipeline_finished)
        self.pipeline.line_received.connect(self._on_line_received)
        self.pipeline.start()

    def _on_abort_pipeline(self):
        if self.pipeline:
            self.pipeline.abort()
        self.btn_abort.setEnabled(False)
        self.btn_run.setEnabled(True)

    def _on_stage_started(self, stage_idx: int, stage_name: str):
        if 0 <= stage_idx < len(self.stages):
            self.stages[stage_idx].status = "running"
            self.canvas.update()
        self.lbl_status.setText(f"Running Stage {stage_idx + 1}/{len(self.stages)}: {stage_name}...")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #38bdf8;")

    def _on_stage_finished(self, stage_idx: int, stage_name: str, success: bool, msg: str):
        if 0 <= stage_idx < len(self.stages):
            self.stages[stage_idx].status = "success" if success else "failed"
            self.stages[stage_idx].status_msg = msg
            self.canvas.update()

    def _on_pipeline_finished(self, success: bool, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        if success:
            self.lbl_status.setText(f"✔ {msg}")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #22c55e;")
            self.btn_adopt.setEnabled(True)
        else:
            self.lbl_status.setText(f"✖ {msg}")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #ef4444;")

    def _on_line_received(self, line: str, is_err: bool):
        self.txt_log.appendPlainText(line)

    def _on_adopt_geometry(self):
        if self.pipeline and self.pipeline.current_coords is not None:
            self.pipelineAdoptRequested.emit(
                self.symbols,
                self.pipeline.current_coords,
                f"Workflow DAG Result ({self.combo_presets.currentText()})",
            )
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                QtWidgets.QMessageBox.information(
                    self, "Geometry Adopted", "Optimized pipeline geometry adopted into main 3D editor!"
                )
            self.accept()
