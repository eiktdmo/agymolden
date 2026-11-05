"""
GUI Dialog for Executing Chained Quantum Chemistry Workflow Pipelines.

Provides visual execution of multi-stage recipes:
- Opt → Freq (with automated imaginary frequency check & true minimum verification)
- Opt → TD-DFT (with direct UV-Vis spectrum visualization)
- TS Opt → Freq → IRC (with 1-imaginary frequency saddle point verification)
"""

import os
from typing import List, Optional
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .pipeline import ChainedWorkflowPipeline, WORKFLOW_TEMPLATES, WorkflowStageConfig
from ..qm_input_generator import QMCalculationParameters
from .base import OptimizationStep


class WorkflowPipelineDialog(QtWidgets.QDialog):
    """Dialog displaying stage progress, live log streaming, and verification for multi-stage QM pipelines."""

    pipeline_completed_with_geom = QtCore.pyqtSignal(list, np.ndarray, str)

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
        self.setWindowTitle("⚡ Chained Quantum Chemistry Workflow Pipeline")
        self.resize(850, 620)

        self.code_key = code_key
        self.symbols = symbols or ["O", "H", "H"]
        self.coords = coords if coords is not None else np.zeros((3, 3))
        self.base_params = base_params or QMCalculationParameters()
        self.work_dir = work_dir or os.path.join(os.getcwd(), "workflow_run")

        self.pipeline: Optional[ChainedWorkflowPipeline] = None
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Top Recipe Selector
        top_group = QtWidgets.QGroupBox("1. Select Workflow Template")
        top_layout = QtWidgets.QHBoxLayout(top_group)

        self.combo_recipe = QtWidgets.QComboBox()
        for name in WORKFLOW_TEMPLATES.keys():
            self.combo_recipe.addItem(name)
        self.combo_recipe.currentTextChanged.connect(self._on_recipe_changed)
        top_layout.addWidget(QtWidgets.QLabel("Recipe:"))
        top_layout.addWidget(self.combo_recipe, stretch=2)

        self.lbl_recipe_desc = QtWidgets.QLabel("")
        self.lbl_recipe_desc.setStyleSheet("color: #94a3b8;")
        top_layout.addWidget(self.lbl_recipe_desc, stretch=3)
        layout.addWidget(top_group)

        # Stage Progress List
        mid_group = QtWidgets.QGroupBox("2. Pipeline Stages")
        mid_layout = QtWidgets.QVBoxLayout(mid_group)

        self.list_stages = QtWidgets.QListWidget()
        self.list_stages.setAlternatingRowColors(True)
        mid_layout.addWidget(self.list_stages)
        layout.addWidget(mid_group)

        # Live Log Console
        log_group = QtWidgets.QGroupBox("3. Active Stage Live Console")
        log_layout = QtWidgets.QVBoxLayout(log_group)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Monospace", 9))
        self.txt_log.setStyleSheet("background-color: #0f172a; color: #f8fafc;")
        log_layout.addWidget(self.txt_log)
        layout.addWidget(log_group, stretch=1)

        # Status and Verification Banner
        self.lbl_status = QtWidgets.QLabel("Ready to launch pipeline.")
        self.lbl_status.setStyleSheet("font-weight: bold; font-size: 13px; color: #38bdf8;")
        layout.addWidget(self.lbl_status)

        # Bottom Buttons
        btn_layout = QtWidgets.QHBoxLayout()

        self.btn_run = QtWidgets.QPushButton("🚀 Launch Workflow Pipeline")
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px 14px; background-color: #0284c7; color: white;")
        self.btn_run.clicked.connect(self._on_launch_pipeline)
        btn_layout.addWidget(self.btn_run)

        self.btn_abort = QtWidgets.QPushButton("⏹ Abort")
        self.btn_abort.setEnabled(False)
        self.btn_abort.clicked.connect(self._on_abort_clicked)
        btn_layout.addWidget(self.btn_abort)

        btn_layout.addStretch()

        self.btn_adopt = QtWidgets.QPushButton("👁️ Adopt Final Geometry into Editor")
        self.btn_adopt.setEnabled(False)
        self.btn_adopt.clicked.connect(self._on_adopt_geometry)
        btn_layout.addWidget(self.btn_adopt)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btn_layout.addWidget(btn_close)

        layout.addLayout(btn_layout)

        self._on_recipe_changed(self.combo_recipe.currentText())

    def _on_recipe_changed(self, recipe_name: str):
        stages = WORKFLOW_TEMPLATES.get(recipe_name, [])
        self.list_stages.clear()
        for idx, st in enumerate(stages, start=1):
            item = QtWidgets.QListWidgetItem(f"⏳ Stage {idx}: {st.name} ({st.job_type}) - {st.description}")
            self.list_stages.addItem(item)
        self.lbl_recipe_desc.setText(f"{len(stages)} chained calculation stages")

    def _on_launch_pipeline(self):
        recipe_name = self.combo_recipe.currentText()
        stages = WORKFLOW_TEMPLATES.get(recipe_name, [])
        if not stages:
            return

        self.txt_log.clear()
        self.btn_run.setEnabled(False)
        self.btn_abort.setEnabled(True)
        self.btn_adopt.setEnabled(False)
        self.combo_recipe.setEnabled(False)

        self.pipeline = ChainedWorkflowPipeline(
            code_key=self.code_key,
            symbols=self.symbols,
            initial_coords=self.coords,
            base_params=self.base_params,
            stages=stages,
            work_dir=self.work_dir,
            parent=self,
        )
        self.pipeline.stage_started.connect(self._on_stage_started)
        self.pipeline.stage_finished.connect(self._on_stage_finished)
        self.pipeline.pipeline_finished.connect(self._on_pipeline_finished)
        self.pipeline.line_received.connect(self._on_line_received)
        self.pipeline.start()

    def _on_abort_clicked(self):
        if self.pipeline:
            self.pipeline.abort()
        self.btn_abort.setEnabled(False)
        self.btn_run.setEnabled(True)
        self.combo_recipe.setEnabled(True)

    def _on_stage_started(self, stage_idx: int, stage_name: str):
        if 0 <= stage_idx < self.list_stages.count():
            item = self.list_stages.item(stage_idx)
            item.setText(f"🔄 Stage {stage_idx + 1}: {stage_name} (Running...)")
            item.setForeground(QtGui.QColor("#38bdf8"))
        self.lbl_status.setText(f"Running Stage {stage_idx + 1}/{self.list_stages.count()}: {stage_name}...")

    def _on_stage_finished(self, stage_idx: int, stage_name: str, success: bool, msg: str):
        if 0 <= stage_idx < self.list_stages.count():
            item = self.list_stages.item(stage_idx)
            icon = "✔" if success else "✖"
            item.setText(f"{icon} Stage {stage_idx + 1}: {stage_name} - {msg}")
            item.setForeground(QtGui.QColor("#22c55e") if success else QtGui.QColor("#ef4444"))

    def _on_pipeline_finished(self, success: bool, msg: str):
        self.btn_run.setEnabled(True)
        self.btn_abort.setEnabled(False)
        self.combo_recipe.setEnabled(True)
        if success:
            self.lbl_status.setText(f"✔ {msg}")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 13px; color: #22c55e;")
            self.btn_adopt.setEnabled(True)
        else:
            self.lbl_status.setText(f"✖ {msg}")
            self.lbl_status.setStyleSheet("font-weight: bold; font-size: 13px; color: #ef4444;")

    def _on_line_received(self, line: str, is_err: bool):
        self.txt_log.appendPlainText(line)

    def _on_adopt_geometry(self):
        if self.pipeline and self.pipeline.current_coords is not None:
            self.pipeline_completed_with_geom.emit(
                self.symbols,
                self.pipeline.current_coords,
                f"Workflow Pipeline Result ({self.combo_recipe.currentText()})",
            )
            QtWidgets.QMessageBox.information(
                self, "Geometry Adopted", "Optimized pipeline geometry adopted into main editor!"
            )
            self.accept()
