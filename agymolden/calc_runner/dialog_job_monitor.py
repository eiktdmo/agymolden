"""
Live Calculation Job Monitor & Controller Dialog.

Features:
- Real-time streaming log console with auto-scroll and search
- Status badge (Running, Converged, Finished, Stopped, Failed)
- Progress & Optimization convergence table (Step, Energy, Delta E)
- Live Viewport Update checkbox: updates main 3D OpenGL viewport at each step
- Pause, Resume, Stop/Kill buttons
- Conda Environment & Wrapper selector bar
"""

import html
import os
import sys
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from .brorsen_runner import BrorsenRunner, detect_conda_environments, find_brorsen_wrappers
from .orca_runner import OrcaRunner


class JobMonitorDialog(QtWidgets.QDialog):
    """Real-time job monitor dialog with streaming logs and live 3D updates."""

    live_step_received = QtCore.pyqtSignal(list, np.ndarray, float)  # (symbols, coords, energy)
    calculation_completed = QtCore.pyqtSignal(str)  # output_path or input_path

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, runner: Optional[BaseCalcRunner] = None):
        super().__init__(parent)
        self.setWindowTitle("Calculation Job Monitor & Live Output")
        self.resize(960, 680)

        self.runner = runner or BrorsenRunner(self)
        self._open_viewers = []
        self._init_ui()
        self._connect_runner_signals()
        self._load_saved_settings()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # 1. Environment & Wrapper Header Group
        self.env_group = QtWidgets.QGroupBox("Calculation Engine & Environment")
        env_layout = QtWidgets.QGridLayout(self.env_group)
        env_layout.setContentsMargins(10, 8, 10, 8)
        env_layout.setHorizontalSpacing(10)

        # Python environment (venv / conda) dropdown
        env_layout.addWidget(QtWidgets.QLabel("<b>Python Environment:</b>"), 0, 0)
        self.combo_conda = QtWidgets.QComboBox()
        self.combo_conda.setToolTip("Select Python interpreter, Virtualenv, or Conda environment")
        self._populate_conda_envs()
        self.combo_conda.currentIndexChanged.connect(self._on_conda_changed)
        env_layout.addWidget(self.combo_conda, 0, 1)

        btn_browse_py = QtWidgets.QPushButton("Browse Python...")
        btn_browse_py.clicked.connect(self._on_browse_python)
        env_layout.addWidget(btn_browse_py, 0, 2)

        # Wrapper script dropdown
        env_layout.addWidget(QtWidgets.QLabel("<b>Brorsen Wrapper:</b>"), 1, 0)
        self.combo_wrapper = QtWidgets.QComboBox()
        self._populate_wrappers()
        env_layout.addWidget(self.combo_wrapper, 1, 1)

        btn_browse_wrap = QtWidgets.QPushButton("Browse Wrapper...")
        btn_browse_wrap.clicked.connect(self._on_browse_wrapper)
        env_layout.addWidget(btn_browse_wrap, 1, 2)

        layout.addWidget(self.env_group)

        # 2. Status & Controls Bar
        status_bar = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Status: <b>IDLE</b>")
        self.lbl_status.setStyleSheet("padding: 4px 8px; border-radius: 4px; background-color: #374151; color: white;")
        status_bar.addWidget(self.lbl_status)

        self.lbl_job_info = QtWidgets.QLabel("No active calculation")
        self.lbl_job_info.setStyleSheet("color: #9ca3af; font-size: 11px;")
        status_bar.addWidget(self.lbl_job_info, stretch=1)

        self.chk_live_3d = QtWidgets.QCheckBox("🔄 Live 3D Viewport Update")
        self.chk_live_3d.setChecked(True)
        self.chk_live_3d.setToolTip("Automatically update the main 3D OpenGL viewport at each optimization step")
        status_bar.addWidget(self.chk_live_3d)

        self.btn_stop = QtWidgets.QPushButton("⏹ Stop Calculation")
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold;")
        self.btn_stop.clicked.connect(self._on_stop_clicked)
        status_bar.addWidget(self.btn_stop)

        layout.addLayout(status_bar)

        # 3. Main Splitter: Tabs (Log Stream / Optimization Steps)
        tabs = QtWidgets.QTabWidget()

        # Tab 1: Live Log Stream
        tab_log = QtWidgets.QWidget()
        log_layout = QtWidgets.QVBoxLayout(tab_log)
        log_layout.setContentsMargins(6, 6, 6, 6)

        # Search bar
        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setPlaceholderText("Filter output text...")
        self.txt_filter.textChanged.connect(self._on_filter_changed)
        search_layout.addWidget(self.txt_filter, stretch=1)

        self.chk_autoscroll = QtWidgets.QCheckBox("Auto-scroll")
        self.chk_autoscroll.setChecked(True)
        search_layout.addWidget(self.chk_autoscroll)

        btn_clear_log = QtWidgets.QPushButton("Clear")
        btn_clear_log.clicked.connect(self._on_clear_log)
        search_layout.addWidget(btn_clear_log)

        log_layout.addLayout(search_layout)

        self.txt_log = QtWidgets.QPlainTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFont(QtGui.QFont("Monospace", 9))
        self.txt_log.setMaximumBlockCount(10000)  # Prevent unbounded memory and UI redraw lag on huge logs
        self._apply_theme_styles()
        log_layout.addWidget(self.txt_log)

        tabs.addTab(tab_log, "📄 Live Console Output")

        # Tab 2: Optimization Trajectory Steps Table
        tab_steps = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(tab_steps)
        steps_layout.setContentsMargins(6, 6, 6, 6)

        self.tbl_steps = QtWidgets.QTableWidget()
        self.tbl_steps.setColumnCount(4)
        self.tbl_steps.setHorizontalHeaderLabels(["Step / Cycle", "Total Energy (Hartree / a.u.)", "ΔE", "Status"])
        self.tbl_steps.horizontalHeader().setStretchLastSection(True)
        self.tbl_steps.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_steps.cellClicked.connect(self._on_step_row_clicked)
        steps_layout.addWidget(self.tbl_steps)

        tabs.addTab(tab_steps, "📈 Optimization Progress")

        # Tab 3: Batch Calculation Queue
        tab_queue = QtWidgets.QWidget()
        queue_layout = QtWidgets.QVBoxLayout(tab_queue)
        queue_layout.setContentsMargins(6, 6, 6, 6)

        q_btn_bar = QtWidgets.QHBoxLayout()
        btn_add_queue = QtWidgets.QPushButton("➕ Add Input Files...")
        btn_add_queue.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 8px;")
        btn_add_queue.clicked.connect(self._on_add_to_queue)
        q_btn_bar.addWidget(btn_add_queue)

        btn_start_queue = QtWidgets.QPushButton("▶️ Start Batch Queue")
        btn_start_queue.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 4px 8px;")
        btn_start_queue.clicked.connect(self._on_start_queue)
        q_btn_bar.addWidget(btn_start_queue)

        btn_clear_queue = QtWidgets.QPushButton("Clear Queue")
        btn_clear_queue.clicked.connect(self._on_clear_queue)
        q_btn_bar.addWidget(btn_clear_queue)
        q_btn_bar.addStretch()

        queue_layout.addLayout(q_btn_bar)

        self.tbl_queue = QtWidgets.QTableWidget()
        self.tbl_queue.setColumnCount(4)
        self.tbl_queue.setHorizontalHeaderLabels(["#", "Input Deck", "Target Engine", "Status"])
        self.tbl_queue.horizontalHeader().setStretchLastSection(True)
        self.tbl_queue.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        queue_layout.addWidget(self.tbl_queue)

        tabs.addTab(tab_queue, "📋 Batch Queue")

        self.batch_queue: List[Dict[str, str]] = []
        self.current_queue_idx: int = -1

        layout.addWidget(tabs, stretch=1)

        # 4. Bottom action buttons
        bottom_layout = QtWidgets.QHBoxLayout()
        self.chk_auto_load = QtWidgets.QCheckBox("Auto-load results into viewer on completion")
        self.chk_auto_load.setChecked(True)
        self.chk_auto_load.setToolTip("Automatically load output log, optimized coordinates, orbitals, and Molden wavefunctions once calculation completes.")
        bottom_layout.addWidget(self.chk_auto_load)

        self.btn_load_results = QtWidgets.QPushButton("📂 Load Results into Viewer")
        self.btn_load_results.setEnabled(False)
        self.btn_load_results.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold;")
        self.btn_load_results.clicked.connect(self._on_load_results)
        bottom_layout.addWidget(self.btn_load_results)

        # Cross-Package Multi-Stage Handoff Menu Button (Option B)
        self.btn_stage_handoff = QtWidgets.QPushButton("➡️ Send to Next Package Stage ▾")
        self.btn_stage_handoff.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_stage_handoff.setToolTip(
            "Multi-Package Calculation Pipeline: Pass final optimized geometry and orbitals from this calculation\n"
            "directly into newBRORSEN, ORCA, PySCF, Gaussian, or openLOWDIN to start the next stage!"
        )
        self.btn_stage_handoff.setEnabled(False)
        self.menu_handoff = QtWidgets.QMenu(self)
        self.btn_stage_handoff.setMenu(self.menu_handoff)
        self._build_handoff_menu()
        bottom_layout.addWidget(self.btn_stage_handoff)

        btn_view_input = QtWidgets.QPushButton("📝 View Input File...")
        btn_view_input.clicked.connect(self._on_open_input_viewer)
        bottom_layout.addWidget(btn_view_input)

        btn_view_full_out = QtWidgets.QPushButton("📄 Open Output Log...")
        btn_view_full_out.clicked.connect(self._on_open_output_viewer)
        bottom_layout.addWidget(btn_view_full_out)

        btn_open_agemacs = QtWidgets.QPushButton("🚀 Open in agemacs")
        btn_open_agemacs.setToolTip("Open the current input/output files in agemacs")
        btn_open_agemacs.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        btn_open_agemacs.clicked.connect(self._on_open_in_agemacs)
        bottom_layout.addWidget(btn_open_agemacs)

        self.btn_auto_fix = QtWidgets.QPushButton("🩹 Auto-Fix & Relaunch")
        self.btn_auto_fix.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 4px 10px;")
        self.btn_auto_fix.setToolTip("QC Doctor: Automatically inject convergence remedies into the input file and restart calculation")
        self.btn_auto_fix.setVisible(False)
        self.btn_auto_fix.clicked.connect(self._on_auto_fix_and_relaunch)
        bottom_layout.addWidget(self.btn_auto_fix)

        bottom_layout.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.hide)
        bottom_layout.addWidget(btn_close)

        layout.addLayout(bottom_layout)

    def _build_handoff_menu(self):
        self.menu_handoff.clear()
        act_brorsen = self.menu_handoff.addAction("✨ newBRORSEN (NEO / Positron / CCSD(T))")
        act_brorsen.triggered.connect(lambda: self._on_handoff_to("brorsen"))
        act_orca = self.menu_handoff.addAction("⚡ ORCA (DLPNO-CCSD(T) / DFT / MD)")
        act_orca.triggered.connect(lambda: self._on_handoff_to("orca"))
        act_pyscf = self.menu_handoff.addAction("🐍 PySCF (Wavefunction / CASSCF / FCI)")
        act_pyscf.triggered.connect(lambda: self._on_handoff_to("pyscf"))
        act_gauss = self.menu_handoff.addAction("⚛️ Gaussian (Opt / Freq / IRC)")
        act_gauss.triggered.connect(lambda: self._on_handoff_to("gaussian"))
        act_xtb = self.menu_handoff.addAction("⚡ xTB / GFN2 (Conformer Screening / Solvation)")
        act_xtb.triggered.connect(lambda: self._on_handoff_to("xtb"))
        act_lowdin = self.menu_handoff.addAction("🔬 openLOWDIN (Multi-Component Propagator)")
        act_lowdin.triggered.connect(lambda: self._on_handoff_to("openlowdin"))

    def _on_handoff_to(self, target_code: str):
        """Pass final optimized geometry and artifacts into target package setup dialog."""
        # 1. First ensure results are loaded into main window
        self._on_load_results()

        main_win = self.parent()
        if main_win is None:
            return

        # 2. Open the appropriate dialog pre-populated with final geometry
        if target_code == "brorsen":
            if hasattr(main_win, "_on_open_brorsen_dialog"):
                main_win._on_open_brorsen_dialog()
        else:
            if hasattr(main_win, "_on_open_qm_dialog"):
                dlg = main_win._on_open_qm_dialog(target_code=target_code)

    def _populate_conda_envs(self):
        self.combo_conda.clear()
        envs = detect_conda_environments()
        for name, path in envs:
            self.combo_conda.addItem(name, path)

    def _populate_wrappers(self):
        self.combo_wrapper.clear()
        wrappers = find_brorsen_wrappers()
        for w in wrappers:
            label = f"{os.path.basename(w)} ({os.path.dirname(w)})"
            self.combo_wrapper.addItem(label, w)

    def _on_browse_python(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Python Executable", "", "Python Executable (python*);;All Files (*)"
        )
        if path and os.path.isfile(path):
            self.combo_conda.addItem(f"Custom: {path}", path)
            self.combo_conda.setCurrentIndex(self.combo_conda.count() - 1)

    def _on_browse_wrapper(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select newBRORSEN Wrapper Script", "", "Python Scripts (*.py);;All Files (*)"
        )
        if path and os.path.isfile(path):
            self.combo_wrapper.addItem(f"Custom: {os.path.basename(path)}", path)
            self.combo_wrapper.setCurrentIndex(self.combo_wrapper.count() - 1)

    def _on_conda_changed(self, idx: int):
        self._save_settings()

    def _load_saved_settings(self):
        from agymolden.session_state import get_settings
        settings = get_settings("JobRunner")
        saved_py = settings.value("python_bin", "")
        saved_wrap = settings.value("brorsen_wrapper", "")

        if saved_py:
            idx = self.combo_conda.findData(saved_py)
            if idx >= 0:
                self.combo_conda.setCurrentIndex(idx)
        if saved_wrap:
            idx = self.combo_wrapper.findData(saved_wrap)
            if idx >= 0:
                self.combo_wrapper.setCurrentIndex(idx)

    def _save_settings(self):
        from agymolden.session_state import get_settings
        settings = get_settings("JobRunner")
        py_bin = self.combo_conda.currentData()
        wrapper = self.combo_wrapper.currentData()
        if py_bin:
            settings.setValue("python_bin", py_bin)
        if wrapper:
            settings.setValue("brorsen_wrapper", wrapper)

    def _connect_runner_signals(self):
        self.runner.line_received.connect(self._on_line_received)
        self.runner.status_changed.connect(self._on_status_changed)
        self.runner.step_updated.connect(self._on_step_updated)
        self.runner.finished.connect(self._on_finished)
        if hasattr(self.runner, "artifacts_downloaded"):
            self.runner.artifacts_downloaded.connect(self._on_artifacts_downloaded)

    def _on_artifacts_downloaded(self, path: str):
        if path and os.path.isfile(path):
            self.lbl_job_info.setText(f"Artifacts transferred: {os.path.basename(path)}")
            self.runner.log_file = path
        else:
            self.lbl_job_info.setText(f"Artifacts transferred to: {os.path.basename(path or self.runner.work_dir)}")
        self.btn_load_results.setEnabled(True)

    def run_calculation(self, input_file: str, runner_type: str = "brorsen"):
        """Start calculation with the selected environment and options."""
        self._apply_theme_styles()
        self._save_settings()
        self.txt_log.clear()
        self.tbl_steps.setRowCount(0)
        self.btn_load_results.setEnabled(False)
        self.btn_stage_handoff.setEnabled(False)

        if hasattr(self, "env_group"):
            self.env_group.setVisible(runner_type in ("brorsen", "pyscf"))
            self.combo_wrapper.setEnabled(runner_type == "brorsen")

        # Use runner's existing configured python/wrapper if already set, otherwise fall back to UI
        if runner_type == "brorsen" and isinstance(self.runner, BrorsenRunner):
            py_bin = self.runner.python_bin or self.combo_conda.currentData() or sys.executable
            wrapper = self.runner.wrapper_path or self.combo_wrapper.currentData() or ""
            self.runner.configure(python_bin=py_bin, wrapper_path=wrapper, extra_flags=self.runner.extra_flags)
        elif runner_type == "pyscf" and hasattr(self.runner, "python_bin"):
            py_bin = getattr(self.runner, "python_bin", None) or self.combo_conda.currentData() or sys.executable
            self.runner.python_bin = py_bin

        from agymolden.session_state import set_last_dir
        set_last_dir(input_file)
        self.lbl_job_info.setText(f"File: {os.path.basename(input_file)} | WorkDir: {os.path.dirname(input_file)}")
        self.btn_stop.setEnabled(True)

        self.show()
        self.raise_()
        self.activateWindow()
        QtWidgets.QApplication.processEvents()

        success = self.runner.start_job(input_file)
        if not success:
            last_err = getattr(self.runner, "last_error_message", "") or getattr(self.runner, "_status_message", "")
            err_details = f"\n\nDetails: {last_err}" if last_err else ""
            QtWidgets.QMessageBox.critical(self, "Launch Error", f"Failed to start calculation for {input_file}{err_details}")
            self.btn_stop.setEnabled(False)

    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)
        self._apply_theme_styles()

    def _is_white_theme(self) -> bool:
        """Check if application or viewer is currently using white theme."""
        # 1. First check parent main window's explicit palette / theme setting
        p = self.parent()
        while p is not None:
            if hasattr(p, "viewer") and getattr(p.viewer, "bg_color_mode", ""):
                return p.viewer.bg_color_mode == "white"
            p = p.parent()
        # 2. Check palette Window color brightness as ground truth
        window_lum = self.palette().color(QtGui.QPalette.ColorRole.Window).lightness()
        if window_lum < 100:
            return False
        # 3. Fall back to QSettings
        from agymolden.session_state import get_settings
        settings = get_settings("AgyMolden")
        return settings.value("color_theme", "white") == "white"

    def _apply_theme_styles(self):
        is_white = self._is_white_theme()
        if hasattr(self, "txt_log"):
            if is_white:
                self.txt_log.setStyleSheet(
                    "background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; "
                    "selection-background-color: #2563eb; selection-color: #ffffff;"
                )
            else:
                self.txt_log.setStyleSheet(
                    "background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; "
                    "selection-background-color: #3b82f6; selection-color: #ffffff;"
                )
        if hasattr(self, "tbl_steps"):
            if is_white:
                self.tbl_steps.setStyleSheet(
                    "QTableWidget { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; gridline-color: #e2e8f0; } "
                    "QHeaderView::section { background-color: #f1f5f9; color: #475569; border: 1px solid #cbd5e1; font-weight: bold; }"
                )
            else:
                self.tbl_steps.setStyleSheet(
                    "QTableWidget { background-color: #1e1e24; color: #f8fafc; border: 1px solid #334155; gridline-color: #334155; } "
                    "QHeaderView::section { background-color: #24242c; color: #94a3b8; border: 1px solid #334155; font-weight: bold; }"
                )

    def _on_line_received(self, line: str, is_stderr: bool):
        escaped_line = html.escape(line)
        is_white = self._is_white_theme()
        if is_stderr:
            color = "#dc2626" if is_white else "#f87171"
        else:
            color = "#0f172a" if is_white else "#f8fafc"

        formatted = f'<span style="color: {color};">{escaped_line}</span>'
        self.txt_log.appendHtml(formatted)
        if self.chk_autoscroll.isChecked():
            cursor = self.txt_log.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.txt_log.setTextCursor(cursor)

    def _on_status_changed(self, status: JobStatus, message: str):
        color_map = {
            JobStatus.IDLE: "#374151",
            JobStatus.STARTING: "#3b82f6",
            JobStatus.RUNNING: "#2563eb",
            JobStatus.CONVERGED: "#059669",
            JobStatus.FINISHED: "#10b981",
            JobStatus.FAILED: "#dc2626",
            JobStatus.STOPPED: "#d97706",
        }
        color = color_map.get(status, "#374151")
        self.lbl_status.setText(f"Status: <b>{status.value.upper()}</b>")
        self.lbl_status.setStyleSheet(f"padding: 4px 8px; border-radius: 4px; background-color: {color}; color: white; font-weight: bold;")
        if message:
            self.lbl_job_info.setText(message)

    def _on_step_updated(self, step: OptimizationStep):
        # Update progress table
        row = self.tbl_steps.rowCount()
        self.tbl_steps.insertRow(row)

        c_item = QtWidgets.QTableWidgetItem(f"Step {step.cycle}")
        e_item = QtWidgets.QTableWidgetItem(f"{step.energy:.8f}" if step.energy is not None else "-")
        delta_str = "-"
        if len(self.runner.steps) > 1 and step.energy is not None and self.runner.steps[-2].energy is not None:
            d_val = step.energy - self.runner.steps[-2].energy
            delta_str = f"{d_val:+.6f}"
        d_item = QtWidgets.QTableWidgetItem(delta_str)
        s_item = QtWidgets.QTableWidgetItem("Converged 🎉" if step.is_converged else "Optimizing...")

        self.tbl_steps.setItem(row, 0, c_item)
        self.tbl_steps.setItem(row, 1, e_item)
        self.tbl_steps.setItem(row, 2, d_item)
        self.tbl_steps.setItem(row, 3, s_item)
        self.tbl_steps.scrollToBottom()

        # Update 3D viewport live if requested
        if self.chk_live_3d.isChecked() and step.symbols and step.coords is not None:
            self.live_step_received.emit(step.symbols, step.coords, step.energy or 0.0)

    def _on_step_row_clicked(self, row: int, col: int):
        if 0 <= row < len(self.runner.steps):
            step = self.runner.steps[row]
            if step.symbols and step.coords is not None:
                self.live_step_received.emit(step.symbols, step.coords, step.energy or 0.0)

    def _on_stop_clicked(self):
        reply = QtWidgets.QMessageBox.question(
            self, "Stop Calculation", "Are you sure you want to terminate the active calculation?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.runner.stop_job()
            self.btn_stop.setEnabled(False)

    def _on_finished(self, success: bool, message: str):
        self.btn_stop.setEnabled(False)
        self.btn_load_results.setEnabled(True)
        self.btn_stage_handoff.setEnabled(True)
        if success:
            if hasattr(self, "chk_auto_load") and self.chk_auto_load.isChecked():
                self._on_load_results()
            if self._has_active_queue():
                self._advance_queue(completed_success=True)
            else:
                QtWidgets.QMessageBox.information(self, "Calculation Completed", f"Job completed successfully!\n{message}")
        else:
            if self._has_active_queue():
                self._advance_queue(completed_success=False)
            # Check for beginner diagnostic recommendations
            from .diagnostics import diagnose_qm_error
            raw_text = self.txt_log.toPlainText()
            eng_hint = getattr(self.runner, "program_name", "orca") or "orca"
            diag = diagnose_qm_error(raw_text, engine=eng_hint)
            if diag:
                self._current_diagnostic = diag
                if diag.suggested_keywords:
                    self.btn_auto_fix.setVisible(True)
                    self.btn_auto_fix.setText(f"🩹 Auto-Fix ({diag.title[:20]}...) & Relaunch")

                msg_box = QtWidgets.QMessageBox(self)
                msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
                msg_box.setWindowTitle(f"QC Doctor: {diag.title}")
                msg_box.setText(f"<b>🩺 QC Doctor Diagnostic: {diag.title}</b><br><br>{diag.summary}")
                msg_box.setInformativeText(f"<b>Suggested Remedies:</b>\n{diag.remedy}")
                msg_box.setDetailedText(raw_text[-2000:] if len(raw_text) > 2000 else raw_text)

                btn_fix = None
                if diag.suggested_keywords:
                    btn_fix = msg_box.addButton("⚡ Apply Fix & Relaunch", QtWidgets.QMessageBox.ButtonRole.ActionRole)
                msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)

                msg_box.exec()
                if btn_fix and msg_box.clickedButton() == btn_fix:
                    self._on_auto_fix_and_relaunch()
            else:
                self.btn_auto_fix.setVisible(False)
                QtWidgets.QMessageBox.warning(self, "Calculation Finished", f"Job finished with notice/error:\n{message}")

    def _on_auto_fix_and_relaunch(self):
        """Automatically inject QC Doctor remedies into input deck and restart job."""
        diag = getattr(self, "_current_diagnostic", None)
        if not diag or not diag.suggested_keywords or not self.runner or not self.runner.input_file:
            return

        inp_path = self.runner.input_file
        if not os.path.isfile(inp_path):
            return

        eng = getattr(self.runner, "program_name", "orca") or "orca"
        try:
            from .diagnostics import patch_input_file_with_remedy
            success = patch_input_file_with_remedy(inp_path, diag, engine=eng)
            if not success:
                return

            self.btn_auto_fix.setVisible(False)
            self.txt_log.append(f"\n[QC Doctor] Injected recovery parameters into {os.path.basename(inp_path)}:\n{diag.suggested_keywords}\nRestarting calculation...\n")
            self.run_calculation(inp_path, runner_type=eng)

        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Auto-Fix Error", f"Failed to auto-apply fix:\n{err}")

    def _on_add_to_queue(self):
        """Add calculation input decks to batch execution queue."""
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Calculation Input Decks", "",
            "All Calculation Inputs (*.inp *.in *.gjf *.com *.lowdin *.py);;ORCA Inputs (*.inp);;Gaussian Inputs (*.gjf *.com);;All Files (*)"
        )
        if not files:
            return

        for f_path in files:
            ext = os.path.splitext(f_path)[1].lower()
            eng = "orca"
            if ext in (".gjf", ".com"):
                eng = "gaussian"
            elif ext == ".py":
                eng = "pyscf"
            elif ext == ".lowdin":
                eng = "openlowdin"

            self.batch_queue.append({
                "path": f_path,
                "engine": eng,
                "status": "Pending",
            })
        self._refresh_queue_table()

    def _refresh_queue_table(self):
        self.tbl_queue.setRowCount(len(self.batch_queue))
        for row, item in enumerate(self.batch_queue):
            self.tbl_queue.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
            self.tbl_queue.setItem(row, 1, QtWidgets.QTableWidgetItem(os.path.basename(item["path"])))
            self.tbl_queue.setItem(row, 2, QtWidgets.QTableWidgetItem(item["engine"].upper()))
            st_item = QtWidgets.QTableWidgetItem(item["status"])
            if item["status"] == "Running":
                st_item.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            elif item["status"] == "Done":
                st_item.setForeground(QtGui.QBrush(QtGui.QColor("#22c55e")))
            elif item["status"] == "Failed":
                st_item.setForeground(QtGui.QBrush(QtGui.QColor("#ef4444")))
            self.tbl_queue.setItem(row, 3, st_item)

    def _on_clear_queue(self):
        self.batch_queue.clear()
        self.current_queue_idx = -1
        self._refresh_queue_table()

    def _has_active_queue(self) -> bool:
        return 0 <= self.current_queue_idx < len(self.batch_queue)

    def _on_start_queue(self):
        if not self.batch_queue:
            QtWidgets.QMessageBox.information(self, "Queue Empty", "Please add calculation input decks first.")
            return
        self.current_queue_idx = 0
        self._run_current_queue_job()

    def _run_current_queue_job(self):
        if not (0 <= self.current_queue_idx < len(self.batch_queue)):
            QtWidgets.QMessageBox.information(self, "Queue Complete", "All batch calculation jobs in queue have finished.")
            return

        item = self.batch_queue[self.current_queue_idx]
        item["status"] = "Running"
        self._refresh_queue_table()
        self.run_calculation(item["path"], runner_type=item["engine"])

    def _advance_queue(self, completed_success: bool):
        if not (0 <= self.current_queue_idx < len(self.batch_queue)):
            return
        self.batch_queue[self.current_queue_idx]["status"] = "Done" if completed_success else "Failed"
        self.current_queue_idx += 1
        self._refresh_queue_table()
        if self.current_queue_idx < len(self.batch_queue):
            QtCore.QTimer.singleShot(1000, self._run_current_queue_job)
        else:
            QtWidgets.QMessageBox.information(self, "Queue Complete", "All batch jobs in the queue have completed.")

    def _launch_fixed_qm_input(self, diag):
        """Open QM Calculation Setup with QC Doctor recommended keywords pre-filled."""
        parent = self.parent()
        if parent and hasattr(parent, "_on_open_qm_calculation_dialog"):
            parent._on_open_qm_calculation_dialog()
            # If a dialog was opened and stored, apply remedy
            active_dlg = getattr(parent, "_active_qm_dialog", None)
            if active_dlg and hasattr(active_dlg, "apply_qc_doctor_remedy"):
                active_dlg.apply_qc_doctor_remedy(diag.suggested_keywords, title=diag.title)

    def _on_load_results(self):
        if self.runner.input_file:
            # Note: Do not permanently close or destroy the monitor, simply hide it while loading results
            self.hide()
            self.calculation_completed.emit(self.runner.input_file)

    def _open_viewer_nonmodal(self, file_path: str):
        from agymolden.dialog_output_viewer import QMOutputViewerDialog

        dlg = QMOutputViewerDialog(None, file_path=file_path)
        dlg.setWindowFlags(
            QtCore.Qt.WindowType.Window
            | QtCore.Qt.WindowType.WindowMinMaxButtonsHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        self._open_viewers.append(dlg)
        dlg.show()
        dlg.raise_()
        return dlg

    def _on_open_input_viewer(self):
        inp_path = getattr(self.runner, "input_file", None)
        if not inp_path or not os.path.isfile(inp_path):
            inp_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Input File to View",
                getattr(self.runner, "work_dir", "") or "",
                "Input Files (*.inp *.in *.gjf *.com *.lowdin *.py);;All Files (*)",
            )
        if inp_path and os.path.isfile(inp_path):
            self._open_viewer_nonmodal(inp_path)

    def _on_open_in_agemacs(self):
        import shutil
        import subprocess

        targets = []
        inp_path = getattr(self.runner, "input_file", None)
        log_path = getattr(self.runner, "log_file", None)
        work_dir = getattr(self.runner, "work_dir", None)
        if inp_path and os.path.isfile(inp_path):
            targets.append(inp_path)
        if log_path and os.path.isfile(log_path):
            targets.append(log_path)
        if not targets and work_dir and os.path.isdir(work_dir):
            targets.append(work_dir)
        if not targets:
            QtWidgets.QMessageBox.information(self, "Open in agemacs", "No active input, output, or working directory to open.")
            return

        from agymolden.companion import send_to_agemacs
        if send_to_agemacs(targets, self):
            return

        # Classroom fallback: if agemacs is not installed on this machine, open file in built-in viewer
        file_targets = [t for t in targets if os.path.isfile(t)]
        if file_targets:
            self._open_viewer_nonmodal(file_targets[0])
            return

        QtWidgets.QMessageBox.information(
            self,
            "Open in agemacs",
            "Could not find 'agemacs' on PATH or in standard project folders.\n\nFiles ready to open:\n" + "\n".join(targets),
        )

    def _on_open_output_viewer(self):
        log_path = getattr(self.runner, "log_file", None)
        if not log_path or not os.path.isfile(log_path):
            if self.runner.input_file:
                base_no_ext = os.path.splitext(self.runner.input_file)[0]
                # Also check in work_dir
                dirs_to_check = [os.path.dirname(self.runner.input_file)]
                if getattr(self.runner, "work_dir", None) and self.runner.work_dir not in dirs_to_check:
                    dirs_to_check.append(self.runner.work_dir)
                for d in dirs_to_check:
                    for ext in (".out", ".log"):
                        cand = os.path.join(d, os.path.basename(base_no_ext) + ext)
                        if os.path.isfile(cand):
                            log_path = cand
                            break
                    if log_path:
                        break
        if not log_path or not os.path.isfile(log_path):
            log_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Output File to View", self.runner.work_dir or "", "Output Files (*.out *.log *.txt);;All Files (*)"
            )
        if log_path and os.path.isfile(log_path):
            self.runner.log_file = log_path
            self._open_viewer_nonmodal(log_path)

    def _on_filter_changed(self, text: str):
        if not text:
            self.txt_log.setExtraSelections([])
            return

        selections = []
        doc = self.txt_log.document()
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QColor("#f59e0b"))
        fmt.setForeground(QtGui.QColor("#000000"))

        cursor = QtGui.QTextCursor(doc)
        count = 0
        while True:
            cursor = doc.find(text, cursor)
            if cursor.isNull():
                break
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = fmt
            selections.append(sel)
            count += 1
            if count > 2000:
                break
        self.txt_log.setExtraSelections(selections)

    def _on_clear_log(self):
        self.txt_log.clear()
