"""
HPC & Cluster Job Submission Script Dialog for AgyMolden.

Enables users to configure, preview, customize, and export production-ready
batch scripts (SLURM, PBS Pro, IBM LSF, SGE) for high-performance computing clusters.
"""

import os
from typing import Optional, List
from PyQt6 import QtCore, QtGui, QtWidgets

from .hpc_batch_generator import (
    ClusterScriptOptions,
    generate_cluster_script,
)
from .dialog_service import dialog_service


class ClusterScriptDialog(QtWidgets.QDialog):
    """Interactive dialog to generate and preview cluster submission scripts (SLURM/PBS/LSF/SGE)."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        input_filename: str = "calculation.inp",
        qm_code: str = "ORCA",
        default_nprocs: int = 8,
        default_mem_gb: int = 16,
    ):
        super().__init__(parent)
        self.setWindowTitle(f"HPC Cluster Submission Script Generator — {qm_code}")
        self.resize(750, 620)

        self.input_filename = input_filename
        self.qm_code = qm_code

        self._init_ui(default_nprocs, default_mem_gb)
        self._update_preview()

    def _init_ui(self, default_nprocs: int, default_mem_gb: int):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # 1. Cluster & Scheduler Configuration Group
        grp_config = QtWidgets.QGroupBox("Cluster & Scheduler Parameters")
        cfg_layout = QtWidgets.QGridLayout(grp_config)
        cfg_layout.setSpacing(8)

        # Scheduler dropdown
        cfg_layout.addWidget(QtWidgets.QLabel("<b>Job Scheduler:</b>"), 0, 0)
        self.combo_sched = QtWidgets.QComboBox()
        self.combo_sched.addItems([
            "SLURM (sbatch)",
            "PBS Pro / TORQUE (qsub)",
            "IBM Spectrum LSF (bsub)",
            "Sun Grid Engine / SGE (qsub)",
        ])
        self.combo_sched.currentIndexChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.combo_sched, 0, 1)

        # Partition / Queue
        cfg_layout.addWidget(QtWidgets.QLabel("Partition / Queue:"), 0, 2)
        self.txt_partition = QtWidgets.QLineEdit("")
        self.txt_partition.setPlaceholderText("e.g. batch, gpu, standard (optional)")
        self.txt_partition.textChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.txt_partition, 0, 3)

        # Nodes
        cfg_layout.addWidget(QtWidgets.QLabel("Compute Nodes:"), 1, 0)
        self.spin_nodes = QtWidgets.QSpinBox()
        self.spin_nodes.setRange(1, 128)
        self.spin_nodes.setValue(1)
        self.spin_nodes.valueChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.spin_nodes, 1, 1)

        # CPUs per node
        cfg_layout.addWidget(QtWidgets.QLabel("CPUs / Cores:"), 1, 2)
        self.spin_cpus = QtWidgets.QSpinBox()
        self.spin_cpus.setRange(1, 256)
        self.spin_cpus.setValue(max(1, default_nprocs))
        self.spin_cpus.valueChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.spin_cpus, 1, 3)

        # Total Memory
        cfg_layout.addWidget(QtWidgets.QLabel("Total RAM (GB):"), 2, 0)
        self.spin_mem = QtWidgets.QSpinBox()
        self.spin_mem.setRange(1, 2048)
        self.spin_mem.setValue(max(1, default_mem_gb))
        self.spin_mem.valueChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.spin_mem, 2, 1)

        # Walltime
        cfg_layout.addWidget(QtWidgets.QLabel("Walltime (Hours):"), 2, 2)
        self.spin_walltime = QtWidgets.QSpinBox()
        self.spin_walltime.setRange(1, 720)
        self.spin_walltime.setValue(24)
        self.spin_walltime.valueChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.spin_walltime, 2, 3)

        # Scratch handling
        self.chk_scratch = QtWidgets.QCheckBox("Use Fast Node-Local Scratch ($TMPDIR) with Automatic Cleanup")
        self.chk_scratch.setChecked(True)
        self.chk_scratch.toggled.connect(self._update_preview)
        cfg_layout.addWidget(self.chk_scratch, 3, 0, 1, 4)

        # Modules to load
        cfg_layout.addWidget(QtWidgets.QLabel("Modules to load:"), 4, 0)
        self.txt_modules = QtWidgets.QLineEdit()
        default_mod = "orca/5.0.4 openmpi/4.1.1" if "ORCA" in self.qm_code.upper() else "gaussian/g16"
        self.txt_modules.setText(default_mod)
        self.txt_modules.setPlaceholderText("Space-separated modules (e.g. orca/5.0.4 openmpi/4.1.1)")
        self.txt_modules.textChanged.connect(self._update_preview)
        cfg_layout.addWidget(self.txt_modules, 4, 1, 1, 3)

        layout.addWidget(grp_config)

        # 2. Script Preview Section
        grp_prev = QtWidgets.QGroupBox("Generated Cluster Batch Script Preview")
        prev_layout = QtWidgets.QVBoxLayout(grp_prev)

        self.text_preview = QtWidgets.QPlainTextEdit()
        self.text_preview.setFont(QtGui.QFont("Monospace", 10))
        prev_layout.addWidget(self.text_preview)
        layout.addWidget(grp_prev, stretch=1)

        # 3. Actions / Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Script")
        btn_copy.clicked.connect(self._on_copy)
        btn_box.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save Script (.sh / .slurm)...")
        btn_save.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 5px 12px;")
        btn_save.clicked.connect(self._on_save)
        btn_box.addWidget(btn_save)

        btn_box.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)

        layout.addLayout(btn_box)

    def _get_scheduler_key(self) -> str:
        text = self.combo_sched.currentText().upper()
        if "PBS" in text:
            return "PBS"
        elif "LSF" in text:
            return "LSF"
        elif "SGE" in text or "SUN" in text:
            return "SGE"
        return "SLURM"

    def _update_preview(self):
        sched = self._get_scheduler_key()
        modules = [m.strip() for m in self.txt_modules.text().split() if m.strip()]
        opts = ClusterScriptOptions(
            scheduler=sched,
            job_name=os.path.splitext(os.path.basename(self.input_filename))[0],
            queue_or_partition=self.txt_partition.text().strip(),
            nodes=self.spin_nodes.value(),
            cpus_per_node=self.spin_cpus.value(),
            memory_gb=self.spin_mem.value(),
            walltime_hours=self.spin_walltime.value(),
            modules_to_load=modules,
            use_scratch=self.chk_scratch.isChecked(),
        )

        script = generate_cluster_script(
            input_file_path=self.input_filename,
            qm_code=self.qm_code,
            options=opts,
        )
        self.text_preview.setPlainText(script)

    def _on_copy(self):
        app = QtWidgets.QApplication.instance()
        if app:
            app.clipboard().setText(self.text_preview.toPlainText())
            dialog_service.info(self, "Copied", "Cluster batch script copied to clipboard!")

    def _on_save(self):
        stem = os.path.splitext(os.path.basename(self.input_filename))[0]
        sched = self._get_scheduler_key()
        ext = ".slurm" if sched == "SLURM" else ".sh"
        default_name = f"submit_{stem}{ext}"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save HPC Batch Submission Script",
            default_name,
            "Shell Scripts (*.sh *.slurm *.bash);;All Files (*)",
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.text_preview.toPlainText())
                dialog_service.info(self, "Saved", f"Cluster submission script saved to:\n{path}")
            except Exception as e:
                dialog_service.error(self, "Save Error", f"Failed to save script:\n{e}", exc=e)
