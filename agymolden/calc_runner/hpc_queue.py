"""
HPC Cluster Job Queue & Workload Manager for AgyMolden.

Provides remote queue monitoring and management for:
- Slurm Workload Manager (squeue / scancel)
- PBS / Torque / Sun Grid Engine (qstat / qdel)

Features:
- Non-blocking remote query over SSH using configured host profiles
- Color-coded job states (RUNNING, PENDING, COMPLETED, FAILED)
- 1-Click remote job cancellation (scancel <id>)
- Automatic periodic polling or on-demand refresh
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import subprocess
import shlex
import logging
from PyQt6 import QtCore, QtGui, QtWidgets

from .remote_ssh import RemoteHostConfig, parse_ssh_config_hosts

log = logging.getLogger(__name__)


@dataclass
class HpcJobRecord:
    job_id: str
    name: str
    user: str
    state: str
    time_elapsed: str
    time_limit: str
    partition: str
    reason_or_nodes: str


def parse_squeue_output(raw_text: str) -> List[HpcJobRecord]:
    """
    Parse pipe-delimited or whitespace-separated Slurm squeue output.
    Expected format: JOBID|NAME|USER|STATE|TIME|TIME_LIMIT|PARTITION|NODELIST(REASON)
    """
    records: List[HpcJobRecord] = []
    lines = raw_text.strip().splitlines()
    if not lines:
        return records

    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("jobid"):
            continue

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                records.append(
                    HpcJobRecord(
                        job_id=parts[0],
                        name=parts[1] if len(parts) > 1 else "",
                        user=parts[2] if len(parts) > 2 else "",
                        state=parts[3] if len(parts) > 3 else "UNKNOWN",
                        time_elapsed=parts[4] if len(parts) > 4 else "",
                        time_limit=parts[5] if len(parts) > 5 else "",
                        partition=parts[6] if len(parts) > 6 else "",
                        reason_or_nodes=parts[7] if len(parts) > 7 else "",
                    )
                )
        else:
            # Fallback standard whitespace-separated squeue
            parts = line.split()
            if len(parts) >= 5:
                records.append(
                    HpcJobRecord(
                        job_id=parts[0],
                        partition=parts[1] if len(parts) > 1 else "",
                        name=parts[2] if len(parts) > 2 else "",
                        user=parts[3] if len(parts) > 3 else "",
                        state=parts[4] if len(parts) > 4 else "UNKNOWN",
                        time_elapsed=parts[5] if len(parts) > 5 else "",
                        reason_or_nodes=parts[6] if len(parts) > 6 else "",
                        time_limit="",
                    )
                )

    return records


class HpcQueueWidget(QtWidgets.QWidget):
    """Interactive GUI widget for remote HPC queue monitoring."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.hosts: List[RemoteHostConfig] = parse_ssh_config_hosts()
        self.jobs: List[HpcJobRecord] = []
        self._init_ui()

        # Polling timer (default 30 seconds, off by default)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30000)
        self.timer.timeout.connect(self.refresh_queue)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Header controls
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)

        top_bar.addWidget(QtWidgets.QLabel("<b>Cluster Host:</b>"))
        self.combo_hosts = QtWidgets.QComboBox()
        self.combo_hosts.setMinimumWidth(160)
        for h in self.hosts:
            label = h.alias or h.hostname
            self.combo_hosts.addItem(f"🖥️ {label}", h)
        if not self.hosts:
            self.combo_hosts.addItem("Local (No SSH hosts)", None)
        top_bar.addWidget(self.combo_hosts)

        self.btn_refresh = QtWidgets.QPushButton("🔄 Refresh")
        self.btn_refresh.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_refresh.clicked.connect(self.refresh_queue)
        top_bar.addWidget(self.btn_refresh)

        self.chk_auto = QtWidgets.QCheckBox("Auto-Refresh (30s)")
        self.chk_auto.toggled.connect(lambda chk: self.timer.start() if chk else self.timer.stop())
        top_bar.addWidget(self.chk_auto)

        top_bar.addStretch()

        self.btn_cancel_job = QtWidgets.QPushButton("⏹ Cancel Job")
        self.btn_cancel_job.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 2px 8px;")
        self.btn_cancel_job.setToolTip("Cancel selected job on remote cluster (scancel <id>)")
        self.btn_cancel_job.clicked.connect(self._on_cancel_selected_job)
        top_bar.addWidget(self.btn_cancel_job)

        layout.addLayout(top_bar)

        # Table of Jobs
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Job ID", "Name", "User", "State", "Time", "Partition", "Nodes / Reason"
        ])
        header = self.table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setStyleSheet(
            "QTableWidget { background-color: #0f172a; color: #f8fafc; gridline-color: #334155; }"
            "QHeaderView::section { background-color: #1e293b; color: #94a3b8; padding: 4px; border: 1px solid #334155; }"
            "QTableWidget::item:selected { background-color: #2563eb; color: #ffffff; }"
        )
        layout.addWidget(self.table, stretch=1)

        # Status footer
        self.lbl_status = QtWidgets.QLabel("Queue idle. Click 'Refresh' to query remote cluster.")
        self.lbl_status.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.lbl_status)

    def refresh_queue(self):
        """Query remote cluster for running/pending jobs."""
        host_cfg = self.combo_hosts.currentData()
        if not host_cfg or not isinstance(host_cfg, RemoteHostConfig):
            self.lbl_status.setText("No remote SSH host configured in ~/.ssh/config.")
            return

        target = host_cfg.alias or host_cfg.hostname
        user_arg = f"-u {host_cfg.user}" if host_cfg.user else ""
        # Format: JOBID|NAME|USER|STATE|TIME|TIME_LIMIT|PARTITION|NODELIST(REASON)
        cmd = f"ssh -o BatchMode=yes -o ConnectTimeout=5 {target} 'squeue {user_arg} -o \"%i|%j|%u|%T|%M|%l|%P|%R\" 2>/dev/null || qstat 2>/dev/null'"

        self.lbl_status.setText(f"Querying {target}...")
        QtWidgets.QApplication.processEvents()

        try:
            res = subprocess.run(
                shlex.split(cmd),
                capture_output=True,
                text=True,
                timeout=8,
            )
            raw = res.stdout.strip()
            self.jobs = parse_squeue_output(raw)
            self._populate_table()
            self.lbl_status.setText(f"Connected to {target}: {len(self.jobs)} job(s) in queue.")
        except subprocess.TimeoutExpired:
            self.lbl_status.setText(f"Connection to {target} timed out (check VPN / SSH).")
        except Exception as e:
            self.lbl_status.setText(f"Failed to query queue: {e}")

    def _populate_table(self):
        self.table.setRowCount(len(self.jobs))
        for row, job in enumerate(self.jobs):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(job.job_id))
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(job.name))
            self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(job.user))

            item_state = QtWidgets.QTableWidgetItem(job.state)
            st_upper = job.state.upper()
            if "RUN" in st_upper:
                item_state.setForeground(QtGui.QBrush(QtGui.QColor("#34d399")))
            elif "PEND" in st_upper or "QUEUED" in st_upper:
                item_state.setForeground(QtGui.QBrush(QtGui.QColor("#f59e0b")))
            elif "FAIL" in st_upper or "CANCEL" in st_upper:
                item_state.setForeground(QtGui.QBrush(QtGui.QColor("#f87171")))
            self.table.setItem(row, 3, item_state)

            self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(job.time_elapsed))
            self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(job.partition))
            self.table.setItem(row, 6, QtWidgets.QTableWidgetItem(job.reason_or_nodes))

    def _on_cancel_selected_job(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self.jobs):
            QtWidgets.QMessageBox.information(self, "Cancel Job", "Please select a job from the table first.")
            return

        job = self.jobs[row]
        host_cfg = self.combo_hosts.currentData()
        if not host_cfg or not isinstance(host_cfg, RemoteHostConfig):
            return

        target = host_cfg.alias or host_cfg.hostname
        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirm Cancellation",
            f"Are you sure you want to cancel Job ID <b>{job.job_id}</b> ({job.name}) on <b>{target}</b>?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Yes:
            cancel_cmd = f"ssh -o BatchMode=yes {target} 'scancel {job.job_id} 2>/dev/null || qdel {job.job_id} 2>/dev/null'"
            try:
                subprocess.run(shlex.split(cancel_cmd), timeout=6)
                self.lbl_status.setText(f"Sent cancellation signal to Job {job.job_id}.")
                QtCore.QTimer.singleShot(1500, self.refresh_queue)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Cancel Failed", f"Failed to cancel job {job.job_id}: {e}")
