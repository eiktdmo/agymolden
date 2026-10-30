"""
Remote File Transfer & File Manager Dialog for AgyMolden.
Enables bidirectional browsing and file transfer between local computer and remote cluster/server
over SSH/SFTP/SCP without touching the console.
"""

import os
import stat
import subprocess
import logging
from typing import Optional, List, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets

from .remote_ssh import RemoteHostConfig, _wrap_ssh_command

log = logging.getLogger(__name__)


class RemoteFileManagerDialog(QtWidgets.QDialog):
    """Dual-pane file manager dialog (Local Machine <---> Remote Server)."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], host_cfg: RemoteHostConfig):
        super().__init__(parent)
        self.host_cfg = host_cfg
        self.setWindowTitle(f"Remote File Transfer: {host_cfg.user}@{host_cfg.hostname}")
        self.resize(920, 560)

        self.local_dir = os.path.abspath(".")
        self.remote_dir = host_cfg.remote_workdir or "~"

        self._init_ui()
        self._refresh_local()
        self._refresh_remote()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Header info
        header = QtWidgets.QHBoxLayout()
        lbl_target = QtWidgets.QLabel(
            f"<b>Remote Host:</b> <code>{self.host_cfg.user}@{self.host_cfg.hostname}</code> &nbsp;|&nbsp; "
            f"<b>Port:</b> {self.host_cfg.port}"
        )
        header.addWidget(lbl_target)
        header.addStretch()

        btn_refresh_all = QtWidgets.QPushButton("🔄 Refresh Both Panes")
        btn_refresh_all.clicked.connect(self._refresh_both)
        header.addWidget(btn_refresh_all)
        layout.addLayout(header)

        # Splitter with Two Panes: Local (Left) and Remote (Right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # --- LEFT PANE: LOCAL ---
        w_local = QtWidgets.QWidget()
        l_layout = QtWidgets.QVBoxLayout(w_local)
        l_layout.setContentsMargins(0, 0, 0, 0)
        l_layout.setSpacing(6)

        l_head = QtWidgets.QHBoxLayout()
        l_head.addWidget(QtWidgets.QLabel("<b>💻 Local Files:</b>"))
        self.txt_local_path = QtWidgets.QLineEdit(self.local_dir)
        self.txt_local_path.returnPressed.connect(self._on_local_path_entered)
        l_head.addWidget(self.txt_local_path, stretch=1)
        btn_local_up = QtWidgets.QPushButton("⬆ Up")
        btn_local_up.setFixedWidth(50)
        btn_local_up.clicked.connect(self._on_local_up)
        l_head.addWidget(btn_local_up)
        l_layout.addLayout(l_head)

        self.tbl_local = QtWidgets.QTableWidget()
        self.tbl_local.setColumnCount(3)
        self.tbl_local.setHorizontalHeaderLabels(["Name", "Size", "Type"])
        self.tbl_local.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tbl_local.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_local.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl_local.itemDoubleClicked.connect(self._on_local_double_clicked)
        l_layout.addWidget(self.tbl_local)

        # Local action bar below files table
        local_bar = QtWidgets.QHBoxLayout()
        self.btn_view_local_log = QtWidgets.QPushButton("📄 View Selected Output Log...")
        self.btn_view_local_log.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 8px;")
        self.btn_view_local_log.setToolTip("Open selected local calculation output log (.out, .log, .txt) in QM Output Viewer")
        self.btn_view_local_log.clicked.connect(self._on_view_selected_local_file)
        local_bar.addWidget(self.btn_view_local_log)
        local_bar.addStretch()
        l_layout.addLayout(local_bar)

        splitter.addWidget(w_local)

        # --- CENTER BUTTONS: Transfer Actions ---
        w_actions = QtWidgets.QWidget()
        a_layout = QtWidgets.QVBoxLayout(w_actions)
        a_layout.addStretch()

        self.btn_upload = QtWidgets.QPushButton("Upload ➡\n(Local to Remote)")
        self.btn_upload.setStyleSheet(
            "background-color: #2563eb; color: white; font-weight: bold; padding: 10px 8px; border-radius: 4px;"
        )
        self.btn_upload.clicked.connect(self._on_upload_clicked)
        a_layout.addWidget(self.btn_upload)

        a_layout.addSpacing(16)

        self.btn_download = QtWidgets.QPushButton("⬅ Download\n(Remote to Local)")
        self.btn_download.setStyleSheet(
            "background-color: #059669; color: white; font-weight: bold; padding: 10px 8px; border-radius: 4px;"
        )
        self.btn_download.clicked.connect(self._on_download_clicked)
        a_layout.addWidget(self.btn_download)

        a_layout.addStretch()
        splitter.addWidget(w_actions)

        # --- RIGHT PANE: REMOTE ---
        w_remote = QtWidgets.QWidget()
        r_layout = QtWidgets.QVBoxLayout(w_remote)
        r_layout.setContentsMargins(0, 0, 0, 0)
        r_layout.setSpacing(6)

        r_head = QtWidgets.QHBoxLayout()
        r_head.addWidget(QtWidgets.QLabel("<b>🌐 Remote Files:</b>"))
        self.txt_remote_path = QtWidgets.QLineEdit(self.remote_dir)
        self.txt_remote_path.returnPressed.connect(self._on_remote_path_entered)
        r_head.addWidget(self.txt_remote_path, stretch=1)
        btn_remote_up = QtWidgets.QPushButton("⬆ Up")
        btn_remote_up.setFixedWidth(50)
        btn_remote_up.clicked.connect(self._on_remote_up)
        r_head.addWidget(btn_remote_up)
        r_layout.addLayout(r_head)

        self.tbl_remote = QtWidgets.QTableWidget()
        self.tbl_remote.setColumnCount(3)
        self.tbl_remote.setHorizontalHeaderLabels(["Name", "Size", "Type"])
        self.tbl_remote.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tbl_remote.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl_remote.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tbl_remote.itemDoubleClicked.connect(self._on_remote_double_clicked)
        r_layout.addWidget(self.tbl_remote)

        splitter.addWidget(w_remote)
        splitter.setSizes([380, 140, 380])
        layout.addWidget(splitter, stretch=1)

        # Status / Log Bar
        self.lbl_status = QtWidgets.QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #64748b; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # Dialog Footer
        footer = QtWidgets.QHBoxLayout()
        footer.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)
        layout.addLayout(footer)

    def _refresh_both(self):
        self._refresh_local()
        self._refresh_remote()

    # -------------------------------------------------------------------------
    # Local File System Operations
    # -------------------------------------------------------------------------
    def _refresh_local(self):
        self.txt_local_path.setText(self.local_dir)
        self.tbl_local.setRowCount(0)

        if not os.path.isdir(self.local_dir):
            return

        try:
            entries = os.listdir(self.local_dir)
        except Exception as e:
            self.lbl_status.setText(f"Local Read Error: {e}")
            return

        # Sort: directories first, then files
        dirs = []
        files = []
        for name in entries:
            p = os.path.join(self.local_dir, name)
            if os.path.isdir(p):
                dirs.append(name)
            else:
                files.append(name)

        dirs.sort(key=str.lower)
        files.sort(key=str.lower)

        for d in dirs:
            self._add_local_row(f"📁 {d}", "-", "Directory", d, is_dir=True)
        for f in files:
            p = os.path.join(self.local_dir, f)
            sz = os.path.getsize(p) if os.path.isfile(p) else 0
            sz_str = f"{sz / 1024:.1f} KB" if sz >= 1024 else f"{sz} B"
            self._add_local_row(f, sz_str, "File", f, is_dir=False)

    def _add_local_row(self, display_name: str, size_str: str, type_str: str, raw_name: str, is_dir: bool):
        row = self.tbl_local.rowCount()
        self.tbl_local.insertRow(row)

        item_name = QtWidgets.QTableWidgetItem(display_name)
        item_name.setData(QtCore.Qt.ItemDataRole.UserRole, (raw_name, is_dir))
        item_size = QtWidgets.QTableWidgetItem(size_str)
        item_type = QtWidgets.QTableWidgetItem(type_str)

        self.tbl_local.setItem(row, 0, item_name)
        self.tbl_local.setItem(row, 1, item_size)
        self.tbl_local.setItem(row, 2, item_type)

    def _on_local_up(self):
        parent = os.path.dirname(self.local_dir)
        if parent and parent != self.local_dir:
            self.local_dir = parent
            self._refresh_local()

    def _on_local_path_entered(self):
        p = self.txt_local_path.text().strip()
        if os.path.isdir(p):
            self.local_dir = os.path.abspath(p)
            self._refresh_local()

    def _on_local_double_clicked(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        item_0 = self.tbl_local.item(row, 0)
        raw_name, is_dir = item_0.data(QtCore.Qt.ItemDataRole.UserRole)
        if is_dir:
            self.local_dir = os.path.abspath(os.path.join(self.local_dir, raw_name))
            self._refresh_local()
        else:
            ext = os.path.splitext(raw_name)[1].lower()
            if ext in (".out", ".log", ".txt", ".xyz", ".molden"):
                full_path = os.path.join(self.local_dir, raw_name)
                self._open_in_output_viewer(full_path)

    def _on_view_selected_local_file(self):
        sel_rows = sorted({idx.row() for idx in self.tbl_local.selectedIndexes()})
        if not sel_rows:
            QtWidgets.QMessageBox.information(self, "View Log", "Please select a transferred output file in the Local list.")
            return
        row = sel_rows[0]
        item_0 = self.tbl_local.item(row, 0)
        raw_name, is_dir = item_0.data(QtCore.Qt.ItemDataRole.UserRole)
        if is_dir:
            return
        full_path = os.path.join(self.local_dir, raw_name)
        self._open_in_output_viewer(full_path)

    def _open_in_output_viewer(self, file_path: str):
        if not os.path.isfile(file_path):
            return
        from agymolden.dialog_output_viewer import QMOutputViewerDialog
        dlg = QMOutputViewerDialog(self, file_path=file_path)
        dlg.exec()

    # -------------------------------------------------------------------------
    # Remote File System Operations (via SSH)
    # -------------------------------------------------------------------------
    def _execute_remote_command(self, cmd_str: str, timeout_sec: int = 10) -> Tuple[int, str, str]:
        target = f"{self.host_cfg.user}@{self.host_cfg.hostname}" if self.host_cfg.user else self.host_cfg.hostname
        ssh_cmd = [
            "ssh",
            "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=5",
            "-o", "ServerAliveCountMax=2",
        ]
        if not self.host_cfg.password:
            ssh_cmd.extend(["-o", "BatchMode=yes"])
        else:
            ssh_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
            ssh_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if self.host_cfg.port != 22:
            ssh_cmd.extend(["-p", str(self.host_cfg.port)])

        ssh_cmd.extend([target, cmd_str])
        full_cmd, env = _wrap_ssh_command(ssh_cmd, self.host_cfg.password)

        try:
            r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout_sec, env=env)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "Remote command timed out."
        except Exception as e:
            return 1, "", str(e)

    def _refresh_remote(self):
        self.tbl_remote.setRowCount(0)
        self.lbl_status.setText(f"Querying remote directory: {self.remote_dir}...")

        from ..workers import run_in_background

        cmd = f"cd {self.remote_dir} 2>/dev/null && pwd && ls -la --time-style=+%s"

        def _worker():
            return self._execute_remote_command(cmd)

        def _on_done(res):
            code, out, err = res
            if code != 0:
                self.lbl_status.setText(f"<span style='color: #ef4444;'>Remote Error: {err.strip()}</span>")
                return

            lines = out.strip().splitlines()
            if not lines:
                return

            resolved_pwd = lines[0].strip()
            self.remote_dir = resolved_pwd
            self.txt_remote_path.setText(resolved_pwd)

            dirs = []
            files = []

            for line in lines[1:]:
                parts = line.split(maxsplit=6)
                if len(parts) < 7:
                    continue
                perms, links, owner, group, size_str, timestamp, name = parts
                if name in (".", ".."):
                    continue

                is_dir = perms.startswith("d")
                try:
                    size_int = int(size_str)
                    s_str = f"{size_int / 1024:.1f} KB" if size_int >= 1024 else f"{size_int} B"
                except ValueError:
                    s_str = "-"

                if is_dir:
                    dirs.append((name, s_str))
                else:
                    files.append((name, s_str))

            dirs.sort(key=lambda x: x[0].lower())
            files.sort(key=lambda x: x[0].lower())

            for name, sz in dirs:
                self._add_remote_row(f"📁 {name}", sz, "Directory", name, is_dir=True)
            for name, sz in files:
                self._add_remote_row(name, sz, "File", name, is_dir=False)

            self.lbl_status.setText(f"Remote directory loaded: {self.remote_dir} ({len(dirs)} dirs, {len(files)} files)")

        def _on_err(exc):
            self.lbl_status.setText(f"<span style='color: #ef4444;'>Remote Error: {exc}</span>")

        run_in_background(self, "Refreshing Remote Directory", f"Querying {self.remote_dir}...", _worker, _on_done, _on_err)
    def _add_remote_row(self, display_name: str, size_str: str, type_str: str, raw_name: str, is_dir: bool):
        row = self.tbl_remote.rowCount()
        self.tbl_remote.insertRow(row)

        item_name = QtWidgets.QTableWidgetItem(display_name)
        item_name.setData(QtCore.Qt.ItemDataRole.UserRole, (raw_name, is_dir))
        item_size = QtWidgets.QTableWidgetItem(size_str)
        item_type = QtWidgets.QTableWidgetItem(type_str)

        self.tbl_remote.setItem(row, 0, item_name)
        self.tbl_remote.setItem(row, 1, item_size)
        self.tbl_remote.setItem(row, 2, item_type)

    def _on_remote_up(self):
        parent = os.path.dirname(self.remote_dir)
        if parent and parent != self.remote_dir:
            self.remote_dir = parent
            self._refresh_remote()

    def _on_remote_path_entered(self):
        p = self.txt_remote_path.text().strip()
        if p:
            self.remote_dir = p
            self._refresh_remote()

    def _on_remote_double_clicked(self, item: QtWidgets.QTableWidgetItem):
        row = item.row()
        item_0 = self.tbl_remote.item(row, 0)
        raw_name, is_dir = item_0.data(QtCore.Qt.ItemDataRole.UserRole)
        if is_dir:
            self.remote_dir = os.path.join(self.remote_dir, raw_name)
            self._refresh_remote()

    # -------------------------------------------------------------------------
    # Bidirectional Transfer Execution
    # -------------------------------------------------------------------------
    def _on_upload_clicked(self):
        """Upload selected local files to remote active directory without freezing GUI."""
        selected_rows = sorted(set(idx.row() for idx in self.tbl_local.selectedIndexes()))
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Upload", "Please select one or more local files to upload.")
            return

        files_to_upload = []
        for r in selected_rows:
            raw_name, is_dir = self.tbl_local.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            files_to_upload.append((os.path.join(self.local_dir, raw_name), is_dir))

        target = f"{self.host_cfg.user}@{self.host_cfg.hostname}" if self.host_cfg.user else self.host_cfg.hostname
        from ..workers import run_in_background

        def _upload_worker():
            for src_path, is_dir in files_to_upload:
                scp_cmd = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
                if is_dir:
                    scp_cmd.append("-r")
                if not self.host_cfg.password:
                    scp_cmd.extend(["-o", "BatchMode=yes"])
                else:
                    scp_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
                if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                    scp_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
                if self.host_cfg.port != 22:
                    scp_cmd.extend(["-P", str(self.host_cfg.port)])

                clean_rem = self.remote_dir.replace("~/", "") if self.remote_dir.startswith("~/") else self.remote_dir
                dest = f"{target}:{clean_rem}/"
                scp_cmd.extend([src_path, dest])
                full_cmd, env = _wrap_ssh_command(scp_cmd, self.host_cfg.password)

                r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60, env=env)
                if r.returncode != 0:
                    err = r.stderr.strip() or r.stdout.strip()
                    raise RuntimeError(f"Failed to upload {os.path.basename(src_path)}:\n{err}")
            return len(files_to_upload)

        def _on_done(count):
            self.lbl_status.setText(f"Upload complete! ({count} item(s) transferred)")
            self._refresh_remote()

        def _on_err(exc):
            QtWidgets.QMessageBox.critical(self, "Upload Failed", str(exc))
            self.lbl_status.setText("<span style=\"color: #ef4444;\">Upload error occurred.</span>")

        run_in_background(self, "Uploading Files", f"Uploading {len(files_to_upload)} file(s) to {target}...", _upload_worker, _on_done, _on_err)

    def _on_download_clicked(self):
        """Download selected remote files to local active directory without freezing GUI."""
        selected_rows = sorted(set(idx.row() for idx in self.tbl_remote.selectedIndexes()))
        if not selected_rows:
            QtWidgets.QMessageBox.information(self, "Download", "Please select one or more remote files to download.")
            return

        files_to_download = []
        for r in selected_rows:
            raw_name, is_dir = self.tbl_remote.item(r, 0).data(QtCore.Qt.ItemDataRole.UserRole)
            files_to_download.append((raw_name, is_dir))

        target = f"{self.host_cfg.user}@{self.host_cfg.hostname}" if self.host_cfg.user else self.host_cfg.hostname
        from ..workers import run_in_background

        def _download_worker():
            for raw_name, is_dir in files_to_download:
                rem_src = f"{target}:{os.path.join(self.remote_dir, raw_name)}"
                scp_cmd = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
                if is_dir:
                    scp_cmd.append("-r")
                if not self.host_cfg.password:
                    scp_cmd.extend(["-o", "BatchMode=yes"])
                else:
                    scp_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
                if self.host_cfg.identity_file and os.path.isfile(self.host_cfg.identity_file):
                    scp_cmd.extend(["-i", self.host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
                if self.host_cfg.port != 22:
                    scp_cmd.extend(["-P", str(self.host_cfg.port)])

                scp_cmd.extend([rem_src, self.local_dir])
                full_cmd, env = _wrap_ssh_command(scp_cmd, self.host_cfg.password)

                r = subprocess.run(full_cmd, capture_output=True, text=True, timeout=60, env=env)
                if r.returncode != 0:
                    err = r.stderr.strip() or r.stdout.strip()
                    raise RuntimeError(f"Failed to download {raw_name}:\n{err}")
            return len(files_to_download)

        def _on_done(count):
            self.lbl_status.setText(f"Download complete! ({count} item(s) downloaded)")
            self._refresh_local()

        def _on_err(exc):
            QtWidgets.QMessageBox.critical(self, "Download Failed", str(exc))
            self.lbl_status.setText("<span style=\"color: #ef4444;\">Download error occurred.</span>")

        run_in_background(self, "Downloading Files", f"Downloading {len(files_to_download)} file(s) from {target}...", _download_worker, _on_done, _on_err)
