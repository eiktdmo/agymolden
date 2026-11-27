"""
Modal launch dialog for choosing Conda environment, wrapper script, and options
BEFORE the calculation starts.
"""

import os
import re
import sys
import subprocess
from typing import List, Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets

from .brorsen_runner import detect_conda_environments, find_brorsen_wrappers
from .remote_ssh import get_all_remote_hosts, test_ssh_connectivity, RemoteHostConfig, _wrap_ssh_command


class BrorsenLaunchDialog(QtWidgets.QDialog):
    """Dialog presented BEFORE running newBRORSEN to select Conda env & wrapper."""

    def __init__(self, parent: Optional[QtWidgets.QWidget], input_file: str):
        super().__init__(parent)
        self.setWindowTitle("Launch newBRORSEN Calculation")
        self.resize(680, 360)
        self.input_file = input_file

        self._init_ui()
        self._load_saved_settings()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Header info
        lbl_info = QtWidgets.QLabel(
            f"<b>Input File:</b> <code>{os.path.basename(self.input_file)}</code><br>"
            f"<span style='color: #6b7280;'>Directory: {os.path.dirname(os.path.abspath(self.input_file))}</span>"
        )
        layout.addWidget(lbl_info)

        # Execution Target Tabs (Local Machine vs Remote Host over VPN)
        self.tab_targets = QtWidgets.QTabWidget()

        # Tab 1: Local Machine
        tab_local = QtWidgets.QWidget()
        t_local_layout = QtWidgets.QVBoxLayout(tab_local)
        t_local_layout.setContentsMargins(6, 6, 6, 6)

        group = QtWidgets.QGroupBox("Local Environment & Script")
        form = QtWidgets.QFormLayout(group)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setSpacing(10)

        # 1. Python Environment (venv / custom)
        conda_layout = QtWidgets.QHBoxLayout()
        self.combo_conda = QtWidgets.QComboBox()
        self.combo_conda.setToolTip("Select Python interpreter or Virtualenv")
        self._populate_conda_envs()
        conda_layout.addWidget(self.combo_conda, stretch=1)

        btn_browse_py = QtWidgets.QPushButton("Browse Python...")
        btn_browse_py.clicked.connect(self._on_browse_python)
        conda_layout.addWidget(btn_browse_py)
        form.addRow("<b>Python Interpreter / Env:</b>", conda_layout)

        # 2. Wrapper Script
        wrapper_layout = QtWidgets.QHBoxLayout()
        self.combo_wrapper = QtWidgets.QComboBox()
        self.combo_wrapper.setToolTip("Select newBRORSEN wrapper script (e.g. v04.08, v04.09)")
        self._populate_wrappers()
        wrapper_layout.addWidget(self.combo_wrapper, stretch=1)

        btn_browse_wrap = QtWidgets.QPushButton("Browse Script...")
        btn_browse_wrap.clicked.connect(self._on_browse_wrapper)
        wrapper_layout.addWidget(btn_browse_wrap)
        form.addRow("<b>newBRORSEN Wrapper:</b>", wrapper_layout)

        # 3. Environment test status
        self.lbl_env_status = QtWidgets.QLabel("")
        self.lbl_env_status.setWordWrap(True)
        form.addRow("Compatibility Check:", self.lbl_env_status)

        btn_test = QtWidgets.QPushButton("🧪 Test Selected Environment & Wrapper")
        btn_test.clicked.connect(self._test_compatibility)
        form.addRow("", btn_test)

        t_local_layout.addWidget(group)
        self.tab_targets.addTab(tab_local, "💻 Local Machine")

        # Tab 2: Remote Machine over SSH / VPN
        tab_remote = QtWidgets.QWidget()
        t_rem_layout = QtWidgets.QVBoxLayout(tab_remote)
        t_rem_layout.setContentsMargins(6, 6, 6, 6)

        rem_group = QtWidgets.QGroupBox("Remote Connection (Configure & Test Before Saving)")
        r_form = QtWidgets.QFormLayout(rem_group)
        r_form.setSpacing(8)

        # VPN Command Field (Blank by default so user can write openconnect, wireguard, etc.)
        vpn_row = QtWidgets.QHBoxLayout()
        self.txt_vpn_cmd = QtWidgets.QLineEdit("")
        self.txt_vpn_cmd.setPlaceholderText("e.g. sudo openconnect --protocol=gp vpnpa.unal.edu.co")
        vpn_row.addWidget(self.txt_vpn_cmd, stretch=1)

        self.btn_launch_vpn = QtWidgets.QPushButton("🚀 Launch VPN Terminal...")
        self.btn_launch_vpn.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 4px 8px;")
        self.btn_launch_vpn.setToolTip("Opens a terminal window running your VPN command to enter sudo and VPN passwords interactively")
        self.btn_launch_vpn.clicked.connect(self._on_launch_vpn_terminal)
        vpn_row.addWidget(self.btn_launch_vpn)
        r_form.addRow("VPN Command:", vpn_row)

        # Host selection or custom entry
        host_row = QtWidgets.QHBoxLayout()
        self.combo_remote_host = QtWidgets.QComboBox()
        self.combo_remote_host.setEditable(True)
        self.combo_remote_host.setInsertPolicy(QtWidgets.QComboBox.InsertPolicy.InsertAtTop)
        self.combo_remote_host.setPlaceholderText("Select known host or type user@hostname...")
        self.remote_hosts = get_all_remote_hosts()
        for h in self.remote_hosts:
            self.combo_remote_host.addItem(f"{h.alias} ({h.hostname})", h)
        self.combo_remote_host.currentIndexChanged.connect(self._on_remote_host_changed)
        host_row.addWidget(self.combo_remote_host, stretch=1)
        r_form.addRow("<b>Remote Host (SSH):</b>", host_row)

        # SSH Key / Identity file
        key_row = QtWidgets.QHBoxLayout()
        self.txt_ssh_key = QtWidgets.QLineEdit("")
        self.txt_ssh_key.setPlaceholderText("Optional path to private key (e.g. ~/.ssh/id_rsa_totalcmd.pem)")
        key_row.addWidget(self.txt_ssh_key, stretch=1)
        btn_browse_key = QtWidgets.QPushButton("Browse Key...")
        btn_browse_key.clicked.connect(self._on_browse_ssh_key)
        key_row.addWidget(btn_browse_key)
        r_form.addRow("SSH Private Key:", key_row)

        # SSH Password (for password-based authentication)
        pwd_row = QtWidgets.QHBoxLayout()
        self.txt_ssh_password = QtWidgets.QLineEdit("")
        self.txt_ssh_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
        self.txt_ssh_password.setPlaceholderText("Optional password (if not using SSH key)")
        pwd_row.addWidget(self.txt_ssh_password, stretch=1)

        self.btn_toggle_pwd = QtWidgets.QPushButton("👁")
        self.btn_toggle_pwd.setFixedWidth(32)
        self.btn_toggle_pwd.setToolTip("Show/Hide password")
        self.btn_toggle_pwd.clicked.connect(self._toggle_password_visibility)
        pwd_row.addWidget(self.btn_toggle_pwd)
        r_form.addRow("SSH Password:", pwd_row)

        self.txt_remote_workdir = QtWidgets.QLineEdit("")
        self.txt_remote_workdir.setPlaceholderText("e.g. ~/pruebareyes or /scratch/username")
        r_form.addRow("Remote Workdir:", self.txt_remote_workdir)

        self.txt_remote_py = QtWidgets.QLineEdit("python3")
        self.txt_remote_py.setPlaceholderText("e.g. python3 or ~/venvs/brorsen/bin/python")
        r_form.addRow("Remote Python:", self.txt_remote_py)

        self.txt_remote_wrap = QtWidgets.QLineEdit("newBRORSEN.v04.09.py")
        self.txt_remote_wrap.setPlaceholderText("e.g. newBRORSEN.v04.09.py or path on remote server")
        r_form.addRow("Remote Wrapper:", self.txt_remote_wrap)

        # Buttons to test remote environment & upload local wrapper
        env_actions_row = QtWidgets.QHBoxLayout()
        self.btn_check_remote_env = QtWidgets.QPushButton("🔍 Check Remote Python & pyscf_neo_cc")
        self.btn_check_remote_env.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 4px 8px;")
        self.btn_check_remote_env.setToolTip("Runs remote python to verify if pyscf and pyscf_neo_cc (Kurt Brorsen branch) are installed")
        self.btn_check_remote_env.clicked.connect(self._on_check_remote_env)
        env_actions_row.addWidget(self.btn_check_remote_env)

        self.btn_check_load = QtWidgets.QPushButton("📊 Server Load & Users")
        self.btn_check_load.setStyleSheet("background-color: #0891b2; color: #fff; font-weight: bold; padding: 4px 8px;")
        self.btn_check_load.setToolTip("Inspect remote CPU usage, load average, logged-in users, and active QM processes")
        self.btn_check_load.clicked.connect(self._on_check_remote_load)
        env_actions_row.addWidget(self.btn_check_load)

        self.btn_upload_wrapper = QtWidgets.QPushButton("📤 Upload Wrapper")
        self.btn_upload_wrapper.setStyleSheet("background-color: #0d9488; color: #fff; font-weight: bold; padding: 4px 8px;")
        self.btn_upload_wrapper.setToolTip("Transfers your local newBRORSEN.v04.09.py directly into the remote workdir via SCP")
        self.btn_upload_wrapper.clicked.connect(self._on_upload_wrapper_to_remote)
        env_actions_row.addWidget(self.btn_upload_wrapper)

        self.btn_file_manager = QtWidgets.QPushButton("📁 Remote File Transfer (Browse / SCP)")
        self.btn_file_manager.setStyleSheet("background-color: #7c3aed; color: #fff; font-weight: bold; padding: 4px 8px;")
        self.btn_file_manager.setToolTip("Open dual-pane graphical file manager to transfer files between local computer and remote server")
        self.btn_file_manager.clicked.connect(self._on_open_file_manager)
        env_actions_row.addWidget(self.btn_file_manager)
        r_form.addRow("Remote Actions:", env_actions_row)

        self.lbl_remote_status = QtWidgets.QLabel("<span style='color: #94a3b8;'>Enter connection parameters and click 'Test Connection' below.</span>")
        self.lbl_remote_status.setWordWrap(True)
        r_form.addRow("Connection Status:", self.lbl_remote_status)

        test_row = QtWidgets.QHBoxLayout()
        btn_test_ssh = QtWidgets.QPushButton("🔌 Test Connection (Verify Reachability)")
        btn_test_ssh.setStyleSheet("background-color: #2563eb; color: #fff; font-weight: bold; padding: 5px;")
        btn_test_ssh.clicked.connect(self._test_remote_ssh)
        test_row.addWidget(btn_test_ssh)

        self.btn_save_remote = QtWidgets.QPushButton("💾 Save As Named Profile...")
        self.btn_save_remote.setStyleSheet("background-color: #059669; color: #fff; font-weight: bold; padding: 5px;")
        self.btn_save_remote.clicked.connect(self._save_verified_remote_profile)
        test_row.addWidget(self.btn_save_remote)

        self.btn_delete_remote = QtWidgets.QPushButton("🗑️ Delete Profile")
        self.btn_delete_remote.clicked.connect(self._delete_remote_profile)
        test_row.addWidget(self.btn_delete_remote)

        r_form.addRow("", test_row)

        t_rem_layout.addWidget(rem_group)

        # SLURM Cluster Options
        self.grp_slurm = QtWidgets.QGroupBox("HPC Batch Queue (SLURM sbatch)")
        self.grp_slurm.setCheckable(True)
        self.grp_slurm.setChecked(False)
        self.grp_slurm.setToolTip("Enable if the remote cluster uses SLURM workload manager rather than direct execution")
        slurm_layout = QtWidgets.QGridLayout(self.grp_slurm)
        slurm_layout.setContentsMargins(8, 8, 8, 8)
        slurm_layout.setSpacing(6)

        slurm_layout.addWidget(QtWidgets.QLabel("Partition / Queue:"), 0, 0)
        self.txt_slurm_partition = QtWidgets.QLineEdit("")
        self.txt_slurm_partition.setPlaceholderText("e.g. general, standard, batch (optional)")
        slurm_layout.addWidget(self.txt_slurm_partition, 0, 1)

        slurm_layout.addWidget(QtWidgets.QLabel("CPUs per Task:"), 0, 2)
        self.spin_slurm_cpus = QtWidgets.QSpinBox()
        self.spin_slurm_cpus.setRange(1, 256)
        self.spin_slurm_cpus.setValue(4)
        slurm_layout.addWidget(self.spin_slurm_cpus, 0, 3)

        slurm_layout.addWidget(QtWidgets.QLabel("Memory:"), 1, 0)
        self.txt_slurm_mem = QtWidgets.QLineEdit("8GB")
        self.txt_slurm_mem.setPlaceholderText("e.g. 8GB, 16GB, 32GB")
        slurm_layout.addWidget(self.txt_slurm_mem, 1, 1)

        slurm_layout.addWidget(QtWidgets.QLabel("Walltime:"), 1, 2)
        self.txt_slurm_time = QtWidgets.QLineEdit("24:00:00")
        self.txt_slurm_time.setPlaceholderText("e.g. 24:00:00 or 48:00:00")
        slurm_layout.addWidget(self.txt_slurm_time, 1, 3)

        t_rem_layout.addWidget(self.grp_slurm)

        self.tab_targets.addTab(tab_remote, "🌐 Remote Server (SSH / VPN)")

        layout.addWidget(self.tab_targets)

        # Extra flags / Options
        opt_group = QtWidgets.QGroupBox("Execution Options")
        opt_layout = QtWidgets.QHBoxLayout(opt_group)
        self.chk_live_3d = QtWidgets.QCheckBox("Update 3D Viewport in Real-Time")
        self.chk_live_3d.setChecked(True)
        opt_layout.addWidget(self.chk_live_3d)

        self.chk_resume = QtWidgets.QCheckBox("--resume (Skip finished stages)")
        opt_layout.addWidget(self.chk_resume)

        self.chk_dry_run = QtWidgets.QCheckBox("--dry-run (Validate only)")
        opt_layout.addWidget(self.chk_dry_run)

        # nice & nohup options
        self.chk_nice = QtWidgets.QCheckBox("Use nice priority")
        self.chk_nice.setChecked(True)
        self.chk_nice.setToolTip("Run process with lowered scheduling priority (nice -n 10) so it does not freeze the machine for others")
        opt_layout.addWidget(self.chk_nice)

        self.chk_nohup = QtWidgets.QCheckBox("Use nohup resilience")
        self.chk_nohup.setChecked(True)
        self.chk_nohup.setToolTip("Ensures calculation continues running on server even if local VPN or network disconnects")
        opt_layout.addWidget(self.chk_nohup)

        layout.addWidget(opt_group)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_run = QtWidgets.QPushButton("▶ Launch Calculation")
        self.btn_run.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_run.clicked.connect(self._on_launch)
        btn_box.addWidget(self.btn_run)

        layout.addLayout(btn_box)

        # Connect changes to re-test
        self.combo_conda.currentIndexChanged.connect(self._on_selection_changed)
        self.combo_wrapper.currentIndexChanged.connect(self._on_selection_changed)

    def _populate_conda_envs(self):
        self.combo_conda.clear()
        envs = detect_conda_environments()
        # Prefer envs with MCCC or brorsen first
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

    def _on_selection_changed(self):
        self._test_compatibility(fast_only=True)

    def _test_compatibility(self, fast_only: bool = False):
        py_bin = self.combo_conda.currentData() or sys.executable
        wrapper = self.combo_wrapper.currentData() or ""

        if not os.path.isfile(wrapper):
            self.lbl_env_status.setText("<span style='color: #ef4444;'>Wrapper script not selected or does not exist.</span>")
            return

        if not os.path.isfile(py_bin):
            self.lbl_env_status.setText("<span style='color: #ef4444;'>Python executable not found.</span>")
            return

        import subprocess
        try:
            # Fast check: verifies interpreter launches and wrapper exists without loading all quantum chemistry modules
            if fast_only:
                chk_code = f"import sys, os; sys.exit(0 if os.path.isfile({repr(wrapper)}) else 1)"
                r = subprocess.run([py_bin, "-c", chk_code], capture_output=True, text=True, timeout=2)
                if r.returncode == 0:
                    self.lbl_env_status.setText("<span style='color: #10b981;'><b>✓ Ready:</b> Python interpreter and wrapper script detected. Click 'Test' for deep verification.</span>")
                else:
                    self.lbl_env_status.setText("<span style='color: #ef4444;'><b>⚠️ Warning:</b> Failed to execute Python interpreter.</span>")
                return

            # Deep check triggered by user button click
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            self.lbl_env_status.setText("<span style='color: #38bdf8;'>Validating environment and wrapper imports...</span>")
            QtWidgets.QApplication.processEvents()
            try:
                r = subprocess.run([py_bin, wrapper, "--help"], capture_output=True, text=True, timeout=8)
                if r.returncode == 0:
                    self.lbl_env_status.setText("<span style='color: #10b981;'><b>✓ Ready:</b> Python environment and wrapper script validated successfully.</span>")
                else:
                    err_line = (r.stderr.strip() or r.stdout.strip()).splitlines()
                    first_err = err_line[0] if err_line else f"Exited with code {r.returncode}"
                    self.lbl_env_status.setText(f"<span style='color: #ef4444;'><b>⚠️ Warning:</b> {first_err}</span>")
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        except Exception as e:
            self.lbl_env_status.setText(f"<span style='color: #ef4444;'>Error testing: {e}</span>")

    def _on_launch_vpn_terminal(self):
        """Spawns an interactive terminal window executing the VPN command directly."""
        cmd_str = self.txt_vpn_cmd.text().strip()
        if not cmd_str:
            QtWidgets.QMessageBox.information(
                self,
                "VPN Command Required",
                "Please enter your VPN command in the field first (e.g. sudo openconnect --protocol=gp vpnpa.unal.edu.co)."
            )
            return

        import shutil
        import subprocess

        # Write a clean executable launcher script in ~/.config/agymolden_vpn.sh so terminal never drops arguments
        script_path = os.path.expanduser("~/.config/agymolden_vpn.sh")
        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(
                    "#!/bin/bash\n"
                    "clear\n"
                    "echo '=========================================================='\n"
                    f"echo ' Executing VPN Command: {cmd_str}'\n"
                    "echo '=========================================================='\n"
                    "echo ''\n"
                    f"{cmd_str}\n"
                    "status=$?\n"
                    "echo ''\n"
                    "if [ $status -eq 0 ]; then\n"
                    "    echo 'VPN closed cleanly.'\n"
                    "else\n"
                    "    echo 'VPN process exited with error.'\n"
                    "fi\n"
                    "echo 'Press Enter to close this window.'\n"
                    "read line\n"
                )
            os.chmod(script_path, 0o755)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not create VPN script:\n{e}")
            return

        # Prefer native desktop terminal (e.g. konsole on KDE Plasma, gnome-terminal on GNOME)
        desktop = (os.environ.get("XDG_CURRENT_DESKTOP") or "").lower()
        if "kde" in desktop and shutil.which("konsole"):
            spawn_args = ["konsole", "-e", script_path]
        elif shutil.which("xfce4-terminal"):
            spawn_args = ["xfce4-terminal", "-e", script_path]
        elif shutil.which("gnome-terminal"):
            spawn_args = ["gnome-terminal", "--", script_path]
        elif shutil.which("konsole"):
            spawn_args = ["konsole", "-e", script_path]
        elif shutil.which("xterm"):
            spawn_args = ["xterm", "-e", script_path]
        else:
            spawn_args = [shutil.which("x-terminal-emulator") or "xterm", "-e", script_path]

        try:
            subprocess.Popen(spawn_args)
            self.lbl_remote_status.setText(
                "<span style='color: #0284c7;'><b>🚀 VPN Terminal launched:</b> Enter your password in the opened window, "
                "then click 'Test Connection' below once connected.</span>"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Failed to Launch Terminal", f"Could not open terminal:\n{e}")

    def _on_browse_ssh_key(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Private SSH Key", os.path.expanduser("~/.ssh"), "Key Files (*.pem id_* *);;All Files (*)"
        )
        if path:
            self.txt_ssh_key.setText(path)

    def _toggle_password_visibility(self):
        if self.txt_ssh_password.echoMode() == QtWidgets.QLineEdit.EchoMode.Password:
            self.txt_ssh_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Normal)
            self.btn_toggle_pwd.setText("🔒")
        else:
            self.txt_ssh_password.setEchoMode(QtWidgets.QLineEdit.EchoMode.Password)
            self.btn_toggle_pwd.setText("👁")

    def _on_remote_host_changed(self):
        h = self.combo_remote_host.currentData()
        if isinstance(h, RemoteHostConfig):
            if h.identity_file:
                self.txt_ssh_key.setText(h.identity_file)
            if h.remote_workdir:
                self.txt_remote_workdir.setText(h.remote_workdir)
            if h.remote_python:
                self.txt_remote_py.setText(h.remote_python)
            if h.remote_wrapper:
                self.txt_remote_wrap.setText(h.remote_wrapper)
            if h.vpn_command:
                self.txt_vpn_cmd.setText(h.vpn_command)
            if getattr(h, "password", ""):
                self.txt_ssh_password.setText(h.password)
            if hasattr(self, "grp_slurm"):
                self.grp_slurm.setChecked(getattr(h, "use_slurm", False))
                self.txt_slurm_partition.setText(getattr(h, "slurm_partition", ""))
                self.spin_slurm_cpus.setValue(getattr(h, "slurm_cpus", 4))
                self.txt_slurm_mem.setText(getattr(h, "slurm_mem", "8GB"))
                self.txt_slurm_time.setText(getattr(h, "slurm_time", "24:00:00"))
        self.btn_save_remote.setEnabled(False)

    def _get_active_remote_config(self) -> RemoteHostConfig:
        raw_host = self.combo_remote_host.currentText().strip()
        data_cfg = self.combo_remote_host.currentData()
        pwd = self.txt_ssh_password.text()

        # If user picked a recognized host and didn't completely change the text
        use_nice = self.chk_nice.isChecked()
        use_nohup = self.chk_nohup.isChecked()
        use_slurm = self.grp_slurm.isChecked() if hasattr(self, "grp_slurm") else False
        slurm_part = self.txt_slurm_partition.text().strip() if hasattr(self, "txt_slurm_partition") else ""
        slurm_cpus = self.spin_slurm_cpus.value() if hasattr(self, "spin_slurm_cpus") else 4
        slurm_mem = self.txt_slurm_mem.text().strip() or "8GB" if hasattr(self, "txt_slurm_mem") else "8GB"
        slurm_time = self.txt_slurm_time.text().strip() or "24:00:00" if hasattr(self, "txt_slurm_time") else "24:00:00"

        if isinstance(data_cfg, RemoteHostConfig) and (data_cfg.hostname in raw_host or data_cfg.alias in raw_host):
            cfg = RemoteHostConfig(
                alias=data_cfg.alias,
                hostname=data_cfg.hostname,
                user=data_cfg.user,
                identity_file=self.txt_ssh_key.text().strip() or data_cfg.identity_file,
                password=pwd,
                port=data_cfg.port,
                remote_workdir=self.txt_remote_workdir.text().strip() or "~/agymolden_scratch",
                remote_python=self.txt_remote_py.text().strip() or "python3",
                remote_wrapper=self.txt_remote_wrap.text().strip() or "newBRORSEN.v04.09.py",
                vpn_command=self.txt_vpn_cmd.text().strip(),
                use_nice=use_nice,
                nice_value=10,
                use_nohup=use_nohup,
                use_slurm=use_slurm,
                slurm_partition=slurm_part,
                slurm_cpus=slurm_cpus,
                slurm_mem=slurm_mem,
                slurm_time=slurm_time,
            )
            return cfg

        # Custom text entered by user (e.g. user@hostname or IP)
        user = ""
        hostname = raw_host
        if "@" in raw_host:
            parts = raw_host.split("@", 1)
            user = parts[0].strip()
            hostname = parts[1].strip()

        return RemoteHostConfig(
            alias=raw_host,
            hostname=hostname,
            user=user or os.environ.get("USER", "andres"),
            identity_file=self.txt_ssh_key.text().strip(),
            password=pwd,
            remote_workdir=self.txt_remote_workdir.text().strip() or "~/agymolden_scratch",
            remote_python=self.txt_remote_py.text().strip() or "python3",
            remote_wrapper=self.txt_remote_wrap.text().strip() or "newBRORSEN.v04.09.py",
            vpn_command=self.txt_vpn_cmd.text().strip(),
            use_nice=use_nice,
            nice_value=10,
            use_nohup=use_nohup,
            use_slurm=use_slurm,
            slurm_partition=slurm_part,
            slurm_cpus=slurm_cpus,
            slurm_mem=slurm_mem,
            slurm_time=slurm_time,
        )

    def _on_upload_wrapper_to_remote(self):
        """Upload the local newBRORSEN wrapper script to the remote workdir."""
        host_cfg = self._get_active_remote_config()
        if not host_cfg.hostname:
            QtWidgets.QMessageBox.warning(self, "Missing Host", "Please enter remote Host or IP first.")
            return

        # Locate local wrapper
        local_cand = os.path.join(os.getcwd(), host_cfg.remote_wrapper)
        if not os.path.isfile(local_cand):
            local_cand = os.path.join(os.getcwd(), "newBRORSEN.v04.09.py")

        local_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Local newBRORSEN Wrapper to Upload",
            local_cand if os.path.isfile(local_cand) else os.getcwd(),
            "Python Scripts (*.py);;All Files (*)",
        )
        if not local_path or not os.path.isfile(local_path):
            return

        target = f"{host_cfg.user}@{host_cfg.hostname}" if host_cfg.user else host_cfg.hostname
        rem_dir = host_cfg.remote_workdir
        filename = os.path.basename(local_path)

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.lbl_remote_status.setText(f"Uploading {filename} to {target}:{rem_dir} ...")
        QtWidgets.QApplication.processEvents()
        # 1. Ensure remote directory exists (resolve tilde safely)
        clean_rem_dir = rem_dir.replace("~/", "") if rem_dir.startswith("~/") else rem_dir
        mkdir_cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        if not host_cfg.password:
            mkdir_cmd.extend(["-o", "BatchMode=yes"])
        else:
            mkdir_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if host_cfg.identity_file and os.path.isfile(host_cfg.identity_file):
            mkdir_cmd.extend(["-i", host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if host_cfg.port != 22:
            mkdir_cmd.extend(["-p", str(host_cfg.port)])
        mkdir_cmd.extend([target, f"mkdir -p {clean_rem_dir}"])
        f_mkdir, env_mkdir = _wrap_ssh_command(mkdir_cmd, host_cfg.password)

        # 2. Transfer file via scp (use 60s timeout for ~450KB scripts across VPN)
        scp_cmd = ["scp", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        if host_cfg.password:
            scp_cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        else:
            scp_cmd.extend(["-o", "BatchMode=yes"])
        if host_cfg.identity_file and os.path.isfile(host_cfg.identity_file):
            scp_cmd.extend(["-i", host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if host_cfg.port != 22:
            scp_cmd.extend(["-P", str(host_cfg.port)])
        scp_cmd.extend([local_path, f"{target}:{clean_rem_dir}/{filename}"])

        f_scp, env_scp = _wrap_ssh_command(scp_cmd, host_cfg.password)

        from ..workers import run_in_background

        def _worker():
            subprocess.run(f_mkdir, capture_output=True, timeout=15, env=env_mkdir)
            return subprocess.run(f_scp, capture_output=True, text=True, timeout=60, env=env_scp)

        def _on_done(r):
            QtWidgets.QApplication.restoreOverrideCursor()
            if r.returncode == 0:
                self.txt_remote_wrap.setText(filename)
                self.lbl_remote_status.setText(
                    f"<span style='color: #10b981;'><b>✓ Wrapper Uploaded:</b> {filename} successfully transferred to {target}:{rem_dir}</span>"
                )
                QtWidgets.QMessageBox.information(
                    self, "Upload Complete", f"Successfully transferred '{filename}' to:\n{target}:{rem_dir}/{filename}"
                )
            else:
                err = r.stderr.strip() or r.stdout.strip()
                self.lbl_remote_status.setText(f"<span style='color: #ef4444;'><b>Upload Failed:</b> {err}</span>")
                QtWidgets.QMessageBox.critical(self, "Upload Failed", f"Failed to transfer wrapper:\n{err}")

        def _on_err(e):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.lbl_remote_status.setText(f"<span style='color: #ef4444;'><b>Upload Error:</b> {e}</span>")
            QtWidgets.QMessageBox.critical(self, "Upload Error", str(e))

        run_in_background(self, "Uploading Script", f"Uploading {filename} to {target}...", _worker, _on_done, _on_err)

    def _on_check_remote_load(self):
        """Query remote machine load average, CPU count, active users, and running QM processes."""
        host_cfg = self._get_active_remote_config()
        if not host_cfg.hostname:
            QtWidgets.QMessageBox.warning(self, "Missing Host", "Please enter remote Host or IP first.")
            return

        target = f"{host_cfg.user}@{host_cfg.hostname}" if host_cfg.user else host_cfg.hostname

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.lbl_remote_status.setText(f"Checking load and active processes on {target}...")
        QtWidgets.QApplication.processEvents()

        # Remote command to inspect load, users, and top chemistry tasks
        stat_cmd = (
            "echo '===UPTIME==='; uptime; "
            "echo '===CPUS==='; nproc 2>/dev/null || grep -c ^processor /proc/cpuinfo; "
            "echo '===USERS==='; who 2>/dev/null; "
            "echo '===CHEM_PROCS==='; "
            "ps -eo user,pid,%cpu,%mem,comm --sort=-%cpu 2>/dev/null | grep -E 'python|pyscf|orca|g16|xtb|gaussian|mpirun' | head -n 12"
        )

        cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        if not host_cfg.password:
            cmd.extend(["-o", "BatchMode=yes"])
        else:
            cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if host_cfg.identity_file and os.path.isfile(host_cfg.identity_file):
            cmd.extend(["-i", host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if host_cfg.port != 22:
            cmd.extend(["-p", str(host_cfg.port)])
        cmd.extend([target, stat_cmd])

        from ..workers import run_in_background
        from .remote_ssh import _wrap_ssh_command

        full_cmd, env = _wrap_ssh_command(cmd, host_cfg.password)

        def _worker():
            return subprocess.run(full_cmd, capture_output=True, text=True, timeout=12, env=env)

        def _on_done(r):
            QtWidgets.QApplication.restoreOverrideCursor()
            if r.returncode != 0:
                err = r.stderr.strip() or r.stdout.strip()
                QtWidgets.QMessageBox.warning(self, "Load Check Failed", f"Could not check server load on {target}:\n{err}")
                return

            out = r.stdout.strip()

            # Parse sections
            uptime_str = "Unknown"
            cpus_str = "Unknown"
            users_list = []
            chem_procs = []

            sections = out.split("===")
            for sec in sections:
                if sec.startswith("UPTIME==="):
                    uptime_str = sec.replace("UPTIME===", "").strip().splitlines()[0] if sec.replace("UPTIME===", "").strip() else ""
                elif sec.startswith("CPUS==="):
                    cpus_str = sec.replace("CPUS===", "").strip().splitlines()[0] if sec.replace("CPUS===", "").strip() else ""
                elif sec.startswith("USERS==="):
                    raw_u = sec.replace("USERS===", "").strip().splitlines()
                    users_list = [u.strip() for u in raw_u if u.strip()]
                elif sec.startswith("CHEM_PROCS==="):
                    raw_p = sec.replace("CHEM_PROCS===", "").strip().splitlines()
                    chem_procs = [p.strip() for p in raw_p if p.strip()]

            # Determine load status color
            load_summary = uptime_str.split("load average:")[-1].strip() if "load average:" in uptime_str else uptime_str
            active_users_str = f"{len(users_list)} active terminal/SSH login(s)" if users_list else "No other users logged in"

            msg = [
                f"<h3>Remote Machine Status: <code>{target}</code></h3>",
                f"<b>Total CPU Cores:</b> {cpus_str}",
                f"<b>Load Average (1, 5, 15 min):</b> <code>{load_summary}</code>",
                f"<b>Logged-in Users:</b> {active_users_str}",
            ]
            if users_list:
                msg.append("<pre style='font-size: 11px;'>" + "\n".join(users_list[:5]) + "</pre>")

            if chem_procs:
                msg.append(f"<b>Active Quantum Chemistry / Python Jobs:</b> ({len(chem_procs)} process(es) running)")
                msg.append("<pre style='font-size: 11px; color: #b91c1c;'>" + "\n".join(chem_procs) + "</pre>")
                msg.append("<i>Tip: Consider using the 'Use nice priority' option so your job politely shares CPU resources.</i>")
            else:
                msg.append("<b>Active Quantum Chemistry Jobs:</b> <span style='color: #059669;'>None (Server is fully idle for calculations)</span>")

            dlg = QtWidgets.QMessageBox(self)
            dlg.setWindowTitle(f"Server Load Report: {target}")
            dlg.setText("<br>".join(msg))
            dlg.setIcon(QtWidgets.QMessageBox.Icon.Information)
            dlg.exec()
            self.lbl_remote_status.setText(f"<span style='color: #10b981;'>✓ Load verified: {load_summary}</span>")

        def _on_err(e):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.lbl_remote_status.setText(f"<span style='color: #ef4444;'><b>Load Check Error:</b> {e}</span>")
            QtWidgets.QMessageBox.critical(self, "Load Check Error", str(e))

        run_in_background(self, "Checking Remote Load", f"Inspecting system load on {target}...", _worker, _on_done, _on_err)

    def _on_open_file_manager(self):
        """Open graphical file transfer dialog to transfer files between local and remote server."""
        host_cfg = self._get_active_remote_config()
        if not host_cfg.hostname:
            QtWidgets.QMessageBox.warning(self, "Missing Host", "Please enter remote Host or IP first.")
            return

        from .dialog_file_transfer import RemoteFileManagerDialog
        dlg = RemoteFileManagerDialog(self, host_cfg)
        dlg.exec()

    def _on_check_remote_env(self):
        """Test remote python environment, PySCF, and Kurt Brorsen's pyscf_neo_cc module."""
        host_cfg = self._get_active_remote_config()
        if not host_cfg.hostname:
            QtWidgets.QMessageBox.warning(self, "Missing Host", "Please enter remote Host or IP first.")
            return

        target = f"{host_cfg.user}@{host_cfg.hostname}" if host_cfg.user else host_cfg.hostname
        py_bin = host_cfg.remote_python or "python3"
        rem_dir = host_cfg.remote_workdir
        wrapper = host_cfg.remote_wrapper

        # Python test code to run on remote host
        py_check = (
            "import sys; "
            "print('PYTHON_VER=' + sys.version.split()[0]); "
            "pyscf_ok = 'NO'; neo_ok = 'NO'; neo_ver = 'N/A'; "
            "try:\n"
            "    import pyscf\n"
            "    pyscf_ok = getattr(pyscf, '__version__', 'OK')\n"
            "except Exception as e: pyscf_ok = f'ERR:{e}'\n"
            "try:\n"
            "    import pyscf_neo_cc\n"
            "    neo_ok = 'YES'\n"
            "    neo_ver = getattr(pyscf_neo_cc, '__version__', 'installed')\n"
            "except Exception as e: neo_ok = f'ERR:{e}'\n"
            "print(f'PYSCF={pyscf_ok}'); "
            "print(f'NEO_CC={neo_ok}|VER={neo_ver}')"
        )

        remote_bash_cmd = f"cd {rem_dir} 2>/dev/null; {py_bin} -c \"{py_check}\""

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.lbl_remote_status.setText("Checking remote Python environment & Brorsen packages...")
        QtWidgets.QApplication.processEvents()
        cmd = ["ssh", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2"]
        if not host_cfg.password:
            cmd.extend(["-o", "BatchMode=yes"])
        else:
            cmd.extend(["-o", "StrictHostKeyChecking=accept-new"])
        if host_cfg.identity_file and os.path.isfile(host_cfg.identity_file):
            cmd.extend(["-i", host_cfg.identity_file, "-o", "IdentitiesOnly=yes"])
        if host_cfg.port != 22:
            cmd.extend(["-p", str(host_cfg.port)])
        cmd.extend([target, remote_bash_cmd])

        from ..workers import run_in_background
        from .remote_ssh import _wrap_ssh_command

        full_cmd, env = _wrap_ssh_command(cmd, host_cfg.password)

        def _worker():
            return subprocess.run(full_cmd, capture_output=True, text=True, timeout=15, env=env)

        def _on_done(r):
            QtWidgets.QApplication.restoreOverrideCursor()
            out = r.stdout.strip()
            lines = out.splitlines()

            # Parse results
            py_ver = "Unknown"
            pyscf_status = "Not installed"
            neo_status = "Not installed"
            for line in lines:
                if line.startswith("PYTHON_VER="):
                    py_ver = line.split("=", 1)[1]
                elif line.startswith("PYSCF="):
                    pyscf_status = line.split("=", 1)[1]
                elif line.startswith("NEO_CC="):
                    neo_status = line.split("=", 1)[1]

            msg = [
                f"<b>Remote Host:</b> {target}",
                f"<b>Python Path:</b> <code>{py_bin}</code> (Python {py_ver})",
                f"<b>PySCF:</b> {pyscf_status}",
                f"<b>Kurt Brorsen Branch (pyscf_neo_cc):</b> {neo_status}",
            ]

            if "YES" in neo_status or "installed" in neo_status:
                self.lbl_remote_status.setText(
                    "<span style='color: #10b981;'><b>✓ Brorsen Environment Ready:</b> "
                    f"Python {py_ver}, PySCF ({pyscf_status}), pyscf_neo_cc ({neo_status}) verified!</span>"
                )
                QtWidgets.QMessageBox.information(self, "Remote Environment Verified", "<br>".join(msg))
            else:
                self.lbl_remote_status.setText(
                    f"<span style='color: #f59e0b;'><b>⚠️ Warning:</b> pyscf_neo_cc missing in remote <code>{py_bin}</code></span>"
                )
                QtWidgets.QMessageBox.warning(
                    self,
                    "Remote Package Missing",
                    "<br>".join(msg) + "<br><br>"
                    "<b>Note:</b> newBRORSEN calculations require the <code>pyscf_neo_cc</code> branch installed in your remote Python virtual environment.<br>"
                    "Update 'Remote Python' with the full path to that environment's python binary (e.g. <code>~/venvs/brorsen/bin/python</code>)."
                )

        def _on_err(e):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.lbl_remote_status.setText(f"<span style='color: #ef4444;'><b>Environment Check Error:</b> {e}</span>")
            QtWidgets.QMessageBox.critical(self, "Check Failed", str(e))

        run_in_background(self, "Checking Brorsen Packages", f"Testing Python & pyscf_neo_cc on {target}...", _worker, _on_done, _on_err)

    def _test_remote_ssh(self):
        host_cfg = self._get_active_remote_config()
        if not host_cfg.hostname:
            self.lbl_remote_status.setText("<span style='color: #ef4444;'>Please enter a remote Host or IP.</span>")
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        self.lbl_remote_status.setText("Testing SSH / Network connection...")
        QtWidgets.QApplication.processEvents()

        from ..workers import run_in_background

        def _worker():
            return test_ssh_connectivity(host_cfg, timeout_sec=5)

        def _on_done(result):
            QtWidgets.QApplication.restoreOverrideCursor()
            ok, msg = result
            if ok:
                self.lbl_remote_status.setText(
                    f"<span style='color: #10b981;'><b>✓ Connected:</b> {msg}</span><br>"
                    "<span style='color: #38bdf8;'>Connection verified! You can now click 'Save Profile'.</span>"
                )
                self.btn_save_remote.setEnabled(True)
            else:
                vpn_cmd = self.txt_vpn_cmd.text().strip()
                vpn_hint = f"<br>If your VPN is disconnected, run your VPN command first:<br><code>{vpn_cmd}</code>" if vpn_cmd else "<br>If remote machine requires a VPN, ensure your VPN client is connected."
                self.lbl_remote_status.setText(
                    f"<span style='color: #ef4444;'><b>⚠️ Connection Failed:</b> {msg}</span>{vpn_hint}"
                )
                self.btn_save_remote.setEnabled(False)

        def _on_err(exc):
            QtWidgets.QApplication.restoreOverrideCursor()
            self.lbl_remote_status.setText(f"<span style='color: #ef4444;'><b>Connection Error:</b> {exc}</span>")
            self.btn_save_remote.setEnabled(False)

        run_in_background(self, "Testing SSH Connection", f"Pinging {host_cfg.hostname} over SSH...", _worker, _on_done, _on_err)

    def _save_verified_remote_profile(self):
        """Save remote configuration as a named profile to QSettings so multiple machines can be saved."""
        cfg = self._get_active_remote_config()
        if not cfg.hostname:
            QtWidgets.QMessageBox.warning(self, "Missing Host", "Please enter a remote Host or IP first.")
            return

        default_name = cfg.alias or f"{cfg.user}@{cfg.hostname}"
        profile_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Remote Profile", "Enter a friendly name for this remote machine/profile:",
            QtWidgets.QLineEdit.EchoMode.Normal, default_name
        )
        if not ok or not profile_name.strip():
            return
        profile_name = profile_name.strip()

        # Generate a safe sanitized key
        safe_key = re.sub(r"[^a-zA-Z0-9_\-]", "_", profile_name)
        prof_key = f"profile_{safe_key}"

        from agymolden.session_state import get_settings
        settings = get_settings("RemoteProfiles")
        settings.setValue(f"{prof_key}/alias", profile_name)
        settings.setValue(f"{prof_key}/hostname", cfg.hostname)
        settings.setValue(f"{prof_key}/user", cfg.user)
        settings.setValue(f"{prof_key}/identity_file", cfg.identity_file)
        settings.setValue(f"{prof_key}/remote_workdir", cfg.remote_workdir)
        settings.setValue(f"{prof_key}/remote_python", cfg.remote_python)
        settings.setValue(f"{prof_key}/remote_wrapper", cfg.remote_wrapper)
        settings.setValue(f"{prof_key}/vpn_command", cfg.vpn_command)
        settings.setValue(f"{prof_key}/use_nice", cfg.use_nice)
        settings.setValue(f"{prof_key}/nice_value", cfg.nice_value)
        settings.setValue(f"{prof_key}/use_nohup", cfg.use_nohup)
        settings.setValue(f"{prof_key}/use_slurm", cfg.use_slurm)
        settings.setValue(f"{prof_key}/slurm_partition", cfg.slurm_partition)
        settings.setValue(f"{prof_key}/slurm_cpus", cfg.slurm_cpus)
        settings.setValue(f"{prof_key}/slurm_mem", cfg.slurm_mem)
        settings.setValue(f"{prof_key}/slurm_time", cfg.slurm_time)
        settings.setValue("last_verified_profile", prof_key)

        # Refresh combo_remote_host with all saved profiles
        self._populate_remote_profiles()
        target_label = f"{profile_name} ({cfg.hostname})"
        idx = self.combo_remote_host.findText(target_label)
        if idx < 0:
            idx = self.combo_remote_host.findText(profile_name)
        if idx >= 0:
            self.combo_remote_host.setCurrentIndex(idx)
        else:
            self.combo_remote_host.setEditText(target_label)

        QtWidgets.QMessageBox.information(
            self, "Profile Saved", f"Remote profile '{profile_name}' saved successfully!"
        )

    def _delete_remote_profile(self):
        """Delete currently selected profile from QSettings."""
        cur_text = self.combo_remote_host.currentText().strip()
        data_cfg = self.combo_remote_host.currentData()
        if not cur_text:
            return

        reply = QtWidgets.QMessageBox.question(
            self, "Delete Profile", f"Are you sure you want to delete profile '{cur_text}'?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        from agymolden.session_state import get_settings
        settings = get_settings("RemoteProfiles")
        # Find matching group
        for group in settings.childGroups():
            if not group.startswith("profile_"):
                continue
            alias = settings.value(f"{group}/alias", "")
            h_name = settings.value(f"{group}/hostname", "")
            if cur_text in (alias, h_name, f"{alias} ({h_name})") or (isinstance(data_cfg, RemoteHostConfig) and data_cfg.hostname == h_name):
                settings.remove(group)
                break

        self._populate_remote_profiles()
        QtWidgets.QMessageBox.information(self, "Profile Deleted", f"Profile '{cur_text}' was removed.")

    def _populate_remote_profiles(self):
        """Populate remote host combo box with all saved profiles and known hosts."""
        self.combo_remote_host.blockSignals(True)
        self.combo_remote_host.clear()
        self.remote_hosts = get_all_remote_hosts()
        for h in self.remote_hosts:
            label = h.alias if h.alias else f"{h.user}@{h.hostname}"
            if h.hostname and h.hostname not in label:
                label = f"{label} ({h.hostname})"
            self.combo_remote_host.addItem(label, h)
        self.combo_remote_host.blockSignals(False)

    def _load_saved_settings(self):
        from agymolden.session_state import get_settings
        settings = get_settings("JobRunner")
        saved_py = settings.value("python_bin", "")
        saved_wrap = settings.value("brorsen_wrapper", "")

        if saved_py:
            idx = self.combo_conda.findData(saved_py)
            if idx >= 0:
                self.combo_conda.setCurrentIndex(idx)
        else:
            # Prioritize andres-venv with latest Brorsen distribution first, then MCCC conda
            for i in range(self.combo_conda.count()):
                text = self.combo_conda.itemText(i)
                if "andres-venv" in text:
                    self.combo_conda.setCurrentIndex(i)
                    break
            else:
                for i in range(self.combo_conda.count()):
                    text = self.combo_conda.itemText(i)
                    if "0926MCCC" in text or "MCCC" in text:
                        self.combo_conda.setCurrentIndex(i)
                        break

        if saved_wrap:
            idx = self.combo_wrapper.findData(saved_wrap)
            if idx >= 0:
                self.combo_wrapper.setCurrentIndex(idx)
        else:
            # Default to v04.09 (matching andres-venv latest Brorsen distribution) or v04.08
            for i in range(self.combo_wrapper.count()):
                text = self.combo_wrapper.itemText(i)
                if "v04.09" in text:
                    self.combo_wrapper.setCurrentIndex(i)
                    break
            else:
                for i in range(self.combo_wrapper.count()):
                    text = self.combo_wrapper.itemText(i)
                    if "v04.08" in text:
                        self.combo_wrapper.setCurrentIndex(i)
                        break

        # Load last verified remote profile if exists
        from agymolden.session_state import get_settings
        rem_settings = get_settings("RemoteProfiles")
        last_prof = rem_settings.value("last_verified_profile", "")
        if last_prof:
            h_name = rem_settings.value(f"{last_prof}/hostname", "")
            if h_name:
                self.combo_remote_host.setEditText(h_name)
                self.txt_ssh_key.setText(rem_settings.value(f"{last_prof}/identity_file", ""))
                self.txt_remote_workdir.setText(rem_settings.value(f"{last_prof}/remote_workdir", "~/agymolden_scratch"))
                self.txt_remote_py.setText(rem_settings.value(f"{last_prof}/remote_python", "python3"))
                self.txt_remote_wrap.setText(rem_settings.value(f"{last_prof}/remote_wrapper", "newBRORSEN.v04.09.py"))
                self.txt_vpn_cmd.setText(rem_settings.value(f"{last_prof}/vpn_command", ""))
                if hasattr(self, "grp_slurm"):
                    self.grp_slurm.setChecked(str(rem_settings.value(f"{last_prof}/use_slurm", "false")).lower() == "true")
                    self.txt_slurm_partition.setText(str(rem_settings.value(f"{last_prof}/slurm_partition", "")))
                    self.spin_slurm_cpus.setValue(int(rem_settings.value(f"{last_prof}/slurm_cpus", 4)))
                    self.txt_slurm_mem.setText(str(rem_settings.value(f"{last_prof}/slurm_mem", "8GB")))
                    self.txt_slurm_time.setText(str(rem_settings.value(f"{last_prof}/slurm_time", "24:00:00")))

        self._test_compatibility(fast_only=True)

    def _save_settings(self):
        from agymolden.session_state import get_settings
        settings = get_settings("JobRunner")
        py_bin = self.combo_conda.currentData()
        wrapper = self.combo_wrapper.currentData()
        if py_bin:
            settings.setValue("python_bin", py_bin)
        if wrapper:
            settings.setValue("brorsen_wrapper", wrapper)

    def _on_launch(self):
        is_remote = (self.tab_targets.currentIndex() == 1)
        if is_remote:
            host_cfg = self._get_active_remote_config()
            if not host_cfg.hostname:
                QtWidgets.QMessageBox.warning(self, "Invalid Remote Host", "Please enter a valid remote Host or IP.")
                return
            self._save_settings()
            self.accept()
            return

        py_bin = self.combo_conda.currentData()
        wrapper = self.combo_wrapper.currentData()

        if not py_bin or not os.path.isfile(py_bin):
            QtWidgets.QMessageBox.warning(self, "Invalid Python", "Please select a valid Python interpreter / Conda environment.")
            return

        if not wrapper or not os.path.isfile(wrapper):
            QtWidgets.QMessageBox.warning(self, "Invalid Wrapper", "Please select a valid newBRORSEN wrapper script.")
            return

        self._save_settings()
        self.accept()

    def get_launch_configuration(self) -> Tuple[str, str, List[str], bool, Optional[RemoteHostConfig]]:
        """Returns (python_bin, wrapper_path, extra_flags, live_3d_update, remote_host_cfg_or_None)."""
        is_remote = (self.tab_targets.currentIndex() == 1)
        flags = []
        if self.chk_resume.isChecked():
            flags.append("--resume")
        if self.chk_dry_run.isChecked():
            flags.append("--dry-run")
        live_3d = self.chk_live_3d.isChecked()

        if is_remote:
            host_cfg = self._get_active_remote_config()
            return "ssh", host_cfg.remote_wrapper, flags, live_3d, host_cfg

        py_bin = self.combo_conda.currentData() or sys.executable
        wrapper = self.combo_wrapper.currentData() or ""
        return py_bin, wrapper, flags, live_3d, None
