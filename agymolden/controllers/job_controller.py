"""
agymolden.controllers.job_controller
====================================
Dedicated controller encapsulating quantum chemistry job preparation,
execution, background monitoring, and artifact ingestion.
Decouples calculation runners and live telemetry from main window mixins.
"""

import os
import re
import logging
from typing import List, Optional, Any, Dict
import numpy as np
from PyQt6 import QtWidgets, QtCore

from ..calc_runner import (
    JobMonitorDialog,
    BrorsenRunner,
    OrcaRunner,
    GenericQMRunner,
    BrorsenLaunchDialog,
)
from ..zmatrix import cartesian_to_zmatrix
from ..topology import norm_bond
from ..forcefield import (
    detect_connectivity,
    detect_bond_orders,
)

log = logging.getLogger(__name__)


class JobController:
    """Manages QM job execution, preflight linting, live convergence telemetry, and artifact loading."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def quick_run_orca(self) -> None:
        """1-Click quick ORCA calculation modal for students and lab exercises."""
        if not self.mw.zatoms:
            QtWidgets.QMessageBox.information(
                self.mw, "Quick ORCA", "Please create or open a molecule before running an ORCA calculation."
            )
            return

        from ..dialog_quick_orca import QuickOrcaDialog
        dlg = QuickOrcaDialog(self.mw, self.mw.zatoms, self.mw.coords)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            saved_path = getattr(dlg, "saved_input_path", None)
            if saved_path and os.path.isfile(saved_path):
                self.launch_calculation_job(saved_path, runner_type="orca")

    def open_preflight_check_dialog(self) -> None:
        """Perform comprehensive pre-flight verification and display detailed inspection dialog."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self.mw, "Pre-Flight Status", "Molecule is empty. Add or open atoms to validate calculation readiness."
            )
            return

        from ..calc_runner.preflight import validate_qm_calculation
        from ..calc_runner.dialog_preflight import PreflightValidationDialog
        from ..qm_input_generator import QMCalculationParameters
        from ..qm_detect import detect_qm_executable

        symbols = [a.symbol for a in self.mw.zatoms]
        params = QMCalculationParameters(
            job_type="Opt",
            method="B3LYP",
            basis="def2-SVP",
            charge=0,
            multiplicity=1,
        )
        info = detect_qm_executable("orca")
        report = validate_qm_calculation(
            code_key="orca",
            symbols=symbols,
            coords=self.mw.coords,
            params=params,
            executable_info=info,
        )
        dlg = PreflightValidationDialog(self.mw, report, qm_name="orca")
        dlg.exec()

    def open_job_monitor(self) -> None:
        """Open or show the Job Monitor Dialog."""
        monitor = getattr(self.mw, "_job_monitor_dlg", None)
        if monitor is None:
            monitor = JobMonitorDialog(self.mw, runner=BrorsenRunner(self.mw))
            monitor.live_step_received.connect(self.mw._on_live_calc_step)
            monitor.calculation_completed.connect(self.mw._on_live_calc_completed)
            self.mw._job_monitor_dlg = monitor
        monitor.show()
        monitor.raise_()
        monitor.activateWindow()

    def open_workflow_dag(self) -> None:
        """Open the Visual Workflow DAG Pipeline Editor (Multi-Stage QM)."""
        from ..dialog_workflow_dag import VisualWorkflowDAGDialog

        symbols = [a.symbol for a in self.mw.zatoms]
        coords = np.array(self.mw.coords, dtype=np.float64, copy=True)
        work_dir = os.path.join(os.getcwd(), "workflow_dag_run")

        dlg = VisualWorkflowDAGDialog(
            parent=self.mw,
            code_key="orca",
            symbols=symbols,
            coords=coords,
            work_dir=work_dir,
        )

        def _on_adopt_geom(new_symbols, new_coords, label):
            self.mw._push_undo_state()
            detected_bonds = detect_connectivity(new_symbols, new_coords, strict=False)
            detected_bo = detect_bond_orders(new_symbols, new_coords, detected_bonds)
            zatoms = cartesian_to_zmatrix(new_symbols, new_coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.mw.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.mw.strict_topology = True
            self.mw.set_molecule_from_zatoms(zatoms, exact_coords=new_coords)
            if self.mw.viewer:
                self.mw.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
                self.mw.viewer.auto_center()
                self.mw.viewer.update()
            self.mw.statusBar().showMessage(f"Adopted geometry from {label} ({len(new_symbols)} atoms).", 5000)

        dlg.pipelineAdoptRequested.connect(_on_adopt_geom)
        dlg.exec()

    def configure_qm_engines(self) -> None:
        """Open the QM Engines Detection & Route Configuration Dialog."""
        from ..qm_detect import QMEnginesStatusDialog
        QMEnginesStatusDialog.show_dialog(self.mw)

    def open_remote_file_manager(self) -> None:
        """Open graphical remote file transfer dialog."""
        from ..calc_runner.remote_ssh import get_all_remote_hosts, RemoteHostConfig
        from ..calc_runner.dialog_file_transfer import RemoteFileManagerDialog

        hosts = get_all_remote_hosts()
        from ..session_state import get_settings
        settings = get_settings("RemoteProfiles")
        last_prof = settings.value("last_verified_profile", "")
        host_cfg = None

        if last_prof:
            host_cfg = RemoteHostConfig(
                hostname=settings.value(f"{last_prof}/hostname", ""),
                user=settings.value(f"{last_prof}/user", os.environ.get("USER", "")),
                identity_file=settings.value(f"{last_prof}/identity_file", ""),
                remote_workdir=settings.value(f"{last_prof}/remote_workdir", "~/agymolden_scratch"),
                remote_python=settings.value(f"{last_prof}/remote_python", "python3"),
                remote_wrapper=settings.value(f"{last_prof}/remote_wrapper", "newBRORSEN.v04.09.py"),
            )
        elif hosts:
            host_cfg = hosts[0]

        if not host_cfg or not host_cfg.hostname:
            QtWidgets.QMessageBox.information(
                self.mw, "Remote File Transfer",
                "Please configure and test your remote host under the 'Remote Server (SSH / VPN)' tab first."
            )
            dlg_launch = BrorsenLaunchDialog(self.mw, input_file="")
            dlg_launch.tab_targets.setCurrentIndex(1)
            dlg_launch.exec()
            return

        dlg = RemoteFileManagerDialog(self.mw, host_cfg)
        dlg.exec()

    def launch_calculation_job(self, input_file: str, runner_type: str = "brorsen") -> None:
        """Launch background QM calculation with live streaming output and 3D viewport updates."""
        py_bin = None
        wrapper_path = None
        extra_flags: List[str] = []
        live_3d = True
        remote_host = None

        if runner_type == "brorsen":
            launch_dlg = BrorsenLaunchDialog(self.mw, input_file)
            if launch_dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            py_bin, wrapper_path, extra_flags, live_3d, remote_host = launch_dlg.get_launch_configuration()

        def _make_runner(parent_obj):
            if runner_type == "orca":
                return OrcaRunner(parent_obj)
            elif runner_type in ("gaussian", "pyscf", "openlowdin", "xtb", "qchem", "psi4"):
                return GenericQMRunner(code_key=runner_type, parent=parent_obj)
            elif remote_host is not None:
                from ..calc_runner.remote_ssh import RemoteSshRunner
                r = RemoteSshRunner(parent_obj)
                r.configure_remote(host_cfg=remote_host, remote_script=wrapper_path or "", extra_flags=extra_flags)
                return r
            else:
                return BrorsenRunner(parent_obj)

        monitor = getattr(self.mw, "_job_monitor_dlg", None)
        if monitor is None:
            runner = _make_runner(self.mw)
            if hasattr(self.mw, "dock_manager"):
                dock = self.mw.dock_manager.get_or_create_job_monitor(runner=runner)
                monitor = dock.monitor
            else:
                monitor = JobMonitorDialog(self.mw, runner=runner)
                monitor.live_step_received.connect(self.mw._on_live_calc_step)
                monitor.calculation_completed.connect(self.mw._on_live_calc_completed)
            self.mw._job_monitor_dlg = monitor
        else:
            monitor.runner = _make_runner(monitor)
            monitor._connect_runner_signals()

        if runner_type == "brorsen" and isinstance(monitor.runner, BrorsenRunner):
            if py_bin and wrapper_path:
                monitor.runner.configure(python_bin=py_bin, wrapper_path=wrapper_path, extra_flags=extra_flags)
                idx_py = monitor.combo_conda.findData(py_bin)
                if idx_py >= 0:
                    monitor.combo_conda.setCurrentIndex(idx_py)
                idx_wr = monitor.combo_wrapper.findData(wrapper_path)
                if idx_wr >= 0:
                    monitor.combo_wrapper.setCurrentIndex(idx_wr)

        monitor.chk_live_3d.setChecked(live_3d)
        monitor.run_calculation(input_file, runner_type=runner_type)
