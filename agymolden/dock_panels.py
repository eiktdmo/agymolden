"""
Unified Dockable Workspace System for AgyMolden.

Provides professional multi-panel workspace docking (QDockWidget) for:
- Calculation Job Monitor & Live Output Stream
- QM Output & Log Viewer with QC Doctor integration
- 1D Dihedral PES Scan & Conformer Plotter

Features:
- Docking to Bottom, Right, or Left workspace areas
- Tabified docks for multi-tasking
- Seamless detaching/floating for dual/triple-monitor scientific workstations
- Layout persistence via QSettings (save/restore dock states)
"""

from typing import Optional, Dict, Any, List
import os
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


class WorkspaceDockWidget(QtWidgets.QDockWidget):
    """Base dock widget with standard scientific workstation features."""

    def __init__(self, title: str, parent: Optional[QtWidgets.QWidget] = None, object_name: str = ""):
        super().__init__(title, parent)
        if object_name:
            self.setObjectName(object_name)

        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.AllDockWidgetAreas)

    def toggle_floating(self):
        """Toggle between docked in main window and floating separate window."""
        self.setFloating(not self.isFloating())


class JobMonitorDock(WorkspaceDockWidget):
    """Dockable panel hosting the real-time QM calculation job monitor."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, runner=None):
        super().__init__("⚡ Calculation Job Monitor", parent, object_name="dock_job_monitor")
        from .calc_runner.dialog_job_monitor import JobMonitorDialog

        self.monitor = JobMonitorDialog(self, runner=runner)
        self.monitor.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWidget(self.monitor)

    @property
    def runner(self):
        return self.monitor.runner

    @runner.setter
    def runner(self, val):
        self.monitor.runner = val

    def run_calculation(self, input_file: str, runner_type: str = "brorsen"):
        self.monitor.run_calculation(input_file, runner_type=runner_type)
        self.show()
        self.raise_()


class QMOutputDock(WorkspaceDockWidget):
    """Dockable panel hosting the QM output log viewer and QC Doctor diagnostics."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, file_path: str = ""):
        super().__init__("📄 QM Output & Log Viewer", parent, object_name="dock_output_viewer")
        from .dialog_output_viewer import QMOutputViewerDialog

        self.viewer_dlg = QMOutputViewerDialog(self, file_path=file_path)
        self.viewer_dlg.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setMinimumWidth(340)
        self.setMinimumHeight(200)
        self.setWidget(self.viewer_dlg)

    def load_file(self, path: str):
        if path and hasattr(self.viewer_dlg, "load_file"):
            self.viewer_dlg.load_file(path)
            self.setWindowTitle(f"📄 QM Output – {os.path.basename(path)}")
        self.viewer_dlg.show()
        self.show()
        self.raise_()


class DihedralScanDock(WorkspaceDockWidget):
    """Dockable panel hosting the 1D Dihedral PES Scan & Conformer Plotter."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget],
        symbols,
        coords,
        bonds,
        atom_indices,
        user_bond_orders=None,
        frozen_indices=None,
    ):
        super().__init__("📈 Dihedral PES Scan", parent, object_name="dock_dihedral_scan")
        from .dialog_conformer_scan import DihedralScanDialog

        self.scan_dlg = DihedralScanDialog(
            self,
            symbols=symbols,
            coords=coords,
            bonds=bonds,
            atom_indices=atom_indices,
            user_bond_orders=user_bond_orders,
            frozen_indices=frozen_indices,
        )
        self.scan_dlg.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWidget(self.scan_dlg)

    @property
    def conformer_selected(self):
        return self.scan_dlg.conformer_selected


class PythonConsoleDock(WorkspaceDockWidget):
    """Dockable panel hosting the interactive Python chemistry console."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, main_window: Optional[QtWidgets.QMainWindow] = None):
        super().__init__("🐍 Python Console", parent, object_name="dock_python_console")
        from .dock_console import PythonConsoleWidget

        self.console_widget = PythonConsoleWidget(self, main_window=main_window)
        self.setWidget(self.console_widget)


class ReactionPathDock(WorkspaceDockWidget):
    """Dockable panel hosting the Reaction Path & Transition State (TS) Interpolator."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_symbols: Optional[List[str]] = None,
        initial_coords: Optional[np.ndarray] = None,
    ):
        super().__init__("🔄 Reaction Path & TS Interpolator", parent, object_name="dock_reaction_path")
        from .dialog_reaction_path import ReactionPathDialog

        self.path_dlg = ReactionPathDialog(
            self,
            initial_symbols=initial_symbols,
            initial_coords=initial_coords,
        )
        self.path_dlg.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWidget(self.path_dlg)

    @property
    def path_adopted(self):
        return self.path_dlg.path_adopted

    @property
    def trajectory_preview(self):
        return self.path_dlg.trajectory_preview


class HpcQueueDock(WorkspaceDockWidget):
    """Dockable panel hosting the remote HPC Cluster & Slurm queue monitor."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__("🖥️ HPC Cluster Queue", parent, object_name="dock_hpc_queue")
        from .calc_runner.hpc_queue import HpcQueueWidget

        self.queue_widget = HpcQueueWidget(self)
        self.setWidget(self.queue_widget)


class QCDoctorDock(WorkspaceDockWidget):
    """Dockable panel hosting the QC Doctor calculation diagnostics and crash auto-repair."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, log_text: str = ""):
        super().__init__("🩺 QC Doctor Diagnostics", parent, object_name="dock_qc_doctor")
        from .dialog_qc_doctor import QCDoctorDialog

        self.doctor_dialog = QCDoctorDialog(self, log_text=log_text)
        self.doctor_dialog.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWidget(self.doctor_dialog)

    def diagnose_log(self, log_text: str):
        from .qc_doctor import diagnose_qm_log
        from .dialog_qc_doctor import QCDoctorDialog

        new_dlg = QCDoctorDialog(self, log_text=log_text)
        new_dlg.setWindowFlags(QtCore.Qt.WindowType.Widget)
        self.setWidget(new_dlg)
        self.doctor_dialog = new_dlg


class WorkspaceDockManager:
    """Manages docks, menu actions, and layout persistence for MainWindow."""

    def __init__(self, main_window: QtWidgets.QMainWindow):
        self.mw = main_window
        self.docks: Dict[str, WorkspaceDockWidget] = {}
        self._init_main_window_docking()

    def _init_main_window_docking(self):
        """Configure main window dock layout options."""
        self.mw.setDockOptions(
            QtWidgets.QMainWindow.DockOption.AnimatedDocks
            | QtWidgets.QMainWindow.DockOption.AllowTabbedDocks
            | QtWidgets.QMainWindow.DockOption.AllowNestedDocks
        )
        self.mw.setCorner(QtCore.Qt.Corner.BottomLeftCorner, QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
        self.mw.setCorner(QtCore.Qt.Corner.BottomRightCorner, QtCore.Qt.DockWidgetArea.RightDockWidgetArea)

    def _tabify_in_area(self, area: QtCore.Qt.DockWidgetArea, new_dock: WorkspaceDockWidget) -> None:
        """Tabify *new_dock* onto an existing non-floating dock in *area* so docks never crush the viewport."""
        for existing in self.docks.values():
            if existing is new_dock or existing.isFloating():
                continue
            if self.mw.dockWidgetArea(existing) == area:
                self.mw.tabifyDockWidget(existing, new_dock)
                return

    def get_or_create_job_monitor(self, runner=None) -> JobMonitorDock:
        if "job_monitor" not in self.docks:
            dock = JobMonitorDock(self.mw, runner=runner)
            # Forward signals to MainWindow handlers if present
            if hasattr(self.mw, "_on_live_calc_step"):
                dock.monitor.live_step_received.connect(self.mw._on_live_calc_step)
            if hasattr(self.mw, "_on_live_calc_completed"):
                dock.monitor.calculation_completed.connect(self.mw._on_live_calc_completed)
            self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._tabify_in_area(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self.docks["job_monitor"] = dock
            bot_h = max(200, min(280, int(self.mw.height() * 0.32)))
            self.mw.resizeDocks([dock], [bot_h], QtCore.Qt.Orientation.Vertical)
        return self.docks["job_monitor"] # type: ignore

    def get_or_create_output_viewer(self, file_path: str = "") -> QMOutputDock:
        created = False
        if "output_viewer" not in self.docks:
            dock = QMOutputDock(self.mw, file_path=file_path)
            self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self._tabify_in_area(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
            self.docks["output_viewer"] = dock
            created = True
        dock = self.docks["output_viewer"]  # type: ignore
        if file_path:
            dock.load_file(file_path)  # type: ignore
        else:
            dock.viewer_dlg.show()  # type: ignore
            dock.show()
            dock.raise_()
        if created:
            right_w = min(max(360, int(self.mw.width() * 0.40)), max(360, self.mw.width() - 420))
            self.mw.resizeDocks([dock], [right_w], QtCore.Qt.Orientation.Horizontal)
        return dock  # type: ignore

    def get_or_create_console(self) -> PythonConsoleDock:
        if "python_console" not in self.docks:
            dock = PythonConsoleDock(self.mw, main_window=self.mw)
            self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._tabify_in_area(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self.docks["python_console"] = dock
        return self.docks["python_console"] # type: ignore

    def get_or_create_hpc_queue(self) -> HpcQueueDock:
        if "hpc_queue" not in self.docks:
            dock = HpcQueueDock(self.mw)
            self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self._tabify_in_area(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)
            self.docks["hpc_queue"] = dock
        return self.docks["hpc_queue"] # type: ignore

    def open_dihedral_scan(
        self,
        symbols,
        coords,
        bonds,
        atom_indices,
        user_bond_orders=None,
        frozen_indices=None,
        on_conformer_selected=None,
    ) -> DihedralScanDock:
        # If an existing scan dock is present, remove it
        if "dihedral_scan" in self.docks:
            old_dock = self.docks.pop("dihedral_scan")
            self.mw.removeDockWidget(old_dock)
            old_dock.deleteLater()

        dock = DihedralScanDock(
            self.mw,
            symbols=symbols,
            coords=coords,
            bonds=bonds,
            atom_indices=atom_indices,
            user_bond_orders=user_bond_orders,
            frozen_indices=frozen_indices,
        )
        if on_conformer_selected:
            dock.conformer_selected.connect(on_conformer_selected)

        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_in_area(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.show()
        dock.raise_()
        self.docks["dihedral_scan"] = dock
        return dock

    def open_reaction_path(
        self,
        initial_symbols=None,
        initial_coords=None,
        on_adopted=None,
        on_preview_traj=None,
    ) -> ReactionPathDock:
        # If an existing reaction path dock is present, remove it
        if "reaction_path" in self.docks:
            old_dock = self.docks.pop("reaction_path")
            self.mw.removeDockWidget(old_dock)
            old_dock.deleteLater()

        dock = ReactionPathDock(
            self.mw,
            initial_symbols=initial_symbols,
            initial_coords=initial_coords,
        )
        if on_adopted:
            dock.path_adopted.connect(on_adopted)
        if on_preview_traj:
            dock.trajectory_preview.connect(on_preview_traj)

        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_in_area(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.show()
        dock.raise_()
        self.docks["reaction_path"] = dock
        return dock

    def open_qc_doctor(self, log_text: str = "") -> QCDoctorDock:
        """Open or activate the QC Doctor calculation diagnostics dock."""
        if "qc_doctor" in self.docks:
            dock = self.docks["qc_doctor"]
            if log_text:
                dock.diagnose_log(log_text)
            dock.show()
            dock.raise_()
            return dock

        dock = QCDoctorDock(self.mw, log_text=log_text)
        self.mw.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        self._tabify_in_area(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)
        dock.show()
        dock.raise_()
        self.docks["qc_doctor"] = dock
        return dock

    def float_all_docks(self):
        """Detach all open docks into separate floating windows (Multi-Monitor Mode)."""
        for dock in self.docks.values():
            dock.setFloating(True)

    def dock_all_panels(self):
        """Snap all floating docks back into the main window docking areas."""
        for dock in self.docks.values():
            dock.setFloating(False)

    def save_layout(self):
        """Save dock states and positions into user preferences."""
        from agymolden.session_state import get_settings
        settings = get_settings("AgyMolden")
        settings.setValue("workspace/dock_state", self.mw.saveState())

    def restore_layout(self):
        """Restore dock states and positions from user preferences."""
        from agymolden.session_state import get_settings
        settings = get_settings("AgyMolden")
        state = settings.value("workspace/dock_state", None)
        if state is not None and isinstance(state, (QtCore.QByteArray, bytes)):
            self.mw.restoreState(state)

    def reset_layout(self):
        """Reset workspace layout to default configuration."""
        for dock in self.docks.values():
            dock.setFloating(False)
            dock.setVisible(False)
