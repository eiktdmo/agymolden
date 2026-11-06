"""
Centralized Event Bus and Signal Dispatcher for AgyMolden.

Provides a decoupled Publish-Subscribe architecture (AppEventBus) connecting:
- 3D OpenGL Viewport
- Dockable Workspace Panels (Job Monitor, QM Log, Dihedral PES Scan)
- Background Calculation Runners (ORCA, newBRORSEN, PySCF)
- User Interface Mixins and Plugins
- External API / Python automation scripts

Thread Safety:
Qt signals automatically marshal across thread boundaries using Qt::QueuedConnection
when emitting from background calculation workers or async tasks.
"""

from typing import Optional, List, Set, Any
import numpy as np
from PyQt6 import QtCore


class AppEventBus(QtCore.QObject):
    """
    Central event dispatcher for AgyMolden.
    Subscribers can connect directly to typed pyqtSignals without holding
    references to MainWindow or child widgets.
    """

    # Molecule & Structure Lifecycle
    molecule_loaded = QtCore.pyqtSignal(list, object, str)  # (symbols: List[str], coords: np.ndarray, title: str)
    geometry_changed = QtCore.pyqtSignal(object)            # coords: np.ndarray
    selection_changed = QtCore.pyqtSignal(object)           # selected_indices: Set[int]
    atom_mutated = QtCore.pyqtSignal(int, str)              # (atom_index: int, new_symbol: str)
    molecule_cleared = QtCore.pyqtSignal()

    # Calculations & Live Job Execution
    job_started = QtCore.pyqtSignal(str, str)               # (input_file: str, runner_type: str)
    job_step_received = QtCore.pyqtSignal(list, object, float)  # (symbols, coords: np.ndarray, energy: float)
    job_completed = QtCore.pyqtSignal(str, bool)            # (output_path: str, success: bool)
    job_error = QtCore.pyqtSignal(str, str)                 # (job_name: str, error_message: str)

    # Workspace & Viewport Configuration
    theme_changed = QtCore.pyqtSignal(str)                  # "dark" or "white"
    style_mode_changed = QtCore.pyqtSignal(str)             # "Ball & Stick", "Spacefill", etc.
    viewport_camera_reset = QtCore.pyqtSignal()
    dock_visibility_toggled = QtCore.pyqtSignal(str, bool)  # (dock_id: str, visible: bool)
    status_message = QtCore.pyqtSignal(str, int)            # (message: str, timeout_ms: int)


# Global singleton instance
_GLOBAL_EVENT_BUS: Optional[AppEventBus] = None


def get_event_bus() -> AppEventBus:
    """Return the application-wide event bus singleton."""
    global _GLOBAL_EVENT_BUS
    if _GLOBAL_EVENT_BUS is None:
        _GLOBAL_EVENT_BUS = AppEventBus()
    return _GLOBAL_EVENT_BUS


def reset_event_bus() -> AppEventBus:
    """Reset the global event bus instance (primarily for isolated test fixtures)."""
    global _GLOBAL_EVENT_BUS
    _GLOBAL_EVENT_BUS = AppEventBus()
    return _GLOBAL_EVENT_BUS
