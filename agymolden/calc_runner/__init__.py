"""
Modular Quantum Chemistry Calculation Runner Subsystem.

Provides:
- BaseCalcRunner: Process management, stdout/stderr streaming, live parsing.
- BrorsenRunner: Dedicated runner for newBRORSEN calculations.
- OrcaRunner: Dedicated runner for ORCA calculations.
- JobMonitorDialog: GUI window for live monitoring, log streaming, and real-time 3D updates.
"""

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from .brorsen_runner import BrorsenRunner, detect_conda_environments, find_brorsen_wrappers
from .orca_runner import OrcaRunner
from .generic_runner import GenericQMRunner
from .dialog_job_monitor import JobMonitorDialog
from .dialog_launch import BrorsenLaunchDialog

__all__ = [
    "BaseCalcRunner",
    "JobStatus",
    "OptimizationStep",
    "BrorsenRunner",
    "OrcaRunner",
    "GenericQMRunner",
    "JobMonitorDialog",
    "BrorsenLaunchDialog",
    "detect_conda_environments",
    "find_brorsen_wrappers",
]
