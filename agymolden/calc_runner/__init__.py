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
from .preflight import validate_qm_calculation, PreflightReport, ValidationIssue
from .dialog_preflight import PreflightValidationDialog
from .pipeline import ChainedWorkflowPipeline, WORKFLOW_TEMPLATES, WorkflowStageConfig
from .dialog_pipeline import WorkflowPipelineDialog

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
    "validate_qm_calculation",
    "PreflightReport",
    "ValidationIssue",
    "PreflightValidationDialog",
    "ChainedWorkflowPipeline",
    "WORKFLOW_TEMPLATES",
    "WorkflowStageConfig",
    "WorkflowPipelineDialog",
]
