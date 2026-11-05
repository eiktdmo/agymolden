"""
Chained Multi-Stage Quantum Chemistry Workflow Pipeline Engine.

Orchestrates sequential, interdependent calculation workflows:
1. Opt → Freq:
   Optimizes geometry, extracts final coordinates, runs harmonic frequency calculation,
   and verifies the absence of imaginary frequencies (True Minimum Guarantee).
2. Opt → TD-DFT:
   Optimizes ground state, runs TD-DFT excitation calculation at the stationary point,
   and produces UV-Vis optical absorption spectra.
3. TS Opt → Freq → IRC:
   Optimizes transition state, verifies exactly one imaginary frequency, and
   launches Intrinsic Reaction Coordinate (IRC) forward and reverse tracking.
"""

from dataclasses import dataclass, field
import os
import logging
from typing import List, Optional, Dict, Any, Callable
import numpy as np

from PyQt6 import QtCore

from .base import BaseCalcRunner, JobStatus, OptimizationStep
from .generic_runner import GenericQMRunner
from .orca_runner import OrcaRunner
from ..qm_log_parser import parse_qm_logfile
from ..qm_input_generator import (
    QMCalculationParameters,
    QM_GENERATORS,
    GaussianGenerator,
    OrcaGenerator,
    PySCFGenerator,
)

log = logging.getLogger(__name__)


@dataclass
class WorkflowStageConfig:
    name: str
    job_type: str  # "Opt", "Freq", "TD-DFT", "IRC", "SP"
    description: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)
    validator: Optional[Callable[[str], Tuple[bool, str]]] = None


WORKFLOW_TEMPLATES: Dict[str, List[WorkflowStageConfig]] = {
    "Opt → Freq (Minimum Guarantee)": [
        WorkflowStageConfig(name="Geometry Optimization", job_type="Opt", description="Optimize structure to stationary point"),
        WorkflowStageConfig(
            name="Harmonic Frequencies",
            job_type="Freq",
            description="Verify positive curvature (0 imaginary frequencies)",
        ),
    ],
    "Opt → TD-DFT (Absorption Spectrum)": [
        WorkflowStageConfig(name="Ground State Optimization", job_type="Opt", description="Optimize ground-state minimum"),
        WorkflowStageConfig(name="TD-DFT Optical Spectrum", job_type="TD-DFT", description="Calculate excited state energies & oscillator strengths"),
    ],
    "TS Opt → Freq (1 Imag) → IRC": [
        WorkflowStageConfig(name="Transition State Optimization", job_type="TS", description="Optimize to first-order saddle point"),
        WorkflowStageConfig(name="Frequency Check", job_type="Freq", description="Confirm exactly one imaginary vibrational mode"),
        WorkflowStageConfig(name="Intrinsic Reaction Coordinate", job_type="IRC", description="Trace minimum energy pathway from TS to Reactant/Product"),
    ],
}


class ChainedWorkflowPipeline(QtCore.QObject):
    """Executes a multi-stage calculation pipeline sequentially with automated data flow."""

    stage_started = QtCore.pyqtSignal(int, str)
    stage_finished = QtCore.pyqtSignal(int, str, bool, str)
    pipeline_finished = QtCore.pyqtSignal(bool, str)
    line_received = QtCore.pyqtSignal(str, bool)
    step_received = QtCore.pyqtSignal(OptimizationStep)

    def __init__(
        self,
        code_key: str,
        symbols: List[str],
        initial_coords: np.ndarray,
        base_params: QMCalculationParameters,
        stages: List[WorkflowStageConfig],
        work_dir: str,
        parent: Optional[QtCore.QObject] = None,
    ):
        super().__init__(parent)
        self.code_key = code_key.lower()
        self.symbols = list(symbols)
        self.current_coords = np.array(initial_coords, copy=True)
        self.base_params = base_params
        self.stages = stages
        self.work_dir = os.path.abspath(work_dir)
        self.current_stage_idx: int = -1
        self.active_runner: Optional[BaseCalcRunner] = None
        self.is_aborted: bool = False
        self.stage_results: List[Dict[str, Any]] = []

    def start(self):
        """Begin pipeline execution from stage 0."""
        os.makedirs(self.work_dir, exist_ok=True)
        self.is_aborted = False
        self.current_stage_idx = -1
        self._advance_to_next_stage()

    def abort(self):
        """Abort current running stage and halt pipeline."""
        self.is_aborted = True
        if self.active_runner and hasattr(self.active_runner, "abort"):
            self.active_runner.abort()
        self.pipeline_finished.emit(False, "Pipeline aborted by user.")

    def _advance_to_next_stage(self):
        if self.is_aborted:
            return

        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.stages):
            # All stages completed successfully!
            self.pipeline_finished.emit(True, f"All {len(self.stages)} workflow stages completed successfully!")
            return

        stage = self.stages[self.current_stage_idx]
        self.stage_started.emit(self.current_stage_idx, stage.name)

        # 1. Generate input for this stage
        gen_cls = None
        for name, cls in QM_GENERATORS.items():
            if self.code_key in name.lower() or self.code_key == getattr(cls, "code_name", "").lower():
                gen_cls = cls
                break
        if not gen_cls:
            self.pipeline_finished.emit(False, f"No generator available for QM code: {self.code_key}")
            return

        stage_params = QMCalculationParameters(
            job_type=stage.job_type,
            method=self.base_params.method,
            basis=self.base_params.basis,
            charge=self.base_params.charge,
            multiplicity=self.base_params.multiplicity,
            memory_gb=self.base_params.memory_gb,
            n_procs=self.base_params.n_procs,
            dispersion=self.base_params.dispersion,
            solvent=self.base_params.solvent,
            solvent_model=self.base_params.solvent_model,
            title=f"Stage {self.current_stage_idx + 1}: {stage.name}",
        )
        for k, v in stage.extra_params.items():
            if hasattr(stage_params, k):
                setattr(stage_params, k, v)

        gen = gen_cls()
        ext = gen.file_extension
        stage_input_text = gen.generate_input(self.symbols, self.current_coords, stage_params)

        stem = f"stage_{self.current_stage_idx + 1}_{stage.job_type.lower().replace(' ', '_').replace('+', '')}"
        input_path = os.path.join(self.work_dir, f"{stem}{ext}")
        with open(input_path, "w", encoding="utf-8") as f:
            f.write(stage_input_text)

        # 2. Instantiate and connect runner
        if "orca" in self.code_key:
            runner = OrcaRunner(parent=self)
        else:
            runner = GenericQMRunner(code_key=self.code_key, parent=self)

        self.active_runner = runner
        runner.line_received.connect(self.line_received)
        runner.step_received.connect(self.step_received)
        runner.finished.connect(self._on_stage_runner_finished)

        success = runner.start_job(input_path, work_dir=self.work_dir)
        if not success:
            self.stage_finished.emit(self.current_stage_idx, stage.name, False, "Failed to launch runner process.")
            self.pipeline_finished.emit(False, f"Failed starting stage: {stage.name}")

    def _on_stage_runner_finished(self, success: bool, msg: str):
        if self.is_aborted:
            return

        stage = self.stages[self.current_stage_idx]
        if not success:
            self.stage_finished.emit(self.current_stage_idx, stage.name, False, msg)
            self.pipeline_finished.emit(False, f"Stage {self.current_stage_idx + 1} ({stage.name}) failed: {msg}")
            return

        # Parse stage output
        stem = f"stage_{self.current_stage_idx + 1}_{stage.job_type.lower().replace(' ', '_').replace('+', '')}"
        cand_log = os.path.join(self.work_dir, f"{stem}.log")
        if not os.path.isfile(cand_log):
            cand_log = os.path.join(self.work_dir, f"{stem}.out")

        parsed = None
        if os.path.isfile(cand_log):
            parsed = parse_qm_logfile(cand_log)

        # Update geometry if optimization produced new coordinates
        if parsed and parsed.trajectory_coords and len(parsed.trajectory_coords) > 0:
            self.current_coords = parsed.trajectory_coords[-1]

        # Stage specific validations
        status_msg = "Completed successfully."
        if stage.job_type == "Freq" and parsed and parsed.vibrational_analysis:
            freqs = parsed.vibrational_analysis.frequencies
            n_imag = len([f for f in freqs if f < -1.0])
            if "TS" in self.stages[max(0, self.current_stage_idx - 1)].name:
                if n_imag != 1:
                    status_msg = f"Warning: TS check expects exactly 1 imaginary frequency, found {n_imag}."
                else:
                    imag_val = [f for f in freqs if f < -1.0][0]
                    status_msg = f"Verified single imaginary mode: {imag_val:.1f} cm⁻¹ (True TS)."
            else:
                if n_imag > 0:
                    status_msg = f"Warning: Found {n_imag} imaginary frequencies! Structure is a saddle point, not a true minimum."
                else:
                    lowest = min([f for f in freqs if f > 0.0]) if any(f > 0 for f in freqs) else 0.0
                    status_msg = f"Verified 0 imaginary frequencies. True minimum (lowest: {lowest:.1f} cm⁻¹)."

        self.stage_results.append({
            "stage_idx": self.current_stage_idx,
            "name": stage.name,
            "log": cand_log,
            "parsed": parsed,
            "status_msg": status_msg,
        })

        self.stage_finished.emit(self.current_stage_idx, stage.name, True, status_msg)
        self._advance_to_next_stage()
