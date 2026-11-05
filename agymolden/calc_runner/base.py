"""
Base Calculation Runner Interface.

Manages external calculation processes (via QProcess or subprocess), captures live stdout/stderr,
parses geometry/energy updates on the fly, and emits Qt signals for UI components.
"""

import os
import sys
import enum
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

from PyQt6 import QtCore

log = logging.getLogger(__name__)


class JobStatus(enum.Enum):
    IDLE = "Idle"
    STARTING = "Starting"
    RUNNING = "Running"
    CONVERGED = "Converged"
    FINISHED = "Finished"
    FAILED = "Failed"
    STOPPED = "Stopped by User"


@dataclass
class OptimizationStep:
    cycle: int
    energy: Optional[float] = None
    delta_e: Optional[float] = None
    gradient_norm: Optional[float] = None
    step_norm: Optional[float] = None
    symbols: Optional[List[str]] = None
    coords: Optional[np.ndarray] = None  # (N, 3) in Angstroms
    is_converged: bool = False


class BaseCalcRunner(QtCore.QObject):
    """
    Abstract base runner managing an external calculation process.

    Signals:
        line_received(str, bool): Emitted on each stdout/stderr line (line, is_stderr).
        status_changed(JobStatus, str): Emitted when calculation status changes (status, message).
        step_updated(OptimizationStep): Emitted when a new geometry or energy step is parsed.
        finished(bool, str): Emitted on calculation completion (success, summary_message).
    """

    line_received = QtCore.pyqtSignal(str, bool)
    status_changed = QtCore.pyqtSignal(object, str)
    step_updated = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(bool, str)
    artifacts_downloaded = QtCore.pyqtSignal(str)

    def __init__(self, parent: Optional[QtCore.QObject] = None):
        super().__init__(parent)
        self.process: Optional[QtCore.QProcess] = None
        self.status: JobStatus = JobStatus.IDLE
        self.work_dir: str = ""
        self.input_file: str = ""
        self.log_file: str = ""
        self.steps: List[OptimizationStep] = []
        self._raw_log_buffer: List[str] = []

    def is_running(self) -> bool:
        return self.process is not None and self.process.state() != QtCore.QProcess.ProcessState.NotRunning

    def start_job(self, input_file: str, work_dir: Optional[str] = None, **kwargs) -> bool:
        """Subclasses implement command line assembly and process launch."""
        raise NotImplementedError

    def stop_job(self):
        """Terminate the active calculation cleanly."""
        if self.process and self.is_running():
            self._set_status(JobStatus.STOPPED, "Stopping calculation process...")
            self.process.terminate()
            if not self.process.waitForFinished(3000):
                self.process.kill()

    def _setup_process(self, program: str, arguments: List[str], work_dir: str, extra_env: Optional[Dict[str, str]] = None):
        self.process = QtCore.QProcess(self)
        self.process.setWorkingDirectory(work_dir)
        if extra_env:
            q_env = QtCore.QProcessEnvironment.systemEnvironment()
            for k, v in extra_env.items():
                q_env.insert(k, v)
            self.process.setProcessEnvironment(q_env)
        if getattr(self, "stdin_file", None) and os.path.isfile(self.stdin_file):
            self.process.setStandardInputFile(self.stdin_file)
        self.process.readyReadStandardOutput.connect(self._on_stdout_ready)
        self.process.readyReadStandardError.connect(self._on_stderr_ready)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self._set_status(JobStatus.STARTING, f"Starting {os.path.basename(program)}...")
        log.info(f"Launching process: {program} {' '.join(arguments)} in {work_dir}")
        self.process.start(program, arguments)

    def _on_stdout_ready(self):
        if not self.process:
            return
        while self.process.canReadLine():
            line_bytes = self.process.readLine()
            line = bytes(line_bytes).decode("utf-8", errors="replace").rstrip("\r\n")
            self._raw_log_buffer.append(line)
            self.line_received.emit(line, False)
            self._parse_line(line)

    def _on_stderr_ready(self):
        if not self.process:
            return
        raw = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            self._raw_log_buffer.append(line)
            self.line_received.emit(line, True)
            if hasattr(self, "_out_fh") and getattr(self, "_out_fh") is not None:
                try:
                    self._out_fh.write(line + "\n")
                    self._out_fh.flush()
                except Exception:
                    logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

    def _parse_line(self, line: str):
        """Subclasses override to extract step info on the fly."""
        pass

    def _on_process_finished(self, exit_code: int, exit_status: QtCore.QProcess.ExitStatus):
        if self.process:
            rem_out = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
            for line in rem_out.splitlines():
                if line:
                    self._raw_log_buffer.append(line)
                    self.line_received.emit(line, False)
                    self._parse_line(line)
            rem_err = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
            for line in rem_err.splitlines():
                if line:
                    self._raw_log_buffer.append(line)
                    self.line_received.emit(line, True)
                    if hasattr(self, "_out_fh") and getattr(self, "_out_fh") is not None:
                        try:
                            self._out_fh.write(line + "\n")
                            self._out_fh.flush()
                        except Exception as e:
                            log.debug("Failed writing stderr line to log file: %s", e)
        success = (exit_code == 0 and exit_status == QtCore.QProcess.ExitStatus.NormalExit)
        if success:
            self._set_status(JobStatus.FINISHED, f"Job finished successfully (code {exit_code})")
        else:
            self._set_status(JobStatus.FAILED, f"Job terminated with exit code {exit_code}")
        self.finished.emit(success, f"Exit code {exit_code}")

    def _on_process_error(self, err: QtCore.QProcess.ProcessError):
        msg = f"Process error: {err}"
        self._set_status(JobStatus.FAILED, msg)
        self.finished.emit(False, msg)

    def _set_status(self, status: JobStatus, message: str = ""):
        self.status = status
        self.last_error_message = message if status == JobStatus.FAILED else ""
        self._status_message = message
        self.status_changed.emit(status, message)
