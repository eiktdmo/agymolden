"""
Pre-flight calculation validation engine for AgyMolden.

Verifies calculation setup, geometry, charge/multiplicity, hardware allocation,
and engine-specific requirements before running or saving input files.
Detects potential issues (e.g., net electron spin parity, memory exceeding RAM,
conflicting keywords, unphysical bond lengths) *before* jobs fail.
"""

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np

from ..elements import ELEMENT_DATA
from ..qm_input_generator import QMCalculationParameters
from ..qm_detect import detect_qm_executable, QMExecutableInfo

log = logging.getLogger(__name__)


@dataclass
class ValidationIssue:
    severity: str  # "error", "warning", "info"
    title: str
    message: str
    remedy: str = ""
    field: Optional[str] = None  # e.g., "charge", "multiplicity", "memory", "basis", "geometry"


@dataclass
class PreflightReport:
    is_valid: bool
    has_errors: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    total_electrons: int = 0
    num_alpha: int = 0
    num_beta: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")


def validate_qm_calculation(
    code_key: str,
    symbols: List[str],
    coords: np.ndarray,
    params: QMCalculationParameters,
    work_dir: Optional[str] = None,
    executable_info: Optional[QMExecutableInfo] = None,
) -> PreflightReport:
    """
    Perform comprehensive pre-flight verification on a planned QM calculation.
    """
    issues: List[ValidationIssue] = []

    # 1. Molecular Geometry Sanity Check
    if len(symbols) == 0:
        issues.append(
            ValidationIssue(
                severity="error",
                title="Empty Molecule",
                message="No atoms are defined in the molecular system.",
                remedy="Build or import a molecule before setting up a calculation.",
                field="geometry",
            )
        )
        return PreflightReport(is_valid=False, has_errors=True, issues=issues)

    if coords is None or len(coords) != len(symbols):
        issues.append(
            ValidationIssue(
                severity="error",
                title="Coordinate Mismatch",
                message=f"Atom count ({len(symbols)}) does not match coordinate count ({len(coords) if coords is not None else 0}).",
                remedy="Re-generate molecular coordinates.",
                field="geometry",
            )
        )
        return PreflightReport(is_valid=False, has_errors=True, issues=issues)

    # Check for overlapping / clashing atoms (distance < 0.5 Angstrom)
    n_atoms = len(symbols)
    if n_atoms > 1:
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                dist = float(np.linalg.norm(coords[i] - coords[j]))
                if dist < 0.45:
                    issues.append(
                        ValidationIssue(
                            severity="error",
                            title="Severe Nuclear Clash",
                            message=f"Atoms {symbols[i]}{i+1} and {symbols[j]}{j+1} are only {dist:.3f} Å apart (< 0.45 Å).",
                            remedy="Adjust geometry or optimize with force field before launching QM calculation.",
                            field="geometry",
                        )
                    )
                    break
            if any(iss.title == "Severe Nuclear Clash" for iss in issues):
                break

    # Check for NaN or Inf in coordinates
    if not np.all(np.isfinite(coords)):
        issues.append(
            ValidationIssue(
                severity="error",
                title="Invalid Coordinates",
                message="One or more atomic coordinates contain NaN or Infinite values.",
                remedy="Reset geometry or reload clean structure.",
                field="geometry",
            )
        )

    # 2. Electron Count, Charge, and Multiplicity Parity
    nuclear_charge = 0
    for s in symbols:
        data = ELEMENT_DATA.get(s, {})
        nuclear_charge += data.get("atomic_number", 0)

    total_electrons = nuclear_charge - params.charge
    num_unpaired = params.multiplicity - 1

    if total_electrons <= 0:
        issues.append(
            ValidationIssue(
                severity="error",
                title="Zero or Negative Electron Count",
                message=f"Total electrons = {total_electrons} (Nuclear charge: {nuclear_charge}, Net charge: {params.charge}).",
                remedy="Check net molecular charge.",
                field="charge",
            )
        )
    else:
        # Spin parity rule:
        # total_electrons and num_unpaired must have the same parity (both even or both odd)
        # i.e., (total_electrons - num_unpaired) % 2 == 0
        if (total_electrons - num_unpaired) % 2 != 0:
            expected_type = "odd" if (total_electrons % 2 != 0) else "even"
            issues.append(
                ValidationIssue(
                    severity="error",
                    title="Spin Multiplicity Parity Error",
                    message=(
                        f"System with {total_electrons} electrons cannot have multiplicity {params.multiplicity} "
                        f"({num_unpaired} unpaired electrons). For {total_electrons} electrons, "
                        f"multiplicity must be an {'odd' if expected_type == 'even' else 'even'} number "
                        f"(e.g., {1 if total_electrons % 2 == 0 else 2}, {3 if total_electrons % 2 == 0 else 4})."
                    ),
                    remedy=(
                        f"Set multiplicity to {1 if total_electrons % 2 == 0 else 2} for ground state, "
                        f"or adjust net charge."
                    ),
                    field="multiplicity",
                )
            )

        if num_unpaired > total_electrons:
            issues.append(
                ValidationIssue(
                    severity="error",
                    title="Multiplicity Exceeds Total Electrons",
                    message=f"Multiplicity {params.multiplicity} requires {num_unpaired} unpaired electrons, but system only has {total_electrons} electrons.",
                    remedy="Reduce multiplicity or change net charge.",
                    field="multiplicity",
                )
            )

    n_alpha = (total_electrons + num_unpaired) // 2 if total_electrons > 0 and (total_electrons - num_unpaired) % 2 == 0 else 0
    n_beta = (total_electrons - num_unpaired) // 2 if total_electrons > 0 and (total_electrons - num_unpaired) % 2 == 0 else 0

    # 3. Hardware Resource Sanity Check (RAM, CPU Cores, Disk Space)
    total_ram_gb = None
    avail_ram_gb = None
    try:
        import psutil
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        avail_ram_gb = psutil.virtual_memory().available / (1024**3)
    except Exception:
        # Fallback to standard library os.sysconf on Linux/macOS
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            total_ram_gb = (pages * page_size) / (1024**3)
        except Exception as e:
            log.debug("System RAM query failed: %s", e)

    if total_ram_gb is not None:
        if params.memory_gb > total_ram_gb:
            issues.append(
                ValidationIssue(
                    severity="error",
                    title="Memory Exceeds Physical RAM",
                    message=(
                        f"Allocated memory ({params.memory_gb} GB) exceeds total physical system RAM "
                        f"({total_ram_gb:.1f} GB). This calculation will be killed by OS OOM killer."
                    ),
                    remedy=f"Reduce memory allocation to {max(2, int(total_ram_gb * 0.75))} GB or less.",
                    field="memory",
                )
            )
        elif avail_ram_gb is not None and params.memory_gb > avail_ram_gb:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title="Memory Exceeds Available RAM",
                    message=(
                        f"Allocated memory ({params.memory_gb} GB) exceeds currently available free RAM "
                        f"({avail_ram_gb:.1f} GB of {total_ram_gb:.1f} GB). System may swap."
                    ),
                    remedy=f"Recommended memory limit: {max(2, int(avail_ram_gb * 0.85))} GB.",
                    field="memory",
                )
            )

    try:
        cpu_count = os.cpu_count() or 1
        if params.n_procs > cpu_count:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title="Oversubscribed CPU Cores",
                    message=f"Requested {params.n_procs} CPU processes/threads, but machine only has {cpu_count} logical cores.",
                    remedy=f"Set number of processors to {cpu_count} or fewer to prevent thrashing.",
                    field="n_procs",
                )
            )
    except Exception as e:
        log.debug("CPU count query failed: %s", e)

    # Check disk space in work_dir / scratch
    target_dir = work_dir or os.getcwd()
    try:
        free_disk_gb = shutil.disk_usage(target_dir).free / (1024**3)
        if free_disk_gb < 1.0:
            issues.append(
                ValidationIssue(
                    severity="error",
                    title="Critically Low Disk Space",
                    message=f"Only {free_disk_gb:.2f} GB of disk space remaining in {target_dir}. Calculation requires scratch storage.",
                    remedy="Free disk space or change working folder.",
                    field="disk",
                )
            )
        elif free_disk_gb < 4.0:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title="Low Free Disk Space",
                    message=f"Working drive has only {free_disk_gb:.1f} GB free. Large integral / two-electron files may fill disk.",
                    remedy="Ensure at least 5-10 GB free for frequency or post-HF calculations.",
                    field="disk",
                )
            )
    except Exception as e:
        log.debug("Disk usage query failed: %s", e)

    # 4. Engine-Specific Validations
    code = (code_key or "").lower()

    # Gaussian Specifics
    if "gaussian" in code or "g16" in code or "g09" in code:
        if params.memory_gb < 2:
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title="Gaussian Minimum Memory Recommendation",
                    message=f"Allocated {params.memory_gb} GB memory. Gaussian Link 101 frequently faults when memory is < 2 GB.",
                    remedy="Increase memory to at least 2 GB (e.g. %mem=2GB).",
                    field="memory",
                )
            )

        # Check basis set compatibility
        basis_raw = params.basis.strip()
        if basis_raw.lower().startswith("def2-"):
            issues.append(
                ValidationIssue(
                    severity="info",
                    title="Ahlrichs Basis Hyphen Auto-Normalized",
                    message=f"Basis set '{basis_raw}' contains a hyphen. AgyMolden will auto-normalize it to '{basis_raw.replace('def2-', 'def2')}' for Gaussian.",
                    remedy="No action needed; handled automatically.",
                    field="basis",
                )
            )

        # Method & Dispersion compatibility
        m_up = params.method.upper().split()[0]
        if m_up in ("HF", "MP2", "CCSD(T)", "CCSD") and params.dispersion and params.dispersion not in ("None", ""):
            issues.append(
                ValidationIssue(
                    severity="info",
                    title="Dispersion Omitted for Wavefunction Method",
                    message=f"D3 dispersion parameters are not parameterized for pure {m_up} in Gaussian. AgyMolden will omit empirical dispersion.",
                    remedy="No action needed; handled automatically.",
                    field="dispersion",
                )
            )
        elif m_up in ("M062X", "M06-2X", "M06HF", "M06L") and params.dispersion and "D3BJ" in params.dispersion.upper():
            issues.append(
                ValidationIssue(
                    severity="info",
                    title="Becke-Johnson Damping Adjusted to GD3",
                    message=f"{params.method} has no Becke-Johnson (BJ) damping parameterization in Gaussian. AgyMolden will auto-switch to GD3 zero-damping.",
                    remedy="No action needed; handled automatically.",
                    field="dispersion",
                )
            )

    # ORCA Specifics
    elif "orca" in code:
        if params.n_procs > 1:
            # Check MPI
            pass
        if params.dispersion == "D4" and params.method.upper().startswith("HF"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    title="D4 with Pure HF",
                    message="D4 dispersion is primarily parameterized for DFT functionals, not pure Hartree-Fock.",
                    remedy="Select a DFT functional (e.g., B3LYP, PBE0, wB97X-D) or choose D3BJ.",
                    field="dispersion",
                )
            )

    # Executable availability
    if executable_info is not None and not executable_info.found:
        issues.append(
            ValidationIssue(
                severity="warning",
                title=f"{executable_info.display_name} Not Found Locally",
                message=f"The executable for {executable_info.display_name} was not detected on this machine. You can save the input file or configure a remote SSH server.",
                remedy="Install the package locally or submit job to a configured HPC remote cluster.",
                field="executable",
            )
        )

    has_errors = any(issue.severity == "error" for issue in issues)
    return PreflightReport(
        is_valid=not has_errors,
        has_errors=has_errors,
        issues=issues,
        total_electrons=total_electrons,
        num_alpha=n_alpha,
        num_beta=n_beta,
    )
