"""
Diagnostic helper for quantum chemistry calculations (especially ORCA and newBRORSEN).
Analyzes log output and provides friendly explanations and actionable remedies for students.
"""

import re
from typing import Optional, Dict


class QMDiagnosticResult:
    def __init__(self, title: str, summary: str, remedy: str, severity: str = "error"):
        self.title = title
        self.summary = summary
        self.remedy = remedy
        self.severity = severity  # "error", "warning", "info"


def diagnose_orca_error(raw_log: str, exit_code: int = 1) -> Optional[QMDiagnosticResult]:
    """Inspect ORCA stdout/stderr log text to identify common beginner errors."""
    if not raw_log:
        return None

    # 1. MPI / Parallel execution issues
    if re.search(r"(ORCA finished by error in ORCA_Init|orterun|mpirun|Open MPI|prun.*failed|No such file or directory.*opal)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Parallel Execution / MPI Configuration Issue",
            summary="ORCA encountered a problem launching its parallel helper processes (OpenMPI / %pal nprocs).",
            remedy=(
                "• Try running with serial execution first (%pal nprocs 1 end).\n"
                "• Verify that OpenMPI binaries (mpirun) match the exact version ORCA was compiled with.\n"
                "• On Linux/macOS, check if full path to ORCA directory is in your PATH and LD_LIBRARY_PATH."
            )
        )

    # 2. SCF Convergence failure
    if re.search(r"(SCF NOT CONVERGED|SCF failed to converge|This wavefunction IS NOT FULLY CONVERGED|DIIS error)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="SCF Did Not Converge",
            summary="The self-consistent field (SCF) calculation failed to reach energy convergence within the allowed iteration limit.",
            remedy=(
                "• In your ORCA input line, add: ! SlowConv or ! TightSCF\n"
                "• Increase maximum SCF iterations: %scf maxiter 150 end\n"
                "• For open-shell or transition metal systems, consider ! KDIIS or ! SOSCF."
            )
        )

    # 3. Basis set missing or unrecognized
    if re.search(r"(Basis set.*not found|Unknown basis set|There are no basis functions|Basis set not available for element)", raw_log, re.IGNORECASE):
        m = re.search(r"Basis set\s+([^\s]+)\s+not (found|available)", raw_log, re.IGNORECASE)
        basis_name = m.group(1) if m else "specified basis set"
        return QMDiagnosticResult(
            title="Basis Set Not Recognized or Unavailable",
            summary=f"ORCA could not locate the {basis_name} for one or more elements in your molecule.",
            remedy=(
                "• Check the spelling of the basis set (e.g. def2-SVP, def2-TZVP, 6-31G*, cc-pVDZ).\n"
                "• If using heavy elements (beyond Kr/Rb), ensure an effective core potential (ECP) or def2/relativistic basis set is used."
            )
        )

    # 4. Memory limit / Out of memory
    if re.search(r"(Out of memory|allocate memory|cannot allocate|insufficient memory|maxcore)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Insufficient Memory Limit (%maxcore)",
            summary="The calculation required more RAM than allocated in the %maxcore setting.",
            remedy=(
                "• In your input file, increase memory per core, for example: %maxcore 4000 (in MB).\n"
                "• If running with multiple processors (%pal nprocs 4), remember total memory required is nprocs × maxcore."
            )
        )

    # 5. Geometry optimization / Max cycles exceeded
    if re.search(r"(GEOMETRY OPTIMIZATION CYCLE.*EXCEEDED|Geometry optimization failed|Optimization did not converge)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Geometry Optimization Did Not Converge",
            summary="The optimization reached the maximum number of geometric steps without satisfying gradient and displacement criteria.",
            remedy=(
                "• In your %geom block, increase MaxIter: %geom MaxIter 100 end\n"
                "• Check the starting geometry: make sure atoms are not overlapping or bond distances are reasonable."
            )
        )

    # 6. Syntax error in input file
    if re.search(r"(INPUT ERROR|Error in input|Unknown keyword|PARSER ERROR)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Input Syntax Error",
            summary="ORCA found unrecognized keywords or syntax formatting issues in your .inp file.",
            remedy=(
                "• Check the top exclamation mark line (e.g. '! B3LYP def2-SVP Opt').\n"
                "• Check the coordinate block: ensure charge and multiplicity ('* xyz 0 1') are specified and closed with an asterisk '*'."
            )
        )

    # 7. Executable not found
    if "not found" in raw_log.lower() or "no such file or directory" in raw_log.lower():
        return QMDiagnosticResult(
            title="ORCA Executable Not Found",
            summary="The system could not locate the ORCA executable.",
            remedy=(
                "• Configure ORCA path in AgyMolden: Menu -> Configure QM Engines & Routes.\n"
                "• Or add the ORCA installation directory to your system PATH."
            )
        )

    return None
