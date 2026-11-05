"""
Diagnostic helper for quantum chemistry calculations (especially ORCA and newBRORSEN).
Analyzes log output and provides friendly explanations and actionable remedies for students.
"""

import re
from typing import Optional, Dict


class QMDiagnosticResult:
    def __init__(
        self,
        title: str,
        summary: str,
        remedy: str,
        severity: str = "error",
        suggested_keywords: Optional[Dict[str, str]] = None,
    ):
        self.title = title
        self.summary = summary
        self.remedy = remedy
        self.severity = severity  # "error", "warning", "info"
        self.suggested_keywords = suggested_keywords or {}


def diagnose_qm_error(raw_log: str, engine: str = "orca", exit_code: int = 1) -> Optional[QMDiagnosticResult]:
    """
    Universal QC Doctor diagnostic parser.
    Inspects stdout/stderr log text from ORCA, Gaussian, or PySCF to identify common
    calculation bottlenecks/crashes and returns structured remedies + keywords for 1-click fixing.
    """
    if not raw_log:
        return None

    eng = (engine or "orca").lower()
    if eng == "gaussian" or ("gaussian" in raw_log.lower() and "orca" not in raw_log.lower()):
        return diagnose_gaussian_error(raw_log, exit_code=exit_code)
    elif eng == "pyscf" or ("pyscf" in raw_log.lower() and "orca" not in raw_log.lower()):
        return diagnose_pyscf_error(raw_log, exit_code=exit_code)
    return diagnose_orca_error(raw_log, exit_code=exit_code)


def diagnose_gaussian_error(raw_log: str, exit_code: int = 1) -> Optional[QMDiagnosticResult]:
    """Diagnose Gaussian calculation failures (Link 502, Link 9999, Link 103, etc.)."""
    if not raw_log:
        return None

    # 1. SCF Convergence failure (l502.exe)
    if re.search(r"(l502\.exe|Convergence failure -- run terminated|SCF failed to converge)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Gaussian SCF Convergence Failure (Error 2070 / Link 502)",
            summary="The self-consistent field (SCF) calculation failed to reach convergence within the allowed cycles.",
            remedy=(
                "• Add 'scf=(qc,maxcycle=200)' or 'scf=(xqc,maxcycle=150)' to the route card for quadratic convergence.\n"
                "• Increase damping with 'scf=(damp,maxcycle=150)'.\n"
                "• If using open-shell transition metals, try 'guess=mix' or unrestricted broken-symmetry."
            ),
            suggested_keywords={"scf": "scf=(xqc,maxcycle=150)"}
        )

    # 2. Geometry optimization exceeded max cycles (l9999.exe)
    if re.search(r"(l9999\.exe|Maximum number of optimization cycles reached|Optimization stopped)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Geometry Optimization Exceeded Max Cycles (Link 9999)",
            summary="Gaussian reached the maximum geometric optimization steps without meeting convergence criteria.",
            remedy=(
                "• Increase max cycles in route card: 'opt=(maxcycles=150)'.\n"
                "• Recalculate exact force constants periodically: 'opt=(calcfc,maxcycles=100)'.\n"
                "• Re-read last geometry with 'geom=check' and re-optimize."
            ),
            suggested_keywords={"opt": "opt=(calcfc,maxcycles=100)"}
        )

    # 3. Coordinate / Z-matrix / Internal coordinate error (l103.exe)
    if re.search(r"(l103\.exe|Linear bend in angle|FormBX had a problem|Error in internal coordinate)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Linear Angle / Coordinate Breakdown (Link 103)",
            summary="Redundant internal coordinates encountered a near-linear bond angle (approaching 180°).",
            remedy=(
                "• Switch to Cartesian coordinate optimization in Gaussian: 'opt=cartesian'.\n"
                "• Or re-orient near-linear angles slightly away from exact 180° in the Z-matrix."
            ),
            suggested_keywords={"opt": "opt=cartesian"}
        )

    # 4. Memory or Disk limit (l1.exe / alloc)
    if re.search(r"(Out-of-memory error|allocation failure|galloc|cannot allocate memory)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Gaussian Memory Allocation Failure",
            summary="Calculation demanded more memory than specified in the '%mem' directive.",
            remedy=(
                "• Increase %mem directive at the very top of your input (e.g. '%mem=8GB').\n"
                "• Ensure physical RAM on your system matches the allocated memory."
            ),
            suggested_keywords={"mem": "%mem=8GB"}
        )

    # 5. Route Syntax Error / Unrecognized Keyword (QPErr / Link 1)
    if re.search(r"(QPErr|syntax error was detected in the input line|ambiguous keyword|unrecognized keyword)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Gaussian Route Card Syntax Error (QPErr / Link 1)",
            summary="Gaussian rejected a keyword, basis set, or functional name in the '#p' route line.",
            remedy=(
                "• Remove hyphens from Ahlrichs basis sets: use 'def2SVP' or 'def2TZVP' (not 'def2-SVP').\n"
                "• Remove hyphens from functionals: use 'wB97XD' (not 'wB97X-D') or 'M062X' (not 'M06-2X').\n"
                "• Specify dispersion as 'EmpiricalDispersion=GD3BJ'.\n"
                "• Check for misspelled keywords or unsupported options in Gaussian."
            ),
            suggested_keywords={"basis": "def2SVP"}
        )

    # 6. Coordinate Format Error (Link 101 / Floating point requirement)
    if re.search(r"(Wanted a floating point number as input|Found an integer as input|RdGeom|routine RdGeom)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Gaussian Coordinate Format Error (Link 101)",
            summary="Gaussian's Fortran reader strictly requires all coordinates to include decimal points (e.g. '0.0', not '0').",
            remedy=(
                "• Ensure all Cartesian coordinates have explicit decimal points.\n"
                "• If using Z-matrix, ensure all distances and angles have decimal points and no illegal header labels exist."
            )
        )

    # 7. Gaussian Environment Misconfigured (GAUSS_EXEDIR)
    if re.search(r"(No executable for file|Search path GAUSS_EXEDIR is|cannot execute.*l\d+\.exe)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Gaussian Environment Misconfigured (GAUSS_EXEDIR)",
            summary="Gaussian cannot locate its link executables because GAUSS_EXEDIR or g16root is not configured.",
            remedy=(
                "• Ensure Gaussian 16 root folder is properly configured under 'Executable & Route'.\n"
                "• AgyMolden automatically synthesizes GAUSS_EXEDIR and GAUSS_SCRDIR when launched through the app."
            )
        )

    return None


def diagnose_pyscf_error(raw_log: str, exit_code: int = 1) -> Optional[QMDiagnosticResult]:
    """Diagnose PySCF calculation failures."""
    if not raw_log:
        return None

    if re.search(r"(scf not converged|scf iteration failed|did not converge)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="PySCF SCF Convergence Failure",
            summary="PySCF SCF solver did not converge to the required energy or density threshold.",
            remedy=(
                "• Enable second-order SCF solver: mf.newton().run()\n"
                "• Increase maximum cycles: mf.max_cycle = 150\n"
                "• Enable level shift for small HOMO-LUMO gaps: mf.level_shift = 0.2"
            ),
            suggested_keywords={"solver": "mf.newton().run()"}
        )

    if re.search(r"(linear dependency|overlap matrix.*singular|near singular)", raw_log, re.IGNORECASE):
        return QMDiagnosticResult(
            title="Basis Set Linear Dependency in PySCF",
            summary="Diffuse basis functions or overlapping atoms produced a near-singular overlap matrix.",
            remedy=(
                "• Remove linear dependencies: mf.lindep_threshold = 1e-4\n"
                "• Switch from diffuse basis (e.g., aug-cc-pVTZ) to def2-TZVP."
            ),
            suggested_keywords={"lindep": "mf.lindep_threshold = 1e-4"}
        )

    return None


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
            ),
            suggested_keywords={"extra": "SlowConv", "scf_block": "%scf maxiter 150 end"}
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
