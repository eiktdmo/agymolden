"""
Quantum Chemistry Calculation Doctor & Crash Diagnostician.

Analyzes aborted, oscillating, or crashed QM log files (ORCA, Gaussian, Q-Chem, xTB)
and provides:
1. Root cause diagnosis (e.g. SCF oscillation, flat PES trough, negative frequencies, linear dependency).
2. Actionable remedies explained in plain language.
3. 1-Click "Auto-Fix Deck & Rerun" recipes.
"""

import re
import os
from typing import Dict, List, Optional, Any, NamedTuple


class DiagnosisResult(NamedTuple):
    error_type: str
    severity: str  # "fatal", "warning", "info"
    headline: str
    explanation: str
    remedy_description: str
    recommended_keywords: List[str]
    input_patch: Optional[Dict[str, str]] = None


DIAGNOSIS_RULES = [
    {
        "pattern": r"(SCF NOT CONVERGED|DIIS error|SCF convergence was not achieved|Convergence failure in the SCF)",
        "type": "SCF Convergence Failure",
        "severity": "fatal",
        "headline": "Self-Consistent Field (SCF) Divergence / Oscillation",
        "explanation": "The electronic wave function oscillated between configurations and failed to meet the energy convergence criterion. This often occurs in transition-metal complexes, open-shell radicals, or diffuse basis systems.",
        "remedy": "Switch to second-order convergence (AutoTRAH / SOSCF), increase maximum iterations, or apply level shifting to break frontier orbital degeneracy.",
        "keywords": ["! SlowConv", "! AutoTRAH", "%scf MaxIter 250 Shift 0.2 end", "! SOSCF"],
    },
    {
        "pattern": r"(GEOMETRY OPTIMIZATION CYCLE LIMIT EXCEEDED|Maximum number of steps reached|Optimization stopped)",
        "type": "Geometry Optimization Stalled",
        "severity": "warning",
        "headline": "Optimization Exceeded Step Limit (Flat PES or Flapping Torsion)",
        "explanation": "The optimizer ran out of cycles before forces reached zero. The molecule is usually stuck in a shallow potential energy well or oscillating along an ultra-soft dihedral torsion angle.",
        "remedy": "Restart optimization from the last trajectory geometry using tighter Cartesian trust radiuses, or increase MaxCycles.",
        "keywords": ["%geom MaxIter 200 end", "%geom Trust 0.1 end"],
    },
    {
        "pattern": r"(\*\*\* OPTIMIZATION FAILED \*\*\*|Linear angle in internal coordinates|Internal coordinate error)",
        "type": "Z-Matrix Coordinate Singularity",
        "severity": "fatal",
        "headline": "Internal Coordinate Singularity (Linear 180° Angle)",
        "explanation": "An angle approached 180 degrees, causing the B-matrix dihedral torsion to become undefined (division by zero in Wilson B-matrix).",
        "remedy": "Switch from standard internal coordinates (Redundant) to Cartesian coordinate optimization, or add dummy ghost atoms (X) at the inflection point.",
        "keywords": ["%geom OptCoords Cartesian end", "! OptCart"],
    },
    {
        "pattern": r"(Basis set linear dependency|Linear dependency in basis set|Smallest eigenvalue of overlap matrix)",
        "type": "Basis Set Linear Dependency",
        "severity": "fatal",
        "headline": "Near Linear Dependency in AO Overlap Matrix",
        "explanation": "The chosen basis set is too diffuse or redundant for the interatomic distances, causing the overlap matrix S to become singular (determinant ~ 0).",
        "remedy": "Remove diffuse augmentation functions (e.g. change ma-def2 to def2), or prune eigenvalues with a stricter canonical threshold.",
        "keywords": ["%scf OverlapTol 1e-6 end", "! def2-SVP"],
    },
    {
        "pattern": r"(VIBRATIONAL FREQUENCIES.*imaginary|Found [2-9] imaginary frequencies|imaginary frequencies)",
        "type": "Higher-Order Saddle Point",
        "severity": "warning",
        "headline": "Multiple Imaginary Frequencies (Higher-Order Saddle Point)",
        "explanation": "A transition state search should have exactly ONE imaginary frequency (negative eigenvalue). Finding multiple imaginary frequencies means the structure is an unphysical hilltop rather than a true transition state.",
        "remedy": "Distort the molecule along the 2nd imaginary normal mode vector using AgyMolden's SmartSlide, then re-optimize TS.",
        "keywords": ["! OptTS CalcHess"],
    },
    {
        "pattern": r"(OUT OF MEMORY|Insufficient memory|Allocation failed|bad_alloc)",
        "type": "Memory Exhaustion",
        "severity": "fatal",
        "headline": "Host Process Out of Memory (RAM)",
        "explanation": "The QM engine attempted an array allocation that exceeded available system physical memory or per-core allocation limit (%maxcore).",
        "remedy": "Increase %maxcore per process, reduce parallel thread count (%pal), or use resolution of identity (RI-J / RIJCOSX) approximations.",
        "keywords": ["%maxcore 4000", "! RIJCOSX"],
    },
]


def diagnose_qm_log(log_text: str, qm_engine: str = "orca") -> List[DiagnosisResult]:
    """Scan QM log file text and return diagnostic assessments with concrete remedies."""
    results: List[DiagnosisResult] = []
    seen_types = set()

    for rule in DIAGNOSIS_RULES:
        pat = str(rule["pattern"])
        if re.search(pat, log_text, re.IGNORECASE):
            r_type = str(rule["type"])
            if r_type not in seen_types:
                seen_types.add(r_type)
                results.append(
                    DiagnosisResult(
                        error_type=r_type,
                        severity=str(rule["severity"]),
                        headline=str(rule["headline"]),
                        explanation=str(rule["explanation"]),
                        remedy_description=str(rule["remedy"]),
                        recommended_keywords=list(rule["keywords"]),  # type: ignore[arg-type]
                    )
                )

    # If no specific rule matched but file ends abruptly
    if not results and ("TERMINATED ABNORMALLY" in log_text or "Error termination" in log_text):
        results.append(
            DiagnosisResult(
                error_type="Unspecified Abnormal Termination",
                severity="fatal",
                headline="Abnormal Process Termination",
                explanation="The calculation terminated with a nonzero exit code before finishing normal termination routines.",
                remedy_description="Check that MPI dependencies are installed, input file paths contain no special characters, and system resources are available.",
                recommended_keywords=["! SlowConv"],
            )
        )

    return results
