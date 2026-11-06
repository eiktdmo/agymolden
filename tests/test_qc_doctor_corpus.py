"""
Comprehensive benchmark suite testing QC Doctor™ diagnostic accuracy against a real corpus
of quantum chemistry failure logs across ORCA, Gaussian, Psi4, PySCF, and xTB.
"""

import pytest
from agymolden.calc_runner.diagnostics import diagnose_qm_error


FAILURES_CORPUS = [
    # --- ORCA FAILURES ---
    {
        "id": "orca_scf_divergence",
        "engine": "orca",
        "log": """
        ------------------
        SCF ITERATIONS
        ------------------
        Iter   Total Energy       Delta-E         RMS-DP      MAX-DP      Time
          98   -154.29810481    -0.0001021481   0.0019208   0.0124810    0.3
          99   -154.29798104     0.0001237700   0.0021491   0.0149012    0.3
         100   -154.29821045    -0.0002294100   0.0020104   0.0138901    0.3
        SCF NOT CONVERGED AFTER 100 CYCLES. Giving up.
        ORCA finished by error termination in SCF
        """,
        "expected_title": "SCF Did Not Converge",
        "expected_remedy_keyword": "SlowConv",
    },
    {
        "id": "orca_geom_maxiter",
        "engine": "orca",
        "log": """
        ---------------------------------
        GEOMETRY OPTIMIZATION CYCLE 50
        ---------------------------------
        Energy change:  -0.00004521  (Tol: -0.00000500)  NO
        RMS gradient:    0.00084512  (Tol:  0.00010000)  NO
        MAX gradient:    0.00294100  (Tol:  0.00030000)  NO
        GEOMETRY OPTIMIZATION CYCLE 50 EXCEEDED
        Optimization did not converge.
        """,
        "expected_title": "Geometry Optimization Did Not Converge",
        "expected_remedy_keyword": "MaxIter",
    },
    {
        "id": "orca_trust_radius",
        "engine": "orca",
        "log": """
        Optimizer Step  14:
        Trust radius too small (0.00001). Line search failed to bracket minimum.
        ORCA finished by error termination in GDIIS
        """,
        "expected_title": "Line Search / Trust Radius",
        "expected_remedy_keyword": "Recalc_Hess",
    },
    {
        "id": "orca_linear_dependency",
        "engine": "orca",
        "log": """
        WARNING: Overlap matrix is singular or near singular!
        Smallest eigenvalue of overlap matrix = 3.4210e-07 (linear dependency detected)
        ORCA finished by error termination in Integral Transformation
        """,
        "expected_title": "Linear Dependency",
        "expected_remedy_keyword": "TolLinDep",
    },
    {
        "id": "orca_memory_limit",
        "engine": "orca",
        "log": """
        Calling ORCA_CPCM ...
        Error in matrix allocation: Out of memory, cannot allocate 32000 MB.
        Please adjust %maxcore.
        """,
        "expected_title": "Insufficient Memory Limit",
        "expected_remedy_keyword": "maxcore",
    },
    {
        "id": "orca_mpi_failure",
        "engine": "orca",
        "log": """
        [mpirun@cluster01] ORCA finished by error in ORCA_Init
        orterun failed to launch child process orca_gto on node node04
        """,
        "expected_title": "Parallel Execution / MPI",
        "expected_remedy_keyword": "nprocs 1",
    },
    # --- GAUSSIAN FAILURES ---
    {
        "id": "gaussian_l502_scf",
        "engine": "gaussian",
        "log": """
        Cycle  128  Pass 1  IDCount 1
        E= -382.4901928340     Delta-E= -0.0000120938 Rises=NO
        Convergence failure -- run terminated.
        Error termination via Lnk1e in /opt/gaussian16/l502.exe
        """,
        "expected_title": "Link 502",
        "expected_remedy_keyword": "xqc",
    },
    {
        "id": "gaussian_l9999_opt",
        "engine": "gaussian",
        "log": """
        GradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGradGrad
        Maximum number of optimization cycles reached.
        -- Stationary point not found --
        Error termination via Lnk1e in /usr/local/g16/l9999.exe
        """,
        "expected_title": "Link 9999",
        "expected_remedy_keyword": "calcfc",
    },
    {
        "id": "gaussian_l103_linear_angle",
        "engine": "gaussian",
        "log": """
        FormBX had a problem. Linear bend in angle 4 - 2 - 8.
        Error in internal coordinate system.
        Error termination via Lnk1e in /software/g16/l103.exe
        """,
        "expected_title": "Link 103",
        "expected_remedy_keyword": "cartesian",
    },
    # --- PSI4 FAILURES ---
    {
        "id": "psi4_scf_fail",
        "engine": "psi4",
        "log": """
        @DF-RHF iter  99:   -154.2891048123   0.000194812   0.00192841 DIIS
        @DF-RHF iter 100:   -154.2891901428  -0.000085330   0.00201940 DIIS
        Psi4Exception: Could not converge SCF iterations in 100 iterations.
        """,
        "expected_title": "Psi4 SCF Convergence Failure",
        "expected_remedy_keyword": "soscf",
    },
    {
        "id": "psi4_optking_fail",
        "engine": "psi4",
        "log": """
        Optking: Optimization failed to converge! Maxiter exceeded in optking.
        """,
        "expected_title": "Optking",
        "expected_remedy_keyword": "geom_maxiter",
    },
    # --- PYSCF FAILURES ---
    {
        "id": "pyscf_scf_fail",
        "engine": "pyscf",
        "log": """
        cycle 50: E = -76.012398412  delta-E = -1.24e-05  norm(gorb) = 0.00194
        WARN: SCF not converged after 50 cycles.
        """,
        "expected_title": "PySCF SCF Convergence",
        "expected_remedy_keyword": "newton",
    },
    # --- XTB FAILURES ---
    {
        "id": "xtb_scc_fail",
        "engine": "xtb",
        "log": """
        iter: 200  E: -12.491024  dE: 0.000021  dq: 0.000491
        [ERROR] SCC not converged within 200 iterations!
        """,
        "expected_title": "xTB SCC Convergence",
        "expected_remedy_keyword": "iterations",
    },
    {
        "id": "xtb_distance_clash",
        "engine": "xtb",
        "log": """
        [ERROR] abnormal distance detected between atom 1 and atom 3 (d = 0.24 A).
        Atoms too close.
        """,
        "expected_title": "Unphysical Interatomic Distance in xTB",
        "expected_remedy_keyword": "check",
    },
]


@pytest.mark.parametrize("case", FAILURES_CORPUS, ids=[c["id"] for c in FAILURES_CORPUS])
def test_qc_doctor_corpus_diagnosis(case):
    """Test that QC Doctor reliably identifies root-cause and outputs actionable remedies."""
    result = diagnose_qm_error(case["log"], engine=case["engine"])
    assert result is not None, f"Failed to diagnose error case: {case['id']}"
    assert case["expected_title"].lower() in result.title.lower(), (
        f"Title mismatch for {case['id']}: expected '{case['expected_title']}' in '{result.title}'"
    )
    assert case["expected_remedy_keyword"].lower() in result.remedy.lower(), (
        f"Remedy mismatch for {case['id']}: expected keyword '{case['expected_remedy_keyword']}' in remedy:\n{result.remedy}"
    )
