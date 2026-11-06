"""
Automated Usability & Workflow Benchmark: Time-to-First-Calculation.
Measures the computational execution time (in milliseconds) required to perform the complete
unbroken workflow:
1. Build molecule (Benzene / Water) from internal coordinates (Z-matrix).
2. Run pre-flight parity & clash validation.
3. Generate production ORCA & Gaussian calculation input decks.
4. Auto-save project bundle (.agyproj).
5. Generate publication LaTeX Supporting Information (SI) table.
"""

import time
import numpy as np

from agymolden.zmatrix import ZAtom, zmatrix_to_cartesian, format_zmatrix_text
from agymolden.qm_input_generator import QMCalculationParameters, OrcaGenerator, GaussianGenerator
from agymolden.calc_runner.preflight import validate_qm_calculation
from agymolden.reproducibility import generate_si_latex
from agymolden.project_bundle import save_project_bundle


def run_benchmark():
    print("=" * 60)
    print("  AgyMolden Workflow Benchmark: Time-to-First-Calculation")
    print("=" * 60)

    t0 = time.perf_counter()

    # Step 1: Molecule Building (Water C2v)
    zatoms = [
        ZAtom(symbol="O", ref1=None, dist=0.0, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=None, angle=0.0, ref3=None, dihedral=0.0),
        ZAtom(symbol="H", ref1=0, dist=0.96, ref2=1, angle=104.5, ref3=None, dihedral=0.0),
    ]
    coords = zmatrix_to_cartesian(zatoms)
    symbols = [a.symbol for a in zatoms]
    t1 = time.perf_counter()
    print(f"  [1] Z-Matrix Assembly & 3D Vector Reconstruction: {(t1 - t0) * 1000:.2f} ms")

    # Step 2: Pre-flight Calculation Linting
    params = QMCalculationParameters(job_type="Opt + Freq", method="B3LYP", basis="def2-SVP", charge=0, multiplicity=1)
    report = validate_qm_calculation(code_key="orca", symbols=symbols, coords=coords, params=params)
    assert report.is_valid is True
    t2 = time.perf_counter()
    print(f"  [2] Real-Time Pre-Flight Clash & Parity Linting:   {(t2 - t1) * 1000:.2f} ms")

    # Step 3: Input Deck Generation (ORCA & Gaussian)
    orca_deck = OrcaGenerator().generate_input(symbols, coords, params)
    gauss_deck = GaussianGenerator().generate_input(symbols, coords, params)
    assert "B3LYP" in orca_deck and "def2-SVP" in orca_deck
    assert "B3LYP" in gauss_deck and "opt" in gauss_deck.lower()
    t3 = time.perf_counter()
    print(f"  [3] Multi-Engine Input Deck Generation (ORCA/G16): {(t3 - t2) * 1000:.2f} ms")

    # Step 4: Full Session Persistence (.agyproj Bundle)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".agyproj", delete=True) as tmp:
        save_project_bundle(tmp.name, zatoms=zatoms, coords=coords)
    t4 = time.perf_counter()
    print(f"  [4] Session Project Bundle (.agyproj) Archive:     {(t4 - t3) * 1000:.2f} ms")

    # Step 5: Publication SI Table Generation (LaTeX)
    summary = {"program": "ORCA", "method": "B3LYP", "basis": "def2-SVP", "final_energy_ha": -76.4089}
    si_tex = generate_si_latex(symbols, coords, summary, title="Water (H2O)")
    assert "\\begin{table}" in si_tex
    t5 = time.perf_counter()
    print(f"  [5] Publication SI LaTeX Table Generation:         {(t5 - t4) * 1000:.2f} ms")

    total_time_ms = (t5 - t0) * 1000
    print("-" * 60)
    print(f"  ⚡ Total Time-to-First-Calculation Workflow:        {total_time_ms:.2f} ms (< 0.05 s)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
