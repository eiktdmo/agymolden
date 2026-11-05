"""
Tests for Tier 1 Features:
1. Universal QM input generator (Gaussian, ORCA, PySCF, openLOWDIN, xTB, Q-Chem)
2. cclib-based QM log parser and trajectory player
3. Synthetic broadened IR spectrum calculation & widget
"""

import tempfile
import numpy as np
import pytest

from agymolden.qm_input_generator import (
    QM_GENERATORS,
    QMCalculationParameters,
    GaussianGenerator,
    OrcaGenerator,
    PySCFGenerator,
    OpenLOWDINGenerator,
    XTBGenerator,
    QChemGenerator,
    register_qm_generator,
    BaseQMGenerator,
)
from agymolden.qm_log_parser import parse_qm_logfile, can_parse_with_cclib
from agymolden.spectrum_widget import calculate_broadened_spectrum


def test_qm_input_generators():
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.09],
        [1.03, 0.0, -0.36],
        [-0.51, -0.89, -0.36],
        [-0.51, 0.89, -0.36],
    ])
    params = QMCalculationParameters(
        job_type="Opt + Freq",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        solvent="Water",
        n_procs=4,
        memory_gb=8,
    )

    # Test all registered generators
    for name, gen_cls in QM_GENERATORS.items():
        gen = gen_cls()
        out = gen.generate_input(symbols, coords, params)
        assert isinstance(out, str)
        assert len(out) > 30

    # Test Gaussian-specific output
    g_out = GaussianGenerator().generate_input(symbols, coords, params)
    assert "#p B3LYP/def2SVP opt freq" in g_out
    assert "%nprocshared=4" in g_out
    assert "scrf=(cpcm,solvent=water)" in g_out

    # Test ORCA-specific output
    o_out = OrcaGenerator().generate_input(symbols, coords, params)
    assert "! B3LYP def2-SVP" in o_out
    assert "D3BJ" in o_out
    assert "Opt Freq" in o_out
    assert "CPCM(Water)" in o_out
    assert "%pal nprocs 4 end" in o_out

    # Test PySCF Python script output
    p_out = PySCFGenerator().generate_input(symbols, coords, params)
    assert "from pyscf import gto, scf, dft" in p_out
    assert "mol.basis = \"def2-SVP\"" in p_out
    assert "dft.RKS(mol)" in p_out

    # Test openLOWDIN APMO input output
    low_out = OpenLOWDINGenerator().generate_input(symbols, coords, params)
    assert "GEOMETRY" in low_out
    assert "END GEOMETRY" in low_out
    assert "TASKS" in low_out
    assert 'method = "RKS"' in low_out
    assert "CONTROL" in low_out
    assert 'electronExchangeCorrelationFunctional = "B3LYP"' in low_out
    assert "OUTPUTS" in low_out
    assert "moldenFile" in low_out
    assert "dirac" in low_out

    # Test custom generator registration
    class MyCustomQM(BaseQMGenerator):
        code_name = "CustomQM"
        file_extension = ".cqm"
        def generate_input(self, syms, crds, p, zatoms=None, variables=None) -> str:
            return f"RUN {p.method} ON {len(syms)} ATOMS"

    register_qm_generator("CustomQM", MyCustomQM)
    assert "CustomQM" in QM_GENERATORS
    assert QM_GENERATORS["CustomQM"]().generate_input(symbols, coords, params) == "RUN B3LYP ON 5 ATOMS"


def test_broadened_spectrum_calculation():
    freqs = [500.0, 1500.0, 3000.0]
    intensities = [10.0, 50.0, 100.0]

    w, curve = calculate_broadened_spectrum(
        freqs, intensities, fwhm=20.0, w_min=400.0, w_max=3500.0, n_points=500, lineshape="Lorentzian"
    )
    assert len(w) == 500
    assert len(curve) == 500
    assert np.all(curve >= 0.0)

    # Peak near 3000 should be the highest
    idx_3000 = np.argmin(np.abs(w - 3000.0))
    idx_500 = np.argmin(np.abs(w - 500.0))
    assert curve[idx_3000] > curve[idx_500]


def test_qm_log_parsing():
    assert can_parse_with_cclib()

    gaussian_sample = """Gaussian, Inc.
 Entering Gaussian System
 #P B3LYP/6-31G*

 Standard orientation:                         
 ---------------------------------------------------------------------
 Center     Atomic      Atomic             Coordinates (Angstroms)
 Number     Number       Type             X           Y           Z
 ---------------------------------------------------------------------
      1          6           0        0.000000    0.000000    0.000000
      2          1           0        0.000000    0.000000    1.089000
      3          1           0        1.026719    0.000000   -0.363000
      4          1           0       -0.513360   -0.889165   -0.363000
      5          1           0       -0.513360    0.889165   -0.363000
 ---------------------------------------------------------------------
 Harmonic frequencies (cm**-1), IR intensities (KM/Mole)
 Frequencies --   1345.1200              1500.5000              3000.2000
 IR Inten    --     32.5000                15.2000               120.0000
 Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z
   1   6      0.00   0.00   0.00     0.00   0.00   0.00     0.00   0.00   0.00
   2   1      0.00   0.00   0.50     0.00   0.50   0.00     0.50   0.00   0.00
   3   1      0.50   0.00  -0.17     0.00  -0.25  -0.43    -0.25  -0.43   0.00
   4   1     -0.25  -0.43  -0.17    -0.43   0.25   0.00     0.25  -0.43   0.00
   5   1     -0.25   0.43  -0.17     0.43   0.25   0.00     0.25   0.43   0.00
 Normal termination of Gaussian 09.
"""
    with tempfile.NamedTemporaryFile("w", suffix=".log", delete=False) as f:
        f.write(gaussian_sample)
        log_path = f.name

    result = parse_qm_logfile(log_path)
    assert result.symbols == ["C", "H", "H", "H", "H"]
    assert len(result.trajectory_coords) == 1
    assert result.vibrational_analysis is not None
    assert len(result.vibrational_analysis.frequencies) == 3
    assert result.vibrational_analysis.frequencies[0] == pytest.approx(1345.12, rel=1e-3)
    assert result.vibrational_analysis.ir_intensities is not None
    assert result.vibrational_analysis.ir_intensities[0] == pytest.approx(32.5, rel=1e-2)
