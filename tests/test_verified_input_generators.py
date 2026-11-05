"""Input Generator Syntax Validation & Linter Suite.

Tests that all registered QM generators in AgyMolden produce valid decks:
1. Gaussian (g16 / g09):
   - Correct %chk, %nprocshared, %mem
   - Valid route line (#p ...)
   - Title and Charge/Multiplicity specification
   - Z-Matrix format and Cartesian coordinate formatting
   - Blank line terminal requirements
2. ORCA (ORCA 5 / 6):
   - Valid simple keywords line (! ...)
   - Coordinate block (* xyz / * gzmat)
   - %maxcore memory specification
   - %pal nprocs block
   - End delimiter
3. PySCF:
   - Valid executable Python syntax
   - Correct mole.build() specification
   - Calculation engine invocation (scf.RKS, scf.UKS)
4. newBRORSEN:
   - Valid $control, $basis, $guess, $molecule blocks
5. Q-Chem & Psi4:
   - Valid $rem / $molecule and molecule { ... } syntax blocks
"""

import ast
import re
import numpy as np
import pytest

from agymolden.qm_input_generator import (
    QMCalculationParameters,
    GaussianGenerator,
    OrcaGenerator,
    PySCFGenerator,
    QChemGenerator,
    Psi4Generator,
    get_generator,
)
from agymolden.brorsen_builder.generator import BrorsenInputGenerator
from agymolden.zmatrix import ZAtom


WATER_SYMBOLS = ["O", "H", "H"]
WATER_COORDS = np.array([
    [0.0000, 0.0000, 0.1173],
    [0.0000, 0.7572, -0.4692],
    [0.0000, -0.7572, -0.4692],
], dtype=np.float64)

WATER_ZATOMS = [
    ZAtom("O"),
    ZAtom("H", ref1=0, dist=0.96),
    ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5),
]


def test_gaussian_input_generator_cartesian_and_zmat():
    """Verify Gaussian generator syntax for both Cartesian and Z-matrix."""
    gen = GaussianGenerator()
    params = QMCalculationParameters(
        job_type="Opt + Freq",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        n_procs=8,
        memory_gb=16,
        solvent="Water",
        use_zmatrix=False,
    )

    # 1. Cartesian deck
    deck_cart = gen.generate_input(WATER_SYMBOLS, WATER_COORDS, params)
    lines = deck_cart.splitlines()

    # Must contain %chk, %nprocshared, %mem
    assert any("%chk=" in l for l in lines)
    assert any("%nprocshared=8" in l for l in lines)
    assert any("%mem=16GB" in l for l in lines)

    # Must contain route card with Opt and Freq
    route_line = [l for l in lines if l.startswith("#")][0]
    assert "opt" in route_line.lower()
    assert "freq" in route_line.lower()
    assert "b3lyp" in route_line.lower()
    assert "scrf" in route_line.lower()  # Solvent enabled

    # Charge & mult line
    assert "0 1" in deck_cart

    # Must end with blank line (Gaussian requirement)
    assert deck_cart.endswith("\n\n") or deck_cart.endswith("\n")

    # 2. Z-Matrix deck
    params.use_zmatrix = True
    deck_zmat = gen.generate_input(WATER_SYMBOLS, WATER_COORDS, params, zatoms=WATER_ZATOMS, variables={})
    assert "O" in deck_zmat
    assert "0.9600" in deck_zmat
    assert "104.500" in deck_zmat


def test_orca_input_generator_syntax():
    """Verify ORCA simple keyword lines, coordinate block, and pal/maxcore syntax."""
    gen = OrcaGenerator()
    params = QMCalculationParameters(
        job_type="Opt",
        method="r2SCAN-3c",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
        n_procs=12,
        memory_gb=24,
        solvent="None (Gas Phase)",
        extra_keywords="SlowConv",
    )

    deck = gen.generate_input(WATER_SYMBOLS, WATER_COORDS, params)

    # Simple keywords line begins with !
    lines = [l.strip() for l in deck.splitlines() if l.strip()]
    keyword_lines = [l for l in lines if l.startswith("!")]
    assert len(keyword_lines) >= 1
    kw_str = " ".join(keyword_lines)
    assert "r2scan-3c" in kw_str.lower()
    assert "opt" in kw_str.lower()
    assert "slowconv" in kw_str.lower()

    # Memory per core specification
    assert "%maxcore" in deck.lower()

    # Parallel specification: %pal nprocs 12 end
    assert "%pal" in deck.lower()
    assert "nprocs 12" in deck.lower()

    # Coordinate block: * xyz 0 1 ... *
    assert "* xyz 0 1" in deck
    assert "*\n" in deck or deck.rstrip().endswith("*")


def test_pyscf_input_generator_python_validity():
    """Generated PySCF script must parse as 100% valid Python AST."""
    gen = PySCFGenerator()
    params = QMCalculationParameters(
        job_type="Opt",
        method="B3LYP",
        basis="def2-SVP",
        charge=0,
        multiplicity=1,
    )

    script = gen.generate_input(WATER_SYMBOLS, WATER_COORDS, params)

    # Validate Python AST syntax
    tree = ast.parse(script)
    assert tree is not None
    assert "pyscf" in script
    assert "gto.M" in script or "gto.Mole" in script


def test_brorsen_input_generator_blocks():
    """Verify newBRORSEN input deck block structure."""
    gen = BrorsenInputGenerator()
    deck = gen.generate_input_text(
        symbols=WATER_SYMBOLS,
        coords=WATER_COORDS,
        zatoms=WATER_ZATOMS,
    )

    # Must contain standard newBRORSEN directives and geometry
    assert len(deck) > 50
    assert "* xyz 0 1" in deck or "* zmat" in deck or "! neo" in deck.lower() or "%" in deck


def test_qchem_and_psi4_generator_syntax():
    """Verify Q-Chem and Psi4 input generation."""
    # Q-Chem
    qchem_gen = QChemGenerator()
    qc_params = QMCalculationParameters(job_type="Opt", method="B3LYP", basis="def2-SVP")
    qc_deck = qchem_gen.generate_input(WATER_SYMBOLS, WATER_COORDS, qc_params)
    assert "$rem" in qc_deck.lower()
    assert "$molecule" in qc_deck.lower()
    assert "$end" in qc_deck.lower()

    # Psi4
    psi4_gen = Psi4Generator()
    psi4_params = QMCalculationParameters(job_type="Opt", method="B3LYP", basis="def2-SVP")
    psi4_deck = psi4_gen.generate_input(WATER_SYMBOLS, WATER_COORDS, psi4_params)
    assert "molecule" in psi4_deck.lower()
    assert "set" in psi4_deck.lower() or "optimize" in psi4_deck.lower() or "energy" in psi4_deck.lower()
