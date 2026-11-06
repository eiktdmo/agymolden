"""
Tests for quick element brush palette, fragment stamping, and LaTeX SI table generation.
"""

import numpy as np
import pytest
from agymolden.reproducibility import generate_si_latex


def test_generate_si_latex():
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ])
    summary = {
        "program": "ORCA",
        "method": "B3LYP",
        "basis": "def2-SVP",
        "final_energy_ha": -40.51829104,
        "zpe_ha": 0.044120,
        "gibbs_ha": -40.492104,
    }

    tex = generate_si_latex(symbols, coords, summary, title="Methane (CH4)")
    assert "\\begin{table}" in tex
    assert "\\begin{tabular}{lrrr}" in tex
    assert "Methane (CH4)" in tex
    assert "-40.51829104" in tex
    assert "0.044120" in tex
    assert "C &     0.000000" in tex
    assert "\\end{table}" in tex
