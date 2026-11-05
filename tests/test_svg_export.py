"""Tests for High-DPI Publication Graphics & Vector SVG Exporter."""

import os
import tempfile
import numpy as np
import pytest

from agymolden.svg_exporter import export_molecule_svg, ELEMENT_COLORS
from agymolden.forcefield import detect_connectivity


def test_export_molecule_svg_water():
    symbols = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.757, 0.586],
        [0.0, -0.757, 0.586],
    ])
    bonds = [(0, 1), (0, 2)]

    svg_str = export_molecule_svg(symbols, coords, bonds=bonds, width=400, height=300)

    assert "<svg" in svg_str
    assert "</svg>" in svg_str
    assert 'viewBox="0 0 400 300"' in svg_str
    # Must contain atom circles and bond lines
    assert "<circle" in svg_str
    assert "<line" in svg_str
    assert ">O</text>" in svg_str
    assert ">H</text>" in svg_str


def test_export_molecule_svg_benzene_publication_white():
    symbols = ["C"] * 6 + ["H"] * 6
    theta = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    coords = np.zeros((12, 3))
    coords[:6, 0] = 1.39 * np.cos(theta)
    coords[:6, 1] = 1.39 * np.sin(theta)
    coords[6:, 0] = 2.47 * np.cos(theta)
    coords[6:, 1] = 2.47 * np.sin(theta)

    bonds = detect_connectivity(symbols, coords)

    svg_str = export_molecule_svg(
        symbols,
        coords,
        bonds=bonds,
        width=800,
        height=800,
        background="#ffffff",
        dark_mode=False,
    )

    assert 'fill="#ffffff"' in svg_str
    assert svg_str.count("<circle") == 12
    assert svg_str.count("<line") == len(bonds)


def test_export_molecule_svg_empty_safe():
    symbols = []
    coords = np.zeros((0, 3))
    svg_str = export_molecule_svg(symbols, coords)
    assert "<svg" in svg_str
    assert "</svg>" in svg_str


def test_export_molecule_svg_file_io(tmp_path):
    symbols = ["C", "C"]
    coords = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]])
    bonds = [(0, 1)]

    svg_data = export_molecule_svg(symbols, coords, bonds=bonds)
    svg_file = tmp_path / "ethane.svg"
    svg_file.write_text(svg_data, encoding="utf-8")

    assert svg_file.exists()
    assert svg_file.stat().st_size > 100
    content = svg_file.read_text(encoding="utf-8")
    assert content.startswith("<svg")
