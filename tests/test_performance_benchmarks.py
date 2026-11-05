"""Performance & Scalability Stress Benchmarks for AgyMolden.

Validates that connectivity detection, 3D instance generation, and volumetric grid
evaluation scale within sub-second interactive budgets across 500, 1000, and 5000 atoms.
"""

import time
import pytest
import numpy as np
from agymolden.forcefield import detect_connectivity


@pytest.mark.benchmark
def test_connectivity_scaling_500_atoms():
    """500 atoms (e.g. large dendrimer, small protein) must detect bonds under 50 ms."""
    symbols = ["C", "C", "H", "H", "O"] * 100
    np.random.seed(42)
    coords = np.random.uniform(-12, 12, (500, 3))

    t0 = time.perf_counter()
    bonds = detect_connectivity(symbols, coords)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(bonds) > 0
    assert elapsed_ms < 50.0, f"500-atom connectivity took {elapsed_ms:.2f} ms (budget: 50 ms)"


@pytest.mark.benchmark
def test_connectivity_scaling_1000_atoms():
    """1000 atoms (e.g. MOF, carbon nanotube) must detect bonds under 80 ms."""
    symbols = ["C", "C", "H", "H"] * 250
    np.random.seed(42)
    coords = np.random.uniform(-18, 18, (1000, 3))

    t0 = time.perf_counter()
    bonds = detect_connectivity(symbols, coords)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(bonds) > 0
    assert elapsed_ms < 80.0, f"1000-atom connectivity took {elapsed_ms:.2f} ms (budget: 80 ms)"


@pytest.mark.benchmark
def test_connectivity_scaling_5000_atoms():
    """5000 atoms (macromolecule/crystal supercell) must detect bonds under 250 ms (was ~11,000 ms)."""
    symbols = ["C", "C", "H", "H", "N"] * 1000
    np.random.seed(42)
    coords = np.random.uniform(-30, 30, (5000, 3))

    t0 = time.perf_counter()
    bonds = detect_connectivity(symbols, coords)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(bonds) > 0
    assert elapsed_ms < 250.0, f"5000-atom connectivity took {elapsed_ms:.2f} ms (budget: 250 ms)"


@pytest.mark.benchmark
def test_volumetric_cube_grid_marching_cubes_latency():
    """Marching cubes isosurface meshing on 60x60x60 (216,000 voxels) grid must complete under 150 ms."""
    from skimage.measure import marching_cubes

    # Synthetic orbital wave: spherical harmonic Gaussian envelope
    N = 60
    x = np.linspace(-4, 4, N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    R = np.sqrt(X**2 + Y**2 + Z**2)
    # pz-like orbital lobe: z * exp(-r)
    grid = Z * np.exp(-R)

    t0 = time.perf_counter()
    verts, faces, normals, values = marching_cubes(grid, level=0.04)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(verts) > 0
    assert len(faces) > 0
    assert elapsed_ms < 150.0, f"Marching cubes on 60^3 grid took {elapsed_ms:.2f} ms (budget: 150 ms)"
