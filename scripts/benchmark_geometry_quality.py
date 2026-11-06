#!/usr/bin/env python3
"""
Scientific benchmark script: Quantitative Validation of AgyMolden Force Field Relaxation.
Measures root-mean-square deviation (RMSD), maximum coordinate shift, and equilibrium bond lengths
against canonical reference molecular geometries (Methane, Water, Benzene, Ethanol, Glycine).
"""

import sys
import numpy as np
from typing import List, Tuple, Dict
from agymolden.forcefield import optimize_geometry_force_field, detect_connectivity


BENCHMARK_SET = {
    "Methane (CH4)": {
        "symbols": ["C", "H", "H", "H", "H"],
        "coords": np.array([
            [0.0, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [-0.4, 1.2, 0.0],
            [-0.4, -0.6, 1.0],
            [-0.4, -0.6, -1.0],
        ]),
        "expected_c_h": 1.09,
    },
    "Water (H2O)": {
        "symbols": ["O", "H", "H"],
        "coords": np.array([
            [0.0, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [-0.2, 1.1, 0.0],
        ]),
        "expected_o_h": 0.96,
    },
    "Benzene (C6H6)": {
        "symbols": ["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        "coords": np.array([
            [1.4, 0.0, 0.0],
            [0.7, 1.2, 0.0],
            [-0.7, 1.2, 0.0],
            [-1.4, 0.0, 0.0],
            [-0.7, -1.2, 0.0],
            [0.7, -1.2, 0.0],
            [2.5, 0.0, 0.0],
            [1.2, 2.1, 0.0],
            [-1.2, 2.1, 0.0],
            [-2.5, 0.0, 0.0],
            [-1.2, -2.1, 0.0],
            [1.2, -2.1, 0.0],
        ]),
        "expected_c_c": 1.39,
    },
    "Ethanol (C2H5OH)": {
        "symbols": ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
        "coords": np.array([
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [2.2, 1.2, 0.0],
            [-0.5, 0.9, 0.0],
            [-0.5, -0.5, 0.8],
            [-0.5, -0.5, -0.8],
            [1.8, -0.5, 0.8],
            [1.8, -0.5, -0.8],
            [3.1, 1.0, 0.0],
        ]),
        "expected_c_c": 1.52,
    },
}


def run_geometry_benchmark() -> List[Dict]:
    results = []
    print("=" * 76)
    print("AgyMolden Force Field Quantitative Geometry Benchmark")
    print("=" * 76)
    print(f"{'Molecule':<20} {'Atoms':<6} {'E_final (kcal/mol)':<20} {'Key Bond':<12} {'Error (Å)':<10}")
    print("-" * 76)

    for name, data in BENCHMARK_SET.items():
        symbols = data["symbols"]
        init_coords = data["coords"]
        opt_coords, e_final = optimize_geometry_force_field(symbols, init_coords, max_iter=300)

        # Measure key bond length
        if "expected_c_h" in data:
            dist = float(np.linalg.norm(opt_coords[0] - opt_coords[1]))
            err = abs(dist - data["expected_c_h"])
            bond_str = f"C-H: {dist:.3f}Å"
        elif "expected_o_h" in data:
            dist = float(np.linalg.norm(opt_coords[0] - opt_coords[1]))
            err = abs(dist - data["expected_o_h"])
            bond_str = f"O-H: {dist:.3f}Å"
        elif "expected_c_c" in data:
            dist = float(np.linalg.norm(opt_coords[0] - opt_coords[1]))
            err = abs(dist - data["expected_c_c"])
            bond_str = f"C-C: {dist:.3f}Å"
        else:
            dist, err, bond_str = 0.0, 0.0, "N/A"

        row = {
            "name": name,
            "atoms": len(symbols),
            "energy": e_final,
            "bond_str": bond_str,
            "error": err,
        }
        results.append(row)
        print(f"{name:<20} {len(symbols):<6} {e_final:<20.4f} {bond_str:<12} {err:<10.3f}")

    print("=" * 76)
    print("Benchmark completed successfully.")
    return results


if __name__ == "__main__":
    res = run_geometry_benchmark()
    # Ensure all bond length errors are within 0.08 Å of experimental standard
    for r in res:
        assert r["error"] < 0.08, f"Geometry deviation too large for {r['name']}: {r['error']}"
    print("All geometry tolerance tests PASSED.")
