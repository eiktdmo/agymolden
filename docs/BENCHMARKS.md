# AgyMolden Empirical Benchmarks

This document records reproducible benchmarks evaluating AgyMolden's geometry optimization quality and workflow ergonomics against standard computational chemistry criteria.

---

## 1. Force Field Geometry Optimization Accuracy

AgyMolden's internal coordinate Cartesian solver and hybrid force field (`uff_mmff`) were tested against standard canonical gas-phase equilibrium bond lengths across representative organic and inorganic systems.

**Benchmark script**: [`scripts/benchmark_geometry_quality.py`](file:///home/andres/Desktop/newagy/scripts/benchmark_geometry_quality.py)  
**Tolerances**: Maximum allowable key bond length deviation $< 0.08$ Å from experimental reference.

| Molecule | Atoms | Final Force Field Energy | Measured Key Bond | Reference Bond Length | Absolute Error ($\Delta r$) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Methane ($\text{CH}_4$)** | 5 | 0.0000 kcal/mol | C–H: 1.095 Å | 1.090 Å | 0.005 Å | **PASSED** |
| **Water ($\text{H}_2\text{O}$)** | 3 | 0.0000 kcal/mol | O–H: 0.960 Å | 0.960 Å | 0.000 Å | **PASSED** |
| **Benzene ($\text{C}_6\text{H}_6$)** | 12 | -0.1673 kcal/mol | C–C: 1.395 Å | 1.390 Å | 0.005 Å | **PASSED** |
| **Ethanol ($\text{C}_2\text{H}_5\text{OH}$)** | 9 | 1.7495 kcal/mol | C–C: 1.536 Å | 1.520 Å | 0.016 Å | **PASSED** |

*Conclusion*: Across all target molecules, AgyMolden's optimizer converges rapidly to physically accurate covalent equilibria with mean absolute error $< 0.01$ Å.

---

## 2. Usability & Time-to-First-Calculation Benchmark

Simulated comparative human interaction steps required to construct a molecule from scratch, configure a density functional theory (DFT) calculation deck, launch it, and monitor results.

**Benchmark script**: [`scripts/benchmark_time_to_calc.py`](file:///home/andres/Desktop/newagy/scripts/benchmark_time_to_calc.py)

| Metric | Classic Fortran Molden | Avogadro 2 | GaussView 6 | AgyMolden v1.1 |
| :--- | :---: | :---: | :---: | :---: |
| **Drawing & Z-Matrix Sync** | Manual Fortran Cards | Cartesians only | Mouse click/drag | Live Bidirectional Sync |
| **Pre-Flight Clash Detection** | None | Manual Inspect | None | Instant HUD Badge (`🟢 Ready` / `⚠️ Clash`) |
| **Quick Element Brush HUD** | No | Sidebar menu | Sub-window picker | 1-Click Ribbon (C, H, O, N, P, S, F, Cl) |
| **QC Doctor™ Diagnostic** | No | No | No | Automated Pattern Match + 1-Click Auto-Fix |
| **Command Palette (`Ctrl+K`)**| No | No | No | Instant Fuzzy Navigation (35+ tools) |
| **Time to First Deck (estimated)** | ~180 s | ~45 s | ~40 s | **~15 s** |
