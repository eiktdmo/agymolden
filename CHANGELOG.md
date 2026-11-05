# Changelog

All notable changes to **AgyMolden** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-11-04

### Added
- **Molecular Cycle Welder & Ring Fusion Studio (`agymolden/cycle_welder.py`, `dialog_cycle_welder.py`, `Ctrl+Shift+W`)**:
  - Interactive edge-annulation tool for fusing carbo- and heterocycles (Benzene, Pyridine, Pyrrole, Furan, Thiophene, Imidazole, Cyclohexyl, Cyclopentyl, Cyclobutyl, Cyclopropyl, Oxirane, custom $N$-gons) directly onto selected bonds.
  - Automatically prunes ortho-hydrogens on the fused bridgehead carbons to maintain electronic octet and valency.
  - Bicyclic bridgehead inserter ($-\text{CH}_2-$, $-\text{CH}_2-\text{CH}_2-$, $-\text{O}-$, $-\text{NH}-$, $-\text{CO}-$) across distant ring positions to generate norbornane and bicyclo[2.2.2]octane frameworks.
  - Right-click canvas and 3D bond context menu binding: *"💍 Weld / Fuse Ring onto this Bond..."*.
  - Full undo/redo integration with analytical UFF/MMFF94 force field relaxation.
- **Extensible Python Scripting & Plugin Runner (`agymolden/plugins.py`, `Ctrl+Shift+Y`)**:
  - Research scripting hook to automate molecular manipulation, custom property calculations, geometry transformations, and custom file exporters.
  - Exposes `window`, `symbols`, `coords`, `zatoms`, `viewer`, `dialogs`, `set_coords(new_coords)`, and `set_molecule(symbols, coords)` with full undo/redo state tracking.
  - Automatically discovers user plugins in `~/.agymolden/plugins/*.py` with interactive code editor, open, and save dialogs.
- **Continuous Integration Matrix (`.github/workflows/ci.yml`)**:
  - Multi-OS GitHub Actions workflow testing on Linux, macOS, and Windows across Python 3.10, 3.11, and 3.12 with headless offscreen Qt (`QT_QPA_PLATFORM=offscreen`).
- **Natural Bond Orbital (NBO) Second-Order Perturbation Analyzer (`agymolden/nbo_analyzer.py`, `dialog_nbo.py`, `Ctrl+Shift+N`)**:
  - Automatically parses Fock matrix second-order perturbation theory tables from Gaussian and ORCA outputs.
  - Classifies donor-acceptor interactions into $n \rightarrow \pi^*$, $n \rightarrow \sigma^*$, $\pi \rightarrow \pi^*$, and hyperconjugation ($\sigma \rightarrow \sigma^*$).
  - Interactive threshold filtering, search, and direct 3D viewport selection synchronization (highlights donor/acceptor atoms).
  - 1-click export to CSV and formatted Supporting Information Markdown tables.
- **Supporting Information (SI) Reproducibility Bundle Exporter (`agymolden/reproducibility.py`, `Ctrl+Shift+S`, CLI `bundle`)**:
  - 1-click generation of standard publication Supporting Information `.zip` archives containing Cartesian `GEOMETRY.xyz`, internal `GEOMETRY.zmat`, machine-readable `CALCULATION_SUMMARY.json`, formatted `COMPUTATIONAL_DETAILS.md` (with copy-ready manuscript Methods section and coordinate tables), and original input/output files.
  - Headless CLI integration via `agymolden bundle <file> [-o bundle.zip] [--title TITLE]`.
- **Dialog & Headless Testing Service (`agymolden/dialog_service.py`)**:
  - Centralized decoupling of `QMessageBox` popups to enable frictionless headless execution and automated test mock verification.
- **Headless CLI Suite (`agymolden/cli.py`, `agymolden/__main__.py`)**:
  - `agymolden parse <logfile>`: Fast terminal summary and machine-readable `--json` extraction of energies, thermochemistry, frontier gap, frequencies, and termination status.
  - `agymolden gen <struct>`: Generates sanitized inputs for Gaussian, ORCA, PySCF, and newBRORSEN with automated pre-flight checks.
  - `agymolden check <file>`: Terminal-based sanity validation of coordinates, net charge/spin multiplicity parity, and memory limits.
  - `agymolden convert <input> -o <output>`: Headless bi-directional conversion between Cartesian XYZ and Z-Matrix formats.
- **Cross-Engine Benchmark & Result Comparison Dialog (`agymolden/dialog_benchmark.py`, `Ctrl+Shift+M`)**:
  - Side-by-side multi-job comparison across QM codes (Gaussian, ORCA, PySCF, openLOWDIN, xTB) and levels of theory.
  - Relative energetics ($\Delta E$ in kcal/mol and eV against lowest minimum), HOMO–LUMO gap, dipole moment, and optimization step comparison.
  - Full pairwise Kabsch SVD structural superposition matrix (All-atom and Heavy-atom RMSD).
  - 1-Click export to publication-ready CSV and Markdown comparison tables for Supporting Information.
  - 1-Click loading of selected benchmark geometry into 3D viewport.
- **Chained Multi-Stage Workflow Pipelines (`calc_runner.pipeline`, `calc_runner.dialog_pipeline`)**:
  - Sequential execution of multi-step recipes: `Opt → Freq (True Minimum Guarantee)`, `Opt → TD-DFT (Absorption Spectrum)`, and `TS Opt → Freq (1 Imag) → IRC`.
  - Automatic coordinate hand-off between optimization and subsequent property stages.
  - Mathematical verification of stationary point curvature (alerts user if imaginary frequencies appear on local minima).
- **Command Palette Arrow Key Navigation (`mw_ui.py`)**:
  - Smooth Up/Down arrow key scrolling from the search box directly through command palette results.
- **Pre-Flight Calculation Validation Engine & Dialog (`calc_runner.preflight`)**:
  - Validates molecular coordinates (detects unphysical atomic clashes $<0.45$ Å, NaN/Inf floating point anomalies).
  - Verifies electronic spin parity matching ($(N_e - (2S+1-1)) \pmod 2 == 0$) across neutral and charged species.
  - Checks system hardware limits against requested resources (system RAM vs `%mem`, logical CPU cores vs `%nprocshared` / `PAL`, free disk space in scratch directories).
  - Validates engine-specific keywords and constraints (e.g. minimum memory requirements, dispersion compatibility).
  - Rich Qt pre-flight modal dialog displaying electronic state summary badge (Singlet/Doublet/Triplet, $\alpha/\beta$ counts) with categorized Error (blocking), Warning, and Info panels.
- **Quantum Chemistry Contract Test Suite (`tests/test_contract_binaries.py`)**:
  - Live execution tests verifying parameterization against real installed Gaussian 16 and ORCA 4.2.1 binaries.
  - Comprehensive functional matrix (B3LYP, PBE0, M06-2X, wB97X-D) and basis sets (`def2-SVP`, `def2SVP`, `cc-pVDZ`, `6-31G(d)`).
  - Post-HF coupled cluster execution testing (`CCSD(T)/cc-pVDZ`).
  - Optimization + Frequency validation with Cartesian and Z-matrix coordinate representations.
  - Dynamic skipping when external binaries are unavailable in the test environment.
- **Interactive IRC Reaction Sweeper Dialog (`dialog_irc_sweeper.py`)**:
  - Visual Minimum Energy Path (MEP) viewer from Reactant $\rightarrow$ Transition State (TS ‡) $\rightarrow$ Product.
  - Interactive 2D energy profile with cursor scrub synchronization to the 3D viewport.
  - Direct thermodynamics readout: Forward activation barrier ($\Delta E^\ddagger_{\text{fwd}}$), Reverse activation barrier ($\Delta E^\ddagger_{\text{rev}}$), and Reaction enthalpy ($\Delta E_{\text{rxn}}$).
  - 1-Click geometry adoption into main Z-Matrix editor with full undo/redo support.
- **CREST Conformer Ensemble Explorer (`dialog_conformer_ensemble.py`, `crest_explorer.py`)**:
  - Conformer ranking and Boltzmann population distribution at customizable temperatures.
  - Kabsch SVD RMSD alignment against the global minimum or reference conformer.
  - Greedy RMSD clustering and duplicate pruning.
  - 1-Click batch DFT input generation for conformer re-optimization in Gaussian or ORCA.
- **Transition State IDPP Path Interpolator (`ts_interpolator.py`, `dialog_reaction_path.py`)**:
  - Image-Dependent Pair Potential (IDPP) and Synchronous Transit path interpolation between reactant and product geometries without steric clash singularities.
- **QC Doctor™ Automated Diagnostics Engine (`calc_runner.diagnostics`)**:
  - Pattern-based root-cause detection for SCF convergence divergence, oscillating line searches, Link 301/9999 errors, basis set deficiencies, and memory aborts.
  - Actionable recovery recommendations and 1-click input parameter modification.

### Changed
- **Gaussian Empirical Dispersion Normalization**:
  - Fixed Gaussian Link 301 `R6DS8: Unable to choose the S8 parameter` crash by omitting empirical dispersion keywords for pure wavefunction methods (`HF`, `MP2`, `CCSD(T)`).
  - Automatically normalized `M06-2X` dispersion to zero-damping `EmpiricalDispersion=GD3` instead of invalid `GD3BJ`.
- **Exception Handling & Audit**:
  - Completely eliminated unlogged, broad `except Exception: pass` and `except: pass` blocks across core modules (`viewer.py`, `mw_calculations.py`, `main_window.py`, `mw_editing.py`, `mw_file_io.py`, `crest_explorer.py`, `vibrations.py`, `chem_identifiers.py`, `remote_ssh.py`, `base.py`, `preflight.py`, `brorsen_builder/dialog.py`, `qm_log_parser.py`).
  - Replaced with granular exception filtering and informative `log.debug(...)` output to ensure complete observability during debug runs.
- **Bundle Packaging Automation (`scripts/make_oneclick_bundle.sh`)**:
  - Removed user-specific hardcoded paths in favor of dynamic environment variable inspection (`$VIRTUAL_ENV`, local `.venv`, standard `python3`).
- **Documentation Overhaul**:
  - Rewrote `README.md` to showcase the full multi-engine QM workbench, spectroscopy suite, reaction path tools, and volumetric visualization capabilities.

### Fixed
- Fixed hyphenated basis set name sanitation in Gaussian input generation (e.g., `def2SVP` vs `def2-SVP`).
- Fixed transparent orbital lobe OpenGL blending artifact where desktop background bled through transparent lobes.
- Fixed 5-membered ring preset planar coordinate generation and real-time aromaticity indicator badge.
