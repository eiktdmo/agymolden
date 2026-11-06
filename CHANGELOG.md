# Changelog

All notable changes to **AgyMolden** are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-11-06

### Added
- **QC Doctor™ Real Failure Diagnostic Corpus & 1-Click Auto-Fix & Relaunch (`agymolden/calc_runner/diagnostics.py`, `dialog_job_monitor.py`, `tests/test_qc_doctor_corpus.py`, `test_qc_doctor_autofix.py`)**:
  - Expanded automated root-cause failure diagnostics across ORCA, Gaussian, Psi4, PySCF, and xTB (SCF non-convergence, trust-radius collapse, basis set linear dependency, Link 9999/502 errors).
  - 1-Click `🩹 Auto-Fix & Relaunch` button in `JobMonitorDialog` that safely parses and injects recovery keywords directly into input decks and restarts the calculation immediately.
- **AgyMolden Project Bundles (`.agyproj`) (`agymolden/project_bundle.py`, `mw_file_io.py`, `tests/test_project_bundle.py`)**:
  - Full portable ZIP/JSON session persistence preserving 3D Cartesian coordinates, internal coordinates (Z-matrix), variables, calculation history, and viewer states.
- **Searchable Command Palette (`Ctrl+K`) (`agymolden/dialog_command_palette.py`, `mw_ui.py`)**:
  - Fuzzy-searchable overlay providing instant keyboard access to all calculation engines, analytical tools, builder modes, and export formats.
- **Interactive Element Brush & Stamp HUD Ribbon (`agymolden/mw_ui.py`, `mw_editing.py`)**:
  - Quick CPK-colored element buttons (**C**, **H**, **O**, **N**, **P**, **S**, **F**, **Cl**) and 1-click functional group / ring stamp menu (**-Phenyl**, **-Methyl**, **-OH**, **-NH2**, **-C=O**, **-NO2**, **-COOH**, **Water**).
- **Live Empirical Molecular Formula & Weight HUD Pill (`agymolden/viewer.py`, `elements.py`)**:
  - Translucent 3D viewport overlay displaying IUPAC Hill-system formula and exact molecular mass in real time during drawing.
- **Pre-Flight Calculation Readiness Badge (`agymolden/mw_ui.py`, `main_window.py`)**:
  - Reactive toolbar badge displaying calculation health (`🟢 Ready`, `⚪ Empty`, `⚠️ Clash`) with automatic nuclear clash detection (< 0.70 Å).
- **Crystallographic CIF & Gaussian Cube Exporter (`agymolden/zmatrix.py`, `dialog_volumetric.py`, `tests/test_cif_and_cube_io.py`)**:
  - Bidirectional CIF export projecting Cartesian coordinates to fractional lattice space via crystallographic metric tensor.
  - Direct 1-click Gaussian `.cube` export for orbitals, total densities, and difference grids ($\Delta \rho = A - B$).
- **Automated LaTeX Supporting Information (SI) Table Export (`agymolden/reproducibility.py`)**:
  - Generates ACS/RSC publication-ready `\begin{table}` LaTeX snippets with energies, zero-point corrections, and coordinates.
- **Session Auto-Save & Crash Recovery (`agymolden/main_window.py`)**:
  - Background 60-second periodic heartbeat saving to `.agyproj` recovery file with startup restoration prompt after abnormal exits.
- **Sequential Batch Calculation Queue (`agymolden/calc_runner/dialog_job_monitor.py`)**:
  - Dedicated `📋 Batch Queue` tab in the job monitor to queue and sequentially execute multiple `.inp` decks across different engines.
- **Opt-In Modern Impostor Shaders (`agymolden/viewer_meshes.py`, `viewer.py`, `dialog_preferences.py`)**:
  - ChimeraX/Avogadro 2 style dual-source Blinn-Phong glossy shading with specular and Fresnel rim highlights, isolated safely behind a preferences toggle.

## [0.7.0] - 2026-11-05

### Added
- **Native Analytical PySCF QTAIM (Quantum Theory of Atoms in Molecules) Bader Topology Engine (`agymolden/multiwfn_analysis.py`, `dialog_multiwfn.py`, `viewer.py`)**:
  - Exact analytical GTO evaluation of electron density $\rho(\mathbf{r})$, gradient $\nabla\rho(\mathbf{r})$, and Hessian matrix $\mathcal{H}(\mathbf{r})$ via PySCF without finite-difference grid noise.
  - Newton-Raphson critical point search with trust-region damping for $(3,-3)$ Nuclear Attractors, $(3,-1)$ Bond Critical Points (BCP), and $(3,+1)$ Ring Critical Points (RCP).
  - Analytical computation of Laplacian $\nabla^2\rho$, Hessian eigenvalues $(\lambda_1 \le \lambda_2 \le \lambda_3)$, bond ellipticity $\epsilon = (\lambda_1 / \lambda_2) - 1$, Lagrangian kinetic energy density $G(\mathbf{r})$, potential energy density $V(\mathbf{r})$, total energy density $H(\mathbf{r}) = G(\mathbf{r}) + V(\mathbf{r})$, and bond character classification (Shared/Covalent, Polar Covalent, Closed-Shell).
  - Bader steepest-ascent gradient bond path trajectory tracing from BCPs to nuclei.
  - 3D Viewport rendering of $(3,-1)$ BCP spheres (vibrant cyan) and continuous gradient bond path cylinders (vibrant emerald).
  - Seamless auto-fallback: Multiwfn is utilized if installed, otherwise high-precision native PySCF analytical solver executes automatically with zero external dependencies.
- **Periodic Boundary Conditions (PBC) & Supercell Engine (`agymolden/pbc.py`, `dialog_pbc.py`, `mw_file_io.py`, `viewer.py`)**:
  - `UnitCell` representation supporting triclinic, monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, and cubic lattices with lengths $(a, b, c)$ and angles $(\alpha, \beta, \gamma)$.
  - Analytical computation of Cartesian lattice vectors $\mathbf{a}, \mathbf{b}, \mathbf{c}$, reciprocal lattice vectors $\mathbf{a}^*, \mathbf{b}^*, \mathbf{c}^*$, crystallographic metric tensor $G$, and cell volume $V$.
  - Fractional $\leftrightarrow$ Cartesian coordinate conversion and boundary wrapping.
  - Interactive 3D wireframe bounding box rendering in the viewport with high-contrast color-coded axes ($\mathbf{a}$ red, $\mathbf{b}$ green, $\mathbf{c}$ blue) and dashed perimeter edges.
  - $N_a \times N_b \times N_c$ supercell replication generator with KDTree boundary duplicate detection and elimination.
  - Direct integration into CIF crystal structure loading and dedicated dialog (`dialog_pbc.py`).
- **Visual Workflow Directed Acyclic Graph (DAG) Pipeline Editor (`agymolden/dialog_workflow_dag.py`, `mw_calculations.py`, `mw_ui.py`, `Ctrl+Shift+D`)**:
  - Interactive visual DAG flowchart canvas rendering stage nodes with status badges (Pending, Running, Succeeded, Failed), connection arrows, and inspector selection.
  - Built-in multi-stage recipe templates:
    - *Opt → Freq (Minimum Guarantee)*
    - *Opt → Freq → TD-DFT (UV-Vis Spectrum)*
    - *TS Opt → Freq (1 Imaginary Mode) → IRC (Reaction Path)*
    - *DFT Opt → Freq → DLPNO-CCSD(T) Benchmark Single Point*
  - Add, delete, and reorder stages (`Move Up` / `Move Down`) with per-stage parameter overrides (Method, Basis, Solvent, Extra Keywords).
  - Recipe persistence: Save and load workflow recipes as structured JSON files.
  - Live execution bridge to `ChainedWorkflowPipeline` with real-time card status updates, live log stream, and 1-click adoption of final optimized geometry into the 3D editor.
- **Interactive Onboarding Tour & First-Launch Wizard (`agymolden/dialog_onboarding.py`, `mw_ui.py`)**:
  - 4-step guided walkthrough of core AgyMolden features with 1-click sample loaders ($H_2O$ and Benzene).
- **Extensible Plugin API v1 (`agymolden/plugin_api.py`, `tests/test_plugins.py`)**:
  - Lifecycle hooks (`on_load`, `run`, `on_unload`), `PluginContext`, and bundled plugins (Radius of Gyration, Principal Axes Alignment).
- **128× Connectivity Detection Acceleration (`agymolden/forcefield.py`)**:
  - Upgraded pairwise distance detection to $O(N \log N)$ `scipy.spatial.cKDTree` spatial indexing, accelerating 5,000-atom connectivity detection from 10.9s to 84.9ms.
- **Modularized UI Dialog Architecture (`agymolden/dialog_steric_descriptors.py`, `dialog_ring_aromaticity.py`, `dialog_isotopologue.py`, `dialog_symmetry_matrices.py`)**:
  - Decoupled major analysis dialogs from large mixin files into dedicated modules.
- **Standalone PyInstaller Build Bundle (`agymolden.spec`)**:
  - Verified standalone executable generation with headless offscreen 3D rendering verification.

## [0.6.0] - 2026-11-05

### Added
- **Psi4 Quantum Chemical Engine Support (`agymolden/qm_input_generator.py`, `qm_detect.py`, `calc_runner/generic_runner.py`)**:
  - Full input deck generator (`Psi4Generator`) supporting Cartesian and Z-matrix coordinate syntax, memory allocation, multi-threaded scaling, and automatic reference selection (`rhf` / `rohf`).
  - Polarizable Continuum Model (PCM) solvation setup and job dispatch (`optimize`, `frequency`, `energy`).
  - Automated executable detection in standard Conda environments and system paths.
  - Streaming execution runner integration in `GenericQMRunner`.
- **Publication-Quality Physical DPI Tagging (`agymolden/mw_file_io.py`)**:
  - Implemented exact physical DPI metadata embedding via `setDotsPerMeterX/Y` (300 and 600 DPI) for drag-and-drop scaling into LaTeX, Word, and InDesign.
  - Journal layout presets: ACS Single Column (3.25 in / 8.25 cm), ACS Double Column (7.0 in), RSC Single Column (8.3 cm), Nature/Springer (89 mm), and TOC Graphic square.
- **Vector Spectrum Plot Exporter (`agymolden/spectrum_widget.py`)**:
  - 1-Click vector figure export (`.pdf`, `.svg`, `.eps`) and high-DPI raster (`.png`) across all spectra (IR/Raman, UV-Vis, NMR Chemical Shifts, and Electronic Density of States).
  - Customizable print background styling (pure white with high-contrast text and dark spines, transparent, or dark canvas).
- **Universal PDB and SDF/MOL Molecular File Formats (`agymolden/zmatrix.py`, `mw_file_io.py`)**:
  - Lossless parsers and writers for Protein Data Bank (`.pdb`) and MDL Molfile / SDF V2000 (`.sdf`, `.mol`).
  - Preserves exact 3D coordinates, atom serials, explicit `CONECT` bonds, and multiple bond orders (single, double, triple).
- **Multiwfn Real-Space 3D Grids & NCI/RDG Studio (`agymolden/multiwfn_analysis.py`, `dialog_multiwfn.py`)**:
  - Automated 3D volumetric grid calculation for Electron Localization Function (ELF), Localized Orbital Locator (LOL), Total Electron Density, Laplacian ($\nabla^2\rho$), and Reduced Density Gradient (RDG / Non-Covalent Interactions).
  - Pure NumPy/SciPy promolecular density fallback when Multiwfn is not installed.
- **1-Click Reaction Transition State Workflow Pipeline (`agymolden/dialog_reaction_path.py`)**:
  - Direct bridge connecting IDPP reaction path interpolation into the multi-stage calculation runner pre-configured with `TS Opt → Freq (Verify 1 Imaginary) → IRC MEP`.
- **Scientific Trust & Mathematical Validation Corpus (`tests/test_numerical_validation.py`, `test_parser_golden.py`)**:
  - Mathematical verification of Kabsch SVD rotation recovery ($\text{RMSD} < 10^{-12}$ Å, $\det(R)=1.0$).
  - Cartesian to Z-Matrix machine-precision roundtrip ($\text{RMSD} = 6.6 \times 10^{-16}$ Å).
  - Golden parser test corpus validating ORCA, Gaussian, and newBRORSEN multi-stage logfiles.

### Changed
- Replaced broad `except Exception` blocks in calculation runners and file I/O operations with typed exceptions (`OSError`, `IOError`, `UnicodeDecodeError`, `SubprocessError`) and structured logging.
- Hardened chemical identifier resolution with offline 3D coordinate templates for standard molecules.

## [0.5.0] - 2026-11-04

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
