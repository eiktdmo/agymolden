# AgyMolden: Modern Quantum Chemistry Workbench & Molecular Builder

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![OpenGL](https://img.shields.io/badge/Graphics-OpenGL%203.3+-orange.svg)](https://www.khronos.org/opengl/)
[![Tests](https://img.shields.io/badge/Tests-349%20Passing-brightgreen.svg)]()

**AgyMolden** is an ultra-fast, modern, native cross-platform quantum chemistry companion and molecular modeling workbench. It combines the intuitive internal coordinate (Z-matrix) editing workflows of classic Molden with a full-fledged computational chemistry environment: multi-package calculation execution, live convergence tracking, automated error diagnostics (QC Doctor), volumetric orbital & electrostatic potential (ESP) visualization, intrinsic reaction coordinate (IRC) reaction profiling, transition state IDPP path interpolation, and CREST conformer ensemble exploration.

Designed from the ground up with **pure Python / NumPy / SciPy / PyQt6 / PyOpenGL** — no bloated binary dependencies, instantaneous startup ($< 1$ s), and a lightweight one-click distribution bundle ($< 1.2$ MB).

---

## 🔬 Core Capabilities

### 1. Quantum Chemistry Calculation Runner & Monitor
- **Multi-Engine Execution**: Native integration with **Gaussian** (G16/G09), **ORCA** (4.x/5.x), **newBRORSEN** (multi-component quantum chemistry), **PySCF**, **openLOWDIN**, **xTB**, and **CREST**.
- **HPC Cluster Batch Script Generator**: 1-click generation and customization of production workload manager submission scripts (**SLURM**, **PBS / Torque**, **LSF**, **SGE**) with scratch directory lifecycles (`$SLURM_TMPDIR`), multi-threading (`$OMP_NUM_THREADS`), automatic module loading, and submission commands.
- **Local & Remote SSH Execution**: Launch jobs locally or stream output live from remote supercomputing clusters over asynchronous SSH with automatic remote directory syncing and artifact retrieval.
- **Live Convergence Tracking**: Real-time visual plots of SCF energy convergence, geometry optimization step energies, and gradient / force RMS and Max thresholds.
- **QC Doctor™ Automated Diagnostics**: Instant root-cause identification of calculation failures (SCF convergence divergence, oscillating line searches, Link 301/9999 errors, basis set deficiencies, memory faults, linear dependency) with actionable recovery recommendations.
- **Pre-Flight Calculation Validation**: Real-time electronic parity matching ($(N_e - (2S+1-1)) \pmod 2 == 0$), atomic clash detection ($< 0.45$ Å), hardware oversubscription detection, and engine-specific keyword linting before launching calculations.

### 2. Volumetric Molecular Orbitals & Real-Space Wavefunction Analysis
- **High-Performance 3D Isosurfaces**: GPU-accelerated marching cubes isosurface rendering for Molecular Orbitals (HOMO, LUMO, core, virtual) and total electron density from Molden format files.
- **Multiwfn Real-Space Grids & NCI/RDG Engine**: Real-space calculation and visualization of Electron Localization Function (ELF), Localized Orbital Locator (LOL), Electrostatic Potential (ESP), and Non-Covalent Interactions (NCI / Reduced Density Gradient) with standard 5-zone $\operatorname{sign}(\lambda_2)\rho$ continuous color mapping (Blue = strong attractive H-bonds, Green = vdW dispersion, Red = steric repulsion).
- **Electrostatic Potential (ESP) Mapping**: Dynamic symmetric color-normalized 3D ESP surfaces and 2D sliced planar contour maps with colorbar gradients and extrema identification ($V_{\min}$, $V_{\max}$).
- **Interactive MO Energy Ladder**: Visual energy diagram of frontier orbitals, orbital symmetry term symbols, Mulliken / Löwdin atomic orbital decomposition, and chemical reactivity metrics (HOMO–LUMO gap, chemical hardness $\eta$, electronegativity $\chi$).

### 3. Reaction Pathway & Transition State Tools
- **Interactive IRC Reaction Sweeper**: Minimum Energy Pathway (MEP) viewer from Reactant $\rightarrow$ Transition State (TS ‡) $\rightarrow$ Product with interactive energy cursor tracking, forward/reverse activation barrier calculation ($\Delta E^\ddagger_{\text{fwd}}$, $\Delta E^\ddagger_{\text{rev}}$), reaction enthalpy ($\Delta E_{\text{rxn}}$), trajectory movie scrubbing, and 1-click geometry adoption into the editor.
- **TS IDPP Path Interpolator**: Image-Dependent Pair Potential (IDPP) and Synchronous Transit linear interpolation between reactant and product geometries, generating smooth reaction paths without unphysical atomic collisions or steric overlaps.

### 4. CREST Conformer Ensemble Explorer
- **Conformer Ranking & Boltzmann Populations**: Parse multi-frame CREST conformer trajectories (`crest_conformers.xyz`, `crest.energies`) with automatic Boltzmann weighting at user-defined temperatures ($T = 298.15$ K).
- **Kabsch SVD RMSD Superposition**: Heavy-atom and all-atom RMSD alignment against the lowest-energy conformer or any selected reference structure.
- **Greedy RMSD Clustering**: Automatic structural deduplication and centroid identification.
- **1-Click DFT Batch Input Generator**: Automatically generate Gaussian or ORCA batch re-optimization input files for the lowest-energy conformers within an energy window (e.g., $\Delta E < 3.0$ kcal/mol).

### 5. Vibrational & Optical Spectroscopy
- **Harmonic IR Spectra**: Synthetic infrared spectra with Lorentzian broadening, peak picking, and automatic lookup of **NIST CCCBDB recommended frequency scaling factors** based on method and basis set.
- **Interactive 3D Normal Modes**: Smooth harmonic vibrational animation with real-time vector arrows, frequency scrubbing, and displacement amplitude control.
- **UV-Vis Electronic Absorption Spectra**: TD-DFT absorption spectra synthesized from excited state energies and oscillator strengths with Gaussian band broadening.

### 6. Natural Bond Orbital (NBO) Perturbation Analysis
- **Second-Order Donor–Acceptor Interactions**: Automatic parsing and interactive visualization of Fock matrix second-order perturbation energies ($E^{(2)}$ in kcal/mol).
- **Physical Classification**: Instant categorization of interactions into $n \rightarrow \pi^*$, $n \rightarrow \sigma^*$, $\pi \rightarrow \pi^*$, and hyperconjugation ($\sigma \rightarrow \sigma^*$).
- **Direct 3D Selection Sync**: Clicking any interaction row highlights the donor and acceptor atoms in the 3D viewport.
- **One-Click CSV & SI Markdown Export**: Export publication-ready interaction tables directly for Supporting Information.

### 7. Cross-Engine Benchmarking & SI Reproducibility
- **Cross-Engine Comparison Matrix (`Ctrl+Shift+M`)**: Side-by-side comparison of calculations across Gaussian, ORCA, PySCF, and xTB with relative energetics ($\Delta E$ in kcal/mol and eV), frontier orbital gap shifts, and pairwise Kabsch SVD structural RMSD matrices.
- **Multi-Stage Workflow Pipelines**: 1-click execution of standard quantum chemistry recipes (Opt $\rightarrow$ Freq guarantee, Opt $\rightarrow$ TD-DFT absorption spectra, TS Opt $\rightarrow$ Freq verification $\rightarrow$ IRC MEP).
- **Reproducibility Bundle Exporter (`Ctrl+Shift+S`)**: 1-click generation of Supporting Information `.zip` packages containing Cartesian `.xyz`, internal `.zmat`, machine-readable `summary.json`, formatted `COMPUTATIONAL_DETAILS.md`, and raw input/output logs.

### 8. True Molden-Style Z-Matrix & Cartesian Modeling
- **Interactive Z-Matrix Editor**: Spreadsheet-style table with immediate 3D visualization; "Add Line" referencing bond distances, angles, and dihedrals.
- **Real-Time Sliders & Spinners**: Adjust geometric parameters interactively with instantaneous 3D reorientation.
- **Shift + Click Geometric Measurements**: Instant Bond Distance (Å), Bond Angle (°), and Dihedral Angle (°) inspection.
- **Fragment & Substituent Library**: 1-click attachment of common functional groups (`-CH3`, `-OH`, `-NH2`, `-C6H5`, `-NO2`, `-COOH`, `-CN`, `-CF3`) and ring presets (benzene, cyclopentadiene, cyclohexane, etc.).
- **Chemical Name / SMILES / InChI 3D Resolver**: Instant conversion of IUPAC names, SMILES, or InChI strings into optimized 3D Cartesian coordinates via NIH Cactus and PubChem REST APIs.

### 9. Molecular Cycle Welder & Ring Fusion Studio (`Ctrl+Shift+W`)
- **Edge Annulation (Bond Welding)**: Select any bond/edge in the 3D viewport or right-click canvas to fuse an aromatic, saturated, or heterocyclic ring (Benzene, Pyridine, Pyrrole, Furan, Thiophene, Imidazole, Cyclohexyl, Cyclopentyl, Cyclobutyl, Cyclopropyl, Oxirane, or custom $N$-gon).
- **Peri-Annulation (3-Atom Bay Fusion)**: Completes open bay pockets across $C_{\text{left}} - C_{\text{bridgehead}} - C_{\text{right}}$ triads to construct Phenalene, Acenaphthene, and extended polyaromatics with zero steric clash and valency conservation.
- **Valency & Ortho-Hydrogen Pruning**: Automatically removes ortho-hydrogens on the fused bridgehead atoms to conserve proper valence, and generates ideal sp2/sp3 hydrogen geometries for the new ring.
- **Bicyclic Bridgehead Formation**: Connect distant ring carbons (e.g. 1,4-cyclohexane positions) with $-\text{CH}_2-$, $-\text{CH}_2-\text{CH}_2-$, $-\text{O}-$, $-\text{NH}-$, or $-\text{CO}-$ bridges to create Norbornanes, Bicyclo[2.2.2]octanes, or Camphor-like systems.
- **1-Click Analytical & Hybrid Force Field Relaxation**: Real-time analytical UFF / MMFF94 relaxation with optional parent-structure freezing and "Auto (Hybrid MMFF / UFF)" backend with organic parameterization.

### 10. Headless CLI Suite for HPC Clusters
Run AgyMolden headless on supercomputing clusters without an X11 display:
```bash
# Parse QM logs to terminal summary or JSON
agymolden parse calculation.log --json

# Generate production calculation inputs with pre-flight checks
agymolden gen molecule.xyz --code orca --job-type Opt --method B3LYP --basis def2-TZVP -o job.inp

# Run sanity and electron-spin parity checks
agymolden check molecule.xyz --charge 0 --mult 1

# Convert geometries between XYZ and internal Z-matrix
agymolden convert molecule.xyz -o molecule.zmat

# Export journal Supporting Information reproducibility bundle
agymolden bundle calculation.log -o SI_bundle.zip
```

---

## 🚀 Quick Start

### 1. Zero-Install One-Click Bundle (Recommended)
Download `AgyMolden_OneClick_Bundle.zip`, extract it anywhere, and launch:
- **Linux / macOS**: Double-click or run `./Run_Mac_or_Linux.sh`
- **Windows**: Double-click `Run_Windows.bat`

The bundle automatically bootstraps an isolated virtual environment and launches AgyMolden in seconds without modifying system packages.

### 2. Installation via Pip from Source
```bash
# Clone the repository
git clone https://github.com/andres/newagy.git
cd newagy

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install AgyMolden in editable mode with all features:
pip install -e ".[all]"

# Launch AgyMolden
agymolden
```

### 3. Quick Automated Setup Scripts
```bash
# Linux / macOS
./setup.sh
python run.py

# Windows
windows-installer.bat
python run.py
```

---

## ⌨️ Essential Keyboard & Mouse Shortcuts

| Action | Shortcut / Mouse Interaction |
| :--- | :--- |
| **Rotate 3D Canvas** | Left Click + Drag |
| **Pan Camera** | Right Click + Drag |
| **Zoom In / Out** | Mouse Wheel or Middle Click + Drag |
| **Select Atom** | Left Click on Atom (synchronizes with Z-Matrix table) |
| **Measure Geometry** | **Shift + Click** on 2 atoms (distance), 3 atoms (angle), 4 atoms (dihedral) |
| **Quick-Add Atom** | **Ctrl + Right-Click & Drag** from an atom to create a new bonded atom |
| **Quick Command Palette** | `Ctrl + K` or `Ctrl + Shift + P` (Search and trigger all 60+ tools & menus) |
| **Example Gallery** | `F2` (Browse and launch bundled Molden wavefunctions, QM logs, and templates) |
| **Open File** | `Ctrl + O` (Supports `.zmat`, `.xyz`, `.pdb`, `.sdf`, `.gjf`, `.inp`, `.out`, `.log`, `.molden`) |
| **Save File** | `Ctrl + S` |
| **Export High-DPI Publication PNG** | `Ctrl + E` / `Ctrl + P` (Physical 300/600 DPI with ACS/RSC/Nature journal presets) |
| **Export SI Reproducibility Bundle** | `Ctrl + Shift + S` (XYZ, ZMAT, Summary JSON, Methods Markdown, logs) |
| **Export Interactive 3D WebGL HTML** | `Ctrl + Shift + H` |
| **Copy 3D Image to Clipboard** | `Ctrl + Shift + C` (Transparent background, publication quality) |
| **Build from Name / SMILES** | `Ctrl + Shift + B` |
| **Quick Optimize Geometry** | `Ctrl + M` or `Ctrl + Shift + O` (UFF / GFN-FF / xTB) |
| **3D Electrostatic Potential (ESP) Map** | `Ctrl + Alt + P` (1-click ESP-mapped total electron density) |
| **Volumetric MO & Isosurfaces Studio** | `Ctrl + Alt + O` (GaussView/Avogadro style Marching Cubes) |
| **Weld / Fuse Molecular Rings** | `Ctrl + Shift + W` (Fuse carbocycles/heterocycles onto bonds or form bridgeheads) |
| **Open QM Calculation Dialog** | `Ctrl + R` |
| **Cross-Engine Benchmark Comparison**| `Ctrl + Shift + M` |
| **NBO Perturbation Analyzer** | `Ctrl + Shift + N` |
| **Python Script & Plugin Runner** | `Ctrl + Shift + Y` (Run custom scripts or user plugins in `~/.agymolden/plugins/`) |
| **Open IRC Reaction Sweeper** | `Ctrl + Shift + I` |
| **Open CREST Conformer Explorer** | `Ctrl + Shift + E` |
| **Open TS IDPP Path Interpolator** | `Ctrl + Shift + T` |
| **Reorder Z-Matrix (ZMEORD)** | `Ctrl + Shift + R` |
| **Toggle Dark / Light Theme** | `Ctrl + T` |
| **User Manual** | `F1` |

---

## 🧪 Supported File Formats & QM Packages

| Format / Software | Input Generation | Execution Runner | Log / Trajectory Parsing | Orbitals & Surfaces |
| :--- | :---: | :---: | :---: | :---: |
| **ORCA (4.x / 5.x / 6.x)** | ✅ Full (Compound, IRC, NumFreq, TD) | ✅ Streaming process | ✅ Output log, `.opt`, `.engrad`, `.hess` | ✅ Native `.molden.input` / `.molden` |
| **Gaussian (G16 / G09)** | ✅ Full (Opt, Freq, IRC, TD-DFT, TS) | ✅ Stdin / Pipe streaming | ✅ Energetics, forces, thermochemistry | ✅ Via `.molden` or `.fchk` conversion |
| **Psi4** | ✅ Cartesian & ZMAT input (`.dat`) | ✅ Streaming process | ✅ Energy, geometry, vibration parser | ✅ Molden wavefunction export |
| **PySCF** | ✅ Python driver scripts | ✅ Python process runner | ✅ Output logs | ✅ Native Molden export |
| **Q-Chem** | ✅ Rem / molecule blocks | ✅ Streaming process | ✅ Output log parsing via cclib | ✅ Molden / Fchk format |
| **newBRORSEN** | ✅ Nuclear-electronic APMO | ✅ Multi-stage runner | ✅ Electronic & positronic logs | ✅ Electronic & positronic Molden |
| **openLOWDIN** | ✅ Direct APMO input generation | ✅ Streaming process | ✅ Output log & properties | ✅ Native Molden support |
| **xTB & CREST** | ✅ GFN2-xTB / Conformer searches | ✅ Subprocess runner | ✅ Multi-frame XYZ & `crest.energies` | — |
| **Standard Formats** | `.zmat`, `.xyz`, `.pdb`, `.sdf`, `.mol`, `.cube` | Headless CLI suite | Multi-XYZ trajectories (`_trj.xyz`) | Gaussian Cube, Molden format |

---

## 🛡️ Running the Test Suite

AgyMolden features an extensive automated test suite covering unit tests, contract tests with real quantum chemistry binaries, GUI smoke tests, and numeric algorithms:

```bash
# Run the complete test suite
pytest -v

# Run calculation runner & pre-flight tests
pytest tests/test_calc_runner.py tests/test_preflight.py -v

# Run contract tests against installed Gaussian / ORCA binaries (auto-skipped if binaries are absent)
pytest tests/test_contract_binaries.py -v
```

---

## 📄 License
This project is licensed under the **MIT License**. See the `LICENSE` file for details.
