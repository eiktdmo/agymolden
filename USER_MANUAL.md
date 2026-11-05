# AgyMolden User Manual & Comparative Evaluation

**Version:** 1.0 (Modern Molden-Inspired Z-Matrix Generator & Molecular Modeling Suite)  
**Platform:** Linux (X11 / Wayland via PyQt6 & Modern OpenGL)  
**Author:** AgyMolden Development Team  

---

## 1. Executive Summary & Overview

**AgyMolden** is a modern molecular editor and Z-matrix builder created to bridge the gap between classical internal-coordinate editors (specifically the classic Fortran/X11 **Molden**) and contemporary interactive chemical sketchers (**Avogadro**, **Gabedit**, and **GaussView**).

While Cartesian coordinate ($XYZ$) sketchers are intuitive for drawing 3D geometry with a mouse, quantum chemistry calculations (Gaussian, GAMESS, ORCA, Molpro, Q-Chem, NWChem) frequently demand or benefit significantly from **internal coordinates (Z-matrix)**:
- Enforcing exact point-group symmetry constraints.
- Freezing specific bond distances, valence angles, or dihedral torsions during partial potential energy surface (PES) scans.
- Describing flexible polymers, biomolecules, and reaction coordinates along intrinsic reaction coordinates (IRC).

AgyMolden preserves the rigorous reference-tree workflow of Molden (`ref1`, `ref2`, `ref3`) while providing an intuitive, interactive OpenGL viewport with Avogadro-style fluid mechanics, automated valence management, force field geometry optimization, harmonic vibrational analysis, and full point group symmetry detection and projection.

---

## 2. Architecture & Core Features of AgyMolden

### 2.1 Dual-Synchronized Representation
AgyMolden maintains bidirectional synchronization across three representations of the molecule:
1. **Interactive 3D OpenGL Viewport**: Ball-and-stick and space-fill (CPK) rendering with mouse picking, translation, elastic bond dragging, and multi-cylinder bond orders.
2. **Spreadsheet Z-Matrix Table (`table_view.py`)**: Real-time table displaying Atom Index, Element Symbol, Bond Reference (`ref1`), Bond Distance ($r$), Angle Reference (`ref2`), Valence Angle ($\theta$), Dihedral Reference (`ref3`), and Torsion ($\phi$). Edits in the table update the 3D model instantly; dragging in 3D updates the table.
3. **Raw ZMAT Text View**: Direct text editor supporting standard Molden and Gaussian format Z-matrices with variable blocks (`Variables:` / `Constants:`).

### 2.2 Molecular Construction & Drawing Methods
- **Double-Click Canvas**: Opens the modern interactive Periodic Table dialog to choose an element and initialize or append atoms.
- **Double-Click Atom**: Replaces hydrogen with the last selected element or extends an open valence along the natural tetrahedral/trigonal geometric vector.
- **Elastic Left-Click Drag from Atom (Avogadro-style)**: Left-clicking an existing atom and dragging into space draws an elastic dashed guide; releasing spawns a new bonded atom along that vector at the covalent equilibrium distance.
- **Ctrl + Left-Click Drag (Move / Translate)**: Translates the clicked atom or the entire multi-selected group rigidly within the 3D camera plane, updating internal coordinates live.
- **Bond Order Adjustment (Double-Click Bond)**:
  - Double-clicking a bond cycles its order ($1 \to 2 \to 3 \to 1$).
  - **Valence Guard**: Monovalent atoms ($\text{H}, \text{F}, \text{Cl}, \text{Br}, \text{I}$) are strictly prevented from exceeding single bond order ($BO = 1$).
  - Chemical valence limits ($\min(\text{val}_1, \text{val}_2)$ minus other heavy bond orders) are strictly enforced.
  - Automatically contracts or expands the equilibrium bond distance and triggers local valence adjustment.
- **Right-Click Atom**: Instantly deletes the targeted atom and re-indexes the Z-matrix.
- **Molden-Style Add Line**: Step-by-step interactive 3D viewport picking for `ref1`, `ref2`, and `ref3`.

### 2.3 Automated Valence & Hydrogen Handling
- **"Adjust Hydrogens" Toggle (`chk_auto_h`)**: When active, newly placed bonds or changed bond orders automatically prune excess hydrogens or spawn geometric hydrogens.
- **Selective vs. Whole-Molecule "Add Hydrogens" (`Ctrl+H`)**:
  - When specific atoms are selected in the viewport or table: only those selected atoms have their valences satisfied with hydrogens.
  - When no atoms are selected: the entire molecular structure is populated with geometric hydrogens.
- **Smart Heteroatom Valences**: Carbonyls ($C=O$), sulfoxides ($S=O$), sulfones ($O=S=O$), phosphates, and nitro groups ($-NO_2$) are recognized so oxygen atoms are not inappropriately protonated.

### 2.4 Internal Coordinate Fine-Tuning Panel
- Selecting an atom or cell activates the parameter control panel:
  - **Bond Length Slider & Spinbox**: Range $0.4\text{ \AA} - 10.0\text{ \AA}$ (resolution $0.001\text{ \AA}$).
  - **Bond Angle Slider & Spinbox**: Range $0.0^\circ - 180.0^\circ$ (resolution $0.1^\circ$).
  - **Dihedral Slider & Spinbox**: Range $-180.0^\circ - 180.0^\circ$ (resolution $0.1^\circ$).
  - **Invert Chirality / Dihedrals**: Flips all dihedral angles to generate enantiomers.

### 2.5 Force Field Optimization & Vibrational Spectroscopy
- **Universal Force Field (UFF/MM) & Hybrid Optimization**:
  - Harmonic bond stretching, angle bending (linear, trigonal planar, tetrahedral), torsional barriers, and non-bonded Lennard-Jones repulsion.
  - Multi-engine optimization dropdown:
    - **Analytical UFF / MMFF94s**: Fast built-in analytical force field with exact symmetry preservation and universal element coverage.
    - **Auto (Hybrid MMFF / UFF)**: Leverages RDKit MMFF94 parameterization for organic frameworks, automatically falling back to analytical UFF for organometallics or atypical valencies.
    - **GFN0-xTB**: Semiempirical quantum mechanics tight-binding optimization with electronic structure calculation.
  - Preserves user-drawn connectivity and frozen atom constraints.
  - Re-aligns the relaxed molecule to standard orientation.
- **Vibrational Normal Mode Calculation**:
  - Computes numerical Hessian matrix from the force field.
  - Mass-weighted diagonalization to obtain $3N-6$ ($3N-5$ for linear) vibrational frequencies and Cartesian normal mode displacements.
  - Interactive normal mode animator: 3D viewport animates atom oscillations with directional displacement arrows (adjustable amplitude and frequency).

### 2.6 HPC Cluster Batch Script Generator
- Accessible directly from the Quantum Chemistry Calculation Runner (`Ctrl+R`) via **"Generate Cluster Script (SLURM/PBS/LSF/SGE)"**:
  - Generates production-ready submission scripts for **SLURM**, **PBS / Torque**, **IBM LSF**, and **Sun Grid Engine (SGE)**.
  - Configurable queue/partition, walltime limits, CPU core allocations, memory quotas (MB/GB), scratch directories (`$SLURM_TMPDIR` / `/tmp`), and email notifications.
  - Built-in scratch lifecycle management: automatic input copying, local SSD execution, output log piping, and result archive sync back to the submission directory.
  - Copy to clipboard or export directly as `.sh` / `.slurm` / `.pbs` file.

### 2.7 Point Group Symmetry & Group Theory Engine
- Detects molecular point groups ($C_1, C_s, C_i, C_n, C_{nv}, C_{nh}, D_n, D_{nd}, D_{nh}, T_d, O_h, I_h, C_{\infty v}, D_{\infty h}$).
- Lists all symmetry operations ($\hat{E}, \hat{C}_n, \hat{\sigma}, \hat{i}, \hat{S}_n$) and their $3\times3$ transformation matrices.
- **Align to Standard Orientation**: Aligns highest $C_n$ axis along $Z$, secondary axes along $X$, and reflection planes in $XZ/YZ$.
- **Enforce Symmetry**: Projects distorted geometries into exact symmetry subspaces.
- **Irreducible Representation Decomposition**: Calculates reducible representation $\Gamma_{\text{tot}}$, subtracts $\Gamma_{\text{trans}} \oplus \Gamma_{\text{rot}}$, and uses the Great Orthogonality Theorem to decompose $\Gamma_{\text{vib}}$ into Mulliken irrep symbols ($A_1, B_2, E_g, T_{2u}$, etc.).

---

## 3. Comparative Evaluation: AgyMolden vs. Molden, Avogadro, Gabedit, and GaussView

| Feature / Capability | **AgyMolden** | **Molden (7.x)** | **Avogadro (1.2 / 2.0)** | **Gabedit (2.5)** | **GaussView (6.x)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Primary Paradigm** | Internal (Z-matrix) + Cartesian | Internal (Z-matrix) | Cartesian (3D Canvas) | Mixed (Z-matrix & Cartesian) | Cartesian + Gaussian builder |
| **Z-Matrix Reference Tree Control** | **Excellent**: Visual highlighting of target, bond, angle, dihedral references with colored pills | **Classic / Rigid**: Command/button driven, cryptic terminal prompts | **Poor / Absent**: Converts to Cartesian internally; no Z-matrix tree editor | **Good**: Has a dedicated Z-matrix builder window | **Moderate**: Can edit redundant internals, but cumbersome |
| **Mouse-Based Atom Dragging** | **Yes** (Avogadro-style elastic drag & Ctrl-translate) | **No** (Rigid parameter inputs) | **Yes** (Industry benchmark for freehand sketch) | **Limited** | **Moderate** (Click-to-place fragments) |
| **Bond Order Control** | **Double-click cycle** with strict chemical valence guards | Text/menu assignment | Click-and-drag or click bond | Menu selection | Context menu / builder panel |
| **Valence Protection on Bond Edit** | **Yes**: Rejects double bonds on H/halogens and saturated centers | **No** (Displays whatever user types) | Partially (auto-adds H, sometimes creates odd valences) | **No** | **Yes**: Enforces standard hybridization |
| **Selective vs. Global Add H** | **Yes** (Selected atoms only, or whole molecule if none selected) | Limited / Global only | Usually global | Global only | Smart / per-valence |
| **Point Group Symmetry & Irreps** | **Complete**: Detection, standard alignment, projection, $\Gamma_{\text{vib}}$ irrep decomposition | Detection & Z-matrix symmetry matching | Basic point group detection | Detection & symmetry constraints | **Gold Standard**: Point group builder & subgroup toggle |
| **Normal Modes & Vibrations** | **Built-in MM Hessian** + interactive 3D arrow animation | Reads calculation outputs only | Reads calculation outputs only | Reads calculation outputs only | Reads Gaussian `.log`/`.chk` outputs only |
| **UI & Experience** | Modern PyQt6, dark theme, responsive splitter, high-DPI | X11 Xlib (1990s retro aesthetic, clunky fonts) | Modern Qt5/Qt6 interface | GTK2/GTK3 (dense toolbars) | Qt commercial interface |
| **Undo / Redo Stack** | **Full** (Ctrl+Z / Ctrl+Y across all actions) | **None** | **Full** | Limited | **Full** |

---

## 4. Critical Usability Analysis: Are We Doing Things Right to Draw Molecules with Ease and Control?

### 4.1 What AgyMolden Gets Exceptionally Right
1. **The "Best of Both Worlds" Hybrid Model**:
   - In pure Cartesian builders (like Avogadro), creating a ring conformation or setting a precise $109.47^\circ$ angle or $180.0^\circ$ dihedral requires either running a force field or cumbersome geometry constraints.
   - In pure Z-matrix tools (like classic Molden), building a branched alkane or ring requires tedious mental accounting of atom indices (`"Atom 16 bonded to 4, angle to 3, dihedral to 1"`).
   - **AgyMolden solves this** by allowing the user to sketch or drag atoms with the mouse while the application automatically maintains a valid, mathematically sound Z-matrix tree.
2. **Interactive Reference Highlighting**:
   - Clicking a table cell (e.g. Dihedral of Atom 7) highlights the 4 involved atoms in distinct, color-coded pills in the 3D viewport (Target = Gold, Bond Ref = Cyan, Angle Ref = Purple, Dihedral Ref = Orange). This eliminates the confusion typical of editing raw Z-matrices.
3. **Valence Integrity**:
   - The recent additions preventing multivalent hydrogens, monovalent halogen over-bonding, and allowing selective hydrogen completion on specific atoms bring AgyMolden on par with professional chemical intelligence engines.
4. **Autonomous Self-Contained Vibrations & Symmetry**:
   - Unlike Molden or Gabedit which require running an external quantum chemistry calculation (Gaussian/GAMESS) just to visualize vibrational modes or irreducible representations, AgyMolden can compute a force field Hessian and normal modes immediately with zero external dependencies.

### 4.2 Where Control Can Be Further Improved (Identified Gaps)
Based on real user testing (e.g., the `cyclo.zmat` cyclohexane issue) and comparison with GaussView and Avogadro, three specific areas need attention:

1. **Ring Closure & Cross-Reference Constraints**:
   - *Problem*: A standard Z-matrix is a mathematical tree spanning $N-1$ directed edges. It cannot natively represent a closed ring in its primary bond references without an explicit distance constraint between the start and end of the ring (e.g. $C_1 \cdots C_6$).
   - *Improvement*: When drawing a bond between two existing atoms (like closing cyclohexane from $C_6$ to $C_1$), provide an explicit **Ring Closure / Cross-Link** indicator so the force field optimizer pulls the ring closed rather than treating them as non-bonded.
2. **Periodic Canvas Double-Click vs. Element Hotkeys**:
   - *Avogadro/GaussView strength*: Pressing keys `C`, `N`, `O`, `F` immediately changes the active drawing brush without opening a modal dialog.
   - *AgyMolden status*: Currently opens the periodic table dialog on canvas double-click. Adding quick element hotkeys (`C`, `N`, `O`, `H`, `S`, `P`, `F`, `Cl`) will speed up sketching significantly.
3. **Hydrogen Pruning on Manual Ring Formation**:
   - When an atom already saturated with hydrogens is manually bonded to another heavy atom, the software should automatically displace one hydrogen rather than leaving a pentacoordinate intermediate.

---

## 5. Quick-Start User Guide

### 5.1 Basic Navigation & Camera Controls
- **Rotate Camera**: Left-Click and drag on empty viewport space.
- **Pan Camera**: Right-Click and drag on empty viewport space.
- **Zoom**: Mouse Scroll Wheel or Middle-Click drag.
- **Select Atom**: Left-Click on any atom sphere.
- **Multi-Select Atoms**: `Shift` + Left-Click on atoms (displays interatomic distance for 2 atoms, valence angle for 3 atoms, dihedral torsion for 4 atoms).
- **Move / Translate Atom(s)**: `Ctrl` + Left-Click and drag an atom in the 3D viewport to reposition it in the camera plane.
- **Quick Element Hotkeys**:
  - `C`: Carbon (or `Shift+C`: Chlorine)
  - `H`: Hydrogen
  - `O`: Oxygen
  - `N`: Nitrogen
  - `S`: Sulfur (or `Shift+S`: Silicon)
  - `P`: Phosphorus
  - `F`: Fluorine
  - `B`: Boron (or `Shift+B`: Bromine)
  - `I`: Iodine
  - *If an atom is selected*: pressing an element key mutates the atom to that element with automatic valence recalculation.
  - *If no atom is selected*: switches the active drawing brush (displayed in the top-left HUD pill).
- **Molden Z-Matrix Reorder Shortcuts**:
  - `Alt+B`: Adjust Bond partner.
  - `Alt+A`: Adjust Angle partner.
  - `Alt+D`: Adjust Dihedral partner.
- **Selection & Viewport Hotkeys**:
  - `Space`: Center & Zoom-to-fit 3D viewport.
  - `Ctrl+A`: Select All atoms.
  - `Ctrl+D`: Deselect All atoms.
  - `Ctrl+E`: Export Publication-Quality High-DPI Image (PNG with transparent, dark, or white background presets).
  - `F2`: Open Interactive QM Output File & Log Viewer (with clickable section outline and regex search).
- **Context-Sensitive Right-Click Menus**:
  - **Right-Click Atom**: Change Element, Attach Functional Group, Invert Chirality, or Delete.
  - **Right-Click Bond**: Set Bond Order (Single, Double, Triple) or Delete Bond.
  - **Right-Click Empty Canvas**: Center View, Add Atom Line, Saturate Hydrogens, or Optimize.
  - *(Note: Right-Click and Drag remains dedicated to smooth camera panning without risk of accidental deletion).*
  - *If no atom is selected*: switches the active drawing brush (displayed in the top-left HUD pill).

### 5.2 Building Molecules Step-by-Step
1. **Starting Fresh**:
   - Double-click the empty viewport canvas to open the Periodic Table, or press any element hotkey (`C`, `O`, `N`, etc.) to choose the brush.
   - If **"Adjust Hydrogens"** is checked in the toolbar, methane ($CH_4$) will be generated automatically.
2. **Adding Atoms & Growing Chains**:
   - **Method A (Avogadro Drag)**: Click an existing atom and drag into empty space. A new atom matching the active brush will appear at standard bond distance ($1.54\text{ \AA}$ for C–C).
   - **Method B (Dedicated Ring Closure Button - Recommended)**:
     1. Select the two atoms to connect into a ring (`Shift` + Click both atoms in the 3D viewport or table).
     2. Click the **💍 Close Ring** button in the toolbar.
     3. AgyMolden automatically prunes excess hydrogens on both closure atoms to preserve valence, adds a cross-bond constraint, and guides the atoms toward each other via a 5-step gradual approach with intermediate force field relaxations at each step.
   - **Method C (Drag-to-Atom Snap)**: Drag from one atom and drop directly onto another existing atom. The green HUD snap indicator will confirm the ring connection.
   - **Method D (Double-Click)**: Double-click an existing hydrogen to replace it with the active element brush.
   - **Method E (Add Line / Molden)**: Click **➕ Add Atom**, select an element, then click 3 reference atoms in the 3D viewport to set `ref1`, `ref2`, and `ref3`.
   - **Interactive 3D Torsion / Rotamer Rotation (`Alt + Left-Click Drag`)**:
     - Hold `Alt` and drag directly across any rotatable single bond to smoothly rotate the downstream functional group or rotamer in real time without altering other coordinates.
3. **Changing Bond Orders**:
   - Double-click any covalent bond cylinder.
   - Single ($1.54\text{ \AA}$) $\to$ Double ($1.34\text{ \AA}$) $\to$ Triple ($1.20\text{ \AA}$) $\to$ Single.
   - Valences adjust automatically (e.g., ethane $\to$ ethylene $\to$ acetylene). Double bonds on hydrogen or monovalent halogens are strictly prohibited.
4. **Adding Hydrogens**:
   - Select one or more specific atoms, then click **💧 Add Hydrogens** (or press `Ctrl+H`) to saturate only those centers.
   - Deselect all atoms and click **💧 Add Hydrogens** to saturate the entire molecule.
5. **Geometry Optimization & Symmetry**:
   - Click **⚡ Optimize (FF)** (or `Ctrl+Shift+O`) to relax bond angles, cross-bond ring strains, and steric clashes using the molecular mechanics force field.
   - Switch to the **Symmetry** tab and click **Detect Point Group Symmetry** to view symmetry operations, or **Align to Standard Orientation** to set the molecule in standard orientation.
6. **Simulating Vibrations**:
   - Switch to the **Vibrational Modes** tab and click **🎵 Calculate Frequencies**.
   - Select any normal mode frequency from the table and click **▶ Simulate Mode** to observe 3D vibrational animation.

### 5.3 Advanced Wavefunction Analysis (Multiwfn-Style)
Open **Analysis $\to$ Volumetric Grid & Isosurface Viewer** (or load a `.molden` file) to access:
- **⚡ 3D ESP Surface**: Exact electrostatic potential grid and 3D isosurfaces.
- **🎨 ESP-Mapped Density**: Isodensity envelope colored by electrostatic potential heatmap (Red = Nucleophilic / Electron-rich, Blue = Electrophilic / Electron-deficient).
- **🌀 Spin Density**: Open-shell radical/triplet distribution ($\rho_\alpha - \rho_\beta$) with dual-color positive/negative lobes.
- **📈 2D ESP Contours**: Slice potential heatmaps and contour lines along XY, XZ, or YZ planes with atom overlays and custom palettes.
- **⚡ ELF Localization (Becke-Edgecombe)**: Real-time kinetic energy density analysis ($\text{ELF} \in [0.0, 1.0]$) mapping covalent bonds and lone pair basins (isovalue $\sim 0.82$).
- **🔍 Localized Orbitals (Boys / Pipek-Mezey)**: Transform canonical delocalized MOs into intuitive localized bond and lone pair orbitals with calculated centroids.
- **🔬 MO Orbital Decomposition (% AO / Atom Contribution)**: Instant Mulliken-type atomic contribution readout in the MO list banner (e.g. `O3: 52.7%, C2: 14.2%`) identifying bonding/antibonding/lone-pair character without external tools.
- **🌿 Non-Covalent Interactions (NCI / RDG)**: Johnson's Reduced Density Gradient isosurfaces ($s=0.5$) colored by $\text{sign}(\lambda_2)\rho$ (Blue = H-bonding/attractive, Green = van der Waals dispersion, Red = steric clash).
- **🎯 Average Local Ionization Energy (ALIE)**: Mapped onto electron isodensity surfaces with automatic detection and labeling of local surface minima (revealing reactive electrophilic and radical attack sites).
- **📍 Multiwfn-Style Surface Extrema ($V_{\text{min}}$ / $V_{\text{max}}$)**: Precise detection of $\sigma$-holes and nucleophilic sites on the ESP surface, projected as interactive colored spheres with kcal/mol and a.u. readout pills in the 3D viewport.
- **🕸️ Surface Render Styles**: Toggle between **Solid Shaded Surface** and **Wireframe Polygon Mesh** to inspect inner core orbitals and electron densities inside transparent outer envelopes.
- **🎯 CHELPG Partial Charges**: Electrostatic potential grid fitting to atomic centers for partial charge mapping in the 3D viewport.

- **⚡ Multiwfn Wavefunction Suite** (`File → Multiwfn Wavefunction Suite…` or the button in the orbital dialog): Mayer/Wiberg bond orders and valences, condensed Fukui indices ($f^+$, $f^-$, $f^0$) with the dual descriptor $\Delta f$, QTAIM critical points (BCPs drawn as spheres in the 3D view), and CM5/Hirshfeld charges. Bond orders and Fukui work out of the box (PySCF). QTAIM and CM5 need the free external program [Multiwfn](http://sobereva.com/multiwfn): put it on your `PATH`, set the `MULTIWFN_PATH` environment variable, or click **Set Multiwfn path…** in the dialog.
- **🗺️ 2D Slicer properties**: the 2D contour dialog can plot ESP, electron density ρ, the Laplacian ∇²ρ, or the reduced density gradient on XY/XZ/YZ planes.
- **🔢 Symmetry matrices**: *Show Symmetry Matrices* has an operation selector and a "display all operations" mode, with trace and determinant of each 3×3 matrix.
- **Orbital isovalue**: the Molecular Orbitals tab has a slider plus spin box; the grid is cached so changing the isovalue re-meshes instantly.
- **Optional dependency note**: ESP/ELF/NCI/CHELPG/Fukui/2D slicing need PySCF (`pip install .[qm]`; not available on Windows). The standalone executable does not bundle PySCF.

---

## 6. Conclusion

AgyMolden successfully unifies the mathematical precision of **Molden's Z-matrix generator** with the modern, fluid ergonomics of **Avogadro** and **GaussView**. Its valence protection, selective hydrogen saturation, cross-bond ring closure snapping, element brush HUD, and autonomous group theory engine make it an effective tool for computational chemistry model preparation on Linux.
