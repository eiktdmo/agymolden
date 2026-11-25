# AgyMolden Student Quickstart Guide

Welcome to your Computational Chemistry laboratory! **AgyMolden** lets you sketch molecules in 2D or 3D, edit internal coordinates (Z-Matrix) and Cartesian coordinates (XYZ), run or export quantum chemistry jobs (ORCA, PySCF, Gaussian, Psi4, xTB), inspect output logs, and visualize 3D orbitals, electron densities, ESP maps, and vibrations.

---

## 🛠️ Getting Started (Linux / macOS / Windows)

1. **Unzip** `AgyMolden_Student_Release.zip` into a folder of your choice.
2. **Install**:
   - **Linux / macOS**: Open a terminal inside the unzipped folder and run:
     ```bash
     ./setup.sh
     ```
   - **Windows (or manual virtual environment)**:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate      # or: source .venv/bin/activate on Linux/macOS
     pip install -e .
     ```
     *(Optional quantum engine on Linux/macOS: `pip install -e ".[qm]"` to enable built-in PySCF orbitals, 3D density color maps, and ESP surfaces without needing external ORCA.)*
3. **Launch AgyMolden**:
   ```bash
   agymolden
   ```
   *(Or `python -m agymolden` inside the activated virtual environment.)*

> **Tip:** Prefer a bright classroom/projector theme or dark mode? Use the **Theme** toggle on the top toolbar to switch between **White** and **Dark** themes at any time.

---

## ✏️ Step 1: Build or Open Your Molecule

### Option A: Open an Existing Molecule or Calculation
- Press **`Ctrl+O`** (or **File → Open File...**) to open `.xyz`, `.zmat`, `.molden`, `.cube`, `.fchk`, or ORCA/Gaussian `.out` / `.log` files.
- Ready-to-use sample molecules are included in the `examples/` folder (e.g., benzene, ethanol, water, ferrocene).

### Option B: Draw in the 2D | 3D Split Sketcher
1. Click **`✏️ 2D | 3D`** on the toolbar to open the live 2D sketcher alongside the 3D viewport.
2. **Elements**: Pick any quick element (`C`, `H`, `O`, `N`, `S`, `P`, `F`, `Cl`, `Br`, `I`) or click **`⚛️ Other...`** to open the full Periodic Table (`B`, `Si`, `Li`, `Fe`, `Pt`, etc.).
3. **Rings (2D Ring Welder)**:
   - Click **`⌬ Benz`**, **`⬡ C6`**, **`⏢ C5`**, **`◻ C4`**, **`△ C3`**, or **`⬡ C7`**.
   - Move over an existing bond to **fuse** two rings together (e.g., naphthalene, anthracene, pyrene), or over an atom to **spiro/exocyclically** attach a ring.
   - Click once to place (auto-returns to Draw mode), or hold **`Shift + Click`** to chain multiple rings!
4. **3D Sync & Undo**: Every stroke updates the 3D structure automatically, and **`Ctrl+Z` / `Ctrl+Y`** undoes/redoes cleanly.
5. Click **`◧ Z-Mat`** (or **`📋 Z-Matrix`**) at any time to switch back to the **Z-Matrix** and **XYZ** coordinate tables.

### Option C: Edit Directly in 3D, Z-Matrix, or XYZ
- **Z-Matrix Tab & XYZ Tab**: Always available in the inspector panel—click any row/cell to highlight reference atoms in 3D and fine-tune bond lengths, angles, and dihedrals.
- **3D Mouse Shortcuts**:
  - **Drag from an atom** into empty space to grow a new atom.
  - **Double-click a bond** to cycle Single → Double → Triple bond order.
  - **`Ctrl + Drag` an atom** to move it; **`Alt + Drag` across a single bond** to rotate a dihedral/torsion.
  - **`Ctrl+H`**: Add missing hydrogens (selected atoms only, or whole molecule if none selected).
  - **`⚡ Optimize (FF)` (`Ctrl+Shift+O`)**: Quick force-field geometry cleanup.

---

## ⚡ Step 2: Run ORCA or Save an Input File (`Ctrl+R`)

1. Click **`⚡ ORCA Studio`** (or press **`Ctrl+R`**) on the toolbar.
2. Choose a lab preset:
   - **📐 Geometry Optimization** (`! B3LYP def2-SVP Opt`)
   - **🎵 Vibrations & IR Spectrum** (`! B3LYP def2-SVP Opt Freq`)
   - **🌐 Single Point / Orbitals** (`! B3LYP def2-SVP SP`)
3. Set **Charge** (`0` for neutral) and **Multiplicity** (`1` for singlet, `2` for doublet, etc.).
4. **Print Molecular Orbitals**: Check **`⚛️ Print Molecular Orbitals & AO Composition`** when you want ORCA's `.out` file to include orbital energies, occupations, and atomic orbital (AO) contributions.
5. **Choose Output Folder**: Pick where `.inp`, `.out`, and `.molden.input` files are saved (AgyMolden remembers your folder automatically).
6. Choose how to run:
   - **Run locally right now**: Click **`🚀 Run ORCA Now`** to watch live 3D optimization steps and convergence plots.
   - **Save for an HPC cluster or external run**: Click **`💾 Save Input Only...`** to export a clean `.inp` file without running ORCA locally. *(Need SLURM/PBS/LSF/SGE scripts? Use **File → QM Calculation Studio** → **Generate Cluster Script**.)*

---

## 📖 Step 3: Analyze Outputs for Class (`F2` Output Viewer)

Press **`F2`** (or click **`📖 Output Viewer`** after a job finishes) to open the **QM Output File Viewer**.

Use the **Teaching Quick-Jump Bar** at the top of the viewer to jump straight to what you need for lab reports:
- **`⭐ Final E`**: Final single-point / total energy (`Eh` and `kcal/mol`).
- **`🧮 E-Parts`**: Separated SCF energy components (Nuclear Repulsion, Electronic, Kinetic, Potential, Exchange/Correlation).
- **`🔬 E-Corr`**: Post-HF correlation energy (`MP2`, `CCSD`, `CCSD(T)`).
- **`⚛️ MOs`**: Molecular Orbital table (orbital number, occupation, energy in `Eh` and `eV`, plus Löwdin/Mulliken AO composition).
- **`🎵 Vibs`**: Harmonic vibrational frequencies (`cm⁻¹`) and IR intensities.

---

## 🌐 Step 4: Visualize 3D Orbitals, Electron Density & Vibrations

Open the **Orbitals** (or **Vibrations**) tab on the right panel:

- **Molecular Orbitals**: Select **HOMO**, **LUMO**, or any MO from the list. Adjust **Isovalue** and switch between **Solid** and **Wireframe** mesh styles.
- **3D Electron Density Color Map (`☁️ ρ(r) ▾`)**:
  - **3D Color Map (Surface)** *(default)*: Displays a smooth 3D molecular van der Waals surface colored by electron density $\rho(r)$—just like the 3D ESP-mapped surface! Use the live inline controls to switch **Palette** (`Plasma`, `Viridis`, `Turbo`, `Inferno`, `RdBu`), toggle **`Log10`** scaling, adjust **Scale**, or switch **Mode** to **3D Slice (Plane)**.
  - **3D Isosurface**: Classic single-isovalue 3D density shell.
  - **2D Contour Map**: Slice $\rho(r)$, ESP, $\nabla^2\rho$, or RDG on `XY` / `XZ` / `YZ` planes.
- **ESP-Mapped Density**: Maps electrostatic potential onto the electron density surface (red = electron-rich/nucleophilic, blue = electron-poor/electrophilic).
- **Vibrations**: Click any frequency in the **Vibrations** tab and press **▶ Simulate Mode** to animate normal modes in 3D.

---

## 🚀 Step 5: Edit Input/Output Files in Agemacs

If **Agemacs** is installed alongside AgyMolden:
- Click **`🚀 Agemacs ▾`** on the top QM toolbar to send your current **Z-Matrix**, **XYZ**, **ORCA Input**, **QM Input**, or **QM Output** directly to Agemacs.
- You can also click **`🚀 Open in Agemacs`** directly inside **ORCA Studio**, **QM Studio**, or the **Output Viewer**.

---

## ⌨️ Essential Shortcuts & Troubleshooting

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+O` / `Ctrl+S` | Open / Save molecule |
| `Ctrl+Z` / `Ctrl+Y` | Undo / Redo (works in both 2D and 3D) |
| `Space` | Center & zoom-to-fit 3D view |
| `Ctrl+R` | Open ORCA Studio |
| `F2` | Open QM Output Viewer (with Teaching Quick-Jump Bar) |
| `Ctrl+H` | Add hydrogens (selected atoms or whole molecule) |
| `Ctrl+E` | Export high-DPI PNG image |

**Common Fixes:**
- **ORCA MPI / Parallel error**: Set **Cores (`%pal`)** to `1` in ORCA Studio.
- **SCF Not Converged**: Add `! SlowConv` or `! KDIIS` in the ORCA route line.
- **ORCA Not Found**: Open **File → Configure QM Engines & Routes** and browse to your `orca` executable.
- **Orbitals/Density button says PySCF is missing**: Install optional QM extras (`pip install -e ".[qm]"` on Linux/macOS) or load a `.molden` / `.molden.input` / `.cube` file from ORCA.
