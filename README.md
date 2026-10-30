# AgyMolden: Molden-Inspired Molecular Z-Matrix Builder

A polished, modern Linux desktop application for generating and manipulating molecular structures using internal coordinates (Z-matrix), following the workflow and spirit of the classic **Molden** package.

---

## 🌟 Key Features

1. **True Molden-Style Z-Matrix Workflow:**
   - **"Add Line"** atom addition referencing existing atoms (bond length, bond angle, dihedral).
   - Direct spreadsheet-style table with immediate 3D visualization.
   - Live synchronization with raw Molden / Gaussian format text files.

2. **Interactive 3D OpenGL Viewport:**
   - **Ball & Stick** and **Spacefill (van der Waals)** rendering modes with CPK coloring and lighting.
   - Dual-mode selection: Clicking an atom in the 3D viewport selects and scrolls to the atom in the Z-matrix table, and selecting in the table highlights the atom in 3D.
   - Real-time parameter tuning: sliders and spin-boxes to adjust bond lengths, angles, and dihedrals interactively with immediate 3D re-orientation.

3. **Measurements & Inspections:**
   - Shift + Click to measure:
     - 2 atoms: **Bond Distance (Å)**
     - 3 atoms: **Bond Angle (°)**
     - 4 atoms: **Dihedral Angle (°)**

4. **Fragment & Substituent Library:**
   - Attach or substitute common functional groups (Methyl `-CH3`, Hydroxyl `-OH`, Amino `-NH2`, Phenyl `-C6H5`, Nitro `-NO2`, Carboxyl `-COOH`, Cyano `-CN`, `-CF3`) to build complex systems quickly.

5. **Interoperability & Presets:**
   - Built-in library of presets (Water, Methane, Ammonia, Benzene, Ethanol, Formaldehyde).
   - Import & Export:
     - **Molden / Gaussian Z-Matrix** (`.zmat`, `.inp`, `.gjf`)
     - **Cartesian XYZ** (`.xyz`)

---

## 🚀 Installation & Source Setup

AgyMolden requires Python 3.9+ with PyQt6, PyOpenGL, NumPy, SciPy, and Matplotlib.

### Prerequisites (Linux)
On Debian/Ubuntu-based distributions, install the basic Qt/OpenGL libraries if they are not already installed:
```bash
sudo apt update && sudo apt install -y libgl1 libegl1 libxkbcommon-x11-0 libxcb-cursor0
```

### Option A – Quick automated setup from source (Linux / macOS)
```bash
./setup.sh          # creates .venv and installs core dependencies
./run.py            # or: python3 run.py
```

### Option B – Quick automated setup from source (Windows)
```bat
setup.bat
python run.py
```

### Option C – Standard `pip install` from source
You can install in editable/development mode directly from the project directory:

```bash
# Minimal installation (core 3D Z-matrix builder, OpenGL viewer, IR spectra):
pip install -e .

# Full installation with all optional features (SMILES/InChI via RDKit, marching cubes surfaces):
pip install -e ".[all]"

# Full QM installation (including cclib QM log parser and PySCF for Linux/macOS):
pip install -e ".[qm]"
```
After installation, you can launch directly anywhere with:
```bash
agymolden [molecule.zmat]
```

### Option D – Standalone executable (no Python required on target machine)
Compile a standalone single binary using PyInstaller (must run on the target OS):
```bash
./build.sh          # Linux/macOS  -> dist/agymolden
build.bat           # Windows      -> dist\agymolden.exe
```

### Linux Desktop Shortcut (optional)
Integrate AgyMolden into your system's application menu:
```bash
./make_desktop_entry.sh
```

---

## 💻 Running the Application

```bash
./run.py [file.zmat|file.xyz]
agymolden --version
```
You can also drag and drop a `.zmat` / `.xyz` file onto the window. Press **F1** (Help → Open User Manual) for the manual.

### Running the tests
```bash
pip install -r requirements-dev.txt
python -m pytest
```
Example molecules are in `examples/`.

### Mouse and Keyboard Shortcuts:
- **Left Click & Drag:** Rotate molecule in 3D.
- **Right Click & Drag:** Pan camera.
- **Ctrl + Right-Click & Drag from an Atom:** Quick-add a new atom of the last added element bound to the anchor atom.
- **Mouse Wheel / Middle Drag:** Zoom in / out.
- **Left Click on Atom:** Select atom for parameter adjustment and table synchronization.
- **Shift + Click on Atoms:** Select multiple atoms to calculate geometric parameters (Distance, Angle, Dihedral).
- **Escape:** Cancel / abort interactive atom connectivity selection.
- **Ctrl + O:** Open Z-matrix file.
- **Ctrl + S:** Save Z-matrix file.
