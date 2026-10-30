# AgyMolden Video Tutorial Script: From Beginner to Advanced

This script is structured into **5 progressive modules** (total duration ~12–15 minutes, or individual 2–3 minute bite-sized clips). It focuses heavily on **newBRORSEN** workflows first, followed by **ORCA** and general computational chemistry features.

---

## 🎬 Module 1: Introduction & One-Click Installation (Beginner)
**Target Duration:** ~2:00 min  
**Screen Display:** Clean terminal / Desktop showing the project folder.

### Video Actions & Voiceover:
- **[0:00 - 0:30] Introduction**
  - *Voiceover:* "Welcome to AgyMolden, an interactive 3D molecular builder, Z-matrix editor, and quantum chemistry workbench. In this tutorial, you will learn how to build molecules, run quantum calculations with newBRORSEN and ORCA, and analyze orbitals, vibrations, and spectra."
- **[0:30 - 1:15] Installation Options**
  - **Linux / macOS:**
    - *Action:* Open terminal, type `./setup.sh` (or simply `./run.py`).
    - *Voiceover:* "To install on Linux or Mac, simply open a terminal in the folder and run `./setup.sh`. It automatically creates a clean virtual environment and installs all dependencies. You can also just type `./run.py`—if it's your first time, it sets up everything automatically."
  - **Windows:**
    - *Action:* Double-click `setup.bat`.
    - *Voiceover:* "On Windows, double-click `setup.bat`. It installs the environment and places an `AgyMolden` shortcut directly on your desktop."
- **[1:15 - 2:00] First Launch & Interface Overview**
  - *Action:* Launch `./run.py`. Show the 3D viewport, toolbar, and the left-hand tab panels (Z-Matrix, Raw ZMAT, Presets & Fragments, Vibrations, Symmetry, Molecular Orbitals).
  - *Voiceover:* "When AgyMolden opens, you see the interactive 3D OpenGL viewport in the center, drawing brushes on top, and specialized tabs on the left."

---

## 🔬 Module 2: The newBRORSEN Workflow (Core Focus)
**Target Duration:** ~3:30 min  
**Screen Display:** Molecular builder and calculation setup dialog.

### Video Actions & Voiceover:
- **[2:00 - 2:45] Building / Loading a System for newBRORSEN**
  - *Action:* Click **Presets & Fragments** → Select **Formaldehyde** (or build $\text{CH}_2\text{O}$).
  - *Voiceover:* "Let’s start with a newBRORSEN calculation. Select Formaldehyde from the Presets tab. In the status bar, notice the live Hill formula, molecular weight, and steric clash monitor."
- **[2:45 - 3:45] Configuring newBRORSEN Multi-Stage Calculations**
  - *Action:* Click the **⚡ newBRORSEN** button on the top toolbar (or press `Ctrl+N`).
  - *Voiceover:* "Click the newBRORSEN button to open the Multi-Stage Engine setup dialog. Here you can configure:"
    - **Stage 1 (Electronic)**: "Method: HF or MP2, Basis: cc-pVDZ, Task: Geometry Optimization or Single Point."
    - **Positronic Sector (+Positron)**: "Check *Add Positron Stage* to configure multi-component NEO/nuclear-electronic calculations, selecting the positronic basis set (e.g. PSX-TZ)."
- **[3:45 - 4:45] Running the Job & Live Job Monitor**
  - *Action:* Click **Run Calculation**. Show the live output log streamer and progress bar.
  - *Voiceover:* "Click Run. AgyMolden executes newBRORSEN in the background with real-time log streaming. When the calculation finishes, AgyMolden automatically loads the optimized geometry, trajectory, and companion wavefunction files."
- **[4:45 - 5:30] Analyzing newBRORSEN Results & Positron Overlays**
  - *Action:* 
    1. Drag the **Trajectory Player** slider to watch the optimization steps.
    2. Click the **Molecular Orbitals** tab. Show the populated MO table, the energy ladder diagram, and the state badge: `State: ¹A₁ (C2v)`.
    3. Click the **⚛️ +Positron MO** button to overlay the positron ground-state density onto the electronic HOMO.
  - *Voiceover:* "Notice that all MO eigenvalues and the HOMO-LUMO gap appear automatically. Clicking '+Positron MO' overlays the gold/green positron cloud directly over the blue/red electron orbital."

---

## ⚡ Module 3: ORCA Calculations & General QM Engines (Intermediate)
**Target Duration:** ~3:00 min  
**Screen Display:** ORCA Calculation Dialog and QM log output viewer.

### Video Actions & Voiceover:
- **[5:30 - 6:15] Quick ORCA Setup**
  - *Action:* Click **⚡ QM Setup** → Choose **ORCA** tab (or **Quick ORCA** preset).
  - *Voiceover:* "For standard electronic structure calculations, AgyMolden provides seamless integration with ORCA, Gaussian, PySCF, and xTB."
  - *Action:* Select **DFT (B3LYP / def2-SVP)**, check **Frequency / Vibrational Modes**, and click **Generate Input / Run**.
- **[6:15 - 7:15] Vibrational Analysis & IR Spectrum**
  - *Action:* Switch to the **Vibrations** tab. Click on a vibrational mode (e.g. carbonyl stretch $\sim 1750\text{ cm}^{-1}$). Click **Play** and toggle **Displacement Vectors**.
  - *Voiceover:* "Once the ORCA frequency job completes, the Vibrations tab displays the IR spectrum. Click any band to animate the 3D molecular vibration with live displacement arrows. You can adjust frequency scaling factors and vector amplitudes."
- **[7:15 - 8:30] Opening Past Calculations**
  - *Action:* Click **📂 Load Molden / QM Log...** in the MO tab and select an old calculation log or `.molden` file.
  - *Voiceover:* "You can open any historical calculation file—ORCA `.out`, Gaussian `.log`, or `.molden`—and AgyMolden instantly reconstructs the MO table, the interactive energy diagram, and the 3D isosurfaces."

---

## 🛠️ Module 4: 3D Molecular Editing & Builder Controls (Intermediate)
**Target Duration:** ~3:00 min  
**Screen Display:** Viewport interactions, mouse shortcuts, hotkeys.

### Video Actions & Voiceover:
- **[8:30 - 9:15] Intuitive Drawing & Element Mutation**
  - *Action:* In Build mode, click canvas to add atoms. Click an atom and press element hotkeys: `C`, `N`, `O`, `F`. Right-click an atom to show the **Change Element** popup menu.
  - *Voiceover:* "Building molecules is fast. Click the canvas to place atoms. Press keys like C, N, or O while an atom is selected to mutate its element with automatic valence saturation."
- **[9:15 - 10:00] Shift+Click Precision Measurements & Ring Centroids**
  - *Action:*
    1. Shift+click 2 atoms $\to$ Bond distance & bond type.
    2. Shift+click 3 atoms $\to$ Bond angle.
    3. Shift+click 4 atoms $\to$ Dihedral angle.
    4. Shift+click 6 atoms of a benzene ring $\to$ Centroid coordinates and span radius.
  - *Voiceover:* "Shift+click two atoms to measure distances, three for angles, and four for dihedrals. Shift+clicking a ring calculates the geometric centroid coordinates and radius."
- **[10:00 - 10:45] Chiral Center Inversion & 3D Manipulator Gizmo**
  - *Action:* Select a chiral center $\to$ click `🔄 Invert Chiral Center (R/S)`. Toggle `3D Gizmo` and drag an axis handle.
  - *Voiceover:* "Need an enantiomer? One click on 'Invert Chiral Center' flips the stereocenter without distorting the rest of the molecule. The 3D translation gizmo allows fine adjustments along Cartesian axes."
- **[10:45 - 11:30] Point Group Symmetry Locking**
  - *Action:* Click **Symmetry** tab $\to$ click **Detect Point Group** $\to$ check **Lock Symmetry**. Move an atom and show symmetric partners moving synchronously.
  - *Voiceover:* "Under the Symmetry tab, detect the point group and check 'Lock Symmetry' to enforce symmetric motion during building."

---

## 🎨 Module 5: Publication Figures, New Features & Pro-Tips (Advanced)
**Target Duration:** ~2:30 min  
**Screen Display:** Viewport styling, snapshot gallery, export dialogs.

### Video Actions & Voiceover:
- **[11:30 - 12:15] Axis Snapping & 180° Flip**
  - *Action:* Press `Z` (XY plane), `Y` (XZ plane), `X` (YZ plane), and backquote `` ` `` to flip 180°. Press `F5` to toggle Orthographic parallel projection.
  - *Voiceover:* "For publication figures, press X, Y, or Z to snap the camera directly along Cartesian axes. Press backquote to flip the view 180 degrees, and press F5 for pure orthographic projection without perspective distortion."
- **[12:15 - 13:00] Representation Styles & Instant Clipboard Copy**
  - *Action:* Switch the Style dropdown between **Ball & Stick**, **Spacefill**, **Licorice / Tube**, and **Wireframe**. Press `Ctrl+Shift+C`. Switch to PowerPoint/Word and press `Ctrl+V` to paste the transparent PNG.
  - *Voiceover:* "Choose between Ball & Stick, Spacefill, Licorice tube, or Wireframe. Press Ctrl+Shift+C to copy a clean, high-resolution transparent render directly to your clipboard for instant pasting into presentations and manuscripts."
- **[13:00 - 13:45] Orbital Snapshot Gallery & 3D Webpage Export**
  - *Action:*
    1. Click **📸 Snapshot Gallery** to display a Gabedit-style contact sheet of all orbitals.
    2. Click **File → Export Interactive 3D HTML...** and open the generated `.html` in a web browser.
  - *Voiceover:* "Use the Snapshot Gallery to preview all molecular orbitals in a single contact sheet. Finally, click 'Export Interactive 3D HTML' to send your students or collaborators a zero-install, fully interactive 3D model that runs on any smartphone or browser."
- **[13:45 - 14:00] Conclusion**
  - *Voiceover:* "AgyMolden gives you the full power of modern quantum chemistry visualization and molecular building right on your laptop. Enjoy your calculations!"
