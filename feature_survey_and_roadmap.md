# Benchmark Analysis: Popular Molecular Visualization Features

A comparative survey of leading molecular modeling and quantum chemistry visualization tools (**GaussView**, **Chemcraft**, **Avogadro 1/2**, **Gabedit**, **VMD**, and **Mercury / CYLview**) reveals several high-impact capabilities that enhance everyday workflows for computational and organic/inorganic chemists.

---

## 🚀 Category 1: "Easy Wins" (High Value, Fast & Clean Implementation)

These features have high daily utility, fit directly into our existing OpenGL renderer, PyQt UI, and QM parser architecture, and can be implemented cleanly with low risk.

| Feature | Inspired By | Description & User Value | Implementation Path |
| :--- | :--- | :--- | :--- |
| **1. Standard Molecular Projections (Snap to Axis)** | GaussView / VMD / Avogadro | Hotkeys or toolbar buttons to snap the camera to pure standard viewports: **XY** (+Z), **XZ** (+Y), **YZ** (+X), and inverted (-Z, -Y, -X). Crucial for publication figure preparation, alignment checking, and viewing planar systems. | In `agymolden/viewer.py`: simple camera orientation presets setting rotation quaternions to standard 90°/180° orientations. |
| **2. Invert / Mirror Stereochemistry (Handedness Inversion)** | Chemcraft / GaussView | One-click button or menu item to invert the stereocenter or entire molecule through a point ($x, y, z \to -x, -y, -z$) or mirror across XY/XZ/YZ planes. Indispensable for chiral molecules, enantiomers, and stereochemical audits. | In `agymolden/mw_editing.py`: negate coordinates of selected atoms or whole system, update Z-matrix. |
| **3. Center of Mass / Centroid Indicator & Marker** | Chemcraft / Mercury | Toggle displaying a glowing marker (and coordinates) at the **Center of Mass (COM)** or geometric **Centroid** of the full molecule or selected group of atoms. Great for measuring distances from an ion/metal to a ring centroid (e.g. cyclopentadienyl, benzene). | In `agymolden/viewer.py`: calculate mass-weighted mean or selected mean; render a small subtle crosshair/sphere with distance readout. |
| **4. Quick Bond Length / Angle Modifying Gizmo Slider** | GaussView (Builder) | When 2 atoms (bond) or 3 atoms (angle) or 4 atoms (dihedral) are selected, show a small floating HUD slider to directly adjust the distance or angle and automatically rotate/translate the dependent fragment. | In `agymolden/viewer_interaction.py` & `mw_editing.py`: connects current measurement label directly to Z-Matrix / coordinate transformation. |
| **5. Multi-Atom Centroid-to-Atom / Centroid-to-Centroid Distance** | Mercury / CYLview | Shift+clicking a group of atoms (e.g. aromatic ring) and then an atom or another ring to measure the distance to the ring centroid (e.g., Fe to Cp ring center in ferrocene). | In `agymolden/viewer_interaction.py`: detect when >3 coplanar atoms are selected as group 1, and compute distance to atom 2. |
| **6. Depth Cueing / Distance Fog** | VMD / PyMOL | Subtle OpenGL depth cueing / fog effect where distant atoms and bonds slightly fade towards the background color. Instantly adds photographic 3D depth perception for molecules with >15 atoms or coordination complexes. | In `agymolden/viewer.py`: standard OpenGL fixed/shader fog (`glEnable(GL_FOG)`, `glFogf(GL_FOG_DENSITY, ...)` or fragment shader depth interpolation). |
| **7. Fast Formula & Molecular Weight Badge** | Chemcraft / Avogadro | A small status badge or sidebar tag showing the empirical formula (Hill system: e.g. $\text{C}_2\text{H}_6\text{O}$) and exact molecular weight in g/mol ($46.069\text{ g/mol}$). | In `agymolden/mw_ui.py`: computed in 1 line via `collections.Counter(self.symbols)` and `get_atomic_weight()`. |

---

## 🏆 Category 2: Interesting Features (Medium Complexity, High Payoff)

These provide distinct workflows that set agymolden apart from conventional viewers.

| Feature | Inspired By | Description & User Value | Complexity & Plan | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **1. CYLview / Nature-Style Illustration Rendering Preset** | CYLview / Chemcraft | Non-photorealistic rendering mode: pastel atom colors, thick outline edges, soft directional lighting, and matte finish that matches organic chemistry journal figures (Angewandte, JACS, Nature). | **Medium**: Custom OpenGL fragment shader or edge-pass rendering outline detection. | **High** |
| **2. Interactive Reaction Path / IRC Profile Plot** | GaussView / Chemcraft | For IRC / optimization trajectories, display a 2D curve of Energy vs Reaction Coordinate. Clicking any point on the curve jumps the 3D viewer directly to that transition state / intermediate geometry. | **Medium**: Matplotlib or PyQtGraph interactive plot synchronized with the trajectory slider. | **High** |
| **3. Non-Covalent Interaction (NCI) Surface Visualizer** | Multiwfn / Avogadro | Isosurface of reduced density gradient (RDG) colored by $\text{sign}(\lambda_2)\rho$ (blue = attractive H-bond, green = van der Waals, red = steric clash). | **Medium**: Integrate directly with existing `dialog_nci_plot.py` and `multiwfn_analysis.py` for automated one-click run and visualization. | **Medium** |
| **4. Dipole Moment Vector Rendering** | GaussView / Avogadro | Render a 3D arrow showing the molecular dipole moment ($\mu$) rooted at the center of mass with magnitude label in Debye. | **Low-Medium**: QM parser already extracts `dipole_debye` and dipole components; add an OpenGL 3D cylinder/cone arrow. | **High** |
| **5. Atom Label Customization (PDB, Element, Index, Partial Charge, Force)** | VMD / GaussView | Toggleable HUD atom labels showing either: element symbol, atom index (1-based), Mulliken/Lowdin partial charge, or atom name. | **Low-Medium**: QPainter 2D overlay during `paintGL` projecting 3D positions to 2D screen text. | **High** |

---

## 🔬 Category 3: Advanced / Complex Features (Longer-Term Roadmap)

These are complex, specialized capabilities seen in heavy research suites:

| Feature | Inspired By | Technical Challenge | Recommendation |
| :--- | :--- | :--- | :--- |
| **1. Electrostatic Potential (ESP) mapped onto Electron Density Isosurface** | GaussView / Gabedit | Requires evaluating two 3D scalar fields simultaneously (grid of electron density for marching cubes isosurface + grid of ESP evaluated via nuclear + electronic potential integrals) and color-mapping vertex by vertex. | Postpone to dedicated volumetric release; existing 2D ESP slice dialog covers quick ESP needs. |
| **2. Real-Time Interactive Molecular Dynamics / Forcefield Pulling** | Avogadro 2 / VMD | User drags an atom in real-time with spring forces while an interactive UFF/MMFF forcefield optimizer runs in a background thread. | Requires continuous async physics loop and fast C++/OpenMM backend to avoid UI stutter. |
| **3. Secondary Structure Ribbon / Cartoon Diagrams for Biopolymers** | VMD / PyMOL / ChimeraX | Spline interpolation of peptide backbones ($\text{C}_\alpha$), $\beta$-sheet ribbons, and $\alpha$-helix cylinders for proteins/nucleic acids. | High complexity, specialized to macromolecules; agymolden is currently optimized for small molecules, organometallics, and quantum chemistry. |

---

## 🎯 Recommended Immediate Action Plan

For the immediate next step, the top easy wins can be implemented rapidly:
1. **Standard Camera Projections (XY, XZ, YZ, reset)** buttons/hotkeys (`Key_X`, `Key_Y`, `Key_Z`, `Key_Home`).
2. **Molecular Formula & Molecular Weight Status Badge** (always visible in bottom bar or title).
3. **Dipole Moment 3D Vector** display (toggleable arrow at center of mass for QM jobs).
4. **Centroid / Ring-to-Atom Distance** in the Shift+Click measurement engine.
5. **Invert / Mirror Coordinates** (chiral inversion tool).
