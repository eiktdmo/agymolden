# Advanced Feature Survey & Prioritization: Popular Molecular Visualization Software

A rigorous comparative review across **GaussView**, **Chemcraft**, **Avogadro 1/2**, **Gabedit**, **VMD**, **PyMOL**, **Mercury**, and **CYLview** identifies practical capabilities that directly improve daily productivity for computational and synthetic chemists.

---

## ⚡ Category 1: Easy Wins (Immediate High-Value Implementations)

These features have high user utility, integrate naturally into our existing PyQt6 / OpenGL / QM parser codebase, and require minimal lines of robust code.

| Feature | Software Benchmark | Description & Regular User Benefit | Implementation in AGYMolden |
| :--- | :--- | :--- | :--- |
| **1. "Copy Viewport Image to Clipboard" (Ctrl+Shift+C)** | GaussView / Chemcraft | Instant one-touch clipboard copy of the 3D molecular viewport (with or without transparent background). Enables pasting high-res 3D structures directly into Slack, PowerPoint, Word, or LaTeX without saving/navigating files. | `QApplication.clipboard().setPixmap(self.viewer.grabFramebuffer())`. |
| **2. Licorice / Tube & Wireframe Representation Modes** | VMD / PyMOL / Chemcraft | Complement existing "Ball & Stick" and "Spacefill" with **Licorice / Tube** (uniform cylinder radius, no atom spheres) and **Wireframe** styles. Essential when viewing large molecules, clusters, or complexes with dense coordination spheres. | Extend `self.style_mode` in `viewer.py`: `inst_data` radius / cylinder width scaling in the GPU instancing pass. |
| **3. Atom Label Display Mode Selector (HUD)** | GaussView / VMD / Mercury | Switch atom 3D HUD labels between: **Symbol only** (`C`), **Index only** (`1`), **Symbol + Index** (`C1`), **Partial Charge** (`+0.25e`), and **Off**. Currently only hardcoded combination is shown. | Dropdown in the Display Menu switching an enum in `paintGL()` / `QPainter` text overlay. |
| **4. Quick Invert Camera / Flip View (180° Backside View)** | Chemcraft / GaussView | A single button or hotkey (`Backquote` or `Alt+Z`) to flip the molecule 180° horizontally to see the exact backside without manual orbiting. Great for inspecting faces of aromatic rings or organometallic complexes. | Rotate `rot_quat` by 180° around the camera Y axis. |
| **5. Atom Element Periodic Table Quick-Replace Hover/Click** | GaussView / Avogadro | Clicking an atom while holding an element key or selecting from an inline quick-picker mutates the atom immediately with valence adjustment. (We have hotkeys `C`, `H`, `O`, `N`; adding an inline right-click "Change Element" popup). | Hook right-click context menu in `viewer_interaction.py` to open a 1-click mini-palette of common elements (C, H, O, N, F, Cl, Br, I, P, S). |

---

## 🏆 Category 2: Interesting Features (Medium Complexity, High Payoff)

These capabilities address major daily quantum chemistry analysis workflows.

| Feature | Software Benchmark | Description & Workflow Payoff | Technical Complexity | Priority |
| :--- | :--- | :--- | :--- | :--- |
| **1. Synchronized IRC / Optimization Energy Profile Plot** | GaussView / Chemcraft | An interactive 2D graph of $E$ vs. Step # below the Trajectory Player. Hovering or clicking points dynamically jumps the 3D viewport to that transition state or intermediate geometry. | **Medium**: Embed a small Matplotlib or PyQtGraph line plot synced to `self.slider_trajectory`. | **High (Priority 1)** |
| **2. CYLview / Nature-Style Illustration Preset** | CYLview / Chemcraft | Distinct non-photorealistic rendering: soft matte lighting, pastel element palettes, and sharp dark silhouette edges matching organic chemistry papers in *JACS* / *Angewandte*. | **Medium**: Custom lighting coefficients + outline shader pass or edge detection on depth buffer. | **High (Priority 2)** |
| **3. Automatic Hydrogen Bond Angle & Distance Table** | Mercury / Avogadro | A dockable table listing all detected non-covalent interactions: Donor—H⋯Acceptor with distances and angles (e.g. $\text{O}-\text{H}\cdots\text{O} = 1.85 \text{ \AA}, 168^\circ$). | **Medium**: Export detected H-bonds from `viewer.py` to a QTableWidget. | **Medium (Priority 3)** |
| **4. Coordinate-Constrained Distance/Angle Measurement HUD** | GaussView | When measuring distances/angles, allow the user to type a new target distance (e.g. change $1.54 \text{ \AA} \to 1.33 \text{ \AA}$) and automatically slide the dependent fragment. | **Medium**: Connect `viewer_interaction.py` measurement label to an editable HUD spinbox. | **Medium (Priority 4)** |

---

## 🔬 Category 3: Advanced / Hard-to-Implement Features (Longer-Term Roadmap)

These are complex features with specialized algorithmic requirements:

| Feature | Benchmark | Complexity & Challenges | Priority |
| :--- | :--- | :--- | :--- |
| **1. Electrostatic Potential (ESP) mapped on 3D Electron Density Isosurface** | GaussView / Gabedit | Requires evaluating two synchronized 3D grids: electron density (for Marching Cubes vertices) + ESP (analytic nuclear + electronic integrals evaluated at every mesh vertex), followed by vertex colormapping. | **Prioritize Later** (2D ESP slice viewer already covers quick qualitative ESP needs). |
| **2. Interactive Real-Time Forcefield / Spring Pulling** | Avogadro 2 / VMD | User drags an atom with a live spring while an MMFF/UFF optimizer runs continuously in a background loop. Requires thread-safe asynchronous geometry synchronization. | **Prioritize Later** (Quick UFF / MMFF clean buttons already provide fast optimization). |
| **3. Secondary Structure Ribbon / Cartoon Rendering** | PyMOL / VMD / ChimeraX | B-spline interpolation of $\text{C}_\alpha$ traces and helical cylinders for biopolymers. High implementation overhead for macromolecules, whereas AGYMolden is specialized for small molecules and organometallics. | **Low Priority** |

---

## 🎯 Recommended Next Immediate Step
Implement the **Top 3 Easy Wins**:
1. **Copy Viewport to Clipboard (`Ctrl+Shift+C`)** with one-touch confirmation.
2. **Licorice / Tube & Wireframe representation modes** in the style selector.
3. **Atom Label Display Mode Selector** (Symbol, Index, Symbol+Index, Charge, Off).
