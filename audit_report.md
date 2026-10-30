# AgyMolden Code Audit & Student Deployment Evaluation

**Target Audience:**
1. **Immediate (Group deployment):** Research group students running **newBRORSEN** and **ORCA** (including advanced NEO/positronic workflows, Conda environments, and cluster/SSH execution).
2. **Upcoming (Class deployment in ~2 weeks):** Computational Chemistry students running **ORCA** (focusing on beginner ergonomics, one-click execution, error guardrails, and zero-configuration setups).

---

## 1. Executive Summary

AgyMolden is an exceptionally rich, feature-dense computational chemistry package (35,000+ lines of Python) that bridges the classic Z-matrix and orbital inspection strengths of **Molden / GaussView** with modern Python Qt6 GUI capabilities, real-time quantum chemistry job execution, and specialized multicomponent/positronic features (newBRORSEN).

- **Test Suite Status:** 107 of 107 tests passing (`test_calc_runner`, `test_gui_smoke`, `test_nist_and_qm_detect`, etc.).
- **Code Architecture:** Modularized into `calc_runner`, `brorsen_builder`, `viewer`, `mw_calculations`, and `qm_detect`.
- **Primary Bottlenecks for Students:**
  1. *Beginner Usability:* Dual menus and deep dialogs for ORCA vs. newBRORSEN can confuse undergraduate students who just want to draw a molecule, pick DFT/B3LYP, and hit "Run".
  2. *Executable Distribution Size:* The onedir dist is ~327 MB (including full SciPy/OpenGL/Qt6 stack). The source tree contains large caches, temporary build artifacts, and an untracked 450 KB monolithic script (`newBRORSEN.v04.09.py`).
  3. *Error Diagnostics:* If ORCA fails due to missing OpenMPI or basis set syntax, the runner captures output to log, but undergraduates need intuitive diagnostic popups.

---

## 2. Speed Audit

### Strengths
- **Native Qt Subprocesses:** Calculation jobs (`orca_runner.py`, `brorsen_runner.py`) use non-blocking `QtCore.QProcess`. Real-time line parsing uses streaming regexes without freezing the GUI.
- **OpenGL 3D Viewport:** Hardware-accelerated OpenGL widget handles molecules and isosurfaces smoothly.
- **Environment & Binary Caching:** `qm_detect.py` and `brorsen_runner.py` implement internal caching (`_ENV_CACHE`, `_WRAPPER_CACHE`) to avoid re-scanning the disk or conda prefixes repeatedly.

### Speed Bottlenecks & Optimization Opportunities
1. **Z-Matrix Re-computation on Large Systems:** Internal-to-Cartesian conversion in `zmatrix.py` is pure Python. For small molecules (under 50 atoms) this is instantaneous (< 1 ms), but for large organic complexes or bio-fragments, vectorized NumPy transforms are recommended.
2. **PyInstaller Startup Time:** A 327 MB onedir application launches in ~1.5–2.5 s on Linux NVMe SSDs, but onefile mode on mechanical hard drives or slower student laptops can take 8–15 seconds to unpack `/tmp/_MEIxxxxxx`. **Retaining `onedir` or distributing via pip/wheel in a micromamba environment is recommended.**
3. **Log File Streaming Overhead:** In `orca_runner.py` and `base.py`, line-by-line Qt signal emissions during high-verbosity ORCA output (e.g. large SCF or TD-DFT runs) can flood the Qt event queue. Adding a 50 ms log throttling buffer prevents UI stutter.

---

## 3. Usability Audit

### Evaluation for Research Group (Phase 1: newBRORSEN + ORCA)
- **Status:** **High Readiness.**
- The research group benefits directly from `brorsen_builder`, the Positronic/NEO schema dialogs, conda environment detection, and remote SSH job launcher (`remote_ssh.py`).
- **Gaps to Address:**
  - `newBRORSEN.v04.09.py` in the root workspace should have its path auto-detected if located next to `agymolden` or in `PATH`.
  - When switching between conda environments, a visual badge indicating the active environment in the status bar will prevent accidental execution with the wrong PySCF/BRORSEN interpreter.

### Evaluation for Computational Chemistry Course (Phase 2: ORCA in ~2 weeks)
- **Status:** **Needs Simplification.**
- Undergraduates are easily overwhelmed by advanced tabs (e.g. active spaces, bespoke basis sets, remote SSH clusters).
- **Key Pain Points:**
  1. *Workflow Discovery:* To run ORCA, students currently have to navigate to **File → Generate QM Calculation Input (Ctrl+G)**, set ORCA, click "Run Now", and then manage the separate **Job Monitor**. A prominent toolbar button: **"🚀 Run ORCA"** or a streamlined "Quick ORCA Run" preset is needed.
  2. *Missing Binary Guidance:* If ORCA is not installed or not in `PATH`, the app should pop up an actionable guide ("ORCA executable not found. Would you like to select your ORCA folder or check our quick installation guide?").
  3. *Presets for Lab Exercises:* Undergraduate labs typically consist of standard calculations:
     - Geometry Optimization (e.g., `B3LYP def2-SVP Opt`)
     - Vibrational Frequencies & IR (e.g., `B3LYP def2-SVP Freq`)
     - Single Point Energy / Frontier Orbitals (HOMO-LUMO)
     Pre-configuring these as 1-click drop-down presets in the QM dialog will eliminate 90% of student syntax errors.

---

## 4. Scope & Competitive Analysis

| Capability | AgyMolden | Molden / GMolden | GaussView 6 | Avogadro 2 | WebMO |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Z-Matrix Building & Live Sliders** | ⭐⭐⭐⭐⭐ (Excellent) | ⭐⭐⭐⭐ (Original, but dated X11) | ⭐⭐⭐ (Secondary) | ⭐⭐ (Poor Z-matrix) | ⭐⭐ (Basic) |
| **newBRORSEN / Positron / NEO Setup** | ⭐⭐⭐⭐⭐ (Unique) | ❌ (None) | ❌ (None) | ❌ (None) | ❌ (None) |
| **ORCA Input Generation & Execution** | ⭐⭐⭐⭐ (Strong) | ⭐ (Manual) | ⭐ (Gaussian-centric) | ⭐⭐⭐ (Good via plugin) | ⭐⭐⭐⭐ (Good) |
| **Live Job Monitoring & Live 3D Trajectory** | ⭐⭐⭐⭐ (Real-time Qt Process) | ❌ (Post-processing only) | ⭐⭐⭐ (Requires Gaussian) | ⭐⭐ (External file reload) | ⭐⭐⭐⭐ (Server-side) |
| **IR / Vibrational Animation** | ⭐⭐⭐⭐ (Interactive spectrum + modes) | ⭐⭐⭐ (Classic) | ⭐⭐⭐⭐⭐ (Industry standard) | ⭐⭐⭐⭐ (Good) | ⭐⭐⭐ (Basic) |
| **Platform Portability & Licensing** | ⭐⭐⭐⭐⭐ (Open Source, Python/Qt6) | ⭐⭐ (Old X11/Motif) | ⭐ (Proprietary / Expensive) | ⭐⭐⭐⭐⭐ (Open source C++) | ⭐⭐ (Commercial server) |

**Competitive Edge:** AgyMolden uniquely combines Molden's pure internal coordinate manipulation with modern GaussView-level interactive editing and a dedicated multi-component/positronic QM engine (newBRORSEN), completely outclassing classic Molden in graphical polish and modern Python extensibility.

---

## 5. Size & Footprint Audit

- **Python Codebase:** 35,167 lines across 50+ files. Well-structured into subpackages.
- **Repository Cleanliness Issues:**
  - `.venv/`: ~1.5 GB (standard for local dev environment; ensure it is strictly ignored).
  - `build/`: ~51 MB (temporary PyInstaller build files).
  - `scratch/`: ~52 MB (stray benchmark files and scratch runs).
  - `dist/agymolden`: 327 MB.
- **Distribution Optimization:**
  - The PyInstaller distribution includes heavy libraries (Qt6, PyOpenGL, NumPy, SciPy).
  - For classroom distribution on student machines (Windows/Mac/Linux), providing a clean single-file installer or a dedicated script (`conda env create -f environment-student.yml`) keeps download sizes under 150 MB compressed.

---

## 6. Actionable Recommendations & Prioritized Roadmap

```mermaid
flowchart TD
    subgraph Priority 1: Research Group (Immediate)
        A1["Verify newBRORSEN execution & conda auto-detection"] --> A2["Auto-locate newBRORSEN.v04.09.py"]
        A2 --> A3["Clean root directory & package sample jobs"]
    end
    subgraph Priority 2: Comp Chem Course (Week 1 - 2)
        B1["Add 'Quick ORCA Run' Toolbar Shortcut & Presets"] --> B2["Add friendly error popup if ORCA / MPI fails"]
        B2 --> B3["Prepare 1-page Student Quickstart Guide"]
    end
    subgraph Priority 3: Distribution & Performance (Ongoing)
        C1["Prune root scratch/build artifacts"] --> C2["Package cross-platform zip/installers"]
    end
```

### Priority 1: Research Group Readiness (Immediate – Next 48 Hours)
1. **Automatic Path Resolution for newBRORSEN:** Ensure `brorsen_runner.py` automatically checks the current workspace directory for `newBRORSEN.v04.09.py` if no custom path is chosen.
2. **Verification of Sample Inputs:** Bundle 2–3 standard test calculations (`examples/water_orca.inp`, `examples/h2o_brorsen.inp`) so group members can run a 10-second verification calculation immediately after setup.
3. **Conda Environment Selector Quick-check:** Ensure students get an immediate green checkmark in the UI confirming `pyscf` and required libraries are present in their chosen conda environment.

### Priority 2: Computational Chemistry Course Readiness (Week 1–2)
1. **"Quick ORCA" 1-Click Workflow:**
   - Add a direct toolbar action: **"Run ORCA Calculation"**.
   - Include 3 educational presets:
     - *Preset 1: Geometry Optimization (`! B3LYP def2-SVP Opt`)*
     - *Preset 2: Vibrational Frequencies & IR (`! B3LYP def2-SVP Freq`)*
     - *Preset 3: Orbital Calculation (`! HF/6-31G* SP`)*
2. **Student-Friendly Error Interceptor:**
   - When ORCA exits with a non-zero code, scan the `.out` file for classic errors (e.g. *Basis set not recognized*, *SCF did not converge*, *Memory limit exceeded*, *Parallel MPI error*) and display a plain-English explanation dialog rather than just a red log line.
3. **Student Quick-Start Card (PDF/Markdown):**
   - A concise 2-page guide: "1. Build molecule -> 2. Click Run ORCA -> 3. View Frequencies / Orbitals".

### Priority 3: Polish & Distribution (Week 2+)
1. **Student Installer Script (`setup_student.sh` / `setup_student.bat`):** A lightweight script that checks Python, installs dependencies, and creates a desktop shortcut without needing developer tools.
2. **Cleanup Repository:** Exclude or archive root-level test input files (`C4H4O_SP_HF_cc-pvdz.inp`, `CH4_SP_HF_cc-pvdz.inp`, `oh.zmat`, `bp.zmat`) into an `examples/student_lab/` directory for tidy distribution.
