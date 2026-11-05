# AgyMolden Engineering Roadmap & Improvement Plan

This document synthesizes the strategic development plan to establish **AgyMolden** as the definitive free, cross-platform Quantum Chemistry workbench and internal-coordinate modeling environment.

---

## 🧭 Core Positioning: The Quantum Chemistry Workflow Companion

AgyMolden focuses on a unified, high-reliability workflow:
$$\text{Build / Z-Matrix} \longrightarrow \text{Pre-flight Lint} \longrightarrow \text{Launch (Local/SSH)} \longrightarrow \text{Monitor} \longrightarrow \text{Diagnose (QC Doctor)} \longrightarrow \text{Analyze (Orbitals/ESP/IRC)} \longrightarrow \text{Publish (SI Bundle)}$$

---

## 📊 Development Tiers (Ordered from Easy to Hard)

### Tier 1: Immediate Polish & UI Quick Wins (Effort: S, < 1 hr each)
- [x] **Repo Hygiene**: Clean loose calculation outputs, checkpoint files, and temporary artifacts out of root into `examples/` and `.gitignore`.
- [x] **Documentation Consolidation**: Unify fragmented audit and roadmap files into `docs/ROADMAP.md` and archive historical audits in `docs/audits/`.
- [ ] **README Accuracy**: Update test count badges (265 passing), verified Python versions, and install instructions.
- [ ] **Startup Diagnostics**: Add a structured one-line startup log with versions (Python, PyQt6, OpenGL) and detected external quantum engines (ORCA, Gaussian, xTB, PySCF, RDKit).
- [ ] **Shortcut Cheat Sheet Sync**: Register `Ctrl+Alt+P` (ESP 3D Map) and `Ctrl+Alt+O` (Volumetric Orbitals Studio) in the in-app shortcuts dialog.
- [ ] **Universal Small-Screen Responsiveness**: Extend compact padding and minimum-width guarantees across all inspector tabs and child dialogs.
- [ ] **ORCA Compatibility Guardrails**: Add tooltips for functionals/basis sets and inline warnings for invalid combinations.

### Tier 2: Code Reliability & Test Hardening (Effort: M, 2–8 hrs each)
- [ ] **Targeted Exception Handling**: Replace 180 broad `except Exception` blocks with specific exceptions and route unexpected errors to a central `report_error()` UI toast/status bar.
- [ ] **Type Annotation Coverage**: Progressively raise return/parameter type annotations from 22% to >60%, integrated with `mypy --check-untyped-defs`.
- [ ] **Automated Linting & Formatting**: Enforce `ruff` and `pre-commit` hooks in CI.
- [ ] **Multi-Resolution Layout Regression Tests**: Headless testing of window geometries (800×600, 1024×768, 1366×768) verifying scroll area engagement and zero widget truncation.
- [ ] **Builder Invariant Property Testing**: Hypothesis/parametrized tests asserting that atom addition, ring fusion, and peri-annulation strictly enforce:
  - Atomic clearance: $d_{\min} \ge 0.80\text{ \AA}$
  - Conservation of valency
  - Exact expected molecular formula (e.g. Naphthalene + Benzene bay fusion $\to$ Phenalene $\text{C}_{13}\text{H}_{10}$).
- [ ] **Round-Trip Format Preservation**: Verify lossless Cartesian round-tripping across ZMAT $\leftrightarrow$ XYZ $\leftrightarrow$ SDF $\leftrightarrow$ PDB.
- [ ] **Verified Input Generators**: Validate generated Gaussian, ORCA, and newBRORSEN decks against keyword syntax linting.
- [ ] **Full Undo/Redo Test Matrix**: Ensure every builder mutation (add atom, delete, bond, reorder, substitute, weld) is reversibly covered by undo/redo.

### Tier 3: Architecture & Modeling Capabilities (Effort: L, 1–3 days each)
- [ ] **Software-OpenGL CI Testing**: Run offscreen GUI tests with Mesa llvmpipe / Xvfb to capture OpenGL framebuffers and verify rendering and picking without physical GPUs.
- [ ] **Decoupled Architecture**: Refactor large mixins (`mw_editing.py`, `mw_calculations.py`, `mw_ui.py`) into independent controllers around a central event bus.
- [ ] **Unified Molecule State**: Establish a single authoritative `MoleculeModel` managing Cartesian coordinates, internal connectivity, bond orders, and stereocenters.
- [ ] **Hybrid Optimization Backends**: Provide seamless fallback between analytical UFF/MMFF94s, RDKit (when installed), and semiempirical GFN0-xTB.
- [ ] **Job Queue & Cluster Schedulers**: Integrated SLURM/PBS script generator, job submission, status polling, and automatic result loading.
- [ ] **Advanced Volumetric Rendering**: GPU ray-casting or depth-sorted impostor spheres/cylinders, ambient occlusion, and publication-ready SVG/vector export.

### Tier 4: Strategic & Ecosystem Growth (Effort: XL, 1–4 weeks each)
- [ ] **Plugin Ecosystem**: Versioned plugin API allowing user-defined Python scripts to add analysis tabs, file importers, and custom calculators.
- [ ] **Deep Wavefunction Analysis**: Broaden integration with Multiwfn and QTAIM topology, ELF/LOL real-space grids, and energy decomposition analysis.
- [ ] **Macromolecular & Extended Solids Support**: Accelerated spatial partitioning for systems $>10,000$ atoms and space-group symmetry replication.
- [ ] **Automated Multi-OS Packaging**: Reproducible GitHub Actions workflow publishing standalone AppImage, macOS `.dmg`, and Windows portable executables.
- [ ] **Empirical Usability Benchmarking**: Real-world user-experience studies measuring time-to-first-calculation for chemistry students and researchers.
