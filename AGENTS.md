# AgyMolden (`agymolden`) Project Rules

## 1. Purpose & Ecosystem Role

- **AgyMolden (`agymol`)** is a Molden-inspired 3D molecular builder, Z-matrix editor, orbital/density/surface viewer, and quantum chemistry (QM) input generator and runner.
- The user refers to `agymolden` as **`agymol`**; use `agymol` in conversation and honor it in new user-facing branding when requested, while keeping existing repo/package/CLI names (`agymolden`) intact for backwards compatibility unless an explicit rename migration is requested.
- Its independent companion application is **Agemacs** (`/home/andres/Programs/agemacs`), a text, LaTeX, and QM editor with a QM Output Navigator.
- Keep the two repositories separate, but treat **cross-app compatibility** as a first-class contract: never change CLI flags, IPC payloads, bundle layouts, or QM diagnostic conventions without checking how `agemacs` interacts with them.

## 2. Codebase Map

- **Main window & mixins**:
  - `agymolden/main_window.py`: `MainWindow` composed of `UIMixin` (`mw_ui.py`), `MenusMixin` (`mw_menus.py`), `EditingMixin` (`mw_editing.py`), `CalculationsMixin` (`mw_calculations.py`), `FileIOMixin` (`mw_file_io.py`), and `ReorderMixin` (`mw_reorder.py`).
- **Domain controllers**:
  - `agymolden/controllers/`: `export_controller.py`, `fragment_controller.py`, `import_controller.py`, `job_controller.py`, `orbital_controller.py`, `ring_controller.py`, `symmetry_controller.py`, `tweezers_controller.py`, `vibration_controller.py`.
- **QM calculation runners & diagnostics**:
  - `agymolden/calc_runner/`: `base.py` (`BaseCalcRunner`), `orca_runner.py`, `generic_runner.py` (Gaussian, PySCF, openLOWDIN, xTB, Q-Chem), `brorsen_runner.py`, `remote_ssh.py`, `preflight.py`, `diagnostics.py`.
- **Companion IPC & discovery**:
  - `agymolden/companion.py`: `QLocalSocket` IPC bridge to `agemacs` (server name `agemacs_companion_ipc`, JSON payload `{"action": "open", "files": [...]}`) and AppImage/CLI discovery (`find_agemacs_command`).
- **Themes & UI dialogs**:
  - `agymolden/theme.py` and `agymolden/dialog_*.py`: Qt6 dark/light theme system and dialogs (Output Viewer, 2D Sketcher, Quick ORCA, QM Input Generator, Command Palette, Onboarding, etc.).
- **Installers & packaging**:
  - `setup.sh`, `windows-installer.bat`, `scripts/make_oneclick_bundle.sh`, `AgyMolden_Student_Release.zip`, `pyproject.toml`.

## 3. Non-Negotiable Invariants ("Never Break These")

1. **No Conda / Anaconda**:
   - All Linux, macOS, and Windows student installers (`setup.sh`, `windows-installer.bat`) must work with standard Python `>=3.9` and `.venv` — never require Conda or Anaconda.
2. **Qt6 Dark & Light Themes**:
   - Every dialog and widget must respect live theme switching (`theme.py`).
   - Avoid QSS double-padding on spinboxes/comboboxes and light-on-white labels.
   - Preserve beginner-friendly menus and full-height right-docked Output Viewer / QM input generators.
3. **QM Subprocess Isolation**:
   - Never leak GUI `LD_PRELOAD` / `libstdc++` or AppImage/PyInstaller environment variables (`APPIMAGE`, `APPDIR`, `LD_LIBRARY_PATH` mounts, `PYTHONHOME`, `_MEIPASS`) into QM engine or companion subprocesses (`calc_runner/base.py`, `companion.py`).
   - Always quote and properly handle file paths containing spaces (especially for ORCA).
4. **Optional Dependencies**:
   - `pyscf` is optional and non-Windows (`sys_platform != 'win32'`); guard tests and imports cleanly so Windows and minimal environments never crash.
5. **GitHub Actions Storage Quota**:
   - Keep CI lightweight and manual-only (`workflow_dispatch`) for heavy or cross-OS builds to stay well under the 0.5 GB Actions storage limit. Never re-enable automatic heavy binary builds on push without asking.
6. **Companion Compatibility Rule**:
   - Any change to CLI arguments, file-open behavior, `companion.py`, bundle paths (`agymolden_env/Scripts/agymolden.exe`, AppImage naming), or QM output diagnostics/parsers MUST be checked against `/home/andres/Programs/agemacs` (`chem_tools.py`, `mainwindow.py`, `qm_navigator.py`, `qm_runner.py`).
7. **Uninstalled QM Codes Rule**:
   - When modifying input generators, CLI runners, or output parsers for QM codes not installed locally (e.g., Gaussian, Psi4, Q-Chem, openLOWDIN, xTB), cross-check authoritative companion reference codes (`cclib`, `ASE`, `QCEngine`, `Avogadro2`, official documentation) and add/update golden-fixture or mocked-runner tests (`tests/test_verified_input_generators.py`, `tests/test_parser_golden.py`, `tests/test_contract_binaries.py`).
8. **Unified Distribution & One-Click Launchers**:
   - Distribution `.zip` bundles for `agymol` (`agymolden`) and `agemacs` must share a consistent naming structure and internal layout/features.
   - Every OS installer must create both a **Desktop one-click launcher** and an **Applications / Start Menu entry** so users can launch directly after installing.
9. **Proactive Self-Service Error & Dependency Notices**:
   - Anticipate common environment failures (e.g., missing MPI/`mpirun` for parallel ORCA, missing QM binaries, missing OpenGL/system libraries, spaces in paths, missing companion app).
   - Surface clear, actionable in-app notices with 1-click fixes or exact OS-specific install commands so students can resolve issues themselves without messaging the maintainer.
10. **Git Sync & On-Request Zip Distribution**:
    - Store all project rules (`AGENTS.md`) and skills (`.agents/skills/*/SKILL.md`) inside the Git repo so `git pull` / `git push` synchronizes them across all machines.
    - Keep GitHub activity clean and low-usage (lightweight/manual-only CI, no unnecessary binary churn).
    - Only generate, update, or upload distribution `.zip` files to GitHub when the user explicitly requests it.
11. **Stringent Testing & Regression Discipline**:
    - Always run stringent tests before declaring work complete. Whenever a bug or issue is found and fixed, **add a targeted regression test** (with isolated state/`QSettings`) so it never recurs.
12. **Clean, Extensible Architecture (Refactor Early)**:
    - Keep code modular and easily extensible; do not wait to refactor when files, mixins, or dialogs become entangled.
13. **Anti-Clutter, Small-Screen & Intuitive Menu Policy**:
    - Proactively prevent UI clutter—especially on small/laptop screens (predict responsive layouts, collapsible panes, clean spacing, UI scaling/font controls).
    - Keep menus intuitive and user-configurable so each user can tailor how the app looks and persist preferences cleanly.
14. **Depth Over Bloat & Top-Tier Benchmarking**:
    - `agymol` already covers a wide feature set—avoid adding disconnected features; instead polish and strengthen existing workflows, benchmarking against top web/standalone tools and checking GitHub/reference ecosystems.
15. **Ample, Sufficient User Guides**:
    - Keep `README.md`, `USER_MANUAL.md`, and `STUDENT_GUIDE.md` thorough, clear, and updated whenever user-facing behavior changes.

## 4. Verification Commands

Run after code changes before declaring work complete:

```bash
# Fast headless test suite
QT_QPA_PLATFORM=offscreen .venv/bin/pytest -m "not slow and not network and not benchmark"

# Targeted UI / ORCA checks
QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_surfaces_and_orca_ui.py tests/test_ui_usability_fixes.py

# Lint (if ruff is installed in .venv)
.venv/bin/ruff check agymolden tests
```
