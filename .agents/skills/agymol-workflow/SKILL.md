---
name: agymol-workflow
description: >-
  Maintainer runbook and cross-OS checklist for AgyMolden (agymol). Use when
  setting up local/global agymol executables and Desktop/Applications launchers,
  auditing Qt6 dark/light themes, debugging ORCA/PySCF/Molden workflows, adding
  proactive self-service dependency notices (e.g., missing MPI/mpirun, OpenGL,
  QM binaries), or generating/uploading student distribution .zip bundles on
  explicit user request.
---

# AgyMolden (`agymol`) Maintainer Workflow

## 1. Canonical Location & Immediate Execution
- **Canonical repo**: `/home/andres/Programs/agymolden` (referred to as `agymol` in conversation; keep `agymolden` package/CLI names intact for compatibility).
- **No Conda / Anaconda**: Installers (`setup.sh` for Linux/macOS, `windows-installer.bat` for Windows) must use standard Python `>=3.9` and `.venv`.
- **One-Click Launchers**: Every OS installer must create **both**:
  1. A **Desktop one-click runner** (`AgyMolden.desktop` / `.bat` / `.command`).
  2. An **Applications / Start Menu entry** so users can launch `agymol` directly from their OS app launcher.
- **Immediate CLI Execution**: Ensure `agymolden` (and `agymol` alias if configured) is runnable from `~/.local/bin` / `.venv/bin` without activating `.venv` manually.

## 2. Proactive Self-Service Dependency Notices
Anticipate common student/user environment failures and surface **actionable in-app notices** (1-click fixes or exact copy-paste OS install commands) so users can solve them without messaging the maintainer:
- **Parallel ORCA / Missing MPI (`mpirun`)**: Detect missing `mpirun` / OpenMPI before or on failure; offer 1-click fallback to serial (`nprocs=1`) or show the exact OS install command (`sudo apt install openmpi-bin`, `brew install open-mpi`, etc.).
- **Paths with Spaces**: Always quote paths in ORCA and subprocess invocations (`calc_runner/orca_runner.py`, `calc_runner/base.py`).
- **Missing QM Binaries / `pyscf` / OpenGL libs**: Detect cleanly via `qm_detect.py` / `preflight.py`, show exact install instructions, and guard optional `pyscf` (`sys_platform != 'win32'`).
- **Subprocess Env Isolation**: Strip GUI `LD_PRELOAD` / `libstdc++` and AppImage/PyInstaller vars (`APPIMAGE`, `APPDIR`, `LD_LIBRARY_PATH` mounts, `PYTHONHOME`, `_MEIPASS`) in `calc_runner/base.py` and `companion.py`.
- **Missing `agemacs` Companion**: Show clear discovery/install hint if `find_agemacs_command()` returns `None`.

## 3. Qt6 Theme & Usability Audit Checklist
- Test both **Dark** and **Light** modes (`agymolden/theme.py`) on every modified dialog/widget:
  - No QSS double-padding on `QSpinBox` / `QComboBox`.
  - No light-on-white or dark-on-dark unreadable labels/buttons.
  - Live theme switching updates open dialogs (Output Viewer, Command Palette, Onboarding, etc.).
- Preserve beginner-friendly menus (`MenusMixin` in `mw_menus.py`) and full-height right-docked Output Viewer / QM input generators.
- Verify bond-order sanitization & stale double-bond clearing on preset/file load (`viewer.py`, `mw_file_io.py`).

## 4. On-Request Distribution Zip & Low-Usage GitHub Checklist
- **Only** generate, update, or upload `AgyMolden_Student_Release.zip` / OneClick bundles (`scripts/make_oneclick_bundle.sh`) when the user **explicitly requests** it.
- Keep `.zip` naming structure and internal layout/features unified with `agemacs`.
- Keep GitHub Actions lightweight and manual-only (`workflow_dispatch`) for heavy/cross-OS builds to stay well under the 0.5 GB Actions storage limit.
- **Verification before commit/push**:
  ```bash
  QT_QPA_PLATFORM=offscreen .venv/bin/pytest -m "not slow and not network and not benchmark"
  QT_QPA_PLATFORM=offscreen .venv/bin/pytest tests/test_surfaces_and_orca_ui.py tests/test_ui_usability_fixes.py
  .venv/bin/ruff check agymolden tests
  ```

## 5. Quality, Anti-Clutter & Full Audits
- **Regression tests**: Always add an isolated regression test (using temporary `QSettings` / temp dirs) whenever fixing a bug.
- **Refactor early**: Keep mixins, controllers, and dialogs modular and extensible—do not wait to refactor when code becomes entangled.
- **Anti-clutter & depth over bloat**: Optimize for small/laptop screens, keep menus intuitive and user-configurable, strengthen existing workflows against top web/standalone tools rather than adding disconnected features, and keep `README.md`, `USER_MANUAL.md`, and `STUDENT_GUIDE.md` ample and up to date.
- **Full audits**: Load `agy-full-audit` when asked to "run a full audit on agymol".
