---
name: agy-companion-ecosystem
description: >-
  Cross-app compatibility and Quantum Chemistry (QM) reference-verification
  runbook for the agymol (agymolden) and agemacs companion ecosystem. Use when
  modifying IPC, CLI launch/discovery, bundle layouts, Desktop/Applications
  launchers, QM input generators, QM runners, or QM output parsers/diagnostics—
  especially for QM codes not installed locally.
---

# AgyMol (`agymolden`) ↔ Agemacs Companion Ecosystem Runbook

`agymol` (`/home/andres/Programs/agymolden`) and `agemacs` (`/home/andres/Programs/agemacs`) are **separate repositories** designed as first-class companions. Any change to shared contracts in one repo must be verified against the other.

---

## 1. Companion Compatibility Contract (`agymol` ↔ `agemacs`)

### `agymol` → `agemacs` (`agymolden/companion.py`)
- **IPC**: `QLocalSocket` connecting to server name `agemacs_companion_ipc`, sending JSON `{"action": "open", "files": [...]}`.
- **Clean Subprocess Env**: `_clean_subprocess_env()` strips `APPIMAGE`, `APPDIR`, `OWD`, `ARGV0`, `PYTHONHOME`, `PYTHONPATH`, `_MEIPASS`, `_MEIPASS2`, and `/tmp/.mount_` `LD_LIBRARY_PATH` entries.
- **Discovery (`find_agemacs_command`)**: Checks `AGEMACS_BIN`, `AGEMACS_APPIMAGE`, `PATH` (`agemacs`), AppImages matching `*agemacs*.appimage` / `*agyeditor*.appimage`, `AGEMACS_DIR`, and standard repo/install folders.

### `agemacs` → `agymol` (`chem_tools.py`, `mainwindow.py`, `qm_navigator.py`)
- **IPC Listener**: `mainwindow.py` runs `QLocalServer("agemacs_companion_ipc")` and handles `{"action": "open", "files": [...]}`.
- **Discovery (`find_agymolden_command`, `_resolve_agymolden_in_dir`)**: Checks `AGYMOLDEN_BIN`, `AGYMOLDEN_APPIMAGE`, `PATH` (`agymolden`), AppImages, `AGYMOLDEN_DIR`, and Windows bundle path `agymolden_env/Scripts/agymolden.exe`.
- **Handoffs**: `open_calculation_in_agymolden`, `_open_xyz_text_in_agymolden`, `_resolve_agymolden_calc_file` (sibling `.out`/`.log` resolution from `.inp`), and QC Doctor diagnostic parity in `qm_navigator.py`.

### Cross-Repo Impact Checklist
Whenever you modify any item on the left, inspect and test the paired files on the right:

| Change Area | `agymol` Files | `agemacs` Files |
| :--- | :--- | :--- |
| IPC / File Open / CLI Args | `agymolden/companion.py`, `cli.py`, `main_window.py` | `mainwindow.py`, `chem_tools.py`, `doctor.py` |
| Bundle Layout / Installers | `windows-installer.bat`, `setup.sh`, `scripts/make_oneclick_bundle.sh` | `install.sh`, `windows-installer.bat`, `scripts/make_oneclick_bundle.sh` |
| QM Diagnostics / Output Nav | `agymolden/calc_runner/diagnostics.py` | `qm_navigator.py`, `test_agemacs.py` |
| QM Runners / Env Isolation | `agymolden/calc_runner/base.py`, `generic_runner.py`, `orca_runner.py` | `qm_runner.py` |

---

## 2. Unified Distribution & One-Click Launcher Standard
- **Same Zip Structure & Features**: Distribution `.zip` bundles for `agymol` and `agemacs` must share consistent naming conventions, internal directory structure, and installer UX (no Conda/Anaconda, Python `>=3.9` + `.venv`).
- **One-Click Launchers on Every OS**: Installers must create both a **Desktop one-click launcher** and an **Applications / Start Menu entry**.
- **On-Request Zip Builds & Clean GitHub**: Only generate or upload distribution `.zip` files when explicitly requested by the user. Keep GitHub Actions manual-only (`workflow_dispatch`) for heavy/cross-OS builds to stay well under the 0.5 GB storage quota.

---

## 3. Uninstalled QM Code Learning & Verification Protocol
Many supported QM codes (e.g., Gaussian, Psi4, Q-Chem, openLOWDIN, xTB, NWChem, GAMESS) may not be installed on the current machine. **Never guess syntax or CLI flags.** Follow this 3-step protocol:

1. **Learn from Authoritative Companion Reference Code**:
   - Inspect `cclib` and `ase` (if installed in `.venv`) or check official documentation / `QCEngine` / `Avogadro2` reference formats for exact input syntax, CLI flags, environment variables, and output markers.
2. **Enforce Parity & Self-Service Error Handling**:
   - Keep diagnostic remedies aligned between `agymolden/calc_runner/diagnostics.py` and `agemacs/qm_navigator.py`.
   - Surface clear, actionable self-service notices (e.g., missing `mpirun`, missing binary, spaces in paths) with 1-click fixes or exact OS install commands.
3. **Add Golden-Fixture & Mocked-Runner Tests**:
   - In `agymol`: update `tests/test_verified_input_generators.py`, `tests/test_parser_golden.py`, and `tests/test_contract_binaries.py` (using `pytest.skip` only for live-binary tests when uninstalled, while keeping mocked/fixture tests always active).
   - In `agemacs`: add/update fixture tests in `test_agemacs.py`.

---

## 4. Cross-Repo Verification Commands
When any companion-contract file changes, run both fast suites:
```bash
# In /home/andres/Programs/agymolden
QT_QPA_PLATFORM=offscreen .venv/bin/pytest -m "not slow and not network and not benchmark"

# In /home/andres/Programs/agemacs
QT_QPA_PLATFORM=offscreen .venv/bin/python test_agemacs.py
```
