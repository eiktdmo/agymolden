---
name: agy-full-audit
description: >-
  Structured full-audit runbook for agymol (agymolden) and agemacs. Use when
  the user asks for a full audit, code quality or de-entanglement review,
  anti-clutter or small-screen UI and menu audit, competitive benchmark against
  top web/standalone tools, stringent test coverage audit, or user-guide
  completeness check.
---

# AgyMol & Agemacs Full-Audit Runbook

Run all 7 pillars systematically when performing a full audit, and present findings ranked **P0 → P1 → P2** with concrete safeguards and regression tests.

## 1. Code Health & De-Entanglement
- Identify oversized, deeply nested, or entangled modules (e.g., large `MainWindow` mixins, monolithic dialogs, duplicated runner/parser logic).
- Propose or execute clean extractions into focused modules/helpers so the codebase stays modular and easy to extend—**refactor early** rather than letting coupling grow.

## 2. Stringent Testing & Regression Coverage
- Run the full fast headless test suite plus lint/AST checks:
  - `agymol`: `QT_QPA_PLATFORM=offscreen .venv/bin/pytest -m "not slow and not network and not benchmark"` and `.venv/bin/ruff check agymolden tests`
  - `agemacs`: `QT_QPA_PLATFORM=offscreen .venv/bin/python test_agemacs.py`, `.venv/bin/python doctor.py`, and Python 3.10 AST check
- Verify every bugfix or newly discovered edge case has an **isolated regression test** (mocking `QSettings` / temp dirs so local user preferences are never read or polluted).

## 3. Anti-Clutter, Small-Screen & Menu Intuitiveness
- Audit layouts at small/laptop resolutions (e.g., `1280×720` / `1366×768`): check for clipped toolbars, cramped docks, overlapping panes, or hidden action buttons.
- Verify **Dark & Light themes** for contrast, readable icons/labels, and zero QSS double-padding.
- Check live font-size / UI-scale controls and user-configurable appearance settings so each user can tailor how the app looks.
- Audit menus for first-time user intuitiveness (logical grouping, clear labels, shortcuts, zero clutter).

## 4. Companion & QM Ecosystem Parity
- Cross-check `agymol` ↔ `agemacs` IPC (`agemacs_companion_ipc`), discovery, bundle layouts, and QC Doctor / Output Navigator parity (see `agy-companion-ecosystem`).
- For QM codes not installed locally, verify input generators, runners, and parsers against authoritative companion reference codes (`cclib`, `ASE`, `QCEngine`, `Avogadro2`) and golden fixtures.

## 5. Competitive Benchmark & Depth-Over-Bloat
- Benchmark core workflows against top web and standalone alternatives (and check GitHub/reference ecosystems):
  - `agymol` vs. Avogadro2, IQmol, GaussView, MolView, Multiwfn, ORCA tools.
  - `agemacs` vs. Overleaf, VS Code, Emacs/AUCTeX.
- **Depth over bloat**: avoid adding disconnected features; focus on making existing workflows faster, cleaner, more reliable, and visually sharper.

## 6. Proactive Self-Service Notices & User Guides
- Verify common environment failures (missing `mpirun`/MPI, LaTeX, QM binaries, OpenGL libs, spaces in paths, missing companion app) produce clear, self-service in-app notices with 1-click fixes or exact OS install commands.
- Audit user documentation (`README.md`, `USER_MANUAL.md`, `STUDENT_GUIDE.md` in `agymol`; `README.md` / `F1` help in `agemacs`) to ensure it is ample, accurate, and easy for students to follow.

## 7. Prioritized Safeguarded Execution Plan
- **P0 (Critical)**: Crashes, data loss, broken IPC/QM runners, failing tests, or unreadable UI.
- **P1 (High)**: Entangled code needing refactoring, small-screen clutter, missing regression tests, or weak self-service notices.
- **P2 (Polish)**: Menu ergonomics, competitive UX polish, and user-guide depth.
- Implement fixes with safeguards, add regression tests for every fix, and re-run stringent tests before declaring done.
