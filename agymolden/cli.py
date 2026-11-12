"""
Headless Command-Line Interface (CLI) for AgyMolden.

Provides fast terminal-based tools for HPC clusters and scripts without requiring an X11 display:
- `agymolden parse <logfile>`: Parse QM logs (Gaussian, ORCA, newBRORSEN, openLOWDIN, PySCF) to terminal summary or JSON.
- `agymolden gen <struct>`: Generate production QM inputs (Gaussian, ORCA, PySCF) with pre-flight checks.
- `agymolden check <input_or_struct>`: Pre-flight sanity validation of coordinates, charge/multiplicity parity, and memory.
- `agymolden convert <input> -o <output>`: Format conversion between XYZ and Z-Matrix.
"""

import argparse
import json
import os
import sys
from typing import List, Optional
import numpy as np

from .qm_log_parser import parse_qm_logfile
from .zmatrix import parse_xyz, parse_zmatrix_text, format_zmatrix_text, cartesian_to_zmatrix, zmatrix_to_cartesian
from .chem_identifiers import format_xyz_string
from .qm_input_generator import (
    QMCalculationParameters,
    QM_GENERATORS,
    GaussianGenerator,
    OrcaGenerator,
    PySCFGenerator,
)
from .calc_runner.preflight import validate_qm_calculation


def cmd_parse(args: argparse.Namespace) -> int:
    """Parse QM output log and output summary or JSON."""
    if not os.path.isfile(args.file):
        sys.stderr.write(f"Error: file not found: {args.file}\n")
        return 1

    result = parse_qm_logfile(args.file)
    if result is None:
        sys.stderr.write(f"Error: unable to parse QM log file: {args.file}\n")
        return 1

    final_e = result.trajectory_energies[-1] if result.trajectory_energies else None
    freqs = list(result.vibrational_analysis.frequencies) if result.vibrational_analysis and getattr(result.vibrational_analysis, "frequencies", None) is not None else []
    gap_ev = (result.lumo_ev - result.homo_ev) if (result.homo_ev is not None and result.lumo_ev is not None) else None
    normal_term = result.converged or ("Normal termination" in result.comment)

    data = {
        "file": os.path.abspath(args.file),
        "program": result.program_name or "Unknown",
        "normal_termination": normal_term,
        "n_atoms": len(result.symbols),
        "formula": "".join(f"{s}{result.symbols.count(s) if result.symbols.count(s) > 1 else ''}" for s in sorted(set(result.symbols))),
        "final_energy_ha": final_e,
        "energies_trajectory": result.trajectory_energies,
        "n_optimization_steps": len(result.trajectory_coords),
        "converged": result.converged,
        "homo_ev": result.homo_ev,
        "lumo_ev": result.lumo_ev,
        "gap_ev": gap_ev,
        "dipole_debye": result.dipole_debye,
        "frequencies_cm1": freqs,
        "n_imaginary_frequencies": len([f for f in freqs if f < -1.0]),
        "zpe_ha": result.zpe_ha,
        "enthalpy_ha": result.enthalpy_ha,
        "gibbs_ha": result.gibbs_ha,
        "is_irc": result.is_irc,
        "comment": result.comment,
    }

    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    # Human-readable summary
    term_status = "✔ Normal Termination" if normal_term else "✖ Abnormal / Incomplete Termination"
    conv_status = "Converged" if result.converged else "Not Converged"
    print(f"============================================================")
    print(f" AgyMolden QM Log Summary: {os.path.basename(args.file)}")
    print(f"============================================================")
    print(f" Program:               {result.program_name}")
    print(f" Status:                {term_status} ({conv_status})")
    print(f" Molecule:              {data['formula']} ({data['n_atoms']} atoms)")
    if final_e is not None:
        print(f" Final Energy:          {final_e:.8f} Ha")
    if len(result.trajectory_coords) > 1:
        print(f" Optimization Steps:    {len(result.trajectory_coords)} steps")
    if gap_ev is not None:
        print(f" Frontier Gap:          HOMO: {result.homo_ev:+.2f} eV  |  LUMO: {result.lumo_ev:+.2f} eV  |  ΔE: {gap_ev:.2f} eV")
    if result.dipole_debye is not None:
        print(f" Dipole Moment:         {result.dipole_debye:.3f} Debye")
    if freqs:
        n_imag = len([f for f in freqs if f < -1.0])
        print(f" Vibrational Modes:     {len(freqs)} frequencies (Imaginary: {n_imag})")
        if n_imag > 0:
            imag_vals = [f"{f:.1f}" for f in freqs if f < -1.0]
            print(f"   * Imaginary Freqs:   {', '.join(imag_vals)} cm⁻¹")
        lowest_pos = [f for f in freqs if f > 0.0]
        if lowest_pos:
            print(f"   * Lowest Mode:       {min(lowest_pos):.1f} cm⁻¹")
    if result.gibbs_ha is not None:
        print(f" Gibbs Free Energy:     {result.gibbs_ha:.8f} Ha")
    print(f"============================================================")
    return 0


def _load_structure(filepath: str):
    """Load symbols and coordinates from XYZ or ZMAT."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if filepath.endswith(".zmat") or "zmat" in filepath.lower():
        zatoms, vars_dict = parse_zmatrix_text(content)
        coords = zmatrix_to_cartesian(zatoms)
        syms = [a.symbol for a in zatoms]
        return syms, coords, zatoms, vars_dict
    else:
        syms, coords = parse_xyz(content, is_path=False)
        return syms, coords, None, {}


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run an automated diagnostic check on the local AgyMolden environment and installed engines."""
    import platform
    import shutil
    from .qm_detect import detect_qm_executable

    print("============================================================")
    print(" AgyMolden Environment & QM Engines Diagnostic (Doctor)")
    print("============================================================")
    print(f" OS / Platform:         {platform.system()} {platform.release()} ({platform.machine()})")
    print(f" Python Executable:     {sys.executable}")
    print(f" Python Version:        {platform.python_version()}")

    # Check Core GUI dependencies
    try:
        import PyQt6
        from PyQt6 import QtCore
        qt_ver = QtCore.PYQT_VERSION_STR
        print(f" PyQt6:                 ✔ Installed (v{qt_ver})")
    except ImportError as e:
        print(f" PyQt6:                 ✖ Missing ({e})")

    try:
        import OpenGL
        print(f" PyOpenGL:              ✔ Installed (v{OpenGL.__version__})")
    except ImportError as e:
        print(f" PyOpenGL:              ✖ Missing ({e})")

    # Check SciPy stack
    import numpy
    import scipy
    import matplotlib
    print(f" NumPy:                 ✔ Installed (v{numpy.__version__})")
    print(f" SciPy:                 ✔ Installed (v{scipy.__version__})")
    print(f" Matplotlib:            ✔ Installed (v{matplotlib.__version__})")

    # Check Optional Packages
    for pkg in ("scikit-image", "cclib", "spglib", "ase", "pyscf", "torchani", "mace"):
        try:
            mod = __import__(pkg.replace("-", "_"))
            ver = getattr(mod, "__version__", "Available")
            print(f" {pkg:<22} ✔ Installed (v{ver})")
        except ImportError:
            print(f" {pkg:<22} ○ Optional (Not installed)")

    print("\n---------------- Installed Quantum Chemistry Engines ----------------")
    for eng in ("xtb", "orca", "gaussian", "pyscf", "psi4", "qchem", "openlowdin"):
        det = detect_qm_executable(eng)
        if det and det.found:
            status = f"✔ Found at: {det.resolved_path} ({det.version_str or 'Detected'})"
        else:
            status = "○ Not detected in PATH / default folders"
        print(f" {eng.upper():<12} {status}")
    print("============================================================")
    return 0


def cmd_install_desktop(args: argparse.Namespace) -> int:
    """Create a native OS application desktop shortcut and launcher without standalone compilation."""
    import platform
    import shutil
    import stat

    os_type = platform.system().lower()
    py_exe = os.path.abspath(sys.executable)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    icon_path = os.path.join(pkg_dir, "icons", "app_icon.png")
    if not os.path.isfile(icon_path):
        icon_path = os.path.join(pkg_dir, "icons", "app.png")

    if "linux" in os_type:
        desktop_dir = os.path.expanduser("~/.local/share/applications")
        os.makedirs(desktop_dir, exist_ok=True)
        desktop_file = os.path.join(desktop_dir, "agymolden.desktop")
        content = f"""[Desktop Entry]
Type=Application
Name=AgyMolden
Comment=Quantum Chemistry Molecular Builder & Orbitals Viewer
Exec={py_exe} -m agymolden %F
Icon={icon_path if os.path.isfile(icon_path) else 'applications-science'}
Terminal=false
Categories=Science;Chemistry;Education;
MimeType=chemical/x-molden;chemical/x-xyz;chemical/x-pdb;
"""
        with open(desktop_file, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(desktop_file, os.stat(desktop_file).st_mode | stat.S_IXUSR | stat.S_IRUSR)
        print(f"✔ Created Linux Desktop Entry: {desktop_file}")
        desktop_user_dir = os.path.expanduser("~/Desktop")
        if os.path.isdir(desktop_user_dir):
            user_shortcut = os.path.join(desktop_user_dir, "agymolden.desktop")
            shutil.copy2(desktop_file, user_shortcut)
            os.chmod(user_shortcut, os.stat(user_shortcut).st_mode | stat.S_IXUSR | stat.S_IRUSR)
            print(f"✔ Copied shortcut to Desktop: {user_shortcut}")
        return 0

    elif "darwin" in os_type:
        desktop_user_dir = os.path.expanduser("~/Desktop")
        launcher_path = os.path.join(desktop_user_dir, "AgyMolden.command")
        content = f"""#!/bin/bash
cd "{os.path.expanduser('~')}"
exec "{py_exe}" -m agymolden "$@"
"""
        with open(launcher_path, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(launcher_path, os.stat(launcher_path).st_mode | stat.S_IXUSR | stat.S_IRUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"✔ Created macOS Launcher on Desktop: {launcher_path}")
        return 0

    elif "win" in os_type:
        desktop_user_dir = os.path.expanduser("~/Desktop")
        bat_path = os.path.join(desktop_user_dir, "AgyMolden.bat")
        py_win = py_exe.replace("python.exe", "pythonw.exe") if os.path.isfile(py_exe.replace("python.exe", "pythonw.exe")) else py_exe
        content = f"""@echo off\r\nstart "" "{py_win}" -m agymolden %*\r\n"""
        with open(bat_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✔ Created Windows Launcher on Desktop: {bat_path}")
        return 0

    else:
        print(f"Desktop shortcut creation not supported on {platform.system()}.")
        return 1


def cmd_gen(args: argparse.Namespace) -> int:
    """Generate quantum chemistry input file."""
    if not os.path.isfile(args.struct):
        sys.stderr.write(f"Error: structure file not found: {args.struct}\n")
        return 1

    syms, coords, zatoms, vars_dict = _load_structure(args.struct)
    if not syms or len(coords) != len(syms):
        sys.stderr.write(f"Error: failed to parse coordinates from: {args.struct}\n")
        return 1

    code_key = (args.code or "orca").lower()
    gen_cls = None
    for name, cls in QM_GENERATORS.items():
        if code_key in name.lower() or code_key == getattr(cls, "code_name", "").lower():
            gen_cls = cls
            break
    if gen_cls is None:
        sys.stderr.write(f"Error: unrecognized QM code '{args.code}'. Supported: orca, gaussian, pyscf, newbrorsen\n")
        return 1

    params = QMCalculationParameters(
        job_type=args.job_type,
        method=args.method,
        basis=args.basis,
        charge=args.charge,
        multiplicity=args.mult,
        memory_gb=args.memory,
        n_procs=args.procs,
        dispersion=args.dispersion,
        use_zmatrix=bool(args.zmatrix and zatoms),
    )

    # Pre-flight check
    if not args.force:
        report = validate_qm_calculation(code_key, syms, coords, params)
        errors = [i for i in report.issues if i.severity == "error"]
        if errors:
            sys.stderr.write("Pre-flight validation failed with errors:\n")
            for err in errors:
                sys.stderr.write(f"  ✖ [{err.title}] {err.message}\n")
            sys.stderr.write("\nUse --force to generate input regardless.\n")
            return 1

    generator = gen_cls()
    text = generator.generate_input(syms, coords, params, zatoms=zatoms, variables=vars_dict)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Generated {code_key.upper()} input: {args.output}")
    else:
        print(text)
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Run pre-flight validation on a structure or input."""
    if not os.path.isfile(args.file):
        sys.stderr.write(f"Error: file not found: {args.file}\n")
        return 1

    syms, coords, _, _ = _load_structure(args.file)
    if not syms:
        sys.stderr.write(f"Error: unable to read atomic coordinates from: {args.file}\n")
        return 1

    params = QMCalculationParameters(
        charge=args.charge,
        multiplicity=args.mult,
        memory_gb=args.memory,
        n_procs=args.procs,
    )
    report = validate_qm_calculation(args.code, syms, coords, params)

    formula = "".join(f"{s}{syms.count(s) if syms.count(s) > 1 else ''}" for s in sorted(set(syms)))
    spin_name = "Singlet" if params.multiplicity == 1 else ("Doublet" if params.multiplicity == 2 else ("Triplet" if params.multiplicity == 3 else f"Mult={params.multiplicity}"))
    print(f"============================================================")
    print(f" Pre-Flight Sanity Report: {os.path.basename(args.file)}")
    print(f"============================================================")
    print(f" Formula:           {formula} ({len(syms)} atoms)")
    print(f" Electronic State:  {spin_name} (Electrons: {report.total_electrons}, α={report.num_alpha}, β={report.num_beta})")
    print(f" Status:            {'✔ All Checks Passed' if report.is_valid else '✖ Validation Errors Detected'}")
    print(f"------------------------------------------------------------")
    if not report.issues:
        print(" No issues found. Ready for calculation!")
    else:
        for iss in report.issues:
            icon = "✖" if iss.severity == "error" else ("⚠" if iss.severity == "warning" else "ℹ")
            print(f" {icon} [{iss.severity.upper()}] {iss.title}")
            print(f"   {iss.message}")
            if iss.remedy:
                print(f"   Remedy: {iss.remedy}")
    print(f"============================================================")
    return 0 if report.is_valid else 1


def cmd_convert(args: argparse.Namespace) -> int:
    """Convert molecular structure between formats."""
    if not os.path.isfile(args.input):
        sys.stderr.write(f"Error: input file not found: {args.input}\n")
        return 1
    if not args.output:
        sys.stderr.write("Error: --output (-o) required for format conversion.\n")
        return 1

    syms, coords, zatoms, vars_dict = _load_structure(args.input)
    if not syms:
        sys.stderr.write(f"Error: unable to read molecular structure from: {args.input}\n")
        return 1

    out_lower = args.output.lower()
    if out_lower.endswith(".zmat"):
        from .zmatrix import cartesian_to_zmatrix
        zat = zatoms if zatoms else cartesian_to_zmatrix(syms, coords)
        txt = format_zmatrix_text(zat, vars_dict or {})
    elif out_lower.endswith(".xyz"):
        txt = format_xyz_string(syms, coords, comment=f"Converted by AgyMolden from {os.path.basename(args.input)}")
    else:
        sys.stderr.write("Error: supported conversion output formats are .xyz and .zmat\n")
        return 1

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"Successfully converted {os.path.basename(args.input)} -> {args.output} ({len(syms)} atoms)")
    return 0


def cmd_bundle(args: argparse.Namespace) -> int:
    """Generate Supporting Information (SI) reproducibility bundle (.zip)."""
    if not os.path.isfile(args.file):
        sys.stderr.write(f"Error: file not found: {args.file}\n")
        return 1

    from .reproducibility import create_reproducibility_bundle

    ext = os.path.splitext(args.file)[1].lower()
    log_res = None
    symbols = []
    coords = None

    if ext in (".log", ".out", ".molden"):
        log_res = parse_qm_logfile(args.file)
        if log_res:
            symbols = log_res.symbols
            coords = log_res.trajectory_coords[-1]

    if not symbols:
        try:
            syms, crds, _, _ = _load_structure(args.file)
            symbols = syms
            coords = crds
        except Exception as e:
            sys.stderr.write(f"Error: unable to extract molecular geometry from {args.file}: {e}\n")
            return 1

    out_zip = args.output
    if not out_zip:
        stem = os.path.splitext(os.path.basename(args.file))[0]
        out_zip = f"{stem}_SI_bundle.zip"

    title = args.title or os.path.splitext(os.path.basename(args.file))[0]
    log_path = args.file if ext in (".log", ".out", ".molden") else None

    input_path = args.input
    if not input_path and log_path:
        stem_full = os.path.splitext(os.path.abspath(args.file))[0]
        for in_ext in (".com", ".gjf", ".inp", ".in"):
            cand = stem_full + in_ext
            if os.path.isfile(cand):
                input_path = cand
                break

    try:
        res_path = create_reproducibility_bundle(
            output_archive_path=out_zip,
            symbols=symbols,
            coords=coords,
            log_result=log_res,
            log_file_path=log_path,
            input_file_path=input_path,
            job_title=title,
        )
        print(f"Created Supporting Information Bundle: {res_path}")
        return 0
    except Exception as e:
        sys.stderr.write(f"Error creating bundle: {e}\n")
        return 1


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run ChemMCP JSON-RPC 2.0 stdio server."""
    from .mcp_server import run_mcp_server_stdio
    run_mcp_server_stdio()
    return 0


def build_cli_parser() -> argparse.ArgumentParser:
    """Create root parser with CLI subcommands."""
    parser = argparse.ArgumentParser(
        prog="agymolden",
        description="AgyMolden: Quantum Chemistry Workbench & Molecular Z-Matrix Builder",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. parse
    p_parse = subparsers.add_parser("parse", help="Parse QM output log and display energy/properties summary")
    p_parse.add_argument("file", help="Path to calculation log file (.log, .out, .molden)")
    p_parse.add_argument("--json", action="store_true", help="Output summary in JSON format")

    # 2. gen
    p_gen = subparsers.add_parser("gen", help="Generate quantum chemistry calculation input file")
    p_gen.add_argument("struct", help="Input structure file (.xyz, .zmat)")
    p_gen.add_argument("--code", default="orca", choices=["orca", "gaussian", "pyscf", "newbrorsen"], help="Target QM package (default: orca)")
    p_gen.add_argument("--method", default="B3LYP", help="Functional or method (default: B3LYP)")
    p_gen.add_argument("--basis", default="def2-SVP", help="Basis set (default: def2-SVP)")
    p_gen.add_argument("--job-type", default="SP", choices=["SP", "Opt", "Freq", "Opt + Freq", "TS", "IRC", "TD-DFT"], help="Job type (default: SP)")
    p_gen.add_argument("--charge", type=int, default=0, help="Molecular net charge (default: 0)")
    p_gen.add_argument("--mult", type=int, default=1, help="Spin multiplicity (default: 1)")
    p_gen.add_argument("--memory", type=int, default=4, help="Memory allocation in GB (default: 4)")
    p_gen.add_argument("--procs", type=int, default=1, help="Number of CPU cores/threads (default: 1)")
    p_gen.add_argument("--dispersion", default="None", choices=["None", "D3BJ", "D3Zero", "D4"], help="Empirical dispersion (default: None)")
    p_gen.add_argument("--zmatrix", action="store_true", help="Format input in internal coordinates (Z-matrix) if supported")
    p_gen.add_argument("-o", "--output", help="Write generated input to destination file")
    p_gen.add_argument("--force", action="store_true", help="Bypass pre-flight sanity checks")

    # 3. check
    p_check = subparsers.add_parser("check", help="Run pre-flight calculation sanity and parity validation")
    p_check.add_argument("file", help="Path to structure file (.xyz, .zmat)")
    p_check.add_argument("--code", default="orca", help="Target QM package")
    p_check.add_argument("--charge", type=int, default=0, help="Net charge (default: 0)")
    p_check.add_argument("--mult", type=int, default=1, help="Spin multiplicity (default: 1)")
    p_check.add_argument("--memory", type=int, default=4, help="Requested memory in GB")
    p_check.add_argument("--procs", type=int, default=1, help="Requested CPU threads")

    # 4. convert
    p_conv = subparsers.add_parser("convert", help="Convert molecular coordinates between formats")
    p_conv.add_argument("input", help="Source molecular structure (.xyz, .zmat)")
    p_conv.add_argument("-o", "--output", required=True, help="Destination file (.xyz, .zmat)")

    # 5. bundle
    p_bundle = subparsers.add_parser("bundle", help="Export Supporting Information (SI) reproducibility bundle (.zip)")
    p_bundle.add_argument("file", help="Path to calculation log or structure file (.log, .out, .molden, .xyz, .zmat)")
    p_bundle.add_argument("-o", "--output", help="Output zip archive name (default: <stem>_SI_bundle.zip)")
    p_bundle.add_argument("--title", help="Job or molecule title for the SI report")
    p_bundle.add_argument("--input", help="Explicit path to associated QM input file")

    # 6. doctor
    subparsers.add_parser("doctor", help="Run diagnostic health-check on Python environment and QM engine binaries")

    # 7. install-desktop
    subparsers.add_parser("install-desktop", help="Install native OS desktop shortcut and launcher without compiling binaries")

    # 8. mcp
    subparsers.add_parser("mcp", help="Run ChemMCP Model Context Protocol JSON-RPC 2.0 server over stdio for AI assistants")

    return parser
