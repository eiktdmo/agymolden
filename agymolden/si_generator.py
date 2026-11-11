"""
Automated Supporting Information (SI) & Computational Methods Generator for AgyMolden.

Synthesizes publication-ready SI sections in:
1. LaTeX (.tex) format formatted for JACS / Angewandte / Nature Chemistry / ACS Catalysis
2. Markdown / HTML (.md / .html) for immediate copy-pasting into Word
3. Exact Cartesian coordinates (XYZ blocks) with energy and frequency headers
4. Standardized "Computational Methods" text with automatic functional, basis set, and software citations.
"""

from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional
import numpy as np


class SISpecies(NamedTuple):
    name: str
    symbols: List[str]
    coords: np.ndarray             # (n_atoms, 3) in Angstrom
    electronic_energy_ha: Optional[float] = None
    zpe_ha: Optional[float] = None
    enthalpy_ha: Optional[float] = None
    gibbs_ha: Optional[float] = None
    lowest_frequencies_cm1: Optional[List[float]] = None
    program_name: str = "ORCA"
    functional: str = "ωB97X-D"
    basis_set: str = "def2-TZVP"
    solvation_model: Optional[str] = "CPCM (dichloromethane)"


def generate_computational_methods_paragraph(
    program: str = "ORCA",
    functional: str = "ωB97X-D",
    basis_set: str = "def2-TZVP",
    solvation: Optional[str] = "CPCM",
    solvent: Optional[str] = "dichloromethane",
    dispersion: Optional[str] = "D3BJ",
    temperature_k: float = 298.15,
) -> str:
    """Generates a peer-reviewed, standard publication methods paragraph."""
    solv_text = f"Solvation effects in {solvent} were accounted for implicitly using the {solvation} model. " if solvation else ""
    disp_text = f"Empirical dispersion corrections were applied via {dispersion}. " if dispersion else ""

    text = (
        f"All density functional theory (DFT) calculations were performed using the {program} quantum chemistry package. "
        f"Geometry optimizations and harmonic vibrational frequency calculations were conducted using the {functional} functional "
        f"with the {basis_set} basis set. {disp_text}{solv_text}"
        f"Stationary points were verified as true local minima (zero imaginary frequencies) or transition states (exactly one imaginary frequency) "
        f"by normal mode frequency analysis at {temperature_k:.2f} K and 1.0 atm. "
        f"Thermodynamic corrections to 298.15 K enthalpy and Gibbs free energy were computed using standard harmonic oscillator approximations."
    )
    return text


def format_xyz_block(species: SISpecies) -> str:
    """Formats standard Cartesian coordinates with energies in comments."""
    lines = [f"{len(species.symbols)}"]
    comment_parts = [f"{species.name}"]
    if species.electronic_energy_ha is not None:
        comment_parts.append(f"E = {species.electronic_energy_ha:.8f} Ha")
    if species.gibbs_ha is not None:
        comment_parts.append(f"G = {species.gibbs_ha:.8f} Ha")
    if species.lowest_frequencies_cm1:
        low_f_str = ", ".join(f"{f:.1f}" for f in species.lowest_frequencies_cm1[:3])
        comment_parts.append(f"Lowest freqs = [{low_f_str}] cm^-1")

    lines.append(" | ".join(comment_parts))
    for sym, coord in zip(species.symbols, species.coords):
        lines.append(f"{sym:<3} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}")
    return "\n".join(lines)


def generate_latex_si_document(
    species_list: List[SISpecies],
    methods_text: str,
    title: str = "Supporting Information: Computational Details and Geometries",
) -> str:
    """Generates a complete standalone LaTeX document with tables and coordinate appendices."""
    doc = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath}",
        r"\usepackage{hyperref}",
        r"\usepackage{longtable}",
        r"\usepackage{listings}",
        r"\lstset{basicstyle=\footnotesize\ttfamily,breaklines=true}",
        "",
        rf"\title{{{title}}}",
        r"\author{Generated automatically via AgyMolden}",
        r"\date{\today}",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\section{Computational Methods}",
        methods_text,
        "",
        r"\section{Summary of Thermochemical Data}",
        r"\begin{table}[h!]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Structure} & \textbf{E (Ha)} & \textbf{ZPE (Ha)} & \textbf{H (Ha)} & \textbf{G (Ha)} \\",
        r"\midrule",
    ]

    for sp in species_list:
        e_str = f"{sp.electronic_energy_ha:.6f}" if sp.electronic_energy_ha is not None else "-"
        z_str = f"{sp.zpe_ha:.6f}" if sp.zpe_ha is not None else "-"
        h_str = f"{sp.enthalpy_ha:.6f}" if sp.enthalpy_ha is not None else "-"
        g_str = f"{sp.gibbs_ha:.6f}" if sp.gibbs_ha is not None else "-"
        doc.append(rf"{sp.name} & {e_str} & {z_str} & {h_str} & {g_str} \\")

    doc.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Absolute electronic energies, zero-point energies, enthalpies, and Gibbs free energies at 298.15 K.}",
        r"\end{table}",
        "",
        r"\section{Cartesian Coordinates (XYZ Blocks)}",
    ])

    for sp in species_list:
        doc.append(rf"\subsection*{{{sp.name}}}")
        doc.append(r"\begin{lstlisting}")
        doc.append(format_xyz_block(sp))
        doc.append(r"\end{lstlisting}")
        doc.append("")

    doc.append(r"\end{document}")
    return "\n".join(doc)


def generate_markdown_si_document(
    species_list: List[SISpecies],
    methods_text: str,
    title: str = "Supporting Information: Computational Details and Geometries",
) -> str:
    """Generates a complete Markdown document for web / Word export."""
    md = [
        f"# {title}",
        "",
        "## 1. Computational Methods",
        "",
        methods_text,
        "",
        "## 2. Summary of Thermochemical Data",
        "",
        "| Structure | Electronic Energy (Ha) | ZPE (Ha) | Enthalpy H (Ha) | Gibbs Free Energy G (Ha) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for sp in species_list:
        e_str = f"{sp.electronic_energy_ha:.6f}" if sp.electronic_energy_ha is not None else "-"
        z_str = f"{sp.zpe_ha:.6f}" if sp.zpe_ha is not None else "-"
        h_str = f"{sp.enthalpy_ha:.6f}" if sp.enthalpy_ha is not None else "-"
        g_str = f"{sp.gibbs_ha:.6f}" if sp.gibbs_ha is not None else "-"
        md.append(f"| **{sp.name}** | {e_str} | {z_str} | {h_str} | {g_str} |")

    md.extend(["", "## 3. Cartesian Coordinates (XYZ Blocks)", ""])

    for sp in species_list:
        md.append(f"### {sp.name}")
        md.append("```xyz")
        md.append(format_xyz_block(sp))
        md.append("```")
        md.append("")

    return "\n".join(md)
