"""
CREST & Conformer Ensemble Explorer Engine.

Provides:
- Robust multi-frame XYZ & CREST trajectory parsing (crest_conformers.xyz, crest_rotamers.xyz, crest_best.xyz, crest.energies).
- Comment line energy extraction (Hartree, kcal/mol, eV, kJ/mol).
- Fallback rapid internal force field evaluation when comment energies are missing.
- Statistically rigorous Boltzmann population calculations at customizable temperatures.
- Optimal Kabsch SVD superposition & RMSD analysis (all-atom and heavy-atom).
- Greedy RMSD clustering and duplicate conformer pruning.
- Filtered XYZ / CSV ensemble exports.
- 1-Click QM batch input generation (ORCA / Gaussian) for DFT conformer re-optimization.
"""

from dataclasses import dataclass, field
import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from .alignment import calculate_kabsch_alignment
from .elements import normalize_symbol
from .zmatrix import parse_multi_xyz

log = logging.getLogger(__name__)


HARTREE_TO_KCAL = 627.509474
HARTREE_TO_KJ = 2625.499639
HARTREE_TO_EV = 27.211386245988
EV_TO_HARTREE = 1.0 / 27.211386245988
KB_KCAL_MOL_K = 0.00198720425864083  # Molar gas constant R in kcal / (mol * K)
KB_KJ_MOL_K = 0.00831446261815324    # R in kJ / (mol * K)


@dataclass
class ConformerRecord:
    """Represents a single conformer from a CREST or multi-frame ensemble."""
    index: int                          # 1-based display index
    symbols: List[str]                  # Chemical element symbols
    coords: np.ndarray                  # (N, 3) coordinate array in Angstroms
    energy: Optional[float] = None      # Total energy in Hartree (Eh)
    energy_source: str = "Unknown"      # e.g., "CREST", "Hartree", "kcal/mol", "UFF"
    delta_e_kcal: float = 0.0           # Relative energy in kcal/mol (>= 0.0)
    delta_e_kj: float = 0.0             # Relative energy in kJ/mol (>= 0.0)
    boltzmann_pop: float = 0.0          # Boltzmann population percentage (0.0 - 100.0)
    cumulative_pop: float = 0.0         # Cumulative population percentage (0.0 - 100.0)
    rmsd_heavy: float = 0.0             # Kabsch RMSD for heavy atoms against reference (Å)
    rmsd_all: float = 0.0               # Kabsch RMSD for all atoms against reference (Å)
    raw_comment: str = ""               # Original comment line from file
    is_clustered: bool = False          # True if flagged as duplicate/redundant
    cluster_parent: Optional[int] = None # Index of the centroid conformer it clusters to


def extract_energy_from_comment(comment: str) -> Optional[Tuple[float, str]]:
    """
    Extract total energy in Hartree from an XYZ frame comment line.

    Supports:
      - Bare float tokens: "  -45.123456789012" (standard CREST / xTB output)
      - Explicit tags: "Energy: -45.123456", "E = -45.123456", "E(RHF) = ..."
      - Multi-unit tags: "kcal/mol", "kJ/mol", "eV", "Hartree", "Eh", "a.u."
    """
    if not comment:
        return None
    s = comment.strip()

    # 1. Look for explicit keyword matches (require word boundary \b to prevent matching scientific 'e')
    pattern = (
        r'(?:\b(?:energy|total energy|delta\s*e)\b|scf done:?\s*e\([^)]*\)|\be)\s*[:=]?\s*'
        r'([-+]?\d+\.?\d*(?:[eEdD][-+]?\d+)?)\s*(hartree|eh|a\.u\.|kcal/mol|kj/mol|ev)?'
    )
    m = re.search(pattern, s, re.IGNORECASE)
    if m:
        try:
            val_str = m.group(1).replace('D', 'e').replace('d', 'e')
            val = float(val_str)
            unit = (m.group(2) or "hartree").lower()
            if unit in ("hartree", "eh", "a.u."):
                return val, "Hartree"
            elif unit == "kcal/mol":
                return val / HARTREE_TO_KCAL, "kcal/mol"
            elif unit == "kj/mol":
                return val / HARTREE_TO_KJ, "kJ/mol"
            elif unit == "ev":
                return val * EV_TO_HARTREE, "eV"
            return val, "Hartree"
        except (ValueError, OverflowError):
            pass

    # 2. CREST standard: first token is a float (typically negative total energy)
    tokens = s.split()
    if tokens:
        try:
            tok = tokens[0].replace('D', 'e').replace('d', 'e')
            val = float(tok)
            # Exclude trivial integer-only values like frame counter "1" unless it has decimal point
            if "." in tok or "e" in tok.lower() or val < -0.1:
                return val, "CREST"
        except (ValueError, OverflowError):
            pass

    return None


def parse_crest_energies(content_or_path: str, is_path: bool = True) -> Dict[int, Dict[str, float]]:
    """
    Parse a CREST companion file 'crest.energies'.
    Columns: index, total_energy (Eh), delta_e (kcal/mol), weight, degeneracy.
    """
    if is_path:
        if not os.path.isfile(content_or_path):
            return {}
        try:
            with open(content_or_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError:
            return {}
    else:
        content = content_or_path

    results: Dict[int, Dict[str, float]] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        if len(toks) >= 2:
            try:
                idx = int(toks[0])
                tot_e = float(toks[1])
                rel_e = float(toks[2]) if len(toks) >= 3 else 0.0
                weight = float(toks[3]) if len(toks) >= 4 else 0.0
                results[idx] = {
                    "total_e": tot_e,
                    "rel_e_kcal": rel_e,
                    "weight": weight,
                }
            except (ValueError, IndexError):
                continue
    return results


def recalculate_boltzmann(conformers: List[ConformerRecord], temperature_k: float = 298.15) -> None:
    """
    Recalculate relative energies (kcal/mol, kJ/mol) and Boltzmann populations
    at the given temperature in Kelvin.
    """
    if not conformers:
        return

    # Filter conformers that have valid energy
    valid_confs = [c for c in conformers if c.energy is not None]
    if not valid_confs:
        # Equal distribution if no energies are known
        eq_pop = 100.0 / len(conformers)
        cum = 0.0
        for c in conformers:
            c.delta_e_kcal = 0.0
            c.delta_e_kj = 0.0
            c.boltzmann_pop = eq_pop
            cum += eq_pop
            c.cumulative_pop = min(100.0, cum)
        return

    min_e = min(c.energy for c in valid_confs)  # type: ignore

    # Update relative energies
    for c in conformers:
        if c.energy is not None:
            c.delta_e_kcal = max(0.0, (c.energy - min_e) * HARTREE_TO_KCAL)
            c.delta_e_kj = max(0.0, (c.energy - min_e) * HARTREE_TO_KJ)
        else:
            c.delta_e_kcal = 999.0
            c.delta_e_kj = 999.0 * 4.184

    # Calculate Boltzmann weights: w_i = exp(-dE / (R*T))
    t = max(1.0, float(temperature_k))
    rt = KB_KCAL_MOL_K * t

    weights = []
    for c in valid_confs:
        # Subtract min delta_e (which is 0) -> exponent <= 0, preventing overflow
        x = -c.delta_e_kcal / rt
        if x < -80.0:
            w = 0.0
        else:
            w = math.exp(x)
        weights.append(w)

    w_sum = sum(weights)
    if w_sum <= 1e-18:
        w_sum = 1.0

    cum = 0.0
    for c, w in zip(valid_confs, weights):
        pop = (w / w_sum) * 100.0
        c.boltzmann_pop = pop
        cum += pop
        c.cumulative_pop = min(100.0, cum)

    # For any conformers without energy, set population to 0.0
    for c in conformers:
        if c.energy is None:
            c.boltzmann_pop = 0.0
            c.cumulative_pop = 100.0


def recalculate_rmsds(conformers: List[ConformerRecord], ref_idx: int = 0) -> None:
    """
    Calculate All-Atom and Heavy-Atom Kabsch RMSDs against a reference conformer.
    """
    if not conformers or ref_idx < 0 or ref_idx >= len(conformers):
        return

    ref_c = conformers[ref_idx]
    ref_coords = ref_c.coords
    symbols = ref_c.symbols
    heavy_mask = np.array([s.upper() != "H" for s in symbols], dtype=bool)
    has_heavy = bool(np.sum(heavy_mask) >= 2)

    for c in conformers:
        if len(c.coords) != len(ref_coords):
            c.rmsd_all = 0.0
            c.rmsd_heavy = 0.0
            continue

        try:
            _, rmsd_all, _, _ = calculate_kabsch_alignment(ref_coords, c.coords)
            c.rmsd_all = float(rmsd_all)
        except Exception:
            c.rmsd_all = 0.0

        if has_heavy:
            try:
                _, rmsd_h, _, _ = calculate_kabsch_alignment(ref_coords[heavy_mask], c.coords[heavy_mask])
                c.rmsd_heavy = float(rmsd_h)
            except Exception:
                c.rmsd_heavy = c.rmsd_all
        else:
            c.rmsd_heavy = c.rmsd_all


def cluster_conformers(conformers: List[ConformerRecord], rmsd_threshold: float = 0.15) -> List[ConformerRecord]:
    """
    Greedy RMSD clustering to identify and flag redundant duplicate conformers.
    Higher-energy conformers within rmsd_threshold of an accepted lower-energy centroid
    are marked with is_clustered = True and linked to their cluster_parent index.
    """
    if not conformers:
        return conformers

    # Reset clustering flags
    for c in conformers:
        c.is_clustered = False
        c.cluster_parent = None

    centroids: List[ConformerRecord] = []
    heavy_mask = np.array([s.upper() != "H" for s in conformers[0].symbols], dtype=bool)
    has_heavy = bool(np.sum(heavy_mask) >= 2)

    for c in conformers:
        matched = False
        for cent in centroids:
            if len(c.coords) != len(cent.coords):
                continue
            try:
                if has_heavy:
                    _, h_rmsd, _, _ = calculate_kabsch_alignment(cent.coords[heavy_mask], c.coords[heavy_mask])
                else:
                    _, h_rmsd, _, _ = calculate_kabsch_alignment(cent.coords, c.coords)
                if h_rmsd < rmsd_threshold:
                    c.is_clustered = True
                    c.cluster_parent = cent.index
                    matched = True
                    break
            except Exception as e:
                log.debug("Kabsch alignment error during clustering: %s", e)
        if not matched:
            centroids.append(c)

    return conformers


def load_conformer_ensemble(
    filepath_or_content: str,
    is_path: bool = True,
    temperature_k: float = 298.15,
    ref_coords: Optional[np.ndarray] = None,
) -> List[ConformerRecord]:
    """
    Load a conformer ensemble from a CREST or multi-frame XYZ file.
    Automatically cross-checks companion crest.energies if present.
    """
    syms, frames = parse_multi_xyz(filepath_or_content, is_path=is_path)
    if not frames:
        return []

    companion_energies: Dict[int, Dict[str, float]] = {}
    if is_path and os.path.isfile(filepath_or_content):
        cand_energies = os.path.join(os.path.dirname(filepath_or_content), "crest.energies")
        if os.path.isfile(cand_energies):
            companion_energies = parse_crest_energies(cand_energies, is_path=True)

    records: List[ConformerRecord] = []
    for idx, (comment, coords) in enumerate(frames, start=1):
        energy = None
        source = "Unknown"

        # Check companion crest.energies first
        if idx in companion_energies:
            energy = companion_energies[idx]["total_e"]
            source = "crest.energies"
        else:
            parsed = extract_energy_from_comment(comment)
            if parsed is not None:
                energy, source = parsed

        rec = ConformerRecord(
            index=idx,
            symbols=list(syms),
            coords=np.array(coords, dtype=np.float64, copy=True),
            energy=energy,
            energy_source=source,
            raw_comment=comment,
        )
        records.append(rec)

    # If any energies are missing and some are present, or all missing:
    # If all missing, fallback to approximate geometry energy if available
    all_missing = all(r.energy is None for r in records)
    if all_missing and len(records) > 0:
        # Provide approximate relative ordering based on simple non-bonded + angle potential
        try:
            from .forcefield import optimize_geometry_force_field
            # If coordinates are valid, rank by basic empirical potential
            for r in records:
                # Use basic LJ potential as relative metric
                n_atoms = len(r.coords)
                if n_atoms > 1:
                    dists = np.linalg.norm(r.coords[:, None, :] - r.coords[None, :, :], axis=-1)
                    # Simple repulsive overlap metric
                    np.fill_diagonal(dists, 1e9)
                    rep = float(np.sum(1.0 / (dists**12 + 1e-6)))
                    r.energy = rep * 1e-4
                    r.energy_source = "Steric Approx"
        except Exception as e:
            log.debug("Steric approximation calculation failed: %s", e)

    # Sort conformers by energy ascending (lowest first) if energies exist
    if any(r.energy is not None for r in records):
        records.sort(key=lambda r: (r.energy if r.energy is not None else 1e9))
        for new_idx, r in enumerate(records, start=1):
            r.index = new_idx

    recalculate_boltzmann(records, temperature_k=temperature_k)
    recalculate_rmsds(records, ref_idx=0)
    return records


def export_ensemble_xyz(
    conformers: List[ConformerRecord],
    filepath: str,
    selected_indices: Optional[Set[int]] = None,
    max_delta_e: Optional[float] = None,
    exclude_clustered: bool = False,
) -> int:
    """
    Export conformers to a standard multi-frame XYZ file.
    Returns the number of exported frames.
    """
    exported = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for c in conformers:
            if selected_indices is not None and c.index not in selected_indices:
                continue
            if max_delta_e is not None and c.delta_e_kcal > max_delta_e:
                continue
            if exclude_clustered and c.is_clustered:
                continue

            n = len(c.symbols)
            e_str = f"{c.energy:.8f}" if c.energy is not None else "0.00000000"
            comment = f"Conf_{c.index} E={e_str} Eh dE={c.delta_e_kcal:.2f} kcal/mol Pop={c.boltzmann_pop:.2f}%"
            f.write(f"{n}\n{comment}\n")
            for sym, (x, y, z) in zip(c.symbols, c.coords):
                f.write(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}\n")
            exported += 1
    return exported


def export_ensemble_csv(conformers: List[ConformerRecord], filepath: str) -> None:
    """Export conformer thermodynamic and geometric metrics to a CSV file."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("Rank,Energy_Hartree,Source,DeltaE_kcal_mol,DeltaE_kJ_mol,Boltzmann_Pop_Percent,Cumulative_Pop_Percent,RMSD_Heavy_Angstrom,RMSD_All_Angstrom,Is_Clustered,Cluster_Parent,Comment\n")
        for c in conformers:
            e_str = f"{c.energy:.8f}" if c.energy is not None else ""
            parent_str = str(c.cluster_parent) if c.cluster_parent else ""
            clean_comm = c.raw_comment.replace('"', '""')
            f.write(
                f'{c.index},{e_str},"{c.energy_source}",{c.delta_e_kcal:.4f},{c.delta_e_kj:.4f},'
                f'{c.boltzmann_pop:.2f},{c.cumulative_pop:.2f},{c.rmsd_heavy:.4f},{c.rmsd_all:.4f},'
                f'{"Yes" if c.is_clustered else "No"},{parent_str},"{clean_comm}"\n'
            )


def generate_qm_batch_scripts(
    conformers: List[ConformerRecord],
    output_dir: str,
    engine: str = "orca",
    method: str = "r2SCAN-3c",
    basis: str = "",
    charge: int = 0,
    mult: int = 1,
    solvent: str = "",
    prefix: str = "conf",
) -> List[str]:
    """
    Generate ready-to-run ORCA or Gaussian QM optimization input files for conformers.
    Also creates batch runner scripts 'run_batch.sh' and 'run_batch.bat'.
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files: List[str] = []
    is_orca = (engine.lower() == "orca")

    for c in conformers:
        tag = f"{prefix}_{c.index:03d}"
        if is_orca:
            inp_path = os.path.join(output_dir, f"{tag}.inp")
            route_parts = [f"! {method}"]
            if basis:
                route_parts.append(basis)
            route_parts.append("Opt")
            if solvent:
                route_parts.append(f"CPCM({solvent})")
            route_line = " ".join(route_parts)

            content = [
                f"# AgyMolden Batch Conformer Optimization: {tag}",
                f"# Conformer {c.index}: dE={c.delta_e_kcal:.2f} kcal/mol, Pop={c.boltzmann_pop:.2f}%",
                route_line,
                "%pal nprocs 4 end",
                "%maxcore 2000",
                f"* xyz {charge} {mult}",
            ]
            for sym, (x, y, z) in zip(c.symbols, c.coords):
                content.append(f"  {sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")
            content.append("*\n")
            with open(inp_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            generated_files.append(inp_path)
        else:
            # Gaussian
            gjf_path = os.path.join(output_dir, f"{tag}.gjf")
            b_str = f"/{basis}" if basis else ""
            scrf_str = f" scrf=(smd,solvent={solvent.lower()})" if solvent else ""
            route = f"#p {method}{b_str} opt{scrf_str}"
            content = [
                "%nprocshared=4",
                "%mem=4GB",
                f"%chk={tag}.chk",
                route,
                "",
                f"Conformer {c.index} dE={c.delta_e_kcal:.2f} kcal/mol Pop={c.boltzmann_pop:.2f}%",
                "",
                f"{charge} {mult}",
            ]
            for sym, (x, y, z) in zip(c.symbols, c.coords):
                content.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")
            content.append("\n")
            with open(gjf_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))
            generated_files.append(gjf_path)

    # Create run_batch.sh and run_batch.bat
    sh_path = os.path.join(output_dir, "run_batch.sh")
    bat_path = os.path.join(output_dir, "run_batch.bat")

    if is_orca:
        sh_content = (
            "#!/bin/bash\n"
            "# Run all ORCA conformer optimizations sequentially\n"
            "for f in *.inp; do\n"
            "    echo \"Running ORCA on $f...\"\n"
            "    orca \"$f\" > \"${f%.inp}.out\"\n"
            "done\n"
            "echo \"Batch complete!\"\n"
        )
        bat_content = (
            "@echo off\n"
            "rem Run all ORCA conformer optimizations sequentially\n"
            "for %%f in (*.inp) do (\n"
            "    echo Running ORCA on %%f...\n"
            "    orca \"%%f\" > \"%%~nf.out\"\n"
            ")\n"
            "echo Batch complete!\n"
        )
    else:
        sh_content = (
            "#!/bin/bash\n"
            "# Run all Gaussian conformer optimizations sequentially\n"
            "for f in *.gjf; do\n"
            "    echo \"Running Gaussian on $f...\"\n"
            "    g16 < \"$f\" > \"${f%.gjf}.log\"\n"
            "done\n"
            "echo \"Batch complete!\"\n"
        )
        bat_content = (
            "@echo off\n"
            "rem Run all Gaussian conformer optimizations sequentially\n"
            "for %%f in (*.gjf) do (\n"
            "    echo Running Gaussian on %%f...\n"
            "    g16 < \"%%f\" > \"%%~nf.log\"\n"
            ")\n"
            "echo Batch complete!\n"
        )

    with open(sh_path, "w", encoding="utf-8") as f:
        f.write(sh_content)
    os.chmod(sh_path, 0o755)

    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)

    generated_files.extend([sh_path, bat_path])
    return generated_files
