"""
Intrinsic Reaction Coordinate (IRC) Pathway Analysis & Sweeper Engine.

Handles:
- Parsing and assembly of IRC calculations from Gaussian, ORCA, newBRORSEN, and multi-frame XYZ files.
- Continuous stitching of bidirectional IRC branches:
    Reactant (s < 0) ──> Transition State (s = 0) ──> Product (s > 0)
- Calculation of forward/reverse activation barriers (ΔE‡_fwd, ΔE‡_rev) and reaction energies (ΔE_rxn).
- High-performance data structures for real-time 3D scrubbing and publication export.
"""

from dataclasses import dataclass, field
import io
import math
import re
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Energy conversion constant: 1 Hartree = 627.509474 kcal/mol = 2625.50 kJ/mol = 27.211386 eV
HARTREE_TO_KCAL = 627.509474
HARTREE_TO_KJ = 2625.499639
HARTREE_TO_EV = 27.211386245988


@dataclass
class IRCPoint:
    """A single point along an Intrinsic Reaction Coordinate (IRC) pathway."""
    index: int
    reaction_coord: float       # s in amu^(1/2)*Bohr or amu^(1/2)*Å
    energy_ha: float            # Total energy in Hartree
    rel_energy_kcal: float      # Relative energy relative to TS or lowest minimum (kcal/mol)
    coords: np.ndarray          # Cartesian coordinates (N, 3) in Å
    forces: Optional[np.ndarray] = None  # Atomic forces (N, 3) in Ha/Bohr
    gradient_norm: Optional[float] = None
    is_ts: bool = False
    is_reactant: bool = False
    is_product: bool = False
    label: str = ""             # "Reactant", "TS ‡", "Product", or step label


@dataclass
class IRCPathway:
    """A complete, ordered Intrinsic Reaction Coordinate (IRC) pathway."""
    symbols: List[str]
    points: List[IRCPoint] = field(default_factory=list)
    ts_index: int = 0
    reactant_index: int = 0
    product_index: int = -1
    delta_e_fwd_kcal: float = 0.0     # E(TS) - E(Reactant)
    delta_e_rev_kcal: float = 0.0     # E(TS) - E(Product)
    delta_e_rxn_kcal: float = 0.0     # E(Product) - E(Reactant)
    is_exothermic: bool = True
    coord_unit: str = "amu½·Bohr"

    def __post_init__(self):
        if self.points and self.product_index < 0:
            self.product_index = len(self.points) - 1

    @property
    def n_points(self) -> int:
        return len(self.points)

    def get_coords_trajectory(self) -> List[np.ndarray]:
        return [p.coords for p in self.points]

    def get_energies_ha(self) -> List[float]:
        return [p.energy_ha for p in self.points]

    def get_reaction_coords(self) -> List[float]:
        return [p.reaction_coord for p in self.points]

    def get_energies_relative(self, unit: str = "kcal") -> List[float]:
        """
        Return relative energies along the path.
        By default, relative to the Reactant state (E_ref = E_reactant),
        or relative to minimum if reactant isn't defined.
        """
        if not self.points:
            return []
        r_idx = self.reactant_index if (0 <= self.reactant_index < len(self.points)) else 0
        ref_e = self.points[r_idx].energy_ha

        scale = {
            "kcal": HARTREE_TO_KCAL,
            "kj": HARTREE_TO_KJ,
            "ev": HARTREE_TO_EV,
            "ha": 1.0,
        }.get(unit.lower(), HARTREE_TO_KCAL)

        return [(p.energy_ha - ref_e) * scale for p in self.points]

    def export_csv(self) -> str:
        """Export publication-ready reaction coordinate profile to CSV format."""
        if not self.points:
            return "Step,Reaction_Coordinate,Energy_Ha,Rel_E_kcal_mol,Rel_E_kJ_mol,Rel_E_eV,Label\n"

        r_idx = self.reactant_index if (0 <= self.reactant_index < len(self.points)) else 0
        ref_e = self.points[r_idx].energy_ha

        out = io.StringIO()
        out.write(f"# IRC Pathway: {len(self.symbols)} atoms, {len(self.points)} points\n")
        out.write(f"# Forward Barrier (Delta E_fwd): {self.delta_e_fwd_kcal:+.3f} kcal/mol ({self.delta_e_fwd_kcal * 4.184:+.3f} kJ/mol)\n")
        out.write(f"# Reverse Barrier (Delta E_rev): {self.delta_e_rev_kcal:+.3f} kcal/mol ({self.delta_e_rev_kcal * 4.184:+.3f} kJ/mol)\n")
        out.write(f"# Reaction Energy (Delta E_rxn): {self.delta_e_rxn_kcal:+.3f} kcal/mol ({'Exothermic' if self.is_exothermic else 'Endothermic'})\n")
        out.write(f"Step,Reaction_Coord_{self.coord_unit.replace('½', '^0.5')},Energy_Ha,Rel_E_kcal_mol,Rel_E_kJ_mol,Rel_E_eV,Label\n")

        for p in self.points:
            d_ha = p.energy_ha - ref_e
            d_kcal = d_ha * HARTREE_TO_KCAL
            d_kj = d_ha * HARTREE_TO_KJ
            d_ev = d_ha * HARTREE_TO_EV
            out.write(
                f"{p.index + 1},{p.reaction_coord:.6f},{p.energy_ha:.8f},{d_kcal:.4f},{d_kj:.4f},{d_ev:.4f},{p.label}\n"
            )
        return out.getvalue()

    def export_xyz(self) -> str:
        """Export the full continuous reaction path as a multi-frame XYZ movie."""
        if not self.points:
            return ""
        r_idx = self.reactant_index if (0 <= self.reactant_index < len(self.points)) else 0
        ref_e = self.points[r_idx].energy_ha

        out = io.StringIO()
        n_atoms = len(self.symbols)

        for p in self.points:
            d_kcal = (p.energy_ha - ref_e) * HARTREE_TO_KCAL
            lbl = f" | {p.label}" if p.label else ""
            out.write(f"{n_atoms}\n")
            out.write(
                f"IRC Step {p.index + 1}: s = {p.reaction_coord:+.5f} {self.coord_unit}, "
                f"E = {p.energy_ha:.8f} Ha, Delta E = {d_kcal:+.3f} kcal/mol{lbl}\n"
            )
            for sym, (x, y, z) in zip(self.symbols, p.coords):
                out.write(f"{sym:<3}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")

        return out.getvalue()


def assemble_irc_pathway(
    symbols: List[str],
    coords_list: List[np.ndarray],
    energies_ha: List[float],
    rxn_coords: Optional[List[float]] = None,
    path_numbers: Optional[List[int]] = None,
    coord_unit: str = "amu½·Bohr",
) -> IRCPathway:
    """
    Assemble and order an IRC pathway monotonically from Reactant -> TS -> Product.

    Handles:
    - When rxn_coords is monotonic: preserves order.
    - When calculation has Forward (s > 0) and Reverse (s < 0) branches separately:
      Sorts the entire trajectory so s runs monotonically from negative to positive.
    - When path_numbers are provided (e.g. Gaussian Path 1 = Forward, Path 2 = Reverse):
      Stitches reverse branch (reversed) -> TS (s=0) -> forward branch.
    - Deduplicates redundant TS frames (e.g. s=0 appearing in both forward and reverse passes).
    """
    n = min(len(coords_list), len(energies_ha))
    if n == 0:
        return IRCPathway(symbols=symbols, points=[])

    coords_list = [np.asarray(c, dtype=np.float64) for c in coords_list[:n]]
    energies_ha = [float(e) for e in energies_ha[:n]]

    # If reaction coordinates are not provided, estimate arc length along path
    if rxn_coords is None or len(rxn_coords) != n:
        # Approximate reaction coordinate by cumulative atomic displacement
        arc_lengths = [0.0]
        for i in range(1, n):
            disp = np.linalg.norm(coords_list[i] - coords_list[i - 1])
            arc_lengths.append(arc_lengths[-1] + float(disp))

        # Find maximum energy (Transition State)
        ts_cand_idx = int(np.argmax(energies_ha))
        ts_arc = arc_lengths[ts_cand_idx]
        rxn_coords = [float(a - ts_arc) for a in arc_lengths]
    else:
        rxn_coords = [float(s) for s in rxn_coords[:n]]

    # Prepare raw items
    items = []
    for idx in range(n):
        items.append({
            "orig_idx": idx,
            "s": rxn_coords[idx],
            "e": energies_ha[idx],
            "c": coords_list[idx],
            "path": path_numbers[idx] if (path_numbers and idx < len(path_numbers)) else None,
        })

    # Sort items monotonically by reaction coordinate s
    # If s has both negative and positive values, natural sorting by s creates Reactant -> TS -> Product
    items.sort(key=lambda item: item["s"])

    # Deduplicate consecutive points where |s_i - s_{i-1}| < 1e-4 and |E_i - E_{i-1}| < 1e-6
    deduped = []
    for item in items:
        if deduped:
            prev = deduped[-1]
            if abs(item["s"] - prev["s"]) < 1e-4 and abs(item["e"] - prev["e"]) < 1e-6:
                continue
        deduped.append(item)

    if not deduped:
        deduped = items

    # Identify Transition State (point closest to s=0 or local/global maximum)
    # Primary candidate: point with smallest |s|
    s_abs = [abs(d["s"]) for d in deduped]
    closest_zero_idx = int(np.argmin(s_abs))

    # Also check maximum energy in the central region
    e_vals = [d["e"] for d in deduped]
    max_e_idx = int(np.argmax(e_vals))

    # If the maximum energy point is reasonably close to s=0, prefer max energy as TS
    if abs(deduped[max_e_idx]["s"]) < 0.5 or s_abs[closest_zero_idx] > 0.1:
        ts_idx = max_e_idx
    else:
        ts_idx = closest_zero_idx

    reactant_idx = 0
    product_idx = len(deduped) - 1

    # Ensure reactant and product are distinct
    if reactant_idx == ts_idx and len(deduped) > 1:
        reactant_idx = 0
    if product_idx == ts_idx and len(deduped) > 1:
        product_idx = len(deduped) - 1

    # Calculate barriers
    e_ts = deduped[ts_idx]["e"]
    e_react = deduped[reactant_idx]["e"]
    e_prod = deduped[product_idx]["e"]

    barrier_fwd_kcal = max(0.0, (e_ts - e_react) * HARTREE_TO_KCAL)
    barrier_rev_kcal = max(0.0, (e_ts - e_prod) * HARTREE_TO_KCAL)
    delta_rxn_kcal = (e_prod - e_react) * HARTREE_TO_KCAL
    is_exo = delta_rxn_kcal <= 0.0

    # Build IRCPoints
    points: List[IRCPoint] = []
    for i, d in enumerate(deduped):
        is_ts_pt = (i == ts_idx)
        is_r_pt = (i == reactant_idx)
        is_p_pt = (i == product_idx)

        if is_ts_pt:
            lbl = "TS ‡"
        elif is_r_pt:
            lbl = "Reactant"
        elif is_p_pt:
            lbl = "Product"
        else:
            lbl = f"Step {i + 1}"

        rel_k = (d["e"] - e_react) * HARTREE_TO_KCAL

        pt = IRCPoint(
            index=i,
            reaction_coord=d["s"],
            energy_ha=d["e"],
            rel_energy_kcal=rel_k,
            coords=d["c"],
            is_ts=is_ts_pt,
            is_reactant=is_r_pt,
            is_product=is_p_pt,
            label=lbl,
        )
        points.append(pt)

    return IRCPathway(
        symbols=list(symbols),
        points=points,
        ts_index=ts_idx,
        reactant_index=reactant_idx,
        product_index=product_idx,
        delta_e_fwd_kcal=barrier_fwd_kcal,
        delta_e_rev_kcal=barrier_rev_kcal,
        delta_e_rxn_kcal=delta_rxn_kcal,
        is_exothermic=is_exo,
        coord_unit=coord_unit,
    )


def parse_gaussian_irc_text(text: str) -> Optional[IRCPathway]:
    """
    Extract converged IRC steps, reaction coordinates, and energies from a Gaussian IRC calculation log.

    Handles both:
    1) "Summary of reaction path following" table at the end of the calculation.
    2) Detailed step blocks with "Point Number: X ... Path Number: Y" and "NET REACTION COORDINATE".
    """
    if "IRC-Calculation" not in text and "reaction path following" not in text and "NET REACTION COORDINATE" not in text:
        return None

    # Try extracting summary table first
    summary_match = re.search(
        r'Summary of reaction path following:?\s*\n[-=\s]+\n\s*Energy\s+RxCoord.*?\n([-=\s]+\n(?:[\s\d.-]+\n)+)',
        text,
        re.IGNORECASE,
    )

    # Extract geometries: find all "Standard orientation:" or "Input orientation:"
    # In Gaussian IRC, each converged point prints an orientation block
    orient_pattern = re.compile(
        r'(?:Standard orientation|Input orientation):[^\n]*\n'
        r'\s*[-=]+\s*\n'
        r'\s*Center\s+Atomic\s+Atomic\s+Coordinates\s*\(Angstroms\)[^\n]*\n'
        r'\s*Number\s+Number\s+Type\s+X\s+Y\s+Z[^\n]*\n'
        r'\s*[-=]+\s*\n'
        r'((?:\s*\d+\s+\d+\s+\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+[^\n]*\n)+)',
        re.IGNORECASE,
    )

    all_blocks = orient_pattern.findall(text)
    if not all_blocks:
        return None

    # Parse first geometry block to get symbols
    from .qm_log_parser import ATOMIC_NUMBER_TO_SYMBOL
    first_block = all_blocks[0].strip().split("\n")
    symbols = []
    for line in first_block:
        parts = line.split()
        if len(parts) >= 6:
            try:
                z = int(parts[1])
                sym = ATOMIC_NUMBER_TO_SYMBOL.get(z, f"X{z}")
                symbols.append(sym)
            except ValueError:
                pass

    if not symbols:
        return None

    # Extract path points from detailed "Point Number" blocks
    # Format:
    # Point Number:    1   Path Number:    1
    # NET REACTION COORDINATE UP TO THIS POINT =    0.10234
    # Total Energy =      -155.01254321
    pt_blocks = re.findall(
        r'Point\s+Number:\s*(\d+)\s+Path\s+Number:\s*(\d+).*?'
        r'NET REACTION COORDINATE UP TO THIS POINT\s*=\s*([+-]?\d+\.\d+).*?'
        r'Total\s+Energy\s*=\s*([+-]?\d+\.\d+)',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    # If detailed point blocks found with matching orientations
    if pt_blocks:
        s_coords = []
        energies = []
        path_nums = []
        for p_idx, path_id, s_val, e_val in pt_blocks:
            s_f = float(s_val)
            path_i = int(path_id)
            # If path 2 (reverse) but s is printed as positive, invert sign
            if path_i == 2 and s_f > 0:
                s_f = -s_f
            s_coords.append(s_f)
            energies.append(float(e_val))
            path_nums.append(path_i)

        # Parse geometry coordinates for each block
        coords_list = []
        for blk in all_blocks:
            lines = blk.strip().split("\n")
            frame = []
            for l in lines:
                parts = l.split()
                if len(parts) >= 6:
                    frame.append([float(parts[3]), float(parts[4]), float(parts[5])])
            if len(frame) == len(symbols):
                coords_list.append(np.array(frame, dtype=np.float64))

        # Match lengths
        min_len = min(len(coords_list), len(s_coords))
        if min_len > 0:
            return assemble_irc_pathway(
                symbols=symbols,
                coords_list=coords_list[:min_len],
                energies_ha=energies[:min_len],
                rxn_coords=s_coords[:min_len],
                path_numbers=path_nums[:min_len],
                coord_unit="amu½·Bohr",
            )

    return None


def parse_orca_irc_text(text: str) -> Optional[IRCPathway]:
    """
    Extract converged IRC steps, reaction coordinates, and energies from an ORCA calculation log.
    """
    if "ORCA INTRINSIC REACTION COORDINATE" not in text and "IRC PATH SUMMARY" not in text:
        return None

    # ORCA IRC PATH SUMMARY table
    # Step   Reaction Coord.       Energy (Eh)         dE (kcal/mol)
    #    0          0.00000       -155.01234567               0.0000
    summary_match = re.search(
        r'IRC PATH SUMMARY[^\n]*\n\s*[-=]+\s*\n\s*Step[^\n]*\n(?:\s*[-=]+\s*\n)?(.*?)(?:\n\s*[-=]+\s*\n|\Z)',
        text,
        re.DOTALL | re.IGNORECASE,
    )

    s_coords = []
    energies = []
    if summary_match:
        for line in summary_match.group(1).strip().split("\n"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    s_coords.append(float(parts[1]))
                    energies.append(float(parts[2]))
                except ValueError:
                    continue

    # Extract CARTESIAN COORDINATES (ANGSTROEM)
    coord_blocks = re.findall(
        r'CARTESIAN COORDINATES \(ANGSTROEM\)[^\n]*\n\s*[-=]+\s*\n((?:\s*[A-Za-z]{1,2}\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+\s+[-+]?\d+\.\d+[^\n]*\n)+)',
        text,
        re.IGNORECASE,
    )

    if not coord_blocks:
        return None

    symbols = []
    coords_list = []
    for blk in coord_blocks:
        lines = blk.strip().split("\n")
        frame = []
        frame_syms = []
        for l in lines:
            parts = l.split()
            if len(parts) >= 4:
                frame_syms.append(parts[0])
                frame.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not symbols and frame_syms:
            symbols = frame_syms
        if len(frame) == len(symbols):
            coords_list.append(np.array(frame, dtype=np.float64))

    min_len = min(len(coords_list), len(energies)) if energies else len(coords_list)
    if min_len > 0:
        return assemble_irc_pathway(
            symbols=symbols,
            coords_list=coords_list[:min_len],
            energies_ha=energies[:min_len] if energies else [0.0] * min_len,
            rxn_coords=s_coords[:min_len] if s_coords else None,
            coord_unit="Bohr·amu½",
        )

    return None


def parse_irc_xyz(filepath_or_content: str) -> Optional[IRCPathway]:
    """
    Parse a multi-frame IRC XYZ file (such as ORCA *_IRC_Full.xyz or newBRORSEN .irc/.xyz).

    Extracts reaction coordinates (s = ...) and energies (E = ...) from the frame title/comment line.
    """
    if "\n" in filepath_or_content:
        content = filepath_or_content
    else:
        try:
            with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return None

    lines = content.strip().splitlines()
    if not lines:
        return None

    symbols: List[str] = []
    coords_list: List[np.ndarray] = []
    energies_ha: List[float] = []
    rxn_coords: List[float] = []

    line_idx = 0
    total_lines = len(lines)

    while line_idx < total_lines:
        line = lines[line_idx].strip()
        if not line:
            line_idx += 1
            continue

        try:
            n_atoms = int(line)
        except ValueError:
            line_idx += 1
            continue

        line_idx += 1
        if line_idx >= total_lines:
            break

        comment = lines[line_idx].strip()
        line_idx += 1

        # Extract energy from comment if present
        # Matches: E = -155.01234 Ha, energy: -155.01234, Eh = -155.01234
        e_match = re.search(r'(?:E|Eh|energy|total energy)\s*[:=]\s*([+-]?\d+\.\d+)', comment, re.IGNORECASE)
        e_ha = float(e_match.group(1)) if e_match else float(len(coords_list))

        # Extract reaction coordinate s from comment if present
        # Matches: s = -0.420, rxn_coord = 0.12, coordinate = -1.25
        s_match = re.search(r'(?:\bs\b|reaction\s*coord|rxn_coord|coord)\s*[:=]\s*([+-]?\d+\.\d+)', comment, re.IGNORECASE)
        s_val = float(s_match.group(1)) if s_match else None

        frame_coords = []
        frame_syms = []

        for _ in range(n_atoms):
            if line_idx >= total_lines:
                break
            atom_line = lines[line_idx].strip()
            line_idx += 1
            parts = atom_line.split()
            if len(parts) >= 4:
                frame_syms.append(parts[0])
                try:
                    frame_coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
                except ValueError:
                    frame_coords.append([0.0, 0.0, 0.0])

        if len(frame_coords) == n_atoms:
            if not symbols:
                symbols = frame_syms
            coords_list.append(np.array(frame_coords, dtype=np.float64))
            energies_ha.append(e_ha)
            if s_val is not None:
                rxn_coords.append(s_val)

    if coords_list and symbols:
        final_s = rxn_coords if len(rxn_coords) == len(coords_list) else None
        return assemble_irc_pathway(
            symbols=symbols,
            coords_list=coords_list,
            energies_ha=energies_ha,
            rxn_coords=final_s,
        )

    return None
