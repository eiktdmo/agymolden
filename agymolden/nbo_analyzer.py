"""
Natural Bond Orbital (NBO) Second-Order Perturbation Theory Analyzer.

Parses and analyzes donor-acceptor interactions from Gaussian and ORCA (GENNBO) outputs:
- Second Order Perturbation Theory Analysis of Fock Matrix in NBO Basis (E(2) in kcal/mol).
- Identifies critical chemical stabilization effects:
    * Lone pair to antibond: n -> σ*, n -> π* (anomeric effect, resonance)
    * Hyperconjugation: σ -> σ*, σ -> π*
    * Conjugation / backbonding: π -> π*, d -> π*
- Extracts atom indices for donor and acceptor NBOs for 3D visualization.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional, Tuple, Set, Dict


@dataclass
class NBOPerturbation:
    donor_idx: int
    donor_label: str
    donor_type: str        # "BD", "LP", "LV", "RY", "CR"
    donor_atoms: List[int] # 0-based atom indices

    acceptor_idx: int
    acceptor_label: str
    acceptor_type: str     # "BD*", "LP*", "LV*", "RY*"
    acceptor_atoms: List[int]

    interaction_type: str  # "n -> σ*", "n -> π*", "σ -> σ*", "π -> π*", "Hyperconjugation", "Other"
    energy_kcal: float
    energy_diff_au: float
    fock_element_au: float


def _classify_interaction(d_type: str, d_sub: int, d_label: str, a_type: str, a_sub: int, a_label: str) -> str:
    """Classify the physical donor-acceptor interaction type."""
    is_lp = "LP" in d_type
    is_bd = "BD" in d_type and "BD*" not in d_type
    is_pi_donor = is_bd and (d_sub >= 2 or "pi" in d_label.lower())
    is_pi_acceptor = "BD*" in a_type and (a_sub >= 2 or "pi" in a_label.lower())
    is_sigma_acceptor = "BD*" in a_type and not is_pi_acceptor

    if is_lp and is_pi_acceptor:
        return "n -> π*"
    elif is_lp and is_sigma_acceptor:
        return "n -> σ*"
    elif is_pi_donor and is_pi_acceptor:
        return "π -> π*"
    elif is_bd and is_sigma_acceptor:
        return "σ -> σ* (Hyperconjugation)"
    elif is_bd and is_pi_acceptor:
        return "σ -> π*"
    elif is_lp:
        return "n -> Acceptor"
    return "Other"


def _extract_atom_indices_from_label(label: str) -> List[int]:
    """
    Extract 0-based atom indices from NBO labels like:
    'C 1 - H 2', 'N 5', 'O 3 - C 4'
    """
    matches = re.findall(r'[A-Za-z]+\s*(\d+)', label)
    indices = []
    for m in matches:
        try:
            indices.append(int(m) - 1)
        except ValueError:
            pass
    return indices


def parse_nbo_perturbation_table(text: str) -> List[NBOPerturbation]:
    """
    Parse Gaussian / ORCA NBO Second Order Perturbation Theory Analysis.
    
    Standard Gaussian / ORCA table format:
      Donor NBO (i)                       Acceptor NBO (j)      kcal/mol   a.u.    a.u.
     =======================================================================================
       1. BD ( 1) C   1 - H   2       /111. RY*( 1) C   1             0.63    2.44   0.035
      13. LP ( 1) N   5               / 74. BD*( 1) C   1 - C   4    12.45    0.42   0.065
    """
    interactions: List[NBOPerturbation] = []
    if not text:
        return interactions

    # Find start of table
    start_match = re.search(
        r"SECOND\s+ORDER\s+PERTURBATION\s+THEORY\s+ANALYSIS\s+OF\s+FOCK\s+MATRIX\s+IN\s+NBO\s+BASIS",
        text,
        re.IGNORECASE,
    )
    if not start_match:
        return interactions

    table_chunk = text[start_match.end():]
    # Find table end (usually dotted line or next section)
    end_match = re.search(r"\n\s*(?:Natural\s+Bond\s+Orbitals|NATURAL\s+LOCALIZED|Summary\s+of\s+Natural)", table_chunk, re.IGNORECASE)
    if end_match:
        table_chunk = table_chunk[:end_match.start()]

    # Line regex for matching table entries
    # E.g. "   1. BD ( 1) C   1 - H   2       / 74. BD*( 1) C   1 - C   4     3.12    1.15   0.054"
    line_pattern = re.compile(
        r"^\s*(\d+)\.\s+([A-Za-z]+\*?)\s*\(\s*(\d+)\s*\)\s*([^\/]+?)\s*\/\s*(\d+)\.\s+([A-Za-z]+\*?)\s*\(\s*(\d+)\s*\)\s*(.+?)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s*$",
        re.MULTILINE,
    )

    for m in line_pattern.finditer(table_chunk):
        try:
            d_idx = int(m.group(1))
            d_type = m.group(2).strip()
            d_sub = int(m.group(3))
            d_desc = m.group(4).strip()

            a_idx = int(m.group(5))
            a_type = m.group(6).strip()
            a_sub = int(m.group(7))
            a_desc = m.group(8).strip()

            e2 = float(m.group(9))
            ediff = float(m.group(10))
            fock = float(m.group(11))

            d_label = f"{d_type}({d_sub}) {d_desc}"
            a_label = f"{a_type}({a_sub}) {a_desc}"

            d_atoms = _extract_atom_indices_from_label(d_desc)
            a_atoms = _extract_atom_indices_from_label(a_desc)

            itype = _classify_interaction(d_type, d_sub, d_label, a_type, a_sub, a_label)

            interactions.append(
                NBOPerturbation(
                    donor_idx=d_idx,
                    donor_label=d_label,
                    donor_type=d_type,
                    donor_atoms=d_atoms,
                    acceptor_idx=a_idx,
                    acceptor_label=a_label,
                    acceptor_type=a_type,
                    acceptor_atoms=a_atoms,
                    interaction_type=itype,
                    energy_kcal=e2,
                    energy_diff_au=ediff,
                    fock_element_au=fock,
                )
            )
        except Exception:
            continue

    # Sort descending by interaction energy E(2)
    interactions.sort(key=lambda x: x.energy_kcal, reverse=True)
    return interactions
