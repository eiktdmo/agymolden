"""
Multiwfn Integration & Wavefunction Analysis Module for AgyMolden.

Provides four essential quantum chemical wavefunction tools:
1. Mayer & Wiberg Lowdin Bond Orders & Atomic Valences
2. Conceptual DFT (CDFT): Condensed Fukui Functions (f+, f-, f0) & Dual Descriptor (Δf)
3. QTAIM / AIM Topology Analysis: Critical Points ((3,-3) Nuclear, (3,-1) Bond CPs)
4. Atomic Partial Charges: CM5 (Charge Model 5) & Hirshfeld charges

Supports both automated execution via local Multiwfn executable and
fast native fallback algorithms (via PySCF / NumPy) when Multiwfn is not installed.
"""

import os
import re
import shutil
import subprocess
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

log = logging.getLogger(__name__)


class MultiwfnNotFoundError(RuntimeError):
    """Raised when a feature needs the external Multiwfn program but it cannot be located."""

    def __init__(self, feature: str = "This analysis"):
        super().__init__(
            f"{feature} requires the external Multiwfn program, which was not found.\n\n"
            "Download it (free) from http://sobereva.com/multiwfn, unpack it anywhere, then either:\n"
            "  - put it on your PATH, or\n"
            "  - click 'Set Multiwfn path...' in the Multiwfn dialog and select the executable.\n\n"
            "Bond orders and Fukui indices still work without Multiwfn."
        )


PYSCF_MISSING_MESSAGE = (
    "This feature needs the optional 'pyscf' package, which is not available in this installation "
    "(the standalone executable does not bundle it, and PySCF has no native Windows build).\n\n"
    "Install it in a source install with:  pip install pyscf   (or: pip install .[qm])\n"
    "On Windows, use WSL or a Linux/macOS machine for this analysis."
)


def friendly_error_text(err: BaseException) -> str:
    """Return a user-friendly message for common analysis failures (missing PySCF / Multiwfn)."""
    if isinstance(err, ImportError) and "pyscf" in str(err).lower():
        return PYSCF_MISSING_MESSAGE
    return str(err)


_MULTIWFN_ROUTE_KEY = "multiwfn"


def _is_executable_file(path: Optional[str]) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def get_multiwfn_custom_path() -> str:
    """Return the user-configured Multiwfn path (empty string if unset)."""
    try:
        from .qm_detect import get_qm_route
        return get_qm_route(_MULTIWFN_ROUTE_KEY)
    except Exception:
        log.debug("Could not read saved Multiwfn path", exc_info=True)
        return ""


def save_multiwfn_custom_path(path: str) -> None:
    """Persist the user-selected Multiwfn executable path."""
    from .qm_detect import save_qm_route
    save_qm_route(_MULTIWFN_ROUTE_KEY, path)


def detect_multiwfn_executable() -> Optional[str]:
    """
    Locate the Multiwfn executable. Search order:
    1. User-configured path (saved in settings) or MULTIWFN_PATH environment variable
    2. System PATH ('Multiwfn', 'multiwfn', 'Multiwfn.exe')
    3. Common install folders in the user's home (~/bin, ~/programs/Multiwfn*, ~/Multiwfn*), and /usr/local/bin
    """
    import glob

    candidates: List[Optional[str]] = [get_multiwfn_custom_path(), os.environ.get("MULTIWFN_PATH")]
    for name in ("Multiwfn", "multiwfn", "Multiwfn.exe"):
        candidates.append(shutil.which(name))

    home = os.path.expanduser("~")
    exe_names = ("Multiwfn", "multiwfn", "Multiwfn.exe")
    for folder in (os.path.join(home, "bin"), "/usr/local/bin", "/opt/Multiwfn"):
        candidates.extend(os.path.join(folder, n) for n in exe_names)
    for pattern in ("programs/Multiwfn*", "Multiwfn*", "Programs/Multiwfn*", "Desktop/Multiwfn*"):
        for folder in sorted(glob.glob(os.path.join(home, pattern)), reverse=True):
            candidates.extend(os.path.join(folder, n) for n in exe_names)

    for p in candidates:
        if p and os.path.isdir(p):
            for n in exe_names:
                if _is_executable_file(os.path.join(p, n)):
                    return os.path.abspath(os.path.join(p, n))
        if _is_executable_file(p):
            return os.path.abspath(p)
    return None


@dataclass
class BondOrderResult:
    bonds: List[Tuple[int, int, str, str, float]]  # (atom_i, atom_j, sym_i, sym_j, order)
    valences: Dict[int, float]                     # atom_index (1-based) -> total valence
    method: str = "Mayer"                          # "Mayer" or "Wiberg (Lowdin)"
    raw_output: str = ""


@dataclass
class FukuiResult:
    atoms: List[Dict[str, Any]]                    # list of dicts: {'index': i, 'symbol': sym, 'f_minus': ..., 'f_plus': ..., 'f_zero': ..., 'dual': ..., 'pred': ...}
    method: str = "Lowdin Frontier Orbital"
    raw_output: str = ""


@dataclass
class CriticalPoint:
    index: int
    cp_type: str                                   # "(3,-3)" Nuclear, "(3,-1)" Bond, "(3,+1)" Ring, "(3,+3)" Cage
    type_name: str                                 # "Nuclear (Attractor)", "Bond Critical Point (BCP)", etc.
    coords_bohr: Tuple[float, float, float]
    coords_angstrom: Tuple[float, float, float]
    connected_atoms: Optional[Tuple[int, int]] = None


@dataclass
class QTAIMResult:
    critical_points: List[CriticalPoint]
    num_nuclear: int = 0
    num_bcp: int = 0
    num_rcp: int = 0
    num_ccp: int = 0
    raw_output: str = ""


@dataclass
class PartialChargesResult:
    atoms: List[Dict[str, Any]]                    # list of dicts: {'index': i, 'symbol': sym, 'hirshfeld': ..., 'cm5': ...}
    dipole_hirshfeld: Optional[Tuple[float, float, float]] = None
    dipole_cm5: Optional[Tuple[float, float, float]] = None
    sum_hirshfeld: float = 0.0
    sum_cm5: float = 0.0
    raw_output: str = ""


# =========================================================================
# Feature 1: Mayer & Wiberg Lowdin Bond Orders
# =========================================================================

def compute_mayer_and_wiberg_orders(molden_path: str, min_threshold: float = 0.05) -> Tuple[BondOrderResult, BondOrderResult]:
    """
    Computes both Mayer bond orders and Wiberg (Lowdin basis) bond orders.
    Uses Multiwfn if available; falls back to PySCF density matrix formulation.
    """
    multiwfn_bin = detect_multiwfn_executable()
    if multiwfn_bin:
        try:
            return _compute_bond_orders_multiwfn(multiwfn_bin, molden_path, min_threshold)
        except Exception as e:
            log.warning(f"Multiwfn bond order execution failed ({e}), attempting PySCF fallback...")

    return _compute_bond_orders_pyscf(molden_path, min_threshold)


def _compute_bond_orders_multiwfn(multiwfn_bin: str, molden_path: str, min_threshold: float = 0.05) -> Tuple[BondOrderResult, BondOrderResult]:
    # Multiwfn commands:
    # 9 (Bond order) -> 1 (Mayer) -> n -> 3 (Wiberg Lowdin) -> n -> 0 (Return) -> q (Quit)
    inp = f"{molden_path}\n9\n1\nn\n3\nn\n0\nq\n"
    res = subprocess.run(
        [multiwfn_bin],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    stdout = res.stdout

    def _parse_chunk(chunk_text: str):
        bonds = []
        valences = {}
        for line in chunk_text.splitlines():
            line_str = line.strip()
            if line_str.startswith("#") and ":" in line_str:
                m = re.search(r'(\d+)\s*\(\s*([A-Za-z]+)\s*\)\s+(\d+)\s*\(\s*([A-Za-z]+)\s*\)\s+([+-]?\d+\.?\d*)', line_str)
                if m:
                    idx1, sym1, idx2, sym2, bo_val = m.groups()
                    order = float(bo_val)
                    if abs(order) >= min_threshold:
                        bonds.append((int(idx1), int(idx2), sym1, sym2, order))
            elif line_str.startswith("Atom ") and ":" in line_str:
                try:
                    atom_part, vals_part = line_str.split(":", 1)
                    idx = int(atom_part.replace("Atom", "").split("(")[0].strip())
                    val = float(vals_part.split()[0])
                    valences[idx] = val
                except Exception:
                    log.debug("Ignored non-fatal exception", exc_info=True)
        return bonds, valences

    parts = stdout.split("Bond orders with absolute value >=")
    mayer_chunk = parts[1] if len(parts) > 1 else ""
    wiberg_chunk = parts[2] if len(parts) > 2 else ""

    mayer_bonds, mayer_valences = _parse_chunk(mayer_chunk)
    wiberg_bonds, wiberg_valences = _parse_chunk(wiberg_chunk)

    res_mayer = BondOrderResult(bonds=mayer_bonds, valences=mayer_valences, method="Mayer (Multiwfn)", raw_output=mayer_chunk)
    res_wiberg = BondOrderResult(bonds=wiberg_bonds, valences=wiberg_valences, method="Wiberg Lowdin (Multiwfn)", raw_output=wiberg_chunk)
    return res_mayer, res_wiberg


def _compute_bond_orders_pyscf(molden_path: str, min_threshold: float = 0.05) -> Tuple[BondOrderResult, BondOrderResult]:
    from pyscf.tools import molden
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(molden_path)
    s = mol.intor_symmetric('int1e_ovlp')

    # Density matrix P
    occ = np.array(mo_occ)
    dm = mo_coeff @ np.diag(occ) @ mo_coeff.T

    # 1. Mayer formulation: B_AB = sum_{i in A, j in B} (PS)_ij * (PS)_ji
    ps = dm @ s
    ao_slices = mol.aoslice_by_atom()
    natm = mol.natm

    mayer_bonds = []
    mayer_val = {i + 1: 0.0 for i in range(natm)}
    for A in range(natm):
        pA0, pA1 = ao_slices[A, 2], ao_slices[A, 3]
        for B in range(A + 1, natm):
            pB0, pB1 = ao_slices[B, 2], ao_slices[B, 3]
            ps_AB = ps[pA0:pA1, pB0:pB1]
            ps_BA = ps[pB0:pB1, pA0:pA1]
            bo = float(np.sum(ps_AB * ps_BA.T))
            if abs(bo) >= min_threshold:
                mayer_bonds.append((A + 1, B + 1, mol.atom_symbol(A), mol.atom_symbol(B), bo))
            mayer_val[A + 1] += bo
            mayer_val[B + 1] += bo

    # 2. Wiberg in Lowdin basis: P_L = S^(1/2) P S^(1/2)
    s_eigval, s_eigvec = np.linalg.eigh(s)
    s_half = s_eigvec @ np.diag(np.sqrt(np.maximum(s_eigval, 1e-14))) @ s_eigvec.T
    p_lowdin = s_half @ dm @ s_half

    wiberg_bonds = []
    wiberg_val = {i + 1: 0.0 for i in range(natm)}
    for A in range(natm):
        pA0, pA1 = ao_slices[A, 2], ao_slices[A, 3]
        for B in range(A + 1, natm):
            pB0, pB1 = ao_slices[B, 2], ao_slices[B, 3]
            pL_AB = p_lowdin[pA0:pA1, pB0:pB1]
            bo = float(np.sum(pL_AB**2))
            if abs(bo) >= min_threshold:
                wiberg_bonds.append((A + 1, B + 1, mol.atom_symbol(A), mol.atom_symbol(B), bo))
            wiberg_val[A + 1] += bo
            wiberg_val[B + 1] += bo

    res_mayer = BondOrderResult(bonds=mayer_bonds, valences=mayer_val, method="Mayer (PySCF)")
    res_wiberg = BondOrderResult(bonds=wiberg_bonds, valences=wiberg_val, method="Wiberg Lowdin (PySCF)")
    return res_mayer, res_wiberg


# =========================================================================
# Feature 2: Conceptual DFT & Condensed Fukui Functions
# =========================================================================

def compute_condensed_fukui(molden_path: str, population_scheme: str = "Lowdin") -> FukuiResult:
    """
    Computes condensed Fukui indices (f-, f+, f0) and Dual Descriptor (Δf = f+ - f-).
    Electrophilic attack favors high f- (HOMO electron donor).
    Nucleophilic attack favors high f+ (LUMO electron acceptor).
    Dual descriptor Δf > 0 indicates electrophilic site, Δf < 0 indicates nucleophilic site.
    """
    from pyscf.tools import molden
    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(molden_path)
    occ = np.array(mo_occ)
    occ_idx = np.where(occ > 0)[0]
    if len(occ_idx) == 0:
        raise ValueError("No occupied orbitals found in Molden file for Fukui analysis.")

    homo_idx = int(occ_idx[-1])
    lumo_idx = homo_idx + 1 if homo_idx + 1 < len(occ) else homo_idx

    s = mol.intor_symmetric('int1e_ovlp')
    c_homo = mo_coeff[:, homo_idx]
    c_lumo = mo_coeff[:, lumo_idx]

    ao_slices = mol.aoslice_by_atom()
    atoms_data = []

    if population_scheme.lower().startswith("mulliken"):
        p_homo = np.outer(c_homo, c_homo) * s
        p_lumo = np.outer(c_lumo, c_lumo) * s
        for ia in range(mol.natm):
            p0, p1 = ao_slices[ia, 2], ao_slices[ia, 3]
            f_minus = float(np.sum(p_homo[p0:p1, :]))
            f_plus = float(np.sum(p_lumo[p0:p1, :]))
            f_zero = 0.5 * (f_plus + f_minus)
            dual = f_plus - f_minus
            pred = "Nucleophilic Site (Δf < 0)" if dual < -0.05 else ("Electrophilic Site (Δf > 0)" if dual > 0.05 else "Neutral")
            atoms_data.append({
                "index": ia + 1,
                "symbol": mol.atom_symbol(ia),
                "f_minus": f_minus,
                "f_plus": f_plus,
                "f_zero": f_zero,
                "dual": dual,
                "site_pred": pred,
            })
    else:  # Default to Lowdin (more basis-set stable and non-negative)
        s_eigval, s_eigvec = np.linalg.eigh(s)
        s_half = s_eigvec @ np.diag(np.sqrt(np.maximum(s_eigval, 1e-14))) @ s_eigvec.T
        c_homo_low = s_half @ c_homo
        c_lumo_low = s_half @ c_lumo

        for ia in range(mol.natm):
            p0, p1 = ao_slices[ia, 2], ao_slices[ia, 3]
            f_minus = float(np.sum(c_homo_low[p0:p1]**2))
            f_plus = float(np.sum(c_lumo_low[p0:p1]**2))
            f_zero = 0.5 * (f_plus + f_minus)
            dual = f_plus - f_minus
            pred = "Nucleophilic Attack (Δf < 0)" if dual < -0.05 else ("Electrophilic Attack (Δf > 0)" if dual > 0.05 else "Neutral / Balanced")
            atoms_data.append({
                "index": ia + 1,
                "symbol": mol.atom_symbol(ia),
                "f_minus": f_minus,
                "f_plus": f_plus,
                "f_zero": f_zero,
                "dual": dual,
                "site_pred": pred,
            })

    return FukuiResult(atoms=atoms_data, method=f"Frontier Orbital {population_scheme}")


# =========================================================================
# Feature 3: QTAIM / AIM Topology Analysis
# =========================================================================

def compute_qtaim_topology(molden_path: str) -> QTAIMResult:
    """
    Runs Multiwfn topology analysis (Option 2) to search for:
    - (3,-3) Nuclear Critical Points
    - (3,-1) Bond Critical Points (BCP) between atom pairs
    """
    multiwfn_bin = detect_multiwfn_executable()
    if not multiwfn_bin:
        raise MultiwfnNotFoundError("QTAIM topology analysis")

    # Multiwfn Option 2 workflow:
    # 2 (Topology) -> 2 (Nuclear CPs) -> 3 (Atomic pair midpoints) -> -4 (Modify/export) -> -1 (Print summary in Angstrom) -> 0 -> -10 (Return) -> q
    inp = f"{molden_path}\n2\n2\n3\n-4\n-1\n0\n-10\nq\n"
    res = subprocess.run(
        [multiwfn_bin],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    stdout = res.stdout

    cps = []
    num_nuc = 0
    num_bcp = 0
    num_rcp = 0
    num_ccp = 0

    lines = stdout.splitlines()
    in_summary = False
    for line in lines:
        line_str = line.strip()
        if "Summary of found CPs:" in line_str:
            in_summary = True
            continue
        if in_summary:
            if "Modify or export found CPs" in line_str or line_str.startswith("----") or "Return" in line_str:
                if cps:
                    break
            parts = line_str.split()
            if len(parts) >= 5:
                try:
                    idx = int(parts[0])
                    x_ang = float(parts[1])
                    y_ang = float(parts[2])
                    z_ang = float(parts[3])
                    cp_type = parts[4]

                    # Convert to bohr (1 Bohr = 0.529177210903 Å)
                    bohr_conv = 1.0 / 0.529177210903
                    x_b = x_ang * bohr_conv
                    y_b = y_ang * bohr_conv
                    z_b = z_ang * bohr_conv

                    type_name = "Critical Point"
                    if "(3,-3)" in cp_type:
                        type_name = "Nuclear CP (Attractor)"
                        num_nuc += 1
                    elif "(3,-1)" in cp_type:
                        type_name = "Bond Critical Point (BCP)"
                        num_bcp += 1
                    elif "(3,+1)" in cp_type:
                        type_name = "Ring Critical Point (RCP)"
                        num_rcp += 1
                    elif "(3,+3)" in cp_type:
                        type_name = "Cage Critical Point (CCP)"
                        num_ccp += 1

                    cps.append(CriticalPoint(
                        index=idx,
                        cp_type=cp_type,
                        type_name=type_name,
                        coords_bohr=(x_b, y_b, z_b),
                        coords_angstrom=(x_ang, y_ang, z_ang),
                    ))
                except (ValueError, IndexError):
                    continue

    return QTAIMResult(
        critical_points=cps,
        num_nuclear=num_nuc,
        num_bcp=num_bcp,
        num_rcp=num_rcp,
        num_ccp=num_ccp,
        raw_output=stdout,
    )


# =========================================================================
# Feature 4: Hirshfeld & CM5 Atomic Charges
# =========================================================================

def compute_hirshfeld_and_cm5_charges(molden_path: str) -> PartialChargesResult:
    """
    Calculates Hirshfeld and Charge Model 5 (CM5) atomic partial charges.
    Uses Multiwfn Option 7 -> 16 -> 1.
    """
    multiwfn_bin = detect_multiwfn_executable()
    if not multiwfn_bin:
        raise MultiwfnNotFoundError("CM5 / Hirshfeld charges")

    inp = f"{molden_path}\n7\n16\n1\nn\n0\nq\n"
    res = subprocess.run(
        [multiwfn_bin],
        input=inp,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    stdout = res.stdout

    atoms_list = []
    sum_cm5 = 0.0
    sum_hirshfeld = 0.0
    dipole_cm5 = None
    dipole_hirsh = None

    lines = stdout.splitlines()
    in_cm5_block = False
    for line in lines:
        line_str = line.strip()
        if "---------- CM5 charges ----------" in line_str:
            in_cm5_block = True
            continue
        if in_cm5_block:
            if line_str.startswith("Summing up all CM5 charges:"):
                try:
                    sum_cm5 = float(line_str.split(":")[-1].strip())
                except Exception:
                    log.debug("Ignored non-fatal exception", exc_info=True)
            elif line_str.startswith("X/Y/Z of dipole moment from CM5 charges"):
                try:
                    parts = line_str.split("charges")[-1].replace("a.u.", "").split()
                    dipole_cm5 = (float(parts[0]), float(parts[1]), float(parts[2]))
                except Exception:
                    log.debug("Ignored non-fatal exception", exc_info=True)
            elif line_str.startswith("Final atomic charges:") or line_str.startswith("Calculation took up"):
                in_cm5_block = False
            elif line_str.startswith("Atom:"):
                # Format: Atom:    1O   CM5 charge:   -0.644900  Hirshfeld charge:   -0.316358
                parts = line_str.split()
                try:
                    atom_raw = parts[1]
                    # separate digits and letters, e.g. '1O' -> 1, 'O'
                    idx_str = "".join([c for c in atom_raw if c.isdigit()])
                    sym_str = "".join([c for c in atom_raw if c.isalpha()])
                    idx = int(idx_str) if idx_str else len(atoms_list) + 1
                    sym = sym_str if sym_str else "X"

                    cm5_idx = parts.index("charge:") + 1
                    cm5_val = float(parts[cm5_idx])

                    hirsh_val = 0.0
                    if "charge:" in parts[cm5_idx + 1:]:
                        hirsh_idx = parts.index("charge:", cm5_idx + 1) + 1
                        hirsh_val = float(parts[hirsh_idx])

                    atoms_list.append({
                        "index": idx,
                        "symbol": sym,
                        "cm5": cm5_val,
                        "hirshfeld": hirsh_val,
                    })
                    sum_hirshfeld += hirsh_val
                except Exception as e:
                    log.debug(f"Could not parse CM5 line: {line_str} ({e})")

    return PartialChargesResult(
        atoms=atoms_list,
        dipole_cm5=dipole_cm5,
        dipole_hirshfeld=dipole_hirsh,
        sum_cm5=sum_cm5,
        sum_hirshfeld=sum_hirshfeld,
        raw_output=stdout,
    )
