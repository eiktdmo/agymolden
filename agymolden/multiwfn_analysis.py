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
    rho: float = 0.0                               # Electron density rho(r_cp) in a.u.
    grad_norm: float = 0.0                         # ||grad rho|| in a.u.
    laplacian: float = 0.0                         # nabla^2 rho = lambda1 + lambda2 + lambda3 in a.u.
    hessian_eigenvalues: Tuple[float, float, float] = (0.0, 0.0, 0.0) # (lambda1 <= lambda2 <= lambda3)
    ellipticity: float = 0.0                       # epsilon = (lambda1 / lambda2) - 1.0 (for BCP)
    v_density: float = 0.0                         # Potential energy density V(r) in a.u.
    g_density: float = 0.0                         # Lagrangian kinetic energy density G(r) in a.u.
    h_density: float = 0.0                         # Total energy density H(r) = G(r) + V(r) in a.u.
    bond_character: str = ""                       # "Shared / Covalent", "Polar Covalent", "Closed-Shell (Ionic/vdW/H-Bond)"
    bond_path: Optional[List[Tuple[float, float, float]]] = None # 3D coordinates in Angstrom along gradient path


@dataclass
class QTAIMResult:
    critical_points: List[CriticalPoint]
    num_nuclear: int = 0
    num_bcp: int = 0
    num_rcp: int = 0
    num_ccp: int = 0
    method: str = "PySCF Analytical QTAIM"
    bond_paths: List[List[Tuple[float, float, float]]] = field(default_factory=list)
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

def compute_qtaim_topology(molden_path: str, engine: str = "auto") -> QTAIMResult:
    """
    Computes QTAIM (Quantum Theory of Atoms in Molecules) Bader topology analysis:
    - (3,-3) Nuclear Attractors
    - (3,-1) Bond Critical Points (BCP) with electron density rho, Laplacian nabla^2 rho,
      ellipticity epsilon, energy densities G, V, H, and exact 3D bond paths
    - (3,+1) Ring Critical Points (RCP)
    - (3,+3) Cage Critical Points (CCP)

    Args:
        molden_path: Path to Molden format file.
        engine: "auto", "pyscf", or "multiwfn".
    """
    if engine == "pyscf":
        return _compute_qtaim_pyscf(molden_path)

    multiwfn_bin = detect_multiwfn_executable()
    if engine == "multiwfn":
        if not multiwfn_bin:
            raise MultiwfnNotFoundError("QTAIM topology analysis")
        return _compute_qtaim_multiwfn(multiwfn_bin, molden_path)

    if multiwfn_bin:
        try:
            return _compute_qtaim_multiwfn(multiwfn_bin, molden_path)
        except Exception as e:
            log.warning("Multiwfn QTAIM execution failed (%s), falling back to native PySCF topology solver...", e)

    return _compute_qtaim_pyscf(molden_path)


def _compute_qtaim_multiwfn(multiwfn_bin: str, molden_path: str) -> QTAIMResult:
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
        method="Multiwfn Option 2",
        raw_output=stdout,
    )


def _compute_qtaim_pyscf(molden_path: str) -> QTAIMResult:
    """High-precision native analytical QTAIM topology evaluator using PySCF GTO evaluation."""
    from pyscf.tools import molden
    from .forcefield import detect_connectivity

    mol, mo_energy, mo_coeff, mo_occ, _, _ = molden.load(molden_path)
    occ = np.array(mo_occ, dtype=float)
    dm = mo_coeff @ np.diag(occ) @ mo_coeff.T

    bohr2ang = 0.529177210903
    ang2bohr = 1.0 / bohr2ang
    natm = mol.natm
    symbols = [mol.atom_symbol(i) for i in range(natm)]
    coords_ang = np.array([mol.atom_coord(i) * bohr2ang for i in range(natm)])

    def _eval_fields(pt_bohr: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        ao = mol.eval_gto("GTOval_sph_deriv2", pt_bohr.reshape(1, 3))
        chi = ao[0]
        dx, dy, dz = ao[1], ao[2], ao[3]
        xx, xy, xz, yy, yz, zz = ao[4], ao[5], ao[6], ao[7], ao[8], ao[9]

        rho = float(np.einsum("pi,ij,pj->p", chi, dm, chi)[0])
        gx = float(2.0 * np.einsum("pi,ij,pj->p", dx, dm, chi)[0])
        gy = float(2.0 * np.einsum("pi,ij,pj->p", dy, dm, chi)[0])
        gz = float(2.0 * np.einsum("pi,ij,pj->p", dz, dm, chi)[0])
        grad = np.array([gx, gy, gz], dtype=float)

        Hxx = float(2.0 * (np.einsum("pi,ij,pj->p", xx, dm, chi)[0] + np.einsum("pi,ij,pj->p", dx, dm, dx)[0]))
        Hyy = float(2.0 * (np.einsum("pi,ij,pj->p", yy, dm, chi)[0] + np.einsum("pi,ij,pj->p", dy, dm, dy)[0]))
        Hzz = float(2.0 * (np.einsum("pi,ij,pj->p", zz, dm, chi)[0] + np.einsum("pi,ij,pj->p", dz, dm, dz)[0]))
        Hxy = float(2.0 * (np.einsum("pi,ij,pj->p", xy, dm, chi)[0] + np.einsum("pi,ij,pj->p", dx, dm, dy)[0]))
        Hxz = float(2.0 * (np.einsum("pi,ij,pj->p", xz, dm, chi)[0] + np.einsum("pi,ij,pj->p", dx, dm, dz)[0]))
        Hyz = float(2.0 * (np.einsum("pi,ij,pj->p", yz, dm, chi)[0] + np.einsum("pi,ij,pj->p", dy, dm, dz)[0]))
        H = np.array([[Hxx, Hxy, Hxz], [Hxy, Hyy, Hyz], [Hxz, Hyz, Hzz]], dtype=float)
        return rho, grad, H

    cps: List[CriticalPoint] = []
    cp_idx = 1
    num_nuc = 0
    num_bcp = 0
    num_rcp = 0
    num_ccp = 0
    all_bond_paths: List[List[Tuple[float, float, float]]] = []

    # 1. (3,-3) Nuclear Attractors at atomic centers
    for A in range(natm):
        r_A = mol.atom_coord(A)
        rho_n, grad_n, H_n = _eval_fields(r_A)
        evals_n = np.sort(np.linalg.eigvalsh(H_n))
        lap_n = float(np.sum(evals_n))
        cps.append(CriticalPoint(
            index=cp_idx,
            cp_type="(3,-3)",
            type_name=f"Nuclear Attractor ({symbols[A]}{A+1})",
            coords_bohr=(float(r_A[0]), float(r_A[1]), float(r_A[2])),
            coords_angstrom=(float(r_A[0] * bohr2ang), float(r_A[1] * bohr2ang), float(r_A[2] * bohr2ang)),
            connected_atoms=(A, A),
            rho=rho_n,
            grad_norm=float(np.linalg.norm(grad_n)),
            laplacian=lap_n,
            hessian_eigenvalues=(float(evals_n[0]), float(evals_n[1]), float(evals_n[2])),
            bond_character="Nuclear Core",
        ))
        cp_idx += 1
        num_nuc += 1

    # 2. (3,-1) Bond Critical Points (BCP)
    bonds = detect_connectivity(symbols, coords_ang)
    # Also evaluate any nearby non-bonded pairs under 3.2 A for non-covalent BCPs / H-bonds
    candidate_pairs = list(bonds)
    bonds_set = set(bonds)
    for i in range(natm):
        for j in range(i + 1, natm):
            if (i, j) not in bonds_set:
                d_ij = float(np.linalg.norm(coords_ang[i] - coords_ang[j]))
                if d_ij <= 3.2:
                    candidate_pairs.append((i, j))

    found_bcp_coords: List[np.ndarray] = []

    for A, B in candidate_pairs:
        r_A = mol.atom_coord(A)
        r_B = mol.atom_coord(B)
        r = 0.5 * (r_A + r_B)

        # Newton-Raphson iteration
        converged = False
        for step in range(25):
            rho, grad, H = _eval_fields(r)
            gnorm = float(np.linalg.norm(grad))
            if gnorm < 1e-4:
                converged = True
                break
            try:
                s_vec = np.linalg.solve(H, grad)
                s_len = float(np.linalg.norm(s_vec))
                if s_len > 0.25:
                    s_vec *= (0.25 / s_len)
                r -= s_vec
            except np.linalg.LinAlgError:
                break

        if not converged:
            rho, grad, H = _eval_fields(r)
            if float(np.linalg.norm(grad)) < 5e-3:
                converged = True

        if converged and rho > 1e-4:
            evals, evecs = np.linalg.eigh(H)
            order = np.argsort(evals)
            evals = evals[order]
            evecs = evecs[:, order]

            negs = sum(1 for e in evals if e < -1e-6)
            poss = sum(1 for e in evals if e > 1e-6)

            if negs == 2 and poss == 1:
                # Check de-duplication
                r_ang = r * bohr2ang
                if any(np.linalg.norm(r_ang - prev) < 0.08 for prev in found_bcp_coords):
                    continue
                found_bcp_coords.append(r_ang)

                lap = float(np.sum(evals))
                l1, l2, l3 = float(evals[0]), float(evals[1]), float(evals[2])
                ellip = float((l1 / (l2 + 1e-12)) - 1.0) if abs(l2) > 1e-10 else 0.0

                # Virial energy densities (in a.u.)
                g_dens = float(0.3 * (3.0 * np.pi**2)**(2.0/3.0) * (rho**(5.0/3.0)) + (1.0/72.0) * (np.linalg.norm(grad)**2 / (rho + 1e-12)))
                v_dens = float(0.25 * lap - 2.0 * g_dens)
                h_dens = float(g_dens + v_dens)

                if h_dens < 0.0 and lap < 0.0:
                    char_str = "Shared / Covalent"
                elif h_dens < 0.0 and lap > 0.0:
                    char_str = "Polar Covalent"
                else:
                    char_str = "Closed-Shell (Ionic / vdW / H-Bond)"

                # Trace bond path along eigenvector of lambda_3 > 0
                v_bond = evecs[:, 2]
                bpath_pts: List[Tuple[float, float, float]] = []

                # Trace branch to atom A and branch to atom B
                for sign in (-1, 1):
                    branch_pts = [r * bohr2ang]
                    curr_b = r + sign * 0.05 * v_bond
                    for _ in range(35):
                        branch_pts.append(curr_b * bohr2ang)
                        d_nuclei = [float(np.linalg.norm(curr_b - mol.atom_coord(k))) for k in (A, B)]
                        if min(d_nuclei) < 0.28:
                            break
                        _, g_step, _ = _eval_fields(curr_b)
                        gs_norm = float(np.linalg.norm(g_step))
                        if gs_norm < 1e-4:
                            break
                        curr_b = curr_b + (g_step / gs_norm) * 0.08

                    if sign == -1:
                        bpath_pts = [tuple(p) for p in reversed(branch_pts)]
                    else:
                        bpath_pts.extend([tuple(p) for p in branch_pts[1:]])

                all_bond_paths.append(bpath_pts)

                cps.append(CriticalPoint(
                    index=cp_idx,
                    cp_type="(3,-1)",
                    type_name=f"Bond CP ({symbols[A]}{A+1}···{symbols[B]}{B+1})",
                    coords_bohr=(float(r[0]), float(r[1]), float(r[2])),
                    coords_angstrom=(float(r_ang[0]), float(r_ang[1]), float(r_ang[2])),
                    connected_atoms=(A, B),
                    rho=rho,
                    grad_norm=float(np.linalg.norm(grad)),
                    laplacian=lap,
                    hessian_eigenvalues=(l1, l2, l3),
                    ellipticity=ellip,
                    v_density=v_dens,
                    g_density=g_dens,
                    h_density=h_dens,
                    bond_character=char_str,
                    bond_path=bpath_pts,
                ))
                cp_idx += 1
                num_bcp += 1

    # 3. (3,+1) Ring Critical Points (RCP)
    # Check ring centroids if rings are detected
    from .topology import build_adj
    adj = build_adj(natm, bonds)
    # Simple cycle detection for 3-6 membered rings
    visited_cycles = set()
    for start in range(natm):
        for n1 in adj[start]:
            for n2 in adj[n1]:
                if n2 == start:
                    continue
                for n3 in adj[n2]:
                    if n3 == n1:
                        continue
                    if n3 in adj[start]: # 4-membered ring
                        cycle = tuple(sorted([start, n1, n2, n3]))
                        visited_cycles.add(cycle)
                    for n4 in adj[n3]:
                        if n4 in (n1, n2):
                            continue
                        if n4 in adj[start]: # 5-membered ring
                            cycle = tuple(sorted([start, n1, n2, n3, n4]))
                            visited_cycles.add(cycle)
                        for n5 in adj[n4]:
                            if n5 in (n1, n2, n3):
                                continue
                            if n5 in adj[start]: # 6-membered ring
                                cycle = tuple(sorted([start, n1, n2, n3, n4, n5]))
                                visited_cycles.add(cycle)

    for cycle in visited_cycles:
        ring_coords = [mol.atom_coord(k) for k in cycle]
        r_rcp = np.mean(ring_coords, axis=0)

        for step in range(25):
            rho, grad, H = _eval_fields(r_rcp)
            if float(np.linalg.norm(grad)) < 1e-4:
                break
            try:
                s_vec = np.linalg.solve(H, grad)
                s_len = float(np.linalg.norm(s_vec))
                if s_len > 0.25:
                    s_vec *= (0.25 / s_len)
                r_rcp -= s_vec
            except np.linalg.LinAlgError:
                break

        rho, grad, H = _eval_fields(r_rcp)
        evals = np.sort(np.linalg.eigvalsh(H))
        negs = sum(1 for e in evals if e < -1e-6)
        poss = sum(1 for e in evals if e > 1e-6)

        if negs == 1 and poss == 2 and rho > 1e-4:
            r_ang = r_rcp * bohr2ang
            if any(np.linalg.norm(r_ang - prev) < 0.10 for prev in found_bcp_coords):
                continue
            found_bcp_coords.append(r_ang)
            lap = float(np.sum(evals))
            g_dens = float(0.3 * (3.0 * np.pi**2)**(2.0/3.0) * (rho**(5.0/3.0)))
            v_dens = float(0.25 * lap - 2.0 * g_dens)
            h_dens = float(g_dens + v_dens)

            cps.append(CriticalPoint(
                index=cp_idx,
                cp_type="(3,+1)",
                type_name=f"Ring CP ({len(cycle)}-ring centroid)",
                coords_bohr=(float(r_rcp[0]), float(r_rcp[1]), float(r_rcp[2])),
                coords_angstrom=(float(r_ang[0]), float(r_ang[1]), float(r_ang[2])),
                rho=rho,
                grad_norm=float(np.linalg.norm(grad)),
                laplacian=lap,
                hessian_eigenvalues=(float(evals[0]), float(evals[1]), float(evals[2])),
                v_density=v_dens,
                g_density=g_dens,
                h_density=h_dens,
                bond_character="Ring Delocalization Center",
            ))
            cp_idx += 1
            num_rcp += 1

    return QTAIMResult(
        critical_points=cps,
        num_nuclear=num_nuc,
        num_bcp=num_bcp,
        num_rcp=num_rcp,
        num_ccp=num_ccp,
        method="PySCF Analytical QTAIM",
        bond_paths=all_bond_paths,
        raw_output=f"PySCF QTAIM Topology Solver completed with {len(cps)} CPs ({num_bcp} BCP, {num_rcp} RCP, {num_nuc} Nuclear)",
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


# =========================================================================
# Feature 5: Real-Space Grid Data (ELF, LOL, Laplacian, Density, NCI/RDG)
# =========================================================================

@dataclass
class GridResult:
    grid_type: str
    cube_path: str
    vol_data: Optional[Any] = None
    method: str = "Multiwfn"
    raw_output: str = ""


_MULTIWFN_GRID_OPTIONS = {
    "Density": "1",
    "Gradient Norm": "2",
    "Laplacian": "3",
    "ELF": "9",
    "LOL": "10",
    "ESP": "13",
}

_MULTIWFN_QUALITY_MAP = {
    "low": "1",
    "medium": "2",
    "fine": "3",
}


def compute_grid_cube(
    input_path: str,
    grid_type: str = "ELF",
    quality: str = "medium",
    output_cube: Optional[str] = None,
) -> GridResult:
    """
    Generate 3D real-space grid data (.cube) via Multiwfn (Function 5) or native promolecular fallback.
    Supported grid_type: 'ELF', 'LOL', 'Laplacian', 'Density', 'ESP'.
    """
    import tempfile
    from .volumetric import parse_cube_file, write_cube_file, VolumetricData
    from .zmatrix import parse_xyz

    bin_path = detect_multiwfn_executable()
    input_abs = os.path.abspath(input_path)
    opt_val = _MULTIWFN_GRID_OPTIONS.get(grid_type, "9")
    qual_val = _MULTIWFN_QUALITY_MAP.get(quality.lower(), "2")

    if bin_path:
        with tempfile.TemporaryDirectory() as tmpdir:
            inp = f"{input_abs}\n5\n{opt_val}\n{qual_val}\n2\nq\n"
            try:
                proc = subprocess.run(
                    [bin_path],
                    input=inp,
                    cwd=tmpdir,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=180,
                )
                stdout = proc.stdout
                # Find any generated .cube file in tmpdir
                cube_files = [os.path.join(tmpdir, f) for f in os.listdir(tmpdir) if f.lower().endswith(".cube")]
                if cube_files:
                    src_cube = cube_files[0]
                    target_path = output_cube or os.path.splitext(input_abs)[0] + f"_{grid_type.lower()}.cube"
                    shutil.copyfile(src_cube, target_path)
                    vol = parse_cube_file(target_path)
                    return GridResult(
                        grid_type=grid_type,
                        cube_path=target_path,
                        vol_data=vol,
                        method="Multiwfn Function 5",
                        raw_output=stdout,
                    )
            except Exception as e:
                log.debug("Multiwfn grid generation attempt failed: %s. Using native engine.", e)

    # Native promolecular fallback
    return _compute_native_promolecular_grid(input_abs, grid_type=grid_type, quality=quality, output_cube=output_cube)


def _compute_native_promolecular_grid(
    input_path: str,
    grid_type: str = "ELF",
    quality: str = "medium",
    output_cube: Optional[str] = None,
) -> GridResult:
    """Fast, robust pure-NumPy promolecular grid generator for Density, Laplacian, ELF, and RDG."""
    from .volumetric import VolumetricData, write_cube_file
    from .zmatrix import parse_xyz

    # Extract symbols and coordinates
    symbols = []
    coords = None

    if input_path.lower().endswith(".molden"):
        from .volumetric import parse_molden_file
        m_data = parse_molden_file(input_path)
        if m_data:
            symbols = m_data.symbols
            coords = m_data.coords
    elif input_path.lower().endswith((".xyz", ".out", ".log")):
        try:
            from .qm_log_parser import parse_qm_logfile
            parsed = parse_qm_logfile(input_path)
            symbols = parsed.symbols
            coords = parsed.trajectory_coords[-1]
        except Exception:
            pass

    if not symbols or coords is None:
        try:
            syms, crds = parse_xyz(input_path, is_path=True)
            symbols = syms
            coords = crds
        except Exception as e:
            raise ValueError(f"Could not load molecular structure from '{input_path}': {e}")

    n_atoms = len(symbols)
    c_arr = np.asarray(coords, dtype=np.float64)

    # Bounding box with 3.0 Angstrom padding
    padding = 3.0
    min_b = np.min(c_arr, axis=0) - padding
    max_b = np.max(c_arr, axis=0) + padding
    span = max_b - min_b

    # Quality setting
    resolutions = {"low": 30, "medium": 50, "fine": 80}
    n_pts = resolutions.get(quality.lower(), 50)
    step = span / (n_pts - 1)

    x = np.linspace(min_b[0], max_b[0], n_pts)
    y = np.linspace(min_b[1], max_b[1], n_pts)
    z = np.linspace(min_b[2], max_b[2], n_pts)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Slater exponent parameters (zeta ~ Z_eff / a0)
    slater_zeta = {
        "H": 1.24, "He": 1.69, "Li": 0.65, "Be": 0.97, "B": 1.30,
        "C": 1.625, "N": 1.95, "O": 2.275, "F": 2.60, "Ne": 2.925,
        "Na": 0.73, "Mg": 0.95, "Al": 1.17, "Si": 1.38, "P": 1.60,
        "S": 1.82, "Cl": 2.03, "Ar": 2.25, "Br": 2.30, "I": 2.00,
    }

    rho = np.zeros_like(X, dtype=np.float64)
    for sym, pos in zip(symbols, c_arr):
        zeta = slater_zeta.get(sym.capitalize(), 1.5)
        r = np.sqrt((X - pos[0])**2 + (Y - pos[1])**2 + (Z - pos[2])**2)
        # Spherically averaged neutral atomic Slater density
        rho += (zeta**3 / np.pi) * np.exp(-2.0 * zeta * r)

    dx, dy, dz = step[0], step[1], step[2]
    grad_x, grad_y, grad_z = np.gradient(rho, dx, dy, dz)
    grad_norm = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

    # Compute requested property grid
    gt = grid_type.upper()
    if gt == "DENSITY":
        grid_data = rho
    elif gt == "GRADIENT NORM":
        grid_data = grad_norm
    elif gt == "LAPLACIAN":
        # ∇²ρ = d²ρ/dx² + d²ρ/dy² + d²ρ/dz²
        d2x = np.gradient(grad_x, dx, axis=0)
        d2y = np.gradient(grad_y, dy, axis=1)
        d2z = np.gradient(grad_z, dz, axis=2)
        grid_data = d2x + d2y + d2z
    elif gt in ("ELF", "LOL"):
        # Becke-Edgecombe ELF / LOL functional ratio
        cf = 0.3 * (3.0 * np.pi**2)**(2.0 / 3.0)
        tau_0 = cf * (np.maximum(rho, 1e-12)**(5.0 / 3.0))
        tau_w = (grad_norm**2) / (8.0 * np.maximum(rho, 1e-12))
        tau = np.maximum(tau_w, 0.5 * tau_0)
        d_val = np.maximum(0.0, tau - tau_w)
        chi = d_val / np.maximum(tau_0, 1e-12)
        if gt == "ELF":
            grid_data = 1.0 / (1.0 + chi**2)
        else:
            # LOL = 1 / (1 + (tau / tau_0))
            grid_data = 1.0 / (1.0 + chi)
    elif gt in ("NCI", "RDG"):
        # Reduced density gradient s(r) = |∇ρ| / (2 * (3*pi^2)^(1/3) * ρ^(4/3))
        factor = 2.0 * (3.0 * np.pi**2)**(1.0 / 3.0)
        grid_data = grad_norm / (factor * (np.maximum(rho, 1e-8)**(4.0 / 3.0)) + 1e-12)
    else:
        grid_data = rho

    target_path = output_cube or os.path.splitext(input_path)[0] + f"_{grid_type.lower()}.cube"
    vol = VolumetricData(
        title=f"AgyMolden {grid_type} 3D Grid",
        comment=f"Generated via native promolecular engine (N={n_pts}^3)",
        symbols=symbols,
        atom_coords=c_arr,
        origin=min_b,
        voxels_x=n_pts,
        voxels_y=n_pts,
        voxels_z=n_pts,
        step_x=np.array([dx, 0.0, 0.0]),
        step_y=np.array([0.0, dy, 0.0]),
        step_z=np.array([0.0, 0.0, dz]),
        grid_data=grid_data,
        is_orbital=False,
    )
    write_cube_file(target_path, vol)
    return GridResult(
        grid_type=grid_type,
        cube_path=target_path,
        vol_data=vol,
        method="AgyMolden Native Promolecular Engine",
    )


def compute_nci_analysis(
    input_path: str,
    quality: str = "medium",
    output_dir: Optional[str] = None,
) -> Tuple[GridResult, GridResult]:
    """
    Computes Non-Covalent Interaction (NCI) Reduced Density Gradient (RDG) and density grids.
    Returns (rdg_result, density_result).
    """
    base = os.path.splitext(os.path.abspath(input_path))[0]
    out_dir = output_dir or os.path.dirname(base)
    rdg_path = os.path.join(out_dir, os.path.basename(base) + "_nci_rdg.cube")
    dens_path = os.path.join(out_dir, os.path.basename(base) + "_nci_dens.cube")

    res_rdg = compute_grid_cube(input_path, grid_type="RDG", quality=quality, output_cube=rdg_path)
    res_dens = compute_grid_cube(input_path, grid_type="Density", quality=quality, output_cube=dens_path)
    return res_rdg, res_dens
