"""
Quantum Theory of Atoms in Molecules (QTAIM) Topology & Critical Point Finder for AgyMolden.

Computes:
1. Bond Critical Points (BCP) (3, -1) where grad rho = 0, with 2 negative and 1 positive Hessian eigenvalue.
2. Electron density at BCP rho_bcp (e/Bohr^3), Laplacian of density nabla^2 rho_bcp.
3. Ellipticity epsilon = (lambda_1 / lambda_2) - 1.
4. Kinetic energy density G_bcp, potential energy density V_bcp, and total energy density H_bcp.
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


class BondCriticalPoint(NamedTuple):
    atom_a_idx: int
    atom_b_idx: int
    atom_a_symbol: str
    atom_b_symbol: str
    coord_angstrom: np.ndarray
    rho_bcp: float               # e / Bohr^3
    laplacian_bcp: float         # nabla^2 rho (e / Bohr^5)
    eigenvalues: Tuple[float, float, float]  # (lambda1, lambda2, lambda3) where lambda1 <= lambda2 < 0 < lambda3
    ellipticity: float           # epsilon = (lambda1 / lambda2) - 1
    h_energy_density: float      # Total electronic energy density H = G + V (Hartree / Bohr^3)


class QTAIMTopologyResult(NamedTuple):
    bond_critical_points: List[BondCriticalPoint]
    atom_symbols: List[str]
    atom_coords_angstrom: np.ndarray


def find_bond_critical_points(molden_path: str, max_bond_distance_angstrom: float = 2.8) -> QTAIMTopologyResult:
    """Finds all (3, -1) Bond Critical Points between bonded atom pairs using Newton-Raphson."""
    from .volumetric import _load_pyscf_molden
    from pyscf import dft

    res = _load_pyscf_molden(molden_path)
    if not res:
        raise ValueError(f"Could not load wavefunction from {molden_path}")

    mol = res[0]
    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
        mo_occ = mo_occ[0]

    occ_idx = np.where(mo_occ > 0.5)[0]
    if len(occ_idx) == 0:
        occ_idx = np.arange(max(1, mol.nelectron // 2))

    c_occ = mo_coeff[:, occ_idx]
    dm = c_occ @ c_occ.T * 2.0  # RHF density matrix

    natm = mol.natm
    atom_symbols = [mol.atom_symbol(i) for i in range(natm)]
    coords_bohr = mol.atom_coords()  # In Bohr
    coords_ang = coords_bohr * BOHR_TO_ANGSTROM

    bcps: List[BondCriticalPoint] = []

    # Helper function to evaluate rho, grad, and hessian at a given point (Bohr)
    def eval_rho_derivatives(pt_bohr: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        ao_val = dft.numint.eval_ao(mol, pt_bohr.reshape(1, 3), deriv=2)
        # ao_val has shape (10, 1, nao)
        # [0]=val, [1]=dx, [2]=dy, [3]=dz
        # [4]=dxx, [5]=dxy, [6]=dxz, [7]=dyy, [8]=dyz, [9]=dzz
        ao = ao_val[0, 0]
        dx = ao_val[1, 0]
        dy = ao_val[2, 0]
        dz = ao_val[3, 0]
        dxx = ao_val[4, 0]
        dxy = ao_val[5, 0]
        dxz = ao_val[6, 0]
        dyy = ao_val[7, 0]
        dyz = ao_val[8, 0]
        dzz = ao_val[9, 0]

        rho = float(ao @ dm @ ao.T)
        gx = 2.0 * float(dx @ dm @ ao.T)
        gy = 2.0 * float(dy @ dm @ ao.T)
        gz = 2.0 * float(dz @ dm @ ao.T)
        grad = np.array([gx, gy, gz])

        hxx = 2.0 * float(dxx @ dm @ ao.T + dx @ dm @ dx.T)
        hxy = 2.0 * float(dxy @ dm @ ao.T + dx @ dm @ dy.T)
        hxz = 2.0 * float(dxz @ dm @ ao.T + dx @ dm @ dz.T)
        hyy = 2.0 * float(dyy @ dm @ ao.T + dy @ dm @ dy.T)
        hyz = 2.0 * float(dyz @ dm @ ao.T + dy @ dm @ dz.T)
        hzz = 2.0 * float(dzz @ dm @ ao.T + dz @ dm @ dz.T)

        hess = np.array([
            [hxx, hxy, hxz],
            [hxy, hyy, hyz],
            [hxz, hyz, hzz],
        ])
        return rho, grad, hess

    # Loop over unique atom pairs within distance cutoff
    max_d_bohr = max_bond_distance_angstrom * ANGSTROM_TO_BOHR
    for i in range(natm):
        for j in range(i + 1, natm):
            dist_bohr = float(np.linalg.norm(coords_bohr[i] - coords_bohr[j]))
            if dist_bohr > max_d_bohr:
                continue

            # 1. Linear scan along the bond axis to find minimum |grad rho| seed
            steps = np.linspace(coords_bohr[i], coords_bohr[j], 15)
            # Exclude points too close to nuclei (avoid core divergence)
            interior_pts = steps[2:-2]
            best_pt = interior_pts[len(interior_pts) // 2]
            best_gnorm = 1e9

            for p in interior_pts:
                _, g, _ = eval_rho_derivatives(p)
                gnorm = np.linalg.norm(g)
                if gnorm < best_gnorm:
                    best_gnorm = gnorm
                    best_pt = p

            # 2. Newton-Raphson refinement: r_{k+1} = r_k - [Hessian]^-1 * grad
            curr_pt = np.array(best_pt, copy=True)
            converged = False
            for it in range(12):
                rho, grad, hess = eval_rho_derivatives(curr_pt)
                gnorm = float(np.linalg.norm(grad))
                if gnorm < 1e-4:
                    converged = True
                    break
                try:
                    step = np.linalg.solve(hess, grad)
                    # Limit step magnitude to prevent divergent jumps
                    s_norm = np.linalg.norm(step)
                    if s_norm > 0.2:
                        step = step * (0.2 / s_norm)
                    curr_pt -= step
                except np.linalg.LinAlgError:
                    break

            if not converged:
                # Accept if gradient norm is suitably small
                if np.linalg.norm(grad) > 0.05:
                    continue

            rho, grad, hess = eval_rho_derivatives(curr_pt)
            w, _ = np.linalg.eigh(hess)
            # Sort eigenvalues
            w = np.sort(w)
            l1, l2, l3 = float(w[0]), float(w[1]), float(w[2])

            # (3, -1) BCP condition: l1 < 0, l2 < 0, l3 > 0
            if l1 < 0 and l2 < 0 and l3 > 0:
                laplacian = l1 + l2 + l3
                ellipticity = (l1 / l2) - 1.0 if abs(l2) > 1e-6 else 0.0
                # Abramov local kinetic energy density approximation G = (3/10)*(3*pi^2)^(2/3)*rho^(5/3) + (1/72)*|grad|^2/rho + (1/6)*laplacian
                # Potential energy density V from local virial theorem: 2*G + V = (1/4)*nabla^2 rho
                # Here we compute Abramov H = G + V
                g_dens = max(0.0, (3.0 / 10.0) * ((3.0 * math.pi ** 2) ** (2.0 / 3.0)) * (rho ** (5.0 / 3.0)) + (1.0 / 6.0) * laplacian)
                v_dens = (0.25 * laplacian) - 2.0 * g_dens
                h_dens = g_dens + v_dens

                bcps.append(
                    BondCriticalPoint(
                        atom_a_idx=i,
                        atom_b_idx=j,
                        atom_a_symbol=atom_symbols[i],
                        atom_b_symbol=atom_symbols[j],
                        coord_angstrom=curr_pt * BOHR_TO_ANGSTROM,
                        rho_bcp=float(rho),
                        laplacian_bcp=float(laplacian),
                        eigenvalues=(l1, l2, l3),
                        ellipticity=float(ellipticity),
                        h_energy_density=float(h_dens),
                    )
                )

    return QTAIMTopologyResult(
        bond_critical_points=bcps,
        atom_symbols=atom_symbols,
        atom_coords_angstrom=coords_ang,
    )
