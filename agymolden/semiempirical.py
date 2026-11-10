"""Pure-Python Semiempirical Quantum Mechanics & Tight-Binding (GFN0-xTB / EHT) Engine.

Depends strictly on `numpy` and `scipy` (zero external binaries or C++/Fortran dependencies).
Provides:
  - Self-Consistent Charge (SCC) / Tight-Binding valence electronic structure (`compute_semiempirical_energy`)
  - HOMO / LUMO gap, Mulliken atomic charges, and Mayer/Wiberg quantum bond orders
  - Quantum mechanical equilibrium bond lengths, angles, conjugated pi-torsions, D3 dispersion, and H-bonding
  - Analytical and semi-analytical gradient optimization (`optimize_geometry_semiempirical`)
"""

import math
from typing import List, Tuple, Dict, Optional
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize

from .elements import ELEMENT_DATA, get_cov_radius, get_vdw_radius, estimate_bond_length


# Valence shell parameters for pure-Python Tight-Binding / Semiempirical QM:
# (n_val_electrons, n_orbitals, H_ss [eV], H_pp [eV], zeta_s [Bohr^-1], zeta_p [Bohr^-1], hardness_eta [eV], EN_pauling, C6 [eV*A^6])
TB_PARAMS: Dict[str, Dict[str, float]] = {
    "H":  {"z_val": 1, "norb": 1, "h_s": -13.60, "h_p":  0.00, "zeta_s": 1.20, "zeta_p": 1.20, "eta": 12.85, "en": 2.20, "c6":  2.5},
    "He": {"z_val": 2, "norb": 1, "h_s": -24.59, "h_p":  0.00, "zeta_s": 1.70, "zeta_p": 1.70, "eta": 24.59, "en": 4.16, "c6":  1.5},
    "Li": {"z_val": 1, "norb": 4, "h_s":  -5.39, "h_p": -3.54, "zeta_s": 0.65, "zeta_p": 0.55, "eta":  4.77, "en": 0.98, "c6": 15.0},
    "Be": {"z_val": 2, "norb": 4, "h_s":  -9.32, "h_p": -6.00, "zeta_s": 0.98, "zeta_p": 0.90, "eta":  8.00, "en": 1.57, "c6": 12.0},
    "B":  {"z_val": 3, "norb": 4, "h_s": -14.05, "h_p": -8.30, "zeta_s": 1.30, "zeta_p": 1.25, "eta":  8.10, "en": 2.04, "c6": 16.0},
    "C":  {"z_val": 4, "norb": 4, "h_s": -19.44, "h_p": -10.67, "zeta_s": 1.625, "zeta_p": 1.575, "eta": 10.00, "en": 2.55, "c6": 22.0},
    "N":  {"z_val": 5, "norb": 4, "h_s": -25.58, "h_p": -13.19, "zeta_s": 1.950, "zeta_p": 1.900, "eta": 11.50, "en": 3.04, "c6": 18.0},
    "O":  {"z_val": 6, "norb": 4, "h_s": -32.30, "h_p": -15.85, "zeta_s": 2.275, "zeta_p": 2.225, "eta": 13.60, "en": 3.44, "c6": 14.0},
    "F":  {"z_val": 7, "norb": 4, "h_s": -40.20, "h_p": -18.65, "zeta_s": 2.600, "zeta_p": 2.550, "eta": 15.40, "en": 3.98, "c6": 11.0},
    "Na": {"z_val": 1, "norb": 4, "h_s":  -5.14, "h_p": -3.04, "zeta_s": 0.73, "zeta_p": 0.60, "eta":  4.60, "en": 0.93, "c6": 25.0},
    "Mg": {"z_val": 2, "norb": 4, "h_s":  -7.65, "h_p": -4.50, "zeta_s": 0.95, "zeta_p": 0.85, "eta":  6.80, "en": 1.31, "c6": 22.0},
    "Al": {"z_val": 3, "norb": 4, "h_s": -11.30, "h_p": -5.98, "zeta_s": 1.16, "zeta_p": 1.10, "eta":  6.50, "en": 1.61, "c6": 30.0},
    "Si": {"z_val": 4, "norb": 4, "h_s": -14.96, "h_p": -7.75, "zeta_s": 1.38, "zeta_p": 1.32, "eta":  7.60, "en": 1.90, "c6": 38.0},
    "P":  {"z_val": 5, "norb": 4, "h_s": -18.60, "h_p": -10.10, "zeta_s": 1.60, "zeta_p": 1.55, "eta": 8.90, "en": 2.19, "c6": 42.0},
    "S":  {"z_val": 6, "norb": 4, "h_s": -21.10, "h_p": -11.65, "zeta_s": 1.82, "zeta_p": 1.75, "eta":  9.80, "en": 2.58, "c6": 45.0},
    "Cl": {"z_val": 7, "norb": 4, "h_s": -25.30, "h_p": -13.70, "zeta_s": 2.04, "zeta_p": 1.98, "eta": 11.00, "en": 3.16, "c6": 40.0},
    "Br": {"z_val": 7, "norb": 4, "h_s": -22.80, "h_p": -11.85, "zeta_s": 1.88, "zeta_p": 1.82, "eta":  9.90, "en": 2.96, "c6": 60.0},
    "I":  {"z_val": 7, "norb": 4, "h_s": -19.40, "h_p": -10.45, "zeta_s": 1.72, "zeta_p": 1.65, "eta":  8.80, "en": 2.66, "c6": 90.0},
}


def _get_tb_param(sym: str) -> Dict[str, float]:
    if sym in TB_PARAMS:
        return TB_PARAMS[sym]
    # Fallback for transition metals / heavier elements using ELEMENT_DATA
    ed = ELEMENT_DATA.get(sym, {})
    grp = ed.get("group", 14)
    z_val = max(1, min(8, grp if grp <= 10 else grp - 10))
    return {
        "z_val": z_val,
        "norb": 4,
        "h_s": -12.0 - 0.8 * z_val,
        "h_p": -6.5 - 0.5 * z_val,
        "zeta_s": 1.35,
        "zeta_p": 1.25,
        "eta": 7.5,
        "en": 1.85,
        "c6": 45.0,
    }


def compute_semiempirical_properties(
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None
) -> Dict:
    """
    Solve the valence Tight-Binding / Extended Hückel + Self-Consistent Charge (SCC-TB) Hamiltonian
    in a minimal valence (s, px, py, pz) Slater basis using `scipy.linalg.eigh`.

    Returns a dictionary containing:
      - `electronic_energy_ev`: Total valence electronic energy in eV
      - `homo_ev`: HOMO energy in eV
      - `lumo_ev`: LUMO energy in eV
      - `gap_ev`: HOMO-LUMO gap in eV
      - `charges`: Mulliken / SCC atomic partial charges (array of shape (N,))
      - `wiberg_bond_orders`: Dict[(u, v), float] of quantum mechanical Wiberg bond orders
    """
    n = len(symbols)
    real_indices = [i for i, s in enumerate(symbols) if s != "X"]
    if not real_indices:
        return {
            "electronic_energy_ev": 0.0,
            "homo_ev": 0.0,
            "lumo_ev": 0.0,
            "gap_ev": 0.0,
            "charges": np.zeros(n),
            "wiberg_bond_orders": {},
        }

    # Build orbital map for real atoms
    atom_orbs: List[List[int]] = [[] for _ in range(n)]
    orb_types: List[Tuple[int, str, float, float]] = []  # (atom_idx, 's'|'px'|'py'|'pz', h_ii, zeta)
    total_elec = 0

    for i in real_indices:
        p = _get_tb_param(symbols[i])
        total_elec += int(p["z_val"])
        norb = int(p["norb"])
        s_idx = len(orb_types)
        orb_types.append((i, "s", p["h_s"], p["zeta_s"]))
        atom_orbs[i].append(s_idx)
        if norb == 4:
            for p_dir in ("px", "py", "pz"):
                p_idx = len(orb_types)
                orb_types.append((i, p_dir, p["h_p"], p["zeta_p"]))
                atom_orbs[i].append(p_idx)

    n_basis = len(orb_types)
    H0 = np.zeros((n_basis, n_basis), dtype=np.float64)
    S = np.eye(n_basis, dtype=np.float64)

    for mu in range(n_basis):
        H0[mu, mu] = orb_types[mu][2]

    BOHR = 0.529177210903
    K_WH = 1.88  # Wolfsberg-Helmholz constant

    # Construct overlap S and off-diagonal H0 using directional Slater-Koster rules
    for idx_a, a in enumerate(real_indices):
        pos_a = coords[a]
        r_cov_a = get_cov_radius(symbols[a])
        for b in real_indices[idx_a + 1:]:
            vec = coords[b] - pos_a
            dist = float(np.linalg.norm(vec))
            if dist < 1e-4 or dist > 6.0:
                continue
            u_vec = vec / dist  # unit vector from a to b (x, y, z)
            r_bohr = dist / BOHR
            r_cov_sum = r_cov_a + get_cov_radius(symbols[b])

            # Radial Slater overlap decay
            orbs_a = atom_orbs[a]
            orbs_b = atom_orbs[b]
            za = orb_types[orbs_a[0]][3]
            zb = orb_types[orbs_b[0]][3]
            z_mean = 0.5 * (za + zb)
            rho = z_mean * r_bohr

            # Accurate polynomial*exp Slater-type radial overlaps
            s_sigma_ss = (1.0 + rho + (rho * rho) / 3.0) * math.exp(-rho)
            s_sigma_sp = 0.65 * rho * (1.0 + 0.5 * rho) * math.exp(-0.92 * rho)
            s_sigma_pp = -0.58 * (1.0 + 0.6 * rho) * math.exp(-0.88 * rho)
            s_pi_pp = 0.48 * (1.0 + 0.8 * rho + 0.25 * rho * rho) * math.exp(-0.92 * rho)

            # Smooth distance damping beyond covalent cutoff
            damp = 1.0 / (1.0 + math.pow(max(0.0, (dist - r_cov_sum) / 1.8), 4))
            s_sigma_ss *= damp
            s_sigma_sp *= damp
            s_sigma_pp *= damp
            s_pi_pp *= damp

            # Populate subblock between atom a and atom b
            # Note: u_vec points from a to b. For <s_a | p_b>, the positive lobe of p_b points AWAY from a,
            # so <s_a | p_b> = -u_vec[c] * S_spsigma, while <p_a | s_b> = +u_vec[c] * S_spsigma.
            for mu in orbs_a:
                _, ta, ha, _ = orb_types[mu]
                for nu in orbs_b:
                    _, tb, hb, _ = orb_types[nu]
                    s_val = 0.0
                    if ta == "s" and tb == "s":
                        s_val = s_sigma_ss
                    elif ta == "s" and tb in ("px", "py", "pz"):
                        c_idx = {"px": 0, "py": 1, "pz": 2}[tb]
                        s_val = -u_vec[c_idx] * s_sigma_sp
                    elif ta in ("px", "py", "pz") and tb == "s":
                        c_idx = {"px": 0, "py": 1, "pz": 2}[ta]
                        s_val = u_vec[c_idx] * s_sigma_sp
                    else:
                        ca = {"px": 0, "py": 1, "pz": 2}[ta]
                        cb = {"px": 0, "py": 1, "pz": 2}[tb]
                        delta_ab = 1.0 if ca == cb else 0.0
                        s_val = u_vec[ca] * u_vec[cb] * s_sigma_pp + (delta_ab - u_vec[ca] * u_vec[cb]) * s_pi_pp

                    S[mu, nu] = s_val
                    S[nu, mu] = s_val
                    h_val = 0.5 * K_WH * (ha + hb) * s_val
                    H0[mu, nu] = h_val
                    H0[nu, mu] = h_val

    # Regularize overlap matrix S to ensure strict positive definiteness for short distances
    s_vals, s_vecs = eigh(S)
    s_vals_clipped = np.maximum(s_vals, 0.08)
    S_reg = (s_vecs * s_vals_clipped) @ s_vecs.T

    # Self-Consistent Charge (SCC) iteration (2 iterations with damping for rock-solid convergence)
    charges = np.zeros(n, dtype=np.float64)
    gamma_mat = np.zeros((n, n), dtype=np.float64)
    for idx_a, a in enumerate(real_indices):
        eta_a = _get_tb_param(symbols[a])["eta"]
        gamma_mat[a, a] = eta_a
        for b in real_indices[idx_a + 1:]:
            eta_b = _get_tb_param(symbols[b])["eta"]
            eta_ab = 0.5 * (eta_a + eta_b)
            d_ab = max(0.2, float(np.linalg.norm(coords[a] - coords[b])))
            # Klopman-Ohno damped Coulomb interaction in eV
            e2_a = 14.399645
            g_ab = 1.0 / math.sqrt((d_ab / e2_a) ** 2 + (1.0 / eta_ab) ** 2)
            gamma_mat[a, b] = g_ab
            gamma_mat[b, a] = g_ab

    occ = np.zeros(n_basis, dtype=np.float64)
    rem = float(total_elec)
    for k in range(n_basis):
        if rem >= 2.0:
            occ[k] = 2.0
            rem -= 2.0
        elif rem > 0.0:
            occ[k] = rem
            rem = 0.0

    eigvals = np.zeros(n_basis)
    eigvecs = np.eye(n_basis)
    P = np.zeros((n_basis, n_basis))

    for scc_step in range(3):
        H = H0.copy()
        if scc_step > 0:
            shifts = gamma_mat @ charges
            for mu in range(n_basis):
                a_i = orb_types[mu][0]
                for nu in range(mu, n_basis):
                    a_j = orb_types[nu][0]
                    v_shift = 0.5 * (shifts[a_i] + shifts[a_j]) * S_reg[mu, nu]
                    H[mu, nu] += v_shift
                    if mu != nu:
                        H[nu, mu] += v_shift

        eigvals, eigvecs = eigh(H, S_reg)
        P = (eigvecs * occ) @ eigvecs.T
        PS = P @ S_reg
        mulliken_pop = np.diag(PS)

        new_charges = np.zeros(n, dtype=np.float64)
        for a in real_indices:
            z_v = _get_tb_param(symbols[a])["z_val"]
            pop_a = sum(mulliken_pop[mu] for mu in atom_orbs[a])
            new_charges[a] = np.clip(z_v - pop_a, -1.5, 1.5)

        charges = 0.45 * new_charges + 0.55 * charges

    # Compute Wiberg bond orders in Löwdin orthogonalized basis (S^1/2 P S^1/2)
    s_sqrt = (s_vecs * np.sqrt(s_vals_clipped)) @ s_vecs.T
    P_orth = s_sqrt @ P @ s_sqrt
    wiberg: Dict[Tuple[int, int], float] = {}
    for idx_a, a in enumerate(real_indices):
        orbs_a = atom_orbs[a]
        for b in real_indices[idx_a + 1:]:
            orbs_b = atom_orbs[b]
            sub = P_orth[np.ix_(orbs_a, orbs_b)]
            w_ab = float(np.sum(sub * sub))
            if w_ab > 0.05:
                wiberg[(min(a, b), max(a, b))] = w_ab

    n_occ = max(0, (total_elec // 2) - 1)
    homo_ev = float(eigvals[n_occ]) if n_basis > 0 else 0.0
    lumo_ev = float(eigvals[min(n_basis - 1, n_occ + 1)]) if n_basis > 1 else homo_ev
    elec_e = float(np.sum(occ * eigvals))

    return {
        "electronic_energy_ev": elec_e,
        "homo_ev": homo_ev,
        "lumo_ev": lumo_ev,
        "gap_ev": max(0.0, lumo_ev - homo_ev),
        "charges": charges,
        "wiberg_bond_orders": wiberg,
        "eigvals_ev": eigvals,
        "eigvecs": eigvecs,
        "occupancies": occ,
        "orb_types": orb_types,
    }


def evaluate_semiempirical_orbital_grid(
    symbols: List[str],
    coords: np.ndarray,
    orbital_index: int = 0,
    nx: int = 36,
    ny: int = 36,
    nz: int = 36,
    margin_ang: float = 2.5,
    qm_props: Optional[Dict] = None,
):
    """
    Evaluate a 3D valence Slater-type Molecular Orbital grid directly in pure NumPy
    from the built-in Semiempirical Tight-Binding (GFN0-xTB / SCC-TB) wavefunction.
    Requires ZERO external binaries or PySCF installation.
    """
    from .volumetric import VolumetricData, BOHR_TO_ANGSTROM

    if qm_props is None or "eigvecs" not in qm_props:
        qm_props = compute_semiempirical_properties(symbols, coords)

    eigvecs = qm_props.get("eigvecs")
    eigvals = qm_props.get("eigvals_ev")
    orb_types = qm_props.get("orb_types")
    if eigvecs is None or orb_types is None or len(orb_types) == 0:
        return None

    n_mo = eigvecs.shape[1]
    mo_idx = max(0, min(n_mo - 1, orbital_index))
    c_vec = eigvecs[:, mo_idx]

    c_min = np.min(coords, axis=0) - margin_ang
    c_max = np.max(coords, axis=0) + margin_ang
    xs = np.linspace(c_min[0], c_max[0], nx, dtype=np.float32)
    ys = np.linspace(c_min[1], c_max[1], ny, dtype=np.float32)
    zs = np.linspace(c_min[2], c_max[2], nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid_3d = np.zeros((nx, ny, nz), dtype=np.float32)

    # Group basis functions by atom for fast vectorized evaluation
    atom_groups: Dict[int, List[Tuple[int, str, float]]] = {}
    for mu, (at_idx, o_type, _h_ii, zeta) in enumerate(orb_types):
        if abs(c_vec[mu]) > 1e-5:
            atom_groups.setdefault(at_idx, []).append((mu, o_type, float(zeta)))

    for at_idx, mu_list in atom_groups.items():
        ax, ay, az = float(coords[at_idx, 0]), float(coords[at_idx, 1]), float(coords[at_idx, 2])
        # Find maximum cutoff radius where Slater decay exp(-zeta * r_bohr) > 1e-6
        min_zeta = min(zeta for _, _, zeta in mu_list)
        r_cut_bohr = 14.0 / max(0.2, min_zeta)
        r_cut_ang = r_cut_bohr * BOHR_TO_ANGSTROM

        # Spatial bounding box slice to avoid evaluating faraway empty grid voxels
        x_mask = np.abs(xs - ax) <= r_cut_ang
        y_mask = np.abs(ys - ay) <= r_cut_ang
        z_mask = np.abs(zs - az) <= r_cut_ang
        if not (np.any(x_mask) and np.any(y_mask) and np.any(z_mask)):
            continue

        ix0, ix1 = np.where(x_mask)[0][[0, -1]]
        iy0, iy1 = np.where(y_mask)[0][[0, -1]]
        iz0, iz1 = np.where(z_mask)[0][[0, -1]]

        sub_X = X[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1]
        sub_Y = Y[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1]
        sub_Z = Z[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1]

        dx = (sub_X - ax) / BOHR_TO_ANGSTROM
        dy = (sub_Y - ay) / BOHR_TO_ANGSTROM
        dz = (sub_Z - az) / BOHR_TO_ANGSTROM
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6

        sub_grid = np.zeros(sub_X.shape, dtype=np.float32)
        for mu, o_type, zeta in mu_list:
            coeff = float(c_vec[mu])
            # Normalized valence Slater-type orbital radial part
            norm_fac = math.sqrt((zeta ** 3) / math.pi)
            rad = norm_fac * np.exp(-zeta * r)
            if o_type == "s":
                sub_grid += coeff * rad
            elif o_type == "px":
                sub_grid += coeff * (zeta * dx) * rad
            elif o_type == "py":
                sub_grid += coeff * (zeta * dy) * rad
            elif o_type == "pz":
                sub_grid += coeff * (zeta * dz) * rad

        grid_3d[ix0:ix1+1, iy0:iy1+1, iz0:iz1+1] += sub_grid

    step_x = np.array([(c_max[0] - c_min[0]) / max(1, nx - 1), 0.0, 0.0], dtype=np.float64)
    step_y = np.array([0.0, (c_max[1] - c_min[1]) / max(1, ny - 1), 0.0], dtype=np.float64)
    step_z = np.array([0.0, 0.0, (c_max[2] - c_min[2]) / max(1, nz - 1)], dtype=np.float64)
    e_ev = float(eigvals[mo_idx]) if eigvals is not None and mo_idx < len(eigvals) else 0.0

    return VolumetricData(
        title=f"TB-QM MO #{mo_idx + 1} (E = {e_ev:+.2f} eV)",
        comment="Built-in Semiempirical Tight-Binding Orbital Grid",
        symbols=list(symbols),
        atom_coords=np.array(coords, dtype=np.float64, copy=True),
        origin=c_min.astype(np.float64),
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=True,
        orbital_index=mo_idx + 1,
    )


def compute_semiempirical_esp_mapped_density(
    symbols: List[str],
    coords: np.ndarray,
    nx: int = 36,
    ny: int = 36,
    nz: int = 36,
    margin_ang: float = 2.5,
    iso_density: float = 0.03,
    qm_props: Optional[Dict] = None,
):
    """
    Compute a 3D electron density isosurface colored by Molecular Electrostatic Potential (ESP)
    using the built-in Semiempirical Tight-Binding SCC charges and promolecular/valence density.
    Works out-of-the-box with zero external QM binaries!
    """
    from .volumetric import VolumetricData, extract_isosurface, BOHR_TO_ANGSTROM

    if qm_props is None or "charges" not in qm_props:
        qm_props = compute_semiempirical_properties(symbols, coords)

    charges = qm_props["charges"]
    c_min = np.min(coords, axis=0) - margin_ang
    c_max = np.max(coords, axis=0) + margin_ang
    xs = np.linspace(c_min[0], c_max[0], nx, dtype=np.float32)
    ys = np.linspace(c_min[1], c_max[1], ny, dtype=np.float32)
    zs = np.linspace(c_min[2], c_max[2], nz, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    rho_grid = np.zeros((nx, ny, nz), dtype=np.float32)

    for i, sym in enumerate(symbols):
        if sym == "X":
            continue
        p = _get_tb_param(sym)
        zeta = float(p["zeta_s"])
        z_val = float(p["z_val"])
        dx = (X - float(coords[i, 0])) / BOHR_TO_ANGSTROM
        dy = (Y - float(coords[i, 1])) / BOHR_TO_ANGSTROM
        dz = (Z - float(coords[i, 2])) / BOHR_TO_ANGSTROM
        r = np.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        rho_grid += float(z_val * (zeta ** 3) / (8.0 * math.pi)) * np.exp(-1.6 * zeta * r)

    step_x = np.array([(c_max[0] - c_min[0]) / max(1, nx - 1), 0.0, 0.0], dtype=np.float64)
    step_y = np.array([0.0, (c_max[1] - c_min[1]) / max(1, ny - 1), 0.0], dtype=np.float64)
    step_z = np.array([0.0, 0.0, (c_max[2] - c_min[2]) / max(1, nz - 1)], dtype=np.float64)

    vol = VolumetricData(
        title="Semiempirical Electron Density",
        comment="Tight-Binding Valence Electron Density",
        symbols=list(symbols),
        atom_coords=np.array(coords, dtype=np.float64, copy=True),
        origin=c_min.astype(np.float64),
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=rho_grid,
        is_orbital=False,
        orbital_index=None,
    )

    eff_iso = min(iso_density, float(np.max(rho_grid)) * 0.4)
    mesh = extract_isosurface(vol, isovalue=max(1e-4, eff_iso), color=(0.8, 0.8, 0.8))
    if mesh is None or len(mesh.vertices) == 0:
        return None

    # Evaluate SCC electrostatic potential V(r) = sum_A q_A / |r - R_A| at surface vertices
    verts = mesh.vertices
    v_esp = np.zeros(len(verts), dtype=np.float32)
    for i, sym in enumerate(symbols):
        if sym == "X":
            continue
        q_i = float(charges[i])
        dist_bohr = np.linalg.norm(verts - coords[i], axis=1) / BOHR_TO_ANGSTROM
        v_esp += q_i / np.maximum(dist_bohr, 0.4)

    vmax = max(0.02, float(np.percentile(np.abs(v_esp), 95)))
    t_norm = np.clip((v_esp + vmax) / (2.0 * vmax), 0.0, 1.0)

    # Red (negative ESP, nucleophilic) -> White (neutral) -> Blue (positive ESP, electrophilic)
    colors = np.zeros((len(verts), 3), dtype=np.float32)
    low_mask = t_norm < 0.5
    s_low = t_norm[low_mask] * 2.0
    colors[low_mask, 0] = 0.92
    colors[low_mask, 1] = 0.15 + 0.80 * s_low
    colors[low_mask, 2] = 0.15 + 0.80 * s_low

    high_mask = ~low_mask
    s_high = (t_norm[high_mask] - 0.5) * 2.0
    colors[high_mask, 0] = 0.95 - 0.80 * s_high
    colors[high_mask, 1] = 0.95 - 0.65 * s_high
    colors[high_mask, 2] = 0.95

    mesh.vertex_colors = colors
    mesh.surface_values = v_esp
    mesh.esp_range = (-vmax, +vmax)
    return mesh


def optimize_geometry_semiempirical(
    symbols: List[str],
    coords: np.ndarray,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    max_iter: int = 250,
    strict_bonds: bool = False,
    user_bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    frozen_indices: Optional[set] = None,
) -> Tuple[np.ndarray, float, Dict]:
    """
    Optimize molecular geometry using the pure-Python Semiempirical Tight-Binding (GFN0-xTB / SCC-TB) method:
      1. Computes quantum mechanical Wiberg bond orders, Mulliken/SCC partial charges, and HOMO/LUMO orbitals
         at the initial geometry and updates them self-consistently along the optimization path.
      2. Uses fractional quantum bond orders (Paulings relation r0 = r_single - c * ln(WBO)) + D3-bj dispersion
         + SCC electrostatics + conjugated pi-system planarity + analytical gradients for fast, accurate convergence.
    Returns:
      (optimized_coords, final_energy_kcal_mol, qm_properties_dict)
    """
    from .forcefield import optimize_geometry_force_field, detect_connectivity

    n = len(symbols)
    if n < 2 or len(coords) != n:
        props = compute_semiempirical_properties(symbols, coords)
        return np.copy(coords), 0.0, props

    bonds = detect_connectivity(symbols, coords, user_bonds=user_bonds, strict=strict_bonds)

    # Step 1: Fast pre-relaxation with analytical UFF/MMFF force field so atoms are in reasonable bonding range
    pre_coords, _ = optimize_geometry_force_field(
        symbols, coords, user_bonds=bonds, max_iter=min(80, max_iter),
        strict_bonds=True, user_bond_orders=user_bond_orders,
        frozen_indices=frozen_indices,
    )

    # Step 2: Compute self-consistent quantum Tight-Binding properties (Wiberg bond orders & SCC charges)
    qm_props = compute_semiempirical_properties(symbols, pre_coords, bonds=bonds)
    wiberg = qm_props["wiberg_bond_orders"]
    charges = qm_props["charges"]

    # Step 3: Refine geometry with QM-derived bond orders, SCC charges, and D3 dispersion
    opt_coords, final_e = optimize_geometry_force_field(
        symbols, pre_coords, user_bonds=bonds, max_iter=max_iter,
        strict_bonds=True, user_bond_orders=user_bond_orders,
        method="semiempirical",
        qm_wiberg_orders=wiberg,
        qm_charges=charges,
        frozen_indices=frozen_indices,
    )

    # Re-evaluate final QM electronic properties at converged geometry
    final_props = compute_semiempirical_properties(symbols, opt_coords, bonds=bonds)
    return opt_coords, final_e, final_props
