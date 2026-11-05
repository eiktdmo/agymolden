"""
Reaction Path and Transition State (TS) Guess Interpolator.

Implements:
1. Kabsch SVD alignment (optimal rigid rotational/translational superposition).
2. Linear transit interpolation between Reactant and Product geometries.
3. Image-Dependent Pair Potential (IDPP) optimization (Jónsson et al., J. Chem. Phys. 140, 214106, 2014)
   yielding collision-free, chemically intuitive reaction trajectories.
4. Input file generation for ORCA NEB-TS, Gaussian QST2/QST3, and xTB path scans.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize


def kabsch_align(reference: np.ndarray, mobile: np.ndarray) -> np.ndarray:
    """
    Rigorously align mobile coordinates onto reference coordinates using the Kabsch algorithm
    (SVD with reflection parity correction) to eliminate rigid translation and rotation.
    """
    p = np.array(reference, dtype=np.float64)
    q = np.array(mobile, dtype=np.float64)
    if len(p) != len(q) or len(p) == 0:
        return q.copy()

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)
    p_c = p - centroid_p
    q_c = q - centroid_q

    h = q_c.T @ p_c
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # Ensure right-handed coordinate system (no reflection)
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T

    return (q_c @ r.T) + centroid_p


def compute_pair_distances(coords: np.ndarray) -> np.ndarray:
    """Compute N x N pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-16)


def interpolate_linear(r_coords: np.ndarray, p_coords: np.ndarray, n_images: int = 7) -> np.ndarray:
    """
    Linear transit interpolation between Reactant and Product.
    Returns array of shape (n_images, N, 3).
    """
    n_images = max(3, n_images)
    p_aligned = kabsch_align(r_coords, p_coords)
    alphas = np.linspace(0.0, 1.0, n_images)
    path = np.zeros((n_images, len(r_coords), 3), dtype=np.float64)
    for k, a in enumerate(alphas):
        path[k] = (1.0 - a) * r_coords + a * p_aligned
    return path


def _idpp_objective_and_grad(coords_flat: np.ndarray, target_dists: np.ndarray, n_atoms: int) -> Tuple[float, np.ndarray]:
    """
    Objective and exact analytical gradient for Image-Dependent Pair Potential (IDPP).
    S = sum_{i < j} ((d_{ij} - D_{ij}) / D_{ij}^2)^2
    """
    coords = coords_flat.reshape((n_atoms, 3))
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (N, N, 3)
    dists = np.sqrt(np.sum(diff ** 2, axis=-1) + 1e-14)         # (N, N)

    # Avoid self-distance division
    safe_targets = np.where(target_dists > 1e-4, target_dists, 1.0)
    weights = 1.0 / (safe_targets ** 2)

    delta = dists - target_dists
    # Triu mask for i < j
    mask = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)

    # Loss
    val = float(np.sum((weights[mask] * delta[mask]) ** 2))

    # Gradient: dS/d(coords_i) = 2 * sum_{j != i} (weights_ij^2 * delta_ij * (coords_i - coords_j) / dists_ij)
    coeff = 2.0 * (weights ** 2) * delta / np.maximum(dists, 1e-8)  # (N, N)
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)          # (N, 3)

    return val, grad.ravel()


def interpolate_idpp(r_coords: np.ndarray, p_coords: np.ndarray, n_images: int = 7, max_iter: int = 100) -> np.ndarray:
    """
    Image-Dependent Pair Potential (IDPP) interpolation.
    Eliminates atom collisions and produces chemically smooth transition trajectories.
    Returns array of shape (n_images, N, 3).
    """
    n_images = max(3, n_images)
    n_atoms = len(r_coords)
    p_aligned = kabsch_align(r_coords, p_coords)

    d_react = compute_pair_distances(r_coords)
    d_prod = compute_pair_distances(p_aligned)

    alphas = np.linspace(0.0, 1.0, n_images)
    path = np.zeros((n_images, n_atoms, 3), dtype=np.float64)
    path[0] = r_coords
    path[-1] = p_aligned

    for k in range(1, n_images - 1):
        a = alphas[k]
        target_d = (1.0 - a) * d_react + a * d_prod
        # Linear initial guess
        init_coords = (1.0 - a) * r_coords + a * p_aligned

        # Check for collinear or overlapping atoms in the initial guess (distance < 0.3 A)
        curr_dists = compute_pair_distances(init_coords)
        mask = np.triu(np.ones((n_atoms, n_atoms), dtype=bool), k=1)
        if np.any(curr_dists[mask] < 0.30):
            # Break collinear symmetry with deterministic orthogonal perturbation
            rng = np.random.RandomState(seed=42 + k)
            perturbation = rng.normal(scale=0.08, size=(n_atoms, 3))
            init_coords = init_coords + perturbation

        res = minimize(
            fun=_idpp_objective_and_grad,
            x0=init_coords.ravel(),
            args=(target_d, n_atoms),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "ftol": 1e-6, "gtol": 1e-4},
        )
        path[k] = res.x.reshape((n_atoms, 3))

    return path


def format_path_xyz(symbols: List[str], path_images: np.ndarray, comment_prefix: str = "Reaction Frame") -> str:
    """Export multi-frame XYZ string for all images along the reaction path."""
    n_atoms = len(symbols)
    blocks = []
    for k, coords in enumerate(path_images):
        lines = [str(n_atoms), f"{comment_prefix} {k + 1}/{len(path_images)}"]
        for s, (x, y, z) in zip(symbols, coords):
            lines.append(f"{s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


def generate_orca_neb_input(
    symbols: List[str],
    path_images: np.ndarray,
    method: str = "B3LYP",
    basis: str = "def2-SVP",
    dispersion: str = "D3BJ",
    charge: int = 0,
    mult: int = 1,
    n_procs: int = 4,
    memory_gb: int = 4,
) -> str:
    """Generate ready-to-run ORCA NEB-TS calculation input file."""
    disp_str = f" {dispersion}" if dispersion and dispersion != "None" else ""
    lines = [
        f"# AgyMolden IDPP Reaction Path -> ORCA NEB-TS",
        f"! {method} {basis}{disp_str} NEB-TS Opt",
        f"%pal nprocs {n_procs} end",
        f"%maxcore {int(memory_gb * 1024 / max(1, n_procs))}",
        "",
        "%neb",
        f"  NImages {len(path_images)}",
        "end",
        "",
        f"* xyz {charge} {mult}",
    ]
    # In ORCA NEB, the reactant is in the main coordinates block, or all images in trajectory
    for s, (x, y, z) in zip(symbols, path_images[0]):
        lines.append(f"  {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("*")
    return "\n".join(lines) + "\n"


def generate_gaussian_qst2_input(
    symbols: List[str],
    r_coords: np.ndarray,
    p_coords: np.ndarray,
    method: str = "B3LYP",
    basis: str = "6-31G*",
    charge: int = 0,
    mult: int = 1,
    n_procs: int = 4,
    memory_gb: int = 4,
) -> str:
    """Generate Gaussian QST2 transition state calculation input file."""
    p_aligned = kabsch_align(r_coords, p_coords)
    lines = [
        f"%nprocshared={n_procs}",
        f"%mem={memory_gb}GB",
        f"#p {method}/{basis} opt=(qst2,calcfc) gfinput",
        "",
        "Reactant Structure",
        "",
        f"{charge} {mult}",
    ]
    for s, (x, y, z) in zip(symbols, r_coords):
        lines.append(f" {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("")
    lines.append("Product Structure")
    lines.append("")
    lines.append(f"{charge} {mult}")
    for s, (x, y, z) in zip(symbols, p_aligned):
        lines.append(f" {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("\n")
    return "\n".join(lines)


def generate_gaussian_qst3_input(
    symbols: List[str],
    r_coords: np.ndarray,
    p_coords: np.ndarray,
    ts_coords: np.ndarray,
    method: str = "B3LYP",
    basis: str = "6-31G*",
    charge: int = 0,
    mult: int = 1,
    n_procs: int = 4,
    memory_gb: int = 4,
) -> str:
    """Generate Gaussian QST3 transition state calculation input with IDPP TS guess."""
    p_aligned = kabsch_align(r_coords, p_coords)
    ts_aligned = kabsch_align(r_coords, ts_coords)
    lines = [
        f"%nprocshared={n_procs}",
        f"%mem={memory_gb}GB",
        f"#p {method}/{basis} opt=(qst3,calcfc) gfinput",
        "",
        "Reactant Structure",
        "",
        f"{charge} {mult}",
    ]
    for s, (x, y, z) in zip(symbols, r_coords):
        lines.append(f" {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("")
    lines.append("Product Structure")
    lines.append("")
    lines.append(f"{charge} {mult}")
    for s, (x, y, z) in zip(symbols, p_aligned):
        lines.append(f" {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("")
    lines.append("Transition State Guess (AgyMolden IDPP Peak)")
    lines.append("")
    lines.append(f"{charge} {mult}")
    for s, (x, y, z) in zip(symbols, ts_aligned):
        lines.append(f" {s:<2}  {x:12.6f}  {y:12.6f}  {z:12.6f}")
    lines.append("\n")
    return "\n".join(lines)
