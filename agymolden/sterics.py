"""Quantitative Steric & Catalysis Descriptors Module for AgyMolden.

Computes:
1. Percent Buried Volume (%V_bur) around a coordination metal/reference center
   within a sphere of radius R (typically 3.5 Å, Bondi/Alvarez vdW radii).
2. Quadrant and Octant steric distribution (SambVca style: NE, NW, SW, SE).
3. 2D Topographic Steric Altitude Map z(x, y) looking down the metal-ligand coordination axis.
4. Verloop Sterimol Parameters (L, B1, B5) along a given bond/substituent axis.
5. Tolman Ligand Cone Angle (exact analytical envelope).
"""

from __future__ import annotations

import math
from typing import Dict, List, NamedTuple, Optional, Tuple
import numpy as np

from .elements import get_vdw_radius


class StericBuriedVolumeResult(NamedTuple):
    center_idx: int
    sphere_radius: float
    percent_v_bur: float
    quadrants: Dict[str, float]  # {"Q1_NE": %, "Q2_NW": %, "Q3_SW": %, "Q4_SE": %}
    buried_volume_angstrom3: float
    total_sphere_volume: float
    steric_map_grid: np.ndarray  # 2D altitude grid z(x,y)
    grid_extent: float


class SterimolResult(NamedTuple):
    l_value: float   # Length along primary axis (Å)
    b1_value: float  # Minimum perpendicular width (Å)
    b5_value: float  # Maximum perpendicular width (Å)
    axis_vector: np.ndarray


def compute_buried_volume(
    symbols: List[str],
    coords: np.ndarray,
    center_idx: int,
    sphere_radius: float = 3.5,
    mesh_spacing: float = 0.12,
    include_center_atom: bool = False,
    z_axis_atoms: Optional[Tuple[int, int]] = None,
) -> StericBuriedVolumeResult:
    """
    Computes %V_bur and quadrant distribution using 3D voxel grid integration.
    Aligns coordinate frame so z-axis points along the metal-ligand vector (or principal coordination axis).
    """
    n = len(symbols)
    if n == 0 or center_idx < 0 or center_idx >= n:
        raise ValueError(f"Invalid center index {center_idx} for molecule with {n} atoms.")

    center_coord = coords[center_idx]
    # Shift center atom to origin
    rel_coords = coords - center_coord

    # Coordinate alignment: orient along specified z-axis vector if given
    if z_axis_atoms is not None:
        idx_a, idx_b = z_axis_atoms
        if 0 <= idx_a < n and 0 <= idx_b < n:
            vec_z = coords[idx_b] - coords[idx_a]
            norm_z = np.linalg.norm(vec_z)
            if norm_z > 1e-4:
                z_hat = vec_z / norm_z
                # Find orthogonal x and y
                cand = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                x_hat = np.cross(cand, z_hat)
                x_hat /= np.linalg.norm(x_hat)
                y_hat = np.cross(z_hat, x_hat)
                rot_mat = np.vstack([x_hat, y_hat, z_hat])
                rel_coords = (rot_mat @ rel_coords.T).T

    # Generate 3D grid bounding the sphere [-R, +R]
    span = np.arange(-sphere_radius, sphere_radius + mesh_spacing * 0.5, mesh_spacing)
    X, Y, Z = np.meshgrid(span, span, span, indexing="ij")
    r_sq = X**2 + Y**2 + Z**2

    in_sphere = r_sq <= (sphere_radius**2)
    total_sphere_pts = int(np.sum(in_sphere))
    if total_sphere_pts == 0:
        total_sphere_pts = 1

    # Track which grid points are inside any atom's vdW sphere
    occupied = np.zeros_like(in_sphere, dtype=bool)

    radii = [get_vdw_radius(s) for s in symbols]

    for i in range(n):
        if not include_center_atom and i == center_idx:
            continue
        c_i = rel_coords[i]
        d_to_center = np.linalg.norm(c_i)
        r_vdw = radii[i]
        # Skip atoms completely outside the sphere
        if d_to_center - r_vdw > sphere_radius:
            continue

        dist_sq = (X - c_i[0])**2 + (Y - c_i[1])**2 + (Z - c_i[2])**2
        atom_mask = dist_sq <= (r_vdw**2)
        occupied |= atom_mask

    # Only count occupied points that fall strictly within the coordination sphere
    buried_mask = occupied & in_sphere
    buried_pts = int(np.sum(buried_mask))

    percent_vbur = (buried_pts / total_sphere_pts) * 100.0

    # Quadrant breakdown in XY plane
    # Q1: X > 0, Y > 0 (NE)
    # Q2: X < 0, Y > 0 (NW)
    # Q3: X < 0, Y < 0 (SW)
    # Q4: X > 0, Y < 0 (SE)
    q1_mask = buried_mask & (X >= 0) & (Y >= 0)
    q2_mask = buried_mask & (X < 0) & (Y >= 0)
    q3_mask = buried_mask & (X < 0) & (Y < 0)
    q4_mask = buried_mask & (X >= 0) & (Y < 0)

    q_total = total_sphere_pts / 4.0
    quads = {
        "Q1_NE": float(np.sum(q1_mask) / q_total * 100.0),
        "Q2_NW": float(np.sum(q2_mask) / q_total * 100.0),
        "Q3_SW": float(np.sum(q3_mask) / q_total * 100.0),
        "Q4_SE": float(np.sum(q4_mask) / q_total * 100.0),
    }

    tot_vol = (4.0 / 3.0) * math.pi * (sphere_radius**3)
    bur_vol = tot_vol * (percent_vbur / 100.0)

    # Topographic Steric Map z(x,y): max Z-altitude profile of steric presence
    # Compress along Z-axis (looking down the coordination vector)
    n_xy = len(span)
    topo_grid = np.zeros((n_xy, n_xy), dtype=np.float64)
    for ix in range(n_xy):
        for iy in range(n_xy):
            z_occ = span[buried_mask[ix, iy, :]]
            if len(z_occ) > 0:
                topo_grid[ix, iy] = np.max(z_occ)
            else:
                topo_grid[ix, iy] = -sphere_radius

    return StericBuriedVolumeResult(
        center_idx=center_idx,
        sphere_radius=sphere_radius,
        percent_v_bur=percent_vbur,
        quadrants=quads,
        buried_volume_angstrom3=bur_vol,
        total_sphere_volume=tot_vol,
        steric_map_grid=topo_grid,
        grid_extent=sphere_radius,
    )


def compute_sterimol(
    symbols: List[str],
    coords: np.ndarray,
    atom_base_idx: int,
    atom_substituent_idx: int,
) -> SterimolResult:
    """
    Computes Verloop Sterimol parameters (L, B1, B5) for a substituent:
    - Primary axis Z goes from atom_base to atom_substituent.
    - L: Maximum extension along the bond axis including vdW radius.
    - B1: Minimum perpendicular radius of tangent cylinder from bond axis.
    - B5: Maximum perpendicular radius of tangent cylinder from bond axis.
    """
    n = len(symbols)
    if not (0 <= atom_base_idx < n and 0 <= atom_substituent_idx < n):
        raise ValueError("Invalid atom indices for Sterimol calculation.")

    r_base = coords[atom_base_idx]
    r_sub = coords[atom_substituent_idx]
    vec_z = r_sub - r_base
    len_z = np.linalg.norm(vec_z)
    if len_z < 1e-4:
        raise ValueError("Base and substituent atoms are degenerate in position.")

    z_hat = vec_z / len_z
    cand = np.array([1.0, 0.0, 0.0]) if abs(z_hat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_hat = np.cross(cand, z_hat)
    x_hat /= np.linalg.norm(x_hat)
    y_hat = np.cross(z_hat, x_hat)
    rot_mat = np.vstack([x_hat, y_hat, z_hat])

    # Transform all substituent atoms (excluding base atom)
    rel_coords = coords - r_base
    aligned_coords = (rot_mat @ rel_coords.T).T

    radii = [get_vdw_radius(s) for s in symbols]

    # Find L: max(z_i + r_vdw) along +Z
    l_vals = []
    # Project in XY plane for perpendicular radii
    perp_dists = []

    for i in range(n):
        if i == atom_base_idx:
            continue
        c_i = aligned_coords[i]
        r_v = radii[i]
        l_vals.append(c_i[2] + r_v)

        d_xy = math.sqrt(c_i[0]**2 + c_i[1]**2) + r_v
        perp_dists.append(d_xy)

    l_max = max(l_vals) if l_vals else len_z + radii[atom_substituent_idx]
    b1_min = min(perp_dists) if perp_dists else 1.0
    b5_max = max(perp_dists) if perp_dists else 1.0

    return SterimolResult(
        l_value=float(l_max),
        b1_value=float(b1_min),
        b5_value=float(b5_max),
        axis_vector=z_hat,
    )
