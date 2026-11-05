"""
Periodic Boundary Conditions (PBC) & Supercell Module for AgyMolden.

Provides:
- UnitCell dataclass: lattice parameters (a, b, c, alpha, beta, gamma),
  Cartesian lattice vectors, metric tensor, reciprocal lattice,
  fractional <-> Cartesian conversion, and 3D bounding box edges.
- Supercell replication: expands crystallographic unit cells to Nx x Ny x Nz supercells
  with boundary duplicate removal.
- Auto-box detection from molecular coordinates with user-defined padding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class UnitCell:
    """
    Represents a crystallographic or simulation unit cell.
    Lattice lengths in Angstrom, angles in degrees.
    """
    a: float = 10.0
    b: float = 10.0
    c: float = 10.0
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def lattice_vectors(self) -> np.ndarray:
        """
        Returns 3x3 array where row 0 is vector a, row 1 is vector b, row 2 is vector c in Cartesian Angstroms.
        Convention:
          a is along X axis: (a, 0, 0)
          b is in XY plane: (b * cos(gamma), b * sin(gamma), 0)
          c has Z component chosen to satisfy alpha, beta, and volume.
        """
        a, b, c = float(self.a), float(self.b), float(self.c)
        rad_a = math.radians(float(self.alpha))
        rad_b = math.radians(float(self.beta))
        rad_g = math.radians(float(self.gamma))

        cos_a = math.cos(rad_a)
        cos_b = math.cos(rad_b)
        cos_g = math.cos(rad_g)
        sin_g = math.sin(rad_g)
        if abs(sin_g) < 1e-7:
            sin_g = 1e-7

        va = np.array([a, 0.0, 0.0], dtype=np.float64)
        vb = np.array([b * cos_g, b * sin_g, 0.0], dtype=np.float64)

        term = 1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
        v_norm = math.sqrt(max(0.0, term))
        vol = a * b * c * v_norm

        cz = vol / (a * b * sin_g) if (a * b * sin_g) != 0 else c
        cy = c * (cos_a - cos_b * cos_g) / sin_g
        cx = c * cos_b
        vc = np.array([cx, cy, cz], dtype=np.float64)

        return np.vstack([va, vb, vc])

    @property
    def volume(self) -> float:
        """Unit cell volume in Angstrom^3."""
        vecs = self.lattice_vectors
        return float(abs(np.dot(vecs[0], np.cross(vecs[1], vecs[2]))))

    @property
    def metric_tensor(self) -> np.ndarray:
        """3x3 metric tensor G_ij = v_i . v_j."""
        M = self.lattice_vectors
        return M @ M.T

    @property
    def reciprocal_lattice_vectors(self) -> np.ndarray:
        """
        Returns 3x3 array of reciprocal lattice vectors b1, b2, b3 in units of 2pi / Angstrom:
        b1 = 2pi * (v2 x v3) / V, etc.
        """
        M = self.lattice_vectors
        v1, v2, v3 = M[0], M[1], M[2]
        vol = self.volume
        if vol < 1e-9:
            return np.zeros((3, 3), dtype=np.float64)
        factor = 2.0 * math.pi / vol
        b1 = factor * np.cross(v2, v3)
        b2 = factor * np.cross(v3, v1)
        b3 = factor * np.cross(v1, v2)
        return np.vstack([b1, b2, b3])

    def cartesian_to_fractional(self, coords: np.ndarray) -> np.ndarray:
        """Converts Cartesian coordinates to fractional coordinates (0 to 1)."""
        if len(coords) == 0:
            return np.empty((0, 3), dtype=np.float64)
        coords_arr = np.asarray(coords, dtype=np.float64) - np.asarray(self.origin, dtype=np.float64)
        M = self.lattice_vectors
        # coords = frac @ M => frac = coords @ inv(M)
        M_inv = np.linalg.inv(M)
        return coords_arr @ M_inv

    def fractional_to_cartesian(self, frac_coords: np.ndarray) -> np.ndarray:
        """Converts fractional coordinates to Cartesian coordinates."""
        if len(frac_coords) == 0:
            return np.empty((0, 3), dtype=np.float64)
        frac_arr = np.asarray(frac_coords, dtype=np.float64)
        M = self.lattice_vectors
        return (frac_arr @ M) + np.asarray(self.origin, dtype=np.float64)

    def wrap_coordinates(self, coords: np.ndarray, to_range: str = "[0,1)") -> np.ndarray:
        """
        Wraps Cartesian coordinates into the unit cell box.
        to_range can be '[0,1)' or '[-0.5,0.5)'.
        """
        frac = self.cartesian_to_fractional(coords)
        if to_range == "[-0.5,0.5)":
            frac = frac - np.round(frac)
        else:
            frac = frac % 1.0
        return self.fractional_to_cartesian(frac)

    def get_box_vertices(self) -> List[np.ndarray]:
        """Returns 8 vertices of the unit cell parallelepiped in Cartesian Angstroms."""
        M = self.lattice_vectors
        o = np.asarray(self.origin, dtype=np.float64)
        v0 = o
        v1 = o + M[0]
        v2 = o + M[1]
        v3 = o + M[0] + M[1]
        v4 = o + M[2]
        v5 = o + M[0] + M[2]
        v6 = o + M[1] + M[2]
        v7 = o + M[0] + M[1] + M[2]
        return [v0, v1, v2, v3, v4, v5, v6, v7]

    def get_box_edges(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Returns 12 edges as (start_pt, end_pt, axis_label) tuples:
        'a' for vector a edges, 'b' for b edges, 'c' for c edges.
        """
        pts = self.get_box_vertices()
        # Bottom face: 0, 1, 3, 2
        # Top face:    4, 5, 7, 6
        edges = [
            # a edges (parallel to vector a)
            (pts[0], pts[1], "a"),
            (pts[2], pts[3], "a"),
            (pts[4], pts[5], "a"),
            (pts[6], pts[7], "a"),
            # b edges (parallel to vector b)
            (pts[0], pts[2], "b"),
            (pts[1], pts[3], "b"),
            (pts[4], pts[6], "b"),
            (pts[5], pts[7], "b"),
            # c edges (parallel to vector c)
            (pts[0], pts[4], "c"),
            (pts[1], pts[5], "c"),
            (pts[2], pts[6], "c"),
            (pts[3], pts[7], "c"),
        ]
        return edges

    def to_dict(self) -> Dict[str, float]:
        """Dictionary representation."""
        return {
            "a": float(self.a),
            "b": float(self.b),
            "c": float(self.c),
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "gamma": float(self.gamma),
            "volume": self.volume,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> UnitCell:
        return cls(
            a=float(d.get("a", 10.0)),
            b=float(d.get("b", 10.0)),
            c=float(d.get("c", 10.0)),
            alpha=float(d.get("alpha", 90.0)),
            beta=float(d.get("beta", 90.0)),
            gamma=float(d.get("gamma", 90.0)),
            origin=tuple(d.get("origin", (0.0, 0.0, 0.0))),
        )


def create_box_from_molecule(coords: np.ndarray, padding: float = 6.0) -> UnitCell:
    """
    Constructs an orthorhombic unit cell enclosing the molecule with the specified padding.
    """
    if len(coords) == 0:
        return UnitCell(a=20.0, b=20.0, c=20.0, alpha=90.0, beta=90.0, gamma=90.0, origin=(-10.0, -10.0, -10.0))

    coords_arr = np.asarray(coords, dtype=np.float64)
    min_xyz = np.min(coords_arr, axis=0) - padding
    max_xyz = np.max(coords_arr, axis=0) + padding
    lengths = max_xyz - min_xyz

    return UnitCell(
        a=float(lengths[0]),
        b=float(lengths[1]),
        c=float(lengths[2]),
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        origin=(float(min_xyz[0]), float(min_xyz[1]), float(min_xyz[2])),
    )


def replicate_supercell(
    symbols: List[str],
    coords: np.ndarray,
    unit_cell: UnitCell,
    nx: int = 1,
    ny: int = 1,
    nz: int = 1,
    remove_duplicates: bool = True,
    duplicate_tol: float = 0.08,
) -> Tuple[List[str], np.ndarray, UnitCell]:
    """
    Replicates the given structure into an Nx x Ny x Nz supercell.
    Returns (new_symbols, new_coords, new_unit_cell).
    """
    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("Supercell replication factors must be >= 1.")

    if len(symbols) == 0 or len(coords) == 0:
        new_cell = UnitCell(
            a=unit_cell.a * nx,
            b=unit_cell.b * ny,
            c=unit_cell.c * nz,
            alpha=unit_cell.alpha,
            beta=unit_cell.beta,
            gamma=unit_cell.gamma,
            origin=unit_cell.origin,
        )
        return [], np.empty((0, 3)), new_cell

    M = unit_cell.lattice_vectors
    va, vb, vc = M[0], M[1], M[2]

    base_coords = np.asarray(coords, dtype=np.float64)
    n_base = len(symbols)

    rep_symbols: List[str] = []
    rep_coords_list: List[np.ndarray] = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = ix * va + iy * vb + iz * vc
                shifted_coords = base_coords + shift
                rep_symbols.extend(symbols)
                rep_coords_list.append(shifted_coords)

    all_coords = np.vstack(rep_coords_list)

    if remove_duplicates and len(all_coords) > 1:
        tree = cKDTree(all_coords)
        pairs = tree.query_pairs(r=duplicate_tol)
        if pairs:
            # Drop the higher index of each duplicate pair
            to_remove = set()
            for i1, i2 in pairs:
                to_remove.add(max(i1, i2))
            keep_mask = np.ones(len(all_coords), dtype=bool)
            for idx in to_remove:
                keep_mask[idx] = False

            all_coords = all_coords[keep_mask]
            rep_symbols = [s for i, s in enumerate(rep_symbols) if keep_mask[i]]

    new_cell = UnitCell(
        a=float(unit_cell.a * nx),
        b=float(unit_cell.b * ny),
        c=float(unit_cell.c * nz),
        alpha=float(unit_cell.alpha),
        beta=float(unit_cell.beta),
        gamma=float(unit_cell.gamma),
        origin=unit_cell.origin,
    )

    return rep_symbols, all_coords, new_cell
