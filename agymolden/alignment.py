"""
Precision 3D Coordinate Centering, Axis/Plane Alignment, and Rigid-Frame Preservation.

This module provides:
  1. Centering functions:
     - center_at_com(symbols, coords): Center of nuclear mass at (0, 0, 0).
     - center_at_centroid(coords): Geometric centroid at (0, 0, 0).
     - center_at_atom(coords, atom_idx): Selected atom i placed at (0, 0, 0).
  2. Axis alignment functions:
     - align_vector_to_axis(coords, vec, target_axis): Rotate molecule around origin so vec
       points along +X, -X, +Y, -Y, +Z, or -Z.
     - align_atoms_to_axis(coords, atom_indices, target_axis, center_first=False):
       * If 1 atom selected: aligns position vector r[i] (from origin) along target_axis.
       * If 2+ atoms selected: aligns vector (r[j] - r[i]) along target_axis.
  3. Plane alignment functions:
     - align_plane_to_cartesian(coords, atom_indices, target_plane, lock_primary_axis=True):
       Rotates molecule so the plane defined by 3 selected atoms (or molecular best-fit plane
       if <3 atoms selected) lies in XY, XZ, or YZ, while optionally preserving the primary
       bond vector direction (e.g. atom0 -> atom1) along the plane's primary axis.
  4. Rigid-Frame Preservation for Z-Matrix Reconstruction:
     - RigidFrame: captures the exact 3D position and orientation of the initial Z-matrix
       reference frame (Atom 0, Atom 1, Atom 2) so that `zmatrix_to_cartesian_in_frame`
       reconstructs updated Z-matrix internal coordinates in the exact user-chosen 3D
       orientation without resetting Atom 0 to origin or Atom 1 to +Z.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from .elements import ELEMENT_DATA, normalize_symbol


AXIS_VECTORS = {
    "X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "+X": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "+Y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "-Y": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "+Z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    "-Z": np.array([0.0, 0.0, -1.0], dtype=np.float64),
}

# Normal vector and primary/secondary in-plane axes for standard Cartesian planes
PLANE_SPECS = {
    "XY": (
        np.array([0.0, 0.0, 1.0], dtype=np.float64),  # normal = Z
        np.array([1.0, 0.0, 0.0], dtype=np.float64),  # primary in-plane = X
        np.array([0.0, 1.0, 0.0], dtype=np.float64),  # secondary in-plane = Y
    ),
    "XZ": (
        np.array([0.0, 1.0, 0.0], dtype=np.float64),  # normal = Y
        np.array([0.0, 0.0, 1.0], dtype=np.float64),  # primary in-plane = Z
        np.array([1.0, 0.0, 0.0], dtype=np.float64),  # secondary in-plane = X
    ),
    "YZ": (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),  # normal = X
        np.array([0.0, 0.0, 1.0], dtype=np.float64),  # secondary/primary = Z
        np.array([0.0, 1.0, 0.0], dtype=np.float64),  # in-plane = Y
    ),
}


def rotation_matrix_from_vectors(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Compute exact 3x3 proper rotation matrix R (det=+1) that rotates unit vector v_from to v_to.
    """
    n_from = float(np.linalg.norm(v_from))
    n_to = float(np.linalg.norm(v_to))
    if n_from < 1e-12 or n_to < 1e-12:
        return np.eye(3, dtype=np.float64)

    a = np.asarray(v_from, dtype=np.float64) / n_from
    b = np.asarray(v_to, dtype=np.float64) / n_to
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if c > 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-10:
        # 180-degree rotation around any axis perpendicular to a
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    axis /= s
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def rodrigues_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Proper rotation matrix around unit axis by angle_rad."""
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(angle_rad) < 1e-12:
        return np.eye(3, dtype=np.float64)
    u = np.asarray(axis, dtype=np.float64) / norm
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    K = np.array([
        [0.0, -u[2], u[1]],
        [u[2], 0.0, -u[0]],
        [-u[1], u[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def get_atomic_masses(symbols: List[str]) -> np.ndarray:
    """Return atomic masses array for a list of element symbols (dummy 'X' gets 0.0)."""
    masses = []
    for s in symbols:
        norm = normalize_symbol(s)
        if norm == "X":
            masses.append(0.0)
        else:
            m = ELEMENT_DATA.get(norm, {}).get("mass", 12.0)
            masses.append(max(0.0, float(m)))
    arr = np.array(masses, dtype=np.float64)
    if np.sum(arr) < 1e-9 and len(arr) > 0:
        arr = np.ones(len(symbols), dtype=np.float64)
    return arr


def compute_center_of_mass(symbols: List[str], coords: np.ndarray) -> np.ndarray:
    """Compute nuclear center of mass (COM) in 3D."""
    if len(coords) == 0:
        return np.zeros(3, dtype=np.float64)
    masses = get_atomic_masses(symbols)
    total_m = float(np.sum(masses))
    if total_m < 1e-12:
        return np.mean(coords, axis=0)
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_m


def center_at_com(symbols: List[str], coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate coordinates so Center of Mass (COM) is at (0, 0, 0). Returns (new_coords, shift)."""
    if len(coords) == 0:
        return np.copy(coords), np.zeros(3, dtype=np.float64)
    com = compute_center_of_mass(symbols, coords)
    return np.asarray(coords, dtype=np.float64) - com, -com


def center_at_centroid(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Translate coordinates so geometric centroid is at (0, 0, 0). Returns (new_coords, shift)."""
    if len(coords) == 0:
        return np.copy(coords), np.zeros(3, dtype=np.float64)
    centroid = np.mean(coords, axis=0)
    return np.asarray(coords, dtype=np.float64) - centroid, -centroid


def center_at_atom(coords: np.ndarray, atom_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Translate coordinates so atom `atom_idx` is placed at (0, 0, 0). Returns (new_coords, shift)."""
    if len(coords) == 0 or atom_idx < 0 or atom_idx >= len(coords):
        return np.copy(coords), np.zeros(3, dtype=np.float64)
    origin_atom = np.array(coords[atom_idx], dtype=np.float64)
    return np.asarray(coords, dtype=np.float64) - origin_atom, -origin_atom


def align_vector_to_axis(
    coords: np.ndarray,
    vec: np.ndarray,
    target_axis: str = "Z",
    pivot: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate coordinates around `pivot` (defaults to origin [0, 0, 0]) so that `vec`
    points along `target_axis` ('X', 'Y', 'Z', '+X', '-X', '+Y', '-Y', '+Z', '-Z').
    Returns (rotated_coords, R_3x3).
    """
    if len(coords) == 0:
        return np.copy(coords), np.eye(3, dtype=np.float64)
    key = target_axis.upper().strip()
    t_vec = AXIS_VECTORS.get(key, AXIS_VECTORS["Z"])
    R = rotation_matrix_from_vectors(vec, t_vec)
    pts = np.asarray(coords, dtype=np.float64)
    if pivot is not None:
        piv = np.asarray(pivot, dtype=np.float64)
        rotated = (R @ (pts - piv).T).T + piv
    else:
        rotated = (R @ pts.T).T
    return rotated, R


def align_atoms_to_axis(
    symbols: List[str],
    coords: np.ndarray,
    selected_indices: List[int],
    target_axis: str = "Z",
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Align molecule to `target_axis` ('X', 'Y', 'Z', '-X', '-Y', '-Z') based on selection:
      - If 2 or more atoms [i, j, ...] are selected: aligns vector (coords[j] - coords[i])
        along target_axis, rotating around the origin (0, 0, 0).
      - If 1 atom [i] is selected: aligns vector coords[i] (from origin to atom i) along
        target_axis; if atom i is at the origin, aligns its bond or principal axis.
      - If 0 atoms are selected: aligns the molecule's principal symmetry axis (highest Cn)
        or principal inertia axis along target_axis.
    Returns (rotated_coords, R_3x3, description_message).
    """
    n = len(coords)
    if n == 0:
        return np.copy(coords), np.eye(3), "No atoms to align."

    valid = [i for i in selected_indices if 0 <= i < n]
    axis_label = target_axis.upper().strip()
    if not axis_label.startswith(("+", "-")):
        axis_label = "+" + axis_label

    if len(valid) >= 2:
        i, j = valid[0], valid[1]
        vec = coords[j] - coords[i]
        if np.linalg.norm(vec) < 1e-8:
            return np.copy(coords), np.eye(3), f"Atoms {symbols[i]}{i+1} and {symbols[j]}{j+1} coincide."
        rot_coords, R = align_vector_to_axis(coords, vec, target_axis)
        desc = f"Aligned vector {symbols[i]}{i+1} → {symbols[j]}{j+1} along {axis_label} axis"
        return rot_coords, R, desc

    if len(valid) == 1:
        i = valid[0]
        vec = coords[i]
        if np.linalg.norm(vec) >= 1e-5:
            rot_coords, R = align_vector_to_axis(coords, vec, target_axis)
            desc = f"Aligned atom {symbols[i]}{i+1} position vector along {axis_label} axis"
            return rot_coords, R, desc

    # Fallback (0 atoms selected, or selected atom is at the origin):
    # Use highest Cn symmetry axis if present, else principal axis of inertia
    from .symmetry import find_symmetry_operations, _calculate_inertia_tensor
    sym_res = find_symmetry_operations(symbols, coords, tolerance=0.20)
    if sym_res.principal_axis is not None and np.linalg.norm(sym_res.principal_axis) > 1e-5:
        vec = sym_res.principal_axis
        src_name = f"principal symmetry axis ({sym_res.symbol})"
    else:
        centered = coords - compute_center_of_mass(symbols, coords)
        _, evecs = _calculate_inertia_tensor(symbols, centered)
        vec = evecs[:, 0]
        src_name = "principal inertia axis"

    rot_coords, R = align_vector_to_axis(coords, vec, target_axis)
    return rot_coords, R, f"Aligned {src_name} along {axis_label} axis"


def align_plane_to_cartesian(
    symbols: List[str],
    coords: np.ndarray,
    selected_indices: List[int],
    target_plane: str = "XZ",
    lock_primary_vector: bool = False,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Rotate molecule around the origin (0, 0, 0) so a reference plane lies in `target_plane`
    ('XY', 'XZ', or 'YZ'):
      - If >= 3 atoms [i, j, k] are selected:
          Plane is defined by v1 = coords[j] - coords[i] and v2 = coords[k] - coords[i],
          with normal n = v1 x v2.
          Uses the shortest arc rotation mapping n onto the target plane normal, or if one of the
          in-plane Cartesian axes already holds a symmetry/bond axis (or lock_primary_vector is True),
          preserves that in-plane axis!
      - If 2 atoms [i, j] are selected (and origin is not collinear with i, j):
          Plane is defined by origin (0,0,0), coords[i], and coords[j] (or molecular plane containing bond i-j).
      - If < 2 atoms are selected:
          Finds the best-fit molecular plane (SVD / largest inertia eigenvector normal or mirror plane)
          and aligns its normal perpendicular to `target_plane`.
    Returns (rotated_coords, R_3x3, description_message).
    """
    n = len(coords)
    if n < 2:
        return np.copy(coords), np.eye(3), "At least 2 atoms are required to define a plane."

    plane_key = target_plane.upper().strip()
    if plane_key not in PLANE_SPECS:
        plane_key = "XZ"
    t_normal, t_prim, t_sec = PLANE_SPECS[plane_key]

    valid = [i for i in selected_indices if 0 <= i < n]
    pts = np.asarray(coords, dtype=np.float64)

    if len(valid) >= 3:
        i, j, k = valid[0], valid[1], valid[2]
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        normal = np.cross(v1, v2)
        norm_n = float(np.linalg.norm(normal))
        if norm_n < 1e-7:
            return np.copy(coords), np.eye(3), (
                f"Selected atoms {symbols[i]}{i+1}, {symbols[j]}{j+1}, {symbols[k]}{k+1} are collinear."
            )
        normal /= norm_n
        # Choose sign of normal closer to t_normal so we don't flip 180° unnecessarily
        if float(np.dot(normal, t_normal)) < -1e-6:
            normal = -normal
        v1_u = v1 / float(np.linalg.norm(v1))

        # Step 1: Shortest-arc rotation of normal onto t_normal
        R1 = rotation_matrix_from_vectors(normal, t_normal)
        v1_rot = R1 @ v1_u

        # Step 2: Only apply an extra in-plane rotation if v1 was already nearly aligned with an
        # in-plane Cartesian axis (within ~8 degrees, dot > 0.99) OR if lock_primary_vector=True
        best_inplane = t_prim
        best_dot = -1.0
        for cand in (t_prim, -t_prim, t_sec, -t_sec):
            d = float(np.dot(v1_u, cand))
            if d > best_dot:
                best_dot = d
                best_inplane = cand

        if lock_primary_vector or best_dot > 0.99:
            target_inplane = best_inplane if best_dot > 0.85 else t_prim
            v1_proj = v1_rot - np.dot(v1_rot, t_normal) * t_normal
            if np.linalg.norm(v1_proj) > 1e-6:
                v1_proj /= np.linalg.norm(v1_proj)
                cross = np.cross(v1_proj, target_inplane)
                sin_a = float(np.dot(cross, t_normal))
                cos_a = float(np.clip(np.dot(v1_proj, target_inplane), -1.0, 1.0))
                ang = math.atan2(sin_a, cos_a)
                R2 = rodrigues_rotation(t_normal, ang)
                R = R2 @ R1
            else:
                R = R1
        else:
            R = R1

        rot_coords = (R @ pts.T).T
        desc = (
            f"Aligned plane ({symbols[i]}{i+1}-{symbols[j]}{j+1}-{symbols[k]}{k+1}) into {plane_key} plane"
        )
        return rot_coords, R, desc

    if len(valid) == 2 and n >= 3:
        i, j = valid[0], valid[1]
        v1 = pts[j] - pts[i]
        n_v1 = float(np.linalg.norm(v1))
        if n_v1 > 1e-7:
            v1_u = v1 / n_v1
            # Pick a third point (either origin if non-collinear, or another atom with largest distance from line i-j)
            cand_vec = -pts[i]  # vector to origin
            if np.linalg.norm(np.cross(v1_u, cand_vec)) < 1e-4:
                # Find atom furthest from line i-j
                best_k = None
                best_h = 0.0
                for m in range(n):
                    if m in (i, j):
                        continue
                    vm = pts[m] - pts[i]
                    h = float(np.linalg.norm(np.cross(v1_u, vm)))
                    if h > best_h:
                        best_h = h
                        best_k = m
                if best_k is not None:
                    return align_plane_to_cartesian(
                        symbols, coords, [i, j, best_k], target_plane, lock_primary_vector
                    )
            else:
                normal = np.cross(v1_u, cand_vec)
                normal /= np.linalg.norm(normal)
                R1 = rotation_matrix_from_vectors(normal, t_normal)
                rot_coords = (R1 @ pts.T).T
                return rot_coords, R1, f"Aligned plane (Origin-{symbols[i]}{i+1}-{symbols[j]}{j+1}) into {plane_key} plane"

    # Whole-molecule plane (0 or 1 atom selected):
    # Use least-squares best-fit plane normal (smallest singular value of centered coordinates)
    centered = pts - np.mean(pts, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    R = rotation_matrix_from_vectors(normal, t_normal)
    rot_coords = (R @ pts.T).T
    return rot_coords, R, f"Aligned molecular plane into {plane_key} plane"


# ==============================================================================
# Rigid-Frame Preservation for Z-Matrix Reconstruction
# ==============================================================================

@dataclass
class RigidFrame:
    """
    Stores the 3D Euclidean frame (origin + orthonormal basis) of the Z-matrix root atoms
    so that internal-coordinate updates preserve the global 3D translation and rotation
    of the molecule (essential after XYZ import or manual Center / Axis / Plane alignment).
    """
    origin: np.ndarray        # 3D position of Atom 0
    rotation: np.ndarray      # 3x3 orthogonal matrix mapping canonical Z-matrix frame -> world frame

    def copy(self) -> "RigidFrame":
        return RigidFrame(origin=np.copy(self.origin), rotation=np.copy(self.rotation))


def capture_rigid_frame(zatoms, coords: np.ndarray) -> Optional[RigidFrame]:
    """
    Given a list of ZAtom objects and their target 3D Cartesian coordinates `coords`,
    compute the exact rigid transformation (origin, rotation) that maps `zmatrix_to_cartesian(zatoms)`
    onto `coords`.
    """
    from .zmatrix import zmatrix_to_cartesian
    n = len(zatoms)
    if n == 0 or len(coords) != n:
        return None

    local_coords = zmatrix_to_cartesian(zatoms)
    origin = np.array(coords[0], dtype=np.float64)
    if n == 1:
        return RigidFrame(origin=origin, rotation=np.eye(3, dtype=np.float64))

    # Shift both sets so Atom 0 is at (0, 0, 0)
    P = local_coords - local_coords[0]
    Q = np.asarray(coords, dtype=np.float64) - origin

    if n == 2:
        R = rotation_matrix_from_vectors(P[1], Q[1])
        return RigidFrame(origin=origin, rotation=R)

    # For n >= 3: first align the Atom 0 -> ref1/Atom 1 vector exactly, then align the Atom 2 plane,
    # or use Kabsch SVD weighted toward the first 3 non-collinear atoms so editing Atom k >= 3
    # keeps Atoms 0, 1, 2 completely stationary in space!
    # Find the first atom >= 2 that is not collinear with P[1]
    v1_loc = P[1]
    v1_wld = Q[1]
    n1_loc = float(np.linalg.norm(v1_loc))
    n1_wld = float(np.linalg.norm(v1_wld))

    if n1_loc > 1e-7 and n1_wld > 1e-7:
        u1_loc = v1_loc / n1_loc
        u1_wld = v1_wld / n1_wld
        plane_idx = None
        for k in range(2, n):
            cross_loc = np.cross(u1_loc, P[k])
            cross_wld = np.cross(u1_wld, Q[k])
            if np.linalg.norm(cross_loc) > 1e-5 and np.linalg.norm(cross_wld) > 1e-5:
                plane_idx = k
                break

        if plane_idx is not None:
            # Orthonormal Triad (e1, e2, e3) in local frame and world frame
            e1_loc = u1_loc
            n_loc = np.cross(e1_loc, P[plane_idx])
            e3_loc = n_loc / np.linalg.norm(n_loc)
            e2_loc = np.cross(e3_loc, e1_loc)
            B_loc = np.column_stack([e1_loc, e2_loc, e3_loc])

            e1_wld = u1_wld
            n_wld = np.cross(e1_wld, Q[plane_idx])
            e3_wld = n_wld / np.linalg.norm(n_wld)
            e2_wld = np.cross(e3_wld, e1_wld)
            B_wld = np.column_stack([e1_wld, e2_wld, e3_wld])

            R = B_wld @ B_loc.T
            return RigidFrame(origin=origin, rotation=R)

    # Fallback: Kabsch SVD alignment over all atoms
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    R = Vt.T @ S @ U.T
    return RigidFrame(origin=origin, rotation=R)


def apply_rigid_frame(local_coords: np.ndarray, frame: Optional[RigidFrame]) -> np.ndarray:
    """Transform canonical Z-matrix coordinates into the stored RigidFrame."""
    if frame is None or len(local_coords) == 0:
        return np.copy(local_coords)
    shifted = np.asarray(local_coords, dtype=np.float64) - local_coords[0]
    return (frame.rotation @ shifted.T).T + frame.origin


def calculate_kabsch_alignment(
    ref_coords: np.ndarray,
    mob_coords: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute optimal rigid superposition of mobile coordinates onto reference coordinates
    using the Kabsch SVD algorithm.

    Returns:
        (aligned_coords, rmsd, R, translation)
        where aligned_coords = (R @ (mob_coords - c_mob).T).T + c_ref
    """
    P = np.asarray(ref_coords, dtype=np.float64)
    Q = np.asarray(mob_coords, dtype=np.float64)

    if len(P) != len(Q) or len(P) == 0:
        raise ValueError("Reference and mobile coordinate sets must have the same non-zero atom count.")

    if weights is None:
        c_ref = np.mean(P, axis=0)
        c_mob = np.mean(Q, axis=0)
        P_cen = P - c_ref
        Q_cen = Q - c_mob
        H = Q_cen.T @ P_cen
    else:
        w = np.asarray(weights, dtype=np.float64)[:, None]
        w_sum = float(np.sum(w)) + 1e-12
        c_ref = np.sum(P * w, axis=0) / w_sum
        c_mob = np.sum(Q * w, axis=0) / w_sum
        P_cen = P - c_ref
        Q_cen = Q - c_mob
        H = (Q_cen * w).T @ P_cen

    U, _, Vt = np.linalg.svd(H)
    d = float(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, 1.0 if d >= 0.0 else -1.0])
    R = Vt.T @ D @ U.T

    aligned = (R @ Q_cen.T).T + c_ref
    diff = P - aligned
    rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
    t = c_ref - (R @ c_mob)

    return aligned, rmsd, R, t
