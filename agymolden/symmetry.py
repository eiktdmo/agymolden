"""Point group symmetry analysis, character tables, irreducible representations, and symmetry enforcement."""

import math
from typing import List, Tuple, Dict, Optional, Set
import numpy as np

from .elements import ELEMENT_DATA, normalize_symbol


class SymmetryOperation:
    """Represents a symmetry operation (Identity, Inversion, Rotation, Reflection, Improper Rotation)."""
    def __init__(self, op_type: str, matrix: np.ndarray, label: str, order: int = 1, axis: Optional[np.ndarray] = None):
        self.op_type = op_type   # 'E', 'i', 'Cn', 'sigma', 'Sn'
        self.matrix = np.array(matrix, dtype=np.float64)
        self.label = label
        self.order = order
        self.axis = axis

    def __repr__(self):
        return f"<SymOp {self.label}>"


class PointGroupResult:
    """Encapsulates the detected point group, its symmetry elements, and representations."""
    def __init__(
        self,
        symbol: str,
        operations: List[SymmetryOperation],
        principal_axis: Optional[np.ndarray] = None,
        horizontal_plane: Optional[np.ndarray] = None,
        vertical_planes: Optional[List[np.ndarray]] = None,
        has_inversion: bool = False,
        horizontal_planes_count: int = 0,
        vertical_planes_count: int = 0,
        linear: bool = False,
        tolerance: float = 0.15
    ):
        self.symbol = symbol                     # e.g., 'C2v', 'D6h', 'Td', 'Oh', 'C1', 'Cs', 'Ci'
        self.operations = operations
        self.principal_axis = principal_axis
        self.horizontal_plane = horizontal_plane
        self.vertical_planes = vertical_planes or []
        self.has_inversion = has_inversion
        self.horizontal_planes_count = horizontal_planes_count
        self.vertical_planes_count = vertical_planes_count
        self.linear = linear
        self.tolerance = tolerance

        # Representations analysis results
        self.gamma_tot: Optional[Dict[str, float]] = None
        self.gamma_trans: Optional[Dict[str, int]] = None
        self.gamma_rot: Optional[Dict[str, int]] = None
        self.gamma_vib: Optional[Dict[str, int]] = None
        self.classes: Optional[List[Tuple[str, int, SymmetryOperation]]] = None

    @property
    def order(self) -> int:
        return len(self.operations)

    def summary(self) -> str:
        return f"Point Group: {self.symbol} (Order {self.order})"


# Character Tables for common molecular point groups
# Format: { 'classes': [(class_name, count)], 'irreps': { irrep_name: [chi_1, chi_2, ...] },
#           'trans': { irrep: count }, 'rot': { irrep: count } }
CHARACTER_TABLES: Dict[str, Dict] = {
    "C1": {
        "classes": [("E", 1)],
        "irreps": {"A": [1]},
        "trans": {"A": 3},
        "rot": {"A": 3},
    },
    "Cs": {
        "classes": [("E", 1), ("sigma_h", 1)],
        "irreps": {
            "A'":  [1,  1],
            'A"':  [1, -1]
        },
        "trans": {"A'": 2, 'A"': 1},
        "rot": {"A'": 1, 'A"': 2},
    },
    "Ci": {
        "classes": [("E", 1), ("i", 1)],
        "irreps": {
            "Ag": [1,  1],
            "Au": [1, -1]
        },
        "trans": {"Au": 3},
        "rot": {"Ag": 3},
    },
    "C2": {
        "classes": [("E", 1), ("C2", 1)],
        "irreps": {
            "A": [1,  1],
            "B": [1, -1]
        },
        "trans": {"A": 1, "B": 2},
        "rot": {"A": 1, "B": 2},
    },
    "C2v": {
        "classes": [("E", 1), ("C2", 1), ("sigma_v(xz)", 1), ("sigma_v(yz)", 1)],
        "irreps": {
            "A1": [1,  1,  1,  1],
            "A2": [1,  1, -1, -1],
            "B1": [1, -1,  1, -1],
            "B2": [1, -1, -1,  1]
        },
        "trans": {"A1": 1, "B1": 1, "B2": 1},
        "rot": {"A2": 1, "B1": 1, "B2": 1},
    },
    "C2h": {
        "classes": [("E", 1), ("C2", 1), ("i", 1), ("sigma_h", 1)],
        "irreps": {
            "Ag": [1,  1,  1,  1],
            "Bg": [1, -1,  1, -1],
            "Au": [1,  1, -1, -1],
            "Bu": [1, -1, -1,  1]
        },
        "trans": {"Au": 1, "Bu": 2},
        "rot": {"Ag": 1, "Bg": 2},
    },
    "C3v": {
        "classes": [("E", 1), ("2C3", 2), ("3sigma_v", 3)],
        "irreps": {
            "A1": [1,  1,  1],
            "A2": [1,  1, -1],
            "E":  [2, -1,  0]
        },
        "trans": {"A1": 1, "E": 1},
        "rot": {"A2": 1, "E": 1},
    },
    "C4v": {
        "classes": [("E", 1), ("2C4", 2), ("C2", 1), ("2sigma_v", 2), ("2sigma_d", 2)],
        "irreps": {
            "A1": [1,  1,  1,  1,  1],
            "A2": [1,  1,  1, -1, -1],
            "B1": [1, -1,  1,  1, -1],
            "B2": [1, -1,  1, -1,  1],
            "E":  [2,  0, -2,  0,  0]
        },
        "trans": {"A1": 1, "E": 1},
        "rot": {"A2": 1, "E": 1},
    },
    "D2": {
        "classes": [("E", 1), ("C2(z)", 1), ("C2(y)", 1), ("C2(x)", 1)],
        "irreps": {
            "A":  [1,  1,  1,  1],
            "B1": [1,  1, -1, -1],
            "B2": [1, -1,  1, -1],
            "B3": [1, -1, -1,  1]
        },
        "trans": {"B1": 1, "B2": 1, "B3": 1},
        "rot": {"B1": 1, "B2": 1, "B3": 1},
    },
    "D2h": {
        "classes": [("E", 1), ("C2(z)", 1), ("C2(y)", 1), ("C2(x)", 1), ("i", 1), ("sigma(xy)", 1), ("sigma(xz)", 1), ("sigma(yz)", 1)],
        "irreps": {
            "Ag":  [1,  1,  1,  1,  1,  1,  1,  1],
            "B1g": [1,  1, -1, -1,  1,  1, -1, -1],
            "B2g": [1, -1,  1, -1,  1, -1,  1, -1],
            "B3g": [1, -1, -1,  1,  1, -1, -1,  1],
            "Au":  [1,  1,  1,  1, -1, -1, -1, -1],
            "B1u": [1,  1, -1, -1, -1, -1,  1,  1],
            "B2u": [1, -1,  1, -1, -1,  1, -1,  1],
            "B3u": [1, -1, -1,  1, -1,  1,  1, -1],
        },
        "trans": {"B1u": 1, "B2u": 1, "B3u": 1},
        "rot": {"B1g": 1, "B2g": 1, "B3g": 1},
    },
    "D2d": {
        "classes": [("E", 1), ("2S4", 2), ("C2", 1), ("2C2'", 2), ("2sigma_d", 2)],
        "irreps": {
            "A1": [1,  1,  1,  1,  1],
            "A2": [1,  1,  1, -1, -1],
            "B1": [1, -1,  1,  1, -1],
            "B2": [1, -1,  1, -1,  1],
            "E":  [2,  0, -2,  0,  0]
        },
        "trans": {"B2": 1, "E": 1},
        "rot": {"A2": 1, "E": 1},
    },
    "D3h": {
        "classes": [("E", 1), ("2C3", 2), ("3C2'", 3), ("sigma_h", 1), ("2S3", 2), ("3sigma_v", 3)],
        "irreps": {
            "A1'":  [1,  1,  1,  1,  1,  1],
            "A2'":  [1,  1, -1,  1,  1, -1],
            "E'":   [2, -1,  0,  2, -1,  0],
            'A1"':  [1,  1,  1, -1, -1, -1],
            'A2"':  [1,  1, -1, -1, -1,  1],
            'E"':   [2, -1,  0, -2,  1,  0],
        },
        "trans": {"A2''": 1, "E'": 1},
        "rot": {"A2'": 1, 'E"': 1},
    },
    "D3d": {
        "classes": [("E", 1), ("2C3", 2), ("3C2'", 3), ("i", 1), ("2S6", 2), ("3sigma_d", 3)],
        "irreps": {
            "A1g": [1,  1,  1,  1,  1,  1],
            "A2g": [1,  1, -1,  1,  1, -1],
            "Eg":  [2, -1,  0,  2, -1,  0],
            "A1u": [1,  1,  1, -1, -1, -1],
            "A2u": [1,  1, -1, -1, -1,  1],
            "Eu":  [2, -1,  0, -2,  1,  0]
        },
        "trans": {"A2u": 1, "Eu": 1},
        "rot": {"A2g": 1, "Eg": 1},
    },
    "D6h": {
        "classes": [("E", 1), ("2C6", 2), ("2C3", 2), ("C2", 1), ("3C2'", 3), ("3C2''", 3),
                    ("i", 1), ("2S3", 2), ("2S6", 2), ("sigma_h", 1), ("3sigma_d", 3), ("3sigma_v", 3)],
        "irreps": {
            "A1g": [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            "A2g": [1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1],
            "B1g": [1, -1,  1, -1,  1, -1,  1, -1,  1, -1,  1, -1],
            "B2g": [1, -1,  1, -1, -1,  1,  1, -1,  1, -1, -1,  1],
            "E1g": [2,  1, -1, -2,  0,  0,  2,  1, -1, -2,  0,  0],
            "E2g": [2, -1, -1,  2,  0,  0,  2, -1, -1,  2,  0,  0],
            "A1u": [1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1],
            "A2u": [1,  1,  1,  1, -1, -1, -1, -1, -1, -1,  1,  1],
            "B1u": [1, -1,  1, -1,  1, -1, -1,  1, -1,  1, -1,  1],
            "B2u": [1, -1,  1, -1, -1,  1, -1,  1, -1,  1,  1, -1],
            "E1u": [2,  1, -1, -2,  0,  0, -2, -1,  1,  2,  0,  0],
            "E2u": [2, -1, -1,  2,  0,  0, -2,  1,  1, -2,  0,  0],
        },
        "trans": {"A2u": 1, "E1u": 1},
        "rot": {"A2g": 1, "E1g": 1},
    },
    "Td": {
        "classes": [("E", 1), ("8C3", 8), ("3C2", 3), ("6S4", 6), ("6sigma_d", 6)],
        "irreps": {
            "A1": [1,  1,  1,  1,  1],
            "A2": [1,  1,  1, -1, -1],
            "E":  [2, -1,  2,  0,  0],
            "T1": [3,  0, -1,  1, -1],
            "T2": [3,  0, -1, -1,  1]
        },
        "trans": {"T2": 1},
        "rot": {"T1": 1},
    },
    "Oh": {
        "classes": [("E", 1), ("8C3", 8), ("6C2'", 6), ("6C4", 6), ("3C2", 3),
                    ("i", 1), ("6S4", 6), ("8S6", 8), ("3sigma_h", 3), ("6sigma_d", 6)],
        "irreps": {
            "A1g": [1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            "A2g": [1,  1, -1, -1,  1,  1, -1,  1,  1, -1],
            "Eg":  [2, -1,  0,  0,  2,  2,  0, -1,  2,  0],
            "T1g": [3,  0, -1,  1, -1,  3,  1,  0, -1, -1],
            "T2g": [3,  0,  1, -1, -1,  3, -1,  0, -1,  1],
            "A1u": [1,  1,  1,  1,  1, -1, -1, -1, -1, -1],
            "A2u": [1,  1, -1, -1,  1, -1,  1, -1, -1,  1],
            "Eu":  [2, -1,  0,  0,  2, -2,  0,  1, -2,  0],
            "T1u": [3,  0, -1,  1, -1, -3, -1,  0,  1,  1],
            "T2u": [3,  0,  1, -1, -1, -3,  1,  0,  1, -1]
        },
        "trans": {"T1u": 1},
        "rot": {"T1g": 1},
    }
}


def _rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix around unit axis."""
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1.0 - c
    return np.array([
        [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c  ]
    ], dtype=np.float64)


def _reflection_matrix(normal: np.ndarray) -> np.ndarray:
    """Reflection across plane passing through origin with given unit normal."""
    n = normal / np.linalg.norm(normal)
    return np.eye(3) - 2.0 * np.outer(n, n)


def _improper_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Improper rotation Sn = reflection perp to axis * rotation by angle."""
    rot = _rotation_matrix(axis, angle_rad)
    ref = _reflection_matrix(axis)
    return ref @ rot


def _center_molecule(symbols: List[str], coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Center molecule at center of nuclear mass."""
    masses = []
    for s in symbols:
        norm = normalize_symbol(s)
        m = ELEMENT_DATA.get(norm, {}).get("mass", 12.0)
        masses.append(max(1.0, m))
    masses = np.array(masses, dtype=np.float64)
    total_mass = np.sum(masses)
    com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    return coords - com, com


def _calculate_inertia_tensor(symbols: List[str], centered_coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate principal moments of inertia and principal axes."""
    I = np.zeros((3, 3), dtype=np.float64)
    for s, (x, y, z) in zip(symbols, centered_coords):
        m = ELEMENT_DATA.get(normalize_symbol(s), {}).get("mass", 12.0)
        I[0, 0] += m * (y*y + z*z)
        I[1, 1] += m * (x*x + z*z)
        I[2, 2] += m * (x*x + y*y)
        I[0, 1] -= m * (x * y)
        I[0, 2] -= m * (x * z)
        I[1, 2] -= m * (y * z)
    I[1, 0] = I[0, 1]
    I[2, 0] = I[0, 2]
    I[2, 1] = I[1, 2]

    evals, evecs = np.linalg.eigh(I)
    idx = np.argsort(evals)
    evals = evals[idx]
    evecs = evecs[:, idx]
    return evals, evecs


def _test_operation(
    centered_coords: np.ndarray,
    symbols: List[str],
    transform_matrix: np.ndarray,
    tolerance: float = 0.15
) -> bool:
    """Test if a symmetry operation maps every atom to an identical element within tolerance."""
    n = len(symbols)
    transformed = (transform_matrix @ centered_coords.T).T

    elem_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(symbols):
        elem_indices.setdefault(s, []).append(i)

    # Pre-extract coordinate subsets per element for fast vectorized matching
    elem_coords = {s: centered_coords[indices] for s, indices in elem_indices.items()}

    used = set()
    tol_sq = tolerance * tolerance
    for i in range(n):
        s_i = symbols[i]
        pos_t = transformed[i]
        candidates = elem_indices[s_i]
        sub_c = elem_coords[s_i]
        diffs = sub_c - pos_t
        dists_sq = np.sum(diffs * diffs, axis=1)

        # Check in order of increasing distance
        matched = False
        sorted_local_indices = np.argsort(dists_sq)
        for loc_idx in sorted_local_indices:
            if dists_sq[loc_idx] > tol_sq:
                break
            j = candidates[loc_idx]
            if j not in used:
                used.add(j)
                matched = True
                break
        if not matched:
            return False

    return len(used) == n


def _refine_operation_matrix(
    centered_coords: np.ndarray,
    symbols: List[str],
    op: SymmetryOperation,
) -> SymmetryOperation:
    """
    Snap symmetry axes/normals that are within ~0.5 degrees of a Cartesian axis or 45-degree bisector
    to exact analytical directions so that aligned molecules yield textbook-exact 3x3 matrices
    (e.g. exact 0.000000 and +/-1.000000 entries).
    """
    if op.axis is not None:
        ax = np.asarray(op.axis, dtype=np.float64)
        norm = float(np.linalg.norm(ax))
        if norm > 1e-8:
            ax = ax / norm
            # Check canonical Cartesian axes and 45-deg bisectors
            s2 = 1.0 / math.sqrt(2.0)
            s3 = 1.0 / math.sqrt(3.0)
            canonical_dirs = [
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([s2, s2, 0.0]),
                np.array([s2, -s2, 0.0]),
                np.array([s2, 0.0, s2]),
                np.array([s2, 0.0, -s2]),
                np.array([0.0, s2, s2]),
                np.array([0.0, s2, -s2]),
                np.array([0.5, math.sqrt(3.0) / 2.0, 0.0]),
                np.array([0.5, -math.sqrt(3.0) / 2.0, 0.0]),
                np.array([math.sqrt(3.0) / 2.0, 0.5, 0.0]),
                np.array([math.sqrt(3.0) / 2.0, -0.5, 0.0]),
                np.array([s3, s3, s3]),
                np.array([s3, s3, -s3]),
                np.array([s3, -s3, s3]),
                np.array([-s3, s3, s3]),
            ]
            for c_dir in canonical_dirs:
                dot = float(np.dot(ax, c_dir))
                if abs(dot) >= math.cos(math.radians(0.8)):
                    snapped = c_dir if dot > 0 else -c_dir
                    if op.op_type == "sigma":
                        cand_mat = _reflection_matrix(snapped)
                    elif op.op_type == "Cn" and op.order >= 2:
                        # Find power k from trace or label
                        k_pow = 1
                        if "^" in op.label:
                            try:
                                k_pow = int(op.label.split("^")[1])
                            except ValueError:
                                k_pow = 1
                        cand_mat = _rotation_matrix(snapped, k_pow * 2.0 * math.pi / op.order)
                        if not np.allclose(cand_mat, op.matrix, atol=0.05):
                            cand_mat = _rotation_matrix(-snapped, k_pow * 2.0 * math.pi / op.order)
                    elif op.op_type == "Sn" and op.order >= 2:
                        k_pow = 1
                        if "^" in op.label:
                            try:
                                k_pow = int(op.label.split("^")[1])
                            except ValueError:
                                k_pow = 1
                        c_k = _rotation_matrix(snapped, k_pow * 2.0 * math.pi / op.order)
                        cand_mat = (_reflection_matrix(snapped) @ c_k) if (k_pow % 2 == 1) else c_k
                        if not np.allclose(cand_mat, op.matrix, atol=0.05):
                            c_k = _rotation_matrix(-snapped, k_pow * 2.0 * math.pi / op.order)
                            cand_mat = (_reflection_matrix(-snapped) @ c_k) if (k_pow % 2 == 1) else c_k
                    else:
                        cand_mat = op.matrix

                    if np.allclose(cand_mat, op.matrix, atol=0.05):
                        op.axis = snapped
                        op.matrix = np.where(np.abs(cand_mat) < 1e-12, 0.0, cand_mat)
                    break

    op.matrix = np.where(np.abs(op.matrix) < 1e-12, 0.0, op.matrix)
    return op


def _deduplicate_axes_vectorized(cands: List[np.ndarray], tol_deg: float = 2.0) -> List[np.ndarray]:
    """Fast vectorized deduplication of 3D unit direction axes."""
    if not cands:
        return []
    cos_thresh = math.cos(math.radians(tol_deg))
    u_arr = np.empty((len(cands), 3), dtype=np.float64)
    n_u = 0
    for c in cands:
        norm = math.sqrt(float(c[0]*c[0] + c[1]*c[1] + c[2]*c[2]))
        if norm < 0.05:
            continue
        c_unit = np.array([c[0] / norm, c[1] / norm, c[2] / norm], dtype=np.float64)
        if n_u == 0:
            u_arr[0] = c_unit
            n_u = 1
        else:
            dots = np.abs(np.dot(u_arr[:n_u], c_unit))
            if np.max(dots) < cos_thresh:
                u_arr[n_u] = c_unit
                n_u += 1
    return [u_arr[k] for k in range(n_u)]


def _are_axes_parallel(a1: np.ndarray, a2: np.ndarray, tol_deg: float = 2.0) -> bool:
    norm1 = math.sqrt(float(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]))
    norm2 = math.sqrt(float(a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2]))
    if norm1 < 1e-12 or norm2 < 1e-12:
        return False
    dot_val = float(a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2])
    cos_th = abs(dot_val / (norm1 * norm2))
    return cos_th >= math.cos(math.radians(tol_deg))


def _are_axes_perpendicular(a1: np.ndarray, a2: np.ndarray, tol_deg: float = 2.0) -> bool:
    norm1 = math.sqrt(float(a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2]))
    norm2 = math.sqrt(float(a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2]))
    if norm1 < 1e-12 or norm2 < 1e-12:
        return False
    dot_val = float(a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2])
    cos_th = abs(dot_val / (norm1 * norm2))
    return cos_th <= math.sin(math.radians(tol_deg))


def find_symmetry_operations(
    symbols: List[str],
    coords: np.ndarray,
    tolerance: float = 0.15
) -> PointGroupResult:
    """
    Analyze molecular 3D geometry and detect its Schoenflies point group symmetry.
    Supports C1, Cs, Ci, Cn, Cnv, Cnh, Dn, Dnd, Dnh, Td, Oh, Ih, C*v, D*h.
    """
    n = len(symbols)
    if n == 0:
        return PointGroupResult("C1", [], tolerance=tolerance)
    if n == 1:
        return PointGroupResult("Kh", [SymmetryOperation("E", np.eye(3), "E")], tolerance=tolerance)

    centered, _ = _center_molecule(symbols, coords)
    evals, evecs = _calculate_inertia_tensor(symbols, centered)

    # 1. Identity
    operations: List[SymmetryOperation] = [
        SymmetryOperation("E", np.eye(3), "E", order=1)
    ]

    # 2. Check linear molecule
    is_linear = (evals[0] < 1e-4) or (evals[2] > 1e-3 and evals[0] / evals[2] < 1e-3)

    # 3. Check Inversion center
    has_inversion = _test_operation(centered, symbols, -np.eye(3), tolerance=tolerance)
    if has_inversion:
        operations.append(SymmetryOperation("i", -np.eye(3), "i", order=2))

    if is_linear:
        symbol = "D*h" if has_inversion else "C*v"
        return PointGroupResult(symbol, operations, has_inversion=has_inversion, linear=True, tolerance=tolerance)

    # 4. Generate candidate symmetry axes
    candidate_axes: List[np.ndarray] = []
    for col in range(3):
        ax = evecs[:, col]
        norm = np.linalg.norm(ax)
        if norm > 1e-4:
            candidate_axes.append(ax / norm)

    # Include face and body diagonals of principal inertia axes (crucial for cubic Oh, Td, Th, O groups)
    for sx in (-1.0, 0.0, 1.0):
        for sy in (-1.0, 0.0, 1.0):
            for sz in (-1.0, 0.0, 1.0):
                if (sx != 0.0) + (sy != 0.0) + (sz != 0.0) >= 2:
                    diag_ax = sx * evecs[:, 0] + sy * evecs[:, 1] + sz * evecs[:, 2]
                    norm_diag = np.linalg.norm(diag_ax)
                    if norm_diag > 1e-4:
                        candidate_axes.append(diag_ax / norm_diag)

    for pos in centered:
        norm = np.linalg.norm(pos)
        if norm > 0.2:
            candidate_axes.append(pos / norm)

    lim = min(n, 14)
    for i in range(lim):
        for j in range(i + 1, lim):
            mid = centered[i] + centered[j]
            norm_m = np.linalg.norm(mid)
            if norm_m > 0.2:
                candidate_axes.append(mid / norm_m)
            diff = centered[i] - centered[j]
            norm_d = np.linalg.norm(diff)
            if norm_d > 0.2:
                candidate_axes.append(diff / norm_d)
            cross = np.cross(centered[i], centered[j])
            norm_c = np.linalg.norm(cross)
            if norm_c > 0.2:
                candidate_axes.append(cross / norm_c)
            # 3-atom sums for triangular face centers (e.g. C3 axes of octahedra/tetrahedra/icosahedra)
            if j < min(n, 10):
                for k in range(j + 1, min(n, 10)):
                    tri = centered[i] + centered[j] + centered[k]
                    norm_t = np.linalg.norm(tri)
                    if norm_t > 0.2:
                        candidate_axes.append(tri / norm_t)

    unique_axes = _deduplicate_axes_vectorized(candidate_axes)

    # 5. Search for Rotational axes Cn
    found_rotations: List[Tuple[np.ndarray, int]] = []
    max_n = 1
    principal_axis = None

    for ax in unique_axes:
        for order in [8, 6, 5, 4, 3, 2]:
            ang = 2.0 * math.pi / order
            mat = _rotation_matrix(ax, ang)
            if _test_operation(centered, symbols, mat, tolerance=tolerance):
                found_rotations.append((ax, order))
                if order > max_n:
                    max_n = order
                    principal_axis = ax
                break

    # Assemble initial operations (E, i, principal Cn)
    if principal_axis is not None and max_n >= 2:
        ang = 2.0 * math.pi / max_n
        for k in range(1, max_n):
            mat_k = _rotation_matrix(principal_axis, k * ang)
            lbl = f"C{max_n}" if k == 1 else f"C{max_n}^{k}"
            operations.append(SymmetryOperation("Cn", mat_k, lbl, order=max_n, axis=principal_axis))

    # Detect perpendicular C2 axes if principal axis exists
    perp_c2_axes: List[np.ndarray] = []
    if principal_axis is not None and max_n >= 2:
        c2_perp_candidates = []
        for i in range(n):
            v = centered[i] - np.dot(centered[i], principal_axis) * principal_axis
            if np.linalg.norm(v) > 0.05:
                u = v / np.linalg.norm(v)
                c2_perp_candidates.append(u)
                c2_perp_candidates.append(np.cross(principal_axis, u))
            for j in range(i + 1, n):
                mid = (centered[i] + centered[j]) / 2.0
                v_mid = mid - np.dot(mid, principal_axis) * principal_axis
                if np.linalg.norm(v_mid) > 0.05:
                    u_mid = v_mid / np.linalg.norm(v_mid)
                    c2_perp_candidates.append(u_mid)
                    c2_perp_candidates.append(np.cross(principal_axis, u_mid))
                diff = centered[i] - centered[j]
                v_diff = diff - np.dot(diff, principal_axis) * principal_axis
                if np.linalg.norm(v_diff) > 0.05:
                    u_diff = v_diff / np.linalg.norm(v_diff)
                    c2_perp_candidates.append(u_diff)
                    c2_perp_candidates.append(np.cross(principal_axis, u_diff))

        unique_perp = []
        for c in c2_perp_candidates:
            if not any(_are_axes_parallel(c, u) for u in unique_perp):
                unique_perp.append(c)

        for ax in unique_perp:
            mat_c2 = _rotation_matrix(ax, math.pi)
            if _test_operation(centered, symbols, mat_c2, tolerance=tolerance):
                if not any(_are_axes_parallel(ax, u) for u in perp_c2_axes):
                    perp_c2_axes.append(ax)
                    operations.append(SymmetryOperation("Cn", mat_c2, "C2'", order=2, axis=ax))
    perp_c2_count = len(perp_c2_axes)

    # 6. Check for cubic symmetry (Td, Oh, Ih)
    c3_count = sum(1 for ax, order in found_rotations if order == 3)
    c4_count = sum(1 for ax, order in found_rotations if order == 4)
    c5_count = sum(1 for ax, order in found_rotations if order == 5)
    cubic_symbol = None
    if c5_count >= 6:
        cubic_symbol = "Ih" if has_inversion else "I"
    elif c3_count >= 4:
        if c4_count >= 3:
            cubic_symbol = "Oh" if has_inversion else "O"
        else:
            cubic_symbol = "Th" if has_inversion else "Td"

    # For cubic groups, assemble all unique proper rotations (C5, C4, C3, C2, C2') across all axes
    if cubic_symbol is not None:
        # Rebuild rotational operations cleanly so C2 (collinear with C4) and C2' (diagonal) are labeled accurately
        operations = [op for op in operations if op.op_type in ("E", "i")]
        c4_axes = [ax for ax, order in found_rotations if order == 4]
        for ax, max_ord in found_rotations:
            for ord_sub in [max_ord, 2] if max_ord == 4 else [max_ord]:
                ang_sub = 2.0 * math.pi / ord_sub
                for k in range(1, ord_sub):
                    if ord_sub == 4 and k == 2:
                        continue  # Handled by ord_sub == 2
                    mat_k = _rotation_matrix(ax, k * ang_sub)
                    if not any(np.allclose(mat_k, op.matrix, atol=1e-3) for op in operations):
                        if ord_sub == 2:
                            is_collinear_c4 = any(_are_axes_parallel(ax, c4_ax) for c4_ax in c4_axes)
                            lbl = "C2" if (is_collinear_c4 or not c4_axes) else "C2'"
                        else:
                            lbl = f"C{ord_sub}" if k == 1 else f"C{ord_sub}^{k}"
                        operations.append(SymmetryOperation("Cn", mat_k, lbl, order=ord_sub, axis=ax))

    # 7. Search for Reflection planes sigma
    sigma_planes: List[np.ndarray] = []
    plane_candidates: List[np.ndarray] = list(unique_axes)
    if principal_axis is not None:
        plane_candidates.append(principal_axis)
        plane_candidates.extend(perp_c2_axes)
        # Vertical planes containing principal axis and an atom: normal is perp to both
        for i in range(n):
            v_perp = centered[i] - np.dot(centered[i], principal_axis) * principal_axis
            if np.linalg.norm(v_perp) > 0.05:
                n_vert = np.cross(principal_axis, v_perp)
                if np.linalg.norm(n_vert) > 0.05:
                    plane_candidates.append(n_vert / np.linalg.norm(n_vert))

    for i in range(n):
        for j in range(i + 1, n):
            if symbols[i] == symbols[j]:
                diff = centered[i] - centered[j]
                if np.linalg.norm(diff) > 0.05:
                    plane_candidates.append(diff / np.linalg.norm(diff))

    unique_plane_cands = _deduplicate_axes_vectorized(plane_candidates)

    c4_axes_cubic = [ax for ax, order in found_rotations if order == 4] if cubic_symbol is not None else []
    for normal in unique_plane_cands:
        mat_ref = _reflection_matrix(normal)
        if _test_operation(centered, symbols, mat_ref, tolerance=tolerance):
            if not any(_are_axes_parallel(normal, p) for p in sigma_planes):
                sigma_planes.append(normal)
                if not any(np.allclose(mat_ref, op.matrix, atol=1e-3) for op in operations):
                    if cubic_symbol is not None:
                        is_sigma_h = any(_are_axes_parallel(normal, c4_ax) for c4_ax in c4_axes_cubic)
                        sig_lbl = "sigma_h" if is_sigma_h else "sigma_d"
                    else:
                        sig_lbl = "sigma"
                    operations.append(SymmetryOperation("sigma", mat_ref, sig_lbl, axis=normal))

    # Improper rotations Sn
    if cubic_symbol is not None:
        # Check improper rotations along all rotational axes (e.g. 6 S4 and 8 S6 in Oh, 6 S4 in Td)
        for ax, max_ord in found_rotations:
            sn_candidates = []
            if max_ord == 4 or (max_ord == 2 and cubic_symbol == "Td"):
                sn_candidates.append(4)
            if max_ord == 3 and has_inversion:
                sn_candidates.append(6)
            if max_ord == 5 and has_inversion:
                sn_candidates.append(10)
            for sn_order in sn_candidates:
                for k in range(1, sn_order):
                    if k % 2 == 0:
                        continue
                    c_k = _rotation_matrix(ax, k * 2.0 * math.pi / sn_order)
                    mat_sn = _reflection_matrix(ax) @ c_k
                    if _test_operation(centered, symbols, mat_sn, tolerance=tolerance):
                        is_e = np.allclose(mat_sn, np.eye(3), atol=1e-3)
                        is_i = np.allclose(mat_sn, -np.eye(3), atol=1e-3)
                        is_rot = np.isclose(np.linalg.det(mat_sn), 1.0)
                        is_ref = np.isclose(np.linalg.det(mat_sn), -1.0) and np.isclose(np.trace(mat_sn), 1.0)
                        if not (is_e or is_i or is_rot or is_ref):
                            if not any(np.allclose(mat_sn, op.matrix, atol=1e-3) for op in operations):
                                lbl_sn = f"S{sn_order}" if k == 1 else f"S{sn_order}^{k}"
                                operations.append(SymmetryOperation("Sn", mat_sn, lbl_sn, order=sn_order, axis=ax))
    elif principal_axis is not None and max_n >= 2:
        for sn_order in [6, 4, 3]:
            if max_n % sn_order == 0 or (2 * max_n) % sn_order == 0:
                max_k = 2 * sn_order if sn_order % 2 != 0 else sn_order
                for k in range(1, max_k):
                    c_k = _rotation_matrix(principal_axis, k * 2.0 * math.pi / sn_order)
                    mat_sn = (_reflection_matrix(principal_axis) @ c_k) if (k % 2 == 1) else c_k
                    if _test_operation(centered, symbols, mat_sn, tolerance=tolerance):
                        is_e = np.allclose(mat_sn, np.eye(3), atol=1e-3)
                        is_i = np.allclose(mat_sn, -np.eye(3), atol=1e-3)
                        is_rot = np.isclose(np.linalg.det(mat_sn), 1.0)
                        is_ref = np.isclose(np.linalg.det(mat_sn), -1.0) and np.isclose(np.trace(mat_sn), 1.0)
                        if not (is_e or is_i or is_rot or is_ref):
                            if not any(np.allclose(mat_sn, op.matrix, atol=1e-3) for op in operations):
                                operations.append(SymmetryOperation("Sn", mat_sn, f"S{sn_order}^{k}", order=sn_order, axis=principal_axis))

    # Clean near-zero entries and snap axes that are aligned with Cartesian/canonical axes
    for op in operations:
        _refine_operation_matrix(centered, symbols, op)
    if principal_axis is not None:
        for c_ax in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])):
            d_p = float(np.dot(principal_axis / np.linalg.norm(principal_axis), c_ax))
            if abs(d_p) >= math.cos(math.radians(0.8)):
                principal_axis = c_ax if d_p > 0 else -c_ax
                break

    if cubic_symbol is not None:
        h_planes_cnt = sum(1 for op in operations if op.op_type == "sigma" and op.label == "sigma_h")
        v_planes_cnt = sum(1 for op in operations if op.op_type == "sigma" and op.label != "sigma_h")
        v_planes_cubic = [op.axis for op in operations if op.op_type == "sigma" and op.axis is not None]
        return PointGroupResult(
            symbol=cubic_symbol,
            operations=operations,
            principal_axis=principal_axis,
            horizontal_plane=principal_axis if h_planes_cnt > 0 else None,
            vertical_planes=v_planes_cubic,
            has_inversion=has_inversion,
            horizontal_planes_count=h_planes_cnt,
            vertical_planes_count=v_planes_cnt,
            tolerance=tolerance,
        )

    # 8. Classify Point Group
    if max_n == 1:
        if len(sigma_planes) > 0:
            return PointGroupResult("Cs", operations, horizontal_plane=sigma_planes[0], has_inversion=has_inversion, vertical_planes_count=len(sigma_planes), tolerance=tolerance)
        if has_inversion:
            return PointGroupResult("Ci", operations, has_inversion=True, tolerance=tolerance)
        return PointGroupResult("C1", operations, tolerance=tolerance)

    has_perpendicular_c2 = (perp_c2_count >= max_n) or (max_n == 2 and perp_c2_count >= 1)

    has_sigma_h = False
    horizontal_plane_norm = None
    v_planes = []

    for normal in sigma_planes:
        if principal_axis is not None and _are_axes_parallel(normal, principal_axis):
            has_sigma_h = True
            horizontal_plane_norm = normal
        else:
            if principal_axis is not None and _are_axes_perpendicular(normal, principal_axis):
                v_planes.append(normal)

    # Check for improper rotation S_2n collinear with Cn
    has_s_2n = False
    if principal_axis is not None:
        mat_s2n = _improper_rotation_matrix(principal_axis, math.pi / max_n)
        has_s_2n = _test_operation(centered, symbols, mat_s2n, tolerance=tolerance)

    if has_perpendicular_c2:
        if has_sigma_h:
            symbol = f"D{max_n}h"
        elif len(v_planes) >= max_n or has_s_2n:
            symbol = f"D{max_n}d"
        else:
            symbol = f"D{max_n}"
    else:
        if has_sigma_h:
            symbol = f"C{max_n}h"
        elif len(v_planes) >= max_n or len(sigma_planes) >= max_n:
            symbol = f"C{max_n}v"
        elif has_s_2n:
            symbol = f"S{2 * max_n}"
        else:
            symbol = f"C{max_n}"

    return PointGroupResult(
        symbol=symbol,
        operations=operations,
        principal_axis=principal_axis,
        horizontal_plane=horizontal_plane_norm,
        vertical_planes=v_planes,
        has_inversion=has_inversion,
        horizontal_planes_count=1 if has_sigma_h else 0,
        vertical_planes_count=len(v_planes),
        tolerance=tolerance
    )


# -------------------------------------------------------------
# 1. Option: Align Molecule to Standard Orientation (Cn along Z, plane in XZ/XY)
# -------------------------------------------------------------
def align_molecule_to_symmetry(
    symbols: List[str],
    coords: np.ndarray,
    result: Optional[PointGroupResult] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Standardize molecular orientation according to IUPAC / quantum chemistry convention:
    - Centers COM at origin (0, 0, 0).
    - Aligns principal symmetry axis (highest Cn) along the +Z axis.
    - If a vertical mirror plane sigma_v exists, aligns it with the XZ plane (normal along Y).
    - If a horizontal mirror plane sigma_h exists, aligns it with the XY plane (normal along Z).
    Returns: (aligned_coords, rotation_matrix)
    """
    n = len(symbols)
    if n == 0:
        return coords, np.eye(3)

    centered, com = _center_molecule(symbols, coords)
    if result is None:
        result = find_symmetry_operations(symbols, coords, tolerance=0.20)

    if result is None or result.symbol in ("C1", "Ci"):
        # Align by principal axes of inertia
        _, evecs = _calculate_inertia_tensor(symbols, centered)
        # Ensure right-handed coordinate system
        if np.linalg.det(evecs) < 0:
            evecs[:, 2] = -evecs[:, 2]
        aligned = centered @ evecs
        return aligned, evecs.T

    p_axis = result.principal_axis
    if p_axis is None:
        if result.horizontal_plane is not None:
            p_axis = result.horizontal_plane
        else:
            _, evecs = _calculate_inertia_tensor(symbols, centered)
            if np.linalg.det(evecs) < 0:
                evecs[:, 2] = -evecs[:, 2]
            return centered @ evecs, evecs.T

    # 1. Rotate principal axis to Z-axis: [0, 0, 1]
    z_target = np.array([0.0, 0.0, 1.0])
    p_norm = p_axis / np.linalg.norm(p_axis)
    cross_pz = np.cross(p_norm, z_target)
    dot_pz = float(np.dot(p_norm, z_target))

    if np.linalg.norm(cross_pz) < 1e-6:
        R_z = np.eye(3) if dot_pz > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        rot_axis = cross_pz / np.linalg.norm(cross_pz)
        rot_angle = math.acos(max(-1.0, min(1.0, dot_pz)))
        R_z = _rotation_matrix(rot_axis, rot_angle)

    rotated = (R_z @ centered.T).T
    total_R = R_z.copy()

    # 2. In-plane orientation around Z:
    # If vertical mirror planes exist, find the plane that contains the most atoms (molecular plane)
    # and orient its normal along the Y-axis so the molecular plane is exactly the XZ plane (screen plane).
    if result.vertical_planes:
        valid_planes = []
        for vp in result.vertical_planes:
            v_norm_rot = R_z @ vp
            v_norm_xy = np.array([v_norm_rot[0], v_norm_rot[1], 0.0])
            norm_len = np.linalg.norm(v_norm_xy)
            if norm_len > 1e-4:
                v_norm_xy /= norm_len
                # Calculate mean squared distance of atoms to this mirror plane (dot product with normal)
                sq_dists = [abs(float(np.dot(p, v_norm_xy)))**2 for p in rotated]
                valid_planes.append((float(np.mean(sq_dists)), v_norm_xy))

        if valid_planes:
            valid_planes.sort(key=lambda item: item[0])
            best_norm_xy = valid_planes[0][1]
            y_target = np.array([0.0, 1.0, 0.0])
            ang_y = math.atan2(float(np.cross(best_norm_xy, y_target)[2]), float(np.dot(best_norm_xy, y_target)))
            R_phi = _rotation_matrix(z_target, ang_y)
            rotated = (R_phi @ rotated.T).T
            total_R = R_phi @ total_R
    else:
        # Align secondary inertia axis in XY plane along X
        I_xy = np.zeros((2, 2))
        for s, p in zip(symbols, rotated):
            m = ELEMENT_DATA.get(normalize_symbol(s), {}).get("mass", 12.0)
            I_xy[0, 0] += m * p[1]**2
            I_xy[1, 1] += m * p[0]**2
            I_xy[0, 1] -= m * p[0] * p[1]
        I_xy[1, 0] = I_xy[0, 1]
        _, evecs_xy = np.linalg.eigh(I_xy)
        ang_xy = math.atan2(evecs_xy[1, 0], evecs_xy[0, 0])
        R_phi = _rotation_matrix(z_target, -ang_xy)
        rotated = (R_phi @ rotated.T).T
        total_R = R_phi @ total_R

    return rotated, total_R


# -------------------------------------------------------------
# 2. Option: Enforce Symmetry by Group Projection Operator & Symmetry Orbits
# -------------------------------------------------------------
def _close_matrix_group(generators: List[SymmetryOperation], max_size: int = 120) -> List[SymmetryOperation]:
    """
    Generate the finite 3x3 orthogonal matrix group closed under multiplication from `generators`.
    Labels newly generated elements according to their trace, determinant, and rotation/reflection axis.
    """
    group: List[SymmetryOperation] = [SymmetryOperation("E", np.eye(3), "E", order=1)]
    for g in generators:
        if not any(np.allclose(g.matrix, ex.matrix, atol=1e-3) for ex in group):
            group.append(g)

    idx = 0
    while idx < len(group) and len(group) < max_size:
        g_i = group[idx]
        for g_j in list(group):
            prod = g_i.matrix @ g_j.matrix
            prod = np.where(np.abs(prod) < 1e-12, 0.0, prod)
            if not any(np.allclose(prod, ex.matrix, atol=1e-3) for ex in group):
                det = float(np.linalg.det(prod))
                tr = float(np.trace(prod))
                if np.allclose(prod, np.eye(3), atol=1e-3):
                    op = SymmetryOperation("E", np.eye(3), "E", order=1)
                elif np.allclose(prod, -np.eye(3), atol=1e-3):
                    op = SymmetryOperation("i", -np.eye(3), "i", order=2)
                elif det > 0.0:
                    # Proper rotation: tr = 1 + 2 cos(theta)
                    cos_th = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
                    th = math.acos(cos_th)
                    ord_guess = max(2, int(round(2.0 * math.pi / th))) if th > 1e-3 else 2
                    # Axis from skew-symmetric part or eigenvector with eigenvalue +1
                    vals, vecs = np.linalg.eig(prod)
                    ax_idx = int(np.argmin(np.abs(vals - 1.0)))
                    ax = np.real(vecs[:, ax_idx])
                    if np.linalg.norm(ax) > 1e-6:
                        ax = ax / np.linalg.norm(ax)
                    op = SymmetryOperation("Cn", prod, f"C{ord_guess}", order=ord_guess, axis=ax)
                else:
                    # Improper operation: reflection (tr == 1) or Sn
                    if abs(tr - 1.0) < 1e-2:
                        vals, vecs = np.linalg.eig(prod)
                        ax_idx = int(np.argmin(np.abs(vals + 1.0)))
                        ax = np.real(vecs[:, ax_idx])
                        if np.linalg.norm(ax) > 1e-6:
                            ax = ax / np.linalg.norm(ax)
                        op = SymmetryOperation("sigma", prod, "sigma", order=2, axis=ax)
                    else:
                        # Sn: prod = sigma_h * C_n => tr = -1 + 2 cos(theta)
                        cos_th = max(-1.0, min(1.0, (tr + 1.0) * 0.5))
                        th = math.acos(cos_th)
                        ord_guess = max(3, int(round(2.0 * math.pi / th))) if th > 1e-3 else 4
                        vals, vecs = np.linalg.eig(prod)
                        ax_idx = int(np.argmin(np.abs(vals + 1.0)))
                        ax = np.real(vecs[:, ax_idx])
                        if np.linalg.norm(ax) > 1e-6:
                            ax = ax / np.linalg.norm(ax)
                        op = SymmetryOperation("Sn", prod, f"S{ord_guess}", order=ord_guess, axis=ax)
                group.append(op)
                if len(group) >= max_size:
                    break
        idx += 1
    return group


def _extract_subgroup_from_parent(
    target_symbol: str,
    parent_res: PointGroupResult,
) -> Optional[PointGroupResult]:
    """
    Construct a valid `PointGroupResult` for `target_symbol` as a subgroup of `parent_res`.
    Returns None if `parent_res` does not contain the required symmetry elements for `target_symbol`.
    """
    if target_symbol == parent_res.symbol:
        return parent_res

    ops = parent_res.operations
    tol = parent_res.tolerance
    if target_symbol == "C1":
        return PointGroupResult("C1", [SymmetryOperation("E", np.eye(3), "E", order=1)], tolerance=tol)

    if target_symbol == "Ci":
        inv_ops = [op for op in ops if op.op_type == "i"]
        if not inv_ops:
            return None
        return PointGroupResult(
            "Ci",
            [SymmetryOperation("E", np.eye(3), "E", order=1), inv_ops[0]],
            has_inversion=True,
            tolerance=tol,
        )

    if target_symbol == "Cs":
        sig_ops = [op for op in ops if op.op_type == "sigma"]
        if not sig_ops:
            return None
        # Prefer horizontal plane if present, otherwise first mirror plane
        chosen_sig = next((op for op in sig_ops if op.label == "sigma_h"), sig_ops[0])
        return PointGroupResult(
            "Cs",
            [SymmetryOperation("E", np.eye(3), "E", order=1), chosen_sig],
            horizontal_plane=chosen_sig.axis,
            horizontal_planes_count=1,
            tolerance=tol,
        )

    # Cubic subgroups: O (order 24), Td (order 24), Th (order 24), T (order 12)
    if target_symbol == "O":
        if parent_res.symbol != "Oh":
            return None
        proper_ops = [op for op in ops if op.op_type in ("E", "Cn")]
        return PointGroupResult(
            "O",
            proper_ops,
            principal_axis=parent_res.principal_axis,
            has_inversion=False,
            tolerance=tol,
        )

    if target_symbol == "Th":
        if parent_res.symbol not in ("Oh", "Th"):
            return None
        # Th has E, 8 C3, 3 C2 (collinear with C4), i, 8 S6, 3 sigma_h (order 24)
        th_ops = [
            op for op in ops
            if op.op_type in ("E", "i")
            or (op.op_type == "Cn" and (op.order == 3 or op.label == "C2"))
            or (op.op_type == "Sn" and op.order == 6)
            or (op.op_type == "sigma" and op.label == "sigma_h")
        ]
        if len(th_ops) == 24:
            return PointGroupResult(
                "Th",
                th_ops,
                principal_axis=parent_res.principal_axis,
                has_inversion=True,
                horizontal_planes_count=3,
                tolerance=tol,
            )
        return None

    if target_symbol == "Td":
        if parent_res.symbol not in ("Oh", "Td"):
            return None
        # Td has E, 8 C3, 3 C2, 6 S4, 6 sigma_d (order 24)
        td_ops = [
            op for op in ops
            if op.op_type == "E"
            or (op.op_type == "Cn" and (op.order == 3 or op.label == "C2"))
            or (op.op_type == "Sn" and op.order == 4)
            or (op.op_type == "sigma" and op.label in ("sigma_d", "sigma"))
        ]
        if len(td_ops) == 24:
            return PointGroupResult(
                "Td",
                td_ops,
                principal_axis=parent_res.principal_axis,
                has_inversion=False,
                vertical_planes_count=6,
                tolerance=tol,
            )
        return None

    if target_symbol == "T":
        if parent_res.symbol not in ("Oh", "O", "Td", "Th"):
            return None
        t_ops = [
            op for op in ops
            if op.op_type == "E"
            or (op.op_type == "Cn" and (op.order == 3 or op.label == "C2"))
        ]
        if len(t_ops) == 12:
            return PointGroupResult(
                "T",
                t_ops,
                principal_axis=parent_res.principal_axis,
                has_inversion=False,
                tolerance=tol,
            )
        return None

    # Parse axial point groups: Cn, Cnv, Cnh, Dn, Dnh, Dnd, S2n
    import re
    m = re.match(r"^(C|D|S)(\d+)(v|h|d)?$", target_symbol)
    if not m:
        return None
    family, n_str, suffix = m.group(1), m.group(2), m.group(3) or ""
    n_ord = int(n_str)

    if family == "S":
        if n_ord % 2 != 0:
            return None
        sn_ops = [op for op in ops if op.op_type == "Sn" and op.order == n_ord and op.axis is not None]
        if not sn_ops:
            return None
        p_ax = sn_ops[0].axis
        gens = []
        for k in range(1, n_ord):
            c_k = _rotation_matrix(p_ax, k * 2.0 * math.pi / n_ord)
            mat_sn = (_reflection_matrix(p_ax) @ c_k) if (k % 2 == 1) else c_k
            if any(np.allclose(mat_sn, op.matrix, atol=1e-2) for op in ops):
                lbl = f"S{n_ord}^{k}" if k % 2 == 1 else f"C{n_ord // 2}^{k // 2}"
                op_t = "Sn" if k % 2 == 1 else "Cn"
                gens.append(SymmetryOperation(op_t, mat_sn, lbl, order=n_ord if k % 2 == 1 else n_ord // 2, axis=p_ax))
        closed = _close_matrix_group(gens, max_size=n_ord)
        if len(closed) == n_ord:
            return PointGroupResult(
                target_symbol,
                closed,
                principal_axis=p_ax,
                has_inversion=any(op.op_type == "i" for op in closed),
                tolerance=tol,
            )
        return None

    # Find candidate principal axes of order n_ord (or a multiple of n_ord) in parent_res
    cand_p_axes: List[np.ndarray] = []
    if parent_res.principal_axis is not None:
        mat_test = _rotation_matrix(parent_res.principal_axis, 2.0 * math.pi / n_ord)
        if any(np.allclose(mat_test, op.matrix, atol=1e-2) for op in ops):
            cand_p_axes.append(parent_res.principal_axis)

    for op in ops:
        if op.op_type == "Cn" and op.axis is not None and op.order >= n_ord and (op.order % n_ord == 0):
            if not any(_are_axes_parallel(op.axis, ax) for ax in cand_p_axes):
                cand_p_axes.append(op.axis)

    for p_ax in cand_p_axes:
        # 1. Cn rotations along p_ax
        cn_ops: List[SymmetryOperation] = []
        valid_cn = True
        for k in range(1, n_ord):
            mat_k = _rotation_matrix(p_ax, k * 2.0 * math.pi / n_ord)
            matched = next((op for op in ops if np.allclose(mat_k, op.matrix, atol=1e-2)), None)
            if matched is None:
                valid_cn = False
                break
            lbl = f"C{n_ord}" if k == 1 else f"C{n_ord}^{k}"
            cn_ops.append(SymmetryOperation("Cn", matched.matrix, lbl, order=n_ord, axis=p_ax))
        if not valid_cn:
            continue

        if family == "C" and suffix == "":
            sub_ops = [SymmetryOperation("E", np.eye(3), "E", order=1)] + cn_ops
            return PointGroupResult(target_symbol, sub_ops, principal_axis=p_ax, tolerance=tol)

        # Find perpendicular C2 axes and mirror planes relative to p_ax
        perp_c2s: List[SymmetryOperation] = []
        vert_sigmas: List[SymmetryOperation] = []
        horiz_sigma: Optional[SymmetryOperation] = None

        for op in ops:
            if op.op_type == "Cn" and op.order == 2 and op.axis is not None:
                if _are_axes_perpendicular(op.axis, p_ax):
                    if not any(_are_axes_parallel(op.axis, existing.axis) for existing in perp_c2s):
                        perp_c2s.append(SymmetryOperation("Cn", op.matrix, "C2'", order=2, axis=op.axis))
            elif op.op_type == "sigma" and op.axis is not None:
                if _are_axes_parallel(op.axis, p_ax):
                    horiz_sigma = SymmetryOperation("sigma", op.matrix, "sigma_h", order=2, axis=p_ax)
                elif _are_axes_perpendicular(op.axis, p_ax):
                    if not any(_are_axes_parallel(op.axis, existing.axis) for existing in vert_sigmas):
                        vert_sigmas.append(SymmetryOperation("sigma", op.matrix, "sigma_v", order=2, axis=op.axis))

        if family == "C" and suffix == "h":
            if horiz_sigma is None:
                continue
            closed = _close_matrix_group(cn_ops + [horiz_sigma], max_size=2 * n_ord)
            if len(closed) == 2 * n_ord:
                return PointGroupResult(
                    target_symbol,
                    closed,
                    principal_axis=p_ax,
                    horizontal_plane=p_ax,
                    has_inversion=any(op.op_type == "i" for op in closed),
                    horizontal_planes_count=1,
                    tolerance=tol,
                )

        if family == "C" and suffix == "v":
            if len(vert_sigmas) < n_ord:
                continue
            # Pick one vertical mirror plane and generate the n_ord planes via Cn
            for vs in vert_sigmas:
                closed = _close_matrix_group(cn_ops + [vs], max_size=2 * n_ord)
                if len(closed) == 2 * n_ord:
                    v_axes = [op.axis for op in closed if op.op_type == "sigma" and op.axis is not None]
                    return PointGroupResult(
                        target_symbol,
                        closed,
                        principal_axis=p_ax,
                        vertical_planes=v_axes,
                        vertical_planes_count=len(v_axes),
                        tolerance=tol,
                    )

        if family == "D":
            if len(perp_c2s) < n_ord:
                continue
            for c2p in perp_c2s:
                dn_closed = _close_matrix_group(cn_ops + [c2p], max_size=2 * n_ord)
                if len(dn_closed) != 2 * n_ord:
                    continue
                if suffix == "":
                    return PointGroupResult(
                        target_symbol,
                        dn_closed,
                        principal_axis=p_ax,
                        tolerance=tol,
                    )
                elif suffix == "h":
                    if horiz_sigma is None:
                        continue
                    dnh_closed = _close_matrix_group(dn_closed + [horiz_sigma], max_size=4 * n_ord)
                    if len(dnh_closed) == 4 * n_ord:
                        v_axes = [
                            op.axis for op in dnh_closed
                            if op.op_type == "sigma" and op.axis is not None and _are_axes_perpendicular(op.axis, p_ax)
                        ]
                        return PointGroupResult(
                            target_symbol,
                            dnh_closed,
                            principal_axis=p_ax,
                            horizontal_plane=p_ax,
                            vertical_planes=v_axes,
                            has_inversion=any(op.op_type == "i" for op in dnh_closed),
                            horizontal_planes_count=1,
                            vertical_planes_count=len(v_axes),
                            tolerance=tol,
                        )
                elif suffix == "d":
                    # Dnd is generated by Dn + a dihedral plane sigma_d bisecting two C2' axes (or S_2n)
                    dn_c2_axes = [
                        op.axis for op in dn_closed
                        if op.op_type == "Cn" and op.order == 2 and op.axis is not None and _are_axes_perpendicular(op.axis, p_ax)
                    ]
                    for vs in vert_sigmas:
                        # sigma_d normal should not be parallel or perpendicular to all dn_c2_axes when n_ord >= 2
                        is_collinear = any(_are_axes_parallel(vs.axis, c2a) or _are_axes_perpendicular(vs.axis, c2a) for c2a in dn_c2_axes)
                        if is_collinear and len(vert_sigmas) > n_ord:
                            continue
                        dnd_closed = _close_matrix_group(dn_closed + [vs], max_size=4 * n_ord)
                        if len(dnd_closed) == 4 * n_ord and not any(
                            op.op_type == "sigma" and op.axis is not None and _are_axes_parallel(op.axis, p_ax)
                            for op in dnd_closed
                        ):
                            v_axes = [op.axis for op in dnd_closed if op.op_type == "sigma" and op.axis is not None]
                            return PointGroupResult(
                                target_symbol,
                                dnd_closed,
                                principal_axis=p_ax,
                                vertical_planes=v_axes,
                                has_inversion=any(op.op_type == "i" for op in dnd_closed),
                                vertical_planes_count=len(v_axes),
                                tolerance=tol,
                            )
    return None


# Standard subgroup lattice ordering for common point groups (highest to lowest)
SUBGROUP_CANDIDATES: Dict[str, List[str]] = {
    "Oh": ["Oh", "O", "Td", "Th", "D4h", "D3d", "D2d", "C4v", "C4h", "D4", "D3", "C3v", "D2h", "C2v", "C2h", "D2", "C4", "S4", "C3", "C2", "Cs", "Ci", "C1"],
    "O":  ["O", "T", "D4", "D3", "D2", "C4", "C3", "C2", "C1"],
    "Td": ["Td", "T", "D2d", "C3v", "D2", "C2v", "S4", "C3", "C2", "Cs", "C1"],
    "Th": ["Th", "T", "D2h", "S6", "D2", "C2v", "C2h", "C3", "C2", "Cs", "Ci", "C1"],
    "D6h": ["D6h", "D6", "C6v", "C6h", "D3h", "D3d", "D2h", "C6", "C3v", "C3h", "D3", "C2v", "C2h", "D2", "C3", "C2", "Cs", "Ci", "C1"],
    "D4h": ["D4h", "D4", "C4v", "C4h", "D2d", "D2h", "C4", "S4", "C2v", "C2h", "D2", "C2", "Cs", "Ci", "C1"],
    "D3h": ["D3h", "D3", "C3v", "C3h", "C2v", "C3", "C2", "Cs", "C1"],
    "D3d": ["D3d", "D3", "C3v", "S6", "C2h", "C3", "C2", "Cs", "Ci", "C1"],
    "D2d": ["D2d", "D2", "C2v", "S4", "C2", "Cs", "C1"],
    "D2h": ["D2h", "D2", "C2v", "C2h", "C2", "Cs", "Ci", "C1"],
    "D2":  ["D2", "C2", "C1"],
    "C6v": ["C6v", "C6", "C3v", "C2v", "C3", "C2", "Cs", "C1"],
    "C4v": ["C4v", "C4", "C2v", "C2", "Cs", "C1"],
    "C3v": ["C3v", "C3", "Cs", "C1"],
    "C2v": ["C2v", "C2", "Cs", "C1"],
    "C2h": ["C2h", "C2", "Cs", "Ci", "C1"],
    "C2":  ["C2", "C1"],
    "Cs":  ["Cs", "C1"],
    "Ci":  ["Ci", "C1"],
    "C1":  ["C1"],
}


def _expected_group_order(symbol: str) -> Optional[int]:
    """Return the exact mathematical order |G| of finite Schoenflies point group `symbol`, or None if infinite."""
    fixed = {
        "C1": 1, "Cs": 2, "Ci": 2,
        "T": 12, "Td": 24, "Th": 24, "O": 24, "Oh": 48, "I": 60, "Ih": 120,
    }
    if symbol in fixed:
        return fixed[symbol]
    import re
    m = re.match(r"^(C|D|S)(\d+)(v|h|d)?$", symbol)
    if not m:
        return None
    fam, n_str, suf = m.group(1), m.group(2), m.group(3) or ""
    n = int(n_str)
    if fam == "S":
        return n
    if fam == "C":
        return n if suf == "" else 2 * n
    if fam == "D":
        return 2 * n if suf == "" else 4 * n
    return None


# LRU Cache for find_available_symmetries to avoid heavy redundant multi-tolerance scans
_AVAILABLE_SYMMETRIES_CACHE: Dict[Tuple, List[Tuple[str, PointGroupResult, str]]] = {}
_AVAILABLE_SYMMETRIES_CACHE_MAX = 64


def find_available_symmetries(
    symbols: List[str],
    coords: np.ndarray,
    base_tolerance: float = 0.15,
) -> List[Tuple[str, PointGroupResult, str]]:
    """
    Discover all point group symmetries available for `(symbols, coords)`:
    - Exact / current point group at `base_tolerance` (tagged `'detected'`)
    - Higher-order point groups (supergroups) satisfied at looser tolerances up to 0.85 Å (tagged `'higher'`)
    - Lower-order subgroups of the highest detected group down to C1 (tagged `'lower'`)

    Returns a list of `(group_symbol, PointGroupResult, role_tag)` sorted from highest group order
    to lowest (`C1`).
    """
    n = len(symbols)
    if n == 0:
        c1 = PointGroupResult("C1", [], tolerance=base_tolerance)
        return [("C1", c1, "detected")]

    # Cache lookup using rounded coordinates and symbols
    c_rounded = np.round(coords, 3)
    cache_key = (tuple(symbols), c_rounded.tobytes(), round(base_tolerance, 3))
    if cache_key in _AVAILABLE_SYMMETRIES_CACHE:
        return _AVAILABLE_SYMMETRIES_CACHE[cache_key]

    exact_res = find_symmetry_operations(symbols, coords, tolerance=base_tolerance)
    found_by_symbol: Dict[str, Tuple[PointGroupResult, str]] = {
        exact_res.symbol: (exact_res, "detected")
    }

    # For large molecules (N >= 40), avoid multiple expensive multi-tolerance scans
    # Use exact_res and scan one looser tolerance (0.35) instead of 5 separate scans
    scan_tols = (base_tolerance, 0.35) if n >= 40 else (0.10, base_tolerance, 0.25, 0.40, 0.60, 0.85)

    # Scan looser tolerances to detect higher (approximate / near) symmetries
    highest_res = exact_res
    for try_tol in scan_tols:
        if abs(try_tol - base_tolerance) < 1e-4:
            cand = exact_res
        else:
            cand = find_symmetry_operations(symbols, coords, tolerance=try_tol)
        exp_ord = _expected_group_order(cand.symbol)
        if exp_ord is not None and cand.order != exp_ord:
            # Try extracting exact closed subgroup of cand.symbol if possible
            sub_clean = _extract_subgroup_from_parent(cand.symbol, cand)
            if sub_clean is not None and sub_clean.order == exp_ord:
                cand = sub_clean
            else:
                continue
        if cand.order > highest_res.order:
            highest_res = cand
        if cand.symbol not in found_by_symbol:
            tag = "higher" if cand.order > exact_res.order else "lower"
            found_by_symbol[cand.symbol] = (cand, tag)

    # Extract all valid subgroups from both highest_res and exact_res
    for parent in (highest_res, exact_res):
        sub_list = SUBGROUP_CANDIDATES.get(
            parent.symbol,
            [parent.symbol, "D2h", "C2v", "C2h", "D2", "C2", "Cs", "Ci", "C1"],
        )
        for sub_sym in sub_list:
            if sub_sym in found_by_symbol:
                continue
            sub_res = _extract_subgroup_from_parent(sub_sym, parent)
            if sub_res is not None:
                tag = "higher" if sub_res.order > exact_res.order else "lower"
                found_by_symbol[sub_sym] = (sub_res, tag)

    # Always ensure C1 is available
    if "C1" not in found_by_symbol:
        c1_res = _extract_subgroup_from_parent("C1", exact_res)
        if c1_res is not None:
            found_by_symbol["C1"] = (c1_res, "lower" if exact_res.order > 1 else "detected")

    results: List[Tuple[str, PointGroupResult, str]] = [
        (sym, res, tag) for sym, (res, tag) in found_by_symbol.items()
    ]
    results.sort(key=lambda item: (item[1].order, item[0] == exact_res.symbol, item[0]), reverse=True)

    if len(_AVAILABLE_SYMMETRIES_CACHE) >= _AVAILABLE_SYMMETRIES_CACHE_MAX:
        _AVAILABLE_SYMMETRIES_CACHE.pop(next(iter(_AVAILABLE_SYMMETRIES_CACHE)))
    _AVAILABLE_SYMMETRIES_CACHE[cache_key] = results

    return results


def get_symmetry_orbits(
    symbols: List[str],
    coords: np.ndarray,
    result: Optional[PointGroupResult] = None,
    tolerance: float = 0.25,
) -> List[List[int]]:
    """
    Partition atom indices [0..N-1] into symmetry equivalence classes (orbits)
    under the point group operations of `result`.
    """
    n = len(symbols)
    if n == 0:
        return []
    if result is None:
        result = find_symmetry_operations(symbols, coords, tolerance=tolerance)
    if result.order <= 1 or not result.operations:
        return [[i] for i in range(n)]

    centered, _ = _center_molecule(symbols, coords)
    tol = max(tolerance, float(getattr(result, "tolerance", 0.15)) * 1.5, 0.20)

    visited = [False] * n
    orbits: List[List[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        orbit_set = {i}
        s_i = symbols[i]
        for op in result.operations:
            pos_t = op.matrix @ centered[i]
            best_j = -1
            best_d = tol
            for j in range(n):
                if symbols[j] != s_i:
                    continue
                d = float(np.linalg.norm(pos_t - centered[j]))
                if d <= best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                orbit_set.add(best_j)
        orbit_list = sorted(orbit_set)
        for j in orbit_list:
            visited[j] = True
        orbits.append(orbit_list)
    return orbits


def get_atom_orbit_and_operations(
    symbols: List[str],
    coords: np.ndarray,
    seed_idx: int,
    result: Optional[PointGroupResult] = None,
    tolerance: float = 0.25,
) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
    """
    Given a seed atom `seed_idx`, return `(com, [(atom_j, R_j), ...])` for all atoms `atom_j`
    in the symmetry orbit of `seed_idx`, where `R_j` is a 3x3 orthogonal matrix from `result.operations`
    that maps `centered[seed_idx]` to `centered[atom_j]`.
    Prefers proper rotations (`det(R) > 0`) when available, and places `seed_idx` first with `I_3`.
    """
    n = len(symbols)
    if n == 0 or seed_idx < 0 or seed_idx >= n:
        return np.zeros(3), []
    centered, com = _center_molecule(symbols, coords)
    if result is None:
        result = find_symmetry_operations(symbols, coords, tolerance=tolerance)
    if result.order <= 1 or not result.operations:
        return com, [(seed_idx, np.eye(3, dtype=np.float64))]

    tol = max(tolerance, float(getattr(result, "tolerance", 0.15)) * 1.5, 0.25)
    seed_sym = symbols[seed_idx]
    seed_pos = centered[seed_idx]

    # Map each target atom j in the orbit to the best operation R (prefer proper rotation det > 0, then smallest distance)
    best_ops: Dict[int, Tuple[float, float, np.ndarray]] = {
        seed_idx: (0.0, 0.0, np.eye(3, dtype=np.float64))
    }
    for op in result.operations:
        R = np.asarray(op.matrix, dtype=np.float64)
        pos_t = R @ seed_pos
        best_j = -1
        best_d = tol
        for j in range(n):
            if symbols[j] != seed_sym:
                continue
            d = float(np.linalg.norm(pos_t - centered[j]))
            if d <= best_d:
                best_d = d
                best_j = j
        if best_j >= 0 and best_j != seed_idx:
            is_improper = 0.0 if float(np.linalg.det(R)) > 0.0 else 1.0
            cand_key = (is_improper, best_d)
            prev = best_ops.get(best_j)
            if prev is None or cand_key < (prev[0], prev[1]):
                best_ops[best_j] = (is_improper, best_d, R.copy())

    orbit_pairs: List[Tuple[int, np.ndarray]] = [(seed_idx, np.eye(3, dtype=np.float64))]
    for j in sorted(best_ops.keys()):
        if j == seed_idx:
            continue
        orbit_pairs.append((j, best_ops[j][2]))
    return com, orbit_pairs


def map_atom_by_operation(
    symbols: List[str],
    coords: np.ndarray,
    atom_idx: int,
    R: np.ndarray,
    com: Optional[np.ndarray] = None,
    tolerance: float = 0.35,
) -> int:
    """Return the index of the atom matching `R @ (coords[atom_idx] - com) + com`."""
    n = len(symbols)
    if atom_idx < 0 or atom_idx >= n:
        return atom_idx
    if com is None:
        _, com = _center_molecule(symbols, coords)
    pos_t = (R @ (coords[atom_idx] - com)) + com
    target_sym = symbols[atom_idx]
    best_j = atom_idx
    best_d = 1e9
    for j in range(n):
        if symbols[j] != target_sym:
            continue
        d = float(np.linalg.norm(pos_t - coords[j]))
        if d < best_d:
            best_d = d
            best_j = j
    return best_j if best_d <= max(tolerance, 0.5) else atom_idx


def enforce_point_group_symmetry(
    symbols: List[str],
    coords: np.ndarray,
    target_group: Optional[str] = None,
    tolerance: float = 0.35,
    reference_result: Optional[PointGroupResult] = None,
) -> Tuple[np.ndarray, PointGroupResult]:
    """
    Symmetrize molecular coordinates using group projection / orbit averaging:
    x_i^{sym} = (1 / |G|) * sum_{g in G} g * x_{pi(g, i)}
    If tolerance was violated, this projects coordinates directly into the exact representation subspace.
    Returns: (symmetrized_coords, updated_point_group_result)
    """
    n = len(symbols)
    if n <= 1:
        return np.copy(coords), find_symmetry_operations(symbols, coords)

    res = reference_result if (reference_result is not None and reference_result.order > 0) else None
    if res is None:
        if target_group is not None:
            avail = find_available_symmetries(symbols, coords, base_tolerance=min(tolerance, 0.15))
            for sym_name, cand_res, _ in avail:
                if sym_name == target_group:
                    res = cand_res
                    break
        if res is None:
            res = find_symmetry_operations(symbols, coords, tolerance=tolerance)
            if (res.symbol == "C1" or (target_group is not None and res.symbol != target_group)) and target_group is not None:
                for try_tol in (0.35, 0.50, 0.75, 0.85):
                    cand_res = find_symmetry_operations(symbols, coords, tolerance=try_tol)
                    if cand_res.symbol == target_group or cand_res.order > res.order:
                        res = cand_res
                    if res.symbol == target_group:
                        break
                if target_group is not None and res.symbol != target_group:
                    sub_res = _extract_subgroup_from_parent(target_group, res)
                    if sub_res is not None:
                        res = sub_res

    if res.order <= 1 or not res.operations:
        return np.copy(coords), res

    centered, com = _center_molecule(symbols, coords)
    symmetrized = np.zeros_like(centered)
    h = res.order

    # Group atoms by element
    elem_indices: Dict[str, List[int]] = {}
    for i, s in enumerate(symbols):
        elem_indices.setdefault(s, []).append(i)

    # For each atom i and each operation g, find matched atom j and project
    for i in range(n):
        s_i = symbols[i]
        sum_vec = np.zeros(3)
        c_i = centered[i]
        c_subset = centered[elem_indices[s_i]]
        elem_list = elem_indices[s_i]

        for op in res.operations:
            R = op.matrix
            # Inverse of orthogonal matrix is R^T
            R_inv = R.T
            pos_t = R_inv @ c_i
            # Vectorized nearest atom search within same element
            diffs = c_subset - pos_t
            dists_sq = np.sum(diffs * diffs, axis=1)
            best_local_idx = int(np.argmin(dists_sq))
            best_j = elem_list[best_local_idx]
            sum_vec += R @ centered[best_j]
        symmetrized[i] = sum_vec / float(h)

    # Restore center of mass
    final_coords = symmetrized + com
    new_res = find_symmetry_operations(symbols, final_coords, tolerance=tolerance)
    # If user explicitly requested a lower subgroup (e.g. D4h or C4v on an Oh molecule),
    # return the requested subgroup PointGroupResult so the chosen lower symmetry stays selected!
    if target_group is not None and new_res.symbol != target_group:
        if reference_result is not None and reference_result.symbol == target_group:
            return final_coords, reference_result
        sub_res = _extract_subgroup_from_parent(target_group, new_res)
        if sub_res is not None:
            return final_coords, sub_res
    return final_coords, new_res


# -------------------------------------------------------------
# 3. Option: Reducible & Irreducible Representations (Rotational & Vibrational Analysis)
# -------------------------------------------------------------
def calculate_symmetry_representations(
    symbols: List[str],
    coords: np.ndarray,
    result: PointGroupResult
) -> Dict[str, any]:
    """
    Calculate reducible representations and decompose into direct sum of irreps:
      - Gamma_tot = 3N representation
      - Gamma_trans = Translational representation
      - Gamma_rot = Rotational representation
      - Gamma_vib = Vibrational representation (Gamma_tot - Gamma_trans - Gamma_rot)
    Returns: dictionary with classes, characters, and direct sum irrep expansions.
    """
    group = result.symbol
    if group not in CHARACTER_TABLES:
        # Fallback to C1
        group = "C1"

    ctable = CHARACTER_TABLES[group]
    classes = ctable["classes"]      # List of (class_label, class_multiplicity)
    irreps = ctable["irreps"]        # Dict of irrep_label -> [chi_1, chi_2, ...]
    trans_decomp = dict(ctable["trans"])
    rot_decomp = dict(ctable["rot"])

    h = sum(g for _, g in classes)   # Order of the group
    centered, _ = _center_molecule(symbols, coords)
    n = len(symbols)

    # Build character vector for Gamma_tot
    gamma_tot_chars = []
    class_ops = []

    p_ax = result.principal_axis if result.principal_axis is not None else np.array([0.0, 0.0, 1.0])

    def _get_op_for_class(class_label: str) -> SymmetryOperation:
        if class_label == "E":
            return SymmetryOperation("E", np.eye(3), "E")
        if class_label == "i":
            return SymmetryOperation("i", -np.eye(3), "i")

        # For cubic groups (Oh, Td, O, Th, Ih), pick an actual operation from result.operations
        if result.symbol in ("Oh", "Td", "O", "Th", "Ih", "I"):
            if "C3" in class_label:
                for op in result.operations:
                    if op.op_type == "Cn" and op.order == 3:
                        return op
            if "C4" in class_label:
                for op in result.operations:
                    if op.op_type == "Cn" and op.order == 4:
                        return op
            if "C2'" in class_label:
                for op in result.operations:
                    if op.op_type == "Cn" and op.label == "C2'":
                        return op
            if "C2" in class_label and "'" not in class_label:
                for op in result.operations:
                    if op.op_type == "Cn" and op.label == "C2":
                        return op
            if "S4" in class_label:
                for op in result.operations:
                    if op.op_type == "Sn" and op.order == 4:
                        return op
            if "S6" in class_label:
                for op in result.operations:
                    if op.op_type == "Sn" and op.order == 6:
                        return op
            if "sigma_h" in class_label:
                for op in result.operations:
                    if op.op_type == "sigma" and op.label == "sigma_h":
                        return op
            if "sigma_d" in class_label:
                for op in result.operations:
                    if op.op_type == "sigma" and op.label == "sigma_d":
                        return op

        if class_label in ("C2", "C2(z)"):
            return SymmetryOperation("Cn", _rotation_matrix(p_ax, math.pi), "C2", order=2, axis=p_ax)
        if "C3" in class_label and "'" not in class_label:
            return SymmetryOperation("Cn", _rotation_matrix(p_ax, 2.0 * math.pi / 3.0), "C3", order=3, axis=p_ax)
        if "C4" in class_label and "'" not in class_label:
            return SymmetryOperation("Cn", _rotation_matrix(p_ax, 2.0 * math.pi / 4.0), "C4", order=4, axis=p_ax)
        if "C6" in class_label and "'" not in class_label:
            return SymmetryOperation("Cn", _rotation_matrix(p_ax, 2.0 * math.pi / 6.0), "C6", order=6, axis=p_ax)

        perp_c2_ops = [
            op for op in result.operations
            if op.op_type == "Cn" and op.order == 2 and
            not np.allclose(op.axis, p_ax, atol=1e-2) and not np.allclose(op.axis, -p_ax, atol=1e-2)
        ]
        if "C2'" in class_label or class_label == "C2(y)":
            if perp_c2_ops:
                return max(perp_c2_ops, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))
        if "C2''" in class_label or class_label == "C2(x)":
            if perp_c2_ops:
                return min(perp_c2_ops, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))

        if "S3" in class_label:
            c3 = _rotation_matrix(p_ax, 2.0 * math.pi / 3.0)
            return SymmetryOperation("Sn", _reflection_matrix(p_ax) @ c3, "S3", order=3, axis=p_ax)
        if "S4" in class_label:
            c4 = _rotation_matrix(p_ax, 2.0 * math.pi / 4.0)
            return SymmetryOperation("Sn", _reflection_matrix(p_ax) @ c4, "S4", order=4, axis=p_ax)
        if "S6" in class_label:
            c6 = _rotation_matrix(p_ax, 2.0 * math.pi / 6.0)
            return SymmetryOperation("Sn", _reflection_matrix(p_ax) @ c6, "S6", order=6, axis=p_ax)

        if "sigma_h" in class_label or "sigma(xy)" in class_label:
            return SymmetryOperation("sigma", _reflection_matrix(p_ax), "sigma_h", axis=p_ax)

        sigma_ops = [op for op in result.operations if op.op_type == "sigma"]
        vert_planes = [op for op in sigma_ops if abs(np.dot(op.axis, p_ax)) < 0.1]

        if result.symbol == "C2v":
            if class_label == "sigma_v(xz)":
                if vert_planes:
                    return min(vert_planes, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))
            elif class_label == "sigma_v(yz)":
                if vert_planes:
                    return max(vert_planes, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))

        if "sigma_v" in class_label or class_label == "sigma(xz)":
            if vert_planes:
                return max(vert_planes, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))
        if "sigma_d" in class_label or class_label == "sigma(yz)":
            if vert_planes:
                return min(vert_planes, key=lambda op: sum(1 for p in centered if np.linalg.norm(op.matrix @ p - p) <= result.tolerance))

        for op in result.operations:
            if op.label == class_label or class_label.endswith(op.label):
                return op
        return SymmetryOperation("E", np.eye(3), "E")

    for class_label, _ in classes:
        rep_op = _get_op_for_class(class_label)
        class_ops.append(rep_op)

        # Count unshifted atoms N_R
        R = rep_op.matrix
        n_unshifted = 0
        for i in range(n):
            pos_t = R @ centered[i]
            if np.linalg.norm(pos_t - centered[i]) <= result.tolerance:
                n_unshifted += 1

        # Character of Cartesian coordinates under operation R: chi_cart(R) = Tr(R)
        chi_cart = float(np.trace(R))
        # Total representation character chi_tot(R) = N_R * Tr(R)
        chi_tot = n_unshifted * chi_cart
        gamma_tot_chars.append(chi_tot)

    # Reduction formula: a_i = 1/h * sum_k g_k * chi_tot(k) * chi_irrep(k)
    tot_decomp: Dict[str, int] = {}
    for irrep_label, chars in irreps.items():
        val = sum(g * chi_tot * chi_i for (c_name, g), chi_tot, chi_i in zip(classes, gamma_tot_chars, chars))
        a_i = int(round(val / float(h)))
        if a_i > 0:
            tot_decomp[irrep_label] = a_i

    # Calculate vibrational representation: Gamma_vib = Gamma_tot - Gamma_trans - Gamma_rot
    vib_decomp: Dict[str, int] = {}
    all_irreps = list(irreps.keys())
    for irr in all_irreps:
        c_tot = tot_decomp.get(irr, 0)
        c_tr = trans_decomp.get(irr, 0)
        c_rot = rot_decomp.get(irr, 0)
        c_vib = c_tot - c_tr - c_rot
        if c_vib > 0:
            vib_decomp[irr] = c_vib

    def format_rep_str(decomp: Dict[str, int]) -> str:
        parts = []
        for k in sorted(decomp.keys()):
            v = decomp[k]
            parts.append(f"{v}{k}" if v > 1 else f"{k}")
        return " ⊕ ".join(parts) if parts else "0"

    result.gamma_tot = {cls_name: char for (cls_name, _), char in zip(classes, gamma_tot_chars)}
    result.gamma_trans = trans_decomp
    result.gamma_rot = rot_decomp
    result.gamma_vib = vib_decomp

    return {
        "classes": [c[0] for c in classes],
        "class_orders": [c[1] for c in classes],
        "class_ops": class_ops,
        "gamma_tot_chars": gamma_tot_chars,
        "gamma_tot_str": format_rep_str(tot_decomp),
        "gamma_trans_str": format_rep_str(trans_decomp),
        "gamma_rot_str": format_rep_str(rot_decomp),
        "gamma_vib_str": format_rep_str(vib_decomp),
        "gamma_tot_dict": tot_decomp,
        "gamma_vib_dict": vib_decomp
    }


def classify_vibrational_mode_irrep(
    symbols: List[str],
    coords: np.ndarray,
    displacement: np.ndarray,
    sym_res: Optional[PointGroupResult] = None,
) -> str:
    """
    Project a vibrational mode displacement vector (N, 3) onto the irreps of the point group.
    Returns the irrep label (e.g. 'A1', 'B2', 'Eg', etc.) with highest projection weight.
    """
    if len(symbols) == 0 or displacement is None or displacement.shape[0] != len(symbols):
        return "-"

    m_norm2 = float(np.sum(displacement * displacement))
    if m_norm2 < 1e-12:
        return "-"

    if sym_res is None or sym_res.order <= 1 or sym_res.symbol == "C1":
        sym_res = find_symmetry_operations(symbols, coords, tolerance=0.15)
        if sym_res.order <= 1 or sym_res.symbol == "C1":
            return "A" if sym_res.symbol == "C1" else "-"

    group = sym_res.symbol
    if group not in CHARACTER_TABLES:
        return "-"

    ctable = CHARACTER_TABLES[group]
    classes = ctable["classes"]
    irreps = ctable["irreps"]
    h = sum(g for _, g in classes)

    # Calculate class ops using calculate_symmetry_representations logic
    reps_info = calculate_symmetry_representations(symbols, coords, sym_res)
    class_ops = reps_info.get("class_ops", [])
    if len(class_ops) != len(classes):
        return "-"

    centered, _ = _center_molecule(symbols, coords)
    n = len(symbols)

    # Compute permutation for each class representative op
    class_perms = []
    for op in class_ops:
        perm = []
        for i in range(n):
            mapped = op.matrix @ centered[i]
            dists = [np.linalg.norm(mapped - centered[j]) for j in range(n)]
            perm.append(int(np.argmin(dists)))
        class_perms.append(perm)

    best_irrep = "-"
    best_weight = 0.0

    for irr_name, irr_chars in irreps.items():
        d_irr = irr_chars[0]
        proj = np.zeros_like(displacement)
        for (c_name, g), chi, op, perm in zip(classes, irr_chars, class_ops, class_perms):
            # Transform displacement under op
            op_m = np.zeros_like(displacement)
            for i in range(n):
                op_m[perm[i]] = op.matrix @ displacement[i]
            # Multiply by class multiplicity g and character chi
            proj += (g * chi) * op_m

        proj = (float(d_irr) / float(h)) * proj
        weight = float(np.sum(proj * proj) / m_norm2)
        if weight > best_weight:
            best_weight = weight
            best_irrep = irr_name

    return best_irrep if best_weight >= 0.25 else "-"


def optimize_with_imposed_symmetry(
    optimize_fn,
    symbols: List[str],
    coords: np.ndarray,
    sym_res: PointGroupResult,
    cycles: int = 6,
    max_projection_shift: float = 0.5,
    min_atom_distance: float = 0.4,
) -> Tuple[Optional[np.ndarray], Optional[object], str]:
    """
    Run a geometry optimization while keeping the point group `sym_res` exact.

    `optimize_fn(coords) -> (new_coords, *extra)` performs a (partial) optimization step.
    Each cycle: optimize, then project the coordinates back into the symmetric subspace
    (`enforce_point_group_symmetry` with `reference_result=sym_res`).

    Returns `(coords, extra, message)`. If the requested symmetry cannot be achieved
    (starting geometry too far from the group, atoms collapsing, or the group not
    holding after projection) returns `(None, None, reason)` so the caller can fall back.
    """
    n = len(symbols)
    if n < 2 or sym_res is None or sym_res.order <= 1:
        return None, None, "No symmetry to impose."

    tol = max(0.05, float(getattr(sym_res, "tolerance", 0.15)))

    def _project(c):
        return enforce_point_group_symmetry(
            symbols, c, target_group=sym_res.symbol, tolerance=tol, reference_result=sym_res
        )[0]

    start = np.asarray(coords, dtype=np.float64)
    cur = _project(start)
    if float(np.max(np.linalg.norm(cur - start, axis=1))) > max_projection_shift:
        return None, None, (
            f"Geometry is too far from {sym_res.symbol} symmetry "
            f"(projection would move atoms by more than {max_projection_shift:.2f} Å)."
        )

    extra = None
    for _ in range(max(1, cycles)):
        out = optimize_fn(cur)
        new_c, extra = out[0], out[1:]
        cur = _project(np.asarray(new_c, dtype=np.float64))

    # Verify result is physically sensible and keeps the symmetry
    d = np.linalg.norm(cur[:, None, :] - cur[None, :, :], axis=2) + np.eye(n) * 10.0
    if float(d.min()) < min_atom_distance:
        return None, None, f"Optimization under {sym_res.symbol} symmetry collapses atoms together."
    check = find_symmetry_operations(symbols, cur, tolerance=0.05)
    if check.order < sym_res.order:
        return None, None, f"{sym_res.symbol} symmetry could not be preserved during optimization."
    return cur, extra, f"Optimized with {sym_res.symbol} symmetry imposed."


def determine_electronic_term_symbol(
    point_group: str,
    orbitals: list,
    mult: Optional[int] = None,
) -> str:
    """
    Determine the spectroscopic term symbol of the single-determinant electronic wavefunction
    from point group symmetry, molecular orbital symmetries, and occupancies.

    Examples:
        - Water (C2v, closed-shell): ¹A₁
        - Water cation H₂O⁺ (C2v, open-shell 1b₁): ²B₁
        - Ethylene (D2h, closed-shell): ¹Ag
        - Triplet excited state (D2h, 1b₁ᵤ × 2b₂_g): ³B₃ᵤ
    """
    import re
    pg = (point_group or "C1").strip()
    table = CHARACTER_TABLES.get(pg)

    singly_occ = []
    tot_unpaired = 0.0
    for mo in orbitals:
        occ = float(getattr(mo, "occupancy", getattr(mo, "occup", 0.0)))
        raw_sym = str(getattr(mo, "sym", "") or "").strip()
        m = re.search(r"([A-Za-z]+[0-9]*[a-z]*)", raw_sym)
        clean_sym = m.group(1) if m else raw_sym
        if 0.05 < occ < 1.95:
            singly_occ.append((clean_sym, occ))
            tot_unpaired += occ

    if mult is not None and mult > 0:
        spin_mult = mult
    else:
        spin_mult = int(round(tot_unpaired)) + 1 if tot_unpaired > 0 else 1

    superscript_map = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    mult_str = str(spin_mult).translate(superscript_map)

    if not table:
        return f"{mult_str}A"

    irreps = table["irreps"]
    classes = table["classes"]
    tot_sym = list(irreps.keys())[0]

    if not singly_occ:
        return f"{mult_str}{tot_sym}"

    h = sum(c[1] for c in classes)
    char_prod = np.ones(len(classes), dtype=np.float64)
    matched_count = 0
    for sym_name, occ in singly_occ:
        for ir in irreps:
            if ir.lower() == sym_name.lower() or sym_name.lower().endswith(ir.lower()):
                char_prod *= np.array(irreps[ir], dtype=np.float64)
                matched_count += 1
                break

    if matched_count == 0:
        return f"{mult_str}{tot_sym}"

    decomposed = []
    for ir, chars in irreps.items():
        val = sum(c[1] * char_prod[idx] * chars[idx] for idx, c in enumerate(classes))
        coeff = val / h
        if round(coeff) > 0:
            count_prefix = str(int(round(coeff))) if round(coeff) > 1 else ""
            decomposed.append(f"{count_prefix}{ir}")

    if decomposed:
        state_sym = " ⊕ ".join(decomposed)
        return f"{mult_str}{state_sym}"
    return f"{mult_str}{tot_sym}"
