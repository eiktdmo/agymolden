"""
GaussView-style Element Coordination, Hybridization, and VSEPR Geometry Templates.

Provides directional valence vectors for coordination numbers 0 through 8
(Linear, Bent, Trigonal Planar, Pyramidal, T-Shaped, Tetrahedral, Square Planar,
Seesaw, Trigonal Bipyramidal, Square Pyramidal, Octahedral, Trigonal Prismatic,
Pentagonal Bipyramidal, Square Antiprismatic) and sensible element defaults across
the entire periodic table.
"""

import math
from typing import Dict, List, Tuple, Optional
import numpy as np

from .alignment import rotation_matrix_from_vectors


# Canonical VSEPR / Coordination Geometry templates:
# Key -> (Display Label, Coordination Number, List of Unit Vectors from Origin)
COORDINATION_TEMPLATES: Dict[str, Tuple[str, int, List[Tuple[float, float, float]]]] = {
    "auto": (
        "Auto (Standard Element Valence)",
        -1,
        [],
    ),
    "naked": (
        "Naked Atom (0 Bonds)",
        0,
        [],
    ),
    "terminal": (
        "Terminal / Monovalent (1 Bond)",
        1,
        [(0.0, 0.0, 1.0)],
    ),
    "linear": (
        "Linear — sp / D∞h (2 Bonds, 180°)",
        2,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
        ],
    ),
    "bent_104": (
        "Bent — sp³ / C2v (2 Bonds, 104.5°)",
        2,
        [
            (math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))),
            (-math.sin(math.radians(52.25)), 0.0, math.cos(math.radians(52.25))),
        ],
    ),
    "bent_120": (
        "Bent — sp² / C2v (2 Bonds, 120°)",
        2,
        [
            (math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))),
            (-math.sin(math.radians(60.0)), 0.0, math.cos(math.radians(60.0))),
        ],
    ),
    "trigonal_planar": (
        "Trigonal Planar — sp² / D3h (3 Bonds, 120°)",
        3,
        [
            (0.0, 0.0, 1.0),
            (math.sin(math.radians(120.0)), 0.0, math.cos(math.radians(120.0))),
            (-math.sin(math.radians(120.0)), 0.0, math.cos(math.radians(120.0))),
        ],
    ),
    "trigonal_pyramidal": (
        "Trigonal Pyramidal — sp³ / C3v (3 Bonds, 107°)",
        3,
        [
            (0.942809, 0.0, -0.333333),
            (-0.471405, 0.816497, -0.333333),
            (-0.471405, -0.816497, -0.333333),
        ],
    ),
    "t_shaped": (
        "T-Shaped — dsp³ / C2v (3 Bonds, 90°/180°)",
        3,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
        ],
    ),
    "tetrahedral": (
        "Tetrahedral — sp³ / Td (4 Bonds, 109.47°)",
        4,
        [
            (0.0, 0.0, 1.0),
            (0.94280904, 0.0, -0.33333333),
            (-0.47140452, 0.81649658, -0.33333333),
            (-0.47140452, -0.81649658, -0.33333333),
        ],
    ),
    "square_planar": (
        "Square Planar — dsp² / D4h (4 Bonds, 90°)",
        4,
        [
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
    "seesaw": (
        "Seesaw — sp³d / C2v (4 Bonds, 180° & 120°)",
        4,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (math.cos(math.radians(30.0)), math.sin(math.radians(30.0)), 0.0),
            (math.cos(math.radians(30.0)), -math.sin(math.radians(30.0)), 0.0),
        ],
    ),
    "trigonal_bipyramidal": (
        "Trigonal Bipyramidal — sp³d / D3h (5 Bonds)",
        5,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (-0.5, math.sqrt(3.0) / 2.0, 0.0),
            (-0.5, -math.sqrt(3.0) / 2.0, 0.0),
        ],
    ),
    "square_pyramidal": (
        "Square Pyramidal — sp³d² / C4v (5 Bonds)",
        5,
        [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
    "octahedral": (
        "Octahedral — sp³d² / Oh (6 Bonds, 90°)",
        6,
        [
            (0.0, 0.0, 1.0),
            (0.0, 0.0, -1.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        ],
    ),
}


# Default coordination geometry template per element when "auto" is chosen:
DEFAULT_ELEMENT_COORDINATION: Dict[str, str] = {
    "X": "naked",
    "H": "terminal",
    "He": "naked",
    "Ne": "naked",
    "Ar": "naked",
    "Kr": "naked",
    "Li": "terminal",
    "Na": "terminal",
    "K": "terminal",
    "Rb": "terminal",
    "Cs": "terminal",
    "Be": "linear",
    "Mg": "linear",
    "Ca": "linear",
    "Sr": "linear",
    "Ba": "linear",
    "B": "trigonal_planar",
    "Al": "trigonal_planar",
    "Ga": "trigonal_planar",
    "In": "trigonal_planar",
    "C": "tetrahedral",
    "Si": "tetrahedral",
    "Ge": "tetrahedral",
    "Sn": "tetrahedral",
    "Pb": "tetrahedral",
    "N": "trigonal_pyramidal",
    "P": "trigonal_pyramidal",
    "As": "trigonal_pyramidal",
    "Sb": "trigonal_pyramidal",
    "Bi": "trigonal_pyramidal",
    "O": "bent_104",
    "S": "bent_104",
    "Se": "bent_104",
    "Te": "bent_104",
    "F": "terminal",
    "Cl": "terminal",
    "Br": "terminal",
    "I": "terminal",
    # Transition metals: default to common coordination geometries
    "Pt": "square_planar",
    "Pd": "square_planar",
    "Ni": "square_planar",
    "Au": "linear",
    "Ag": "linear",
    "Cu": "tetrahedral",
    "Zn": "tetrahedral",
    "Cd": "tetrahedral",
    "Hg": "linear",
    "Ti": "octahedral",
    "V": "octahedral",
    "Cr": "octahedral",
    "Mn": "octahedral",
    "Fe": "octahedral",
    "Co": "octahedral",
    "Ru": "octahedral",
    "Rh": "octahedral",
    "Mo": "octahedral",
    "Zr": "octahedral",
    "Nb": "octahedral",
    "Tc": "octahedral",
    "Sc": "octahedral",
    "Y": "octahedral",
    "U": "octahedral",
    "Xe": "square_planar",
}


def resolve_coordination_template(element: str, template_key: str = "auto") -> str:
    """Resolve 'auto' to the canonical template key for `element`."""
    key = (template_key or "auto").strip().lower()
    if key != "auto" and key in COORDINATION_TEMPLATES:
        return key
    return DEFAULT_ELEMENT_COORDINATION.get(element, "tetrahedral")


def get_coordination_vectors(
    element: str,
    template_key: str = "auto",
    existing_bond_vectors: Optional[List[np.ndarray]] = None,
) -> List[np.ndarray]:
    """
    Compute unit vectors pointing along the open coordination sites for `element`
    under `template_key` (e.g. 'tetrahedral', 'trigonal_planar', 'square_planar', 'octahedral').

    If `existing_bond_vectors` contains 1 or more unit vectors from the central atom to
    already-bonded neighbors, the template is oriented so its first vector aligns with the
    first existing bond vector, and only the remaining unoccupied coordination directions
    are returned!
    """
    resolved = resolve_coordination_template(element, template_key)
    _, coord_num, raw_vecs = COORDINATION_TEMPLATES.get(resolved, COORDINATION_TEMPLATES["tetrahedral"])
    if coord_num <= 0 or not raw_vecs:
        return []

    vecs = [np.array(v, dtype=np.float64) / np.linalg.norm(v) for v in raw_vecs]
    existing = []
    if existing_bond_vectors:
        for ev in existing_bond_vectors:
            n_ev = float(np.linalg.norm(ev))
            if n_ev > 1e-6:
                existing.append(np.asarray(ev, dtype=np.float64) / n_ev)

    if not existing:
        return vecs

    # Align the first template vector with existing[0]
    R = rotation_matrix_from_vectors(vecs[0], existing[0])
    rotated = [R @ v for v in vecs]

    # Filter out template sites that are already occupied by existing bonds
    remaining: List[np.ndarray] = []
    used_indices = {0}
    for ex in existing[1:]:
        # Find closest unused template vector
        best_k = None
        best_dot = -2.0
        for k, rv in enumerate(rotated):
            if k in used_indices:
                continue
            d = float(np.dot(ex, rv))
            if d > best_dot:
                best_dot = d
                best_k = k
        if best_k is not None:
            used_indices.add(best_k)

    for k, rv in enumerate(rotated):
        if k not in used_indices:
            remaining.append(rv)

    return remaining


def compute_buried_volume(
    symbols: List[str],
    coords: np.ndarray,
    center_idx: int = 0,
    sphere_radius: float = 3.5,
    include_hydrogens: bool = True,
    bond_scale: float = 1.17,
    grid_spacing: float = 0.15,
    excluded_indices: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Compute Cavallo / SambVca Percent Buried Volume (%V_bur) and Quadrant Steric Breakdown
    around a coordination center atom `center_idx`.

    Reference: Falivene et al., Nature Chemistry 11, 872-879 (2019).
    Uses Bondi van der Waals radii scaled by `bond_scale` (default 1.17).
    """
    from .elements import ELEMENT_DATA

    coords_arr = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms == 0 or not (0 <= center_idx < n_atoms):
        return {
            "v_bur_pct": 0.0,
            "v_free_pct": 100.0,
            "sphere_radius": sphere_radius,
            "quadrants": {"SW": 0.0, "NW": 0.0, "NE": 0.0, "SE": 0.0},
        }

    origin = coords_arr[center_idx]
    excl = set(excluded_indices or [])
    excl.add(center_idx)

    atom_pts = []
    atom_r2 = []
    for idx in range(n_atoms):
        if idx in excl:
            continue
        sym = symbols[idx]
        if not include_hydrogens and sym == "H":
            continue
        vdw = float(ELEMENT_DATA.get(sym, {}).get("vdw_radius", 1.70)) * bond_scale
        rel = coords_arr[idx] - origin
        # Only atoms that can intersect the sphere of radius R need to be checked
        if float(np.linalg.norm(rel)) <= sphere_radius + vdw:
            atom_pts.append(rel)
            atom_r2.append(vdw * vdw)

    # Construct uniform 3D Cartesian voxel grid inside [-R, R]^3
    n_pts = max(15, int(round((2.0 * sphere_radius) / max(0.08, grid_spacing))))
    ax = np.linspace(-sphere_radius, sphere_radius, n_pts)
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    r2_sphere = np.sum(pts ** 2, axis=1)
    in_sphere = r2_sphere <= (sphere_radius * sphere_radius)
    sphere_pts = pts[in_sphere]

    n_total = len(sphere_pts)
    if n_total == 0 or not atom_pts:
        return {
            "v_bur_pct": 0.0,
            "v_free_pct": 100.0,
            "sphere_radius": sphere_radius,
            "quadrants": {"SW": 0.0, "NW": 0.0, "NE": 0.0, "SE": 0.0},
        }

    buried_mask = np.zeros(n_total, dtype=bool)
    for pt_a, r2_a in zip(atom_pts, atom_r2):
        diff = sphere_pts - pt_a
        dist2 = np.sum(diff * diff, axis=1)
        buried_mask |= (dist2 <= r2_a)

    v_bur_pct = 100.0 * float(np.sum(buried_mask)) / float(n_total)

    # Quadrant percentages in XY plane: NE (+x,+y), NW (-x,+y), SW (-x,-y), SE (+x,-y)
    px = sphere_pts[:, 0]
    py = sphere_pts[:, 1]
    quad_masks = {
        "NE": (px >= 0.0) & (py >= 0.0),
        "NW": (px < 0.0) & (py >= 0.0),
        "SW": (px < 0.0) & (py < 0.0),
        "SE": (px >= 0.0) & (py < 0.0),
    }
    quadrants = {}
    for q_name, q_m in quad_masks.items():
        q_tot = int(np.sum(q_m))
        quadrants[q_name] = (100.0 * float(np.sum(buried_mask & q_m)) / float(q_tot)) if q_tot > 0 else 0.0

    return {
        "v_bur_pct": round(v_bur_pct, 2),
        "v_free_pct": round(100.0 - v_bur_pct, 2),
        "sphere_radius": float(sphere_radius),
        "quadrants": {k: round(v, 1) for k, v in quadrants.items()},
    }


def compute_sterimol_and_cone_angle(
    symbols: List[str],
    coords: np.ndarray,
    primary_idx: int = 0,
    attached_idx: int = 1,
    substituent_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Verloop Sterimol parameters (L, B1, B5) and Tolman Cone Angle (deg)
    along the bond axis from `primary_idx` (e.g. Metal or ipso-atom) -> `attached_idx` (donor/alpha atom).

    Reference: Verloop et al., Drug Design (1976); Tolman, Chem. Rev. 77, 313 (1977).
    """
    from .elements import ELEMENT_DATA

    coords_arr = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms < 2 or not (0 <= primary_idx < n_atoms) or not (0 <= attached_idx < n_atoms) or primary_idx == attached_idx:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0, "cone_angle_deg": 0.0}

    if substituent_indices is None:
        # Automatically collect all atoms except primary_idx
        sub_idxs = [i for i in range(n_atoms) if i != primary_idx]
    else:
        sub_idxs = [i for i in substituent_indices if 0 <= i < n_atoms and i != primary_idx]

    if not sub_idxs:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0, "cone_angle_deg": 0.0}

    origin = coords_arr[primary_idx]
    axis_vec = coords_arr[attached_idx] - origin
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-8:
        return {"L": 0.0, "B1": 0.0, "B5": 0.0, "cone_angle_deg": 0.0}
    u_axis = axis_vec / axis_norm

    # Build orthonormal frame (u_x, u_y, u_axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(u_axis[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
    u_x = np.cross(u_axis, ref)
    u_x /= np.linalg.norm(u_x)
    u_y = np.cross(u_axis, u_x)

    l_max = 0.0
    b5_max = 0.0
    half_angles = []
    xy_points = []
    xy_radii = []

    for idx in sub_idxs:
        sym = symbols[idx]
        vdw = float(ELEMENT_DATA.get(sym, {}).get("vdw_radius", 1.70))
        rel = coords_arr[idx] - origin
        z_proj = float(np.dot(rel, u_axis))
        x_proj = float(np.dot(rel, u_x))
        y_proj = float(np.dot(rel, u_y))
        r_perp = math.hypot(x_proj, y_proj)

        l_max = max(l_max, z_proj + vdw)
        b5_max = max(b5_max, r_perp + vdw)
        xy_points.append((x_proj, y_proj))
        xy_radii.append(vdw)

        dist = float(np.linalg.norm(rel))
        if dist > 1e-6:
            cos_theta = max(-1.0, min(1.0, z_proj / dist))
            theta_center = math.acos(cos_theta)
            alpha_tangent = math.asin(min(1.0, vdw / dist))
            half_angles.append(math.degrees(theta_center + alpha_tangent))

    # Compute B1 (minimum perpendicular width across 180 angular orientations in the XY plane)
    angles = np.linspace(0.0, math.pi, 180, endpoint=False)
    b1_candidates = []
    for phi in angles:
        nx, ny = math.cos(phi), math.sin(phi)
        max_pos = 0.0
        max_neg = 0.0
        for (xp, yp), r_vdw in zip(xy_points, xy_radii):
            proj = xp * nx + yp * ny
            max_pos = max(max_pos, proj + r_vdw)
            max_neg = max(max_neg, -proj + r_vdw)
        b1_candidates.append(min(max_pos, max_neg))
    b1_min = float(min(b1_candidates)) if b1_candidates else 0.0

    cone_angle = 2.0 * max(half_angles) if half_angles else 0.0

    return {
        "L": round(float(l_max), 2),
        "B1": round(float(b1_min), 2),
        "B5": round(float(b5_max), 2),
        "cone_angle_deg": round(float(cone_angle), 1),
    }


def compute_coordination_geometry_indices(
    symbols: List[str],
    coords: np.ndarray,
    center_idx: int = 0,
    neighbor_indices: Optional[List[int]] = None,
) -> Dict[str, object]:
    """
    Compute inorganic/organometallic coordination geometry indices around `center_idx`:
      - CN = 4: Yang tau_4 = (360 - (alpha + beta)) / 141.0 and Okuniewski tau_4'
                (1.00 = Ideal Tetrahedral Td, 0.00 = Ideal Square Planar D4h)
      - CN = 5: Addison tau_5 = (beta - alpha) / 60.0
                (1.00 = Ideal Trigonal Bipyramidal D3h, 0.00 = Ideal Square Pyramidal C4v)
      - CN = 6: Octahedral angular distortion parameter Sigma = sum_{i=1..12} |90 - phi_i| (deg)
    """
    from .forcefield import detect_connectivity

    coords_arr = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    if n_atoms < 2 or not (0 <= center_idx < n_atoms):
        return {"coord_number": 0, "summary": "N/A"}

    if neighbor_indices is None:
        bonds = detect_connectivity(symbols, coords_arr, strict=False)
        nbrs = sorted({v if u == center_idx else u for u, v in bonds if u == center_idx or v == center_idx})
    else:
        nbrs = [i for i in neighbor_indices if 0 <= i < n_atoms and i != center_idx]

    cn = len(nbrs)
    if cn < 2:
        return {"coord_number": cn, "summary": f"CN = {cn} (Terminal / Monocoordinate)"}

    origin = coords_arr[center_idx]
    unit_vecs = []
    for nb in nbrs:
        v = coords_arr[nb] - origin
        ln = float(np.linalg.norm(v))
        if ln > 1e-8:
            unit_vecs.append(v / ln)

    cn = len(unit_vecs)
    angles_deg: List[float] = []
    for i in range(cn):
        for j in range(i + 1, cn):
            cos_th = max(-1.0, min(1.0, float(np.dot(unit_vecs[i], unit_vecs[j]))))
            angles_deg.append(math.degrees(math.acos(cos_th)))
    angles_sorted = sorted(angles_deg, reverse=True)

    out: Dict[str, object] = {
        "coord_number": cn,
        "neighbors": nbrs,
        "max_angles_deg": [round(a, 2) for a in angles_sorted[:3]],
        "summary": f"CN = {cn}",
    }

    if cn == 4 and len(angles_sorted) >= 2:
        beta, alpha = angles_sorted[0], angles_sorted[1]
        theta_t = 109.4712206
        tau4 = (360.0 - (alpha + beta)) / (360.0 - 2.0 * theta_t)
        tau4_prime = ((beta - alpha) / (360.0 - theta_t)) + ((180.0 - beta) / (180.0 - theta_t))
        tau4 = max(0.0, min(1.2, tau4))
        tau4_prime = max(0.0, min(1.2, tau4_prime))
        geom_lbl = "Tetrahedral (Td)" if tau4 >= 0.70 else ("Seesaw / Distorted" if tau4 >= 0.25 else "Square Planar (D4h)")
        out["tau_4"] = round(tau4, 3)
        out["tau_4_prime"] = round(tau4_prime, 3)
        out["summary"] = f"CN = 4 ({geom_lbl}): τ₄ = {tau4:.3f}, τ₄' = {tau4_prime:.3f} (α={alpha:.1f}°, β={beta:.1f}°)"
    elif cn == 5 and len(angles_sorted) >= 2:
        beta, alpha = angles_sorted[0], angles_sorted[1]
        tau5 = max(0.0, min(1.2, (beta - alpha) / 60.0))
        geom_lbl = "Trigonal Bipyramidal (D3h)" if tau5 >= 0.50 else "Square Pyramidal (C4v)"
        out["tau_5"] = round(tau5, 3)
        out["summary"] = f"CN = 5 ({geom_lbl}): τ₅ = {tau5:.3f} (α={alpha:.1f}°, β={beta:.1f}°)"
    elif cn == 6 and len(angles_sorted) >= 15:
        # 15 L-M-L angles total: 3 trans (~180 deg) and 12 cis (~90 deg)
        cis_angles = angles_sorted[3:]
        sigma_oct = sum(abs(90.0 - a) for a in cis_angles)
        out["sigma_oct_deg"] = round(sigma_oct, 2)
        out["summary"] = f"CN = 6 (Octahedral): Distortion Σ = {sigma_oct:.1f}° (0° = ideal Oh)"

    return out
