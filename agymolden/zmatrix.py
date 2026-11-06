"""Z-Matrix data structures, parsing, conversion to Cartesian coordinates, and file I/O."""

import re
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


@dataclass
class ZAtom:
    symbol: str
    ref1: Optional[int] = None       # 0-indexed index of bond partner
    dist: float = 0.0                # bond distance in Angstroms
    dist_var: Optional[str] = None   # optional variable name
    ref2: Optional[int] = None       # 0-indexed index of angle partner
    angle: float = 0.0               # angle in degrees
    angle_var: Optional[str] = None  # optional variable name
    ref3: Optional[int] = None       # 0-indexed index of dihedral partner
    dihedral: float = 0.0            # dihedral in degrees
    dihedral_var: Optional[str] = None # optional variable name


def zmatrix_to_cartesian(zatoms: List[ZAtom]) -> np.ndarray:
    """
    Convert a list of Z-matrix atoms into 3D Cartesian coordinates (N x 3 np.ndarray).
    Standard Z-matrix conventions:
      - Atom 0 is placed at (0, 0, 0)
      - Atom 1 is placed along +Z or +X from Atom 0 at dist
      - Atom 2 is placed in the XZ or XY plane at (dist, angle) relative to ref1, ref2
      - Atom >= 3 is placed via standard NeRF / vector reconstruction using ref1, ref2, ref3
    """
    n = len(zatoms)
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    coords = np.zeros((n, 3), dtype=np.float64)
    if n == 1:
        return coords

    # Atom 1: relative to ref1
    r0 = coords[zatoms[1].ref1 if zatoms[1].ref1 is not None else 0]
    coords[1] = r0 + np.array([0.0, 0.0, zatoms[1].dist])

    if n == 2:
        return coords

    # Atom 2: relative to ref1, ref2
    r_bond = zatoms[2].dist
    a_deg = zatoms[2].angle
    a_rad = math.radians(a_deg)
    ref1_idx = zatoms[2].ref1 if zatoms[2].ref1 is not None else 0
    ref2_idx = zatoms[2].ref2 if zatoms[2].ref2 is not None else 1

    p1 = coords[ref1_idx]
    p2 = coords[ref2_idx]

    # Vector along bond p2 -> p1
    v21 = p1 - p2
    norm21 = np.linalg.norm(v21)
    if norm21 < 1e-7:
        u_z = np.array([0.0, 0.0, 1.0])
    else:
        u_z = v21 / norm21

    # Choose perpendicular unit vector
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(u_z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u_x = np.cross(u_z, arbitrary)
    u_x /= np.linalg.norm(u_x)
    u_y = np.cross(u_z, u_x)

    # Place atom 2 at angle from u_z in u_x direction
    # angle is angle(atom2, p1, p2)
    # So vector from p1 to atom2 has:
    # component along -u_z is cos(180 - a) = -cos(a), or along -u_z
    coords[2] = p1 + r_bond * (-np.cos(a_rad) * u_z + np.sin(a_rad) * u_x)

    # Atom 3 to N
    for i in range(3, n):
        atom = zatoms[i]
        i1 = atom.ref1 if atom.ref1 is not None else (i - 1)
        i2 = atom.ref2 if atom.ref2 is not None else (i - 2)
        i3 = atom.ref3 if atom.ref3 is not None else (i - 3)

        p1 = coords[i1]
        p2 = coords[i2]
        p3 = coords[i3]

        r = atom.dist
        theta = math.radians(atom.angle)
        phi = math.radians(atom.dihedral)

        # Standard NeRF / Natural Extension of Reference Frame construction:
        # bc = p1 - p2 (vector along bond 2 -> 1)
        # ab = p2 - p3 (vector along bond 3 -> 2)
        bc = p1 - p2
        l_bc = np.linalg.norm(bc)
        if l_bc < 1e-7:
            bc = np.array([0.0, 0.0, 1.0])
        else:
            bc /= l_bc

        # Normal vector to plane 3-2-1
        n_plane = np.cross(p2 - p3, bc)
        l_np = np.linalg.norm(n_plane)
        if l_np < 1e-7:
            # Collinear, pick perpendicular
            arb = np.array([1.0, 0.0, 0.0]) if abs(bc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n_plane = np.cross(arb, bc)
            l_np = np.linalg.norm(n_plane)
            if l_np < 1e-7:
                n_plane = np.array([0.0, 1.0, 0.0])
            else:
                n_plane /= l_np
        else:
            n_plane /= l_np

        n_x = np.cross(n_plane, bc)

        # Vector pointing from p1 to new atom:
        # Angle theta is angle between (new_atom - p1) and (p2 - p1) = -bc.
        # So projection onto (-bc) is cos(theta) => projection onto bc is -cos(theta).
        # Perpendicular component in plane is sin(theta).
        # Rotation around bc by dihedral angle phi:
        # phi = 0 corresponds to cis to p3 (trans is 180).
        # Standard right-handed NeRF formulation:
        d_vec = (
            -np.cos(theta) * bc
            + np.sin(theta) * np.cos(phi) * n_x
            + np.sin(theta) * np.sin(phi) * n_plane
        )

        coords[i] = p1 + r * d_vec

    return coords


def calculate_bond_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculate Euclidean distance between two 3D points."""
    return float(np.linalg.norm(p1 - p2))


def calculate_angle(p_i: np.ndarray, p_ref1: np.ndarray, p_ref2: np.ndarray) -> float:
    """Calculate angle in degrees for p_i - p_ref1 - p_ref2."""
    v1 = p_i - p_ref1
    v2 = p_ref2 - p_ref1
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = np.dot(v1, v2) / (n1 * n2)
    c = np.clip(c, -1.0, 1.0)
    return float(math.degrees(math.acos(c)))


def calculate_dihedral(p_i: np.ndarray, p_ref1: np.ndarray, p_ref2: np.ndarray, p_ref3: np.ndarray) -> float:
    """Calculate dihedral angle in degrees for p_i - p_ref1 - p_ref2 - p_ref3."""
    p0 = p_ref3
    p1 = p_ref2
    p2 = p_ref1
    p3 = p_i
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm < 1e-12:
        return 0.0
    b1_u = b1 / b1_norm
    v = b0 - np.dot(b0, b1_u) * b1_u
    w = b2 - np.dot(b2, b1_u) * b1_u
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_u, v), w)
    return float(math.degrees(math.atan2(y, x)))


def cartesian_to_zmatrix(
    symbols: List[str],
    coords: np.ndarray,
    existing_zatoms: Optional[List[ZAtom]] = None,
    user_bonds: Optional[List[Tuple[int, int]]] = None,
    strict_bonds: bool = False
) -> List[ZAtom]:
    """
    Build a connectivity-aware canonical Z-matrix from 3D Cartesian coordinates.
    Hydrogens and substituent atoms bond to their true chemical parent atom
    rather than daisy-chaining sequentially.
    If existing_zatoms is provided and its length and symbols match, preserves the existing
    reference structure (ref1, ref2, ref3) and variable names, only updating the numerical
    bond distances, angles, and dihedrals from coords.
    """
    from .forcefield import detect_connectivity

    n = len(symbols)
    zatoms: List[ZAtom] = []
    if n == 0:
        return zatoms

    # If existing zatoms match, attempt in-place reference preservation
    if existing_zatoms is not None and len(existing_zatoms) == n:
        matches = all(existing_zatoms[k].symbol == symbols[k] for k in range(n))
        if matches:
            # Check validity: each atom k must only reference prior atoms (< k)
            valid_refs = True
            for k in range(n):
                a = existing_zatoms[k]
                if k == 1 and (a.ref1 is None or a.ref1 >= 1):
                    valid_refs = False; break
                elif k == 2 and (a.ref1 is None or a.ref1 >= 2 or a.ref2 is None or a.ref2 >= 2):
                    valid_refs = False; break
                elif k >= 3 and (a.ref1 is None or a.ref1 >= k or a.ref2 is None or a.ref2 >= k or a.ref3 is None or a.ref3 >= k):
                    valid_refs = False; break

            if valid_refs:
                for k in range(n):
                    old_a = existing_zatoms[k]
                    new_a = ZAtom(symbol=symbols[k])
                    if k >= 1:
                        new_a.ref1 = old_a.ref1
                        new_a.dist = calculate_bond_distance(coords[k], coords[old_a.ref1])
                        new_a.dist_var = old_a.dist_var
                    if k >= 2:
                        new_a.ref2 = old_a.ref2
                        new_a.angle = calculate_angle(coords[k], coords[old_a.ref1], coords[old_a.ref2])
                        new_a.angle_var = old_a.angle_var
                    if k >= 3:
                        new_a.ref3 = old_a.ref3
                        new_a.dihedral = calculate_dihedral(coords[k], coords[old_a.ref1], coords[old_a.ref2], coords[old_a.ref3])
                        new_a.dihedral_var = old_a.dihedral_var
                    zatoms.append(new_a)
                return zatoms

    bonds = detect_connectivity(symbols, coords, user_bonds=user_bonds, strict=strict_bonds)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)

    for i in range(n):
        if i == 0:
            zatoms.append(ZAtom(symbol=symbols[0]))
        elif i == 1:
            prior_bonded = [j for j in adj[i] if j < i]
            if prior_bonded:
                # Prefer heavy atoms over H, and closer distance
                prior_bonded.sort(key=lambda j: (0 if symbols[j] != "H" else 1, calculate_bond_distance(coords[i], coords[j])))
                ref1 = prior_bonded[0]
            else:
                ref1 = 0
            d = calculate_bond_distance(coords[i], coords[ref1])
            zatoms.append(ZAtom(symbol=symbols[1], ref1=ref1, dist=d))
        elif i == 2:
            prior_bonded = [j for j in adj[i] if j < i]
            if prior_bonded:
                prior_bonded.sort(key=lambda j: (0 if symbols[j] != "H" else 1, calculate_bond_distance(coords[i], coords[j])))
                ref1 = prior_bonded[0]
            else:
                ref1 = 1
            # ref2: prefer bonded to ref1
            cands = [j for j in range(i) if j != ref1]
            cands.sort(key=lambda j: (0 if j in adj[ref1] else 1, abs(j - ref1)))
            ref2 = cands[0]
            d = calculate_bond_distance(coords[i], coords[ref1])
            ang = calculate_angle(coords[i], coords[ref1], coords[ref2])
            zatoms.append(ZAtom(symbol=symbols[2], ref1=ref1, dist=d, ref2=ref2, angle=ang))
        else:
            prior_bonded = [j for j in adj[i] if j < i]
            if prior_bonded:
                prior_bonded.sort(key=lambda j: (0 if symbols[j] != "H" else 1, calculate_bond_distance(coords[i], coords[j])))
                ref1 = prior_bonded[0]
            else:
                # Fallback to closest prior atom
                prior_atoms = list(range(i))
                prior_atoms.sort(key=lambda j: (0 if symbols[j] != "H" else 1, calculate_bond_distance(coords[i], coords[j])))
                ref1 = prior_atoms[0]

            # ref2: candidate from < i, distinct from ref1. Prefer bonded to ref1 and non-collinear
            cands2 = [j for j in range(i) if j != ref1]
            def angle_score2(j: int):
                ang = calculate_angle(coords[i], coords[ref1], coords[j])
                is_bonded = 0 if j in adj[ref1] else 1
                collinear = 10 if (ang < 5.0 or ang > 175.0) else 0
                return (collinear, is_bonded, abs(j - ref1))
            cands2.sort(key=angle_score2)
            ref2 = cands2[0]

            # ref3: candidate from < i, distinct from ref1 and ref2. Prefer bonded to ref2/ref1 and non-collinear
            cands3 = [j for j in range(i) if j not in (ref1, ref2)]
            def angle_score3(j: int):
                ang = calculate_angle(coords[ref1], coords[ref2], coords[j])
                is_bonded = 0 if j in adj[ref2] else (1 if j in adj[ref1] else 2)
                collinear = 10 if (ang < 5.0 or ang > 175.0) else 0
                return (collinear, is_bonded, abs(j - ref2))
            cands3.sort(key=angle_score3)
            ref3 = cands3[0]

            d = calculate_bond_distance(coords[i], coords[ref1])
            ang = calculate_angle(coords[i], coords[ref1], coords[ref2])
            dih = calculate_dihedral(coords[i], coords[ref1], coords[ref2], coords[ref3])
            zatoms.append(ZAtom(symbol=symbols[i], ref1=ref1, dist=d, ref2=ref2, angle=ang, ref3=ref3, dihedral=dih))

    return zatoms


def parse_zmatrix_text(text: str) -> Tuple[List[ZAtom], Dict[str, float]]:
    """
    Parse Molden/Gaussian style Z-matrix text including variable definitions.
    Example:
      C
      O  1 1.22
      H  1 1.09 2 120.0
      H  1 1.09 2 120.0 3 180.0
    or with variables:
      C
      O 1 rCO
      ...
      Variables:
      rCO = 1.22
    """
    raw_lines = [line.strip() for line in text.strip().splitlines() if not line.strip().startswith("#")]
    zatoms: List[ZAtom] = []
    variables: Dict[str, float] = {}

    atom_lines: List[str] = []
    var_lines: List[str] = []
    parsing_vars = False

    def looks_like_var_line(line: str) -> bool:
        toks = line.replace("=", " ").replace(",", " ").split()
        if len(toks) != 2 or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", toks[0]):
            return False
        try:
            float(toks[1])
        except ValueError:
            return False
        return "." in toks[1] or "=" in line

    for line in raw_lines:
        if not line:
            # Gaussian style: blank line separates atoms from variables
            if atom_lines:
                parsing_vars = True
            continue
        lower = line.lower()
        if lower.startswith("variables") or lower.startswith("constants") or "=" in line:
            parsing_vars = True
            if "=" in line:
                var_lines.append(line)
            continue
        if parsing_vars or (atom_lines and looks_like_var_line(line)):
            parsing_vars = True
            var_lines.append(line)
        else:
            atom_lines.append(line)

    # Parse variables first
    for line in var_lines:
        line_clean = line.replace("=", " ").replace(",", " ")
        tokens = line_clean.split()
        if len(tokens) >= 2:
            var_name = tokens[0].strip()
            try:
                var_val = float(tokens[1])
                variables[var_name] = var_val
            except ValueError:
                pass

    def resolve_val(token: str) -> float:
        try:
            return float(token)
        except ValueError:
            base, sgn = _split_var(token)
            return sgn * variables.get(base or "", 0.0)

    def var_name_of(token: str) -> Optional[str]:
        base, _ = _split_var(token)
        return token if (base and base in variables) else None

    atom_labels: Dict[str, int] = {}  # Map label string (e.g. 'C1', 'O2', '1') to 0-based index

    def resolve_ref(token: str, current_idx: int) -> int:
        clean = token.strip()
        # Direct integer
        try:
            val = int(clean)
            return max(0, val - 1)
        except ValueError:
            pass
        # Named label
        if clean in atom_labels:
            return atom_labels[clean]
        # Element symbol only or fallback
        for lbl, idx in atom_labels.items():
            if lbl.lower() == clean.lower():
                return idx
        return max(0, current_idx - 1)

    for i, line in enumerate(atom_lines):
        # Replace commas with whitespace
        tokens = line.replace(",", " ").split()
        if not tokens:
            continue
        raw_sym = tokens[0]

        # Extract pure element symbol (e.g., C1 -> C, Cl2 -> Cl, Fe -> Fe)
        match = re.match(r"^([A-Za-z]{1,2})\d*$", raw_sym)
        pure_sym = match.group(1).capitalize() if match else raw_sym[:2].capitalize()
        atom_labels[raw_sym] = i
        atom_labels[str(i + 1)] = i

        atom = ZAtom(symbol=pure_sym)
        if len(tokens) >= 3:
            # ref1 dist
            atom.ref1 = resolve_ref(tokens[1], i)
            atom.dist_var = var_name_of(tokens[2])
            atom.dist = resolve_val(tokens[2])

        if len(tokens) >= 5:
            # ref2 angle
            atom.ref2 = resolve_ref(tokens[3], i)
            atom.angle_var = var_name_of(tokens[4])
            atom.angle = resolve_val(tokens[4])

        if len(tokens) >= 7:
            # ref3 dihedral
            atom.ref3 = resolve_ref(tokens[5], i)
            atom.dihedral_var = var_name_of(tokens[6])
            atom.dihedral = resolve_val(tokens[6])

        zatoms.append(atom)

    return zatoms, variables


def format_zmatrix_text(zatoms: List[ZAtom], variables: Optional[Dict[str, float]] = None) -> str:
    """Format Z-matrix into Molden / Gaussian readable string."""
    lines = []
    for i, a in enumerate(zatoms):
        sym = a.symbol
        if i == 0:
            lines.append(f"{sym:<4}")
        elif i == 1:
            ref1 = (a.ref1 + 1) if a.ref1 is not None else 1
            d_str = a.dist_var if a.dist_var else f"{a.dist:8.4f}"
            lines.append(f"{sym:<4}  {ref1:>3}  {d_str}")
        elif i == 2:
            ref1 = (a.ref1 + 1) if a.ref1 is not None else 1
            d_str = a.dist_var if a.dist_var else f"{a.dist:8.4f}"
            ref2 = (a.ref2 + 1) if a.ref2 is not None else 2
            a_str = a.angle_var if a.angle_var else f"{a.angle:8.3f}"
            lines.append(f"{sym:<4}  {ref1:>3}  {d_str}  {ref2:>3}  {a_str}")
        else:
            ref1 = (a.ref1 + 1) if a.ref1 is not None else 1
            d_str = a.dist_var if a.dist_var else f"{a.dist:8.4f}"
            ref2 = (a.ref2 + 1) if a.ref2 is not None else 2
            a_str = a.angle_var if a.angle_var else f"{a.angle:8.3f}"
            ref3 = (a.ref3 + 1) if a.ref3 is not None else 3
            di_str = a.dihedral_var if a.dihedral_var else f"{a.dihedral:8.3f}"
            lines.append(f"{sym:<4}  {ref1:>3}  {d_str}  {ref2:>3}  {a_str}  {ref3:>3}  {di_str}")

    if variables:
        lines.append("\nVariables:")
        for k, v in variables.items():
            lines.append(f"{k} = {v:.4f}")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Variables (symbolic Z-matrix) support
# ----------------------------------------------------------------------
_FIELDS = (("dist", "dist_var"), ("angle", "angle_var"), ("dihedral", "dihedral_var"))


def _split_var(name: Optional[str]) -> Tuple[Optional[str], float]:
    """'-D1' -> ('D1', -1.0); 'R1' -> ('R1', 1.0)."""
    if not name:
        return None, 1.0
    if name.startswith("-"):
        return name[1:], -1.0
    if name.startswith("+"):
        return name[1:], 1.0
    return name, 1.0


def format_zmatrix_numeric(zatoms: List[ZAtom]) -> str:
    """Z-matrix with explicit numbers for every bond, angle and dihedral (no variables)."""
    plain = [ZAtom(a.symbol, a.ref1, a.dist, None, a.ref2, a.angle, None, a.ref3, a.dihedral, None)
             for a in zatoms]
    return format_zmatrix_text(plain, None)


def format_zmatrix_variables(zatoms: List[ZAtom], variables: Optional[Dict[str, float]] = None) -> str:
    """Z-matrix with a variable name for each coordinate and a 'Variables:' block with values.
    Coordinates that already have a variable keep it (shared variables stay shared); every
    other coordinate gets its own new variable."""
    variables = dict(variables or {})
    out: List[ZAtom] = []
    counters = {"dist": 0, "angle": 0, "dihedral": 0}
    prefix = {"dist": "R", "angle": "A", "dihedral": "D"}

    def new_name(kind: str) -> str:
        while True:
            counters[kind] += 1
            nm = f"{prefix[kind]}{counters[kind]}"
            if nm not in variables:
                return nm

    for i, a in enumerate(zatoms):
        b = ZAtom(a.symbol, a.ref1, a.dist, a.dist_var, a.ref2, a.angle, a.angle_var,
                  a.ref3, a.dihedral, a.dihedral_var)
        nfields = min(i, 3)
        for k, (val_attr, var_attr) in enumerate(_FIELDS[:nfields]):
            base, _sgn = _split_var(getattr(b, var_attr))
            if base is None or base not in variables:
                nm = new_name(val_attr)
                variables[nm] = getattr(b, val_attr)
                setattr(b, var_attr, nm)
        out.append(b)
    used = []
    for b in out:
        for _v, var_attr in _FIELDS:
            base, _ = _split_var(getattr(b, var_attr))
            if base and base not in used:
                used.append(base)
    return format_zmatrix_text(out, {k: variables[k] for k in used})


def apply_variables(zatoms: List[ZAtom], variables: Dict[str, float]) -> None:
    """Set the numeric values of every coordinate that references a variable."""
    for a in zatoms:
        for val_attr, var_attr in _FIELDS:
            base, sgn = _split_var(getattr(a, var_attr))
            if base is not None and base in variables:
                setattr(a, val_attr, sgn * variables[base])


def sync_variables_from_atoms(zatoms: List[ZAtom], variables: Dict[str, float]) -> Dict[str, float]:
    """Update each variable to the average (sign-corrected) value of the coordinates using it,
    then push that value back so linked coordinates stay identical. Unused variables are dropped."""
    acc: Dict[str, List[float]] = {}
    for a in zatoms:
        for val_attr, var_attr in _FIELDS:
            base, sgn = _split_var(getattr(a, var_attr))
            if base is not None:
                acc.setdefault(base, []).append(sgn * getattr(a, val_attr))
    new_vars = {}
    for k in list(variables.keys()) + [k for k in acc if k not in variables]:
        if k in acc:
            new_vars[k] = float(np.mean(acc[k]))
    apply_variables(zatoms, new_vars)
    return new_vars


def assign_symmetric_variables(zatoms: List[ZAtom], tol_dist: float = 0.01, tol_ang: float = 0.5,
                               include_angles: bool = True, include_dihedrals: bool = True) -> Dict[str, float]:
    """
    Replace numbers with variables, sharing one variable among symmetry-equivalent coordinates
    (same atom types and same value within tolerance). E.g. in benzene all C-C bonds share R1,
    all C-H bonds share R2, which reduces the number of parameters to optimize.
    If include_angles is False, angles are kept as explicit numerical constants.
    If include_dihedrals is False, dihedrals are kept as explicit numerical constants.
    Dihedrals of opposite sign share a variable using the '-D' notation. Dihedrals of 0/180
    get a variable too (they are kept constant by symmetry). Returns the variables dict.
    """
    variables: Dict[str, float] = {}
    groups: Dict[str, List[Tuple[tuple, float, str]]] = {"dist": [], "angle": [], "dihedral": []}
    prefix = {"dist": "R", "angle": "A", "dihedral": "D"}
    syms = [a.symbol for a in zatoms]

    def find_or_create(kind: str, key: tuple, value: float, tol: float, allow_neg: bool):
        for gkey, gval, name in groups[kind]:
            if gkey != key:
                continue
            if abs(gval - value) <= tol:
                return name
            if allow_neg and abs(gval + value) <= tol and abs(value) > tol:
                return "-" + name
        name = f"{prefix[kind]}{len(groups[kind]) + 1}"
        groups[kind].append((key, value, name))
        variables[name] = value
        return name

    for i, a in enumerate(zatoms):
        if i >= 1 and a.ref1 is not None and 0 <= a.ref1 < len(syms):
            key = tuple(sorted((syms[i], syms[a.ref1])))
            a.dist_var = find_or_create("dist", key, a.dist, tol_dist, False)
        if i >= 2 and a.ref2 is not None and a.ref1 is not None and 0 <= a.ref2 < len(syms) and 0 <= a.ref1 < len(syms):
            if include_angles:
                ends = sorted((syms[i], syms[a.ref2]))
                key = (ends[0], syms[a.ref1], ends[1])
                a.angle_var = find_or_create("angle", key, a.angle, tol_ang, False)
            else:
                a.angle_var = None
        if i >= 3 and a.ref3 is not None and a.ref1 is not None and 0 <= a.ref1 < len(syms):
            if include_dihedrals:
                key = (syms[i], syms[a.ref1])  # same moving atom type on same center type
                a.dihedral_var = find_or_create("dihedral", key, a.dihedral, tol_ang, True)
            else:
                a.dihedral_var = None
    # Snap linked coordinates to the shared value
    apply_variables(zatoms, variables)
    return variables


def clear_variables(zatoms: List[ZAtom]) -> None:
    for a in zatoms:
        a.dist_var = a.angle_var = a.dihedral_var = None


def _normalize_xyz_token(token: str) -> Optional[str]:
    """
    Convert an XYZ atom token (element symbol like 'C', 'CL', 'Fe', 'C1', or atomic number '6')
    into a canonical element symbol ('C', 'Cl', 'Fe', 'X'). Returns None if invalid.
    """
    from .elements import ELEMENT_DATA, normalize_symbol
    t = token.strip()
    if not t:
        return None
    # Check if integer atomic number (e.g. '6' -> 'C', '0' -> 'X')
    try:
        z_num = int(t)
        if z_num == 0:
            return "X"
        for sym, data in ELEMENT_DATA.items():
            if data.get("atomic_number") == z_num:
                return sym
        return None
    except ValueError:
        pass

    # Strip trailing digits/labels (e.g. 'C1' -> 'C', 'Cl2' -> 'Cl', 'Bq' -> 'X')
    m = re.match(r"^([A-Za-z]{1,2})\d*$", t)
    if not m:
        return None
    cand = normalize_symbol(m.group(1))
    if cand in ("Bq", "Du", "Xx"):
        return "X"
    if cand in ELEMENT_DATA:
        return cand
    return None


def _parse_xyz_float(token: str) -> float:
    """Parse float supporting standard and Fortran 'D'/'d' scientific notation."""
    return float(token.replace("D", "e").replace("d", "e"))


def parse_xyz(filepath_or_content: str, is_path: bool = True) -> Tuple[List[str], np.ndarray]:
    """
    Read a standard or headerless XYZ format molecule.
    Properly handles:
      - Blank comment lines on line 2 (without shifting or dropping Atom 1!)
      - Leading blank lines before the atom count header
      - Atomic numbers (e.g. 6 -> C, 1 -> H) as well as element symbols and numbered labels (C1, Fe2)
      - Fortran D/d exponents (e.g. 1.234D-01)
      - Multi-frame XYZ files (reads the first complete frame cleanly)
    """
    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    else:
        content = filepath_or_content

    raw_lines = content.splitlines()
    # Skip leading completely blank lines while preserving internal blank lines (such as the comment line!)
    start_idx = 0
    while start_idx < len(raw_lines) and not raw_lines[start_idx].strip():
        start_idx += 1
    if start_idx >= len(raw_lines):
        return [], np.empty((0, 3), dtype=np.float64)

    first_line = raw_lines[start_idx].strip()
    first_tokens = first_line.split()

    # Check if first non-empty line is the XYZ atom count header (single integer N > 0)
    n_atoms: Optional[int] = None
    if len(first_tokens) == 1:
        try:
            val = int(first_tokens[0])
            if val > 0:
                n_atoms = val
        except ValueError:
            n_atoms = None

    if n_atoms is not None:
        # Line `start_idx` is atom count; line `start_idx + 1` is comment line (even if blank!)
        candidate_lines = raw_lines[start_idx + 2:]
    else:
        # Headerless XYZ (only atom coordinate lines)
        candidate_lines = raw_lines[start_idx:]

    symbols: List[str] = []
    coords: List[List[float]] = []

    for line in candidate_lines:
        s_line = line.strip()
        if not s_line or s_line.startswith("#"):
            continue
        tokens = s_line.replace(",", " ").split()
        if len(tokens) < 4:
            # If we already read atoms in a headerless file and hit a non-coordinate line, stop
            if n_atoms is None and symbols:
                break
            continue
        sym = _normalize_xyz_token(tokens[0])
        if sym is None:
            continue
        try:
            x = _parse_xyz_float(tokens[1])
            y = _parse_xyz_float(tokens[2])
            z = _parse_xyz_float(tokens[3])
        except ValueError:
            continue
        symbols.append(sym)
        coords.append([x, y, z])
        if n_atoms is not None and len(symbols) >= n_atoms:
            break

    if not symbols:
        return [], np.empty((0, 3), dtype=np.float64)
    return symbols, np.array(coords, dtype=np.float64)


def parse_multi_xyz(filepath_or_content: str, is_path: bool = True) -> Tuple[List[str], List[Tuple[str, np.ndarray]]]:
    """
    Parse a multi-frame trajectory XYZ file into (symbols, [(comment, coords_array), ...]).
    Each frame contains an (N, 3) coordinate array.
    """
    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    else:
        content = filepath_or_content

    lines = content.splitlines()
    frames: List[Tuple[str, np.ndarray]] = []
    base_symbols: List[str] = []
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        s = lines[i].strip()
        if not s:
            i += 1
            continue
        try:
            n_atoms = int(s)
        except ValueError:
            i += 1
            continue

        comment = lines[i + 1].strip() if i + 1 < n_lines else ""
        frame_coords = []
        frame_syms = []

        for line_idx in range(i + 2, min(i + 2 + n_atoms, n_lines)):
            toks = lines[line_idx].strip().replace(",", " ").split()
            if len(toks) >= 4:
                sym = _normalize_xyz_token(toks[0])
                if sym:
                    try:
                        x = _parse_xyz_float(toks[1])
                        y = _parse_xyz_float(toks[2])
                        z = _parse_xyz_float(toks[3])
                        frame_syms.append(sym)
                        frame_coords.append([x, y, z])
                    except ValueError:
                        pass

        if len(frame_coords) == n_atoms:
            if not base_symbols:
                base_symbols = frame_syms
            frames.append((comment, np.array(frame_coords, dtype=np.float64)))
            i += 2 + n_atoms
        else:
            i += 1

    return base_symbols, frames


def write_xyz(filepath: str, symbols: List[str], coords: np.ndarray, comment: str = "Generated by AgyMolden"):
    """Export coordinates to XYZ file."""
    with open(filepath, "w") as f:
        f.write(f"{len(symbols)}\n")
        f.write(f"{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords):
            f.write(f"{sym:<4} {x:12.6f} {y:12.6f} {z:12.6f}\n")


def parse_pdb(filepath_or_content: str, is_path: bool = True) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]]]:
    """Parse a Protein Data Bank (PDB) file or string, returning (symbols, coords, bonds)."""
    from .elements import normalize_symbol

    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = filepath_or_content.splitlines()

    symbols: List[str] = []
    coords_list: List[Tuple[float, float, float]] = []
    bonds_set = set()
    serial_to_idx: Dict[int, int] = {}

    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                serial = int(line[6:11].strip())
            except ValueError:
                serial = len(symbols) + 1

            sym_str = line[76:78].strip() if len(line) >= 78 else ""
            if not sym_str:
                raw_name = line[12:16].strip()
                sym_str = re.sub(r'[^A-Za-z]', '', raw_name)
            sym = normalize_symbol(sym_str) or "C"

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue

            curr_idx = len(symbols)
            serial_to_idx[serial] = curr_idx
            symbols.append(sym)
            coords_list.append((x, y, z))

        elif line.startswith("CONECT"):
            tokens = line[6:].split()
            if not tokens:
                continue
            try:
                src_serial = int(tokens[0])
                if src_serial not in serial_to_idx:
                    continue
                src_idx = serial_to_idx[src_serial]
                for tok in tokens[1:]:
                    dst_serial = int(tok)
                    if dst_serial in serial_to_idx:
                        dst_idx = serial_to_idx[dst_serial]
                        if src_idx != dst_idx:
                            bonds_set.add((min(src_idx, dst_idx), max(src_idx, dst_idx)))
            except ValueError:
                continue

    coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3), dtype=np.float64)
    return symbols, coords, sorted(list(bonds_set))


def write_pdb(
    filepath: str,
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    title: str = "Generated by AgyMolden",
) -> None:
    """Export molecular coordinates and connectivity to standard PDB format."""
    lines = [
        f"HEADER    {title[:40]:<40}",
        f"TITLE     {title}",
        "COMPND    MOL",
    ]
    for i, (sym, (x, y, z)) in enumerate(zip(symbols, coords)):
        atom_name = f"{sym}{i+1}"[:4]
        lines.append(
            f"HETATM{i+1:5d} {atom_name:<4s} MOL A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {sym:>2s}"
        )

    if bonds:
        conect_dict: Dict[int, List[int]] = {}
        for a, b in bonds:
            conect_dict.setdefault(a + 1, []).append(b + 1)
            conect_dict.setdefault(b + 1, []).append(a + 1)
        for a_serial in sorted(conect_dict.keys()):
            partners = conect_dict[a_serial]
            for chunk_start in range(0, len(partners), 4):
                chunk = partners[chunk_start : chunk_start + 4]
                rec = f"CONECT{a_serial:5d}" + "".join(f"{p:5d}" for p in chunk)
                lines.append(rec)

    lines.append("END\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_mol_sdf(
    filepath_or_content: str, is_path: bool = True
) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Parse MDL Molfile / SDF (V2000), returning (symbols, coords, bonds, bond_orders)."""
    from .elements import normalize_symbol

    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = filepath_or_content.splitlines()

    if len(lines) < 4:
        return [], np.empty((0, 3)), [], {}

    counts = lines[3]
    try:
        n_atoms = int(counts[:3].strip())
        n_bonds = int(counts[3:6].strip())
    except ValueError:
        return [], np.empty((0, 3)), [], {}

    symbols: List[str] = []
    coords_list: List[Tuple[float, float, float]] = []
    bonds: List[Tuple[int, int]] = []
    bond_orders: Dict[Tuple[int, int], int] = {}

    idx = 4
    for _ in range(n_atoms):
        if idx >= len(lines):
            break
        aline = lines[idx]
        try:
            x = float(aline[:10].strip())
            y = float(aline[10:20].strip())
            z = float(aline[20:30].strip())
            sym = normalize_symbol(aline[31:34].strip()) or "C"
            symbols.append(sym)
            coords_list.append((x, y, z))
        except (ValueError, IndexError):
            pass
        idx += 1

    for _ in range(n_bonds):
        if idx >= len(lines):
            break
        bline = lines[idx]
        try:
            a1 = int(bline[:3].strip()) - 1
            a2 = int(bline[3:6].strip()) - 1
            order = int(bline[6:9].strip())
            pair = (min(a1, a2), max(a1, a2))
            bonds.append(pair)
            bond_orders[pair] = order
        except (ValueError, IndexError):
            pass
        idx += 1

    coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3), dtype=np.float64)
    return symbols, coords, bonds, bond_orders


def write_mol_sdf(
    filepath: str,
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    title: str = "AgyMolden Molecular Workbench",
) -> None:
    """Export molecular coordinates and connectivity to MDL Molfile / SDF V2000 format."""
    n_atoms = len(symbols)
    n_bonds = len(bonds) if bonds else 0
    lines = [
        f"{title[:80]}",
        "  AgyMolden V2000 3D Export",
        "",
        f"{n_atoms:3d}{n_bonds:3d}  0  0  0  0  0  0  0  0999 V2000",
    ]
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{x:10.4f}{y:10.4f}{z:10.4f} {sym:<3s} 0  0  0  0  0  0  0  0  0  0  0  0")

    if bonds:
        for a, b in bonds:
            order = 1
            if bond_orders:
                order = bond_orders.get((min(a, b), max(a, b)), 1)
            lines.append(f"{a+1:3d}{b+1:3d}{order:3d}  0  0  0  0")

    lines.append("M  END")
    lines.append("$$$$\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_mol2(
    filepath_or_content: str, is_path: bool = True
) -> Tuple[List[str], np.ndarray, List[Tuple[int, int]], Dict[Tuple[int, int], int]]:
    """Parse Tripos Mol2 format, returning (symbols, coords, bonds, bond_orders)."""
    from .elements import normalize_symbol

    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    else:
        lines = filepath_or_content.splitlines()

    symbols: List[str] = []
    coords_list: List[List[float]] = []
    bonds: List[Tuple[int, int]] = []
    bond_orders: Dict[Tuple[int, int], int] = {}

    in_atom = False
    in_bond = False

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("@<TRIPOS>ATOM"):
            in_atom = True
            in_bond = False
            continue
        elif s.startswith("@<TRIPOS>BOND"):
            in_atom = False
            in_bond = True
            continue
        elif s.startswith("@<TRIPOS>"):
            in_atom = False
            in_bond = False
            continue

        if in_atom:
            parts = s.split()
            if len(parts) >= 5:
                # Format: atom_id atom_name x y z [atom_type [subst_id [subst_name [charge]]]]
                raw_sym = parts[5].split(".")[0] if len(parts) > 5 else parts[1]
                # Filter out numbers from atom names (e.g. C1 -> C)
                clean_sym = re.sub(r"[0-9]+", "", raw_sym).capitalize()
                sym = normalize_symbol(clean_sym) if normalize_symbol(clean_sym) else (clean_sym[:2] if len(clean_sym) >= 2 else clean_sym)
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                    symbols.append(sym if sym else "C")
                    coords_list.append([x, y, z])
                except ValueError:
                    continue

        elif in_bond:
            parts = s.split()
            if len(parts) >= 4:
                try:
                    b1 = int(parts[1]) - 1
                    b2 = int(parts[2]) - 1
                    btype = parts[3].lower()
                    order = 1
                    if btype == "2":
                        order = 2
                    elif btype == "3":
                        order = 3
                    elif btype in ("ar", "am"):
                        order = 1  # Standard aromatic representation
                    pair = (min(b1, b2), max(b1, b2))
                    bonds.append(pair)
                    bond_orders[pair] = order
                except ValueError:
                    continue

    coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3), dtype=np.float64)
    return symbols, coords, bonds, bond_orders


def write_mol2(
    filepath: str,
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    molecule_name: str = "MOL",
) -> None:
    """Export molecular coordinates and connectivity to Tripos Mol2 format."""
    n_atoms = len(symbols)
    n_bonds = len(bonds) if bonds else 0
    lines = [
        "@<TRIPOS>MOLECULE",
        f"{molecule_name[:40]}",
        f" {n_atoms:5d} {n_bonds:5d}     1     0     0",
        "SMALL",
        "NO_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]

    for i, (sym, (x, y, z)) in enumerate(zip(symbols, coords)):
        atom_name = f"{sym}{i+1}"
        atom_type = sym
        lines.append(f"{i+1:7d} {atom_name:<8s} {x:10.4f} {y:10.4f} {z:10.4f} {atom_type:<8s} 1 <1>       0.0000")

    if bonds:
        lines.append("@<TRIPOS>BOND")
        for bond_idx, (a, b) in enumerate(bonds):
            order = 1
            if bond_orders:
                order = bond_orders.get((min(a, b), max(a, b)), 1)
            btype_str = str(order) if order in (1, 2, 3) else "1"
            lines.append(f"{bond_idx+1:6d} {a+1:5d} {b+1:5d} {btype_str:>4s}")

    lines.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_cif(
    filepath_or_content: str, is_path: bool = True
) -> Tuple[List[str], np.ndarray, Dict[str, float]]:
    """
    Parse Crystallographic Information File (CIF) returning (symbols, cartesian_coords, cell_parameters).
    Converts fractional coordinates to Cartesian Angstroms via the crystallographic metric tensor.
    """
    from .elements import normalize_symbol

    if is_path:
        with open(filepath_or_content, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = filepath_or_content

    # Helper to strip ESD parentheses: e.g. "10.234(5)" -> 10.234
    def _clean_num(val_str: str) -> float:
        cleaned = re.sub(r'\([0-9]+\)', '', val_str.strip())
        return float(cleaned)

    # 1. Parse unit cell dimensions
    cell = {"a": 10.0, "b": 10.0, "c": 10.0, "alpha": 90.0, "beta": 90.0, "gamma": 90.0}
    cell_keys = {
        "_cell_length_a": "a",
        "_cell_length_b": "b",
        "_cell_length_c": "c",
        "_cell_angle_alpha": "alpha",
        "_cell_angle_beta": "beta",
        "_cell_angle_gamma": "gamma",
    }
    for line in text.splitlines():
        line_s = line.strip()
        for k, prop in cell_keys.items():
            if line_s.lower().startswith(k):
                parts = line_s.split()
                if len(parts) >= 2:
                    try:
                        cell[prop] = _clean_num(parts[1])
                    except ValueError:
                        pass

    # 2. Compute Fractional -> Cartesian transformation matrix
    a = cell["a"]
    b = cell["b"]
    c = cell["c"]
    alpha = math.radians(cell["alpha"])
    beta = math.radians(cell["beta"])
    gamma = math.radians(cell["gamma"])

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-6:
        sin_g = 1e-6

    term = 1.0 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2.0 * cos_a * cos_b * cos_g
    v_norm = math.sqrt(max(0.0, term))
    vol = a * b * c * v_norm

    m_frac_to_cart = np.array([
        [a, b * cos_g, c * cos_b],
        [0.0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
        [0.0, 0.0, vol / (a * b * sin_g) if (a * b * sin_g) != 0 else c],
    ], dtype=np.float64)

    # 3. Locate loop_ containing _atom_site_
    lines = text.splitlines()
    symbols: List[str] = []
    coords_list: List[Tuple[float, float, float]] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if line.startswith("loop_"):
            idx += 1
            headers = []
            while idx < len(lines) and lines[idx].strip().startswith("_"):
                headers.append(lines[idx].strip().lower())
                idx += 1

            if any("_atom_site" in h for h in headers):
                col_type = -1
                col_label = -1
                col_fx = -1
                col_fy = -1
                col_fz = -1
                col_cx = -1
                col_cy = -1
                col_cz = -1

                for h_i, h_name in enumerate(headers):
                    if h_name == "_atom_site_type_symbol":
                        col_type = h_i
                    elif h_name == "_atom_site_label":
                        col_label = h_i
                    elif h_name in ("_atom_site_fract_x", "_atom_site_fract_x_"):
                        col_fx = h_i
                    elif h_name in ("_atom_site_fract_y", "_atom_site_fract_y_"):
                        col_fy = h_i
                    elif h_name in ("_atom_site_fract_z", "_atom_site_fract_z_"):
                        col_fz = h_i
                    elif h_name == "_atom_site_cartn_x":
                        col_cx = h_i
                    elif h_name == "_atom_site_cartn_y":
                        col_cy = h_i
                    elif h_name == "_atom_site_cartn_z":
                        col_cz = h_i

                while idx < len(lines):
                    row = lines[idx].strip()
                    if not row or row.startswith("loop_") or row.startswith("_") or row.startswith("#"):
                        break
                    parts = row.split()
                    if len(parts) >= len(headers):
                        raw_sym = ""
                        if 0 <= col_type < len(parts):
                            raw_sym = parts[col_type]
                        elif 0 <= col_label < len(parts):
                            raw_sym = parts[col_label]
                        clean_sym = re.sub(r'[^A-Za-z]', '', raw_sym)
                        norm_sym = normalize_symbol(clean_sym) or "C"

                        try:
                            if col_cx >= 0 and col_cy >= 0 and col_cz >= 0:
                                x = _clean_num(parts[col_cx])
                                y = _clean_num(parts[col_cy])
                                z = _clean_num(parts[col_cz])
                            elif col_fx >= 0 and col_fy >= 0 and col_fz >= 0:
                                xf = _clean_num(parts[col_fx])
                                yf = _clean_num(parts[col_fy])
                                zf = _clean_num(parts[col_fz])
                                cart = m_frac_to_cart @ np.array([xf, yf, zf])
                                x, y, z = float(cart[0]), float(cart[1]), float(cart[2])
                            else:
                                idx += 1
                                continue
                            symbols.append(norm_sym)
                            coords_list.append((x, y, z))
                        except (ValueError, IndexError):
                            pass
                    idx += 1
                break
        idx += 1

    coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3), dtype=np.float64)
    return symbols, coords, cell


def format_cif_text(
    symbols: List[str],
    coords: np.ndarray,
    cell: Optional[Dict[str, float]] = None,
    data_name: str = "AgyMolden_Structure",
) -> str:
    """
    Export molecular coordinates and crystallographic unit cell parameters to standard CIF syntax.
    Converts Cartesian coordinates (Å) to fractional coordinates [x, y, z] using cell parameters.
    """
    if cell is None:
        # Default cubic bounding cell with 10 Å margin if no periodic cell specified
        min_c = np.min(coords, axis=0) if len(coords) > 0 else np.zeros(3)
        max_c = np.max(coords, axis=0) if len(coords) > 0 else np.ones(3)
        span = np.maximum(max_c - min_c + 10.0, 15.0)
        cell = {
            "a": float(span[0]),
            "b": float(span[1]),
            "c": float(span[2]),
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        }

    a = float(cell.get("a", 10.0))
    b = float(cell.get("b", 10.0))
    c = float(cell.get("c", 10.0))
    alpha = math.radians(float(cell.get("alpha", 90.0)))
    beta = math.radians(float(cell.get("beta", 90.0)))
    gamma = math.radians(float(cell.get("gamma", 90.0)))

    cos_a = math.cos(alpha)
    cos_b = math.cos(beta)
    cos_g = math.cos(gamma)
    sin_g = math.sin(gamma)
    if abs(sin_g) < 1e-6:
        sin_g = 1e-6

    term = 1.0 - cos_a ** 2 - cos_b ** 2 - cos_g ** 2 + 2.0 * cos_a * cos_b * cos_g
    v_norm = math.sqrt(max(0.0, term))
    vol = a * b * c * v_norm

    m_frac_to_cart = np.array([
        [a, b * cos_g, c * cos_b],
        [0.0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
        [0.0, 0.0, vol / (a * b * sin_g) if (a * b * sin_g) != 0 else c],
    ], dtype=np.float64)

    # Invert to convert Cart -> Frac
    try:
        m_cart_to_frac = np.linalg.inv(m_frac_to_cart)
    except np.linalg.LinAlgError:
        m_cart_to_frac = np.diag([1.0 / a, 1.0 / b, 1.0 / c])

    lines = [
        f"data_{re.sub(r'[^A-Za-z0-9_]', '_', data_name)}",
        "_audit_creation_method  'AgyMolden Molecular Workbench'",
        "_symmetry_space_group_name_H-M  'P 1'",
        "_symmetry_Int_Tables_number     1",
        "",
        f"_cell_length_a     {a:10.4f}",
        f"_cell_length_b     {b:10.4f}",
        f"_cell_length_c     {c:10.4f}",
        f"_cell_angle_alpha  {cell.get('alpha', 90.0):10.4f}",
        f"_cell_angle_beta   {cell.get('beta', 90.0):10.4f}",
        f"_cell_angle_gamma  {cell.get('gamma', 90.0):10.4f}",
        "",
        "loop_",
        "  _atom_site_label",
        "  _atom_site_type_symbol",
        "  _atom_site_fract_x",
        "  _atom_site_fract_y",
        "  _atom_site_fract_z",
        "  _atom_site_occupancy",
    ]

    for i, (sym, pos) in enumerate(zip(symbols, coords)):
        frac = m_cart_to_frac @ pos
        # Wrap or keep fractional
        lbl = f"{sym}{i + 1}"
        lines.append(f"  {lbl:<6s} {sym:<3s} {frac[0]:12.6f} {frac[1]:12.6f} {frac[2]:12.6f}   1.0000")

    lines.append("")
    return "\n".join(lines)


def write_cif_file(
    path: str,
    symbols: List[str],
    coords: np.ndarray,
    cell: Optional[Dict[str, float]] = None,
    data_name: str = "AgyMolden_Structure",
):
    """Write molecular geometry and unit cell parameters to a .cif file."""
    txt = format_cif_text(symbols, coords, cell=cell, data_name=data_name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)



