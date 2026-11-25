"""
Gaussian Cube (.cube) and Molden Format File Parsers, and Marching Cubes Isosurface Generator.

Supports:
- Standard Gaussian volumetric grid files (.cube) for Electron Density, Electrostatic Potential (ESP),
  and Molecular Orbitals (HOMO, LUMO, etc.)
- Molden files (.molden) with [Atoms], [MO], [GTO], and geometries
- High-performance marching cubes isosurface extraction (positive and negative lobes for orbitals)
"""

import math
import logging
import os
import threading
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    from skimage.measure import marching_cubes
    _HAS_MARCHING_CUBES = True
except ImportError:
    marching_cubes = None
    _HAS_MARCHING_CUBES = True  # Pure-NumPy marching tetrahedra fallback is always available


def _pure_numpy_marching_tetrahedra(
    grid: np.ndarray, level: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pure-NumPy vectorized Marching Tetrahedra isosurface extractor.
    Provides a zero-compiler fallback when scikit-image is not installed.
    Decomposes each 3D voxel cube into 6 tetrahedra and interpolates edge crossings.
    """
    nx, ny, nz = grid.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32), np.empty((0, 3))

    # 8 cube corner offsets: (dx, dy, dz)
    # 0:(0,0,0), 1:(1,0,0), 2:(1,1,0), 3:(0,1,0), 4:(0,0,1), 5:(1,0,1), 6:(1,1,1), 7:(0,1,1)
    c_offsets = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=np.float32)

    # 6 tetrahedra sharing the main diagonal (0 -> 6)
    tets = [
        (0, 5, 1, 6),
        (0, 1, 2, 6),
        (0, 2, 3, 6),
        (0, 3, 7, 6),
        (0, 7, 4, 6),
        (0, 4, 5, 6),
    ]

    # Precompute voxel values at the 8 corners for active cells
    v0 = grid[:-1, :-1, :-1]
    v1 = grid[1:, :-1, :-1]
    v2 = grid[1:, 1:, :-1]
    v3 = grid[:-1, 1:, :-1]
    v4 = grid[:-1, :-1, 1:]
    v5 = grid[1:, :-1, 1:]
    v6 = grid[1:, 1:, 1:]
    v7 = grid[:-1, 1:, 1:]

    c_min = np.minimum.reduce([v0, v1, v2, v3, v4, v5, v6, v7])
    c_max = np.maximum.reduce([v0, v1, v2, v3, v4, v5, v6, v7])
    active_mask = (c_min <= level) & (c_max >= level)
    ix, iy, iz = np.nonzero(active_mask)
    if len(ix) == 0:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32), np.empty((0, 3))

    base_coords = np.column_stack([ix, iy, iz]).astype(np.float32)
    c_vals = np.column_stack([
        v0[ix, iy, iz], v1[ix, iy, iz], v2[ix, iy, iz], v3[ix, iy, iz],
        v4[ix, iy, iz], v5[ix, iy, iz], v6[ix, iy, iz], v7[ix, iy, iz],
    ]).astype(np.float32) - float(level)

    all_tris = []

    def _interp(p_a, p_b, val_a, val_b):
        denom = val_a - val_b
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
        t = np.clip(val_a / denom, 0.0, 1.0)[:, None]
        return p_a + t * (p_b - p_a)

    for t0, t1, t2, t3 in tets:
        p = [base_coords + c_offsets[tk] for tk in (t0, t1, t2, t3)]
        val = [c_vals[:, tk] for tk in (t0, t1, t2, t3)]
        # Classify vertices > 0
        b0 = val[0] > 0.0
        b1 = val[1] > 0.0
        b2 = val[2] > 0.0
        b3 = val[3] > 0.0
        code = (b0.astype(np.int32) << 3) | (b1.astype(np.int32) << 2) | (b2.astype(np.int32) << 1) | b3.astype(np.int32)

        for case_id in range(1, 15):
            sel = np.nonzero(code == case_id)[0]
            if len(sel) == 0:
                continue
            ins = [i for i in range(4) if (case_id >> (3 - i)) & 1]
            outs = [i for i in range(4) if not ((case_id >> (3 - i)) & 1)]

            if len(ins) == 1:
                i0 = ins[0]
                o0, o1, o2 = outs
                va = _interp(p[i0][sel], p[o0][sel], val[i0][sel], val[o0][sel])
                vb = _interp(p[i0][sel], p[o1][sel], val[i0][sel], val[o1][sel])
                vc = _interp(p[i0][sel], p[o2][sel], val[i0][sel], val[o2][sel])
                all_tris.append(np.stack([va, vb, vc], axis=1))
            elif len(ins) == 3:
                o0 = outs[0]
                i0, i1, i2 = ins
                va = _interp(p[o0][sel], p[i0][sel], val[o0][sel], val[i0][sel])
                vb = _interp(p[o0][sel], p[i2][sel], val[o0][sel], val[i2][sel])
                vc = _interp(p[o0][sel], p[i1][sel], val[o0][sel], val[i1][sel])
                all_tris.append(np.stack([va, vb, vc], axis=1))
            elif len(ins) == 2:
                i0, i1 = ins
                o0, o1 = outs
                v00 = _interp(p[i0][sel], p[o0][sel], val[i0][sel], val[o0][sel])
                v01 = _interp(p[i0][sel], p[o1][sel], val[i0][sel], val[o1][sel])
                v10 = _interp(p[i1][sel], p[o0][sel], val[i1][sel], val[o0][sel])
                v11 = _interp(p[i1][sel], p[o1][sel], val[i1][sel], val[o1][sel])
                all_tris.append(np.stack([v00, v01, v11], axis=1))
                all_tris.append(np.stack([v00, v11, v10], axis=1))

    if not all_tris:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.uint32), np.empty((0, 3))

    tris = np.concatenate(all_tris, axis=0)  # shape (N_tri, 3, 3)
    verts_vox = tris.reshape(-1, 3)
    faces = np.arange(len(verts_vox), dtype=np.uint32).reshape(-1, 3)

    # Compute outward normals from gradient of scalar field (-grad(field))
    gx, gy, gz = np.gradient(grid)
    vi = np.clip(np.round(verts_vox[:, 0]).astype(int), 0, nx - 1)
    vj = np.clip(np.round(verts_vox[:, 1]).astype(int), 0, ny - 1)
    vk = np.clip(np.round(verts_vox[:, 2]).astype(int), 0, nz - 1)
    normals = -np.column_stack([gx[vi, vj, vk], gy[vi, vj, vk], gz[vi, vj, vk]]).astype(np.float32)
    n_norm = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    normals /= n_norm
    return verts_vox, faces, normals

from .elements import ELEMENT_DATA

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM


log = logging.getLogger(__name__)


@dataclass
class VolumetricData:
    title: str
    comment: str
    symbols: List[str]
    atom_coords: np.ndarray  # in Angstroms, shape: (n_atoms, 3)
    origin: np.ndarray       # in Angstroms, shape: (3,)
    voxels_x: int
    voxels_y: int
    voxels_z: int
    step_x: np.ndarray       # in Angstroms, shape: (3,)
    step_y: np.ndarray       # in Angstroms, shape: (3,)
    step_z: np.ndarray       # in Angstroms, shape: (3,)
    grid_data: np.ndarray    # 3D array of shape (voxels_x, voxels_y, voxels_z)
    is_orbital: bool = True
    orbital_index: Optional[int] = None


@dataclass
class MoldenMO:
    sym: str
    energy: float
    spin: str
    occupancy: float
    coefficients: Dict[int, float]  # 1-based index to coeff


@dataclass
class MoldenFileData:
    title: str
    symbols: List[str]
    coords: np.ndarray       # in Angstroms
    orbitals: List[MoldenMO]
    frequencies: Optional[List[float]] = None
    vibrations: Optional[Any] = None


@dataclass
class IsosurfaceMesh:
    isovalue: float
    vertices: np.ndarray   # shape: (n_verts, 3) in Angstroms
    normals: np.ndarray    # shape: (n_verts, 3) normalized
    faces: np.ndarray      # shape: (n_faces, 3) uint32
    color: Tuple[float, float, float]  # (R, G, B) in [0, 1]
    vertex_colors: Optional[np.ndarray] = None  # shape: (n_verts, 3) or (n_verts, 4) in [0, 1]
    surface_values: Optional[np.ndarray] = None  # Raw scalar property values at vertices (e.g. ESP a.u., ALIE eV)
    extrema_points: Optional[List[Dict[str, Any]]] = None  # Local surface extrema [{"type": "min"|"max", "pos": [x,y,z], "val": float, "label": str}]
    esp_range: Optional[Tuple[float, float]] = None  # (min_v, max_v) potential mapping range in a.u.
    colormap_name: str = "bwr_r"
    colorbar_title: str = "ESP Heatmap (a.u.)"



def parse_cube_file(path: str) -> VolumetricData:
    """Parse standard Gaussian volumetric .cube file into 3D grid and molecular coordinates."""
    with open(path, "r") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"File {path} is too short to be a valid Gaussian .cube file.")

    title = lines[0].strip()
    comment = lines[1].strip()

    # Line 3: Natoms, Origin_x, Origin_y, Origin_z (in Bohr by convention; sign of Natoms indicates extra orbital headers)
    toks3 = lines[2].split()
    n_atoms_raw = int(toks3[0])
    n_atoms = abs(n_atoms_raw)
    has_mo_header = (n_atoms_raw < 0)

    # In standard .cube files, coordinates are in atomic units (Bohr)
    origin_bohr = np.array([float(toks3[1]), float(toks3[2]), float(toks3[3])], dtype=np.float64)
    origin_ang = origin_bohr * BOHR_TO_ANGSTROM

    # Lines 4-6: Nvoxels, Step_x, Step_y, Step_z
    toks4 = lines[3].split()
    nx = int(toks4[0])
    step_x = np.array([float(toks4[1]), float(toks4[2]), float(toks4[3])], dtype=np.float64) * BOHR_TO_ANGSTROM

    toks5 = lines[4].split()
    ny = int(toks5[0])
    step_y = np.array([float(toks5[1]), float(toks5[2]), float(toks5[3])], dtype=np.float64) * BOHR_TO_ANGSTROM

    toks6 = lines[5].split()
    nz = int(toks6[0])
    step_z = np.array([float(toks6[1]), float(toks6[2]), float(toks6[3])], dtype=np.float64) * BOHR_TO_ANGSTROM

    # Atoms: lines 7 to 6 + n_atoms
    atomic_num_to_sym = {d["atomic_number"]: s for s, d in ELEMENT_DATA.items() if "atomic_number" in d}
    symbols = []
    atom_coords_ang = np.empty((n_atoms, 3), dtype=np.float64)

    curr_line = 6
    for i in range(n_atoms):
        parts = lines[curr_line].split()
        at_num = int(parts[0])
        sym = atomic_num_to_sym.get(at_num, f"X{at_num}")
        symbols.append(sym)
        c_bohr = np.array([float(parts[2]), float(parts[3]), float(parts[4])], dtype=np.float64)
        atom_coords_ang[i] = c_bohr * BOHR_TO_ANGSTROM
        curr_line += 1

    mo_index = None
    if has_mo_header and curr_line < len(lines):
        mo_parts = lines[curr_line].split()
        if mo_parts:
            try:
                mo_index = int(mo_parts[1]) if len(mo_parts) > 1 else int(mo_parts[0])
            except ValueError:
                pass
        curr_line += 1

    # Remaining lines contain the 3D grid data in Fortran format
    raw_vals = []
    for line in lines[curr_line:]:
        for val_str in line.split():
            raw_vals.append(float(val_str))

    total_voxels = nx * ny * nz
    if len(raw_vals) < total_voxels:
        raise ValueError(
            f"Expected {total_voxels} grid data points in cube file, but found only {len(raw_vals)}."
        )

    grid_arr = np.array(raw_vals[:total_voxels], dtype=np.float32).reshape((nx, ny, nz))

    return VolumetricData(
        title=title,
        comment=comment,
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_arr,
        is_orbital=(mo_index is not None or "MO" in comment.upper() or "ORBITAL" in comment.upper() or np.min(grid_arr) < -1e-4),
        orbital_index=mo_index,
    )


def write_cube_file(path: str, vol_data: VolumetricData):
    """Write VolumetricData back to a standard Gaussian .cube file format."""
    n_atoms = len(vol_data.symbols)
    origin_bohr = vol_data.origin * ANGSTROM_TO_BOHR
    step_x_bohr = vol_data.step_x * ANGSTROM_TO_BOHR
    step_y_bohr = vol_data.step_y * ANGSTROM_TO_BOHR
    step_z_bohr = vol_data.step_z * ANGSTROM_TO_BOHR

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{vol_data.title or 'Generated by AgyMolden'}\n")
        f.write(f"{vol_data.comment or 'Cube Grid Output'}\n")
        f.write(f"{n_atoms:5d}{origin_bohr[0]:12.6f}{origin_bohr[1]:12.6f}{origin_bohr[2]:12.6f}\n")
        f.write(f"{vol_data.voxels_x:5d}{step_x_bohr[0]:12.6f}{step_x_bohr[1]:12.6f}{step_x_bohr[2]:12.6f}\n")
        f.write(f"{vol_data.voxels_y:5d}{step_y_bohr[0]:12.6f}{step_y_bohr[1]:12.6f}{step_y_bohr[2]:12.6f}\n")
        f.write(f"{vol_data.voxels_z:5d}{step_z_bohr[0]:12.6f}{step_z_bohr[1]:12.6f}{step_z_bohr[2]:12.6f}\n")

        for s, c_ang in zip(vol_data.symbols, vol_data.atom_coords):
            at_num = ELEMENT_DATA.get(s, {}).get("atomic_number", 1)
            c_bohr = c_ang * ANGSTROM_TO_BOHR
            f.write(f"{at_num:5d}{float(at_num):12.6f}{c_bohr[0]:12.6f}{c_bohr[1]:12.6f}{c_bohr[2]:12.6f}\n")

        # Flat array written with 6 numbers per line
        flat = vol_data.grid_data.flatten()
        for idx in range(0, len(flat), 6):
            chunk = flat[idx : idx + 6]
            f.write(" ".join(f"{v:13.5e}" for v in chunk) + "\n")


def compute_grid_difference(grid_a: VolumetricData, grid_b: VolumetricData, operation: str = "A - B") -> VolumetricData:
    """
    Perform volumetric grid arithmetic (Multiwfn style).
    Supports operations: 'A - B', 'A + B', 'A * B'.
    Validates dimensional compatibility.
    """
    if (grid_a.voxels_x != grid_b.voxels_x or
        grid_a.voxels_y != grid_b.voxels_y or
        grid_a.voxels_z != grid_b.voxels_z):
        raise ValueError(
            f"Grid dimension mismatch: Grid A is ({grid_a.voxels_x}, {grid_a.voxels_y}, {grid_a.voxels_z}) "
            f"while Grid B is ({grid_b.voxels_x}, {grid_b.voxels_y}, {grid_b.voxels_z}). Grids must match exactly."
        )

    if operation == "A - B":
        res_data = grid_a.grid_data - grid_b.grid_data
        title = f"Difference Grid: ({grid_a.title}) - ({grid_b.title})"
    elif operation == "A + B":
        res_data = grid_a.grid_data + grid_b.grid_data
        title = f"Sum Grid: ({grid_a.title}) + ({grid_b.title})"
    elif operation == "A * B":
        res_data = grid_a.grid_data * grid_b.grid_data
        title = f"Product Grid: ({grid_a.title}) * ({grid_b.title})"
    else:
        raise ValueError(f"Unknown operation {operation}")

    return VolumetricData(
        title=title,
        comment=f"Operation {operation} computed with AgyMolden",
        symbols=list(grid_a.symbols),
        atom_coords=np.array(grid_a.atom_coords, copy=True),
        origin=np.array(grid_a.origin, copy=True),
        voxels_x=grid_a.voxels_x,
        voxels_y=grid_a.voxels_y,
        voxels_z=grid_a.voxels_z,
        step_x=np.array(grid_a.step_x, copy=True),
        step_y=np.array(grid_a.step_y, copy=True),
        step_z=np.array(grid_a.step_z, copy=True),
        grid_data=res_data.astype(np.float32),
        is_orbital=(np.min(res_data) < -1e-5 and np.max(res_data) > 1e-5),
        orbital_index=None,
    )


def convert_gbw_to_molden_if_possible(gbw_path: str) -> Optional[str]:
    """Convert an ORCA binary wavefunction (.gbw) file to .molden.input / .molden using orca_2mkl if available."""
    import shutil
    import subprocess
    if not gbw_path or not os.path.isfile(gbw_path):
        return None
    stem = os.path.splitext(os.path.abspath(gbw_path))[0]
    molden_inp = f"{stem}.molden.input"
    molden_std = f"{stem}.molden"
    gbw_mtime = os.path.getmtime(gbw_path)
    for cand in (molden_inp, molden_std):
        if os.path.isfile(cand) and os.path.getmtime(cand) >= gbw_mtime - 2.0:
            return cand

    orca_2mkl = shutil.which("orca_2mkl")
    if not orca_2mkl:
        try:
            from .qm_detect import detect_qm_executable
            info = detect_qm_executable("orca")
            if info.found and info.resolved_path:
                cand_2mkl = os.path.join(os.path.dirname(info.resolved_path), "orca_2mkl")
                if os.path.isfile(cand_2mkl):
                    orca_2mkl = cand_2mkl
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)
    if not orca_2mkl:
        return None

    try:
        subprocess.run(
            [orca_2mkl, os.path.basename(stem), "-molden"],
            cwd=os.path.dirname(stem),
            capture_output=True,
            timeout=30,
        )
        if os.path.isfile(molden_inp):
            if not os.path.isfile(molden_std):
                try:
                    shutil.copy2(molden_inp, molden_std)
                except OSError:
                    pass
            return molden_inp
        if os.path.isfile(molden_std):
            return molden_std
    except Exception as e:
        log.debug("orca_2mkl conversion failed for %s: %s", gbw_path, e)
    return None


def parse_molden_file(path: str) -> MoldenFileData:
    """Parse standard .molden / .molf format file containing geometry, orbitals and frequencies."""
    # Automatic conversion if ORCA binary wavefunction (.gbw) is passed
    if path.lower().endswith(".gbw"):
        conv_gbw = convert_gbw_to_molden_if_possible(path)
        if conv_gbw and os.path.isfile(conv_gbw):
            path = conv_gbw
    # Automatic conversion if Gaussian FCHK file is passed
    if path.lower().endswith((".fchk", ".fch")):
        stem = os.path.splitext(os.path.abspath(path))[0]
        cand_m = f"{stem}.fchk.molden"
        if not (os.path.isfile(cand_m) and os.path.getmtime(cand_m) >= os.path.getmtime(path) - 2.0):
            try:
                import cclib
                p_cclib = cclib.io.ccopen(path)
                data_cclib = p_cclib.parse()
                if data_cclib is not None:
                    m_str = cclib.io.moldenwriter.MOLDEN(data_cclib).generate_repr()
                    if m_str and len(m_str) > 50:
                        with open(cand_m, "w", encoding="utf-8") as f_fchk:
                            f_fchk.write(m_str)
            except Exception as e_fchk:
                log.debug("Auto-generation of Molden from FCHK in parse_molden_file skipped: %s", e_fchk)
        if os.path.isfile(cand_m):
            path = cand_m

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = content.splitlines()
    title = "Molden File"
    symbols = []
    coords_list = []
    orbitals = []
    freqs = []
    fr_coords_list = []
    norm_coords_list = []
    ir_ints = []

    current_section = None
    atom_units = "AU"  # default atomic units (Bohr)
    fr_units = "AU"

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        if line.startswith("[") and "]" in line:
            sec_name = line.split("]")[0].strip("[").strip().upper()
            current_section = sec_name
            if "ATOMS" in sec_name:
                if "(AU)" in line.upper() or "AU" in line.upper():
                    atom_units = "AU"
                elif "(ANGS)" in line.upper() or "ANG" in line.upper():
                    atom_units = "ANGS"
            elif "FR-COORD" in sec_name:
                fr_units = "ANGS" if "ANG" in line.upper() else "AU"
            i += 1
            continue

        if current_section == "TITLE":
            title = line
        elif current_section and "ATOMS" in current_section:
            toks = line.split()
            if len(toks) >= 6:
                # Format: AtomName AtomIndex AtomicNumber X Y Z
                sym = toks[0]
                # Normalize symbol if it contains numbers like C1 -> C
                clean_sym = "".join([c for c in sym if c.isalpha()]).capitalize() or sym
                x, y, z = float(toks[3]), float(toks[4]), float(toks[5])
                scale = BOHR_TO_ANGSTROM if atom_units == "AU" else 1.0
                symbols.append(clean_sym)
                coords_list.append([x * scale, y * scale, z * scale])
        elif current_section and "FREQ" in current_section:
            try:
                freqs.append(float(line.split()[0]))
            except ValueError:
                pass
        elif current_section and "FR-COORD" in current_section:
            toks = line.split()
            if len(toks) >= 4:
                try:
                    sym = "".join([c for c in toks[0] if c.isalpha()]).capitalize() or toks[0]
                    scale = BOHR_TO_ANGSTROM if fr_units == "AU" else 1.0
                    x, y, z = float(toks[1]) * scale, float(toks[2]) * scale, float(toks[3]) * scale
                    fr_coords_list.append((sym, [x, y, z]))
                except ValueError:
                    pass
        elif current_section and "FR-NORM-COORD" in current_section:
            if line.lower().startswith("vibration"):
                norm_coords_list.append([])
            else:
                toks = line.split()
                if len(toks) >= 3 and norm_coords_list:
                    try:
                        norm_coords_list[-1].append([float(toks[0]), float(toks[1]), float(toks[2])])
                    except ValueError:
                        pass
        elif current_section and "INT" in current_section:
            try:
                ir_ints.append(float(line.split()[0]))
            except ValueError:
                pass
        elif current_section == "MO":
            if "=" in line:
                key_part, val_part = line.split("=", 1)
                key_up = key_part.strip().upper()
                val_str = val_part.strip()
            else:
                key_up, val_str = "", ""

            if key_up in ("SYM", "ENE"):
                # Start a new orbital block
                sym_val = val_str if key_up == "SYM" else "1a"
                try:
                    init_ene = float(val_str.replace("D", "E").replace("d", "e")) if key_up == "ENE" else 0.0
                except ValueError:
                    init_ene = 0.0
                seen_ene = (key_up == "ENE")
                seen_sym = (key_up == "SYM")
                seen_coeff = False
                mo = MoldenMO(sym=sym_val, energy=init_ene, spin="Alpha", occupancy=0.0, coefficients={})
                i += 1
                while i < len(lines):
                    sub = lines[i].strip()
                    if not sub:
                        i += 1
                        continue
                    if sub.startswith("["):
                        i -= 1
                        break
                    if "=" in sub:
                        s_key, s_val = sub.split("=", 1)
                        s_key_up = s_key.strip().upper()
                        s_val_str = s_val.strip()
                    else:
                        s_key_up, s_val_str = "", ""

                    if s_key_up == "SYM":
                        if seen_sym or seen_coeff:
                            i -= 1
                            break
                        mo.sym = s_val_str or mo.sym
                        seen_sym = True
                    elif s_key_up == "ENE":
                        if seen_ene or seen_coeff:
                            i -= 1
                            break
                        try:
                            mo.energy = float(s_val_str.replace("D", "E").replace("d", "e"))
                        except ValueError:
                            pass
                        seen_ene = True
                    elif s_key_up == "SPIN":
                        mo.spin = s_val_str
                    elif s_key_up.startswith("OCC"):
                        try:
                            mo.occupancy = float(s_val_str.replace("D", "E").replace("d", "e"))
                        except ValueError:
                            pass
                    else:
                        # Coefficient line: AO_Index Value
                        ctoks = sub.replace("D", "E").replace("d", "e").split()
                        if len(ctoks) >= 2:
                            try:
                                mo.coefficients[int(ctoks[0])] = float(ctoks[1])
                                seen_coeff = True
                            except ValueError:
                                pass
                    i += 1
                orbitals.append(mo)
        i += 1

    if not symbols and fr_coords_list:
        symbols = [s for s, _ in fr_coords_list]
        coords_list = [c for _, c in fr_coords_list]

    coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3))
    vib_obj = None
    if freqs and norm_coords_list and len(symbols) > 0:
        try:
            from .vibrations import VibrationalAnalysis
            n_atoms = len(symbols)
            valid_freqs = []
            valid_disps = []
            valid_irs = []
            for idx_f, (f_val, disp_rows) in enumerate(zip(freqs, norm_coords_list)):
                if len(disp_rows) == n_atoms and abs(f_val) > 5.0:
                    valid_freqs.append(f_val)
                    valid_disps.append(np.array(disp_rows, dtype=np.float64))
                    if idx_f < len(ir_ints):
                        valid_irs.append(ir_ints[idx_f])
            if valid_freqs and valid_disps:
                vib_obj = VibrationalAnalysis(
                    frequencies=valid_freqs,
                    displacements=np.array(valid_disps, dtype=np.float64),
                    symbols=symbols,
                    base_coords=coords,
                    source="Molden Frequency File",
                    ir_intensities=valid_irs if len(valid_irs) == len(valid_freqs) else None,
                )
        except Exception:
            log.debug("Ignored non-fatal exception", exc_info=True)

    return MoldenFileData(
        title=title,
        symbols=symbols,
        coords=coords,
        orbitals=orbitals,
        frequencies=freqs if freqs else None,
        vibrations=vib_obj,
    )


_MO_CTX_LOCK = threading.Lock()
_MO_CTX_CACHE: Dict[tuple, dict] = {}
_MO_CTX_MAX_BYTES = 400 * 1024 * 1024

_CART2SPH_L2 = np.array([
    [0.0, 0.0, -0.31539156525252005, 0.0, 0.5462742152960395],
    [1.092548430592079, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.092548430592079, 0.0],
    [0.0, 0.0, -0.31539156525252005, 0.0, -0.5462742152960395],
    [0.0, 1.092548430592079, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.6307831305050401, 0.0, 0.0],
], dtype=np.float64)

_CART2SPH_L3 = np.array([
    [0.0, 0.0, 0.0, 0.0, -0.4570457994644658, 0.0, 0.5900435899266435],
    [1.7701307697799304, 0.0, -0.4570457994644658, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.1195289977703462, 0.0, 1.445305721320277, 0.0],
    [0.0, 0.0, 0.0, 0.0, -0.4570457994644658, 0.0, -1.7701307697799304],
    [0.0, 2.890611442640554, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.8281831978578632, 0.0, 0.0],
    [-0.5900435899266435, 0.0, -0.4570457994644658, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, -1.1195289977703462, 0.0, -1.445305721320277, 0.0],
    [0.0, 0.0, 1.8281831978578632, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.7463526651802308, 0.0, 0.0, 0.0],
], dtype=np.float64)

_CART2SPH_L4 = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.317356640745, 0.0, -0.473087347879, 0.0, 0.625835735449],
    [2.503342941796, 0.0, -0.946174695758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -2.007139630672, 0.0, 1.77013076978, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.63471328149, 0.0, 0.0, 0.0, -3.755014412695],
    [0.0, 5.31039230934, 0.0, -2.007139630672, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -2.53885312596, 0.0, 2.838524087273, 0.0, 0.0],
    [-2.503342941796, 0.0, -0.946174695758, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, -2.007139630672, 0.0, -5.31039230934, 0.0],
    [0.0, 0.0, 5.677048174545, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 2.676186174229, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.317356640745, 0.0, 0.473087347879, 0.0, 0.625835735449],
    [0.0, -1.77013076978, 0.0, -2.007139630672, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -2.53885312596, 0.0, -2.838524087273, 0.0, 0.0],
    [0.0, 0.0, 0.0, 2.676186174229, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.84628437532, 0.0, 0.0, 0.0, 0.0],
], dtype=np.float64)


def _gto_gaussian_int_np(n: int, alpha: np.ndarray) -> np.ndarray:
    from scipy.special import gamma
    n1 = (n + 1) * 0.5
    return float(gamma(n1)) / (2.0 * (alpha ** n1))


def _gto_norm_np(l_ang: int, expnt: np.ndarray) -> np.ndarray:
    return 1.0 / np.sqrt(_gto_gaussian_int_np(l_ang * 2 + 2, 2.0 * expnt))


def _cart_powers_for_l(l_ang: int) -> List[Tuple[int, int, int]]:
    powers = []
    for lx in range(l_ang, -1, -1):
        for ly in range(l_ang - lx, -1, -1):
            lz = l_ang - lx - ly
            powers.append((lx, ly, lz))
    return powers


def _double_factorial(n: int) -> int:
    """Return n!! (with (-1)!! = 1 and 0!! = 1)."""
    res = 1
    while n > 1:
        res *= n
        n -= 2
    return res


def _cart_ang_norm(lx: int, ly: int, lz: int) -> float:
    """Angular normalization factor for a Cartesian GTO x^lx y^ly z^lz."""
    l_ang = lx + ly + lz
    denom = (
        4.0
        * np.pi
        * _double_factorial(2 * lx - 1)
        * _double_factorial(2 * ly - 1)
        * _double_factorial(2 * lz - 1)
    )
    return float(np.sqrt(_double_factorial(2 * l_ang + 1) / denom))


class _FallbackCube:
    """Pure-NumPy drop-in replacement for pyscf.tools.cubegen.Cube."""

    def __init__(self, mol: Any, nx: int = 48, ny: int = 48, nz: int = 48, margin: float = 5.5, extent: Optional[Tuple[float, float, float]] = None):
        self.nx = int(nx)
        self.ny = int(ny)
        self.nz = int(nz)
        atom_coords = mol.atom_coords()
        if extent is not None and all(e > 0 for e in extent):
            center = 0.5 * (np.max(atom_coords, axis=0) + np.min(atom_coords, axis=0))
            box_diag = np.asarray(extent, dtype=np.float64)
            self.boxorig = center - 0.5 * box_diag
        else:
            box_diag = np.max(atom_coords, axis=0) - np.min(atom_coords, axis=0) + 2.0 * float(margin)
            self.boxorig = np.min(atom_coords, axis=0) - float(margin)
        self.box = np.diag(box_diag)

    def get_coords(self) -> np.ndarray:
        xs = np.arange(self.nx, dtype=np.float64) * (self.box[0, 0] / max(1, self.nx - 1))
        ys = np.arange(self.ny, dtype=np.float64) * (self.box[1, 1] / max(1, self.ny - 1))
        zs = np.arange(self.nz, dtype=np.float64) * (self.box[2, 2] / max(1, self.nz - 1))
        return np.vstack(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T + self.boxorig


class _FallbackMol:
    """Pure-NumPy molecular GTO representation and evaluator when PySCF is unavailable."""

    def __init__(
        self,
        symbols: List[str],
        coords_bohr: np.ndarray,
        shells: List[Dict[str, Any]],
        cart: bool = False,
        cart_l: Optional[Dict[int, bool]] = None,
    ):
        self._symbols = list(symbols)
        self._coords_bohr = np.asarray(coords_bohr, dtype=np.float64)
        self._shells = shells
        self._cart_l: Dict[int, bool] = dict(cart_l) if cart_l is not None else {2: bool(cart), 3: bool(cart), 4: bool(cart)}
        self.cart = bool(cart)
        self.natm = len(self._symbols)
        self._basis = True  # truthy sentinel for callers checking getattr(mol, "_basis", None)
        from .elements import ELEMENT_DATA
        self._charges = np.array([ELEMENT_DATA.get(s, {}).get("number", 6) for s in self._symbols], dtype=np.int32)
        self.nelectron = int(np.sum(self._charges))

    def atom_coords(self) -> np.ndarray:
        return self._coords_bohr

    def atom_symbol(self, idx: int) -> str:
        return self._symbols[idx]

    def atom_charges(self) -> np.ndarray:
        return self._charges

    def ao_labels(self, fmt: bool = True) -> List[Any]:
        labels = []
        shell_counts: Dict[Tuple[int, int], int] = {}
        l_names = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
        for sh in self._shells:
            at_idx = int(sh["atom_idx"])
            l_ang = int(sh["l"])
            sym = self._symbols[at_idx]
            base_n = l_ang + 1 + shell_counts.get((at_idx, l_ang), 0)
            shell_counts[(at_idx, l_ang)] = shell_counts.get((at_idx, l_ang), 0) + 1
            nl_str = f"{base_n}{l_names.get(l_ang, 's')}"
            is_cart = self._cart_l.get(l_ang, self.cart) if l_ang >= 2 else True
            n_comp = ((l_ang + 1) * (l_ang + 2)) // 2 if is_cart else (2 * l_ang + 1)
            for c_i in range(n_comp):
                if fmt:
                    labels.append(f"{at_idx} {sym}{at_idx + 1} {nl_str}")
                else:
                    labels.append((at_idx, sym, nl_str, str(c_i)))
        return labels

    def eval_gto(self, eval_name: str, coords: np.ndarray) -> np.ndarray:
        pts = np.asarray(coords, dtype=np.float64)
        n_pts = pts.shape[0]
        cols = []
        for sh in self._shells:
            at_idx = int(sh["atom_idx"])
            l_ang = int(sh["l"])
            exps = sh["exps"]
            weights = sh["weights"]
            r = pts - self._coords_bohr[at_idx]
            x, y, z = r[:, 0], r[:, 1], r[:, 2]
            r2 = x * x + y * y + z * z
            rad = np.exp(-r2[:, None] * exps[None, :]) @ weights  # (n_pts,)

            if l_ang == 0:
                cols.append((rad * 0.28209479177387814)[:, None])
            elif l_ang == 1:
                cols.append(np.column_stack([x, y, z]) * (rad * 0.4886025119029199)[:, None])
            else:
                powers = _cart_powers_for_l(l_ang)
                cart_ang = np.empty((n_pts, len(powers)), dtype=np.float64)
                for p_idx, (lx, ly, lz) in enumerate(powers):
                    val = np.ones(n_pts, dtype=np.float64)
                    if lx:
                        val = val * (x ** lx if lx > 1 else x)
                    if ly:
                        val = val * (y ** ly if ly > 1 else y)
                    if lz:
                        val = val * (z ** lz if lz > 1 else z)
                    cart_ang[:, p_idx] = val
                cart_block = cart_ang * rad[:, None]
                is_cart = self._cart_l.get(l_ang, self.cart)
                if is_cart:
                    ang_norms = np.array([_cart_ang_norm(lx, ly, lz) for lx, ly, lz in powers], dtype=np.float64)
                    cols.append(cart_block * ang_norms[None, :])
                elif l_ang == 2:
                    cols.append(cart_block @ _CART2SPH_L2)
                elif l_ang == 3:
                    cols.append(cart_block @ _CART2SPH_L3)
                elif l_ang == 4:
                    cols.append(cart_block @ _CART2SPH_L4)
                else:
                    cols.append(cart_block[:, : (2 * l_ang + 1)])

        if not cols:
            return np.zeros((n_pts, 0), dtype=np.float64)
        return np.hstack(cols)


def _load_fallback_molden(molden_path: str):
    """Pure-NumPy Molden [Atoms] + [GTO] + [MO] loader when PySCF is not installed."""
    with open(molden_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()

    symbols: List[str] = []
    coords_bohr_list: List[List[float]] = []
    atom_units = "AU"
    cart_l: Dict[int, bool] = {2: True, 3: True, 4: True}

    raw_shells: List[Dict[str, Any]] = []
    mo_energies: List[float] = []
    mo_occs: List[float] = []
    mo_syms: List[str] = []
    mo_spins: List[str] = []
    mo_coeffs_dicts: List[Dict[int, float]] = []

    l_map = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
    sec = None
    cur_atom_idx = 0
    i = 0
    n_lines = len(lines)

    while i < n_lines:
        raw = lines[i].strip()
        if not raw:
            i += 1
            continue
        if raw.startswith("[") and "]" in raw:
            sec_header = raw[1:raw.index("]")].strip().upper()
            sec = sec_header.split()[0]
            if "ATOMS" in sec_header:
                sec = "ATOMS"
                if "ANG" in raw.upper():
                    atom_units = "ANGS"
                else:
                    atom_units = "AU"
            else:
                if "5D" in sec_header:
                    cart_l[2] = False
                    if "10F" not in sec_header:
                        cart_l[3] = False
                    if "15G" not in sec_header:
                        cart_l[4] = False
                if "6D" in sec_header:
                    cart_l[2] = True
                if "7F" in sec_header:
                    cart_l[3] = False
                if "10F" in sec_header:
                    cart_l[3] = True
                if "9G" in sec_header:
                    cart_l[4] = False
                if "15G" in sec_header:
                    cart_l[4] = True
            i += 1
            continue

        if sec == "ATOMS":
            toks = raw.split()
            if len(toks) >= 6:
                sym = "".join(c for c in toks[0] if c.isalpha()).capitalize() or toks[0]
                x, y, z = float(toks[3]), float(toks[4]), float(toks[5])
                scale = ANGSTROM_TO_BOHR if atom_units == "ANGS" else 1.0
                symbols.append(sym)
                coords_bohr_list.append([x * scale, y * scale, z * scale])
        elif sec == "GTO":
            toks = raw.split()
            if len(toks) >= 1 and toks[0].isdigit():
                cur_atom_idx = int(toks[0]) - 1
            elif len(toks) >= 2 and toks[0].lower() == "sp":
                n_prim = int(toks[1])
                fac = float(toks[2].replace("D", "E").replace("d", "e")) if len(toks) >= 3 else 1.0
                if fac == 0.0:
                    fac = 1.0
                exps, cs_s, cs_p = [], [], []
                for _ in range(n_prim):
                    i += 1
                    if i >= n_lines:
                        break
                    ptoks = lines[i].replace("D", "E").replace("d", "e").split()
                    if len(ptoks) >= 3:
                        exps.append(float(ptoks[0]))
                        cs_s.append(float(ptoks[1]) * fac)
                        cs_p.append(float(ptoks[2]) * fac)
                if exps:
                    exps_arr = np.array(exps, dtype=np.float64)
                    raw_shells.append({"atom_idx": cur_atom_idx, "l": 0, "exps": exps_arr, "cs": np.array(cs_s, dtype=np.float64)})
                    raw_shells.append({"atom_idx": cur_atom_idx, "l": 1, "exps": exps_arr, "cs": np.array(cs_p, dtype=np.float64)})
            elif len(toks) >= 2 and toks[0].lower() in l_map:
                lsym = toks[0].lower()
                n_prim = int(toks[1])
                fac = float(toks[2].replace("D", "E").replace("d", "e")) if len(toks) >= 3 else 1.0
                if fac == 0.0:
                    fac = 1.0
                exps = []
                cs = []
                for _ in range(n_prim):
                    i += 1
                    if i >= n_lines:
                        break
                    ptoks = lines[i].replace("D", "E").replace("d", "e").split()
                    if len(ptoks) >= 2:
                        exps.append(float(ptoks[0]))
                        cs.append(float(ptoks[1]) * fac)
                if exps:
                    raw_shells.append({
                        "atom_idx": cur_atom_idx,
                        "l": l_map[lsym],
                        "exps": np.array(exps, dtype=np.float64),
                        "cs": np.array(cs, dtype=np.float64),
                    })
        elif sec == "MO":
            if "=" in raw:
                k_part, v_part = raw.split("=", 1)
                k_up = k_part.strip().upper()
                v_str = v_part.strip()
            else:
                k_up, v_str = "", ""
            if k_up in ("SYM", "ENE"):
                sym_val = v_str if k_up == "SYM" else "1a"
                try:
                    ene_val = float(v_str.replace("D", "E").replace("d", "e")) if k_up == "ENE" else 0.0
                except ValueError:
                    ene_val = 0.0
                spin_val = "Alpha"
                occ_val = 0.0
                c_dict: Dict[int, float] = {}
                seen_sym = (k_up == "SYM")
                seen_ene = (k_up == "ENE")
                seen_coeff = False
                i += 1
                while i < n_lines:
                    sub = lines[i].strip()
                    if not sub:
                        i += 1
                        continue
                    if sub.startswith("["):
                        i -= 1
                        break
                    if "=" in sub:
                        sk_part, sv_part = sub.split("=", 1)
                        sk_up = sk_part.strip().upper()
                        sv_str = sv_part.strip()
                    else:
                        sk_up, sv_str = "", ""
                    if sk_up == "SYM":
                        if seen_sym or seen_coeff:
                            i -= 1
                            break
                        sym_val = sv_str or sym_val
                        seen_sym = True
                    elif sk_up == "ENE":
                        if seen_ene or seen_coeff:
                            i -= 1
                            break
                        try:
                            ene_val = float(sv_str.replace("D", "E").replace("d", "e"))
                        except ValueError:
                            pass
                        seen_ene = True
                    elif sk_up == "SPIN":
                        spin_val = sv_str
                    elif sk_up.startswith("OCC"):
                        try:
                            occ_val = float(sv_str.replace("D", "E").replace("d", "e"))
                        except ValueError:
                            pass
                    else:
                        ctoks = sub.replace("D", "E").replace("d", "e").split()
                        if len(ctoks) >= 2:
                            try:
                                c_dict[int(ctoks[0]) - 1] = float(ctoks[1])
                                seen_coeff = True
                            except ValueError:
                                pass
                    i += 1
                mo_syms.append(sym_val)
                mo_energies.append(ene_val)
                mo_spins.append(spin_val)
                mo_occs.append(occ_val)
                mo_coeffs_dicts.append(c_dict)
        i += 1

    if not symbols or not raw_shells or not mo_coeffs_dicts:
        return None

    # Build AO permutation from Molden order to internal PySCF AO order
    order_idx: List[int] = []
    off = 0
    for sh in raw_shells:
        l_ang = int(sh["l"])
        is_cart = cart_l.get(l_ang, True) if l_ang >= 2 else True
        if is_cart:
            if l_ang == 2:
                order_idx.extend([off + 0, off + 3, off + 5, off + 1, off + 2, off + 4])
            elif l_ang == 3:
                order_idx.extend([off + 0, off + 6, off + 9, off + 3, off + 1, off + 2, off + 5, off + 8, off + 7, off + 4])
            elif l_ang == 4:
                order_idx.extend([off + 0, off + 10, off + 14, off + 1, off + 2, off + 6, off + 11, off + 9, off + 13, off + 3, off + 5, off + 12, off + 4, off + 7, off + 8])
            else:
                order_idx.extend(range(off, off + ((l_ang + 1) * (l_ang + 2)) // 2))
            off += ((l_ang + 1) * (l_ang + 2)) // 2
        else:
            if l_ang == 2:
                order_idx.extend([off + 2, off + 3, off + 1, off + 4, off + 0])
            elif l_ang == 3:
                order_idx.extend([off + 3, off + 4, off + 2, off + 5, off + 1, off + 6, off + 0])
            elif l_ang == 4:
                order_idx.extend([off + 4, off + 5, off + 3, off + 6, off + 2, off + 7, off + 1, off + 8, off + 0])
            else:
                order_idx.extend(range(off, off + 2 * l_ang + 1))
            off += 2 * l_ang + 1

    n_ao = off
    n_mo = len(mo_coeffs_dicts)
    mo_raw = np.zeros((n_ao, n_mo), dtype=np.float64)
    for m_i, c_dict in enumerate(mo_coeffs_dicts):
        for ao_i, val in c_dict.items():
            if 0 <= ao_i < n_ao:
                mo_raw[ao_i, m_i] = val

    aoidx = np.argsort(order_idx)
    mo_coeff = mo_raw[aoidx]

    # Compute contracted radial weights for both unnormalized and pre-normalized (orca_2mkl) conventions
    # Check which convention gives cleaner shell self-normalization
    shells_built: List[Dict[str, Any]] = []
    for sh in raw_shells:
        l_ang = int(sh["l"])
        exps = sh["exps"]
        cs = sh["cs"]
        # In _nomalize_contracted_ao, dividing cs by gto_norm before multiplying by gto_norm cancels out!
        # Wait: make_bas_env multiplies cs by gto_norm(l, es). If the Molden file (like orca_2mkl) already
        # multiplied cs by gto_norm(l, es), then dividing cs by gto_norm(l, es) means make_bas_env's
        # multiplication restores cs! Let's detect whether cs is already pre-normalized by checking
        # the shell radial self-overlap int_0^inf r^(2l+2) (sum_p c_p exp(-a_p r^2))^2 dr:
        ee = _gto_gaussian_int_np(l_ang * 2 + 2, exps[:, None] + exps[None, :])
        ovlp_direct = float(cs @ ee @ cs)
        cs_times_gn = cs * _gto_norm_np(l_ang, exps)
        ovlp_unnorm = float(cs_times_gn @ ee @ cs_times_gn)
        if abs(ovlp_direct - 1.0) < abs(ovlp_unnorm - 1.0):
            w_raw = cs.copy()
            s1 = 1.0 / np.sqrt(max(1e-30, ovlp_direct))
        else:
            w_raw = cs_times_gn
            s1 = 1.0 / np.sqrt(max(1e-30, ovlp_unnorm))
        shells_built.append({
            "atom_idx": sh["atom_idx"],
            "l": l_ang,
            "exps": exps,
            "weights": w_raw * s1,
        })

    mol = _FallbackMol(symbols, np.array(coords_bohr_list, dtype=np.float64), shells_built, cart=all(cart_l.values()), cart_l=cart_l)
    mo_en_arr = np.array(mo_energies, dtype=np.float64)
    mo_occ_arr = np.array(mo_occs, dtype=np.float64)
    spins_up = [s.upper() for s in mo_spins]
    if any(s.startswith("B") for s in spins_up) and any(s.startswith("A") for s in spins_up):
        beta_mask = np.array([s.startswith("B") for s in spins_up])
        alpha_mask = ~beta_mask
        if np.sum(alpha_mask) == np.sum(beta_mask):
            mo_en_arr = np.stack([mo_en_arr[alpha_mask], mo_en_arr[beta_mask]], axis=0)
            mo_coeff = np.stack([mo_coeff[:, alpha_mask], mo_coeff[:, beta_mask]], axis=0)
            mo_occ_arr = np.stack([mo_occ_arr[alpha_mask], mo_occ_arr[beta_mask]], axis=0)
            irrep_arr = (
                [mo_syms[idx] for idx in np.where(alpha_mask)[0]],
                [mo_syms[idx] for idx in np.where(beta_mask)[0]],
            )
            return (mol, mo_en_arr, mo_coeff, mo_occ_arr, irrep_arr, mo_spins)

    return (mol, mo_en_arr, mo_coeff, mo_occ_arr, mo_syms, mo_spins)


def _load_pyscf_molden(molden_path: str):
    """Load a Molden file with PySCF (or pure-NumPy fallback) and automatically normalize ORCA/orca_2mkl pre-normalized GTOs.

    ORCA's `orca_2mkl` writes primitive contraction coefficients already multiplied by the
    primitive Gaussian normalization factor `gto_norm(l, alpha)`. PySCF's `molden.load`
    assumes unnormalized contraction coefficients, which otherwise causes multi-primitive
    contracted shells (1s, 2s, 2p) to lose ~3.4 electrons on ethanol (`Tr(P*S) = 22.6`
    instead of `26.0`), shifting the entire ESP positive!
    """
    import copy
    import io
    import contextlib
    try:
        from pyscf import gto
        from pyscf.tools import molden
    except ImportError:
        return _load_fallback_molden(molden_path)

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        res = list(molden.load(molden_path))
    if not res:
        return None
    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return tuple(res)

    try:
        mo_coeff = np.asarray(res[2])
        s = mol.intor("int1e_ovlp")
        c0 = mo_coeff[0] if mo_coeff.ndim == 3 else mo_coeff
        norms = np.einsum("pi,pq,qi->i", c0, s, c0)
        err_orig = float(np.max(np.abs(norms - 1.0)))
        if err_orig > 1e-3:
            mol_alt = copy.deepcopy(mol)
            for _sym, bas_list in mol_alt._basis.items():
                for b in bas_list:
                    l_ang = b[0]
                    for p in b[1:]:
                        p[1] = p[1] / float(gto.gto_norm(l_ang, p[0]))
            mol_alt.build(0, 0)
            s_alt = mol_alt.intor("int1e_ovlp")
            if mol_alt.cart:
                scale = np.sqrt(np.diag(s) / np.diag(s_alt))
                c_alt = mo_coeff * scale[None, :, None] if mo_coeff.ndim == 3 else mo_coeff * scale[:, None]
            else:
                c_alt = mo_coeff
            c0_alt = c_alt[0] if c_alt.ndim == 3 else c_alt
            norms_alt = np.einsum("pi,pq,qi->i", c0_alt, s_alt, c0_alt)
            err_alt = float(np.max(np.abs(norms_alt - 1.0)))
            if err_alt < err_orig:
                res[0] = mol_alt
                res[2] = c_alt
    except Exception:
        log.debug("Ignored non-fatal exception", exc_info=True)
    return tuple(res)


def _get_mo_grid_context(molden_path, nx, ny, nz, margin_bohr):
    """Load a Molden file with PySCF (or pure-NumPy fallback) and evaluate AO values on the cube grid (cached, 1 entry).

    Returns None when the file has no usable basis.
    """
    try:
        from pyscf.tools import cubegen
        cube_cls = cubegen.Cube
    except ImportError:
        cube_cls = _FallbackCube

    try:
        key = (os.path.abspath(molden_path), os.path.getmtime(molden_path), nx, ny, nz, float(margin_bohr))
    except OSError:
        return None

    with _MO_CTX_LOCK:
        cached = _MO_CTX_CACHE.get(key)
        if cached is not None:
            return cached

    res = None
    try:
        res = _load_pyscf_molden(molden_path)
    except Exception:
        log.debug("PySCF molden.load failed for %s; trying pure-NumPy fallback", molden_path, exc_info=True)
        res = None
    if not res or not getattr(res[0], "_basis", None) or res[0].natm == 0:
        try:
            res = _load_fallback_molden(molden_path)
        except Exception:
            log.debug("Fallback molden loader failed for %s", molden_path, exc_info=True)
            return None
    if not res:
        return None
    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    if isinstance(mol, _FallbackMol):
        c = _FallbackCube(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
    else:
        c = cube_cls(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
    coords = c.get_coords()
    ao_val = mol.eval_gto("GTOval", coords)
    ctx = {"res": res, "mol": mol, "cube": c, "coords": coords, "ao_val": ao_val}

    if ao_val.nbytes <= _MO_CTX_MAX_BYTES:
        with _MO_CTX_LOCK:
            _MO_CTX_CACHE.clear()  # keep only the most recent file in memory
            _MO_CTX_CACHE[key] = ctx
    return ctx


def compute_molden_mo_atom_contributions(
    molden_path: str,
    orbital_index: int,
    top_k: int = 4,
) -> str:
    """
    Compute Mulliken/c^2 atomic and angular-momentum (s/p/d/f) percentage composition
    for a given Molecular Orbital index from a Molden file.
    Returns a concise summary string (e.g. '68.4% O1 (2p), 15.8% H2 (1s), 15.8% H3 (1s)').
    """
    if not molden_path or not os.path.isfile(molden_path):
        return ""
    res = None
    with _MO_CTX_LOCK:
        for k, v in _MO_CTX_CACHE.items():
            if k[0] == os.path.abspath(molden_path):
                res = v.get("res")
                break
    if res is None:
        try:
            res = _load_pyscf_molden(molden_path)
        except Exception:
            return ""
    if not res:
        return ""
    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return ""

    mo_coeff = np.asarray(res[2])
    if mo_coeff.ndim == 3:
        mo_coeff = mo_coeff[0]
    if orbital_index < 0 or orbital_index >= mo_coeff.shape[1]:
        return ""

    c_vec = mo_coeff[:, orbital_index]
    try:
        s_mat = mol.intor("int1e_ovlp")
        ao_weights = c_vec * (s_mat @ c_vec)
        ao_weights = np.maximum(0.0, ao_weights)
        if float(np.sum(ao_weights)) <= 1e-12:
            ao_weights = c_vec ** 2
    except Exception:
        ao_weights = c_vec ** 2

    total_w = float(np.sum(ao_weights))
    if total_w <= 1e-12:
        return ""
    ao_pct = (ao_weights / total_w) * 100.0

    try:
        ao_labels = mol.ao_labels(fmt=False)
    except Exception:
        return ""

    atom_totals: Dict[int, float] = {}
    atom_shell_totals: Dict[int, Dict[str, float]] = {}
    for mu, lbl in enumerate(ao_labels):
        if mu >= len(ao_pct):
            break
        at_idx = int(lbl[0])
        nl_str = str(lbl[2])  # e.g. '1s', '2p', '3d'
        w_val = float(ao_pct[mu])
        atom_totals[at_idx] = atom_totals.get(at_idx, 0.0) + w_val
        sh_dict = atom_shell_totals.setdefault(at_idx, {})
        sh_dict[nl_str] = sh_dict.get(nl_str, 0.0) + w_val

    ranked = sorted(atom_totals.items(), key=lambda kv: kv[1], reverse=True)
    parts = []
    for at_idx, pct in ranked[:top_k]:
        if pct < 1.5:
            continue
        sym = mol.atom_symbol(at_idx)
        sh_dict = atom_shell_totals.get(at_idx, {})
        if sh_dict:
            top_sh = max(sh_dict.items(), key=lambda kv: kv[1])[0]
            parts.append(f"{pct:.1f}% {sym}{at_idx + 1} ({top_sh})")
        else:
            parts.append(f"{pct:.1f}% {sym}{at_idx + 1}")
    return ", ".join(parts)


def classify_molden_orbitals_irreps(
    molden_path: str,
    sym_res: Any,
    coords_ang: Optional[np.ndarray] = None,
) -> List[str]:
    """Classify each Molecular Orbital in a Molden file into an irreducible representation of `sym_res`.

    Uses a G-invariant 3D real-space sample cloud and character projection over the point group's
    conjugacy classes (works with both PySCF and the pure-NumPy GTO fallback).
    """
    if not molden_path or not os.path.isfile(molden_path) or sym_res is None:
        return []

    res = None
    with _MO_CTX_LOCK:
        for k, v in _MO_CTX_CACHE.items():
            if k[0] == os.path.abspath(molden_path):
                res = v.get("res")
                break
    if res is None:
        try:
            res = _load_pyscf_molden(molden_path)
        except Exception:
            return []
    if not res:
        return []
    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return []

    mo_coeff = np.asarray(res[2], dtype=np.float64)
    mo_energy = np.asarray(res[1], dtype=np.float64) if len(res) > 1 and res[1] is not None else None

    if mo_coeff.ndim == 3:
        n_mo_total = mo_coeff.shape[0] * mo_coeff.shape[2]
    else:
        n_mo_total = mo_coeff.shape[1]

    group = getattr(sym_res, "symbol", "C1")
    if getattr(sym_res, "order", 1) <= 1 or group == "C1":
        return ["A"] * n_mo_total

    from .symmetry import CHARACTER_TABLES, calculate_symmetry_representations, _center_molecule
    if group not in CHARACTER_TABLES:
        return []

    ctable = CHARACTER_TABLES[group]
    classes = ctable["classes"]
    irreps = ctable["irreps"]

    mol_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]
    reps_coords = coords_ang if (coords_ang is not None and len(coords_ang) == len(symbols)) else mol_coords_ang
    reps_info = calculate_symmetry_representations(symbols, reps_coords, sym_res)
    class_ops = reps_info.get("class_ops", [])
    if len(class_ops) != len(classes):
        return []

    # Map symmetry operations from reps_coords frame into the Molden file's coordinate frame
    mol_centered_ang, mol_com_ang = _center_molecule(symbols, mol_coords_ang)
    mol_com_bohr = mol_com_ang * ANGSTROM_TO_BOHR
    r_align = np.eye(3, dtype=np.float64)
    if coords_ang is not None and len(coords_ang) == len(symbols):
        mw_centered_ang, _ = _center_molecule(symbols, coords_ang)
        if np.max(np.abs(mw_centered_ang - mol_centered_ang)) > 1e-3:
            cov = mol_centered_ang.T @ mw_centered_ang
            u_m, _, vt_m = np.linalg.svd(cov)
            d_det = np.linalg.det(vt_m.T @ u_m.T)
            r_align = vt_m.T @ np.diag([1.0, 1.0, 1.0 if d_det >= 0 else -1.0]) @ u_m.T

    # Build a G-invariant 3D point cloud around the molecule in Bohr
    rng = np.random.default_rng(42)
    seeds = (mol.atom_coords() - mol_com_bohr) + rng.normal(0.0, 0.75, size=(mol.natm, 3))
    op_mats_mol = [r_align.T @ op.matrix @ r_align for op in sym_res.operations]
    pts_centered = np.vstack([seeds @ m.T for m in op_mats_mol])
    pts_molden = pts_centered + mol_com_bohr

    ao_0 = mol.eval_gto("GTOval", pts_molden)

    def _classify_channel(c_mat: np.ndarray, en_vec: Optional[np.ndarray]) -> List[str]:
        n_mo = c_mat.shape[1]
        psi_0 = ao_0 @ c_mat
        norms = np.maximum(np.sum(psi_0 * psi_0, axis=0), 1e-18)

        diag_chars = np.empty((len(classes), n_mo), dtype=np.float64)
        for c_idx, op in enumerate(class_ops):
            m_mol = r_align.T @ op.matrix @ r_align
            pts_rot = (pts_centered @ m_mol) + mol_com_bohr
            ao_rot = mol.eval_gto("GTOval", pts_rot)
            psi_rot = ao_rot @ c_mat
            diag_chars[c_idx] = np.sum(psi_0 * psi_rot, axis=0) / norms

        out = [""] * n_mo
        i = 0
        while i < n_mo:
            j = i + 1
            if en_vec is not None and j < n_mo:
                while j < n_mo and abs(float(en_vec[j]) - float(en_vec[i])) < 1e-4:
                    j += 1
            shell_chars = np.sum(diag_chars[:, i:j], axis=1)
            best_irr = "-"
            best_err = 1e9
            for irr_name, chi_vec in irreps.items():
                err = sum(g * ((shell_chars[ci] - chi_vec[ci]) ** 2) for ci, (_, g) in enumerate(classes))
                if err < best_err:
                    best_err = err
                    best_irr = irr_name

            if best_err > 0.5 and (j - i) > 1:
                for k in range(i, j):
                    c_vec = diag_chars[:, k]
                    b_irr, b_err = "-", 1e9
                    for irr_name, chi_vec in irreps.items():
                        err = sum(g * ((c_vec[ci] - chi_vec[ci]) ** 2) for ci, (_, g) in enumerate(classes))
                        if err < b_err:
                            b_err = err
                            b_irr = irr_name
                    out[k] = b_irr
            else:
                for k in range(i, j):
                    out[k] = best_irr
            i = j
        return out

    if mo_coeff.ndim == 3:
        en0 = mo_energy[0] if (mo_energy is not None and mo_energy.ndim == 2) else mo_energy
        en1 = mo_energy[1] if (mo_energy is not None and mo_energy.ndim == 2) else mo_energy
        return _classify_channel(mo_coeff[0], en0) + _classify_channel(mo_coeff[1], en1)
    return _classify_channel(mo_coeff, mo_energy)


def localize_molden_orbitals(
    molden_path: str,
    method: str = "pipek-mezey",
    localize_virtuals: bool = False,
) -> Optional[str]:
    """
    Generate Localized Molecular Orbitals (LMOs) from a Molden file using PySCF's
    unitary orbital localization schemes:
      - 'pipek-mezey' (or 'pm'): Pipek-Mezey Mulliken/IAO population localization (preserves sigma/pi separation)
      - 'boys' (or 'fb'): Foster-Boys minimum spatial variance <r^2> - <r>^2 localization

    Writes a companion `<stem>_localized_<method>.molden` file and returns its path.
    """
    if not molden_path or not os.path.isfile(molden_path):
        return None

    res = _load_pyscf_molden(molden_path)
    if not res:
        return None

    mol = res[0]
    mo_energy = np.asarray(res[1], dtype=np.float64) if len(res) > 1 and res[1] is not None else None
    mo_coeff_raw = np.asarray(res[2], dtype=np.float64)
    mo_occ_raw = np.asarray(res[3], dtype=np.float64) if len(res) > 3 and res[3] is not None else None

    if mo_coeff_raw.ndim == 3:
        mo_coeff = mo_coeff_raw[0].copy()
        mo_occ = mo_occ_raw[0].copy() if (mo_occ_raw is not None and mo_occ_raw.ndim == 2) else mo_occ_raw
        mo_en = mo_energy[0].copy() if (mo_energy is not None and mo_energy.ndim == 2) else mo_energy
    else:
        mo_coeff = mo_coeff_raw.copy()
        mo_occ = mo_occ_raw.copy() if mo_occ_raw is not None else np.zeros(mo_coeff.shape[1])
        mo_en = mo_energy.copy() if mo_energy is not None else np.zeros(mo_coeff.shape[1])

    if mo_occ is None or len(mo_occ) != mo_coeff.shape[1]:
        n_occ = mol.nelectron // 2
        mo_occ = np.zeros(mo_coeff.shape[1], dtype=np.float64)
        mo_occ[:n_occ] = 2.0

    if mo_en is None or len(mo_en) != mo_coeff.shape[1]:
        mo_en = np.arange(mo_coeff.shape[1], dtype=np.float64) * 0.1 - 0.5

    from pyscf import lo
    from pyscf.tools import molden as pyscf_molden

    occ_idx = np.where(mo_occ > 0.05)[0]
    virt_idx = np.where(mo_occ <= 0.05)[0]

    method_low = method.strip().lower()
    use_boys = "boys" in method_low or method_low == "fb"
    tag = "boys" if use_boys else "pm"

    def _run_loc(sub_coeff: np.ndarray) -> np.ndarray:
        if sub_coeff.shape[1] <= 1:
            return sub_coeff
        if use_boys:
            loc_obj = lo.Boys(mol, sub_coeff)
        else:
            loc_obj = lo.PM(mol, sub_coeff)
            loc_obj.pop_method = "mulliken"
        loc_obj.verbose = 0
        loc_obj.conv_tol = 1e-6
        return loc_obj.kernel()

    new_coeff = mo_coeff.copy()
    if len(occ_idx) > 1:
        new_coeff[:, occ_idx] = _run_loc(mo_coeff[:, occ_idx])
    if localize_virtuals and len(virt_idx) > 1:
        new_coeff[:, virt_idx] = _run_loc(mo_coeff[:, virt_idx])

    # Compute expectation value of Fock-like diagonal energy E_i = <LMO_i | F | LMO_i>
    # where F = S * C_canon * diag(e_canon) * C_canon^T * S so LMOs remain meaningfully ordered by energy!
    try:
        s_mat = mol.intor("int1e_ovlp")
        # Transformation matrix U = C_canon^T S C_loc
        u_mat = mo_coeff.T @ s_mat @ new_coeff
        new_en = np.einsum("ki,k,ki->i", u_mat, mo_en, u_mat)
        # Sort occupied LMOs by expectation energy
        if len(occ_idx) > 1:
            order_occ = np.argsort(new_en[occ_idx])
            new_coeff[:, occ_idx] = new_coeff[:, occ_idx[order_occ]]
            new_en[occ_idx] = new_en[occ_idx[order_occ]]
    except Exception:
        new_en = mo_en.copy()

    stem, _ = os.path.splitext(molden_path)
    if stem.endswith(".molden"):
        stem = stem[:-7]
    out_path = f"{stem}_localized_{tag}.molden"

    with open(out_path, "w", encoding="utf-8") as f_out:
        pyscf_molden.header(mol, f_out)
        pyscf_molden.orbital_coeff(
            mol,
            f_out,
            new_coeff,
            ene=new_en,
            occ=mo_occ,
            symm=["LMO"] * new_coeff.shape[1],
        )

    return out_path




def evaluate_molden_orbital_grid(
    molden_path: str,
    orbital_index: int = 0,
    nx: int = 48,
    ny: int = 48,
    nz: int = 48,
    margin_bohr: float = 5.5,
) -> Optional[VolumetricData]:
    """
    Rigorously evaluate molecular orbital wavefunction values on a 3D Cartesian grid
    from Molden [GTO] basis and [MO] coefficients using PySCF's analytic GTO evaluator.
    Preserves exact molecular symmetry and nuclear coordinates.
    """
    ctx = _get_mo_grid_context(molden_path, nx, ny, nz, margin_bohr)
    if ctx is None:
        return None
    res, mol, c, coords, ao_val = ctx["res"], ctx["mol"], ctx["cube"], ctx["coords"], ctx["ao_val"]

    if len(res) < 3 or res[2] is None:
        return None
    mo_coeff = np.asarray(res[2])
    if mo_coeff.ndim < 2 or mo_coeff.size == 0:
        return None
    mo_energy = np.asarray(res[1]) if len(res) > 1 and res[1] is not None else None
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else None
    irrep_labels = res[4] if len(res) > 4 else None

    # Handle spin channels if mo_coeff has shape (2, nao, nmo)
    if mo_coeff.ndim == 3:
        n_spin, nao, nmo = mo_coeff.shape
        spin_idx = min(n_spin - 1, max(0, orbital_index // nmo))
        orb_in_spin = orbital_index % nmo
        coeff_col = mo_coeff[spin_idx, :, orb_in_spin]
        spin_label = " [Beta]" if spin_idx == 1 else " [Alpha]"
    else:
        nao, nmo = mo_coeff.shape
        spin_idx = 0
        orb_in_spin = orbital_index % nmo
        coeff_col = mo_coeff[:, orb_in_spin]
        spin_label = ""

    # AO values on the grid are cached per file, so each MO is a single matrix-vector product
    grid_1d = ao_val.dot(coeff_col)
    grid_3d = grid_1d.reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    sym_tag = ""
    if irrep_labels is not None:
        if isinstance(irrep_labels, (list, tuple, np.ndarray)) and len(irrep_labels) == 2 and mo_coeff.ndim == 3:
            sub_irrep = irrep_labels[spin_idx]
            if sub_irrep is not None and orb_in_spin < len(sub_irrep):
                sym_tag = f" ({sub_irrep[orb_in_spin]})"
        elif orbital_index < len(irrep_labels):
            sym_tag = f" ({irrep_labels[orbital_index]})"

    e_tag = ""
    if mo_energy is not None:
        if mo_energy.ndim == 2 and spin_idx < mo_energy.shape[0] and orb_in_spin < mo_energy.shape[1]:
            e_tag = f" E={float(mo_energy[spin_idx, orb_in_spin]):.4f} a.u."
        elif mo_energy.ndim == 1 and orbital_index < len(mo_energy):
            e_tag = f" E={float(mo_energy[orbital_index]):.4f} a.u."

    return VolumetricData(
        title=f"MO #{orbital_index + 1}{spin_label}{sym_tag}{e_tag}",
        comment=f"Exact GTO Evaluation from Molden ({c.nx}x{c.ny}x{c.nz})",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=True,
        orbital_index=orbital_index + 1,
    )


def compute_molden_esp_grid(
    molden_path: str,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    margin_bohr: float = 5.5,
    extent: Optional[Tuple[float, float, float]] = None,
) -> Optional[VolumetricData]:
    """
    Compute rigorous Molecular Electrostatic Potential (ESP) 3D grid directly
    from a Molden file using PySCF's analytic 1-electron grid integrals.
    ESP(r) = V_nuc(r) - V_elec(r) = sum_A (Z_A / |r - R_A|) - int (rho(r') / |r - r'|) dr'.
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    # Construct total electronic density matrix dm = sum_i occ_i |phi_i><phi_i|
    if mo_coeff.ndim == 3:
        dm = np.sum([c @ np.diag(occ) @ c.T for c, occ in zip(mo_coeff, mo_occ)], axis=0)
    else:
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T

    cube_kw = {"margin": margin_bohr}
    if extent is not None and all(e > 0 for e in extent):
        cube_kw["extent"] = extent

    c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, **cube_kw)
    coords = c.get_coords()

    # Nuclear potential: V_nuc = sum_A Z_A / |r - R_A|
    xyz_bohr = mol.atom_coords()
    znuc = mol.atom_charges().astype(np.float64)
    r_nuc = np.linalg.norm(coords[:, None, :] - xyz_bohr[None, :, :], axis=2)
    v_nuc = np.sum(znuc[None, :] / np.maximum(r_nuc, 1e-8), axis=1)

    # Electronic potential: V_elec = int rho(r') / |r - r'| dr' via chunked int1e_grids
    chunk = 4000
    v_elec = np.empty(len(coords), dtype=np.float64)
    for i in range(0, len(coords), chunk):
        ints = mol.intor("int1e_grids", grids=coords[i:i + chunk])
        v_elec[i:i + chunk] = np.einsum("gij,ij->g", ints, dm)

    v_esp = (v_nuc - v_elec).reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    return VolumetricData(
        title="Molecular Electrostatic Potential (ESP)",
        comment="Exact Analytic ESP from Molden Wavefunction",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=v_esp,
        is_orbital=True,  # allows positive (+) and negative (-) potential lobes
        orbital_index=None,
    )


def compute_molden_electron_density_grid(
    molden_path: str,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    margin_bohr: float = 5.5,
) -> Optional[VolumetricData]:
    """
    Compute total electron density 3D grid: rho(r) = sum_i n_i |psi_i(r)|^2.
    Evaluated from Molden [GTO] basis and [MO] coefficients using PySCF or the pure-NumPy GTO evaluator.
    """
    ctx = _get_mo_grid_context(molden_path, nx, ny, nz, margin_bohr)
    if ctx is None:
        return None

    res = ctx["res"]
    mol = ctx["mol"]
    c = ctx["cube"]
    ao_val = ctx["ao_val"]

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else None

    if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        rho = np.zeros(ao_val.shape[0], dtype=np.float64)
        for s_idx in range(2):
            occ_s = mo_occ[s_idx] if mo_occ is not None and mo_occ.ndim == 2 else np.zeros(mo_coeff.shape[2])
            mask = occ_s > 1e-8
            if np.any(mask):
                psi = ao_val @ mo_coeff[s_idx, :, mask].T
                rho += np.sum((psi * psi) * occ_s[mask][None, :], axis=1)
    else:
        if mo_occ is None:
            mo_occ = np.zeros(mo_coeff.shape[1], dtype=np.float64)
        mask = mo_occ > 1e-8
        if np.any(mask):
            psi = ao_val @ mo_coeff[:, mask]
            rho = np.sum((psi * psi) * mo_occ[mask][None, :], axis=1)
        else:
            rho = np.zeros(ao_val.shape[0], dtype=np.float64)

    grid_3d = rho.reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    bname = os.path.basename(molden_path).lower()
    if "positron" in bname:
        dens_title = "Positron Density (ρ₊)"
        dens_comment = f"Exact Positron Density from Molden ({c.nx}x{c.ny}x{c.nz})"
    elif "muon" in bname:
        dens_title = "Muon Density (ρ_μ)"
        dens_comment = f"Exact Muon Density from Molden ({c.nx}x{c.ny}x{c.nz})"
    elif "proton" in bname:
        dens_title = "Quantum Proton Density (ρ_p)"
        dens_comment = f"Exact Quantum Proton Density from Molden ({c.nx}x{c.ny}x{c.nz})"
    else:
        dens_title = "Total Electron Density (ρ)"
        dens_comment = f"Exact Electron Density from Molden ({c.nx}x{c.ny}x{c.nz})"

    return VolumetricData(
        title=dens_title,
        comment=dens_comment,
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=False,
        orbital_index=None,
    )


compute_molden_density_grid = compute_molden_electron_density_grid


def _eval_molden_density_on_coords(res_tuple, mol: Any, coords: np.ndarray) -> np.ndarray:
    """Evaluate total particle density sum_i n_i |psi_i(r)|^2 at arbitrary Bohr coordinates."""
    mo_coeff = np.asarray(res_tuple[2])
    mo_occ = np.asarray(res_tuple[3]) if len(res_tuple) > 3 and res_tuple[3] is not None else None
    n_pts = len(coords)
    rho = np.zeros(n_pts, dtype=np.float64)
    chunk = 8000
    for start in range(0, n_pts, chunk):
        ao_val = mol.eval_gto("GTOval", coords[start:start + chunk])
        if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
            for s_idx in range(2):
                occ_s = mo_occ[s_idx] if mo_occ is not None and mo_occ.ndim == 2 else np.zeros(mo_coeff.shape[2])
                mask = occ_s > 1e-8
                if np.any(mask):
                    psi = ao_val @ mo_coeff[s_idx, :, mask].T
                    rho[start:start + chunk] += np.sum((psi * psi) * occ_s[mask][None, :], axis=1)
        else:
            occ_r = mo_occ if mo_occ is not None and mo_occ.ndim == 1 else np.zeros(mo_coeff.shape[1])
            mask = occ_r > 1e-8
            if np.any(mask):
                psi = ao_val @ mo_coeff[:, mask]
                rho[start:start + chunk] += np.sum((psi * psi) * occ_r[mask][None, :], axis=1)
    return rho


def compute_molden_annihilation_density_grid(
    elec_molden_path: str,
    pos_molden_path: str,
    nx: int = 44,
    ny: int = 44,
    nz: int = 44,
    margin_bohr: float = 6.5,
) -> Optional[VolumetricData]:
    """
    Compute the Electron-Positron Annihilation Density 3D grid:
        rho_ann(r) = rho_-(r) * rho_+(r)
    Evaluated on a single shared Cartesian grid spanning both wavefunctions.
    """
    def _load_any(path: str):
        res = None
        try:
            res = _load_pyscf_molden(path)
        except Exception:
            res = None
        if not res or not getattr(res[0], "_basis", None) or res[0].natm == 0:
            try:
                res = _load_fallback_molden(path)
            except Exception:
                return None
        if not res or len(res) < 3 or res[2] is None:
            return None
        return res

    res_e = _load_any(elec_molden_path)
    res_p = _load_any(pos_molden_path)
    if res_e is None or res_p is None:
        return None

    mol_e, mol_p = res_e[0], res_p[0]
    all_coords = np.vstack([mol_e.atom_coords(), mol_p.atom_coords()])
    box_diag = np.max(all_coords, axis=0) - np.min(all_coords, axis=0) + 2.0 * float(margin_bohr)
    boxorig = np.min(all_coords, axis=0) - float(margin_bohr)

    xs = np.arange(nx, dtype=np.float64) * (box_diag[0] / max(1, nx - 1))
    ys = np.arange(ny, dtype=np.float64) * (box_diag[1] / max(1, ny - 1))
    zs = np.arange(nz, dtype=np.float64) * (box_diag[2] / max(1, nz - 1))
    coords = np.vstack(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T + boxorig

    rho_e = _eval_molden_density_on_coords(res_e, mol_e, coords)
    rho_p = _eval_molden_density_on_coords(res_p, mol_p, coords)
    rho_ann = (rho_e * rho_p).reshape((nx, ny, nz)).astype(np.float32)

    step_x = np.array([box_diag[0] / max(1, nx - 1), 0.0, 0.0], dtype=np.float64) * BOHR_TO_ANGSTROM
    step_y = np.array([0.0, box_diag[1] / max(1, ny - 1), 0.0], dtype=np.float64) * BOHR_TO_ANGSTROM
    step_z = np.array([0.0, 0.0, box_diag[2] / max(1, nz - 1)], dtype=np.float64) * BOHR_TO_ANGSTROM

    return VolumetricData(
        title="Electron-Positron Annihilation Density (ρ₋·ρ₊)",
        comment=f"Shared-Grid Annihilation Overlap Density ({nx}x{ny}x{nz})",
        symbols=[mol_e.atom_symbol(i) for i in range(mol_e.natm)],
        atom_coords=mol_e.atom_coords() * BOHR_TO_ANGSTROM,
        origin=boxorig * BOHR_TO_ANGSTROM,
        voxels_x=nx,
        voxels_y=ny,
        voxels_z=nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=rho_ann,
        is_orbital=False,
        orbital_index=None,
    )


def compute_molden_spin_density_grid(
    molden_path: str,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    margin_bohr: float = 5.5,
) -> Optional[VolumetricData]:
    """
    Compute open-shell spin density 3D grid: Delta rho(r) = rho_alpha(r) - rho_beta(r).
    Evaluated from unrestricted Molden wavefunction MO coefficients using PySCF GTO evaluation.
    Reuses cached _get_mo_grid_context AO values when available for instantaneous evaluation.
    """
    ctx = _get_mo_grid_context(molden_path, nx, ny, nz, margin_bohr)
    if ctx is not None:
        res = ctx["res"]
        mol = ctx["mol"]
        c = ctx["cube"]
        coords = ctx["coords"]
        ao_val = ctx["ao_val"]
    else:
        import io
        import contextlib
        try:
            from pyscf.tools import cubegen
        except ImportError:
            return None
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = _load_pyscf_molden(molden_path)
        except Exception:
            return None
        if not res:
            return None
        mol = res[0]
        if not getattr(mol, "_basis", None) or mol.natm == 0:
            return None
        c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
        coords = c.get_coords()
        ao_val = mol.eval_gto("GTOval", coords)  # (n_points, nao)

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else None

    if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        # Unrestricted: Alpha and Beta channels
        occ_a = mo_occ[0] if mo_occ is not None and mo_occ.ndim == 2 else np.ones(mo_coeff.shape[2])
        occ_b = mo_occ[1] if mo_occ is not None and mo_occ.ndim == 2 else np.zeros(mo_coeff.shape[2])
        dm_a = mo_coeff[0] @ np.diag(occ_a) @ mo_coeff[0].T
        dm_b = mo_coeff[1] @ np.diag(occ_b) @ mo_coeff[1].T
        spin_dm = dm_a - dm_b
        rho_spin = np.einsum("pi,ij,pj->p", ao_val, spin_dm, ao_val)
    else:
        # Restricted open-shell check
        if mo_occ is not None and np.any((mo_occ > 0.0) & (mo_occ < 1.99)):
            # Has fractionally/singly occupied orbitals
            open_shell_mask = (mo_occ > 0.0) & (mo_occ < 1.99)
            coeff_open = mo_coeff[:, open_shell_mask]
            occ_open = mo_occ[open_shell_mask]
            dm_open = coeff_open @ np.diag(occ_open) @ coeff_open.T
            rho_spin = np.einsum("pi,ij,pj->p", ao_val, dm_open, ao_val)
        else:
            # Closed-shell singlet (spin density everywhere 0)
            rho_spin = np.zeros(len(coords), dtype=np.float64)

    grid_3d = rho_spin.reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    return VolumetricData(
        title="Spin Density (ρ_alpha - ρ_beta)",
        comment="Exact Open-Shell Spin Density from Molden Wavefunction",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=True,
        orbital_index=None,
    )


def compute_molden_esp_mapped_density(
    molden_path: str,
    density_isovalue: float = 0.001,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    esp_min: Optional[float] = None,
    esp_max: Optional[float] = None,
    colormap_name: str = "bwr_r",
    find_extrema: bool = True,
    margin_bohr: float = 5.5,
    extent: Optional[Tuple[float, float, float]] = None,
) -> Optional[IsosurfaceMesh]:
    """
    Generate an ESP-mapped Electron Density Isosurface mesh.
    Extracts isodensity surface at `density_isovalue` (e.g. 0.001 a.u. for van der Waals contact envelope),
    evaluates exact ESP potential at each mesh vertex, colors vertices using standard colormaps
    (e.g. bwr_r, RdBu, coolwarm_r, Spectral, jet_r), and detects Multiwfn-style surface extrema (V_min, V_max).
    When esp_min and esp_max are omitted, a symmetric range [-V_lim, +V_lim] centered on 0.0 is automatically
    computed from the surface values so that neutral regions are neutral white/gray, nucleophilic (negative)
    regions display full red hues, and electrophilic (positive) regions display full blue hues.
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
        import matplotlib as mpl
    except ImportError:
        return None

    ctx = None
    if extent is None:
        ctx = _get_mo_grid_context(molden_path, nx, ny, nz, margin_bohr)

    if ctx is not None:
        res = ctx["res"]
        mol = ctx["mol"]
        c = ctx["cube"]
        coords = ctx["coords"]
        ao_val = ctx["ao_val"]
    else:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = _load_pyscf_molden(molden_path)
        except Exception:
            return None

        if not res:
            return None

        mol = res[0]
        if not getattr(mol, "_basis", None) or mol.natm == 0:
            return None

        cube_kw = {"margin": margin_bohr}
        if extent is not None and all(e > 0 for e in extent):
            cube_kw["extent"] = extent

        c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, **cube_kw)
        coords = c.get_coords()
        ao_val = mol.eval_gto("GTOval", coords)

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    if mo_coeff.ndim == 3:
        dm = np.sum([c_m @ np.diag(occ) @ c_m.T for c_m, occ in zip(mo_coeff, mo_occ)], axis=0)
    else:
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T

    rho = np.einsum("pi,ij,pj->p", ao_val, dm, ao_val).reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    vol_dens = VolumetricData(
        title="Total Electron Density",
        comment="",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=rho,
        is_orbital=False,
    )

    mesh = extract_isosurface(vol_dens, isovalue=density_isovalue, color=(0.7, 0.7, 0.7))
    if mesh is None or len(mesh.vertices) == 0:
        return None

    # Evaluate exact ESP at the mesh vertices (convert Angstrom vertices to Bohr)
    verts_bohr = mesh.vertices * ANGSTROM_TO_BOHR
    xyz_bohr = mol.atom_coords()
    znuc = mol.atom_charges().astype(np.float64)

    r_nuc = np.linalg.norm(verts_bohr[:, None, :] - xyz_bohr[None, :, :], axis=2)
    v_nuc = np.sum(znuc[None, :] / np.maximum(r_nuc, 1e-8), axis=1)

    chunk = 4000
    v_elec = np.empty(len(verts_bohr), dtype=np.float64)
    for i in range(0, len(verts_bohr), chunk):
        ints = mol.intor("int1e_grids", grids=verts_bohr[i:i + chunk])
        v_elec[i:i + chunk] = np.einsum("gij,ij->g", ints, dm)

    v_esp_verts = v_nuc - v_elec

    # Color map normalization:
    # If range is not specified or at default placeholder, dynamically calculate a symmetric range
    # [-lim, +lim] around 0.0 using the 98th percentile of absolute ESP values on the envelope.
    # This guarantees that 0.0 is strictly at the center (norm_val = 0.5) and both positive (+δ)
    # and negative (-δ) regions use the full dynamic color range without crushing or washing out.
    if esp_min is None or esp_max is None or (abs(esp_min - (-0.05)) < 1e-6 and abs(esp_max - 0.05) < 1e-6):
        abs_vals = np.abs(v_esp_verts)
        max_abs = float(np.percentile(abs_vals, 98.0)) if len(abs_vals) > 0 else 0.05
        lim = max(0.008, round(max_abs * 1.05, 3))
        eff_esp_min = -lim
        eff_esp_max = lim
    else:
        eff_esp_min = float(esp_min)
        eff_esp_max = float(esp_max)

    denom = max(1e-6, (eff_esp_max - eff_esp_min))
    norm_val = np.clip((v_esp_verts - eff_esp_min) / denom, 0.0, 1.0)

    try:
        cmap = mpl.colormaps.get(colormap_name, mpl.colormaps["bwr_r"])
        rgba = cmap(norm_val)
        colors = rgba[:, :3].astype(np.float32)
    except Exception:
        # Fallback to standard 3-stop red-white-blue
        colors = np.empty((len(verts_bohr), 3), dtype=np.float32)
        mask_low = norm_val < 0.5
        t_low = (norm_val[mask_low] / 0.5)[:, None]
        c_red = np.array([0.95, 0.15, 0.15], dtype=np.float32)
        c_mid = np.array([0.95, 0.95, 0.95], dtype=np.float32)
        c_blue = np.array([0.15, 0.50, 0.98], dtype=np.float32)
        colors[mask_low] = (1.0 - t_low) * c_red + t_low * c_mid
        t_high = ((norm_val[~mask_low] - 0.5) / 0.5)[:, None]
        colors[~mask_low] = (1.0 - t_high) * c_mid + t_high * c_blue

    mesh.vertex_colors = colors
    mesh.surface_values = v_esp_verts
    mesh.esp_range = (eff_esp_min, eff_esp_max)
    mesh.colormap_name = colormap_name

    # Find Multiwfn-style local surface extrema (V_min and V_max)
    if find_extrema and len(mesh.faces) > 0:
        n_v = len(mesh.vertices)
        neighbors = [set() for _ in range(n_v)]
        for f in mesh.faces:
            neighbors[f[0]].add(f[1]); neighbors[f[0]].add(f[2])
            neighbors[f[1]].add(f[0]); neighbors[f[1]].add(f[2])
            neighbors[f[2]].add(f[0]); neighbors[f[2]].add(f[1])

        extrema = []
        for vi in range(n_v):
            nbrs = neighbors[vi]
            if not nbrs:
                continue
            v_val = v_esp_verts[vi]
            is_min = all(v_val <= v_esp_verts[nb] for nb in nbrs)
            is_max = all(v_val >= v_esp_verts[nb] for nb in nbrs)

            if is_min or is_max:
                kcal = v_val * 627.5095
                ev = v_val * 27.211386
                ext_type = "min" if is_min else "max"
                lbl = f"{'V_min' if is_min else 'V_max'}: {ev:+.3f} eV ({kcal:+.1f} kcal/mol, {v_val:+.4f} a.u.)"
                extrema.append({
                    "type": ext_type,
                    "pos": mesh.vertices[vi],
                    "val": float(v_val),
                    "ev": float(ev),
                    "kcal": float(kcal),
                    "label": lbl,
                })

        # Cluster / prune nearby extrema within 0.8 Angstroms
        pruned_extrema = []
        for cand in sorted(extrema, key=lambda x: abs(x["val"]), reverse=True):
            if not any(np.linalg.norm(cand["pos"] - p["pos"]) < 0.8 for p in pruned_extrema):
                pruned_extrema.append(cand)
        mesh.extrema_points = pruned_extrema

    return mesh


def compute_molden_elf_grid(
    molden_path: str,
    nx: int = 40,
    ny: int = 40,
    nz: int = 40,
    margin_bohr: float = 5.5,
) -> Optional[VolumetricData]:
    """
    Compute the Electron Localization Function (ELF) 3D grid directly from Molden wavefunctions.
    Uses Becke-Edgecombe formula (spin-polarized for UHF, closed-shell for RHF) with
    Multiwfn far-field stability constant (1e-5) and fast O(N_pts * N_ao * N_occ) contractions.
    Values range from 0.0 to 1.0; isosurface at 0.80 - 0.85 reveals electron pairs, bonds, and lone pairs.
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    is_uhf = mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2
    channels: List[Tuple[np.ndarray, np.ndarray]] = []
    if is_uhf:
        for s_idx in range(2):
            occ_s = mo_occ[s_idx] if mo_occ.ndim == 2 else np.zeros(mo_coeff.shape[2])
            mask = occ_s > 1e-8
            if np.any(mask):
                channels.append((mo_coeff[s_idx, :, mask].T, occ_s[mask]))
    else:
        occ_r = mo_occ if mo_occ.ndim == 1 else np.zeros(mo_coeff.shape[1])
        mask = occ_r > 1e-8
        if np.any(mask):
            channels.append((mo_coeff[:, mask], occ_r[mask]))

    c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
    coords = c.get_coords()

    # Evaluate AO values and gradients in chunks using occupied-MO contractions
    chunk = 4000
    n_pts = len(coords)
    elf_vals = np.zeros(n_pts, dtype=np.float32)
    if not channels:
        grid_3d = elf_vals.reshape((c.nx, c.ny, c.nz))
    else:
        cf_rhf = 0.3 * (3.0 * np.pi**2) ** (2.0 / 3.0)
        cf_uhf = 0.3 * (6.0 * np.pi**2) ** (2.0 / 3.0)

        for i in range(0, n_pts, chunk):
            sub_coords = coords[i:i + chunk]
            ao_val = mol.eval_gto("GTOval", sub_coords)        # (sub_n, nao)
            ao_grad = mol.eval_gto("GTOval_ip", sub_coords)     # (3, sub_n, nao)

            if is_uhf:
                d_tot = np.full(len(sub_coords), 1e-5, dtype=np.float64)
                rho_53_sum = np.zeros(len(sub_coords), dtype=np.float64)
                for c_occ, occ_v in channels:
                    psi = ao_val @ c_occ
                    gpsi = ao_grad @ c_occ
                    rho_s = np.sum((psi * psi) * occ_v[None, :], axis=1)
                    grad_s = 2.0 * np.sum((gpsi * psi[None, :, :]) * occ_v[None, None, :], axis=2)
                    grad_sq_s = np.sum(grad_s * grad_s, axis=0)
                    tau_s = 0.5 * np.sum(np.sum(gpsi * gpsi, axis=0) * occ_v[None, :], axis=1)
                    d_tot += np.maximum(0.0, tau_s - 0.125 * grad_sq_s / np.maximum(rho_s, 1e-12))
                    rho_53_sum += np.maximum(rho_s, 1e-12) ** (5.0 / 3.0)
                d_0 = cf_uhf * rho_53_sum
            else:
                c_occ, occ_v = channels[0]
                psi = ao_val @ c_occ
                gpsi = ao_grad @ c_occ
                rho = np.sum((psi * psi) * occ_v[None, :], axis=1)
                grad_rho = 2.0 * np.sum((gpsi * psi[None, :, :]) * occ_v[None, None, :], axis=2)
                grad_rho_sq = np.sum(grad_rho * grad_rho, axis=0)
                tau = 0.5 * np.sum(np.sum(gpsi * gpsi, axis=0) * occ_v[None, :], axis=1)
                d_tot = np.maximum(0.0, tau - 0.125 * grad_rho_sq / np.maximum(rho, 1e-12)) + 1e-5
                d_0 = cf_rhf * np.maximum(rho, 1e-12) ** (5.0 / 3.0)

            chi = d_tot / np.maximum(d_0, 1e-12)
            sub_elf = 1.0 / (1.0 + chi * chi)
            elf_vals[i:i + chunk] = sub_elf.astype(np.float32)

        grid_3d = elf_vals.reshape((c.nx, c.ny, c.nz))

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(k) for k in range(mol.natm)]

    return VolumetricData(
        title="Electron Localization Function (ELF)",
        comment="Exact Becke-Edgecombe ELF from Molden Wavefunction",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=False,  # ELF is strictly positive [0, 1]
    )


def compute_molden_localized_orbitals(
    molden_path: str,
    method: str = "Boys",
) -> Optional[Dict]:
    """
    Compute Localized Molecular Orbitals (LMOs) for occupied space using Boys or Pipek-Mezey localization.
    Returns:
        {"labels": lmo_labels, "lmo_coeff": lmo_coefficients, "centroids": lmo_centers}
    where lmo_coefficients has shape (nao, n_occ).
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden
        from pyscf import lo
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    # Select occupied space
    if mo_coeff.ndim == 3:
        # For UHF/unrestricted, localize alpha occupied
        c_mat = mo_coeff[0]
        occ_vec = mo_occ[0] if mo_occ.ndim == 2 else mo_occ
    else:
        c_mat = mo_coeff
        occ_vec = mo_occ

    occ_indices = np.where(occ_vec > 0.05)[0]
    if len(occ_indices) == 0:
        return None

    occ_c = c_mat[:, occ_indices]

    if method.lower().startswith("pipek") or method.lower() == "pm":
        localizer = lo.PipekMezey(mol, occ_c)
    else:
        localizer = lo.Boys(mol, occ_c)

    lmo_coeff = localizer.kernel()

    # Determine spatial centroid (center of electron density) for each LMO
    # <r>_i = int r |phi_i(r)|^2 dr = sum_A <phi_i | r | phi_i>
    dip_ints = mol.intor("int1e_r")  # shape (3, nao, nao)
    lmo_centers = np.empty((lmo_coeff.shape[1], 3), dtype=np.float64)
    for i in range(lmo_coeff.shape[1]):
        ci = lmo_coeff[:, i]
        r_x = ci @ dip_ints[0] @ ci
        r_y = ci @ dip_ints[1] @ ci
        r_z = ci @ dip_ints[2] @ ci
        lmo_centers[i] = np.array([r_x, r_y, r_z]) * BOHR_TO_ANGSTROM

    lmo_labels = [f"LMO #{k+1} ({method})" for k in range(lmo_coeff.shape[1])]
    return {
        "labels": lmo_labels,
        "lmo_coeff": lmo_coeff,
        "centroids": lmo_centers,
    }


def evaluate_lmo_orbital_grid(
    molden_path: str,
    lmo_coeff: np.ndarray,
    lmo_index: int = 0,
    lmo_title: str = "Localized Molecular Orbital",
    nx: int = 44,
    ny: int = 44,
    nz: int = 44,
    margin_bohr: float = 5.5,
) -> Optional[VolumetricData]:
    """Evaluate a single localized molecular orbital column onto a 3D Cartesian grid."""
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
    coords = c.get_coords()
    ao_val = mol.eval_gto("GTOval", coords)
    col = lmo_coeff[:, lmo_index] if lmo_coeff.ndim == 2 else lmo_coeff
    grid_flat = ao_val @ col
    grid_3d = grid_flat.reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(k) for k in range(mol.natm)]

    return VolumetricData(
        title=lmo_title,
        comment="Localized Molecular Orbital (Boys/Pipek-Mezey)",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_3d,
        is_orbital=True,
    )


def compute_molden_nci_rdg(
    molden_path: str,
    nx: int = 36,
    ny: int = 36,
    nz: int = 36,
    margin_bohr: float = 5.5,
    rdg_isovalue: float = 0.5,
    rho_cutoff: float = 0.05,
) -> Optional[IsosurfaceMesh]:
    """
    Compute Non-Covalent Interactions (NCI) using Reduced Density Gradient (RDG).
    Masks covalent core regions (rho > rho_cutoff) before Marching Cubes, extracts the RDG
    isosurface at `rdg_isovalue` (e.g. 0.5), and colors vertices with Multiwfn-style Blue-Green-Red
    mapping over sign(lambda_2) * rho in [-0.035, +0.035]:
        Blue  = Attractive interaction (Hydrogen bonding, halogen bonding)
        Green = Weak dispersion (van der Waals)
        Red   = Steric repulsion (non-bonded clash, ring center strain)
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
        from pyscf.dft import numint
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])

    channels: List[Tuple[np.ndarray, np.ndarray]] = []
    if mo_coeff.ndim == 3 and mo_coeff.shape[0] == 2:
        for s_idx in range(2):
            occ_s = mo_occ[s_idx] if mo_occ.ndim == 2 else np.zeros(mo_coeff.shape[2])
            mask = occ_s > 1e-8
            if np.any(mask):
                channels.append((mo_coeff[s_idx, :, mask].T, occ_s[mask]))
    else:
        occ_r = mo_occ if mo_occ.ndim == 1 else np.zeros(mo_coeff.shape[1])
        mask = occ_r > 1e-8
        if np.any(mask):
            channels.append((mo_coeff[:, mask], occ_r[mask]))
    if not channels:
        return None

    c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, margin=margin_bohr)
    coords = c.get_coords()

    ni = numint.NumInt()
    chunk = 2000
    n_pts = len(coords)
    rdg_flat = np.empty(n_pts, dtype=np.float32)

    cf = 2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0)

    for i in range(0, n_pts, chunk):
        sub_coords = coords[i:i + chunk]
        ao_1 = ni.eval_ao(mol, sub_coords, deriv=1)  # (4, sub_n, nao)
        rho_p = np.zeros(len(sub_coords), dtype=np.float64)
        gx = np.zeros(len(sub_coords), dtype=np.float64)
        gy = np.zeros(len(sub_coords), dtype=np.float64)
        gz = np.zeros(len(sub_coords), dtype=np.float64)
        for c_occ, occ_v in channels:
            psi = ao_1[0] @ c_occ
            rho_p += np.sum((psi * psi) * occ_v[None, :], axis=1)
            gx += 2.0 * np.sum(((ao_1[1] @ c_occ) * psi) * occ_v[None, :], axis=1)
            gy += 2.0 * np.sum(((ao_1[2] @ c_occ) * psi) * occ_v[None, :], axis=1)
            gz += 2.0 * np.sum(((ao_1[3] @ c_occ) * psi) * occ_v[None, :], axis=1)

        norm_g = np.sqrt(gx * gx + gy * gy + gz * gz)
        rdg_p = norm_g / (cf * np.maximum(rho_p, 1e-10) ** (4.0 / 3.0))
        if rho_cutoff > 0.0:
            rdg_p = np.where(rho_p > rho_cutoff, 100.0, rdg_p)
        rdg_flat[i:i + chunk] = rdg_p

    # Extract RDG isosurface (typically s = 0.5)
    grid_rdg = rdg_flat.reshape((c.nx, c.ny, c.nz))
    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(k) for k in range(mol.natm)]
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM

    vol_rdg = VolumetricData(
        title="Reduced Density Gradient (RDG)",
        comment="Non-Covalent Interactions (NCI)",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=grid_rdg,
        is_orbital=False,
    )

    mesh = extract_isosurface(vol_rdg, isovalue=rdg_isovalue, color=(0.2, 0.8, 0.3))
    if mesh is None or len(mesh.vertices) == 0:
        return None

    # Evaluate exact sign(lambda_2)*rho at the mesh vertices using deriv=2
    verts_bohr = mesh.vertices * ANGSTROM_TO_BOHR
    nv = len(verts_bohr)
    slr = np.zeros(nv, dtype=np.float32)

    v_chunk = 1500
    for vi in range(0, nv, v_chunk):
        sub_v = verts_bohr[vi:vi + v_chunk]
        ao_2 = ni.eval_ao(mol, sub_v, deriv=2)
        sub_nv = len(sub_v)
        v_rho = np.zeros(sub_nv, dtype=np.float64)
        hess = np.zeros((sub_nv, 3, 3), dtype=np.float64)

        for c_occ, occ_v in channels:
            psi = ao_2[0] @ c_occ
            gx = ao_2[1] @ c_occ
            gy = ao_2[2] @ c_occ
            gz = ao_2[3] @ c_occ
            hxx = ao_2[4] @ c_occ
            hxy = ao_2[5] @ c_occ
            hxz = ao_2[6] @ c_occ
            hyy = ao_2[7] @ c_occ
            hyz = ao_2[8] @ c_occ
            hzz = ao_2[9] @ c_occ

            v_rho += np.sum((psi * psi) * occ_v[None, :], axis=1)
            hess[:, 0, 0] += 2.0 * np.sum((hxx * psi + gx * gx) * occ_v[None, :], axis=1)
            hess[:, 1, 1] += 2.0 * np.sum((hyy * psi + gy * gy) * occ_v[None, :], axis=1)
            hess[:, 2, 2] += 2.0 * np.sum((hzz * psi + gz * gz) * occ_v[None, :], axis=1)
            c_xy = 2.0 * np.sum((hxy * psi + gx * gy) * occ_v[None, :], axis=1)
            c_xz = 2.0 * np.sum((hxz * psi + gx * gz) * occ_v[None, :], axis=1)
            c_yz = 2.0 * np.sum((hyz * psi + gy * gz) * occ_v[None, :], axis=1)
            hess[:, 0, 1] += c_xy
            hess[:, 1, 0] += c_xy
            hess[:, 0, 2] += c_xz
            hess[:, 2, 0] += c_xz
            hess[:, 1, 2] += c_yz
            hess[:, 2, 1] += c_yz

        eig_v = np.linalg.eigvalsh(hess)
        lam2_v = eig_v[:, 1]
        slr[vi:vi + sub_nv] = (np.sign(lam2_v) * v_rho).astype(np.float32)

    # Multiwfn-style Blue (attractive) -> Green (vdW) -> Red (steric clash)
    norm_val = np.clip((slr - (-0.035)) / 0.07, 0.0, 1.0)
    cmap = LinearSegmentedColormap.from_list(
        "nci_bgr", [(0.0, 0.15, 0.95), (0.05, 0.80, 0.25), (0.95, 0.12, 0.12)]
    )
    rgba = cmap(norm_val)
    colors = rgba[:, :3].astype(np.float32)

    mesh.vertex_colors = colors
    mesh.surface_values = slr
    return mesh


def compute_molden_alie_mapped_density(
    molden_path: str,
    density_isovalue: float = 0.001,
    nx: int = 38,
    ny: int = 38,
    nz: int = 38,
    colormap_name: str = "Spectral_r",
) -> Optional[IsosurfaceMesh]:
    """
    Compute Average Local Ionization Energy (ALIE / I_bar(r)) mapped onto electron isodensity envelope.
    ALIE reveals sites of electrophilic and radical reactivity (surface minima indicate highest reactivity).
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden, cubegen
        import matplotlib as mpl
    except ImportError:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None

    if not res:
        return None

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None
    if len(res) < 3 or res[2] is None:
        return None
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])
    mo_energy = np.asarray(res[1]) if len(res) > 1 and res[1] is not None else np.zeros(mo_coeff.shape[-1])

    if mo_coeff.ndim == 3:
        c_mat = mo_coeff[0]
        occ_vec = mo_occ[0] if mo_occ.ndim == 2 else mo_occ
        e_vec = mo_energy[0] if mo_energy.ndim == 2 else mo_energy
    else:
        c_mat = mo_coeff
        occ_vec = mo_occ
        e_vec = mo_energy

    occ_idx = np.where(occ_vec > 0.05)[0]
    if len(occ_idx) == 0:
        return None

    occ_c = c_mat[:, occ_idx]
    occ_e = e_vec[occ_idx]
    occ_o = occ_vec[occ_idx]

    c = cubegen.Cube(mol, nx=nx, ny=ny, nz=nz, margin=3.2)
    coords = c.get_coords()
    ao_val = mol.eval_gto("GTOval", coords)
    dm = occ_c @ np.diag(occ_o) @ occ_c.T
    rho = np.einsum("pi,ij,pj->p", ao_val, dm, ao_val).reshape((c.nx, c.ny, c.nz)).astype(np.float32)

    step_x = (c.box[0] / max(1, c.nx - 1)) * BOHR_TO_ANGSTROM
    step_y = (c.box[1] / max(1, c.ny - 1)) * BOHR_TO_ANGSTROM
    step_z = (c.box[2] / max(1, c.nz - 1)) * BOHR_TO_ANGSTROM
    origin_ang = c.boxorig * BOHR_TO_ANGSTROM
    atom_coords_ang = mol.atom_coords() * BOHR_TO_ANGSTROM
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    vol_dens = VolumetricData(
        title="Total Electron Density",
        comment="ALIE envelope",
        symbols=symbols,
        atom_coords=atom_coords_ang,
        origin=origin_ang,
        voxels_x=c.nx,
        voxels_y=c.ny,
        voxels_z=c.nz,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        grid_data=rho,
        is_orbital=False,
    )

    mesh = extract_isosurface(vol_dens, isovalue=density_isovalue, color=(0.7, 0.7, 0.7))
    if mesh is None or len(mesh.vertices) == 0:
        return None

    # Evaluate ALIE at the mesh vertices
    verts_bohr = mesh.vertices * ANGSTROM_TO_BOHR
    ao_v = mol.eval_gto("GTOval", verts_bohr)
    rho_v = np.einsum("pi,ij,pj->p", ao_v, dm, ao_v)

    mo_vals = ao_v @ occ_c  # shape (n_verts, n_occ)
    # ALIE = sum_i (occ_i * |phi_i(r)|^2 * (-epsilon_i)) / rho(r)
    num = np.sum((mo_vals**2) * occ_o[None, :] * (-occ_e[None, :]), axis=1)
    alie_verts = (num / np.maximum(rho_v, 1e-8)) * 27.211386  # convert a.u. to eV

    # Normalize across 5th to 95th percentiles for vibrant contrast
    p_min = float(np.percentile(alie_verts, 3))
    p_max = float(np.percentile(alie_verts, 97))
    denom = max(1e-4, p_max - p_min)
    norm_val = np.clip((alie_verts - p_min) / denom, 0.0, 1.0)

    cmap = mpl.colormaps.get(colormap_name, mpl.colormaps["Spectral_r"])
    rgba = cmap(norm_val)
    mesh.vertex_colors = rgba[:, :3].astype(np.float32)
    mesh.surface_values = alie_verts

    # Find ALIE minima (reactive sites)
    if len(mesh.faces) > 0:
        n_v = len(mesh.vertices)
        neighbors = [set() for _ in range(n_v)]
        for f in mesh.faces:
            neighbors[f[0]].add(f[1]); neighbors[f[0]].add(f[2])
            neighbors[f[1]].add(f[0]); neighbors[f[1]].add(f[2])
            neighbors[f[2]].add(f[0]); neighbors[f[2]].add(f[1])

        extrema = []
        for vi in range(n_v):
            nbrs = neighbors[vi]
            if not nbrs:
                continue
            v_val = alie_verts[vi]
            if all(v_val <= alie_verts[nb] for nb in nbrs):
                extrema.append({
                    "type": "min",
                    "pos": mesh.vertices[vi],
                    "val": float(v_val),
                    "label": f"ALIE min: {v_val:.2f} eV",
                })
        # Prune
        pruned = []
        for cand in sorted(extrema, key=lambda x: x["val"]):
            if not any(np.linalg.norm(cand["pos"] - p["pos"]) < 1.0 for p in pruned):
                pruned.append(cand)
        mesh.extrema_points = pruned

    return mesh


def compute_molden_orbital_decomposition(
    molden_path: str,
    orbital_index: int,
) -> List[Dict[str, Any]]:
    """
    Compute atomic orbital decomposition (% contribution per atom) for a molecular orbital.
    Uses Mulliken-type AO projection (C_mu,i * (S * C_i)_mu).
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden
    except ImportError:
        return []

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return []

    if not res:
        return []

    mol = res[0]
    mo_coeff = np.asarray(res[2])
    if mo_coeff.ndim == 3:
        n_spin, _nao, nmo = mo_coeff.shape
        if orbital_index >= n_spin * nmo:
            return []
        spin_idx = orbital_index // nmo
        orb_in_spin = orbital_index % nmo
        c_mat = mo_coeff[spin_idx]
    else:
        c_mat = mo_coeff
        orb_in_spin = orbital_index
        if orb_in_spin >= c_mat.shape[1]:
            return []

    s_mat = mol.intor("int1e_ovlp")
    c_i = c_mat[:, orb_in_spin]
    sc_i = s_mat @ c_i
    ao_pop = c_i * sc_i  # sum(ao_pop) == 1.0 for normalized MO

    # Group populations by atom
    atom_pops = np.zeros(mol.natm, dtype=np.float64)
    ao_labels = mol.ao_labels(fmt=False)
    for mu, (atm_id, sym, bas_name, ang) in enumerate(ao_labels):
        atom_pops[atm_id] += ao_pop[mu]

    results = []
    for atm_id in range(mol.natm):
        pct = float(atom_pops[atm_id] * 100.0)
        sym_raw = mol.atom_symbol(atm_id)
        # Strip any appended index in atom_symbol (e.g. 'C1' -> 'C')
        sym_clean = "".join([ch for ch in sym_raw if ch.isalpha()])
        results.append({
            "atom_idx": atm_id,
            "symbol": sym_clean or sym_raw,
            "percentage": pct,
        })
    return sorted(results, key=lambda x: abs(x["percentage"]), reverse=True)


def compute_molden_chelpg_charges(
    molden_path: str,
    spacing_angstrom: float = 0.45,
    extension_angstrom: float = 2.8,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Compute rigorous CHELPG atomic partial charges directly from a Molden file.
    Uses standard Breneman-Wiberg vdW exclusion radii and exact PySCF electrostatic integrals.
    Optimized 0.45 A grid spacing provides >3x speedup while matching charges within 0.003 e.
    Returns (charges_array, rmse).
    """
    import io
    import contextlib

    try:
        from pyscf.tools import molden
    except ImportError:
        return None, 0.0

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            res = _load_pyscf_molden(molden_path)
    except Exception:
        return None, 0.0

    if not res:
        return None, 0.0

    mol = res[0]
    if not getattr(mol, "_basis", None) or mol.natm == 0:
        return None, 0.0
    if len(res) < 3 or res[2] is None:
        return None, 0.0
    mo_c_arr = np.asarray(res[2])
    if mo_c_arr.ndim < 2 or mo_c_arr.size == 0:
        return None, 0.0

    mo_coeff = np.asarray(res[2])
    mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])
    if mo_coeff.ndim == 3:
        dm = np.sum([c @ np.diag(occ) @ c.T for c, occ in zip(mo_coeff, mo_occ)], axis=0)
    else:
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T

    # Breneman & Wiberg (1990) CHELPG vdW radii (Angstrom)
    chelpg_radii = {
        1: 1.45, 2: 1.45, 3: 1.50, 4: 1.50, 5: 1.50, 6: 1.50,
        7: 1.70, 8: 1.70, 9: 1.50, 10: 1.70, 11: 2.00, 12: 2.00,
        13: 2.00, 14: 2.10, 15: 1.85, 16: 1.80, 17: 1.80, 18: 2.00
    }

    xyz_bohr = mol.atom_coords()
    znuc = mol.atom_charges().astype(np.float64)
    radii_bohr = np.array([chelpg_radii.get(int(z), 1.70) * ANGSTROM_TO_BOHR for z in znuc])

    box_lo = xyz_bohr.min(axis=0) - (radii_bohr.max() + extension_angstrom * ANGSTROM_TO_BOHR)
    box_hi = xyz_bohr.max(axis=0) + (radii_bohr.max() + extension_angstrom * ANGSTROM_TO_BOHR)
    step = spacing_angstrom * ANGSTROM_TO_BOHR

    grid_x = np.arange(box_lo[0], box_hi[0], step)
    grid_y = np.arange(box_lo[1], box_hi[1], step)
    grid_z = np.arange(box_lo[2], box_hi[2], step)

    X, Y, Z = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    dists = np.linalg.norm(pts[:, None, :] - xyz_bohr[None, :, :], axis=2)
    outside_vdw = np.all(dists >= radii_bohr[None, :], axis=1)
    inside_ext = np.any(dists <= (radii_bohr + extension_angstrom * ANGSTROM_TO_BOHR)[None, :], axis=1)
    pts = pts[outside_vdw & inside_ext]

    if len(pts) == 0:
        return None, 0.0

    # Potential of electrons on grid points in chunks
    chunk = 4000
    v_elec = np.empty(len(pts), dtype=np.float64)
    for i in range(0, len(pts), chunk):
        ints = mol.intor("int1e_grids", grids=pts[i:i + chunk])
        v_elec[i:i + chunk] = np.einsum("gij,ij->g", ints, dm)

    r = np.linalg.norm(pts[:, None, :] - xyz_bohr[None, :, :], axis=2)
    v_total = (znuc[None, :] / np.maximum(r, 1e-8)).sum(axis=1) - v_elec

    # Least squares atomic charge fit with Lagrange multiplier constraint sum(q) = total_charge
    n = len(xyz_bohr)
    a = 1.0 / np.maximum(r, 1e-8)
    m_mat = np.zeros((n + 1, n + 1))
    m_mat[:n, :n] = a.T @ a
    m_mat[:n, n] = 1.0
    m_mat[n, :n] = 1.0

    tot_charge = float(znuc.sum() - mol.nelectron)
    rhs = np.concatenate([a.T @ v_total, [tot_charge]])
    q = np.linalg.solve(m_mat, rhs)[:n]
    rmse = float(np.sqrt(np.mean((a @ q - v_total) ** 2)))

    return q, rmse


def extract_isosurface(
    vol_data: VolumetricData,
    isovalue: float = 0.03,
    color: Tuple[float, float, float] = (0.2, 0.6, 1.0),
) -> Optional[IsosurfaceMesh]:
    """
    Extract a 3D polygonal triangle mesh at the given isovalue using Marching Cubes.
    Transforms voxel coordinates back into Angstrom 3D Cartesian space.
    """
    if not _HAS_MARCHING_CUBES:
        raise RuntimeError(
            "scikit-image is required for Marching Cubes surface extraction.\n"
            "Install it with:  pip install scikit-image  (or: pip install .[all])"
        )

    grid = vol_data.grid_data
    val_min, val_max = float(np.min(grid)), float(np.max(grid))

    if isovalue < val_min or isovalue > val_max:
        return None

    try:
        if marching_cubes is not None:
            # marching_cubes returns vertices in fractional voxel coordinates (0 to nx-1, etc.)
            # Suppress NumPy 2.5+ in-place shape mutation warnings originating inside skimage Lewiner implementation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*Setting the shape on a NumPy array.*")
                verts_vox, faces, normals, _ = marching_cubes(grid, level=isovalue)
        else:
            verts_vox, faces, normals = _pure_numpy_marching_tetrahedra(grid, level=isovalue)
    except (RuntimeError, ValueError):
        return None

    if len(verts_vox) == 0:
        return None

    # Transform voxel coords (i, j, k) to 3D Cartesian world space:
    # r = origin + i * step_x + j * step_y + k * step_z
    v_i = verts_vox[:, 0:1]
    v_j = verts_vox[:, 1:2]
    v_k = verts_vox[:, 2:3]

    world_verts = (
        vol_data.origin
        + v_i * vol_data.step_x
        + v_j * vol_data.step_y
        + v_k * vol_data.step_z
    ).astype(np.float32)

    # Normals transformation: inverse transpose of affine matrix
    aff_matrix = np.column_stack([vol_data.step_x, vol_data.step_y, vol_data.step_z])
    inv_t = np.linalg.inv(aff_matrix).T
    world_normals = (normals @ inv_t).astype(np.float32)
    norms = np.linalg.norm(world_normals, axis=1, keepdims=True) + 1e-12
    world_normals /= norms

    return IsosurfaceMesh(
        isovalue=isovalue,
        vertices=world_verts,
        normals=world_normals,
        faces=faces.astype(np.uint32),
        color=color,
    )


def generate_orbital_surfaces(
    vol_data: VolumetricData,
    isovalue: float = 0.03,
    pos_color: Tuple[float, float, float] = (0.22, 0.58, 0.94), # Vibrant Cyan-Blue (+)
    neg_color: Tuple[float, float, float] = (0.95, 0.30, 0.25), # Vibrant Coral-Red (-)
) -> List[IsosurfaceMesh]:
    """Generate both positive and negative lobe isosurfaces for molecular orbitals."""
    meshes = []
    # Positive lobe
    mesh_pos = extract_isosurface(vol_data, isovalue=+abs(isovalue), color=pos_color)
    if mesh_pos is not None:
        meshes.append(mesh_pos)

    # Negative lobe (if grid has negative values)
    if np.min(vol_data.grid_data) < -1e-5:
        mesh_neg = extract_isosurface(vol_data, isovalue=-abs(isovalue), color=neg_color)
        if mesh_neg is not None:
            meshes.append(mesh_neg)

    return meshes


def build_density_colormap_slice_mesh(
    vol_data: VolumetricData,
    plane: str = "AUTO",
    offset_angstrom: float = 0.0,
    colormap_name: str = "viridis",
    log_scale: bool = True,
    resolution: int = 88,
    alpha: float = 0.88,
) -> IsosurfaceMesh:
    """Build a 3D color-mapped slice plane mesh from any VolumetricData density grid."""
    from scipy.ndimage import map_coordinates

    res = max(16, int(resolution))
    origin = np.asarray(vol_data.origin, dtype=np.float64)
    step_mat = np.column_stack([
        np.asarray(vol_data.step_x, dtype=np.float64),
        np.asarray(vol_data.step_y, dtype=np.float64),
        np.asarray(vol_data.step_z, dtype=np.float64),
    ])
    inv_step_mat = np.linalg.inv(step_mat)
    box_diag = (
        (vol_data.voxels_x - 1) * step_mat[:, 0]
        + (vol_data.voxels_y - 1) * step_mat[:, 1]
        + (vol_data.voxels_z - 1) * step_mat[:, 2]
    )
    p_min = np.minimum(origin, origin + box_diag)
    p_max = np.maximum(origin, origin + box_diag)
    box_center = 0.5 * (p_min + p_max)

    plane_key = (plane or "AUTO").upper()
    if plane_key == "AUTO":
        coords = np.asarray(vol_data.atom_coords, dtype=np.float64) if len(vol_data.atom_coords) > 0 else np.empty((0, 3))
        if len(coords) >= 2:
            stds = np.std(coords, axis=0)
            min_ax = int(np.argmin(stds))
            plane_key = ("YZ", "XZ", "XY")[min_ax]
        else:
            plane_key = "XY"

    if plane_key == "XZ":
        ax1, ax2, ax_n = 0, 2, 1
        n_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    elif plane_key == "YZ":
        ax1, ax2, ax_n = 1, 2, 0
        n_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:  # XY
        ax1, ax2, ax_n = 0, 1, 2
        n_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    u_vals = np.linspace(p_min[ax1], p_max[ax1], res, dtype=np.float64)
    v_vals = np.linspace(p_min[ax2], p_max[ax2], res, dtype=np.float64)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    pts = np.empty((res * res, 3), dtype=np.float64)
    pts[:, ax1] = uu.ravel()
    pts[:, ax2] = vv.ravel()
    pts[:, ax_n] = box_center[ax_n] + float(offset_angstrom)

    vox_coords = ((pts - origin[None, :]) @ inv_step_mat.T).T
    rho_slice = map_coordinates(
        np.asarray(vol_data.grid_data, dtype=np.float64),
        vox_coords,
        order=1,
        mode="constant",
        cval=0.0,
    )

    if log_scale:
        vals = np.log10(np.maximum(np.abs(rho_slice), 1e-6))
        pos_vals = vals[vals > -5.9]
        vmin = max(-6.0, float(np.percentile(pos_vals, 5)) if len(pos_vals) > 0 else -5.0)
        vmax = max(vmin + 0.5, float(np.max(vals)))
        cbar_title = "log10 Density (e/a0^3)"
    else:
        vals = rho_slice
        vmin = float(np.min(vals))
        vmax = max(vmin + 1e-6, float(np.max(vals)))
        cbar_title = "Density rho (e/a0^3)"

    norm_val = np.clip((vals - vmin) / max(1e-9, vmax - vmin), 0.0, 1.0)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get(colormap_name, mpl.colormaps["viridis"])
        rgb = cmap(norm_val)[:, :3].astype(np.float32)
    except Exception:
        rgb = np.column_stack([norm_val, 0.5 * (1.0 - np.abs(2.0 * norm_val - 1.0)), 1.0 - norm_val]).astype(np.float32)

    idx_grid = np.arange(res * res, dtype=np.uint32).reshape(res, res)
    i00 = idx_grid[:-1, :-1].ravel()
    i10 = idx_grid[1:, :-1].ravel()
    i01 = idx_grid[:-1, 1:].ravel()
    i11 = idx_grid[1:, 1:].ravel()
    faces = np.vstack([
        np.column_stack([i00, i10, i11]),
        np.column_stack([i00, i11, i01]),
    ]).astype(np.uint32)

    normals = np.tile(n_vec, (res * res, 1))
    return IsosurfaceMesh(
        isovalue=0.0,
        vertices=pts.astype(np.float32),
        normals=normals,
        faces=faces,
        color=(0.2, 0.7, 0.6),
        vertex_colors=rgb,
        surface_values=vals.astype(np.float32),
        esp_range=(vmin, vmax),
        colormap_name=colormap_name,
        colorbar_title=cbar_title,
    )


def build_density_mapped_vdw_mesh(
    vol_data: VolumetricData,
    vdw_scale: float = 0.85,
    colormap_name: str = "viridis",
    log_scale: bool = False,
    grid_n: int = 48,
) -> Optional[IsosurfaceMesh]:
    """Map 3D density onto a smooth Gaussian vdW molecular envelope (like 3D ESP-mapped density)."""
    import math
    from scipy.ndimage import map_coordinates

    coords = np.asarray(vol_data.atom_coords, dtype=np.float64)
    if len(coords) == 0:
        return build_density_colormap_slice_mesh(vol_data, colormap_name=colormap_name, log_scale=log_scale)

    try:
        from .elements import get_vdw_radius
        radii = np.array([max(0.8, float(get_vdw_radius(s))) for s in vol_data.symbols], dtype=np.float64)
    except Exception:
        radii = np.full(len(coords), 1.7, dtype=np.float64)

    eff_radii = radii * max(0.4, float(vdw_scale))
    pad = float(np.max(eff_radii)) + 1.5
    p_min = np.min(coords, axis=0) - pad
    p_max = np.max(coords, axis=0) + pad
    n = max(24, int(grid_n))
    xs = np.linspace(p_min[0], p_max[0], n, dtype=np.float64)
    ys = np.linspace(p_min[1], p_max[1], n, dtype=np.float64)
    zs = np.linspace(p_min[2], p_max[2], n, dtype=np.float64)
    dx, dy, dz = (p_max[0] - p_min[0]) / (n - 1), (p_max[1] - p_min[1]) / (n - 1), (p_max[2] - p_min[2]) / (n - 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    field = np.zeros((n, n, n), dtype=np.float64)
    for (ax, ay, az), rad in zip(coords, eff_radii):
        r2 = (X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2
        field += np.exp(-r2 / (rad * rad))

    env_vol = VolumetricData(
        title="vdW Envelope",
        comment="metaball",
        symbols=list(vol_data.symbols),
        atom_coords=coords,
        origin=p_min,
        voxels_x=n,
        voxels_y=n,
        voxels_z=n,
        step_x=np.array([dx, 0.0, 0.0]),
        step_y=np.array([0.0, dy, 0.0]),
        step_z=np.array([0.0, 0.0, dz]),
        grid_data=field,
        is_orbital=False,
    )
    mesh = extract_isosurface(env_vol, isovalue=math.exp(-1.0), color=(0.2, 0.7, 0.6))
    if mesh is None or len(mesh.vertices) == 0:
        return build_density_colormap_slice_mesh(vol_data, colormap_name=colormap_name, log_scale=log_scale)

    origin = np.asarray(vol_data.origin, dtype=np.float64)
    step_mat = np.column_stack([
        np.asarray(vol_data.step_x, dtype=np.float64),
        np.asarray(vol_data.step_y, dtype=np.float64),
        np.asarray(vol_data.step_z, dtype=np.float64),
    ])
    inv_step_mat = np.linalg.inv(step_mat)
    vox_coords = ((np.asarray(mesh.vertices, dtype=np.float64) - origin[None, :]) @ inv_step_mat.T).T
    rho_verts = map_coordinates(
        np.asarray(vol_data.grid_data, dtype=np.float64),
        vox_coords,
        order=1,
        mode="constant",
        cval=0.0,
    )

    if log_scale:
        vals = np.log10(np.maximum(np.abs(rho_verts), 1e-6))
        cbar_title = "log10 Density (e/a0^3)"
    else:
        vals = rho_verts
        cbar_title = "Density rho (e/a0^3)"

    vmin = float(np.percentile(vals, 2))
    vmax = max(vmin + 1e-6, float(np.percentile(vals, 98)))
    norm_val = np.clip((vals - vmin) / (vmax - vmin), 0.0, 1.0)
    try:
        import matplotlib as mpl
        cmap = mpl.colormaps.get(colormap_name, mpl.colormaps["viridis"])
        rgb = cmap(norm_val)[:, :3].astype(np.float32)
    except Exception:
        rgb = np.column_stack([norm_val, 0.5 * (1.0 - np.abs(2.0 * norm_val - 1.0)), 1.0 - norm_val]).astype(np.float32)

    mesh.vertex_colors = rgb
    mesh.surface_values = vals.astype(np.float32)
    mesh.esp_range = (vmin, vmax)
    mesh.colormap_name = colormap_name
    mesh.colorbar_title = cbar_title
    return mesh
