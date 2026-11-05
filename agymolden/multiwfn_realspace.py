"""
Multiwfn & Real-Space Grid Wavefunction Analysis Engine for AgyMolden.

Provides algorithms for:
1. Real-space scalar field classification (ELF, LOL, ESP, RDG/NCI, Laplacians, Electron Density).
2. Non-Covalent Interactions (NCI) analysis:
   - Reduced density gradient: s(r) = 1 / (2 * (3*pi^2)^(1/3)) * |grad rho| / rho^(4/3)
   - sign(lambda_2) * rho(r) mapping:
       * Blue (sign(lambda_2) * rho < -0.01 a.u.): Strong attractive interaction (H-bonds, halogen bonds)
       * Green (-0.01 <= sign(lambda_2) * rho <= 0.01 a.u.): Van der Waals / dispersion interactions
       * Red (sign(lambda_2) * rho > 0.01 a.u.): Strong steric clash / non-bonding repulsion (ring centers)
3. Multiwfn Cube File Reader & Metadata Extractor:
   - Identifies grid type from comment lines or value distributions
   - Generates specialized isosurface meshes with custom colormaps and boundary limits
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from .volumetric import VolumetricData, IsosurfaceMesh, extract_isosurface


@dataclass
class RealSpaceFieldInfo:
    """Identified metadata for a real-space scalar grid."""
    field_type: str         # "ELF", "LOL", "NCI_RDG", "ESP", "DENSITY", "LAPLACIAN", "UNKNOWN"
    recommended_isovalue: float
    unit: str
    colormap: str           # "bwr_r", "nci_bgr", "viridis", "coolwarm"
    description: str


def classify_real_space_grid(vol: VolumetricData) -> RealSpaceFieldInfo:
    """Classify a 3D volumetric grid based on header comments and value ranges."""
    txt = (vol.title + " " + vol.comment).upper()
    g_min = float(np.min(vol.grid_data))
    g_max = float(np.max(vol.grid_data))

    # 1. Electron Localization Function (ELF)
    if "ELF" in txt or ("LOCALIZATION" in txt and g_min >= 0.0 and g_max <= 1.05):
        return RealSpaceFieldInfo(
            field_type="ELF",
            recommended_isovalue=0.80,
            unit="dimensionless [0, 1]",
            colormap="viridis",
            description="Electron Localization Function (ELF): reveals core, valence, and lone pair electron shells (0.80 = localized pairs)",
        )

    # 2. Localized Orbital Locator (LOL)
    if "LOL" in txt or ("LOCATOR" in txt and g_min >= 0.0 and g_max <= 1.05):
        return RealSpaceFieldInfo(
            field_type="LOL",
            recommended_isovalue=0.55,
            unit="dimensionless [0, 1]",
            colormap="plasma",
            description="Localized Orbital Locator (LOL): localized electron basins based on kinetic energy density",
        )

    # 3. Reduced Density Gradient (RDG) / Non-Covalent Interactions (NCI)
    if "RDG" in txt or "NCI" in txt or "REDUCED GRADIENT" in txt:
        return RealSpaceFieldInfo(
            field_type="NCI_RDG",
            recommended_isovalue=0.50,
            unit="dimensionless a.u.",
            colormap="nci_bgr",
            description="Reduced Density Gradient (RDG): low-density low-gradient regions indicate non-covalent interactions (0.50 a.u.)",
        )

    # 4. Electrostatic Potential (ESP)
    if "ESP" in txt or "ELECTROSTATIC" in txt or "POTENTIAL" in txt:
        return RealSpaceFieldInfo(
            field_type="ESP",
            recommended_isovalue=0.001,
            unit="a.u. (Hartree/e)",
            colormap="bwr_r",
            description="Molecular Electrostatic Potential (ESP): electrophilic (blue) vs nucleophilic (red) reactivity regions",
        )

    # 5. Total Electron Density
    if "DENSITY" in txt or "RHO" in txt or (g_min >= 0.0 and g_max > 1.0):
        return RealSpaceFieldInfo(
            field_type="DENSITY",
            recommended_isovalue=0.001,
            unit="e/Bohr³",
            colormap="coolwarm",
            description="Total Electron Density: molecular van der Waals envelope (0.001 a.u. = standard vdW boundary)",
        )

    return RealSpaceFieldInfo(
        field_type="UNKNOWN",
        recommended_isovalue=0.05,
        unit="a.u.",
        colormap="coolwarm",
        description="General real-space scalar grid",
    )


def compute_nci_color_mapping(
    sign_lambda2_rho: np.ndarray,
    v_min: float = -0.04,
    v_max: float = 0.04,
) -> np.ndarray:
    """
    Map sign(lambda_2)*rho scalar values into RGB colors using the standard Multiwfn/NCIPLOT palette:
    - Blue  (sign(lambda_2)*rho <= -0.015): Strong attractive H-bond / halogen bond
    - Cyan  (-0.015 < val < -0.005): Weak attractive interaction
    - Green (-0.005 <= val <= 0.005): Van der Waals dispersion region
    - Yellow (0.005 < val < 0.015): Weak repulsive interaction
    - Red   (val >= 0.015): Strong steric clash / ring strain
    Returns (N, 3) float array of RGB colors in [0, 1].
    """
    n = len(sign_lambda2_rho)
    colors = np.zeros((n, 3), dtype=np.float32)

    # Multiwfn standard NCI color interpolation
    for i, val in enumerate(sign_lambda2_rho):
        if val <= -0.015:
            # Blue (#2563eb)
            colors[i] = [0.145, 0.388, 0.922]
        elif val < -0.005:
            # Cyan / Sky blue (#06b6d4)
            t = (val - (-0.015)) / 0.010
            colors[i] = (1.0 - t) * np.array([0.145, 0.388, 0.922]) + t * np.array([0.024, 0.714, 0.831])
        elif val <= 0.005:
            # Pure Green (#22c55e)
            t = (val - (-0.005)) / 0.010
            colors[i] = (1.0 - t) * np.array([0.024, 0.714, 0.831]) + t * np.array([0.133, 0.773, 0.369])
        elif val < 0.015:
            # Yellow / Orange (#f59e0b)
            t = (val - 0.005) / 0.010
            colors[i] = (1.0 - t) * np.array([0.133, 0.773, 0.369]) + t * np.array([0.961, 0.620, 0.043])
        else:
            # Red (#dc2626)
            colors[i] = [0.863, 0.149, 0.149]

    return colors


def create_nci_isosurface_mesh(
    rdg_grid: VolumetricData,
    sign_lambda2_rho_grid: VolumetricData,
    rdg_isovalue: float = 0.50,
) -> Optional[IsosurfaceMesh]:
    """
    Extract an RDG isosurface (typically s = 0.50 a.u.) and color each vertex
    by the companion sign(lambda_2)*rho scalar field (Blue = Attractive, Green = vdW, Red = Steric).
    """
    # 1. Extract base geometric mesh at s = rdg_isovalue
    base_mesh = extract_isosurface(rdg_grid, isovalue=rdg_isovalue, color=(0.2, 0.8, 0.2))
    if base_mesh is None or len(base_mesh.vertices) == 0:
        return None

    # 2. Interpolate sign(lambda_2)*rho values at each vertex position
    verts = base_mesh.vertices
    origin = sign_lambda2_rho_grid.origin
    step_x = sign_lambda2_rho_grid.step_x
    step_y = sign_lambda2_rho_grid.step_y
    step_z = sign_lambda2_rho_grid.step_z
    data = sign_lambda2_rho_grid.grid_data
    nx, ny, nz = sign_lambda2_rho_grid.voxels_x, sign_lambda2_rho_grid.voxels_y, sign_lambda2_rho_grid.voxels_z

    dx_norm = float(np.linalg.norm(step_x))
    dy_norm = float(np.linalg.norm(step_y))
    dz_norm = float(np.linalg.norm(step_z))

    sampled_vals = np.zeros(len(verts), dtype=np.float32)
    for i, v in enumerate(verts):
        rel = v - origin
        # Orthogonal grid index mapping
        gx = int(round(rel[0] / dx_norm)) if dx_norm > 0 else 0
        gy = int(round(rel[1] / dy_norm)) if dy_norm > 0 else 0
        gz = int(round(rel[2] / dz_norm)) if dz_norm > 0 else 0
        gx = max(0, min(nx - 1, gx))
        gy = max(0, min(ny - 1, gy))
        gz = max(0, min(nz - 1, gz))
        sampled_vals[i] = data[gx, gy, gz]

    # 3. Color vertex points using NCI color scale
    vertex_colors = compute_nci_color_mapping(sampled_vals)

    base_mesh.vertex_colors = vertex_colors
    base_mesh.surface_values = sampled_vals
    base_mesh.esp_range = (-0.035, 0.035)

    return base_mesh
