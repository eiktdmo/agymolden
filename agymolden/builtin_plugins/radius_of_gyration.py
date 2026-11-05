"""
Radius of Gyration & Inertial Shape Descriptors Plugin for AgyMolden.

Calculates molecular mass-weighted center of mass, radius of gyration (Rg),
principal moments of inertia (I1 <= I2 <= I3), asphericity, and acylindricity.
"""

import numpy as np
from agymolden.plugin_api import PluginBase, PluginMetadata, PluginContext
from agymolden.elements import ELEMENT_DATA


class GyrationShapeDescriptorPlugin(PluginBase):
    metadata = PluginMetadata(
        name="Gyration Radius & Shape Descriptors",
        version="1.0.0",
        author="AgyMolden Core Team",
        description="Calculates mass-weighted Radius of Gyration (Rg), principal moments of inertia, and molecular asphericity.",
        menu_label="📐 Radius of Gyration & Shape Descriptors...",
        category="Analysis",
    )

    def run(self, context: PluginContext) -> dict:
        symbols = context.symbols
        coords = context.coords
        n = len(symbols)

        if n == 0 or len(coords) != n:
            context.show_message("Shape Descriptors", "Please load or build a molecule first.")
            return {}

        masses = np.array([ELEMENT_DATA.get(s, {}).get("mass", 12.011) for s in symbols], dtype=float)
        tot_mass = float(np.sum(masses))

        # Center of mass
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / tot_mass
        rel_coords = coords - com

        # Radius of Gyration (Rg)
        sq_dist = np.sum(rel_coords**2, axis=1)
        rg = float(np.sqrt(np.sum(masses * sq_dist) / tot_mass))

        # Inertia tensor (in amu * Angstrom^2)
        I_mat = np.zeros((3, 3), dtype=float)
        for i in range(n):
            m = masses[i]
            x, y, z = rel_coords[i]
            I_mat[0, 0] += m * (y**2 + z**2)
            I_mat[1, 1] += m * (x**2 + z**2)
            I_mat[2, 2] += m * (x**2 + y**2)
            I_mat[0, 1] -= m * x * y
            I_mat[0, 2] -= m * x * z
            I_mat[1, 2] -= m * y * z
        I_mat[1, 0] = I_mat[0, 1]
        I_mat[2, 0] = I_mat[0, 2]
        I_mat[2, 1] = I_mat[1, 2]

        evals, _ = np.linalg.eigh(I_mat)
        I1, I2, I3 = sorted([float(e) for e in evals])

        # Shape descriptors
        asphericity = float(I3 - 0.5 * (I1 + I2))
        acylindricity = float(I2 - I1)
        rel_anisotropy = float((asphericity**2 + 0.75 * acylindricity**2) / ((I1 + I2 + I3)**2 + 1e-12))

        # Molecular rotor classification
        if abs(I1 - I3) < 1e-3:
            rotor_type = "Spherical Top (I1 = I2 = I3)"
        elif abs(I1 - I2) < 1e-3 and abs(I2 - I3) > 1e-3:
            rotor_type = "Prolate Symmetric Top (I1 = I2 < I3)"
        elif abs(I2 - I3) < 1e-3 and abs(I1 - I2) > 1e-3:
            rotor_type = "Oblate Symmetric Top (I1 < I2 = I3)"
        elif abs(I1) < 1e-3:
            rotor_type = "Linear Molecule (I1 = 0, I2 = I3)"
        else:
            rotor_type = "Asymmetric Top (I1 < I2 < I3)"

        report = (
            f"<b>Molecular Shape & Inertial Analysis</b><br>"
            f"<hr>"
            f"<b>Atoms:</b> {n} &nbsp;|&nbsp; <b>Total Mass:</b> {tot_mass:.3f} amu<br>"
            f"<b>Center of Mass:</b> [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] Å<br><br>"
            f"<b>Radius of Gyration (Rg):</b> <span style='color:#0284c7; font-weight:bold; font-size:13px;'>{rg:.4f} Å</span><br><br>"
            f"<b>Principal Moments of Inertia (amu·Å²):</b><br>"
            f"&nbsp;&nbsp;• I₁ (Ia) = {I1:10.4f}<br>"
            f"&nbsp;&nbsp;• I₂ (Ib) = {I2:10.4f}<br>"
            f"&nbsp;&nbsp;• I₃ (Ic) = {I3:10.4f}<br><br>"
            f"<b>Rotor Classification:</b> <span style='color:#7c3aed; font-weight:bold;'>{rotor_type}</span><br>"
            f"<b>Asphericity (b):</b> {asphericity:.4f} amu·Å²<br>"
            f"<b>Relative Shape Anisotropy (κ²):</b> {rel_anisotropy:.5f}"
        )

        context.show_message("Radius of Gyration & Shape Descriptors", report)
        context.status_message(f"Shape analysis: Rg = {rg:.4f} Å | Top: {rotor_type}", 5000)

        return {
            "rg": rg,
            "com": com,
            "moments": (I1, I2, I3),
            "asphericity": asphericity,
            "acylindricity": acylindricity,
            "rotor_type": rotor_type,
        }
