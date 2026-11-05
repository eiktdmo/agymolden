"""
Inertial Frame Principal Axis Aligner Plugin for AgyMolden.

Computes the molecular moment of inertia tensor, shifts center-of-mass to origin,
and rigidly rotates the molecule so that principal axes align exactly with the standard
Cartesian X, Y, and Z axes.
"""

import numpy as np
from agymolden.plugin_api import PluginBase, PluginMetadata, PluginContext
from agymolden.elements import ELEMENT_DATA


class PrincipalAxesAlignerPlugin(PluginBase):
    metadata = PluginMetadata(
        name="Align Molecule to Principal Inertial Axes",
        version="1.0.0",
        author="AgyMolden Core Team",
        description="Translates molecule COM to origin and rotates coordinates so principal inertial axes coincide with X, Y, and Z axes.",
        menu_label="🔄 Align to Principal Inertial Axes",
        category="Geometry",
    )

    def run(self, context: PluginContext) -> np.ndarray:
        symbols = context.symbols
        coords = context.coords
        n = len(symbols)

        if n < 2 or len(coords) != n:
            context.show_message("Principal Axes Alignment", "Please load a molecule with at least 2 atoms.")
            return coords

        masses = np.array([ELEMENT_DATA.get(s, {}).get("mass", 12.011) for s in symbols], dtype=float)
        tot_mass = float(np.sum(masses))

        # 1. Translate COM to origin
        com = np.sum(coords * masses[:, np.newaxis], axis=0) / tot_mass
        shifted = coords - com

        # 2. Compute 3x3 inertia tensor
        I_mat = np.zeros((3, 3), dtype=float)
        for i in range(n):
            m = masses[i]
            x, y, z = shifted[i]
            I_mat[0, 0] += m * (y**2 + z**2)
            I_mat[1, 1] += m * (x**2 + z**2)
            I_mat[2, 2] += m * (x**2 + y**2)
            I_mat[0, 1] -= m * x * y
            I_mat[0, 2] -= m * x * z
            I_mat[1, 2] -= m * y * z
        I_mat[1, 0] = I_mat[0, 1]
        I_mat[2, 0] = I_mat[0, 2]
        I_mat[2, 1] = I_mat[1, 2]

        evals, evecs = np.linalg.eigh(I_mat)
        # Sort eigenvectors by increasing eigenvalue (Ia <= Ib <= Ic)
        order = np.argsort(evals)
        R = evecs[:, order]  # Columns are principal axes

        # Ensure right-handed coordinate system (det(R) == +1)
        if np.linalg.det(R) < 0:
            R[:, 2] = -R[:, 2]

        # Rotate coordinates into principal frame: r_new = shifted @ R
        new_coords = np.dot(shifted, R)

        # Apply coordinates via context (pushes undo state automatically)
        context.set_coords(new_coords)
        context.status_message("Aligned molecule to principal inertial axes (COM at origin, I1 || X, I2 || Y, I3 || Z)", 4500)
        return new_coords
