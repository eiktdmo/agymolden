"""
AgyMolden Project Bundle (.agyproj) format reader and writer.
Preserves molecular structure, internal coordinates, calculation history,
QM inputs, and camera/visualization state in an open, portable JSON/ZIP format.
"""

import json
import logging
import os
import zipfile
from typing import Dict, Any, Optional, List
import numpy as np

log = logging.getLogger(__name__)

from .zmatrix import ZAtom, cartesian_to_zmatrix, zmatrix_to_cartesian


PROJECT_SCHEMA_VERSION = "1.0"


def save_project_bundle(
    filepath: str,
    zatoms: List[ZAtom],
    coords: np.ndarray,
    variables: Optional[Dict[str, float]] = None,
    cross_bonds: Optional[List[List[int]]] = None,
    calculation_history: Optional[List[Dict[str, Any]]] = None,
    viewer_state: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Save complete modeling session into an .agyproj bundle archive.
    """
    manifest = {
        "format": "AgyMolden Project Bundle",
        "schema_version": PROJECT_SCHEMA_VERSION,
        "metadata": metadata or {},
        "molecule": {
            "num_atoms": len(zatoms),
            "zatoms": [
                {
                    "symbol": a.symbol,
                    "ref1": a.ref1,
                    "dist": a.dist,
                    "dist_var": a.dist_var,
                    "ref2": a.ref2,
                    "angle": a.angle,
                    "angle_var": a.angle_var,
                    "ref3": a.ref3,
                    "dihedral": a.dihedral,
                    "dihedral_var": a.dihedral_var,
                }
                for a in zatoms
            ],
            "coords": coords.tolist() if isinstance(coords, np.ndarray) else (coords or []),
            "variables": variables or {},
            "cross_bonds": [list(b) for b in (cross_bonds or [])],
        },
        "calculation_history": calculation_history or [],
        "viewer_state": viewer_state or {},
    }

    tmp_path = filepath + ".tmp"
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("project.json", json.dumps(manifest, indent=2))
        os.replace(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return True


def load_project_bundle(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Load an .agyproj bundle archive and unpack molecular state, calculation history, and view settings.
    """
    if not os.path.isfile(filepath):
        return None

    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "project.json" not in zf.namelist():
                return None
            data = json.loads(zf.read("project.json").decode("utf-8"))

        mol_data = data.get("molecule", {})
        zatoms_raw = mol_data.get("zatoms", [])
        zatoms = []
        for d in zatoms_raw:
            zatoms.append(
                ZAtom(
                    symbol=d.get("symbol", "C"),
                    ref1=d.get("ref1"),
                    dist=float(d.get("dist", 0.0)),
                    dist_var=d.get("dist_var"),
                    ref2=d.get("ref2"),
                    angle=float(d.get("angle", 0.0)),
                    angle_var=d.get("angle_var"),
                    ref3=d.get("ref3"),
                    dihedral=float(d.get("dihedral", 0.0)),
                    dihedral_var=d.get("dihedral_var"),
                )
            )

        coords_list = mol_data.get("coords", [])
        coords = np.array(coords_list, dtype=np.float64) if coords_list else np.empty((0, 3), dtype=np.float64)

        return {
            "zatoms": zatoms,
            "coords": coords,
            "variables": mol_data.get("variables", {}),
            "cross_bonds": [tuple(b) for b in mol_data.get("cross_bonds", [])],
            "calculation_history": data.get("calculation_history", []),
            "viewer_state": data.get("viewer_state", {}),
            "metadata": data.get("metadata", {}),
        }
    except Exception:
        log.warning("Failed to load project bundle %s", filepath, exc_info=True)
        return None
