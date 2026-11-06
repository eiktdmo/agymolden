"""
agymolden.controllers.export_controller
=======================================
Dedicated controller encapsulating molecular serialization and export operations.
Decouples file generation and format translation from GUI main window mixins.
"""

import os
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PyQt6 import QtWidgets

from ..zmatrix import (
    write_xyz,
    write_pdb,
    write_mol_sdf,
    write_mol2,
    write_cif_file,
    format_zmatrix_numeric,
    format_zmatrix_variables,
    ZAtom,
)
from ..dialog_service import dialog_service


class ExportController:
    """Handles all molecular and structural file export actions."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def export_xyz(self) -> Optional[str]:
        """Prompt user and write Cartesian coordinates to .xyz."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(self.mw, "Export XYZ", "No molecule to export.")
            return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Cartesian XYZ", "molecule.xyz", "XYZ Molecular Files (*.xyz);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.mw.zatoms]
                write_xyz(path, symbols, self.mw.coords)
                self.mw.statusBar().showMessage(f"Exported Cartesian XYZ to {os.path.basename(path)}", 4000)
                return path
            except (OSError, IOError) as e:
                dialog_service.error(self.mw, "Error Exporting XYZ", str(e), exc=e)
        return None

    def export_pdb(self) -> Optional[str]:
        """Prompt user and write coordinates & bonds to standard .pdb."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(self.mw, "Export PDB", "No molecule to export.")
            return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Protein Data Bank (PDB)", "molecule.pdb", "PDB Molecular Files (*.pdb);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.mw.zatoms]
                bonds = list(self.mw._current_explicit_bonds())
                write_pdb(path, symbols, self.mw.coords, bonds=bonds)
                self.mw.statusBar().showMessage(f"Exported PDB structure to {os.path.basename(path)}", 4000)
                return path
            except (OSError, IOError) as e:
                dialog_service.error(self.mw, "Error Exporting PDB", str(e), exc=e)
        return None

    def export_sdf(self) -> Optional[str]:
        """Prompt user and write coordinates, bonds & bond orders to MDL .sdf / .mol."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(self.mw, "Export SDF", "No molecule to export.")
            return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Structure-Data File (SDF/MOL)", "molecule.sdf", "SDF Files (*.sdf);;MOL Files (*.mol);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.mw.zatoms]
                bonds = list(self.mw._current_explicit_bonds())
                bo = getattr(self.mw.viewer, "bond_orders", {})
                write_mol_sdf(path, symbols, self.mw.coords, bonds=bonds, bond_orders=bo)
                self.mw.statusBar().showMessage(f"Exported SDF/MOL structure to {os.path.basename(path)}", 4000)
                return path
            except (OSError, IOError) as e:
                dialog_service.error(self.mw, "Error Exporting SDF/MOL", str(e), exc=e)
        return None

    def export_mol2(self) -> Optional[str]:
        """Prompt user and write coordinates, bonds & bond orders to Tripos .mol2."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(self.mw, "Export Mol2", "No molecule to export.")
            return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Tripos Mol2 File", "molecule.mol2", "Mol2 Files (*.mol2);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.mw.zatoms]
                bonds = list(self.mw._current_explicit_bonds())
                bo = getattr(self.mw.viewer, "bond_orders", {})
                write_mol2(path, symbols, self.mw.coords, bonds=bonds, bond_orders=bo)
                self.mw.statusBar().showMessage(f"Exported Tripos Mol2 structure to {os.path.basename(path)}", 4000)
                return path
            except (OSError, IOError) as e:
                dialog_service.error(self.mw, "Error Exporting Mol2", str(e), exc=e)
        return None

    def export_cif(self) -> Optional[str]:
        """Prompt user and write coordinates & unit cell to crystallographic .cif."""
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(self.mw, "Export CIF", "No molecule to export.")
            return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Crystallographic Information File (CIF)", "structure.cif", "CIF Files (*.cif);;All Files (*)"
        )
        if not path:
            return None

        symbols = [a.symbol for a in self.mw.zatoms]
        cell_dict = None
        if hasattr(self.mw, "unit_cell") and self.mw.unit_cell is not None:
            cell_dict = {
                "a": self.mw.unit_cell.a,
                "b": self.mw.unit_cell.b,
                "c": self.mw.unit_cell.c,
                "alpha": self.mw.unit_cell.alpha,
                "beta": self.mw.unit_cell.beta,
                "gamma": self.mw.unit_cell.gamma,
            }

        try:
            write_cif_file(path, symbols, self.mw.coords, cell=cell_dict)
            self.mw.statusBar().showMessage(f"Exported CIF crystal file to {os.path.basename(path)}", 4000)
            return path
        except Exception as e:
            dialog_service.error(self.mw, "Error Exporting CIF", str(e), exc=e)
        return None
