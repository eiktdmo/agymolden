"""
agymolden.controllers.tweezers_controller
=========================================
Dedicated controller encapsulating Molecular Tweezers operations:
- Live force field (UFF/VSEPR) interactive relaxation during atom dragging
- Tweezers state toggles and cursor handling
- Mouse release reconstruction of canonical Z-matrix geometry
"""

from typing import Any, Set
import numpy as np
from PyQt6 import QtCore
from ..forcefield import optimize_geometry_force_field
from ..zmatrix import cartesian_to_zmatrix
from ..alignment import capture_rigid_frame


class TweezersController:
    """Manages real-time molecular tweezers and force field relaxation during dragging."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def toggle_tweezers(self, checked: bool) -> None:
        """Toggle interactive live force field relaxation during atom drag."""
        self.mw.live_tweezers_enabled = checked
        if hasattr(self.mw, "viewer") and self.mw.viewer:
            self.mw.viewer.tweezers_active = checked
            if checked:
                self.mw.viewer.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            else:
                self.mw.viewer.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        status = "enabled (Left-click & drag any atom to relax live)" if checked else "disabled"
        self.mw.statusBar().showMessage(f"🧲 Molecular Tweezers: {status}", 4000)

    def handle_atom_drag(self, atom_idx: int, new_x: float, new_y: float, new_z: float) -> None:
        """Execute live force-field relaxation while atom is being moved."""
        if not getattr(self.mw, "live_tweezers_enabled", False) or len(self.mw.zatoms) <= 2:
            return

        try:
            symbols = [a.symbol for a in self.mw.zatoms]
            drag_indices: Set[int] = set(
                getattr(self.mw.viewer, "atom_move_drag_indices", [atom_idx]) or [atom_idx]
            )
            sym = symbols[atom_idx] if 0 <= atom_idx < len(symbols) else ""

            if len(drag_indices) < len(symbols):
                # Use intact initial bonds snapshot captured when drag began, or current bonds
                user_bonds = getattr(self.mw.viewer, "_tweezers_initial_bonds", None)
                if not user_bonds:
                    user_bonds = list(getattr(self.mw.viewer, "bonds", []))
                opt_c, e_val = optimize_geometry_force_field(
                    symbols,
                    self.mw.coords,
                    user_bonds=user_bonds,
                    strict_bonds=True,
                    max_iter=15,
                    frozen_indices=drag_indices,
                )
                self.mw.coords = np.array(opt_c, copy=True)
                self.mw.viewer.coords = np.array(opt_c, copy=True)
                # Keep active bond visualization rendered with stretched bond during drag
                if user_bonds:
                    self.mw.viewer.bonds = list(user_bonds)
                else:
                    self.mw.viewer.compute_bonds()
                self.mw.viewer.update()
                self.mw.statusBar().showMessage(
                    f"🧲 Tweezers: Dragging {sym}{atom_idx+1} at ({new_x:.2f}, {new_y:.2f}, {new_z:.2f}) Å | UFF: {e_val:.1f} kcal/mol",
                    1000,
                )
            else:
                self.mw.statusBar().showMessage(
                    f"🧲 Tweezers: Translating selected {len(drag_indices)} atom(s)", 1000
                )
        except Exception:
            pass

    def finish_atom_drag(self, atom_idx: int) -> None:
        """Called when mouse release completes atom dragging. Reconstructs canonical Z-matrix."""
        if hasattr(self.mw, "viewer") and hasattr(self.mw.viewer, "coords") and len(self.mw.viewer.coords) == len(self.mw.coords):
            self.mw.coords = np.array(self.mw.viewer.coords, copy=True)
            if hasattr(self.mw.viewer, "_tweezers_initial_bonds"):
                self.mw.viewer._tweezers_initial_bonds = None
            self.mw.viewer.compute_bonds()
            self.mw.viewer.update()
        symbols = [a.symbol for a in self.mw.zatoms]
        if self.mw.strict_topology:
            new_zatoms = cartesian_to_zmatrix(symbols, self.mw.coords, existing_zatoms=self.mw.zatoms)
        else:
            new_zatoms = cartesian_to_zmatrix(symbols, self.mw.coords)
        self.mw.zatoms = new_zatoms
        self.mw._rigid_frame = capture_rigid_frame(self.mw.zatoms, self.mw.coords)
        if hasattr(self.mw, "table_widget"):
            self.mw.table_widget.load_zatoms(self.mw.zatoms)
        if hasattr(self.mw, "_sync_raw_text"):
            self.mw._sync_raw_text()
        if hasattr(self.mw, "_update_adjust_panel"):
            self.mw._update_adjust_panel()
        sym = symbols[atom_idx] if 0 <= atom_idx < len(symbols) else ""
        self.mw.statusBar().showMessage(
            f"Moved atom {sym}{atom_idx+1}. Updated Z-matrix geometry.", 4000
        )
