"""
agymolden.controllers.ring_controller
====================================
Dedicated controller encapsulating molecular ring operations:
- Molecular Cycle Welder & Ring Fusion (bicyclic / spiro / fused rings)
- Ring Aromaticity analysis (HOMA index / planarity / Kekulé bond alternation)
"""

from typing import Any, Optional
import numpy as np
from PyQt6 import QtWidgets


class RingController:
    """Handles molecular ring fusion, annulation, and aromaticity inspection."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def open_cycle_welder(self) -> None:
        """Open Molecular Cycle Welder & Ring Fusion Studio (Ctrl+Shift+W)."""
        from ..dialog_cycle_welder import CycleWelderDialog
        dlg = CycleWelderDialog(self.mw)
        dlg.exec()

    def open_cycle_welder_for_bond(self, u: int, v: int) -> None:
        """Select bond (u, v) and open Cycle Welder Dialog."""
        self.mw.selected_atoms = [u, v]
        if hasattr(self.mw, "viewer") and self.mw.viewer:
            self.mw.viewer.selected_bond = (u, v)
            self.mw.viewer.multi_selected = [u, v]
            self.mw.viewer.selected_atom_idx = u
            self.mw.viewer.update()
        self.open_cycle_welder()

    def open_ring_aromaticity_dialog(self) -> None:
        """Open the Ring Aromaticity (HOMA Index) & Planarity Inspector Dialog (Multiwfn / Krygowski style)."""
        if len(self.mw.zatoms) < 3:
            QtWidgets.QMessageBox.information(
                self.mw, "Ring Aromaticity (HOMA)", "Please load or build a ring-containing molecule first."
            )
            return

        from ..dialog_ring_aromaticity import RingAromaticityDialog
        symbols = [a.symbol for a in self.mw.zatoms]
        coords = np.array(self.mw.coords, dtype=np.float64, copy=True)
        bonds = self.mw._current_explicit_bonds()
        bo = self.mw._current_bond_orders()

        def _highlight_ring(ring_atoms):
            if hasattr(self.mw, "viewer") and self.mw.viewer:
                self.mw.viewer.multi_selected = ring_atoms
                self.mw.viewer.selected_atom_idx = ring_atoms[0] if ring_atoms else None
                self.mw.viewer.update()

        dlg = RingAromaticityDialog(
            parent=self.mw,
            symbols=symbols,
            coords=coords,
            bonds=bonds,
            bond_orders=bo,
            on_ring_highlight=_highlight_ring,
        )
        dlg.exec()
