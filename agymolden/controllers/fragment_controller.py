"""
agymolden.controllers.fragment_controller
=========================================
Dedicated controller encapsulating molecular fragment operations:
- Categorized fragment library retrieval & custom user fragment persistence
- Fragment stamping onto canvas and atomic anchors
- Chirality and stereocenter inversion
- 1D Dihedral rotor conformer scan invocation
"""

import os
from typing import List, Tuple, Dict, Optional, Any, Set
import numpy as np
from PyQt6 import QtWidgets, QtCore

from ..topology import norm_bond
from ..zmatrix import cartesian_to_zmatrix
from ..smartslides import invert_chiral_center


class FragmentController:
    """Handles fragment stamping, stereochemistry, and conformer dihedral scanning."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def on_fragment_category_changed(self, category: str) -> None:
        """Filter the fragment combo box by the chosen category."""
        from ..fragment_library import get_fragments_by_category
        if not hasattr(self.mw, "combo_fragment"):
            return
        names = get_fragments_by_category(category)
        prev = self.mw.combo_fragment.currentText()
        self.mw.combo_fragment.blockSignals(True)
        self.mw.combo_fragment.clear()
        self.mw.combo_fragment.addItems(names)
        if prev in names:
            self.mw.combo_fragment.setCurrentText(prev)
        self.mw.combo_fragment.blockSignals(False)
        if self.mw.combo_fragment.count() > 0:
            self.mw._on_fragment_combo_changed(self.mw.combo_fragment.currentText())

    def save_custom_fragment(self) -> None:
        """Save the selected subgraph (or entire molecule) to the persistent Custom User Library."""
        from ..fragment_library import save_custom_fragment
        if len(self.mw.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self.mw, "Save Custom Fragment", "Please build or load a molecule first."
            )
            return

        sel = self.mw._get_ordered_selection()
        if not sel:
            sel = list(range(len(self.mw.zatoms)))

        default_name = f"Custom ({'-'.join(self.mw.zatoms[i].symbol for i in sel[:3])})"
        name, ok = QtWidgets.QInputDialog.getText(
            self.mw,
            "Save to Custom Fragment Library",
            f"Enter a name for this fragment ({len(sel)} atoms; Atom #{sel[0]+1} {self.mw.zatoms[sel[0]].symbol} is anchor):",
            QtWidgets.QLineEdit.EchoMode.Normal,
            default_name,
        )
        if not ok or not name.strip():
            return

        clean_name = name.strip()
        symbols = [self.mw.zatoms[i].symbol for i in sel]
        sub_coords = self.mw.coords[sel]
        pos_map = {old: k for k, old in enumerate(sel)}
        sub_bonds = [
            (pos_map[u], pos_map[v])
            for u, v in self.mw._current_explicit_bonds()
            if u in pos_map and v in pos_map
        ]
        sub_bo = {
            norm_bond(pos_map[u], pos_map[v]): o
            for (u, v), o in self.mw._current_bond_orders().items()
            if u in pos_map and v in pos_map
        }

        save_custom_fragment(
            clean_name, symbols, sub_coords, bonds=sub_bonds, bond_orders=sub_bo, connect_idx=0
        )

        # Refresh fragment dropdown
        if hasattr(self.mw, "combo_frag_category"):
            self.on_fragment_category_changed(self.mw.combo_frag_category.currentText())
        if hasattr(self.mw, "combo_fragment"):
            idx_f = self.mw.combo_fragment.findText(clean_name)
            if idx_f < 0 and hasattr(self.mw, "combo_frag_category"):
                self.mw.combo_frag_category.setCurrentText("Custom User Library")
                idx_f = self.mw.combo_fragment.findText(clean_name)
            if idx_f >= 0:
                self.mw.combo_fragment.setCurrentIndex(idx_f)

        self.mw.statusBar().showMessage(
            f"Saved '{clean_name}' ({len(sel)} atoms) to Custom User Library (~/.config/AgyMolden/custom_fragments.json).",
            5000,
        )

    def invert_stereocenter(self) -> None:
        """Invert the chiral stereocenter configuration at selected atom or active selection."""
        sel = self.mw._get_ordered_selection()
        if not sel:
            self.mw._on_invert_chirality()
            return
        self.mw._push_undo_state()
        symbols = [a.symbol for a in self.mw.zatoms]
        bonds = self.mw._current_explicit_bonds()
        old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
        new_coords, desc = invert_chiral_center(symbols, self.mw.coords, bonds, sel)
        new_zatoms = cartesian_to_zmatrix(
            symbols, new_coords, existing_zatoms=self.mw.zatoms, user_bonds=bonds
        )
        self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        if old_bo:
            self.mw.viewer.bond_orders = old_bo
        self.mw.viewer.multi_selected = list(sel)
        self.mw.viewer.selected_atom_idx = sel[0]
        self.mw.viewer.update()
        self.mw.statusBar().showMessage(desc, 4500)

    def open_dihedral_scan_dialog(self) -> None:
        """Open interactive 1D Dihedral Rotor Conformer Scan dialog with embedded PES plotter."""
        from ..dialog_conformer_scan import DihedralScanDialog
        n = len(self.mw.zatoms)
        if n < 4:
            QtWidgets.QMessageBox.information(
                self.mw,
                "1D Dihedral Rotor Scan",
                "At least 4 atoms are required to perform a dihedral torsion scan.",
            )
            return

        sel = self.mw._get_ordered_selection()
        if len(sel) < 4:
            idx = self.mw.viewer.selected_atom_idx
            if idx is not None and idx >= 3:
                a = self.mw.zatoms[idx]
                if a.ref1 is not None and a.ref2 is not None and a.ref3 is not None:
                    sel = [a.ref3, a.ref2, a.ref1, idx]
        if len(sel) < 4:
            QtWidgets.QMessageBox.information(
                self.mw,
                "1D Dihedral Rotor Scan",
                "Please select 4 atoms (Shift+Click in the 3D viewport) defining the dihedral torsion (i - j - k - l), "
                "or select an atom (row >= 4) in the Z-Matrix table.",
            )
            return

        i1, i2, i3, i4 = sel[0], sel[1], sel[2], sel[3]
        symbols = [a.symbol for a in self.mw.zatoms]
        bonds = self.mw._current_explicit_bonds()
        bo = self.mw._current_bond_orders()
        frozen = set(getattr(self.mw, "frozen_atoms", set()))

        self.mw._push_undo_state()
        def on_conformer_selected(new_coords):
            self.mw._apply_smartslide_coords(new_coords, [i1, i2, i3, i4])
            self.mw._update_adjust_panel()

        if hasattr(self.mw, "dock_manager"):
            dock = self.mw.dock_manager.open_dihedral_scan(
                symbols=symbols,
                coords=self.mw.coords,
                bonds=bonds,
                atom_indices=(i1, i2, i3, i4),
                user_bond_orders=bo,
                frozen_indices=frozen,
                on_conformer_selected=on_conformer_selected,
            )
        else:
            dlg = DihedralScanDialog(
                self.mw,
                symbols=symbols,
                coords=self.mw.coords,
                bonds=bonds,
                atom_indices=(i1, i2, i3, i4),
                user_bond_orders=bo,
                frozen_indices=frozen,
            )
            dlg.conformer_selected.connect(on_conformer_selected)
            dlg.exec()
