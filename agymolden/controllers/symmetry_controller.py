"""
agymolden.controllers.symmetry_controller
=========================================
Dedicated controller managing point group symmetry detection, Schoenflies operation tables,
supergroup/subgroup stepping, alignment to standard orientations, coordinate projection,
and GaussView-style Impose Symmetry constraints.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..symmetry import (
    PointGroupResult,
    find_available_symmetries,
    enforce_point_group_symmetry,
    align_molecule_to_symmetry,
    get_symmetry_orbits,
)
from ..zmatrix import cartesian_to_zmatrix
from ..dialog_service import dialog_service


class SymmetryController:
    """Encapsulates all point group symmetry business logic and UI synchronization."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def populate_available_symmetries(
        self,
        avail: List[Tuple[str, PointGroupResult, str]],
        selected_symbol: Optional[str] = None,
    ) -> None:
        """Populate Symmetry tab and 3D viewport symmetry combo boxes with Higher / Detected / Lower groups."""
        self.mw._available_symmetries = list(avail)
        detected_sym = next((sym for sym, _, tag in avail if tag == "detected"), avail[0][0] if avail else "C1")
        target_sym = selected_symbol or detected_sym

        role_labels = {
            "higher": "⬆ Higher (approx. supergroup)",
            "detected": "★ Detected (exact)",
            "lower": "⬇ Lower (subgroup)",
        }

        for combo_attr in ("combo_sym_group", "combo_view_sym_group"):
            combo = getattr(self.mw, combo_attr, None)
            if combo is None:
                continue
            combo.blockSignals(True)
            combo.clear()
            sel_idx = 0
            for idx, (sym, res, tag) in enumerate(avail):
                role_txt = role_labels.get(tag, tag)
                if combo_attr == "combo_view_sym_group":
                    short_tag = "⬆ Higher" if tag == "higher" else ("★ Detected" if tag == "detected" else "⬇ Lower")
                    item_txt = f"{sym} (Ord {res.order}) [{short_tag}]"
                else:
                    item_txt = f"{sym}  (Order {res.order})  —  {role_txt}"
                combo.addItem(item_txt, sym)
                if sym == target_sym:
                    sel_idx = idx
            if 0 <= sel_idx < combo.count():
                combo.setCurrentIndex(sel_idx)
            combo.blockSignals(False)

        self.update_sym_step_buttons()

    def update_sym_step_buttons(self) -> None:
        """Enable or disable the Higher / Lower step buttons based on current index in _available_symmetries."""
        avail = getattr(self.mw, "_available_symmetries", [])
        combo = getattr(self.mw, "combo_sym_group", None)
        if not avail or combo is None or combo.count() == 0:
            if hasattr(self.mw, "btn_sym_higher"):
                self.mw.btn_sym_higher.setEnabled(False)
            if hasattr(self.mw, "btn_sym_lower"):
                self.mw.btn_sym_lower.setEnabled(False)
            return
        idx = combo.currentIndex()
        if hasattr(self.mw, "btn_sym_higher"):
            self.mw.btn_sym_higher.setEnabled(idx > 0)
        if hasattr(self.mw, "btn_sym_lower"):
            self.mw.btn_sym_lower.setEnabled(0 <= idx < len(avail) - 1)

    def select_symmetry_group(self, group_symbol: str, enforce_if_higher: bool = True) -> Optional[PointGroupResult]:
        """Select a higher, detected, or lower point group symmetry group_symbol."""
        if len(self.mw.zatoms) == 0:
            return None
        tol = self.mw.spin_sym_tol.value() if hasattr(self.mw, "spin_sym_tol") else 0.15
        symbols = [a.symbol for a in self.mw.zatoms]
        avail = getattr(self.mw, "_available_symmetries", None)
        if not avail:
            avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
            self.mw._available_symmetries = avail

        chosen_entry = next((item for item in avail if item[0] == group_symbol), None)
        if chosen_entry is None:
            avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
            self.mw._available_symmetries = avail
            chosen_entry = next((item for item in avail if item[0] == group_symbol), None)
        if chosen_entry is None:
            return getattr(self.mw, "_current_sym_res", None)

        sym_name, chosen_res, role_tag = chosen_entry

        if role_tag == "higher" and enforce_if_higher and chosen_res.order > 1:
            self.mw._push_undo_state()
            user_bonds = self.mw._current_explicit_bonds()
            old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
            sym_coords, enforced_res = enforce_point_group_symmetry(
                symbols, self.mw.coords, target_group=sym_name, tolerance=tol, reference_result=chosen_res
            )
            new_zatoms = cartesian_to_zmatrix(
                symbols, sym_coords, existing_zatoms=self.mw.zatoms, user_bonds=user_bonds
            )
            prev_imposed = getattr(self.mw, "symmetry_imposed", False)
            self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
            if old_bo:
                self.mw.viewer.bond_orders = old_bo
                self.mw.viewer.update()
            avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
            chosen_entry = next((item for item in avail if item[0] == sym_name), (sym_name, enforced_res, "detected"))
            sym_name, chosen_res, role_tag = chosen_entry
            if prev_imposed:
                self.mw.symmetry_imposed = True

        self.mw._current_sym_res = chosen_res
        self.populate_available_symmetries(avail, selected_symbol=sym_name)
        self.refresh_symmetry_ui(chosen_res, role_tag=role_tag)

        if getattr(self.mw, "symmetry_imposed", False):
            self.set_impose_symmetry(True, chosen_res)
        else:
            self.set_impose_symmetry(True, chosen_res)
            orbits = get_symmetry_orbits(symbols, self.mw.coords, chosen_res)
            self.mw.statusBar().showMessage(
                f"Selected & Imposed {sym_name} symmetry (Order {chosen_res.order}, {role_tag}, {len(orbits)} atom orbit(s)). Edits replicate across equivalent atoms.",
                5000,
            )
        return chosen_res

    def refresh_symmetry_ui(self, res: PointGroupResult, role_tag: Optional[str] = None) -> None:
        """Populate symmetry labels and operations table from PointGroupResult without switching tabs."""
        self.mw.btn_sym_align.setEnabled(True)
        self.mw.btn_sym_enforce.setEnabled(True)
        self.mw.btn_sym_matrices.setEnabled(True)
        self.mw.btn_calc_reps.setEnabled(True)

        avail = getattr(self.mw, "_available_symmetries", [])
        if role_tag is None and avail:
            role_tag = next((tag for sym, _, tag in avail if sym == res.symbol), "detected")

        role_badge = ""
        if role_tag == "higher":
            role_badge = " [Higher / Approx. Supergroup — click 'Enforce Symmetry' to snap]"
        elif role_tag == "lower":
            detected_sym = next((sym for sym, _, tag in avail if tag == "detected"), "")
            role_badge = f" [Lower Subgroup of {detected_sym}]" if detected_sym else " [Lower Subgroup]"
        elif role_tag == "detected":
            role_badge = " [Detected]"

        self.mw.lbl_sym_group.setText(f"Point Group: {res.symbol}{role_badge}")
        symbols = [a.symbol for a in self.mw.zatoms]
        orbits = get_symmetry_orbits(symbols, self.mw.coords, res) if len(symbols) == len(self.mw.coords) else []
        orbit_summary = ", ".join(
            f"{{{', '.join(f'{symbols[i]}{i+1}' for i in orb)}}}" for orb in orbits
        ) if len(orbits) <= 12 else f"{len(orbits)} equivalence classes"

        details_lines = [
            f"Schoenflies Group: <b>{res.symbol}</b> (Group Order: {res.order})",
            f"Inversion Center (i): {'Present' if res.has_inversion else 'Absent'}",
            f"Horizontal Mirror Plane (σh): {'Present' if res.horizontal_planes_count > 0 else 'None'}",
            f"Vertical/Dihedral Planes (σv/σd): {res.vertical_planes_count}",
        ]
        if orbit_summary:
            details_lines.append(f"Symmetry Orbits ({len(orbits)}): <span style='color:#38bdf8;'>{orbit_summary}</span>")
        if res.principal_axis is not None:
            details_lines.append(
                f"Principal Axis: [{res.principal_axis[0]:.3f}, {res.principal_axis[1]:.3f}, {res.principal_axis[2]:.3f}]"
            )
        self.mw.lbl_sym_details.setText("<br>".join(details_lines))

        prev_row = self.mw.table_symmetry_ops.currentRow()
        self.mw.table_symmetry_ops.setRowCount(res.order)
        for idx, op in enumerate(res.operations):
            it_num = QtWidgets.QTableWidgetItem(str(idx + 1))
            it_num.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            it_num.setForeground(QtGui.QBrush(QtGui.QColor("#a78bfa")))

            it_op = QtWidgets.QTableWidgetItem(op.label)
            it_op.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            it_op.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))

            type_desc = op.op_type
            if op.axis is not None:
                ax = op.axis
                type_desc += f" along [{ax[0]:.2f}, {ax[1]:.2f}, {ax[2]:.2f}]"
            it_desc = QtWidgets.QTableWidgetItem(type_desc)

            self.mw.table_symmetry_ops.setItem(idx, 0, it_num)
            self.mw.table_symmetry_ops.setItem(idx, 1, it_op)
            self.mw.table_symmetry_ops.setItem(idx, 2, it_desc)

        if res.order > 0:
            self.mw.table_symmetry_ops.selectRow(prev_row if 0 <= prev_row < res.order else 0)
        self.update_sym_step_buttons()
        if hasattr(self.mw, "_update_orbital_symmetries"):
            self.mw._update_orbital_symmetries(res)

    def detect_symmetry(
        self, checked_or_switch: bool = True, *, switch_tab: bool = True, preferred_symbol: Optional[str] = None
    ) -> None:
        """Analyze molecular 3D coordinates, discover higher/detected/lower symmetries, and update UI."""
        if len(self.mw.zatoms) == 0:
            if switch_tab:
                QtWidgets.QMessageBox.information(
                    self.mw, "Point Group Symmetry", "Please add or load a molecule to analyze symmetry."
                )
            return

        tol = self.mw.spin_sym_tol.value()
        if switch_tab:
            self.mw.statusBar().showMessage(f"Detecting point group symmetry (tol={tol:.2f} Å)...", 3000)
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))

        try:
            symbols = [a.symbol for a in self.mw.zatoms]
            avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
            detected_entry = next((item for item in avail if item[2] == "detected"), avail[0])
            chosen_entry = detected_entry
            if preferred_symbol:
                chosen_entry = next((item for item in avail if item[0] == preferred_symbol), detected_entry)

            res = chosen_entry[1]
            role_tag = chosen_entry[2]
            self.mw._current_sym_res = res
            self.populate_available_symmetries(avail, selected_symbol=chosen_entry[0])
            self.refresh_symmetry_ui(res, role_tag=role_tag)

            if switch_tab:
                idx = self.mw.tabs.indexOf(getattr(self.mw, "tab_symmetry", None))
                if idx >= 0:
                    self.mw.tabs.setCurrentIndex(idx)
                else:
                    self.mw.tabs.setCurrentIndex(4)
                n_higher = sum(1 for _, _, t in avail if t == "higher")
                n_lower = sum(1 for _, _, t in avail if t == "lower")
                self.mw.statusBar().showMessage(
                    f"Detected: {detected_entry[0]} (Order {detected_entry[1].order}) | "
                    f"Available: {n_higher} higher, {n_lower} lower subgroup(s)",
                    5500,
                )
        except Exception as e:
            if switch_tab:
                dialog_service.error(self.mw, "Symmetry Detection Error", f"Failed to detect symmetry:\n{e}", exc=e)
        finally:
            if switch_tab:
                QtWidgets.QApplication.restoreOverrideCursor()

    def align_symmetry(self) -> None:
        """Align molecule according to highest C_n (along Z) and principal plane (in XZ or XY)."""
        if len(self.mw.zatoms) == 0:
            return

        res = getattr(self.mw, "_current_sym_res", None)
        symbols = [a.symbol for a in self.mw.zatoms]
        self.mw._push_undo_state()
        user_bonds = self.mw._current_explicit_bonds()
        old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
        aligned_coords, _ = align_molecule_to_symmetry(symbols, self.mw.coords, res)

        new_zatoms = cartesian_to_zmatrix(
            symbols, aligned_coords, existing_zatoms=self.mw.zatoms, user_bonds=user_bonds
        )
        self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=aligned_coords)
        if old_bo:
            self.mw.viewer.bond_orders = old_bo
        self.mw.viewer.align_camera_standard()

        self.detect_symmetry()
        self.mw.statusBar().showMessage(
            "Aligned geometry to standard symmetry orientation (highest C_n along Z, displayed in screen plane).", 5000
        )

    def enforce_symmetry(self) -> None:
        """Project coordinates into exact point group subspace if tolerance is not fulfilled."""
        if len(self.mw.zatoms) == 0:
            return

        res = getattr(self.mw, "_current_sym_res", None)
        target = res.symbol if res else None
        tol = self.mw.spin_sym_tol.value()

        self.mw._push_undo_state()
        symbols = [a.symbol for a in self.mw.zatoms]
        user_bonds = self.mw._current_explicit_bonds()
        old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
        sym_coords, new_res = enforce_point_group_symmetry(
            symbols, self.mw.coords, target_group=target, tolerance=tol, reference_result=res
        )

        new_zatoms = cartesian_to_zmatrix(
            symbols, sym_coords, existing_zatoms=self.mw.zatoms, user_bonds=user_bonds
        )
        self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
        if old_bo:
            self.mw.viewer.bond_orders = old_bo
            self.mw.viewer.update()

        self.detect_symmetry(switch_tab=False, preferred_symbol=target or new_res.symbol)
        self.set_impose_symmetry(True, self.mw._current_sym_res)
        self.mw.statusBar().showMessage(
            f"Enforced & Imposed {new_res.symbol} symmetry (Order {new_res.order}). Subsequent edits will preserve {new_res.symbol} symmetry!",
            5500,
        )

    def set_impose_symmetry(self, enabled: bool, sym_res: Optional[PointGroupResult] = None) -> None:
        """Enable or disable symmetry-constrained editing (Impose Symmetry)."""
        if enabled and len(self.mw.zatoms) > 0:
            tol = self.mw.spin_sym_tol.value() if hasattr(self.mw, "spin_sym_tol") else 0.15
            symbols = [a.symbol for a in self.mw.zatoms]
            avail = getattr(self.mw, "_available_symmetries", None)
            if not avail:
                avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
                self.mw._available_symmetries = avail
            if sym_res is None:
                sym_res = getattr(self.mw, "_current_sym_res", None)
                if sym_res is None:
                    detected_entry = next((item for item in avail if item[2] == "detected"), avail[0])
                    sym_res = detected_entry[1]

            matching_entry = next((item for item in avail if item[0] == sym_res.symbol), None)
            if matching_entry is not None and matching_entry[2] == "higher" and sym_res.order > 1:
                user_bonds = self.mw._current_explicit_bonds()
                old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
                sym_coords, sym_res = enforce_point_group_symmetry(
                    symbols, self.mw.coords, target_group=sym_res.symbol, tolerance=tol, reference_result=sym_res
                )
                new_zatoms = cartesian_to_zmatrix(
                    symbols, sym_coords, existing_zatoms=self.mw.zatoms, user_bonds=user_bonds
                )
                self.mw.symmetry_imposed = False
                self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
                if old_bo:
                    self.mw.viewer.bond_orders = old_bo
                    self.mw.viewer.update()
                avail = find_available_symmetries(symbols, self.mw.coords, base_tolerance=tol)
                self.mw._available_symmetries = avail

            self.mw._current_sym_res = sym_res
            self.mw.imposed_sym_res = sym_res
            self.mw.symmetry_imposed = True
            self.populate_available_symmetries(avail, selected_symbol=sym_res.symbol)
            self.refresh_symmetry_ui(sym_res)
        else:
            self.mw.symmetry_imposed = False
            self.mw.imposed_sym_res = None

        grp_label = self.mw.imposed_sym_res.symbol if (self.mw.symmetry_imposed and self.mw.imposed_sym_res) else ""
        if hasattr(self.mw, "chk_impose_symmetry"):
            self.mw.chk_impose_symmetry.blockSignals(True)
            self.mw.chk_impose_symmetry.setChecked(self.mw.symmetry_imposed)
            self.mw.chk_impose_symmetry.setText(
                f"🔒 Impose Symmetry ({grp_label})" if grp_label else "🔒 Impose Symmetry"
            )
            self.mw.chk_impose_symmetry.blockSignals(False)

        if hasattr(self.mw, "btn_impose_symmetry"):
            self.mw.btn_impose_symmetry.blockSignals(True)
            self.mw.btn_impose_symmetry.setChecked(self.mw.symmetry_imposed)
            if self.mw.symmetry_imposed and grp_label:
                self.mw.btn_impose_symmetry.setText(f"🔒 Impose Symmetry: {grp_label}")
                self.mw.btn_impose_symmetry.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 5px;")
            else:
                self.mw.btn_impose_symmetry.setText("🔒 Impose Symmetry: OFF")
                self.mw.btn_impose_symmetry.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 5px;")
            self.mw.btn_impose_symmetry.blockSignals(False)

        if self.mw.symmetry_imposed and self.mw.imposed_sym_res is not None:
            self.mw.statusBar().showMessage(
                f"Imposed {self.mw.imposed_sym_res.symbol} symmetry (Order {self.mw.imposed_sym_res.order}): edits will replicate across symmetry-equivalent atoms.",
                5000,
            )
        elif not enabled:
            self.mw.statusBar().showMessage("Symmetry constraint (Impose Symmetry) turned OFF.", 3500)
