"""Mixin modules extracted from MainWindow for clean separation of concerns."""

import os
import math
import copy
import logging
from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .workers import run_in_background
from .elements import ELEMENT_DATA, normalize_symbol, estimate_bond_length
from .zmatrix import (
    ZAtom,
    zmatrix_to_cartesian,
    cartesian_to_zmatrix,
    parse_zmatrix_text,
    format_zmatrix_text,
    parse_xyz,
    write_xyz,
    calculate_bond_distance,
    calculate_angle,
    calculate_dihedral,
    format_zmatrix_numeric,
    format_zmatrix_variables,
    apply_variables,
    sync_variables_from_atoms,
    assign_symmetric_variables,
    clear_variables,
    _split_var,
)
from .fragments import FRAGMENTS, SAMPLE_MOLECULES
from .viewer import MoleculeViewer
from .table_view import ZMatrixTableWidget
from .periodic_table import PeriodicTableDialog
from .topology import (
    norm_bond, build_adj, remove_atoms, fix_valence_explicit, bfs_reorder,
    rotation_between, fragment_outward_axis, form_single_bond, BondFormationError,
)
from .forcefield import (
    DEFAULT_VALENCES,
    add_hydrogens_to_valence,
    adjust_hydrogens_to_valence,
    optimize_geometry_force_field,
    detect_connectivity,
    detect_bond_orders,
    get_cov_radius,
    _calculate_hydrogen_directions,
)
from .vibrations import VibrationalAnalysis, compute_force_field_vibrations
from .symmetry import (
    find_symmetry_operations,
    find_available_symmetries,
    PointGroupResult,
    align_molecule_to_symmetry,
    enforce_point_group_symmetry,
    calculate_symmetry_representations,
    classify_vibrational_mode_irrep,
    get_symmetry_orbits,
    optimize_with_imposed_symmetry,
)
from .dialog_service import dialog_service

log = logging.getLogger(__name__)


class CalculationsMixin:
    def _get_selected_opt_method(self) -> str:
        if hasattr(self, "combo_opt_method") and self.combo_opt_method.currentIndex() >= 0:
            data = self.combo_opt_method.currentData()
            if data in ("auto_hybrid", "uff_mmff", "semiempirical", "mlip"):
                return data
        return "uff_mmff"

    def _on_opt_method_changed(self, idx: int):
        method = self._get_selected_opt_method()
        if method == "semiempirical":
            label = "Semiempirical QM (GFN0-xTB)"
        elif method == "mlip":
            label = "MLIP (ANI-2x / MACE / Hybrid QM)"
        elif method == "auto_hybrid":
            label = "Auto Hybrid (MMFF / UFF)"
        else:
            label = "Analytical UFF/MMFF94s"
        if hasattr(self, "btn_calc_vibrations"):
            self.btn_calc_vibrations.setText(f"⚡ Calculate Frequencies ({label})")
        self.statusBar().showMessage(f"Selected calculation engine: {label}", 3500)

    def _on_optimize_geometry(self):
        """Optimize molecular geometry using selected force field (UFF/MMFF94s), Semiempirical QM, or MLIP."""
        if len(self.zatoms) < 2:
            QtWidgets.QMessageBox.information(
                self, "Optimize Geometry", "At least 2 atoms are required for geometry optimization."
            )
            return

        method = self._get_selected_opt_method()
        if method == "semiempirical":
            method_label = "Semiempirical QM (GFN0-xTB)"
        elif method == "mlip":
            method_label = "MLIP (ANI-2x / MACE / Hybrid QM)"
        elif method == "auto_hybrid":
            method_label = "Auto Hybrid (MMFF / UFF)"
        else:
            method_label = "Analytical UFF/MMFF94s"
        self._push_undo_state()
        self.statusBar().showMessage(f"Optimizing geometry using {method_label}...", 3000)

        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        bo = dict(getattr(self.viewer, 'bond_orders', {}))
        start_coords = np.array(self.coords, copy=True)
        frozen = {i for i in getattr(self, "frozen_atoms", set()) if 0 <= i < len(symbols)}

        imposed = self._get_active_imposed_symmetry() if not frozen else None

        def run_opt(c0, iters):
            if method == "mlip":
                from .mlip_engine import optimize_geometry_mlip
                return optimize_geometry_mlip(
                    symbols, c0, user_bonds=user_bonds, max_iter=iters,
                    user_bond_orders=bo, frozen_indices=frozen
                )
            if method == "semiempirical":
                from .semiempirical import optimize_geometry_semiempirical
                return optimize_geometry_semiempirical(
                    symbols, c0, user_bonds=user_bonds, max_iter=iters,
                    user_bond_orders=bo, frozen_indices=frozen
                )
            opt_coords, final_e = optimize_geometry_force_field(
                symbols, c0, user_bonds=user_bonds, max_iter=iters,
                user_bond_orders=bo, method=method, frozen_indices=frozen
            )
            return opt_coords, final_e, {}

        def compute():
            if imposed is not None:
                sym_coords, extra, msg = optimize_with_imposed_symmetry(
                    lambda c: run_opt(c, 60), symbols, start_coords, imposed, cycles=6
                )
                if sym_coords is not None:
                    return sym_coords, extra[0], extra[1], msg
                return (*run_opt(start_coords, 250), f"Symmetry {imposed.symbol} not achievable: {msg} Ran unconstrained optimization.")
            return (*run_opt(start_coords, 250), "")

        def done(result):
            try:
                self._apply_optimization(symbols, user_bonds, method, method_label, result)
            except Exception as e:
                dialog_service.error(self, "Optimization Error", f"Geometry optimization failed:\n{e}", exc=e)

        def failed(exc):
            dialog_service.error(self, "Optimization Error", f"Geometry optimization failed:\n{exc}", exc=exc)

        run_in_background(self, "Optimize Geometry", f"Optimizing geometry ({method_label})...",
                          compute, done, failed)

    def _apply_optimization(self, symbols, user_bonds, method, method_label, result):
        opt_coords, final_e, qm_props, sym_msg = result
        sym_kept = sym_msg.startswith("Optimized with")
        qm_summary = ""
        if method in ("semiempirical", "mlip"):
            gap = qm_props.get("gap_ev", 0.0)
            homo = qm_props.get("homo_ev", 0.0)
            lumo = qm_props.get("lumo_ev", 0.0)
            backend = qm_props.get("mlip_backend", "")
            b_tag = f" [{backend}]" if backend else ""
            qm_summary = f"{b_tag} | HOMO: {homo:.2f} eV, LUMO: {lumo:.2f} eV, Gap: {gap:.2f} eV"
            self._semiempirical_qm_props = qm_props

        # Reconstruct Z-matrix from optimized Cartesian coordinates preserving references
        new_zatoms = cartesian_to_zmatrix(
            symbols, opt_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
        )
        frozen = {i for i in getattr(self, "frozen_atoms", set()) if 0 <= i < len(symbols)}
        if frozen or sym_kept:
            # Keep exact (symmetrized) coordinates and frame so the imposed point group stays exact
            self.set_molecule_from_zatoms(new_zatoms, exact_coords=opt_coords)
            self.viewer.fit_camera()
        else:
            self.set_molecule_from_zatoms(new_zatoms)
            # Align optimized geometry to standard orientation (highest C_n along Z, molecular plane in XZ)
            # and align camera to present the molecule directly in the screen plane
            try:
                sym_res = find_symmetry_operations(symbols, opt_coords, tolerance=0.20)
                aligned_coords, _ = align_molecule_to_symmetry(symbols, opt_coords, sym_res)
                aligned_zatoms = cartesian_to_zmatrix(symbols, aligned_coords, user_bonds=user_bonds)
                self.set_molecule_from_zatoms(aligned_zatoms, exact_coords=aligned_coords)
                self.viewer.align_camera_standard()
            except Exception:
                log.warning("Symmetry alignment after optimization failed; keeping unaligned geometry", exc_info=True)
                self.viewer.align_camera_standard()

        self.statusBar().showMessage(
            f"{method_label} converged! E = {final_e:.3f} kcal/mol{qm_summary}"
            + (f" | {sym_msg}" if sym_msg else ""), 8000
        )
        if sym_msg and not sym_kept:
            QtWidgets.QMessageBox.information(self, "Impose Symmetry", sym_msg)

    def _on_conformer_search(self):
        """
        Avogadro / Gabedit-style Conformational Search & Rotamer Generator:
        Detects rotatable bonds, generates and minimizes candidate conformers using the active
        force field / GFN0-xTB engine, adopts the lowest-energy conformer, and populates the
        Trajectory Player with the ranked conformer ensemble.
        """
        if len(self.zatoms) < 3:
            QtWidgets.QMessageBox.information(
                self, "Conformational Search", "At least 3 atoms are required for a conformational search."
            )
            return

        from .smartslides import find_rotatable_bonds, generate_conformers

        method = self._get_selected_opt_method()
        method_label = "Semiempirical QM (GFN0-xTB)" if method == "semiempirical" else "Analytical UFF/MMFF94s"
        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        bo = dict(getattr(self.viewer, "bond_orders", {}))
        start_coords = np.array(self.coords, copy=True)
        frozen = {i for i in getattr(self, "frozen_atoms", set()) if 0 <= i < len(symbols)}
        rotors = find_rotatable_bonds(symbols, user_bonds, bo)

        self._push_undo_state()
        self.statusBar().showMessage(
            f"Searching conformers ({len(rotors)} rotatable bond(s), {method_label})...", 4000
        )

        def compute():
            return generate_conformers(
                symbols, start_coords, user_bonds, bond_orders=bo,
                n_samples=12, method=method, frozen_indices=frozen
            )

        def done(result):
            conf_coords, conf_energies_ev = result
            if not conf_coords:
                return
            best_coords = conf_coords[0]
            new_zatoms = cartesian_to_zmatrix(
                symbols, best_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
            )
            self.set_molecule_from_zatoms(new_zatoms, exact_coords=best_coords)
            if bo:
                self.viewer.bond_orders = bo
                self.viewer.update()

            self.trajectory_coords = conf_coords
            self.trajectory_energies = conf_energies_ev
            self.trajectory_symbols = symbols
            n_conf = len(conf_coords)
            if hasattr(self, "grp_trajectory"):
                if n_conf > 1:
                    self.grp_trajectory.setTitle(f"Conformer Ensemble ({n_conf} Ranked Conformers — Step 1 is Lowest Energy)")
                    self.grp_trajectory.setVisible(True)
                    self.slider_trajectory.blockSignals(True)
                    self.slider_trajectory.setRange(0, n_conf - 1)
                    self.slider_trajectory.setValue(0)
                    self.slider_trajectory.blockSignals(False)
                    self.lbl_trajectory_step.setText(f"Conf 1/{n_conf}")
                    self.lbl_trajectory_energy.setText("ΔE = 0.00 kcal/mol (Lowest)")
                else:
                    self.grp_trajectory.setVisible(False)

            e_min_kcal = conf_energies_ev[0] * 23.060548
            span_kcal = (conf_energies_ev[-1] - conf_energies_ev[0]) * 23.060548 if n_conf > 1 else 0.0
            self.statusBar().showMessage(
                f"Conformer search complete ({len(rotors)} rotor(s)): found {n_conf} unique conformer(s) "
                f"(E_min = {e_min_kcal:.2f} kcal/mol, ΔE span = {span_kcal:.2f} kcal/mol). Adopted lowest-energy conformer!",
                7000,
            )

        def failed(exc):
            dialog_service.error(self, "Conformer Search Error", f"Conformational search failed:\n{exc}", exc=exc)

        run_in_background(
            self,
            "Conformational Search",
            f"Sampling & minimizing rotamers ({len(rotors)} rotatable bond(s), {method_label})...",
            compute,
            done,
            failed,
        )

    def _on_calculate_vibrations(self):
        """Calculate normal vibrational frequencies and displacements using selected UFF/MMFF94s or Semiempirical QM engine."""
        from .controllers import VibrationController
        VibrationController(self).calculate_vibrations()

    def _apply_vibrations(self, result):
        from .controllers import VibrationController
        VibrationController(self).apply_vibrations(result)

    def _on_open_isotopologue_dialog(self):
        """Open the Isotopologue Vibrational Shift & Kinetic Isotope Effect (KIE) Dialog."""
        if not getattr(self, "vibrational_analysis", None) or not self.vibrational_analysis.frequencies:
            QtWidgets.QMessageBox.information(
                self,
                "Isotopologue & KIE Calculator",
                "Please calculate or load vibrational frequencies first.",
            )
            return

        from .dialog_isotopologues import IsotopologueDialog
        dlg = IsotopologueDialog(self, vibrational_analysis=self.vibrational_analysis)
        dlg.exec()

    def _on_spectrum_scale_changed(self, scale_fac: float):
        """Update the Scaled Frequency column in table_vibrations when NIST scale factor changes."""
        from .controllers import VibrationController
        VibrationController(self).on_spectrum_scale_changed(scale_fac)

    def _on_spectrum_mode_clicked(self, mode_idx: int):
        """Called when user clicks on an IR/Raman spectrum peak in the matplotlib plot."""
        from .controllers import VibrationController
        VibrationController(self).on_spectrum_mode_clicked(mode_idx)

    def _on_nmr_atom_clicked(self, atom_idx: int):
        """Highlight the clicked NMR nucleus in the 3D viewport and Z-Matrix table."""
        if 0 <= atom_idx < len(self.zatoms):
            self.viewer.selected_atom_idx = atom_idx
            self.viewer.multi_selected = [atom_idx]
            self.viewer.update()
            sym = self.zatoms[atom_idx].symbol
            self.statusBar().showMessage(f"🧲 Selected NMR nucleus: {sym}{atom_idx + 1}", 3500)

    def _on_dos_orbital_clicked(self, mo_idx: int):
        """Select and render the clicked Molecular Orbital from the Density of States (DOS) plot."""
        if hasattr(self, "table_orbitals") and 0 <= mo_idx < self.table_orbitals.rowCount():
            self.table_orbitals.selectRow(mo_idx)
            self._render_mo_orbital(mo_idx)

    def _on_vibration_mode_selected(self):
        """Called when a mode row is selected in the vibrations table."""
        from .controllers import VibrationController
        VibrationController(self).on_vibration_mode_selected()

    def _on_toggle_vibration_play(self):
        """Toggle simulation playback of selected vibrational mode."""
        from .controllers import VibrationController
        VibrationController(self).toggle_vibration_play()

    def _on_stop_vibration(self):
        """Stop vibration and restore equilibrium structure."""
        from .controllers import VibrationController
        VibrationController(self).stop_vibration()

    def _on_vib_amplitude_changed(self, val: int):
        amp = val / 100.0
        self.lbl_vib_amp.setText(f"{amp:.2f}")
        self.viewer.set_vibration_amplitude(amp)

    def _on_vib_vec_scale_changed(self, val: int):
        scale = val / 100.0
        if hasattr(self, "lbl_vib_vec_scale"):
            self.lbl_vib_vec_scale.setText(f"{scale:.1f}×")
        if hasattr(self.viewer, "set_vibration_vector_scale"):
            self.viewer.set_vibration_vector_scale(scale)

    def _on_filter_vibrations(self, idx: int = 0):
        """Filter table_vibrations rows by frequency category (genuine, imaginary, fingerprint, etc.)."""
        if not hasattr(self, "table_vibrations") or not getattr(self, "vibrational_analysis", None):
            return
        crit = self.combo_vib_filter.currentData() if hasattr(self, "combo_vib_filter") else "all"
        analysis = self.vibrational_analysis
        for row, freq in enumerate(analysis.frequencies):
            show = True
            if crit == "genuine":
                show = (freq > 15.0)
            elif crit == "imag":
                show = (freq < -5.0)
            elif crit == "fingerprint":
                show = (500.0 <= freq <= 1500.0)
            elif crit == "functional":
                show = (1500.0 <= freq <= 4000.0)
            self.table_vibrations.setRowHidden(row, not show)

    def _on_toggle_vib_vectors(self, checked: bool):
        self.viewer.set_vibration_show_vectors(checked)

    def _on_vib_scrub_changed(self, val: int):
        """Interactively deform the molecule along the selected normal mode coordinate."""
        from .controllers import VibrationController
        VibrationController(self).scrub_vibration(val)

    def _on_adopt_vibrational_displacement(self, sign: float = 1.0):
        """
        GaussView-style 'Manual Displacement / Follow Imaginary Mode':
        Displaces the equilibrium molecular geometry along the currently selected normal mode
        by `sign * amplitude * disp` and adopts it as the new molecular coordinates (with full Undo support).
        """
        from .controllers import VibrationController
        VibrationController(self).adopt_vibrational_displacement(sign)

    def _on_export_vibration_gif(self):
        """Export one full oscillation cycle of the active vibrational mode as an animated GIF."""
        from .controllers import VibrationController
        VibrationController(self).export_vibration_gif()

    def _populate_available_symmetries(
        self,
        avail: List[Tuple[str, PointGroupResult, str]],
        selected_symbol: Optional[str] = None,
    ):
        """Populate both the Symmetry tab and 3D viewport symmetry combo boxes with Higher / Detected / Lower groups."""
        from .controllers import SymmetryController
        SymmetryController(self).populate_available_symmetries(avail, selected_symbol=selected_symbol)

    def _update_sym_step_buttons(self):
        """Enable or disable the Higher / Lower step buttons based on current index in _available_symmetries."""
        from .controllers import SymmetryController
        SymmetryController(self).update_sym_step_buttons()

    def select_symmetry_group(self, group_symbol: str, enforce_if_higher: bool = True) -> Optional[PointGroupResult]:
        """Programmatically or interactively select a higher, detected, or lower point group symmetry group_symbol."""
        from .controllers import SymmetryController
        return SymmetryController(self).select_symmetry_group(group_symbol, enforce_if_higher=enforce_if_higher)

    def _on_sym_group_combo_changed(self, idx: int):
        combo = getattr(self, "combo_sym_group", None)
        if combo is None or idx < 0:
            return
        sym = combo.itemData(idx)
        if sym:
            self.select_symmetry_group(str(sym), enforce_if_higher=False)

    def _on_view_sym_group_changed(self, idx: int):
        combo = getattr(self, "combo_view_sym_group", None)
        if combo is None or idx < 0:
            return
        sym = combo.itemData(idx)
        if sym:
            self.select_symmetry_group(str(sym), enforce_if_higher=False)

    def _on_step_higher_symmetry(self):
        """Step to the next higher-order symmetry group in _available_symmetries."""
        avail = getattr(self, "_available_symmetries", [])
        combo = getattr(self, "combo_sym_group", None)
        if not avail or combo is None:
            return
        idx = combo.currentIndex()
        if idx > 0:
            next_sym = avail[idx - 1][0]
            self.select_symmetry_group(next_sym, enforce_if_higher=False)

    def _on_step_lower_symmetry(self):
        """Step to the next lower-order symmetry subgroup in _available_symmetries."""
        avail = getattr(self, "_available_symmetries", [])
        combo = getattr(self, "combo_sym_group", None)
        if not avail or combo is None:
            return
        idx = combo.currentIndex()
        if 0 <= idx < len(avail) - 1:
            next_sym = avail[idx + 1][0]
            self.select_symmetry_group(next_sym, enforce_if_higher=False)

    def _refresh_symmetry_ui(self, res: PointGroupResult, role_tag: Optional[str] = None):
        """Populate symmetry labels and operations table from PointGroupResult without switching tabs."""
        from .controllers import SymmetryController
        SymmetryController(self).refresh_symmetry_ui(res, role_tag=role_tag)

    def _on_detect_symmetry(self, checked_or_switch=True, *, switch_tab: bool = True, preferred_symbol: Optional[str] = None):
        """Analyze molecular 3D coordinates, discover higher/detected/lower symmetries, and update UI."""
        from .controllers import SymmetryController
        SymmetryController(self).detect_symmetry(checked_or_switch, switch_tab=switch_tab, preferred_symbol=preferred_symbol)

    def _on_sym_slider_tol_changed(self, val: int):
        tol = val / 100.0
        self.spin_sym_tol.blockSignals(True)
        self.spin_sym_tol.setValue(tol)
        self.spin_sym_tol.blockSignals(False)
        if len(self.zatoms) > 0:
            self._on_detect_symmetry(switch_tab=False)

    def _on_sym_spin_tol_changed(self, val: float):
        s_val = int(round(val * 100.0))
        if hasattr(self, "slider_sym_tol"):
            self.slider_sym_tol.blockSignals(True)
            self.slider_sym_tol.setValue(s_val)
            self.slider_sym_tol.blockSignals(False)
        if len(self.zatoms) > 0:
            self._on_detect_symmetry(switch_tab=False)

    def _on_align_symmetry(self):
        """Align molecule according to highest C_n (along Z) and principal plane (in XZ or XY)."""
        from .controllers import SymmetryController
        SymmetryController(self).align_symmetry()

    def _on_enforce_symmetry(self):
        """Project coordinates into exact point group subspace if tolerance is not fulfilled."""
        from .controllers import SymmetryController
        SymmetryController(self).enforce_symmetry()

    def _on_toggle_impose_symmetry(self, checked: bool):
        """Toggle GaussView-style Impose / Constrain Symmetry mode."""
        self.set_impose_symmetry(checked)

    def set_impose_symmetry(self, enabled: bool, sym_res: Optional[PointGroupResult] = None):
        """Enable or disable symmetry-constrained editing (Impose Symmetry)."""
        from .controllers import SymmetryController
        SymmetryController(self).set_impose_symmetry(enabled, sym_res=sym_res)

    def _on_show_symmetry_matrices(self):
        """Produce and display 3x3 symmetry operation matrices with interactive selector and view-all capability."""
        res = getattr(self, "_current_sym_res", None)
        if not res or not res.operations:
            QtWidgets.QMessageBox.information(self, "Symmetry Matrices", "Please detect symmetry first.")
            return

        from .dialog_symmetry_matrices import SymmetryMatricesDialog
        bg_mode = getattr(self.viewer, "bg_color_mode", "")
        curr_row = self.table_symmetry_ops.currentRow()
        dlg = SymmetryMatricesDialog(self, sym_res=res, initial_row=curr_row, bg_color_mode=bg_mode)
        dlg.exec()

    def _on_calculate_symmetry_representations(self):
        """Calculate reducible representations and direct sum decomposition of irreps."""
        if len(self.zatoms) == 0:
            return

        res = getattr(self, "_current_sym_res", None)
        if not res:
            self._on_detect_symmetry()
            res = getattr(self, "_current_sym_res", None)

        if not res:
            return

        symbols = [a.symbol for a in self.zatoms]
        reps = calculate_symmetry_representations(symbols, self.coords, res)

        # Format output message
        classes_str = "  ".join(f"{c_lbl}" for c_lbl in reps["classes"])
        chars_str = "  ".join(f"{int(c_val) if abs(c_val - round(c_val)) < 1e-4 else f'{c_val:.1f}'}" for c_val in reps["gamma_tot_chars"])

        is_white = getattr(self.viewer, "bg_color_mode", "") == "white"
        group_col = "#0284c7" if is_white else "#38bdf8"
        tot_col = "#b45309" if is_white else "#facc15"
        trans_col = "#0284c7" if is_white else "#38bdf8"
        rot_col = "#6d28d9" if is_white else "#a78bfa"
        vib_col = "#047857" if is_white else "#4ade80"

        html_text = (
            f"<b>Point Group:</b> <span style='color:{group_col};'>{res.symbol}</span> (Order {res.order})<br><br>"
            f"<b>Symmetry Classes:</b><br>"
            f"&nbsp;&nbsp;Classes:&nbsp;&nbsp;{classes_str}<br>"
            f"&nbsp;&nbsp;Γ_tot:&nbsp;&nbsp;&nbsp;&nbsp;{chars_str}<br><br>"
            f"<b>Reducible Representations Decomposition:</b><br>"
            f"&nbsp;&nbsp;• <b>Γ_tot (3N)</b>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= <span style='color:{tot_col};'>{reps['gamma_tot_str']}</span><br>"
            f"&nbsp;&nbsp;• <b>Γ_trans (x,y,z)</b>&nbsp;= <span style='color:{trans_col};'>{reps['gamma_trans_str']}</span><br>"
            f"&nbsp;&nbsp;• <b>Γ_rot (Rx,Ry,Rz)</b>= <span style='color:{rot_col};'>{reps['gamma_rot_str']}</span><br><br>"
            f"<b>Vibrational Modes Analysis (Normal Modes):</b><br>"
            f"&nbsp;&nbsp;• <b>Γ_vib</b> = Γ_tot ⊖ Γ_trans ⊖ Γ_rot<br>"
            f"&nbsp;&nbsp;• <b>Γ_vib</b> = <span style='color:{vib_col}; font-size:12px; font-weight:bold;'>{reps['gamma_vib_str']}</span>"
        )
        self.lbl_sym_reps.setText(html_text)
        self.box_sym_reps.setVisible(True)
        self.statusBar().showMessage(
            f"Symmetry analysis: Γ_vib = {reps['gamma_vib_str']} | Γ_rot = {reps['gamma_rot_str']}", 6000
        )

    # -------------------------------------------------------------------------
    # Calculation Runner & Live Job Monitor
    # -------------------------------------------------------------------------
    def _on_quick_run_orca(self):
        """1-Click quick ORCA calculation modal for students and lab exercises."""
        from .controllers import JobController
        JobController(self).quick_run_orca()

    def _on_open_preflight_check_dialog(self):
        """Perform comprehensive pre-flight verification and display detailed inspection dialog."""
        from .controllers import JobController
        JobController(self).open_preflight_check_dialog()

    def _on_open_job_monitor(self):
        """Open or show the Job Monitor Dialog."""
        from .controllers import JobController
        JobController(self).open_job_monitor()

    def _on_open_workflow_dag(self):
        """Open the Visual Workflow DAG Pipeline Editor (Multi-Stage QM)."""
        from .controllers import JobController
        JobController(self).open_workflow_dag()

    def _on_configure_qm_engines(self):
        """Open the QM Engines Detection & Route Configuration Dialog."""
        from .controllers import JobController
        JobController(self).configure_qm_engines()

    def _on_open_remote_file_manager(self):
        """Open graphical remote file transfer dialog."""
        from .controllers import JobController
        JobController(self).open_remote_file_manager()

    def launch_calculation_job(self, input_file: str, runner_type: str = "brorsen"):
        """Launch background QM calculation with live streaming output and 3D viewport updates."""
        from .controllers import JobController
        JobController(self).launch_calculation_job(input_file, runner_type=runner_type)

    def _on_live_calc_step(self, symbols: List[str], coords: np.ndarray, energy: float):
        """Update 3D viewport geometry and status bar live as optimization step arrives."""
        if len(symbols) == 0 or coords is None or len(coords) != len(symbols):
            return

        # If molecule topology matches, update coords smoothly
        if len(self.zatoms) == len(symbols):
            user_bonds = self._current_explicit_bonds()
            bo = getattr(self.viewer, "bond_orders", {})
            try:
                new_zatoms = cartesian_to_zmatrix(symbols, coords, existing_zatoms=self.zatoms, user_bonds=user_bonds)
                self.set_molecule_from_zatoms(new_zatoms, exact_coords=coords)
                if bo:
                    self.viewer.bond_orders = bo
            except Exception:
                self.coords = coords.copy()
        else:
            # Rebuild molecule if atom count changed
            bonds = detect_connectivity(symbols, coords)
            bo = detect_bond_orders(symbols, coords, bonds)
            zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
            self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
            self.viewer.bond_orders = bo

        self.viewer.update()
        if energy:
            self.statusBar().showMessage(f"Live Optimization Step: E = {energy:.8f} Hartree", 3000)
        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                bus.job_step_received.emit(symbols, coords, float(energy or 0.0))
        except (RuntimeError, AttributeError):
            pass

    def _on_live_calc_completed(self, input_file: str):
        """Handle completion of live job: locate and open log / trajectory / molden / cube files."""
        try:
            bus = getattr(self, "event_bus", None)
            if bus:
                bus.job_completed.emit(input_file, True)
        except (RuntimeError, AttributeError):
            pass
        # Check if runner has a dedicated work_dir (e.g. RemoteSshRunner downloading to work_dir)
        job_mon = getattr(self, "_job_monitor_dlg", None)
        runner_work_dir = getattr(job_mon.runner, "work_dir", None) if job_mon and getattr(job_mon, "runner", None) else None
        runner_log_file = getattr(job_mon.runner, "log_file", None) if job_mon and getattr(job_mon, "runner", None) else None

        base_dir = runner_work_dir if runner_work_dir and os.path.isdir(runner_work_dir) else os.path.dirname(input_file)
        stem = os.path.splitext(os.path.basename(input_file))[0]

        # 1. Check for .out or .log or trajectory xyz
        candidates = []
        if runner_log_file and os.path.isfile(runner_log_file):
            candidates.append(runner_log_file)
        candidates.extend([
            os.path.join(base_dir, f"{stem}.out"),
            os.path.join(base_dir, f"{stem}.log"),
            os.path.join(base_dir, f"{stem}_final.xyz"),
            os.path.join(base_dir, f"{stem}_opt.xyz"),
            os.path.join(base_dir, "xtbopt.xyz"),
            os.path.join(base_dir, f"{stem}.xyz"),
        ])
        if runner_work_dir and runner_work_dir != os.path.dirname(input_file):
            inp_dir = os.path.dirname(input_file)
            candidates.extend([
                os.path.join(inp_dir, f"{stem}.out"),
                os.path.join(inp_dir, f"{stem}.log"),
                os.path.join(inp_dir, f"{stem}_final.xyz"),
            ])
        opened = False
        for c in candidates:
            if os.path.isfile(c):
                try:
                    self._suppress_companion_prompt = True
                    self.open_path(c)
                    opened = True
                    break
                except Exception as e:
                    log.warning(f"Could not open result file {c}: {e}")
                finally:
                    self._suppress_companion_prompt = False

        # 2. Check for generated molden files: parse [Artifact Generated] lines from .out/.log and scan for %jobname and stem
        artifact_moldens = []
        out_path = runner_log_file if runner_log_file and os.path.isfile(runner_log_file) else None
        if not out_path or not os.path.isfile(out_path):
            out_path = os.path.join(base_dir, f"{stem}.out")
        if not os.path.isfile(out_path):
            out_path = os.path.join(base_dir, f"{stem}.log")

        candidate_stems = [stem]
        if os.path.isfile(input_file):
            try:
                import re
                with open(input_file, "r", encoding="utf-8", errors="ignore") as f_inp:
                    inp_txt = f_inp.read()
                for m_jn in re.finditer(r"^\s*%jobname\s+([^\s]+)", inp_txt, re.MULTILINE | re.IGNORECASE):
                    jn = m_jn.group(1).strip()
                    if jn and jn not in candidate_stems:
                        candidate_stems.append(jn)
            except Exception as e:
                log.debug("Failed parsing %%jobname from %s: %s", input_file, e)

        if os.path.isfile(out_path):
            try:
                import re
                with open(out_path, "r", encoding="utf-8", errors="ignore") as f_out:
                    out_txt = f_out.read()
                for m_art in re.finditer(r"\[Artifact Generated\]\s+([^\s]+\.molden(?:\.input)?)", out_txt):
                    art_file = m_art.group(1).strip()
                    art_full = art_file if os.path.isabs(art_file) else os.path.join(base_dir, art_file)
                    if os.path.isfile(art_full) and art_full not in artifact_moldens:
                        artifact_moldens.append(art_full)
            except Exception:
                log.debug("Ignored non-fatal exception", exc_info=True)

        # Prefer the latest electronic molden from the final stage, while recording all artifact moldens
        elec_artifacts = [p for p in artifact_moldens if "positron" not in os.path.basename(p).lower()]
        pos_artifacts = [p for p in artifact_moldens if "positron" in os.path.basename(p).lower()]

        molden_cand = list(reversed(elec_artifacts)) + list(reversed(pos_artifacts))
        for c_stem in candidate_stems:
            for s_dir in filter(None, [base_dir, os.path.dirname(input_file)]):
                molden_cand.extend([
                    os.path.join(s_dir, f"{c_stem}_hf_electrons.molden"),
                    os.path.join(s_dir, f"{c_stem}_electrons.molden"),
                    os.path.join(s_dir, f"{c_stem}.molden.input"),
                    os.path.join(s_dir, f"{c_stem}.molden"),
                    os.path.join(s_dir, f"{c_stem}.e-.molden"),
                    os.path.join(s_dir, f"{c_stem}.E-.molden"),
                    os.path.join(s_dir, f"{c_stem}.E-ALPHA.molden"),
                    os.path.join(s_dir, f"{c_stem}_hf_positrons.molden"),
                    os.path.join(s_dir, f"{c_stem}_positrons.molden"),
                    os.path.join(s_dir, f"{c_stem}_hf_positron.molden"),
                    os.path.join(s_dir, f"{c_stem}_positron.molden"),
                    os.path.join(s_dir, f"{c_stem}.POSITRON.molden"),
                    os.path.join(s_dir, f"{c_stem}.e+.molden"),
                    os.path.join(s_dir, f"{c_stem}.E+.molden"),
                    os.path.join(s_dir, f"{c_stem}.gbw"),
                ])
        molden_cand.append(os.path.join(base_dir, "orbitals.molden"))
        self._related_molden_files = list(artifact_moldens)
        loaded_molden = False
        active_found_molden = None
        for m in molden_cand:
            if os.path.isfile(m):
                if m.lower().endswith(".gbw"):
                    self._on_open_volumetric_dialog()
                    if hasattr(self, "_volumetric_dlg"):
                        conv = self._volumetric_dlg._convert_gbw_to_molden(m)
                        if conv and os.path.isfile(conv):
                            m = conv
                        else:
                            continue
                self.statusBar().showMessage(f"Calculation finished! Molden file available: {os.path.basename(m)}", 6000)
                try:
                    self._active_molden_path = m
                    self.load_orbitals_from_molden(m, auto_render=True)
                    if hasattr(self, "_volumetric_dlg") and self._volumetric_dlg:
                        self._volumetric_dlg.load_molden_path(m)
                    loaded_molden = True
                    active_found_molden = m
                except Exception as e:
                    log.warning(f"Could not load orbitals from {m}: {e}")
                break

        # Also search for companion .cube files generated in the working directory
        cube_files = []
        try:
            for s_dir in set(filter(None, [base_dir, os.path.dirname(input_file)])):
                if os.path.isdir(s_dir):
                    for fname in os.listdir(s_dir):
                        if fname.lower().endswith(".cube"):
                            c_full = os.path.join(s_dir, fname)
                            if c_full not in cube_files:
                                cube_files.append(c_full)
        except Exception as e:
            log.debug("Error discovering companion cube files: %s", e)
        self._related_cube_files = cube_files

        # Store comprehensive calculation results bundle for instant 1-click access across all property dialogs
        self._last_calculation_results = {
            "input_file": input_file,
            "out_file": out_path if os.path.isfile(out_path) else None,
            "molden_file": active_found_molden,
            "cube_files": cube_files,
            "base_dir": base_dir,
        }

        # Also check if a companion positron molden file exists alongside electron molden
        for pos_cand in (
            list(reversed(pos_artifacts))
            + [
                os.path.join(base_dir, f"{stem}_positrons.molden"),
                os.path.join(base_dir, f"{stem}_hf_positrons.molden"),
                os.path.join(base_dir, f"{stem}_positron.molden"),
                os.path.join(base_dir, f"{stem}_hf_positron.molden"),
                os.path.join(base_dir, f"{stem}.POSITRON.molden"),
            ]
        ):
            if os.path.isfile(pos_cand):
                self._companion_positron_molden_path = pos_cand
                if hasattr(self, "btn_positron_overlay"):
                    self.btn_positron_overlay.setVisible(True)
                break

        # Automatically switch to the most informative inspector tab
        if hasattr(self, "tabs"):
            if getattr(self, "vibrational_analysis", None) or getattr(self, "_vib_modes", None):
                idx = self.tabs.indexOf(getattr(self, "tab_vibrations", None))
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
            elif loaded_molden:
                idx = self.tabs.indexOf(getattr(self, "tab_orbitals", None))
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)

        # 3. Present Publication-Ready Calculation Summary Card
        try:
            res_log = out_path if out_path and os.path.isfile(out_path) else None
            if res_log:
                from .qm_log_parser import parse_qm_logfile
                from .dialog_summary_card import QMSummaryCardDialog
                summary_data = parse_qm_logfile(res_log)
                if summary_data and (summary_data.trajectory_energies or summary_data.symbols):
                    pt_grp = getattr(getattr(self, "_current_sym_res", None), "symbol", "C1")
                    dlg_card = QMSummaryCardDialog(self, summary_data, res_log, point_group=pt_grp)
                    dlg_card.show()
                    dlg_card.raise_()
        except Exception as e:
            log.warning(f"Could not display calculation summary card: {e}", exc_info=True)

    def _on_open_summary_card(self):
        """Open publication-ready 1-page summary card of current or last-loaded QM calculation results."""
        from .dialog_summary_card import QMSummaryCardDialog
        last_qm = getattr(self, "_last_qm_res", None)
        path = getattr(self, "_last_loaded_qm_log_path", "Current Molecule")
        if last_qm:
            pt_grp = getattr(getattr(self, "_current_sym_res", None), "symbol", "C1")
            dlg = QMSummaryCardDialog(self, last_qm, path, point_group=pt_grp)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        else:
            self.statusBar().showMessage("No QM calculation log loaded yet. Run a calculation or open an output file.", 3500)

    def _on_open_nci_plot_dialog(self):
        """Open 2D Reduced Density Gradient (NCI / RDG) scatter plot dialog."""
        from .dialog_nci_plot import NCIRdgPlotDialog
        molden_path = getattr(self, "_active_molden_path", None)
        if not molden_path:
            last_res = getattr(self, "_last_calculation_results", {})
            if last_res.get("molden_file") and os.path.isfile(last_res["molden_file"]):
                molden_path = last_res["molden_file"]
                self._active_molden_path = molden_path
        if not molden_path:
            # Check last QM log companion
            last_qm = getattr(self, "_last_loaded_qm_log_path", None)
            if last_qm:
                base = os.path.splitext(last_qm)[0]
                for ext in (".molden", ".molden.input", "_electrons.molden"):
                    if os.path.isfile(base + ext):
                        molden_path = base + ext
                        break
        if not molden_path or not os.path.isfile(molden_path):
            molden_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Molden File for NCI / RDG Analysis", "", "Molden Files (*.molden *.molden.input *.molf);;All Files (*)"
            )
        if molden_path and os.path.isfile(molden_path):
            dlg = NCIRdgPlotDialog(self, molden_path=molden_path)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
        else:
            self.statusBar().showMessage("NCI plot requires an electronic wavefunction (.molden file).", 3500)

    # -------------------------------------------------------------------------
    # Quick Molecular Orbital Inspector
    # -------------------------------------------------------------------------
    def load_orbitals_from_molden(self, path: str, auto_render: bool = True):
        """Parse Molden file and populate the Molecular Orbitals table."""
        from .volumetric import parse_molden_file
        data = parse_molden_file(path)
        if not data or not data.orbitals:
            return

        self._active_molden_path = path
        self.set_orbitals_from_data(data, auto_render=auto_render, source_path=path)

    def set_orbitals_from_data(
        self,
        data: Any,
        homo_idx: Optional[int] = None,
        auto_render: bool = False,
        source_path: Optional[str] = None,
        source_label: Optional[str] = None,
    ):
        """Populate the Molecular Orbitals table and energy diagram from MoldenFileData or QM log."""
        if not data or not getattr(data, "orbitals", None):
            return

        self._active_molden_data = data
        if source_path:
            self._active_molden_path = source_path
        elif not getattr(self, "_active_molden_path", None):
            # Check if a companion molden exists in the same folder as the last loaded QM log
            last_qm = getattr(self, "_last_loaded_qm_log_path", None)
            if last_qm:
                q_dir = os.path.dirname(last_qm)
                q_base = os.path.splitext(os.path.basename(last_qm))[0]
                for cand_ext in (".molden", "_electrons.molden", "_hf_electrons.molden", ".molden.input"):
                    c_path = os.path.join(q_dir, q_base + cand_ext)
                    if os.path.isfile(c_path):
                        self._active_molden_path = c_path
                        break

        self.table_orbitals.blockSignals(True)
        self.table_orbitals.setRowCount(len(data.orbitals))
        if homo_idx is None:
            for idx, mo in enumerate(data.orbitals):
                if mo.occupancy > 0.05:
                    homo_idx = idx

        for idx, mo in enumerate(data.orbitals):
            item_num = QtWidgets.QTableWidgetItem(str(idx + 1))
            item_num.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            tag = getattr(mo, "sym", f"MO {idx+1}")
            tag_color = "#94a3b8"
            if homo_idx is not None:
                if idx == homo_idx:
                    tag = "HOMO"
                    tag_color = "#3b82f6"  # Blue
                elif idx == homo_idx + 1:
                    tag = "LUMO"
                    tag_color = "#a855f7"  # Purple
                elif idx == homo_idx - 1:
                    tag = "HOMO-1"
                elif idx == homo_idx + 2:
                    tag = "LUMO+1"

            item_tag = QtWidgets.QTableWidgetItem(tag)
            item_tag.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_tag.setForeground(QtGui.QBrush(QtGui.QColor(tag_color)))
            item_tag.setFont(QtGui.QFont("Monospace", 9, QtGui.QFont.Weight.Bold))

            # Energies: 1 a.u. = 27.211386245988 eV
            ev_val = mo.energy * 27.211386
            item_ev = QtWidgets.QTableWidgetItem(f"{ev_val:+.3f}")
            item_ev.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            item_au = QtWidgets.QTableWidgetItem(f"{mo.energy:+.5f}")
            item_au.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            item_occ = QtWidgets.QTableWidgetItem(f"{mo.occupancy:.1f}")
            item_occ.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

            self.table_orbitals.setItem(idx, 0, item_num)
            self.table_orbitals.setItem(idx, 1, item_tag)
            self.table_orbitals.setItem(idx, 2, item_ev)
            self.table_orbitals.setItem(idx, 3, item_au)
            self.table_orbitals.setItem(idx, 4, item_occ)
        self.table_orbitals.blockSignals(False)

        gap_str = ""
        if homo_idx is not None and (homo_idx + 1) < len(data.orbitals):
            e_homo = data.orbitals[homo_idx].energy
            e_lumo = data.orbitals[homo_idx + 1].energy
            gap_au = e_lumo - e_homo
            gap_ev = gap_au * 27.211386
            gap_str = f" | <b>ΔE(HOMO–LUMO) = {gap_ev:.2f} eV</b> ({gap_au:.4f} a.u.)"

        display_name = source_label or (os.path.basename(source_path) if source_path else "Calculation Orbitals")
        if hasattr(self, "lbl_mo_status") and self.lbl_mo_status:
            self.lbl_mo_status.setText(
                f"<b>{display_name}</b>: {len(data.orbitals)} MOs loaded. "
                + (f"HOMO is MO #{homo_idx+1}{gap_str}" if homo_idx is not None else "")
            )

        path = source_path or getattr(self, "_active_molden_path", "") or ""

        # Auto-detect companion positron/electron Molden files in the same directory
        self._companion_positron_molden_path = None
        base_dir = os.path.dirname(path) if path else ""
        fname = os.path.basename(path) if path else ""
        pos_candidates = []
        elec_candidates = []
        if "_electrons" in fname:
            pos_candidates.append(os.path.join(base_dir, fname.replace("_electrons", "_positrons")))
            pos_candidates.append(os.path.join(base_dir, fname.replace("_electrons", "_positron")))
        if "_positrons" in fname:
            elec_candidates.append(os.path.join(base_dir, fname.replace("_positrons", "_electrons")))
        elif "_positron" in fname:
            elec_candidates.append(os.path.join(base_dir, fname.replace("_positron", "_electrons")))
        if ".E-ALPHA" in fname:
            pos_candidates.append(os.path.join(base_dir, fname.replace(".E-ALPHA", ".POSITRON")))
        if ".POSITRON" in fname:
            elec_candidates.append(os.path.join(base_dir, fname.replace(".POSITRON", ".E-ALPHA")))
        stem = os.path.splitext(fname)[0]
        pos_candidates.extend([
            os.path.join(base_dir, f"{stem}_positrons.molden"),
            os.path.join(base_dir, f"{stem}_hf_positrons.molden"),
            os.path.join(base_dir, f"{stem}_positron.molden"),
            os.path.join(base_dir, f"{stem}_hf_positron.molden"),
            os.path.join(base_dir, f"{stem}.POSITRON.molden"),
        ])
        # Also include any positron molden discovered in _related_molden_files from multi-stage .out logs
        for rel in getattr(self, "_related_molden_files", []):
            if "positron" in os.path.basename(rel).lower():
                pos_candidates.append(rel)

        for pc in pos_candidates:
            if os.path.isfile(pc) and os.path.abspath(pc) != os.path.abspath(path):
                self._companion_positron_molden_path = pc
                break

        if hasattr(self, "btn_positron_overlay"):
            self.btn_positron_overlay.setVisible(bool(self._companion_positron_molden_path))

        # Populate the MO Source Switcher dropdown (Electrons vs Positrons / Multi-Stage Molden files)
        if hasattr(self, "combo_mo_file_source"):
            related = []
            for p_cand in [path] + elec_candidates + pos_candidates + list(getattr(self, "_related_molden_files", [])):
                if p_cand and os.path.isfile(p_cand):
                    abs_p = os.path.abspath(p_cand)
                    if all(os.path.abspath(x) != abs_p for x in related):
                        related.append(p_cand)
            self.combo_mo_file_source.blockSignals(True)
            self.combo_mo_file_source.clear()
            for r_path in related:
                bname = os.path.basename(r_path)
                if "positron" in bname.lower():
                    lbl = f"⚛️ Positron MOs ({bname})"
                elif "electron" in bname.lower() or ".e-alpha" in bname.lower():
                    lbl = f"⚡ Electron MOs ({bname})"
                else:
                    lbl = f"📄 {bname}"
                self.combo_mo_file_source.addItem(lbl, r_path)
                if os.path.abspath(r_path) == os.path.abspath(path):
                    self.combo_mo_file_source.setCurrentIndex(self.combo_mo_file_source.count() - 1)
            self.combo_mo_file_source.setVisible(len(related) > 1)
            self.combo_mo_file_source.blockSignals(False)

        if hasattr(self, "mo_ladder_widget"):
            self.mo_ladder_widget.set_orbitals(data.orbitals, homo_idx=homo_idx)
        if hasattr(self, "dos_widget"):
            self.dos_widget.set_orbitals(data.orbitals, homo_idx=homo_idx)

        # Update HOMO-LUMO Gap Summary Badge (Chemcraft style) with Electronic Term Symbol
        if hasattr(self, "lbl_mo_gap_badge"):
            if homo_idx is not None and 0 <= homo_idx < len(data.orbitals) - 1:
                e_homo = float(data.orbitals[homo_idx].energy) * 27.211386
                e_lumo = float(data.orbitals[homo_idx + 1].energy) * 27.211386
                gap_ev = e_lumo - e_homo
                gap_kcal = gap_ev * 23.0605
                hardness = gap_ev / 2.0
                mu_chem = (e_homo + e_lumo) / 2.0

                # Compute electronic term symbol from point group and MO occupancies
                term_str = ""
                try:
                    from .symmetry import find_symmetry_operations, determine_electronic_term_symbol
                    pt_grp = "C1"
                    coords_to_use = getattr(data, "coords", None)
                    symbols_to_use = getattr(data, "symbols", None)
                    if coords_to_use is None or len(coords_to_use) == 0:
                        coords_to_use = self.coords
                        symbols_to_use = self.symbols
                    if symbols_to_use is not None and len(symbols_to_use) > 0 and len(coords_to_use) == len(symbols_to_use):
                        pg_res = find_symmetry_operations(symbols_to_use, coords_to_use, tolerance=0.1)
                        if pg_res:
                            pt_grp = pg_res.symbol
                    term = determine_electronic_term_symbol(pt_grp, data.orbitals)
                    if term:
                        term_str = f"  |  <b>State: {term}</b> ({pt_grp})"
                except Exception as e:
                    log.debug("Failed determining electronic term symbol: %s", e)

                self.lbl_mo_gap_badge.setText(
                    f"⚡ HOMO: {e_homo:+.2f} eV  |  LUMO: {e_lumo:+.2f} eV  |  "
                    f"ΔE_gap = {gap_ev:.2f} eV ({gap_kcal:.1f} kcal/mol)  |  "
                    f"Hardness (η): {hardness:.2f} eV  |  μ: {mu_chem:+.2f} eV{term_str}"
                )
                self.lbl_mo_gap_badge.setVisible(True)
            else:
                self.lbl_mo_gap_badge.setVisible(False)

        # Select HOMO by default if found
        if homo_idx is not None:
            if auto_render:
                self.table_orbitals.selectRow(homo_idx)
            else:
                self.table_orbitals.blockSignals(True)
                self.table_orbitals.selectRow(homo_idx)
                self.table_orbitals.blockSignals(False)
                if hasattr(self, "mo_ladder_widget"):
                    self.mo_ladder_widget.set_selected_orbital(homo_idx)

    def _on_mo_source_combo_changed(self, idx: int):
        if not hasattr(self, "combo_mo_file_source") or idx < 0:
            return
        target_path = self.combo_mo_file_source.itemData(idx)
        if target_path and os.path.isfile(target_path) and os.path.abspath(target_path) != os.path.abspath(getattr(self, "_active_molden_path", "")):
            self.load_orbitals_from_molden(target_path, auto_render=True)
            if hasattr(self, "_volumetric_dlg") and self._volumetric_dlg and self._volumetric_dlg.isVisible():
                self._volumetric_dlg.load_molden_path(target_path)

    def _on_mo_table_selected(self):
        selected = self.table_orbitals.selectedItems()
        if not selected:
            return
        row = self.table_orbitals.row(selected[0])
        if hasattr(self, "mo_ladder_widget"):
            self.mo_ladder_widget.set_selected_orbital(row)
        if getattr(self, "_active_molden_path", None) and os.path.isfile(self._active_molden_path):
            try:
                from .volumetric import compute_molden_mo_atom_contributions
                comp_str = compute_molden_mo_atom_contributions(self._active_molden_path, row)
                if comp_str and hasattr(self, "lbl_mo_status"):
                    mo_type = self.table_orbitals.item(row, 1).text() if self.table_orbitals.item(row, 1) else f"MO #{row + 1}"
                    self.lbl_mo_status.setText(f"🧬 <b>MO #{row + 1} ({mo_type}) Composition:</b> {comp_str}")
            except Exception as e:
                log.debug("Failed computing MO atom contributions: %s", e)
        self._render_mo_orbital(row)

    def _render_mo_orbital(self, mo_index: int):
        from .volumetric import evaluate_molden_orbital_grid, generate_orbital_surfaces
        has_molden = bool(getattr(self, "_active_molden_path", None) and os.path.isfile(self._active_molden_path))
        if not has_molden:
            # Built-in Semiempirical Tight-Binding MO evaluation (zero external dependencies)
            if not getattr(self, "zatoms", None) or len(self.zatoms) == 0:
                return
            from .semiempirical import evaluate_semiempirical_orbital_grid
            isovalue = self.spin_quick_iso.value() if hasattr(self, "spin_quick_iso") else 0.030
            opacity = (self.slider_quick_opacity.value() / 100.0) if hasattr(self, "slider_quick_opacity") else 0.80
            n_pts = int(self.combo_quick_grid.currentData() or 36) if hasattr(self, "combo_quick_grid") else 36
            symbols = [a.symbol for a in self.zatoms]
            coords = np.array(self.coords, dtype=np.float64, copy=True)
            qm_props = getattr(self, "_semiempirical_qm_props", None)
            vol = evaluate_semiempirical_orbital_grid(
                symbols, coords, orbital_index=mo_index, nx=n_pts, ny=n_pts, nz=n_pts, qm_props=qm_props
            )
            if vol is None:
                return
            max_abs = float(np.max(np.abs(vol.grid_data)))
            eff_iso = isovalue if (isovalue < max_abs and max_abs > 1e-5) else round(max_abs * 0.45, 4)
            pos_c, neg_c = self._get_mo_phase_colors()
            meshes = generate_orbital_surfaces(vol, isovalue=max(1e-4, eff_iso), pos_color=pos_c, neg_color=neg_c)
            self._current_rendered_mo_vol = vol
            self._current_rendered_mo_index = mo_index
            if meshes:
                self.viewer.active_isosurfaces = meshes
                self.viewer.surface_opacity = opacity
                self.viewer.update()
                self.statusBar().showMessage(
                    f"Rendered Instant Semiempirical MO #{mo_index + 1} (Isovalue={eff_iso:.4f})", 4000
                )
            return

        isovalue = self.spin_quick_iso.value() if hasattr(self, "spin_quick_iso") else 0.030
        opacity = (self.slider_quick_opacity.value() / 100.0) if hasattr(self, "slider_quick_opacity") else 0.80
        n_pts = int(self.combo_quick_grid.currentData() or 40) if hasattr(self, "combo_quick_grid") else 40
        margin_b = float(self.spin_quick_margin.value()) if hasattr(self, "spin_quick_margin") else 3.0

        molden_path = self._active_molden_path
        self.statusBar().showMessage(f"Computing 3D orbital mesh for MO #{mo_index+1} ({n_pts}³ grid, margin {margin_b:.1f} a₀)...", 3000)

        def _compute():
            vol = evaluate_molden_orbital_grid(molden_path, orbital_index=mo_index, nx=n_pts, ny=n_pts, nz=n_pts, margin_bohr=margin_b)
            if not vol:
                return [], isovalue, None
            max_abs = float(np.max(np.abs(vol.grid_data)))
            eff_iso = isovalue
            # Only auto-clamp if user has not explicitly set a smaller isovalue or if eff_iso is out of range
            if eff_iso >= max_abs and max_abs > 1e-5:
                eff_iso = round(max_abs * 0.45, 4)
            is_pos_file = "positron" in os.path.basename(molden_path).lower()
            if is_pos_file:
                pos_c, neg_c = (0.95, 0.72, 0.12), (0.16, 0.78, 0.44)
            else:
                pos_c, neg_c = self._get_mo_phase_colors()
            meshes = generate_orbital_surfaces(vol, isovalue=eff_iso, pos_color=pos_c, neg_color=neg_c)
            return meshes, eff_iso, vol

        def _on_done(res_tuple):
            meshes, eff_iso, vol = res_tuple
            if vol:
                self._current_rendered_mo_vol = vol
                self._current_rendered_mo_index = mo_index
                max_abs = float(np.max(np.abs(vol.grid_data)))
                # Dynamically set max slider and spin range to match orbital density
                if hasattr(self, "spin_quick_iso"):
                    self.spin_quick_iso.blockSignals(True)
                    self.spin_quick_iso.setMaximum(max(0.5, round(max_abs * 1.2, 3)))
                    self.spin_quick_iso.setValue(eff_iso)
                    self.spin_quick_iso.blockSignals(False)
                if hasattr(self, "slider_quick_iso"):
                    self.slider_quick_iso.blockSignals(True)
                    s_max = max(100, int(round(max_abs * 1200)))
                    self.slider_quick_iso.setMaximum(s_max)
                    self.slider_quick_iso.setValue(int(round(eff_iso * 1000)))
                    self.slider_quick_iso.blockSignals(False)

            if meshes:
                self.viewer.active_isosurfaces = meshes
                self.viewer.surface_opacity = opacity
                self.viewer.update()
                self.statusBar().showMessage(f"Rendered MO #{mo_index+1} (Isovalue={eff_iso:.4f})", 4000)

        def _on_error(e):
            log.warning(f"Error computing MO #{mo_index+1}: {e}")

        run_in_background(
            self,
            "Rendering Orbital",
            f"Computing 3D GTO grid for MO #{mo_index+1}...",
            _compute,
            _on_done,
            _on_error,
        )

    def _on_load_orbitals_prompt(self):
        """Prompt user to open any calculation file (.molden, .out, .log, .cube) to populate MOs and energy ladder."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Molecular Orbitals & Energy Diagram",
            getattr(self, "_last_opened_dir", ""),
            "Calculation & Orbital Files (*.molden *.molden.input *.molf *.out *.log *.cube *.gbw);;"
            "Molden Files (*.molden *.molden.input *.molf);;"
            "QM Output Logs (*.out *.log);;"
            "Gaussian Cubes (*.cube);;"
            "All Files (*)",
        )
        if not path:
            return

        low = path.lower()
        if low.endswith((".molden", ".molden.input", ".molf", ".gbw")):
            self.load_orbitals_from_molden(path, auto_render=False)
            self._remember_file(path)
        elif low.endswith((".out", ".log")):
            if hasattr(self, "_load_qm_log_path"):
                self._load_qm_log_path(path)
        elif low.endswith(".cube"):
            self._load_cube_path(path)

        idx = self.tabs.indexOf(getattr(self, "tab_orbitals", None))
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    def _on_quick_view_homo(self):
        from .controllers import OrbitalController
        OrbitalController(self).quick_view_homo()

    def _on_quick_view_lumo(self):
        from .controllers import OrbitalController
        OrbitalController(self).quick_view_lumo()

    def _on_quick_view_spin_density(self):
        """1-Click Open-Shell Spin Density (rho_alpha - rho_beta) evaluation and 3D rendering."""
        from .controllers import OrbitalController
        OrbitalController(self).quick_view_spin_density()

    def _on_quick_view_esp_map(self):
        """1-Click Electrostatic Potential (ESP) mapped onto Total Electron Density isosurface."""
        from .controllers import OrbitalController
        OrbitalController(self).quick_view_esp_map()

    def _on_localize_orbitals(self, method: str = "pipek-mezey"):
        """1-Click PySCF orbital localization (Pipek-Mezey or Foster-Boys) from the active Molden file."""
        from .controllers import OrbitalController
        OrbitalController(self).localize_orbitals(method=method)

    def _on_restore_canonical_orbitals(self):
        """Restore the original canonical delocalized MOs after viewing Localized Molecular Orbitals."""
        from .controllers import OrbitalController
        OrbitalController(self).restore_canonical_orbitals()

    def _on_clear_orbital_surface(self):
        from .controllers import OrbitalController
        OrbitalController(self).clear_orbital_surface()

    def _on_overlay_positron_orbital(self):
        """Evaluate the companion positron ground-state orbital and overlay it (gold/green) onto the 3D viewport."""
        from .controllers import OrbitalController
        OrbitalController(self).overlay_positron_orbital()

    def _on_quick_slider_iso_changed(self, val: int):
        iso = val / 1000.0
        if hasattr(self, "spin_quick_iso"):
            self.spin_quick_iso.blockSignals(True)
            self.spin_quick_iso.setValue(iso)
            self.spin_quick_iso.blockSignals(False)
        self._remesh_current_mo(iso)

    def _on_quick_spin_iso_changed(self, val: float):
        if hasattr(self, "slider_quick_iso"):
            self.slider_quick_iso.blockSignals(True)
            self.slider_quick_iso.setValue(int(round(val * 1000.0)))
            self.slider_quick_iso.blockSignals(False)
        self._remesh_current_mo(val)

    def _get_mo_phase_colors(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        from .controllers import OrbitalController
        return OrbitalController(self).get_mo_phase_colors()

    def _remesh_current_mo(self, iso: float):
        """Fast Marching Cubes remeshing using cached 3D grid data if available."""
        from .controllers import OrbitalController
        OrbitalController(self).remesh_current_mo(iso)

    def _on_quick_iso_changed(self, val: float):
        self._on_quick_spin_iso_changed(val)

    def _on_quick_opacity_changed(self, val: int):
        self.viewer.surface_opacity = val / 100.0
        self.viewer.update()

    def _on_open_orbital_snapshot_gallery(self):
        """Open the Orbital Snapshot & Gallery dialog to batch capture all or selected orbitals."""
        from .controllers import OrbitalController
        OrbitalController(self).open_orbital_snapshot_gallery()

    def _on_open_benchmark_dialog(self):
        """Open Cross-Engine Benchmark & Calculation Comparison dialog (Ctrl+Shift+M)."""
        from .dialog_benchmark import BenchmarkComparisonDialog

        initial_files = []
        last_log = getattr(self, "_last_loaded_qm_log_path", None)
        if last_log and os.path.isfile(last_log):
            initial_files.append(last_log)

        dlg = BenchmarkComparisonDialog(parent=self, initial_files=initial_files)

        def _on_structure_selected(syms, coords, label):
            from .zmatrix import cartesian_to_zmatrix
            zatoms = cartesian_to_zmatrix(syms, coords)
            self.set_molecule_from_zatoms(zatoms)
            self.statusBar().showMessage(f"Loaded benchmark structure: {label}", 4000)

        dlg.structure_selected.connect(_on_structure_selected)
        dlg.exec()

    def _on_open_reaction_thermo_dialog(self):
        """Open Reaction Thermochemistry & Eyring Kinetics Workbench (Alt+Shift+R)."""
        from .dialog_reaction_thermo import ReactionThermoDialog

        dlg = ReactionThermoDialog(parent=self)
        last_log = getattr(self, "_last_loaded_qm_log_path", None)
        if last_log and os.path.isfile(last_log):
            from .thermochemistry import extract_species_from_log_file
            try:
                sp = extract_species_from_log_file(last_log, role="reactant")
                dlg.species.append(sp)
                dlg._refresh_table()
                dlg._recalculate()
            except Exception:
                pass
        dlg.exec()

    def _on_open_nbo_dialog(self):
        """Open NBO Donor-Acceptor Perturbation Visualizer (Ctrl+Shift+N)."""
        from .dialog_nbo import NBODonorAcceptorDialog

        last_log = getattr(self, "_last_loaded_qm_log_path", None)
        sel_atoms = getattr(self, "selected_atoms", [])

        dlg = NBODonorAcceptorDialog(
            parent=self,
            log_filepath_or_text=last_log,
            is_path=True,
            selected_atoms=sel_atoms,
        )

        def _on_interaction_highlighted(d_atoms, a_atoms, e2, label):
            all_involved = list(set(d_atoms + a_atoms))
            if hasattr(self, "viewer") and hasattr(self.viewer, "set_selected_atoms"):
                self.viewer.set_selected_atoms(all_involved)
            self.statusBar().showMessage(f"NBO Interaction: {label}", 5000)

        dlg.interaction_highlighted.connect(_on_interaction_highlighted)
        dlg.exec()

    def _on_quick_in_memory_pyscf_dft(self, method: str = "b3lyp", basis: str = "def2-svp"):
        """Fast in-process PySCF DFT / HF calculation directly in RAM with zero temporary file pollution."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(self, "Quick DFT (PySCF)", "Please build or load a molecule first.")
            return

        try:
            import pyscf
            from pyscf import gto, dft, scf
            from pyscf.tools import molden
        except ImportError:
            QtWidgets.QMessageBox.warning(
                self, "PySCF Missing", "PySCF is not installed in the active environment.\nRun: pip install pyscf"
            )
            return

        symbols = [a.symbol for a in self.zatoms]
        coords = np.array(self.coords, copy=True)
        atom_spec = "; ".join(f"{s} {c[0]} {c[1]} {c[2]}" for s, c in zip(symbols, coords))

        import tempfile
        from .async_task import run_in_background

        def _worker():
            mol = gto.M(atom=atom_spec, basis=basis, verbose=0)
            if method.lower() == "hf":
                mf = scf.RHF(mol).run()
            else:
                mf = dft.RKS(mol, xc=method).run()
            
            with tempfile.NamedTemporaryFile("w", suffix=".molden", delete=False) as f:
                molden.from_scf(mf, f.name)
                mpath = f.name
            
            from .volumetric import parse_molden_file
            mdata = parse_molden_file(mpath)
            try:
                os.unlink(mpath)
            except Exception:
                pass
            return mf, mdata

        def _on_done(result):
            mf, mdata = result
            if mdata and mdata.orbitals:
                self.set_orbitals_from_data(mdata, auto_render=True, source_label=f"In-Memory PySCF ({method.upper()}/{basis})")
                e_ha = float(mf.e_tot)
                self.statusBar().showMessage(
                    f"✔ PySCF {method.upper()}/{basis} complete! E = {e_ha:.8f} Ha | Converged: {mf.converged}",
                    6000,
                )

        def _on_error(err):
            QtWidgets.QMessageBox.critical(self, "PySCF DFT Error", f"In-memory DFT calculation failed:\n{err}")

        run_in_background(
            self,
            "Quick PySCF DFT",
            f"Solving {method.upper()}/{basis} in-process ({len(symbols)} atoms)...",
            _worker,
            _on_done,
            _on_error,
        )

    def _on_open_script_runner(self):
        """Open custom Python Script and Plugin Runner dialog (Ctrl+Shift+Y)."""
        from .plugins import PythonScriptRunnerDialog
        dlg = PythonScriptRunnerDialog(self)
        dlg.exec()

    def _on_open_addons_manager(self):
        """Open in-app scientific backend and extensions manager."""
        from .dialog_addons import ExtensionManagerDialog
        dlg = ExtensionManagerDialog(self)
        dlg.exec()

    def _on_open_sample_gallery(self):
        """Open curated chemistry sample and tutorial gallery dialog."""
        from .dialog_samples import SampleGalleryDialog
        dlg = SampleGalleryDialog(self)
        dlg.sample_selected.connect(self._load_sample_xyz_string)
        dlg.exec()

    def _load_sample_xyz_string(self, xyz_text: str):
        """Parse raw XYZ text and load it into the active molecule canvas."""
        lines = [line.strip() for line in xyz_text.strip().splitlines() if line.strip()]
        if not lines:
            return
        try:
            # Skip atom count and comment line if present
            start_idx = 0
            if lines[0].isdigit():
                start_idx = 2
            symbols = []
            coords = []
            for line in lines[start_idx:]:
                parts = line.split()
                if len(parts) >= 4:
                    symbols.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if symbols:
                self.load_cartesian_molecule(symbols, np.array(coords, dtype=np.float64))
                self.statusBar().showMessage(f"✔ Successfully loaded {len(symbols)} atoms from Sample Gallery.", 5000)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load Error", f"Failed to parse sample geometry:\n{e}")

    def _on_open_fmo_alignment(self):
        """Open Frontier Molecular Orbital (FMO) Alignment dialog."""
        from .dialog_fmo_alignment import FMOAlignmentDialog

        # Extract active molecule's HOMO/LUMO if available
        homo_a = -5.80
        lumo_a = -1.20
        name_a = "Active Molecule"

        res = getattr(self, "_last_qm_res", None) or getattr(self, "_last_qm_result", None)
        if res:
            if getattr(res, "homo_ev", None) is not None:
                homo_a = float(res.homo_ev)
            if getattr(res, "lumo_ev", None) is not None:
                lumo_a = float(res.lumo_ev)
        elif len(self.zatoms) > 0:
            name_a = f"{self.zatoms[0].symbol}-System"

        dlg = FMOAlignmentDialog(
            parent=self,
            sys1_name=name_a,
            sys1_homo=homo_a,
            sys1_lumo=lumo_a,
        )
        dlg.exec()

    def _on_open_citations_dialog(self):
        """Open automated BibTeX citation and computational methods dialog."""
        from .dialog_citations import CitationDialog
        dlg = CitationDialog(self)
        dlg.exec()

    def _on_open_cdft_workbench(self):
        """Open Conceptual DFT (CDFT) & Fukui Reactivity Workbench (Ctrl+Shift+F)."""
        from .dialog_cdft import CDFTWorkbenchDialog

        symbols = [a.symbol for a in self.zatoms] if self.zatoms else self.symbols
        coords = self.coords

        homo_val = -6.20
        lumo_val = -1.40
        res = getattr(self, "_last_qm_res", None) or getattr(self, "_last_qm_result", None)
        if res:
            if getattr(res, "homo_ev", None) is not None:
                homo_val = float(res.homo_ev)
            if getattr(res, "lumo_ev", None) is not None:
                lumo_val = float(res.lumo_ev)

        dlg = CDFTWorkbenchDialog(
            parent=self,
            symbols=symbols,
            coords=coords,
            homo_ev=homo_val,
            lumo_ev=lumo_val,
        )

        def _on_apply_dual(syms, dual_vals):
            # Map dual descriptor values into custom_charges for viewport rendering
            if hasattr(self, "viewer"):
                self.viewer.custom_charges = np.array(dual_vals, copy=True)
                self.viewer.show_charges = True
                self.viewer.update()
                if hasattr(self, "chk_charges"):
                    self.chk_charges.blockSignals(True)
                    self.chk_charges.setChecked(True)
                    self.chk_charges.blockSignals(False)
            self.statusBar().showMessage("Applied CDFT Dual Descriptor (Δf) to 3D Viewport", 4000)

        dlg.apply_dual_descriptor_to_viewer.connect(_on_apply_dual)
        dlg.exec()







