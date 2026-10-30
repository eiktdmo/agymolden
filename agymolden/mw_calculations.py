"""Mixin modules extracted from MainWindow for clean separation of concerns."""

import os
import math
import copy
import logging
from typing import List, Optional, Dict, Tuple
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

log = logging.getLogger(__name__)


class CalculationsMixin:
    def _get_selected_opt_method(self) -> str:
        if hasattr(self, "combo_opt_method") and self.combo_opt_method.currentIndex() >= 0:
            data = self.combo_opt_method.currentData()
            if data in ("uff_mmff", "semiempirical"):
                return data
        return "uff_mmff"

    def _on_opt_method_changed(self, idx: int):
        method = self._get_selected_opt_method()
        label = "Semiempirical QM (GFN0-xTB)" if method == "semiempirical" else "Analytical UFF/MMFF94s"
        if hasattr(self, "btn_calc_vibrations"):
            self.btn_calc_vibrations.setText(f"⚡ Calculate Frequencies ({label})")
        self.statusBar().showMessage(f"Selected calculation engine: {label}", 3500)

    def _on_optimize_geometry(self):
        """Optimize molecular geometry using selected force field (UFF/MMFF94s) or Semiempirical QM (GFN0-xTB)."""
        if len(self.zatoms) < 2:
            QtWidgets.QMessageBox.information(
                self, "Optimize Geometry", "At least 2 atoms are required for geometry optimization."
            )
            return

        method = self._get_selected_opt_method()
        method_label = "Semiempirical QM (GFN0-xTB)" if method == "semiempirical" else "Analytical UFF/MMFF94s"
        self._push_undo_state()
        self.statusBar().showMessage(f"Optimizing geometry using {method_label}...", 3000)

        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        bo = dict(getattr(self.viewer, 'bond_orders', {}))
        start_coords = np.array(self.coords, copy=True)
        frozen = {i for i in getattr(self, "frozen_atoms", set()) if 0 <= i < len(symbols)}

        imposed = self._get_active_imposed_symmetry() if not frozen else None

        def run_opt(c0, iters):
            if method == "semiempirical":
                from .semiempirical import optimize_geometry_semiempirical
                return optimize_geometry_semiempirical(
                    symbols, c0, user_bonds=user_bonds, max_iter=iters,
                    user_bond_orders=bo, frozen_indices=frozen
                )
            opt_coords, final_e = optimize_geometry_force_field(
                symbols, c0, user_bonds=user_bonds, max_iter=iters,
                user_bond_orders=bo, method="uff_mmff", frozen_indices=frozen
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
                log.exception("Unhandled error in GUI action")
                log.exception("Applying optimization result failed")
                QtWidgets.QMessageBox.critical(self, "Optimization Error", f"Geometry optimization failed:\n{str(e)}")

        def failed(exc):
            log.error("Geometry optimization failed", exc_info=exc)
            QtWidgets.QMessageBox.critical(self, "Optimization Error", f"Geometry optimization failed:\n{str(exc)}")

        run_in_background(self, "Optimize Geometry", f"Optimizing geometry ({method_label})...",
                          compute, done, failed)

    def _apply_optimization(self, symbols, user_bonds, method, method_label, result):
        opt_coords, final_e, qm_props, sym_msg = result
        sym_kept = sym_msg.startswith("Optimized with")
        qm_summary = ""
        if method == "semiempirical":
            gap = qm_props.get("gap_ev", 0.0)
            homo = qm_props.get("homo_ev", 0.0)
            lumo = qm_props.get("lumo_ev", 0.0)
            qm_summary = f" | HOMO: {homo:.2f} eV, LUMO: {lumo:.2f} eV, Gap: {gap:.2f} eV"

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
            log.error("Conformer search failed", exc_info=exc)
            QtWidgets.QMessageBox.critical(self, "Conformer Search Error", f"Conformational search failed:\n{str(exc)}")

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
        if len(self.zatoms) < 2:
            QtWidgets.QMessageBox.information(
                self, "Vibrational Modes", "At least 2 atoms are required to calculate vibrational modes."
            )
            return

        method = self._get_selected_opt_method()
        method_label = "Semiempirical QM (GFN0-xTB)" if method == "semiempirical" else "Analytical UFF/MMFF94s"
        self.statusBar().showMessage(f"Calculating vibrational Hessian and normal modes ({method_label})...", 4000)

        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        bo = dict(getattr(self.viewer, 'bond_orders', {}))
        start_coords = np.array(self.coords, copy=True)

        def compute():
            # Compute vibrations at current geometry
            return compute_force_field_vibrations(
                symbols, start_coords, user_bonds=user_bonds, optimize_first=False,
                method=method, user_bond_orders=bo
            )

        def done(result):
            try:
                self._apply_vibrations(result)
            except Exception as e:
                log.exception("Unhandled error in GUI action")
                log.exception("Displaying vibrations failed")
                QtWidgets.QMessageBox.critical(self, "Vibration Calculation Error", f"Failed to compute vibrations:\n{str(e)}")

        def failed(exc):
            log.error("Vibration calculation failed", exc_info=exc)
            QtWidgets.QMessageBox.critical(self, "Vibration Calculation Error", f"Failed to compute vibrations:\n{str(exc)}")

        run_in_background(self, "Vibrational Modes", f"Calculating normal modes ({method_label})...",
                          compute, done, failed)

    def _apply_vibrations(self, result):
        eq_coords, analysis = result
        self.vibrational_analysis = analysis

        symbols = [a.symbol for a in self.zatoms]
        sym_res = getattr(self, "_current_sym_res", None)
        scale_fac = self.spectrum_widget.scale_factor if hasattr(self, "spectrum_widget") else 1.0

        # Populate table
        self.table_vibrations.setRowCount(analysis.n_modes)
        for row, freq in enumerate(analysis.frequencies):
            item_idx = QtWidgets.QTableWidgetItem(str(row + 1))
            item_idx.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_idx.setForeground(QtGui.QBrush(QtGui.QColor("#90caf9")))

            item_freq = QtWidgets.QTableWidgetItem(f"{freq:.2f}")
            item_freq.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

            scaled_freq = freq * scale_fac
            item_scaled = QtWidgets.QTableWidgetItem(f"{scaled_freq:.2f}")
            item_scaled.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            item_scaled.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))

            # Classify mode symmetry irrep
            irrep_label = "-"
            if row < len(analysis.displacements):
                disp = analysis.displacements[row]
                irrep_label = classify_vibrational_mode_irrep(symbols, self.coords, disp, sym_res)
            item_irrep = QtWidgets.QTableWidgetItem(irrep_label)
            item_irrep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_irrep.setForeground(QtGui.QBrush(QtGui.QColor("#f59e0b" if irrep_label != "-" else "#64748b")))
            item_irrep.setFont(QtGui.QFont("Monospace", 9, QtGui.QFont.Weight.Bold))

            if freq < -5.0:
                status_text = "Imaginary / Transition State"
                color_status = "#f87171"  # red
            elif abs(freq) <= 15.0:
                status_text = "Translation / Rotation"
                color_status = "#94a3b8"  # slate
            else:
                status_text = "Genuine Vibration"
                color_status = "#4ade80"  # green

            item_status = QtWidgets.QTableWidgetItem(status_text)
            item_status.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_status.setForeground(QtGui.QBrush(QtGui.QColor(color_status)))

            self.table_vibrations.setItem(row, 0, item_idx)
            self.table_vibrations.setItem(row, 1, item_freq)
            self.table_vibrations.setItem(row, 2, item_scaled)
            self.table_vibrations.setItem(row, 3, item_irrep)
            self.table_vibrations.setItem(row, 4, item_status)

        # Switch to Vibrational Modes tab
        idx = self.tabs.indexOf(getattr(self, "tab_vibrations", None))
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
        else:
            self.tabs.setCurrentIndex(3)

        # Select first genuine vibrational mode (skip first 6 translations/rotations if present)
        first_vib_row = min(6, max(0, analysis.n_modes - 1))
        for r, f in enumerate(analysis.frequencies):
            if f > 20.0:
                first_vib_row = r
                break
        self.table_vibrations.selectRow(first_vib_row)

        # Update synthetic IR / Raman spectrum widget
        if hasattr(self, "spectrum_widget"):
            intensities = getattr(analysis, "ir_intensities", None)
            raman = getattr(analysis, "raman_activities", None)
            method = getattr(self, "_last_qm_method", "")
            basis = getattr(self, "_last_qm_basis", "")
            self.spectrum_widget.set_vibrational_data(
                analysis.frequencies,
                intensities,
                method=method,
                basis=basis,
                raman_activities=raman,
            )
            self.spectrum_widget.set_active_mode(first_vib_row)

        self.statusBar().showMessage(
            f"Calculated {analysis.n_modes} normal modes ({analysis.source}). Select a mode to simulate.", 5000
        )

    def _on_spectrum_scale_changed(self, scale_fac: float):
        """Update the Scaled Frequency column in table_vibrations when NIST scale factor changes."""
        if not self.vibrational_analysis:
            return
        for row, freq in enumerate(self.vibrational_analysis.frequencies):
            scaled_freq = freq * scale_fac
            it = self.table_vibrations.item(row, 2)
            if it:
                it.setText(f"{scaled_freq:.2f}")
            else:
                new_it = QtWidgets.QTableWidgetItem(f"{scaled_freq:.2f}")
                new_it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                new_it.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                self.table_vibrations.setItem(row, 2, new_it)

    def _on_spectrum_mode_clicked(self, mode_idx: int):
        """Called when user clicks on an IR spectrum peak in the matplotlib plot."""
        if hasattr(self, "table_vibrations") and 0 <= mode_idx < self.table_vibrations.rowCount():
            self.table_vibrations.selectRow(mode_idx)

    def _on_vibration_mode_selected(self):
        """Called when a mode row is selected in the vibrations table."""
        if not self.vibrational_analysis:
            return

        selected = self.table_vibrations.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        if row < 0 or row >= self.vibrational_analysis.n_modes:
            return

        freq = self.vibrational_analysis.frequencies[row]
        disp = self.vibrational_analysis.displacements[row]

        if hasattr(self, "spectrum_widget"):
            self.spectrum_widget.set_active_mode(row)

        self.btn_play_vibration.setEnabled(True)
        self.lbl_vib_info.setText(
            f"Mode #{row + 1}: Frequency = {freq:.2f} cm⁻¹ (Max atom displacement: {np.max(np.linalg.norm(disp, axis=1)):.3f})"
        )

        amp = self.slider_vib_amp.value() / 100.0
        # If already vibrating, seamlessly update mode; otherwise set mode so static 3D vectors are displayed
        if self.viewer.vibrating:
            self.viewer.start_vibration(row, freq, disp, amplitude=amp)
            self.btn_play_vibration.setText("⏸ Pause Simulation")
            self.btn_play_vibration.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
            self.btn_stop_vibration.setEnabled(True)
        else:
            self.viewer.set_vibration_mode(row, freq, disp, amplitude=amp)

    def _on_toggle_vibration_play(self):
        """Toggle simulation playback of selected vibrational mode."""
        if not self.vibrational_analysis:
            return

        selected = self.table_vibrations.selectedItems()
        if not selected:
            return
        row = selected[0].row()

        if self.viewer.vibrating:
            # Stop simulation
            self.viewer.stop_vibration()
            self.btn_play_vibration.setText("▶ Start Simulation")
            self.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
            self.btn_stop_vibration.setEnabled(False)
            self.statusBar().showMessage("Paused vibrational mode simulation.", 2500)
        else:
            freq = self.vibrational_analysis.frequencies[row]
            disp = self.vibrational_analysis.displacements[row]
            amp = self.slider_vib_amp.value() / 100.0
            self.viewer.start_vibration(row, freq, disp, amplitude=amp)
            self.btn_play_vibration.setText("⏸ Pause Simulation")
            self.btn_play_vibration.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
            self.btn_stop_vibration.setEnabled(True)
            self.statusBar().showMessage(f"Simulating Mode #{row + 1}: {freq:.1f} cm⁻¹", 4000)

    def _on_stop_vibration(self):
        """Stop vibration and restore equilibrium structure."""
        self.viewer.stop_vibration()
        self.btn_play_vibration.setText("▶ Start Simulation")
        self.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        self.btn_stop_vibration.setEnabled(False)
        self.statusBar().showMessage("Stopped vibrational simulation. Structure at equilibrium.", 2500)

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

    def _on_adopt_vibrational_displacement(self, sign: float = 1.0):
        """
        GaussView-style 'Manual Displacement / Follow Imaginary Mode':
        Displaces the equilibrium molecular geometry along the currently selected normal mode
        by `sign * amplitude * disp` and adopts it as the new molecular coordinates (with full Undo support).
        """
        if not getattr(self, "vibrational_analysis", None) or self.viewer.vib_displacements is None:
            QtWidgets.QMessageBox.information(
                self, "Follow Normal Mode", "Please calculate or load vibrational modes and select a mode in the table first."
            )
            return

        disp = np.asarray(self.viewer.vib_displacements, dtype=np.float64)
        if len(disp) != len(self.zatoms):
            return

        amp = (self.slider_vib_amp.value() / 100.0) if hasattr(self, "slider_vib_amp") else 0.50
        mode_idx = (self.viewer.vib_mode_idx or 0) + 1
        freq = self.viewer.vib_frequency

        self._on_stop_vibration()
        self._push_undo_state()

        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        old_bo = dict(getattr(self.viewer, "bond_orders", {}))
        new_coords = np.array(self.coords, dtype=np.float64) + (float(sign) * amp) * disp
        new_zatoms = cartesian_to_zmatrix(
            symbols, new_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
        )
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        if old_bo:
            self.viewer.bond_orders = old_bo
            self.viewer.update()

        dir_str = "+" if sign >= 0 else "-"
        self.statusBar().showMessage(
            f"Adopted {dir_str}{amp:.2f}× displaced geometry along Mode #{mode_idx} ({freq:.1f} cm⁻¹). Ready to re-optimize!",
            5000,
        )

    def _on_export_vibration_gif(self):
        """Export one full oscillation cycle of the active vibrational mode as an animated GIF."""
        if not self.vibrational_analysis or self.viewer.vib_displacements is None:
            QtWidgets.QMessageBox.information(
                self, "Export Vibrational Animation", "Please calculate or load vibrational modes and select a mode first."
            )
            return

        mode_num = (self.viewer.vib_mode_idx or 0) + 1
        default_name = f"vibration_mode_{mode_num}.gif"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Vibrational Animation (GIF)", default_name, "Animated GIF (*.gif);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"

        was_vibrating = self.viewer.vibrating
        saved_phase = self.viewer.vib_phase
        self.viewer.vib_timer.stop()
        self.viewer.vibrating = True

        n_frames = 24
        frames_rgb = []
        try:
            for k in range(n_frames):
                self.viewer.vib_phase = (2.0 * math.pi * k) / n_frames
                self.viewer.repaint()
                if self.viewer.isValid() and self.viewer.context() is not None:
                    qimg = self.viewer.grabFramebuffer().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
                else:
                    w = max(400, self.viewer.width())
                    h = max(300, self.viewer.height())
                    qimg = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGBA8888)
                    qimg.fill(QtGui.QColor(30, 33, 41))
                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                ptr.setsize(h * w * 4)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
                frames_rgb.append(arr[:, :, :3])

            import matplotlib.animation as mpl_anim
            from matplotlib.figure import Figure
            h, w, _ = frames_rgb[0].shape
            fig = Figure(figsize=(w / 100.0, h / 100.0), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            im = ax.imshow(frames_rgb[0])

            def _update_frame(idx):
                im.set_data(frames_rgb[idx])
                return [im]

            ani = mpl_anim.FuncAnimation(fig, _update_frame, frames=len(frames_rgb), interval=50, blit=True)
            ani.save(path, writer="pillow", fps=20)
            self.statusBar().showMessage(f"Saved vibrational animation GIF to {os.path.basename(path)}", 5000)
        except Exception as e:
            log.exception("Failed to export vibration GIF")
            QtWidgets.QMessageBox.critical(self, "GIF Export Error", f"Could not export GIF:\n{e}")
        finally:
            self.viewer.vib_phase = saved_phase
            self.viewer.vibrating = was_vibrating
            if was_vibrating:
                self.viewer.vib_timer.start()
            self.viewer.update()

    def _populate_available_symmetries(
        self,
        avail: List[Tuple[str, PointGroupResult, str]],
        selected_symbol: Optional[str] = None,
    ):
        """Populate both the Symmetry tab and 3D viewport symmetry combo boxes with Higher / Detected / Lower groups."""
        self._available_symmetries = list(avail)
        detected_sym = next((sym for sym, _, tag in avail if tag == "detected"), avail[0][0] if avail else "C1")
        target_sym = selected_symbol or detected_sym

        role_labels = {
            "higher": "⬆ Higher (approx. supergroup)",
            "detected": "★ Detected (exact)",
            "lower": "⬇ Lower (subgroup)",
        }

        for combo_attr in ("combo_sym_group", "combo_view_sym_group"):
            combo = getattr(self, combo_attr, None)
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

        self._update_sym_step_buttons()

    def _update_sym_step_buttons(self):
        """Enable or disable the Higher / Lower step buttons based on current index in _available_symmetries."""
        avail = getattr(self, "_available_symmetries", [])
        combo = getattr(self, "combo_sym_group", None)
        if not avail or combo is None or combo.count() == 0:
            if hasattr(self, "btn_sym_higher"):
                self.btn_sym_higher.setEnabled(False)
            if hasattr(self, "btn_sym_lower"):
                self.btn_sym_lower.setEnabled(False)
            return
        idx = combo.currentIndex()
        # avail is sorted from highest order (index 0) to lowest order C1 (index len-1)
        if hasattr(self, "btn_sym_higher"):
            self.btn_sym_higher.setEnabled(idx > 0)
        if hasattr(self, "btn_sym_lower"):
            self.btn_sym_lower.setEnabled(0 <= idx < len(avail) - 1)

    def select_symmetry_group(self, group_symbol: str, enforce_if_higher: bool = True) -> Optional[PointGroupResult]:
        """
        Programmatically or interactively select a higher, detected, or lower point group symmetry `group_symbol`.
        - Updates `_current_sym_res` and the Symmetry tab operations table.
        - If `group_symbol` is a higher (approximate supergroup) symmetry and `enforce_if_higher` is True,
          projects the coordinates into that higher point group so the higher symmetry holds exactly.
        - If `symmetry_imposed` is enabled, updates `imposed_sym_res` to the selected group.
        """
        if len(self.zatoms) == 0:
            return None
        tol = self.spin_sym_tol.value() if hasattr(self, "spin_sym_tol") else 0.15
        symbols = [a.symbol for a in self.zatoms]
        avail = getattr(self, "_available_symmetries", None)
        if not avail:
            avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
            self._available_symmetries = avail

        chosen_entry = next((item for item in avail if item[0] == group_symbol), None)
        if chosen_entry is None:
            # Recompute available symmetries in case geometry changed
            avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
            self._available_symmetries = avail
            chosen_entry = next((item for item in avail if item[0] == group_symbol), None)
        if chosen_entry is None:
            return getattr(self, "_current_sym_res", None)

        sym_name, chosen_res, role_tag = chosen_entry

        # If a higher (supergroup) symmetry was chosen and enforce_if_higher is requested,
        # symmetrize coordinates into that higher group's exact representation subspace!
        if role_tag == "higher" and enforce_if_higher and chosen_res.order > 1:
            self._push_undo_state()
            user_bonds = self._current_explicit_bonds()
            old_bo = dict(getattr(self.viewer, "bond_orders", {}))
            sym_coords, enforced_res = enforce_point_group_symmetry(
                symbols, self.coords, target_group=sym_name, tolerance=tol, reference_result=chosen_res
            )
            new_zatoms = cartesian_to_zmatrix(
                symbols, sym_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
            )
            prev_imposed = getattr(self, "symmetry_imposed", False)
            self.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
            if old_bo:
                self.viewer.bond_orders = old_bo
                self.viewer.update()
            avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
            chosen_entry = next((item for item in avail if item[0] == sym_name), (sym_name, enforced_res, "detected"))
            sym_name, chosen_res, role_tag = chosen_entry
            if prev_imposed:
                self.symmetry_imposed = True

        self._current_sym_res = chosen_res
        self._populate_available_symmetries(avail, selected_symbol=sym_name)
        self._refresh_symmetry_ui(chosen_res, role_tag=role_tag)

        if getattr(self, "symmetry_imposed", False):
            self.set_impose_symmetry(True, chosen_res)
        else:
            # Automatically enable Impose Symmetry when user selects a specific symmetry group
            self.set_impose_symmetry(True, chosen_res)
            orbits = get_symmetry_orbits(symbols, self.coords, chosen_res)
            self.statusBar().showMessage(
                f"Selected & Imposed {sym_name} symmetry (Order {chosen_res.order}, {role_tag}, {len(orbits)} atom orbit(s)). Edits replicate across equivalent atoms.",
                5000,
            )
        return chosen_res

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
        self.btn_sym_align.setEnabled(True)
        self.btn_sym_enforce.setEnabled(True)
        self.btn_sym_matrices.setEnabled(True)
        self.btn_calc_reps.setEnabled(True)

        avail = getattr(self, "_available_symmetries", [])
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

        self.lbl_sym_group.setText(f"Point Group: {res.symbol}{role_badge}")
        symbols = [a.symbol for a in self.zatoms]
        orbits = get_symmetry_orbits(symbols, self.coords, res) if len(symbols) == len(self.coords) else []
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
        self.lbl_sym_details.setText("<br>".join(details_lines))

        prev_row = self.table_symmetry_ops.currentRow()
        self.table_symmetry_ops.setRowCount(res.order)
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

            self.table_symmetry_ops.setItem(idx, 0, it_num)
            self.table_symmetry_ops.setItem(idx, 1, it_op)
            self.table_symmetry_ops.setItem(idx, 2, it_desc)

        if res.order > 0:
            self.table_symmetry_ops.selectRow(prev_row if 0 <= prev_row < res.order else 0)
        self._update_sym_step_buttons()

    def _on_detect_symmetry(self, checked_or_switch=True, *, switch_tab: bool = True, preferred_symbol: Optional[str] = None):
        """Analyze molecular 3D coordinates, discover higher/detected/lower symmetries, and update UI."""
        # Note: QPushButton.clicked / QToolButton.clicked / QAction.triggered pass `checked: bool = False`
        # as the first positional argument for non-checkable buttons!
        if switch_tab is True and checked_or_switch is False:
            # Determine if caller explicitly passed switch_tab=False positionally or if Qt signal passed checked=False:
            # Callers wanting silent refresh pass keyword `switch_tab=False`.
            pass
        if len(self.zatoms) == 0:
            if switch_tab:
                QtWidgets.QMessageBox.information(
                    self, "Point Group Symmetry", "Please add or load a molecule to analyze symmetry."
                )
            return

        tol = self.spin_sym_tol.value()
        if switch_tab:
            self.statusBar().showMessage(f"Detecting point group symmetry (tol={tol:.2f} Å)...", 3000)

        if switch_tab:
            QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        try:
            symbols = [a.symbol for a in self.zatoms]
            avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
            detected_entry = next((item for item in avail if item[2] == "detected"), avail[0])
            chosen_entry = detected_entry
            if preferred_symbol:
                chosen_entry = next((item for item in avail if item[0] == preferred_symbol), detected_entry)

            res = chosen_entry[1]
            role_tag = chosen_entry[2]
            self._current_sym_res = res
            self._populate_available_symmetries(avail, selected_symbol=chosen_entry[0])
            self._refresh_symmetry_ui(res, role_tag=role_tag)

            if switch_tab:
                # Switch to Symmetry tab
                idx = self.tabs.indexOf(getattr(self, "tab_symmetry", None))
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)
                else:
                    self.tabs.setCurrentIndex(4)
                n_higher = sum(1 for _, _, t in avail if t == "higher")
                n_lower = sum(1 for _, _, t in avail if t == "lower")
                self.statusBar().showMessage(
                    f"Detected: {detected_entry[0]} (Order {detected_entry[1].order}) | "
                    f"Available: {n_higher} higher, {n_lower} lower subgroup(s)",
                    5500,
                )
        except Exception as e:
            log.exception("Unhandled error in GUI action")
            if switch_tab:
                QtWidgets.QMessageBox.critical(self, "Symmetry Detection Error", f"Failed to detect symmetry:\n{str(e)}")
        finally:
            if switch_tab:
                QtWidgets.QApplication.restoreOverrideCursor()

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
        if len(self.zatoms) == 0:
            return

        res = getattr(self, "_current_sym_res", None)
        symbols = [a.symbol for a in self.zatoms]
        self._push_undo_state()
        user_bonds = self._current_explicit_bonds()
        old_bo = dict(getattr(self.viewer, "bond_orders", {}))
        aligned_coords, _ = align_molecule_to_symmetry(symbols, self.coords, res)

        # Update Z-matrix and keep exact aligned Cartesian coordinates
        new_zatoms = cartesian_to_zmatrix(
            symbols, aligned_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
        )
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=aligned_coords)
        if old_bo:
            self.viewer.bond_orders = old_bo
        self.viewer.align_camera_standard()

        # Re-detect symmetry with aligned orientation
        self._on_detect_symmetry()
        self.statusBar().showMessage(
            f"Aligned geometry to standard symmetry orientation (highest C_n along Z, displayed in screen plane).", 5000
        )

    def _on_enforce_symmetry(self):
        """Project coordinates into exact point group subspace if tolerance is not fulfilled."""
        if len(self.zatoms) == 0:
            return

        res = getattr(self, "_current_sym_res", None)
        target = res.symbol if res else None
        tol = self.spin_sym_tol.value()

        self._push_undo_state()
        symbols = [a.symbol for a in self.zatoms]
        user_bonds = self._current_explicit_bonds()
        old_bo = dict(getattr(self.viewer, "bond_orders", {}))
        sym_coords, new_res = enforce_point_group_symmetry(
            symbols, self.coords, target_group=target, tolerance=tol, reference_result=res
        )

        # Update Z-matrix and keep exact symmetrized coordinates
        new_zatoms = cartesian_to_zmatrix(
            symbols, sym_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
        )
        self.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
        if old_bo:
            self.viewer.bond_orders = old_bo
            self.viewer.update()

        # Re-detect symmetry with symmetrized geometry while keeping user's chosen target group
        self._on_detect_symmetry(switch_tab=False, preferred_symbol=target or new_res.symbol)
        self.set_impose_symmetry(True, self._current_sym_res)
        self.statusBar().showMessage(
            f"Enforced & Imposed {new_res.symbol} symmetry (Order {new_res.order}). Subsequent edits will preserve {new_res.symbol} symmetry!", 5500
        )

    def _on_toggle_impose_symmetry(self, checked: bool):
        """Toggle GaussView-style Impose / Constrain Symmetry mode."""
        self.set_impose_symmetry(checked)

    def set_impose_symmetry(self, enabled: bool, sym_res: Optional[PointGroupResult] = None):
        """
        Enable or disable symmetry-constrained editing (Impose Symmetry).
        When enabled, automatically detects the current point group (if not provided)
        and locks its symmetry operations so subsequent structural edits (replacing atoms,
        attaching ligands like CO, modifying bonds) replicate across all symmetry-equivalent atoms.
        """
        if enabled and len(self.zatoms) > 0:
            tol = self.spin_sym_tol.value() if hasattr(self, "spin_sym_tol") else 0.15
            symbols = [a.symbol for a in self.zatoms]
            avail = getattr(self, "_available_symmetries", None)
            if not avail:
                avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
                self._available_symmetries = avail
            if sym_res is None:
                sym_res = getattr(self, "_current_sym_res", None)
                if sym_res is None:
                    detected_entry = next((item for item in avail if item[2] == "detected"), avail[0])
                    sym_res = detected_entry[1]
            # Check if sym_res is a higher (approximate supergroup) symmetry: if so, snap coordinates to it!
            matching_entry = next((item for item in avail if item[0] == sym_res.symbol), None)
            if matching_entry is not None and matching_entry[2] == "higher" and sym_res.order > 1:
                user_bonds = self._current_explicit_bonds()
                old_bo = dict(getattr(self.viewer, "bond_orders", {}))
                sym_coords, sym_res = enforce_point_group_symmetry(
                    symbols, self.coords, target_group=sym_res.symbol, tolerance=tol, reference_result=sym_res
                )
                new_zatoms = cartesian_to_zmatrix(
                    symbols, sym_coords, existing_zatoms=self.zatoms, user_bonds=user_bonds
                )
                self.symmetry_imposed = False  # avoid recursive re-detection during set_molecule_from_zatoms
                self.set_molecule_from_zatoms(new_zatoms, exact_coords=sym_coords)
                if old_bo:
                    self.viewer.bond_orders = old_bo
                    self.viewer.update()
                avail = find_available_symmetries(symbols, self.coords, base_tolerance=tol)
                self._available_symmetries = avail

            self._current_sym_res = sym_res
            self.imposed_sym_res = sym_res
            self.symmetry_imposed = True
            self._populate_available_symmetries(avail, selected_symbol=sym_res.symbol)
            self._refresh_symmetry_ui(sym_res)
        else:
            self.symmetry_imposed = False
            self.imposed_sym_res = None

        grp_label = self.imposed_sym_res.symbol if (self.symmetry_imposed and self.imposed_sym_res) else ""
        if hasattr(self, "chk_impose_symmetry"):
            self.chk_impose_symmetry.blockSignals(True)
            self.chk_impose_symmetry.setChecked(self.symmetry_imposed)
            self.chk_impose_symmetry.setText(
                f"🔒 Impose Symmetry ({grp_label})" if grp_label else "🔒 Impose Symmetry"
            )
            self.chk_impose_symmetry.blockSignals(False)

        if hasattr(self, "btn_impose_symmetry"):
            self.btn_impose_symmetry.blockSignals(True)
            self.btn_impose_symmetry.setChecked(self.symmetry_imposed)
            if self.symmetry_imposed and grp_label:
                self.btn_impose_symmetry.setText(f"🔒 Impose Symmetry: {grp_label}")
                self.btn_impose_symmetry.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 5px;")
            else:
                self.btn_impose_symmetry.setText("🔒 Impose Symmetry: OFF")
                self.btn_impose_symmetry.setStyleSheet("background-color: #475569; color: #ffffff; font-weight: bold; padding: 5px;")
            self.btn_impose_symmetry.blockSignals(False)

        if self.symmetry_imposed and self.imposed_sym_res is not None:
            self.statusBar().showMessage(
                f"Imposed {self.imposed_sym_res.symbol} symmetry (Order {self.imposed_sym_res.order}): edits will replicate across symmetry-equivalent atoms.",
                5000,
            )
        elif not enabled:
            self.statusBar().showMessage("Symmetry constraint (Impose Symmetry) turned OFF.", 3500)

    def _on_show_symmetry_matrices(self):
        """Produce and display 3x3 symmetry operation matrices with interactive selector and view-all capability."""
        res = getattr(self, "_current_sym_res", None)
        if not res or not res.operations:
            QtWidgets.QMessageBox.information(self, "Symmetry Matrices", "Please detect symmetry first.")
            return

        is_white = getattr(self.viewer, "bg_color_mode", "") == "white"
        tbl_bg = "#ffffff" if is_white else "#0f172a"
        tbl_fg = "#0f172a" if is_white else "#f8fafc"
        border_col = "#cbd5e1" if is_white else "#334155"

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(f"3×3 Cartesian Symmetry Matrices — Point Group {res.symbol}")
        dlg.resize(520, 460)
        d_layout = QtWidgets.QVBoxLayout(dlg)
        d_layout.setSpacing(10)

        # Header info
        hdr = QtWidgets.QLabel(
            f"<b>Point Group:</b> <span style='color:#7c3aed; font-size:14px;'>{res.symbol}</span> "
            f"&nbsp;|&nbsp; <b>Order (h):</b> {res.order} operations"
        )
        d_layout.addWidget(hdr)

        # Combo selector for operations + Show All option
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Select Operation:"))
        combo_ops = QtWidgets.QComboBox()
        for idx, op in enumerate(res.operations):
            details = f" along [{op.axis[0]:.2f}, {op.axis[1]:.2f}, {op.axis[2]:.2f}]" if op.axis is not None else ""
            combo_ops.addItem(f"#{idx+1}: {op.label} ({op.op_type}{details})", idx)
        combo_ops.addItem("📑 Display All Operations Side-by-Side", -1)

        # Pre-select table's currently selected row (or 0)
        curr_row = self.table_symmetry_ops.currentRow()
        if 0 <= curr_row < len(res.operations):
            combo_ops.setCurrentIndex(curr_row)
        else:
            combo_ops.setCurrentIndex(0)

        sel_row.addWidget(combo_ops, stretch=1)
        d_layout.addLayout(sel_row)

        # Scroll area containing rendered matrix / matrices
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_w = QtWidgets.QWidget()
        content_layout = QtWidgets.QVBoxLayout(content_w)
        content_layout.setContentsMargins(8, 8, 8, 8)
        content_layout.setSpacing(12)
        scroll.setWidget(content_w)
        d_layout.addWidget(scroll, stretch=1)

        def _format_matrix_html(idx: int, op) -> str:
            M = op.matrix
            tr = float(np.trace(M))
            det = float(np.linalg.det(M))
            details = f" along [{op.axis[0]:.2f}, {op.axis[1]:.2f}, {op.axis[2]:.2f}]" if op.axis is not None else ""
            h = [
                f"<div style='margin-bottom:12px; padding:8px; border:1px solid {border_col}; border-radius:6px; background:{tbl_bg};'>",
                f"<div style='font-size:13px; font-weight:bold; color:#0284c7; margin-bottom:6px;'>"
                f"Operation #{idx+1}: {op.label} <span style='color:#64748b; font-size:11px;'>({op.op_type}{details})</span></div>",
                f"<table border='1' cellpadding='5' cellspacing='0' style='border-collapse:collapse; font-family:monospace; background:{tbl_bg}; color:{tbl_fg}; font-size:12px; margin-bottom:6px;'>",
            ]
            for r in range(3):
                h.append("<tr>")
                for c in range(3):
                    val = M[r, c]
                    # Clean display for clean zeroes or ±1
                    if abs(val) < 1e-7:
                        val_str = "0.000000"
                    elif abs(val - 1.0) < 1e-7:
                        val_str = "1.000000"
                    elif abs(val + 1.0) < 1e-7:
                        val_str = "-1.000000"
                    else:
                        val_str = f"{val:10.6f}"
                    h.append(f"<td align='right' style='padding:6px 12px;'>{val_str}</td>")
                h.append("</tr>")
            h.append("</table>")
            h.append(f"<div style='color:#94a3b8; font-size:11px;'>Trace = {tr:+.3f} &nbsp;|&nbsp; Determinant = {det:+.3f}</div>")
            h.append("</div>")
            return "".join(h)

        def _update_view():
            # Clear previous contents
            while content_layout.count():
                item = content_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()

            sel_idx = combo_ops.currentData()
            if sel_idx == -1:
                # Show all matrices
                html_chunks = [_format_matrix_html(i, op) for i, op in enumerate(res.operations)]
                lbl = QtWidgets.QLabel("".join(html_chunks))
                lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
                lbl.setWordWrap(True)
                content_layout.addWidget(lbl)
            else:
                op = res.operations[sel_idx]
                lbl = QtWidgets.QLabel(_format_matrix_html(sel_idx, op))
                lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
                content_layout.addWidget(lbl)
            content_layout.addStretch()

        combo_ops.currentIndexChanged.connect(_update_view)
        _update_view()

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dlg.close)
        d_layout.addWidget(btn_box)
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
        if not self.zatoms:
            QtWidgets.QMessageBox.information(
                self, "Quick ORCA", "Please create or open a molecule before running an ORCA calculation."
            )
            return

        from .dialog_quick_orca import QuickOrcaDialog
        dlg = QuickOrcaDialog(self, self.zatoms, self.coords)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            saved_path = getattr(dlg, "saved_input_path", None)
            if saved_path and os.path.isfile(saved_path):
                self.launch_calculation_job(saved_path, runner_type="orca")

    def _on_open_job_monitor(self):
        """Open or show the Job Monitor Dialog."""
        from .calc_runner import JobMonitorDialog, BrorsenRunner
        monitor = getattr(self, "_job_monitor_dlg", None)
        if monitor is None:
            monitor = JobMonitorDialog(self, runner=BrorsenRunner(self))
            monitor.live_step_received.connect(self._on_live_calc_step)
            monitor.calculation_completed.connect(self._on_live_calc_completed)
            self._job_monitor_dlg = monitor
        monitor.show()
        monitor.raise_()
        monitor.activateWindow()

    def _on_configure_qm_engines(self):
        """Open the QM Engines Detection & Route Configuration Dialog."""
        from .qm_detect import QMEnginesStatusDialog
        QMEnginesStatusDialog.show_dialog(self)

    def _on_open_remote_file_manager(self):
        """Open graphical remote file transfer dialog."""
        from .calc_runner.remote_ssh import get_all_remote_hosts, RemoteHostConfig
        from .calc_runner.dialog_file_transfer import RemoteFileManagerDialog

        hosts = get_all_remote_hosts()
        # Fall back to saved QSettings profile if available
        settings = QtCore.QSettings("AgyMolden", "RemoteProfiles")
        last_prof = settings.value("last_verified_profile", "")
        host_cfg = None

        if last_prof:
            host_cfg = RemoteHostConfig(
                hostname=settings.value(f"{last_prof}/hostname", ""),
                user=settings.value(f"{last_prof}/user", "andres"),
                identity_file=settings.value(f"{last_prof}/identity_file", ""),
                remote_workdir=settings.value(f"{last_prof}/remote_workdir", "~/agymolden_scratch"),
                remote_python=settings.value(f"{last_prof}/remote_python", "python3"),
                remote_wrapper=settings.value(f"{last_prof}/remote_wrapper", "newBRORSEN.v04.09.py"),
            )
        elif hosts:
            host_cfg = hosts[0]

        if not host_cfg or not host_cfg.hostname:
            # Open launch dialog to configure host
            QtWidgets.QMessageBox.information(
                self, "Remote File Transfer",
                "Please configure and test your remote host under the 'Remote Server (SSH / VPN)' tab first."
            )
            from .calc_runner.dialog_launch import BrorsenLaunchDialog
            dlg = BrorsenLaunchDialog(self, input_file="")
            dlg.tab_targets.setCurrentIndex(1)
            dlg.exec()
            return

        dlg = RemoteFileManagerDialog(self, host_cfg)
        dlg.exec()

    def launch_calculation_job(self, input_file: str, runner_type: str = "brorsen"):
        """Launch background QM calculation with live streaming output and 3D viewport updates."""
        from .calc_runner import JobMonitorDialog, BrorsenRunner, OrcaRunner, GenericQMRunner, BrorsenLaunchDialog

        # If it's a brorsen calculation, prompt for Conda environment and wrapper FIRST
        py_bin = None
        wrapper_path = None
        extra_flags = []
        live_3d = True
        remote_host = None

        if runner_type == "brorsen":
            launch_dlg = BrorsenLaunchDialog(self, input_file)
            if launch_dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return  # User cancelled launch
            py_bin, wrapper_path, extra_flags, live_3d, remote_host = launch_dlg.get_launch_configuration()

        def _make_runner(parent_obj):
            if runner_type == "orca":
                return OrcaRunner(parent_obj)
            elif runner_type in ("gaussian", "pyscf", "openlowdin", "xtb", "qchem"):
                return GenericQMRunner(code_key=runner_type, parent=parent_obj)
            elif remote_host is not None:
                from .calc_runner.remote_ssh import RemoteSshRunner
                r = RemoteSshRunner(parent_obj)
                r.configure_remote(host_cfg=remote_host, remote_script=wrapper_path, extra_flags=extra_flags)
                return r
            else:
                return BrorsenRunner(parent_obj)

        monitor = getattr(self, "_job_monitor_dlg", None)
        if monitor is None:
            runner = _make_runner(self)
            monitor = JobMonitorDialog(self, runner=runner)
            monitor.live_step_received.connect(self._on_live_calc_step)
            monitor.calculation_completed.connect(self._on_live_calc_completed)
            self._job_monitor_dlg = monitor
        else:
            monitor.runner = _make_runner(monitor)
            monitor._connect_runner_signals()

        if runner_type == "brorsen" and isinstance(monitor.runner, BrorsenRunner):
            if py_bin and wrapper_path:
                monitor.runner.configure(python_bin=py_bin, wrapper_path=wrapper_path, extra_flags=extra_flags)
                # Synchronize monitor UI dropdowns
                idx_py = monitor.combo_conda.findData(py_bin)
                if idx_py >= 0:
                    monitor.combo_conda.setCurrentIndex(idx_py)
                idx_wr = monitor.combo_wrapper.findData(wrapper_path)
                if idx_wr >= 0:
                    monitor.combo_wrapper.setCurrentIndex(idx_wr)

        monitor.chk_live_3d.setChecked(live_3d)
        monitor.run_calculation(input_file, runner_type=runner_type)

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

    def _on_live_calc_completed(self, input_file: str):
        """Handle completion of live job: locate and open log / trajectory / molden / cube files."""
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
            except Exception:
                pass

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
                    os.path.join(s_dir, f"{c_stem}_hf_positrons.molden"),
                    os.path.join(s_dir, f"{c_stem}_positrons.molden"),
                    os.path.join(s_dir, f"{c_stem}_hf_positron.molden"),
                    os.path.join(s_dir, f"{c_stem}_positron.molden"),
                    os.path.join(s_dir, f"{c_stem}.POSITRON.molden"),
                    os.path.join(s_dir, f"{c_stem}.E-ALPHA.molden"),
                    os.path.join(s_dir, f"{c_stem}.gbw"),
                ])
        molden_cand.append(os.path.join(base_dir, "orbitals.molden"))
        self._related_molden_files = list(artifact_moldens)
        loaded_molden = False
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
                    self.load_orbitals_from_molden(m, auto_render=True)
                    if hasattr(self, "_volumetric_dlg") and self._volumetric_dlg and self._volumetric_dlg.isVisible():
                        self._volumetric_dlg.load_molden_path(m)
                    loaded_molden = True
                except Exception as e:
                    log.warning(f"Could not load orbitals from {m}: {e}")
                break

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
        self.lbl_mo_status.setText(
            f"<b>{display_name}</b>: {len(data.orbitals)} MOs loaded. "
            + (f"HOMO is MO #{homo_idx+1}{gap_str}" if homo_idx is not None else "")
        )

        path = source_path or getattr(self, "_active_molden_path", "")

        # Auto-detect companion positron/electron Molden files in the same directory
        self._companion_positron_molden_path = None
        base_dir = os.path.dirname(path)
        fname = os.path.basename(path)
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
                except Exception:
                    pass

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
        if not selected or not hasattr(self, "_active_molden_path") or not self._active_molden_path:
            return
        row = self.table_orbitals.row(selected[0])
        if hasattr(self, "mo_ladder_widget"):
            self.mo_ladder_widget.set_selected_orbital(row)
        self._render_mo_orbital(row)

    def _render_mo_orbital(self, mo_index: int):
        from .volumetric import evaluate_molden_orbital_grid, generate_orbital_surfaces
        if not hasattr(self, "_active_molden_path") or not os.path.isfile(self._active_molden_path):
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
        if hasattr(self, "_active_molden_data") and self._active_molden_data:
            homo_idx = 0
            for idx, mo in enumerate(self._active_molden_data.orbitals):
                if mo.occupancy > 0.05:
                    homo_idx = idx
            idx = self.tabs.indexOf(getattr(self, "tab_orbitals", None))
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
            else:
                self.tabs.setCurrentIndex(5)
            self.table_orbitals.selectRow(homo_idx)
            self._render_mo_orbital(homo_idx)
        else:
            self._on_open_volumetric_dialog()

    def _on_quick_view_lumo(self):
        if hasattr(self, "_active_molden_data") and self._active_molden_data:
            lumo_idx = 0
            for idx, mo in enumerate(self._active_molden_data.orbitals):
                if mo.occupancy <= 0.05:
                    lumo_idx = idx
                    break
            idx = self.tabs.indexOf(getattr(self, "tab_orbitals", None))
            if idx >= 0:
                self.tabs.setCurrentIndex(idx)
            else:
                self.tabs.setCurrentIndex(5)
            self.table_orbitals.selectRow(lumo_idx)
            self._render_mo_orbital(lumo_idx)
        else:
            self._on_open_volumetric_dialog()

    def _on_clear_orbital_surface(self):
        self.viewer.active_isosurfaces = []
        self.viewer.clear_surface_rect = None
        self.viewer.clear_surface_hovered = False
        self.viewer.update()
        self.statusBar().showMessage("Cleared 3D orbital / isosurface.", 2500)

    def _on_overlay_positron_orbital(self):
        """Evaluate the companion positron ground-state orbital and overlay it (gold/green) onto the 3D viewport."""
        from .volumetric import evaluate_molden_orbital_grid, generate_orbital_surfaces
        pos_path = getattr(self, "_companion_positron_molden_path", None)
        if not pos_path or not os.path.isfile(pos_path):
            pos_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Select Positron Molden File", "", "Molden Files (*.molden *.molden.input);;All Files (*)"
            )
            if not pos_path:
                return
            self._companion_positron_molden_path = pos_path

        isovalue = self.spin_quick_iso.value() if hasattr(self, "spin_quick_iso") else 0.030

        def _compute():
            vol = evaluate_molden_orbital_grid(pos_path, orbital_index=0, nx=40, ny=40, nz=40)
            if not vol:
                return [], isovalue
            max_abs = float(np.max(np.abs(vol.grid_data)))
            eff_iso = isovalue
            if eff_iso >= max_abs * 0.92 and max_abs > 1e-5:
                eff_iso = round(max_abs * 0.45, 4)
            return generate_orbital_surfaces(
                vol,
                isovalue=eff_iso,
                pos_color=(0.95, 0.72, 0.12),  # Gold
                neg_color=(0.16, 0.78, 0.44),  # Emerald green
            ), eff_iso

        def _on_done(res_tuple):
            pos_meshes, eff_iso = res_tuple
            if pos_meshes:
                existing = list(getattr(self.viewer, "active_isosurfaces", []) or [])
                self.viewer.active_isosurfaces = existing + pos_meshes
                self.viewer.update()
                self.statusBar().showMessage(
                    f"Overlaid Positron MO #1 from {os.path.basename(pos_path)} (Isovalue={eff_iso:.4f}, Gold/Green)", 5000
                )

        run_in_background(
            self,
            "Rendering Positron Orbital",
            f"Computing 3D Positron orbital grid from {os.path.basename(pos_path)}...",
            _compute,
            _on_done,
        )

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
        """Return (pos_color, neg_color) based on combo_mo_phase_colors selection."""
        scheme = self.combo_mo_phase_colors.currentData() if hasattr(self, "combo_mo_phase_colors") else "blue_red"
        if scheme == "cyan_orange":
            return (0.06, 0.75, 0.88), (0.98, 0.45, 0.08)  # Cyan (+), Orange (-)
        elif scheme == "green_purple":
            return (0.16, 0.80, 0.44), (0.65, 0.24, 0.85)  # Green (+), Purple (-)
        elif scheme == "gold_emerald":
            return (0.95, 0.72, 0.12), (0.16, 0.78, 0.44)  # Gold (+), Emerald (-)
        else:
            return (0.22, 0.58, 0.94), (0.95, 0.30, 0.25)  # Classic Blue (+), Coral Red (-)

    def _remesh_current_mo(self, iso: float):
        """Fast Marching Cubes remeshing using cached 3D grid data if available."""
        vol = getattr(self, "_current_rendered_mo_vol", None)
        if vol is not None:
            from .volumetric import generate_orbital_surfaces
            molden_path = getattr(self, "_active_molden_path", "")
            is_pos_file = "positron" in os.path.basename(molden_path).lower()
            if is_pos_file:
                pos_c, neg_c = (0.95, 0.72, 0.12), (0.16, 0.78, 0.44)
            else:
                pos_c, neg_c = self._get_mo_phase_colors()

            meshes = generate_orbital_surfaces(vol, isovalue=iso, pos_color=pos_c, neg_color=neg_c)

            self.viewer.active_isosurfaces = meshes
            self.viewer.update()
            mo_idx = getattr(self, "_current_rendered_mo_index", 0)
            self.statusBar().showMessage(f"Updated MO #{mo_idx+1} Isovalue: {iso:.4f}", 2500)
        else:
            # Fall back to full re-evaluation
            selected = self.table_orbitals.selectedItems()
            if selected:
                row = self.table_orbitals.row(selected[0])
                self._render_mo_orbital(row)

    def _on_quick_iso_changed(self, val: float):
        self._on_quick_spin_iso_changed(val)

    def _on_quick_opacity_changed(self, val: int):
        self.viewer.surface_opacity = val / 100.0
        self.viewer.update()

    def _on_open_orbital_snapshot_gallery(self):
        """Open the Orbital Snapshot & Gallery dialog to batch capture all or selected orbitals."""
        molden_path = getattr(self, "_active_molden_path", None)
        molden_data = getattr(self, "_active_molden_data", None)

        if not molden_path or not molden_data or not getattr(molden_data, "orbitals", None):
            # Prompt user to select a Molden file
            molden_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Molden File for Orbital Snapshots",
                getattr(self, "_last_opened_dir", ""),
                "Molden Files (*.molden *.molden.input);;All Files (*)",
            )
            if not molden_path:
                return
            from .volumetric import parse_molden_file
            try:
                molden_data = parse_molden_file(molden_path)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error Reading Molden File", str(e))
                return

        # Determine homo index
        homo_idx = None
        for i, o in enumerate(molden_data.orbitals):
            if getattr(o, "occupancy", 0.0) > 0.05:
                homo_idx = i

        from .dialog_orbital_gallery import OrbitalSnapshotDialog
        dlg = OrbitalSnapshotDialog(self, molden_path, molden_data.orbitals, homo_idx=homo_idx)
        dlg.exec()



