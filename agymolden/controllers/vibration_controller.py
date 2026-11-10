"""
agymolden.controllers.vibration_controller
==========================================
Dedicated controller managing normal vibrational modes, harmonic frequency calculations,
interactive 3D normal-mode animations, IR/Raman spectrum peak linking, mode scrubbing,
geometry displacement along imaginary/transition modes, and animated GIF exports.
"""

import os
import math
from typing import Optional, Any
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from ..vibrations import compute_force_field_vibrations
from ..symmetry import classify_vibrational_mode_irrep
from ..zmatrix import cartesian_to_zmatrix
from ..dialog_service import dialog_service
from ..workers import run_in_background


class VibrationController:
    """Encapsulates all vibrational mode business logic and UI synchronization."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def calculate_vibrations(self) -> None:
        """Calculate normal vibrational frequencies and displacements using selected UFF/MMFF94s or Semiempirical QM engine."""
        if len(self.mw.zatoms) < 2:
            QtWidgets.QMessageBox.information(
                self.mw, "Vibrational Modes", "At least 2 atoms are required to calculate vibrational modes."
            )
            return

        method = self.mw._get_selected_opt_method()
        eff_method = "semiempirical" if method in ("semiempirical", "mlip") else method
        if method == "mlip":
            method_label = "MLIP / Semiempirical QM"
        elif method == "semiempirical":
            method_label = "Semiempirical QM (GFN0-xTB)"
        else:
            method_label = "Analytical UFF/MMFF94s"
        self.mw.statusBar().showMessage(f"Calculating vibrational Hessian and normal modes ({method_label})...", 4000)

        symbols = [a.symbol for a in self.mw.zatoms]
        user_bonds = self.mw._current_explicit_bonds()
        bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
        start_coords = np.array(self.mw.coords, copy=True)

        def compute():
            return compute_force_field_vibrations(
                symbols, start_coords, user_bonds=user_bonds, optimize_first=False,
                method=eff_method, user_bond_orders=bo
            )

        def done(result):
            try:
                self.apply_vibrations(result)
            except Exception as e:
                dialog_service.error(self.mw, "Vibration Calculation Error", f"Failed to compute vibrations:\n{e}", exc=e)

        def failed(exc):
            dialog_service.error(self.mw, "Vibration Calculation Error", f"Failed to compute vibrations:\n{exc}", exc=exc)

        run_in_background(
            self.mw,
            "Vibrational Modes",
            f"Calculating normal modes ({method_label})...",
            compute,
            done,
            failed,
        )

    def apply_vibrations(self, result: Any) -> None:
        """Apply computed or loaded vibrational analysis to UI tables and spectrum plot."""
        eq_coords, analysis = result
        self.mw.vibrational_analysis = analysis

        symbols = [a.symbol for a in self.mw.zatoms]
        sym_res = getattr(self.mw, "_current_sym_res", None)
        scale_fac = self.mw.spectrum_widget.scale_factor if hasattr(self.mw, "spectrum_widget") else 1.0

        self.mw.table_vibrations.setRowCount(analysis.n_modes)
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

            irrep_label = "-"
            if row < len(analysis.displacements):
                disp = analysis.displacements[row]
                irrep_label = classify_vibrational_mode_irrep(symbols, self.mw.coords, disp, sym_res)
            item_irrep = QtWidgets.QTableWidgetItem(irrep_label)
            item_irrep.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_irrep.setForeground(QtGui.QBrush(QtGui.QColor("#f59e0b" if irrep_label != "-" else "#64748b")))
            item_irrep.setFont(QtGui.QFont("Monospace", 9, QtGui.QFont.Weight.Bold))

            motion_label = ""
            if row < len(analysis.displacements):
                try:
                    from ..vibrations import assign_normal_mode_motion
                    motion_label = assign_normal_mode_motion(symbols, self.mw.coords, analysis.displacements[row], freq_cm1=freq)
                except Exception:
                    motion_label = ""

            if freq < -5.0:
                status_text = f"TS Imaginary ({motion_label})" if motion_label else "Imaginary / Transition State"
                color_status = "#f87171"
            elif abs(freq) <= 15.0:
                status_text = "Translation / Rotation"
                color_status = "#94a3b8"
            else:
                status_text = motion_label if motion_label else "Genuine Vibration"
                color_status = "#4ade80"

            item_status = QtWidgets.QTableWidgetItem(status_text)
            item_status.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_status.setForeground(QtGui.QBrush(QtGui.QColor(color_status)))

            self.mw.table_vibrations.setItem(row, 0, item_idx)
            self.mw.table_vibrations.setItem(row, 1, item_freq)
            self.mw.table_vibrations.setItem(row, 2, item_scaled)
            self.mw.table_vibrations.setItem(row, 3, item_irrep)
            self.mw.table_vibrations.setItem(row, 4, item_status)

        idx = self.mw.tabs.indexOf(getattr(self.mw, "tab_vibrations", None))
        if idx >= 0:
            self.mw.tabs.setCurrentIndex(idx)
        else:
            self.mw.tabs.setCurrentIndex(3)

        first_vib_row = min(6, max(0, analysis.n_modes - 1))
        for r, f in enumerate(analysis.frequencies):
            if f > 20.0:
                first_vib_row = r
                break
        self.mw.table_vibrations.selectRow(first_vib_row)

        if hasattr(self.mw, "spectrum_widget"):
            intensities = getattr(analysis, "ir_intensities", None)
            raman = getattr(analysis, "raman_activities", None)
            method = getattr(self.mw, "_last_qm_method", "")
            basis = getattr(self.mw, "_last_qm_basis", "")
            self.mw.spectrum_widget.set_vibrational_data(
                analysis.frequencies,
                intensities,
                method=method,
                basis=basis,
                raman_activities=raman,
            )
            self.mw.spectrum_widget.set_active_mode(first_vib_row)

        self.mw.statusBar().showMessage(
            f"Calculated {analysis.n_modes} normal modes ({analysis.source}). Select a mode to simulate.", 5000
        )

    def on_spectrum_scale_changed(self, scale_fac: float) -> None:
        """Update Scaled Frequency column in table_vibrations when scaling factor changes."""
        if not self.mw.vibrational_analysis:
            return
        for row, freq in enumerate(self.mw.vibrational_analysis.frequencies):
            scaled_freq = freq * scale_fac
            it = self.mw.table_vibrations.item(row, 2)
            if it:
                it.setText(f"{scaled_freq:.2f}")
            else:
                new_it = QtWidgets.QTableWidgetItem(f"{scaled_freq:.2f}")
                new_it.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                new_it.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
                self.mw.table_vibrations.setItem(row, 2, new_it)

    def on_spectrum_mode_clicked(self, mode_idx: int) -> None:
        """Triggered upon clicking an IR/Raman peak in the spectrum widget."""
        if hasattr(self.mw, "table_vibrations") and 0 <= mode_idx < self.mw.table_vibrations.rowCount():
            self.mw.table_vibrations.selectRow(mode_idx)
            if self.mw.vibrational_analysis and 0 <= mode_idx < self.mw.vibrational_analysis.n_modes:
                freq = self.mw.vibrational_analysis.frequencies[mode_idx]
                disp = self.mw.vibrational_analysis.displacements[mode_idx]
                amp = getattr(self.mw, "slider_vib_amp", None)
                amp_val = (amp.value() / 100.0) if amp is not None else 0.5
                self.mw.viewer.start_vibration(mode_idx, freq, disp, amplitude=amp_val)
                if hasattr(self.mw, "btn_play_vibration"):
                    self.mw.btn_play_vibration.setText("⏸ Pause Simulation")
                    self.mw.btn_play_vibration.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
                if hasattr(self.mw, "btn_stop_vibration"):
                    self.mw.btn_stop_vibration.setEnabled(True)
                ir_int = ""
                if self.mw.vibrational_analysis.ir_intensities and mode_idx < len(self.mw.vibrational_analysis.ir_intensities):
                    ir_int = f" | IR Int: {self.mw.vibrational_analysis.ir_intensities[mode_idx]:.1f} km/mol"
                self.mw.statusBar().showMessage(f"Animating Mode #{mode_idx + 1}: {freq:.1f} cm⁻¹{ir_int}", 4000)

    def on_vibration_mode_selected(self) -> None:
        """Handle normal mode selection in table_vibrations."""
        if not self.mw.vibrational_analysis:
            return
        selected = self.mw.table_vibrations.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        if row < 0 or row >= self.mw.vibrational_analysis.n_modes:
            return

        freq = self.mw.vibrational_analysis.frequencies[row]
        disp = self.mw.vibrational_analysis.displacements[row]

        if hasattr(self.mw, "spectrum_widget"):
            self.mw.spectrum_widget.set_active_mode(row)

        self.mw.btn_play_vibration.setEnabled(True)
        self.mw.lbl_vib_info.setText(
            f"Mode #{row + 1}: Frequency = {freq:.2f} cm⁻¹ (Max atom displacement: {np.max(np.linalg.norm(disp, axis=1)):.3f})"
        )

        amp = self.mw.slider_vib_amp.value() / 100.0
        if self.mw.viewer.vibrating:
            self.mw.viewer.start_vibration(row, freq, disp, amplitude=amp)
            self.mw.btn_play_vibration.setText("⏸ Pause Simulation")
            self.mw.btn_play_vibration.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
            self.mw.btn_stop_vibration.setEnabled(True)
        else:
            self.mw.viewer.set_vibration_mode(row, freq, disp, amplitude=amp)

    def toggle_vibration_play(self) -> None:
        """Toggle animation playback of active vibrational mode."""
        if not self.mw.vibrational_analysis:
            return

        selected = self.mw.table_vibrations.selectedItems()
        if not selected:
            return
        row = selected[0].row()

        if self.mw.viewer.vibrating:
            self.mw.viewer.stop_vibration()
            self.mw.btn_play_vibration.setText("▶ Start Simulation")
            self.mw.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
            self.mw.btn_stop_vibration.setEnabled(False)
            self.mw.statusBar().showMessage("Paused vibrational mode simulation.", 2500)
        else:
            freq = self.mw.vibrational_analysis.frequencies[row]
            disp = self.mw.vibrational_analysis.displacements[row]
            amp = self.mw.slider_vib_amp.value() / 100.0
            self.mw.viewer.start_vibration(row, freq, disp, amplitude=amp)
            self.mw.btn_play_vibration.setText("⏸ Pause Simulation")
            self.mw.btn_play_vibration.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold;")
            self.mw.btn_stop_vibration.setEnabled(True)
            self.mw.statusBar().showMessage(f"Simulating Mode #{row + 1}: {freq:.1f} cm⁻¹", 4000)

    def stop_vibration(self) -> None:
        """Stop vibration simulation and restore equilibrium molecular coordinates."""
        self.mw.viewer.stop_vibration()
        self.mw.btn_play_vibration.setText("▶ Start Simulation")
        self.mw.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")
        self.mw.btn_stop_vibration.setEnabled(False)
        self.mw.statusBar().showMessage("Stopped vibrational simulation. Structure at equilibrium.", 2500)

    def scrub_vibration(self, val: int) -> None:
        """Deform the molecule interactively along the selected normal mode coordinate."""
        frac = val / 100.0
        if hasattr(self.mw, "lbl_vib_scrub"):
            self.mw.lbl_vib_scrub.setText(f"{frac:+.2f}×")

        if not getattr(self.mw, "vibrational_analysis", None) or self.mw.viewer.vib_displacements is None:
            return

        if getattr(self.mw.viewer, "vibrating", False):
            self.mw.viewer.stop_vibration()
            if hasattr(self.mw, "btn_play_vibration"):
                self.mw.btn_play_vibration.setText("▶ Start Simulation")
                self.mw.btn_play_vibration.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold;")

        amp = (self.mw.slider_vib_amp.value() / 100.0) if hasattr(self.mw, "slider_vib_amp") else 0.50
        disp = np.asarray(self.mw.viewer.vib_displacements, dtype=np.float64)
        if len(disp) == len(self.mw.coords):
            self.mw.viewer.vib_phase = math.asin(np.clip(frac, -1.0, 1.0))
            self.mw.viewer.vib_amplitude = amp
            self.mw.viewer.update()

    def adopt_vibrational_displacement(self, sign: float = 1.0) -> None:
        """Adopt displaced geometry along normal mode as new equilibrium coordinates (Undo-supported)."""
        if not getattr(self.mw, "vibrational_analysis", None) or self.mw.viewer.vib_displacements is None:
            QtWidgets.QMessageBox.information(
                self.mw, "Follow Normal Mode", "Please calculate or load vibrational modes and select a mode in the table first."
            )
            return

        disp = np.asarray(self.mw.viewer.vib_displacements, dtype=np.float64)
        if len(disp) != len(self.mw.zatoms):
            return

        amp = (self.mw.slider_vib_amp.value() / 100.0) if hasattr(self.mw, "slider_vib_amp") else 0.50
        mode_idx = (self.mw.viewer.vib_mode_idx or 0) + 1
        freq = self.mw.viewer.vib_frequency

        self.stop_vibration()
        self.mw._push_undo_state()

        symbols = [a.symbol for a in self.mw.zatoms]
        user_bonds = self.mw._current_explicit_bonds()
        old_bo = dict(getattr(self.mw.viewer, "bond_orders", {}))
        new_coords = np.array(self.mw.coords, dtype=np.float64) + (float(sign) * amp) * disp
        new_zatoms = cartesian_to_zmatrix(
            symbols, new_coords, existing_zatoms=self.mw.zatoms, user_bonds=user_bonds
        )
        self.mw.set_molecule_from_zatoms(new_zatoms, exact_coords=new_coords)
        if old_bo:
            self.mw.viewer.bond_orders = old_bo
            self.mw.viewer.update()

        dir_str = "+" if sign >= 0 else "-"
        self.mw.statusBar().showMessage(
            f"Adopted {dir_str}{amp:.2f}× displaced geometry along Mode #{mode_idx} ({freq:.1f} cm⁻¹). Ready to re-optimize!",
            5000,
        )

    def export_vibration_gif(self) -> None:
        """Export one full oscillation cycle of the active normal mode as an animated GIF."""
        if not self.mw.vibrational_analysis or self.mw.viewer.vib_displacements is None:
            QtWidgets.QMessageBox.information(
                self.mw, "Export Vibrational Animation", "Please calculate or load vibrational modes and select a mode first."
            )
            return

        mode_num = (self.mw.viewer.vib_mode_idx or 0) + 1
        default_name = f"vibration_mode_{mode_num}.gif"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.mw, "Export Vibrational Animation (GIF)", default_name, "Animated GIF (*.gif);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".gif"):
            path += ".gif"

        was_vibrating = self.mw.viewer.vibrating
        saved_phase = self.mw.viewer.vib_phase
        self.mw.viewer.vib_timer.stop()
        self.mw.viewer.vibrating = True

        n_frames = 24
        frames_rgb = []
        try:
            for k in range(n_frames):
                self.mw.viewer.vib_phase = (2.0 * math.pi * k) / n_frames
                self.mw.viewer.repaint()
                if self.mw.viewer.isValid() and self.mw.viewer.context() is not None:
                    qimg = self.mw.viewer.grabFramebuffer().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
                else:
                    w = max(400, self.mw.viewer.width())
                    h = max(300, self.mw.viewer.height())
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
            ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
            ax.axis("off")
            im = ax.imshow(frames_rgb[0])

            def _update_frame(idx):
                im.set_data(frames_rgb[idx])
                return [im]

            ani = mpl_anim.FuncAnimation(fig, _update_frame, frames=len(frames_rgb), interval=50, blit=True)
            ani.save(path, writer="pillow", fps=20)
            self.mw.statusBar().showMessage(f"Saved vibrational animation GIF to {os.path.basename(path)}", 5000)
        except Exception as e:
            dialog_service.error(self.mw, "GIF Export Error", f"Could not export GIF:\n{e}", exc=e)
        finally:
            self.mw.viewer.vib_phase = saved_phase
            self.mw.viewer.vibrating = was_vibrating
            if was_vibrating:
                self.mw.viewer.vib_timer.start()
            self.mw.viewer.update()
