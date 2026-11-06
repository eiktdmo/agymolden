"""
agymolden.controllers.import_controller
=======================================
Dedicated controller encapsulating molecular import, log ingestion,
and multi-format translation:
- QM Log file parsing (Gaussian, ORCA, Q-Chem, xTB via cclib / native parsers)
- Cartesian XYZ parsing (single and multi-frame CREST / trajectory geometries)
- Crystallographic CIF and PDB coordinate loading
- Companion file discovery (.molden, .cube, positron densities)
"""

import os
import logging
from typing import Any, Optional, List, Tuple
import numpy as np
from PyQt6 import QtWidgets

from ..qm_log_parser import parse_qm_logfile, can_parse_with_cclib
from ..forcefield import detect_connectivity, detect_bond_orders
from ..zmatrix import cartesian_to_zmatrix, parse_xyz
from ..topology import norm_bond
from ..dialog_service import dialog_service

log = logging.getLogger(__name__)


class ImportController:
    """Manages file imports, trajectory parsing, and companion artifact synchronization."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def import_qm_log(self) -> None:
        """Import quantum chemistry output file (.log, .out) from Gaussian, ORCA, Q-Chem, or xTB."""
        if not can_parse_with_cclib():
            QtWidgets.QMessageBox.warning(
                self.mw, "cclib Required", "cclib is required to parse QM output files. Please install cclib."
            )
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mw,
            "Import QM Log / Output File",
            self.mw._last_dir(),
            "QM Output Files (*.log *.out *.output);;All Files (*)",
        )
        if path:
            self.load_qm_log_path(path)

    def load_qm_log_path(self, path: str) -> None:
        """Parse QM log file and update geometry, vibrational frequencies, and orbitals."""
        try:
            res = parse_qm_logfile(path)
            self.mw._push_undo_state()
            self.mw.variables = {}

            # Store optimization trajectory
            self.mw._last_qm_res = res
            self.mw.trajectory_coords = res.trajectory_coords
            self.mw.trajectory_energies = res.trajectory_energies
            self.mw.trajectory_symbols = res.symbols

            final_coords = res.trajectory_coords[-1]
            symbols = res.symbols

            detected_bonds = detect_connectivity(symbols, final_coords, strict=False)
            detected_bo = detect_bond_orders(symbols, final_coords, detected_bonds)

            zatoms = cartesian_to_zmatrix(symbols, final_coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.mw.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.mw.strict_topology = True

            self.mw.set_molecule_from_zatoms(zatoms, exact_coords=final_coords)
            self.mw.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
            if res.charges is not None and len(res.charges) == len(symbols):
                self.mw.viewer.custom_charges = res.charges
                self.mw.viewer.show_charges = True
                if hasattr(self.mw, "chk_charges"):
                    self.mw.chk_charges.blockSignals(True)
                    self.mw.chk_charges.setChecked(True)
                    self.mw.chk_charges.blockSignals(False)
            self.mw.viewer.update()

            # Apply nuclear forces / gradients if present
            if getattr(res, "nuclear_forces", None) is not None:
                self.mw.viewer.nuclear_forces = res.nuclear_forces
                if hasattr(self.mw, "chk_forces"):
                    self.mw.chk_forces.blockSignals(True)
                    self.mw.chk_forces.setChecked(True)
                    self.mw.chk_forces.blockSignals(False)
                self.mw.viewer.show_forces = True

            # Enable Trajectory Player if multiple steps exist
            n_steps = len(res.trajectory_coords)
            if hasattr(self.mw, "grp_trajectory"):
                if n_steps > 1:
                    is_irc = getattr(res, "is_irc", False)
                    init_step = n_steps - 1
                    if is_irc:
                        self.mw.grp_trajectory.setTitle("⚛️ Intrinsic Reaction Coordinate (IRC Pathway Player)")
                        if hasattr(self.mw, "combo_traj_mode"):
                            idx_bounce = self.mw.combo_traj_mode.findData("bounce")
                            if idx_bounce >= 0:
                                self.mw.combo_traj_mode.setCurrentIndex(idx_bounce)
                        if hasattr(self.mw, "btn_irc_sweeper"):
                            self.mw.btn_irc_sweeper.setStyleSheet(
                                "background-color: #6366f1; color: #ffffff; padding: 2px 8px; font-weight: bold; border: 1.5px solid #a5b4fc;"
                            )
                        if getattr(res, "irc_pathway", None) is not None and hasattr(res.irc_pathway, "ts_index"):
                            init_step = getattr(res.irc_pathway, "ts_index", init_step)
                    else:
                        self.mw.grp_trajectory.setTitle("Trajectory Player (Optimization Steps / MD)")
                        if hasattr(self.mw, "btn_irc_sweeper"):
                            self.mw.btn_irc_sweeper.setStyleSheet(
                                "background-color: #6366f1; color: #ffffff; padding: 2px 8px; font-weight: bold;"
                            )
                    self.mw.grp_trajectory.setVisible(True)
                    self.mw.slider_trajectory.blockSignals(True)
                    self.mw.slider_trajectory.setRange(0, n_steps - 1)
                    self.mw.slider_trajectory.setValue(init_step)
                    self.mw.slider_trajectory.blockSignals(False)
                    self.mw.lbl_trajectory_step.setText(f"Step {init_step + 1}/{n_steps}")
                    if res.trajectory_energies and len(res.trajectory_energies) == n_steps:
                        e_val = res.trajectory_energies[init_step]
                        self.mw.lbl_trajectory_energy.setText(f"E = {e_val:.6f} eV")
                    else:
                        self.mw.lbl_trajectory_energy.setText("")
                else:
                    self.mw.grp_trajectory.setVisible(False)

            # Apply vibrational modes
            if res.vibrational_analysis and res.vibrational_analysis.n_modes > 0:
                self.mw._apply_vibrations((final_coords, res.vibrational_analysis))

            # Apply UV-Vis transitions
            if hasattr(res, "et_energies") and res.et_energies and hasattr(self.mw, "uvvis_widget"):
                self.mw.uvvis_widget.set_uv_vis_data(res.et_energies, res.et_oscs)
                if hasattr(self.mw, "tabs_spectra"):
                    self.mw.tabs_spectra.setCurrentIndex(1)

            # Apply NMR chemical shieldings
            if getattr(res, "nmr_shieldings", None) and hasattr(self.mw, "nmr_widget"):
                self.mw.nmr_widget.set_nmr_data(res.nmr_shieldings)
                if hasattr(self.mw, "tabs_spectra") and not (res.vibrational_analysis and res.vibrational_analysis.n_modes > 0):
                    self.mw.tabs_spectra.setCurrentIndex(2)

            # Apply Molecular Orbitals
            if getattr(res, "orbitals", None) and hasattr(self.mw, "set_orbitals_from_data"):
                from ..volumetric import MoldenFileData
                m_dummy = MoldenFileData(
                    title=f"{res.program_name} log orbitals",
                    symbols=symbols,
                    coords=final_coords,
                    orbitals=res.orbitals or [],
                )
                self.mw.set_orbitals_from_data(m_dummy, homo_idx=res.homo_idx, source_label=os.path.basename(path))
            else:
                q_dir = os.path.dirname(path)
                q_base = os.path.splitext(os.path.basename(path))[0]
                for cand_ext in ("_hf_electrons.molden", "_electrons.molden", ".molden", ".molden.input", ".e-.molden", ".E-.molden", ".E-ALPHA.molden", ".molf"):
                    cand_m = os.path.join(q_dir, q_base + cand_ext)
                    if os.path.isfile(cand_m):
                        try:
                            self.mw.load_orbitals_from_molden(cand_m, auto_render=False)
                            break
                        except Exception:
                            log.debug("Auto-loading companion molden failed", exc_info=True)

            if hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data:
                idx = self.mw.tabs.indexOf(getattr(self.mw, "tab_orbitals", None))
                if idx >= 0:
                    self.mw.tabs.setCurrentIndex(idx)

            self.mw._last_loaded_qm_log_path = path
            self.mw._remember_file(path)
            if hasattr(self.mw, "set_interaction_mode"):
                self.mw.set_interaction_mode("inspect")
            if hasattr(self.mw, "_update_qm_summary_banner"):
                self.mw._update_qm_summary_banner(res)
            self.mw.statusBar().showMessage(
                f"Loaded {res.program_name} log: {len(symbols)} atoms, {n_steps} step(s)"
                + (f", {res.vibrational_analysis.n_modes} frequency modes" if res.vibrational_analysis else ""),
                6000,
            )
            if not getattr(self.mw, "_suppress_companion_prompt", False):
                self.mw._check_and_prompt_companion_files(path)
        except Exception as e:
            dialog_service.error(self.mw, "Error Reading QM Output", str(e), exc=e)

    def import_xyz(self) -> None:
        """Prompt user and import molecular coordinates from Cartesian .xyz."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.mw, "Import XYZ File", self.mw._last_dir(), "XYZ Molecular Files (*.xyz);;All Files (*)"
        )
        if path:
            self.load_xyz_path(path)

    def load_xyz_path(self, path: str) -> None:
        """Parse Cartesian XYZ coordinates and detect connectivity / multi-step trajectory."""
        try:
            symbols, coords = parse_xyz(path)
            if len(symbols) == 0:
                QtWidgets.QMessageBox.warning(self.mw, "Empty File", "No atoms found in XYZ file.")
                return
            self.load_xyz_data(symbols, coords, source_label=os.path.basename(path))
            self.mw._remember_file(path)

            try:
                from ..zmatrix import parse_multi_xyz
                from ..crest_explorer import extract_energy_from_comment
                _, multi_frames = parse_multi_xyz(path, is_path=True)
                if len(multi_frames) > 1:
                    traj_energies = []
                    for c_comm, _ in multi_frames:
                        p_e = extract_energy_from_comment(c_comm)
                        traj_energies.append(p_e[0] if p_e is not None else 0.0)
                    self.mw.trajectory_coords = [f[1] for f in multi_frames]
                    self.mw.trajectory_symbols = symbols
                    self.mw.trajectory_energies = traj_energies
                    if hasattr(self.mw, "grp_trajectory"):
                        self.mw.grp_trajectory.setVisible(True)
                        self.mw.slider_trajectory.blockSignals(True)
                        self.mw.slider_trajectory.setRange(0, len(multi_frames) - 1)
                        self.mw.slider_trajectory.setValue(0)
                        self.mw.slider_trajectory.blockSignals(False)
                        self.mw.lbl_trajectory_step.setText(f"Step 1/{len(multi_frames)}")
            except Exception:
                pass

        except Exception as e:
            dialog_service.error(self.mw, "Error Reading XYZ", str(e), exc=e)

    def load_xyz_data(
        self,
        symbols: List[str],
        coords: np.ndarray,
        source_label: str = "Pasted XYZ",
    ) -> None:
        """Load Cartesian symbols and coordinates directly into the active molecule canvas."""
        if len(symbols) == 0:
            QtWidgets.QMessageBox.warning(self.mw, "Empty Data", "No atoms provided.")
            return

        self.mw._push_undo_state()
        self.mw.variables = {}

        detected_bonds = detect_connectivity(symbols, coords, strict=False)
        detected_bo = detect_bond_orders(symbols, coords, detected_bonds)

        zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=detected_bonds, strict_bonds=True)
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
        self.mw.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
        self.mw.strict_topology = True

        self.mw.set_molecule_from_zatoms(zatoms, exact_coords=coords)
        self.mw.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
        self.mw.viewer.update()
        self.mw.statusBar().showMessage(f"Loaded {source_label} ({len(symbols)} atoms)", 4000)
