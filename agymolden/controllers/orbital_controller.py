"""
agymolden.controllers.orbital_controller
========================================
Dedicated controller managing 3D molecular orbital visualization, HOMO/LUMO quick views,
open-shell spin density isosurfaces, ESP-mapped electron density surfaces, PySCF Pipek-Mezey
and Foster-Boys orbital localization, positron orbital overlays, fast Marching Cubes remeshing,
and batch orbital snapshot galleries.
"""

import os
import logging
from typing import Optional, Any, Tuple
import numpy as np
from PyQt6 import QtWidgets

from ..volumetric import (
    compute_molden_spin_density_grid,
    generate_orbital_surfaces,
    compute_molden_esp_mapped_density,
    localize_molden_orbitals,
    evaluate_molden_orbital_grid,
    parse_molden_file,
)
from ..dialog_service import dialog_service
from ..workers import run_in_background

log = logging.getLogger(__name__)


class OrbitalController:
    """Encapsulates molecular orbital and volumetric surface rendering logic."""

    def __init__(self, main_window: Any):
        self.mw = main_window

    def _populate_semiempirical_mos_if_needed(self) -> bool:
        """Compute and populate built-in Semiempirical Tight-Binding (GFN0-xTB / SCC-TB) MOs when no Molden file is loaded."""
        if not getattr(self.mw, "zatoms", None) or len(self.mw.zatoms) == 0:
            return False
        from ..semiempirical import compute_semiempirical_properties
        from ..volumetric import MoldenFileData, MoldenMO

        symbols = [a.symbol for a in self.mw.zatoms]
        coords = np.array(self.mw.coords, dtype=np.float64, copy=True)
        props = compute_semiempirical_properties(symbols, coords)
        self.mw._semiempirical_qm_props = props

        eigvals_ev = props.get("eigvals_ev", [])
        eigvecs = props.get("eigvecs")
        occ = props.get("occupancies", [])
        if eigvecs is None or len(eigvals_ev) == 0:
            return False

        orbitals = []
        for idx, e_ev in enumerate(eigvals_ev):
            e_au = float(e_ev) / 27.211386
            occ_val = float(occ[idx]) if idx < len(occ) else 0.0
            coefs = {mu + 1: float(eigvecs[mu, idx]) for mu in range(eigvecs.shape[0])}
            orbitals.append(
                MoldenMO(sym=f"A{idx + 1}", energy=e_au, spin="Alpha", occupancy=occ_val, coefficients=coefs)
            )

        data = MoldenFileData(
            title="Built-in Semiempirical Tight-Binding Wavefunction",
            symbols=symbols,
            coords=coords,
            orbitals=orbitals,
        )
        self.mw._active_molden_path = None
        self.mw.set_orbitals_from_data(
            data,
            auto_render=False,
            source_path="",
            source_label="Built-in Semiempirical QM (GFN0-xTB)",
        )
        return True

    def quick_view_homo(self) -> None:
        """1-Click quick view of the Highest Occupied Molecular Orbital (HOMO)."""
        if not (hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data):
            self._populate_semiempirical_mos_if_needed()

        if hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data:
            homo_idx = 0
            for idx, mo in enumerate(self.mw._active_molden_data.orbitals):
                if mo.occupancy > 0.05:
                    homo_idx = idx
            idx = self.mw.tabs.indexOf(getattr(self.mw, "tab_orbitals", None))
            if idx >= 0:
                self.mw.tabs.setCurrentIndex(idx)
            else:
                self.mw.tabs.setCurrentIndex(5)
            self.mw.table_orbitals.selectRow(homo_idx)
            self.mw._render_mo_orbital(homo_idx)
        else:
            self.mw._on_open_volumetric_dialog()

    def quick_view_lumo(self) -> None:
        """1-Click quick view of the Lowest Unoccupied Molecular Orbital (LUMO)."""
        if not (hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data):
            self._populate_semiempirical_mos_if_needed()

        if hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data:
            lumo_idx = 0
            for idx, mo in enumerate(self.mw._active_molden_data.orbitals):
                if mo.occupancy <= 0.05:
                    lumo_idx = idx
                    break
            idx = self.mw.tabs.indexOf(getattr(self.mw, "tab_orbitals", None))
            if idx >= 0:
                self.mw.tabs.setCurrentIndex(idx)
            else:
                self.mw.tabs.setCurrentIndex(5)
            self.mw.table_orbitals.selectRow(lumo_idx)
            self.mw._render_mo_orbital(lumo_idx)
        else:
            self.mw._on_open_volumetric_dialog()

    def quick_view_homo_lumo_duality(self) -> None:
        """1-Click Dual Frontier Overlay: Simultaneously render HOMO (blue/gold) and LUMO (purple/emerald)."""
        if not (hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data):
            self._populate_semiempirical_mos_if_needed()

        if not (hasattr(self.mw, "_active_molden_data") and self.mw._active_molden_data):
            self.mw._on_open_volumetric_dialog()
            return

        homo_idx = 0
        lumo_idx = 1
        for idx, mo in enumerate(self.mw._active_molden_data.orbitals):
            if mo.occupancy > 0.05:
                homo_idx = idx
        if homo_idx + 1 < len(self.mw._active_molden_data.orbitals):
            lumo_idx = homo_idx + 1

        symbols = [a.symbol for a in self.mw.zatoms] if self.mw.zatoms else self.mw.symbols
        coords = self.mw.coords
        qm_props = getattr(self.mw, "_semiempirical_qm_props", None)

        from ..volumetric import evaluate_semiempirical_orbital_grid, generate_orbital_surfaces
        n_pts = 36
        vol_h = evaluate_semiempirical_orbital_grid(symbols, coords, orbital_index=homo_idx, nx=n_pts, ny=n_pts, nz=n_pts, qm_props=qm_props)
        vol_l = evaluate_semiempirical_orbital_grid(symbols, coords, orbital_index=lumo_idx, nx=n_pts, ny=n_pts, nz=n_pts, qm_props=qm_props)

        if not vol_h or not vol_l:
            self.mw.statusBar().showMessage("Could not evaluate HOMO/LUMO duality grids.", 3000)
            return

        iso_h = float(np.max(np.abs(vol_h.grid_data))) * 0.45
        iso_l = float(np.max(np.abs(vol_l.grid_data))) * 0.45

        # HOMO colors: Cyan & Amber
        mesh_h = generate_orbital_surfaces(vol_h, isovalue=max(1e-4, iso_h), pos_color=(0.14, 0.70, 0.95), neg_color=(0.95, 0.65, 0.15))
        # LUMO colors: Violet & Emerald
        mesh_l = generate_orbital_surfaces(vol_l, isovalue=max(1e-4, iso_l), pos_color=(0.65, 0.25, 0.95), neg_color=(0.15, 0.85, 0.45))

        combined_meshes = list(mesh_h) + list(mesh_l)
        if hasattr(self.mw, "viewer"):
            self.mw.viewer.active_isosurfaces = combined_meshes
            self.mw.viewer.surface_opacity = 0.75
            self.mw.viewer.update()
            self.mw.statusBar().showMessage(f"Dual Frontier View: HOMO #{homo_idx+1} (Cyan/Amber) + LUMO #{lumo_idx+1} (Violet/Emerald)", 5000)

    def quick_view_spin_density(self) -> None:
        """1-Click Open-Shell Spin Density (rho_alpha - rho_beta) evaluation and 3D rendering."""
        molden_path = getattr(self.mw, "_active_molden_path", None)
        if not molden_path or not os.path.isfile(molden_path):
            self.mw._on_open_volumetric_dialog()
            return

        isovalue = self.mw.spin_quick_iso.value() if hasattr(self.mw, "spin_quick_iso") else 0.005
        opacity = (self.mw.slider_quick_opacity.value() / 100.0) if hasattr(self.mw, "slider_quick_opacity") else 0.80
        n_pts = int(self.mw.combo_quick_grid.currentData() or 40) if hasattr(self.mw, "combo_quick_grid") else 40
        margin_b = float(self.mw.spin_quick_margin.value()) if hasattr(self.mw, "spin_quick_margin") else 3.0

        self.mw.statusBar().showMessage(f"Computing 3D spin density Δρ = ρ_α − ρ_β ({n_pts}³ grid)...", 3000)

        def _compute():
            vol = compute_molden_spin_density_grid(molden_path, nx=n_pts, ny=n_pts, nz=n_pts, margin_bohr=margin_b)
            if not vol:
                return [], isovalue, None
            max_abs = float(np.max(np.abs(vol.grid_data)))
            eff_iso = min(isovalue, 0.01) if isovalue >= 0.025 else isovalue
            if eff_iso >= max_abs and max_abs > 1e-6:
                eff_iso = round(max_abs * 0.35, 5)
            if max_abs <= 1e-6:
                return [], eff_iso, vol
            pos_c, neg_c = (0.14, 0.65, 0.95), (0.95, 0.55, 0.12)
            meshes = generate_orbital_surfaces(vol, isovalue=max(1e-5, eff_iso), pos_color=pos_c, neg_color=neg_c)
            return meshes, eff_iso, vol

        def _on_done(res_tuple):
            meshes, eff_iso, vol = res_tuple
            if vol is None:
                self.mw.statusBar().showMessage("Could not evaluate spin density from the active Molden file.", 4000)
                return
            max_abs = float(np.max(np.abs(vol.grid_data)))
            if max_abs <= 1e-6 or not meshes:
                self.mw.statusBar().showMessage(
                    "Spin density Δρ(r) is zero everywhere (closed-shell singlet wavefunction).", 5000
                )
                return
            self.mw._current_rendered_mo_vol = vol
            self.mw.viewer.active_isosurfaces = meshes
            self.mw.viewer.surface_opacity = opacity
            self.mw.viewer.update()
            self.mw.statusBar().showMessage(
                f"Rendered Spin Density Δρ = ρ_α − ρ_β (Isovalue={eff_iso:.4f}, max |Δρ|={max_abs:.4f} e/a₀³)", 5000
            )

        def _on_error(e):
            log.warning(f"Error computing spin density: {e}")

        run_in_background(
            self.mw,
            "Computing Spin Density",
            "Evaluating open-shell spin density Δρ = ρ_α − ρ_β on 3D grid...",
            _compute,
            _on_done,
            _on_error,
        )

    def quick_view_esp_map(self) -> None:
        """1-Click Electrostatic Potential (ESP) mapped onto Total Electron Density isosurface."""
        molden_path = getattr(self.mw, "_active_molden_path", None)
        if not molden_path:
            last_res = getattr(self.mw, "_last_calculation_results", {})
            if last_res.get("molden_file") and os.path.isfile(last_res["molden_file"]):
                molden_path = last_res["molden_file"]
                self.mw._active_molden_path = molden_path
        if not molden_path or not os.path.isfile(molden_path):
            if getattr(self.mw, "zatoms", None) and len(self.mw.zatoms) > 0:
                from ..semiempirical import compute_semiempirical_esp_mapped_density
                symbols = [a.symbol for a in self.mw.zatoms]
                coords = np.array(self.mw.coords, dtype=np.float64, copy=True)
                opacity = (self.mw.slider_quick_opacity.value() / 100.0) if hasattr(self.mw, "slider_quick_opacity") else 0.85
                mesh = compute_semiempirical_esp_mapped_density(symbols, coords)
                if mesh is not None:
                    self.mw.viewer.active_isosurfaces = [mesh]
                    self.mw.viewer.surface_opacity = opacity
                    self.mw.viewer.update()
                    e_rng = getattr(mesh, "esp_range", (-0.05, 0.05))
                    self.mw.statusBar().showMessage(
                        f"Rendered Instant Built-in SCC-TB ESP Surface (V_ESP ∈ [{e_rng[0]:+.3f}, {e_rng[1]:+.3f}] a.u.)",
                        5000,
                    )
                    return
            self.mw._on_open_volumetric_dialog()
            return

        opacity = (self.mw.slider_quick_opacity.value() / 100.0) if hasattr(self.mw, "slider_quick_opacity") else 0.85
        n_pts = int(self.mw.combo_quick_grid.currentData() or 40) if hasattr(self.mw, "combo_quick_grid") else 40
        margin_b = max(3.5, float(self.mw.spin_quick_margin.value())) if hasattr(self.mw, "spin_quick_margin") else 3.5

        self.mw.statusBar().showMessage(f"Computing 3D ESP-mapped electron density surface ({n_pts}³ grid)...", 3000)

        def _compute():
            return compute_molden_esp_mapped_density(
                molden_path,
                density_isovalue=0.002,
                nx=n_pts,
                ny=n_pts,
                nz=n_pts,
                margin_bohr=margin_b,
            )

        def _on_done(mesh):
            if mesh is None:
                self.mw.statusBar().showMessage("Could not evaluate ESP-mapped density from the active Molden file.", 4000)
                return
            self.mw.viewer.active_isosurfaces = [mesh]
            self.mw.viewer.surface_opacity = opacity
            self.mw.viewer.update()
            e_rng = getattr(mesh, "esp_range", (-0.05, 0.05))
            self.mw.statusBar().showMessage(
                f"Rendered ESP-Mapped Electron Density (ρ = 0.002 e/a₀³, V_ESP ∈ [{e_rng[0]:+.3f}, {e_rng[1]:+.3f}] a.u.)",
                5000,
            )

        def _on_error(e):
            log.warning(f"Error computing ESP-mapped density: {e}")

        run_in_background(
            self.mw,
            "Computing ESP Map",
            "Evaluating Total Electron Density & Electrostatic Potential V_ESP(r)...",
            _compute,
            _on_done,
            _on_error,
        )

    def localize_orbitals(self, method: str = "pipek-mezey") -> None:
        """1-Click PySCF orbital localization (Pipek-Mezey or Foster-Boys) from the active Molden file."""
        molden_path = getattr(self.mw, "_active_molden_path", None)
        if not molden_path or not os.path.isfile(molden_path):
            self.mw._on_open_volumetric_dialog()
            return

        if "_localized_" not in os.path.basename(molden_path):
            self.mw._canonical_molden_path = molden_path
        src_path = getattr(self.mw, "_canonical_molden_path", None) or molden_path

        label = "Foster-Boys" if "boys" in method.lower() else "Pipek-Mezey"
        self.mw.statusBar().showMessage(f"Localizing occupied orbitals via PySCF {label}...", 3000)

        def _compute():
            return localize_molden_orbitals(src_path, method=method, localize_virtuals=False)

        def _on_done(loc_path):
            if not loc_path or not os.path.isfile(loc_path):
                self.mw.statusBar().showMessage(f"Could not localize orbitals from {os.path.basename(src_path)}.", 4000)
                return
            self.mw.load_orbitals_from_molden(loc_path, auto_render=True)
            self.mw.statusBar().showMessage(
                f"🧲 Loaded {label} Localized Molecular Orbitals (LMOs) from {os.path.basename(loc_path)}",
                5000,
            )

        def _on_error(e):
            log.warning(f"Error localizing orbitals ({label}): {e}")
            self.mw.statusBar().showMessage(f"Orbital localization error: {e}", 5000)

        run_in_background(
            self.mw,
            f"Localizing Orbitals ({label})",
            f"Running PySCF {label} unitary orbital localization...",
            _compute,
            _on_done,
            _on_error,
        )

    def restore_canonical_orbitals(self) -> None:
        """Restore original canonical delocalized MOs after viewing Localized Molecular Orbitals."""
        canon_path = getattr(self.mw, "_canonical_molden_path", None)
        if canon_path and os.path.isfile(canon_path):
            self.mw.load_orbitals_from_molden(canon_path, auto_render=True)
            self.mw.statusBar().showMessage(
                f"↩️ Restored canonical delocalized MOs from {os.path.basename(canon_path)}", 4000
            )
        else:
            self.mw.statusBar().showMessage("Canonical MOs are already active.", 3000)

    def clear_orbital_surface(self) -> None:
        """Clear active 3D orbital / isosurface from the viewport."""
        self.mw.viewer.active_isosurfaces = []
        self.mw.viewer.clear_surface_rect = None
        self.mw.viewer.clear_surface_hovered = False
        self.mw.viewer.update()
        self.mw.statusBar().showMessage("Cleared 3D orbital / isosurface.", 2500)

    def overlay_positron_orbital(self) -> None:
        """Evaluate companion positron ground-state orbital and overlay it (gold/green) onto 3D view."""
        pos_path = getattr(self.mw, "_companion_positron_molden_path", None)
        if not pos_path or not os.path.isfile(pos_path):
            pos_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.mw, "Select Positron Molden File", "", "Molden Files (*.molden *.molden.input);;All Files (*)"
            )
            if not pos_path:
                return
            self.mw._companion_positron_molden_path = pos_path

        isovalue = self.mw.spin_quick_iso.value() if hasattr(self.mw, "spin_quick_iso") else 0.030

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
                pos_color=(0.95, 0.72, 0.12),
                neg_color=(0.16, 0.78, 0.44),
            ), eff_iso

        def _on_done(res_tuple):
            pos_meshes, eff_iso = res_tuple
            if pos_meshes:
                existing = list(getattr(self.mw.viewer, "active_isosurfaces", []) or [])
                self.mw.viewer.active_isosurfaces = existing + pos_meshes
                self.mw.viewer.update()
                self.mw.statusBar().showMessage(
                    f"Overlaid Positron MO #1 from {os.path.basename(pos_path)} (Isovalue={eff_iso:.4f}, Gold/Green)", 5000
                )

        run_in_background(
            self.mw,
            "Rendering Positron Orbital",
            f"Computing 3D Positron orbital grid from {os.path.basename(pos_path)}...",
            _compute,
            _on_done,
            lambda e: log.warning(f"Error computing positron orbital: {e}"),
        )

    def get_mo_phase_colors(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Return (pos_color, neg_color) based on combo_mo_phase_colors selection."""
        scheme = self.mw.combo_mo_phase_colors.currentData() if hasattr(self.mw, "combo_mo_phase_colors") else "blue_red"
        if scheme == "cyan_orange":
            return (0.06, 0.75, 0.88), (0.98, 0.45, 0.08)
        elif scheme == "green_purple":
            return (0.16, 0.80, 0.44), (0.65, 0.24, 0.85)
        elif scheme == "gold_emerald":
            return (0.95, 0.72, 0.12), (0.16, 0.78, 0.44)
        else:
            return (0.22, 0.58, 0.94), (0.95, 0.30, 0.25)

    def remesh_current_mo(self, iso: float) -> None:
        """Fast Marching Cubes remeshing using cached 3D grid data if available."""
        vol = getattr(self.mw, "_current_rendered_mo_vol", None)
        if vol is not None:
            molden_path = getattr(self.mw, "_active_molden_path", "")
            is_pos_file = "positron" in os.path.basename(molden_path).lower()
            if is_pos_file:
                pos_c, neg_c = (0.95, 0.72, 0.12), (0.16, 0.78, 0.44)
            else:
                pos_c, neg_c = self.get_mo_phase_colors()

            meshes = generate_orbital_surfaces(vol, isovalue=iso, pos_color=pos_c, neg_color=neg_c)
            self.mw.viewer.active_isosurfaces = meshes
            self.mw.viewer.update()
            mo_idx = getattr(self.mw, "_current_rendered_mo_index", 0)
            self.mw.statusBar().showMessage(f"Updated MO #{mo_idx+1} Isovalue: {iso:.4f}", 2500)
        else:
            selected = self.mw.table_orbitals.selectedItems()
            if selected:
                row = self.mw.table_orbitals.row(selected[0])
                self.mw._render_mo_orbital(row)

    def open_orbital_snapshot_gallery(self) -> None:
        """Open Orbital Snapshot & Gallery dialog to batch capture all or selected orbitals."""
        molden_path = getattr(self.mw, "_active_molden_path", None)
        molden_data = getattr(self.mw, "_active_molden_data", None)

        if not molden_path or not molden_data or not getattr(molden_data, "orbitals", None):
            molden_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.mw,
                "Select Molden File for Orbital Snapshots",
                getattr(self.mw, "_last_opened_dir", ""),
                "Molden Files (*.molden *.molden.input);;All Files (*)",
            )
            if not molden_path:
                return
            try:
                molden_data = parse_molden_file(molden_path)
            except Exception as e:
                dialog_service.error(self.mw, "Error Reading Molden File", str(e), exc=e)
                return

        homo_idx = None
        for i, o in enumerate(molden_data.orbitals):
            if getattr(o, "occupancy", 0.0) > 0.05:
                homo_idx = i

        from ..dialog_orbital_gallery import OrbitalSnapshotDialog
        dlg = OrbitalSnapshotDialog(self.mw, molden_path, molden_data.orbitals, homo_idx=homo_idx)
        dlg.exec()
