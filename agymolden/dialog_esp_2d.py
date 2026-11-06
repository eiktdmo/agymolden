"""
Interactive 2D Electrostatic Potential (ESP) Contour Map Dialog.

Provides:
- 2D contour slice plotting on XY, XZ, or YZ planes with adjustable slice offset
- Real-time adjustment of color palette, contour density, and color scale range
- Atom projections and labels overlaid on the 2D contour slice
- Export high-resolution publication-quality PNG / SVG / PDF figures
"""

import math
from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from .volumetric import BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, VolumetricData


class ESP2DContourDialog(QtWidgets.QDialog):
    """Dialog for interactive 2D slicing and contour plotting of Molecular Electrostatic Potential or Volumetric Grids."""

    def __init__(self, parent: QtWidgets.QWidget, molden_path: Optional[str] = None, vol_data: Optional[VolumetricData] = None):
        super().__init__(parent)
        self.setWindowTitle("2D Slicer & Contour Map (Gabedit / Multiwfn style)")
        self.resize(880, 680)
        self.molden_path = molden_path
        self.vol_data = vol_data

        self.mol = None
        self.dm = None
        self.current_fig = None
        if self.molden_path:
            try:
                self._load_wavefunction()
            except ImportError as exc:
                from .multiwfn_analysis import friendly_error_text
                if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("QT_QPA_PLATFORM") == "offscreen"):
                    QtWidgets.QMessageBox.warning(parent, "PySCF Required", friendly_error_text(exc))
                self.molden_path = None
        self._init_ui()
        self._recompute_and_plot()

    def _load_wavefunction(self):
        from .volumetric import _load_pyscf_molden

        res = _load_pyscf_molden(self.molden_path)
        self.mol = res[0]
        mo_coeff = np.asarray(res[2])
        mo_occ = np.asarray(res[3]) if len(res) > 3 and res[3] is not None else np.zeros(mo_coeff.shape[-1])
        if mo_coeff.ndim == 3:
            self.dm = np.sum([c @ np.diag(occ) @ c.T for c, occ in zip(mo_coeff, mo_occ)], axis=0)
        else:
            self.dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T

    def _init_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.setSpacing(12)

        # Left: Controls sidebar
        ctrl_panel = QtWidgets.QWidget()
        ctrl_panel.setFixedWidth(290)
        ctrl_layout = QtWidgets.QVBoxLayout(ctrl_panel)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(10)

        # Property Selector Group
        grp_prop = QtWidgets.QGroupBox("Property to Slice")
        prop_form = QtWidgets.QFormLayout(grp_prop)
        prop_form.setSpacing(8)

        self.combo_property = QtWidgets.QComboBox()
        self.combo_property.addItem("Electrostatic Potential (ESP)", "ESP")
        self.combo_property.addItem("Electron Density (ρ)", "RHO")
        self.combo_property.addItem("Laplacian of Density (∇²ρ)", "LAPL")
        self.combo_property.addItem("Reduced Density Gradient (RDG)", "RDG")
        self.combo_property.currentIndexChanged.connect(self._on_property_changed)
        prop_form.addRow("Function:", self.combo_property)
        ctrl_layout.addWidget(grp_prop)

        grp_plane = QtWidgets.QGroupBox("Slicing Plane & Orientation")
        p_form = QtWidgets.QFormLayout(grp_plane)
        p_form.setSpacing(8)

        self.combo_plane = QtWidgets.QComboBox()
        self.combo_plane.addItem("XZ Plane (Top/Ring View)", "XZ")
        self.combo_plane.addItem("XY Plane", "XY")
        self.combo_plane.addItem("YZ Plane (Side View)", "YZ")
        # Auto-detect best initial plane with largest coordinate spread
        if self.mol is not None:
            coords = self.mol.atom_coords() * BOHR_TO_ANGSTROM
        elif self.vol_data is not None and len(self.vol_data.atom_coords) > 0:
            coords = np.array(self.vol_data.atom_coords)
        else:
            coords = np.zeros((1, 3))
        stds = np.std(coords, axis=0) if len(coords) > 1 else np.zeros(3)
        if stds[1] < stds[0] and stds[1] < stds[2]:
            self.combo_plane.setCurrentIndex(0) # XZ
        elif stds[2] < stds[0] and stds[2] < stds[1]:
            self.combo_plane.setCurrentIndex(1) # XY
        else:
            self.combo_plane.setCurrentIndex(2) # YZ
        self.combo_plane.currentIndexChanged.connect(self._recompute_and_plot)
        p_form.addRow("Slice Plane:", self.combo_plane)

        self.spin_offset = QtWidgets.QDoubleSpinBox()
        self.spin_offset.setRange(-15.0, 15.0)
        self.spin_offset.setSingleStep(0.2)
        self.spin_offset.setValue(0.0)
        self.spin_offset.setSuffix(" Å")

        self.slider_offset = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_offset.setRange(-150, 150)
        self.slider_offset.setValue(0)
        self.slider_offset.valueChanged.connect(lambda v: self.spin_offset.setValue(v / 10.0))
        self.spin_offset.valueChanged.connect(lambda v: (self.slider_offset.setValue(int(round(v * 10.0))), self._recompute_and_plot()))
        
        offset_layout = QtWidgets.QHBoxLayout()
        offset_layout.addWidget(self.slider_offset, stretch=1)
        offset_layout.addWidget(self.spin_offset)
        p_form.addRow("Slice Offset:", offset_layout)

        self.spin_grid_pts = QtWidgets.QSpinBox()
        self.spin_grid_pts.setRange(40, 200)
        self.spin_grid_pts.setValue(80)
        self.spin_grid_pts.setSingleStep(10)
        
        self.slider_grid_pts = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_grid_pts.setRange(40, 200)
        self.slider_grid_pts.setValue(80)
        self.slider_grid_pts.valueChanged.connect(self.spin_grid_pts.setValue)
        self.spin_grid_pts.valueChanged.connect(lambda v: (self.slider_grid_pts.setValue(v), self._recompute_and_plot()))

        res_layout = QtWidgets.QHBoxLayout()
        res_layout.addWidget(self.slider_grid_pts, stretch=1)
        res_layout.addWidget(self.spin_grid_pts)
        p_form.addRow("Resolution:", res_layout)

        # Slice Box Margin / Extent (Bohr)
        self.spin_box_margin = QtWidgets.QDoubleSpinBox()
        self.spin_box_margin.setRange(1.0, 15.0)
        self.spin_box_margin.setSingleStep(0.5)
        self.spin_box_margin.setValue(4.2)
        self.spin_box_margin.setSuffix(" a₀")

        self.slider_box_margin = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_box_margin.setRange(10, 150)
        self.slider_box_margin.setValue(42)
        self.slider_box_margin.valueChanged.connect(lambda v: self.spin_box_margin.setValue(v / 10.0))
        self.spin_box_margin.valueChanged.connect(lambda v: (self.slider_box_margin.setValue(int(round(v * 10.0))), self._recompute_and_plot()))

        margin_layout = QtWidgets.QHBoxLayout()
        margin_layout.addWidget(self.slider_box_margin, stretch=1)
        margin_layout.addWidget(self.spin_box_margin)
        p_form.addRow("Box Size Margin:", margin_layout)

        ctrl_layout.addWidget(grp_plane)

        # Contour Styling Group
        grp_style = QtWidgets.QGroupBox("Contour Style & Colormap")
        s_form = QtWidgets.QFormLayout(grp_style)
        s_form.setSpacing(8)

        self.combo_cmap = QtWidgets.QComboBox()
        for cm in ["bwr_r", "RdBu_r", "coolwarm_r", "bwr", "coolwarm", "viridis", "plasma", "Spectral_r", "seismic"]:
            self.combo_cmap.addItem(cm)
        self.combo_cmap.currentIndexChanged.connect(self._recompute_and_plot)
        s_form.addRow("Colormap:", self.combo_cmap)

        self.spin_range = QtWidgets.QDoubleSpinBox()
        self.spin_range.setRange(0.005, 0.50)
        self.spin_range.setSingleStep(0.005)
        self.spin_range.setValue(0.05)
        self.spin_range.setPrefix("± ")
        self.spin_range.setSuffix(" a.u.")

        self.slider_range = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_range.setRange(5, 500)
        self.slider_range.setValue(50)
        self.slider_range.valueChanged.connect(lambda v: self.spin_range.setValue(v / 1000.0))
        self.spin_range.valueChanged.connect(lambda v: (self.slider_range.setValue(int(round(v * 1000.0))), self._recompute_and_plot()))

        range_layout = QtWidgets.QHBoxLayout()
        range_layout.addWidget(self.slider_range, stretch=1)
        range_layout.addWidget(self.spin_range)
        s_form.addRow("Color Limits:", range_layout)

        self.spin_levels = QtWidgets.QSpinBox()
        self.spin_levels.setRange(10, 80)
        self.spin_levels.setValue(32)

        self.slider_levels = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_levels.setRange(10, 80)
        self.slider_levels.setValue(32)
        self.slider_levels.valueChanged.connect(self.spin_levels.setValue)
        self.spin_levels.valueChanged.connect(lambda v: (self.slider_levels.setValue(v), self._recompute_and_plot()))

        levels_layout = QtWidgets.QHBoxLayout()
        levels_layout.addWidget(self.slider_levels, stretch=1)
        levels_layout.addWidget(self.spin_levels)
        s_form.addRow("Contours:", levels_layout)

        self.chk_atoms = QtWidgets.QCheckBox("Show projected atoms")
        self.chk_atoms.setChecked(True)
        self.chk_atoms.toggled.connect(self._recompute_and_plot)
        s_form.addRow("", self.chk_atoms)

        self.chk_lines = QtWidgets.QCheckBox("Show contour line overlays")
        self.chk_lines.setChecked(True)
        self.chk_lines.toggled.connect(self._recompute_and_plot)
        s_form.addRow("", self.chk_lines)

        ctrl_layout.addWidget(grp_style)
        ctrl_layout.addStretch()

        btn_save = QtWidgets.QPushButton("💾 Save 2D Figure (PNG/PDF)...")
        btn_save.setStyleSheet("background-color: #2563eb; color: #fff; font-weight: bold; padding: 7px;")
        btn_save.clicked.connect(self._on_save_figure)
        ctrl_layout.addWidget(btn_save)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        ctrl_layout.addWidget(btn_close)

        main_layout.addWidget(ctrl_panel)

        # Right: Matplotlib Canvas
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(7, 6), dpi=100)
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        self.figure.patch.set_facecolor("#ffffff" if is_white else "#1e293b")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)

        main_layout.addWidget(plot_widget, stretch=1)

    def _on_property_changed(self):
        prop = self.combo_property.currentData() or "ESP"
        # Adjust colormap and default range based on property
        self.combo_cmap.blockSignals(True)
        self.spin_range.blockSignals(True)
        if prop == "ESP":
            self.combo_cmap.setCurrentText("bwr_r")
            self.spin_range.setValue(0.05)
            self.spin_range.setPrefix("± ")
            self.spin_range.setSuffix(" a.u.")
        elif prop == "RHO":
            self.combo_cmap.setCurrentText("viridis")
            self.spin_range.setValue(0.20)
            self.spin_range.setPrefix("Max ")
            self.spin_range.setSuffix(" e/Bohr³")
        elif prop == "LAPL":
            self.combo_cmap.setCurrentText("seismic")
            self.spin_range.setValue(0.50)
            self.spin_range.setPrefix("± ")
            self.spin_range.setSuffix(" a.u.")
        elif prop == "RDG":
            self.combo_cmap.setCurrentText("plasma")
            self.spin_range.setValue(2.0)
            self.spin_range.setPrefix("Max ")
            self.spin_range.setSuffix(" (dimensionless)")
        self.combo_cmap.blockSignals(False)
        self.spin_range.blockSignals(False)
        self._recompute_and_plot()

    def _recompute_and_plot(self):
        if self.mol is None and self.vol_data is None:
            return

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            plane_mode = self.combo_plane.currentData() or "XZ"
            axis_map = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}
            u_ax, v_ax, n_ax = axis_map.get(plane_mode, (0, 2, 1))

            if self.mol is not None:
                # Wavefunction ESP mode
                coords_bohr = self.mol.atom_coords()
                offset_bohr = float(self.spin_offset.value()) * ANGSTROM_TO_BOHR
                plane_n = np.mean(coords_bohr[:, n_ax]) + offset_bohr

                pad_bohr = float(self.spin_box_margin.value()) if hasattr(self, "spin_box_margin") else 4.2
                u_min = coords_bohr[:, u_ax].min() - pad_bohr
                u_max = coords_bohr[:, u_ax].max() + pad_bohr
                v_min = coords_bohr[:, v_ax].min() - pad_bohr
                v_max = coords_bohr[:, v_ax].max() + pad_bohr

                n_pts = self.spin_grid_pts.value()
                u_vals = np.linspace(u_min, u_max, n_pts)
                v_vals = np.linspace(v_min, v_max, n_pts)
                U_grid, V_grid = np.meshgrid(u_vals, v_vals, indexing="ij")

                pts = np.zeros((n_pts * n_pts, 3), dtype=np.float64)
                pts[:, u_ax] = U_grid.ravel()
                pts[:, v_ax] = V_grid.ravel()
                pts[:, n_ax] = plane_n

                # Nuclear ESP:
                znuc = self.mol.atom_charges().astype(np.float64)
                r_nuc = np.linalg.norm(pts[:, None, :] - coords_bohr[None, :, :], axis=2)
                v_nuc = np.sum(znuc[None, :] / np.maximum(r_nuc, 1e-5), axis=1)

                # Electronic ESP via chunked int1e_grids or other properties:
                selected_prop = self.combo_property.currentData() or "ESP"
                prop_title = "Electrostatic Potential (ESP)"
                val_unit = "Hartree / a.u."

                if selected_prop == "ESP":
                    chunk = 4000
                    v_elec = np.empty(len(pts), dtype=np.float64)
                    for i in range(0, len(pts), chunk):
                        ints = self.mol.intor("int1e_grids", grids=pts[i:i + chunk])
                        v_elec[i:i + chunk] = np.einsum("gij,ij->g", ints, self.dm)
                    v_2d = (v_nuc - v_elec).reshape((n_pts, n_pts))
                elif selected_prop == "RHO":
                    prop_title = "Electron Density (ρ)"
                    val_unit = "e / Bohr³"
                    from pyscf.dft import numint
                    ao = numint.eval_ao(self.mol, pts, deriv=0)
                    rho = numint.eval_rho(self.mol, ao, self.dm, xctype="LDA")
                    v_2d = rho.reshape((n_pts, n_pts))
                elif selected_prop == "LAPL":
                    prop_title = "Laplacian of Density (∇²ρ)"
                    val_unit = "a.u."
                    from pyscf.dft import numint
                    ao = numint.eval_ao(self.mol, pts, deriv=2)
                    rho_mgga = numint.eval_rho(self.mol, ao, self.dm, xctype="MGGA")
                    # rho_mgga[5] is the exact laplacian of electron density
                    v_2d = rho_mgga[5].reshape((n_pts, n_pts))
                elif selected_prop == "RDG":
                    prop_title = "Reduced Density Gradient (RDG)"
                    val_unit = "dimensionless"
                    from pyscf.dft import numint
                    ao = numint.eval_ao(self.mol, pts, deriv=1)
                    rho_gga = numint.eval_rho(self.mol, ao, self.dm, xctype="GGA")
                    norm_grad = np.linalg.norm(rho_gga[1:4], axis=0)
                    cf = 2.0 * (3.0 * np.pi**2)**(1.0 / 3.0)
                    rdg = norm_grad / (cf * np.maximum(rho_gga[0], 1e-12)**(4.0 / 3.0))
                    v_2d = rdg.reshape((n_pts, n_pts))
                else:
                    v_2d = np.zeros((n_pts, n_pts))

                u_ang = u_vals * BOHR_TO_ANGSTROM
                v_ang = v_vals * BOHR_TO_ANGSTROM
                atom_symbols = [self.mol.atom_symbol(i) for i in range(self.mol.natm)]
                atom_coords_ang = coords_bohr * BOHR_TO_ANGSTROM
                slice_pos_ang = plane_n * BOHR_TO_ANGSTROM
            else:
                # VolumetricData grid slicing mode (Cube file / Difference Density)
                vol = self.vol_data
                n_pts = self.spin_grid_pts.value()
                
                # Bounding box of the 3D grid
                box_min = vol.origin
                box_max = vol.origin + np.array([
                    vol.voxels_x * vol.step_x[0],
                    vol.voxels_y * vol.step_y[1],
                    vol.voxels_z * vol.step_z[2]
                ])
                center_n = 0.5 * (box_min[n_ax] + box_max[n_ax])
                slice_pos_ang = center_n + float(self.spin_offset.value())

                u_ang = np.linspace(box_min[u_ax], box_max[u_ax], n_pts)
                v_ang = np.linspace(box_min[v_ax], box_max[v_ax], n_pts)
                U_mesh, V_mesh = np.meshgrid(u_ang, v_ang, indexing="ij")

                # Trilinear interpolation on 3D grid
                norm_u = (U_mesh - vol.origin[u_ax]) / max(1e-5, (box_max[u_ax] - box_min[u_ax]))
                norm_v = (V_mesh - vol.origin[v_ax]) / max(1e-5, (box_max[v_ax] - box_min[v_ax]))
                norm_n = (slice_pos_ang - vol.origin[n_ax]) / max(1e-5, (box_max[n_ax] - box_min[n_ax]))

                dims = np.array([vol.voxels_x, vol.voxels_y, vol.voxels_z])
                idx_n = np.clip(int(norm_n * (dims[n_ax] - 1)), 0, dims[n_ax] - 1)
                
                # Take slice directly along n_ax
                if n_ax == 0:
                    raw_slice = vol.grid_data[idx_n, :, :]
                    u_src = np.linspace(box_min[1], box_max[1], vol.voxels_y)
                    v_src = np.linspace(box_min[2], box_max[2], vol.voxels_z)
                elif n_ax == 1:
                    raw_slice = vol.grid_data[:, idx_n, :]
                    u_src = np.linspace(box_min[0], box_max[0], vol.voxels_x)
                    v_src = np.linspace(box_min[2], box_max[2], vol.voxels_z)
                else:
                    raw_slice = vol.grid_data[:, :, idx_n]
                    u_src = np.linspace(box_min[0], box_max[0], vol.voxels_x)
                    v_src = np.linspace(box_min[1], box_max[1], vol.voxels_y)

                from scipy.interpolate import RegularGridInterpolator
                interp = RegularGridInterpolator((u_src, v_src), raw_slice, bounds_error=False, fill_value=0.0)
                eval_pts = np.stack([U_mesh.ravel(), V_mesh.ravel()], axis=-1)
                v_2d = interp(eval_pts).reshape((n_pts, n_pts))

                atom_symbols = vol.symbols
                atom_coords_ang = vol.atom_coords
                prop_title = vol.title or "Volumetric Grid"
                val_unit = "a.u."

            # Plot onto Matplotlib Figure
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
            ax.set_facecolor("#f8fafc" if is_white else "#0f172a")

            lim = float(self.spin_range.value())
            n_levels = self.spin_levels.value()
            selected_prop = self.combo_property.currentData() or "ESP"
            if selected_prop in ("RHO", "RDG"):
                # Non-negative properties: 0 to lim
                levels = np.linspace(0.0, lim, n_levels)
            else:
                # Bipolar properties (-lim to +lim)
                levels = np.linspace(-lim, lim, n_levels)
            cmap_name = self.combo_cmap.currentText()

            cp = ax.contourf(
                u_ang, v_ang, v_2d.T,
                levels=levels, cmap=cmap_name, extend="both"
            )
            if self.chk_lines.isChecked():
                ax.contour(
                    u_ang, v_ang, v_2d.T,
                    levels=levels[::max(1, len(levels) // 10)],
                    colors="#334155", linewidths=0.6, alpha=0.7
                )
                if np.min(v_2d) <= 0.0 <= np.max(v_2d):
                    ax.contour(
                        u_ang, v_ang, v_2d.T,
                        levels=[0.0],
                        colors="#ffffff", linewidths=1.2, linestyles="dashed"
                    )

            # Draw atom projections
            if self.chk_atoms.isChecked() and len(atom_coords_ang) > 0:
                for sym, cat in zip(atom_symbols, atom_coords_ang):
                    at_u = cat[u_ax]
                    at_v = cat[v_ax]
                    dist_from_plane = abs(cat[n_ax] - slice_pos_ang)
                    alpha = max(0.25, min(1.0, 1.0 - (dist_from_plane / 2.0)))
                    ax.scatter([at_u], [at_v], color="#f8fafc", s=60, edgecolors="#0f172a", linewidth=1.2, alpha=alpha, zorder=6)
                    ax.text(at_u + 0.12, at_v + 0.12, sym, color="#f8fafc", fontsize=9, fontweight="bold", alpha=alpha, zorder=7)

            axis_labels = {"XY": ("X (Å)", "Y (Å)"), "XZ": ("X (Å)", "Z (Å)"), "YZ": ("Y (Å)", "Z (Å)")}
            lx, ly = axis_labels.get(plane_mode, ("U (Å)", "V (Å)"))
            text_color = "#0f172a" if is_white else "#f8fafc"
            tick_color = "#475569" if is_white else "#cbd5e1"
            spine_color = "#cbd5e1" if is_white else "#475569"
            ax.set_xlabel(lx, color=text_color, fontsize=10)
            ax.set_ylabel(ly, color=text_color, fontsize=10)
            ax.set_title(f"2D {prop_title} ({plane_mode} Plane, Offset: {self.spin_offset.value():+.2f} Å)", color="#2563eb" if is_white else "#38bdf8", fontsize=11, fontweight="bold")
            ax.tick_params(colors=tick_color)
            for spine in ax.spines.values():
                spine.set_color(spine_color)
            ax.set_aspect("equal")

            cbar = self.figure.colorbar(cp, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(f"Value ({val_unit})", color=text_color, fontsize=9)
            cbar.ax.tick_params(colors=tick_color)

            self.figure.tight_layout()
            self.canvas.draw()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_save_figure(self):
        prop = (self.combo_property.currentData() or "contour").lower()
        plane = (self.combo_plane.currentData() or "slice").lower()
        default_name = f"{prop}_2d_{plane}_slice.png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save 2D Contour Figure", default_name, "PNG Image (*.png);;PDF Vector File (*.pdf);;SVG Image (*.svg)"
        )
        if path:
            self.figure.savefig(path, dpi=300, facecolor=self.figure.get_facecolor(), bbox_inches="tight")
            QtWidgets.QMessageBox.information(self, "Figure Saved", f"Exported 2D contour map to:\n{path}")
