"""
Volumetric Grid & Isosurface Controller Dialog (GaussView & Avogadro style).

Provides controls for:
- Loading Gaussian .cube files (Electron density, ESP, Molecular Orbitals)
- Loading Molden (.molden) files with MO coefficients
- Interactive isovalue adjustment slider with real-time Marching Cubes updates
- Surface opacity / transparency slider
- Positive (+) and Negative (-) orbital lobe color pickers
- Toggling surface visibility
"""

import logging
import os
from typing import List, Optional, Tuple, Dict
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .volumetric import (
    VolumetricData,
    MoldenFileData,
    MoldenMO,
    IsosurfaceMesh,
    extract_isosurface,
    generate_orbital_surfaces,
    evaluate_molden_orbital_grid,
    compute_molden_esp_grid,
    compute_molden_spin_density_grid,
    compute_molden_esp_mapped_density,
    compute_molden_nci_rdg,
    compute_molden_alie_mapped_density,
    compute_molden_orbital_decomposition,
    compute_molden_elf_grid,
    compute_molden_localized_orbitals,
    evaluate_lmo_orbital_grid,
    compute_molden_chelpg_charges,
    parse_cube_file,
    parse_molden_file,
    write_cube_file,
    compute_grid_difference,
    _HAS_MARCHING_CUBES,
)
from .dialog_esp_2d import ESP2DContourDialog
from .workers import run_in_background


class _CubeLoaderThread(QtCore.QThread):
    loaded = QtCore.pyqtSignal(object, str)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path: str, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            vol = parse_cube_file(self.path)
            self.loaded.emit(vol, self.path)
        except Exception as e:
            self.failed.emit(str(e))


class VolumetricSurfaceDialog(QtWidgets.QDialog):
    surface_updated = QtCore.pyqtSignal(list, float)  # (List[IsosurfaceMesh], opacity)
    load_molecule_requested = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords)

    def __init__(self, parent: QtWidgets.QWidget, viewer=None):
        super().__init__(parent)
        self.setWindowTitle("Volumetric Grid & Isosurface Viewer (.cube / .molden)")
        self.resize(520, 480)
        self.viewer = viewer

        self.current_vol_data: Optional[VolumetricData] = None
        self.current_molden_data: Optional[MoldenFileData] = None
        self.current_molden_path: Optional[str] = None
        self.current_lmo_orbitals = None  # Holds localized orbital coefficients if calculated
        self.pos_color = (0.22, 0.58, 0.94)  # Vibrant Cyan-Blue (+)
        self.neg_color = (0.95, 0.30, 0.25)  # Vibrant Coral-Red (-)
        self._surface_undo_stack: List[Tuple[List[IsosurfaceMesh], float]] = []
        self._surface_redo_stack: List[Tuple[List[IsosurfaceMesh], float]] = []

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # File Loader Section
        grp_file = QtWidgets.QGroupBox("Load Volumetric Data")
        f_layout = QtWidgets.QHBoxLayout(grp_file)

        btn_load_cube = QtWidgets.QPushButton("📂 Open .cube File...")
        btn_load_cube.setStyleSheet("background-color: #2563eb; font-weight: bold; padding: 6px;")
        btn_load_cube.clicked.connect(self._on_open_cube)
        f_layout.addWidget(btn_load_cube)

        btn_grid_math = QtWidgets.QPushButton("➗ Grid Math (A - B)...")
        btn_grid_math.setStyleSheet("background-color: #7c3aed; font-weight: bold; padding: 6px;")
        btn_grid_math.setToolTip("Multiwfn style: Compute difference density (Δρ = A - B), sum, or product between two .cube grids")
        btn_grid_math.clicked.connect(self._on_grid_arithmetic)
        f_layout.addWidget(btn_grid_math)

        btn_load_molden = QtWidgets.QPushButton("📂 Open .molden / ORCA (.molden.input, .gbw)...")
        btn_load_molden.setStyleSheet("background-color: #0d9488; font-weight: bold; padding: 6px;")
        btn_load_molden.setToolTip("Open Molden wavefunction (.molden, .molden.input) or ORCA binary wavefunction (.gbw)")
        btn_load_molden.clicked.connect(self._on_open_molden)
        f_layout.addWidget(btn_load_molden)

        layout.addWidget(grp_file)

        # Molecular Property Buttons (ESP, Spin Density, CHELPG)
        self.grp_properties = QtWidgets.QGroupBox("Wavefunction Analysis (Multiwfn-Style)")
        prop_layout = QtWidgets.QGridLayout(self.grp_properties)
        prop_layout.setSpacing(6)

        self.btn_esp_mapped = QtWidgets.QPushButton("🎨 ESP-Mapped Density (Red − / Blue +)")
        self.btn_esp_mapped.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_esp_mapped.setToolTip("GaussView / Multiwfn style: Electron isodensity envelope colored by electrostatic potential heatmap (Red = Negative δ-, Blue = Positive δ+)")
        self.btn_esp_mapped.clicked.connect(self._on_compute_esp_mapped_density)
        prop_layout.addWidget(self.btn_esp_mapped, 0, 0)

        self.btn_calc_esp = QtWidgets.QPushButton("⚡ 3D ESP Isopotential Lobes (±V)")
        self.btn_calc_esp.setStyleSheet("background-color: #4f46e5; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_esp.setToolTip("Calculate exact 3D electrostatic potential grid and display positive (+V) and negative (-V) isopotential lobes")
        self.btn_calc_esp.clicked.connect(self._on_compute_esp)
        prop_layout.addWidget(self.btn_calc_esp, 0, 1)

        self.btn_spin_density = QtWidgets.QPushButton("🌀 Spin Density")
        self.btn_spin_density.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_spin_density.setToolTip("Compute open-shell spin density distribution (rho_alpha - rho_beta)")
        self.btn_spin_density.clicked.connect(self._on_compute_spin_density)
        prop_layout.addWidget(self.btn_spin_density, 1, 0)

        self.btn_esp_2d = QtWidgets.QPushButton("📈 2D ESP Contours...")
        self.btn_esp_2d.setStyleSheet("background-color: #7c3aed; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_esp_2d.setToolTip("Plot 2D ESP slice contours on XY, XZ, or YZ planes with atom overlays")
        self.btn_esp_2d.clicked.connect(self._on_open_esp_2d)
        prop_layout.addWidget(self.btn_esp_2d, 1, 1)

        self.btn_calc_elf = QtWidgets.QPushButton("⚡ ELF Localization")
        self.btn_calc_elf.setStyleSheet("background-color: #0d9488; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_elf.setToolTip("Becke-Edgecombe Electron Localization Function (ELF, 0.0 - 1.0) for chemical bond & lone pair visualization")
        self.btn_calc_elf.clicked.connect(self._on_compute_elf)
        prop_layout.addWidget(self.btn_calc_elf, 2, 0)

        self.btn_localize_mo = QtWidgets.QPushButton("🔍 Localize Orbitals")
        self.btn_localize_mo.setStyleSheet("background-color: #b45309; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_localize_mo.setToolTip("Transform canonical molecular orbitals to localized MOs (Boys or Pipek-Mezey bonds & lone pairs)")
        self.btn_localize_mo.clicked.connect(self._on_compute_localized_orbitals)
        prop_layout.addWidget(self.btn_localize_mo, 2, 1)

        self.btn_calc_nci = QtWidgets.QPushButton("🌿 NCI / RDG (Interactions)")
        self.btn_calc_nci.setStyleSheet("background-color: #15803d; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_nci.setToolTip("Reduced Density Gradient (RDG): Visualizes non-covalent interactions (Blue = H-bond, Green = vdW, Red = steric clash)")
        self.btn_calc_nci.clicked.connect(self._on_compute_nci_rdg)

        self.btn_nci_plot = QtWidgets.QPushButton("📈 NCI 2D Plot...")
        self.btn_nci_plot.setStyleSheet("background-color: #047857; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_nci_plot.setToolTip("Open interactive 2D Reduced Density Gradient (RDG) vs sign(λ₂)ρ scatter plot")
        self.btn_nci_plot.clicked.connect(self._on_open_nci_plot)

        nci_h_box = QtWidgets.QHBoxLayout()
        nci_h_box.addWidget(self.btn_calc_nci, stretch=2)
        nci_h_box.addWidget(self.btn_nci_plot, stretch=1)
        prop_layout.addLayout(nci_h_box, 3, 0)

        self.btn_calc_alie = QtWidgets.QPushButton("🎯 ALIE Reactive Sites")
        self.btn_calc_alie.setStyleSheet("background-color: #be185d; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_alie.setToolTip("Average Local Ionization Energy (ALIE): Identifies electrophilic attack & radical reaction sites (surface minima)")
        self.btn_calc_alie.clicked.connect(self._on_compute_alie)
        prop_layout.addWidget(self.btn_calc_alie, 3, 1)

        self.btn_calc_chelpg = QtWidgets.QPushButton("🎯 CHELPG Partial Charges")
        self.btn_calc_chelpg.setStyleSheet("background-color: #059669; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_calc_chelpg.setToolTip("Fit electrostatic potential to atomic centers (Fast / Accurate)")
        self.btn_calc_chelpg.clicked.connect(self._on_compute_chelpg)
        prop_layout.addWidget(self.btn_calc_chelpg, 4, 0)

        self.btn_multiwfn_suite = QtWidgets.QPushButton("⚡ Multiwfn Suite (Bonds/Fukui/QTAIM/CM5)")
        self.btn_multiwfn_suite.setStyleSheet("background-color: #4f46e5; color: #ffffff; font-weight: bold; padding: 5px;")
        self.btn_multiwfn_suite.setToolTip("Open full Multiwfn Electronic Structure suite: Mayer/Wiberg bond orders, Conceptual DFT Fukui/Dual descriptors, QTAIM BCPs, CM5/Hirshfeld charges")
        self.btn_multiwfn_suite.clicked.connect(self._on_open_multiwfn_suite)
        prop_layout.addWidget(self.btn_multiwfn_suite, 4, 1)

        self.grp_properties.setVisible(False)
        layout.addWidget(self.grp_properties)

        # Info Banner
        self.lbl_info = QtWidgets.QLabel("No volumetric grid loaded. Open a Gaussian .cube or Molden file.")
        self.lbl_info.setStyleSheet("color: #94a3b8; font-style: italic;")
        self.lbl_info.setWordWrap(True)
        layout.addWidget(self.lbl_info)

        # Molecular Orbital Picker (for .molden files or multi-MO files)
        self.grp_mo = QtWidgets.QGroupBox("Molecular Orbitals (MO List)")
        mo_layout = QtWidgets.QVBoxLayout(self.grp_mo)
        self.combo_mo = QtWidgets.QComboBox()
        self.combo_mo.currentIndexChanged.connect(self._on_mo_selected)
        mo_layout.addWidget(self.combo_mo)
        self.grp_mo.setVisible(False)
        layout.addWidget(self.grp_mo)

        # Isosurface Parameters Group
        grp_params = QtWidgets.QGroupBox("Marching Cubes Isosurface Controls")
        p_layout = QtWidgets.QFormLayout(grp_params)
        p_layout.setSpacing(8)

        # Isovalue Slider & Spinbox
        row_iso = QtWidgets.QHBoxLayout()
        self.slider_iso = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_iso.setRange(1, 500)  # 0.001 to 0.500
        self.slider_iso.setValue(30)      # 0.030 default
        self.slider_iso.valueChanged.connect(self._on_slider_iso_changed)
        row_iso.addWidget(self.slider_iso, stretch=1)

        self.spin_iso = QtWidgets.QDoubleSpinBox()
        self.spin_iso.setRange(0.0001, 2.0)
        self.spin_iso.setDecimals(4)
        self.spin_iso.setSingleStep(0.005)
        self.spin_iso.setValue(0.0300)
        self.spin_iso.valueChanged.connect(self._on_spin_iso_changed)
        row_iso.addWidget(self.spin_iso)
        p_layout.addRow("Isovalue (|ψ|):", row_iso)

        # Opacity Slider
        row_op = QtWidgets.QHBoxLayout()
        self.slider_opacity = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_opacity.setRange(10, 100)
        self.slider_opacity.setValue(82)
        self.slider_opacity.valueChanged.connect(self._on_opacity_changed)
        row_op.addWidget(self.slider_opacity, stretch=1)

        self.lbl_opacity = QtWidgets.QLabel("82%")
        self.lbl_opacity.setFixedWidth(36)
        row_op.addWidget(self.lbl_opacity)
        p_layout.addRow("Surface Opacity:", row_op)

        # Color Customization Buttons
        row_col = QtWidgets.QHBoxLayout()
        self.btn_pos_col = QtWidgets.QPushButton("(+) Lobe Color")
        self.btn_pos_col.setStyleSheet("background-color: #38bdf8; color: #000; font-weight: bold;")
        self.btn_pos_col.clicked.connect(self._on_pick_pos_color)
        row_col.addWidget(self.btn_pos_col)

        self.btn_neg_col = QtWidgets.QPushButton("(-) Lobe Color")
        self.btn_neg_col.setStyleSheet("background-color: #f87171; color: #000; font-weight: bold;")
        self.btn_neg_col.clicked.connect(self._on_pick_neg_color)
        row_col.addWidget(self.btn_neg_col)
        p_layout.addRow("Phase Colors:", row_col)

        # Visibility Checkbox
        self.chk_visible = QtWidgets.QCheckBox("Show Isosurface in 3D Viewport")
        self.chk_visible.setChecked(True)
        self.chk_visible.toggled.connect(self._update_surface)
        p_layout.addRow("Visibility:", self.chk_visible)

        # Multi-Component Dual Overlay Checkbox (Positron + Electron)
        self.chk_overlay_mode = QtWidgets.QCheckBox("Keep Previous Surface (Dual Positron/Electron Overlay)")
        self.chk_overlay_mode.setChecked(False)
        self.chk_overlay_mode.setToolTip(
            "newBRORSEN style: Overlay multiple isosurfaces simultaneously.\n"
            "For example, display the electronic density/orbital lobes and overlay the bound positron density cloud."
        )
        p_layout.addRow("Dual Overlay:", self.chk_overlay_mode)

        # Wavefunction Grid Resolution (Low 30^3 / Medium 40^3 / High 60^3)
        row_grid = QtWidgets.QHBoxLayout()
        self.combo_grid_points = QtWidgets.QComboBox()
        self.combo_grid_points.addItem("Low (30³ — Fast for large molecules)", 30)
        self.combo_grid_points.addItem("Medium (40³ — Standard)", 40)
        self.combo_grid_points.addItem("High (60³ — Publication quality)", 60)
        self.combo_grid_points.setCurrentIndex(1)
        self.combo_grid_points.setToolTip(
            "Select 3D voxel grid resolution (N×N×N) when evaluating orbitals, ESP, ELF, or NCI from a Molden wavefunction"
        )
        self.combo_grid_points.currentIndexChanged.connect(self._on_grid_points_changed)
        row_grid.addWidget(self.combo_grid_points, stretch=1)
        p_layout.addRow("Grid Resolution:", row_grid)

        # Mesh Visual Quality / Resolution Selector
        row_res = QtWidgets.QHBoxLayout()
        self.combo_mesh_res = QtWidgets.QComboBox()
        self.combo_mesh_res.addItem("Normal (Fast)", 1.0)
        self.combo_mesh_res.addItem("High Resolution (1.5× Grid)", 1.5)
        self.combo_mesh_res.addItem("Ultra-Fine (2.0× Smooth Mesh)", 2.0)
        self.combo_mesh_res.currentIndexChanged.connect(self._update_surface)
        row_res.addWidget(self.combo_mesh_res, stretch=1)
        p_layout.addRow("Mesh Quality:", row_res)

        # Mesh Style (Solid vs Wireframe)
        row_style = QtWidgets.QHBoxLayout()
        self.combo_surface_style = QtWidgets.QComboBox()
        self.combo_surface_style.addItem("Solid Shaded Surface", "Solid")
        self.combo_surface_style.addItem("Wireframe Polygon Mesh", "Wireframe")
        self.combo_surface_style.currentIndexChanged.connect(self._on_surface_style_changed)
        row_style.addWidget(self.combo_surface_style, stretch=1)
        p_layout.addRow("Render Style:", row_style)

        layout.addWidget(grp_params)

        # Stats readout
        self.lbl_stats = QtWidgets.QLabel("")
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        stats_col = "#047857" if is_white else "#a7f3d0"
        self.lbl_stats.setStyleSheet(f"font-family: monospace; color: {stats_col}; font-size: 11px;")
        layout.addWidget(self.lbl_stats)

        # Background Calculation Progress Bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 2px;
                background-color: #334155;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 2px;
            }
        """)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addStretch()

        # Bottom Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_clear = QtWidgets.QPushButton("Clear Surface")
        btn_clear.clicked.connect(self._on_clear_surface)
        btn_box.addWidget(btn_clear)

        self.btn_undo_surf = QtWidgets.QPushButton("↩️ Undo Surface")
        self.btn_undo_surf.setToolTip("Revert to the previous multi-surface overlay snapshot")
        self.btn_undo_surf.setEnabled(False)
        self.btn_undo_surf.clicked.connect(self._on_undo_surface)
        btn_box.addWidget(self.btn_undo_surf)

        self.btn_redo_surf = QtWidgets.QPushButton("↪️ Redo Surface")
        self.btn_redo_surf.setToolTip("Restore the undone surface overlay snapshot")
        self.btn_redo_surf.setEnabled(False)
        self.btn_redo_surf.clicked.connect(self._on_redo_surface)
        btn_box.addWidget(self.btn_redo_surf)

        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _on_open_cube(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Gaussian Volumetric Cube File", "", "Gaussian Cube Files (*.cube);;All Files (*)"
        )
        if not path:
            return
        self.load_cube_path(path)

    def _apply_cube_data(self, vol: VolumetricData, path: str):
        """Configure UI state and render loaded volumetric data."""
        self._last_cube_path = path
        self.current_vol_data = vol
        self.current_molden_data = None
        self.current_molden_path = None
        self.current_lmo_orbitals = None
        self.grp_mo.setVisible(False)

        # Auto-scale isovalue based on grid standard deviation / max
        g_abs = np.abs(vol.grid_data)
        max_val = float(np.max(g_abs))
        sugg_iso = max(0.005, min(0.1, float(np.percentile(g_abs[g_abs > 1e-4], 85)))) if np.any(g_abs > 1e-4) else 0.03
        self.spin_iso.blockSignals(True)
        self.spin_iso.setValue(sugg_iso)
        self.spin_iso.blockSignals(False)

        self.lbl_info.setText(
            f"<b>{vol.title or 'Cube File'}</b>: {vol.voxels_x}×{vol.voxels_y}×{vol.voxels_z} grid "
            f"({vol.symbols and len(vol.symbols)} atoms). Min: {float(np.min(vol.grid_data)):.4f}, Max: {max_val:.4f}"
        )

        # Prompt to load coordinates if molecular viewer is empty
        if self.viewer and len(self.viewer.coords) == 0 and len(vol.symbols) > 0:
            self.load_molecule_requested.emit(vol.symbols, vol.atom_coords)

        self._update_surface()

    def load_cube_path(self, path: str, synchronous: bool = False):
        """Load and display a Gaussian .cube volumetric file in the background."""
        from .workers import SYNCHRONOUS
        if synchronous or SYNCHRONOUS:
            try:
                vol = parse_cube_file(path)
                self._apply_cube_data(vol, path)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Cube Parse Error", f"Failed to load {path}:\n{e}")
            return

        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(True)
        self.lbl_stats.setText(f"Loading cube file '{os.path.basename(path)}' in background...")

        def on_loaded(vol, p):
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
            self._apply_cube_data(vol, p)

        def on_failed(err):
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
            self.lbl_stats.setText(f"Failed to load cube: {err}")
            QtWidgets.QMessageBox.warning(self, "Cube Parse Error", f"Failed to load {path}:\n{err}")

        self._cube_loader = _CubeLoaderThread(path, self)
        self._cube_loader.loaded.connect(on_loaded)
        self._cube_loader.failed.connect(on_failed)
        self._cube_loader.start()

    def _on_grid_arithmetic(self):
        """Perform grid operations between two Gaussian .cube files (Multiwfn style)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Volumetric Grid Arithmetic (A - B / Difference Density)")
        dlg.resize(480, 240)
        vbox = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        
        row_a = QtWidgets.QHBoxLayout()
        edit_a = QtWidgets.QLineEdit()
        if hasattr(self, "_last_cube_path") and self._last_cube_path:
            edit_a.setText(self._last_cube_path)
        btn_browse_a = QtWidgets.QPushButton("Browse...")
        btn_browse_a.clicked.connect(lambda: edit_a.setText(
            QtWidgets.QFileDialog.getOpenFileName(dlg, "Select Grid A (.cube)", "", "Cube Files (*.cube)")[0] or edit_a.text()
        ))
        row_a.addWidget(edit_a, stretch=1)
        row_a.addWidget(btn_browse_a)
        form.addRow("Grid A (e.g. Complex / Total):", row_a)

        row_b = QtWidgets.QHBoxLayout()
        edit_b = QtWidgets.QLineEdit()
        btn_browse_b = QtWidgets.QPushButton("Browse...")
        btn_browse_b.clicked.connect(lambda: edit_b.setText(
            QtWidgets.QFileDialog.getOpenFileName(dlg, "Select Grid B (.cube)", "", "Cube Files (*.cube)")[0] or edit_b.text()
        ))
        row_b.addWidget(edit_b, stretch=1)
        row_b.addWidget(btn_browse_b)
        form.addRow("Grid B (e.g. Monomer / Reference):", row_b)

        combo_op = QtWidgets.QComboBox()
        combo_op.addItem("Difference (A - B) [Density difference / Interaction]", "A - B")
        combo_op.addItem("Sum (A + B)", "A + B")
        combo_op.addItem("Product (A * B)", "A * B")
        form.addRow("Operation:", combo_op)

        vbox.addLayout(form)
        vbox.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        path_a = edit_a.text().strip()
        path_b = edit_b.text().strip()
        if not path_a or not os.path.isfile(path_a) or not path_b or not os.path.isfile(path_b):
            QtWidgets.QMessageBox.warning(self, "Invalid Files", "Please select valid files for both Grid A and Grid B.")
            return

        try:
            grid_a = parse_cube_file(path_a)
            grid_b = parse_cube_file(path_b)
            op = combo_op.currentData() or "A - B"
            diff_vol = compute_grid_difference(grid_a, grid_b, operation=op)
            
            self.current_vol_data = diff_vol
            self.current_molden_data = None
            self.current_molden_path = None
            self.current_lmo_orbitals = None
            self.grp_mo.setVisible(False)

            max_val = float(np.max(np.abs(diff_vol.grid_data)))
            sugg_iso = max(0.001, round(max_val * 0.15, 4))
            self.spin_iso.setValue(sugg_iso)
            self.lbl_info.setText(
                f"<b>{diff_vol.title}</b>: {diff_vol.voxels_x}×{diff_vol.voxels_y}×{diff_vol.voxels_z} grid. "
                f"Min: {float(np.min(diff_vol.grid_data)):.4e}, Max: {float(np.max(diff_vol.grid_data)):.4e}"
            )
            self._update_surface()
            QtWidgets.QMessageBox.information(
                self, "Grid Calculation Complete",
                f"Calculated grid operation '{op}' successfully!\n\n"
                f"Min value: {float(np.min(diff_vol.grid_data)):.4e}\n"
                f"Max value: {float(np.max(diff_vol.grid_data)):.4e}\n"
                "The 3D isosurface has been updated in the viewport."
            )
        except Exception as err:
            QtWidgets.QMessageBox.critical(self, "Grid Calculation Error", str(err))

    def _get_default_molden_dir_or_file(self) -> str:
        """Return the most recently generated/used Molden file or directory so it is pre-selected in the file dialog."""
        if self.current_molden_path and os.path.isfile(self.current_molden_path):
            return self.current_molden_path
        p = self.parent()
        if p is not None:
            last_res = getattr(p, "_last_calculation_results", {})
            if last_res.get("molden_file") and os.path.isfile(last_res["molden_file"]):
                return last_res["molden_file"]
            active_m = getattr(p, "_active_molden_path", None)
            if active_m and os.path.isfile(active_m):
                return active_m
            last_log = getattr(p, "_last_loaded_qm_log_path", None)
            if last_log and os.path.isfile(last_log):
                stem = os.path.splitext(last_log)[0]
                for ext in (".molden.input", ".molden", "_electrons.molden"):
                    if os.path.isfile(stem + ext):
                        return stem + ext
                return os.path.dirname(last_log)
            if hasattr(p, "_last_dir"):
                d = p._last_dir()
                if d and os.path.isdir(d):
                    return d
        return os.path.abspath(".")

    def _on_open_molden(self):
        start_target = self._get_default_molden_dir_or_file()
        dlg = QtWidgets.QFileDialog(
            self,
            "Open Molden / ORCA Wavefunction File",
            os.path.dirname(start_target) if os.path.isfile(start_target) else start_target,
            "Molden & ORCA Wavefunction Files (*.molden.input *.molden *.molf *.gbw);;All Files (*)",
        )
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        # Sort by modification time (newest calculation first) so the latest ORCA run is at the top
        for proxy in dlg.findChildren(QtCore.QSortFilterProxyModel):
            proxy.sort(3, QtCore.Qt.SortOrder.DescendingOrder)
        if os.path.isfile(start_target):
            dlg.selectFile(start_target)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        selected = dlg.selectedFiles()
        if not selected:
            return
        self.load_molden_path(selected[0])

    def _convert_gbw_to_molden(self, gbw_path: str) -> Optional[str]:
        """Convert an ORCA .gbw file to .molden.input using orca_2mkl (or return existing .molden.input if newer)."""
        import shutil
        import subprocess
        from .qm_detect import detect_qm_executable

        stem = os.path.splitext(os.path.abspath(gbw_path))[0]
        molden_inp = f"{stem}.molden.input"
        molden_std = f"{stem}.molden"

        # If .molden.input already exists and is at least as new as the .gbw, use it directly
        if os.path.isfile(molden_inp) and os.path.getmtime(molden_inp) >= os.path.getmtime(gbw_path) - 2.0:
            return molden_inp
        if os.path.isfile(molden_std) and os.path.getmtime(molden_std) >= os.path.getmtime(gbw_path) - 2.0:
            return molden_std

        info = detect_qm_executable("orca")
        orca_bin = info.resolved_path.split()[0] if (info.found and info.resolved_path) else (shutil.which("orca") or "")
        resolved_orca = shutil.which(orca_bin) or (orca_bin if os.path.isabs(orca_bin) else "")
        orca_dir = os.path.dirname(resolved_orca) if resolved_orca else ""
        orca_2mkl = (os.path.join(orca_dir, "orca_2mkl") if orca_dir else None) or shutil.which("orca_2mkl")

        if not orca_2mkl or not os.path.isfile(orca_2mkl):
            QtWidgets.QMessageBox.warning(
                self,
                "orca_2mkl Not Found",
                f"Cannot convert '{os.path.basename(gbw_path)}' because 'orca_2mkl' was not found alongside ORCA.\n"
                "Please configure your ORCA executable path under File → Configure QM Engines & Routes.",
            )
            return None

        try:
            subprocess.run(
                [orca_2mkl, os.path.basename(stem), "-molden"],
                cwd=os.path.dirname(stem),
                capture_output=True,
                timeout=30,
            )
            if os.path.isfile(molden_inp):
                if not os.path.isfile(molden_std):
                    shutil.copy2(molden_inp, molden_std)
                return molden_inp
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "orca_2mkl Error", f"Failed to convert .gbw to Molden format:\n{e}")
        return None

    def load_molden_path(self, path: str):
        # If the user selected an ORCA binary wavefunction (.gbw), convert it on the fly via orca_2mkl
        if path.lower().endswith(".gbw"):
            converted = self._convert_gbw_to_molden(path)
            if not converted:
                return
            path = converted
        def _parse():
            m = parse_molden_file(path)
            return m

        def _on_parsed(m):
            self.current_molden_data = m
            self.current_molden_path = path
            self.current_lmo_orbitals = None
            if len(m.symbols) > 0:
                self.load_molecule_requested.emit(m.symbols, m.coords)

            self.combo_mo.blockSignals(True)
            self.combo_mo.clear()
            for idx, mo in enumerate(m.orbitals):
                occ_tag = f"Occ={mo.occupancy:.1f}" if mo.occupancy > 0 else "Virtual"
                self.combo_mo.addItem(f"MO #{idx+1} ({mo.sym}): E={mo.energy:.4f} a.u. [{occ_tag}]", idx)
            self.combo_mo.blockSignals(False)

            if m.orbitals:
                self.grp_mo.setVisible(True)
                self.grp_properties.setVisible(True)
                self.lbl_info.setText(
                    f"<b>{m.title}</b>: Loaded {len(m.symbols)} atoms and {len(m.orbitals)} Molecular Orbitals from Molden file."
                )
                # Automatically evaluate HOMO or first orbital
                homo_idx = 0
                for i_orb, mo in enumerate(m.orbitals):
                    if mo.occupancy > 0.05:
                        homo_idx = i_orb
                self.combo_mo.setCurrentIndex(homo_idx)
                self._on_mo_selected(homo_idx)
            else:
                self.grp_mo.setVisible(False)
                self.grp_properties.setVisible(False)
                self.lbl_info.setText(f"Loaded {len(m.symbols)} atoms from Molden file (no MO section found).")

        def _on_error(err):
            QtWidgets.QMessageBox.critical(self, "Error Loading Molden File", str(err))

        run_in_background(self, "Loading Molden File", f"Parsing {os.path.basename(path)}...", _parse, _on_parsed, _on_error)

    def _get_grid_pts(self, default: int = 40) -> int:
        if hasattr(self, "combo_grid_points"):
            return int(self.combo_grid_points.currentData() or default)
        return default

    def _on_grid_points_changed(self):
        if self.current_molden_path and self.grp_mo.isVisible():
            self._on_mo_selected(self.combo_mo.currentIndex())

    def _on_compute_esp(self):
        """Compute Molecular Electrostatic Potential (ESP) 3D grid and display isosurfaces."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "ESP Calculation", "Please load a .molden file first.")
            return

        # Settings dialog with sliders for grid resolution and box margin
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("3D ESP Isopotential Grid Settings")
        dlg.resize(460, 220)
        vbox = QtWidgets.QVBoxLayout(dlg)
        form = QtWidgets.QFormLayout()
        form.setSpacing(10)

        # Box margin (padding in Bohr)
        margin_layout = QtWidgets.QHBoxLayout()
        slider_margin = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_margin.setRange(15, 120)  # 1.5 to 12.0 Bohr
        slider_margin.setValue(35)
        spin_margin = QtWidgets.QDoubleSpinBox()
        spin_margin.setRange(1.5, 12.0)
        spin_margin.setDecimals(1)
        spin_margin.setSingleStep(0.5)
        spin_margin.setValue(3.5)
        spin_margin.setSuffix(" a₀")
        slider_margin.valueChanged.connect(lambda v: spin_margin.setValue(v / 10.0))
        spin_margin.valueChanged.connect(lambda v: slider_margin.setValue(int(round(v * 10.0))))
        margin_layout.addWidget(slider_margin, stretch=1)
        margin_layout.addWidget(spin_margin)
        form.addRow("Box Size Margin:", margin_layout)

        # Grid resolution (voxels/axis)
        grid_layout = QtWidgets.QHBoxLayout()
        default_pts = self._get_grid_pts(40)
        slider_grid = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_grid.setRange(20, 80)
        slider_grid.setValue(default_pts)
        spin_grid = QtWidgets.QSpinBox()
        spin_grid.setRange(20, 100)
        spin_grid.setSingleStep(5)
        spin_grid.setValue(default_pts)
        spin_grid.setSuffix(" voxels/axis")
        slider_grid.valueChanged.connect(spin_grid.setValue)
        spin_grid.valueChanged.connect(slider_grid.setValue)
        grid_layout.addWidget(slider_grid, stretch=1)
        grid_layout.addWidget(spin_grid)
        form.addRow("Grid Resolution:", grid_layout)

        vbox.addLayout(form)
        vbox.addStretch()

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        n_pts = int(spin_grid.value())
        margin_b = float(spin_margin.value())

        def _compute():
            return compute_molden_esp_grid(self.current_molden_path, nx=n_pts, ny=n_pts, nz=n_pts, margin_bohr=margin_b)

        def _on_done(esp_vol):
            if esp_vol is None:
                QtWidgets.QMessageBox.warning(self, "ESP Calculation", "Could not calculate ESP from current Molden file.")
                return

            self.current_vol_data = esp_vol
            # Set chemical ESP lobe colors: Blue (+V electrophilic) and Red (-V nucleophilic)
            self.pos_color = (0.15, 0.50, 0.98) # Vibrant Sky Blue (+V)
            self.neg_color = (0.95, 0.15, 0.15) # Vibrant Crimson Red (-V)
            self.btn_pos_col.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold;")
            self.btn_neg_col.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold;")

            # Auto-scale suggested isovalue based on ESP range
            esp_data = esp_vol.grid_data
            min_v = float(np.min(esp_data))
            max_v = float(np.max(esp_data))

            # Choose sensible default isovalue within the grid's range
            if min_v < 0 < max_v:
                sugg_iso = min(abs(min_v), abs(max_v)) * 0.35
            elif min_v >= 0:
                sugg_iso = min_v + (max_v - min_v) * 0.25
            else:
                sugg_iso = max_v - (max_v - min_v) * 0.25

            sugg_iso = float(np.clip(sugg_iso, 0.001, 10.0))

            # Expand range for ESP
            self.spin_iso.setRange(-50.0, 50.0)
            self.slider_iso.setRange(1, 2000)

            self.spin_iso.blockSignals(True)
            self.spin_iso.setValue(sugg_iso)
            self.spin_iso.blockSignals(False)
            self.slider_iso.blockSignals(True)
            self.slider_iso.setValue(int(round(sugg_iso * 1000.0)))
            self.slider_iso.blockSignals(False)

            self.lbl_info.setText(
                f"<b>{esp_vol.title}</b>: Evaluated exact ESP grid ({esp_vol.voxels_x}×{esp_vol.voxels_y}×{esp_vol.voxels_z}). "
                f"<span style='color: #2563eb;'>Positive (+V) = Blue</span>, <span style='color: #dc2626;'>Negative (-V) = Red</span>. "
                f"Min: {min_v:.4f} a.u., Max: {max_v:.4f} a.u."
            )
            self._update_surface()

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing Electrostatic Potential (ESP) requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "ESP Calculation Error", f"Failed to compute ESP:\n{e}")

        run_in_background(
            self,
            "ESP Calculation",
            "Evaluating exact molecular electrostatic potential integrals (PySCF)...",
            _compute,
            _on_done,
            _on_error,
        )

    def _on_compute_spin_density(self):
        """Compute open-shell spin density 3D grid and display isosurfaces."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "Spin Density", "Please load a .molden file first.")
            return

        n_pts = self._get_grid_pts(40)

        def _compute():
            return compute_molden_spin_density_grid(self.current_molden_path, nx=n_pts, ny=n_pts, nz=n_pts)

        def _on_done(spin_vol):
            if spin_vol is None:
                QtWidgets.QMessageBox.warning(self, "Spin Density", "Could not calculate spin density from current Molden file.")
                return

            self.current_vol_data = spin_vol
            # Set default isovalue for spin density
            self.spin_iso.blockSignals(True)
            self.spin_iso.setValue(0.005)
            self.spin_iso.blockSignals(False)
            self.slider_iso.blockSignals(True)
            self.slider_iso.setValue(5)
            self.slider_iso.blockSignals(False)

            self.lbl_info.setText(
                f"<b>{spin_vol.title}</b>: Evaluated exact spin density grid ({spin_vol.voxels_x}×{spin_vol.voxels_y}×{spin_vol.voxels_z}). "
                f"Min: {float(np.min(spin_vol.grid_data)):.5f}, Max: {float(np.max(spin_vol.grid_data)):.5f}"
            )
            self._update_surface()

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing Spin Density requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "Spin Density Error", f"Failed to compute spin density:\n{e}")

        run_in_background(self, "Spin Density", "Evaluating spin density grid (PySCF)...", _compute, _on_done, _on_error)

    def _on_surface_style_changed(self):
        style = self.combo_surface_style.currentData() or "Solid"
        if self.viewer:
            self.viewer.surface_mesh_style = style
            self.viewer.update()

    def _on_compute_esp_mapped_density(self):
        """Compute ESP-mapped electron density isosurface and update 3D viewer."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "ESP-Mapped Density", "Please load a .molden file first.")
            return

        # Prompt for colormap and range options
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("3D ESP-Mapped Electron Density Settings")
        dlg.resize(520, 480)
        vbox = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        form.setSpacing(10)

        combo_cmap = QtWidgets.QComboBox()
        combo_cmap.addItem("bwr_r (Red = Negative δ−, White = Neutral, Blue = Positive δ+)", "bwr_r")
        combo_cmap.addItem("RdBu (Red = Negative δ−, Light Gray = Neutral, Dark Blue = Positive δ+)", "RdBu")
        combo_cmap.addItem("coolwarm_r (Balanced Soft Red δ− to Blue δ+)", "coolwarm_r")
        combo_cmap.addItem("jet_r (Rainbow / Multiwfn Red δ− to Blue δ+)", "jet_r")
        combo_cmap.addItem("Spectral (Spectral Rainbow Red δ− to Violet δ+)", "Spectral")
        form.addRow("Colormap Palette:", combo_cmap)

        # Electron Density Isovalue: Slider + Spinbox pair
        dens_layout = QtWidgets.QHBoxLayout()
        slider_dens = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_dens.setRange(1, 100)  # 0.0001 to 0.0100 a.u. (step 0.0001)
        slider_dens.setValue(10)  # default 0.0010
        spin_dens = QtWidgets.QDoubleSpinBox()
        spin_dens.setRange(0.0001, 0.0500)
        spin_dens.setDecimals(4)
        spin_dens.setSingleStep(0.0005)
        spin_dens.setValue(0.0010)
        spin_dens.setSuffix(" a.u.")
        slider_dens.valueChanged.connect(lambda v: spin_dens.setValue(v / 10000.0))
        spin_dens.valueChanged.connect(lambda v: slider_dens.setValue(int(round(v * 10000.0))))
        dens_layout.addWidget(slider_dens, stretch=1)
        dens_layout.addWidget(spin_dens)
        form.addRow("Density Isovalue (ρ):", dens_layout)

        # Color Scaling Range: Auto or Slider/Spinbox Manual Limit
        chk_auto_range = QtWidgets.QCheckBox("Auto-scale symmetric potential range [-V_lim, +V_lim] (Recommended)")
        chk_auto_range.setChecked(True)
        form.addRow("Color Scaling:", chk_auto_range)

        range_layout = QtWidgets.QHBoxLayout()
        slider_manual_range = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_manual_range.setRange(5, 500)  # 0.005 to 0.500 a.u.
        slider_manual_range.setValue(50)  # default 0.050
        slider_manual_range.setEnabled(False)
        spin_manual_range = QtWidgets.QDoubleSpinBox()
        spin_manual_range.setRange(0.005, 0.500)
        spin_manual_range.setDecimals(3)
        spin_manual_range.setSingleStep(0.005)
        spin_manual_range.setValue(0.050)
        spin_manual_range.setPrefix("± ")
        spin_manual_range.setSuffix(" a.u.")
        spin_manual_range.setEnabled(False)
        slider_manual_range.valueChanged.connect(lambda v: spin_manual_range.setValue(v / 1000.0))
        spin_manual_range.valueChanged.connect(lambda v: slider_manual_range.setValue(int(round(v * 1000.0))))
        chk_auto_range.toggled.connect(lambda chk: (slider_manual_range.setEnabled(not chk), spin_manual_range.setEnabled(not chk)))
        range_layout.addWidget(slider_manual_range, stretch=1)
        range_layout.addWidget(spin_manual_range)
        form.addRow("Manual Potential Limit:", range_layout)

        # Surface Opacity: Slider + Spinbox pair
        opacity_layout = QtWidgets.QHBoxLayout()
        slider_op = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_op.setRange(10, 100)
        slider_op.setValue(int(round((self.slider_opacity.value() if hasattr(self, "slider_opacity") else 85))))
        spin_op = QtWidgets.QSpinBox()
        spin_op.setRange(10, 100)
        spin_op.setValue(slider_op.value())
        spin_op.setSuffix(" %")
        slider_op.valueChanged.connect(spin_op.setValue)
        spin_op.valueChanged.connect(slider_op.setValue)
        opacity_layout.addWidget(slider_op, stretch=1)
        opacity_layout.addWidget(spin_op)
        form.addRow("Surface Opacity:", opacity_layout)

        # Evaluation Box Size Margin (Padding beyond outer atoms in Bohr): Slider + Spinbox pair
        margin_layout = QtWidgets.QHBoxLayout()
        slider_margin = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_margin.setRange(15, 100)  # 1.5 to 10.0 Bohr
        slider_margin.setValue(35)  # 3.5 Bohr default
        spin_margin = QtWidgets.QDoubleSpinBox()
        spin_margin.setRange(1.5, 12.0)
        spin_margin.setDecimals(1)
        spin_margin.setSingleStep(0.5)
        spin_margin.setValue(3.5)
        spin_margin.setSuffix(" a₀")
        slider_margin.valueChanged.connect(lambda v: spin_margin.setValue(v / 10.0))
        spin_margin.valueChanged.connect(lambda v: slider_margin.setValue(int(round(v * 10.0))))
        margin_layout.addWidget(slider_margin, stretch=1)
        margin_layout.addWidget(spin_margin)
        form.addRow("Box Size Margin:", margin_layout)

        # 3D Grid Resolution: Slider + Spinbox pair (20³ to 90³)
        grid_layout = QtWidgets.QHBoxLayout()
        default_pts = self._get_grid_pts(40)
        slider_grid = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider_grid.setRange(20, 90)
        slider_grid.setValue(default_pts)
        spin_grid = QtWidgets.QSpinBox()
        spin_grid.setRange(20, 100)
        spin_grid.setSingleStep(5)
        spin_grid.setValue(default_pts)
        spin_grid.setSuffix(" voxels/axis")
        slider_grid.valueChanged.connect(spin_grid.setValue)
        spin_grid.valueChanged.connect(slider_grid.setValue)
        grid_layout.addWidget(slider_grid, stretch=1)
        grid_layout.addWidget(spin_grid)
        form.addRow("Grid Resolution:", grid_layout)

        chk_extrema = QtWidgets.QCheckBox("Detect & label surface extrema (V_min nucleophilic, V_max electrophilic)")
        chk_extrema.setChecked(True)
        form.addRow("Surface Extrema:", chk_extrema)

        vbox.addLayout(form)
        vbox.addStretch()

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        vbox.addWidget(btns)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        cmap_clean = combo_cmap.currentData() or "bwr_r"
        dens_iso = float(spin_dens.value())
        selected_margin = float(spin_margin.value())
        selected_opacity = float(slider_op.value()) / 100.0
        do_extrema = chk_extrema.isChecked()
        if chk_auto_range.isChecked():
            esp_lim_min = None
            esp_lim_max = None
        else:
            val_lim = float(spin_manual_range.value())
            esp_lim_min = -val_lim
            esp_lim_max = val_lim

        n_pts = int(spin_grid.value())

        def _compute():
            return compute_molden_esp_mapped_density(
                self.current_molden_path,
                density_isovalue=dens_iso,
                colormap_name=cmap_clean,
                esp_min=esp_lim_min,
                esp_max=esp_lim_max,
                find_extrema=do_extrema,
                margin_bohr=selected_margin,
                nx=n_pts,
                ny=n_pts,
                nz=n_pts,
            )

        def _on_done(mesh):
            if mesh is None:
                QtWidgets.QMessageBox.warning(self, "ESP-Mapped Density", "Could not generate ESP-mapped surface from current Molden file.")
                return

            opacity = selected_opacity
            if hasattr(self, "slider_opacity"):
                self.slider_opacity.blockSignals(True)
                self.slider_opacity.setValue(int(round(opacity * 100.0)))
                self.slider_opacity.blockSignals(False)
            self.surface_updated.emit([mesh], opacity)
            n_triangles = len(mesh.faces)
            n_ext = len(mesh.extrema_points) if mesh.extrema_points else 0
            r_str = f"Range: [{mesh.esp_range[0]:+.3f}, {mesh.esp_range[1]:+.3f}] a.u." if getattr(mesh, "esp_range", None) else ""
            self.lbl_stats.setText(f"Generated ESP-Mapped Density Envelope: {n_triangles:,} triangles, {n_ext} surface extrema. {r_str}")
            self.lbl_info.setText(
                f"<b>ESP-Mapped Isodensity Surface ({cmap_clean}, ρ={dens_iso}):</b> "
                f"<span style='color: #ef4444;'>Red = Negative (Nucleophilic δ−)</span>, "
                f"<span style='color: #38bdf8;'>Blue = Positive (Electrophilic δ+)</span>. {r_str} "
                f"Found {n_ext} local extrema (V_min / V_max) marked on surface."
            )

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing ESP-mapped surface requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "ESP-Mapped Density Error", f"Failed to generate surface:\n{e}")

        run_in_background(
            self,
            "ESP-Mapped Density",
            "Computing electron density and mapping electrostatic potential (PySCF)...",
            _compute,
            _on_done,
            _on_error,
        )

    def _on_compute_nci_rdg(self):
        """Compute Non-Covalent Interactions (NCI / RDG) isosurface."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "NCI / RDG Analysis", "Please load a .molden file first.")
            return

        n_pts = self._get_grid_pts(36)

        def _compute():
            return compute_molden_nci_rdg(self.current_molden_path, rdg_isovalue=0.5, nx=n_pts, ny=n_pts, nz=n_pts)

        def _on_done(mesh):
            if mesh is None:
                QtWidgets.QMessageBox.warning(self, "NCI / RDG Analysis", "Could not generate NCI/RDG surface from current Molden file.")
                return

            opacity = self.slider_opacity.value() / 100.0
            self.surface_updated.emit([mesh], opacity)
            n_triangles = len(mesh.faces)
            self.lbl_stats.setText(f"Generated NCI/RDG Isosurface (s=0.5): {n_triangles:,} triangles.")
            self.lbl_info.setText(
                "<b>NCI / Reduced Density Gradient:</b> Blue = H-Bonding / Attractive, Green = van der Waals dispersion, Red = Steric clash."
            )

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing NCI/RDG requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "NCI / RDG Error", f"Failed to compute NCI/RDG:\n{e}")

        run_in_background(self, "NCI / RDG Analysis", "Computing reduced density gradient (RDG) and Hessian (PySCF)...", _compute, _on_done, _on_error)

    def _on_open_nci_plot(self):
        """Open interactive 2D Reduced Density Gradient (RDG) vs sign(λ₂)ρ scatter plot."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "NCI 2D Plot", "Please load a .molden file first.")
            return
        from .dialog_nci_plot import NCIRdgPlotDialog
        dlg = NCIRdgPlotDialog(self, molden_path=self.current_molden_path)
        dlg.exec()

    def _on_compute_alie(self):
        """Compute Average Local Ionization Energy (ALIE) mapped density surface."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "ALIE Analysis", "Please load a .molden file first.")
            return

        n_pts = self._get_grid_pts(38)

        def _compute():
            return compute_molden_alie_mapped_density(self.current_molden_path, density_isovalue=0.002, nx=n_pts, ny=n_pts, nz=n_pts)

        def _on_done(mesh):
            if mesh is None:
                QtWidgets.QMessageBox.warning(self, "ALIE Analysis", "Could not generate ALIE surface from current Molden file.")
                return

            opacity = self.slider_opacity.value() / 100.0
            self.surface_updated.emit([mesh], opacity)
            n_triangles = len(mesh.faces)
            n_min = len(mesh.extrema_points) if mesh.extrema_points else 0
            self.lbl_stats.setText(f"Generated ALIE Surface: {n_triangles:,} triangles, {n_min} reactive minima.")
            self.lbl_info.setText(
                f"<b>Average Local Ionization Energy (ALIE):</b> Red/Orange = Low ALIE (Most reactive electrophilic / radical attack sites). Found {n_min} minima."
            )

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing ALIE requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "ALIE Error", f"Failed to compute ALIE:\n{e}")

        run_in_background(self, "ALIE Analysis", "Computing average local ionization energy surface (PySCF)...", _compute, _on_done, _on_error)

    def _on_compute_chelpg(self):
        """Compute CHELPG atomic partial charges from Molden and apply to 3D viewer."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "CHELPG Charges", "Please load a .molden file first.")
            return

        def _compute():
            return compute_molden_chelpg_charges(self.current_molden_path)

        def _on_done(res):
            if res is None:
                QtWidgets.QMessageBox.warning(self, "CHELPG Charges", "Could not calculate CHELPG charges from current Molden file.")
                return

            charges, rmse = res
            if self.viewer:
                self.viewer.custom_charges = charges
                self.viewer.show_charges = True
                self.viewer.update()

            msg_lines = [f"Calculated CHELPG partial charges (RMSE: {rmse:.5f} a.u.):\n"]
            syms = self.current_molden_data.symbols if self.current_molden_data else []
            for i, q in enumerate(charges):
                s = syms[i] if i < len(syms) else f"Atom {i+1}"
                msg_lines.append(f"  {i+1:>3}: {s:<4} {q:+.4f} e")
            msg_lines.append(f"\nSum of charges: {charges.sum():+.4f} e")

            QtWidgets.QMessageBox.information(self, "CHELPG Partial Charges", "\n".join(msg_lines))

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing CHELPG partial charges requires the 'pyscf' package:\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "CHELPG Error", f"Failed to compute CHELPG charges:\n{e}")

        run_in_background(self, "CHELPG Charges", "Fitting CHELPG electrostatic potential charges (PySCF)...", _compute, _on_done, _on_error)

    def _on_open_multiwfn_suite(self):
        """Open the multi-feature Multiwfn Electronic Structure Dialog."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "Multiwfn Suite", "Please load a .molden file first.")
            return

        from .dialog_multiwfn import MultiwfnAnalysisDialog
        dlg = MultiwfnAnalysisDialog(self, molden_path=self.current_molden_path, viewer=self.viewer)
        dlg.exec()

    def _on_compute_elf(self):
        """Compute Becke-Edgecombe Electron Localization Function (ELF) 3D grid."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "ELF Calculation", "Please load a .molden file first.")
            return

        n_pts = self._get_grid_pts(40)

        def _compute():
            return compute_molden_elf_grid(self.current_molden_path, nx=n_pts, ny=n_pts, nz=n_pts)

        def _on_done(elf_vol):
            if elf_vol is None:
                QtWidgets.QMessageBox.warning(self, "ELF Calculation", "Could not calculate ELF from current Molden file.")
                return

            self.current_vol_data = elf_vol
            # Standard chemical ELF isovalues are around 0.80 - 0.85 (showing covalent bonds and lone pairs)
            self.spin_iso.blockSignals(True)
            self.spin_iso.setValue(0.82)
            self.spin_iso.blockSignals(False)
            self.slider_iso.blockSignals(True)
            self.slider_iso.setValue(min(500, int(0.82 * 1000.0)))
            self.slider_iso.blockSignals(False)

            self.lbl_info.setText(
                f"<b>{elf_vol.title}</b>: Evaluated exact Becke-Edgecombe ELF grid ({elf_vol.voxels_x}×{elf_vol.voxels_y}×{elf_vol.voxels_z}). "
                f"Isovalue set to 0.82 (Bonds / Lone Pairs)."
            )
            self._update_surface()

        def _on_error(e):
            if isinstance(e, ImportError):
                QtWidgets.QMessageBox.warning(
                    self,
                    "PySCF Required",
                    f"Computing Electron Localization Function (ELF) requires 'pyscf':\n\n{e}\n\n"
                    "Install it with:  pip install pyscf  (or: pip install .[qm])"
                )
            else:
                QtWidgets.QMessageBox.critical(self, "ELF Calculation Error", f"Failed to compute ELF:\n{e}")

        run_in_background(self, "ELF Calculation", "Computing Becke-Edgecombe ELF grid (PySCF)...", _compute, _on_done, _on_error)

    def _on_compute_localized_orbitals(self):
        """Compute localized molecular orbitals (Boys or Pipek-Mezey) from Molden file."""
        if not self.current_molden_path:
            QtWidgets.QMessageBox.information(self, "Localized Orbitals", "Please load a .molden file first.")
            return

        method, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Orbital Localization Method",
            "Select Localization Procedure:",
            ["Boys (Foster-Boys - minimizes spatial extent)", "Pipek-Mezey (PM - preserves σ/π separation)"],
            0,
            False,
        )
        if not ok:
            return

        loc_method = "Boys" if "Boys" in method else "PipekMezey"

        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            res = compute_molden_localized_orbitals(self.current_molden_path, method=loc_method)
            if res is None:
                QtWidgets.QMessageBox.warning(self, "Localized Orbitals", "Could not localize orbitals from current Molden file.")
                return

            self.current_lmo_orbitals = res
            lmo_coeff = res["lmo_coeff"]
            centroids = res["centroids"]
            n_lmo = lmo_coeff.shape[1]

            # Populate combo box with LMOs
            self.combo_mo.blockSignals(True)
            self.combo_mo.clear()
            for i in range(n_lmo):
                c = centroids[i] if i < len(centroids) else [0.0, 0.0, 0.0]
                self.combo_mo.addItem(f"LMO #{i+1} ({loc_method}) [Centroid: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})]", f"LMO_{i}")
            self.combo_mo.blockSignals(False)

            self.lbl_info.setText(
                f"<b>Localized MOs ({loc_method}):</b> Generated {n_lmo} localized orbitals. "
                "Select any LMO from the dropdown above to view 3D lobes."
            )
            self.combo_mo.setCurrentIndex(0)
            self._on_mo_selected(0)
        except ImportError as e:
            QtWidgets.QMessageBox.warning(
                self,
                "PySCF Required",
                f"Orbital localization requires 'pyscf':\n\n{e}\n\n"
                "Install it with:  pip install pyscf  (or: pip install .[qm])"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Localization Error", f"Failed to compute localized orbitals:\n{e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _on_open_esp_2d(self):
        """Open 2D Plane Slice & Contour Slicer Dialog (from .molden or loaded .cube grid)."""
        if not self.current_molden_path and not self.current_vol_data:
            QtWidgets.QMessageBox.information(
                self, "2D Slicer", "Please load a .molden file or a .cube volumetric grid first."
            )
            return

        try:
            if self.current_molden_path:
                try:
                    import pyscf
                    dlg = ESP2DContourDialog(self, molden_path=self.current_molden_path)
                except ImportError:
                    if self.current_vol_data:
                        dlg = ESP2DContourDialog(self, vol_data=self.current_vol_data)
                    else:
                        QtWidgets.QMessageBox.warning(
                            self,
                            "PySCF Required",
                            "Direct wavefunction 2D ESP slicing requires the 'pyscf' package.\n\n"
                            "Install it with:  pip install pyscf  (or load a .cube file)."
                        )
                        return
            else:
                dlg = ESP2DContourDialog(self, vol_data=self.current_vol_data)

            dlg.exec()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "2D Slicer Error", f"Could not launch 2D Slicer:\n{e}")
            QtWidgets.QMessageBox.critical(self, "2D ESP Slicer Error", f"Could not launch 2D ESP Slicer:\n{e}")

    def _on_mo_selected(self, idx: int):
        if idx < 0:
            return

        n_pts = self._get_grid_pts(40)

        # Check if we are currently browsing Localized Molecular Orbitals (LMOs)
        if self.current_lmo_orbitals is not None and self.current_molden_path:
            lmo_coeff = self.current_lmo_orbitals["lmo_coeff"]
            if idx < lmo_coeff.shape[1]:
                def _eval_lmo():
                    return evaluate_lmo_orbital_grid(
                        self.current_molden_path,
                        lmo_coeff,
                        lmo_index=idx,
                        nx=n_pts,
                        ny=n_pts,
                        nz=n_pts,
                    )
                def _on_lmo_done(vol):
                    if vol is not None:
                        self.current_vol_data = vol
                        self.lbl_info.setText(
                            f"<b>{vol.title}</b>: Evaluated exact localized orbital GTO grid "
                            f"({vol.voxels_x}×{vol.voxels_y}×{vol.voxels_z})."
                        )
                        self._update_surface()
                run_in_background(self, "Orbital Evaluation", f"Evaluating localized orbital #{idx+1}...", _eval_lmo, _on_lmo_done, lambda e: None)
                return

        if not self.current_molden_data or idx >= len(self.current_molden_data.orbitals):
            return

        # Attempt exact analytic GTO evaluation first if basis is available
        if self.current_molden_path:
            def _eval_mo():
                vol = evaluate_molden_orbital_grid(self.current_molden_path, orbital_index=idx, nx=n_pts, ny=n_pts, nz=n_pts)
                decomp = compute_molden_orbital_decomposition(self.current_molden_path, orbital_index=idx) if vol else []
                return vol, decomp

            def _on_mo_done(res):
                vol, decomp = res
                if vol is not None:
                    self.current_vol_data = vol
                    # Auto-scale isovalue if current spin_iso exceeds the orbital's max amplitude (e.g. diffuse positron MOs)
                    max_abs = float(np.max(np.abs(vol.grid_data)))
                    if max_abs > 1e-5 and self.spin_iso.value() >= max_abs * 0.92:
                        sugg_iso = round(max_abs * 0.45, 4)
                        self.spin_iso.blockSignals(True)
                        self.spin_iso.setValue(sugg_iso)
                        self.spin_iso.blockSignals(False)
                        self.slider_iso.blockSignals(True)
                        self.slider_iso.setValue(max(1, int(round(sugg_iso * 1000.0))))
                        self.slider_iso.blockSignals(False)
                    decomp_str = ""
                    if decomp:
                        top_items = [f"{d['symbol']}{d['atom_idx']+1}: {d['percentage']:.1f}%" for d in decomp[:4] if abs(d['percentage']) >= 2.0]
                        if top_items:
                            decomp_str = f"<br><span style='color: #fbbf24;'>Composition:</span> {', '.join(top_items)}"

                    self.lbl_info.setText(
                        f"<b>{vol.title}</b>: Evaluated exact quantum mechanical GTO grid "
                        f"({vol.voxels_x}×{vol.voxels_y}×{vol.voxels_z}) from Molden basis.{decomp_str}"
                    )
                    self._update_surface()
                else:
                    mo = self.current_molden_data.orbitals[idx]
                    self._generate_synthetic_mo_grid(mo)

            run_in_background(self, "Orbital Evaluation", f"Evaluating MO #{idx+1} on 3D grid...", _eval_mo, _on_mo_done, lambda e: None)
            return

        # Fallback to synthetic basis representation if GTO section is missing or unparseable
        mo = self.current_molden_data.orbitals[idx]
        self._generate_synthetic_mo_grid(mo)

    def _generate_synthetic_mo_grid(self, mo: MoldenMO):
        """Construct a 3D volumetric grid from atom coordinates and atomic orbital coefficients."""
        coords = self.current_molden_data.coords
        symbols = self.current_molden_data.symbols
        if len(coords) == 0:
            return

        # Voxel bounding box around molecule
        pad = 3.5
        min_c = np.min(coords, axis=0) - pad
        max_c = np.max(coords, axis=0) + pad
        nx, ny, nz = 36, 36, 36

        step = (max_c - min_c) / np.array([nx - 1, ny - 1, nz - 1])
        x = np.linspace(min_c[0], max_c[0], nx)
        y = np.linspace(min_c[1], max_c[1], ny)
        z = np.linspace(min_c[2], max_c[2], nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

        grid = np.zeros((nx, ny, nz), dtype=np.float32)

        # Approximate MO density by superposition of atomic center Gaussian basis lobes
        coeff_list = list(mo.coefficients.values())
        for a_idx, c_pos in enumerate(coords):
            # Distance squared from atom
            r2 = (X - c_pos[0])**2 + (Y - c_pos[1])**2 + (Z - c_pos[2])**2
            c_val = coeff_list[a_idx % len(coeff_list)] if coeff_list else (1.0 if a_idx == 0 else -0.7)
            sigma = 0.95
            grid += c_val * np.exp(-0.5 * r2 / (sigma**2)).astype(np.float32)

        vol = VolumetricData(
            title=f"MO {mo.sym} (E={mo.energy:.3f})",
            comment="Synthesized from Molden MO coefficients",
            symbols=symbols,
            atom_coords=coords,
            origin=min_c,
            voxels_x=nx,
            voxels_y=ny,
            voxels_z=nz,
            step_x=np.array([step[0], 0.0, 0.0]),
            step_y=np.array([0.0, step[1], 0.0]),
            step_z=np.array([0.0, 0.0, step[2]]),
            grid_data=grid,
            is_orbital=True,
        )
        self.current_vol_data = vol
        self._update_surface()

    def _on_slider_iso_changed(self, val: int):
        iso = val / 1000.0
        self.spin_iso.blockSignals(True)
        self.spin_iso.setValue(iso)
        self.spin_iso.blockSignals(False)
        self._schedule_surface_update()

    def _on_spin_iso_changed(self, val: float):
        s_val = int(round(val * 1000.0))
        self.slider_iso.blockSignals(True)
        self.slider_iso.setValue(s_val)
        self.slider_iso.blockSignals(False)
        self._schedule_surface_update()

    def _schedule_surface_update(self, delay_ms: int = 150):
        """Debounce rapid isovalue changes so dragging the slider does not spawn many meshing threads."""
        timer = getattr(self, "_iso_debounce_timer", None)
        if timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._update_surface)
            self._iso_debounce_timer = timer
        timer.start(delay_ms)

    def _on_opacity_changed(self, val: int):
        self.lbl_opacity.setText(f"{val}%")
        self._update_surface()

    def _on_pick_pos_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*self.pos_color), self, "Select (+) Lobe Color")
        if col.isValid():
            self.pos_color = (col.redF(), col.greenF(), col.blueF())
            lum = 0.299 * col.redF() + 0.587 * col.greenF() + 0.114 * col.blueF()
            txt_col = "#000000" if lum > 0.5 else "#ffffff"
            self.btn_pos_col.setStyleSheet(f"background-color: {col.name()}; color: {txt_col}; font-weight: bold;")
            self._update_surface()

    def _on_pick_neg_color(self):
        col = QtWidgets.QColorDialog.getColor(QtGui.QColor.fromRgbF(*self.neg_color), self, "Select (-) Lobe Color")
        if col.isValid():
            self.neg_color = (col.redF(), col.greenF(), col.blueF())
            lum = 0.299 * col.redF() + 0.587 * col.greenF() + 0.114 * col.blueF()
            txt_col = "#000000" if lum > 0.5 else "#ffffff"
            self.btn_neg_col.setStyleSheet(f"background-color: {col.name()}; color: {txt_col}; font-weight: bold;")
            self._update_surface()

    def _push_surface_state(self, new_meshes: list, new_opacity: float):
        """Record current viewer active isosurfaces before applying new meshes for undo/redo."""
        current_meshes = list(getattr(self.viewer, "active_isosurfaces", [])) if self.viewer else []
        current_opacity = float(getattr(self.viewer, "surface_opacity", 0.8)) if self.viewer else 0.8
        self._surface_undo_stack.append((current_meshes, current_opacity))
        if len(self._surface_undo_stack) > 25:
            self._surface_undo_stack.pop(0)
        self._surface_redo_stack.clear()
        self._update_undo_redo_buttons()
        self.surface_updated.emit(new_meshes, new_opacity)

    def _update_undo_redo_buttons(self):
        if hasattr(self, "btn_undo_surf"):
            self.btn_undo_surf.setEnabled(len(self._surface_undo_stack) > 0)
        if hasattr(self, "btn_redo_surf"):
            self.btn_redo_surf.setEnabled(len(self._surface_redo_stack) > 0)

    def _on_undo_surface(self):
        if not self._surface_undo_stack:
            return
        current_meshes = list(getattr(self.viewer, "active_isosurfaces", [])) if self.viewer else []
        current_opacity = float(getattr(self.viewer, "surface_opacity", 0.8)) if self.viewer else 0.8
        self._surface_redo_stack.append((current_meshes, current_opacity))
        prev_meshes, prev_opacity = self._surface_undo_stack.pop()
        self._update_undo_redo_buttons()
        self.surface_updated.emit(prev_meshes, prev_opacity)
        self.lbl_stats.setText(f"Surface restored from undo: {len(prev_meshes)} active mesh layer(s).")

    def _on_redo_surface(self):
        if not self._surface_redo_stack:
            return
        current_meshes = list(getattr(self.viewer, "active_isosurfaces", [])) if self.viewer else []
        current_opacity = float(getattr(self.viewer, "surface_opacity", 0.8)) if self.viewer else 0.8
        self._surface_undo_stack.append((current_meshes, current_opacity))
        next_meshes, next_opacity = self._surface_redo_stack.pop()
        self._update_undo_redo_buttons()
        self.surface_updated.emit(next_meshes, next_opacity)
        self.lbl_stats.setText(f"Surface restored from redo: {len(next_meshes)} active mesh layer(s).")

    def _on_clear_surface(self):
        self._push_surface_state([], 1.0)
        self.lbl_stats.setText("Isosurface cleared.")

    def _update_surface(self):
        if not self.chk_visible.isChecked() or not self.current_vol_data:
            self.surface_updated.emit([], 1.0)
            self.lbl_stats.setText("Surface hidden.")
            return

        iso = self.spin_iso.value()
        opacity = self.slider_opacity.value() / 100.0
        quality_scale = float(self.combo_mesh_res.currentData() or 1.0)
        vol_target = self.current_vol_data
        pos_col = self.pos_color
        neg_col = self.neg_color

        # If a background worker is already running, wait or disconnect
        if hasattr(self, "_active_mc_thread") and self._active_mc_thread and self._active_mc_thread.isRunning():
            try:
                self._active_mc_thread.mesh_ready.disconnect()
                self._active_mc_thread.mesh_failed.disconnect()
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        if hasattr(self, "progress_bar"):
            self.progress_bar.setVisible(True)
        self.lbl_stats.setText("Computing Marching Cubes isosurface in background...")
        if self.parent() and hasattr(self.parent(), "statusBar") and self.parent().statusBar():
            self.parent().statusBar().showMessage("Computing orbital isosurfaces in background...", 3000)

        class _MCSurfaceThread(QtCore.QThread):
            mesh_ready = QtCore.pyqtSignal(list, str)
            mesh_failed = QtCore.pyqtSignal(str)

            def run(self):
                try:
                    target = vol_target
                    if quality_scale > 1.01:
                        import scipy.ndimage
                        zoomed_grid = scipy.ndimage.zoom(target.grid_data, zoom=quality_scale, order=2).astype(np.float32)
                        target = VolumetricData(
                            title=target.title,
                            comment=target.comment,
                            symbols=target.symbols,
                            atom_coords=target.atom_coords,
                            origin=target.origin,
                            voxels_x=zoomed_grid.shape[0],
                            voxels_y=zoomed_grid.shape[1],
                            voxels_z=zoomed_grid.shape[2],
                            step_x=target.step_x / quality_scale,
                            step_y=target.step_y / quality_scale,
                            step_z=target.step_z / quality_scale,
                            grid_data=zoomed_grid,
                            is_orbital=target.is_orbital,
                            orbital_index=target.orbital_index,
                        )

                    if target.is_orbital:
                        meshes = generate_orbital_surfaces(
                            target,
                            isovalue=iso,
                            pos_color=pos_col,
                            neg_color=neg_col,
                        )
                    else:
                        m = extract_isosurface(target, isovalue=iso, color=pos_col)
                        meshes = [m] if m is not None else []

                    n_triangles = sum(len(m.faces) for m in meshes)
                    n_verts = sum(len(m.vertices) for m in meshes)
                    stat = f"Extracted {len(meshes)} lobe(s): {n_triangles:,} triangles ({n_verts:,} vertices)."
                    self.mesh_ready.emit(meshes, stat)
                except Exception as e:
                    self.mesh_failed.emit(str(e))

        def on_done(meshes, stat_text):
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
            if self.chk_overlay_mode.isChecked() and self.viewer and getattr(self.viewer, "active_isosurfaces", None):
                combined = list(self.viewer.active_isosurfaces) + list(meshes)
            else:
                combined = meshes
            self._push_surface_state(combined, opacity)
            self.lbl_stats.setText(stat_text)
            if self.parent() and hasattr(self.parent(), "statusBar") and self.parent().statusBar():
                self.parent().statusBar().showMessage(stat_text, 4000)

        def on_error(err_msg):
            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
            self.lbl_stats.setText(f"Marching cubes error: {err_msg}")

        self._active_mc_thread = _MCSurfaceThread(self)
        self._active_mc_thread.mesh_ready.connect(on_done)
        self._active_mc_thread.mesh_failed.connect(on_error)
        self._active_mc_thread.start()
