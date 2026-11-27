"""
Preferences dialog for AgyMolden.
Allows configuring default viewport colors, background modes, rendering options,
and default calculation parameters with persistence via QSettings.
"""

from PyQt6 import QtCore, QtGui, QtWidgets


class PreferencesDialog(QtWidgets.QDialog):
    """User preferences dialog."""

    def __init__(self, parent: QtWidgets.QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("⚙️ AgyMolden Preferences")
        self.resize(520, 380)

        from .session_state import get_settings
        self.settings = get_settings("Preferences")
        self._init_ui()
        self._load_settings()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        tabs = QtWidgets.QTabWidget()

        # Tab 1: 3D Viewport & Rendering
        tab_view = QtWidgets.QWidget()
        v_layout = QtWidgets.QFormLayout(tab_view)
        v_layout.setSpacing(10)

        self.combo_bg = QtWidgets.QComboBox()
        self.combo_bg.addItem("White Background", "white")
        self.combo_bg.addItem("Dark Slate Background", "dark")
        v_layout.addRow("Default Background:", self.combo_bg)

        self.chk_labels = QtWidgets.QCheckBox("Show atom labels by default")
        v_layout.addRow("", self.chk_labels)

        self.chk_indices = QtWidgets.QCheckBox("Show atom indices by default")
        v_layout.addRow("", self.chk_indices)

        self.spin_msaa = QtWidgets.QComboBox()
        self.spin_msaa.addItem("4x MSAA (Smooth / Balanced)", 4)
        self.spin_msaa.addItem("8x MSAA (Ultra Sharp)", 8)
        self.spin_msaa.addItem("Disabled (Fastest / Low-end GPU)", 0)
        v_layout.addRow("Anti-Aliasing (MSAA):", self.spin_msaa)

        self.chk_modern_shaders = QtWidgets.QCheckBox("Enable Modern Impostor Shaders (Dual-Source Glossy)")
        self.chk_modern_shaders.setToolTip(
            "Render glossy, depth-correct sphere and cylinder impostors using GPU fragment shaders (ChimeraX/PyMOL style).\n"
            "Uncheck to keep 100% legacy OpenGL fixed-function rendering."
        )
        v_layout.addRow("", self.chk_modern_shaders)

        self.chk_depth_cue = QtWidgets.QCheckBox("Enable Screen-Space Depth Cueing & Cavity Ambient Occlusion")
        self.chk_depth_cue.setToolTip(
            "Apply soft contact shadowing in molecular cavities and atmospheric depth falloff (QuteMol/PyMOL style)\n"
            "to maximize 3D depth perception for macromolecules and complexes."
        )
        v_layout.addRow("", self.chk_depth_cue)

        self.combo_default_dpi = QtWidgets.QComboBox()
        self.combo_default_dpi.addItem("300 DPI (Standard Journal Submission)", 300)
        self.combo_default_dpi.addItem("600 DPI (High-Density Print / Line Art)", 600)
        self.combo_default_dpi.addItem("150 DPI (Web & Presentation)", 150)
        self.combo_default_dpi.addItem("72 DPI (Standard Screen)", 72)
        v_layout.addRow("Default Export DPI:", self.combo_default_dpi)

        self.combo_mouse_mode = QtWidgets.QComboBox()
        self.combo_mouse_mode.addItem("AgyMolden / Standard (Left: Rotate, Right: Pan, Wheel/Mid: Zoom)", "standard")
        self.combo_mouse_mode.addItem("GaussView Preset (Left: Rotate, Right: Zoom, Mid: Pan)", "gaussview")
        self.combo_mouse_mode.addItem("Chemcraft / PyMOL Preset (Left: Rotate, Right: Pan, Shift+Right: Zoom)", "chemcraft")
        v_layout.addRow("Mouse Navigation Style:", self.combo_mouse_mode)

        from .ui_scale import MAX_FONT_PT, MIN_FONT_PT, UI_SCALE_PRESETS

        self.spin_font_size = QtWidgets.QSpinBox()
        self.spin_font_size.setRange(MIN_FONT_PT, MAX_FONT_PT)
        self.spin_font_size.setSuffix(" pt")
        self.spin_font_size.setToolTip("Live UI font size across all menus, buttons, and dialogs (no restart needed)")
        v_layout.addRow("UI Font Size (Live):", self.spin_font_size)

        self.combo_ui_scale = QtWidgets.QComboBox()
        for sc in UI_SCALE_PRESETS:
            pct = int(round(sc * 100))
            if abs(sc - 1.0) < 1e-3:
                label = "100% (Default)"
            elif sc >= 1.5:
                label = f"{pct}% (High-DPI / Small Screen)"
            else:
                label = f"{pct}%"
            self.combo_ui_scale.addItem(label, sc)
        self.combo_ui_scale.setToolTip("Global Qt UI scale factor for icons, padding, and layouts (applied on next launch)")
        v_layout.addRow("Startup UI Scale (Restart):", self.combo_ui_scale)

        tabs.addTab(tab_view, "🖥️ 3D Viewport")

        # Tab 2: Calculation & Symmetry Defaults
        tab_calc = QtWidgets.QWidget()
        c_layout = QtWidgets.QFormLayout(tab_calc)
        c_layout.setSpacing(10)

        self.combo_default_engine = QtWidgets.QComboBox()
        self.combo_default_engine.addItem("ORCA (4.x / 5.x / 6.x)", "orca")
        self.combo_default_engine.addItem("Gaussian (G16 / G09)", "gaussian")
        self.combo_default_engine.addItem("Psi4", "psi4")
        self.combo_default_engine.addItem("PySCF (Python Script)", "pyscf")
        self.combo_default_engine.addItem("Q-Chem", "qchem")
        self.combo_default_engine.addItem("xTB (Tight-Binding)", "xtb")
        self.combo_default_engine.addItem("openLOWDIN (APMO)", "openlowdin")
        self.combo_default_engine.addItem("newBRORSEN", "newbrorsen")
        c_layout.addRow("Preferred Default QM Engine:", self.combo_default_engine)

        self.spin_cores = QtWidgets.QSpinBox()
        self.spin_cores.setRange(1, 128)
        self.spin_cores.setValue(1)
        c_layout.addRow("Default CPU Cores (%nprocs):", self.spin_cores)

        self.spin_mem = QtWidgets.QSpinBox()
        self.spin_mem.setRange(500, 512000)
        self.spin_mem.setSingleStep(1000)
        self.spin_mem.setValue(4000)
        self.spin_mem.setSuffix(" MB")
        c_layout.addRow("Default Memory Target:", self.spin_mem)

        self.spin_sym_tol = QtWidgets.QDoubleSpinBox()
        self.spin_sym_tol.setRange(0.01, 1.0)
        self.spin_sym_tol.setSingleStep(0.05)
        self.spin_sym_tol.setValue(0.15)
        self.spin_sym_tol.setSuffix(" Å")
        c_layout.addRow("Default Symmetry Tolerance:", self.spin_sym_tol)

        tabs.addTab(tab_calc, "⚡ Calculations")

        layout.addWidget(tabs)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QtWidgets.QPushButton("💾 Save Preferences")
        btn_save.setStyleSheet("background-color: #2563eb; color: #fff; font-weight: bold; padding: 6px 14px;")
        btn_save.clicked.connect(self._save_settings)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

    def _load_settings(self):
        bg = self.settings.value("viewport/background", "white")
        idx = self.combo_bg.findData(bg)
        if idx >= 0:
            self.combo_bg.setCurrentIndex(idx)

        self.chk_labels.setChecked(self.settings.value("viewport/show_labels", False, type=bool))
        self.chk_indices.setChecked(self.settings.value("viewport/show_indices", False, type=bool))

        msaa = self.settings.value("viewport/msaa", 4, type=int)
        idx_msaa = self.spin_msaa.findData(msaa)
        if idx_msaa >= 0:
            self.spin_msaa.setCurrentIndex(idx_msaa)

        self.chk_modern_shaders.setChecked(self.settings.value("viewport/modern_shaders", True, type=bool))
        self.chk_depth_cue.setChecked(self.settings.value("viewport/depth_cue", True, type=bool))

        dpi = self.settings.value("export/default_dpi", 300, type=int)
        idx_dpi = self.combo_default_dpi.findData(dpi)
        if idx_dpi >= 0:
            self.combo_default_dpi.setCurrentIndex(idx_dpi)

        engine = self.settings.value("calc/default_engine", "orca", type=str)
        idx_eng = self.combo_default_engine.findData(engine)
        if idx_eng >= 0:
            self.combo_default_engine.setCurrentIndex(idx_eng)

        mouse_m = self.settings.value("viewport/mouse_mode", "standard", type=str)
        idx_mouse = self.combo_mouse_mode.findData(mouse_m)
        if idx_mouse >= 0:
            self.combo_mouse_mode.setCurrentIndex(idx_mouse)

        self.spin_cores.setValue(self.settings.value("calc/nprocs", 1, type=int))
        self.spin_mem.setValue(self.settings.value("calc/memory", 4000, type=int))
        self.spin_sym_tol.setValue(self.settings.value("calc/sym_tol", 0.15, type=float))

        from .ui_scale import get_saved_font_size, get_saved_ui_scale

        self.spin_font_size.setValue(get_saved_font_size())
        saved_scale = get_saved_ui_scale()
        best_idx = 0
        best_diff = 999.0
        for i in range(self.combo_ui_scale.count()):
            val = float(self.combo_ui_scale.itemData(i))
            diff = abs(val - saved_scale)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        self.combo_ui_scale.setCurrentIndex(best_idx)

    def _save_settings(self):
        self.settings.setValue("viewport/background", self.combo_bg.currentData())
        self.settings.setValue("viewport/show_labels", self.chk_labels.isChecked())
        self.settings.setValue("viewport/show_indices", self.chk_indices.isChecked())
        self.settings.setValue("viewport/msaa", self.spin_msaa.currentData())
        self.settings.setValue("viewport/modern_shaders", self.chk_modern_shaders.isChecked())
        self.settings.setValue("viewport/depth_cue", self.chk_depth_cue.isChecked())
        self.settings.setValue("export/default_dpi", self.combo_default_dpi.currentData())
        self.settings.setValue("calc/default_engine", self.combo_default_engine.currentData())
        self.settings.setValue("viewport/mouse_mode", self.combo_mouse_mode.currentData())

        self.settings.setValue("calc/nprocs", self.spin_cores.value())
        self.settings.setValue("calc/memory", self.spin_mem.value())
        self.settings.setValue("calc/sym_tol", self.spin_sym_tol.value())

        from .ui_scale import apply_live_font_size, save_ui_scale

        apply_live_font_size(self.spin_font_size.value(), persist=True)
        save_ui_scale(float(self.combo_ui_scale.currentData()))

        # Apply immediately to parent main window if active
        parent = self.parent()
        if parent:
            if hasattr(parent, "viewer") and parent.viewer:
                bg = self.combo_bg.currentData()
                parent.viewer.bg_color_mode = bg
                parent.viewer.show_labels = self.chk_labels.isChecked()
                parent.viewer.show_indices = self.chk_indices.isChecked()
                parent.viewer.use_modern_shaders = self.chk_modern_shaders.isChecked()
                parent.viewer.use_depth_cueing = self.chk_depth_cue.isChecked()
                parent.viewer.mouse_mode = self.combo_mouse_mode.currentData()
                parent.viewer.update()
            if hasattr(parent, "spin_sym_tol"):
                parent.spin_sym_tol.setValue(self.spin_sym_tol.value())

        self.accept()
