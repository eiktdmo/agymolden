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

        self.settings = QtCore.QSettings("AgyMolden", "Preferences")
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

        self.combo_mouse_mode = QtWidgets.QComboBox()
        self.combo_mouse_mode.addItem("AgyMolden / Standard (Left: Rotate, Right: Pan, Wheel/Mid: Zoom)", "standard")
        self.combo_mouse_mode.addItem("GaussView Preset (Left: Rotate, Right: Zoom, Mid: Pan)", "gaussview")
        self.combo_mouse_mode.addItem("Chemcraft / PyMOL Preset (Left: Rotate, Right: Pan, Shift+Right: Zoom)", "chemcraft")
        v_layout.addRow("Mouse Navigation Style:", self.combo_mouse_mode)

        tabs.addTab(tab_view, "🖥️ 3D Viewport")

        # Tab 2: Calculation & Symmetry Defaults
        tab_calc = QtWidgets.QWidget()
        c_layout = QtWidgets.QFormLayout(tab_calc)
        c_layout.setSpacing(10)

        self.spin_cores = QtWidgets.QSpinBox()
        self.spin_cores.setRange(1, 128)
        self.spin_cores.setValue(4)
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

        mouse_m = self.settings.value("viewport/mouse_mode", "standard", type=str)
        idx_mouse = self.combo_mouse_mode.findData(mouse_m)
        if idx_mouse >= 0:
            self.combo_mouse_mode.setCurrentIndex(idx_mouse)

        self.spin_cores.setValue(self.settings.value("calc/nprocs", 4, type=int))
        self.spin_mem.setValue(self.settings.value("calc/memory", 4000, type=int))
        self.spin_sym_tol.setValue(self.settings.value("calc/sym_tol", 0.15, type=float))

    def _save_settings(self):
        self.settings.setValue("viewport/background", self.combo_bg.currentData())
        self.settings.setValue("viewport/show_labels", self.chk_labels.isChecked())
        self.settings.setValue("viewport/show_indices", self.chk_indices.isChecked())
        self.settings.setValue("viewport/msaa", self.spin_msaa.currentData())
        self.settings.setValue("viewport/mouse_mode", self.combo_mouse_mode.currentData())

        self.settings.setValue("calc/nprocs", self.spin_cores.value())
        self.settings.setValue("calc/memory", self.spin_mem.value())
        self.settings.setValue("calc/sym_tol", self.spin_sym_tol.value())

        # Apply immediately to parent main window if active
        parent = self.parent()
        if parent:
            if hasattr(parent, "viewer") and parent.viewer:
                bg = self.combo_bg.currentData()
                parent.viewer.bg_color_mode = bg
                parent.viewer.show_labels = self.chk_labels.isChecked()
                parent.viewer.show_indices = self.chk_indices.isChecked()
                parent.viewer.mouse_mode = self.combo_mouse_mode.currentData()
                parent.viewer.update()
            if hasattr(parent, "spin_sym_tol"):
                parent.spin_sym_tol.setValue(self.spin_sym_tol.value())

        self.accept()
