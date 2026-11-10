"""
Lighting, Shading, and Publication Figure Studio Dialog for AgyMolden.
Provides interactive controls for ambient intensity, specular shininess,
outline rendering, depth cueing, and custom color palettes (Okabe-Ito, Colorblind-safe, Jmol).
"""

from __future__ import annotations

from PyQt6 import QtCore, QtGui, QtWidgets


class LightingStudioDialog(QtWidgets.QDialog):
    """Interactive Lighting, Shaders & Color Accessibility Studio."""

    settings_changed = QtCore.pyqtSignal()

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.setWindowTitle("3D Lighting, Shaders & Publication Studio")
        self.resize(460, 380)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)

        group_shading = QtWidgets.QGroupBox("Shading & Silhouette Effects")
        lay_shading = QtWidgets.QVBoxLayout(group_shading)

        self.chk_modern = QtWidgets.QCheckBox("✨ ChimeraX / Avogadro 2 Modern Glossy Shading")
        self.chk_modern.setChecked(getattr(self.viewer, "use_modern_shaders", False))
        self.chk_modern.toggled.connect(self._on_modern_toggled)
        lay_shading.addWidget(self.chk_modern)

        self.chk_outline = QtWidgets.QCheckBox("🖋️ IboView / QuteMol Publication Ink Outlines")
        self.chk_outline.setChecked(getattr(self.viewer, "show_publication_outlines", False))
        self.chk_outline.toggled.connect(self._on_outline_toggled)
        lay_shading.addWidget(self.chk_outline)

        self.chk_depth = QtWidgets.QCheckBox("🌫️ Depth Cueing / Distance Fog")
        self.chk_depth.setChecked(getattr(self.viewer, "use_depth_cueing", True))
        self.chk_depth.toggled.connect(self._on_depth_toggled)
        lay_shading.addWidget(self.chk_depth)

        layout.addWidget(group_shading)

        # Accessibility & Palettes
        group_palette = QtWidgets.QGroupBox("Color Palettes & Accessibility")
        lay_pal = QtWidgets.QFormLayout(group_palette)

        self.combo_palette = QtWidgets.QComboBox()
        self.combo_palette.addItem("Default CPK / Jmol Standard", "jmol")
        self.combo_palette.addItem("Okabe-Ito (Deuteranopia & Protanopia Safe)", "okabe_ito")
        self.combo_palette.addItem("Viridis High-Contrast Gradient", "viridis")
        self.combo_palette.addItem("Monochrome Publication Grayscale", "grayscale")
        self.combo_palette.currentIndexChanged.connect(self._on_palette_changed)
        lay_pal.addRow("<b>Atom Palette:</b>", self.combo_palette)

        layout.addWidget(group_palette)

        # Quick Journal Presets
        group_presets = QtWidgets.QGroupBox("1-Click Journal Figure Presets")
        lay_pre = QtWidgets.QHBoxLayout(group_presets)

        btn_jacs = QtWidgets.QPushButton("🏛️ JACS / ACS")
        btn_jacs.clicked.connect(lambda: self._apply_preset("jacs"))
        lay_pre.addWidget(btn_jacs)

        btn_nat = QtWidgets.QPushButton("🌿 Nature / Springer")
        btn_nat.clicked.connect(lambda: self._apply_preset("nature"))
        lay_pre.addWidget(btn_nat)

        btn_dark = QtWidgets.QPushButton("🌙 Dark Cyberpunk")
        btn_dark.clicked.connect(lambda: self._apply_preset("dark"))
        lay_pre.addWidget(btn_dark)

        layout.addWidget(group_presets)

        layout.addStretch()

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _on_modern_toggled(self, checked: bool):
        self.viewer.use_modern_shaders = checked
        self.viewer.update()

    def _on_outline_toggled(self, checked: bool):
        self.viewer.show_publication_outlines = checked
        self.viewer.update()

    def _on_depth_toggled(self, checked: bool):
        self.viewer.use_depth_cueing = checked
        self.viewer.update()

    def _on_palette_changed(self, idx: int):
        val = self.combo_palette.currentData()
        # Set custom element color map in viewer if implemented
        setattr(self.viewer, "active_color_palette", val)
        self.viewer.update()

    def _apply_preset(self, name: str):
        if name == "jacs":
            self.viewer.bg_color = (1.0, 1.0, 1.0, 1.0)
            self.chk_outline.setChecked(True)
            self.chk_modern.setChecked(True)
        elif name == "nature":
            self.viewer.bg_color = (0.97, 0.98, 0.99, 1.0)
            self.chk_outline.setChecked(False)
            self.chk_modern.setChecked(True)
        elif name == "dark":
            self.viewer.bg_color = (0.06, 0.09, 0.16, 1.0)
            self.chk_outline.setChecked(False)
            self.chk_modern.setChecked(True)
        self.viewer.update()
