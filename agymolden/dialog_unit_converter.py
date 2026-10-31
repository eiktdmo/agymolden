"""
Interactive Quantum Chemistry Unit & Energy Converter Tool.
Converts simultaneously between Hartree (Eh), eV, kcal/mol, kJ/mol, cm^-1, and Kelvin.
"""

from PyQt6 import QtCore, QtGui, QtWidgets


# Conversion factors relative to 1 Hartree (Eh)
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KJ = 2625.4996394799
HARTREE_TO_CM = 219474.63136320
HARTREE_TO_KELVIN = 315775.02480407


class UnitConverterDialog(QtWidgets.QDialog):
    """Clean, instant multi-unit energy converter for quantum chemists."""

    def __init__(self, parent: QtWidgets.QWidget = None, initial_hartree: float = 1.0):
        super().__init__(parent)
        self.setWindowTitle("⚡ Quantum Chemistry Energy Unit Converter")
        self.resize(460, 360)
        self._updating = False

        self._init_ui()
        self.set_from_hartree(initial_hartree)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        info = QtWidgets.QLabel(
            "<b>Physical Constant Exact Energy Conversions (CODATA):</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Type into any field to convert simultaneously across all energy units."
            "</span>"
        )
        layout.addWidget(info)

        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        self.edit_ha = QtWidgets.QLineEdit()
        self.edit_ev = QtWidgets.QLineEdit()
        self.edit_kcal = QtWidgets.QLineEdit()
        self.edit_kj = QtWidgets.QLineEdit()
        self.edit_cm = QtWidgets.QLineEdit()
        self.edit_k = QtWidgets.QLineEdit()

        for ed in (self.edit_ha, self.edit_ev, self.edit_kcal, self.edit_kj, self.edit_cm, self.edit_k):
            ed.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: bold; padding: 4px;")

        self.edit_ha.textEdited.connect(lambda t: self._on_edited("ha", t))
        self.edit_ev.textEdited.connect(lambda t: self._on_edited("ev", t))
        self.edit_kcal.textEdited.connect(lambda t: self._on_edited("kcal", t))
        self.edit_kj.textEdited.connect(lambda t: self._on_edited("kj", t))
        self.edit_cm.textEdited.connect(lambda t: self._on_edited("cm", t))
        self.edit_k.textEdited.connect(lambda t: self._on_edited("k", t))

        form.addRow("Hartree (Eₕ / a.u.):", self.edit_ha)
        form.addRow("Electron-Volt (eV):", self.edit_ev)
        form.addRow("Kilocalories / mol (kcal/mol):", self.edit_kcal)
        form.addRow("Kilojoules / mol (kJ/mol):", self.edit_kj)
        form.addRow("Wavenumber (cm⁻¹):", self.edit_cm)
        form.addRow("Temperature (Kelvin):", self.edit_k)

        layout.addLayout(form)

        # Quick reference presets
        presets_box = QtWidgets.QHBoxLayout()
        presets_box.addWidget(QtWidgets.QLabel("Presets:"))
        for lbl, val in [("1 kcal/mol", 1.0 / HARTREE_TO_KCAL), ("1 eV", 1.0 / HARTREE_TO_EV), ("Room Temp (298.15 K)", 298.15 / HARTREE_TO_KELVIN), ("1 Hartree", 1.0)]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda _c=False, v=val: self.set_from_hartree(v))
            presets_box.addWidget(btn)
        presets_box.addStretch()
        layout.addLayout(presets_box)

        layout.addStretch()

        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def set_from_hartree(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_ha.setText(f"{val:.8f}")
            self.edit_ev.setText(f"{val * HARTREE_TO_EV:.6f}")
            self.edit_kcal.setText(f"{val * HARTREE_TO_KCAL:.6f}")
            self.edit_kj.setText(f"{val * HARTREE_TO_KJ:.6f}")
            self.edit_cm.setText(f"{val * HARTREE_TO_CM:.3f}")
            self.edit_k.setText(f"{val * HARTREE_TO_KELVIN:.2f}")
        finally:
            self._updating = False

    def _on_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            clean = text.strip().replace(",", "")
            v = float(clean)
        except ValueError:
            return

        if unit == "ha":
            ha = v
        elif unit == "ev":
            ha = v / HARTREE_TO_EV
        elif unit == "kcal":
            ha = v / HARTREE_TO_KCAL
        elif unit == "kj":
            ha = v / HARTREE_TO_KJ
        elif unit == "cm":
            ha = v / HARTREE_TO_CM
        elif unit == "k":
            ha = v / HARTREE_TO_KELVIN
        else:
            ha = v

        self._updating = True
        try:
            if unit != "ha": self.edit_ha.setText(f"{ha:.8f}")
            if unit != "ev": self.edit_ev.setText(f"{ha * HARTREE_TO_EV:.6f}")
            if unit != "kcal": self.edit_kcal.setText(f"{ha * HARTREE_TO_KCAL:.6f}")
            if unit != "kj": self.edit_kj.setText(f"{ha * HARTREE_TO_KJ:.6f}")
            if unit != "cm": self.edit_cm.setText(f"{ha * HARTREE_TO_CM:.3f}")
            if unit != "k": self.edit_k.setText(f"{ha * HARTREE_TO_KELVIN:.2f}")
        finally:
            self._updating = False
