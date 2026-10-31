"""
Comprehensive Quantum Chemistry Unit & Physical Quantity Converter Tool.
Converts quantities across 5 key quantum chemical domains:
1. Energy & Thermochemistry (Hartree, eV, kcal/mol, kJ/mol, cm⁻¹, Kelvin)
2. Length & Distance (Bohr a₀, Ångström, Picometer, Nanometer)
3. Electric Dipole Moment (Debye, a.u. e·a₀, C·m)
4. Polarizability (a.u. Bohr³, Å³, C·m²/V)
5. Electric Field & Pressure (a.u. field, V/Å, V/m, MV/cm; atm, bar, Pa, Torr/mmHg)
"""

from PyQt6 import QtCore, QtGui, QtWidgets


# --- CODATA 2018 / Exact Constants ---
# Energy relative to 1 Hartree (Eh)
HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCAL = 627.5094740631
HARTREE_TO_KJ = 2625.4996394799
HARTREE_TO_CM = 219474.63136320
HARTREE_TO_KELVIN = 315775.02480407

# Length relative to 1 Bohr (a0)
BOHR_TO_ANGSTROM = 0.529177210903
BOHR_TO_PM = 52.9177210903
BOHR_TO_NM = 0.0529177210903

# Dipole relative to 1 Debye (D)
# 1 a.u. (e*a0) = 2.541746473 Debye
# 1 Debye = 3.33564095198e-30 C*m
DEBYE_TO_AU = 1.0 / 2.541746473
DEBYE_TO_CM = 3.33564095198e-30

# Polarizability volume relative to 1 Bohr^3 (a.u.)
# 1 Bohr^3 = (0.529177210903)^3 Angstrom^3 = 0.148184711 Angstrom^3
# SI: 1 a.u. of polarizability = e^2 * a0^2 / Eh = 1.648777274e-41 C^2*m^2/J (or C*m^2/V)
BOHR3_TO_ANG3 = 0.529177210903 ** 3
BOHR3_TO_SI = 1.64877727436e-41

# Electric Field relative to 1 a.u. (Eh / (e * a0))
# 1 a.u. = 5.14220674763e11 V/m = 51.4220674763 V/Angstrom = 5142.20674763 MV/cm
AU_FIELD_TO_V_M = 5.14220674763e11
AU_FIELD_TO_V_ANG = 51.4220674763
AU_FIELD_TO_MV_CM = 5142.20674763

# Pressure relative to 1 standard atmosphere (atm)
ATM_TO_BAR = 1.01325
ATM_TO_PA = 101325.0
ATM_TO_TORR = 760.0


class UnitConverterDialog(QtWidgets.QDialog):
    """Multi-quantity converter dialog for quantum chemistry."""

    def __init__(self, parent: QtWidgets.QWidget = None, initial_hartree: float = 1.0):
        super().__init__(parent)
        self.setWindowTitle("⚡ Quantum Chemistry Multitool Unit Converter")
        self.resize(540, 440)
        self._updating = False

        self._init_ui()
        self.set_from_hartree(initial_hartree)
        self.set_from_bohr(1.0)
        self.set_from_debye(1.85)  # water dipole ~1.85 D
        self.set_from_bohr3(10.0)
        self.set_from_au_field(0.001)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        header = QtWidgets.QLabel(
            "<b>Physical Constants & Quantum Conversions (CODATA Recommended):</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Instant bidirectional conversion. Edit any box to update all other units in real time."
            "</span>"
        )
        layout.addWidget(header)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Energy
        self._init_energy_tab()
        # Tab 2: Length / Geometry
        self._init_length_tab()
        # Tab 3: Dipole Moment
        self._init_dipole_tab()
        # Tab 4: Polarizability
        self._init_polarizability_tab()
        # Tab 5: Electric Field & Pressure
        self._init_field_pressure_tab()

        # Bottom Close button
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setDefault(True)
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _style_edit(self, edit: QtWidgets.QLineEdit):
        edit.setStyleSheet("font-family: monospace; font-size: 12px; font-weight: bold; padding: 4px;")

    # ==================== 1. ENERGY TAB ====================
    def _init_energy_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        self.edit_ha = QtWidgets.QLineEdit()
        self.edit_ev = QtWidgets.QLineEdit()
        self.edit_kcal = QtWidgets.QLineEdit()
        self.edit_kj = QtWidgets.QLineEdit()
        self.edit_cm = QtWidgets.QLineEdit()
        self.edit_k = QtWidgets.QLineEdit()

        for ed in (self.edit_ha, self.edit_ev, self.edit_kcal, self.edit_kj, self.edit_cm, self.edit_k):
            self._style_edit(ed)

        self.edit_ha.textEdited.connect(lambda t: self._on_energy_edited("ha", t))
        self.edit_ev.textEdited.connect(lambda t: self._on_energy_edited("ev", t))
        self.edit_kcal.textEdited.connect(lambda t: self._on_energy_edited("kcal", t))
        self.edit_kj.textEdited.connect(lambda t: self._on_energy_edited("kj", t))
        self.edit_cm.textEdited.connect(lambda t: self._on_energy_edited("cm", t))
        self.edit_k.textEdited.connect(lambda t: self._on_energy_edited("k", t))

        form.addRow("Hartree (Eₕ / a.u.):", self.edit_ha)
        form.addRow("Electron-Volt (eV):", self.edit_ev)
        form.addRow("Kilocalories / mol (kcal/mol):", self.edit_kcal)
        form.addRow("Kilojoules / mol (kJ/mol):", self.edit_kj)
        form.addRow("Wavenumber (cm⁻¹):", self.edit_cm)
        form.addRow("Temperature (Kelvin):", self.edit_k)
        vbox.addLayout(form)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for lbl, val in [
            ("1 kcal/mol", 1.0 / HARTREE_TO_KCAL),
            ("1 eV", 1.0 / HARTREE_TO_EV),
            ("Room Temp (298.15 K)", 298.15 / HARTREE_TO_KELVIN),
            ("1 Hartree", 1.0)
        ]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda _c=False, v=val: self.set_from_hartree(v))
            presets.addWidget(btn)
        presets.addStretch()
        vbox.addLayout(presets)
        vbox.addStretch()

        self.tabs.addTab(tab, "Energy & Thermo")

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

    def _on_energy_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "ha": ha = v
        elif unit == "ev": ha = v / HARTREE_TO_EV
        elif unit == "kcal": ha = v / HARTREE_TO_KCAL
        elif unit == "kj": ha = v / HARTREE_TO_KJ
        elif unit == "cm": ha = v / HARTREE_TO_CM
        elif unit == "k": ha = v / HARTREE_TO_KELVIN
        else: ha = v

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

    # ==================== 2. LENGTH TAB ====================
    def _init_length_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        self.edit_bohr = QtWidgets.QLineEdit()
        self.edit_ang = QtWidgets.QLineEdit()
        self.edit_pm = QtWidgets.QLineEdit()
        self.edit_nm = QtWidgets.QLineEdit()

        for ed in (self.edit_bohr, self.edit_ang, self.edit_pm, self.edit_nm):
            self._style_edit(ed)

        self.edit_bohr.textEdited.connect(lambda t: self._on_length_edited("bohr", t))
        self.edit_ang.textEdited.connect(lambda t: self._on_length_edited("ang", t))
        self.edit_pm.textEdited.connect(lambda t: self._on_length_edited("pm", t))
        self.edit_nm.textEdited.connect(lambda t: self._on_length_edited("nm", t))

        form.addRow("Bohr Radius (a₀ / a.u.):", self.edit_bohr)
        form.addRow("Ångström (Å):", self.edit_ang)
        form.addRow("Picometer (pm):", self.edit_pm)
        form.addRow("Nanometer (nm):", self.edit_nm)
        vbox.addLayout(form)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for lbl, val_bohr in [
            ("1.0 Å", 1.0 / BOHR_TO_ANGSTROM),
            ("1.0 a₀ (Bohr)", 1.0),
            ("C-C bond (1.54 Å)", 1.54 / BOHR_TO_ANGSTROM),
            ("C-H bond (1.09 Å)", 1.09 / BOHR_TO_ANGSTROM),
        ]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda _c=False, v=val_bohr: self.set_from_bohr(v))
            presets.addWidget(btn)
        presets.addStretch()
        vbox.addLayout(presets)
        vbox.addStretch()

        self.tabs.addTab(tab, "Length & Distance")

    def set_from_bohr(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_bohr.setText(f"{val:.8f}")
            self.edit_ang.setText(f"{val * BOHR_TO_ANGSTROM:.6f}")
            self.edit_pm.setText(f"{val * BOHR_TO_PM:.4f}")
            self.edit_nm.setText(f"{val * BOHR_TO_NM:.6f}")
        finally:
            self._updating = False

    def _on_length_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "bohr": bohr = v
        elif unit == "ang": bohr = v / BOHR_TO_ANGSTROM
        elif unit == "pm": bohr = v / BOHR_TO_PM
        elif unit == "nm": bohr = v / BOHR_TO_NM
        else: bohr = v

        self._updating = True
        try:
            if unit != "bohr": self.edit_bohr.setText(f"{bohr:.8f}")
            if unit != "ang": self.edit_ang.setText(f"{bohr * BOHR_TO_ANGSTROM:.6f}")
            if unit != "pm": self.edit_pm.setText(f"{bohr * BOHR_TO_PM:.4f}")
            if unit != "nm": self.edit_nm.setText(f"{bohr * BOHR_TO_NM:.6f}")
        finally:
            self._updating = False

    # ==================== 3. DIPOLE MOMENT TAB ====================
    def _init_dipole_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        self.edit_debye = QtWidgets.QLineEdit()
        self.edit_dipole_au = QtWidgets.QLineEdit()
        self.edit_dipole_cm = QtWidgets.QLineEdit()

        for ed in (self.edit_debye, self.edit_dipole_au, self.edit_dipole_cm):
            self._style_edit(ed)

        self.edit_debye.textEdited.connect(lambda t: self._on_dipole_edited("debye", t))
        self.edit_dipole_au.textEdited.connect(lambda t: self._on_dipole_edited("au", t))
        self.edit_dipole_cm.textEdited.connect(lambda t: self._on_dipole_edited("cm", t))

        form.addRow("Debye (D):", self.edit_debye)
        form.addRow("Atomic Units (e·a₀):", self.edit_dipole_au)
        form.addRow("Coulomb-meter (C·m):", self.edit_dipole_cm)
        vbox.addLayout(form)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for lbl, val_d in [
            ("Water (1.85 D)", 1.854),
            ("1.0 a.u. (2.5417 D)", 2.541746),
            ("Ammonia (1.47 D)", 1.47),
            ("HF (1.82 D)", 1.826),
        ]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda _c=False, v=val_d: self.set_from_debye(v))
            presets.addWidget(btn)
        presets.addStretch()
        vbox.addLayout(presets)
        vbox.addStretch()

        self.tabs.addTab(tab, "Dipole Moment")

    def set_from_debye(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_debye.setText(f"{val:.6f}")
            self.edit_dipole_au.setText(f"{val * DEBYE_TO_AU:.6f}")
            self.edit_dipole_cm.setText(f"{val * DEBYE_TO_CM:.6e}")
        finally:
            self._updating = False

    def _on_dipole_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "debye": debye = v
        elif unit == "au": debye = v / DEBYE_TO_AU
        elif unit == "cm": debye = v / DEBYE_TO_CM
        else: debye = v

        self._updating = True
        try:
            if unit != "debye": self.edit_debye.setText(f"{debye:.6f}")
            if unit != "au": self.edit_dipole_au.setText(f"{debye * DEBYE_TO_AU:.6f}")
            if unit != "cm": self.edit_dipole_cm.setText(f"{debye * DEBYE_TO_CM:.6e}")
        finally:
            self._updating = False

    # ==================== 4. POLARIZABILITY TAB ====================
    def _init_polarizability_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)
        form = QtWidgets.QFormLayout()
        form.setSpacing(8)

        self.edit_pol_bohr3 = QtWidgets.QLineEdit()
        self.edit_pol_ang3 = QtWidgets.QLineEdit()
        self.edit_pol_si = QtWidgets.QLineEdit()

        for ed in (self.edit_pol_bohr3, self.edit_pol_ang3, self.edit_pol_si):
            self._style_edit(ed)

        self.edit_pol_bohr3.textEdited.connect(lambda t: self._on_pol_edited("bohr3", t))
        self.edit_pol_ang3.textEdited.connect(lambda t: self._on_pol_edited("ang3", t))
        self.edit_pol_si.textEdited.connect(lambda t: self._on_pol_edited("si", t))

        form.addRow("Atomic Units (Bohr³ / e²a₀²/Eₕ):", self.edit_pol_bohr3)
        form.addRow("Polarizability Volume (Å³):", self.edit_pol_ang3)
        form.addRow("SI Units (C·m²/V):", self.edit_pol_si)
        vbox.addLayout(form)

        presets = QtWidgets.QHBoxLayout()
        presets.addWidget(QtWidgets.QLabel("Presets:"))
        for lbl, val_b3 in [
            ("1.0 Bohr³", 1.0),
            ("1.0 Å³", 1.0 / BOHR3_TO_ANG3),
            ("Water (9.64 a.u.)", 9.64),
            ("Benzene (70.4 a.u.)", 70.4),
        ]:
            btn = QtWidgets.QPushButton(lbl)
            btn.setStyleSheet("padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda _c=False, v=val_b3: self.set_from_bohr3(v))
            presets.addWidget(btn)
        presets.addStretch()
        vbox.addLayout(presets)
        vbox.addStretch()

        self.tabs.addTab(tab, "Polarizability")

    def set_from_bohr3(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_pol_bohr3.setText(f"{val:.6f}")
            self.edit_pol_ang3.setText(f"{val * BOHR3_TO_ANG3:.6f}")
            self.edit_pol_si.setText(f"{val * BOHR3_TO_SI:.6e}")
        finally:
            self._updating = False

    def _on_pol_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "bohr3": b3 = v
        elif unit == "ang3": b3 = v / BOHR3_TO_ANG3
        elif unit == "si": b3 = v / BOHR3_TO_SI
        else: b3 = v

        self._updating = True
        try:
            if unit != "bohr3": self.edit_pol_bohr3.setText(f"{b3:.6f}")
            if unit != "ang3": self.edit_pol_ang3.setText(f"{b3 * BOHR3_TO_ANG3:.6f}")
            if unit != "si": self.edit_pol_si.setText(f"{b3 * BOHR3_TO_SI:.6e}")
        finally:
            self._updating = False

    # ==================== 5. FIELD & PRESSURE TAB ====================
    def _init_field_pressure_tab(self):
        tab = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(tab)

        # Electric Field Box
        box_f = QtWidgets.QGroupBox("Electric Field (Stark effect, External QM bias)")
        form_f = QtWidgets.QFormLayout(box_f)
        self.edit_f_au = QtWidgets.QLineEdit()
        self.edit_f_vang = QtWidgets.QLineEdit()
        self.edit_f_mvcm = QtWidgets.QLineEdit()
        self.edit_f_vm = QtWidgets.QLineEdit()

        for ed in (self.edit_f_au, self.edit_f_vang, self.edit_f_mvcm, self.edit_f_vm):
            self._style_edit(ed)

        self.edit_f_au.textEdited.connect(lambda t: self._on_field_edited("au", t))
        self.edit_f_vang.textEdited.connect(lambda t: self._on_field_edited("vang", t))
        self.edit_f_mvcm.textEdited.connect(lambda t: self._on_field_edited("mvcm", t))
        self.edit_f_vm.textEdited.connect(lambda t: self._on_field_edited("vm", t))

        form_f.addRow("Atomic Field (a.u.):", self.edit_f_au)
        form_f.addRow("Volt / Ångström (V/Å):", self.edit_f_vang)
        form_f.addRow("Megavolt / cm (MV/cm):", self.edit_f_mvcm)
        form_f.addRow("Volt / meter (V/m):", self.edit_f_vm)
        vbox.addWidget(box_f)

        # Pressure Box
        box_p = QtWidgets.QGroupBox("Pressure (High-pressure QM, Diamond Anvil Cell)")
        form_p = QtWidgets.QFormLayout(box_p)
        self.edit_p_atm = QtWidgets.QLineEdit()
        self.edit_p_bar = QtWidgets.QLineEdit()
        self.edit_p_gpa = QtWidgets.QLineEdit()
        self.edit_p_torr = QtWidgets.QLineEdit()

        for ed in (self.edit_p_atm, self.edit_p_bar, self.edit_p_gpa, self.edit_p_torr):
            self._style_edit(ed)

        self.edit_p_atm.textEdited.connect(lambda t: self._on_press_edited("atm", t))
        self.edit_p_bar.textEdited.connect(lambda t: self._on_press_edited("bar", t))
        self.edit_p_gpa.textEdited.connect(lambda t: self._on_press_edited("gpa", t))
        self.edit_p_torr.textEdited.connect(lambda t: self._on_press_edited("torr", t))

        form_p.addRow("Atmosphere (atm):", self.edit_p_atm)
        form_p.addRow("Bar (bar):", self.edit_p_bar)
        form_p.addRow("Gigapascal (GPa):", self.edit_p_gpa)
        form_p.addRow("Torr / mmHg:", self.edit_p_torr)
        vbox.addWidget(box_p)

        self._set_from_atm(1.0)
        vbox.addStretch()

        self.tabs.addTab(tab, "Electric Field & Pressure")

    def set_from_au_field(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_f_au.setText(f"{val:.6f}")
            self.edit_f_vang.setText(f"{val * AU_FIELD_TO_V_ANG:.4f}")
            self.edit_f_mvcm.setText(f"{val * AU_FIELD_TO_MV_CM:.4f}")
            self.edit_f_vm.setText(f"{val * AU_FIELD_TO_V_M:.4e}")
        finally:
            self._updating = False

    def _on_field_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "au": f_au = v
        elif unit == "vang": f_au = v / AU_FIELD_TO_V_ANG
        elif unit == "mvcm": f_au = v / AU_FIELD_TO_MV_CM
        elif unit == "vm": f_au = v / AU_FIELD_TO_V_M
        else: f_au = v

        self._updating = True
        try:
            if unit != "au": self.edit_f_au.setText(f"{f_au:.6f}")
            if unit != "vang": self.edit_f_vang.setText(f"{f_au * AU_FIELD_TO_V_ANG:.4f}")
            if unit != "mvcm": self.edit_f_mvcm.setText(f"{f_au * AU_FIELD_TO_MV_CM:.4f}")
            if unit != "vm": self.edit_f_vm.setText(f"{f_au * AU_FIELD_TO_V_M:.4e}")
        finally:
            self._updating = False

    def _set_from_atm(self, val: float):
        if self._updating:
            return
        self._updating = True
        try:
            self.edit_p_atm.setText(f"{val:.6f}")
            self.edit_p_bar.setText(f"{val * ATM_TO_BAR:.6f}")
            self.edit_p_gpa.setText(f"{(val * ATM_TO_PA) / 1e9:.6e}")
            self.edit_p_torr.setText(f"{val * ATM_TO_TORR:.2f}")
        finally:
            self._updating = False

    def _on_press_edited(self, unit: str, text: str):
        if self._updating:
            return
        try:
            v = float(text.strip().replace(",", ""))
        except ValueError:
            return

        if unit == "atm": atm = v
        elif unit == "bar": atm = v / ATM_TO_BAR
        elif unit == "gpa": atm = (v * 1e9) / ATM_TO_PA
        elif unit == "torr": atm = v / ATM_TO_TORR
        else: atm = v

        self._updating = True
        try:
            if unit != "atm": self.edit_p_atm.setText(f"{atm:.6f}")
            if unit != "bar": self.edit_p_bar.setText(f"{atm * ATM_TO_BAR:.6f}")
            if unit != "gpa": self.edit_p_gpa.setText(f"{(atm * ATM_TO_PA) / 1e9:.6e}")
            if unit != "torr": self.edit_p_torr.setText(f"{atm * ATM_TO_TORR:.2f}")
        finally:
            self._updating = False
