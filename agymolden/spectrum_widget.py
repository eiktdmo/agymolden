"""
Interactive IR Spectrum Plotting Widget with Peak Broadening (Lorentzian/Gaussian).

Allows users to view a synthetic IR spectrum with Lorentzian broadening,
see stick spectrum lines, and click on any peak to jump to and animate
that vibrational mode in the 3D viewport.
"""

from typing import List, Optional, Callable, Tuple
import numpy as np
from PyQt6 import QtCore, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def calculate_broadened_spectrum(
    frequencies: List[float],
    intensities: Optional[List[float]] = None,
    fwhm: float = 15.0,
    w_min: float = 400.0,
    w_max: float = 4000.0,
    n_points: int = 1500,
    lineshape: str = "Lorentzian",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute synthetic continuous spectrum via Lorentzian or Gaussian convolution.
    
    Returns (wavenumbers, absorbance).
    """
    w = np.linspace(w_min, w_max, n_points)
    abs_curve = np.zeros_like(w)

    if not frequencies:
        return w, abs_curve

    ints = intensities if intensities is not None else [1.0] * len(frequencies)
    gamma = fwhm / 2.0  # HWHM
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    for freq, intensity in zip(frequencies, ints):
        if freq < 20.0:  # Skip translations/rotations and negative frequencies
            continue
        if lineshape == "Gaussian":
            abs_curve += intensity * np.exp(-0.5 * ((w - freq) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))
        else:  # Lorentzian
            abs_curve += intensity * (gamma / np.pi) / ((w - freq) ** 2 + gamma ** 2)

    return w, abs_curve


class IRSpectrumWidget(QtWidgets.QWidget):
    """Interactive Matplotlib canvas widget for IR spectra."""
    mode_clicked = QtCore.pyqtSignal(int)  # mode index 0-based
    scale_factor_changed = QtCore.pyqtSignal(float)  # emitted when scale factor is modified

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frequencies: List[float] = []
        self.intensities: Optional[List[float]] = None
        self.raman_activities: Optional[List[float]] = None
        self.plot_mode: str = "IR"  # "IR" or "Raman"
        self.active_mode_idx: Optional[int] = None
        self.fwhm: float = 15.0
        self.scale_factor: float = 1.0000
        self.calc_method: str = ""
        self.calc_basis: str = ""

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Plot Controls Ribbon
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(6)

        self.combo_spec_type = QtWidgets.QComboBox()
        self.combo_spec_type.addItems(["IR Spectrum", "Raman Spectrum"])
        self.combo_spec_type.setToolTip("Switch between Infrared (IR) absorbance and Raman scattering activity")
        self.combo_spec_type.currentIndexChanged.connect(self._on_spec_type_changed)
        ctrl_layout.addWidget(self.combo_spec_type)

        ctrl_layout.addWidget(QtWidgets.QLabel("Peak Width (FWHM):"))
        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(2.0, 50.0)
        self.spin_fwhm.setValue(15.0)
        self.spin_fwhm.setSingleStep(2.0)
        self.spin_fwhm.setSuffix(" cm⁻¹")
        self.spin_fwhm.valueChanged.connect(self._on_fwhm_changed)
        ctrl_layout.addWidget(self.spin_fwhm)

        self.chk_invert_x = QtWidgets.QCheckBox("Invert X (Chemistry Standard)")
        self.chk_invert_x.setChecked(True)
        self.chk_invert_x.toggled.connect(self._replot)
        ctrl_layout.addWidget(self.chk_invert_x)

        self.chk_sticks = QtWidgets.QCheckBox("Show Sticks")
        self.chk_sticks.setChecked(True)
        self.chk_sticks.toggled.connect(self._replot)
        ctrl_layout.addWidget(self.chk_sticks)

        # Scale Factor controls (NIST CCCBDB)
        ctrl_layout.addWidget(QtWidgets.QLabel("Scale Factor:"))
        self.spin_scale = QtWidgets.QDoubleSpinBox()
        self.spin_scale.setRange(0.500, 1.500)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setSingleStep(0.005)
        self.spin_scale.setValue(1.0000)
        self.spin_scale.setToolTip("Multiplicative scale factor applied to harmonic vibrational frequencies (e.g. from NIST CCCBDB)")
        self.spin_scale.valueChanged.connect(self._on_scale_changed)
        ctrl_layout.addWidget(self.spin_scale)

        self.btn_nist_scale = QtWidgets.QPushButton("NIST CCCBDB...")
        self.btn_nist_scale.setToolTip("Look up and apply official vibrational scale factor from NIST CCCBDB based on method and basis set")
        self.btn_nist_scale.setStyleSheet("background-color: #2563eb; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_nist_scale.clicked.connect(self._on_nist_scale_clicked)
        ctrl_layout.addWidget(self.btn_nist_scale)

        self.lbl_scale_info = QtWidgets.QLabel("")
        self.lbl_scale_info.setStyleSheet("color: #38bdf8; font-size: 10px;")
        ctrl_layout.addWidget(self.lbl_scale_info)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(5, 3), dpi=90, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        layout.addWidget(self.canvas, stretch=1)

    def _style_axes(self):
        is_white = False
        p = self.parent()
        while p is not None:
            v = getattr(p, "viewer", None)
            if v and getattr(v, "bg_color_mode", "") == "white":
                is_white = True
                break
            p = getattr(p, "parent", lambda: None)()

        ax_face = "#f8fafc" if is_white else "#141418"
        tick_col = "#475569" if is_white else "#94a3b8"
        lbl_col = "#0f172a" if is_white else "#cbd5e1"
        spine_col = "#cbd5e1" if is_white else "#2e2e38"

        self.ax.set_facecolor(ax_face)
        self.ax.tick_params(colors=tick_col, labelsize=8)
        self.ax.xaxis.label.set_color(lbl_col)
        self.ax.yaxis.label.set_color(lbl_col)
        self.ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        self.ax.set_ylabel("Absorbance / Intensity", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def set_vibrational_data(
        self,
        frequencies: List[float],
        intensities: Optional[List[float]] = None,
        method: str = "",
        basis: str = "",
        raman_activities: Optional[List[float]] = None,
    ):
        self.frequencies = list(frequencies)
        self.intensities = list(intensities) if intensities is not None else None
        self.raman_activities = list(raman_activities) if raman_activities is not None else None
        self.calc_method = method
        self.calc_basis = basis
        self.active_mode_idx = None

        if method:
            from .vibrations import get_nist_scale_factor
            fac, cite = get_nist_scale_factor(method, basis)
            if abs(fac - 1.0) > 1e-4:
                self.spin_scale.blockSignals(True)
                self.spin_scale.setValue(fac)
                self.spin_scale.blockSignals(False)
                self.scale_factor = fac
                self.lbl_scale_info.setText(cite)
        self._replot()

    def _on_spec_type_changed(self, idx: int):
        self.plot_mode = "Raman" if idx == 1 else "IR"
        self._replot()

    def set_active_mode(self, mode_idx: int):
        self.active_mode_idx = mode_idx
        self._replot()

    def _on_fwhm_changed(self, val: float):
        self.fwhm = val
        self._replot()

    def _on_scale_changed(self, val: float):
        self.scale_factor = val
        self.scale_factor_changed.emit(val)
        self._replot()

    def _on_nist_scale_clicked(self):
        from .vibrations import get_nist_scale_factor, NIST_VIBRATIONAL_SCALE_FACTORS
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("NIST CCCBDB Vibrational Frequency Scaling")
        dlg.resize(460, 240)
        vbox = QtWidgets.QVBoxLayout(dlg)
        vbox.setSpacing(10)

        info = QtWidgets.QLabel(
            "<b>Select Method & Basis Set for NIST CCCBDB Scale Factor:</b><br>"
            "<span style='color: #64748b; font-size: 11px;'>"
            "Corrects harmonic frequencies for anharmonicity and electronic model deficiency.<br>"
            "Reference: <a href='https://cccbdb.nist.gov/vibscalex.asp'>https://cccbdb.nist.gov/vibscalex.asp</a>"
            "</span>"
        )
        info.setOpenExternalLinks(True)
        vbox.addWidget(info)

        form = QtWidgets.QFormLayout()
        cb_meth = QtWidgets.QComboBox()
        cb_meth.setEditable(True)
        cb_meth.addItems(["B3LYP", "PBE0", "wB97X-D", "M06-2X", "HF", "MP2", "CAM-B3LYP", "B97-D3", "AM1", "PM6", "xTB"])
        if self.calc_method:
            cb_meth.setCurrentText(self.calc_method)
        form.addRow("Method / Functional:", cb_meth)

        cb_bas = QtWidgets.QComboBox()
        cb_bas.setEditable(True)
        cb_bas.addItems(["6-31G*", "6-31+G(d,p)", "def2-SVP", "def2-TZVP", "cc-pVDZ", "cc-pVTZ", "6-31G", "sto-3g"])
        if self.calc_basis:
            cb_bas.setCurrentText(self.calc_basis)
        form.addRow("Basis Set:", cb_bas)

        lbl_res = QtWidgets.QLabel("")
        lbl_res.setStyleSheet("font-weight: bold; color: #0284c7;")
        form.addRow("NIST Factor:", lbl_res)

        vbox.addLayout(form)

        def update_val():
            fac, cite = get_nist_scale_factor(cb_meth.currentText(), cb_bas.currentText())
            lbl_res.setText(f"λ = {fac:.4f} ({cite})")

        cb_meth.currentTextChanged.connect(update_val)
        cb_bas.currentTextChanged.connect(update_val)
        update_val()

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(dlg.accept)
        btn_box.rejected.connect(dlg.reject)
        vbox.addWidget(btn_box)

        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            fac, cite = get_nist_scale_factor(cb_meth.currentText(), cb_bas.currentText())
            self.spin_scale.setValue(fac)
            self.lbl_scale_info.setText(cite)

    def get_effective_frequencies(self) -> List[float]:
        return [f * self.scale_factor for f in self.frequencies]

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.frequencies:
            self.ax.text(
                0.5, 0.5, "No vibrational frequencies calculated yet.\nClick 'Calculate Frequencies' or load a QM log file.",
                ha="center", va="center", color="#64748b", transform=self.ax.transAxes, fontsize=9
            )
            self.canvas.draw()
            return

        eff_freqs = self.get_effective_frequencies()
        valid_freqs = [f for f in eff_freqs if f > 20.0]
        if not valid_freqs:
            w_min, w_max = 400.0, 4000.0
        else:
            w_min = max(200.0, min(valid_freqs) - 150.0)
            w_max = max(4000.0, max(valid_freqs) + 200.0)

        active_ints = self.raman_activities if self.plot_mode == "Raman" else self.intensities
        if self.plot_mode == "Raman":
            self.ax.set_ylabel("Raman Activity (Å⁴/amu)", fontsize=9)
            line_col = "#34d399"
            fill_col = "#059669"
        else:
            self.ax.set_ylabel("IR Absorbance / Intensity", fontsize=9)
            line_col = "#38bdf8"
            fill_col = "#0284c7"

        w, curve = calculate_broadened_spectrum(
            eff_freqs, active_ints, fwhm=self.fwhm, w_min=w_min, w_max=w_max
        )

        max_curve = np.max(curve) if len(curve) > 0 and np.max(curve) > 0 else 1.0
        # Normalize curve to 1.0
        norm_curve = curve / max_curve
        lbl_curve = f"Lorentzian Envelope (λ={self.scale_factor:.4f})" if abs(self.scale_factor - 1.0) > 1e-4 else "Lorentzian Envelope"
        self.ax.plot(w, norm_curve, color=line_col, lw=1.6, label=lbl_curve)
        self.ax.fill_between(w, norm_curve, color=fill_col, alpha=0.25)

        # Plot sticks
        if self.chk_sticks.isChecked():
            ints = active_ints if active_ints is not None else [1.0] * len(eff_freqs)
            max_int = max(ints) if ints and max(ints) > 0 else 1.0
            for idx, (f, inten) in enumerate(zip(eff_freqs, ints)):
                if f < 20.0:
                    continue
                h = inten / max_int
                is_active = (idx == self.active_mode_idx)
                col = "#f59e0b" if is_active else "#94a3b8"
                lw = 2.2 if is_active else 1.0
                self.ax.vlines(f, 0, h, color=col, lw=lw, alpha=0.85)

        if self.chk_invert_x.isChecked():
            self.ax.set_xlim(w_max, w_min)
        else:
            self.ax.set_xlim(w_min, w_max)
        self.ax.set_ylim(0, 1.05)

        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.frequencies:
            return
        clicked_w = event.xdata
        eff_freqs = self.get_effective_frequencies()
        # Find nearest vibrational mode
        best_idx = None
        min_dist = float("inf")
        for idx, f in enumerate(eff_freqs):
            if f < 20.0:
                continue
            dist = abs(f - clicked_w)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        # Pick tolerance of +/- 50 cm^-1 or 2*fwhm
        if best_idx is not None and min_dist < max(35.0, 2.0 * self.fwhm):
            self.active_mode_idx = best_idx
            self.mode_clicked.emit(best_idx)
            self._replot()


def calculate_uv_vis_spectrum(
    wavelengths_nm: List[float],
    oscillator_strengths: Optional[List[float]] = None,
    fwhm_nm: float = 25.0,
    w_min: float = 190.0,
    w_max: float = 800.0,
    n_points: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian convolution of electronic transitions (UV-Vis absorption spectrum).
    Returns (wavelengths_in_nm, extinction_coefficient_curve).
    """
    w = np.linspace(w_min, w_max, n_points)
    abs_curve = np.zeros_like(w)

    if not wavelengths_nm:
        return w, abs_curve

    oscs = oscillator_strengths if oscillator_strengths is not None else [1.0] * len(wavelengths_nm)
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    for wl, f in zip(wavelengths_nm, oscs):
        if wl < 10.0 or f <= 0.0:
            continue
        # Gaussian peak: f * exp(-0.5 * ((w - wl)/sigma)^2)
        abs_curve += f * np.exp(-0.5 * ((w - wl) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    return w, abs_curve


class UVVisSpectrumWidget(QtWidgets.QWidget):
    """Interactive Matplotlib canvas widget for UV-Vis Electronic Absorption Spectra (GaussView / Multiwfn style)."""
    transition_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wavelengths: List[float] = []
        self.oscs: Optional[List[float]] = None
        self.active_idx: Optional[int] = None
        self.fwhm_nm: float = 25.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Ribbon controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QtWidgets.QLabel("Peak Width (FWHM):"))
        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(5.0, 100.0)
        self.spin_fwhm.setValue(25.0)
        self.spin_fwhm.setSingleStep(5.0)
        self.spin_fwhm.setSuffix(" nm")
        self.spin_fwhm.valueChanged.connect(self._on_fwhm_changed)
        ctrl_layout.addWidget(self.spin_fwhm)

        self.chk_sticks = QtWidgets.QCheckBox("Show Transitions (Sticks)")
        self.chk_sticks.setChecked(True)
        self.chk_sticks.toggled.connect(self._replot)
        ctrl_layout.addWidget(self.chk_sticks)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(5, 3), dpi=90, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        layout.addWidget(self.canvas, stretch=1)

    def _style_axes(self):
        is_white = False
        p = self.parent()
        while p is not None:
            v = getattr(p, "viewer", None)
            if v and getattr(v, "bg_color_mode", "") == "white":
                is_white = True
                break
            p = getattr(p, "parent", lambda: None)()

        ax_face = "#f8fafc" if is_white else "#141418"
        tick_col = "#475569" if is_white else "#94a3b8"
        lbl_col = "#0f172a" if is_white else "#cbd5e1"
        spine_col = "#cbd5e1" if is_white else "#2e2e38"

        self.ax.set_facecolor(ax_face)
        self.ax.tick_params(colors=tick_col, labelsize=8)
        self.ax.xaxis.label.set_color(lbl_col)
        self.ax.yaxis.label.set_color(lbl_col)
        self.ax.set_xlabel("Wavelength λ (nm)", fontsize=9)
        self.ax.set_ylabel("Absorbance / Oscillator Strength f", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def set_uv_vis_data(self, wavelengths_nm: List[float], oscillator_strengths: Optional[List[float]] = None):
        self.wavelengths = list(wavelengths_nm)
        self.oscs = list(oscillator_strengths) if oscillator_strengths is not None else None
        self.active_idx = None
        self._replot()

    def _on_fwhm_changed(self, val: float):
        self.fwhm_nm = val
        self._replot()

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.wavelengths:
            self.ax.text(
                0.5, 0.5, "No UV-Vis excited state transitions loaded.\nOpen a TD-DFT / CIS output file.",
                ha="center", va="center", color="#64748b", transform=self.ax.transAxes, fontsize=9
            )
            self.canvas.draw()
            return

        w_min = max(180.0, min(self.wavelengths) - 50.0)
        w_max = min(1200.0, max(self.wavelengths) + 80.0)

        w, curve = calculate_uv_vis_spectrum(
            self.wavelengths, self.oscs, fwhm_nm=self.fwhm_nm, w_min=w_min, w_max=w_max
        )

        max_curve = np.max(curve) if len(curve) > 0 and np.max(curve) > 0 else 1.0
        norm_curve = curve / max_curve
        self.ax.plot(w, norm_curve, color="#a855f7", lw=1.8, label="UV-Vis Envelope")
        self.ax.fill_between(w, norm_curve, color="#9333ea", alpha=0.25)

        if self.chk_sticks.isChecked():
            oscs = self.oscs if self.oscs is not None else [1.0] * len(self.wavelengths)
            max_osc = max(oscs) if oscs and max(oscs) > 0 else 1.0
            for idx, (wl, f) in enumerate(zip(self.wavelengths, oscs)):
                h = f / max_osc
                is_active = (idx == self.active_idx)
                col = "#38bdf8" if is_active else "#f43f5e"
                lw = 2.4 if is_active else 1.2
                self.ax.vlines(wl, 0, h, color=col, lw=lw, alpha=0.9)

        self.ax.set_xlim(w_min, w_max)
        self.ax.set_ylim(0, 1.05)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.wavelengths:
            return
        clicked_w = event.xdata
        best_idx = None
        min_dist = float("inf")
        for idx, wl in enumerate(self.wavelengths):
            dist = abs(wl - clicked_w)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        if best_idx is not None and min_dist < max(15.0, 1.5 * self.fwhm_nm):
            self.active_idx = best_idx
            self.transition_clicked.emit(best_idx)
            self._replot()
