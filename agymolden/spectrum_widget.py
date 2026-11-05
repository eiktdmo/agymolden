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


def export_spectrum_figure_dialog(parent: QtWidgets.QWidget, fig: Figure, default_stem: str = "spectrum") -> Optional[str]:
    """Export Matplotlib spectrum figure to publication-quality vector (PDF, SVG, EPS) or high-DPI raster (PNG)."""
    filter_str = (
        "Portable Document Format (*.pdf);;"
        "Scalable Vector Graphics (*.svg);;"
        "High-DPI Publication PNG (*.png);;"
        "Encapsulated PostScript (*.eps);;"
        "All Files (*)"
    )
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        parent,
        f"Export Publication Spectrum Plot ({default_stem.upper()})",
        f"{default_stem}.pdf",
        filter_str,
    )
    if not path:
        return None

    dlg = QtWidgets.QDialog(parent)
    dlg.setWindowTitle("Publication Figure Options")
    dlg.resize(360, 160)
    layout = QtWidgets.QVBoxLayout(dlg)

    form = QtWidgets.QFormLayout()
    combo_bg = QtWidgets.QComboBox()
    combo_bg.addItem("Pure White (Standard for Papers / Print)", "white")
    combo_bg.addItem("Transparent Background", "transparent")
    combo_bg.addItem("Dark Theme (Current Screen View)", "dark")
    form.addRow("Background:", combo_bg)

    spin_dpi = QtWidgets.QSpinBox()
    spin_dpi.setRange(72, 1200)
    spin_dpi.setValue(300)
    spin_dpi.setSingleStep(50)
    spin_dpi.setSuffix(" DPI")
    form.addRow("Raster DPI (for PNG):", spin_dpi)
    layout.addLayout(form)

    btn_box = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel
    )
    btn_box.accepted.connect(dlg.accept)
    btn_box.rejected.connect(dlg.reject)
    layout.addWidget(btn_box)

    bg_choice = "white"
    dpi_val = 300
    if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
        bg_choice = combo_bg.currentData()
        dpi_val = spin_dpi.value()
    else:
        return None

    orig_fig_fc = fig.get_facecolor()
    orig_ax_fc = [ax.get_facecolor() for ax in fig.axes]

    is_trans = (bg_choice == "transparent")
    save_fc = "white" if bg_choice == "white" else (orig_fig_fc if bg_choice == "dark" else "none")

    if bg_choice == "white":
        fig.set_facecolor("white")
        for ax in fig.axes:
            ax.set_facecolor("white")
            ax.tick_params(colors="black")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.title.set_color("black")
            for spine in ax.spines.values():
                spine.set_color("#333333")

    try:
        fig.savefig(
            path,
            dpi=dpi_val,
            bbox_inches="tight",
            facecolor=save_fc,
            transparent=is_trans,
        )
    finally:
        fig.set_facecolor(orig_fig_fc)
        for i, ax in enumerate(fig.axes):
            if i < len(orig_ax_fc):
                ax.set_facecolor(orig_ax_fc[i])
        if hasattr(parent, "_style_axes"):
            parent._style_axes()
        if hasattr(parent, "canvas") and hasattr(parent.canvas, "draw_idle"):
            parent.canvas.draw_idle()

    return path


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

        self.btn_export_csv = QtWidgets.QPushButton("💾 CSV")
        self.btn_export_csv.setToolTip("Export broadened IR/Raman spectrum curve to CSV")
        self.btn_export_csv.setStyleSheet("background-color: #0d9488; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        ctrl_layout.addWidget(self.btn_export_csv)

        self.btn_export_plot = QtWidgets.QPushButton("📷 Plot...")
        self.btn_export_plot.setToolTip("Export publication-quality vector (PDF, SVG, EPS) or raster (PNG) spectrum figure")
        self.btn_export_plot.setStyleSheet("background-color: #4f46e5; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_plot.clicked.connect(self._on_export_plot_clicked)
        ctrl_layout.addWidget(self.btn_export_plot)

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

    def get_csv_string(self) -> str:
        """Return broadened IR/Raman spectrum curve as a CSV string."""
        eff_freqs = self.get_effective_frequencies()
        active_ints = self.raman_activities if self.plot_mode == "Raman" else self.intensities
        w, curve = calculate_broadened_spectrum(eff_freqs, active_ints, fwhm=self.fwhm)
        max_c = float(np.max(curve)) if len(curve) > 0 and np.max(curve) > 0 else 1.0
        norm_c = curve / max_c
        lines = [f"Wavenumber_cm-1,{self.plot_mode}_Intensity,{self.plot_mode}_Normalized"]
        for wn, raw_v, norm_v in zip(w, curve, norm_c):
            lines.append(f"{wn:.2f},{raw_v:.6f},{norm_v:.6f}")
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.frequencies:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Export {self.plot_mode} Spectrum to CSV", f"{self.plot_mode.lower()}_spectrum.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())

    def _on_export_plot_clicked(self):
        export_spectrum_figure_dialog(self, self.fig, f"{self.plot_mode.lower()}_spectrum")


def calculate_uv_vis_spectrum(
    wavelengths_nm: List[float],
    oscillator_strengths: Optional[List[float]] = None,
    fwhm: float = 25.0,
    unit: str = "nm",
    profile: str = "gaussian",
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    n_points: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian or Lorentzian convolution of electronic excited-state transitions (UV-Vis spectrum).
    Supports Wavelength (nm), Transition Energy (eV), and Wavenumbers (cm⁻¹).
    """
    if not wavelengths_nm:
        w = np.linspace(190.0, 800.0, n_points)
        return w, np.zeros_like(w)

    oscs = oscillator_strengths if oscillator_strengths is not None else [1.0] * len(wavelengths_nm)

    # Convert wavelengths to target x unit
    unit_l = unit.lower()
    if "ev" in unit_l:
        x_trans = [1239.84193 / max(1.0, float(wl)) for wl in wavelengths_nm]
        def_min = max(0.5, min(x_trans) - 0.6)
        def_max = min(12.0, max(x_trans) + 0.8)
    elif "cm" in unit_l:
        x_trans = [1e7 / max(1.0, float(wl)) for wl in wavelengths_nm]
        def_min = max(4000.0, min(x_trans) - 3000.0)
        def_max = min(75000.0, max(x_trans) + 5000.0)
    else:  # default nm
        x_trans = [float(wl) for wl in wavelengths_nm]
        def_min = max(180.0, min(x_trans) - 50.0)
        def_max = min(1200.0, max(x_trans) + 80.0)

    cur_min = x_min if x_min is not None else def_min
    cur_max = x_max if x_max is not None else def_max
    if cur_min >= cur_max:
        cur_min, cur_max = cur_max - 1.0, cur_min + 1.0

    x = np.linspace(cur_min, cur_max, n_points)
    abs_curve = np.zeros_like(x)

    is_lorentzian = "lorentz" in profile.lower()
    if is_lorentzian:
        gamma = max(1e-4, fwhm / 2.0)
        for xt, f in zip(x_trans, oscs):
            if f <= 0.0:
                continue
            abs_curve += f * (gamma / (np.pi * ((x - xt) ** 2 + gamma ** 2)))
    else:
        sigma = max(1e-4, fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))
        for xt, f in zip(x_trans, oscs):
            if f <= 0.0:
                continue
            abs_curve += f * np.exp(-0.5 * ((x - xt) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))

    return x, abs_curve


class UVVisSpectrumWidget(QtWidgets.QWidget):
    """Interactive Matplotlib canvas widget for UV-Vis Electronic Absorption Spectra (GaussView / Multiwfn style)."""
    transition_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wavelengths: List[float] = []
        self.oscs: Optional[List[float]] = None
        self.active_idx: Optional[int] = None
        self.current_unit: str = "nm"
        self.current_profile: str = "gaussian"
        self.fwhm_val: float = 25.0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Ribbon controls
        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QtWidgets.QLabel("X-Axis:"))
        self.combo_unit = QtWidgets.QComboBox()
        self.combo_unit.addItems(["Wavelength (nm)", "Energy (eV)", "Wavenumber (cm⁻¹)"])
        self.combo_unit.currentIndexChanged.connect(self._on_unit_changed)
        ctrl_layout.addWidget(self.combo_unit)

        ctrl_layout.addWidget(QtWidgets.QLabel("Profile:"))
        self.combo_profile = QtWidgets.QComboBox()
        self.combo_profile.addItems(["Gaussian", "Lorentzian"])
        self.combo_profile.currentIndexChanged.connect(self._on_profile_changed)
        ctrl_layout.addWidget(self.combo_profile)

        ctrl_layout.addWidget(QtWidgets.QLabel("FWHM:"))
        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(2.0, 200.0)
        self.spin_fwhm.setValue(25.0)
        self.spin_fwhm.setSingleStep(5.0)
        self.spin_fwhm.setSuffix(" nm")
        self.spin_fwhm.valueChanged.connect(self._on_fwhm_changed)
        ctrl_layout.addWidget(self.spin_fwhm)

        self.chk_sticks = QtWidgets.QCheckBox("Show Sticks")
        self.chk_sticks.setChecked(True)
        self.chk_sticks.toggled.connect(self._replot)
        ctrl_layout.addWidget(self.chk_sticks)

        self.btn_sample = QtWidgets.QPushButton("✨ Sample")
        self.btn_sample.setToolTip("Load sample TD-DFT excited-state electronic spectrum")
        self.btn_sample.setStyleSheet("background-color: #6366f1; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_sample.clicked.connect(self.load_sample_uv_vis)
        ctrl_layout.addWidget(self.btn_sample)

        self.btn_export_csv = QtWidgets.QPushButton("💾 CSV")
        self.btn_export_csv.setToolTip("Export broadened UV-Vis absorption spectrum curve to CSV")
        self.btn_export_csv.setStyleSheet("background-color: #0d9488; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        ctrl_layout.addWidget(self.btn_export_csv)

        self.btn_export_plot = QtWidgets.QPushButton("📷 Plot...")
        self.btn_export_plot.setToolTip("Export publication-quality UV-Vis spectrum figure (PDF, SVG, PNG, EPS)")
        self.btn_export_plot.setStyleSheet("background-color: #4f46e5; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_plot.clicked.connect(self._on_export_plot_clicked)
        ctrl_layout.addWidget(self.btn_export_plot)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        # Matplotlib Figure & Canvas
        self.fig = Figure(figsize=(5, 3), dpi=90, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        layout.addWidget(self.canvas, stretch=1)

        # Bottom Readout Label
        self.lbl_info = QtWidgets.QLabel("Click any transition stick or peak to inspect energy and oscillator strength.")
        self.lbl_info.setStyleSheet("color: #38bdf8; font-family: monospace; font-size: 11px; padding: 2px;")
        layout.addWidget(self.lbl_info)

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

        if self.current_unit == "eV":
            self.ax.set_xlabel("Transition Energy (eV)", fontsize=9)
        elif self.current_unit == "cm-1":
            self.ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        else:
            self.ax.set_xlabel("Wavelength λ (nm)", fontsize=9)

        self.ax.set_ylabel("Absorbance / Oscillator Strength f", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def set_uv_vis_data(self, wavelengths_nm: List[float], oscillator_strengths: Optional[List[float]] = None):
        self.wavelengths = list(wavelengths_nm)
        self.oscs = list(oscillator_strengths) if oscillator_strengths is not None else None
        self.active_idx = None
        self.lbl_info.setText(f"Loaded {len(self.wavelengths)} electronic transition(s). Click stick to inspect.")
        self._replot()

    def load_sample_uv_vis(self):
        """Load realistic sample TD-DFT electronic transitions (Anthracene / Ru-complex inspired)."""
        sample_wls = [520.4, 452.1, 378.6, 356.2, 340.1, 285.4, 254.2, 220.8]
        sample_oscs = [0.015, 0.082, 0.041, 0.185, 0.240, 0.035, 1.250, 0.420]
        self.set_uv_vis_data(sample_wls, sample_oscs)

    def _on_unit_changed(self, idx: int):
        units = ["nm", "eV", "cm-1"]
        self.current_unit = units[idx] if 0 <= idx < len(units) else "nm"
        self.spin_fwhm.blockSignals(True)
        if self.current_unit == "eV":
            self.spin_fwhm.setRange(0.02, 2.0)
            self.spin_fwhm.setDecimals(2)
            self.spin_fwhm.setSingleStep(0.05)
            self.spin_fwhm.setSuffix(" eV")
            self.spin_fwhm.setValue(0.25)
            self.fwhm_val = 0.25
        elif self.current_unit == "cm-1":
            self.spin_fwhm.setRange(100.0, 10000.0)
            self.spin_fwhm.setDecimals(0)
            self.spin_fwhm.setSingleStep(200.0)
            self.spin_fwhm.setSuffix(" cm⁻¹")
            self.spin_fwhm.setValue(2000.0)
            self.fwhm_val = 2000.0
        else:
            self.spin_fwhm.setRange(2.0, 200.0)
            self.spin_fwhm.setDecimals(1)
            self.spin_fwhm.setSingleStep(5.0)
            self.spin_fwhm.setSuffix(" nm")
            self.spin_fwhm.setValue(25.0)
            self.fwhm_val = 25.0
        self.spin_fwhm.blockSignals(False)
        self._replot()

    def _on_profile_changed(self, idx: int):
        self.current_profile = "gaussian" if idx == 0 else "lorentzian"
        self._replot()

    def _on_fwhm_changed(self, val: float):
        self.fwhm_val = val
        self._replot()

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.wavelengths:
            self.ax.text(
                0.5, 0.5, "No UV-Vis excited state transitions loaded.\nOpen a TD-DFT / CIS output file or click ✨ Sample.",
                ha="center", va="center", color="#64748b", transform=self.ax.transAxes, fontsize=9
            )
            self.canvas.draw()
            return

        x, curve = calculate_uv_vis_spectrum(
            self.wavelengths,
            self.oscs,
            fwhm=self.fwhm_val,
            unit=self.current_unit,
            profile=self.current_profile,
        )

        max_curve = np.max(curve) if len(curve) > 0 and np.max(curve) > 0 else 1.0
        norm_curve = curve / max_curve
        prof_label = "Gaussian" if self.current_profile == "gaussian" else "Lorentzian"
        self.ax.plot(x, norm_curve, color="#a855f7", lw=1.8, label=f"UV-Vis ({prof_label})")
        self.ax.fill_between(x, norm_curve, color="#9333ea", alpha=0.25)

        if self.chk_sticks.isChecked():
            oscs = self.oscs if self.oscs is not None else [1.0] * len(self.wavelengths)
            max_osc = max(oscs) if oscs and max(oscs) > 0 else 1.0

            if self.current_unit == "eV":
                x_trans = [1239.84193 / max(1.0, float(wl)) for wl in self.wavelengths]
            elif self.current_unit == "cm-1":
                x_trans = [1e7 / max(1.0, float(wl)) for wl in self.wavelengths]
            else:
                x_trans = list(self.wavelengths)

            for idx, (xt, f) in enumerate(zip(x_trans, oscs)):
                h = f / max_osc
                is_active = (idx == self.active_idx)
                col = "#38bdf8" if is_active else "#f43f5e"
                lw = 2.4 if is_active else 1.2
                self.ax.vlines(xt, 0, h, color=col, lw=lw, alpha=0.9)

        if len(x) > 0:
            self.ax.set_xlim(x[0], x[-1])
        self.ax.set_ylim(0, 1.05)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.wavelengths:
            return
        clicked_x = event.xdata

        if self.current_unit == "eV":
            x_trans = [1239.84193 / max(1.0, float(wl)) for wl in self.wavelengths]
            tol = max(0.1, 1.2 * self.fwhm_val)
        elif self.current_unit == "cm-1":
            x_trans = [1e7 / max(1.0, float(wl)) for wl in self.wavelengths]
            tol = max(800.0, 1.2 * self.fwhm_val)
        else:
            x_trans = list(self.wavelengths)
            tol = max(15.0, 1.2 * self.fwhm_val)

        best_idx = None
        min_dist = float("inf")
        for idx, xt in enumerate(x_trans):
            dist = abs(xt - clicked_x)
            if dist < min_dist:
                min_dist = dist
                best_idx = idx

        if best_idx is not None and min_dist < tol:
            self.active_idx = best_idx
            wl = self.wavelengths[best_idx]
            ev = 1239.84193 / max(1.0, wl)
            cm = 1e7 / max(1.0, wl)
            f = self.oscs[best_idx] if self.oscs and best_idx < len(self.oscs) else 0.0
            self.lbl_info.setText(
                f"State #{best_idx+1}: λ = {wl:.1f} nm  |  E = {ev:.2f} eV  |  ṽ = {cm:,.0f} cm⁻¹  |  f_osc = {f:.4f}"
            )
            self.transition_clicked.emit(best_idx)
            self._replot()

    def get_csv_string(self) -> str:
        """Return broadened UV-Vis absorption spectrum curve as a CSV string."""
        x, curve = calculate_uv_vis_spectrum(
            self.wavelengths,
            self.oscs,
            fwhm=self.fwhm_val,
            unit=self.current_unit,
            profile=self.current_profile,
        )
        max_c = float(np.max(curve)) if len(curve) > 0 and np.max(curve) > 0 else 1.0
        norm_c = curve / max_c
        col_x = "Wavelength_nm" if self.current_unit == "nm" else f"X_{self.current_unit}"
        header = f"{col_x},Absorbance_Raw,Absorbance_Normalized"
        lines = [header]
        for xv, raw_v, norm_v in zip(x, curve, norm_c):
            lines.append(f"{xv:.3f},{raw_v:.6f},{norm_v:.6f}")
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.wavelengths:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export UV-Vis Spectrum to CSV", "uvvis_spectrum.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())

    def _on_export_plot_clicked(self):
        export_spectrum_figure_dialog(self, self.fig, "uv_vis_spectrum")


# Standard isotropic shielding references (ppm) for Tetramethylsilane (TMS) / common nuclei
DEFAULT_NMR_TMS_REF: dict = {
    "H": 31.88,
    "C": 186.4,
    "N": -135.8,
    "F": 196.0,
    "P": 328.4,
    "Si": 374.5,
}


class NMRSpectrumWidget(QtWidgets.QWidget):
    """
    Interactive Matplotlib canvas widget for Simulated Isotropic NMR Chemical Shift Spectra
    (1H, 13C, 19F, 31P, etc., GaussView / Chemcraft style).
    Converts isotropic shielding sigma_iso (ppm) to chemical shift delta = sigma_ref - sigma_iso (ppm).
    """

    atom_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.nmr_shieldings: List[dict] = []
        self.active_atom_idx: Optional[int] = None
        self.fwhm_ppm: float = 0.15

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QtWidgets.QLabel("Nucleus:"))
        self.combo_nucleus = QtWidgets.QComboBox()
        self.combo_nucleus.addItems(["H (1H)", "C (13C)", "F (19F)", "P (31P)", "N (15N)"])
        self.combo_nucleus.currentTextChanged.connect(self._on_nucleus_changed)
        ctrl_layout.addWidget(self.combo_nucleus)

        ctrl_layout.addWidget(QtWidgets.QLabel("σ_ref (TMS):"))
        self.spin_ref = QtWidgets.QDoubleSpinBox()
        self.spin_ref.setRange(-1000.0, 2000.0)
        self.spin_ref.setDecimals(2)
        self.spin_ref.setValue(31.88)
        self.spin_ref.setSuffix(" ppm")
        self.spin_ref.setToolTip("Isotropic shielding of reference standard (e.g. TMS: ~31.88 ppm for 1H, ~186.4 ppm for 13C): δ = σ_ref − σ_iso")
        self.spin_ref.valueChanged.connect(lambda _v: self._replot())
        ctrl_layout.addWidget(self.spin_ref)

        ctrl_layout.addWidget(QtWidgets.QLabel("FWHM:"))
        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(0.02, 5.0)
        self.spin_fwhm.setDecimals(2)
        self.spin_fwhm.setSingleStep(0.05)
        self.spin_fwhm.setValue(0.15)
        self.spin_fwhm.setSuffix(" ppm")
        self.spin_fwhm.valueChanged.connect(self._on_fwhm_changed)
        ctrl_layout.addWidget(self.spin_fwhm)

        self.btn_export_csv = QtWidgets.QPushButton("💾 CSV")
        self.btn_export_csv.setToolTip("Export NMR chemical shifts and shielding table to CSV")
        self.btn_export_csv.setStyleSheet("background-color: #0d9488; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        ctrl_layout.addWidget(self.btn_export_csv)

        self.btn_export_plot = QtWidgets.QPushButton("📷 Plot...")
        self.btn_export_plot.setToolTip("Export publication-quality NMR spectrum figure (PDF, SVG, PNG, EPS)")
        self.btn_export_plot.setStyleSheet("background-color: #4f46e5; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_plot.clicked.connect(self._on_export_plot_clicked)
        ctrl_layout.addWidget(self.btn_export_plot)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        self.fig = Figure(figsize=(5, 3), dpi=90, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        layout.addWidget(self.canvas, stretch=1)

    def _active_element(self) -> str:
        txt = self.combo_nucleus.currentText()
        return txt.split()[0].strip() if txt else "H"

    def _on_nucleus_changed(self, _txt: str):
        elem = self._active_element()
        ref_val = DEFAULT_NMR_TMS_REF.get(elem, 0.0)
        self.spin_ref.blockSignals(True)
        self.spin_ref.setValue(ref_val)
        self.spin_ref.blockSignals(False)
        self.spin_fwhm.blockSignals(True)
        self.spin_fwhm.setValue(0.15 if elem == "H" else 1.20)
        self.spin_fwhm.blockSignals(False)
        self.fwhm_ppm = self.spin_fwhm.value()
        self._replot()

    def _on_fwhm_changed(self, val: float):
        self.fwhm_ppm = float(val)
        self._replot()

    def set_nmr_data(self, nmr_shieldings: List[dict]):
        """Load parsed NMR shielding dicts: [{'atom_index': int, 'symbol': str, 'sigma_iso': float, 'anisotropy': float}]."""
        self.nmr_shieldings = list(nmr_shieldings or [])
        self.active_atom_idx = None
        if self.nmr_shieldings:
            elems = sorted({str(d.get("symbol", "H")) for d in self.nmr_shieldings})
            self.combo_nucleus.blockSignals(True)
            self.combo_nucleus.clear()
            # Put H and C first if present
            ordered = [e for e in ("H", "C", "F", "P", "N", "O", "Si") if e in elems] + [
                e for e in elems if e not in ("H", "C", "F", "P", "N", "O", "Si")
            ]
            for e in ordered:
                lbl = {"H": "H (1H)", "C": "C (13C)", "F": "F (19F)", "P": "P (31P)", "N": "N (15N)"}.get(e, e)
                self.combo_nucleus.addItem(lbl)
            self.combo_nucleus.blockSignals(False)
            self._on_nucleus_changed(self.combo_nucleus.currentText())
        else:
            self._replot()

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
        self.ax.set_xlabel("Chemical Shift δ (ppm)", fontsize=9)
        self.ax.set_ylabel("Relative Intensity", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def _current_peaks(self) -> List[Tuple[int, str, float, float]]:
        """Return list of (atom_index, symbol, sigma_iso, delta_ppm) for currently selected nucleus."""
        elem = self._active_element()
        ref = float(self.spin_ref.value())
        peaks = []
        for item in self.nmr_shieldings:
            if str(item.get("symbol", "")) == elem:
                at_idx = int(item.get("atom_index", 0))
                sig = float(item.get("sigma_iso", 0.0))
                delta = ref - sig
                peaks.append((at_idx, elem, sig, delta))
        return peaks

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        peaks = self._current_peaks()
        if not peaks:
            self.ax.text(
                0.5,
                0.5,
                "No NMR chemical shieldings loaded.\nOpen an ORCA or Gaussian NMR (GIAO) output file.",
                ha="center",
                va="center",
                color="#64748b",
                transform=self.ax.transAxes,
                fontsize=9,
            )
            self.canvas.draw()
            return

        deltas = [p[3] for p in peaks]
        margin = max(1.5, 8.0 * self.fwhm_ppm)
        d_min = min(deltas) - margin
        d_max = max(deltas) + margin
        x_grid = np.linspace(d_min, d_max, 1000)
        y_curve = np.zeros_like(x_grid)
        gamma = max(0.01, self.fwhm_ppm * 0.5)

        for _at_idx, _sym, _sig, d_val in peaks:
            y_curve += (gamma * gamma) / ((x_grid - d_val) ** 2 + gamma * gamma)

        max_y = float(np.max(y_curve)) if len(y_curve) > 0 and np.max(y_curve) > 0 else 1.0
        norm_y = y_curve / max_y

        self.ax.plot(x_grid, norm_y, color="#10b981", lw=1.6, label="Lorentzian NMR Envelope")
        self.ax.fill_between(x_grid, norm_y, color="#059669", alpha=0.22)

        for at_idx, sym, _sig, d_val in peaks:
            is_act = (at_idx == self.active_atom_idx)
            col = "#f59e0b" if is_act else "#38bdf8"
            lw = 2.4 if is_act else 1.3
            self.ax.vlines(d_val, 0.0, 0.85, color=col, lw=lw, alpha=0.9)
            self.ax.text(
                d_val,
                0.88,
                f"{sym}{at_idx + 1}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color=col,
                fontweight="bold" if is_act else "normal",
            )

        # Standard NMR convention: high ppm on the left, 0 ppm on the right
        self.ax.set_xlim(d_max, d_min)
        self.ax.set_ylim(0.0, 1.08)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None:
            return
        peaks = self._current_peaks()
        if not peaks:
            return
        clicked_d = float(event.xdata)
        best_at = None
        min_dist = float("inf")
        for at_idx, _sym, _sig, d_val in peaks:
            dist = abs(d_val - clicked_d)
            if dist < min_dist:
                min_dist = dist
                best_at = at_idx
        if best_at is not None:
            self.active_atom_idx = best_at
            self.atom_clicked.emit(best_at)
            self._replot()

    def get_csv_string(self) -> str:
        """Return NMR shielding and chemical shift table as CSV string."""
        ref = float(self.spin_ref.value())
        lines = ["Atom_Index,Symbol,Sigma_Iso_ppm,Anisotropy_ppm,Delta_Shift_ppm"]
        for item in self.nmr_shieldings:
            at_idx = int(item.get("atom_index", 0)) + 1
            sym = str(item.get("symbol", "X"))
            sig = float(item.get("sigma_iso", 0.0))
            aniso = float(item.get("anisotropy", 0.0))
            ref_e = DEFAULT_NMR_TMS_REF.get(sym, ref)
            delta = ref_e - sig
            lines.append(f"{at_idx},{sym},{sig:.4f},{aniso:.4f},{delta:.4f}")
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.nmr_shieldings:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export NMR Shifts to CSV", "nmr_shifts.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())

    def _on_export_plot_clicked(self):
        export_spectrum_figure_dialog(self, self.fig, f"nmr_{self._active_element()}_spectrum")


class DOSSpectrumWidget(QtWidgets.QWidget):
    """
    Interactive Matplotlib canvas widget for Electronic Density of States (TDOS) Spectra
    (GaussSum / Multiwfn style).
    Convolves occupied and virtual Molecular Orbital eigenvalues (eV) with Gaussian functions
    and marks the HOMO-LUMO gap.
    """

    orbital_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.energies_ev: List[float] = []
        self.occupancies: List[float] = []
        self.homo_idx: Optional[int] = None
        self.fwhm_ev: float = 0.50

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setSpacing(6)

        ctrl_layout.addWidget(QtWidgets.QLabel("Gaussian Broadening (FWHM):"))
        self.spin_fwhm = QtWidgets.QDoubleSpinBox()
        self.spin_fwhm.setRange(0.05, 3.00)
        self.spin_fwhm.setDecimals(2)
        self.spin_fwhm.setSingleStep(0.10)
        self.spin_fwhm.setValue(0.50)
        self.spin_fwhm.setSuffix(" eV")
        self.spin_fwhm.valueChanged.connect(self._on_fwhm_changed)
        ctrl_layout.addWidget(self.spin_fwhm)

        self.chk_sticks = QtWidgets.QCheckBox("Show MO Levels (Sticks)")
        self.chk_sticks.setChecked(True)
        self.chk_sticks.toggled.connect(self._replot)
        ctrl_layout.addWidget(self.chk_sticks)

        self.btn_export_csv = QtWidgets.QPushButton("💾 CSV")
        self.btn_export_csv.setToolTip("Export Electronic Density of States (TDOS) curve to CSV")
        self.btn_export_csv.setStyleSheet("background-color: #0d9488; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_csv.clicked.connect(self._on_export_csv_clicked)
        ctrl_layout.addWidget(self.btn_export_csv)

        self.btn_export_plot = QtWidgets.QPushButton("📷 Plot...")
        self.btn_export_plot.setToolTip("Export publication-quality DOS spectrum figure (PDF, SVG, PNG, EPS)")
        self.btn_export_plot.setStyleSheet("background-color: #4f46e5; color: white; padding: 2px 6px; font-weight: bold;")
        self.btn_export_plot.clicked.connect(self._on_export_plot_clicked)
        ctrl_layout.addWidget(self.btn_export_plot)

        ctrl_layout.addStretch()
        layout.addLayout(ctrl_layout)

        self.fig = Figure(figsize=(5, 3), dpi=90, facecolor="#1e1e24")
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._style_axes()
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)

        layout.addWidget(self.canvas, stretch=1)

    def _on_fwhm_changed(self, val: float):
        self.fwhm_ev = float(val)
        self._replot()

    def set_orbitals(self, orbitals: List, homo_idx: Optional[int] = None):
        """Populate TDOS from a list of MoldenMO objects (where mo.energy is in Hartree)."""
        self.energies_ev = [float(getattr(mo, "energy", 0.0)) * 27.211386245988 for mo in (orbitals or [])]
        self.occupancies = [float(getattr(mo, "occupancy", 0.0)) for mo in (orbitals or [])]
        self.homo_idx = homo_idx
        self._replot()

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
        self.ax.set_xlabel("Orbital Energy E (eV)", fontsize=9)
        self.ax.set_ylabel("Density of States ( states / eV )", fontsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(spine_col)
        self.ax.grid(True, color=spine_col, linestyle="--", alpha=0.6)

    def _compute_dos_curves(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.energies_ev:
            z = np.zeros(10)
            return z, z, z, z
        # Focus around valence window [-30 eV, +15 eV] if wide basis set is present
        valence_e = [e for e in self.energies_ev if -35.0 <= e <= 20.0]
        if len(valence_e) >= 2:
            e_min = min(valence_e) - 3.0
            e_max = max(valence_e) + 3.0
        else:
            e_min = min(self.energies_ev) - 3.0
            e_max = max(self.energies_ev) + 3.0

        grid = np.linspace(e_min, e_max, 800)
        occ_dos = np.zeros_like(grid)
        virt_dos = np.zeros_like(grid)
        sigma = max(0.02, self.fwhm_ev / (2.0 * np.sqrt(2.0 * np.log(2.0))))
        pref = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

        for e_val, occ in zip(self.energies_ev, self.occupancies):
            if e_val < e_min - 4.0 or e_val > e_max + 4.0:
                continue
            g_pk = pref * np.exp(-0.5 * ((grid - e_val) / sigma) ** 2)
            if occ > 0.05:
                occ_dos += g_pk
            else:
                virt_dos += g_pk

        tot_dos = occ_dos + virt_dos
        return grid, tot_dos, occ_dos, virt_dos

    def _replot(self):
        self.ax.clear()
        self._style_axes()

        if not self.energies_ev:
            self.ax.text(
                0.5,
                0.5,
                "No Molecular Orbitals loaded for Density of States.\nOpen a .molden or QM output file.",
                ha="center",
                va="center",
                color="#64748b",
                transform=self.ax.transAxes,
                fontsize=9,
            )
            self.canvas.draw()
            return

        grid, tot_dos, occ_dos, virt_dos = self._compute_dos_curves()
        max_d = float(np.max(tot_dos)) if len(tot_dos) > 0 and np.max(tot_dos) > 0 else 1.0

        self.ax.plot(grid, tot_dos, color="#94a3b8", lw=1.2, label="Total DOS")
        self.ax.fill_between(grid, occ_dos, color="#2563eb", alpha=0.35, label="Occupied MOs")
        self.ax.fill_between(grid, virt_dos, color="#db2777", alpha=0.22, label="Virtual MOs")

        if self.chk_sticks.isChecked():
            e_min, e_max = float(grid[0]), float(grid[-1])
            for e_val, occ in zip(self.energies_ev, self.occupancies):
                if e_min <= e_val <= e_max:
                    col = "#3b82f6" if occ > 0.05 else "#f43f5e"
                    self.ax.vlines(e_val, 0.0, max_d * 0.28, color=col, lw=1.1, alpha=0.8)

        if self.homo_idx is not None and 0 <= self.homo_idx < len(self.energies_ev):
            e_h = self.energies_ev[self.homo_idx]
            self.ax.axvline(e_h, color="#38bdf8", linestyle="--", lw=1.2, label=f"HOMO ({e_h:+.2f} eV)")
            if self.homo_idx + 1 < len(self.energies_ev):
                e_l = self.energies_ev[self.homo_idx + 1]
                self.ax.axvline(e_l, color="#f43f5e", linestyle="--", lw=1.2, label=f"LUMO ({e_l:+.2f} eV)")

        self.ax.set_xlim(float(grid[0]), float(grid[-1]))
        self.ax.set_ylim(0.0, max_d * 1.12)
        self.ax.legend(fontsize=7.5, loc="upper right", framealpha=0.75)
        self.fig.tight_layout()
        self.canvas.draw()

    def _on_canvas_click(self, event):
        if event.xdata is None or not self.energies_ev:
            return
        clicked_e = float(event.xdata)
        best_idx = int(np.argmin(np.abs(np.asarray(self.energies_ev) - clicked_e)))
        self.orbital_clicked.emit(best_idx)

    def get_csv_string(self) -> str:
        """Return Electronic Density of States (TDOS, Occupied DOS, Virtual DOS) curve as CSV."""
        grid, tot_dos, occ_dos, virt_dos = self._compute_dos_curves()
        lines = ["Energy_eV,Total_DOS,Occupied_DOS,Virtual_DOS"]
        for e_v, t_d, o_d, v_d in zip(grid, tot_dos, occ_dos, virt_dos):
            lines.append(f"{e_v:.4f},{t_d:.6f},{o_d:.6f},{v_d:.6f}")
        return "\n".join(lines) + "\n"

    def _on_export_csv_clicked(self):
        if not self.energies_ev:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Electronic DOS to CSV", "electronic_dos.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())

    def _on_export_plot_clicked(self):
        export_spectrum_figure_dialog(self, self.fig, "dos_spectrum")
