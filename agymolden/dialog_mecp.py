"""Interactive Minimum Energy Crossing Point (MECP) & Two-State Reactivity Dialog."""

from __future__ import annotations

from typing import List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .mecp_tools import interpolate_mecp_seam, generate_orca_mecp_input


class MECPExplorerDialog(QtWidgets.QDialog):
    """Interactive Two-State Reactivity (TSR) & MECP Crossing Seam Explorer."""

    crossing_selected = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, coords) to send to viewer

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🔀 Minimum Energy Crossing Point (MECP) & Spin-Inversion Explorer")
        self.resize(850, 620)
        self.setMinimumSize(750, 520)

        self.symbols = list(symbols) if symbols else ["Fe", "O", "C", "H", "H", "H"]
        self.coords = np.array(coords, copy=True) if coords is not None else np.zeros((len(self.symbols), 3))

        self._build_ui()
        self._recompute_seam()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Two-State Reactivity (TSR) & Minimum Energy Crossing Point (MECP) Explorer</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Analyzes spin-forbidden reaction pathways that cross between two spin states (e.g., Singlet ⇄ Triplet, Low-Spin ⇄ High-Spin). "
            "Interpolates the crossing seam geometry (ΔE → 0) and generates ORCA / Gaussian MECP search decks."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Controls Row
        ctrl_box = QtWidgets.QGroupBox("Spin States & Diabatic Well Energetics")
        c_lay = QtWidgets.QGridLayout(ctrl_box)

        c_lay.addWidget(QtWidgets.QLabel("State 1 Multiplicity:"), 0, 0)
        self.spin_mult1 = QtWidgets.QSpinBox()
        self.spin_mult1.setRange(1, 15)
        self.spin_mult1.setValue(1)
        self.spin_mult1.valueChanged.connect(self._recompute_seam)
        c_lay.addWidget(self.spin_mult1, 0, 1)

        c_lay.addWidget(QtWidgets.QLabel("State 1 Reference Energy [eV]:"), 0, 2)
        self.spin_e1 = QtWidgets.QDoubleSpinBox()
        self.spin_e1.setRange(-1000.0, 1000.0)
        self.spin_e1.setDecimals(3)
        self.spin_e1.setValue(0.000)
        self.spin_e1.valueChanged.connect(self._recompute_seam)
        c_lay.addWidget(self.spin_e1, 0, 3)

        c_lay.addWidget(QtWidgets.QLabel("State 2 Multiplicity:"), 1, 0)
        self.spin_mult2 = QtWidgets.QSpinBox()
        self.spin_mult2.setRange(1, 15)
        self.spin_mult2.setValue(3)
        self.spin_mult2.valueChanged.connect(self._recompute_seam)
        c_lay.addWidget(self.spin_mult2, 1, 1)

        c_lay.addWidget(QtWidgets.QLabel("State 2 Reference Energy [eV]:"), 1, 2)
        self.spin_e2 = QtWidgets.QDoubleSpinBox()
        self.spin_e2.setRange(-1000.0, 1000.0)
        self.spin_e2.setDecimals(3)
        self.spin_e2.setValue(0.350)
        self.spin_e2.valueChanged.connect(self._recompute_seam)
        c_lay.addWidget(self.spin_e2, 1, 3)

        layout.addWidget(ctrl_box)

        # Plot Canvas
        self.fig = Figure(facecolor="#0f172a")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

        self.lbl_crossing = QtWidgets.QLabel()
        self.lbl_crossing.setStyleSheet("color: #facc15; font-weight: bold; font-size: 12px;")
        layout.addWidget(self.lbl_crossing)

        # Bottom Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_deck = QtWidgets.QPushButton("📝 Export ORCA MECP Input...")
        btn_deck.clicked.connect(self._export_orca_deck)
        btn_adopt = QtWidgets.QPushButton("📥 Send Seam Geometry to 3D Viewer")
        btn_adopt.setStyleSheet("background-color: #0284c7; font-weight: bold;")
        btn_adopt.clicked.connect(self._adopt_geometry)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)

        btn_box.addWidget(btn_deck)
        btn_box.addWidget(btn_adopt)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _recompute_seam(self):
        e1 = self.spin_e1.value()
        e2 = self.spin_e2.value()

        # Generate slightly perturbed coordinate for State 2 if starting from single geometry
        coords1 = self.coords
        coords2 = self.coords + np.random.RandomState(42).normal(0.0, 0.08, self.coords.shape)

        self.res = interpolate_mecp_seam(coords1, coords2, e1, e2, n_points=31)

        # Render PES plot
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor("#1e293b")

        x = self.res.coordinates
        ax.plot(x, self.res.energies_state1_ev, color="#38bdf8", linewidth=2.5, label=f"State 1 (Mult={self.spin_mult1.value()})")
        ax.plot(x, self.res.energies_state2_ev, color="#f43f5e", linewidth=2.5, label=f"State 2 (Mult={self.spin_mult2.value()})")

        # Crossing Point marker
        cx = self.res.crossing_lambda
        cy = self.res.crossing_energy_ev
        ax.plot(cx, cy, "o", color="#facc15", markersize=9, label="MECP Crossing Seam")
        ax.axvline(cx, color="#facc15", linestyle="--", alpha=0.6)

        ax.set_xlabel("Normalized Intersystem Reaction Coordinate (λ)", color="#cbd5e1", fontsize=10)
        ax.set_ylabel("Relative Energy (eV)", color="#cbd5e1", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#f8fafc", loc="upper center")

        self.fig.tight_layout()
        self.canvas.draw()

        de_barrier_kcal = (cy - e1) * 23.0605
        self.lbl_crossing.setText(
            f"⚡ Located Crossing Seam: λ = {cx:.2f} | E_seam = {cy:+.3f} eV | Barrier from State 1: +{de_barrier_kcal:.1f} kcal/mol"
        )

    def _export_orca_deck(self):
        deck = generate_orca_mecp_input(
            symbols=self.symbols,
            coords=self.res.geometries[self.res.crossing_index],
            mult_1=self.spin_mult1.value(),
            mult_2=self.spin_mult2.value(),
        )
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("ORCA MECP Search Deck")
        dlg.resize(600, 450)
        lay = QtWidgets.QVBoxLayout(dlg)
        txt = QtWidgets.QPlainTextEdit(deck)
        txt.setFont(QtGui.QFont("DejaVu Sans Mono", 9))
        lay.addWidget(txt)
        b_box = QtWidgets.QHBoxLayout()
        btn_c = QtWidgets.QPushButton("Copy to Clipboard")
        btn_c.clicked.connect(lambda: (QtWidgets.QApplication.clipboard().setText(deck), dlg.accept()))
        b_box.addWidget(btn_c)
        b_box.addStretch()
        lay.addLayout(b_box)
        dlg.exec()

    def _adopt_geometry(self):
        seam_coords = self.res.geometries[self.res.crossing_index]
        self.crossing_selected.emit(self.symbols, seam_coords)
        QtWidgets.QMessageBox.information(
            self, "Adopted", "Seam crossing geometry loaded into the 3D molecular viewer."
        )
