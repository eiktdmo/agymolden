"""Interactive Hückel Molecular Orbital (HMO) Teaching Workbench Dialog.

Features:
- Live Secular Determinant / Matrix Inspector (|H - E·I| = 0)
- Frost-Musulin Circle visualizer for cyclic [N]annulenes
- Aufbau MO Energy Ladder with spin-paired arrows (↑↓)
- Aromaticity & Delocalization Energy (DE) calculator
- Coulson Pi-Bond Order and Pi-Electron Density Table
- Real-time 3D Viewport orbital lobe and net charge projection
- Preset Classroom Demonstrations: Benzene, Butadiene, Cyclobutadiene, Allyl, Azulene, Pyridine, etc.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from .huckel_engine import (
    detect_pi_system,
    solve_huckel_system,
    HuckelResult,
    HuckelMO,
)

def _fused_bicyclic(n_left: int, n_right: int):
    """Planar fused bicyclic pi-system sharing one C-C bond (naphthalene 6+6, azulene 5+7).

    Atoms 0/1 are the top/bottom fusion carbons; the right ring's remaining atoms follow
    (top to bottom), then the left ring's.
    """
    side = 1.40
    coords = [(0.0, side / 2.0), (0.0, -side / 2.0)]
    bonds = [(0, 1)]

    def _ring(n: int, sign: float) -> None:
        apothem = side / (2.0 * math.tan(math.pi / n))
        radius = side / (2.0 * math.sin(math.pi / n))
        first_idx = len(coords)
        chain = [0]
        for k in range(1, n - 1):
            theta = math.pi - math.pi / n - k * 2.0 * math.pi / n
            coords.append((sign * (apothem + radius * math.cos(theta)), radius * math.sin(theta)))
            chain.append(first_idx + k - 1)
        chain.append(1)
        bonds.extend(zip(chain[:-1], chain[1:]))

    _ring(n_right, +1.0)
    _ring(n_left, -1.0)
    arr = np.array([[x, y, 0.0] for x, y in coords])
    return ["C"] * len(coords), arr, sorted((min(a, b), max(a, b)) for a, b in bonds)


_AZULENE = _fused_bicyclic(5, 7)
_NAPHTHALENE = _fused_bicyclic(6, 6)


# Classic Classroom Presets (Symbols, Coords, Pi Bonds)
HUCKEL_PRESETS: Dict[str, Tuple[List[str], np.ndarray, List[Tuple[int, int]], int]] = {
    "Ethylene (C2H4)": (
        ["C", "C"],
        np.array([[-0.67, 0.0, 0.0], [0.67, 0.0, 0.0]]),
        [(0, 1)],
        2,
    ),
    "1,3-Butadiene (C4H6)": (
        ["C", "C", "C", "C"],
        np.array([[-1.85, -0.4, 0.0], [-0.65, 0.35, 0.0], [0.65, -0.35, 0.0], [1.85, 0.4, 0.0]]),
        [(0, 1), (1, 2), (2, 3)],
        4,
    ),
    "Benzene (C6H6) [Aromatic 6π]": (
        ["C"] * 6,
        np.array([
            [1.40 * math.cos(i * math.pi / 3.0), 1.40 * math.sin(i * math.pi / 3.0), 0.0]
            for i in range(6)
        ]),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        6,
    ),
    "Cyclobutadiene (C4H4) [Antiaromatic 4π]": (
        ["C"] * 4,
        np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]),
        [(0, 1), (1, 2), (2, 3), (3, 0)],
        4,
    ),
    "Allyl Radical / Ion (C3H5)": (
        ["C", "C", "C"],
        np.array([[-1.25, -0.3, 0.0], [0.0, 0.4, 0.0], [1.25, -0.3, 0.0]]),
        [(0, 1), (1, 2)],
        3,  # 3 electrons for radical, 2 for cation, 4 for anion
    ),
    "Cyclopentadienyl Anion (C5H5⁻) [Aromatic 6π]": (
        ["C"] * 5,
        np.array([
            [1.25 * math.cos(i * 2.0 * math.pi / 5.0), 1.25 * math.sin(i * 2.0 * math.pi / 5.0), 0.0]
            for i in range(5)
        ]),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)],
        6,
    ),
    "Tropylium Cation (C7H7⁺) [Aromatic 6π]": (
        ["C"] * 7,
        np.array([
            [1.45 * math.cos(i * 2.0 * math.pi / 7.0), 1.45 * math.sin(i * 2.0 * math.pi / 7.0), 0.0]
            for i in range(7)
        ]),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0)],
        6,
    ),
    "Naphthalene (C10H8) [Fused 10π]": (_NAPHTHALENE[0], _NAPHTHALENE[1], _NAPHTHALENE[2], 10),
    "Azulene (C10H8) [Non-Alternant 5+7]": (_AZULENE[0], _AZULENE[1], _AZULENE[2], 10),
    "Pyridine (C5H5N) [Heterocycle]": (
        ["N", "C", "C", "C", "C", "C"],
        np.array([
            [1.40 * math.cos(i * math.pi / 3.0), 1.40 * math.sin(i * math.pi / 3.0), 0.0]
            for i in range(6)
        ]),
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)],
        6,
    ),
}


class HuckelTeachingDialog(QtWidgets.QDialog):
    """Undergraduate Quantum Chemistry Hückel Theory Interactive Teaching Dialog."""

    project_orbital_to_viewer = QtCore.pyqtSignal(list, np.ndarray)  # (symbols, custom_weights)
    project_charges_to_viewer = QtCore.pyqtSignal(list, np.ndarray)

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        initial_symbols: Optional[List[str]] = None,
        initial_coords: Optional[np.ndarray] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🎓 Hückel Molecular Orbital (HMO) Theory Teaching Workbench")
        self.resize(1000, 720)
        self.setMinimumSize(850, 600)

        self.current_symbols = list(initial_symbols) if initial_symbols else ["C"] * 6
        self.current_coords = np.array(initial_coords, copy=True) if initial_coords is not None else np.zeros((len(self.current_symbols), 3))
        self.current_bonds: List[Tuple[int, int]] = []
        self.current_res: Optional[HuckelResult] = None

        self._build_ui()
        self._load_from_parent_or_preset()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header Frame
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>🎓 Hückel Molecular Orbital (HMO) Theory Teaching Workbench</b>")
        h_title.setStyleSheet("font-size: 15px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "An interactive quantum mechanics learning suite: Explore the secular determinant |H - E·I| = 0, "
            "Frost-Musulin aromaticity circles, Hund's rule Aufbau electron filling, Coulson π-bond orders, "
            "and LCAO 3D π-orbital lobe nodal planes."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_desc.setWordWrap(True)
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Ribbon Controls (Presets + Electron Slider)
        ribbon = QtWidgets.QHBoxLayout()
        ribbon.addWidget(QtWidgets.QLabel("<b>Classroom Preset:</b>"))
        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItem("From Current 3D Viewport Structure", "viewport")
        for name in HUCKEL_PRESETS.keys():
            self.combo_presets.addItem(name, name)
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        ribbon.addWidget(self.combo_presets)

        ribbon.addSpacing(15)
        ribbon.addWidget(QtWidgets.QLabel("<b>π-Electrons (N_π):</b>"))
        self.spin_electrons = QtWidgets.QSpinBox()
        self.spin_electrons.setRange(0, 30)
        self.spin_electrons.setValue(6)
        self.spin_electrons.valueChanged.connect(self._recalculate)
        ribbon.addWidget(self.spin_electrons)

        ribbon.addSpacing(15)
        ribbon.addWidget(QtWidgets.QLabel("<b>|β| (eV):</b>"))
        self.spin_beta = QtWidgets.QDoubleSpinBox()
        self.spin_beta.setRange(0.5, 5.0)
        self.spin_beta.setSingleStep(0.1)
        self.spin_beta.setValue(2.70)
        self.spin_beta.valueChanged.connect(self._recalculate)
        ribbon.addWidget(self.spin_beta)

        ribbon.addStretch()
        layout.addLayout(ribbon)

        # Main Splitter: Left (Plots: MO Ladder & Frost Circle) | Right (Inspectors & Tables)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left Container: Matplotlib Canvas
        left_container = QtWidgets.QWidget()
        l_lay = QtWidgets.QVBoxLayout(left_container)
        l_lay.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(facecolor="#0f172a")
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        l_lay.addWidget(self.canvas)

        self.lbl_aromatic_pill = QtWidgets.QLabel()
        self.lbl_aromatic_pill.setStyleSheet("background-color: #1e293b; color: #4ade80; font-weight: bold; padding: 6px; border-radius: 4px;")
        l_lay.addWidget(self.lbl_aromatic_pill)

        splitter.addWidget(left_container)

        # Right Container: Tabs
        tabs = QtWidgets.QTabWidget()

        # Tab 1: Secular Determinant & Matrix
        tab_sec = QtWidgets.QWidget()
        t_sec_lay = QtWidgets.QVBoxLayout(tab_sec)
        self.txt_secular = QtWidgets.QTextBrowser()
        self.txt_secular.setFont(QtGui.QFont("DejaVu Sans Mono", 10))
        self.txt_secular.setStyleSheet("background-color: #090d16; color: #e2e8f0;")
        t_sec_lay.addWidget(self.txt_secular)
        tabs.addTab(tab_sec, "📐 Secular Determinant")

        # Tab 2: LCAO Molecular Orbitals & Nodal Planes
        tab_mo = QtWidgets.QWidget()
        t_mo_lay = QtWidgets.QVBoxLayout(tab_mo)

        self.tbl_mo = QtWidgets.QTableWidget()
        self.tbl_mo.setColumnCount(6)
        self.tbl_mo.setHorizontalHeaderLabels(["Orbital", "Energy (α + mβ)", "Occupancy", "Type", "Nodes", "Action"])
        self.tbl_mo.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tbl_mo.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        t_mo_lay.addWidget(self.tbl_mo)

        btn_project_mo = QtWidgets.QPushButton("🔮 Project Selected MO Lobes onto 3D Molecule")
        btn_project_mo.setStyleSheet("background-color: #0284c7; font-weight: bold; padding: 6px;")
        btn_project_mo.clicked.connect(self._on_project_mo)
        t_mo_lay.addWidget(btn_project_mo)

        tabs.addTab(tab_mo, "🌊 LCAO Orbitals")

        # Tab 3: Coulson Bond Orders & Net Pi-Charges
        tab_coulson = QtWidgets.QWidget()
        t_c_lay = QtWidgets.QVBoxLayout(tab_coulson)
        self.txt_coulson = QtWidgets.QTextBrowser()
        self.txt_coulson.setFont(QtGui.QFont("DejaVu Sans Mono", 10))
        self.txt_coulson.setStyleSheet("background-color: #090d16; color: #e2e8f0;")
        t_c_lay.addWidget(self.txt_coulson)

        btn_project_q = QtWidgets.QPushButton("⚡ Project Net π-Charges to 3D Atoms")
        btn_project_q.clicked.connect(self._on_project_charges)
        t_c_lay.addWidget(btn_project_q)

        tabs.addTab(tab_coulson, "🧪 Bond Orders & Densities")

        # Tab 4: UV-Vis & Electronic Color
        tab_uv = QtWidgets.QWidget()
        t_uv_lay = QtWidgets.QVBoxLayout(tab_uv)
        self.txt_uv = QtWidgets.QTextBrowser()
        self.txt_uv.setStyleSheet("background-color: #090d16; color: #e2e8f0;")
        t_uv_lay.addWidget(self.txt_uv)
        tabs.addTab(tab_uv, "🌈 UV-Vis Transition")

        splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)
        layout.addWidget(splitter)

        # Bottom Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Full HMO Summary")
        btn_copy.clicked.connect(self._copy_summary)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_copy)
        btn_box.addStretch()
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)

    def _load_from_parent_or_preset(self):
        # Default to Benzene preset for demonstration if current coords are empty
        pi_idx, pi_bonds = detect_pi_system(self.current_symbols, self.current_coords)
        if len(pi_idx) >= 2:
            self.combo_presets.setCurrentIndex(0)
            self._recalculate()
        else:
            self.combo_presets.setCurrentText("Benzene (C6H6) [Aromatic 6π]")

    def _on_preset_changed(self, idx: int):
        key = self.combo_presets.currentData()
        if key == "viewport":
            main_win = self.parent()
            if main_win and hasattr(main_win, "symbols") and hasattr(main_win, "coords"):
                self.current_symbols = list(main_win.symbols)
                self.current_coords = np.array(main_win.coords, copy=True)
                pi_idx, self.current_bonds = detect_pi_system(self.current_symbols, self.current_coords)
                self.spin_electrons.setValue(len(pi_idx))
        elif key in HUCKEL_PRESETS:
            syms, crds, bnds, n_e = HUCKEL_PRESETS[key]
            self.current_symbols = list(syms)
            self.current_coords = np.array(crds, copy=True)
            self.current_bonds = list(bnds)
            self.spin_electrons.blockSignals(True)
            self.spin_electrons.setValue(n_e)
            self.spin_electrons.blockSignals(False)

        self._recalculate()

    def _recalculate(self):
        key = self.combo_presets.currentData()
        if key == "viewport" or not self.current_bonds:
            pi_idx, pi_bonds = detect_pi_system(self.current_symbols, self.current_coords)
        else:
            pi_idx = list(range(len(self.current_symbols)))
            pi_bonds = list(self.current_bonds)

        if len(pi_idx) < 2:
            self.lbl_aromatic_pill.setText("⚠️ No conjugated π-system detected. Choose a classroom preset.")
            return

        n_e = self.spin_electrons.value()
        beta_val = -self.spin_beta.value()

        try:
            self.current_res = solve_huckel_system(
                self.current_symbols,
                pi_idx,
                pi_bonds,
                custom_electrons=n_e,
                beta_ev=beta_val,
            )
            self._render_plots(self.current_res)
            self._render_secular_tab(self.current_res)
            self._render_mo_table(self.current_res)
            self._render_coulson_tab(self.current_res)
            self._render_uv_tab(self.current_res)
        except Exception as e:
            self.lbl_aromatic_pill.setText(f"Calculation Error: {e}")

    def _render_plots(self, res: HuckelResult):
        self.fig.clear()

        if res.frost_polygon_n is not None:
            # 2 Subplots: Left = Frost-Musulin Circle, Right = MO Aufbau Ladder
            ax_frost = self.fig.add_subplot(121)
            ax_ladder = self.fig.add_subplot(122)
            self._draw_frost_circle(ax_frost, res)
            self._draw_mo_ladder(ax_ladder, res)
        else:
            # Single wide MO Aufbau Ladder for acyclic polyenes
            ax_ladder = self.fig.add_subplot(111)
            self._draw_mo_ladder(ax_ladder, res)

        self.fig.tight_layout()
        self.canvas.draw()

        color = "#4ade80" if "Aromatic" in res.aromaticity_status else ("#f43f5e" if "Antiaromatic" in res.aromaticity_status else "#facc15")
        self.lbl_aromatic_pill.setText(
            f"⚡ {res.aromaticity_status} | E_π = {len(res.pi_symbols)}α + {res.total_pi_energy_beta:.3f}β | Delocalization Energy (DE) = {res.delocalization_energy_beta:+.3f}β"
        )
        self.lbl_aromatic_pill.setStyleSheet(f"background-color: #1e293b; color: {color}; font-weight: bold; padding: 6px; border-radius: 4px;")

    def _draw_frost_circle(self, ax, res: HuckelResult):
        ax.set_facecolor("#1e293b")
        n = res.frost_polygon_n or 6

        # Circle of radius 2 centered at (0, 0)
        theta = np.linspace(0, 2.0 * math.pi, 200)
        ax.plot(2.0 * np.cos(theta), 2.0 * np.sin(theta), color="#64748b", linestyle="--", linewidth=1.2)

        # Inscribed polygon pointing DOWN: vertices at theta_k = -pi/2 + k * (2*pi/n)
        # Note: In Frost's mnemonic, energy increases UPWARDS.
        # Bottom-most vertex is at -2 (E = alpha + 2*beta)
        v_angles = [-math.pi / 2.0 + k * (2.0 * math.pi / n) for k in range(n + 1)]
        px = [2.0 * math.cos(a) for a in v_angles]
        py = [2.0 * math.sin(a) for a in v_angles]
        ax.plot(px, py, color="#38bdf8", linewidth=2.0)

        # Plot energy horizontal ticks at each unique vertex height
        unique_y = sorted(list({round(y, 4) for y in py[:-1]}))
        for y_val in unique_y:
            ax.hlines(y_val, -2.3, 2.3, colors="#94a3b8", linestyles=":", alpha=0.5)
            # m_beta = -y_val (since lowest y = -2 corresponds to m = +2)
            ax.text(2.35, y_val, f"m={-y_val:+.2f}", color="#38bdf8", va="center", fontsize=8)

        ax.axhline(0, color="#f43f5e", linestyle="-.", linewidth=0.8, alpha=0.7)
        ax.text(-2.3, 0.08, "Non-bonding (α)", color="#f43f5e", fontsize=8)

        ax.set_xlim(-2.8, 3.2)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect("equal")
        ax.set_title(f"Frost-Musulin Circle ([{n}]annulene)", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.axis("off")

    def _draw_mo_ladder(self, ax, res: HuckelResult):
        ax.set_facecolor("#1e293b")
        n_mo = len(res.orbitals)

        # Group degenerate orbitals
        grouped: Dict[float, List[HuckelMO]] = {}
        for mo in res.orbitals:
            key = round(mo.m_beta, 3)
            grouped.setdefault(key, []).append(mo)

        unique_m = sorted(grouped.keys(), reverse=True)

        for m_key in unique_m:
            orbs = grouped[m_key]
            n_deg = len(orbs)
            y_val = m_key  # higher m_beta = lower energy visually? Standard chemistry plots lower energy at BOTTOM!
            # So plot -m_key so that m = +2 (stable) is at bottom -2, and m = -2 (antibonding) is at top +2!
            plot_y = -m_key

            width = 0.5
            spacing = 0.7
            start_x = -((n_deg - 1) * spacing) / 2.0

            for i, mo in enumerate(orbs):
                cx = start_x + i * spacing
                col = "#22c55e" if mo.occupancy == 2 else ("#facc15" if mo.occupancy == 1 else "#38bdf8")
                ax.hlines(plot_y, cx - width / 2.0, cx + width / 2.0, colors=col, linewidth=3.5)

                # Spin arrows
                if mo.occupancy == 2:
                    ax.text(cx - 0.08, plot_y, "↑", color="#ffffff", fontweight="bold", fontsize=12, va="center")
                    ax.text(cx + 0.08, plot_y, "↓", color="#ffffff", fontweight="bold", fontsize=12, va="center")
                elif mo.occupancy == 1:
                    ax.text(cx, plot_y, "↑", color="#ffffff", fontweight="bold", fontsize=12, va="center", ha="center")

                # Label
                tag = f"ψ_{mo.index}"
                if mo.is_homo:
                    tag += " (HOMO)"
                elif mo.is_lumo:
                    tag += " (LUMO)"
                elif mo.is_somo:
                    tag += " (SOMO)"
                ax.text(cx + width / 2.0 + 0.08, plot_y, tag, color=col, va="center", fontsize=8, fontweight="bold")

        ax.axhline(0, color="#f43f5e", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(1.2, 0.05, "Non-bonding (α)", color="#f43f5e", fontsize=8)

        ax.set_ylabel("Energy (units of -β, Higher Energy ↑)", color="#cbd5e1", fontsize=9)
        ax.set_title("π-MO Energy Ladder & Aufbau", color="#f8fafc", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_color("#475569")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle=":", alpha=0.3, color="#64748b")

    def _render_secular_tab(self, res: HuckelResult):
        n = len(res.pi_symbols)
        # Render LaTeX / Text Secular Determinant in x = (alpha - E) / beta
        # Matrix H has H_ii = h_i, H_ij = k_ij
        lines = []
        lines.append("<b>Hückel Secular Determinant:</b><br>")
        lines.append("Setting <i>x = (α - E) / β</i>:<br><br>")

        det_lines = []
        for i in range(n):
            row_items = []
            for j in range(n):
                if i == j:
                    h_val = res.secular_matrix_h[i, i]
                    item = "x" if abs(h_val) < 1e-4 else f"(x + {h_val:.2f})"
                else:
                    k_val = res.secular_matrix_h[i, j]
                    item = "1" if abs(k_val - 1.0) < 1e-4 else ("0" if abs(k_val) < 1e-4 else f"{k_val:.2f}")
                row_items.append(f"{item:>8}")
            det_lines.append("│ " + "  ".join(row_items) + " │ = 0")

        lines.append("<pre style='color: #38bdf8; font-size: 11px;'>" + "\n".join(det_lines) + "</pre>")
        lines.append("<br><b>Secular Roots (E = α + m·β):</b><br><ul>")
        for mo in res.orbitals:
            lines.append(f"<li>ψ_{mo.index}: m = {mo.m_beta:+.4f} (E = {mo.energy_relative_beta})</li>")
        lines.append("</ul>")

        self.txt_secular.setHtml("".join(lines))

    def _render_mo_table(self, res: HuckelResult):
        self.tbl_mo.setRowCount(len(res.orbitals))
        for row, mo in enumerate(res.orbitals):
            self.tbl_mo.setItem(row, 0, QtWidgets.QTableWidgetItem(f"ψ_{mo.index}"))
            self.tbl_mo.setItem(row, 1, QtWidgets.QTableWidgetItem(mo.energy_relative_beta))
            self.tbl_mo.setItem(row, 2, QtWidgets.QTableWidgetItem(f"{mo.occupancy} e⁻"))

            m_type = "Bonding" if mo.m_beta > 0.05 else ("Antibonding" if mo.m_beta < -0.05 else "Non-bonding")
            if mo.is_homo:
                m_type += " (HOMO)"
            elif mo.is_lumo:
                m_type += " (LUMO)"
            self.tbl_mo.setItem(row, 3, QtWidgets.QTableWidgetItem(m_type))
            self.tbl_mo.setItem(row, 4, QtWidgets.QTableWidgetItem(f"{mo.nodal_planes} nodes"))

            btn = QtWidgets.QPushButton("View Lobes")
            btn.clicked.connect(lambda _, idx=row: self._project_specific_mo(idx))
            self.tbl_mo.setCellWidget(row, 5, btn)

    def _render_coulson_tab(self, res: HuckelResult):
        lines = []
        lines.append("<h3>Coulson π-Bond Orders & Free Valence:</h3>")
        lines.append("<table cellpadding='4' border='1' style='border-collapse: collapse; border-color: #334155; width: 100%;'>")
        lines.append("<tr style='background-color: #0f172a; color: #38bdf8;'><th>Bond</th><th>π-Bond Order (p_rs)</th><th>Estimated Length (Å)</th></tr>")

        for (u, v), p_val in sorted(res.coulson_bond_orders.items()):
            su, sv = res.pi_symbols[res.pi_atom_indices.index(u)], res.pi_symbols[res.pi_atom_indices.index(v)]
            # Estimated length via Coulson formula: R = 1.54 - 0.21 * (p / (p + 1)) approx
            r_est = 1.54 - (1.54 - 1.34) * (p_val / (p_val + 0.8 * (1.0 - p_val))) if p_val > 0 else 1.54
            lines.append(f"<tr><td><b>{su}{u+1} — {sv}{v+1}</b></td><td>{p_val:.4f}</td><td>{r_est:.3f} Å</td></tr>")
        lines.append("</table><br>")

        lines.append("<h3>π-Electron Densities & Net Charges:</h3>")
        lines.append("<table cellpadding='4' border='1' style='border-collapse: collapse; border-color: #334155; width: 100%;'>")
        lines.append("<tr style='background-color: #0f172a; color: #38bdf8;'><th>Atom</th><th>π-Density (q_r)</th><th>Net π-Charge (ζ_r)</th><th>Free Valence (F_r)</th></tr>")

        for loc, orig in enumerate(res.pi_atom_indices):
            sym = res.pi_symbols[loc]
            q_val = res.pi_electron_densities[loc]
            z_val = res.pi_charges[loc]
            f_val = res.free_valence_indices[loc]
            col = "#f43f5e" if z_val > 0.05 else ("#38bdf8" if z_val < -0.05 else "#cbd5e1")
            lines.append(
                f"<tr><td><b>{sym}{orig+1}</b></td><td>{q_val:.4f}</td>"
                f"<td><span style='color: {col}; font-weight: bold;'>{z_val:+.4f}</span></td>"
                f"<td>{f_val:.4f}</td></tr>"
            )
        lines.append("</table>")
        self.txt_coulson.setHtml("".join(lines))

    def _render_uv_tab(self, res: HuckelResult):
        r, g, b = res.predicted_color_rgb
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        uv_str = f"{res.uv_vis_transition_nm:.1f} nm" if res.uv_vis_transition_nm else "N/A (Forbidden / Degenerate)"

        html = f"""
        <h3>Electronic π → π* Transition Predictor:</h3>
        <ul>
          <li><b>HOMO–LUMO Energy Gap:</b> ΔE = <b>{res.homo_lumo_gap_beta:.4f} |β|</b></li>
          <li><b>Transition Energy (|β| = {self.spin_beta.value():.2f} eV):</b> ΔE = <b>{res.homo_lumo_gap_beta * self.spin_beta.value():.2f} eV</b></li>
          <li><b>Absorption Wavelength (λ_max):</b> <b style="color: #38bdf8; font-size: 14px;">{uv_str}</b></li>
        </ul>
        <div style="margin-top: 15px; padding: 12px; background-color: {hex_col}; border-radius: 6px; color: #ffffff; font-weight: bold; text-align: center;">
          Approximate Visual Color Swatch (Transmitted / Reflected Light)
        </div>
        <p style="color: #94a3b8; font-size: 11px; margin-top: 10px;">
        Pedagogical note: As polyene conjugation length increases (Ethylene → Butadiene → Hexatriene → Carotene),
        the HOMO-LUMO gap ΔE decreases proportional to 1/N, shifting λ_max from the ultraviolet into the visible spectrum!
        </p>
        """
        self.txt_uv.setHtml(html)

    def _project_specific_mo(self, row: int):
        if not self.current_res or row >= len(self.current_res.orbitals):
            return
        mo = self.current_res.orbitals[row]
        n_tot = len(self.current_symbols)
        weights = np.zeros(n_tot, dtype=np.float64)
        for loc, orig in enumerate(self.current_res.pi_atom_indices):
            if orig < n_tot:
                weights[orig] = mo.coefficients[loc]

        self.project_orbital_to_viewer.emit(self.current_symbols, weights)
        QtWidgets.QMessageBox.information(
            self, "Projected to 3D Viewport",
            f"Orbital ψ_{mo.index} ({mo.energy_relative_beta}, {mo.nodal_planes} nodes) projected to 3D atoms as signed atomic charges/lobes."
        )

    def _on_project_mo(self):
        selected = self.tbl_mo.selectedIndexes()
        row = selected[0].row() if selected else 0
        self._project_specific_mo(row)

    def _on_project_charges(self):
        if not self.current_res:
            return
        n_tot = len(self.current_symbols)
        charges = np.zeros(n_tot, dtype=np.float64)
        for loc, orig in enumerate(self.current_res.pi_atom_indices):
            if orig < n_tot:
                charges[orig] = self.current_res.pi_charges[loc]

        self.project_charges_to_viewer.emit(self.current_symbols, charges)
        QtWidgets.QMessageBox.information(
            self, "Net Charges Projected",
            "Net π-electron charges (ζ_r) projected onto 3D atoms in the main viewport."
        )

    def _copy_summary(self):
        if not self.current_res:
            return
        clip = QtWidgets.QApplication.clipboard()
        summary = (
            f"HMO Calculation: {self.current_res.system_name}\n"
            f"Status: {self.current_res.aromaticity_status}\n"
            f"Total Pi Energy: {self.current_res.total_pi_energy_beta:.3f} beta\n"
            f"Delocalization Energy: {self.current_res.delocalization_energy_beta:+.3f} beta\n"
            f"HOMO-LUMO Gap: {self.current_res.homo_lumo_gap_beta:.4f} beta\n"
            f"Predicted UV-Vis: {self.current_res.uv_vis_transition_nm or 'N/A'} nm\n"
        )
        clip.setText(summary)
        QtWidgets.QMessageBox.information(self, "Copied", "HMO Summary copied to clipboard.")
