"""Steric & Coordination Descriptors Dialog (%V_bur, Sterimol, Tolman Cone Angle, tau4/tau5)."""

import numpy as np
from typing import List, Optional
from PyQt6 import QtWidgets

from .coordination import (
    compute_buried_volume,
    compute_sterimol_and_cone_angle,
    compute_coordination_geometry_indices,
)


class StericDescriptorsDialog(QtWidgets.QDialog):
    """Catalysis & Organometallic Steric Descriptors Inspector (SambVca / SEQCROW style)."""

    def __init__(self, parent=None, symbols: Optional[List[str]] = None, coords: Optional[np.ndarray] = None, center_idx: int = 0, attached_idx: int = 1):
        super().__init__(parent)
        self.setWindowTitle("🛡️ Catalysis Steric & Coordination Descriptors (%V_bur, Sterimol, τ₄/τ₅)")
        self.resize(620, 480)

        self.symbols = symbols or []
        self.coords = coords if coords is not None else np.empty((0, 3))
        self.center_idx = center_idx
        self.attached_idx = attached_idx

        self._init_ui()
        self._recalc()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        ctrl = QtWidgets.QGridLayout()
        ctrl.addWidget(QtWidgets.QLabel("Central / Metal Atom:"), 0, 0)
        self.combo_center = QtWidgets.QComboBox()
        for i, s in enumerate(self.symbols):
            self.combo_center.addItem(f"#{i + 1}: {s}", i)
        if 0 <= self.center_idx < len(self.symbols):
            self.combo_center.setCurrentIndex(self.center_idx)
        self.combo_center.currentIndexChanged.connect(self._recalc)
        ctrl.addWidget(self.combo_center, 0, 1)

        ctrl.addWidget(QtWidgets.QLabel("Attached / Donor Atom (for Sterimol):"), 0, 2)
        self.combo_att = QtWidgets.QComboBox()
        for i, s in enumerate(self.symbols):
            self.combo_att.addItem(f"#{i + 1}: {s}", i)
        if 0 <= self.attached_idx < len(self.symbols):
            self.combo_att.setCurrentIndex(self.attached_idx)
        self.combo_att.currentIndexChanged.connect(self._recalc)
        ctrl.addWidget(self.combo_att, 0, 3)

        ctrl.addWidget(QtWidgets.QLabel("Sphere Radius R (Å):"), 1, 0)
        self.spin_r = QtWidgets.QDoubleSpinBox()
        self.spin_r.setRange(1.5, 8.0)
        self.spin_r.setDecimals(2)
        self.spin_r.setSingleStep(0.25)
        self.spin_r.setValue(3.50)
        self.spin_r.valueChanged.connect(self._recalc)
        ctrl.addWidget(self.spin_r, 1, 1)

        self.chk_h = QtWidgets.QCheckBox("Include Hydrogens in %V_bur")
        self.chk_h.setChecked(True)
        self.chk_h.toggled.connect(self._recalc)
        ctrl.addWidget(self.chk_h, 1, 2, 1, 2)
        layout.addLayout(ctrl)

        self.txt_out = QtWidgets.QTextEdit()
        self.txt_out.setReadOnly(True)
        self.txt_out.setStyleSheet("font-family: monospace; font-size: 12px;")
        layout.addWidget(self.txt_out, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        btn_recalc = QtWidgets.QPushButton("🔄 Recalculate")
        btn_recalc.clicked.connect(self._recalc)
        btn_row.addWidget(btn_recalc)

        btn_copy = QtWidgets.QPushButton("📋 Copy Results")
        btn_copy.clicked.connect(self._copy_results)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _recalc(self):
        c_i = self.combo_center.currentData()
        a_i = self.combo_att.currentData()
        if c_i is None or not (0 <= c_i < len(self.symbols)):
            return
        radius = self.spin_r.value()
        inc_h = self.chk_h.isChecked()

        # 1. Buried volume
        res_vbur = compute_buried_volume(
            self.symbols, self.coords, center_idx=c_i, sphere_radius=radius, include_hydrogens=inc_h
        )
        v_bur_pct = float(res_vbur.get("v_bur_pct", 0.0))
        v_free_pct = float(res_vbur.get("v_free_pct", 100.0))
        v_total_sphere = (4.0 / 3.0) * np.pi * (radius ** 3)
        v_buried = v_total_sphere * (v_bur_pct / 100.0)
        quads = res_vbur.get("quadrants", {})
        q_vals = [float(v) for v in quads.values()] if quads else [0.0]
        q_inequity = max(q_vals) - min(q_vals) if q_vals else 0.0

        # 2. Sterimol & Tolman cone angle
        res_sterimol = None
        if a_i is not None and a_i != c_i and 0 <= a_i < len(self.symbols):
            sub_idxs = [
                i for i in range(len(self.symbols))
                if i != c_i and (inc_h or self.symbols[i] != "H")
            ]
            res_sterimol = compute_sterimol_and_cone_angle(
                self.symbols, self.coords, primary_idx=c_i, attached_idx=a_i, substituent_indices=sub_idxs
            )

        # 3. Coordination geometry indices (tau4, tau4', tau5)
        res_coord = compute_coordination_geometry_indices(self.symbols, self.coords, center_idx=c_i)

        lines = [
            f"=== STERIC & COORDINATION DESCRIPTORS (Center: #{c_i+1} {self.symbols[c_i]}) ===",
            f"Sphere Radius R: {radius:.2f} Å  |  Include Hydrogens: {inc_h}",
            "",
            "--- 1. PERCENT BURIED VOLUME (%V_bur / SambVca style) ---",
            f"  Total Buried Volume (%V_bur): {v_bur_pct:6.2f} %",
            f"  Free Volume (%V_free):        {v_free_pct:6.2f} %",
            f"  Sphere Volume:                {v_total_sphere:6.2f} Å³",
            f"  Total Buried Volume (V_bur):  {v_buried:6.2f} Å³",
            "",
            "  Steric Quadrant Breakdown (Quadrant Steric Map):",
            f"    NW Quadrant (Q1):           {float(quads.get('NW', 0.0)):6.2f} %",
            f"    NE Quadrant (Q2):           {float(quads.get('NE', 0.0)):6.2f} %",
            f"    SW Quadrant (Q3):           {float(quads.get('SW', 0.0)):6.2f} %",
            f"    SE Quadrant (Q4):           {float(quads.get('SE', 0.0)):6.2f} %",
            f"    Steric Map Inequity ΔQ:     {q_inequity:6.2f} %",
            "",
        ]

        if res_sterimol:
            lines.extend([
                f"--- 2. VERLOOP STERIMOL PARAMETERS & TOLMAN CONE ANGLE (Axis #{c_i+1}➔#{a_i+1}) ---",
                f"  L  (Primary Axis Length):     {float(res_sterimol.get('L', 0.0)):6.2f} Å",
                f"  B1 (Minimal Perpendicular):   {float(res_sterimol.get('B1', 0.0)):6.2f} Å",
                f"  B5 (Maximal Perpendicular):   {float(res_sterimol.get('B5', 0.0)):6.2f} Å",
                f"  Tolman Cone Angle (θ):        {float(res_sterimol.get('cone_angle_deg', 0.0)):6.2f} °",
                "",
            ])

        if res_coord and res_coord.get("coord_number", 0) >= 2:
            cn = res_coord["coord_number"]
            lines.append(f"--- 3. COORDINATION GEOMETRY INDICES (CN = {cn}) ---")
            lines.append(f"  Summary:                      {res_coord.get('summary', '')}")
            if "tau_4" in res_coord:
                lines.append(f"  Yang τ₄ Parameter:            {res_coord['tau_4']:.3f}  (1.00 = Tetrahedron, 0.00 = Square Planar)")
                lines.append(f"  Okuniewski τ₄' Parameter:     {res_coord['tau_4_prime']:.3f}")
            if "tau_5" in res_coord:
                lines.append(f"  Addison τ₅ Parameter:         {res_coord['tau_5']:.3f}  (1.00 = Trigonal Bipyramid, 0.00 = Square Pyramid)")
            lines.append("")

        self.txt_out.setPlainText("\n".join(lines))

    def _copy_results(self):
        QtWidgets.QApplication.clipboard().setText(self.txt_out.toPlainText())
