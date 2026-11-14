"""Self-contained Interactive Lab Exercise (.agylab) dialog for classroom use."""

import json
import os
from typing import Optional
from PyQt6 import QtCore, QtWidgets


DEFAULT_LAB = {
    "title": "Lab 1: Geometry Optimization & IR of Water",
    "instructions": (
        "1. Load the starting H2O geometry into the 3D viewer.\n"
        "2. Run ORCA with '! B3LYP def2-SVP Opt Freq' (Charge 0, Mult 1).\n"
        "3. Inspect the optimized bond length, angle, and IR frequencies."
    ),
    "route": "! B3LYP def2-SVP Opt Freq",
    "charge": 0,
    "multiplicity": 1,
    "xyz": "3\nWater starting geometry\nO  0.000  0.000  0.000\nH  0.960  0.000  0.000\nH -0.240  0.930  0.000\n",
}


class LabExerciseDialog(QtWidgets.QDialog):
    """Interactive classroom lab exercise viewer/runner (.agylab JSON)."""

    def __init__(self, parent=None, lab_data: Optional[dict] = None):
        super().__init__(parent)
        self.setWindowTitle("🎓 Interactive Lab Exercise (.agylab)")
        self.resize(560, 440)
        self.lab_data = dict(lab_data or DEFAULT_LAB)

        layout = QtWidgets.QVBoxLayout(self)

        self.lbl_title = QtWidgets.QLabel(f"<h3>{self.lab_data.get('title', 'Lab Exercise')}</h3>")
        layout.addWidget(self.lbl_title)

        self.txt_instructions = QtWidgets.QPlainTextEdit()
        self.txt_instructions.setReadOnly(True)
        self.txt_instructions.setPlainText(self.lab_data.get("instructions", ""))
        self.txt_instructions.setMaximumHeight(110)
        layout.addWidget(self.txt_instructions)

        self.lbl_preset = QtWidgets.QLabel(
            f"<b>ORCA Preset:</b> <code>{self.lab_data.get('route', '! B3LYP def2-SVP Opt')}</code> "
            f"| <b>Charge:</b> {self.lab_data.get('charge', 0)} "
            f"| <b>Mult:</b> {self.lab_data.get('multiplicity', 1)}"
        )
        layout.addWidget(self.lbl_preset)

        grp_check = QtWidgets.QGroupBox("Student Lab Checklist")
        chk_layout = QtWidgets.QVBoxLayout(grp_check)
        self.chk_step1 = QtWidgets.QCheckBox("1. Starting geometry loaded into 3D viewer")
        self.chk_step2 = QtWidgets.QCheckBox("2. ORCA calculation launched & converged")
        self.chk_step3 = QtWidgets.QCheckBox("3. Results inspected (Geometry / Orbitals / IR)")
        chk_layout.addWidget(self.chk_step1)
        chk_layout.addWidget(self.chk_step2)
        chk_layout.addWidget(self.chk_step3)
        layout.addWidget(grp_check)

        btn_row = QtWidgets.QHBoxLayout()
        btn_open = QtWidgets.QPushButton("📂 Open .agylab...")
        btn_open.clicked.connect(self._on_open_lab)
        btn_row.addWidget(btn_open)

        btn_load_geom = QtWidgets.QPushButton("🧪 Load Geometry")
        btn_load_geom.clicked.connect(self._on_load_geometry)
        btn_row.addWidget(btn_load_geom)

        btn_run_orca = QtWidgets.QPushButton("🚀 Run ORCA")
        btn_run_orca.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 4px 10px;")
        btn_run_orca.clicked.connect(self._on_run_orca)
        btn_row.addWidget(btn_run_orca)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _on_open_lab(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Lab Exercise", "", "AgyLab (*.agylab *.json);;All Files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.lab_data = data
            self.lbl_title.setText(f"<h3>{data.get('title', os.path.basename(path))}</h3>")
            self.txt_instructions.setPlainText(data.get("instructions", ""))
            self.lbl_preset.setText(
                f"<b>ORCA Preset:</b> <code>{data.get('route', '! B3LYP def2-SVP Opt')}</code> "
                f"| <b>Charge:</b> {data.get('charge', 0)} "
                f"| <b>Mult:</b> {data.get('multiplicity', 1)}"
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Error", f"Could not load lab file:\n{exc}")

    def _on_load_geometry(self):
        xyz = self.lab_data.get("xyz", "")
        parent = self.parent()
        if xyz and parent and hasattr(parent, "_load_xyz_path"):
            import tempfile
            with tempfile.NamedTemporaryFile("w", suffix=".xyz", delete=False, encoding="utf-8") as tmp:
                tmp.write(xyz)
                tmp_path = tmp.name
            try:
                parent._load_xyz_path(tmp_path)
                self.chk_step1.setChecked(True)
            finally:
                if os.path.isfile(tmp_path):
                    os.remove(tmp_path)

    def _on_run_orca(self):
        parent = self.parent()
        if parent and hasattr(parent, "_on_quick_run_orca"):
            self.chk_step2.setChecked(True)
            parent._on_quick_run_orca()
