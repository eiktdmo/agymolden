"""Interactive Example Molecule & Calculation Gallery Dialog for AgyMolden."""

import os
from typing import List, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets
from .resources import find_resource


EXAMPLES: List[Tuple[str, str, str, str]] = [
    (
        "Water (H₂O) — Orbitals & ESP",
        "examples/h2o_orca.molden",
        "Molden Wavefunction",
        "Water molecule computed with ORCA (HF/cc-pVDZ). Explore HOMO/LUMO molecular orbitals, electron density, and 3D electrostatic potential surfaces.",
    ),
    (
        "Benzene (C₆H₆) — π-System",
        "examples/C6H6_OPT_HF_cc-pvdz_hf_electrons.molden",
        "Molden Wavefunction",
        "Aromatic benzene ring with full canonical molecular orbital expansion, nodal planes, and delocalized π orbitals.",
    ),
    (
        "Ethanol (C₂H₆O) — Conformation & MOs",
        "examples/C2H6O_OPT_HF_cc-pvdz_hf_electrons.molden",
        "Molden Wavefunction",
        "Ethanol molecule with orbital eigenvalues, electronic states, and molecular electrostatic potential envelope.",
    ),
    (
        "Water Frequency & Vibrations (Gaussian 16)",
        "examples/AgyMolden_generated_calculation.log",
        "Gaussian Log",
        "Gaussian 16 Opt+Freq calculation. Inspect animated normal vibrational modes, harmonic IR frequencies, and synthesized spectrum.",
    ),
    (
        "Ethylene Geometry Optimization (newBRORSEN)",
        "C2H4_OPT_HF_cc-pvdz.out",
        "newBRORSEN Log",
        "Complete geometry optimization trajectory computed with newBRORSEN showing iterative energy convergence and gradient minimization.",
    ),
    (
        "LiH Nuclear-Electronic Positronic State",
        "HLi_electronic_positronic.out",
        "Multi-Component QM",
        "Groundbreaking non-Born-Oppenheimer APMO positronic lithium hydride calculation featuring simultaneous electronic and positronic wavefunctions.",
    ),
    (
        "Water Reaction & Stationary Point (ORCA)",
        "orca_calc.out",
        "ORCA Log",
        "ORCA quantum chemistry calculation output with thermochemistry, Mulliken/Loewdin atomic charges, and orbital energies.",
    ),
    (
        "Toluene (C₇H₈) — Internal Coordinates",
        "examples/toluene.zmat",
        "Z-Matrix",
        "Clean internal coordinate definition of toluene demonstrating symmetric variables and dihedral angles.",
    ),
    (
        "Ethane (C₂H₆) — Staggered Conformation",
        "examples/ethane.zmat",
        "Z-Matrix",
        "Prototypical staggered alkane conformation suitable for torsional scans and internal coordinate manipulation.",
    ),
    (
        "Cyclopentadiene (C₅H₆) — Ring Structure",
        "examples/c5h6.zmat",
        "Z-Matrix",
        "Five-membered conjugated diene ring demonstrating ring closures and sp² vs sp³ carbon centers.",
    ),
]


class ExampleGalleryDialog(QtWidgets.QDialog):
    """Browse and load bundled educational and benchmark quantum chemistry examples."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Example Molecule & Calculation Gallery")
        self.resize(750, 480)
        self.parent_window = parent

        layout = QtWidgets.QVBoxLayout(self)

        # Header description
        header = QtWidgets.QLabel(
            "<h3>🧪 AgyMolden Example Gallery</h3>"
            "<p>Select any bundled molecule or quantum chemistry calculation to immediately load and explore in the 3D workbench.</p>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Table of examples
        self.table = QtWidgets.QTableWidget(len(EXAMPLES), 3)
        self.table.setHorizontalHeaderLabels(["Title", "Type", "Description"])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)

        type_colors = {
            "Molden Wavefunction": "#0284c7",
            "Gaussian Log": "#16a34a",
            "newBRORSEN Log": "#7c3aed",
            "Multi-Component QM": "#db2777",
            "ORCA Log": "#d97706",
            "Z-Matrix": "#4b5563",
        }

        for row, (title, rel_path, ex_type, desc) in enumerate(EXAMPLES):
            it_title = QtWidgets.QTableWidgetItem(f"<b>{title}</b>")
            it_type = QtWidgets.QTableWidgetItem(ex_type)
            color = type_colors.get(ex_type, "#374151")
            it_type.setForeground(QtGui.QBrush(QtGui.QColor(color)))

            it_desc = QtWidgets.QTableWidgetItem(desc)

            # Store the relative path in UserRole
            it_title.setData(QtCore.Qt.ItemDataRole.UserRole, rel_path)

            self.table.setItem(row, 0, it_title)
            self.table.setItem(row, 1, it_type)
            self.table.setItem(row, 2, it_desc)

        self.table.selectRow(0)
        layout.addWidget(self.table)

        # Bottom buttons
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_open = QtWidgets.QPushButton("🚀 Open in Workbench")
        self.btn_open.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 14px;")
        self.btn_open.clicked.connect(self._on_open_selected)
        btn_layout.addWidget(self.btn_open)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    def _get_selected_path(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        it = self.table.item(row, 0)
        rel_path = it.data(QtCore.Qt.ItemDataRole.UserRole)
        # Resolve via find_resource
        abs_path = find_resource(rel_path)
        if not abs_path or not os.path.isfile(abs_path):
            # Check relative to workspace
            candidate = os.path.abspath(rel_path)
            if os.path.isfile(candidate):
                abs_path = candidate
        return abs_path or ""

    def _on_item_double_clicked(self, _item):
        self._on_open_selected()

    def _on_open_selected(self):
        path = self._get_selected_path()
        if not path:
            QtWidgets.QMessageBox.warning(self, "File Not Found", "The selected example file could not be located.")
            return

        self.accept()
        if self.parent_window and hasattr(self.parent_window, "open_path"):
            self.parent_window.open_path(path)
