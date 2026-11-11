"""
Natural Bond Orbital (NBO) Donor-Acceptor Perturbation Visualizer & Analyzer Dialog.

Features:
- Visualizes Second-Order Perturbation Theory Fock Matrix interactions:
    Donor (i) -> Acceptor (j), E(2) (kcal/mol), ΔE (a.u.), F(i,j) (a.u.).
- Interactive chemical filtering:
    * Minimum E(2) threshold (kcal/mol)
    * Interaction type (n -> σ*, n -> π*, σ -> σ* Hyperconjugation, π -> π*)
    * Filter by selected atom(s)
- Direct 3D coordination sync: Selecting an interaction highlights the donor and acceptor atoms.
- 1-Click export to publication CSV and GitHub-flavored Markdown tables.
"""

import os
from typing import List, Optional
from PyQt6 import QtCore, QtGui, QtWidgets

from .nbo_analyzer import NBOPerturbation, parse_nbo_perturbation_table


class NBODonorAcceptorDialog(QtWidgets.QDialog):
    """Dialog for inspecting and filtering NBO donor-acceptor interactions."""

    interaction_highlighted = QtCore.pyqtSignal(list, list, float, str)

    def __init__(
        self,
        parent=None,
        log_filepath_or_text: Optional[str] = None,
        is_path: bool = True,
        selected_atoms: Optional[List[int]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("🧪 Natural Bond Orbital (NBO) Donor–Acceptor Interactions")
        self.resize(920, 600)

        self.all_interactions: List[NBOPerturbation] = []
        self.filtered_interactions: List[NBOPerturbation] = []
        self.selected_atoms: List[int] = selected_atoms or []

        self._init_ui()

        if log_filepath_or_text:
            if is_path and os.path.isfile(log_filepath_or_text):
                with open(log_filepath_or_text, "r", encoding="utf-8", errors="replace") as f:
                    txt = f.read()
                self.load_nbo_text(txt, filename=os.path.basename(log_filepath_or_text))
            elif not is_path:
                self.load_nbo_text(log_filepath_or_text)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Top Filter Toolbar
        filter_group = QtWidgets.QGroupBox("Filters & Controls")
        f_layout = QtWidgets.QHBoxLayout(filter_group)
        f_layout.setContentsMargins(8, 8, 8, 8)
        f_layout.setSpacing(10)

        f_layout.addWidget(QtWidgets.QLabel("Min E(2):"))
        self.spin_min_e = QtWidgets.QDoubleSpinBox()
        self.spin_min_e.setRange(0.1, 500.0)
        self.spin_min_e.setValue(2.0)
        self.spin_min_e.setSuffix(" kcal/mol")
        self.spin_min_e.valueChanged.connect(self._apply_filters)
        f_layout.addWidget(self.spin_min_e)

        f_layout.addWidget(QtWidgets.QLabel("Interaction Type:"))
        self.combo_type = QtWidgets.QComboBox()
        self.combo_type.addItems([
            "All Types",
            "n -> σ* (Anomeric / Non-covalent)",
            "n -> π* (Resonance / Conjugation)",
            "σ -> σ* (Hyperconjugation)",
            "π -> π* (Conjugation)",
        ])
        self.combo_type.currentTextChanged.connect(self._apply_filters)
        f_layout.addWidget(self.combo_type)

        self.chk_selected_only = QtWidgets.QCheckBox("Selected Atoms Only")
        self.chk_selected_only.toggled.connect(self._apply_filters)
        f_layout.addWidget(self.chk_selected_only)

        f_layout.addStretch()

        btn_load = QtWidgets.QPushButton("📂 Open Log...")
        btn_load.clicked.connect(self._on_browse_log)
        f_layout.addWidget(btn_load)

        layout.addWidget(filter_group)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Donor NBO (i)",
            "Acceptor NBO (j)",
            "Type",
            "E(2) (kcal/mol)",
            "E(j) - E(i) (a.u.)",
            "F(i,j) (a.u.)",
            "Atoms Involved",
        ])
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        layout.addWidget(self.table, stretch=1)

        # Status & Export bar
        bot_layout = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Ready. Open a Gaussian or ORCA log with POP=NBO.")
        self.lbl_status.setStyleSheet("font-weight: bold; color: #38bdf8;")
        bot_layout.addWidget(self.lbl_status)
        bot_layout.addStretch()

        btn_csv = QtWidgets.QPushButton("📊 Export CSV")
        btn_csv.clicked.connect(self._on_export_csv)
        bot_layout.addWidget(btn_csv)

        btn_net = QtWidgets.QPushButton("🕸️ View E(2) Network Graph...")
        btn_net.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 10px;")
        btn_net.setToolTip("Open interactive topological Chord/Bipartite network diagram of donor-acceptor interactions")
        btn_net.clicked.connect(self._on_view_network_graph)
        bot_layout.addWidget(btn_net)

        btn_md = QtWidgets.QPushButton("📝 Copy Markdown")
        btn_md.clicked.connect(self._on_copy_markdown)
        bot_layout.addWidget(btn_md)

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.close)
        bot_layout.addWidget(btn_close)

        layout.addLayout(bot_layout)

    def load_nbo_text(self, text: str, filename: str = ""):
        """Parse NBO text and populate the table."""
        self.all_interactions = parse_nbo_perturbation_table(text)
        self._apply_filters()
        fn_str = f" from {filename}" if filename else ""
        self.lbl_status.setText(f"Loaded {len(self.all_interactions)} NBO donor-acceptor interactions{fn_str}.")

    def _apply_filters(self):
        min_e = self.spin_min_e.value()
        type_filter = self.combo_type.currentText()
        sel_only = self.chk_selected_only.isChecked()

        filtered = []
        for item in self.all_interactions:
            if item.energy_kcal < min_e:
                continue
            if type_filter.startswith("n -> σ*") and item.interaction_type != "n -> σ*":
                continue
            if type_filter.startswith("n -> π*") and item.interaction_type != "n -> π*":
                continue
            if type_filter.startswith("σ -> σ*") and "Hyperconjugation" not in item.interaction_type:
                continue
            if type_filter.startswith("π -> π*") and item.interaction_type != "π -> π*":
                continue
            if sel_only and self.selected_atoms:
                all_at = set(item.donor_atoms + item.acceptor_atoms)
                if not any(a in all_at for a in self.selected_atoms):
                    continue
            filtered.append(item)

        self.filtered_interactions = filtered
        self._populate_table()

    def _populate_table(self):
        self.table.setRowCount(len(self.filtered_interactions))
        for r, item in enumerate(self.filtered_interactions):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(f"{item.donor_idx}. {item.donor_label}"))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(f"{item.acceptor_idx}. {item.acceptor_label}"))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(item.interaction_type))

            item_e = QtWidgets.QTableWidgetItem(f"{item.energy_kcal:.2f}")
            item_e.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            if item.energy_kcal >= 10.0:
                item_e.setForeground(QtGui.QColor("#22c55e"))
                item_e.setFont(QtGui.QFont("", -1, QtGui.QFont.Weight.Bold))
            elif item.energy_kcal >= 5.0:
                item_e.setForeground(QtGui.QColor("#f59e0b"))
            self.table.setItem(r, 3, item_e)

            item_diff = QtWidgets.QTableWidgetItem(f"{item.energy_diff_au:.2f}")
            item_diff.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 4, item_diff)

            item_fock = QtWidgets.QTableWidgetItem(f"{item.fock_element_au:.3f}")
            item_fock.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
            self.table.setItem(r, 5, item_fock)

            atoms_desc = f"Donor: {list(map(lambda x: x+1, item.donor_atoms))} → Acceptor: {list(map(lambda x: x+1, item.acceptor_atoms))}"
            self.table.setItem(r, 6, QtWidgets.QTableWidgetItem(atoms_desc))

        self.lbl_status.setText(f"Showing {len(self.filtered_interactions)} of {len(self.all_interactions)} interactions.")

    def _on_row_selected(self):
        indexes = self.table.selectedIndexes()
        if not indexes:
            return
        row = indexes[0].row()
        if 0 <= row < len(self.filtered_interactions):
            item = self.filtered_interactions[row]
            label = f"{item.donor_label} → {item.acceptor_label} (E(2) = {item.energy_kcal:.1f} kcal/mol)"
            self.interaction_highlighted.emit(
                item.donor_atoms,
                item.acceptor_atoms,
                item.energy_kcal,
                label,
            )

    def _on_browse_log(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Output Log with NBO Analysis",
            "",
            "QM Logs (*.log *.out *.txt);;All Files (*)",
        )
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                txt = f.read()
            self.load_nbo_text(txt, filename=os.path.basename(path))

    def get_csv_string(self) -> str:
        lines = ["Donor_NBO,Acceptor_NBO,Type,E2_kcal_mol,Delta_E_au,Fock_au,Donor_Atoms,Acceptor_Atoms"]
        for i in self.filtered_interactions:
            d_at = ";".join(str(a + 1) for a in i.donor_atoms)
            a_at = ";".join(str(a + 1) for a in i.acceptor_atoms)
            lines.append(
                f'"{i.donor_label}","{i.acceptor_label}","{i.interaction_type}",{i.energy_kcal:.2f},{i.energy_diff_au:.2f},{i.fock_element_au:.3f},"{d_at}","{a_at}"'
            )
        return "\n".join(lines) + "\n"

    def get_markdown_string(self) -> str:
        lines = [
            "| Donor NBO (i) | Acceptor NBO (j) | Type | E(2) (kcal/mol) | ΔE (a.u.) | F(i,j) (a.u.) |",
            "| :--- | :--- | :--- | :---: | :---: | :---: |",
        ]
        for i in self.filtered_interactions:
            lines.append(
                f"| {i.donor_label} | {i.acceptor_label} | {i.interaction_type} | **{i.energy_kcal:.2f}** | {i.energy_diff_au:.2f} | {i.fock_element_au:.3f} |"
            )
        return "\n".join(lines) + "\n"

    def _on_export_csv(self):
        if not self.filtered_interactions:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export NBO Interactions to CSV", "nbo_interactions.csv", "CSV Files (*.csv);;All Files (*)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.get_csv_string())
            QtWidgets.QMessageBox.information(self, "Exported", f"Exported NBO interactions to:\n{path}")

    def _on_copy_markdown(self):
        if not self.filtered_interactions:
            return
        QtWidgets.QApplication.clipboard().setText(self.get_markdown_string())
        QtWidgets.QMessageBox.information(self, "Copied", "NBO Markdown table copied to clipboard for publication SI!")

    def _on_view_network_graph(self):
        """Open interactive NBO E(2) network graph dialog."""
        from .dialog_nbo_network import NBONetworkDialog
        interactions = self.filtered_interactions if self.filtered_interactions else self.all_interactions
        dlg = NBONetworkDialog(parent=self, interactions=interactions)
        dlg.exec()

