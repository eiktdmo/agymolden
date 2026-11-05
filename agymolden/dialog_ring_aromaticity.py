"""Ring Aromaticity (HOMA Index) & Planarity Inspector Dialog (Multiwfn / Krygowski style)."""

import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from PyQt6 import QtCore, QtGui, QtWidgets

from .aromatic import compute_ring_homa_and_planarity


class RingAromaticityDialog(QtWidgets.QDialog):
    """Kruszewski-Krygowski HOMA Aromaticity Index & Ring Planarity Inspector."""

    def __init__(
        self,
        parent=None,
        symbols: Optional[List[str]] = None,
        coords: Optional[np.ndarray] = None,
        bonds: Optional[List[Tuple[int, int]]] = None,
        bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
        on_ring_highlight: Optional[Callable[[List[int]], None]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("💍 Ring Aromaticity (HOMA Index) & Planarity Inspector")
        self.resize(720, 430)

        self.symbols = symbols or []
        self.coords = coords if coords is not None else np.empty((0, 3))
        self.bonds = bonds or []
        self.bond_orders = bond_orders or {}
        self.on_ring_highlight = on_ring_highlight

        self.rings_info = compute_ring_homa_and_planarity(
            self.symbols, self.coords, self.bonds, self.bond_orders
        )
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        lbl_head = QtWidgets.QLabel(
            "<b>Kruszewski–Krygowski HOMA Aromaticity Index & Ring Planarity</b><br>"
            "<span style='color:#94a3b8; font-size:11px;'>"
            "HOMA = 1 − (1/n) Σ α(R_opt − R_i)²  (1.00 = Ideal Benzene, ~0.00 = Non-Aromatic, &lt;0.00 = Anti-Aromatic). "
            "Click any ring row to highlight its atoms in the 3D viewport."
            "</span>"
        )
        lbl_head.setWordWrap(True)
        layout.addWidget(lbl_head)

        self.tbl = QtWidgets.QTableWidget(len(self.rings_info), 7)
        self.tbl.setHorizontalHeaderLabels([
            "Ring #", "Size", "Ring Atoms", "HOMA Index", "Mean R (Å)", "Planarity RMSD (Å)", "Classification"
        ])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tbl.setAlternatingRowColors(True)

        for r_i, info in enumerate(self.rings_info):
            self.tbl.setItem(r_i, 0, QtWidgets.QTableWidgetItem(str(info["ring_index"])))
            self.tbl.setItem(r_i, 1, QtWidgets.QTableWidgetItem(f"{info['size']}-ring"))
            self.tbl.setItem(r_i, 2, QtWidgets.QTableWidgetItem(str(info["label"])))
            h_str = f"{info['homa']:.4f}" if info["homa"] is not None else "N/A"
            it_h = QtWidgets.QTableWidgetItem(h_str)
            if info["homa"] is not None and info["homa"] >= 0.70:
                it_h.setForeground(QtGui.QBrush(QtGui.QColor("#4ade80")))
            self.tbl.setItem(r_i, 3, it_h)
            self.tbl.setItem(r_i, 4, QtWidgets.QTableWidgetItem(f"{info['mean_bond_ang']:.4f}"))
            self.tbl.setItem(r_i, 5, QtWidgets.QTableWidgetItem(f"{info['planarity_rmsd_ang']:.4f}"))
            self.tbl.setItem(r_i, 6, QtWidgets.QTableWidgetItem(str(info["status"])))

        self.tbl.itemSelectionChanged.connect(self._on_row_clicked)
        layout.addWidget(self.tbl)

        btn_row = QtWidgets.QHBoxLayout()
        btn_copy = QtWidgets.QPushButton("📋 Copy Table to Clipboard")
        btn_copy.clicked.connect(self._copy_table)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _on_row_clicked(self):
        row = self.tbl.currentRow()
        if 0 <= row < len(self.rings_info):
            ring_atoms = self.rings_info[row]["atom_indices"]
            if self.on_ring_highlight:
                self.on_ring_highlight(ring_atoms)

    def _copy_table(self):
        lines = ["Ring #\tSize\tRing Atoms\tHOMA Index\tMean R (Å)\tPlanarity RMSD (Å)\tClassification"]
        for info in self.rings_info:
            h_str = f"{info['homa']:.4f}" if info["homa"] is not None else "N/A"
            lines.append(
                f"{info['ring_index']}\t{info['size']}-ring\t{info['label']}\t{h_str}\t"
                f"{info['mean_bond_ang']:.4f}\t{info['planarity_rmsd_ang']:.4f}\t{info['status']}"
            )
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
