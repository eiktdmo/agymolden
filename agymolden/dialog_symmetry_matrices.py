"""Symmetry operation matrices dialog for AgyMolden."""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtWidgets


class SymmetryMatricesDialog(QtWidgets.QDialog):
    """Dialog displaying 3x3 Cartesian symmetry operation matrices with interactive selector."""

    def __init__(self, parent=None, sym_res=None, initial_row: int = 0, bg_color_mode: str = "dark"):
        super().__init__(parent)
        self.sym_res = sym_res
        self.setWindowTitle(f"3×3 Cartesian Symmetry Matrices — Point Group {sym_res.symbol}")
        self.resize(520, 460)

        is_white = bg_color_mode == "white"
        self.tbl_bg = "#ffffff" if is_white else "#0f172a"
        self.tbl_fg = "#0f172a" if is_white else "#f8fafc"
        self.border_col = "#cbd5e1" if is_white else "#334155"

        d_layout = QtWidgets.QVBoxLayout(self)
        d_layout.setSpacing(10)

        # Header info
        hdr = QtWidgets.QLabel(
            f"<b>Point Group:</b> <span style='color:#7c3aed; font-size:14px;'>{sym_res.symbol}</span> "
            f"&nbsp;|&nbsp; <b>Order (h):</b> {sym_res.order} operations"
        )
        d_layout.addWidget(hdr)

        # Combo selector for operations + Show All option
        sel_row = QtWidgets.QHBoxLayout()
        sel_row.addWidget(QtWidgets.QLabel("Select Operation:"))
        self.combo_ops = QtWidgets.QComboBox()
        for idx, op in enumerate(sym_res.operations):
            details = f" along [{op.axis[0]:.2f}, {op.axis[1]:.2f}, {op.axis[2]:.2f}]" if op.axis is not None else ""
            self.combo_ops.addItem(f"#{idx+1}: {op.label} ({op.op_type}{details})", idx)
        self.combo_ops.addItem("📑 Display All Operations Side-by-Side", -1)

        if 0 <= initial_row < len(sym_res.operations):
            self.combo_ops.setCurrentIndex(initial_row)
        else:
            self.combo_ops.setCurrentIndex(0)

        sel_row.addWidget(self.combo_ops, stretch=1)
        d_layout.addLayout(sel_row)

        # Scroll area containing rendered matrix / matrices
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        content_w = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout(content_w)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(12)
        scroll.setWidget(content_w)
        d_layout.addWidget(scroll, stretch=1)

        self.combo_ops.currentIndexChanged.connect(self._update_view)
        self._update_view()

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.close)
        d_layout.addWidget(btn_box)

    def _format_matrix_html(self, idx: int, op) -> str:
        M = op.matrix
        tr = float(np.trace(M))
        det = float(np.linalg.det(M))
        details = f" along [{op.axis[0]:.2f}, {op.axis[1]:.2f}, {op.axis[2]:.2f}]" if op.axis is not None else ""
        h = [
            f"<div style='margin-bottom:12px; padding:8px; border:1px solid {self.border_col}; border-radius:6px; background:{self.tbl_bg};'>",
            f"<div style='font-size:13px; font-weight:bold; color:#0284c7; margin-bottom:6px;'>"
            f"Operation #{idx+1}: {op.label} <span style='color:#64748b; font-size:11px;'>({op.op_type}{details})</span></div>",
            f"<table border='1' cellpadding='5' cellspacing='0' style='border-collapse:collapse; font-family:monospace; background:{self.tbl_bg}; color:{self.tbl_fg}; font-size:12px; margin-bottom:6px;'>",
        ]
        for r in range(3):
            h.append("<tr>")
            for c in range(3):
                val = M[r, c]
                if abs(val) < 1e-7:
                    val_str = "0.000000"
                elif abs(val - 1.0) < 1e-7:
                    val_str = "1.000000"
                elif abs(val + 1.0) < 1e-7:
                    val_str = "-1.000000"
                else:
                    val_str = f"{val:10.6f}"
                h.append(f"<td align='right' style='padding:6px 12px;'>{val_str}</td>")
            h.append("</tr>")
        h.append("</table>")
        h.append(f"<div style='color:#94a3b8; font-size:11px;'>Trace = {tr:+.3f} &nbsp;|&nbsp; Determinant = {det:+.3f}</div>")
        h.append("</div>")
        return "".join(h)

    def _update_view(self):
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        sel_idx = self.combo_ops.currentData()
        if sel_idx == -1:
            html_chunks = [self._format_matrix_html(i, op) for i, op in enumerate(self.sym_res.operations)]
            lbl = QtWidgets.QLabel("".join(html_chunks))
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            lbl.setWordWrap(True)
            self.content_layout.addWidget(lbl)
        else:
            op = self.sym_res.operations[sel_idx]
            lbl = QtWidgets.QLabel(self._format_matrix_html(sel_idx, op))
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            self.content_layout.addWidget(lbl)
        self.content_layout.addStretch()
