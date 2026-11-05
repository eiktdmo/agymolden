"""
Interactive Pre-Flight Validation Dialog for Quantum Chemistry calculations.
Displays issues grouped by severity (Error, Warning, Info) with clear remedies.
"""

from PyQt6 import QtWidgets, QtCore, QtGui
from typing import Optional

from .preflight import PreflightReport, ValidationIssue


class PreflightValidationDialog(QtWidgets.QDialog):
    """
    Shows calculation pre-flight verification summary.
    If fatal errors exist, blocks calculation launch and guides user to fix.
    If only warnings exist, allows user to review and proceed or cancel.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget], report: PreflightReport, qm_name: str):
        super().__init__(parent)
        self.report = report
        self.qm_name = qm_name
        self.setWindowTitle(f"Pre-Flight Inspection — {qm_name}")
        self.resize(620, 440)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Header Banner
        header = QtWidgets.QFrame()
        if self.report.has_errors:
            header.setStyleSheet("background-color: #fef2f2; border: 1px solid #f87171; border-radius: 6px; padding: 10px;")
            icon_txt = "⛔"
            title_txt = f"<b style='color: #991b1b; font-size: 14px;'>Pre-Flight Checks Failed ({self.report.error_count} Error{'s' if self.report.error_count != 1 else ''})</b>"
            sub_txt = "<span style='color: #7f1d1d;'>Critical issues must be resolved before this calculation can be launched.</span>"
        elif self.report.warning_count > 0:
            header.setStyleSheet("background-color: #fffbeb; border: 1px solid #fcd34d; border-radius: 6px; padding: 10px;")
            icon_txt = "⚠️"
            title_txt = f"<b style='color: #92400e; font-size: 14px;'>Pre-Flight Warnings ({self.report.warning_count} Warning{'s' if self.report.warning_count != 1 else ''})</b>"
            sub_txt = "<span style='color: #78350f;'>Potential issues detected. Please review before proceeding.</span>"
        else:
            header.setStyleSheet("background-color: #f0fdf4; border: 1px solid #86efac; border-radius: 6px; padding: 10px;")
            icon_txt = "✅"
            title_txt = "<b style='color: #166534; font-size: 14px;'>All Pre-Flight Checks Passed</b>"
            sub_txt = "<span style='color: #14532d;'>Calculation parameters, spin parity, and hardware requirements are validated.</span>"

        h_layout = QtWidgets.QHBoxLayout(header)
        lbl_icon = QtWidgets.QLabel(f"<span style='font-size: 24px;'>{icon_txt}</span>")
        h_layout.addWidget(lbl_icon)

        text_layout = QtWidgets.QVBoxLayout()
        text_layout.addWidget(QtWidgets.QLabel(title_txt))
        text_layout.addWidget(QtWidgets.QLabel(sub_txt))
        h_layout.addLayout(text_layout)
        h_layout.addStretch()
        layout.addWidget(header)

        # Electronic Overview Pill Box
        info_bar = QtWidgets.QHBoxLayout()
        spin_label = "Singlet" if self.report.num_alpha == self.report.num_beta else (
            "Doublet" if abs(self.report.num_alpha - self.report.num_beta) == 1 else (
                "Triplet" if abs(self.report.num_alpha - self.report.num_beta) == 2 else "Open-Shell"
            )
        )
        pill = QtWidgets.QLabel(
            f"<b>Electrons:</b> {self.report.total_electrons} "
            f"({self.report.num_alpha} α, {self.report.num_beta} β)  |  "
            f"<b>Spin State:</b> {spin_label} (2S+1 = {self.report.num_alpha - self.report.num_beta + 1})"
        )
        pill.setStyleSheet("background: #f1f5f9; border-radius: 4px; padding: 6px 12px; font-family: monospace; font-size: 11px;")
        info_bar.addWidget(pill)
        layout.addLayout(info_bar)

        # Issues Scroll List
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        c_layout = QtWidgets.QVBoxLayout(container)
        c_layout.setSpacing(8)

        for issue in self.report.issues:
            card = QtWidgets.QFrame()
            if issue.severity == "error":
                card.setStyleSheet("background: #fff5f5; border-left: 4px solid #ef4444; border-top: 1px solid #fee2e2; border-right: 1px solid #fee2e2; border-bottom: 1px solid #fee2e2; border-radius: 4px; padding: 8px;")
                badge = "<span style='color: #dc2626; font-weight: bold;'>[ERROR]</span>"
            elif issue.severity == "warning":
                card.setStyleSheet("background: #fffdf5; border-left: 4px solid #f59e0b; border-top: 1px solid #fef3c7; border-right: 1px solid #fef3c7; border-bottom: 1px solid #fef3c7; border-radius: 4px; padding: 8px;")
                badge = "<span style='color: #d97706; font-weight: bold;'>[WARNING]</span>"
            else:
                card.setStyleSheet("background: #f8fafc; border-left: 4px solid #3b82f6; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0; border-radius: 4px; padding: 8px;")
                badge = "<span style='color: #2563eb; font-weight: bold;'>[INFO]</span>"

            card_layout = QtWidgets.QVBoxLayout(card)
            card_layout.setSpacing(4)
            card_layout.addWidget(QtWidgets.QLabel(f"{badge} <b>{issue.title}</b>"))
            lbl_msg = QtWidgets.QLabel(issue.message)
            lbl_msg.setWordWrap(True)
            lbl_msg.setStyleSheet("color: #334155; font-size: 12px;")
            card_layout.addWidget(lbl_msg)
            if issue.remedy:
                lbl_rem = QtWidgets.QLabel(f"💡 <i>Remedy:</i> {issue.remedy}")
                lbl_rem.setWordWrap(True)
                lbl_rem.setStyleSheet("color: #0f766e; font-size: 11px;")
                card_layout.addWidget(lbl_rem)

            c_layout.addWidget(card)

        c_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Action Buttons
        btn_bar = QtWidgets.QHBoxLayout()
        btn_bar.addStretch()

        if self.report.has_errors:
            self.btn_action = QtWidgets.QPushButton("Fix Calculation Setup")
            self.btn_action.setStyleSheet("background-color: #dc2626; color: white; font-weight: bold; padding: 6px 16px;")
            self.btn_action.clicked.connect(self.reject)
            btn_bar.addWidget(self.btn_action)
        else:
            btn_cancel = QtWidgets.QPushButton("Cancel")
            btn_cancel.clicked.connect(self.reject)
            btn_bar.addWidget(btn_cancel)

            self.btn_proceed = QtWidgets.QPushButton("Proceed to Launch ▶")
            self.btn_proceed.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 6px 16px;")
            self.btn_proceed.clicked.connect(self.accept)
            btn_bar.addWidget(self.btn_proceed)

        layout.addLayout(btn_bar)
