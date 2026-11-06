"""
QC Doctor (Calculation Failure Diagnostician & Auto-Repair) Dialog.
Presents clear root-cause explanations and 1-click recipes for repairing stalled calculations.
"""

from typing import List, Optional
from PyQt6 import QtCore, QtGui, QtWidgets
from .qc_doctor import DiagnosisResult, diagnose_qm_log


class QCDoctorDialog(QtWidgets.QDialog):
    """Interactive modal inspecting failed/aborted QM calculations and offering 1-click remedies."""

    def __init__(self, parent=None, results: Optional[List[DiagnosisResult]] = None, log_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle("🩺 AgyMolden QC Doctor — Calculation Diagnostics & Auto-Repair")
        self.resize(750, 520)
        self.results = results or (diagnose_qm_log(log_text) if log_text else [])

        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Header banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #1e1e2e; border: 1px solid #3b4261; border-radius: 6px; padding: 10px;")
        h_layout = QtWidgets.QHBoxLayout(header)

        icon_lbl = QtWidgets.QLabel("🩺")
        icon_lbl.setStyleSheet("font-size: 32px;")
        h_layout.addWidget(icon_lbl)

        txt_v = QtWidgets.QVBoxLayout()
        title_lbl = QtWidgets.QLabel("Quantum Chemistry Calculation Diagnostics")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #f38ba8;")
        sub_lbl = QtWidgets.QLabel("AgyMolden QC Doctor analyzed your log file to detect failure root causes and generate fixes.")
        sub_lbl.setStyleSheet("color: #a6adc8; font-size: 12px;")
        txt_v.addWidget(title_lbl)
        txt_v.addWidget(sub_lbl)
        h_layout.addLayout(txt_v)
        h_layout.addStretch(1)
        layout.addWidget(header)

        if not self.results:
            no_err_lbl = QtWidgets.QLabel("<b>✓ No known critical calculation errors detected in this log file.</b><br>The calculation appears to have completed normally or encountered an operating system signal.")
            no_err_lbl.setStyleSheet("color: #a6e3a1; font-size: 13px; padding: 20px;")
            layout.addWidget(no_err_lbl)
        else:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background: transparent; border: none;")

            card_container = QtWidgets.QWidget()
            card_layout = QtWidgets.QVBoxLayout(card_container)
            card_layout.setSpacing(12)

            for i, res in enumerate(self.results):
                card = QtWidgets.QFrame()
                border_color = "#f38ba8" if res.severity == "fatal" else "#f9e2af"
                card.setStyleSheet(f"background-color: #181825; border: 1.5px solid {border_color}; border-radius: 6px; padding: 12px;")
                c_layout = QtWidgets.QVBoxLayout(card)
                c_layout.setSpacing(6)

                h_line = QtWidgets.QHBoxLayout()
                badge = QtWidgets.QLabel(f" {res.severity.upper()} ")
                badge_bg = "#dc2626" if res.severity == "fatal" else "#d97706"
                badge.setStyleSheet(f"background-color: {badge_bg}; color: white; font-weight: bold; border-radius: 3px; font-size: 11px; padding: 2px 4px;")
                h_line.addWidget(badge)

                t_lbl = QtWidgets.QLabel(res.headline)
                t_lbl.setStyleSheet("font-weight: bold; font-size: 13px; color: #cdd6f4;")
                h_line.addWidget(t_lbl)
                h_line.addStretch(1)
                c_layout.addLayout(h_line)

                exp_lbl = QtWidgets.QLabel(f"<b>What happened:</b> {res.explanation}")
                exp_lbl.setWordWrap(True)
                exp_lbl.setStyleSheet("color: #bac2de; font-size: 12px; margin-top: 4px;")
                c_layout.addWidget(exp_lbl)

                rem_lbl = QtWidgets.QLabel(f"<b>How to fix:</b> {res.remedy_description}")
                rem_lbl.setWordWrap(True)
                rem_lbl.setStyleSheet("color: #89b4fa; font-size: 12px; margin-top: 2px;")
                c_layout.addWidget(rem_lbl)

                if res.recommended_keywords:
                    kw_box = QtWidgets.QHBoxLayout()
                    kw_title = QtWidgets.QLabel("<b>Recommended Keywords:</b>")
                    kw_title.setStyleSheet("font-size: 11px; color: #a6adc8;")
                    kw_box.addWidget(kw_title)
                    for kw in res.recommended_keywords:
                        kw_tag = QtWidgets.QLabel(f"<code>{kw}</code>")
                        kw_tag.setStyleSheet("background-color: #313244; color: #a6e3a1; padding: 2px 6px; border-radius: 4px; font-size: 11px;")
                        kw_box.addWidget(kw_tag)
                    kw_box.addStretch(1)
                    c_layout.addLayout(kw_box)

                card_layout.addWidget(card)

            card_layout.addStretch(1)
            scroll.setWidget(card_container)
            layout.addWidget(scroll, stretch=1)

        # Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch(1)
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        layout.addLayout(btn_box)
