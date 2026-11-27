"""Curated Sample & Benchmark Gallery Dialog for AgyMolden."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from .data_samples import SAMPLE_CATALOG, ChemistrySample


class SampleGalleryDialog(QtWidgets.QDialog):
    """Visual gallery dialog allowing users to browse and open curated chemical systems in 1 click."""

    sample_selected = QtCore.pyqtSignal(str)  # emits raw XYZ string

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("📚 Chemistry Sample & Benchmark Gallery — AgyMolden")
        self.resize(800, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header Info Banner
        is_white = False
        parent_w = self.parent()
        if parent_w is not None and hasattr(parent_w, "viewer") and getattr(parent_w.viewer, "bg_color_mode", None) == "white":
            is_white = True
        elif self.palette().window().color().lightness() > 128:
            is_white = True

        header = QtWidgets.QFrame()
        if is_white:
            header.setStyleSheet("""
                QFrame {
                    background-color: #f1f5f9;
                    border: 1px solid #cbd5e1;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        else:
            header.setStyleSheet("""
                QFrame {
                    background-color: #1e293b;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    padding: 10px;
                }
            """)
        h_layout = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Curated Benchmark & Tutorial Systems</b>")
        h_title.setStyleSheet(f"font-size: 14px; color: {'#0369a1' if is_white else '#38bdf8'};")
        h_desc = QtWidgets.QLabel(
            "Select any sample structure below to immediately load it into the 3D viewport, "
            "inspect orbitals, or run calculations without needing to build from scratch."
        )
        h_desc.setStyleSheet(f"color: {'#334155' if is_white else '#cbd5e1'}; font-size: 11px;")
        h_desc.setWordWrap(True)
        h_layout.addWidget(h_title)
        h_layout.addWidget(h_desc)
        layout.addWidget(header)

        # Splitter: List on Left, Details & Preview on Right
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        self.list_samples = QtWidgets.QListWidget()
        if is_white:
            self.list_samples.setStyleSheet("""
                QListWidget {
                    background-color: #ffffff;
                    color: #0f172a;
                    border: 1px solid #cbd5e1;
                    border-radius: 6px;
                    font-size: 12px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #e2e8f0;
                }
                QListWidget::item:selected {
                    background-color: #0284c7;
                    color: #ffffff;
                }
            """)
        else:
            self.list_samples.setStyleSheet("""
                QListWidget {
                    background-color: #0f172a;
                    color: #f8fafc;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    font-size: 12px;
                }
                QListWidget::item {
                    padding: 8px;
                    border-bottom: 1px solid #1e293b;
                }
                QListWidget::item:selected {
                    background-color: #0284c7;
                    color: #ffffff;
                }
            """)
        for sample in SAMPLE_CATALOG:
            item = QtWidgets.QListWidgetItem(f"⚛ {sample.title} ({sample.formula})")
            self.list_samples.addItem(item)
        splitter.addWidget(self.list_samples)

        # Right side detail panel
        right_panel = QtWidgets.QFrame()
        r_layout = QtWidgets.QVBoxLayout(right_panel)
        r_layout.setContentsMargins(10, 0, 0, 0)

        self.lbl_title = QtWidgets.QLabel("Select a sample...")
        self.lbl_title.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {'#0f172a' if is_white else '#f8fafc'};")
        r_layout.addWidget(self.lbl_title)

        self.lbl_category = QtWidgets.QLabel("")
        self.lbl_category.setStyleSheet(f"font-size: 11px; color: {'#475569' if is_white else '#94a3b8'};")
        r_layout.addWidget(self.lbl_category)

        self.lbl_description = QtWidgets.QLabel("")
        self.lbl_description.setStyleSheet(f"font-size: 12px; color: {'#1e293b' if is_white else '#e2e8f0'}; line-height: 1.4;")
        self.lbl_description.setWordWrap(True)
        r_layout.addWidget(self.lbl_description)

        r_layout.addSpacing(10)
        self.txt_xyz = QtWidgets.QPlainTextEdit()
        self.txt_xyz.setReadOnly(True)
        self.txt_xyz.setStyleSheet("""
            QPlainTextEdit {
                background-color: #020617;
                color: #38bdf8;
                font-family: monospace;
                font-size: 11px;
                border: 1px solid #334155;
                border-radius: 4px;
            }
        """)
        r_layout.addWidget(self.txt_xyz, 1)

        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        # Bottom Buttons
        b_bar = QtWidgets.QHBoxLayout()
        b_bar.addStretch()

        self.btn_load = QtWidgets.QPushButton("🚀 Open in 3D Viewport")
        self.btn_load.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                font-weight: bold;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_load.clicked.connect(self._on_load_clicked)
        b_bar.addWidget(self.btn_load)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        b_bar.addWidget(btn_cancel)
        layout.addLayout(b_bar)

        self.list_samples.currentRowChanged.connect(self._on_row_changed)
        if len(SAMPLE_CATALOG) > 0:
            self.list_samples.setCurrentRow(0)

    def _on_row_changed(self, row: int):
        if 0 <= row < len(SAMPLE_CATALOG):
            s = SAMPLE_CATALOG[row]
            self.lbl_title.setText(s.title)
            self.lbl_category.setText(f"Category: {s.category}")
            self.lbl_description.setText(s.description)
            self.txt_xyz.setPlainText(s.xyz_data)

    def _on_load_clicked(self):
        row = self.list_samples.currentRow()
        if 0 <= row < len(SAMPLE_CATALOG):
            s = SAMPLE_CATALOG[row]
            self.sample_selected.emit(s.xyz_data)
            self.accept()
