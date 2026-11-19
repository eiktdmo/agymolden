"""
Interactive Quantum Chemistry Output & Log File Viewer Dialog.

Features:
- Fast read-only log viewing via QPlainTextEdit.
- Automatic section indexing (Header, SCF, Geometry Cycles, Normal Modes, Errors/Warnings).
- Clickable section list on the left to jump directly to corresponding log lines.
- Real-time text search and occurrence highlighting.
- Integration with external text editors and system clipboard.
"""

import os
import re
from typing import List, Tuple, Optional
from PyQt6 import QtCore, QtGui, QtWidgets


class QMOutputViewerDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, file_path: str = ""):
        super().__init__(parent)
        self.file_path = os.path.abspath(file_path) if file_path else ""
        self.line_indices: List[Tuple[str, int]] = []  # (Section title, 0-based line index)

        title = f"QM Output Viewer – {os.path.basename(self.file_path)}" if self.file_path else "QM Output Viewer"
        self.setWindowTitle(title)
        self.resize(920, 640)

        self._init_ui()
        if self.file_path and os.path.isfile(self.file_path):
            self.load_file(self.file_path)

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        # 1. Top control banner (Path, Search bar, Match counter)
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(8)

        lbl_path = QtWidgets.QLabel(f"<b>File:</b> {os.path.basename(self.file_path)}" if self.file_path else "<b>File:</b> None")
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        path_col = "#0284c7" if is_white else "#38bdf8"
        lbl_path.setStyleSheet(f"color: {path_col}; font-size: 12px;")
        top_bar.addWidget(lbl_path)
        top_bar.addStretch()

        top_bar.addWidget(QtWidgets.QLabel("🔍 Search:"))
        self.txt_search = QtWidgets.QLineEdit()
        self.txt_search.setPlaceholderText("Find text in log...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.setFixedWidth(220)
        self.txt_search.textChanged.connect(self._on_search_changed)
        self.txt_search.returnPressed.connect(self._find_next)
        top_bar.addWidget(self.txt_search)

        btn_find_prev = QtWidgets.QPushButton("▲")
        btn_find_prev.setFixedWidth(28)
        btn_find_prev.setToolTip("Previous match (Shift+F3)")
        btn_find_prev.clicked.connect(self._find_prev)
        top_bar.addWidget(btn_find_prev)

        btn_find_next = QtWidgets.QPushButton("▼")
        btn_find_next.setFixedWidth(28)
        btn_find_next.setToolTip("Next match (F3)")
        btn_find_next.clicked.connect(self._find_next)
        top_bar.addWidget(btn_find_next)

        self.lbl_match_count = QtWidgets.QLabel("")
        self.lbl_match_count.setStyleSheet("color: #94a3b8; font-size: 11px;")
        top_bar.addWidget(self.lbl_match_count)

        layout.addLayout(top_bar)

        # 2. Main Horizontal Splitter (Sections Tree on left, Log Text on right)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left Card: Sections & Key Milestones
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        lbl_sec = QtWidgets.QLabel("<b>📑 Key Sections & Steps:</b>")
        left_layout.addWidget(lbl_sec)

        self.list_sections = QtWidgets.QListWidget()
        self.list_sections.itemClicked.connect(self._on_section_clicked)
        left_layout.addWidget(self.list_sections)

        splitter.addWidget(left_widget)

        # Right Card: Plain Text Viewport
        self.txt_viewer = QtWidgets.QPlainTextEdit()
        self.txt_viewer.setReadOnly(True)
        self.txt_viewer.setFont(QtGui.QFont("Monospace", 9))
        self.txt_viewer.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        is_white = self._is_white_theme()
        if is_white:
            self.txt_viewer.setStyleSheet(
                "background-color: #ffffff; color: #0f172a; font-family: monospace; "
                "selection-background-color: #2563eb; selection-color: #ffffff; border: 1px solid #cbd5e1;"
            )
            self.list_sections.setStyleSheet(
                "QListWidget { background-color: #ffffff; color: #0f172a; border: 1px solid #cbd5e1; font-size: 11px; } "
                "QListWidget::item { padding: 4px 6px; border-radius: 3px; } "
                "QListWidget::item:selected { background-color: #2563eb; color: #ffffff; } "
                "QListWidget::item:hover:!selected { background-color: #f1f5f9; }"
            )
        else:
            self.txt_viewer.setStyleSheet(
                "background-color: #0f172a; color: #f8fafc; font-family: monospace; "
                "selection-background-color: #2563eb; selection-color: #ffffff; border: 1px solid #334155;"
            )
            self.list_sections.setStyleSheet(
                "QListWidget { background-color: #1e1e24; color: #f8fafc; border: 1px solid #334155; font-size: 11px; } "
                "QListWidget::item { padding: 4px 6px; border-radius: 3px; } "
                "QListWidget::item:selected { background-color: #3b82f6; color: #ffffff; } "
                "QListWidget::item:hover:!selected { background-color: #2e2e38; }"
            )
        splitter.addWidget(self.txt_viewer)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([260, 640])
        layout.addWidget(splitter, stretch=1)

        # 3. Bottom Action Buttons
        bot_bar = QtWidgets.QHBoxLayout()

        self.btn_doctor = QtWidgets.QPushButton("🩺 QC Doctor Diagnose & Fix")
        self.btn_doctor.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold;")
        self.btn_doctor.setToolTip("Scan this output log with QC Doctor to detect convergence failures, errors, and remedies.")
        self.btn_doctor.clicked.connect(self._on_qc_doctor_clicked)
        bot_bar.addWidget(self.btn_doctor)

        btn_copy = QtWidgets.QPushButton("📋 Copy All to Clipboard")
        btn_copy.clicked.connect(self._on_copy_all)
        bot_bar.addWidget(btn_copy)

        btn_ext = QtWidgets.QPushButton("📂 Open in System Editor")
        btn_ext.clicked.connect(self._on_open_external)
        bot_bar.addWidget(btn_ext)

        bot_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bot_bar.addWidget(btn_close)

        layout.addLayout(bot_bar)

    def _is_white_theme(self) -> bool:
        """Check if application or viewer is currently using white theme."""
        p = self.parent()
        while p is not None:
            if hasattr(p, "viewer") and getattr(p.viewer, "bg_color_mode", ""):
                return p.viewer.bg_color_mode == "white"
            p = p.parent()
        # Check palette Window color brightness
        window_lum = self.palette().color(QtGui.QPalette.ColorRole.Window).lightness()
        if window_lum < 100:
            return False
        from agymolden.session_state import get_settings
        settings = get_settings("AgyMolden")
        return settings.value("color_theme", "white") == "white"

    def load_file(self, path: str):
        """Read and parse the output file into the text view and index sections."""
        if not os.path.isfile(path):
            self.txt_viewer.setPlainText(f"Error: File not found: {path}")
            return

        self.file_path = os.path.abspath(path)
        self.setWindowTitle(f"QM Output Viewer – {os.path.basename(self.file_path)}")

        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            self.txt_viewer.setPlainText(f"Error opening file:\n{e}")
            return

        self.txt_viewer.setPlainText(content)
        self._index_sections(content)

    def _index_sections(self, content: str):
        """Scan file lines for key milestones (Cycles, SCF, Frequencies, Warnings, Convergence)."""
        self.list_sections.clear()
        self.line_indices.clear()

        lines = content.splitlines()
        first_step = True

        for idx, line in enumerate(lines):
            s = line.strip()
            # Match optimization steps (ORCA: GEOMETRY OPTIMIZATION CYCLE 1, Gaussian: Cycle 1, Step 1)
            m_cycle = re.search(
                r"(?:GEOMETRY\s+OPTIMIZATION\s+CYCLE|Geometry\s+optimization\s+cycle|^Cycle\s+|^Step\s+)\s*(\d+)",
                s,
                re.IGNORECASE,
            )
            if m_cycle and len(s) < 80 and not re.search(r"(?:RMS|MAX|step\s+\d+\.\d+)", s, re.IGNORECASE):
                c_num = m_cycle.group(1)
                self._add_section_item(f"🔄 Optimization Step {c_num}", idx)
                continue

            # Match SCF / Energy
            e_match = re.search(
                r"(?:FINAL\s+SINGLE\s+POINT\s+ENERGY|Total\s+Energy|SCF\s+Done|E\(HF\)|E\(DFT\)|E\(CCSD\))\s*[:=]?\s*([+-]?\d+\.\d+)",
                s,
                re.IGNORECASE,
            )
            if e_match:
                if len(self.line_indices) < 35 or "Done" in s or "FINAL" in s:
                    self._add_section_item(f"⚡ E = {e_match.group(1)}", idx)
                continue

            # Match Convergence
            s_l = s.lower()
            if "the optimization has converged" in s_l:
                self._add_section_item("🎉 Optimization Converged", idx)
                continue
            if "scf converged" in s_l:
                self._add_section_item("✔ SCF Converged", idx)
                continue
            if "converged" in s_l and not any(k in s_l for k in ["not yet converged", "failed to converge", "item", "tolerance"]) and len(s) < 80:
                self._add_section_item(f"✔ {s[:45]}", idx)
                continue

            # Match Vibrational Analysis
            if re.search(r"(?:Harmonic frequencies|Vibrational Frequencies|NORMAL MODES)", s, re.IGNORECASE):
                self._add_section_item("🎵 Harmonic Frequencies / Modes", idx)
                continue

            # Match Normal Termination
            if "terminated normally" in s_l:
                self._add_section_item("🏁 Terminated Normally", idx)
                continue

            # Match Errors / Warnings
            if any(k in s_l for k in ["startup error", "diis error", "||err||", "tolerr", "grad/error", "cause some error"]):
                pass
            elif re.search(r"\b(?:error(?:\s+termination|\s+in\s+|\s+has\s+occurred|!)|fatal|abnormal|crash|failed to converge)\b", s_l) and len(s) < 95:
                self._add_section_item(f"⚠️ {s[:45]}", idx)
                continue

        if not self.line_indices:
            self.list_sections.addItem("No specific section markers detected")

    def _add_section_item(self, label: str, line_no: int):
        self.line_indices.append((label, line_no))
        item = QtWidgets.QListWidgetItem(f"{label}  (L:{line_no+1})")
        item.setData(QtCore.Qt.ItemDataRole.UserRole, line_no)
        self.list_sections.addItem(item)

    def _on_section_clicked(self, item: QtWidgets.QListWidgetItem):
        line_no = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if line_no is not None:
            self._jump_to_line(line_no)

    def _jump_to_line(self, line_no: int):
        doc = self.txt_viewer.document()
        block = doc.findBlockByLineNumber(line_no)
        if block.isValid():
            cursor = QtGui.QTextCursor(block)
            self.txt_viewer.setTextCursor(cursor)
            self.txt_viewer.centerCursor()

    def _on_search_changed(self, text: str):
        if not text:
            self.lbl_match_count.setText("")
            # Clear search highlight
            self.txt_viewer.setExtraSelections([])
            return

        # Highlight all occurrences
        selections = []
        doc = self.txt_viewer.document()
        fmt = QtGui.QTextCharFormat()
        fmt.setBackground(QtGui.QColor("#ca8a04"))  # Warm golden yellow
        fmt.setForeground(QtGui.QColor("#ffffff"))

        cursor = QtGui.QTextCursor(doc)
        count = 0
        while True:
            cursor = doc.find(text, cursor)
            if cursor.isNull():
                break
            sel = QtWidgets.QTextEdit.ExtraSelection()
            sel.cursor = cursor
            sel.format = fmt
            selections.append(sel)
            count += 1
            if count > 2000:
                break

        self.txt_viewer.setExtraSelections(selections)
        self.lbl_match_count.setText(f"{count} match(es)" if count <= 2000 else ">2000 matches")
        self._find_next()

    def _find_next(self):
        query = self.txt_search.text()
        if not query:
            return
        if not self.txt_viewer.find(query):
            # Wrap around to start
            cursor = self.txt_viewer.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
            self.txt_viewer.setTextCursor(cursor)
            self.txt_viewer.find(query)

    def _find_prev(self):
        query = self.txt_search.text()
        if not query:
            return
        flags = QtGui.QTextDocument.FindFlag.FindBackward
        if not self.txt_viewer.find(query, flags):
            # Wrap around to end
            cursor = self.txt_viewer.textCursor()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
            self.txt_viewer.setTextCursor(cursor)
            self.txt_viewer.find(query, flags)

    def _on_copy_all(self):
        QtWidgets.QApplication.clipboard().setText(self.txt_viewer.toPlainText())
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Copied entire output log to clipboard!", self)

    def _on_qc_doctor_clicked(self):
        """Analyze the current output file with QC Doctor."""
        from agymolden.calc_runner.diagnostics import diagnose_qm_error
        raw_text = self.txt_viewer.toPlainText()
        if not raw_text.strip():
            QtWidgets.QMessageBox.information(self, "QC Doctor", "No log output is loaded to analyze.")
            return

        diag = diagnose_qm_error(raw_text)
        if diag:
            msg_box = QtWidgets.QMessageBox(self)
            msg_box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            msg_box.setWindowTitle(f"QC Doctor: {diag.title}")
            msg_box.setText(f"<b>🩺 QC Doctor Diagnosis:</b><br><h3>{diag.title}</h3><p>{diag.summary}</p>")
            msg_box.setInformativeText(f"<b>Recommended Fix / Recovery Strategy:</b>\n{diag.remedy}")
            msg_box.setDetailedText(raw_text[-2500:] if len(raw_text) > 2500 else raw_text)
            msg_box.exec()
        else:
            QtWidgets.QMessageBox.information(
                self,
                "QC Doctor",
                "<b>🩺 QC Doctor Diagnosis:</b><br><br>"
                "No critical convergence failures, allocation crashes, or fatal syntax errors detected.<br>"
                "The calculation appears normal or healthy!"
            )

    def _on_open_external(self):
        if self.file_path and os.path.isfile(self.file_path):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(self.file_path))
        else:
            QtWidgets.QMessageBox.information(self, "External Editor", "File path is not available on disk.")
