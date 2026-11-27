"""
Modern searchable Command Palette (Ctrl+K) for AgyMolden.
Provides fuzzy filtering and instantaneous keyboard navigation to all calculation engines,
analytical tools, builder operations, view modes, and exporters.
"""

from typing import List, Tuple, Callable, Optional
from PyQt6 import QtCore, QtGui, QtWidgets


class CommandItem:
    def __init__(self, title: str, category: str, shortcut: str, callback: Callable, keywords: str = ""):
        self.title = title
        self.category = category
        self.shortcut = shortcut
        self.callback = callback
        self.keywords = keywords.lower()

    def matches(self, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        full_text = f"{self.title.lower()} {self.category.lower()} {self.keywords}"
        # All space-separated words must match
        words = q.split()
        return all(w in full_text for w in words)


class CommandPaletteDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, commands: Optional[List[CommandItem]] = None):
        super().__init__(parent)
        self.commands = commands or []
        self.filtered_commands: List[CommandItem] = []
        self.setWindowTitle("AgyMolden Command Palette")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMaximumWidth(700)
        self.setMinimumHeight(400)

        # Use standard Dialog frame on Wayland so the palette cannot trap focus without a title bar
        if not QtGui.QGuiApplication.platformName().startswith("wayland"):
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.Dialog)
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._init_ui()
        self._filter_commands("")

    def _init_ui(self):
        container = QtWidgets.QFrame(self)
        container.setObjectName("paletteContainer")
        # Use the application's palette roles so the overlay follows the light/dark theme.
        container.setStyleSheet("""
            QFrame#paletteContainer {
                background-color: palette(window);
                border: 2px solid palette(highlight);
                border-radius: 8px;
            }
            QLineEdit {
                background-color: palette(base);
                color: palette(text);
                font-size: 14px;
                padding: 10px 14px;
                border: 1px solid palette(mid);
                border-radius: 6px;
            }
            QLineEdit:focus {
                border: 1px solid palette(highlight);
            }
            QListWidget {
                background-color: transparent;
                border: none;
                color: palette(window-text);
                font-size: 13px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: palette(highlight);
                color: palette(highlighted-text);
            }
            QListWidget::item:hover:!selected {
                background-color: palette(midlight);
            }
            QLabel#hintLabel {
                color: palette(placeholder-text);
                font-size: 11px;
            }
        """)

        outer_layout = QtWidgets.QVBoxLayout(self)
        outer_layout.setContentsMargins(8, 8, 8, 8)
        outer_layout.addWidget(container)

        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Search Bar
        self.search_edit = QtWidgets.QLineEdit()
        self.search_edit.setPlaceholderText("Type a command, tool, or calculation engine (e.g., ORCA, ESP, Optimize, Export)...")
        self.search_edit.textChanged.connect(self._filter_commands)
        layout.addWidget(self.search_edit)

        # List of Matching Commands
        self.list_widget = QtWidgets.QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list_widget.itemActivated.connect(self._execute_selected)
        layout.addWidget(self.list_widget, stretch=1)

        # Footer hint
        footer = QtWidgets.QHBoxLayout()
        hint = QtWidgets.QLabel("↑/↓ to navigate  •  Enter to select  •  Esc to close")
        hint.setObjectName("hintLabel")
        footer.addWidget(hint)
        footer.addStretch(1)
        layout.addLayout(footer)

    def keyPressEvent(self, event: Optional[QtGui.QKeyEvent]):
        if event is None:
            return
        key = event.key()
        if key == QtCore.Qt.Key.Key_Escape:
            self.reject()
            return
        elif key == QtCore.Qt.Key.Key_Down:
            cur = self.list_widget.currentRow()
            if cur < self.list_widget.count() - 1:
                self.list_widget.setCurrentRow(cur + 1)
            return
        elif key == QtCore.Qt.Key.Key_Up:
            cur = self.list_widget.currentRow()
            if cur > 0:
                self.list_widget.setCurrentRow(cur - 1)
            return
        elif key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            self._execute_selected()
            return
        super().keyPressEvent(event)

    def _filter_commands(self, text: str):
        self.list_widget.clear()
        self.filtered_commands = [c for c in self.commands if c.matches(text)]
        for cmd in self.filtered_commands:
            item = QtWidgets.QListWidgetItem()
            short_tag = f"  [{cmd.shortcut}]" if cmd.shortcut else ""
            item.setText(f"[{cmd.category}]  {cmd.title}{short_tag}")
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def _execute_selected(self):
        row = self.list_widget.currentRow()
        if 0 <= row < len(self.filtered_commands):
            cmd = self.filtered_commands[row]
            self.accept()
            cmd.callback()
