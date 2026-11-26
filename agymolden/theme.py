"""
Centralized Qt Stylesheet (QSS) and Theme Definitions for AgyMolden.

Provides DARK_THEME_QSS and LIGHT_THEME_QSS along with resource arrow paths.
"""

import os

RESOURCES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
ARROW_DARK_PATH = os.path.join(RESOURCES_DIR, "arrow_dark.svg").replace("\\", "/")
ARROW_LIGHT_PATH = os.path.join(RESOURCES_DIR, "arrow_light.svg").replace("\\", "/")
CHECKMARK_PATH = os.path.join(RESOURCES_DIR, "check_white.svg").replace("\\", "/")
SPIN_UP_LIGHT_PATH = os.path.join(RESOURCES_DIR, "spin_up_light.svg").replace("\\", "/")
SPIN_DOWN_LIGHT_PATH = os.path.join(RESOURCES_DIR, "spin_down_light.svg").replace("\\", "/")
SPIN_UP_DARK_PATH = os.path.join(RESOURCES_DIR, "spin_up_dark.svg").replace("\\", "/")
SPIN_DOWN_DARK_PATH = os.path.join(RESOURCES_DIR, "spin_down_dark.svg").replace("\\", "/")

DARK_THEME_QSS = """
QMainWindow, QDialog {
    background-color: #1e1e24;
    color: #e0e0e0;
}
QLabel {
    color: #e0e0e0;
}
QCheckBox, QRadioButton {
    color: #f1f5f9;
    spacing: 6px;
    font-size: 12px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #64748b;
    border-radius: 3px;
    background-color: #1e1e28;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #60a5fa;
    background-color: #282838;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    border-color: #3b82f6;
    background-color: #2563eb;
    image: url("__CHECKMARK_PATH__");
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #475569;
    background-color: #1e1e24;
}
QMenuBar {
    background-color: #18181c;
    color: #cccccc;
    border-bottom: 1px solid #2a2a32;
    padding: 2px;
}
QMenuBar::item:selected {
    background-color: #2d3748;
    border-radius: 4px;
}
QMenu {
    background-color: #24242c;
    color: #e0e0e0;
    border: 1px solid #383844;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #3b82f6;
    color: #ffffff;
    border-radius: 3px;
}
QToolBar {
    background-color: #22222a;
    border-bottom: 1px solid #2e2e38;
    spacing: 6px;
    padding: 4px;
}
QToolButton {
    background-color: #2b2b36;
    color: #eeeeee;
    border: 1px solid #3d3d4c;
    border-radius: 4px;
    padding: 4px;
    font-weight: 500;
}
QFileDialog QToolButton {
    min-width: 24px;
    min-height: 24px;
    padding: 2px;
}
QToolButton:hover {
    background-color: #383848;
    border-color: #4f526b;
}
QToolButton:pressed {
    background-color: #2563eb;
    border-color: #1d4ed8;
}
QTabWidget::pane {
    border: 1px solid #2e2e38;
    background-color: #1a1a20;
}
QTabBar::tab {
    background-color: #24242c;
    color: #a0a0b0;
    border: 1px solid #32323e;
    padding: 6px 10px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #1a1a20;
    color: #ffffff;
    border-bottom-color: #1a1a20;
    font-weight: bold;
}
QTableWidget {
    background-color: #1a1a20;
    color: #e2e8f0;
    gridline-color: #2d3748;
    border: 1px solid #2e2e38;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #24242c;
    color: #94a3b8;
    padding: 5px;
    border: 1px solid #2e2e38;
    font-weight: 600;
}
QTextEdit, QPlainTextEdit {
    background-color: #141418;
    color: #a7f3d0;
    font-family: 'Fira Code', 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 13px;
    border: 1px solid #2e2e38;
    border-radius: 4px;
}
QGroupBox {
    border: 1px solid #333342;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    color: #93c5fd;
    padding: 16px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #3b82f6;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #2563eb;
}
QPushButton:pressed {
    background-color: #1d4ed8;
}
QPushButton:disabled {
    background-color: #374151;
    color: #9ca3af;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #374151;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3b82f6;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #24242c;
    color: #f1f5f9;
    border: 1px solid #3d3d4c;
    border-radius: 4px;
    padding: 3px 8px;
    min-height: 26px;
}
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    border: none;
    padding: 0px 2px;
    min-height: 0px;
    background: transparent;
}
QComboBox QAbstractItemView, QComboBoxPrivateContainer, QComboBoxPrivateContainer QFrame, QListView, QTreeView, QTableView, QListWidget {
    background-color: #24242c;
    color: #f1f5f9;
    selection-background-color: #3b82f6;
    selection-color: #ffffff;
    border: 1px solid #383844;
    outline: none;
    padding: 2px;
}
QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 3px 8px;
}
QListWidget::item {
    padding: 3px 6px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #3b82f6;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background-color: #2e2e38;
}
QSplitter::handle {
    background-color: #383844;
}
QSplitter::handle:horizontal {
    width: 5px;
}
QSplitter::handle:vertical {
    height: 5px;
}
QSplitter::handle:hover {
    background-color: #3b82f6;
}
QScrollBar:vertical {
    background: #1e1e24;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #3d3d4c;
    min-height: 24px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #525266;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #1e1e24;
    height: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #3d3d4c;
    min-width: 24px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal:hover {
    background: #525266;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #3d3d4c;
}
QComboBox::down-arrow {
    image: url("__ARROW_LIGHT_PATH__");
    width: 10px;
    height: 6px;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 20px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #3d3d4c;
    border-bottom: 1px solid #3d3d4c;
    background-color: #2b2b36;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #3b82f6;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("__SPIN_UP_LIGHT_PATH__");
    width: 8px;
    height: 5px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #3d3d4c;
    background-color: #2b2b36;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #3b82f6;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("__SPIN_DOWN_LIGHT_PATH__");
    width: 8px;
    height: 5px;
}
QStatusBar {
    background-color: #141418;
    color: #94a3b8;
    border-top: 1px solid #24242c;
    font-size: 12px;
}
""".replace("__ARROW_LIGHT_PATH__", ARROW_LIGHT_PATH).replace("__CHECKMARK_PATH__", CHECKMARK_PATH).replace("__SPIN_UP_LIGHT_PATH__", SPIN_UP_LIGHT_PATH).replace("__SPIN_DOWN_LIGHT_PATH__", SPIN_DOWN_LIGHT_PATH)

LIGHT_THEME_QSS = """
QMainWindow, QDialog {
    background-color: #f8fafc;
    color: #1e293b;
}
QLabel {
    color: #1e293b;
}
QCheckBox, QRadioButton {
    color: #0f172a;
    spacing: 6px;
    font-size: 12px;
}
QCheckBox::indicator, QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #94a3b8;
    border-radius: 3px;
    background-color: #ffffff;
}
QRadioButton::indicator {
    border-radius: 8px;
}
QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: #2563eb;
    background-color: #f1f5f9;
}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {
    border-color: #2563eb;
    background-color: #2563eb;
    image: url("__CHECKMARK_PATH__");
}
QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: #cbd5e1;
    background-color: #f8fafc;
}
QMenuBar {
    background-color: #f1f5f9;
    color: #1e293b;
    border-bottom: 1px solid #cbd5e1;
    padding: 2px;
}
QMenuBar::item:selected {
    background-color: #e2e8f0;
    border-radius: 4px;
}
QMenu {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    padding: 4px;
}
QMenu::item:selected {
    background-color: #2563eb;
    color: #ffffff;
    border-radius: 3px;
}
QToolBar {
    background-color: #f1f5f9;
    border-bottom: 1px solid #cbd5e1;
    spacing: 6px;
    padding: 4px;
}
QToolButton {
    background-color: #ffffff;
    color: #1e293b;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 4px;
    font-weight: 500;
}
QFileDialog QToolButton {
    min-width: 24px;
    min-height: 24px;
    padding: 2px;
}
QToolButton:hover {
    background-color: #e2e8f0;
    border-color: #94a3b8;
}
QToolButton:pressed {
    background-color: #2563eb;
    color: #ffffff;
    border-color: #1d4ed8;
}
QTabWidget::pane {
    border: 1px solid #cbd5e1;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #f1f5f9;
    color: #475569;
    border: 1px solid #cbd5e1;
    padding: 6px 10px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-size: 12px;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #0f172a;
    border-bottom-color: #ffffff;
    font-weight: bold;
}
QTableWidget {
    background-color: #ffffff;
    color: #0f172a;
    gridline-color: #e2e8f0;
    border: 1px solid #cbd5e1;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
}
QHeaderView::section {
    background-color: #f1f5f9;
    color: #475569;
    padding: 5px;
    border: 1px solid #cbd5e1;
    font-weight: 600;
}
QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    color: #0f172a;
    font-family: 'Fira Code', 'DejaVu Sans Mono', 'Courier New', monospace;
    font-size: 13px;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
}
QGroupBox {
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    color: #2563eb;
    padding: 16px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}
QPushButton {
    background-color: #2563eb;
    color: white;
    border: none;
    border-radius: 4px;
    padding: 6px 10px;
    font-weight: 600;
    font-size: 12px;
}
QPushButton:hover {
    background-color: #1d4ed8;
}
QPushButton:pressed {
    background-color: #1e40af;
}
QPushButton:disabled {
    background-color: #e2e8f0;
    color: #94a3b8;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #e2e8f0;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #2563eb;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ffffff;
    border: 1px solid #94a3b8;
    width: 14px;
    margin-top: -4px;
    margin-bottom: -4px;
    border-radius: 7px;
}
QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 3px 8px;
    min-height: 26px;
}
QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QComboBox QLineEdit {
    border: none;
    padding: 0px 2px;
    min-height: 0px;
    background: transparent;
}
QComboBox QAbstractItemView, QComboBoxPrivateContainer, QComboBoxPrivateContainer QFrame, QListView, QTreeView, QTableView, QListWidget {
    background-color: #ffffff;
    color: #0f172a;
    selection-background-color: #2563eb;
    selection-color: #ffffff;
    border: 1px solid #cbd5e1;
    outline: none;
    padding: 2px;
}
QComboBox QAbstractItemView::item {
    min-height: 24px;
    padding: 3px 8px;
}
QListWidget::item {
    padding: 3px 6px;
    border-radius: 3px;
}
QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}
QListWidget::item:hover:!selected {
    background-color: #f1f5f9;
}
QSplitter::handle {
    background-color: #cbd5e1;
}
QSplitter::handle:horizontal {
    width: 5px;
}
QSplitter::handle:vertical {
    height: 5px;
}
QSplitter::handle:hover {
    background-color: #2563eb;
}
QScrollBar:vertical {
    background: #f8fafc;
    width: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:vertical {
    background: #cbd5e1;
    min-height: 24px;
    border-radius: 6px;
}
QScrollBar::handle:vertical:hover {
    background: #94a3b8;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background: #f8fafc;
    height: 12px;
    margin: 0px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal {
    background: #cbd5e1;
    min-width: 24px;
    border-radius: 6px;
}
QScrollBar::handle:horizontal:hover {
    background: #94a3b8;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid #cbd5e1;
}
QComboBox::down-arrow {
    image: url("__ARROW_DARK_PATH__");
    width: 10px;
    height: 6px;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 20px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid #cbd5e1;
    border-bottom: 1px solid #cbd5e1;
    background-color: #f1f5f9;
}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
    background-color: #e2e8f0;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    image: url("__SPIN_UP_DARK_PATH__");
    width: 8px;
    height: 5px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid #cbd5e1;
    background-color: #f1f5f9;
}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
    background-color: #e2e8f0;
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    image: url("__SPIN_DOWN_DARK_PATH__");
    width: 8px;
    height: 5px;
}
QStatusBar {
    background-color: #f1f5f9;
    color: #475569;
    border-top: 1px solid #cbd5e1;
    font-size: 12px;
}
""".replace("__ARROW_DARK_PATH__", ARROW_DARK_PATH).replace("__CHECKMARK_PATH__", CHECKMARK_PATH).replace("__SPIN_UP_DARK_PATH__", SPIN_UP_DARK_PATH).replace("__SPIN_DOWN_DARK_PATH__", SPIN_DOWN_DARK_PATH)


def fit_dialog_to_screen(dialog, pref_w: int, pref_h: int, full_height: bool = False):
    """Resize a dialog to preferred dimensions, clamped cleanly to the primary screen's available area."""
    try:
        from PyQt6 import QtGui
        scr = QtGui.QGuiApplication.primaryScreen()
        if scr is not None:
            avail = scr.availableGeometry()
            if full_height:
                w = min(pref_w, max(520, int(avail.width() * 0.90)))
                h = max(520, int(avail.height() * 0.94))
                dialog.resize(w, h)
                dialog.move(
                    avail.left() + max(0, (avail.width() - w) // 2),
                    avail.top() + max(8, (avail.height() - h) // 2),
                )
            else:
                w = min(pref_w, max(480, int(avail.width() * 0.90)))
                h = min(pref_h, max(420, int(avail.height() * 0.88)))
                dialog.resize(w, h)
            return
    except Exception:
        pass
    dialog.resize(pref_w, pref_h)
