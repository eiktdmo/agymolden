"""
Shared base dialog and theme-variant helpers for AgyMolden tool dialogs.

Provides consistent margins, spacing, header banners, and standard button
boxes so new and refactored dialogs avoid boilerplate duplication.
"""

from __future__ import annotations

from typing import Optional
from PyQt6 import QtWidgets

_VARIANT_STYLES = {
    "primary": "background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 5px 12px; border-radius: 4px;",
    "success": "background-color: #059669; color: #ffffff; font-weight: bold; padding: 5px 12px; border-radius: 4px;",
    "danger": "background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 5px 12px; border-radius: 4px;",
    "muted": "color: #94a3b8; font-size: 11px;",
}


def apply_style_variant(widget: QtWidgets.QWidget, variant: str) -> None:
    """Apply a standard semantic style variant to a Qt widget."""
    widget.setProperty("agyVariant", variant)
    qss = _VARIANT_STYLES.get(variant)
    if qss:
        widget.setStyleSheet(qss)


class BaseToolDialog(QtWidgets.QDialog):
    """Reusable base class for AgyMolden analysis and builder dialogs."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        title: str = "AgyMolden Tool",
        width: int = 640,
        height: int = 480,
        subtitle: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(width, height)

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        if subtitle:
            lbl = QtWidgets.QLabel(subtitle)
            lbl.setWordWrap(True)
            apply_style_variant(lbl, "muted")
            self.main_layout.addWidget(lbl)

    def add_standard_buttons(
        self,
        buttons: QtWidgets.QDialogButtonBox.StandardButton = (
            QtWidgets.QDialogButtonBox.StandardButton.Close
        ),
    ) -> QtWidgets.QDialogButtonBox:
        """Append a standard QDialogButtonBox wired to accept/reject."""
        box = QtWidgets.QDialogButtonBox(buttons)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        self.main_layout.addWidget(box)
        return box
