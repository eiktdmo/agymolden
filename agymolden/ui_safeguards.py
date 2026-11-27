"""Low-resolution screen safeguards for AgyMolden windows and dialogs.

Ensures that no QDialog or QMainWindow exceeds the available screen geometry
on low-resolution displays (e.g., 800x600, 1024x600, 1366x768), preventing
action buttons at the bottom of dialogs from becoming unreachable.
"""

from __future__ import annotations

import logging
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

logger = logging.getLogger(__name__)

# Leave a small margin for window decorations / taskbar
_DEFAULT_MARGIN_W = 16
_DEFAULT_MARGIN_H = 40
_MIN_CLAMP_W = 320
_MIN_CLAMP_H = 240


def get_available_screen_geometry(
    widget: QtWidgets.QWidget,
) -> Optional[QtCore.QRect]:
    """Return the available geometry for the widget's screen, or None."""
    screen = widget.screen()
    if screen is None:
        app = QtWidgets.QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
    if screen is None:
        return None
    avail = screen.availableGeometry()
    if not avail.isValid() or avail.width() <= 0 or avail.height() <= 0:
        return None
    return avail


def clamp_window_to_screen(
    widget: QtWidgets.QWidget,
    avail: Optional[QtCore.QRect] = None,
    margin_w: int = _DEFAULT_MARGIN_W,
    margin_h: int = _DEFAULT_MARGIN_H,
) -> bool:
    """Clamp a top-level window's minimum size, current size, and position to fit `avail`.

    Returns True if any dimension or position was adjusted.
    """
    if widget.property("_skip_screen_clamp"):
        return False

    if avail is None:
        avail = get_available_screen_geometry(widget)
    if avail is None:
        return False

    max_w = max(_MIN_CLAMP_W, avail.width() - margin_w)
    max_h = max(_MIN_CLAMP_H, avail.height() - margin_h)
    changed = False

    # 1. Clamp minimumSize so Qt's layout engine does not force an oversized window.
    min_w = widget.minimumWidth()
    min_h = widget.minimumHeight()
    if min_w > max_w or min_h > max_h:
        widget.setMinimumSize(min(min_w, max_w), min(min_h, max_h))
        changed = True

    # 2. If the layout's minimumSizeHint exceeds the screen, relax sizeConstraint
    #    so the window can actually shrink below the layout's preferred minimum.
    layout = widget.layout()
    if layout is not None:
        lmin = layout.minimumSize()
        if lmin.width() > max_w or lmin.height() > max_h:
            if layout.sizeConstraint() != QtWidgets.QLayout.SizeConstraint.SetNoConstraint:
                layout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetNoConstraint)
                changed = True

    # 3. Clamp current size.
    cur_w = widget.width()
    cur_h = widget.height()
    target_w = min(cur_w, max_w) if cur_w > max_w else cur_w
    target_h = min(cur_h, max_h) if cur_h > max_h else cur_h
    if target_w != cur_w or target_h != cur_h:
        widget.resize(target_w, target_h)
        changed = True

    # 4. Keep window position inside the available screen area (skip on Wayland where move() is a no-op/unsafe).
    if not QtGui.QGuiApplication.platformName().startswith("wayland"):
        pos = widget.pos()
        new_x = max(avail.left(), min(pos.x(), avail.right() - widget.width() + 1))
        new_y = max(avail.top(), min(pos.y(), avail.bottom() - widget.height() + 1))
        if new_x != pos.x() or new_y != pos.y():
            widget.move(new_x, new_y)
            changed = True

    return changed


class LowResolutionSafeguardFilter(QtCore.QObject):
    """Application-wide event filter that clamps QDialog and QMainWindow on Show."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if event.type() == QtCore.QEvent.Type.Show:
            if isinstance(obj, (QtWidgets.QMessageBox, QtWidgets.QFileDialog)):
                return super().eventFilter(obj, event)
            if isinstance(obj, (QtWidgets.QDialog, QtWidgets.QMainWindow)) and obj.isWindow():
                if not obj.property("_ui_safeguard_active"):
                    obj.setProperty("_ui_safeguard_active", True)
                    try:
                        clamp_window_to_screen(obj)
                    finally:
                        obj.setProperty("_ui_safeguard_active", False)
        return super().eventFilter(obj, event)


def install_low_resolution_safeguards(app: QtWidgets.QApplication) -> LowResolutionSafeguardFilter:
    """Install the global low-resolution safeguard event filter on `app`."""
    guard = LowResolutionSafeguardFilter(app)
    app.installEventFilter(guard)
    # Keep a reference on the app object so it is never garbage-collected.
    app._low_res_safeguard_filter = guard  # type: ignore[attr-defined]
    return guard
