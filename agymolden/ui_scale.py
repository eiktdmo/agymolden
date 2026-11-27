"""Built-in UI scaling and live font-size management for AgyMolden.

Provides two complementary readability controls for small/high-DPI screens:
1. Live font size (pt): updates QApplication.font() and scales hardcoded
   `font-size: Npx` / `Npt` rules across all widget stylesheets immediately
   without restarting, re-baselining cleanly when themes change.
2. Startup UI scale factor: persisted via QSettings and applied via
   `QT_SCALE_FACTOR` before QApplication construction (or via `--scale`).
"""

from __future__ import annotations

import os
import re
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

DEFAULT_FONT_PT = 10
MIN_FONT_PT = 8
MAX_FONT_PT = 22

DEFAULT_UI_SCALE = 1.0
MIN_UI_SCALE = 0.75
MAX_UI_SCALE = 2.5

UI_SCALE_PRESETS = (1.0, 1.15, 1.25, 1.4, 1.5, 1.75)
FONT_SIZE_PRESETS = (10, 12, 14, 16, 18)

_FONT_SIZE_RE = re.compile(r"(font-size\s*:\s*)(\d+(?:\.\d+)?)(px|pt)", re.IGNORECASE)
_current_font_pt: int = DEFAULT_FONT_PT


def _clamp_font_pt(pt: int) -> int:
    return max(MIN_FONT_PT, min(MAX_FONT_PT, int(round(pt))))


def _clamp_ui_scale(scale: float) -> float:
    return max(MIN_UI_SCALE, min(MAX_UI_SCALE, float(scale)))


def get_saved_font_size() -> int:
    """Return the persisted base font size in points, clamped to [MIN_FONT_PT, MAX_FONT_PT]."""
    from .session_state import get_settings

    settings = get_settings("AgyMolden")
    raw = settings.value("ui/font_size_pt", DEFAULT_FONT_PT)
    try:
        return _clamp_font_pt(int(raw))
    except (TypeError, ValueError):
        return DEFAULT_FONT_PT


def save_font_size(pt: int) -> int:
    """Persist the base font size in points and return the clamped value."""
    from .session_state import get_settings

    clamped = _clamp_font_pt(pt)
    settings = get_settings("AgyMolden")
    settings.setValue("ui/font_size_pt", clamped)
    settings.sync()
    return clamped


def get_saved_ui_scale() -> float:
    """Return the persisted startup UI scale factor, clamped to [MIN_UI_SCALE, MAX_UI_SCALE]."""
    from .session_state import get_settings

    settings = get_settings("AgyMolden")
    raw = settings.value("ui/scale_factor", DEFAULT_UI_SCALE)
    try:
        return _clamp_ui_scale(float(raw))
    except (TypeError, ValueError):
        return DEFAULT_UI_SCALE


def save_ui_scale(scale: float) -> float:
    """Persist the startup UI scale factor and return the clamped value."""
    from .session_state import get_settings

    clamped = _clamp_ui_scale(scale)
    settings = get_settings("AgyMolden")
    settings.setValue("ui/scale_factor", round(clamped, 2))
    settings.sync()
    return clamped


def apply_startup_scale_env(cli_scale: Optional[float] = None) -> float:
    """Apply QT_SCALE_FACTOR before QApplication is constructed.

    Priority: explicit CLI `--scale` > existing `QT_SCALE_FACTOR` env var >
    saved `ui/scale_factor` in QSettings.
    """
    if cli_scale is not None:
        scale = _clamp_ui_scale(cli_scale)
        os.environ["QT_SCALE_FACTOR"] = f"{scale:.2f}"
        return scale

    existing = os.environ.get("QT_SCALE_FACTOR", "").strip()
    if existing:
        try:
            return _clamp_ui_scale(float(existing))
        except ValueError:
            return DEFAULT_UI_SCALE

    scale = get_saved_ui_scale()
    if abs(scale - 1.0) > 1e-3:
        os.environ["QT_SCALE_FACTOR"] = f"{scale:.2f}"
    return scale


def _scale_stylesheet_fonts(qss: str, ratio: float) -> str:
    """Scale numeric `font-size: Npx|Npt` rules in *qss* by *ratio* (minimum 7)."""
    if not qss or abs(ratio - 1.0) < 1e-3:
        return qss

    def _repl(match: re.Match) -> str:
        prefix, num_str, unit = match.groups()
        scaled = max(7, int(round(float(num_str) * ratio)))
        return f"{prefix}{scaled}{unit}"

    return _FONT_SIZE_RE.sub(_repl, qss)


def _apply_qss_scale(obj: QtCore.QObject, ratio: float) -> None:
    """Scale widget/application stylesheet fonts without compound-scaling on repeated calls."""
    if not hasattr(obj, "styleSheet") or not hasattr(obj, "setStyleSheet"):
        return
    if obj.property("_agymolden_qss_scaling"):
        return

    cur_qss = obj.styleSheet() or ""
    if not cur_qss:
        if obj.property("_agymolden_base_qss") is not None:
            obj.setProperty("_agymolden_base_qss", None)
            obj.setProperty("_agymolden_scaled_qss", None)
        return

    if _FONT_SIZE_RE.search(cur_qss) is None:
        return

    last_scaled = obj.property("_agymolden_scaled_qss")
    if last_scaled is None or cur_qss != last_scaled:
        base_qss = cur_qss
        obj.setProperty("_agymolden_base_qss", base_qss)
    else:
        base_qss = obj.property("_agymolden_base_qss") or cur_qss

    new_qss = _scale_stylesheet_fonts(base_qss, ratio)
    obj.setProperty("_agymolden_scaled_qss", new_qss)
    if new_qss != cur_qss:
        obj.setProperty("_agymolden_qss_scaling", True)
        try:
            obj.setStyleSheet(new_qss)
        finally:
            obj.setProperty("_agymolden_qss_scaling", False)


class FontScaleEventFilter(QtCore.QObject):
    """Application-wide filter that scales stylesheets on newly polished/restyled widgets."""

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        etype = event.type()
        if etype in (QtCore.QEvent.Type.Polish, QtCore.QEvent.Type.StyleChange):
            if isinstance(obj, QtWidgets.QWidget):
                if _current_font_pt != DEFAULT_FONT_PT or obj.property("_agymolden_base_qss") is not None:
                    _apply_qss_scale(obj, _current_font_pt / float(DEFAULT_FONT_PT))
        return super().eventFilter(obj, event)


def install_ui_scale_filter(app: QtWidgets.QApplication) -> FontScaleEventFilter:
    """Install the global FontScaleEventFilter on *app* (idempotent)."""
    existing = getattr(app, "_ui_scale_filter", None)
    if isinstance(existing, FontScaleEventFilter):
        return existing
    filt = FontScaleEventFilter(app)
    app.installEventFilter(filt)
    app._ui_scale_filter = filt
    return filt


def get_current_font_size() -> int:
    """Return the active live UI font size in points."""
    return _current_font_pt


def apply_live_font_size(
    pt: int,
    app: Optional[QtWidgets.QApplication] = None,
    persist: bool = True,
) -> int:
    """Apply *pt* live to the entire application and all widget stylesheets."""
    global _current_font_pt
    clamped = _clamp_font_pt(pt)
    _current_font_pt = clamped

    if persist:
        save_font_size(clamped)

    target_app = app or QtWidgets.QApplication.instance()
    if target_app is None:
        return clamped

    install_ui_scale_filter(target_app)

    font = QtGui.QFont(target_app.font())
    font.setPointSize(clamped)
    target_app.setFont(font)

    ratio = clamped / float(DEFAULT_FONT_PT)
    _apply_qss_scale(target_app, ratio)
    for widget in list(target_app.allWidgets()):
        _apply_qss_scale(widget, ratio)

    from .ui_safeguards import clamp_window_to_screen

    for top in list(target_app.topLevelWidgets()):
        if isinstance(top, (QtWidgets.QMainWindow, QtWidgets.QDialog)) and top.isVisible():
            top.updateGeometry()
            clamp_window_to_screen(top)

    return clamped


def step_live_font_size(
    delta: int,
    app: Optional[QtWidgets.QApplication] = None,
    persist: bool = True,
) -> int:
    """Adjust the live UI font size by *delta* points."""
    return apply_live_font_size(_current_font_pt + delta, app=app, persist=persist)
