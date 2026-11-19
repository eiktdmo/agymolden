"""Session-level state helpers for AgyMolden (e.g., remembering last working folder)."""

from __future__ import annotations

import os
from typing import Optional

_last_dir: str = ""
_installed: bool = False


def get_settings(app: str = "AgyMolden"):
    """Return a QSettings instance under the unified 'AgyMolden' organization."""
    from PyQt6 import QtCore
    return QtCore.QSettings("AgyMolden", app)


def get_last_dir(fallback: str = "") -> str:
    """Return the last working directory used in this session, or *fallback*."""
    if _last_dir and os.path.isdir(_last_dir):
        return _last_dir
    return fallback


def set_last_dir(path: Optional[str]) -> None:
    """Update the session's last working directory from a file or directory path."""
    global _last_dir
    if not path:
        return
    p = os.path.abspath(path)
    if os.path.isdir(p):
        _last_dir = p
    else:
        parent = os.path.dirname(p)
        if parent and os.path.isdir(parent):
            _last_dir = parent


def _resolve_initial_dir(directory: str) -> str:
    """If session *last_dir* is set, override stale initial directories while keeping any default filename."""
    last = get_last_dir("")
    if not last:
        return directory
    if not directory:
        return last
    # If caller passed an existing directory path (e.g. self._last_dir() or os.getcwd()), prefer session last_dir
    if os.path.isdir(directory):
        return last
    # If caller passed a default filename (with or without a directory prefix, e.g. "calc.inp" or "/old/calc.inp")
    base = os.path.basename(directory)
    if base:
        return os.path.join(last, base)
    return last


def install_file_dialog_session_memory() -> None:
    """Wrap QFileDialog static helpers so all dialogs remember the last folder in-session."""
    global _installed
    if _installed:
        return
    from PyQt6 import QtWidgets

    orig_open = QtWidgets.QFileDialog.getOpenFileName
    orig_open_multi = QtWidgets.QFileDialog.getOpenFileNames
    orig_save = QtWidgets.QFileDialog.getSaveFileName
    orig_dir = QtWidgets.QFileDialog.getExistingDirectory

    def _get_open_file_name(parent=None, caption="", directory="", filter="", initialFilter="", options=QtWidgets.QFileDialog.Option(0)):
        res = orig_open(parent, caption, _resolve_initial_dir(directory), filter, initialFilter, options)
        if res and res[0]:
            set_last_dir(res[0])
        return res

    def _get_open_file_names(parent=None, caption="", directory="", filter="", initialFilter="", options=QtWidgets.QFileDialog.Option(0)):
        res = orig_open_multi(parent, caption, _resolve_initial_dir(directory), filter, initialFilter, options)
        if res and res[0]:
            set_last_dir(res[0][0])
        return res

    def _get_save_file_name(parent=None, caption="", directory="", filter="", initialFilter="", options=QtWidgets.QFileDialog.Option(0)):
        res = orig_save(parent, caption, _resolve_initial_dir(directory), filter, initialFilter, options)
        if res and res[0]:
            set_last_dir(res[0])
        return res

    def _get_existing_directory(parent=None, caption="", directory="", options=QtWidgets.QFileDialog.Option.ShowDirsOnly):
        res = orig_dir(parent, caption, _resolve_initial_dir(directory), options)
        if res:
            set_last_dir(res)
        return res

    QtWidgets.QFileDialog.getOpenFileName = staticmethod(_get_open_file_name)
    QtWidgets.QFileDialog.getOpenFileNames = staticmethod(_get_open_file_names)
    QtWidgets.QFileDialog.getSaveFileName = staticmethod(_get_save_file_name)
    QtWidgets.QFileDialog.getExistingDirectory = staticmethod(_get_existing_directory)
    _installed = True
