"""
Extensible Python Scripting & Plugin Runner for AgyMolden.

Provides researchers with a scripting hook to automate molecular manipulation,
custom property calculations, geometry transformations, and custom file exporters.
"""

import os
import sys
import glob
import logging
import traceback
from typing import Dict, Any, Optional, List
import numpy as np

from PyQt6 import QtWidgets, QtGui, QtCore
from .dialog_service import dialog_service
from .plugin_api import PluginBase, PluginMetadata, PluginContext, PluginManager

log = logging.getLogger(__name__)


def get_user_plugin_dir() -> str:
    """Return user plugin directory (~/.agymolden/plugins/)."""
    user_home = os.path.expanduser("~")
    pdir = os.path.join(user_home, ".agymolden", "plugins")
    os.makedirs(pdir, exist_ok=True)
    return pdir


def discover_user_plugins() -> List[str]:
    """Find all .py plugin scripts in ~/.agymolden/plugins/."""
    pdir = get_user_plugin_dir()
    return sorted(glob.glob(os.path.join(pdir, "*.py")))


def execute_plugin_script(script_text: str, window: Any, filename: str = "<plugin>") -> Dict[str, Any]:
    """
    Execute a user Python script with full access to the AgyMolden API and context.
    
    Exposed namespace:
    - window: MainWindow instance
    - zatoms: current ZAtoms list
    - coords: numpy array of Cartesian coordinates
    - symbols: list of element symbols
    - viewer: MoleculeViewer instance
    - dialogs: DialogService singleton
    - np: numpy module
    - set_coords(new_coords): update Cartesian coordinates with undo history
    - set_molecule(symbols, coords): update molecule with undo history
    """
    symbols = [a.symbol for a in getattr(window, "zatoms", [])]
    coords = getattr(window, "coords", np.zeros((0, 3)))

    def set_coords(new_coords: np.ndarray):
        arr = np.array(new_coords, dtype=float)
        if arr.shape != coords.shape:
            raise ValueError(f"Shape mismatch: expected {coords.shape}, got {arr.shape}")
        if hasattr(window, "_push_undo_state"):
            window._push_undo_state()
        window.coords = arr
        if hasattr(window, "viewer") and window.viewer:
            if hasattr(window.viewer, "update_coords"):
                user_bonds = window._current_explicit_bonds() if hasattr(window, "_current_explicit_bonds") else None
                window.viewer.update_coords(arr, user_bonds=user_bonds)
            elif hasattr(window.viewer, "set_coords"):
                window.viewer.set_coords(arr)
        if hasattr(window, "table_widget") and window.table_widget:
            window.table_widget.sync_coords_to_table(arr)
        if hasattr(window, "_sync_raw_text"):
            window._sync_raw_text()

    def set_molecule(new_symbols: List[str], new_coords: np.ndarray):
        arr = np.array(new_coords, dtype=float)
        if hasattr(window, "_push_undo_state"):
            window._push_undo_state()
        from .zmatrix import cartesian_to_zmatrix
        zatoms = cartesian_to_zmatrix(new_symbols, arr)
        window.zatoms = zatoms
        window.coords = arr
        if hasattr(window, "table_widget") and window.table_widget:
            window.table_widget.load_zatoms(zatoms)
        if hasattr(window, "viewer") and window.viewer:
            window.viewer.set_molecule(list(new_symbols), arr)
        if hasattr(window, "_sync_raw_text"):
            window._sync_raw_text()

    script_globals = {
        "__name__": "__main__",
        "__file__": filename,
        "window": window,
        "zatoms": getattr(window, "zatoms", []),
        "symbols": symbols,
        "coords": np.copy(coords),
        "viewer": getattr(window, "viewer", None),
        "dialogs": dialog_service,
        "np": np,
        "set_coords": set_coords,
        "set_molecule": set_molecule,
    }

    from .plugin_api import verify_plugin_trust
    if filename not in ("<plugin>", "<inline>") and not verify_plugin_trust(filename, script_text=script_text, parent=window):
        raise PermissionError(f"Execution of untrusted plugin script cancelled: {filename}")

    compiled = compile(script_text, filename, "exec")
    exec(compiled, script_globals)
    return script_globals


class PythonScriptRunnerDialog(QtWidgets.QDialog):
    """Interactive dialog for executing custom Python scripts and plugins inside AgyMolden."""

    def __init__(self, window: Any = None, parent: Optional[QtWidgets.QWidget] = None):
        qparent = parent if parent is not None else (window if isinstance(window, QtWidgets.QWidget) else None)
        super().__init__(qparent)
        self.window = window
        self.plugin_manager = PluginManager(window or qparent)
        self.plugin_manager.discover_and_load_all()
        self.setWindowTitle("🐍 Python Script & Plugin Runner")
        self.resize(680, 520)
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Header info
        lbl_info = QtWidgets.QLabel(
            "Execute custom Python scripts and formal plugins to manipulate geometries, inspect properties, or automate workflows.<br>"
            "Available variables: <code>window</code>, <code>symbols</code>, <code>coords</code>, <code>set_coords(new_coords)</code>, <code>set_molecule(syms, coords)</code>, <code>np</code>, <code>dialogs</code>."
        )
        lbl_info.setWordWrap(True)
        lbl_info.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(lbl_info)

        # Plugin preset selector
        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addWidget(QtWidgets.QLabel("Plugins:"))
        self.combo_plugins = QtWidgets.QComboBox()
        self.combo_plugins.addItem("(Select a discovered or built-in plugin...)")

        # Add registered PluginBase plugins first
        for name, p_obj in self.plugin_manager.plugins.items():
            meta = getattr(p_obj, "metadata", None)
            desc = meta.description if meta else ""
            label = f"✨ {name}"
            mod_file = getattr(sys.modules.get(p_obj.__class__.__module__), "__file__", "")
            self.combo_plugins.addItem(label, mod_file or name)

        # Add any raw user script files
        user_plugins = discover_user_plugins()
        for p in user_plugins:
            base = os.path.basename(p)
            if not any(self.combo_plugins.itemText(i).endswith(base) for i in range(self.combo_plugins.count())):
                self.combo_plugins.addItem(f"📄 {base}", p)

        self.combo_plugins.currentIndexChanged.connect(self._on_plugin_selected)
        top_bar.addWidget(self.combo_plugins, 1)

        btn_browse = QtWidgets.QPushButton("📁 Open Script...")
        btn_browse.clicked.connect(self._on_browse_file)
        top_bar.addWidget(btn_browse)
        layout.addLayout(top_bar)

        # Editor
        self.editor = QtWidgets.QPlainTextEdit()
        self.editor.setPlaceholderText(
            "# Example:\n"
            "# Translate entire molecule by +1.0 Angstrom along Z\n"
            "new_coords = np.copy(coords)\n"
            "new_coords[:, 2] += 1.0\n"
            "set_coords(new_coords)\n"
            "dialogs.info(window, 'Done', 'Molecule translated by +1.0 A along Z.')\n"
        )
        font = QtGui.QFont("Monospace", 10)
        font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.editor.setFont(font)
        self.editor.setStyleSheet("background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; padding: 6px;")
        layout.addWidget(self.editor, 1)

        # Bottom buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_save = QtWidgets.QPushButton("💾 Save to Plugins...")
        btn_save.clicked.connect(self._on_save_plugin)
        btn_box.addWidget(btn_save)

        btn_box.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        btn_box.addWidget(btn_close)

        btn_run = QtWidgets.QPushButton("▶ Run Script")
        btn_run.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px;")
        btn_run.clicked.connect(self._on_run_script)
        btn_box.addWidget(btn_run)

        layout.addLayout(btn_box)

    def _on_plugin_selected(self, idx: int):
        if idx <= 0:
            return
        path = self.combo_plugins.itemData(idx)
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Read Error", f"Unable to read plugin: {e}")

    def _on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Python Script", "", "Python Files (*.py);;All Files (*)"
        )
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.editor.setPlainText(f.read())
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Read Error", f"Unable to read script: {e}")

    def _on_save_plugin(self):
        text = self.editor.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "Empty Script", "Cannot save an empty script.")
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Plugin", "Plugin filename (e.g. center_mass.py):"
        )
        if ok and name:
            if not name.endswith(".py"):
                name += ".py"
            pdir = get_user_plugin_dir()
            target = os.path.join(pdir, name)
            try:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(text)
                QtWidgets.QMessageBox.information(self, "Plugin Saved", f"Plugin saved to:\n{target}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save Error", f"Failed to save plugin:\n{e}")

    def _on_run_script(self):
        text = self.editor.toPlainText().strip()
        if not text:
            QtWidgets.QMessageBox.warning(self, "Empty Script", "Please enter or load Python script code to run.")
            return
        try:
            execute_plugin_script(text, self.window or self.parent(), filename="<user_script>")
        except Exception as e:
            tb = traceback.format_exc()
            log.exception("Error executing user script")
            QtWidgets.QMessageBox.critical(
                self, "Script Execution Error", f"An error occurred while executing the script:\n\n{e}\n\nTraceback:\n{tb}"
            )
