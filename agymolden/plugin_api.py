"""
Formal Plugin API v1 for AgyMolden.

Provides a structured, typed plugin lifecycle with hooks for calculation analysis,
geometry modification, menu action registration, and event callbacks.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import importlib.util
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Metadata describing an AgyMolden plugin."""
    name: str
    version: str = "1.0.0"
    author: str = "Community"
    description: str = ""
    menu_label: Optional[str] = None
    shortcut: Optional[str] = None
    category: str = "General"
    api_version: str = "1.1.0"


def register_plugin(
    name: str,
    version: str = "1.0.0",
    author: str = "Community",
    description: str = "",
    menu_label: Optional[str] = None,
    shortcut: Optional[str] = None,
    category: str = "General",
) -> Callable[[Any], Any]:
    """Decorator to declare and register an AgyMolden plugin class with metadata."""
    def decorator(cls: Any) -> Any:
        cls.metadata = PluginMetadata(
            name=name,
            version=version,
            author=author,
            description=description,
            menu_label=menu_label,
            shortcut=shortcut,
            category=category,
        )
        return cls
    return decorator


class PluginContext:
    """Safe facade providing plugins access to molecule state and UI interactions."""

    def __init__(self, window: Any):
        self._window = window

    @property
    def window(self) -> Any:
        return self._window

    @property
    def symbols(self) -> List[str]:
        return [a.symbol for a in getattr(self._window, "zatoms", [])]

    @property
    def coords(self) -> np.ndarray:
        c = getattr(self._window, "coords", None)
        return np.copy(c) if c is not None and len(c) > 0 else np.zeros((0, 3))

    @property
    def zatoms(self) -> List[Any]:
        return getattr(self._window, "zatoms", [])

    def get_bonds(self) -> List[Tuple[int, int]]:
        if hasattr(self._window, "_current_explicit_bonds"):
            return self._window._current_explicit_bonds()
        return getattr(self._window, "bonds", [])

    def set_coords(self, new_coords: np.ndarray) -> None:
        """Update Cartesian coordinates with undo history."""
        arr = np.array(new_coords, dtype=float)
        curr = getattr(self._window, "coords", None)
        if curr is not None and arr.shape != curr.shape:
            raise ValueError(f"Shape mismatch: expected {curr.shape}, got {arr.shape}")
        if hasattr(self._window, "_push_undo_state"):
            self._window._push_undo_state()
        self._window.coords = arr
        if hasattr(self._window, "viewer") and self._window.viewer:
            if hasattr(self._window.viewer, "update_coords"):
                user_bonds = self._window._current_explicit_bonds() if hasattr(self._window, "_current_explicit_bonds") else None
                self._window.viewer.update_coords(arr, user_bonds=user_bonds)
            elif hasattr(self._window.viewer, "set_coords"):
                self._window.viewer.set_coords(arr)
        if hasattr(self._window, "table_widget") and self._window.table_widget:
            if hasattr(self._window.table_widget, "sync_coords_to_table"):
                self._window.table_widget.sync_coords_to_table(arr)
        if hasattr(self._window, "_sync_raw_text"):
            self._window._sync_raw_text()

    def set_molecule(self, symbols: List[str], coords: np.ndarray) -> None:
        """Replace the active molecule with new symbols and coordinates with undo history."""
        arr = np.array(coords, dtype=float)
        if hasattr(self._window, "_push_undo_state"):
            self._window._push_undo_state()
        from .zmatrix import cartesian_to_zmatrix
        zatoms = cartesian_to_zmatrix(symbols, arr)
        self._window.zatoms = zatoms
        self._window.coords = arr
        if hasattr(self._window, "table_widget") and self._window.table_widget:
            if hasattr(self._window.table_widget, "load_zatoms"):
                self._window.table_widget.load_zatoms(zatoms)
        if hasattr(self._window, "viewer") and self._window.viewer:
            self._window.viewer.set_molecule(list(symbols), arr)
        if hasattr(self._window, "_sync_raw_text"):
            self._window._sync_raw_text()

    def show_message(self, title: str, text: str) -> None:
        """Display an informational popup message dialog."""
        import os
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AGYMOLDEN_HEADLESS"):
            log.info("[%s] %s", title, text)
            return

        if hasattr(self._window, "dialogs"):
            if hasattr(self._window.dialogs, "info"):
                self._window.dialogs.info(self._window, title, text)
                return
            if hasattr(self._window.dialogs, "show_info"):
                self._window.dialogs.show_info(self._window, title, text)
                return
        if True:
            try:
                from PyQt6 import QtWidgets
                app = QtWidgets.QApplication.instance()
                if app is not None:
                    QtWidgets.QMessageBox.information(
                        self._window if isinstance(self._window, QtWidgets.QWidget) else None,
                        title, text
                    )
                else:
                    log.info("[%s] %s", title, text)
            except Exception:
                log.info("[%s] %s", title, text)

    def status_message(self, text: str, timeout_ms: int = 4000) -> None:
        """Display a temporary message in the status bar."""
        if hasattr(self._window, "statusBar") and self._window.statusBar():
            self._window.statusBar().showMessage(text, timeout_ms)
        else:
            log.info("Status: %s", text)


class PluginBase:
    """Base class for all AgyMolden plugins."""

    metadata: PluginMetadata = PluginMetadata(name="Unnamed Plugin")

    def __init__(self):
        self._enabled = True

    def on_load(self, context: PluginContext) -> None:
        """Called when the plugin is initially discovered and loaded into AgyMolden."""
        pass

    def run(self, context: PluginContext) -> Any:
        """Entry point executed when the user activates the plugin from the menu or dialog."""
        raise NotImplementedError("Plugins must implement the run() method.")

    def on_geometry_change(self, context: PluginContext, symbols: List[str], coords: np.ndarray) -> None:
        """Optional hook called when the molecule geometry is edited or refreshed."""
        pass

    def on_calculation_completed(self, context: PluginContext, result: Dict[str, Any]) -> None:
        """Optional hook called when a QM or semiempirical calculation terminates."""
        pass

    def on_file_loaded(self, context: PluginContext, filepath: str) -> None:
        """Optional hook called when a new molecule file or trajectory is opened."""
        pass

    def on_unload(self, context: PluginContext) -> None:
        """Called when the plugin is deactivated or AgyMolden is closing."""
        pass


def verify_plugin_trust(
    filepath: str,
    script_text: Optional[str] = None,
    parent: Any = None,
) -> bool:
    """Verify user trust before executing an external plugin/script.

    Built-in plugins under ``agymolden/builtin_plugins`` are always trusted.
    External scripts are hashed (SHA-256) and checked against QSettings
    ``TrustedPlugins/trusted_hashes``. In headless/test environments, execution
    is allowed automatically unless ``AGYMOLDEN_ENFORCE_PLUGIN_TRUST=1``.
    """
    enforce = os.environ.get("AGYMOLDEN_ENFORCE_PLUGIN_TRUST") == "1"
    if not enforce and (
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("AGYMOLDEN_TRUST_ALL_PLUGINS") == "1"
    ):
        return True

    builtin_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "builtin_plugins"))
    if filepath and os.path.isfile(filepath):
        abs_fp = os.path.normpath(os.path.abspath(filepath))
        if abs_fp.startswith(builtin_dir + os.sep):
            return True
        if script_text is None:
            with open(abs_fp, "r", encoding="utf-8", errors="replace") as f:
                script_text = f.read()
    elif script_text is None:
        return False

    digest = hashlib.sha256(script_text.encode("utf-8", errors="replace")).hexdigest()
    try:
        from .session_state import get_settings
        settings = get_settings("TrustedPlugins")
        raw = settings.value("trusted_hashes", "[]")
        trusted = set(json.loads(raw) if isinstance(raw, str) else (raw or []))
    except Exception:
        settings = None
        trusted = set()

    if digest in trusted:
        return True

    try:
        from PyQt6 import QtWidgets
        app = QtWidgets.QApplication.instance()
        if app is None or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            if enforce:
                return False
            return True
        qparent = parent if isinstance(parent, QtWidgets.QWidget) else None
        label = filepath or "<inline script>"
        reply = QtWidgets.QMessageBox.question(
            qparent,
            "Trust External Plugin?",
            f"AgyMolden is about to execute an external Python plugin/script:\n\n"
            f"Path: {label}\nSHA-256: {digest[:16]}...\n\n"
            "Only run scripts from sources you trust. Trust and remember this script?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            trusted.add(digest)
            if settings is not None:
                settings.setValue("trusted_hashes", json.dumps(sorted(trusted)))
            return True
        return False
    except Exception:
        log.debug("Plugin trust prompt fallback", exc_info=True)
        return not enforce


class PluginManager:
    """Singleton managing discovery, lifecycle, and invocation of AgyMolden plugins."""

    def __init__(self, window: Any = None):
        self.window = window
        self.context = PluginContext(window) if window else None
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_dirs: List[str] = [
            os.path.join(os.path.expanduser("~"), ".agymolden", "plugins"),
            os.path.join(os.path.dirname(__file__), "builtin_plugins"),
        ]

    def set_window(self, window: Any) -> None:
        self.window = window
        self.context = PluginContext(window)

    def register_plugin(self, plugin: PluginBase) -> None:
        """Register an instantiated plugin object."""
        name = plugin.metadata.name
        self.plugins[name] = plugin
        if self.context:
            try:
                plugin.on_load(self.context)
            except Exception as e:
                log.error("Error loading plugin '%s': %s", name, e, exc_info=True)

    def load_plugin_from_file(self, filepath: str) -> Optional[PluginBase]:
        """Dynamically load and instantiate a PluginBase subclass from a python file."""
        if not os.path.isfile(filepath) or not filepath.endswith(".py"):
            return None
        if not verify_plugin_trust(filepath, parent=self.window):
            log.warning("Skipped untrusted plugin: %s", filepath)
            return None
        mod_name = f"agymolden_plugin_{os.path.splitext(os.path.basename(filepath))[0]}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, filepath)
            if not spec or not spec.loader:
                return None
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # Find PluginBase subclass or legacy hook
            for attr_name in dir(mod):
                attr = getattr(mod, attr_name)
                if isinstance(attr, type) and issubclass(attr, PluginBase) and attr is not PluginBase:
                    plugin_instance = attr()
                    self.register_plugin(plugin_instance)
                    return plugin_instance

            # Fallback for simple legacy script with a metadata dict or run() function
            if hasattr(mod, "run"):
                name = getattr(mod, "PLUGIN_NAME", os.path.splitext(os.path.basename(filepath))[0])
                desc = getattr(mod, "PLUGIN_DESC", getattr(mod, "__doc__", "") or "")
                meta = PluginMetadata(name=name, description=desc.strip())

                class SimpleScriptPlugin(PluginBase):
                    metadata = meta
                    def run(self, ctx):
                        return mod.run(ctx)

                plugin_inst = SimpleScriptPlugin()
                self.register_plugin(plugin_inst)
                return plugin_inst

        except Exception as e:
            log.error("Failed to load plugin from %s: %s", filepath, e, exc_info=True)
            return None
        return None

    def discover_and_load_all(self) -> Dict[str, PluginBase]:
        """Search all plugin directories and load valid plugins."""
        for pdir in self.plugin_dirs:
            if not os.path.isdir(pdir):
                continue
            for py_path in sorted(glob.glob(os.path.join(pdir, "*.py"))):
                if os.path.basename(py_path).startswith("__"):
                    continue
                self.load_plugin_from_file(py_path)
        return self.plugins

    def execute_plugin(self, name: str) -> Any:
        """Run a plugin by name using the current context."""
        plugin = self.plugins.get(name)
        if not plugin:
            raise KeyError(f"Plugin '{name}' not found.")
        if not self.context:
            raise RuntimeError("PluginManager has no active context / window.")
        return plugin.run(self.context)

    def notify_geometry_changed(self, symbols: List[str], coords: np.ndarray) -> None:
        """Dispatch geometry change events to all active plugins."""
        if not self.context:
            return
        for name, plugin in self.plugins.items():
            try:
                plugin.on_geometry_change(self.context, symbols, coords)
            except Exception as e:
                log.warning("Plugin '%s' error in on_geometry_change: %s", name, e)

    def notify_calculation_completed(self, result: Dict[str, Any]) -> None:
        """Dispatch calculation completion events to all active plugins."""
        if not self.context:
            return
        for name, plugin in self.plugins.items():
            try:
                plugin.on_calculation_completed(self.context, result)
            except Exception as e:
                log.warning("Plugin '%s' error in on_calculation_completed: %s", name, e)

    def notify_file_loaded(self, filepath: str) -> None:
        """Dispatch file loaded events to all active plugins."""
        if not self.context:
            return
        for name, plugin in self.plugins.items():
            try:
                plugin.on_file_loaded(self.context, filepath)
            except Exception as e:
                log.warning("Plugin '%s' error in on_file_loaded: %s", name, e)
