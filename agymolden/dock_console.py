"""
Interactive Embedded Python Scripting Console for AgyMolden.

Provides a dockable, live Python REPL exposing high-level chemical manipulation
APIs (`agy.*`) directly connected to the 3D viewport, undo stack, and event bus.
"""

from typing import Optional, List, Any, Dict
import html
import io
import sys
import traceback
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


class AgyScriptingAPI:
    """Safe, high-level Python API exposed to user scripts in the embedded console."""

    def __init__(self, main_window: Optional[QtWidgets.QMainWindow] = None):
        self._mw = main_window

    @property
    def symbols(self) -> List[str]:
        """Return list of atomic element symbols in the active molecule."""
        if self._mw and hasattr(self._mw, "zatoms"):
            return [a.symbol for a in self._mw.zatoms]
        return []

    @property
    def coords(self) -> np.ndarray:
        """Return Nx3 numpy array of Cartesian coordinates (Å)."""
        if self._mw and hasattr(self._mw, "coords") and self._mw.coords is not None:
            return np.copy(self._mw.coords)
        return np.empty((0, 3), dtype=np.float64)

    def set_coords(self, new_coords: np.ndarray):
        """Update Cartesian coordinates and refresh 3D OpenGL viewport."""
        if self._mw is None:
            return
        arr = np.asarray(new_coords, dtype=np.float64)
        if len(arr) != len(self.symbols):
            raise ValueError(f"Coordinate count {len(arr)} does not match atom count {len(self.symbols)}")
        if hasattr(self._mw, "_push_undo_state"):
            self._mw._push_undo_state() # type: ignore
        setattr(self._mw, "coords", arr)
        if hasattr(self._mw, "viewer") and getattr(self._mw, "viewer", None):
            self._mw.viewer.update_coords(arr)
        if hasattr(self._mw, "event_bus") and self._mw.event_bus:
            self._mw.event_bus.geometry_changed.emit(arr)

    def center(self):
        """Translate the molecule centroid to Cartesian origin (0, 0, 0)."""
        c = self.coords
        if len(c) > 0:
            centroid = np.mean(c, axis=0)
            self.set_coords(c - centroid)

    def translate(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """Translate all atoms by Cartesian vector [dx, dy, dz] in Ångströms."""
        c = self.coords
        if len(c) > 0:
            shift = np.array([dx, dy, dz], dtype=np.float64)
            self.set_coords(c + shift)

    def rotate(self, angle_degrees: float, axis: str = "z"):
        """Rotate molecule around Cartesian axis ('x', 'y', or 'z') in degrees."""
        c = self.coords
        if len(c) == 0:
            return
        rad = np.radians(angle_degrees)
        ax = axis.lower()
        if ax == "x":
            R = np.array([
                [1.0, 0.0, 0.0],
                [0.0, np.cos(rad), -np.sin(rad)],
                [0.0, np.sin(rad), np.cos(rad)]
            ])
        elif ax == "y":
            R = np.array([
                [np.cos(rad), 0.0, np.sin(rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(rad), 0.0, np.cos(rad)]
            ])
        else: # z
            R = np.array([
                [np.cos(rad), -np.sin(rad), 0.0],
                [np.sin(rad), np.cos(rad), 0.0],
                [0.0, 0.0, 1.0]
            ])
        centroid = np.mean(c, axis=0)
        c_centered = c - centroid
        c_rot = np.dot(c_centered, R.T) + centroid
        self.set_coords(c_rot)

    def distance(self, i: int, j: int) -> float:
        """Calculate interatomic distance (Å) between atom i and j (0-based)."""
        c = self.coords
        return float(np.linalg.norm(c[i] - c[j]))

    @property
    def event_bus(self):
        """Access the application-wide event bus singleton."""
        from .events import get_event_bus
        return get_event_bus()

    def __repr__(self) -> str:
        n = len(self.symbols)
        return f"<AgyScriptingAPI: {n} atoms ({''.join(self.symbols[:8])}{'...' if n > 8 else ''})>"


class PythonConsoleWidget(QtWidgets.QWidget):
    """Interactive Python REPL console with command history and rich syntax styling."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, main_window: Optional[QtWidgets.QMainWindow] = None):
        super().__init__(parent)
        self.mw = main_window
        self.history: List[str] = []
        self.history_idx: int = 0
        self.api = AgyScriptingAPI(main_window)
        self.namespace: Dict[str, Any] = {
            "agy": self.api,
            "np": np,
            "print": self._custom_print,
        }
        self._init_ui()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Output text area
        self.txt_output = QtWidgets.QTextEdit()
        self.txt_output.setReadOnly(True)
        self.txt_output.setFont(QtGui.QFont("Monospace", 9))
        self.txt_output.setStyleSheet(
            "QTextEdit { background-color: #0f172a; color: #f8fafc; border: 1px solid #334155; border-radius: 4px; }"
        )
        layout.addWidget(self.txt_output, stretch=1)

        # Input line layout with prompt
        in_layout = QtWidgets.QHBoxLayout()
        in_layout.setSpacing(4)

        lbl_prompt = QtWidgets.QLabel(">>>")
        lbl_prompt.setStyleSheet("font-weight: bold; color: #38bdf8; font-family: monospace;")
        in_layout.addWidget(lbl_prompt)

        self.txt_input = QtWidgets.QLineEdit()
        self.txt_input.setFont(QtGui.QFont("Monospace", 9))
        self.txt_input.setPlaceholderText("Enter Python command (e.g. agy.rotate(45, 'z'), agy.center(), print(agy.coords))...")
        self.txt_input.setStyleSheet(
            "QLineEdit { background-color: #1e293b; color: #f8fafc; border: 1px solid #475569; border-radius: 4px; padding: 3px 6px; }"
        )
        self.txt_input.returnPressed.connect(self._on_execute)
        self.txt_input.installEventFilter(self)
        in_layout.addWidget(self.txt_input, stretch=1)

        btn_run = QtWidgets.QPushButton("Run")
        btn_run.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 3px 10px;")
        btn_run.clicked.connect(self._on_execute)
        in_layout.addWidget(btn_run)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self.txt_output.clear)
        in_layout.addWidget(btn_clear)

        layout.addLayout(in_layout)

        # Welcome banner
        self._append_html(
            "<span style='color: #38bdf8;'><b>AgyMolden Interactive Python Console</b></span><br>"
            "<span style='color: #94a3b8;'>Type Python expressions or commands. "
            "Use <b>agy</b> for molecular manipulation (e.g. <code>agy.symbols</code>, <code>agy.rotate(45, 'z')</code>, <code>agy.center()</code>).</span><br>"
        )

    def eventFilter(self, obj, event):
        """Handle Up/Down arrow keys for command history navigation."""
        if obj == self.txt_input and event.type() == QtCore.QEvent.Type.KeyPress:
            key = event.key()
            if key == QtCore.Qt.Key.Key_Up:
                if self.history and self.history_idx > 0:
                    self.history_idx -= 1
                    self.txt_input.setText(self.history[self.history_idx])
                return True
            elif key == QtCore.Qt.Key.Key_Down:
                if self.history and self.history_idx < len(self.history) - 1:
                    self.history_idx += 1
                    self.txt_input.setText(self.history[self.history_idx])
                else:
                    self.history_idx = len(self.history)
                    self.txt_input.clear()
                return True
        return super().eventFilter(obj, event)

    def _append_html(self, html: str):
        self.txt_output.append(html)
        cursor = self.txt_output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.txt_output.setTextCursor(cursor)

    def _custom_print(self, *args, **kwargs):
        buf = io.StringIO()
        print(*args, file=buf, **kwargs)
        text = html.escape(buf.getvalue().rstrip()).replace("\n", "<br>")
        self._append_html(f"<span style='color: #cbd5e1;'>{text}</span>")

    def _on_execute(self):
        cmd = self.txt_input.text().strip()
        if not cmd:
            return

        self.history.append(cmd)
        self.history_idx = len(self.history)
        self.txt_input.clear()

        self._append_html(f"<span style='color: #38bdf8;'><b>&gt;&gt;&gt;</b></span> <span style='color: #f1f5f9;'>{html.escape(cmd)}</span>")

        # Try evaluating as an expression first, then fall back to exec
        try:
            try:
                code = compile(cmd, "<console>", "eval")
                res = eval(code, self.namespace)
                if res is not None:
                    self._append_html(f"<span style='color: #34d399;'>{html.escape(repr(res))}</span>")
            except SyntaxError:
                # Execute statement
                exec(cmd, self.namespace)
        except Exception:
            err = traceback.format_exc().rstrip()
            # Show concise error trace
            lines = err.split("\n")
            short_err = html.escape("\n".join(lines[-3:])).replace("\n", "<br>")
            self._append_html(f"<span style='color: #f87171;'>{short_err}</span>")
