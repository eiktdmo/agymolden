"""Companion IPC bridge between AgyMolden and agemacs."""

import json
import os
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple
from PyQt6 import QtWidgets
try:
    from PyQt6 import QtNetwork
except ImportError:
    QtNetwork = None  # type: ignore[assignment]


def _clean_subprocess_env() -> Dict[str, str]:
    """Return a clean environment stripped of AppImage/PyInstaller wrapper variables."""
    env = os.environ.copy()
    for var in ("APPIMAGE", "APPDIR", "OWD", "ARGV0", "PYTHONHOME", "PYTHONPATH", "_MEIPASS", "_MEIPASS2"):
        env.pop(var, None)
    if "LD_LIBRARY_PATH_ORIG" in env:
        env["LD_LIBRARY_PATH"] = env.pop("LD_LIBRARY_PATH_ORIG")
    elif "LD_LIBRARY_PATH" in env:
        parts = [p for p in env["LD_LIBRARY_PATH"].split(os.pathsep) if "/tmp/.mount_" not in p]
        if parts:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
        else:
            env.pop("LD_LIBRARY_PATH", None)
    return env


def _find_agemacs_appimage(search_dirs: List[str]) -> Optional[str]:
    """Search candidate directories for an Agemacs / AgyEditor AppImage."""
    matches: List[Tuple[float, str]] = []
    seen_dirs = set()
    for d in search_dirs:
        if not d:
            continue
        norm_d = os.path.abspath(os.path.expanduser(d))
        if norm_d in seen_dirs or not os.path.isdir(norm_d):
            continue
        seen_dirs.add(norm_d)
        try:
            for entry in os.scandir(norm_d):
                if not entry.is_file():
                    continue
                low = entry.name.lower()
                if low.endswith(".appimage") and ("agemacs" in low or "agyeditor" in low):
                    try:
                        mtime = entry.stat().st_mtime
                    except OSError:
                        mtime = 0.0
                    matches.append((mtime, entry.path))
        except OSError:
            continue
    if not matches:
        return None
    matches.sort(reverse=True)
    best = matches[0][1]
    if os.name != "nt" and not os.access(best, os.X_OK):
        try:
            os.chmod(best, os.stat(best).st_mode | 0o111)
        except OSError:
            pass
    return best


def find_agemacs_command(extra_dirs: Optional[List[str]] = None) -> Optional[Tuple[List[str], Optional[str]]]:
    """Locate an agemacs executable, AppImage, or source launcher.

    Returns ``(cmd_prefix, cwd)`` or ``None`` if not found.
    """
    for env_key in ("AGEMACS_BIN", "AGEMACS_APPIMAGE"):
        explicit = os.environ.get(env_key, "").strip()
        if explicit and os.path.isfile(explicit):
            if os.name != "nt" and not os.access(explicit, os.X_OK):
                try:
                    os.chmod(explicit, os.stat(explicit).st_mode | 0o111)
                except OSError:
                    pass
            return ([explicit], os.path.dirname(os.path.abspath(explicit)))

    agemacs_bin = shutil.which("agemacs")
    if agemacs_bin:
        return ([agemacs_bin], None)

    appimage_path = os.environ.get("APPIMAGE", "").strip()
    appimage_dir = os.path.dirname(os.path.abspath(appimage_path)) if appimage_path else ""
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    appimage_dirs: List[str] = []
    if extra_dirs:
        appimage_dirs.extend(extra_dirs)
    if appimage_dir:
        appimage_dirs.append(appimage_dir)
    appimage_dirs.extend([
        os.environ.get("AGEMACS_DIR", ""),
        os.path.expanduser("~/Applications"),
        os.path.expanduser("~/Programs"),
        os.path.expanduser("~/Programs/agemacs"),
        os.path.expanduser("~/Programs/agemacs/dist"),
        os.path.expanduser("~/Downloads"),
        os.path.expanduser("~/Desktop"),
        os.path.expanduser("~/.local/bin"),
        os.path.dirname(proj_root),
    ])
    found_appimage = _find_agemacs_appimage(appimage_dirs)
    if found_appimage:
        return ([found_appimage], os.path.dirname(os.path.abspath(found_appimage)))

    candidate_dirs = [
        os.environ.get("AGEMACS_DIR", ""),
        os.path.join(appimage_dir, "agemacs") if appimage_dir else "",
        os.path.expanduser("~/Programs/agemacs"),
        os.path.expanduser("~/Projects/agemacs"),
        os.path.expanduser("~/Insync/Desktop/agyprojects/agemacs"),
        os.path.expanduser("~/Desktop/agyprojects/agemacs"),
        os.path.expanduser("~/agyprojects/agemacs"),
        os.path.expanduser("~/agemacs"),
        os.path.abspath(os.path.join(proj_root, "..", "agemacs")),
        os.path.abspath(os.path.join(proj_root, "..", "agyprojects", "agemacs")),
    ]
    for agemacs_dir in candidate_dirs:
        if not agemacs_dir or not os.path.isdir(agemacs_dir):
            continue
        if os.name == "nt":
            bat = os.path.join(agemacs_dir, "agemacs.bat")
            if os.path.isfile(bat):
                return ([bat], agemacs_dir)
        else:
            launcher = os.path.join(agemacs_dir, "agemacs")
            if os.path.isfile(launcher) and os.access(launcher, os.X_OK):
                return ([launcher], agemacs_dir)
        for cand in ("editor.py", "agemacs.py", "main.py"):
            script = os.path.join(agemacs_dir, cand)
            if os.path.isfile(script):
                return ([sys.executable, script], agemacs_dir)

    return None


def send_to_agemacs(targets: List[str], parent: Optional[QtWidgets.QWidget] = None) -> bool:
    """Open target files in a running agemacs instance via IPC, or launch agemacs (including AppImage)."""
    clean_targets = [t for t in targets if t]
    if not clean_targets:
        return False

    if QtNetwork is not None:
        sock = QtNetwork.QLocalSocket(parent)
        sock.connectToServer("agemacs_companion_ipc")
        if sock.waitForConnected(400):
            payload = json.dumps({"action": "open", "files": clean_targets}).encode("utf-8")
            sock.write(payload)
            sock.flush()
            sock.waitForBytesWritten(400)
            sock.disconnectFromServer()
            return True

    found = find_agemacs_command()
    if found is not None:
        cmd, cwd = found
        subprocess.Popen(cmd + clean_targets, cwd=cwd, env=_clean_subprocess_env())
        return True

    return False


def send_text_to_agemacs(
    filename: str,
    content: str,
    parent: Optional[QtWidgets.QWidget] = None,
) -> bool:
    """Write generated text (XYZ, ZMAT, ORCA/Gaussian input, etc.) to a temp file and open it in agemacs."""
    import tempfile

    safe_name = os.path.basename(filename or "agymolden_export.txt")
    out_dir = os.path.join(tempfile.gettempdir(), "agymolden_agemacs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, safe_name)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)
    return send_to_agemacs([out_path], parent)


def start_agymolden_ipc(window: QtWidgets.QMainWindow) -> None:
    """Start QLocalServer on 'agymolden_companion_ipc' to accept files from agemacs."""
    if QtNetwork is None:
        return
    try:
        server_name = "agymolden_companion_ipc"
        QtNetwork.QLocalServer.removeServer(server_name)
        server = QtNetwork.QLocalServer(window)
    except Exception:
        return

    def _on_connection() -> None:
        while server.hasPendingConnections():
            sock = server.nextPendingConnection()
            if sock is None:
                continue
            sock.waitForReadyRead(500)
            raw = bytes(sock.readAll()).decode("utf-8", errors="ignore").strip()
            sock.disconnectFromServer()
            if not raw:
                continue
            try:
                msg = json.loads(raw)
            except Exception:
                msg = {"action": "open", "file": raw}
            if msg.get("action") == "open":
                path = msg.get("file") or (msg.get("files") or [None])[0]
                if path and os.path.isfile(path):
                    window.open_path(path)
                    window.showNormal()
                    window.raise_()
                    window.activateWindow()

    server.newConnection.connect(_on_connection)
    server.listen(server_name)
    window._companion_server = server
