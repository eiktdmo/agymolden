"""Companion IPC bridge between AgyMolden and agemacs."""

import json
import os
import shutil
import subprocess
import sys
from typing import List, Optional
from PyQt6 import QtNetwork, QtWidgets


def send_to_agemacs(targets: List[str], parent: Optional[QtWidgets.QWidget] = None) -> bool:
    """Open target files in a running agemacs instance via IPC, or launch agemacs."""
    clean_targets = [t for t in targets if t]
    if not clean_targets:
        return False

    sock = QtNetwork.QLocalSocket(parent)
    sock.connectToServer("agemacs_companion_ipc")
    if sock.waitForConnected(400):
        payload = json.dumps({"action": "open", "files": clean_targets}).encode("utf-8")
        sock.write(payload)
        sock.flush()
        sock.waitForBytesWritten(400)
        sock.disconnectFromServer()
        return True

    agemacs_bin = shutil.which("agemacs")
    if agemacs_bin:
        subprocess.Popen([agemacs_bin] + clean_targets)
        return True

    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_dirs = [
        os.environ.get("AGEMACS_DIR", ""),
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
                subprocess.Popen([bat] + clean_targets, cwd=agemacs_dir)
                return True
        else:
            launcher = os.path.join(agemacs_dir, "agemacs")
            if os.path.isfile(launcher) and os.access(launcher, os.X_OK):
                subprocess.Popen([launcher] + clean_targets, cwd=agemacs_dir)
                return True
        for cand in ("editor.py", "agemacs.py", "main.py"):
            script = os.path.join(agemacs_dir, cand)
            if os.path.isfile(script):
                subprocess.Popen([sys.executable, script] + clean_targets, cwd=agemacs_dir)
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
