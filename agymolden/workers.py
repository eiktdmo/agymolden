"""Run slow calculations off the GUI thread with a busy/cancel dialog."""

import logging
from typing import Any, Callable

from PyQt6 import QtCore, QtWidgets

log = logging.getLogger(__name__)

# Keep strong references to running background QThreads until their finished signal fires
# so parentless dialogs or canceled jobs never destroy a QThread while it is still running.
_ACTIVE_WORKERS: set = set()

# When True, jobs run inline on the calling thread (used by unit tests that
# need deterministic, immediate results). Never enabled in the real program.
SYNCHRONOUS = False


class _Worker(QtCore.QThread):
    succeeded = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(object)

    def __init__(self, fn: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self):
        try:
            self.succeeded.emit(self._fn())
        except Exception as exc:  # reported to the GUI thread
            log.debug("worker failed", exc_info=True)
            self.failed.emit(exc)


def run_in_background(
    parent: Any,
    title: str,
    message: str,
    fn: Callable[[], Any],
    on_done: Callable[[Any], None],
    on_error: Callable[[Exception], None],
) -> None:
    """Run ``fn()`` in a thread; call ``on_done(result)`` or ``on_error(exc)`` in the GUI thread.

    ``fn`` must not touch any Qt object. The window stays responsive (it repaints)
    but input is blocked by a modal dialog. Cancel discards the result: the
    numerical routines cannot be interrupted, so the thread finishes silently.
    """
    if SYNCHRONOUS:
        try:
            result = fn()
        except Exception as exc:
            on_error(exc)
        else:
            on_done(result)
        return

    widget_parent = parent if isinstance(parent, QtWidgets.QWidget) else None
    dlg = QtWidgets.QProgressDialog(message, "Cancel", 0, 0, widget_parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
    dlg.setMinimumDuration(350)
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)

    worker = _Worker(fn, None)
    state = {"cancelled": False, "finished": False}

    if parent is not None:
        try:
            if not hasattr(parent, "_active_workers"):
                parent._active_workers = set()
            parent._active_workers.add(worker)
        except Exception:
            pass

    def close_dialog():
        dlg.hide()
        dlg.reset()
        dlg.close()
        dlg.deleteLater()

    def finish(callback, payload):
        if state["finished"]:
            return
        state["finished"] = True
        close_dialog()
        if not state["cancelled"]:
            callback(payload)

    def cancel():
        if not state["finished"]:
            state["cancelled"] = True
            finish(lambda _: None, None)

    def cleanup_worker(w=worker):
        _ACTIVE_WORKERS.discard(w)
        if parent is not None:
            try:
                active = getattr(parent, "_active_workers", None)
                if active is not None:
                    active.discard(w)
            except RuntimeError:
                pass

    worker.succeeded.connect(lambda r: finish(on_done, r))
    worker.failed.connect(lambda e: finish(on_error, e))
    dlg.canceled.connect(cancel)
    # Keep the QThread alive until it truly ends, then free it
    _ACTIVE_WORKERS.add(worker)
    worker.finished.connect(cleanup_worker)
    worker.finished.connect(worker.deleteLater)
    worker.start()
    QtCore.QTimer.singleShot(350, lambda: dlg.show() if not state["finished"] else None)
