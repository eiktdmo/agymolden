"""Run slow calculations off the GUI thread with a busy/cancel dialog."""

import logging
from typing import Any, Callable

from PyQt6 import QtCore, QtWidgets

log = logging.getLogger(__name__)

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
    parent: QtWidgets.QWidget,
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

    dlg = QtWidgets.QProgressDialog(message, "Cancel", 0, 0, parent)
    dlg.setWindowTitle(title)
    dlg.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
    dlg.setMinimumDuration(300)  # no flicker for fast jobs
    dlg.setAutoClose(False)
    dlg.setAutoReset(False)

    worker = _Worker(fn, parent)
    state = {"cancelled": False, "finished": False}

    def close_dialog():
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

    worker.succeeded.connect(lambda r: finish(on_done, r))
    worker.failed.connect(lambda e: finish(on_error, e))
    dlg.canceled.connect(cancel)
    # keep the QThread alive until it truly ends, then free it
    worker.finished.connect(worker.deleteLater)
    parent._active_workers = getattr(parent, "_active_workers", [])
    parent._active_workers.append(worker)
    worker.finished.connect(lambda: parent._active_workers.remove(worker) if worker in parent._active_workers else None)
    worker.start()
    dlg.show()
