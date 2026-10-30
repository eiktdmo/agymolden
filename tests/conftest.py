import os

# Run Qt without a display (CI, SSH, headless machines).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest


@pytest.fixture(autouse=True)
def _synchronous_jobs(monkeypatch):
    """Run background jobs inline so tests see results immediately.

    Tests of the real threading use the ``threaded_jobs`` fixture instead.
    """
    from agymolden import workers
    monkeypatch.setattr(workers, "SYNCHRONOUS", True)


@pytest.fixture
def threaded_jobs(monkeypatch):
    from agymolden import workers
    monkeypatch.setattr(workers, "SYNCHRONOUS", False)


@pytest.fixture(scope="session")
def qapp():
    from PyQt6 import QtWidgets
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app
