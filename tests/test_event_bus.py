"""
Unit tests for AgyMolden Decoupled Event-Bus Architecture (Phase D).
"""

import pytest
import numpy as np
from PyQt6 import QtWidgets

from agymolden.events import get_event_bus, reset_event_bus, AppEventBus


@pytest.fixture(scope="module")
def qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


@pytest.fixture(autouse=True)
def clean_event_bus():
    bus = reset_event_bus()
    yield bus
    reset_event_bus()


def test_event_bus_singleton(qapp):
    bus1 = get_event_bus()
    bus2 = get_event_bus()
    assert bus1 is bus2
    assert isinstance(bus1, AppEventBus)


def test_molecule_lifecycle_signals(qapp):
    bus = get_event_bus()
    received_mol = []
    cleared_count = []

    bus.molecule_loaded.connect(lambda syms, coords, title: received_mol.append((syms, coords, title)))
    bus.molecule_cleared.connect(lambda: cleared_count.append(1))

    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    bus.molecule_loaded.emit(symbols, coords, "H2")

    assert len(received_mol) == 1
    assert received_mol[0][0] == ["H", "H"]
    assert np.allclose(received_mol[0][1], coords)
    assert received_mol[0][2] == "H2"

    bus.molecule_cleared.emit()
    assert len(cleared_count) == 1


def test_job_execution_signals(qapp):
    bus = get_event_bus()
    steps = []
    completed = []

    bus.job_step_received.connect(lambda syms, coords, energy: steps.append((syms, energy)))
    bus.job_completed.connect(lambda path, success: completed.append((path, success)))

    bus.job_step_received.emit(["C", "O"], np.zeros((2, 3)), -113.25)
    bus.job_completed.emit("/tmp/opt.out", True)

    assert len(steps) == 1
    assert steps[0][0] == ["C", "O"]
    assert steps[0][1] == -113.25

    assert len(completed) == 1
    assert completed[0] == ("/tmp/opt.out", True)


def test_theme_and_style_signals(qapp):
    bus = get_event_bus()
    themes = []
    styles = []

    bus.theme_changed.connect(themes.append)
    bus.style_mode_changed.connect(styles.append)

    bus.theme_changed.emit("white")
    bus.style_mode_changed.emit("Spacefill")

    assert themes == ["white"]
    assert styles == ["Spacefill"]
