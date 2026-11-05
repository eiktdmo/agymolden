"""Tests for Onboarding Tour & First-Launch Walkthrough Dialog."""

import pytest
import numpy as np
from PyQt6 import QtWidgets, QtCore
from agymolden.dialog_onboarding import OnboardingTourDialog
from agymolden.zmatrix import ZAtom


class DummyWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.zatoms = []
        self.coords = np.zeros((0, 3))
        self.molecule_set = False

    def set_molecule_from_zatoms(self, zatoms, exact_coords=None):
        self.zatoms = zatoms
        self.coords = exact_coords if exact_coords is not None else np.zeros((len(zatoms), 3))
        self.molecule_set = True


def test_onboarding_tour_navigation(qapp):
    win = DummyWindow()
    dlg = OnboardingTourDialog(win)

    # 1. Starts at step 0
    assert dlg.pages_stack.count() == 4
    assert dlg.current_step == 0
    assert "Step 1 of 4" in dlg.lbl_step_indicator.text()
    assert dlg.btn_prev.isEnabled() is False
    assert dlg.btn_next.text() == "Next ▶"

    # 2. Advance to step 1
    dlg._on_next()
    assert dlg.current_step == 1
    assert "Step 2 of 4" in dlg.lbl_step_indicator.text()
    assert dlg.btn_prev.isEnabled() is True

    # 3. Advance to final step (step 3)
    dlg._on_next()
    dlg._on_next()
    assert dlg.current_step == 3
    assert "Step 4 of 4" in dlg.lbl_step_indicator.text()
    assert dlg.btn_next.text() == "Finish & Explore 🎉"

    # 4. Previous navigation
    dlg._on_prev()
    assert dlg.current_step == 2
    assert "Step 3 of 4" in dlg.lbl_step_indicator.text()

    dlg.close()


def test_onboarding_sample_loading(qapp):
    win = DummyWindow()
    dlg = OnboardingTourDialog(win)

    # Test loading Water
    dlg._load_sample("h2o")
    assert win.molecule_set is True
    assert len(win.zatoms) == 3
    assert [a.symbol for a in win.zatoms] == ["O", "H", "H"]

    # Test loading Benzene
    dlg._load_sample("benzene")
    assert len(win.zatoms) == 12
    assert win.zatoms[0].symbol == "C"

    dlg.close()


def test_onboarding_startup_preference_toggle(qapp):
    dlg = OnboardingTourDialog()
    initial_val = dlg.chk_show_on_startup.isChecked()

    dlg.chk_show_on_startup.setChecked(not initial_val)
    saved_val = dlg.settings.value("show_onboarding_on_startup", type=bool)
    assert saved_val == (not initial_val)

    # Restore initial state
    dlg.chk_show_on_startup.setChecked(initial_val)
    dlg.close()
