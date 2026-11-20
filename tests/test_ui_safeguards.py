"""Tests for low-resolution window and dialog safeguards."""

import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets

from agymolden.ui_safeguards import (
    clamp_window_to_screen,
    install_low_resolution_safeguards,
)


class TestUISafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_oversized_dialog_is_clamped_to_small_screen(self) -> None:
        dlg = QtWidgets.QDialog()
        dlg.setMinimumSize(1200, 900)
        dlg.resize(1200, 900)
        dlg.move(500, 500)

        small_screen = QtCore.QRect(0, 0, 800, 600)
        changed = clamp_window_to_screen(dlg, avail=small_screen)

        self.assertTrue(changed)
        self.assertLessEqual(dlg.minimumWidth(), 800 - 16)
        self.assertLessEqual(dlg.minimumHeight(), 600 - 40)
        self.assertLessEqual(dlg.width(), 800 - 16)
        self.assertLessEqual(dlg.height(), 600 - 40)
        self.assertLessEqual(dlg.x() + dlg.width(), 800)
        self.assertLessEqual(dlg.y() + dlg.height(), 600)

    def test_normal_dialog_on_large_screen_is_unchanged(self) -> None:
        dlg = QtWidgets.QDialog()
        dlg.setMinimumSize(400, 300)
        dlg.resize(500, 350)
        dlg.move(100, 100)

        large_screen = QtCore.QRect(0, 0, 1920, 1080)
        changed = clamp_window_to_screen(dlg, avail=large_screen)

        self.assertFalse(changed)
        self.assertEqual(dlg.width(), 500)
        self.assertEqual(dlg.height(), 350)

    def test_dialog_with_tall_layout_relaxes_constraint(self) -> None:
        dlg = QtWidgets.QDialog()
        layout = QtWidgets.QVBoxLayout(dlg)
        for i in range(30):
            layout.addWidget(QtWidgets.QPushButton(f"Button {i}"))
        dlg.resize(400, 900)

        small_screen = QtCore.QRect(0, 0, 800, 600)
        changed = clamp_window_to_screen(dlg, avail=small_screen)

        self.assertTrue(changed)
        self.assertEqual(
            layout.sizeConstraint(),
            QtWidgets.QLayout.SizeConstraint.SetNoConstraint,
        )
        self.assertLessEqual(dlg.height(), 600 - 40)

    def test_skip_screen_clamp_property_respected(self) -> None:
        win = QtWidgets.QMainWindow()
        win.setProperty("_skip_screen_clamp", True)
        win.resize(1600, 1200)

        small_screen = QtCore.QRect(0, 0, 800, 600)
        changed = clamp_window_to_screen(win, avail=small_screen)

        self.assertFalse(changed)
        self.assertEqual(win.width(), 1600)
        self.assertEqual(win.height(), 1200)

    def test_event_filter_clamps_on_show(self) -> None:
        guard = install_low_resolution_safeguards(self.app)
        try:
            dlg = QtWidgets.QDialog()
            dlg.setMinimumSize(1024, 768)
            dlg.resize(1024, 768)

            small_screen = QtCore.QRect(0, 0, 800, 600)
            with patch(
                "agymolden.ui_safeguards.get_available_screen_geometry",
                return_value=small_screen,
            ):
                dlg.show()
                self.app.processEvents()

            self.assertLessEqual(dlg.width(), 800 - 16)
            self.assertLessEqual(dlg.height(), 600 - 40)
            dlg.close()
        finally:
            self.app.removeEventFilter(guard)


if __name__ == "__main__":
    unittest.main()
