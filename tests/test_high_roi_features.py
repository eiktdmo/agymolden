"""Tests for high-ROI features: batch MLIP energies/forces and camera focus."""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtGui

from agymolden.mlip_engine import batch_energies_and_forces_mlip
from agymolden.viewer import MoleculeViewer


class TestHighROIFeatures(unittest.TestCase):
    def test_batch_energies_empty(self) -> None:
        self.assertEqual(batch_energies_and_forces_mlip(["H", "H"], []), [])

    def test_batch_energies_valid_and_invalid_frames_isolated(self) -> None:
        symbols = ["H", "H"]
        valid_frame = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        bad_shape = np.array([[0.0, 0.0, 0.0]])
        nan_frame = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.74]])

        res = batch_energies_and_forces_mlip(
            symbols, [valid_frame, bad_shape, nan_frame]
        )
        self.assertEqual(len(res), 3)
        self.assertIsNone(res[0]["error"])
        self.assertIsInstance(res[0]["energy"], float)

        self.assertIsNotNone(res[1]["error"])
        self.assertIsNone(res[1]["energy"])

        self.assertIsNotNone(res[2]["error"])
        self.assertIsNone(res[2]["energy"])

    def test_focus_on_selection_safeguards(self) -> None:
        class _DummyViewer:
            coords = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
            selected_atoms = [1]
            rot_quat = QtGui.QQuaternion()
            pan_x = 0.0
            pan_y = 0.0
            updated = False

            def update(self):
                self.updated = True

            def auto_center(self):
                self.pan_x = 0.0
                self.pan_y = 0.0

        v = _DummyViewer()
        MoleculeViewer.focus_on_selection(v)
        self.assertTrue(v.updated)
        self.assertAlmostEqual(v.pan_x, -2.0, places=5)
        self.assertAlmostEqual(v.pan_y, 0.0, places=5)

        # Out-of-range selection falls back to auto_center
        v.selected_atoms = [99]
        MoleculeViewer.focus_on_selection(v)
        self.assertAlmostEqual(v.pan_x, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
