"""Every dialog added in the apex/quintet/Hückel releases must open without raising."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6 import QtWidgets

SYMS = ["C", "C", "H", "H"]
XYZ = np.array([[0.0, 0.0, 0.0], [1.34, 0.0, 0.0], [-0.5, 0.9, 0.0], [1.84, 0.9, 0.0]])


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _factories():
    from agymolden.dialog_eda import EDADialog
    from agymolden.dialog_huckel import HuckelTeachingDialog
    from agymolden.dialog_marcus import MarcusElectronTransferDialog
    from agymolden.dialog_mecp import MECPExplorerDialog
    from agymolden.dialog_pes_2d_builder import PES2DScanBuilderDialog
    from agymolden.dialog_redox_potentials import RedoxPotentialsDialog
    from agymolden.dialog_spin_diagnostics import SpinContaminationDialog
    from agymolden.dialog_sterics import StericsDialog

    return {
        "eda": lambda: EDADialog(None),
        "huckel": lambda: HuckelTeachingDialog(None, SYMS, XYZ),
        "marcus": lambda: MarcusElectronTransferDialog(None),
        "mecp": lambda: MECPExplorerDialog(None, SYMS, XYZ),
        "pes2d": lambda: PES2DScanBuilderDialog(None, SYMS, XYZ, []),
        "redox": lambda: RedoxPotentialsDialog(None),
        "spin": lambda: SpinContaminationDialog(None),
        "sterics": lambda: StericsDialog(None, SYMS, XYZ),
    }


@pytest.mark.parametrize("name", sorted(["eda", "huckel", "marcus", "mecp", "pes2d", "redox", "spin", "sterics"]))
def test_dialog_opens(app, name):
    dlg = _factories()[name]()
    assert dlg is not None
    dlg.close()


def test_huckel_every_preset_solves(app):
    from agymolden.dialog_huckel import HuckelTeachingDialog

    dlg = HuckelTeachingDialog(None, SYMS, XYZ)
    for i in range(dlg.combo_presets.count()):
        dlg.combo_presets.setCurrentIndex(i)
        assert dlg.current_res is not None, dlg.combo_presets.itemText(i)
