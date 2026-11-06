import pytest
from PyQt6 import QtWidgets, QtCore, QtGui
from agymolden.mo_ladder_widget import MOEnergyLadderWidget

class DummyMO:
    def __init__(self, energy, occupancy, sym):
        self.energy = energy
        self.occupancy = occupancy
        self.sym = sym

def test_mo_ladder_widget(qapp):
    w = MOEnergyLadderWidget()
    mos = [
        DummyMO(energy=-19.5, occupancy=2.0, sym="1a1"),
        DummyMO(energy=-1.2, occupancy=2.0, sym="2a1"),
        DummyMO(energy=-0.7, occupancy=2.0, sym="1b2"),
        DummyMO(energy=-0.5, occupancy=2.0, sym="3a1"),  # HOMO
        DummyMO(energy=0.1, occupancy=0.0, sym="1b1"),   # LUMO
        DummyMO(energy=0.3, occupancy=0.0, sym="4a1"),
    ]
    w.set_orbitals(mos, homo_idx=3)
    assert len(w._orbitals) == 6
    assert w._homo_idx == 3

    clicked = []
    w.orbital_clicked.connect(clicked.append)

    # Trigger click on first rung
    w.resize(200, 300)
    w.show()
    QtWidgets.QApplication.processEvents()

    assert len(w._rung_rects) > 0
    mo_target, rect_target = w._rung_rects[0]
    center = rect_target.center()
    ev = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseButtonPress,
        QtCore.QPointF(center.x(), center.y()),
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    w.mousePressEvent(ev)
    assert clicked == [mo_target]
    w.close()
