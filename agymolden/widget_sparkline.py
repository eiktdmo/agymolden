"""
agymolden.widget_sparkline
==========================
Lightweight interactive mini energy curve (sparkline) widget for the Trajectory Player.
Clicking or dragging on the curve seeks directly to the corresponding trajectory frame.
"""

from typing import List, Optional
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets


class SparklinePlot(QtWidgets.QWidget):
    """Interactive mini energy trajectory profile graph."""

    step_clicked = QtCore.pyqtSignal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.energies: List[float] = []
        self.active_step: int = 0
        self.setFixedHeight(32)
        self.setMouseTracking(True)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click any point on the energy curve to seek trajectory frame")

    def set_data(self, energies: List[float], active_step: int = 0):
        self.energies = list(energies) if energies else []
        self.active_step = max(0, min(len(self.energies) - 1, active_step)) if self.energies else 0
        self.update()

    def set_active_step(self, step: int):
        if self.energies:
            self.active_step = max(0, min(len(self.energies) - 1, step))
            self.update()

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        n = len(self.energies)

        # Background
        painter.setBrush(QtGui.QColor(15, 23, 42, 210))
        painter.setPen(QtGui.QPen(QtGui.QColor("#334155"), 1.0))
        painter.drawRoundedRect(QtCore.QRect(0, 0, w - 1, h - 1), 4, 4)

        if n < 2:
            painter.setPen(QtGui.QColor("#64748b"))
            painter.setFont(QtGui.QFont("DejaVu Sans", 8))
            painter.drawText(QtCore.QRect(0, 0, w, h), QtCore.Qt.AlignmentFlag.AlignCenter, "Energy Profile")
            painter.end()
            return

        e_arr = np.array(self.energies)
        e_min = float(np.min(e_arr))
        e_max = float(np.max(e_arr))
        e_span = max(1e-6, e_max - e_min)

        pad_x = 8
        pad_y = 5
        plot_w = w - 2 * pad_x
        plot_h = h - 2 * pad_y

        points: List[QtCore.QPointF] = []
        for i, val in enumerate(e_arr):
            x = pad_x + (i / (n - 1)) * plot_w
            y = pad_y + (1.0 - (val - e_min) / e_span) * plot_h
            points.append(QtCore.QPointF(x, y))

        # Filled energy gradient area under the curve
        poly = QtGui.QPolygonF(points)
        poly.append(QtCore.QPointF(pad_x + plot_w, h - pad_y))
        poly.append(QtCore.QPointF(pad_x, h - pad_y))

        grad = QtGui.QLinearGradient(0, pad_y, 0, h - pad_y)
        grad.setColorAt(0.0, QtGui.QColor(56, 189, 248, 120))
        grad.setColorAt(1.0, QtGui.QColor(56, 189, 248, 10))
        painter.setBrush(QtGui.QBrush(grad))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawPolygon(poly)

        # Plot line
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        pen_curve = QtGui.QPen(QtGui.QColor("#38bdf8"), 1.8)
        painter.setPen(pen_curve)
        painter.drawPolyline(QtGui.QPolygonF(points))

        # Active step cursor point and vertical guideline
        if 0 <= self.active_step < len(points):
            cur_pt = points[self.active_step]
            # Vertical needle
            pen_needle = QtGui.QPen(QtGui.QColor(234, 179, 8, 180), 1.2, QtCore.Qt.PenStyle.DashLine)
            painter.setPen(pen_needle)
            painter.drawLine(QtCore.QPointF(cur_pt.x(), pad_y), QtCore.QPointF(cur_pt.x(), h - pad_y))

            # Golden needle point
            painter.setBrush(QtGui.QColor("#eab308"))
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.5))
            painter.drawEllipse(cur_pt, 3.5, 3.5)

        painter.end()

    def _seek_from_pos(self, pos: QtCore.QPoint):
        if len(self.energies) < 2:
            return
        w = self.width()
        pad_x = 8
        plot_w = max(1, w - 2 * pad_x)
        rel_x = max(0.0, min(1.0, (pos.x() - pad_x) / plot_w))
        step = int(round(rel_x * (len(self.energies) - 1)))
        if step != self.active_step:
            self.active_step = step
            self.update()
            self.step_clicked.emit(step)

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._seek_from_pos(event.pos())

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            self._seek_from_pos(event.pos())
