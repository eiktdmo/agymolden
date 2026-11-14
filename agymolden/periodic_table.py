"""Molden-style Periodic Table picker dialog/widget."""

from PyQt6 import QtCore, QtGui, QtWidgets
from .elements import MOLDEN_PERIODIC_TABLE, ELEMENT_DATA, get_color
from .coordination import COORDINATION_TEMPLATES, resolve_coordination_template


class PeriodicTableDialog(QtWidgets.QDialog):
    element_selected = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, initial_coordination: str = "auto"):
        super().__init__(parent)
        self.setWindowTitle("Molden / GaussView - Select Element & Coordination")
        self.setModal(True)
        self.resize(740, 380)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header instructions
        lbl_info = QtWidgets.QLabel("Select an element to add to the Z-matrix (or press Esc to cancel):")
        lbl_info.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(lbl_info)

        # Grid of elements
        grid_widget = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(grid_widget)
        grid.setSpacing(3)
        grid.setContentsMargins(0, 0, 0, 0)

        for sym, row, col in MOLDEN_PERIODIC_TABLE:
            data = ELEMENT_DATA.get(sym, {})
            name = data.get("name", sym)
            z_num = data.get("atomic_number", 0)
            c = data.get("color", (0.5, 0.5, 0.5))

            # CPK-tinted background with good contrast
            r_i, g_i, b_i = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
            # Darken or blend for button background
            bg_col = f"rgb({int(r_i * 0.45 + 30)}, {int(g_i * 0.45 + 30)}, {int(b_i * 0.45 + 30)})"
            fg_col = "#ffffff" if (r_i * 0.299 + g_i * 0.587 + b_i * 0.114) < 180 else "#000000"

            btn = QtWidgets.QPushButton(sym)
            btn.setProperty("class", "elem-btn")
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {bg_col};
                    color: {fg_col};
                }}
            """)
            btn.setToolTip(f"{name} (Z={z_num})")
            # Connect
            btn.clicked.connect(lambda checked, s=sym: self._choose_element(s))
            btn.installEventFilter(self)
            btn.setProperty("elem_sym", sym)
            btn.setProperty("elem_name", name)
            btn.setProperty("elem_z", z_num)
            grid.addWidget(btn, row, col)

        layout.addWidget(grid_widget, stretch=1)

        # Compact Element Info Strip (updates on hover)
        self.lbl_elem_info = QtWidgets.QLabel("Hover any element to inspect atomic mass, covalent & vdW radii.")
        self.lbl_elem_info.setStyleSheet("color: #cbd5e1; font-size: 11px; padding: 4px 6px; background: #1e293b; border-radius: 4px;")
        layout.addWidget(self.lbl_elem_info)

        # GaussView-style Coordination / Hybridization Template Selector
        coord_row = QtWidgets.QHBoxLayout()
        coord_row.setSpacing(8)
        lbl_coord = QtWidgets.QLabel("Coordination / Hybridization Template:")
        lbl_coord.setStyleSheet("font-weight: bold; color: #0284c7;")
        coord_row.addWidget(lbl_coord)

        self.combo_coordination = QtWidgets.QComboBox()
        self.combo_coordination.setMinimumWidth(300)
        for key, (label, _, _) in COORDINATION_TEMPLATES.items():
            self.combo_coordination.addItem(label, key)
        idx_init = self.combo_coordination.findData(initial_coordination or "auto")
        if idx_init >= 0:
            self.combo_coordination.setCurrentIndex(idx_init)
        coord_row.addWidget(self.combo_coordination, stretch=1)

        if parent is not None and hasattr(parent, "_on_add_hydrogens"):
            btn_add_h = QtWidgets.QPushButton("💧 Add H (Complete Valence)")
            btn_add_h.setStyleSheet("background-color: #0284c7; color: #ffffff; font-weight: bold; padding: 4px 10px;")
            btn_add_h.setToolTip("Complete valence of all atoms with standard geometric hydrogens (Ctrl+H)")
            btn_add_h.clicked.connect(lambda: (parent._on_add_hydrogens(), self.reject()))
            coord_row.addWidget(btn_add_h)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        coord_row.addWidget(btn_cancel)
        layout.addLayout(coord_row)

        self.chosen_element: str = ""
        self.chosen_coordination: str = initial_coordination or "auto"

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Type.Enter:
            sym = obj.property("elem_sym")
            if sym:
                name = obj.property("elem_name") or sym
                z_num = obj.property("elem_z") or ""
                try:
                    from .elements import get_mass, get_cov_radius, get_vdw_radius
                    mass = get_mass(sym)
                    rcov = get_cov_radius(sym)
                    rvdw = get_vdw_radius(sym)
                    self.lbl_elem_info.setText(
                        f"<b>{sym}</b> — {name} (Z={z_num}) &nbsp;|&nbsp; Mass: {mass:.3f} u &nbsp;|&nbsp; r<sub>cov</sub>: {rcov:.2f} Å &nbsp;|&nbsp; r<sub>vdW</sub>: {rvdw:.2f} Å"
                    )
                except Exception as exc:
                    self.lbl_elem_info.setText(f"<b>{sym}</b> — {name} (Z={z_num}) [info error: {exc}]")
        return super().eventFilter(obj, event)

    def _choose_element(self, sym: str):
        self.chosen_element = sym
        self.chosen_coordination = str(self.combo_coordination.currentData() or "auto")
        self.element_selected.emit(sym)
        self.accept()
