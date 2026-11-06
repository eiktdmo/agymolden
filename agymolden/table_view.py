"""Z-matrix table widget with inline editing, variable support, and Molden interactive workflow."""

from typing import List, Optional, Dict
from PyQt6 import QtCore, QtGui, QtWidgets

from .zmatrix import ZAtom
from .elements import ELEMENT_DATA


class ZMatrixTableWidget(QtWidgets.QTableWidget):
    before_data_changed = QtCore.pyqtSignal()
    data_changed = QtCore.pyqtSignal()
    atom_selected = QtCore.pyqtSignal(int)
    cell_selected = QtCore.pyqtSignal(int, int) # (row, col)

    COL_IDX = 0
    COL_SYM = 1
    COL_REF1 = 2
    COL_DIST = 3
    COL_REF2 = 4
    COL_ANG = 5
    COL_REF3 = 6
    COL_DIH = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(8)
        self.setHorizontalHeaderLabels([
            "#", "Sym", "Ref 1", "Bond (Å)", "Ref 2", "Angle (°)", "Ref 3", "Dihed (°)"
        ])
        header = self.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
            for col in (self.COL_IDX, self.COL_SYM, self.COL_REF1, self.COL_REF2, self.COL_REF3):
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
            header.setMinimumSectionSize(38)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        v_header = self.verticalHeader()
        if v_header is not None:
            v_header.setVisible(False)
            v_header.setDefaultSectionSize(22)  # Compact row height (22px) maximizes visible lines
            v_header.setMinimumSectionSize(18)
        self.setStyleSheet(
            "QTableWidget::item { padding: 1px 4px; font-size: 11px; }"
        )
        self.setAlternatingRowColors(True)

        self._updating = False
        self.zatoms: List[ZAtom] = []

        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self._on_selection_changed)
        self.cellClicked.connect(self._on_cell_clicked)

    def filter_rows(self, query: str):
        """Filter table rows by atom symbol, index, or index range (e.g. 'C', '1-20', '>10')."""
        query = query.strip().lower()
        if not query:
            for row in range(self.rowCount()):
                self.setRowHidden(row, False)
            return

        for row in range(self.rowCount()):
            idx_str = str(row + 1)
            sym = self.zatoms[row].symbol.lower() if row < len(self.zatoms) else ""
            match = False
            if query == sym or query in sym:
                match = True
            elif query == idx_str:
                match = True
            elif "-" in query:
                parts = query.split("-", 1)
                try:
                    start, end = int(parts[0]), int(parts[1])
                    if start <= (row + 1) <= end:
                        match = True
                except ValueError:
                    pass
            elif query.startswith(">"):
                try:
                    val = int(query[1:])
                    if (row + 1) > val:
                        match = True
                except ValueError:
                    pass
            elif query.startswith("<"):
                try:
                    val = int(query[1:])
                    if (row + 1) < val:
                        match = True
                except ValueError:
                    pass
            self.setRowHidden(row, not match)

    def load_zatoms(self, zatoms: List[ZAtom]):
        self._updating = True
        self.zatoms = zatoms
        self.setRowCount(len(zatoms))

        for row, a in enumerate(zatoms):
            # Index (read-only)
            item_idx = QtWidgets.QTableWidgetItem(str(row + 1))
            item_idx.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable)
            item_idx.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            item_idx.setForeground(QtGui.QBrush(QtGui.QColor("#90caf9")))
            self.setItem(row, self.COL_IDX, item_idx)

            # Symbol
            item_sym = QtWidgets.QTableWidgetItem(a.symbol)
            item_sym.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            font = item_sym.font()
            font.setBold(True)
            item_sym.setFont(font)
            self.setItem(row, self.COL_SYM, item_sym)

            # Ref 1 & Dist
            if row >= 1:
                ref1_str = str(a.ref1 + 1) if a.ref1 is not None else "1"
                dist_str = f"{a.dist_var} ({a.dist:.4f})" if a.dist_var else f"{a.dist:.4f}"
                item_ref1 = QtWidgets.QTableWidgetItem(ref1_str)
                item_ref1.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item_dist = QtWidgets.QTableWidgetItem(dist_str)
                item_dist.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if a.dist_var:
                    item_dist.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            else:
                item_ref1 = QtWidgets.QTableWidgetItem("-")
                item_ref1.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                item_dist = QtWidgets.QTableWidgetItem("-")
                item_dist.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.setItem(row, self.COL_REF1, item_ref1)
            self.setItem(row, self.COL_DIST, item_dist)

            # Ref 2 & Angle
            if row >= 2:
                ref2_str = str(a.ref2 + 1) if a.ref2 is not None else "1"
                ang_str = f"{a.angle_var} ({a.angle:.3f})" if a.angle_var else f"{a.angle:.3f}"
                item_ref2 = QtWidgets.QTableWidgetItem(ref2_str)
                item_ref2.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item_ang = QtWidgets.QTableWidgetItem(ang_str)
                item_ang.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if a.angle_var:
                    item_ang.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            else:
                item_ref2 = QtWidgets.QTableWidgetItem("-")
                item_ref2.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                item_ang = QtWidgets.QTableWidgetItem("-")
                item_ang.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.setItem(row, self.COL_REF2, item_ref2)
            self.setItem(row, self.COL_ANG, item_ang)

            # Ref 3 & Dihedral
            if row >= 3:
                ref3_str = str(a.ref3 + 1) if a.ref3 is not None else "1"
                dih_str = f"{a.dihedral_var} ({a.dihedral:.3f})" if a.dihedral_var else f"{a.dihedral:.3f}"
                item_ref3 = QtWidgets.QTableWidgetItem(ref3_str)
                item_ref3.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                item_dih = QtWidgets.QTableWidgetItem(dih_str)
                item_dih.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
                if a.dihedral_var:
                    item_dih.setForeground(QtGui.QBrush(QtGui.QColor("#38bdf8")))
            else:
                item_ref3 = QtWidgets.QTableWidgetItem("-")
                item_ref3.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
                item_dih = QtWidgets.QTableWidgetItem("-")
                item_dih.setFlags(QtCore.Qt.ItemFlag.NoItemFlags)
            self.setItem(row, self.COL_REF3, item_ref3)
            self.setItem(row, self.COL_DIH, item_dih)

        self._updating = False

    def highlight_row(self, row: int):
        if 0 <= row < self.rowCount():
            self._updating = True
            self.selectRow(row)
            self.scrollToItem(self.item(row, 0))
            self._updating = False

    def _on_selection_changed(self):
        if self._updating:
            return
        selected = self.selectedItems()
        if selected:
            row = selected[0].row()
            col = selected[0].column()
            self.atom_selected.emit(row)
            self.cell_selected.emit(row, col)

    def _on_cell_clicked(self, row: int, col: int):
        if not self._updating:
            self.cell_selected.emit(row, col)

    @staticmethod
    def _parse_coord_cell(text: str, current_val: float, current_var: Optional[str], min_val: Optional[float] = None):
        """Parse cell input as either a number, a variable name ('rCC'), 'rCC=1.397', or 'rCC (1.397)'."""
        import re
        t = text.strip()
        if not t:
            return current_val, current_var
        # Pure number
        try:
            v = float(t)
            if min_val is not None:
                v = max(min_val, v)
            return v, current_var
        except ValueError:
            pass
        # Format: var = val or var (val)
        m = re.match(r"^([+-]?[A-Za-z_][A-Za-z0-9_]*)\s*(?:=|\()\s*([+-]?\d+(?:\.\d*)?)\s*\)?$", t)
        if m:
            var_name = m.group(1)
            v = float(m.group(2))
            if min_val is not None:
                v = max(min_val, v)
            return v, var_name
        # Pure variable name: e.g. rCC, R1, -D1
        m2 = re.match(r"^([+-]?[A-Za-z_][A-Za-z0-9_]*)$", t)
        if m2:
            return current_val, m2.group(1)
        return current_val, current_var

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem):
        if self._updating:
            return

        row = item.row()
        col = item.column()
        if row >= len(self.zatoms):
            return

        text = item.text().strip()
        atom = self.zatoms[row]
        self.before_data_changed.emit()

        self._updating = True
        try:
            if col == self.COL_SYM:
                norm_sym = text.capitalize() if text else "C"
                old_sym = atom.symbol
                self.last_raw_sym_text = text.strip()
                self.last_old_sym = old_sym
                atom.symbol = norm_sym
                item.setText(norm_sym)
                # If replacing hydrogen with a heavy element (e.g. H -> O), scale bond distance
                if old_sym == "H" and norm_sym != "H" and atom.ref1 is not None and atom.ref1 < len(self.zatoms):
                    from .forcefield import estimate_bond_length
                    partner_sym = self.zatoms[atom.ref1].symbol
                    nat_dist = estimate_bond_length(norm_sym, partner_sym)
                    atom.dist = nat_dist
                    dist_item = self.item(row, self.COL_DIST)
                    if dist_item:
                        dist_item.setText(f"{nat_dist:.4f}")
            elif col == self.COL_REF1 and row >= 1:
                ref = max(1, min(row, int(text)))
                atom.ref1 = ref - 1
                item.setText(str(ref))
            elif col == self.COL_DIST and row >= 1:
                atom.dist, atom.dist_var = self._parse_coord_cell(text, atom.dist, atom.dist_var, min_val=0.01)
            elif col == self.COL_REF2 and row >= 2:
                ref = max(1, min(row, int(text)))
                atom.ref2 = ref - 1
                item.setText(str(ref))
            elif col == self.COL_ANG and row >= 2:
                atom.angle, atom.angle_var = self._parse_coord_cell(text, atom.angle, atom.angle_var)
            elif col == self.COL_REF3 and row >= 3:
                ref = max(1, min(row, int(text)))
                atom.ref3 = ref - 1
                item.setText(str(ref))
            elif col == self.COL_DIH and row >= 3:
                atom.dihedral, atom.dihedral_var = self._parse_coord_cell(text, atom.dihedral, atom.dihedral_var)
        except ValueError:
            pass
        finally:
            self._updating = False

        self.last_edit = {
            self.COL_SYM: (row, "symbol"),
            self.COL_DIST: (row, "dist"),
            self.COL_ANG: (row, "angle"),
            self.COL_DIH: (row, "dihedral"),
        }.get(col)
        self.data_changed.emit()
