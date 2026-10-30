"""
Smart Electronic Basis Set Selector Dialog connected to Basis Set Exchange (BSE).

Provides:
- Live filtering of orbital basis sets matching molecule elements
- Categorization by basis set family (Dunning, Ahlrichs/Karlsruhe, Pople, Jensen, etc.)
- Search / filter by name, description, and keywords
- Detailed info panel showing citations, description, and element coverage
"""

from typing import List, Optional, Dict
from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import basis_set_exchange as bse
    _HAS_BSE = True
except ImportError:
    _HAS_BSE = False

FAMILY_DISPLAY_NAMES = {
    "dunning": "Dunning Correlation-Consistent (cc-pVnZ / aug-cc-pVnZ)",
    "ahlrichs": "Ahlrichs / Karlsruhe (def2-SVP, def2-TZVP, def2-QZVP)",
    "pople": "Pople Split-Valence (6-31G*, 6-311+G**, etc.)",
    "jensen": "Jensen Polarization Consistent (pc-n / aug-pc-n)",
    "ano": "Atomic Natural Orbital (ANO-RCC / ANO-DK3)",
    "sto": "Slater-Type Minimal (STO-3G, STO-6G)",
    "truhlar": "Truhlar Calendar sets (jun-cc-pVTZ, etc.)",
    "dyall": "Dyall Relativistic",
    "jorge": "Jorge / DKH / ZORA",
    "sapporo": "Sapporo Relativistic",
    "lehtola_emd": "Lehtola Completeness-Optimized (coemd)",
    "lehtola_hgbs": "Lehtola Harmonic Gaussian (AHGBS)",
    "sigmanz": "Sigma-nZ",
}

POPULAR_BASIS_SETS = [
    "def2-svp", "def2-tzvp", "def2-tzvpp", "def2-qzvp", "def2-qzvpp",
    "cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz",
    "6-31g*", "6-31+g*", "6-311g**", "6-311++g**", "sto-3g", "ano-rcc-dzp"
]


class SmartBasisDialog(QtWidgets.QDialog):
    basis_selected = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget, elements: Optional[List[str]] = None, current_basis: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Smart Electronic Basis Set Selector (Basis Set Exchange)")
        self.resize(750, 560)
        self.elements = sorted(list(set(elements))) if elements else []
        self.current_basis = current_basis.strip()
        self.selected_basis_name = self.current_basis

        self._init_ui()
        self._populate_basis_sets()

    def _init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)

        # Header Banner
        header = QtWidgets.QHBoxLayout()
        el_str = ", ".join(self.elements) if self.elements else "All Elements"
        lbl_target = QtWidgets.QLabel(f"Target Molecule Elements: <b>{el_str}</b>")
        lbl_target.setStyleSheet("font-size: 13px; color: #38bdf8;")
        header.addWidget(lbl_target)
        header.addStretch()

        self.chk_match_only = QtWidgets.QCheckBox("Show only basis sets supporting all target elements")
        self.chk_match_only.setChecked(bool(self.elements))
        self.chk_match_only.toggled.connect(self._populate_basis_sets)
        header.addWidget(self.chk_match_only)
        layout.addLayout(header)

        # Search Bar & Family Filter
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Search:"))
        self.txt_search = QtWidgets.QLineEdit()
        self.txt_search.setPlaceholderText("Filter by name (e.g. def2, aug-cc, 6-311, ano)...")
        self.txt_search.setClearButtonEnabled(True)
        self.txt_search.textChanged.connect(self._filter_tree)
        filter_row.addWidget(self.txt_search, stretch=2)

        filter_row.addWidget(QtWidgets.QLabel("Family:"))
        self.combo_family = QtWidgets.QComboBox()
        self.combo_family.addItem("All Families", "")
        for fam_code, fam_title in sorted(FAMILY_DISPLAY_NAMES.items(), key=lambda x: x[1]):
            self.combo_family.addItem(fam_title, fam_code)
        self.combo_family.currentIndexChanged.connect(self._filter_tree)
        filter_row.addWidget(self.combo_family, stretch=2)
        layout.addLayout(filter_row)

        # Main splitter: Tree View on left, Details on right
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: Tree Widget
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Basis Set", "Family", "Elements Covered"])
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 140)
        self.tree.itemSelectionChanged.connect(self._on_tree_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        splitter.addWidget(self.tree)

        # Right: Detail & Citation Card
        detail_widget = QtWidgets.QWidget()
        d_layout = QtWidgets.QVBoxLayout(detail_widget)
        d_layout.setContentsMargins(6, 0, 0, 0)
        d_layout.addWidget(QtWidgets.QLabel("<b>Basis Set Description & Coverage:</b>"))
        
        self.txt_details = QtWidgets.QTextEdit()
        self.txt_details.setReadOnly(True)
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        if is_white:
            self.txt_details.setStyleSheet("background-color: #ffffff; color: #0f172a; font-family: sans-serif; font-size: 11px; border: 1px solid #cbd5e1;")
        else:
            self.txt_details.setStyleSheet("background-color: #0f172a; color: #e2e8f0; font-family: sans-serif; font-size: 11px;")
        d_layout.addWidget(self.txt_details)
        splitter.addWidget(detail_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # Bottom Action Buttons
        btn_box = QtWidgets.QHBoxLayout()
        self.lbl_selected = QtWidgets.QLabel(f"Selected: <b>{self.current_basis or 'None'}</b>")
        is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
        sel_col = "#047857" if is_white else "#a7f3d0"
        self.lbl_selected.setStyleSheet(f"color: {sel_col}; font-size: 12px;")
        btn_box.addWidget(self.lbl_selected)

        btn_box.addStretch()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        btn_select = QtWidgets.QPushButton("✔ Apply Basis Set")
        btn_select.setStyleSheet("background-color: #2563eb; color: #fff; font-weight: bold; padding: 6px 14px;")
        btn_select.clicked.connect(self._on_apply)
        btn_box.addWidget(btn_select)

        layout.addLayout(btn_box)

    def _populate_basis_sets(self):
        self.tree.clear()
        if not _HAS_BSE:
            item = QtWidgets.QTreeWidgetItem(["basis-set-exchange library not installed", "", ""])
            self.tree.addTopLevelItem(item)
            return

        meta = bse.get_metadata()
        filter_elements = self.elements if self.chk_match_only.isChecked() and self.elements else None
        if filter_elements:
            try:
                allowed_keys = set(bse.filter_basis_sets(elements=filter_elements, role="orbital").keys())
            except Exception:
                allowed_keys = set(meta.keys())
        else:
            allowed_keys = {k for k, v in meta.items() if v.get("role") == "orbital"}

        # Group basis sets by family
        family_groups: Dict[str, List[QtWidgets.QTreeWidgetItem]] = {}

        # Add a special "⭐ Recommended / Popular" top-level group
        rec_items = []
        for bs_key in POPULAR_BASIS_SETS:
            clean_key = bs_key.lower().replace("*", "_st_")
            match_k = None
            if bs_key in meta:
                match_k = bs_key
            elif clean_key in meta:
                match_k = clean_key
            else:
                for k in meta:
                    if k.lower() == bs_key.lower() or meta[k].get("display_name", "").lower() == bs_key.lower():
                        match_k = k
                        break
            if match_k and match_k in allowed_keys:
                v = meta[match_k]
                disp = v.get("display_name", match_k)
                fam = v.get("family", "General")
                ver = v.get("versions", {}).get(v.get("latest_version", "1"), {})
                n_el = len(ver.get("elements", []))
                it = QtWidgets.QTreeWidgetItem([disp, fam, f"{n_el} elements"])
                it.setData(0, QtCore.Qt.ItemDataRole.UserRole, match_k)
                it.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, v)
                rec_items.append(it)

        if rec_items:
            rec_parent = QtWidgets.QTreeWidgetItem(["⭐ Standard & Popular Sets", "Quick Pick", f"{len(rec_items)} sets"])
            rec_parent.setExpanded(True)
            font = rec_parent.font(0)
            font.setBold(True)
            rec_parent.setFont(0, font)
            rec_parent.addChildren(rec_items)
            self.tree.addTopLevelItem(rec_parent)

        # Group the rest
        for bs_key in sorted(allowed_keys):
            v = meta[bs_key]
            fam = v.get("family", "other")
            disp = v.get("display_name", bs_key)
            ver = v.get("versions", {}).get(v.get("latest_version", "1"), {})
            n_el = len(ver.get("elements", []))

            it = QtWidgets.QTreeWidgetItem([disp, fam, f"{n_el} elements"])
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole, bs_key)
            it.setData(0, QtCore.Qt.ItemDataRole.UserRole + 1, v)

            family_groups.setdefault(fam, []).append(it)

        for fam_code, items in sorted(family_groups.items(), key=lambda x: FAMILY_DISPLAY_NAMES.get(x[0], x[0])):
            fam_title = FAMILY_DISPLAY_NAMES.get(fam_code, fam_code.capitalize())
            parent_it = QtWidgets.QTreeWidgetItem([fam_title, fam_code, f"{len(items)} sets"])
            parent_it.addChildren(items)
            self.tree.addTopLevelItem(parent_it)

        self._filter_tree()

    def _filter_tree(self):
        query = self.txt_search.text().strip().lower()
        fam_filter = self.combo_family.currentData()

        for i in range(self.tree.topLevelItemCount()):
            top = self.tree.topLevelItem(i)
            visible_children = 0
            for j in range(top.childCount()):
                child = top.child(j)
                name = child.text(0).lower()
                fam = child.text(1).lower()
                v_data = child.data(0, QtCore.Qt.ItemDataRole.UserRole + 1) or {}
                desc = v_data.get("description", "").lower()

                match_text = (not query) or (query in name) or (query in desc) or (query in fam)
                match_fam = (not fam_filter) or (fam == fam_filter) or (v_data.get("family") == fam_filter)

                child.setHidden(not (match_text and match_fam))
                if match_text and match_fam:
                    visible_children += 1

            top.setHidden(visible_children == 0)
            if query and visible_children > 0:
                top.setExpanded(True)

    def _on_tree_selection_changed(self):
        items = self.tree.selectedItems()
        if not items:
            return
        item = items[0]
        bs_key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        v_data = item.data(0, QtCore.Qt.ItemDataRole.UserRole + 1)
        if not bs_key or not v_data:
            return

        self.selected_basis_name = item.text(0)
        self.lbl_selected.setText(f"Selected: <b>{self.selected_basis_name}</b>")

        # Update Detail Pane
        disp = v_data.get("display_name", bs_key)
        fam = v_data.get("family", "")
        desc = v_data.get("description", "No description available.")
        ver = v_data.get("versions", {}).get(v_data.get("latest_version", "1"), {})
        covered_els = ver.get("elements", [])

        # Check coverage of current molecule elements
        missing = [el for el in self.elements if str(bse.lut.element_data_from_symbol(el)["atomic_number"]) not in covered_els] if _HAS_BSE and self.elements else []
        coverage_html = "<span style='color: #4ade80;'>✔ Covers all target elements!</span>" if not missing else f"<span style='color: #f87171;'>⚠ Missing target elements: {', '.join(missing)}</span>"

        info_html = f"""
        <h3 style='color: #38bdf8; margin-bottom: 4px;'>{disp}</h3>
        <p><b>Family:</b> {fam.capitalize()} | <b>Role:</b> Orbital</p>
        <p><b>Molecule Coverage:</b> {coverage_html}</p>
        <p><b>Description:</b><br>{desc}</p>
        <p><b>Total Elements Supported ({len(covered_els)}):</b><br>
        <span style='color: #94a3b8; font-size: 10px;'>{', '.join([bse.lut.element_symbol_from_Z(int(z)) for z in covered_els if int(z) <= 118]) if _HAS_BSE else ''}</span></p>
        """
        self.txt_details.setHtml(info_html)

    def _on_item_double_clicked(self, item: QtWidgets.QTreeWidgetItem, col: int):
        bs_key = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if bs_key:
            self._on_apply()

    def _on_apply(self):
        if self.selected_basis_name:
            self.basis_selected.emit(self.selected_basis_name)
            self.accept()
