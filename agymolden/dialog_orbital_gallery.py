"""
Orbital Snapshot & Gallery Generator Dialog.

Provides 1-click batch snapshot generation for Molecular Orbitals (as in Gabedit / Multiwfn):
- Contact Sheet / Collage (single composite PNG of all or selected orbitals in a neat NxM grid)
- Batch PNG Snapshot Export (saves high-res individual images like MO_08_HOMO.png into a folder)
- Interactive Orbital Slideshow / Gallery preview
"""

import os
import math
from typing import List, Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets


class OrbitalSnapshotDialog(QtWidgets.QDialog):
    """Dialog to preview and batch export all or selected molecular orbitals as snapshots or grid collage."""

    def __init__(self, parent: QtWidgets.QMainWindow, molden_path: str, orbitals: List, homo_idx: Optional[int] = None):
        super().__init__(parent)
        self.setWindowTitle("📸 Molecular Orbital Snapshots & Gallery (Gabedit-style)")
        self.resize(880, 680)

        self._parent_mw = parent
        self._molden_path = molden_path
        self._orbitals = orbitals
        self._homo_idx = homo_idx

        # Rendered orbital thumbnails cache: mo_idx -> (QImage, energy_ev, label)
        self._cache_images = {}
        self._is_generating = False
        self._cancel_requested = False

        self._setup_ui()
        self._populate_orbital_table()

    def _setup_ui(self):
        main_layout = QtWidgets.QHBoxLayout(self)

        # Left Column: Orbital selection & parameters
        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_panel.setFixedWidth(320)

        # Selection presets
        grp_scope = QtWidgets.QGroupBox("1. Orbitals to Capture")
        vbox_scope = QtWidgets.QVBoxLayout(grp_scope)

        self.radio_valence = QtWidgets.QRadioButton("Frontier / Valence Orbitals (HOMO±5)")
        self.radio_valence.setChecked(True)
        self.radio_valence.toggled.connect(self._apply_preset_selection)
        vbox_scope.addWidget(self.radio_valence)

        self.radio_frontier4 = QtWidgets.QRadioButton("Publication Frontier Quartet (HOMO-1 to LUMO+1)")
        self.radio_frontier4.toggled.connect(self._apply_preset_selection)
        vbox_scope.addWidget(self.radio_frontier4)

        self.radio_occupied = QtWidgets.QRadioButton("All Occupied Orbitals")
        self.radio_occupied.toggled.connect(self._apply_preset_selection)
        vbox_scope.addWidget(self.radio_occupied)

        self.radio_all = QtWidgets.QRadioButton("All Orbitals (Entire Basis Set)")
        self.radio_all.toggled.connect(self._apply_preset_selection)
        vbox_scope.addWidget(self.radio_all)

        self.radio_custom = QtWidgets.QRadioButton("Custom Selection (below):")
        self.radio_custom.toggled.connect(self._apply_preset_selection)
        vbox_scope.addWidget(self.radio_custom)

        # Table of orbitals with checkboxes
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["MO #", "Type", "Energy (eV)", "Occ."])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        vbox_scope.addWidget(self.table, stretch=1)

        # Selection helpers
        btn_sel_row = QtWidgets.QHBoxLayout()
        btn_sel_all = QtWidgets.QPushButton("Select All")
        btn_sel_all.clicked.connect(self._select_all_rows)
        btn_desel_all = QtWidgets.QPushButton("Deselect All")
        btn_desel_all.clicked.connect(self._deselect_all_rows)
        btn_sel_row.addWidget(btn_sel_all)
        btn_sel_row.addWidget(btn_desel_all)
        vbox_scope.addLayout(btn_sel_row)

        left_layout.addWidget(grp_scope, stretch=1)

        # Surface & Render Settings
        grp_settings = QtWidgets.QGroupBox("2. Snapshot Settings")
        form_settings = QtWidgets.QFormLayout(grp_settings)

        self.spin_iso = QtWidgets.QDoubleSpinBox()
        self.spin_iso.setRange(0.001, 1.0)
        self.spin_iso.setDecimals(3)
        self.spin_iso.setSingleStep(0.005)
        self.spin_iso.setValue(0.030)
        form_settings.addRow("Isovalue:", self.spin_iso)

        self.combo_res = QtWidgets.QComboBox()
        self.combo_res.addItem("Fast / Preview (30³)", 30)
        self.combo_res.addItem("Standard (40³)", 40)
        self.combo_res.addItem("High Quality (55³)", 55)
        self.combo_res.setCurrentIndex(1)
        form_settings.addRow("Grid Mesh:", self.combo_res)

        self.spin_margin = QtWidgets.QDoubleSpinBox()
        self.spin_margin.setRange(1.0, 15.0)
        self.spin_margin.setDecimals(1)
        self.spin_margin.setSingleStep(0.5)
        self.spin_margin.setValue(3.0)
        self.spin_margin.setSuffix(" a₀ (Bohr)")
        self.spin_margin.setToolTip("Plot box padding beyond the outer atoms (Bohr). Increase if orbital lobes are clipped.")
        form_settings.addRow("Plot Box Size:", self.spin_margin)

        self.combo_bg = QtWidgets.QComboBox()
        self.combo_bg.addItems(["Dark (#1e2129)", "White", "Transparent"])
        self.combo_bg.setCurrentIndex(0)
        form_settings.addRow("Background:", self.combo_bg)

        self.combo_cols = QtWidgets.QComboBox()
        self.combo_cols.addItems(["Auto (Square-ish)", "2 Columns", "3 Columns", "4 Columns", "5 Columns", "6 Columns"])
        form_settings.addRow("Grid Columns:", self.combo_cols)

        left_layout.addWidget(grp_settings)

        # Progress bar
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Action Buttons
        self.btn_generate = QtWidgets.QPushButton("⚡ Generate Snapshots")
        self.btn_generate.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 7px;")
        self.btn_generate.clicked.connect(self._on_start_generate)
        left_layout.addWidget(self.btn_generate)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_save_sheet = QtWidgets.QPushButton("💾 Save Contact Sheet...")
        self.btn_save_sheet.setEnabled(False)
        self.btn_save_sheet.clicked.connect(self._on_save_contact_sheet)
        btn_row.addWidget(self.btn_save_sheet)

        self.btn_copy_sheet = QtWidgets.QPushButton("📋 Copy Sheet")
        self.btn_copy_sheet.setToolTip("Copy composite publication contact sheet directly to system clipboard")
        self.btn_copy_sheet.setEnabled(False)
        self.btn_copy_sheet.clicked.connect(self._on_copy_contact_sheet)
        btn_row.addWidget(self.btn_copy_sheet)

        self.btn_save_zip = QtWidgets.QPushButton("📁 Export All...")
        self.btn_save_zip.setEnabled(False)
        self.btn_save_zip.clicked.connect(self._on_export_individual_images)
        btn_row.addWidget(self.btn_save_zip)
        left_layout.addLayout(btn_row)

        main_layout.addWidget(left_panel)

        # Right Panel: Interactive Scrollable Gallery & Preview
        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        header_right = QtWidgets.QHBoxLayout()
        self.lbl_gallery_title = QtWidgets.QLabel("<b>Orbital Gallery Preview</b> (Run Generate to create snapshots)")
        header_right.addWidget(self.lbl_gallery_title)
        header_right.addStretch()

        self.lbl_stats = QtWidgets.QLabel("")
        self.lbl_stats.setStyleSheet("color: #64748b; font-size: 11px;")
        header_right.addWidget(self.lbl_stats)
        right_layout.addLayout(header_right)

        # Scroll Area for Thumbnails Grid
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QtWidgets.QWidget()
        self.grid_layout = QtWidgets.QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(12)
        self.scroll_area.setWidget(self.scroll_content)
        right_layout.addWidget(self.scroll_area, stretch=1)

        main_layout.addWidget(right_panel, stretch=1)

    def _populate_orbital_table(self):
        self.table.setRowCount(len(self._orbitals))
        for idx, mo in enumerate(self._orbitals):
            ev_val = float(mo.energy) * 27.211386
            occ = float(getattr(mo, "occupancy", 0.0))

            # Label tag
            if idx == self._homo_idx:
                tag = "HOMO"
            elif self._homo_idx is not None and idx == self._homo_idx + 1:
                tag = "LUMO"
            elif self._homo_idx is not None and idx < self._homo_idx:
                tag = f"HOMO-{self._homo_idx - idx}"
            elif self._homo_idx is not None and idx > self._homo_idx + 1:
                tag = f"LUMO+{idx - (self._homo_idx + 1)}"
            else:
                tag = "MO"

            item_no = QtWidgets.QTableWidgetItem(f"#{idx+1}")
            item_no.setFlags(item_no.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            item_no.setCheckState(QtCore.Qt.CheckState.Unchecked)

            item_type = QtWidgets.QTableWidgetItem(tag)
            item_ev = QtWidgets.QTableWidgetItem(f"{ev_val:+.2f}")
            item_occ = QtWidgets.QTableWidgetItem(f"{occ:.2f}")

            if idx == self._homo_idx:
                for it in [item_no, item_type, item_ev, item_occ]:
                    it.setForeground(QtGui.QColor("#3b82f6"))
                    it.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Weight.Bold))
            elif self._homo_idx is not None and idx == self._homo_idx + 1:
                for it in [item_no, item_type, item_ev, item_occ]:
                    it.setForeground(QtGui.QColor("#a855f7"))
                    it.setFont(QtGui.QFont("Sans", 9, QtGui.QFont.Weight.Bold))

            self.table.setItem(idx, 0, item_no)
            self.table.setItem(idx, 1, item_type)
            self.table.setItem(idx, 2, item_ev)
            self.table.setItem(idx, 3, item_occ)

        self._apply_preset_selection()

    def _apply_preset_selection(self):
        n = len(self._orbitals)
        center = self._homo_idx if self._homo_idx is not None else (n // 2)

        for row in range(n):
            item = self.table.item(row, 0)
            if not item:
                continue

            if self.radio_valence.isChecked():
                # HOMO-5 to LUMO+5
                lo = max(0, center - 5)
                hi = min(n, center + 6)
                item.setCheckState(QtCore.Qt.CheckState.Checked if lo <= row < hi else QtCore.Qt.CheckState.Unchecked)
            elif hasattr(self, "radio_frontier4") and self.radio_frontier4.isChecked():
                # HOMO-1, HOMO, LUMO, LUMO+1 (2x2 Publication Frontier Quartet)
                lo = max(0, center - 1)
                hi = min(n, center + 3)
                item.setCheckState(QtCore.Qt.CheckState.Checked if lo <= row < hi else QtCore.Qt.CheckState.Unchecked)
            elif self.radio_occupied.isChecked():
                item.setCheckState(QtCore.Qt.CheckState.Checked if row <= center else QtCore.Qt.CheckState.Unchecked)
            elif self.radio_all.isChecked():
                item.setCheckState(QtCore.Qt.CheckState.Checked)

    def _select_all_rows(self):
        self.radio_custom.setChecked(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                it.setCheckState(QtCore.Qt.CheckState.Checked)

    def _deselect_all_rows(self):
        self.radio_custom.setChecked(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it:
                it.setCheckState(QtCore.Qt.CheckState.Unchecked)

    def _get_selected_indices(self) -> List[int]:
        selected = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it and it.checkState() == QtCore.Qt.CheckState.Checked:
                selected.append(r)
        return selected

    def _on_start_generate(self):
        if self._is_generating:
            self._cancel_requested = True
            return

        selected = self._get_selected_indices()
        if not selected:
            QtWidgets.QMessageBox.warning(self, "No Orbitals Selected", "Please select at least one orbital to snapshot.")
            return

        self._is_generating = True
        self._cancel_requested = False
        self.btn_generate.setText("🛑 Cancel")
        self.btn_generate.setStyleSheet("background-color: #dc2626; color: #ffffff; font-weight: bold; padding: 7px;")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(selected))
        self.progress_bar.setValue(0)
        self.btn_save_sheet.setEnabled(False)
        self.btn_save_zip.setEnabled(False)

        # Clear existing gallery items
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._cache_images.clear()

        # Render loop in background thread or sequential timer steps
        isovalue = self.spin_iso.value()
        n_pts = int(self.combo_res.currentData() or 40)
        margin_b = float(self.spin_margin.value()) if hasattr(self, "spin_margin") else 3.0
        bg_mode_text = self.combo_bg.currentText().lower()
        if "white" in bg_mode_text:
            bg_mode = "white"
            transparent = False
        elif "trans" in bg_mode_text:
            bg_mode = "dark"
            transparent = True
        else:
            bg_mode = "dark"
            transparent = False

        self._run_batch_render_sequence(selected, isovalue, n_pts, bg_mode, transparent, margin_bohr=margin_b)

    def _run_batch_render_sequence(self, selected_indices: List[int], isovalue: float, n_pts: int, bg_mode: str, transparent: bool, margin_bohr: float = 3.0):
        import numpy as np
        from .volumetric import evaluate_molden_orbital_grid, generate_orbital_surfaces, parse_molden_file
        viewer = getattr(self._parent_mw, "viewer", None)
        if not viewer:
            QtWidgets.QMessageBox.critical(self, "Error", "3D Molecule Viewer is not available.")
            self._finish_generate()
            return

        # Ensure the 3D viewer has molecular coordinates loaded from the Molden file if the canvas was empty
        if getattr(viewer, "coords", None) is None or len(viewer.coords) == 0:
            try:
                m_data = parse_molden_file(self._molden_path)
                if m_data and len(m_data.symbols) > 0 and len(m_data.coords) == len(m_data.symbols):
                    if hasattr(self._parent_mw, "load_cartesian_molecule"):
                        self._parent_mw.load_cartesian_molecule(
                            m_data.symbols, m_data.coords, title=os.path.basename(self._molden_path)
                        )
            except Exception:
                pass

        orig_surfaces = getattr(viewer, "active_isosurfaces", [])
        orig_clean = getattr(viewer, "export_clean_mode", False)
        orig_bg = getattr(viewer, "bg_color_mode", "dark")
        orig_trans = getattr(viewer, "transparent_bg", False)

        viewer.export_clean_mode = True
        viewer.bg_color_mode = bg_mode
        viewer.transparent_bg = transparent

        is_pos_file = "positron" in os.path.basename(self._molden_path).lower()

        # Step-by-step generator using QTimer so UI remains completely responsive with progress updates
        idx_cursor = [0]
        self._last_render_error = None

        def _step():
            if self._cancel_requested or idx_cursor[0] >= len(selected_indices):
                # Restore viewer state
                viewer.active_isosurfaces = orig_surfaces
                viewer.export_clean_mode = orig_clean
                viewer.bg_color_mode = orig_bg
                viewer.transparent_bg = orig_trans
                viewer.update()
                self._finish_generate()
                return

            mo_idx = selected_indices[idx_cursor[0]]
            self.lbl_stats.setText(f"Calculating MO #{mo_idx+1} ({idx_cursor[0]+1}/{len(selected_indices)})...")

            # 1. Compute orbital grid
            try:
                vol = evaluate_molden_orbital_grid(self._molden_path, orbital_index=mo_idx, nx=n_pts, ny=n_pts, nz=n_pts, margin_bohr=margin_bohr)
                if vol and vol.grid_data is not None:
                    max_abs = float(np.max(np.abs(vol.grid_data)))
                    eff_iso = isovalue
                    if eff_iso >= max_abs * 0.90 and max_abs > 1e-5:
                        eff_iso = round(max_abs * 0.45, 4)

                    if is_pos_file:
                        meshes = generate_orbital_surfaces(
                            vol,
                            isovalue=eff_iso,
                            pos_color=(0.95, 0.72, 0.12),
                            neg_color=(0.16, 0.78, 0.44),
                        )
                    else:
                        meshes = generate_orbital_surfaces(vol, isovalue=eff_iso)

                    viewer.active_isosurfaces = meshes
                    viewer.update()
                    QtWidgets.QApplication.processEvents()

                    # Grab framebuffer
                    img = viewer.grabFramebuffer()
                    if not img.isNull():
                        mo_obj = self._orbitals[mo_idx]
                        ev_val = float(mo_obj.energy) * 27.211386

                        tag = f"MO #{mo_idx+1}"
                        if mo_idx == self._homo_idx:
                            tag += " (HOMO)"
                        elif self._homo_idx is not None and mo_idx == self._homo_idx + 1:
                            tag += " (LUMO)"

                        label = f"{tag}\nE = {ev_val:+.2f} eV"
                        self._cache_images[mo_idx] = (img.copy(), ev_val, label)
                        self._add_thumbnail_to_grid(mo_idx, img, label)
                else:
                    self._last_render_error = (
                        f"Could not evaluate 3D basis/MO grid from '{os.path.basename(self._molden_path)}'."
                    )
            except Exception as ex:
                self._last_render_error = str(ex)

            idx_cursor[0] += 1
            self.progress_bar.setValue(idx_cursor[0])
            QtCore.QTimer.singleShot(15, _step)

        QtCore.QTimer.singleShot(10, _step)

    def _add_thumbnail_to_grid(self, mo_idx: int, qimg: QtGui.QImage, label_text: str):
        col_setting = self.combo_cols.currentIndex()
        if col_setting == 0:
            num_items = len(self._cache_images)
            cols = max(2, min(5, math.ceil(math.sqrt(max(4, num_items)))))
        else:
            cols = col_setting + 1

        curr_count = len(self._cache_images) - 1
        row = curr_count // cols
        col = curr_count % cols

        card = QtWidgets.QFrame()
        card.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        card.setStyleSheet(
            "QFrame { background-color: #1e293b; border: 1px solid #334155; border-radius: 6px; padding: 4px; }"
            "QFrame:hover { border: 1.5px solid #38bdf8; }"
        )
        card_layout = QtWidgets.QVBoxLayout(card)
        card_layout.setContentsMargins(4, 4, 4, 4)
        card_layout.setSpacing(4)

        # Scaled thumbnail image label
        lbl_img = QtWidgets.QLabel()
        lbl_img.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        thumb_size = 180
        pix = QtGui.QPixmap.fromImage(qimg).scaled(
            thumb_size, thumb_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation
        )
        lbl_img.setPixmap(pix)
        lbl_img.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        # Clicking thumbnail activates that orbital in main window
        lbl_img.mousePressEvent = lambda e, idx=mo_idx: self._select_orbital_in_main(idx)
        card_layout.addWidget(lbl_img)

        # Text caption
        lbl_txt = QtWidgets.QLabel(label_text)
        lbl_txt.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        lbl_txt.setStyleSheet("color: #e2e8f0; font-size: 11px; font-weight: bold; background: transparent; border: none;")
        card_layout.addWidget(lbl_txt)

        self.grid_layout.addWidget(card, row, col)

    def _select_orbital_in_main(self, mo_idx: int):
        if hasattr(self._parent_mw, "table_orbitals"):
            self._parent_mw.table_orbitals.selectRow(mo_idx)

    def _finish_generate(self):
        self._is_generating = False
        self.btn_generate.setText("⚡ Re-Generate Snapshots")
        self.btn_generate.setStyleSheet("background-color: #2563eb; color: #ffffff; font-weight: bold; padding: 7px;")
        self.progress_bar.setVisible(False)
        has_imgs = bool(self._cache_images)
        self.btn_save_sheet.setEnabled(has_imgs)
        if hasattr(self, "btn_copy_sheet"):
            self.btn_copy_sheet.setEnabled(has_imgs)
        self.btn_save_zip.setEnabled(has_imgs)
        if not has_imgs and getattr(self, "_last_render_error", None):
            self.lbl_stats.setText(f"⚠️ 0 snapshots generated: {self._last_render_error}")
        else:
            self.lbl_stats.setText(f"Completed {len(self._cache_images)} orbital snapshots.")

    def _create_composite_contact_sheet(self) -> QtGui.QImage:
        """Compose all cached orbital snapshots into a high-res publication-ready contact sheet grid."""
        if not self._cache_images:
            return QtGui.QImage()

        items = sorted(self._cache_images.items(), key=lambda x: x[0])
        num_items = len(items)

        col_setting = self.combo_cols.currentIndex()
        if col_setting == 0:
            cols = max(2, min(5, math.ceil(math.sqrt(num_items))))
        else:
            cols = col_setting + 1

        rows = math.ceil(num_items / cols)

        cell_w = 400
        cell_h = 400
        caption_h = 50
        header_h = 60
        margin = 20

        total_w = margin * 2 + cols * cell_w + (cols - 1) * margin
        total_h = header_h + margin * 2 + rows * (cell_h + caption_h) + (rows - 1) * margin

        is_white = ("white" in self.combo_bg.currentText().lower())
        bg_col = QtGui.QColor("#ffffff") if is_white else QtGui.QColor("#0f172a")
        card_bg = QtGui.QColor("#f8fafc") if is_white else QtGui.QColor("#1e293b")
        card_border = QtGui.QColor("#cbd5e1") if is_white else QtGui.QColor("#334155")
        text_col = QtGui.QColor("#0f172a") if is_white else QtGui.QColor("#f8fafc")
        subtext_col = QtGui.QColor("#475569") if is_white else QtGui.QColor("#94a3b8")

        sheet = QtGui.QImage(total_w, total_h, QtGui.QImage.Format.Format_ARGB32)
        sheet.fill(bg_col)

        painter = QtGui.QPainter(sheet)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)

        # Main Title Banner
        bname = os.path.basename(self._molden_path)
        f_title = QtGui.QFont("Sans", 16, QtGui.QFont.Weight.Bold)
        painter.setFont(f_title)
        painter.setPen(text_col)
        painter.drawText(QtCore.QRectF(margin, margin // 2, total_w - 2 * margin, 32), QtCore.Qt.AlignmentFlag.AlignCenter, f"Molecular Orbital Contact Sheet — {bname}")

        f_sub = QtGui.QFont("Sans", 10)
        painter.setFont(f_sub)
        painter.setPen(subtext_col)
        iso = self.spin_iso.value()
        painter.drawText(QtCore.QRectF(margin, margin // 2 + 30, total_w - 2 * margin, 20), QtCore.Qt.AlignmentFlag.AlignCenter, f"Isovalue = ±{iso:.3f} | Rendered with AgyMolden")

        # Draw Cells
        f_card = QtGui.QFont("Sans", 11, QtGui.QFont.Weight.Bold)
        f_energy = QtGui.QFont("Monospace", 10)

        for i, (mo_idx, (qimg, ev_val, label)) in enumerate(items):
            r = i // cols
            c = i % cols

            x = margin + c * (cell_w + margin)
            y = header_h + margin + r * (cell_h + caption_h + margin)

            cell_rect = QtCore.QRectF(x, y, cell_w, cell_h + caption_h)

            # Draw card background
            painter.setPen(QtGui.QPen(card_border, 1.5))
            painter.setBrush(QtGui.QBrush(card_bg))
            painter.drawRoundedRect(cell_rect, 8, 8)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

            # Draw 3D orbital snapshot image
            img_rect = QtCore.QRectF(x + 6, y + 6, cell_w - 12, cell_h - 12)
            painter.drawImage(img_rect, qimg)

            # Draw caption
            tag = f"MO #{mo_idx+1}"
            if mo_idx == self._homo_idx:
                tag += " [HOMO]"
                tag_col = QtGui.QColor("#2563eb")
            elif self._homo_idx is not None and mo_idx == self._homo_idx + 1:
                tag += " [LUMO]"
                tag_col = QtGui.QColor("#9333ea")
            else:
                tag_col = text_col

            painter.setFont(f_card)
            painter.setPen(tag_col)
            painter.drawText(QtCore.QRectF(x, y + cell_h, cell_w, 22), QtCore.Qt.AlignmentFlag.AlignCenter, tag)

            painter.setFont(f_energy)
            painter.setPen(subtext_col)
            painter.drawText(QtCore.QRectF(x, y + cell_h + 22, cell_w, 20), QtCore.Qt.AlignmentFlag.AlignCenter, f"E = {ev_val:+.2f} eV")

        painter.end()
        return sheet

    def _on_save_contact_sheet(self):
        """Save the composite contact sheet grid to a high-res PNG image."""
        sheet = self._create_composite_contact_sheet()
        if sheet.isNull():
            return

        base_name = os.path.splitext(os.path.basename(self._molden_path))[0]
        default_name = f"{base_name}_all_orbitals_sheet.png"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Orbital Contact Sheet Collage",
            default_name,
            "PNG Image (*.png);;JPEG Image (*.jpg)",
        )
        if path:
            sheet.save(path)
            QtWidgets.QMessageBox.information(
                self,
                "Contact Sheet Saved",
                f"Successfully exported orbital snapshot collage ({sheet.width()}x{sheet.height()} px) to:\n{path}",
            )

    def _on_copy_contact_sheet(self):
        """Copy the composite contact sheet grid directly to the system clipboard."""
        sheet = self._create_composite_contact_sheet()
        if sheet.isNull():
            return
        QtWidgets.QApplication.clipboard().setImage(sheet)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "📋 Copied orbital contact sheet to clipboard!", self)
        self.lbl_stats.setText(f"📋 Copied {sheet.width()}×{sheet.height()} px contact sheet to clipboard.")

    def _on_export_individual_images(self):
        """Batch export each individual orbital snapshot image into a target folder."""
        if not self._cache_images:
            return

        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder to Save Individual Orbital Snapshots")
        if not folder:
            return

        base_name = os.path.splitext(os.path.basename(self._molden_path))[0]
        saved_count = 0
        items = sorted(self._cache_images.items(), key=lambda x: x[0])

        for mo_idx, (qimg, ev_val, _) in items:
            tag = f"MO_{mo_idx+1:03d}"
            if mo_idx == self._homo_idx:
                tag += "_HOMO"
            elif self._homo_idx is not None and mo_idx == self._homo_idx + 1:
                tag += "_LUMO"
            fname = f"{base_name}_{tag}_{ev_val:+.2f}eV.png".replace(" ", "_")
            out_path = os.path.join(folder, fname)
            qimg.save(out_path, "PNG")
            saved_count += 1

        QtWidgets.QMessageBox.information(
            self,
            "Export Complete",
            f"Successfully saved {saved_count} individual orbital snapshots to:\n{folder}",
        )
