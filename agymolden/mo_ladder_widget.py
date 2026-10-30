"""
Interactive Molecular Orbital Energy-Level Ladder Diagram Widget.

Displays frontier molecular orbitals (HOMO-n to LUMO+m) as horizontal energy rungs
along a vertical energy axis (in eV), with electron spin arrows (↑↓) for occupied MOs,
highlighted HOMO-LUMO gap, degenerate/near-degenerate orbital splitting, core-orbital
exclusion / toggle, and click-to-select interaction synced with the MO table.
"""

from typing import List, Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets


class MOEnergyLadderWidget(QtWidgets.QWidget):
    """Interactive graphical energy-level diagram for Molecular Orbitals."""

    orbital_clicked = QtCore.pyqtSignal(int)  # 0-based MO index

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setMinimumWidth(165)
        self.setMinimumHeight(240)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.setToolTip(
            "MO Energy Ladder:\n"
            "• Click any rung to select and render that orbital\n"
            "• Degenerate/close orbitals are displayed side-by-side\n"
            "• Core orbitals (< -50 eV) are grouped to prevent scale compression\n"
            "• Click top-right [Val / All] button to toggle core orbital display"
        )

        # List of (mo_idx, energy_ev, occupancy, label_sym, is_core)
        self._orbitals: List[Tuple[int, float, float, str, bool]] = []
        self._homo_idx: Optional[int] = None
        self._selected_idx: Optional[int] = None
        self._show_all: bool = False  # False = Valence/Frontier focus; True = Show all (including deep core)

        # Clickable areas: list of (mo_idx, QRectF)
        self._rung_rects: List[Tuple[int, QtCore.QRectF]] = []
        self._btn_mode_rect: Optional[QtCore.QRectF] = None
        self._core_badge_rect: Optional[QtCore.QRectF] = None

    def set_orbitals(self, orbitals: List, homo_idx: Optional[int] = None):
        """Populate diagram from MoldenMO objects."""
        raw_list = []
        for idx, mo in enumerate(orbitals):
            ev_val = float(mo.energy) * 27.211386
            occ = float(getattr(mo, "occupancy", 0.0))
            sym = str(getattr(mo, "sym", f"MO {idx+1}"))
            raw_list.append((idx, ev_val, occ, sym))

        # Determine HOMO reference energy
        homo_energy = None
        if homo_idx is not None and 0 <= homo_idx < len(raw_list):
            homo_energy = raw_list[homo_idx][1]
        elif raw_list:
            # Fallback: highest occupied
            for i, e, occ, _ in raw_list:
                if occ > 0.05:
                    homo_energy = e

        # Classify core orbitals:
        # Quantum chemical core threshold: E < -50 eV AND at least 30 eV deeper than HOMO
        self._orbitals = []
        for idx, ev_val, occ, sym in raw_list:
            is_core = False
            if occ > 0.05:  # Only occupied MOs are considered core
                if ev_val < -50.0:
                    if homo_energy is not None:
                        is_core = (homo_energy - ev_val) > 28.0
                    else:
                        is_core = True
            self._orbitals.append((idx, ev_val, occ, sym, is_core))

        self._homo_idx = homo_idx
        self._selected_idx = homo_idx
        self.update()

    def set_selected_orbital(self, mo_idx: int):
        if self._selected_idx != mo_idx:
            self._selected_idx = mo_idx
            # If the selected orbital is a core orbital and we are in valence mode, auto-expand to show it
            if 0 <= mo_idx < len(self._orbitals) and self._orbitals[mo_idx][4] and not self._show_all:
                self._show_all = True
            self.update()

    def toggle_show_all(self):
        """Toggle between Valence/Frontier view and Full energy range."""
        self._show_all = not self._show_all
        self.update()

    def _get_visible_orbitals(self) -> List[Tuple[int, float, float, str, bool]]:
        """Return orbitals to display based on mode."""
        if not self._orbitals:
            return []

        if self._show_all:
            # When viewing all, limit to at most 15 below HOMO and 16 above LUMO if molecule is huge
            n = len(self._orbitals)
            center = self._homo_idx if self._homo_idx is not None else (n // 2)
            lo = max(0, center - 15)
            hi = min(n, center + 16)
            return self._orbitals[lo:hi]

        # Valence / Frontier mode: exclude core orbitals
        # Center around HOMO
        n = len(self._orbitals)
        center = self._homo_idx if self._homo_idx is not None else (n // 2)

        # Filter out core
        valence_mos = [item for item in self._orbitals if not item[4]]
        if not valence_mos:
            # In case all were somehow classified core (rare edge case), fall back to raw slice
            lo = max(0, center - 5)
            hi = min(n, center + 6)
            return self._orbitals[lo:hi]

        # In the valence list, find the window around HOMO
        try:
            val_center_idx = next(i for i, item in enumerate(valence_mos) if item[0] == center)
        except StopIteration:
            val_center_idx = len(valence_mos) // 2

        # Show up to 6 occupied valence MOs below HOMO and up to 8 virtual MOs above LUMO
        lo_v = max(0, val_center_idx - 6)
        hi_v = min(len(valence_mos), val_center_idx + 8)
        subset = valence_mos[lo_v:hi_v]

        # Ensure selected index is included if it's not a core orbital
        if self._selected_idx is not None and not any(item[0] == self._selected_idx for item in subset):
            sel_item = next((item for item in self._orbitals if item[0] == self._selected_idx), None)
            if sel_item and not sel_item[4]:
                subset.append(sel_item)
                subset.sort(key=lambda x: x[0])

        return subset

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        is_white = False
        p = self.parent()
        if p is not None:
            v = getattr(p, "viewer", None)
            if v and getattr(v, "bg_color_mode", "") == "white":
                is_white = True
            elif hasattr(p, "styleSheet") and "#f8fafc" in (p.styleSheet() or ""):
                is_white = True

        bg_col = QtGui.QColor("#ffffff") if is_white else QtGui.QColor("#0f172a")
        border_col = QtGui.QColor("#cbd5e1") if is_white else QtGui.QColor("#334155")
        text_title_col = QtGui.QColor("#0f172a") if is_white else QtGui.QColor("#cbd5e1")
        text_dim_col = QtGui.QColor("#64748b") if is_white else QtGui.QColor("#94a3b8")

        painter.fillRect(rect, bg_col)
        painter.setPen(QtGui.QPen(border_col, 1))
        painter.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 6, 6)

        self._rung_rects.clear()
        self._btn_mode_rect = None
        self._core_badge_rect = None

        if not self._orbitals:
            painter.setPen(text_dim_col)
            f = painter.font()
            f.setPointSize(8)
            painter.setFont(f)
            painter.drawText(
                rect.adjusted(8, 8, -8, -8),
                QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
                "MO Energy\nLadder\n(Load .molden)",
            )
            return

        visible = self._get_visible_orbitals()
        core_count = sum(1 for item in self._orbitals if item[4])

        # Header Title
        f_title = QtGui.QFont("Sans", 8, QtGui.QFont.Weight.Bold)
        painter.setFont(f_title)
        painter.setPen(text_title_col)
        painter.drawText(
            QtCore.QRectF(4, 5, rect.width() - 56, 16),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            " MO Ladder (eV)",
        )

        # Toggle Button [Valence / All]
        btn_w = 46.0
        btn_h = 16.0
        btn_x = rect.width() - btn_w - 6.0
        btn_y = 5.0
        self._btn_mode_rect = QtCore.QRectF(btn_x, btn_y, btn_w, btn_h)

        btn_bg = QtGui.QColor("#e2e8f0") if is_white else QtGui.QColor("#1e293b")
        btn_border = QtGui.QColor("#94a3b8") if is_white else QtGui.QColor("#475569")
        btn_text = QtGui.QColor("#0284c7") if is_white else QtGui.QColor("#38bdf8")

        painter.setPen(QtGui.QPen(btn_border, 1))
        painter.setBrush(QtGui.QBrush(btn_bg))
        painter.drawRoundedRect(self._btn_mode_rect, 3, 3)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        f_btn = QtGui.QFont("Sans", 7, QtGui.QFont.Weight.Bold)
        painter.setFont(f_btn)
        painter.setPen(btn_text)
        mode_label = "ALL" if self._show_all else "VAL"
        painter.drawText(self._btn_mode_rect, QtCore.Qt.AlignmentFlag.AlignCenter, mode_label)

        if not visible:
            return

        top_pad = 28.0
        bot_pad = 24.0 if (core_count > 0 and not self._show_all) else 14.0
        avail_h = max(60.0, rect.height() - top_pad - bot_pad)

        # Group orbitals by energy degeneracy: orbitals within 0.06 eV of each other share the same level cluster
        sorted_visible = sorted(visible, key=lambda x: x[1])
        energy_groups: List[List[Tuple[int, float, float, str, bool]]] = []
        for mo in sorted_visible:
            if not energy_groups:
                energy_groups.append([mo])
            else:
                last_group = energy_groups[-1]
                avg_e = sum(item[1] for item in last_group) / len(last_group)
                if abs(mo[1] - avg_e) <= 0.06:
                    last_group.append(mo)
                else:
                    energy_groups.append([mo])

        group_energies = [sum(item[1] for item in g) / len(g) for g in energy_groups]
        e_min = min(group_energies)
        e_max = max(group_energies)
        span = max(0.5, e_max - e_min)

        def _calc_raw_y(e_val: float) -> float:
            frac = (e_val - e_min) / span
            return top_pad + avail_h * (1.0 - frac)

        # Compute initial group Y positions
        group_y_map = {}
        for g, avg_e in zip(energy_groups, group_energies):
            group_y_map[avg_e] = _calc_raw_y(avg_e)

        # Enforce minimum vertical separation between distinct energy levels (at least 15 px)
        min_spacing = 15.0
        for i in range(1, len(group_energies)):
            prev_e = group_energies[i - 1]
            curr_e = group_energies[i]
            y_prev = group_y_map[prev_e]
            y_curr = group_y_map[curr_e]
            if (y_prev - y_curr) < min_spacing:
                group_y_map[curr_e] = y_prev - min_spacing

        # If highest level drifted past top_pad, normalize back into [top_pad, top_pad + avail_h]
        min_y_found = min(group_y_map.values())
        max_y_found = max(group_y_map.values())
        y_span = max_y_found - min_y_found
        if y_span > avail_h or min_y_found < top_pad:
            for k in group_y_map:
                rel = (group_y_map[k] - min_y_found) / max(1.0, y_span)
                group_y_map[k] = top_pad + rel * avail_h

        def _y_for_e(e_val: float) -> float:
            closest_avg = min(group_energies, key=lambda ge: abs(ge - e_val))
            return group_y_map[closest_avg]

        # Draw HOMO-LUMO gap shading if both are present in visible subset
        if self._homo_idx is not None:
            homo_item = next((item for item in visible if item[0] == self._homo_idx), None)
            lumo_item = next((item for item in visible if item[0] == self._homo_idx + 1), None)
            if homo_item is not None and lumo_item is not None:
                y_h = _y_for_e(homo_item[1])
                y_l = _y_for_e(lumo_item[1])
                y_top = min(y_h, y_l)
                h_gap = max(4.0, abs(y_h - y_l))
                gap_rect = QtCore.QRectF(10, y_top, rect.width() - 20, h_gap)
                gap_bg = QtGui.QColor(2, 132, 199, 32) if is_white else QtGui.QColor(56, 189, 248, 26)
                painter.fillRect(gap_rect, gap_bg)

                gap_ev = lumo_item[1] - homo_item[1]
                f_gap = QtGui.QFont("Monospace", 7, QtGui.QFont.Weight.Bold)
                painter.setFont(f_gap)
                painter.setPen(QtGui.QColor("#0284c7") if is_white else QtGui.QColor("#38bdf8"))
                painter.drawText(
                    gap_rect,
                    QtCore.Qt.AlignmentFlag.AlignCenter,
                    f"ΔE={gap_ev:.2f} eV",
                )

        total_rung_x1 = 38.0
        total_rung_x2 = max(total_rung_x1 + 36.0, rect.width() - 44.0)
        rung_total_width = total_rung_x2 - total_rung_x1

        f_lbl = QtGui.QFont("Monospace", 7)
        painter.setFont(f_lbl)

        # Draw each energy level group
        for g in energy_groups:
            k = len(g)
            y = _y_for_e(g[0][1])
            avg_e = sum(item[1] for item in g) / k

            # Divide horizontal span into k sub-rungs with small gaps
            gap_w = 4.0 if k > 1 else 0.0
            sub_w = (rung_total_width - (k - 1) * gap_w) / k

            for seg_i, (idx, e_ev, occ, _sym, is_c) in enumerate(g):
                x1 = total_rung_x1 + seg_i * (sub_w + gap_w)
                x2 = x1 + sub_w
                mid_x = 0.5 * (x1 + x2)

                is_sel = (idx == self._selected_idx)
                is_occ = (occ > 0.05)

                if is_sel:
                    col = QtGui.QColor("#f59e0b")
                    lw = 3.0
                elif idx == self._homo_idx:
                    col = QtGui.QColor("#3b82f6")
                    lw = 2.4
                elif self._homo_idx is not None and idx == self._homo_idx + 1:
                    col = QtGui.QColor("#a855f7")
                    lw = 2.4
                elif is_c:
                    col = QtGui.QColor("#64748b")
                    lw = 1.4
                elif is_occ:
                    col = QtGui.QColor("#60a5fa")
                    lw = 1.8
                else:
                    col = QtGui.QColor("#475569") if is_white else QtGui.QColor("#94a3b8")
                    lw = 1.4

                painter.setPen(QtGui.QPen(col, lw))
                painter.drawLine(QtCore.QPointF(x1, y), QtCore.QPointF(x2, y))

                # Draw occupancy arrows on sub-rung
                arrow_col = QtGui.QColor("#0f172a") if is_white else QtGui.QColor("#f8fafc")
                painter.setPen(arrow_col)
                if occ >= 1.5:
                    painter.drawText(QtCore.QRectF(mid_x - 10, y - 9, 20, 10), QtCore.Qt.AlignmentFlag.AlignCenter, "↑↓")
                elif occ >= 0.5:
                    painter.drawText(QtCore.QRectF(mid_x - 10, y - 9, 20, 10), QtCore.Qt.AlignmentFlag.AlignCenter, "↑")

                hit_rect = QtCore.QRectF(x1 - 2, y - 7, sub_w + 4, 14)
                self._rung_rects.append((idx, hit_rect))

            # Left Label: MO tags (#, H, L)
            lbl_text_col = QtGui.QColor("#0f172a") if is_white else QtGui.QColor("#e2e8f0")
            painter.setPen(lbl_text_col)

            tags = []
            for idx, _, _, _, _ in g:
                if idx == self._homo_idx:
                    tags.append("H")
                elif self._homo_idx is not None and idx == self._homo_idx + 1:
                    tags.append("L")
                else:
                    tags.append(f"{idx+1}")
            tag_str = ",".join(tags)
            painter.drawText(
                QtCore.QRectF(2, y - 6, total_rung_x1 - 4, 12),
                QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
                tag_str,
            )

            # Right Label: Average energy of the group
            painter.drawText(
                QtCore.QRectF(total_rung_x2 + 3, y - 6, rect.width() - total_rung_x2 - 5, 12),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                f"{avg_e:+.1f}",
            )

        # Core Orbital Indicator Badge (at bottom if core orbitals exist and not in show_all mode)
        if core_count > 0 and not self._show_all:
            badge_h = 16.0
            badge_y = rect.height() - badge_h - 4.0
            self._core_badge_rect = QtCore.QRectF(8, badge_y, rect.width() - 16, badge_h)

            badge_bg = QtGui.QColor(226, 232, 240, 200) if is_white else QtGui.QColor(30, 41, 59, 200)
            badge_pen = QtGui.QColor("#94a3b8") if is_white else QtGui.QColor("#475569")
            painter.setPen(QtGui.QPen(badge_pen, 1, QtCore.Qt.PenStyle.DashLine))
            painter.setBrush(QtGui.QBrush(badge_bg))
            painter.drawRoundedRect(self._core_badge_rect, 3, 3)
            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

            painter.setPen(QtGui.QColor("#475569") if is_white else QtGui.QColor("#94a3b8"))
            f_core = QtGui.QFont("Sans", 7)
            painter.setFont(f_core)
            painter.drawText(
                self._core_badge_rect,
                QtCore.Qt.AlignmentFlag.AlignCenter,
                f"▼ {core_count} Core MOs (< -50 eV) hidden",
            )

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        pos = event.position()

        # Check toggle button click
        if self._btn_mode_rect and self._btn_mode_rect.contains(pos):
            self.toggle_show_all()
            return

        # Check core badge click
        if self._core_badge_rect and self._core_badge_rect.contains(pos):
            self.toggle_show_all()
            return

        # Check orbital rung clicks: find closest hit_rect
        best_idx = None
        best_dist = 999.0
        for idx, r in self._rung_rects:
            if r.contains(pos):
                best_idx = idx
                break
            dy = abs(pos.y() - r.center().y())
            dx = max(0.0, r.left() - pos.x(), pos.x() - r.right())
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist and dy <= 12.0 and dx <= 16.0:
                best_dist = dist
                best_idx = idx

        if best_idx is not None:
            self._selected_idx = best_idx
            self.orbital_clicked.emit(best_idx)
            self.update()
