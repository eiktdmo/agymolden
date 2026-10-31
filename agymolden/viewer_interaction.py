"""Mouse interaction, ray-casting atom/bond picking, and measurement helpers for MoleculeViewer."""

import math
from typing import List, Optional, Tuple
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from .elements import get_cov_radius, get_vdw_radius
from .zmatrix import calculate_bond_distance, calculate_angle, calculate_dihedral


class ViewerInteractionMixin:
    def mousePressEvent(self, event: QtGui.QMouseEvent):
        self.last_mouse_pos = event.pos()

        # Right-Click: Can be atom deletion (click on atom) or camera pan (drag on canvas)
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            clicked_atom = self._pick_atom(event.pos(), tolerance_px=32.0)
            self.right_press_pos = event.pos()
            self.right_press_atom_idx = clicked_atom
            self.right_dragged = False
            return

        # Left-Click
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check if user clicked on the in-viewport "Clear 3D Surface" HUD button
            clear_rect = getattr(self, "clear_surface_rect", None)
            if clear_rect is not None and getattr(self, "active_isosurfaces", None) and clear_rect.contains(event.pos()):
                self.active_isosurfaces = []
                self.clear_surface_rect = None
                self.clear_surface_hovered = False
                if hasattr(self, "surface_cleared"):
                    self.surface_cleared.emit()
                self.update()
                return

            # Check if user clicked on a 3D manipulation gizmo handle (X, Y, or Z)
            gizmo_rects = getattr(self, "gizmo_screen_rects", {}) if getattr(self, "show_manipulator_gizmo", False) else {}
            for ax_name, rect in gizmo_rects.items():
                if rect.contains(event.pos()):
                    self.active_gizmo_axis = ax_name
                    self.gizmo_drag_start_screen = event.pos()
                    valid_sel = [i for i in self.multi_selected if 0 <= i < len(self.coords)]
                    if not valid_sel and self.selected_atom_idx is not None and 0 <= self.selected_atom_idx < len(self.coords):
                        valid_sel = [self.selected_atom_idx]
                    self.gizmo_drag_indices = valid_sel
                    self.gizmo_drag_start_coords = {i: np.copy(self.coords[i]) for i in valid_sel}
                    self.update()
                    return

            # Alt + Left-Click on a bond: Start 3D Torsion / Rotamer drag
            app_mods = QtWidgets.QApplication.keyboardModifiers()
            is_alt_down = bool(
                (event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) or
                (app_mods & QtCore.Qt.KeyboardModifier.AltModifier)
            )
            if is_alt_down:
                clicked_bond_oriented = self._pick_bond(event.pos(), tolerance_px=24.0, oriented=True)
                if clicked_bond_oriented is not None:
                    u, v = clicked_bond_oriented
                    self.torsion_drag_active = True
                    self.torsion_drag_bond = (u, v)
                    self.torsion_drag_start_pos = event.pos()
                    self.torsion_drag_last_pos = event.pos()
                    self.selected_bond = (min(u, v), max(u, v))
                    self.multi_selected = [u, v]
                    self.update()
                    return

            clicked_bond = self._pick_bond(event.pos(), tolerance_px=24.0)

            clicked_atom = self._pick_atom(event.pos(), tolerance_px=32.0)

            if clicked_atom is not None:
                self.empty_press_pos = None
                if self.partial_reorder_mode:
                    self.atom_picked_for_partial_reorder.emit(clicked_atom)
                    self.update()
                    return

                if self.reordering_active:
                    self.atom_picked_for_reorder.emit(clicked_atom)
                    self.update()
                    return

                if self.picking_active:
                    self.atom_picked_for_connectivity.emit(clicked_atom)
                    self.update()
                    return

                app_mods = QtWidgets.QApplication.keyboardModifiers()
                is_ctrl_down = bool(
                    (event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier) or
                    (app_mods & QtCore.Qt.KeyboardModifier.ControlModifier)
                )

                if is_ctrl_down:
                    self.highlighted_feature_atoms = {}
                    # Filter any out-of-bounds indices in multi_selected
                    n_coords = len(self.coords)
                    self.multi_selected = [i for i in self.multi_selected if 0 <= i < n_coords]

                    # Ctrl + Left-Click: Multi-select/toggle atoms and prepare to drag if moved >= 6 px
                    was_already_sel = clicked_atom in self.multi_selected
                    self._ctrl_press_atom = clicked_atom
                    self._ctrl_press_was_selected = was_already_sel
                    self._ctrl_drag_started_emitted = False
                    if not was_already_sel:
                        self.multi_selected.append(clicked_atom)
                    self.selected_atom_idx = clicked_atom

                    self.atom_move_drag_active = True
                    self.atom_move_drag_idx = clicked_atom
                    self.atom_move_drag_indices = [idx for idx in self.multi_selected if 0 <= idx < n_coords] or [clicked_atom]
                    self.atom_move_drag_start_screen = event.pos()
                    self.atom_move_drag_start_coord = np.copy(self.coords[clicked_atom])
                    self.atom_move_drag_start_coords = {idx: np.copy(self.coords[idx]) for idx in self.atom_move_drag_indices if 0 <= idx < n_coords}
                    self.smartslide_selection_active = len(self.multi_selected) >= 2
                    self._update_measurement_info()
                elif event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                    self.highlighted_feature_atoms = {}
                    # Toggle in multi_selected
                    if clicked_atom in self.multi_selected:
                        self.multi_selected.remove(clicked_atom)
                        if self.selected_atom_idx == clicked_atom:
                            self.selected_atom_idx = self.multi_selected[0] if self.multi_selected else None
                    else:
                        self.multi_selected.append(clicked_atom)
                        self.selected_atom_idx = clicked_atom
                    self.smartslide_selection_active = len(self.multi_selected) >= 2
                    self._update_measurement_info()
                else:
                    # Plain Left-Click on an atom:
                    self.selected_bond = None
                    self.highlighted_feature_atoms = {}
                    self.smartslide_selection_active = False
                    # If clicked on the already-selected single atom -> unselect it!
                    if clicked_atom == self.selected_atom_idx and len(self.multi_selected) <= 1:
                        self.selected_atom_idx = None
                        self.multi_selected = []
                        self._update_measurement_info()
                        self.atom_clicked.emit(clicked_atom, event.button())
                        self.update()
                        return
                    else:
                        self.selected_atom_idx = clicked_atom
                        self.multi_selected = [clicked_atom]
                        self._update_measurement_info()
                        # Only start Avogadro-style atom addition drag when in Build mode
                        if getattr(self, "interaction_mode", "build") == "build":
                            self.atom_add_drag_active = True
                            self.atom_add_drag_anchor_idx = clicked_atom
                            self.atom_add_drag_start_pos = event.pos()
                            self.atom_add_drag_current_pos = event.pos()
                        else:
                            self.atom_add_drag_active = False
                            self.atom_add_drag_anchor_idx = None

                self.atom_clicked.emit(clicked_atom, event.button())
                self.update()
            else:
                self.empty_press_pos = event.pos()
                # Shift/Ctrl + drag on empty canvas: rubber-band region selection
                mods = event.modifiers() | QtWidgets.QApplication.keyboardModifiers()
                if (mods & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.ControlModifier)) \
                        and not (self.partial_reorder_mode or self.reordering_active or self.picking_active):
                    self.band_active = True
                    self.band_additive = bool(mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
                    self.band_start = event.pos()
                    self.band_current = event.pos()

    def _finish_band_selection(self):
        """Select every atom whose projected screen position lies inside the rubber band."""
        self.band_active = False
        if self.band_start is None or self.band_current is None or len(self.coords) == 0:
            self.update()
            return
        x0, x1 = sorted((self.band_start.x(), self.band_current.x()))
        y0, y1 = sorted((self.band_start.y(), self.band_current.y()))
        if (x1 - x0) < 3 and (y1 - y0) < 3:
            self.update()
            return
        mvp = self._get_projection_matrix() * self._get_view_matrix()
        w, h = self.width(), self.height()
        inside = []
        for i, p in enumerate(self.coords):
            p4 = mvp * QtGui.QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0)
            if p4.w() <= 0.01:
                continue
            ndc = p4 / p4.w()
            sx = (ndc.x() * 0.5 + 0.5) * w
            sy = (-ndc.y() * 0.5 + 0.5) * h
            if x0 <= sx <= x1 and y0 <= sy <= y1:
                inside.append(i)
        n = len(self.coords)
        self.highlighted_feature_atoms = {}
        base = [i for i in self.multi_selected if 0 <= i < n] if self.band_additive else []
        self.multi_selected = base + [i for i in inside if i not in base]
        self.selected_atom_idx = self.multi_selected[0] if self.multi_selected else None
        self.smartslide_selection_active = len(self.multi_selected) >= 2
        self._update_measurement_info()
        self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        # 1. Active 3D Gizmo Drag (Constrained Translation along X, Y, or Z in World Space)
        if getattr(self, "active_gizmo_axis", None) and hasattr(self, "gizmo_drag_start_coords"):
            ax_name = self.active_gizmo_axis
            dx = float(event.pos().x() - self.gizmo_drag_start_screen.x())
            dy = float(event.pos().y() - self.gizmo_drag_start_screen.y())
            right_w, up_w = self._get_camera_axes()
            h = max(1, self.height())
            scale = (self.zoom * 0.8) / h
            mouse_disp_w = (dx * scale) * right_w - (dy * scale) * up_w

            # Constrain displacement strictly onto the chosen world Cartesian axis
            unit_ax = {"X": np.array([1.0, 0.0, 0.0]), "Y": np.array([0.0, 1.0, 0.0]), "Z": np.array([0.0, 0.0, 1.0])}.get(ax_name, np.zeros(3))
            proj_dist = float(np.dot(mouse_disp_w, unit_ax))
            constrained_disp = proj_dist * unit_ax

            for idx in self.gizmo_drag_indices:
                orig_c = self.gizmo_drag_start_coords.get(idx, self.coords[idx])
                self.coords[idx] = orig_c + constrained_disp

            self.compute_bonds()
            self.update()
            if self.gizmo_drag_indices:
                lead_idx = self.gizmo_drag_indices[0]
                nc = self.coords[lead_idx]
                self.atom_moved_by_drag.emit(lead_idx, float(nc[0]), float(nc[1]), float(nc[2]))
            return

        # Check hover on 3D gizmo handles, Clear Surface HUD button, and clickable atoms when not dragging
        if not (event.buttons() & (QtCore.Qt.MouseButton.LeftButton | QtCore.Qt.MouseButton.RightButton)):
            clear_rect = getattr(self, "clear_surface_rect", None)
            hover_clear = bool(clear_rect is not None and getattr(self, "active_isosurfaces", None) and clear_rect.contains(event.pos()))
            if hover_clear != getattr(self, "clear_surface_hovered", False):
                self.clear_surface_hovered = hover_clear
                self.update()
            if hover_clear:
                self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                return

            hover_ax = None
            if getattr(self, "show_manipulator_gizmo", False):
                for name, rect in getattr(self, "gizmo_screen_rects", {}).items():
                    if rect.contains(event.pos()):
                        hover_ax = name
                        break
            if hover_ax != getattr(self, "gizmo_hover_axis", None):
                self.gizmo_hover_axis = hover_ax
                self.update()

            # Set intuitive cursor shape on hover
            if hover_ax is not None:
                self.setCursor(QtCore.Qt.CursorShape.SizeAllCursor)
            else:
                hover_atom = self._pick_atom(event.pos(), tolerance_px=28.0)
                if hover_atom is not None:
                    self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                else:
                    hover_bond = self._pick_bond(event.pos(), tolerance_px=14.0)
                    if hover_bond is not None:
                        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
                    else:
                        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        if getattr(self, "band_active", False):
            self.band_current = event.pos()
            self.update()
            return
        # Handle active Alt + Left-Click torsion / rotamer drag around a bond
        if getattr(self, "torsion_drag_active", False) and hasattr(self, "torsion_drag_bond"):
            u, v = self.torsion_drag_bond
            dx = float(event.pos().x() - self.torsion_drag_last_pos.x())
            dy = float(event.pos().y() - self.torsion_drag_last_pos.y())
            self.torsion_drag_last_pos = event.pos()
            delta_deg = dx * 0.8 - dy * 0.4
            if abs(delta_deg) > 0.01:
                self.bond_torsion_dragged.emit(u, v, delta_deg)
            return

        # Handle active Ctrl + Left-Click drag to move existing atom or selected group in 3D camera plane
        if self.atom_move_drag_active and self.atom_move_drag_idx is not None:
            dx = float(event.pos().x() - self.atom_move_drag_start_screen.x())
            dy = float(event.pos().y() - self.atom_move_drag_start_screen.y())
            if not getattr(self, "_ctrl_drag_started_emitted", False):
                if abs(dx) + abs(dy) < 6.0:
                    return
                self._ctrl_drag_started_emitted = True
                self.atom_drag_started.emit(self.atom_move_drag_idx)
            right_w, up_w = self._get_camera_axes()
            h = max(1, self.height())
            scale = (self.zoom * 0.8) / h
            delta_3d = (dx * scale) * right_w - (dy * scale) * up_w

            # Translate all atoms in the active drag group
            indices = self.atom_move_drag_indices or [self.atom_move_drag_idx]
            for idx in indices:
                orig_pos = self.atom_move_drag_start_coords.get(idx, self.coords[idx])
                self.coords[idx] = orig_pos + delta_3d

            self.compute_bonds()
            self.update()
            new_c = self.coords[self.atom_move_drag_idx]
            self.atom_moved_by_drag.emit(self.atom_move_drag_idx, float(new_c[0]), float(new_c[1]), float(new_c[2]))
            return

        # Handle active Left-Click drag from an atom (elastic line to add atom)
        if self.atom_add_drag_active:
            self.atom_add_drag_current_pos = event.pos()
            self.update()
            return

        dx = event.position().x() - self.last_mouse_pos.x()
        dy = event.position().y() - self.last_mouse_pos.y()
        self.last_mouse_pos = event.pos()

        if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
            # Plain Left-Drag ALWAYS orbits the 3D camera (GaussView / Avogadro / Gabedit standard).
            # Alt + Left-Drag rotates the selected subgroup (> 1 atom) around its centroid.
            app_mods = QtWidgets.QApplication.keyboardModifiers()
            is_alt_down = bool(
                (event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) or
                (app_mods & QtCore.Qt.KeyboardModifier.AltModifier)
            )
            valid_sel = [i for i in self.multi_selected if 0 <= i < len(self.coords)]
            if is_alt_down and len(valid_sel) > 1:
                right_w, up_w = self._get_camera_axes()
                d_ang_pitch = dy * 0.008  # rotation around screen right axis
                d_ang_yaw = dx * 0.008    # rotation around screen up axis

                # Compute centroid of selected region
                sel_coords = self.coords[valid_sel]
                centroid = np.mean(sel_coords, axis=0)

                # Rodrigues rotation matrix for pitch and yaw
                def rodrigues_rot(axis: np.ndarray, theta: float) -> np.ndarray:
                    axis = axis / (np.linalg.norm(axis) + 1e-12)
                    c = math.cos(theta)
                    s = math.sin(theta)
                    k = np.array([
                        [0.0, -axis[2], axis[1]],
                        [axis[2], 0.0, -axis[0]],
                        [-axis[1], axis[0], 0.0]
                    ])
                    return np.eye(3) + s * k + (1.0 - c) * (k @ k)

                r_pitch = rodrigues_rot(right_w, d_ang_pitch)
                r_yaw = rodrigues_rot(up_w, d_ang_yaw)
                r_comb = r_yaw @ r_pitch

                # Apply rotation to selected atoms around centroid
                for idx in valid_sel:
                    v = self.coords[idx] - centroid
                    self.coords[idx] = centroid + r_comb @ v

                self.region_rotated_in_drag = True
                self.compute_bonds()
                self.update()
            else:
                # Orbit rotation when clicking/dragging empty canvas (whole system)
                # Apply screen-space rotation to quaternion so rotation direction is always intuitive and gimbal-lock free
                q_pitch = QtGui.QQuaternion.fromAxisAndAngle(QtGui.QVector3D(1.0, 0.0, 0.0), dy * 0.5)
                q_yaw = QtGui.QQuaternion.fromAxisAndAngle(QtGui.QVector3D(0.0, 1.0, 0.0), dx * 0.5)
                if not hasattr(self, "rot_quat") or self.rot_quat.isNull():
                    self.rot_quat = (
                        QtGui.QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, self.rot_x) *
                        QtGui.QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, self.rot_y)
                    )
                self.rot_quat = q_pitch * q_yaw * self.rot_quat
                self.rot_x = (self.rot_x + dy * 0.5) % 360.0
                self.rot_y = (self.rot_y + dx * 0.5) % 360.0
                self.update()
        elif event.buttons() & QtCore.Qt.MouseButton.RightButton:
            # Mark that right button has moved beyond threshold for camera panning/zooming
            r_dist = (event.pos() - self.right_press_pos).manhattanLength()
            if r_dist > 14:
                self.right_dragged = True

            mode = getattr(self, "mouse_mode", "standard")
            if mode == "gaussview":
                # GaussView style: Right-drag zooms camera
                self.zoom = max(1.0, min(150.0, self.zoom - dy * 0.15))
                self.update()
            else:
                # Standard / Chemcraft style: Right-drag pans camera
                factor = self.zoom * 0.0025
                self.pan_x += dx * factor
                self.pan_y -= dy * factor
                self.update()
        elif event.buttons() & QtCore.Qt.MouseButton.MiddleButton:
            mode = getattr(self, "mouse_mode", "standard")
            if mode == "gaussview":
                # GaussView style: Middle-drag pans camera
                factor = self.zoom * 0.0025
                self.pan_x += dx * factor
                self.pan_y -= dy * factor
                self.update()
            else:
                # Standard style: Middle-drag zooms
                self.zoom = max(1.0, min(150.0, self.zoom - dy * 0.1))
                self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        # Left-Click release
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            if getattr(self, "active_gizmo_axis", None):
                self.active_gizmo_axis = None
                self.gizmo_hover_axis = None
                self.update()
                if hasattr(self, "gizmo_drag_indices") and self.gizmo_drag_indices:
                    self.atom_drag_finished.emit(self.gizmo_drag_indices[0])
                return

            if getattr(self, "band_active", False):
                self.band_current = event.pos()
                self._finish_band_selection()
                return

            # Check if region was rotated during drag
            if getattr(self, "region_rotated_in_drag", False):
                self.region_rotated_in_drag = False
                self.region_rotation_finished.emit()
                self.update()
                # Clear empty_press_pos so selection is not cleared
                self.empty_press_pos = None
                return

            if getattr(self, "torsion_drag_active", False):
                self.torsion_drag_active = False
                self.torsion_drag_bond = None
                self.update()
                return

            if self.atom_move_drag_active:
                dragged_idx = self.atom_move_drag_idx
                was_dragged = bool(getattr(self, "_ctrl_drag_started_emitted", False))
                was_already_sel = bool(getattr(self, "_ctrl_press_was_selected", False))
                self.atom_move_drag_active = False
                self.atom_move_drag_idx = None
                self.atom_move_drag_indices = []
                self.atom_move_drag_start_coords = {}
                self._ctrl_press_atom = None
                self._ctrl_press_was_selected = False
                self._ctrl_drag_started_emitted = False
                if not was_dragged and was_already_sel and dragged_idx in self.multi_selected:
                    # Pure Ctrl+Click on an already-selected atom toggles it off
                    self.multi_selected.remove(dragged_idx)
                    if self.selected_atom_idx == dragged_idx:
                        self.selected_atom_idx = self.multi_selected[0] if self.multi_selected else None
                    self.smartslide_selection_active = len(self.multi_selected) >= 2
                    self._update_measurement_info()
                self.update()
                if was_dragged and dragged_idx is not None:
                    self.atom_drag_finished.emit(dragged_idx)
                return

            if self.atom_add_drag_active:
                anchor_idx = self.atom_add_drag_anchor_idx
                start_p = self.atom_add_drag_start_pos
                end_p = event.pos()
                self.atom_add_drag_active = False
                self.atom_add_drag_anchor_idx = None
                self.update()

                drag_dist = (end_p - start_p).manhattanLength()
                if anchor_idx is not None and drag_dist >= 12:
                    # Check if release position lands on another existing atom (Cross-bond / Ring closure)
                    target_atom = self._pick_atom(end_p, tolerance_px=30.0)
                    if target_atom is not None and target_atom != anchor_idx:
                        # Dropped directly on another existing atom! Connect them!
                        self.cross_bond_drawn.emit(anchor_idx, target_atom)
                    else:
                        # Meaningful drag into empty space: Add new atom bound to anchor_idx along the drag direction
                        dx = float(end_p.x() - start_p.x())
                        dy = float(end_p.y() - start_p.y())
                        self.quick_add_atom_dragged.emit(anchor_idx, dx, dy)
                elif anchor_idx is not None:
                    # Simple single click on atom (no significant drag): atom is already selected
                    pass
                return

            # Check if this was a plain click on empty canvas or bond (release without drag)
            if getattr(self, "empty_press_pos", None) is not None:
                start_p = self.empty_press_pos
                end_p = event.pos()
                drag_d = (end_p - start_p).manhattanLength()
                self.empty_press_pos = None
                app_mods = QtWidgets.QApplication.keyboardModifiers()
                mods = event.modifiers() | app_mods
                if drag_d < 6 and not (mods & (QtCore.Qt.KeyboardModifier.ShiftModifier | QtCore.Qt.KeyboardModifier.ControlModifier)):
                    clicked_bond = self._pick_bond(end_p, tolerance_px=18.0)
                    if clicked_bond is not None:
                        u, v = clicked_bond
                        self.selected_bond = (min(u, v), max(u, v))
                        bo_str = self._bond_order_name(u, v)
                        d = float(np.linalg.norm(self.coords[u] - self.coords[v]))
                        msg = f"Bond {self.symbols[u]}{u+1} - {self.symbols[v]}{v+1}: {d:.4f} Å (Bond Order: {bo_str})"
                        self.selected_atom_idx = u
                        self.multi_selected = [u, v]
                        self.measurement_updated.emit(msg)
                        self.bond_clicked.emit(u, v)
                        self.update()
                    else:
                        self.selected_bond = None
                        self.selected_atom_idx = None
                        self.multi_selected = []
                        self._update_measurement_info()
                        self.update()
                self.canvas_dragged_or_clicked.emit(start_p, end_p, drag_d)

        # Right-Click release: open context menu or emit right-click signal if not dragged
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            # Re-pick at release position to guarantee we catch small atoms or minor release shifts
            atom_at_release = self._pick_atom(event.pos(), tolerance_px=36.0)
            atom_target = self.right_press_atom_idx if self.right_press_atom_idx is not None else atom_at_release
            was_dragged = self.right_dragged
            self.right_press_atom_idx = None
            self.right_dragged = False
            if not was_dragged:
                global_pos = self.mapToGlobal(event.pos())
                if atom_target is not None:
                    # Select the atom if not already selected
                    self.selected_atom_idx = atom_target
                    if atom_target not in self.multi_selected:
                        self.multi_selected = [atom_target]
                    self.update()
                    self.atom_right_clicked.emit(atom_target, global_pos)
                else:
                    # Check if right-clicked directly on a bond
                    clicked_bond = self._pick_bond(event.pos(), tolerance_px=18.0)
                    if clicked_bond is not None:
                        u, v = clicked_bond
                        self.selected_bond = (min(u, v), max(u, v))
                        self.multi_selected = [u, v]
                        self.update()
                        self.bond_right_clicked.emit(u, v, global_pos)
                    else:
                        self.canvas_right_clicked.emit(global_pos)
                return

        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            # Check bond double-click first with a focused screen tolerance around the bond cylinder
            clicked_bond = self._pick_bond(event.pos(), tolerance_px=18.0)
            if clicked_bond is not None:
                self.bond_double_clicked.emit(clicked_bond[0], clicked_bond[1])
                return

            # Alt + Left-Click on a bond: Start 3D Torsion / Rotamer drag
            app_mods = QtWidgets.QApplication.keyboardModifiers()
            is_alt_down = bool(
                (event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier) or
                (app_mods & QtCore.Qt.KeyboardModifier.AltModifier)
            )
            if is_alt_down:
                clicked_bond_oriented = self._pick_bond(event.pos(), tolerance_px=24.0, oriented=True)
                if clicked_bond_oriented is not None:
                    u, v = clicked_bond_oriented
                    self.torsion_drag_active = True
                    self.torsion_drag_bond = (u, v)
                    self.torsion_drag_start_pos = event.pos()
                    self.torsion_drag_last_pos = event.pos()
                    self.selected_bond = (min(u, v), max(u, v))
                    self.multi_selected = [u, v]
                    self.update()
                    return

            clicked_bond = self._pick_bond(event.pos(), tolerance_px=24.0)

            clicked_atom = self._pick_atom(event.pos(), tolerance_px=32.0)

            if clicked_atom is not None:
                self.atom_double_clicked.emit(clicked_atom)
            else:
                self.canvas_double_clicked.emit(event.pos())
            return
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent):
        delta = event.angleDelta().y() / 120.0
        self.zoom = max(1.0, min(150.0, self.zoom - delta * 0.8))
        self.update()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent):
        # Forward dropped file to MainWindow
        main_win = self.window()
        if hasattr(main_win, "open_path"):
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    main_win.open_path(url.toLocalFile())
                    event.acceptProposedAction()
                    return
        super().dropEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Quick Camera Hotkeys: 1 (Front/XY), 2 (Side/YZ), 3 (Top/XZ), 0 or Space (Reset/Center)
        key = event.key()
        if key == QtCore.Qt.Key.Key_1:
            self.set_camera_plane_view("XY")
            return
        elif key == QtCore.Qt.Key.Key_2:
            self.set_camera_plane_view("YZ")
            return
        elif key == QtCore.Qt.Key.Key_3:
            self.set_camera_plane_view("XZ")
            return
        elif key in (QtCore.Qt.Key.Key_0, QtCore.Qt.Key.Key_Space):
            self.auto_center()
            return
        super().keyPressEvent(event)

    def _pick_bond(self, pos: QtCore.QPoint, tolerance_px: float = 18.0, oriented: bool = False) -> Optional[Tuple[int, int]]:
        """
        Find the covalent bond whose cylinder segment is closest to the mouse cursor.
        If oriented=True, returns (u, v) where v is the atom end closer to the cursor,
        so that group/substituent operations rotate the side the user clicked on.
        """
        if len(self.coords) == 0 or len(self.bonds) == 0:
            return None

        render_coords = self.get_render_coords()
        proj_mat = self._get_projection_matrix()
        view_mat = self._get_view_matrix()
        mvp = proj_mat * view_mat
        w, h = self.width(), self.height()
        mx, my = float(pos.x()), float(pos.y())

        best_bond = None
        min_score = 1e9

        for i1, i2 in self.bonds:
            if i1 >= len(render_coords) or i2 >= len(render_coords):
                continue
            p1 = render_coords[i1]
            p2 = render_coords[i2]

            p4_1 = mvp * QtGui.QVector4D(float(p1[0]), float(p1[1]), float(p1[2]), 1.0)
            p4_2 = mvp * QtGui.QVector4D(float(p2[0]), float(p2[1]), float(p2[2]), 1.0)
            if p4_1.w() <= 0.01 or p4_2.w() <= 0.01:
                continue

            ndc1 = p4_1 / p4_1.w()
            ndc2 = p4_2 / p4_2.w()

            sx1 = (ndc1.x() * 0.5 + 0.5) * w
            sy1 = (-ndc1.y() * 0.5 + 0.5) * h
            sx2 = (ndc2.x() * 0.5 + 0.5) * w
            sy2 = (-ndc2.y() * 0.5 + 0.5) * h

            dx = sx2 - sx1
            dy = sy2 - sy1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq < 1e-4:
                dist = math.hypot(mx - sx1, my - sy1)
                t = 0.5
            else:
                t = ((mx - sx1) * dx + (my - sy1) * dy) / seg_len_sq
                if t < 0.08 or t > 0.92:
                    # Near atom centers, prefer atom picking
                    continue
                proj_x = sx1 + t * dx
                proj_y = sy1 + t * dy
                dist = math.hypot(mx - proj_x, my - proj_y)

            if dist <= tolerance_px:
                avg_depth = (ndc1.z() + ndc2.z()) * 0.5
                score = avg_depth * 40.0 + dist
                if score < min_score:
                    min_score = score
                    if oriented:
                        # If t > 0.5, the cursor is closer to i2, so i2 is the moving end: (i1, i2)
                        # If t <= 0.5, closer to i1, so i1 is the moving end: (i2, i1)
                        best_bond = (i1, i2) if t > 0.5 else (i2, i1)
                    else:
                        best_bond = (min(i1, i2), max(i1, i2))

        return best_bond

    def _pick_atom(self, pos: QtCore.QPoint, tolerance_px: float = 30.0) -> Optional[int]:
        """Find the atom closest to mouse position under depth test with dynamic screen tolerance."""
        if len(self.coords) == 0:
            return None

        proj_mat = self._get_projection_matrix()
        view_mat = self._get_view_matrix()
        mvp = proj_mat * view_mat
        w, h = self.width(), self.height()

        best_idx = None
        min_depth = 1e9
        best_dist_sq = 1e9

        for i, p in enumerate(self.coords):
            p4 = mvp * QtGui.QVector4D(float(p[0]), float(p[1]), float(p[2]), 1.0)
            if p4.w() <= 0.01:
                continue
            ndc = p4 / p4.w()
            sx = (ndc.x() * 0.5 + 0.5) * w
            sy = (-ndc.y() * 0.5 + 0.5) * h
            dist_sq = (sx - pos.x())**2 + (sy - pos.y())**2

            # Dynamic hit radius based on atom radius projected onto screen
            cov_r = get_cov_radius(self.symbols[i])
            approx_screen_radius = max(tolerance_px, (cov_r / max(self.zoom, 1.0)) * h * 0.45)
            allowed_dist_sq = approx_screen_radius * approx_screen_radius

            if dist_sq <= allowed_dist_sq:
                # Combine depth and 2D pixel distance: favor foreground atom and direct cursor hits
                score = ndc.z() * 50.0 + math.sqrt(dist_sq)
                if score < min_depth:
                    min_depth = score
                    best_idx = i
                    best_dist_sq = dist_sq

        return best_idx

    def _update_measurement_info(self):
        n = min(len(self.symbols), len(self.coords))
        valid_sel = [i for i in self.multi_selected if 0 <= i < n]
        if valid_sel != self.multi_selected:
            self.multi_selected = valid_sel
            if self.selected_atom_idx is not None and (self.selected_atom_idx < 0 or self.selected_atom_idx >= n):
                self.selected_atom_idx = valid_sel[0] if valid_sel else None

        k = len(self.multi_selected)
        atoms_str = ", ".join(f"{self.symbols[idx]}{idx+1}" for idx in self.multi_selected)
        if k == 2:
            a, b = self.multi_selected
            d = np.linalg.norm(self.coords[a] - self.coords[b])
            from .elements import get_cov_radius, get_vdw_radius
            sum_cov = get_cov_radius(self.symbols[a]) + get_cov_radius(self.symbols[b])
            sum_vdw = get_vdw_radius(self.symbols[a]) + get_vdw_radius(self.symbols[b])
            bond_state = "Bonded" if d <= sum_cov * 1.25 else ("Non-covalent/VdW" if d <= sum_vdw else "Separated")
            msg = f"Distance ({self.symbols[a]}{a+1} - {self.symbols[b]}{b+1}): {d:.4f} Å [{bond_state}, Σr_cov={sum_cov:.2f} Å, Σr_vdw={sum_vdw:.2f} Å]"
        elif k == 3:
            a, b, c = self.multi_selected
            v1 = self.coords[a] - self.coords[b]
            v2 = self.coords[c] - self.coords[b]
            cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
            deg = math.degrees(math.acos(np.clip(cos_th, -1.0, 1.0)))
            msg = f"Angle ({self.symbols[a]}{a+1} - {self.symbols[b]}{b+1} - {self.symbols[c]}{c+1}): {deg:.2f}° [Selected: {atoms_str}]"
        elif k == 4:
            a, b, c, d = self.multi_selected
            # Dihedral
            p0, p1, p2, p3 = self.coords[a], self.coords[b], self.coords[c], self.coords[d]
            b0 = -1.0 * (p1 - p0)
            b1 = p2 - p1
            b2 = p3 - p2
            b1_u = b1 / (np.linalg.norm(b1) + 1e-12)
            v = b0 - np.dot(b0, b1_u) * b1_u
            w = b2 - np.dot(b2, b1_u) * b1_u
            x = np.dot(v, w)
            y = np.dot(np.cross(b1_u, v), w)
            deg = math.degrees(math.atan2(y, x))
            msg = f"Dihedral ({self.symbols[a]}{a+1} - {self.symbols[b]}{b+1} - {self.symbols[c]}{c+1} - {self.symbols[d]}{d+1}): {deg:.2f}° [Selected: {atoms_str}]"
        elif k > 4:
            # Check if group has a centroid or is a ring
            sel_pts = self.coords[self.multi_selected]
            centroid = np.mean(sel_pts, axis=0)
            rg = np.mean(np.linalg.norm(sel_pts - centroid, axis=1))
            msg = f"Selected {k} atoms: {atoms_str} | Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}) Å (r_span={rg:.2f} Å) | Press Ctrl+C to copy"
        elif k == 1:
            idx = self.multi_selected[0]
            p = self.coords[idx]
            from .elements import get_cov_radius, get_vdw_radius, get_atomic_number
            cov = get_cov_radius(self.symbols[idx])
            vdw = get_vdw_radius(self.symbols[idx])
            z = get_atomic_number(self.symbols[idx])
            charge_info = ""
            charges_arr = self._charges_and_dipole()[0] if hasattr(self, "_charges_and_dipole") else None
            if charges_arr is not None and idx < len(charges_arr):
                charge_info = f" | Charge: {charges_arr[idx]:+.3f} e"
            msg = f"Atom #{idx+1} {self.symbols[idx]} (Z={z}): pos=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) Å | r_cov={cov:.2f} Å, r_vdw={vdw:.2f} Å{charge_info}"
        else:
            msg = "Shift+Click atoms to measure Dist (2), Angle (3), Dihedral (4) or group selection"

        self.measurement_updated.emit(msg)
