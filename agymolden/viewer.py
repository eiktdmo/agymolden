"""Modern OpenGL 3D Molecule Viewport using PyQt6 QtOpenGLWidgets and modern shaders."""

import ctypes
import logging
import math
import os
from typing import List, Optional, Tuple, Set, Dict, Any
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_ONE_MINUS_SRC_ALPHA, GL_SRC_ALPHA, GL_ZERO, GL_ONE, GL_COLOR_BUFFER_BIT, GL_COMPILE_STATUS, GL_DEPTH_BUFFER_BIT,
    GL_DEPTH_TEST, GL_ELEMENT_ARRAY_BUFFER, GL_FALSE, GL_FLOAT, GL_FRAGMENT_SHADER,
    GL_LINK_STATUS, GL_MULTISAMPLE, GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_TRIANGLES, GL_UNSIGNED_INT,
    GL_VERTEX_SHADER, glAttachShader, glBindBuffer, glBindVertexArray, glBlendFunc, glBlendFuncSeparate, glBufferData,
    glClear, glClearColor, glCompileShader, glCreateProgram, glCreateShader,
    glDeleteBuffers, glDeleteShader, glDeleteVertexArrays, glDrawElements, glDrawElementsInstanced, glEnable, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glGetProgramInfoLog, glGetProgramiv,
    glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram,
    glShaderSource, glUniform1i, glUniform1f, glUniform3f, glUniformMatrix3fv,
    glUniformMatrix4fv, glUseProgram, glVertexAttribPointer, glVertexAttribDivisor, glViewport,
    glDepthFunc, glDepthMask, glDisable, GL_LESS, GL_TRUE,
    GL_CULL_FACE, GL_SCISSOR_TEST, GL_STENCIL_TEST,
    glPolygonMode, GL_FRONT_AND_BACK, GL_LINE, GL_FILL,
    glCullFace, GL_FRONT, GL_BACK, GL_RENDERER, glGetString,
)

from .aromatic import find_delocalized_bonds
from .elements import get_color, get_cov_radius, get_vdw_radius
from .viewer_meshes import (
    create_sphere_mesh, create_cylinder_mesh, create_cone_mesh,
    VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC,
    SURFACE_VERTEX_SHADER, SURFACE_FRAGMENT_SHADER,
)
from .viewer_interaction import ViewerInteractionMixin

log = logging.getLogger(__name__)


class MoleculeViewer(ViewerInteractionMixin, QOpenGLWidget):
    atom_clicked = QtCore.pyqtSignal(int, QtCore.Qt.MouseButton) # atom_index, mouse_button
    atom_right_clicked = QtCore.pyqtSignal(int, QtCore.QPoint) # atom_index, global_screen_pos
    bond_right_clicked = QtCore.pyqtSignal(int, int, QtCore.QPoint) # atom1, atom2, global_screen_pos
    canvas_right_clicked = QtCore.pyqtSignal(QtCore.QPoint) # global_screen_pos
    atom_right_clicked_delete = QtCore.pyqtSignal(int) # atom_index to delete (legacy/direct)
    atom_picked_for_connectivity = QtCore.pyqtSignal(int)
    atom_picked_for_reorder = QtCore.pyqtSignal(int)
    atom_picked_for_partial_reorder = QtCore.pyqtSignal(int) # clicked atom during 'b', 'a', or 'd' mode
    quick_add_atom_dragged = QtCore.pyqtSignal(int, float, float) # anchor_atom_idx, screen_dx, screen_dy
    cross_bond_drawn = QtCore.pyqtSignal(int, int) # (atom1_idx, atom2_idx) when dragging from atom1 and dropping on atom2
    atom_drag_started = QtCore.pyqtSignal(int) # atom_idx when starting to drag an atom
    atom_moved_by_drag = QtCore.pyqtSignal(int, float, float, float) # atom_idx, new_x, new_y, new_z
    atom_drag_finished = QtCore.pyqtSignal(int) # atom_idx after drag completes
    region_rotation_finished = QtCore.pyqtSignal() # emitted when rotating a selected region finishes
    atom_double_clicked = QtCore.pyqtSignal(int) # atom_index double-clicked
    bond_clicked = QtCore.pyqtSignal(int, int) # (atom1_idx, atom2_idx) clicked
    bond_double_clicked = QtCore.pyqtSignal(int, int) # (atom1_idx, atom2_idx) double-clicked
    bond_torsion_dragged = QtCore.pyqtSignal(int, int, float) # (atom_u, atom_v, delta_deg)
    canvas_double_clicked = QtCore.pyqtSignal(QtCore.QPoint) # screen pos where empty canvas was double-clicked
    canvas_dragged_or_clicked = QtCore.pyqtSignal(QtCore.QPoint, QtCore.QPoint, int) # start_pos, end_pos, drag_distance
    surface_cleared = QtCore.pyqtSignal() # emitted when user clicks the in-viewport Clear Surface pill
    measurement_updated = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setUpdateBehavior(QOpenGLWidget.UpdateBehavior.PartialUpdate)

        # Molecule Data
        self.symbols: List[str] = []
        self.coords: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.bonds: List[Tuple[int, int]] = []
        self.bond_orders: Dict[Tuple[int, int], int] = {}
        self.strict_bonds: bool = False
        self._deloc_key = None
        self._deloc_cache: Dict = {}
        self.show_dipole: bool = False
        self.show_charges: bool = False
        self.show_hbonds: bool = False
        self.show_forces: bool = False
        self.nuclear_forces: Optional[np.ndarray] = None  # shape: (n_atoms, 3) in Ha/Bohr or eV/Angstrom
        self.force_scale: float = 2.5
        self.qtaim_critical_points: List[Any] = []  # List of CriticalPoint objects from Multiwfn/QTAIM
        self.show_qtaim_cps: bool = False
        self.qtaim_bond_paths: List[Any] = []        # List of 3D gradient bond path trajectories
        self.show_qtaim_bond_paths: bool = False
        self.unit_cell: Optional[Any] = None         # UnitCell object (Periodic Boundary Conditions)
        self.show_unit_cell: bool = True
        self.show_ghost_atoms: bool = False          # Translucent periodic boundary ghost atoms (VESTA / Mercury style)
        self.transparent_bg: bool = False
        self.bg_color_mode: str = "white"  # "dark" or "white" (default: white)
        self.mouse_mode: str = "standard"  # "standard", "gaussview", or "chemcraft"
        self._charge_key = None
        self._charge_cache = (np.empty(0), np.zeros(3), 0.0)
        self.custom_charges: Optional[np.ndarray] = None
        self.is_orthographic: bool = False  # False = Perspective, True = Orthographic (VMD/GaussView style)
        # Rubber-band (Shift/Ctrl + drag on empty canvas) selection state
        self.band_active: bool = False
        self.band_additive: bool = False
        self.band_start: Optional[QtCore.QPoint] = None
        self.band_current: Optional[QtCore.QPoint] = None
        # Track whether selected region was rotated during current mouse drag
        self.region_rotated_in_drag: bool = False
        # When True, bonds are exactly user_bonds (no distance-based detection)
        self.strict_bonds: bool = False
        self.user_bonds: Optional[List[Tuple[int, int]]] = None

        # Selection & Picking
        self.selected_atom_idx: Optional[int] = None
        self.multi_selected: List[int] = [] # For distance / angle / dihedral measurement
        self.pinned_measurements: List[List[int]] = [] # Pinned persistent measurements [[idx1, idx2], [idx1, idx2, idx3], ...] for publication figures
        self.frozen_atoms: set = set() # Constrained / frozen atom indices (Chemcraft/GaussView style)
        self.show_publication_outlines: bool = False # IboView/QuteMol style silhouette outlines & hemisphere lighting
        self.show_manipulator_gizmo: bool = False # 3D translation & rotation manipulator gizmo on selected atoms (off by default)
        self.active_gizmo_axis: Optional[str] = None # 'x', 'y', 'z' or None
        self.gizmo_hover_axis: Optional[str] = None
        self.gizmo_screen_rects: Dict[str, QtCore.QRect] = {}
        self.clear_surface_rect: Optional[QtCore.QRect] = None
        self.selected_bond: Optional[Tuple[int, int]] = None # Currently clicked bond (u, v) for bond order hotkeys (1, 2, 3)
        # Highlight atoms involved in currently selected Z-matrix feature:
        # dict mapping atom_idx -> role ("target", "bond", "angle", "dihedral")
        self.highlighted_feature_atoms: Dict[int, str] = {}

        # Molden interactive connectivity picking state
        self.picking_active: bool = False
        self.picking_banner: str = ""
        self.picked_references: List[int] = []

        # Molden interactive reorder Z-matrix state (ZMEORD)
        self.reordering_active: bool = False
        self.reorder_list: List[int] = []

        # Interactive partial Z-matrix adjustment mode ('b' for bond, 'a' for angle, 'd' for dihedral)
        self.partial_reorder_mode: Optional[str] = None # 'bond', 'angle', 'dihedral' or None
        self.partial_reorder_picks: List[int] = []
        self.partial_reorder_needed: int = 0
        self.partial_reorder_banner: str = ""

        # Left-Click Drag from atom state (Avogadro style create new atom + drag)
        self.atom_add_drag_active: bool = False
        self.atom_add_drag_anchor_idx: Optional[int] = None
        self.atom_add_drag_start_pos = QtCore.QPoint()
        self.atom_add_drag_current_pos = QtCore.QPoint()
        self.active_drag_label: str = ""

        # Ctrl + Left-Click Drag to translate/move existing atom or multi-selected atoms in 3D camera plane
        self.atom_move_drag_active: bool = False
        self.atom_move_drag_idx: Optional[int] = None
        self.atom_move_drag_indices: List[int] = []
        self.atom_move_drag_start_screen = QtCore.QPoint()
        self.atom_move_drag_start_coord = np.zeros(3, dtype=np.float64)
        self.atom_move_drag_start_coords: Dict[int, np.ndarray] = {}

        # Right-Click Press tracking for atom deletion vs view panning
        self.right_press_pos = QtCore.QPoint()
        self.right_press_atom_idx: Optional[int] = None
        self.right_dragged: bool = False

        # Active Element Brush (e.g. C, H, O, N, S, P, F, Cl)
        self.brush_element: str = "C"
        # Interaction Mode: "build" (click+drag atom sprouts bond/atom) vs "inspect" (click selects/measures, drag orbits camera)
        self.interaction_mode: str = "build"
        self._ctrl_press_atom: Optional[int] = None
        self._ctrl_press_was_selected: bool = False
        self._ctrl_drag_started_emitted: bool = False

        # Display Options
        self.show_labels: bool = True
        self.show_indices: bool = True
        self.show_measurements: bool = True
        self.show_bond_lengths: bool = False
        self.show_axis_triad: bool = True
        self.show_scale_bar: bool = False
        self.sphere_scale: float = 0.35
        self.cylinder_radius: float = 0.08
        self.style_mode: str = "Ball & Stick" # Ball & Stick, Spacefill, Wireframe
        self.use_modern_shaders: bool = True
        self.use_depth_cueing: bool = True
        self.show_publication_outlines: bool = False

        # Camera state
        self.rot_x: float = 15.0
        self.rot_y: float = -20.0
        self.rot_quat: QtGui.QQuaternion = (
            QtGui.QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, 15.0) *
            QtGui.QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, -20.0)
        )
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.zoom: float = 15.0

        # Mouse interaction
        self.last_mouse_pos = QtCore.QPoint()

        # OpenGL objects
        self.shader_program = 0
        self.sphere_vao = 0
        self.sphere_vbo = 0
        self.sphere_ebo = 0
        self.sphere_indices_count = 0
        self.sphere_inst_vbo = 0

        self.cyl_vao = 0
        self.cyl_vbo = 0
        self.cyl_ebo = 0
        self.cyl_indices_count = 0
        self.cyl_inst_vbo = 0

        self.cone_vao = 0
        self.cone_vbo = 0
        self.cone_ebo = 0
        self.cone_indices_count = 0

        # Vibrational mode animation state
        self._active_isosurfaces: List[Any] = []
        self._retired_surface_gl_caches: List[Tuple[int, int, int, int]] = []
        self.vibrating: bool = False
        self.vib_mode_idx: Optional[int] = None
        self.vib_frequency: float = 0.0
        self.vib_displacements: Optional[np.ndarray] = None  # shape: (n_atoms, 3)
        self.vib_amplitude: float = 0.5
        self.vib_vector_scale: float = 1.0
        self.vib_show_vectors: bool = True
        self.vib_phase: float = 0.0

        # Animation timer (approx 50-60 FPS)
        self.vib_timer = QtCore.QTimer(self)
        self.vib_timer.setInterval(20)  # 20 ms
        self.vib_timer.timeout.connect(self._on_vib_timer_tick)

    def _on_vib_timer_tick(self):
        if self.vibrating:
            # Advance phase smoothly
            self.vib_phase = (self.vib_phase + 0.15) % (2.0 * math.pi)
            self.update()

    def set_vibration_mode(self, mode_idx: int, freq: float, displacements: np.ndarray, amplitude: Optional[float] = None):
        """Set the active vibrational mode displacements and frequency without necessarily animating."""
        self.vib_mode_idx = mode_idx
        self.vib_frequency = freq
        self.vib_displacements = np.array(displacements, dtype=np.float64)
        if amplitude is not None:
            self.vib_amplitude = amplitude
        self.update()

    def start_vibration(self, mode_idx: int, freq: float, displacements: np.ndarray, amplitude: float = 0.5):
        """Start animated simulation of a vibrational normal mode."""
        self.vib_mode_idx = mode_idx
        self.vib_frequency = freq
        self.vib_displacements = np.array(displacements, dtype=np.float64)
        self.vib_amplitude = amplitude
        self.vibrating = True
        self.vib_phase = 0.0
        self.vib_timer.start()
        self.update()

    def stop_vibration(self, clear_mode: bool = False):
        """Stop vibrational simulation and return to equilibrium geometry."""
        self.vibrating = False
        self.vib_timer.stop()
        self.vib_phase = 0.0
        if clear_mode:
            self.vib_mode_idx = None
            self.vib_displacements = None
        self.update()

    def set_vibration_amplitude(self, amp: float):
        self.vib_amplitude = max(0.01, min(5.0, amp))
        self.update()

    def set_vibration_vector_scale(self, scale: float):
        self.vib_vector_scale = max(0.1, min(5.0, scale))
        self.update()

    def set_vibration_show_vectors(self, show: bool):
        self.vib_show_vectors = show
        self.update()

    @property
    def active_isosurfaces(self) -> List[Any]:
        return getattr(self, "_active_isosurfaces", [])

    @active_isosurfaces.setter
    def active_isosurfaces(self, value: List[Any]):
        new_list = list(value) if value else []
        new_ids = {id(m) for m in new_list}
        old_list = getattr(self, "_active_isosurfaces", [])
        retired = getattr(self, "_retired_surface_gl_caches", [])
        for old_m in old_list:
            if id(old_m) not in new_ids:
                gl_cache = getattr(old_m, "_gl_cache", None)
                if gl_cache is not None:
                    retired.append(gl_cache)
                    try:
                        old_m._gl_cache = None
                    except Exception as e:
                        log.debug("Failed to clear _gl_cache on surface: %s", e)
        self._retired_surface_gl_caches = retired
        self._active_isosurfaces = new_list

    def get_render_coords(self) -> np.ndarray:
        """Return instantaneous atom coordinates considering active vibrational mode displacement."""
        if (
            self.vibrating
            and self.vib_displacements is not None
            and len(self.vib_displacements) == len(self.coords)
        ):
            disp = math.sin(self.vib_phase) * self.vib_amplitude * self.vib_displacements
            return self.coords + disp
        return self.coords

    def _charges_and_dipole(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Return (charges_in_e, dipole_vector_in_Debye, dipole_norm_in_Debye), cached."""
        n = len(self.symbols)
        if n == 0 or len(self.coords) != n:
            return np.zeros(0), np.zeros(3), 0.0

        # If user/file provided custom quantum charges (e.g. CHELPG / Mulliken from .out or .molden)
        if self.custom_charges is not None and len(self.custom_charges) == n:
            q = np.array(self.custom_charges, dtype=np.float64)
            center = np.mean(self.coords, axis=0)
            mu_vec = 4.8032047 * np.sum(q[:, None] * (self.coords - center), axis=0)
            mu_mag = float(np.linalg.norm(mu_vec))
            return (q, mu_vec, mu_mag)

        try:
            key = (tuple(self.symbols), tuple(sorted(self.bonds)), np.round(self.coords, 2).tobytes())
        except Exception:
            key = None
        if key is not None and key == self._charge_key:
            return self._charge_cache
        try:
            from .forcefield import compute_gasteiger_charges
            adj = {i: [] for i in range(n)}
            for u, w in self.bonds:
                if 0 <= u < n and 0 <= w < n:
                    adj[u].append(w)
                    adj[w].append(u)
            hyb = {}
            for i, s in enumerate(self.symbols):
                deg = len(adj[i])
                if s == "C":
                    hyb[i] = "sp" if deg <= 2 else ("sp2" if deg == 3 else "sp3")
                elif s in ("N", "B"):
                    hyb[i] = "sp2" if deg <= 2 else "sp3"
                elif s == "O":
                    hyb[i] = "sp2" if deg == 1 else "sp3"
                else:
                    hyb[i] = "sp3"
            q = compute_gasteiger_charges(self.symbols, self.bonds, hyb)
            # 1 e * Angstrom = 4.8032047 Debye (pointing from negative to positive charge center: sum q_i r_i)
            center = np.mean(self.coords, axis=0)
            mu_vec = 4.8032047 * np.sum(q[:, None] * (self.coords - center), axis=0)
            mu_mag = float(np.linalg.norm(mu_vec))
            res = (q, mu_vec, mu_mag)
        except Exception as e:
            log.debug(f"Failed to compute Gasteiger charges: {e}", exc_info=True)
            res = (np.zeros(n), np.zeros(3), 0.0)
        self._charge_key, self._charge_cache = key, res
        return res

    @staticmethod
    def _charge_to_rgb(q_val: float) -> Tuple[float, float, float]:
        """Map partial charge in [-0.45, +0.45] e to Red (delta-) -> White (0) -> Blue (delta+)."""
        t = max(-1.0, min(1.0, float(q_val) / 0.40))
        if t < 0.0:
            # Red (0.95, 0.22, 0.22) to White (0.94, 0.94, 0.96)
            a = -t
            return (0.94 + 0.01 * a, 0.94 * (1.0 - 0.76 * a), 0.96 * (1.0 - 0.77 * a))
        else:
            # White (0.94, 0.94, 0.96) to Blue (0.20, 0.48, 0.98)
            a = t
            return (0.94 * (1.0 - 0.78 * a), 0.94 * (1.0 - 0.49 * a), 0.96 + 0.02 * a)

    def _delocalized_bonds(self):
        """Delocalized (aromatic / resonance) bonds for the current structure, cached."""
        try:
            key = (
                tuple(self.symbols),
                tuple(sorted(self.bonds)),
                tuple(sorted(self.bond_orders.items())),
                np.round(self.coords, 1).tobytes(),
            )
        except Exception:
            key = None
        if key is not None and key == getattr(self, "_deloc_key", None):
            return self._deloc_cache
        try:
            result = find_delocalized_bonds(self.symbols, self.coords, self.bonds, self.bond_orders)
        except Exception as e:
            log.debug(f"Failed to find delocalized bonds: {e}", exc_info=True)
            result = {}
        self._deloc_key, self._deloc_cache = key, result
        return result

    def _detect_hydrogen_bonds(self, coords: np.ndarray) -> List[Tuple[int, int, int]]:
        """
        Detect classic hydrogen bonds: D-H ... A
        where D and A are electronegative atoms (O, N, F, Cl, S),
        distance(H, A) is between 1.45 and 2.70 Angstroms,
        and angle(D-H...A) >= 110 degrees.
        Returns list of (idx_donor, idx_hydrogen, idx_acceptor).
        """
        hbonds = []
        n = len(self.symbols)
        if n < 3:
            return hbonds

        electronegative = {"O", "N", "F", "CL", "Cl", "S"}

        # Find covalent D-H bonds
        dh_pairs = []
        for u, v in self.bonds:
            if 0 <= u < n and 0 <= v < n:
                if self.symbols[u].upper() == "H" and self.symbols[v].upper() in electronegative:
                    dh_pairs.append((v, u))  # (D, H)
                elif self.symbols[v].upper() == "H" and self.symbols[u].upper() in electronegative:
                    dh_pairs.append((u, v))  # (D, H)

        for idx_d, idx_h in dh_pairs:
            ph = coords[idx_h]
            pd = coords[idx_d]
            v_dh = pd - ph
            len_dh = np.linalg.norm(v_dh)
            if len_dh < 1e-4:
                continue

            for idx_a, sym_a in enumerate(self.symbols):
                if idx_a == idx_d or idx_a == idx_h:
                    continue
                if sym_a.upper() not in electronegative:
                    continue

                pa = coords[idx_a]
                v_ha = pa - ph
                dist_ha = np.linalg.norm(v_ha)
                if 1.40 <= dist_ha <= 2.70:
                    # Angle D-H...A: angle between (pd - ph) and (pa - ph)
                    cos_angle = np.dot(v_dh, v_ha) / (len_dh * dist_ha)
                    angle_deg = math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))
                    if angle_deg >= 110.0:
                        hbonds.append((idx_d, idx_h, idx_a))

        return hbonds

    def _detect_non_covalent_interactions(self, coords: np.ndarray) -> List[Tuple[str, np.ndarray, np.ndarray, float]]:
        """
        Detect non-covalent interactions (NCI):
        1. Halogen bonds: C-X ... Y (where X = Cl, Br, I; Y = O, N, S; angle C-X...Y >= 140°, dist 2.4 - 3.8 A)
        2. Pi-Pi stacking: Centroid to centroid of aromatic rings (dist 3.2 - 4.4 A, angle between normals <= 30° or >= 60° for T-shaped)
        Returns list of (type_str, start_pos, end_pos, distance).
        """
        ncis = []
        n = len(self.symbols)
        if n < 4 or len(coords) != n:
            return ncis

        # 1. Halogen Bonds (sigma-hole)
        halogens = {"CL", "Cl", "BR", "Br", "I"}
        acceptors = {"O", "N", "S"}
        carbon_halogens = []
        for u, v in self.bonds:
            if 0 <= u < n and 0 <= v < n:
                if self.symbols[u].upper() == "C" and self.symbols[v] in halogens:
                    carbon_halogens.append((u, v))
                elif self.symbols[v].upper() == "C" and self.symbols[u] in halogens:
                    carbon_halogens.append((v, u))

        for idx_c, idx_x in carbon_halogens:
            pc, px = coords[idx_c], coords[idx_x]
            v_cx = px - pc
            len_cx = np.linalg.norm(v_cx)
            if len_cx < 1e-4:
                continue
            for idx_y, sym_y in enumerate(self.symbols):
                if idx_y == idx_c or idx_y == idx_x:
                    continue
                if sym_y not in acceptors:
                    continue
                py = coords[idx_y]
                v_xy = py - px
                dist_xy = float(np.linalg.norm(v_xy))
                if 2.4 <= dist_xy <= 3.8:
                    cos_th = np.dot(v_cx, v_xy) / (len_cx * dist_xy)
                    ang_deg = math.degrees(math.acos(np.clip(cos_th, -1.0, 1.0)))
                    if ang_deg >= 135.0:
                        ncis.append(("Halogen Bond", px, py, dist_xy))

        # 2. Pi-Pi Ring Stacking
        from .ring_perception import RingPerception
        try:
            rp = RingPerception(self.symbols, coords, self.bonds)
            rings = rp.find_rings(max_size=6)
            aromatic_rings = [r for r in rings if len(r) in (5, 6)]
            if len(aromatic_rings) >= 2:
                centroids = []
                normals = []
                for ring in aromatic_rings:
                    pts = coords[list(ring)]
                    c_pt = np.mean(pts, axis=0)
                    # SVD normal
                    svd = np.linalg.svd(pts - c_pt)
                    norm_vec = svd[2][2]
                    centroids.append(c_pt)
                    normals.append(norm_vec / np.linalg.norm(norm_vec))

                for r1 in range(len(centroids)):
                    for r2 in range(r1 + 1, len(centroids)):
                        c1, c2 = centroids[r1], centroids[r2]
                        d_cent = float(np.linalg.norm(c1 - c2))
                        if 3.2 <= d_cent <= 4.4:
                            dot_n = abs(float(np.dot(normals[r1], normals[r2])))
                            ang_n = math.degrees(math.acos(np.clip(dot_n, -1.0, 1.0)))
                            if ang_n <= 32.0 or ang_n >= 60.0:  # Parallel or T-shaped
                                ncis.append(("π-π Stacking", c1, c2, d_cent))
        except Exception:
            logging.getLogger(__name__).debug("Ignored exception", exc_info=True)

        return ncis

    def _bond_order_name(self, u: int, v: int) -> str:
        key = (min(u, v), max(u, v))
        if key in self._delocalized_bonds():
            return "Aromatic / delocalized, 1.5"
        bo = self.bond_orders.get(key, 1)
        return {1: "Single", 2: "Double", 3: "Triple"}.get(bo, f"{bo}-fold")

    def set_molecule(self, symbols: List[str], coords: np.ndarray, user_bonds: Optional[List[Tuple[int, int]]] = None):
        self.stop_vibration(clear_mode=True)
        self.symbols = list(symbols)
        self.coords = np.array(coords, dtype=np.float64) if len(coords) > 0 else np.empty((0, 3))
        self.user_bonds = list(user_bonds) if user_bonds is not None else None
        n = len(self.symbols)
        # Clean up any stale selection indices that exceed the new molecule size
        if self.selected_atom_idx is not None and (self.selected_atom_idx < 0 or self.selected_atom_idx >= n):
            self.selected_atom_idx = None
        self.multi_selected = [i for i in self.multi_selected if 0 <= i < n]
        if self.selected_bond is not None:
            bu, bv = self.selected_bond
            if bu < 0 or bu >= n or bv < 0 or bv >= n:
                self.selected_bond = None
        self.atom_move_drag_active = False
        self.atom_move_drag_idx = None
        self.atom_move_drag_indices = []
        self.atom_move_drag_start_coords = {}
        self.compute_bonds()
        self.fit_camera()
        self.update()

    def update_coords(self, coords: np.ndarray, user_bonds: Optional[List[Tuple[int, int]]] = None):
        """Update coordinates without resetting camera or view parameters."""
        self.coords = np.array(coords, dtype=np.float64) if len(coords) > 0 else np.empty((0, 3))
        if user_bonds is not None:
            self.user_bonds = list(user_bonds)
        n = len(self.coords)
        if self.selected_atom_idx is not None and (self.selected_atom_idx < 0 or self.selected_atom_idx >= n):
            self.selected_atom_idx = None
        self.multi_selected = [i for i in self.multi_selected if 0 <= i < n]
        if self.selected_bond is not None:
            bu, bv = self.selected_bond
            if bu < 0 or bu >= n or bv < 0 or bv >= n:
                self.selected_bond = None
        self.compute_bonds()
        self.update()

    def _sanitize_bond_orders(self):
        """Remove out-of-range, non-existent, or chemically impossible (e.g. H=C) bond orders."""
        if not getattr(self, "bond_orders", None):
            return
        _MONOVALENT = {
            "H", "D", "T", "F", "Cl", "Br", "I",
            "Li", "Na", "K", "Rb", "Cs", "He", "Ne", "Ar", "Kr", "Xe",
        }
        n = len(self.symbols)
        bond_set = {(min(u, v), max(u, v)) for u, v in self.bonds}
        cleaned: Dict[Tuple[int, int], int] = {}
        for (u, v), order in list(self.bond_orders.items()):
            a, b = (u, v) if u <= v else (v, u)
            try:
                o_int = int(order)
            except Exception:
                continue
            if o_int <= 1 or a < 0 or b >= n or a == b:
                continue
            if (a, b) not in bond_set:
                continue
            if self.symbols[a] in _MONOVALENT or self.symbols[b] in _MONOVALENT:
                continue
            cleaned[(a, b)] = o_int
        self.bond_orders = cleaned

    def compute_bonds(self):
        """Auto detect covalent bonds based on pairwise covalent radii, strictly honoring user_bonds."""
        from .forcefield import detect_connectivity
        self.bonds = detect_connectivity(self.symbols, self.coords, user_bonds=self.user_bonds,
                                         strict=bool(getattr(self, 'strict_bonds', False)))
        self._sanitize_bond_orders()

    def fit_camera(self):
        """Fit camera zoom and pan to the current coordinates WITHOUT modifying self.coords."""
        if len(self.coords) == 0:
            self.pan_x = 0.0
            self.pan_y = 0.0
            self.zoom = 12.0
            return

        center = np.mean(self.coords, axis=0)
        max_dist = np.max(np.linalg.norm(self.coords - center, axis=1)) if len(self.coords) > 0 else 5.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.zoom = max(max_dist * 3.2, 5.0)

    def get_camera_state(self) -> Dict[str, Any]:
        """Return a JSON-serializable snapshot of the current camera orientation, pan, zoom, and projection."""
        q = self.rot_quat
        return {
            "rot_x": float(self.rot_x),
            "rot_y": float(self.rot_y),
            "rot_quat": [float(q.scalar()), float(q.x()), float(q.y()), float(q.z())],
            "pan_x": float(self.pan_x),
            "pan_y": float(self.pan_y),
            "zoom": float(self.zoom),
            "is_orthographic": bool(getattr(self, "is_orthographic", False)),
        }

    def set_camera_state(self, state: Dict[str, Any]) -> bool:
        """Restore camera orientation, pan, zoom, and projection from a saved snapshot."""
        if not isinstance(state, dict) or "rot_quat" not in state:
            return False
        q_vals = state.get("rot_quat")
        if not isinstance(q_vals, (list, tuple)) or len(q_vals) != 4:
            return False
        try:
            self.rot_x = float(state.get("rot_x", self.rot_x))
            self.rot_y = float(state.get("rot_y", self.rot_y))
            q = QtGui.QQuaternion(float(q_vals[0]), float(q_vals[1]), float(q_vals[2]), float(q_vals[3]))
            if not q.isNull():
                self.rot_quat = q.normalized()
            self.pan_x = float(state.get("pan_x", 0.0))
            self.pan_y = float(state.get("pan_y", 0.0))
            self.zoom = max(0.5, float(state.get("zoom", 15.0)))
            self.is_orthographic = bool(state.get("is_orthographic", False))
            self.update()
            return True
        except (TypeError, ValueError):
            return False

    def auto_center(self):
        """Reset camera pan and zoom to perfectly frame and center the molecule in the canvas."""
        self.fit_camera()
        self.update()

    def focus_on_selection(self):
        """Center the camera pan on the selected atoms, or fall back to auto_center()."""
        if self.coords is None or len(self.coords) == 0:
            return
        sel = [i for i in getattr(self, "selected_atoms", []) if 0 <= i < len(self.coords)]
        if not sel:
            self.auto_center()
            return
        mol_center = np.mean(self.coords, axis=0)
        sel_center = np.mean(self.coords[sel], axis=0)
        delta = sel_center - mol_center
        # Project world-space offset into view space using current rotation
        rot_mat = self.rot_quat.toRotationMatrix()
        vx = rot_mat[0, 0] * delta[0] + rot_mat[0, 1] * delta[1] + rot_mat[0, 2] * delta[2]
        vy = rot_mat[1, 0] * delta[0] + rot_mat[1, 1] * delta[1] + rot_mat[1, 2] * delta[2]
        self.pan_x = -float(vx)
        self.pan_y = -float(vy)
        self.update()

    def align_camera_standard(self):
        """
        Set camera orientation to standard viewing position:
        rot_x = -90.0, rot_y = 0.0, so the Z axis points vertically upwards on screen,
        the X axis points horizontally to the right on screen, and the Y axis points into the screen.
        The XZ plane (where standard-aligned molecules lie) is directly coplanar with the screen.
        """
        self.rot_x = -90.0
        self.rot_y = 0.0
        self.rot_quat = QtGui.QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, -90.0)
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def set_camera_plane_view(self, plane: str):
        """Snap camera orientation to view directly onto the XY, XZ, or YZ Cartesian plane or along Principal Inertial Axes."""
        p = plane.upper()
        if p == "XY":
            self.rot_x = 0.0
            self.rot_y = 0.0
            self.rot_quat = QtGui.QQuaternion()
        elif p == "XZ":
            self.rot_x = -90.0
            self.rot_y = 0.0
            self.rot_quat = QtGui.QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, -90.0)
        elif p == "YZ":
            self.rot_x = 0.0
            self.rot_y = -90.0
            self.rot_quat = QtGui.QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, -90.0)
        elif p in ("INERTIA", "INERTIAL", "AXES"):
            self.align_to_principal_axes()
            return
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def flip_camera_180(self):
        """Flip the camera view 180 degrees horizontally (Chemcraft style) to view opposite face."""
        flip_q = QtGui.QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, 180.0)
        self.rot_quat = flip_q * self.rot_quat
        self.update()

    def align_to_principal_axes(self):
        """Align camera view to look along the molecule's principal axes of inertia (Chemcraft/VMD style)."""
        if len(self.coords) < 2:
            return
        # Compute principal axes of inertia
        from .elements import get_mass
        masses = np.array([get_mass(s) for s in self.symbols], dtype=np.float64)
        c = self.coords - np.average(self.coords, axis=0, weights=masses)
        I = np.zeros((3, 3), dtype=np.float64)
        for i, m in enumerate(masses):
            x, y, z = c[i]
            I[0, 0] += m * (y * y + z * z)
            I[1, 1] += m * (x * x + z * z)
            I[2, 2] += m * (x * x + y * y)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        evals, evecs = np.linalg.eigh(I)
        # evecs columns are principal axes. Ensure right-handed coordinate system:
        R = evecs.copy()
        if np.linalg.det(R) < 0:
            R[:, 2] *= -1.0
        # QQuaternion from orthonormal principal axes (X=axis 0, Y=axis 1, Z=axis 2)
        self.rot_quat = QtGui.QQuaternion.fromAxes(
            QtGui.QVector3D(float(R[0, 0]), float(R[1, 0]), float(R[2, 0])),
            QtGui.QVector3D(float(R[0, 1]), float(R[1, 1]), float(R[2, 1])),
            QtGui.QVector3D(float(R[0, 2]), float(R[1, 2]), float(R[2, 2])),
        )
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def toggle_orthographic(self) -> bool:
        """Toggle between Perspective and Orthographic projection."""
        self.is_orthographic = not self.is_orthographic
        self.update()
        return self.is_orthographic

    def set_bg_color_mode(self, mode: str):
        """Set background color mode ('dark', 'white', or 'transparent') and request repaint."""
        self.bg_color_mode = "white" if mode.lower() == "white" else "dark"
        self.update()

    def apply_journal_preset(self, preset_name: str) -> str:
        """
        Apply publication-grade visual preset for journal figures and presentations.
        Supported presets:
          - 'jacs': JACS Pastel (white background, crisp publication outlines ON)
          - 'nature': Nature / Wiley Glossy (crisp white background, high-contrast specular Phong, outlines OFF)
          - 'angewandte': Angewandte Bold (white background, bold silhouette outlines ON)
          - 'monochrome': Monochrome Print (white background, crisp ink outlines, high contrast)
          - 'dark': Dark Presentation Mode (dark slate background, vibrant colors, outlines OFF)
        """
        p = preset_name.lower()
        if "jacs" in p:
            self.set_bg_color_mode("white")
            self.show_publication_outlines = True
            label = "JACS Pastel"
        elif "nature" in p:
            self.set_bg_color_mode("white")
            self.show_publication_outlines = False
            label = "Nature Glossy"
        elif "angewandte" in p:
            self.set_bg_color_mode("white")
            self.show_publication_outlines = True
            label = "Angewandte Bold"
        elif "mono" in p or "print" in p or "gray" in p:
            self.set_bg_color_mode("white")
            self.show_publication_outlines = True
            label = "Monochrome Print"
        else:  # "dark"
            self.set_bg_color_mode("dark")
            self.show_publication_outlines = False
            label = "Dark Presentation"
        self.update()
        return label

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)
        glClearColor(0.12, 0.13, 0.16, 1.0) # Modern dark slate background

        # Compile shaders
        self.shader_program = self._build_shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
        self.surface_program = self._build_shader(SURFACE_VERTEX_SHADER, SURFACE_FRAGMENT_SHADER)

        # Adaptive LOD for software renderers (no dedicated GPU)
        try:
            raw_renderer = glGetString(GL_RENDERER)
            renderer_str = (raw_renderer.decode("utf-8", errors="ignore") if isinstance(raw_renderer, bytes) else str(raw_renderer or "")).lower()
        except Exception:
            renderer_str = ""
        is_software_gl = any(k in renderer_str for k in ("llvmpipe", "swiftshader", "gdi generic", "softpipe"))
        sphere_stacks, sphere_sectors = (16, 24) if is_software_gl else (48, 64)

        # Build sphere geometry
        s_verts, s_inds = create_sphere_mesh(stacks=sphere_stacks, sectors=sphere_sectors)
        self.sphere_indices_count = len(s_inds)

        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        self.sphere_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo)
        glBufferData(GL_ARRAY_BUFFER, s_verts.nbytes, s_verts, GL_STATIC_DRAW)

        self.sphere_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, s_inds.nbytes, s_inds, GL_STATIC_DRAW)

        # Pos
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        # Normals equal positions for unit sphere
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))

        # Dynamic instance buffer for fast batch rendering of spheres
        self.sphere_inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, 256, None, GL_DYNAMIC_DRAW)
        # 8 floats per instance: pos(3), radius(1), color(3), hl_flag(1) = 32 bytes
        inst_stride = 8 * 4
        # Attribute 2: instance_pos (vec3)
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)
        # Attribute 3: instance_radius (float)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(3, 1)
        # Attribute 4: instance_color (vec3)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(4 * 4))
        glVertexAttribDivisor(4, 1)
        # Attribute 5: instance_hl (float)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, inst_stride, ctypes.c_void_p(7 * 4))
        glVertexAttribDivisor(5, 1)

        # Build cylinder geometry
        c_verts, c_inds = create_cylinder_mesh(sectors=36)
        self.cyl_indices_count = len(c_inds)

        self.cyl_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cyl_vao)

        self.cyl_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cyl_vbo)
        glBufferData(GL_ARRAY_BUFFER, c_verts.nbytes, c_verts, GL_STATIC_DRAW)

        self.cyl_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cyl_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, c_inds.nbytes, c_inds, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))

        # Dynamic instance buffer for fast batch rendering of bond cylinders
        self.cyl_inst_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cyl_inst_vbo)
        glBufferData(GL_ARRAY_BUFFER, 512, None, GL_DYNAMIC_DRAW)
        # 11 floats per cylinder instance: start_pos(3), radius(1), color(3), hl(1), end_pos(3) = 44 bytes
        cyl_inst_stride = 11 * 4
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, cyl_inst_stride, ctypes.c_void_p(0))
        glVertexAttribDivisor(2, 1)
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, cyl_inst_stride, ctypes.c_void_p(3 * 4))
        glVertexAttribDivisor(3, 1)
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, cyl_inst_stride, ctypes.c_void_p(4 * 4))
        glVertexAttribDivisor(4, 1)
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, cyl_inst_stride, ctypes.c_void_p(7 * 4))
        glVertexAttribDivisor(5, 1)
        glEnableVertexAttribArray(6)
        glVertexAttribPointer(6, 3, GL_FLOAT, GL_FALSE, cyl_inst_stride, ctypes.c_void_p(8 * 4))
        glVertexAttribDivisor(6, 1)

        glBindVertexArray(0)

        # Build cone geometry for vector arrowheads
        cone_verts, cone_inds = create_cone_mesh(sectors=16)
        self.cone_indices_count = len(cone_inds)

        self.cone_vao = glGenVertexArrays(1)
        glBindVertexArray(self.cone_vao)

        self.cone_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.cone_vbo)
        glBufferData(GL_ARRAY_BUFFER, cone_verts.nbytes, cone_verts, GL_STATIC_DRAW)

        self.cone_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.cone_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cone_inds.nbytes, cone_inds, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, ctypes.c_void_p(3 * 4))

        glBindVertexArray(0)
        self._uniform_locs: Dict[str, int] = {}

    def _uloc(self, name: str) -> int:
        """Return cached GL uniform location for ``self.shader_program``."""
        locs = getattr(self, "_uniform_locs", None)
        if locs is None:
            locs = {}
            self._uniform_locs = locs
        loc = locs.get(name)
        if loc is None:
            loc = glGetUniformLocation(self.shader_program, name)
            locs[name] = loc
        return loc

    def _get_cached_sphere_props(self, is_white_bg: bool):
        """Cache per-atom sphere radii and base RGB colors keyed by symbols and display style."""
        key = (
            tuple(self.symbols),
            self.style_mode,
            float(self.sphere_scale),
            float(self.cylinder_radius),
            bool(is_white_bg),
        )
        if getattr(self, "_cached_sphere_key", None) != key:
            n = len(self.symbols)
            radii = np.empty(n, dtype=np.float32)
            colors = np.empty((n, 3), dtype=np.float32)
            bg_mode = "white" if is_white_bg else "dark"
            style = self.style_mode
            if style == "Spacefill":
                for i, sym in enumerate(self.symbols):
                    radii[i] = get_vdw_radius(sym) * 0.75
                    colors[i] = get_color(sym, bg_mode=bg_mode)
            elif style == "Licorice / Tube":
                r_tube = float(self.cylinder_radius) * 1.05
                radii.fill(r_tube)
                for i, sym in enumerate(self.symbols):
                    colors[i] = get_color(sym, bg_mode=bg_mode)
            elif style == "Wireframe":
                radii.fill(0.05)
                for i, sym in enumerate(self.symbols):
                    colors[i] = get_color(sym, bg_mode=bg_mode)
            else:
                s_scale = float(self.sphere_scale)
                for i, sym in enumerate(self.symbols):
                    radii[i] = max(0.2, get_cov_radius(sym) * s_scale)
                    colors[i] = get_color(sym, bg_mode=bg_mode)
            self._cached_sphere_key = key
            self._cached_sphere_props = (radii, colors)
        return self._cached_sphere_props

    def _compute_atom_ao(self, coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Compute cached per-atom cavity ambient occlusion factor in [0.58, 1.0]."""
        n = len(coords)
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        key = (n, float(coords[0, 0]), float(coords[-1, -1]), float(radii[0]))
        if getattr(self, "_ao_cache_key", None) == key and getattr(self, "_ao_cache_val", None) is not None:
            return self._ao_cache_val
        diff = coords[:, None, :] - coords[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        mask = dist2 < (4.5 * 4.5)
        r2 = (radii[None, :] ** 2) * mask
        occ = np.sum(r2 / np.maximum(dist2, 0.5), axis=1)
        ao = np.clip(1.0 - 0.55 * occ, 0.58, 1.0).astype(np.float32)
        self._ao_cache_key = key
        self._ao_cache_val = ao
        return ao

    def _build_sphere_instance_data(
        self,
        render_coords: np.ndarray,
        charges_arr: Optional[np.ndarray],
        is_white_bg: bool,
        clean_export: bool,
    ) -> np.ndarray:
        """Pack (N, 8) float32 sphere instance buffer [x, y, z, radius, r, g, b, highlight]."""
        n_atoms = len(self.symbols)
        inst_data = np.empty((n_atoms, 8), dtype=np.float32)
        radii, base_colors = self._get_cached_sphere_props(is_white_bg)
        inst_data[:, 0:3] = render_coords
        inst_data[:, 3] = radii
        if charges_arr is not None and len(charges_arr) >= n_atoms:
            for i in range(n_atoms):
                inst_data[i, 4:7] = self._charge_to_rgb(charges_arr[i])
        else:
            inst_data[:, 4:7] = base_colors
        if getattr(self, "use_depth_cueing", True) and n_atoms > 1:
            ao = self._compute_atom_ao(render_coords, radii)
            inst_data[:, 4:7] *= ao[:, None]

        inst_data[:, 7] = 0.0
        if not clean_export:
            for i in self.multi_selected:
                if 0 <= i < n_atoms:
                    inst_data[i, 7] = 2.0
            sel_idx = self.selected_atom_idx
            if sel_idx is not None and 0 <= sel_idx < n_atoms:
                inst_data[sel_idx, 7] = 1.0
            for i in self.picked_references:
                if 0 <= i < n_atoms:
                    inst_data[i, 7] = 3.0
            role_map = {"target": 4.0, "bond": 5.0, "angle": 6.0, "dihedral": 7.0}
            for i, role in self.highlighted_feature_atoms.items():
                if 0 <= i < n_atoms and role in role_map:
                    inst_data[i, 7] = role_map[role]
        return inst_data

    def _build_ghost_instance_data(
        self,
        render_coords: np.ndarray,
        face_shifts: List[np.ndarray],
        is_white_bg: bool,
    ) -> np.ndarray:
        """Pack vectorized (26*N, 8) float32 PBC ghost atom instance buffer."""
        n_atoms = len(self.symbols)
        if n_atoms == 0 or not face_shifts:
            return np.empty((0, 8), dtype=np.float32)
        shifts = np.asarray(face_shifts, dtype=np.float32)
        n_shifts = len(shifts)
        coords_f32 = np.asarray(render_coords, dtype=np.float32)
        ghost_pos = (coords_f32[None, :, :] + shifts[:, None, :]).reshape(-1, 3)

        bg_mode = "white" if is_white_bg else "dark"
        s_scale = float(self.sphere_scale)
        g_rad = np.array([max(0.18, get_cov_radius(sym) * s_scale * 0.85) for sym in self.symbols], dtype=np.float32)
        g_col = np.array([get_color(sym, bg_mode=bg_mode) for sym in self.symbols], dtype=np.float32) * 0.75 + 0.25

        ghost_arr = np.empty((n_shifts * n_atoms, 8), dtype=np.float32)
        ghost_arr[:, 0:3] = ghost_pos
        ghost_arr[:, 3] = np.tile(g_rad, n_shifts)
        ghost_arr[:, 4:7] = np.tile(g_col, (n_shifts, 1))
        ghost_arr[:, 7] = 0.0
        return ghost_arr

    def _get_cached_topo(self, n_rc: int):
        """Return cached (delocalized_bonds, adj_graph), recomputing only when topology changes."""
        bo_items = tuple(sorted(getattr(self, "bond_orders", {}).items()))
        key = (n_rc, tuple(self.bonds), bo_items)
        if getattr(self, "_cached_topo_key", None) != key:
            deloc = self._delocalized_bonds()
            adj_graph: Dict[int, List[int]] = {}
            for u, w in self.bonds:
                if 0 <= u < n_rc and 0 <= w < n_rc:
                    adj_graph.setdefault(u, []).append(w)
                    adj_graph.setdefault(w, []).append(u)
            self._cached_topo_key = key
            self._cached_topo_val = (deloc, adj_graph)
        return self._cached_topo_val

    def _build_shader(self, v_src: str, f_src: str) -> int:
        def compile(src, s_type):
            shader = glCreateShader(s_type)
            glShaderSource(shader, src)
            glCompileShader(shader)
            status = glGetShaderiv(shader, GL_COMPILE_STATUS)
            if not status:
                err = glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
                raise RuntimeError(f"Shader compilation failed: {err}")
            return shader

        vs = compile(v_src, GL_VERTEX_SHADER)
        fs = compile(f_src, GL_FRAGMENT_SHADER)
        prog = glCreateProgram()
        glAttachShader(prog, vs)
        glAttachShader(prog, fs)
        glLinkProgram(prog)
        status = glGetProgramiv(prog, GL_LINK_STATUS)
        if not status:
            err = glGetProgramInfoLog(prog).decode("utf-8", errors="ignore")
            raise RuntimeError(f"Shader link failed: {err}")
        glDeleteShader(vs)
        glDeleteShader(fs)
        return prog

    def resizeGL(self, w: int, h: int):
        glViewport(0, 0, w, h)

    def _get_projection_matrix(self) -> QtGui.QMatrix4x4:
        mat = QtGui.QMatrix4x4()
        aspect = self.width() / max(1, self.height())
        if getattr(self, "is_orthographic", False):
            # Orthographic box matched to current zoom level and aspect ratio
            # At 45 deg FOV, visible half-height at distance self.zoom is zoom * tan(22.5 deg) ~ zoom * 0.4142
            ortho_h = max(1.0, self.zoom * 0.4142)
            ortho_w = ortho_h * aspect
            mat.ortho(-ortho_w, ortho_w, -ortho_h, ortho_h, 0.1, 300.0)
        else:
            mat.perspective(45.0, aspect, 0.5, 300.0)
        return mat

    def _get_view_matrix(self) -> QtGui.QMatrix4x4:
        mat = QtGui.QMatrix4x4()
        mat.translate(self.pan_x, self.pan_y, -self.zoom)
        if hasattr(self, "rot_quat") and not self.rot_quat.isNull():
            mat.rotate(self.rot_quat)
        else:
            mat.rotate(self.rot_x, 1.0, 0.0, 0.0)
            mat.rotate(self.rot_y, 0.0, 1.0, 0.0)
        if len(self.coords) > 0:
            c = np.mean(self.coords, axis=0)
            mat.translate(-float(c[0]), -float(c[1]), -float(c[2]))
        return mat

    def _get_camera_axes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the world-space unit vectors for screen right and screen up."""
        if hasattr(self, "rot_quat") and not self.rot_quat.isNull():
            inv_q = self.rot_quat.inverted()
            rw = inv_q.rotatedVector(QtGui.QVector3D(1.0, 0.0, 0.0))
            uw = inv_q.rotatedVector(QtGui.QVector3D(0.0, 1.0, 0.0))
            right_w = np.array([rw.x(), rw.y(), rw.z()], dtype=np.float64)
            up_w = np.array([uw.x(), uw.y(), uw.z()], dtype=np.float64)
            r_norm = np.linalg.norm(right_w)
            u_norm = np.linalg.norm(up_w)
            if r_norm > 1e-6:
                right_w /= r_norm
            if u_norm > 1e-6:
                up_w /= u_norm
            return right_w, up_w

        rad_x = math.radians(self.rot_x)
        rad_y = math.radians(self.rot_y)
        # Rotation matrices Rx and Ry
        rx = np.array([
            [1.0, 0.0, 0.0],
            [0.0, math.cos(rad_x), -math.sin(rad_x)],
            [0.0, math.sin(rad_x), math.cos(rad_x)]
        ])
        ry = np.array([
            [math.cos(rad_y), 0.0, math.sin(rad_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(rad_y), 0.0, math.cos(rad_y)]
        ])
        # Camera rotation transpose maps camera directions back to world
        r_cam_to_world = (rx @ ry).T
        right_w = r_cam_to_world @ np.array([1.0, 0.0, 0.0])
        up_w = r_cam_to_world @ np.array([0.0, 1.0, 0.0])
        return right_w, up_w

    def screen_displacement_to_world(self, dx: float, dy: float) -> np.ndarray:
        """Convert a 2D screen mouse delta (dx, dy) into a normalized 3D world direction vector."""
        right_w, up_w = self._get_camera_axes()
        # In screen coords, positive dy is downwards, so -dy is upwards
        dir_w = dx * right_w - dy * up_w
        norm = np.linalg.norm(dir_w)
        if norm < 1e-6:
            return right_w
        return dir_w / norm

    def paintGL(self):
        is_white_bg = (getattr(self, "bg_color_mode", "dark") == "white")
        is_white_vp = (is_white_bg and not getattr(self, "transparent_bg", False))

        # QPainter overlays (labels/HUD) leave GL state modified (depth test off, etc.), which made
        # atoms draw in index order instead of by depth. Restore the full 3D state every frame.
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glDepthMask(GL_TRUE)
        glDisable(GL_CULL_FACE)
        glDisable(GL_SCISSOR_TEST)
        glDisable(GL_STENCIL_TEST)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)

        if getattr(self, "transparent_bg", False):
            glClearColor(0.0, 0.0, 0.0, 0.0)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        elif getattr(self, "bg_color_mode", "dark") == "white":
            glClearColor(1.0, 1.0, 1.0, 1.0)
        else:
            glClearColor(0.12, 0.13, 0.16, 1.0) # Modern dark slate background

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if len(self.coords) == 0:
            # Render Clean Jumpstart / Welcome Overlay when viewport is empty (Newcomer Friendly)
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            w, h = self.width(), self.height()
            is_white_vp = (getattr(self, "bg_color_mode", "dark") == "white" and not getattr(self, "transparent_bg", False))

            recent_paths: List[str] = []
            try:
                from .session_state import get_settings
                raw_rec = get_settings("AgyMolden").value("recent_files", []) or []
                if isinstance(raw_rec, str):
                    raw_rec = [raw_rec]
                for rp in raw_rec:
                    if isinstance(rp, str) and os.path.isfile(rp) and rp not in recent_paths:
                        recent_paths.append(rp)
                    if len(recent_paths) >= 3:
                        break
            except Exception:
                logging.getLogger(__name__).debug("Ignored exception reading recent_files", exc_info=True)

            card_w = min(480, w - 40)
            card_h = 195 + (24 + len(recent_paths) * 19 if recent_paths else 0)
            card_x = (w - card_w) // 2
            card_y = max(10, (h - card_h) // 2)

            # Background card
            card_rect = QtCore.QRect(card_x, card_y, card_w, card_h)
            painter.setBrush(QtGui.QColor(248, 250, 252, 235) if is_white_vp else QtGui.QColor(15, 23, 42, 225))
            painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1" if is_white_vp else "#334155"), 1.5))
            painter.drawRoundedRect(card_rect, 10, 10)

            # Logo & Welcome Title
            painter.setFont(QtGui.QFont("DejaVu Sans", 14, QtGui.QFont.Weight.Bold))
            painter.setPen(QtGui.QColor("#0284c7" if is_white_vp else "#38bdf8"))
            painter.drawText(card_x + 20, card_y + 36, "⚛️ AgyMolden Quantum Workbench")

            # Quick Tips / Shortcuts subtitle
            painter.setFont(QtGui.QFont("DejaVu Sans", 9))
            painter.setPen(QtGui.QColor("#475569" if is_white_vp else "#94a3b8"))
            painter.drawText(card_x + 20, card_y + 60, "Click any option below or use its shortcut to begin:")

            # Interactive Clickable Hotspots
            shortcuts = [
                ("📂 Open Molecular File...", "Ctrl+O", "open_file"),
                ("✏️ Build from Name / SMILES...", "Ctrl+Shift+B", "build_name"),
                ("📚 Open Example Gallery...", "Gallery", "open_gallery"),
                ("🚀 Quick ORCA Calculation Studio...", "F5", "quick_orca"),
                ("⚡ Command Palette...", "Ctrl+K", "cmd_palette"),
            ]
            sy = card_y + 82
            self._welcome_hotspots = []
            for label, key, action_key in shortcuts:
                row_rect = QtCore.QRect(card_x + 16, sy - 14, card_w - 32, 20)
                self._welcome_hotspots.append((row_rect, action_key))

                # Pill / hover highlight
                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                painter.setPen(QtGui.QColor("#0284c7" if is_white_vp else "#38bdf8"))
                painter.drawText(card_x + 24, sy, f"• {label}")
                painter.setPen(QtGui.QColor("#475569" if is_white_vp else "#cbd5e1"))
                painter.drawText(card_x + card_w - 95, sy, f"[{key}]")
                sy += 21

            if recent_paths:
                sy += 4
                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                painter.setPen(QtGui.QColor("#64748b" if is_white_vp else "#94a3b8"))
                painter.drawText(card_x + 20, sy, "Recent Files:")
                sy += 18
                for rp in recent_paths:
                    row_rect = QtCore.QRect(card_x + 16, sy - 13, card_w - 32, 18)
                    self._welcome_hotspots.append((row_rect, f"recent:{rp}"))
                    painter.setFont(QtGui.QFont("DejaVu Sans", 8))
                    painter.setPen(QtGui.QColor("#0284c7" if is_white_vp else "#7dd3fc"))
                    bname = os.path.basename(rp)
                    painter.drawText(card_x + 24, sy, f"↺ {bname}")
                    sy += 19

            if getattr(self, "_drag_hover", False):
                self._draw_drop_overlay(painter, w, h)
            painter.end()
            return

        render_coords = self.get_render_coords()

        glUseProgram(self.shader_program)

        proj_mat = self._get_projection_matrix()
        view_mat = self._get_view_matrix()

        # Set uniforms (cached locations)
        loc_proj = self._uloc("u_proj")
        loc_view = self._uloc("u_view")
        loc_model = self._uloc("u_model")
        loc_norm = self._uloc("u_normal_mat")
        loc_color = self._uloc("u_color")
        loc_light = self._uloc("u_light_pos")
        loc_view_pos = self._uloc("u_view_pos")
        loc_hl = self._uloc("u_is_highlighted")
        loc_use_inst = self._uloc("u_use_instancing")

        c_x, c_y, c_z = 0.0, 0.0, 0.0
        if len(self.coords) > 0:
            c = np.mean(self.coords, axis=0)
            c_x, c_y, c_z = float(c[0]), float(c[1]), float(c[2])
        if hasattr(self, "rot_quat") and not self.rot_quat.isNull():
            inv_q = self.rot_quat.inverted()
            cw = inv_q.rotatedVector(QtGui.QVector3D(0.0, 0.0, float(self.zoom)))
            cam_world_pos = (cw.x() + c_x, cw.y() + c_y, cw.z() + c_z)
            lw = inv_q.rotatedVector(QtGui.QVector3D(15.0, 25.0, 20.0))
            light_world_pos = (cam_world_pos[0] + lw.x(), cam_world_pos[1] + lw.y(), cam_world_pos[2] + lw.z())
        else:
            cam_world_pos = (c_x, c_y, self.zoom + c_z)
            light_world_pos = (cam_world_pos[0] + 15.0, cam_world_pos[1] + 25.0, cam_world_pos[2] + 20.0)

        glUniformMatrix4fv(loc_proj, 1, GL_FALSE, proj_mat.data())
        glUniformMatrix4fv(loc_view, 1, GL_FALSE, view_mat.data())
        glUniform3f(loc_light, light_world_pos[0], light_world_pos[1], light_world_pos[2])
        glUniform3f(loc_view_pos, cam_world_pos[0], cam_world_pos[1], cam_world_pos[2])
        glUniform1i(loc_use_inst, 0)
        loc_pub_outline = self._uloc("u_publication_outline")
        if loc_pub_outline != -1:
            glUniform1i(loc_pub_outline, 1 if getattr(self, "show_publication_outlines", False) else 0)

        loc_modern_glossy = self._uloc("u_modern_glossy")
        if loc_modern_glossy != -1:
            glUniform1i(loc_modern_glossy, 1 if getattr(self, "use_modern_shaders", False) else 0)

        loc_depth_cue = self._uloc("u_depth_cue")
        if loc_depth_cue != -1:
            glUniform1i(loc_depth_cue, 1 if getattr(self, "use_depth_cueing", True) else 0)

        loc_bg_col = self._uloc("u_bg_color")
        if loc_bg_col != -1:
            if getattr(self, "bg_color_mode", "white") == "white":
                glUniform3f(loc_bg_col, 0.97, 0.98, 0.99)
            else:
                glUniform3f(loc_bg_col, 0.07, 0.09, 0.14)

        loc_cam_dist = self._uloc("u_cam_dist")
        if loc_cam_dist != -1:
            glUniform1f(loc_cam_dist, float(getattr(self, "zoom", 15.0)))

        # 1. Render Bonds (Cylinders) via GPU instancing
        if self.style_mode != "Spacefill":
            glBindVertexArray(self.cyl_vao)
            # Reference vectors for camera orientation
            right_cam, up_cam = self._get_camera_axes()
            view_cam = np.cross(right_cam, up_cam)
            n_rc = len(render_coords)
            deloc, adj_graph = self._get_cached_topo(n_rc)

            cyl_instances = []
            for i1, i2 in self.bonds:
                if not (0 <= i1 < n_rc and 0 <= i2 < n_rc):
                    continue
                p1 = render_coords[i1]
                p2 = render_coords[i2]
                v = p2 - p1
                length = np.linalg.norm(v)
                if length < 1e-4:
                    continue
                v_norm = v / length

                # Determine bond order: 1, 2, or 3
                b_key = (min(i1, i2), max(i1, i2))
                order = self.bond_orders.get(b_key, 1)

                # Determine cylinder offsets perpendicular to bond vector:
                # 1. Primary candidate: In-plane molecular perpendicular direction (Chemcraft / GaussView standard)
                #    In planar fragments (like carbonyls, alkenes, aromatics), the double bond should lie
                #    in the plane formed by the bonded atoms and their substituent neighbors.
                nbrs = [k for k in adj_graph.get(i1, []) if k != i2] + [k for k in adj_graph.get(i2, []) if k != i1]
                in_plane_candidate = None
                if nbrs:
                    normals = []
                    for k in nbrs:
                        if not (0 <= k < n_rc):
                            continue
                        pk = render_coords[k]
                        vec_k = pk - p1
                        n_k = np.cross(v_norm, vec_k)
                        len_nk = np.linalg.norm(n_k)
                        if len_nk > 1e-3:
                            normals.append(n_k / len_nk)
                    if normals:
                        avg_n = normals[0].copy()
                        for nk in normals[1:]:
                            if np.dot(nk, avg_n) < 0:
                                avg_n -= nk
                            else:
                                avg_n += nk
                        len_avg = np.linalg.norm(avg_n)
                        if len_avg > 1e-3:
                            avg_n /= len_avg
                            # In-plane perpendicular is perpendicular to both bond axis and molecular plane normal
                            cand = np.cross(avg_n, v_norm)
                            len_cand = np.linalg.norm(cand)
                            if len_cand > 1e-3:
                                in_plane_candidate = cand / len_cand

                # 2. Camera-plane perpendicular (billboard fallback for edge-on views or isolated bonds)
                cam_perp = np.cross(v_norm, up_cam)
                if np.linalg.norm(cam_perp) < 1e-3:
                    cam_perp = np.cross(v_norm, right_cam)
                len_cp = np.linalg.norm(cam_perp)
                cam_perp = (cam_perp / len_cp) if len_cp > 1e-4 else np.array([0.0, 1.0, 0.0])

                if in_plane_candidate is not None:
                    # Check how clearly the in-plane vector is visible from current camera angle:
                    # Project onto screen plane (perpendicular to view line of sight)
                    proj = in_plane_candidate - np.dot(in_plane_candidate, view_cam) * view_cam
                    proj_mag = np.linalg.norm(proj)
                    if proj_mag > 0.35:
                        # Clear view: use in-plane molecular orientation so double bonds lie flat with atoms
                        perp = in_plane_candidate
                    else:
                        # Edge-on view: blend smoothly to billboarded camera perp so double bonds don't eclipse/vanish
                        alpha = float(np.clip((proj_mag - 0.10) / 0.25, 0.0, 1.0))
                        blended = alpha * in_plane_candidate + (1.0 - alpha) * cam_perp
                        len_b = np.linalg.norm(blended)
                        perp = (blended / len_b) if len_b > 1e-4 else cam_perp
                else:
                    perp = cam_perp

                # Each offset is (displacement, length_scale)
                if b_key in deloc:
                    # Aromatic / resonance bond (order 1.5): solid line + shorter parallel line
                    inward = deloc[b_key]
                    side = inward if inward is not None else perp
                    # Rings: the second line sits cleanly inside the ring (Avogadro 2 / GaussView proportions)
                    offsets = [(np.zeros(3), 1.0), (side * 0.12, 0.78)]
                    cyl_rad = self.cylinder_radius * 0.65
                elif order == 1:
                    offsets = [(np.zeros(3), 1.0)]
                    cyl_rad = self.cylinder_radius
                elif order == 2:
                    # GaussView / Avogadro 2 style: twin parallel cylinders separated by 0.11 A with refined radius
                    shift = 0.10
                    offsets = [(perp * shift, 1.0), (-perp * shift, 1.0)]
                    cyl_rad = self.cylinder_radius * 0.70
                else:  # order >= 3
                    shift = 0.13
                    offsets = [(np.zeros(3), 1.0), (perp * shift, 1.0), (-perp * shift, 1.0)]
                    cyl_rad = self.cylinder_radius * 0.60

                # Apply representation style modifiers
                if self.style_mode == "Licorice / Tube":
                    cyl_rad *= 1.35
                elif self.style_mode == "Wireframe":
                    cyl_rad *= 0.35

                half_len = 0.5 * length
                is_white_bg = (getattr(self, "bg_color_mode", "dark") == "white")
                raw_c1 = get_color(self.symbols[i1], bg_mode="white" if is_white_bg else "dark")
                raw_c2 = get_color(self.symbols[i2], bg_mode="white" if is_white_bg else "dark")
                # Ensure bond halves have crisp visibility and contrast against both dark and white backgrounds
                def _bond_contrast(c):
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    if is_white_bg and lum < 0.42:
                        # Lighten dark Carbon bonds in white mode so balls and bonds are clearly distinct
                        return (min(1.0, c[0] + 0.16), min(1.0, c[1] + 0.16), min(1.0, c[2] + 0.16))
                    if is_white_bg:
                        # Against pure white, darken near-white bonds slightly for clear definition
                        if lum > 0.80:
                            return (min(c[0], 0.74), min(c[1], 0.76), min(c[2], 0.79))
                    else:
                        # Against dark backgrounds, make Carbon bonds slightly lighter than Carbon balls
                        if lum < 0.60 and max(c) - min(c) < 0.12:
                            return (min(1.0, c[0] + 0.12), min(1.0, c[1] + 0.12), min(1.0, c[2] + 0.12))
                    return c
                c1 = _bond_contrast(raw_c1)
                c2 = _bond_contrast(raw_c2)

                is_bond_selected = (
                    not getattr(self, "export_clean_mode", False)
                    and self.selected_bond is not None
                    and (min(i1, i2), max(i1, i2)) == (min(self.selected_bond), max(self.selected_bond))
                )
                if is_bond_selected:
                    cyl_rad *= 1.25

                for off, len_scale in offsets:
                    # shorter lines are centred on the bond: shift start inward by (1 - scale) * L / 2
                    p1_off = p1 + off + v_norm * (0.5 * length * (1.0 - len_scale))
                    mid_off = 0.5 * (p1 + p2) + off
                    p2_off = mid_off + v_norm * (0.5 * length * len_scale)

                    if is_bond_selected:
                        col1 = (1.0, 0.84, 0.0)
                        col2 = (1.0, 0.84, 0.0)
                        hl_val = 1.0
                    else:
                        col1 = c1
                        col2 = c2
                        hl_val = 0.0

                    cyl_instances.append((
                        float(p1_off[0]), float(p1_off[1]), float(p1_off[2]),
                        float(cyl_rad),
                        float(col1[0]), float(col1[1]), float(col1[2]),
                        hl_val,
                        float(mid_off[0]), float(mid_off[1]), float(mid_off[2]),
                    ))
                    cyl_instances.append((
                        float(mid_off[0]), float(mid_off[1]), float(mid_off[2]),
                        float(cyl_rad),
                        float(col2[0]), float(col2[1]), float(col2[2]),
                        hl_val,
                        float(p2_off[0]), float(p2_off[1]), float(p2_off[2]),
                    ))

            if cyl_instances and self.cyl_vao and self.cyl_inst_vbo:
                cyl_inst_arr = np.array(cyl_instances, dtype=np.float32)
                glBindBuffer(GL_ARRAY_BUFFER, self.cyl_inst_vbo)
                glBufferData(GL_ARRAY_BUFFER, cyl_inst_arr.nbytes, cyl_inst_arr, GL_DYNAMIC_DRAW)
                glUniform1i(loc_use_inst, 2)
                glDrawElementsInstanced(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None, len(cyl_instances))
                glUniform1i(loc_use_inst, 0)

        # 1b. Render Hydrogen Bonds (Dashed / segmented cyan cylinders) (Avogadro style)
        if getattr(self, "show_hbonds", False) and len(self.symbols) > 2 and len(render_coords) == len(self.symbols):
            hbonds = self._detect_hydrogen_bonds(render_coords)
            cyl_rad_hb = 0.032
            glUniform1i(loc_hl, 0)
            glUniform3f(loc_color, 0.22, 0.85, 0.95) # Cyan dash

            for idx_d, idx_h, idx_a in hbonds:
                ph = render_coords[idx_h]
                pa = render_coords[idx_a]
                v_ha = pa - ph
                dist_ha = float(np.linalg.norm(v_ha))
                if dist_ha < 1e-4:
                    continue
                u_ha = v_ha / dist_ha

                # Rotate cylinder from z-axis to u_ha
                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, u_ha)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_ha), -1.0, 1.0)))
                elif np.dot(z_axis, u_ha) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                # 4-5 dashes along H...A
                n_dashes = max(3, int(dist_ha / 0.38))
                dash_len = (dist_ha / n_dashes) * 0.55
                step_len = dist_ha / n_dashes

                for step_i in range(n_dashes):
                    dash_start = ph + u_ha * (step_i * step_len + 0.1 * step_len)
                    m_dash = QtGui.QMatrix4x4()
                    m_dash.translate(float(dash_start[0]), float(dash_start[1]), float(dash_start[2]))
                    if angle_deg != 0:
                        m_dash.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m_dash.scale(cyl_rad_hb, cyl_rad_hb, float(dash_len))

                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_dash.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_dash.normalMatrix().data())
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

        # 1c. Render Non-Covalent Interactions (NCI / Halogen Bonds & Pi-Pi Stacking)
        if getattr(self, "show_nci", False) and len(self.symbols) > 3 and len(render_coords) == len(self.symbols):
            ncis = self._detect_non_covalent_interactions(render_coords)
            glUniform1i(loc_hl, 0)
            cyl_rad_nci = 0.030

            for nci_type, p1, p2, dist_nci in ncis:
                v_nci = p2 - p1
                dist = float(np.linalg.norm(v_nci))
                if dist < 1e-4:
                    continue
                u_nci = v_nci / dist

                # Colors: Orange for Halogen, Purple for Pi-Pi Stacking
                if "Halogen" in nci_type:
                    glUniform3f(loc_color, 0.98, 0.55, 0.15)  # Orange
                else:
                    glUniform3f(loc_color, 0.75, 0.35, 0.95)  # Purple

                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, u_nci)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_nci), -1.0, 1.0)))
                elif np.dot(z_axis, u_nci) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                n_dashes = max(3, int(dist / 0.42))
                dash_len = (dist / n_dashes) * 0.50
                step_len = dist / n_dashes

                for step_i in range(n_dashes):
                    dash_start = p1 + u_nci * (step_i * step_len + 0.1 * step_len)
                    m_dash = QtGui.QMatrix4x4()
                    m_dash.translate(float(dash_start[0]), float(dash_start[1]), float(dash_start[2]))
                    if angle_deg != 0:
                        m_dash.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m_dash.scale(cyl_rad_nci, cyl_rad_nci, float(dash_len))

                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_dash.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_dash.normalMatrix().data())
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

        # 2. Render Atoms (Spheres) via GPU Instancing (single draw call for all spheres)
        charges_arr = self._charges_and_dipole()[0] if getattr(self, 'show_charges', False) else None
        clean_export = getattr(self, "export_clean_mode", False)
        n_atoms = len(self.symbols)

        if n_atoms > 0:
            # Build instance data array: 8 float32 per atom [pos.xyz, rad, col.rgb, hl]
            inst_data = self._build_sphere_instance_data(render_coords, charges_arr, is_white_bg, clean_export)

            if self.sphere_vao and self.sphere_inst_vbo:
                glBindVertexArray(self.sphere_vao)
                glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
                glBufferData(GL_ARRAY_BUFFER, inst_data.nbytes, inst_data, GL_DYNAMIC_DRAW)
                glUniform1i(loc_use_inst, 1)

                glDrawElementsInstanced(GL_TRIANGLES, self.sphere_indices_count, GL_UNSIGNED_INT, None, n_atoms)

                # 2b. Optional: Render Translucent Periodic Boundary Ghost Atoms (VESTA / Mercury packing style)
                if (
                    getattr(self, "show_ghost_atoms", False)
                    and getattr(self, "show_unit_cell", True)
                    and getattr(self, "unit_cell", None) is not None
                    and not clean_export
                ):
                    uc = self.unit_cell
                    M = uc.lattice_vectors
                    face_shifts = [
                        sx * M[0] + sy * M[1] + sz * M[2]
                        for sx in (-1, 0, 1)
                        for sy in (-1, 0, 1)
                        for sz in (-1, 0, 1)
                        if not (sx == 0 and sy == 0 and sz == 0)
                    ]
                    ghost_arr = self._build_ghost_instance_data(render_coords, face_shifts, is_white_bg)
                    if len(ghost_arr) > 0:
                        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
                        glBufferData(GL_ARRAY_BUFFER, ghost_arr.nbytes, ghost_arr, GL_DYNAMIC_DRAW)
                        glDrawElementsInstanced(GL_TRIANGLES, self.sphere_indices_count, GL_UNSIGNED_INT, None, len(ghost_instances))

                glUniform1i(loc_use_inst, 0)
                glBindVertexArray(0)

        # 3. Render 3D Vibrational Normal Mode Displacement Vectors (Arrows)
        if (
            (self.vibrating or self.vib_displacements is not None)
            and self.vib_show_vectors
            and self.vib_displacements is not None
            and len(self.vib_displacements) == len(self.coords)
        ):
            v_scale = getattr(self, "vib_vector_scale", 1.0)
            # Shaft radius and cone dimensions
            shaft_r = 0.04 * max(0.4, v_scale)
            cone_r = 0.09 * max(0.4, v_scale)
            cone_len_ratio = 0.35  # fraction of vector length for arrowhead

            for i in range(len(self.coords)):
                d_vec = self.vib_displacements[i]
                d_len = float(np.linalg.norm(d_vec))
                if d_len < 0.02:
                    continue

                # Scale vector visually by amplitude and vector scale
                vec_len = d_len * self.vib_amplitude * 2.2 * v_scale
                if vec_len < 0.15:
                    vec_len = 0.15
                u_dir = d_vec / d_len

                # Arrow originates at base equilibrium position
                base_p = self.coords[i]
                shaft_len = vec_len * (1.0 - cone_len_ratio)
                head_len = vec_len * cone_len_ratio

                # Rotation from z-axis to u_dir
                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, u_dir)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_dir), -1.0, 1.0)))
                elif np.dot(z_axis, u_dir) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                # Render Shaft (Cylinder)
                glBindVertexArray(self.cyl_vao)
                m_shaft = QtGui.QMatrix4x4()
                m_shaft.translate(float(base_p[0]), float(base_p[1]), float(base_p[2]))
                if angle_deg != 0:
                    m_shaft.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_shaft.scale(shaft_r, shaft_r, float(shaft_len))

                # Color: Vibrant Gold / Amber (38, 189, 248 or 245, 158, 11)
                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_shaft.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_shaft.normalMatrix().data())
                glUniform3f(loc_color, 0.98, 0.75, 0.15)
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

                # Render Arrowhead (Cone)
                glBindVertexArray(self.cone_vao)
                head_base = base_p + u_dir * shaft_len
                m_cone = QtGui.QMatrix4x4()
                m_cone.translate(float(head_base[0]), float(head_base[1]), float(head_base[2]))
                if angle_deg != 0:
                    m_cone.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_cone.scale(cone_r, cone_r, float(head_len))

                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_cone.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_cone.normalMatrix().data())
                glUniform3f(loc_color, 0.98, 0.85, 0.25)
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cone_indices_count, GL_UNSIGNED_INT, None)

        # 3a. Render 3D Nuclear Force / Gradient Vectors (-dE/dx, Optimization Diagnostics)
        if (
            getattr(self, "show_forces", False)
            and getattr(self, "nuclear_forces", None) is not None
            and len(self.nuclear_forces) == len(self.coords)
        ):
            shaft_r = 0.045
            cone_r = 0.105
            cone_len_ratio = 0.32
            f_scale = getattr(self, "force_scale", 2.5)

            for i in range(len(self.coords)):
                f_vec = self.nuclear_forces[i]
                f_len = float(np.linalg.norm(f_vec))
                if f_len < 1e-4:
                    continue

                vec_len = max(0.20, min(3.5, f_len * f_scale))
                u_dir = f_vec / f_len

                base_p = self.coords[i]
                shaft_len = vec_len * (1.0 - cone_len_ratio)
                head_len = vec_len * cone_len_ratio

                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, u_dir)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_dir), -1.0, 1.0)))
                elif np.dot(z_axis, u_dir) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                # Render Shaft (Cylinder) in Vibrant Coral/Orange-Red
                glBindVertexArray(self.cyl_vao)
                m_shaft = QtGui.QMatrix4x4()
                m_shaft.translate(float(base_p[0]), float(base_p[1]), float(base_p[2]))
                if angle_deg != 0:
                    m_shaft.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_shaft.scale(shaft_r, shaft_r, float(shaft_len))

                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_shaft.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_shaft.normalMatrix().data())
                glUniform3f(loc_color, 0.95, 0.35, 0.15)  # Vibrant fiery orange-red
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

                # Render Arrowhead (Cone)
                glBindVertexArray(self.cone_vao)
                head_base = base_p + u_dir * shaft_len
                m_cone = QtGui.QMatrix4x4()
                m_cone.translate(float(head_base[0]), float(head_base[1]), float(head_base[2]))
                if angle_deg != 0:
                    m_cone.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_cone.scale(cone_r, cone_r, float(head_len))

                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_cone.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_cone.normalMatrix().data())
                glUniform3f(loc_color, 1.00, 0.50, 0.20)
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cone_indices_count, GL_UNSIGNED_INT, None)

        # 3b. Render 3D Electric Dipole Moment Vector (Magenta/Cyan Arrow)
        if getattr(self, "show_dipole", False) and len(render_coords) > 0:
            _, mu_vec, mu_mag = self._charges_and_dipole()
            if mu_mag >= 0.05:
                u_dir = mu_vec / mu_mag
                # Scale arrow length nicely (1.4 A base + 0.95 A per Debye, capped at 4.8 A)
                arrow_len = min(4.8, 1.4 + 0.95 * mu_mag)
                center_p = np.mean(render_coords, axis=0)
                base_p = center_p - 0.5 * arrow_len * u_dir
                shaft_len = arrow_len * 0.72
                head_len = arrow_len * 0.28
                shaft_r = 0.065
                cone_r = 0.16

                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, u_dir)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_dir), -1.0, 1.0)))
                elif np.dot(z_axis, u_dir) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                glBindVertexArray(self.cyl_vao)
                m_shaft = QtGui.QMatrix4x4()
                m_shaft.translate(float(base_p[0]), float(base_p[1]), float(base_p[2]))
                if angle_deg != 0:
                    m_shaft.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_shaft.scale(shaft_r, shaft_r, float(shaft_len))
                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_shaft.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_shaft.normalMatrix().data())
                glUniform3f(loc_color, 0.93, 0.28, 0.60)  # Vibrant magenta-pink
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

                glBindVertexArray(self.cone_vao)
                head_base = base_p + u_dir * shaft_len
                m_cone = QtGui.QMatrix4x4()
                m_cone.translate(float(head_base[0]), float(head_base[1]), float(head_base[2]))
                if angle_deg != 0:
                    m_cone.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                m_cone.scale(cone_r, cone_r, float(head_len))
                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_cone.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_cone.normalMatrix().data())
                glUniform3f(loc_color, 0.98, 0.45, 0.73)
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.cone_indices_count, GL_UNSIGNED_INT, None)

        # 3b-1. Render Dynamic 3D IR Transition Dipole Moment Derivative Vector during vibration
        if getattr(self, "vibrating", False) and getattr(self, "show_vib_dipole", True) and len(render_coords) > 0:
            vib_disp = getattr(self, "vib_displacements", None)
            if vib_disp is None:
                vib_disp = getattr(self, "vibration_displacements", None)
            if vib_disp is not None and len(vib_disp) == len(render_coords):
                # Transition dipole derivative approx: sum(q_i * dr_i)
                charges_arr = self._charges_and_dipole()[0] if hasattr(self, "_charges_and_dipole") else None
                if charges_arr is not None and len(charges_arr) == len(render_coords):
                    q = charges_arr
                else:
                    q = np.array([1.0 if s != "H" else 0.5 for s in self.symbols])
                disp = vib_disp
                # Oscillating displacement factor
                phase_val = getattr(self, "vib_phase", getattr(self, "vibration_phase", 0.0))
                cos_phase = math.cos(phase_val)
                d_mu = np.sum((q[:, None] * disp), axis=0) * cos_phase
                d_mag = np.linalg.norm(d_mu)
                if d_mag >= 0.01:
                    u_dir = d_mu / d_mag
                    arrow_len = min(3.8, max(0.8, d_mag * 1.5))
                    center_p = np.mean(render_coords, axis=0)
                    base_p = center_p - 0.5 * arrow_len * u_dir
                    shaft_len = arrow_len * 0.72
                    head_len = arrow_len * 0.28
                    shaft_r = 0.05
                    cone_r = 0.13

                    z_axis = np.array([0.0, 0.0, 1.0])
                    axis = np.cross(z_axis, u_dir)
                    axis_len = np.linalg.norm(axis)
                    angle_deg = 0.0
                    if axis_len > 1e-6:
                        axis = axis / axis_len
                        angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, u_dir), -1.0, 1.0)))
                    elif np.dot(z_axis, u_dir) < 0:
                        axis = np.array([1.0, 0.0, 0.0])
                        angle_deg = 180.0

                    glBindVertexArray(self.cyl_vao)
                    m_shaft = QtGui.QMatrix4x4()
                    m_shaft.translate(float(base_p[0]), float(base_p[1]), float(base_p[2]))
                    if angle_deg != 0:
                        m_shaft.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m_shaft.scale(shaft_r, shaft_r, float(shaft_len))
                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_shaft.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_shaft.normalMatrix().data())
                    glUniform3f(loc_color, 0.20, 0.85, 0.40)  # Vibrant emerald green
                    glUniform1i(loc_hl, 0)
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

                    glBindVertexArray(self.cone_vao)
                    head_base = base_p + u_dir * shaft_len
                    m_cone = QtGui.QMatrix4x4()
                    m_cone.translate(float(head_base[0]), float(head_base[1]), float(head_base[2]))
                    if angle_deg != 0:
                        m_cone.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m_cone.scale(cone_r, cone_r, float(head_len))
                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_cone.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_cone.normalMatrix().data())
                    glUniform3f(loc_color, 0.40, 0.95, 0.55)
                    glUniform1i(loc_hl, 0)
                    glDrawElements(GL_TRIANGLES, self.cone_indices_count, GL_UNSIGNED_INT, None)

        # 3b-2. Render QTAIM / AIM Critical Points (BCP cyan, RCP yellow, CCP orange)
        if getattr(self, "show_qtaim_cps", False) and getattr(self, "qtaim_critical_points", None):
            glBindVertexArray(self.sphere_vao)
            for cp in self.qtaim_critical_points:
                # coords in angstrom
                pos = getattr(cp, "coords_angstrom", None)
                if pos is None:
                    continue
                cp_type = getattr(cp, "cp_type", "")
                # Nuclear (3,-3) are already atoms, so emphasize BCP (3,-1), RCP (3,+1), CCP (3,+3)
                if "(3,-1)" in cp_type:
                    # Bond Critical Point: Vibrant cyan/teal sphere
                    col = (0.05, 0.85, 0.85)
                    rad = 0.16
                elif "(3,+1)" in cp_type:
                    # Ring Critical Point: Vibrant amber/yellow sphere
                    col = (0.95, 0.80, 0.10)
                    rad = 0.14
                elif "(3,+3)" in cp_type:
                    # Cage Critical Point: Vibrant purple sphere
                    col = (0.75, 0.25, 0.90)
                    rad = 0.14
                else:
                    # (3,-3) Nuclear Attractor: Small silver marker
                    col = (0.80, 0.80, 0.85)
                    rad = 0.12

                m_cp = QtGui.QMatrix4x4()
                m_cp.translate(float(pos[0]), float(pos[1]), float(pos[2]))
                m_cp.scale(rad)
                glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_cp.data())
                glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_cp.normalMatrix().data())
                glUniform3f(loc_color, col[0], col[1], col[2])
                glUniform1i(loc_hl, 0)
                glDrawElements(GL_TRIANGLES, self.sphere_indices_count, GL_UNSIGNED_INT, None)

        # 3b-3. Render QTAIM / Bader Gradient Bond Paths (Thin cylinders)
        if getattr(self, "show_qtaim_bond_paths", False) and getattr(self, "qtaim_bond_paths", None):
            glBindVertexArray(self.cyl_vao)
            path_col = (0.10, 0.85, 0.65)  # Vibrant emerald/cyan bond path
            shaft_r = 0.035
            z_axis = np.array([0.0, 0.0, 1.0])
            for bpath in self.qtaim_bond_paths:
                if not bpath or len(bpath) < 2:
                    continue
                for seg_i in range(len(bpath) - 1):
                    p1 = np.array(bpath[seg_i], dtype=float)
                    p2 = np.array(bpath[seg_i + 1], dtype=float)
                    v_dir = p2 - p1
                    dist = float(np.linalg.norm(v_dir))
                    if dist < 1e-4:
                        continue
                    u_dir = v_dir / dist
                    cross = np.cross(z_axis, u_dir)
                    c_len = float(np.linalg.norm(cross))
                    if c_len > 1e-6:
                        axis = cross / c_len
                        angle_deg = math.degrees(math.acos(np.clip(float(np.dot(z_axis, u_dir)), -1.0, 1.0)))
                    elif float(np.dot(z_axis, u_dir)) < 0.0:
                        axis = np.array([1.0, 0.0, 0.0])
                        angle_deg = 180.0
                    else:
                        axis = np.array([0.0, 1.0, 0.0])
                        angle_deg = 0.0

                    m_seg = QtGui.QMatrix4x4()
                    m_seg.translate(float(p1[0]), float(p1[1]), float(p1[2]))
                    if angle_deg != 0.0:
                        m_seg.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m_seg.scale(shaft_r, shaft_r, dist)

                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m_seg.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m_seg.normalMatrix().data())
                    glUniform3f(loc_color, path_col[0], path_col[1], path_col[2])
                    glUniform1i(loc_hl, 0)
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

        glBindVertexArray(0)
        glUseProgram(0)

        # 3c. Render 3D Volumetric Isosurfaces (Molecular Orbitals / ESP / Density)
        if getattr(self, "_retired_surface_gl_caches", None):
            for old_vao, old_vbo, old_ebo, _ in self._retired_surface_gl_caches:
                try:
                    glDeleteVertexArrays(1, [int(old_vao)])
                    glDeleteBuffers(2, [int(old_vbo), int(old_ebo)])
                except Exception as e:
                    log.debug("Failed deleting old GL buffers: %s", e)
            self._retired_surface_gl_caches.clear()
        active_surfaces = getattr(self, "active_isosurfaces", [])
        if active_surfaces:
            glUseProgram(self.surface_program)
            loc_s_proj = glGetUniformLocation(self.surface_program, "u_proj")
            loc_s_view = glGetUniformLocation(self.surface_program, "u_view")
            loc_s_col = glGetUniformLocation(self.surface_program, "u_color")
            loc_s_light = glGetUniformLocation(self.surface_program, "u_light_pos")
            loc_s_view_pos = glGetUniformLocation(self.surface_program, "u_view_pos")
            loc_s_opacity = glGetUniformLocation(self.surface_program, "u_opacity")
            loc_s_use_vcol = glGetUniformLocation(self.surface_program, "u_use_vertex_color")

            glUniformMatrix4fv(loc_s_proj, 1, GL_FALSE, proj_mat.data())
            glUniformMatrix4fv(loc_s_view, 1, GL_FALSE, view_mat.data())
            glUniform3f(loc_s_light, cam_world_pos[0] + 15.0, cam_world_pos[1] + 25.0, cam_world_pos[2] + 20.0)
            glUniform3f(loc_s_view_pos, cam_world_pos[0], cam_world_pos[1], cam_world_pos[2])
            opacity = getattr(self, "surface_opacity", 0.82)
            glUniform1f(loc_s_opacity, float(opacity))

            # Enable blending for transparent orbital lobes.
            # Use glBlendFuncSeparate so RGB blends smoothly while alpha remains opaque (1.0),
            # preventing desktop / background windows from bleeding through transparent lobes.
            glEnable(GL_BLEND)
            if getattr(self, "transparent_bg", False):
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)

            # Disable depth writes during transparent surface rendering to eliminate Z-fighting & flicker.
            # Depth testing remains enabled so atoms and bonds inside the orbital are properly occluded.
            glDepthMask(GL_FALSE)

            mesh_style = getattr(self, "surface_mesh_style", "Solid")  # "Solid", "Wireframe"
            if mesh_style == "Wireframe":
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Two-pass rendering for transparent solid meshes:
            # Pass 1: Render back-faces (cull front)
            # Pass 2: Render front-faces (cull back)
            # This ensures front and back transparent lobes blend consistently from any camera angle without flicker.
            passes = ["back", "front"] if mesh_style != "Wireframe" else ["both"]
            if mesh_style != "Wireframe":
                glEnable(GL_CULL_FACE)

            for p in passes:
                if p == "back":
                    glCullFace(GL_FRONT)
                elif p == "front":
                    glCullFace(GL_BACK)

                for mesh_obj in active_surfaces:
                    has_vcol = (getattr(mesh_obj, "vertex_colors", None) is not None and len(mesh_obj.vertex_colors) == len(mesh_obj.vertices))
                    glUniform1i(loc_s_use_vcol, 1 if has_vcol else 0)
                    glUniform3f(loc_s_col, float(mesh_obj.color[0]), float(mesh_obj.color[1]), float(mesh_obj.color[2]))

                    # Cache VAO / VBO / EBO on mesh_obj so rotating the camera does not re-upload meshes to GPU every frame
                    gl_cache = getattr(mesh_obj, "_gl_cache", None)
                    if gl_cache is None:
                        vao = glGenVertexArrays(1)
                        glBindVertexArray(vao)

                        vbo = glGenBuffers(1)
                        glBindBuffer(GL_ARRAY_BUFFER, vbo)
                        stride = 9 * 4 if has_vcol else 6 * 4
                        packed_v = np.empty((len(mesh_obj.vertices), 9 if has_vcol else 6), dtype=np.float32)
                        packed_v[:, 0:3] = mesh_obj.vertices
                        packed_v[:, 3:6] = mesh_obj.normals
                        if has_vcol:
                            packed_v[:, 6:9] = mesh_obj.vertex_colors
                        glBufferData(GL_ARRAY_BUFFER, packed_v.nbytes, packed_v, GL_STATIC_DRAW)

                        ebo = glGenBuffers(1)
                        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
                        faces_flat = mesh_obj.faces.flatten().astype(np.uint32)
                        glBufferData(GL_ELEMENT_ARRAY_BUFFER, faces_flat.nbytes, faces_flat, GL_STATIC_DRAW)

                        glEnableVertexAttribArray(0)
                        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
                        glEnableVertexAttribArray(1)
                        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(3 * 4))
                        if has_vcol:
                            glEnableVertexAttribArray(2)
                            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(6 * 4))

                        gl_cache = (vao, vbo, ebo, len(faces_flat))
                        try:
                            mesh_obj._gl_cache = gl_cache
                        except Exception as e:
                            log.debug("Failed to cache GL buffers on mesh_obj: %s", e)
                    else:
                        vao, vbo, ebo, n_indices = gl_cache
                        glBindVertexArray(vao)

                    glDrawElements(GL_TRIANGLES, gl_cache[3], GL_UNSIGNED_INT, None)
                    glBindVertexArray(0)

            if mesh_style != "Wireframe":
                glDisable(GL_CULL_FACE)
            else:
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

            # Re-enable depth writes for subsequent 3D passes / overlays
            glDepthMask(GL_TRUE)
            glUseProgram(0)

        # In clean PNG export mode, render strictly the 3D molecule (no 2D labels, rings, or HUD badges)
        if clean_export:
            return

        # Rubber-band selection rectangle
        if getattr(self, "band_active", False) and self.band_start is not None and self.band_current is not None:
            bp = QtGui.QPainter(self)
            bp.setPen(QtGui.QPen(QtGui.QColor(80, 190, 255), 1, QtCore.Qt.PenStyle.DashLine))
            bp.setBrush(QtGui.QColor(80, 190, 255, 45))
            bp.drawRect(QtCore.QRect(self.band_start, self.band_current).normalized())
            bp.end()

        # 4. Render 2D overlays (Atom labels & indices, active status banners, selection rings)
        is_white_vp = (getattr(self, "bg_color_mode", "dark") == "white")
        has_selection = (self.selected_atom_idx is not None) or bool(self.multi_selected) or (self.selected_bond is not None)
        frozen_set = set(getattr(self, "frozen_atoms", set()) or set())
        surface_extrema = []
        for s in active_surfaces:
            if getattr(s, "show_extrema_labels", False) and getattr(s, "extrema_points", None):
                surface_extrema.extend(s.extrema_points)

        if (
            has_selection
            or bool(frozen_set)
            or bool(active_surfaces)
            or self.show_labels
            or self.show_indices
            or self.show_charges
            or self.show_dipole
            or getattr(self, "show_bond_lengths", False)
            or getattr(self, "show_axis_triad", True)
            or getattr(self, "show_scale_bar", False)
            or (getattr(self, "show_unit_cell", True) and getattr(self, "unit_cell", None) is not None)
            or self.vibrating
            or self.partial_reorder_mode
            or self.reordering_active
            or self.picking_active
            or surface_extrema
            or bool(getattr(self, "tweezers_active", False))
            or bool(getattr(self, "atom_move_drag_active", False))
        ):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            fm = painter.fontMetrics()

            mvp = proj_mat * view_mat
            w = self.width()
            h = self.height()

            # Render Periodic Boundary Conditions (PBC) Unit Cell wireframe & axis labels
            if getattr(self, "show_unit_cell", True) and getattr(self, "unit_cell", None) is not None:
                uc = self.unit_cell
                edges = uc.get_box_edges() if hasattr(uc, "get_box_edges") else []
                is_white_vp = (self.bg_color_mode == "white" and not self.transparent_bg)

                pen_a = QtGui.QPen(QtGui.QColor("#ef4444"), 2.2)
                pen_b = QtGui.QPen(QtGui.QColor("#10b981"), 2.2)
                pen_c = QtGui.QPen(QtGui.QColor("#3b82f6"), 2.2)
                pen_box = QtGui.QPen(
                    QtGui.QColor("#94a3b8" if is_white_vp else "#475569"),
                    1.4,
                    QtCore.Qt.PenStyle.DashLine,
                )

                origin_pt = getattr(uc, "origin", (0.0, 0.0, 0.0))
                for start_pt, end_pt, ax_label in edges:
                    p4_s = mvp * QtGui.QVector4D(float(start_pt[0]), float(start_pt[1]), float(start_pt[2]), 1.0)
                    p4_e = mvp * QtGui.QVector4D(float(end_pt[0]), float(end_pt[1]), float(end_pt[2]), 1.0)
                    if p4_s.w() <= 0.05 or p4_e.w() <= 0.05:
                        continue
                    ndc_s = p4_s / p4_s.w()
                    ndc_e = p4_e / p4_e.w()
                    sx_s = int((ndc_s.x() * 0.5 + 0.5) * w)
                    sy_s = int((-ndc_s.y() * 0.5 + 0.5) * h)
                    sx_e = int((ndc_e.x() * 0.5 + 0.5) * w)
                    sy_e = int((-ndc_e.y() * 0.5 + 0.5) * h)

                    is_origin_edge = np.allclose(start_pt, origin_pt, atol=1e-5)
                    if is_origin_edge and ax_label == "a":
                        painter.setPen(pen_a)
                    elif is_origin_edge and ax_label == "b":
                        painter.setPen(pen_b)
                    elif is_origin_edge and ax_label == "c":
                        painter.setPen(pen_c)
                    else:
                        painter.setPen(pen_box)

                    painter.drawLine(sx_s, sy_s, sx_e, sy_e)

                    if is_origin_edge:
                        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                        if ax_label == "a":
                            painter.setPen(QtGui.QColor("#dc2626"))
                            painter.drawText(sx_e + 5, sy_e - 5, "a")
                        elif ax_label == "b":
                            painter.setPen(QtGui.QColor("#059669"))
                            painter.drawText(sx_e + 5, sy_e - 5, "b")
                        elif ax_label == "c":
                            painter.setPen(QtGui.QColor("#2563eb"))
                            painter.drawText(sx_e + 5, sy_e - 5, "c")

            # Render glowing high-contrast selection rings & frozen constraint rings around atoms
            if (has_selection or bool(frozen_set)) and len(render_coords) == len(self.symbols):
                for i in range(len(self.symbols)):
                    is_prim = (i == self.selected_atom_idx)
                    is_multi = (i in self.multi_selected)
                    is_froz = (i in frozen_set)
                    if not (is_prim or is_multi or is_froz):
                        continue
                    pos = render_coords[i]
                    p4 = mvp * QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
                    if p4.w() <= 0.05:
                        continue
                    ndc = p4 / p4.w()
                    sx = int((ndc.x() * 0.5 + 0.5) * w)
                    sy = int((-ndc.y() * 0.5 + 0.5) * h)
                    cov_r = get_cov_radius(self.symbols[i])
                    r_px = max(13.0, (cov_r / max(self.zoom, 1.0)) * h * 0.45)

                    painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                    if is_froz:
                        # Frozen atom constraint halo (dashed icy blue-violet ring)
                        froz_pen = QtGui.QPen(QtGui.QColor(99, 102, 241, 215), 2.2, QtCore.Qt.PenStyle.DashLine)
                        painter.setPen(froz_pen)
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 9), int(r_px + 9))
                    if is_prim:
                        # Primary: bright radiant gold ring + 4 tick marks
                        painter.setPen(QtGui.QPen(QtGui.QColor(250, 204, 21, 55), 7))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 7), int(r_px + 7))
                        painter.setPen(QtGui.QPen(QtGui.QColor(254, 240, 138), 2.5))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 5), int(r_px + 5))
                        painter.setPen(QtGui.QPen(QtGui.QColor(234, 179, 8), 2.0))
                        d_in = r_px + 3
                        d_out = r_px + 9
                        for ang in (45, 135, 225, 315):
                            rad = math.radians(ang)
                            x1 = int(sx + d_in * math.cos(rad))
                            y1 = int(sy + d_in * math.sin(rad))
                            x2 = int(sx + d_out * math.cos(rad))
                            y2 = int(sy + d_out * math.sin(rad))
                            painter.drawLine(x1, y1, x2, y2)
                    elif is_multi:
                        # Secondary: vibrant cyan ring
                        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 55), 5))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 6), int(r_px + 6))
                        painter.setPen(QtGui.QPen(QtGui.QColor(186, 230, 253), 2.0))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 4), int(r_px + 4))

            # Render 3D Elastic Strain Tether & Spring Line while dragging in Tweezers mode
            if (
                getattr(self, "atom_move_drag_active", False)
                and getattr(self, "atom_move_drag_idx", None) is not None
                and hasattr(self, "atom_move_drag_start_coord")
                and len(render_coords) == len(self.symbols)
            ):
                drag_i = self.atom_move_drag_idx
                if 0 <= drag_i < len(render_coords):
                    curr_pt = render_coords[drag_i]
                    orig_pt = self.atom_move_drag_start_coord
                    p4_curr = mvp * QtGui.QVector4D(float(curr_pt[0]), float(curr_pt[1]), float(curr_pt[2]), 1.0)
                    p4_orig = mvp * QtGui.QVector4D(float(orig_pt[0]), float(orig_pt[1]), float(orig_pt[2]), 1.0)
                    if p4_curr.w() > 0.05 and p4_orig.w() > 0.05:
                        ndc_c = p4_curr / p4_curr.w()
                        ndc_o = p4_orig / p4_orig.w()
                        sc_x = int((ndc_c.x() * 0.5 + 0.5) * w)
                        sc_y = int((-ndc_c.y() * 0.5 + 0.5) * h)
                        so_x = int((ndc_o.x() * 0.5 + 0.5) * w)
                        so_y = int((-ndc_o.y() * 0.5 + 0.5) * h)

                        dist_ang = float(np.linalg.norm(curr_pt - orig_pt))
                        if dist_ang > 0.05:
                            # Dynamic color: Green (0 Å) -> Yellow (0.8 Å) -> Red (> 1.5 Å)
                            t_strain = min(1.0, max(0.0, dist_ang / 1.6))
                            red = int(min(255, t_strain * 2.0 * 255))
                            green = int(min(255, (1.0 - max(0.0, t_strain - 0.5) * 2.0) * 255))
                            strain_col = QtGui.QColor(red, green, 40)

                            # Draw tether dashed line
                            pen_tether = QtGui.QPen(strain_col, 2.4, QtCore.Qt.PenStyle.DashLine)
                            painter.setPen(pen_tether)
                            painter.drawLine(so_x, so_y, sc_x, sc_y)

                            # Draw origin anchor ring
                            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                            painter.setPen(QtGui.QPen(QtGui.QColor(strain_col.red(), strain_col.green(), strain_col.blue(), 180), 1.5))
                            painter.drawEllipse(QtCore.QPoint(so_x, so_y), 6, 6)

                            # Strain badge at midpoint
                            mid_x = (sc_x + so_x) // 2
                            mid_y = (sc_y + so_y) // 2
                            badge_txt = f"Δr = {dist_ang:.2f} Å"
                            painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                            bw = painter.fontMetrics().horizontalAdvance(badge_txt) + 8
                            bh = painter.fontMetrics().height() + 4
                            b_rect = QtCore.QRect(mid_x - bw // 2, mid_y - bh // 2, bw, bh)
                            painter.setBrush(QtGui.QColor(15, 23, 42, 220))
                            painter.setPen(QtGui.QPen(strain_col, 1.2))
                            painter.drawRoundedRect(b_rect, 4, 4)
                            painter.setPen(QtGui.QColor("#ffffff"))
                            painter.drawText(b_rect, QtCore.Qt.AlignmentFlag.AlignCenter, badge_txt)

            if self.show_labels or self.show_indices or self.show_charges or bool(frozen_set):
                for i, (sym, pos) in enumerate(zip(self.symbols, render_coords)):
                    if not (self.show_labels or self.show_indices or self.show_charges or (i in frozen_set)):
                        continue
                    p4 = mvp * QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
                    if p4.w() <= 0.05:
                        continue
                    ndc = p4 / p4.w()
                    screen_x = int((ndc.x() * 0.5 + 0.5) * w)
                    screen_y = int((-ndc.y() * 0.5 + 0.5) * h)

                    label_text = ""
                    if self.reordering_active:
                        if i in self.reorder_list:
                            order_num = self.reorder_list.index(i) + 1
                            label_text = f"#{order_num} ({sym})"
                        else:
                            label_text = f"{sym}{i+1}"
                    elif self.show_labels and self.show_indices:
                        label_text = f"{sym}{i+1}"
                    elif self.show_labels:
                        label_text = sym
                    elif self.show_indices:
                        label_text = f"{i+1}"
                    if self.show_charges and charges_arr is not None and i < len(charges_arr):
                        q_str = f"{charges_arr[i]:+.2f}e"
                        label_text = f"{label_text} ({q_str})" if label_text else q_str
                    if i in frozen_set:
                        label_text = f"{label_text} 🔒".strip() if label_text else f"{sym}{i+1} 🔒"

                    # Role tag when feature box is clicked in Z-Matrix
                    if i in self.highlighted_feature_atoms:
                        role_name = self.highlighted_feature_atoms[i].capitalize()
                        label_text = f"[{role_name}] {label_text}"

                    # Text pill styling: high-contrast background and crisp border adapted to dark/white viewport
                    is_white_vp = (getattr(self, "bg_color_mode", "dark") == "white")
                    if is_white_vp:
                        bg_color = QtGui.QColor(248, 250, 252, 235)  # Light porcelain background
                        border_color = QtGui.QColor("#0284c7")
                        text_color = QtGui.QColor("#0f172a")          # Deep charcoal
                    else:
                        bg_color = QtGui.QColor(15, 23, 42, 220)      # Dark slate semi-transparent
                        border_color = QtGui.QColor("#38bdf8")
                        text_color = QtGui.QColor("#ffffff")

                    if self.partial_reorder_mode and i in self.partial_reorder_picks:
                        step_idx = self.partial_reorder_picks.index(i)
                        role = ["Target", "Bond", "Angle", "Dihedral"][step_idx] if step_idx < 4 else f"#{step_idx+1}"
                        label_text = f"[{role}] {sym}{i+1}"
                        text_color = QtGui.QColor("#854d0e") if is_white_vp else QtGui.QColor("#fef08a")
                        border_color = QtGui.QColor("#d97706") if is_white_vp else QtGui.QColor("#f59e0b")
                        bg_color = QtGui.QColor(254, 243, 199, 240) if is_white_vp else QtGui.QColor(69, 26, 3, 230)
                    elif i in self.highlighted_feature_atoms:
                        role = self.highlighted_feature_atoms[i]
                        if role == "target":
                            text_color = QtGui.QColor("#854d0e") if is_white_vp else QtGui.QColor("#fef08a")
                            border_color = QtGui.QColor("#ca8a04") if is_white_vp else QtGui.QColor("#eab308")
                            bg_color = QtGui.QColor(254, 249, 195, 240) if is_white_vp else QtGui.QColor(50, 40, 5, 235)
                        elif role == "bond":
                            text_color = QtGui.QColor("#0e7490") if is_white_vp else QtGui.QColor("#a5f3fc")
                            border_color = QtGui.QColor("#0891b2") if is_white_vp else QtGui.QColor("#06b6d4")
                            bg_color = QtGui.QColor(207, 250, 254, 240) if is_white_vp else QtGui.QColor(8, 47, 73, 235)
                        elif role == "angle":
                            text_color = QtGui.QColor("#86198f") if is_white_vp else QtGui.QColor("#f5d0fe")
                            border_color = QtGui.QColor("#a21caf") if is_white_vp else QtGui.QColor("#c026d3")
                            bg_color = QtGui.QColor(250, 232, 255, 240) if is_white_vp else QtGui.QColor(59, 7, 100, 235)
                        elif role == "dihedral":
                            text_color = QtGui.QColor("#9a3412") if is_white_vp else QtGui.QColor("#fed7aa")
                            border_color = QtGui.QColor("#ea580c") if is_white_vp else QtGui.QColor("#f97316")
                            bg_color = QtGui.QColor(255, 237, 213, 240) if is_white_vp else QtGui.QColor(67, 20, 7, 235)
                    elif self.reordering_active and i in self.reorder_list:
                        text_color = QtGui.QColor("#15803d") if is_white_vp else QtGui.QColor("#86efac")
                        border_color = QtGui.QColor("#16a34a") if is_white_vp else QtGui.QColor("#22c55e")
                        if is_white_vp: bg_color = QtGui.QColor(220, 252, 231, 240)
                    elif i in self.picked_references:
                        text_color = QtGui.QColor("#15803d") if is_white_vp else QtGui.QColor("#86efac")
                        border_color = QtGui.QColor("#16a34a") if is_white_vp else QtGui.QColor("#22c55e")
                        if is_white_vp: bg_color = QtGui.QColor(220, 252, 231, 240)
                    elif i == self.selected_atom_idx:
                        text_color = QtGui.QColor("#854d0e") if is_white_vp else QtGui.QColor("#fef08a")
                        border_color = QtGui.QColor("#d97706") if is_white_vp else QtGui.QColor("#fbbf24")
                        if is_white_vp: bg_color = QtGui.QColor(254, 243, 199, 245)
                    elif i in self.multi_selected:
                        text_color = QtGui.QColor("#0369a1") if is_white_vp else QtGui.QColor("#a5f3fc")
                        border_color = QtGui.QColor("#0284c7") if is_white_vp else QtGui.QColor("#38bdf8")
                        if is_white_vp: bg_color = QtGui.QColor(224, 242, 254, 245)
                    else:
                        text_color = QtGui.QColor("#0f172a") if is_white_vp else QtGui.QColor("#f1f5f9")
                        border_color = QtGui.QColor(148, 163, 184, 200) if is_white_vp else QtGui.QColor(100, 116, 139, 180)

                    # Draw pill background
                    t_w = fm.horizontalAdvance(label_text)
                    t_h = fm.height()
                    pill_x = screen_x + 8
                    pill_y = screen_y - t_h - 2
                    pill_rect = QtCore.QRect(pill_x, pill_y, t_w + 10, t_h + 4)

                    painter.setBrush(bg_color)
                    painter.setPen(QtGui.QPen(border_color, 1.2))
                    painter.drawRoundedRect(pill_rect, 4, 4)

                    painter.setPen(text_color)
                    painter.drawText(pill_x + 5, pill_y + t_h - 1, label_text)

            # Render Chemcraft-style "Show All Bond Lengths" 3D overlays at bond midpoints
            if getattr(self, "show_bond_lengths", False) and self.bonds and len(render_coords) == len(self.symbols):
                bl_font = QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold)
                painter.setFont(bl_font)
                bl_fm = painter.fontMetrics()
                vp_w = float(self.width())
                vp_h = float(self.height())
                for i1, i2 in self.bonds:
                    if not (0 <= i1 < len(render_coords) and 0 <= i2 < len(render_coords)):
                        continue
                    p1_arr = render_coords[i1]
                    p2_arr = render_coords[i2]
                    mid = 0.5 * (p1_arr + p2_arr)
                    p4 = mvp * QtGui.QVector4D(float(mid[0]), float(mid[1]), float(mid[2]), 1.0)
                    if p4.w() <= 0.05:
                        continue
                    ndc = p4 / p4.w()
                    sx = int((ndc.x() * 0.5 + 0.5) * vp_w)
                    sy = int((-ndc.y() * 0.5 + 0.5) * vp_h)
                    dist_val = float(np.linalg.norm(p2_arr - p1_arr))
                    bl_txt = f"{dist_val:.3f} Å"
                    tw = bl_fm.horizontalAdvance(bl_txt)
                    th = bl_fm.height()
                    bx = sx - tw // 2 - 4
                    by = sy - th // 2 - 2
                    b_rect = QtCore.QRect(bx, by, tw + 8, th + 4)
                    if is_white_vp:
                        painter.setBrush(QtGui.QColor(255, 255, 255, 225))
                        painter.setPen(QtGui.QPen(QtGui.QColor(15, 118, 110, 210), 1.2))
                    else:
                        painter.setBrush(QtGui.QColor(18, 22, 34, 215))
                        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 200), 1.2))
                    painter.drawRoundedRect(b_rect, 4, 4)
                    painter.setPen(QtGui.QColor(15, 23, 42) if is_white_vp else QtGui.QColor(186, 230, 253))
                    painter.drawText(bx + 4, by + th - 2, bl_txt)
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))

            # Render GaussView-style 3D floating measurement overlays (Distance, Angle, Dihedral)
            active_measurements = []
            if self.show_measurements and len(self.multi_selected) >= 2:
                active_measurements.append(list(self.multi_selected))
            for pm in getattr(self, "pinned_measurements", []):
                if pm not in active_measurements:
                    active_measurements.append(pm)

            if active_measurements and len(render_coords) == len(self.symbols):
                def _proj_pos(pos_arr):
                    p4 = mvp * QtGui.QVector4D(float(pos_arr[0]), float(pos_arr[1]), float(pos_arr[2]), 1.0)
                    if p4.w() <= 0.05:
                        return None
                    ndc = p4 / p4.w()
                    vp_w = float(self.width())
                    vp_h = float(self.height())
                    return int((ndc.x() * 0.5 + 0.5) * vp_w), int((-ndc.y() * 0.5 + 0.5) * vp_h)

                def _draw_badge(sx, sy, text, border_col_hex, bg_alpha=230):
                    b_font = QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold)
                    painter.setFont(b_font)
                    b_fm = painter.fontMetrics()
                    tw = b_fm.horizontalAdvance(text)
                    th = b_fm.height()
                    bx = sx - tw // 2 - 6
                    by = sy - th // 2 - 3
                    b_rect = QtCore.QRect(bx, by, tw + 12, th + 6)
                    if is_white_vp:
                        painter.setBrush(QtGui.QColor(248, 250, 252, 240))
                        painter.setPen(QtGui.QPen(QtGui.QColor(border_col_hex), 1.5))
                        painter.drawRoundedRect(b_rect, 5, 5)
                        painter.setPen(QtGui.QColor("#0f172a"))
                    else:
                        painter.setBrush(QtGui.QColor(15, 23, 42, bg_alpha))
                        painter.setPen(QtGui.QPen(QtGui.QColor(border_col_hex), 1.5))
                        painter.drawRoundedRect(b_rect, 5, 5)
                        painter.setPen(QtGui.QColor("#ffffff"))
                    painter.drawText(bx + 6, by + th - 2, text)

                for meas in active_measurements:
                    k_sel = len(meas)
                    if not all(0 <= idx < len(render_coords) for idx in meas):
                        continue

                    if k_sel == 2:
                        a, b = meas[0], meas[1]
                        pa, pb = render_coords[a], render_coords[b]
                        dist = float(np.linalg.norm(pa - pb))
                        s_a = _proj_pos(pa)
                        s_b = _proj_pos(pb)
                        if s_a and s_b:
                            painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 200), 2.0, QtCore.Qt.PenStyle.DashLine))
                            painter.drawLine(s_a[0], s_a[1], s_b[0], s_b[1])
                            s_mid = _proj_pos((pa + pb) * 0.5)
                            if s_mid:
                                # Educational reference: color-code if bond length is strained or clashing
                                badge_col = "#38bdf8"
                                if dist < 0.70:
                                    badge_col = "#ef4444"  # Severe clash (red)
                                elif dist < 0.95:
                                    badge_col = "#f59e0b"  # Short bond (amber)
                                _draw_badge(s_mid[0], s_mid[1], f"{dist:.3f} Å", badge_col)

                    elif k_sel == 3:
                        a, b, c = meas[0], meas[1], meas[2]
                        pa, pb, pc = render_coords[a], render_coords[b], render_coords[c]
                        v1 = pa - pb
                        v2 = pc - pb
                        cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                        deg = float(math.degrees(math.acos(np.clip(cos_th, -1.0, 1.0))))
                        s_a, s_b, s_c = _proj_pos(pa), _proj_pos(pb), _proj_pos(pc)
                        if s_a and s_b and s_c:
                            painter.setPen(QtGui.QPen(QtGui.QColor(192, 38, 211, 200), 2.0, QtCore.Qt.PenStyle.DashLine))
                            painter.drawLine(s_a[0], s_a[1], s_b[0], s_b[1])
                            painter.drawLine(s_c[0], s_c[1], s_b[0], s_b[1])
                            u1 = v1 / (np.linalg.norm(v1) + 1e-12)
                            u2 = v2 / (np.linalg.norm(v2) + 1e-12)
                            bisect = u1 + u2
                            norm_b = np.linalg.norm(bisect)
                            pos_badge = pb + (bisect / norm_b) * 0.65 if norm_b > 1e-5 else pb + np.array([0.0, 0.5, 0.0])
                            s_badge = _proj_pos(pos_badge) or (s_b[0] + 25, s_b[1] - 25)
                            # Educational hint: compare with ideal tetrahedral (109.5°), planar (120°), or linear (180°)
                            ref_txt = ""
                            if abs(deg - 109.5) <= 3.5:
                                ref_txt = " (~sp³)"
                            elif abs(deg - 120.0) <= 3.5:
                                ref_txt = " (~sp²)"
                            elif abs(deg - 180.0) <= 3.5:
                                ref_txt = " (~sp)"
                            _draw_badge(s_badge[0], s_badge[1], f"θ = {deg:.1f}°{ref_txt}", "#c026d3")

                    elif k_sel >= 4:
                        a, b, c, d = meas[:4]
                        pa, pb, pc, pd = render_coords[a], render_coords[b], render_coords[c], render_coords[d]
                        b0 = -1.0 * (pb - pa)
                        b1 = pc - pb
                        b2 = pd - pc
                        b1_u = b1 / (np.linalg.norm(b1) + 1e-12)
                        v_vec = b0 - np.dot(b0, b1_u) * b1_u
                        w_vec = b2 - np.dot(b2, b1_u) * b1_u
                        deg = float(math.degrees(math.atan2(np.dot(np.cross(b1_u, v_vec), w_vec), np.dot(v_vec, w_vec))))
                        s_a, s_b, s_c, s_d = _proj_pos(pa), _proj_pos(pb), _proj_pos(pc), _proj_pos(pd)
                        if s_a and s_b and s_c and s_d:
                            painter.setPen(QtGui.QPen(QtGui.QColor(249, 115, 22, 200), 2.0, QtCore.Qt.PenStyle.DashLine))
                            painter.drawLine(s_a[0], s_a[1], s_b[0], s_b[1])
                            painter.drawLine(s_b[0], s_b[1], s_c[0], s_c[1])
                            painter.drawLine(s_c[0], s_c[1], s_d[0], s_d[1])
                            s_mid = _proj_pos((pb + pc) * 0.5)
                            if s_mid:
                                _draw_badge(s_mid[0], s_mid[1], f"φ = {deg:.1f}°", "#f97316")

            # Render Multiwfn-style Surface Extrema (V_min, V_max, ALIE min)
            if surface_extrema:
                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                e_fm = painter.fontMetrics()
                for ext in surface_extrema[:16]:
                    pos = ext["pos"]
                    p4 = mvp * QtGui.QVector4D(float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
                    if p4.w() <= 0.05:
                        continue
                    ndc = p4 / p4.w()
                    sx = int((ndc.x() * 0.5 + 0.5) * w)
                    sy = int((-ndc.y() * 0.5 + 0.5) * h)

                    is_min = (ext.get("type") == "min")
                    lbl = ext.get("label", "")
                    tw = e_fm.horizontalAdvance(lbl)
                    th = e_fm.height()

                    # Dot marker
                    dot_color = QtGui.QColor("#38bdf8") if is_min else QtGui.QColor("#ef4444")
                    painter.setBrush(dot_color)
                    painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1.2))
                    painter.drawEllipse(QtCore.QPoint(sx, sy), 4, 4)

                    # Text label pill
                    lx = sx + 6
                    ly = sy - th - 2
                    p_rect = QtCore.QRect(lx, ly, tw + 8, th + 3)
                    painter.setBrush(QtGui.QColor(15, 23, 42, 225))
                    painter.setPen(QtGui.QPen(dot_color, 1))
                    painter.drawRoundedRect(p_rect, 3, 3)
                    painter.setPen(QtGui.QColor("#f8fafc"))
                    painter.drawText(lx + 4, ly + th - 2, lbl)

            # Interactive / Vibrations instructions banner at the top
            banner_msg = ""
            border_banner = QtGui.QColor("#38bdf8")

            if self.vibrating:
                mode_str = f"Mode #{self.vib_mode_idx + 1}: " if self.vib_mode_idx is not None else ""
                banner_msg = f"🎵 Vibrating Mode: {mode_str}{self.vib_frequency:.1f} cm⁻¹ (Amp: {self.vib_amplitude:.2f})"
                border_banner = QtGui.QColor("#f59e0b")
            elif self.partial_reorder_mode and self.partial_reorder_banner:
                banner_msg = self.partial_reorder_banner
                border_banner = QtGui.QColor("#f59e0b")
            elif self.reordering_active:
                banner_msg = f"Reordering Z-Matrix: Click atoms in desired sequence ({len(self.reorder_list)}/{len(self.symbols)} selected). Esc to cancel."
                border_banner = QtGui.QColor("#22c55e")
            elif self.picking_active and self.picking_banner:
                banner_msg = self.picking_banner
                border_banner = QtGui.QColor("#38bdf8")

            if banner_msg:
                banner_rect = QtCore.QRect(10, 38, w - 20, 32)
                painter.setBrush(QtGui.QColor(30, 41, 59, 235))
                painter.setPen(QtGui.QPen(border_banner, 2))
                painter.drawRoundedRect(banner_rect, 6, 6)

                painter.setPen(QtGui.QColor("#f8fafc"))
                painter.setFont(QtGui.QFont("DejaVu Sans", 10, QtGui.QFont.Weight.Bold))
                painter.drawText(banner_rect, QtCore.Qt.AlignmentFlag.AlignCenter, banner_msg)

            # Top-right HUD badge: Dipole moment & partial charge summary
            if (self.show_dipole or self.show_charges) and len(self.symbols) > 0:
                _, mu_v, mu_d = self._charges_and_dipole()
                dip_txt = f"Dipole |μ| = {mu_d:.2f} D  (δ⁻ Red → δ⁺ Blue)"
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                d_fm = painter.fontMetrics()
                d_tw = d_fm.horizontalAdvance(dip_txt)
                d_rect = QtCore.QRect(w - d_tw - 26, 8, d_tw + 16, 24)
                painter.setBrush(QtGui.QColor(248, 250, 252, 240) if is_white_vp else QtGui.QColor(15, 23, 42, 220))
                painter.setPen(QtGui.QPen(QtGui.QColor("#db2777" if is_white_vp else "#ec4899"), 1.2))
                painter.drawRoundedRect(d_rect, 5, 5)
                painter.setPen(QtGui.QColor("#831843" if is_white_vp else "#fdf2f8"))
                painter.drawText(d_rect, QtCore.Qt.AlignmentFlag.AlignCenter, dip_txt)

            # Top-left HUD badge: Active Element Brush indicator
            brush_sym = getattr(self, "brush_element", "C") or "C"
            brush_text = f"Brush: {brush_sym}"
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            b_fm = painter.fontMetrics()
            b_tw = b_fm.horizontalAdvance(brush_text)
            # Layout: 8px pad | 10px dot | 6px gap | text | 10px pad
            b_rect = QtCore.QRect(10, 8, 8 + 10 + 6 + b_tw + 10, 24)
            painter.setBrush(QtGui.QColor(248, 250, 252, 240) if is_white_vp else QtGui.QColor(15, 23, 42, 215))
            painter.setPen(QtGui.QPen(QtGui.QColor("#0284c7" if is_white_vp else "#38bdf8"), 1))
            painter.drawRoundedRect(b_rect, 5, 5)
            # Dot in element color
            brush_col = get_color(brush_sym, bg_mode="white" if is_white_vp else "dark")
            painter.setBrush(QtGui.QColor(int(brush_col[0]*255), int(brush_col[1]*255), int(brush_col[2]*255)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(b_rect.left() + 8, b_rect.center().y() - 4, 10, 10)
            painter.setPen(QtGui.QColor("#0f172a" if is_white_vp else "#f1f5f9"))
            text_rect = QtCore.QRect(b_rect.left() + 24, b_rect.top(), b_tw + 10, b_rect.height())
            painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignVCenter | QtCore.Qt.AlignmentFlag.AlignLeft, brush_text)

            # Left-Click drag visual line (elastic rubberband indicating new atom addition or cross-bond)
            if self.atom_add_drag_active:
                # Check if hovered over another atom
                hovered_target = self._pick_atom(self.atom_add_drag_current_pos, tolerance_px=30.0)
                is_cross = (hovered_target is not None and hovered_target != self.atom_add_drag_anchor_idx)
                line_color = QtGui.QColor("#4ade80") if is_cross else QtGui.QColor("#38bdf8") # green for connect, blue for new
                pen = QtGui.QPen(line_color, 2, QtCore.Qt.PenStyle.DashLine)
                painter.setPen(pen)
                painter.drawLine(self.atom_add_drag_start_pos, self.atom_add_drag_current_pos)
                # Circle at end of drag
                painter.setBrush(QtGui.QColor(line_color.red(), line_color.green(), line_color.blue(), 190))
                painter.drawEllipse(self.atom_add_drag_current_pos, 8, 8)
                if is_cross:
                    # Label indicating connection / ring closure
                    tgt_sym = self.symbols[hovered_target] if hovered_target < len(self.symbols) else ""
                    snap_txt = f"Connect to {tgt_sym}{hovered_target+1} (Ring / Cross-Bond)"
                    painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                    snap_rect = QtCore.QRect(self.atom_add_drag_current_pos.x() + 12, self.atom_add_drag_current_pos.y() - 12, 230, 22)
                    painter.setBrush(QtGui.QColor(22, 101, 52, 220))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#86efac"), 1.2))
                    painter.drawRoundedRect(snap_rect, 4, 4)
                    painter.setPen(QtGui.QColor("#ffffff"))
                    painter.drawText(snap_rect, QtCore.Qt.AlignmentFlag.AlignCenter, snap_txt)
                elif self.active_drag_label:
                    # Label indicating element or fragment being added
                    painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                    d_fm = painter.fontMetrics()
                    d_tw = d_fm.horizontalAdvance(self.active_drag_label)
                    snap_rect = QtCore.QRect(self.atom_add_drag_current_pos.x() + 12, self.atom_add_drag_current_pos.y() - 12, d_tw + 16, 22)
                    painter.setBrush(QtGui.QColor(15, 23, 42, 220))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 1.2))
                    painter.drawRoundedRect(snap_rect, 4, 4)
                    painter.setPen(QtGui.QColor("#f8fafc"))
                    painter.drawText(snap_rect, QtCore.Qt.AlignmentFlag.AlignCenter, self.active_drag_label)

            # Top-left HUD badge: Publication Molecule Watermark / Figure Overlay (positioned below Element Brush badge)
            if getattr(self, "show_watermark", True) and len(self.symbols) > 0:
                # Formula & symmetry info
                sym_counts: Dict[str, int] = {}
                for s in self.symbols:
                    sym_counts[s] = sym_counts.get(s, 0) + 1
                # Hill system sorting: C first, then H, then alphabetical
                sorted_elems = sorted(sym_counts.keys(), key=lambda e: (0, e) if e == "C" else ((1, e) if e == "H" else (2, e)))
                formula = "".join(f"{e}{sym_counts[e] if sym_counts[e] > 1 else ''}" for e in sorted_elems)
                pt_grp = getattr(self, "current_point_group_label", "")
                wm_txt = f"{formula}  [{pt_grp}]" if pt_grp else formula

                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                wfm = painter.fontMetrics()
                wtw = wfm.horizontalAdvance(wm_txt)
                wm_y = 74 if banner_msg else 38
                wrect = QtCore.QRect(10, wm_y, wtw + 14, 22)
                if is_white_vp:
                    painter.setBrush(QtGui.QColor(248, 250, 252, 230))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1.0))
                else:
                    painter.setBrush(QtGui.QColor(15, 23, 42, 190))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1.0))
                painter.drawRoundedRect(wrect, 4, 4)
                painter.setPen(QtGui.QColor("#0f172a" if is_white_vp else "#e2e8f0"))
                painter.drawText(wrect, QtCore.Qt.AlignmentFlag.AlignCenter, wm_txt)

            # Bottom-left HUD badge: Selection summary
            if has_selection and len(self.symbols) > 0:
                n_syms = len(self.symbols)
                if self.selected_bond is not None:
                    u, v = self.selected_bond
                    if 0 <= u < n_syms and 0 <= v < n_syms:
                        bo_name = self._bond_order_name(u, v)
                        sel_str = f"Selected Bond: {self.symbols[u]}{u+1} — {self.symbols[v]}{v+1} ({bo_name}) | Press 1, 2, 3 to change order"
                    else:
                        sel_str = f"Selected: Bond"
                else:
                    valid_sel = [i for i in self.multi_selected if 0 <= i < n_syms]
                    if not valid_sel and self.selected_atom_idx is not None and 0 <= self.selected_atom_idx < n_syms:
                        valid_sel = [self.selected_atom_idx]
                    sel_count = len(valid_sel)
                    if sel_count > 0:
                        atoms_list = [f"{self.symbols[i]}{i+1}" for i in valid_sel]
                        if sel_count <= 8:
                            sel_str = f"Selected ({sel_count}): {', '.join(atoms_list)}"
                        else:
                            shown = ", ".join(atoms_list[:6])
                            sel_str = f"Selected ({sel_count}): {shown}, ... (+{sel_count - 6} more)"
                    else:
                        sel_str = f"Selected: 0 atoms"

                s_tw = fm.horizontalAdvance(sel_str)
                s_th = fm.height()
                badge_rect = QtCore.QRect(10, self.height() - s_th - 16, s_tw + 20, s_th + 8)
                if is_white_vp:
                    painter.setBrush(QtGui.QColor(254, 252, 232, 245))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#d97706"), 1.5))
                    painter.drawRoundedRect(badge_rect, 5, 5)
                    painter.setPen(QtGui.QColor("#78350f"))
                else:
                    painter.setBrush(QtGui.QColor(15, 23, 42, 230))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1.5))
                    painter.drawRoundedRect(badge_rect, 5, 5)
                    painter.setPen(QtGui.QColor("#fef08a"))
                painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, sel_str)

            # Active Tweezers HUD Mode Pill (Bottom Center-Left)
            if getattr(self, "tweezers_active", False) and len(self.symbols) > 0:
                twz_txt = "🧲 Tweezers Active: Left-Click & Drag any atom to move & relax"
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                twz_fm = painter.fontMetrics()
                twz_tw = twz_fm.horizontalAdvance(twz_txt)
                twz_th = twz_fm.height()
                twz_rect = QtCore.QRect(12, 12, twz_tw + 20, twz_th + 10)
                painter.setBrush(QtGui.QColor(2, 132, 199, 235))
                painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 1.8))
                painter.drawRoundedRect(twz_rect, 6, 6)
                painter.setPen(QtGui.QColor("#ffffff"))
                painter.drawText(twz_rect, QtCore.Qt.AlignmentFlag.AlignCenter, twz_txt)
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))

            # Top-right HUD badge: Clear 3D Orbital / Surface Pill (positioned cleanly below the Formula/MW pill)
            if active_surfaces:
                surf_txt = "✖ Clear 3D Surface (Esc)"
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                s_fm = painter.fontMetrics()
                s_tw = s_fm.horizontalAdvance(surf_txt)
                has_formula_pill = len(self.symbols) > 0
                has_extra_badge = (self.show_dipole or self.show_charges) and has_formula_pill
                if has_extra_badge:
                    y_surf = 72
                elif has_formula_pill:
                    y_surf = 44
                else:
                    y_surf = 12
                s_w = s_tw + 18
                s_rect = QtCore.QRect(w - s_w - 12, y_surf, s_w, 26)
                self.clear_surface_rect = s_rect
                is_hover_clear = getattr(self, "clear_surface_hovered", False)
                painter.setBrush(QtGui.QColor(220, 38, 38, 235) if is_hover_clear else QtGui.QColor(15, 23, 42, 225))
                painter.setPen(QtGui.QPen(QtGui.QColor("#f87171"), 1.5))
                painter.drawRoundedRect(s_rect, 5, 5)
                painter.setPen(QtGui.QColor("#ffffff") if is_hover_clear else QtGui.QColor("#fecaca"))
                painter.drawText(s_rect, QtCore.Qt.AlignmentFlag.AlignCenter, surf_txt)

                # If an ESP-mapped surface with a potential range is active, render a colorbar legend
                esp_mesh = next((m for m in active_surfaces if getattr(m, "esp_range", None) is not None), None)
                if esp_mesh is not None:
                    e_min, e_max = esp_mesh.esp_range
                    cbar_w = 170
                    cbar_h = 12
                    cbar_x = w - cbar_w - 20
                    cbar_y = y_surf + 48
                    box_rect = QtCore.QRect(cbar_x - 8, cbar_y - 16, cbar_w + 16, cbar_h + 36)
                    if is_white_vp:
                        painter.setBrush(QtGui.QColor(248, 250, 252, 240))
                        painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1"), 1.2))
                    else:
                        painter.setBrush(QtGui.QColor(15, 23, 42, 230))
                        painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1.2))
                    painter.drawRoundedRect(box_rect, 6, 6)

                    # Dynamic multi-stop gradient for realistic colormap preview
                    grad = QtGui.QLinearGradient(cbar_x, cbar_y, cbar_x + cbar_w, cbar_y)
                    cmap_raw = getattr(esp_mesh, "colormap_name", "bwr_r")
                    cmap_loaded = False
                    try:
                        import matplotlib as _mpl
                        _cmap = _mpl.colormaps.get(cmap_raw)
                        if _cmap is not None:
                            for i_s in range(9):
                                t_s = i_s / 8.0
                                r_c, g_c, b_c, _ = _cmap(t_s)
                                grad.setColorAt(t_s, QtGui.QColor(int(r_c * 255), int(g_c * 255), int(b_c * 255)))
                            cmap_loaded = True
                    except Exception:
                        cmap_loaded = False

                    if not cmap_loaded:
                        cmap_hint = str(cmap_raw).lower()
                        if "jet" in cmap_hint:
                            grad.setColorAt(0.00, QtGui.QColor("#ef4444"))
                            grad.setColorAt(0.25, QtGui.QColor("#f97316"))
                            grad.setColorAt(0.50, QtGui.QColor("#22c55e"))
                            grad.setColorAt(0.75, QtGui.QColor("#06b6d4"))
                            grad.setColorAt(1.00, QtGui.QColor("#2563eb"))
                        elif "spectral" in cmap_hint:
                            grad.setColorAt(0.00, QtGui.QColor("#d73027"))
                            grad.setColorAt(0.25, QtGui.QColor("#fee08b"))
                            grad.setColorAt(0.50, QtGui.QColor("#ffffbf"))
                            grad.setColorAt(0.75, QtGui.QColor("#91bfdb"))
                            grad.setColorAt(1.00, QtGui.QColor("#4575b4"))
                        else:  # bwr_r / RdBu / coolwarm_r standard
                            grad.setColorAt(0.00, QtGui.QColor("#dc2626"))
                            grad.setColorAt(0.48, QtGui.QColor("#f8fafc"))
                            grad.setColorAt(0.52, QtGui.QColor("#f8fafc"))
                            grad.setColorAt(1.00, QtGui.QColor("#2563eb"))

                    painter.setBrush(QtGui.QBrush(grad))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1.0))
                    painter.drawRect(cbar_x, cbar_y, cbar_w, cbar_h)

                    cbar_title = getattr(esp_mesh, "colorbar_title", "ESP Heatmap (a.u.)")
                    is_sym = (e_min < 0.0 < e_max) and ("log" not in cbar_title.lower())
                    painter.setFont(QtGui.QFont("DejaVu Sans", 7, QtGui.QFont.Weight.Bold))
                    painter.setPen(QtGui.QColor("#dc2626" if is_sym and is_white_vp else ("#f87171" if is_sym else ("#334155" if is_white_vp else "#cbd5e1"))))
                    painter.drawText(cbar_x - 2, cbar_y + cbar_h + 13, f"{e_min:+.2f}" if is_sym else f"{e_min:.2g}")
                    painter.setPen(QtGui.QColor("#475569" if is_white_vp else "#cbd5e1"))
                    mid_lbl = "0" if is_sym else f"{0.5 * (e_min + e_max):.2g}"
                    painter.drawText(cbar_x + cbar_w // 2 - 8, cbar_y + cbar_h + 13, mid_lbl)
                    painter.setPen(QtGui.QColor("#2563eb" if is_sym and is_white_vp else ("#60a5fa" if is_sym else ("#334155" if is_white_vp else "#cbd5e1"))))
                    e_max_str = f"{e_max:+.2f}" if is_sym else f"{e_max:.2g}"
                    painter.drawText(cbar_x + cbar_w - 42, cbar_y + cbar_h + 13, e_max_str)
                    painter.setPen(QtGui.QColor("#334155" if is_white_vp else "#94a3b8"))
                    painter.drawText(cbar_x, cbar_y - 4, cbar_title)
            else:
                self.clear_surface_rect = None

            # 4. Render 3D Translation & Manipulation Gizmo (X=Red, Y=Green, Z=Blue) on selection center
            # Also render circular Torsion / Dihedral ring gizmo when exactly a single bond or 2 bonded atoms are selected
            if (
                has_selection
                and len(render_coords) > 0
                and not self.atom_add_drag_active
            ):
                valid_sel = [i for i in self.multi_selected if 0 <= i < len(render_coords)]
                if not valid_sel and self.selected_atom_idx is not None and 0 <= self.selected_atom_idx < len(render_coords):
                    valid_sel = [self.selected_atom_idx]

                # 4a. 3D Torsion Ring Gizmo around selected bond axis
                selected_bond = getattr(self, "selected_bond", None)
                if not selected_bond and len(valid_sel) == 2:
                    u, v = valid_sel[0], valid_sel[1]
                    b_pair = (min(u, v), max(u, v))
                    if b_pair in getattr(self, "bonds", []):
                        selected_bond = b_pair

                if selected_bond:
                    u, v = selected_bond
                    if 0 <= u < len(render_coords) and 0 <= v < len(render_coords):
                        p_u = render_coords[u]
                        p_v = render_coords[v]
                        mid_w = (p_u + p_v) * 0.5
                        p4_mid = mvp * QtGui.QVector4D(float(mid_w[0]), float(mid_w[1]), float(mid_w[2]), 1.0)
                        if p4_mid.w() > 0.05:
                            ndc_mid = p4_mid / p4_mid.w()
                            mid_sx = int((ndc_mid.x() * 0.5 + 0.5) * w)
                            mid_sy = int((-ndc_mid.y() * 0.5 + 0.5) * h)

                            # Torsion ring radius on screen
                            ring_r = 28
                            is_torsion_active = getattr(self, "torsion_drag_active", False)
                            ring_col = QtGui.QColor("#38bdf8") if not is_torsion_active else QtGui.QColor("#fbbf24")

                            painter.save()
                            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
                            pen_ring = QtGui.QPen(ring_col, 2.5 if is_torsion_active else 1.8, QtCore.Qt.PenStyle.DashLine if not is_torsion_active else QtCore.Qt.PenStyle.SolidLine)
                            painter.setPen(pen_ring)
                            painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
                            painter.drawEllipse(QtCore.QPoint(mid_sx, mid_sy), ring_r, ring_r)

                            # Central torsion badge / angle indicator
                            painter.setBrush(QtGui.QColor(15, 23, 42, 210))
                            painter.setPen(QtGui.QPen(ring_col, 1.2))
                            badge_rect = QtCore.QRect(mid_sx - 28, mid_sy - 10, 56, 20)
                            painter.drawRoundedRect(badge_rect, 4, 4)
                            painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                            painter.setPen(QtGui.QColor("#f8fafc"))
                            painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "Alt+Drag ↻")
                            painter.restore()

                # 4b. Translation gizmo (when show_manipulator_gizmo is enabled)
                if getattr(self, "show_manipulator_gizmo", False) and valid_sel:
                    center_world = np.mean(render_coords[valid_sel], axis=0)
                    p4_c = mvp * QtGui.QVector4D(float(center_world[0]), float(center_world[1]), float(center_world[2]), 1.0)
                    if p4_c.w() > 0.05:
                        ndc_c = p4_c / p4_c.w()
                        c_sx = int((ndc_c.x() * 0.5 + 0.5) * w)
                        c_sy = int((-ndc_c.y() * 0.5 + 0.5) * h)

                        # Arm length in world space
                        arm_len_w = 1.6
                        axes_data = [
                            ("X", np.array([arm_len_w, 0.0, 0.0]), QtGui.QColor("#ef4444"), QtGui.QColor("#fca5a5")),
                            ("Y", np.array([0.0, arm_len_w, 0.0]), QtGui.QColor("#22c55e"), QtGui.QColor("#86efac")),
                            ("Z", np.array([0.0, 0.0, arm_len_w]), QtGui.QColor("#3b82f6"), QtGui.QColor("#93c5fd")),
                        ]
                        self.gizmo_screen_rects = {}
                        hover_ax = getattr(self, "gizmo_hover_axis", None)
                        active_ax = getattr(self, "active_gizmo_axis", None)

                        # Draw subtle central hub
                        painter.setBrush(QtGui.QColor(255, 255, 255, 180))
                        painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), 1.5))
                        painter.drawEllipse(QtCore.QPoint(c_sx, c_sy), 4, 4)

                        for ax_name, ax_vec, ax_col, ax_light in axes_data:
                            tip_w = center_world + ax_vec
                            p4_tip = mvp * QtGui.QVector4D(float(tip_w[0]), float(tip_w[1]), float(tip_w[2]), 1.0)
                            if p4_tip.w() <= 0.05:
                                continue
                            ndc_tip = p4_tip / p4_tip.w()
                            t_sx = int((ndc_tip.x() * 0.5 + 0.5) * w)
                            t_sy = int((-ndc_tip.y() * 0.5 + 0.5) * h)

                            is_highlight = (hover_ax == ax_name) or (active_ax == ax_name)
                            col = ax_light if is_highlight else ax_col
                            line_w = 3.5 if is_highlight else 2.2

                            # Shaft
                            painter.setPen(QtGui.QPen(col, line_w))
                            painter.drawLine(c_sx, c_sy, t_sx, t_sy)

                            # Handle handle / box at tip
                            box_size = 14 if is_highlight else 10
                            handle_rect = QtCore.QRect(t_sx - box_size // 2, t_sy - box_size // 2, box_size, box_size)
                            self.gizmo_screen_rects[ax_name] = handle_rect

                            painter.setBrush(col)
                            painter.setPen(QtGui.QPen(QtGui.QColor("#0f172a"), 1.2))
                            painter.drawRoundedRect(handle_rect, 3, 3)

                            # Axis label
                            painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                            painter.setPen(QtGui.QColor("#ffffff"))
                            painter.drawText(t_sx + 8, t_sy + 4, ax_name)

            # 5. Render Corner 3D Orientation Axis Triad (Avogadro 2 / PyMOL / VMD style)
            if getattr(self, "show_axis_triad", True):
                triad_cx = w - 48
                triad_cy = h - (58 if getattr(self, "show_scale_bar", False) else 42)
                triad_len = 26.0
                # Background disc
                painter.setBrush(QtGui.QColor(248, 250, 252, 200) if is_white_vp else QtGui.QColor(15, 23, 42, 185))
                painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1" if is_white_vp else "#334155"), 1.0))
                painter.drawEllipse(QtCore.QPoint(triad_cx, triad_cy), 34, 34)

                q_rot = self.rot_quat if (hasattr(self, "rot_quat") and not self.rot_quat.isNull()) else (
                    QtGui.QQuaternion.fromAxisAndAngle(1.0, 0.0, 0.0, self.rot_x)
                    * QtGui.QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, self.rot_y)
                )
                triad_axes = [
                    ("X", QtGui.QVector3D(1.0, 0.0, 0.0), QtGui.QColor("#ef4444")),
                    ("Y", QtGui.QVector3D(0.0, 1.0, 0.0), QtGui.QColor("#22c55e")),
                    ("Z", QtGui.QVector3D(0.0, 0.0, 1.0), QtGui.QColor("#3b82f6")),
                ]
                # Sort back-to-front by rotated z so front axes draw on top
                rotated_list = []
                for ax_lbl, v3, col in triad_axes:
                    rv = q_rot.rotatedVector(v3)
                    rotated_list.append((float(rv.z()), ax_lbl, float(rv.x()), float(rv.y()), col))
                rotated_list.sort(key=lambda item: item[0])

                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                for _rz, ax_lbl, rx_v, ry_v, col in rotated_list:
                    ex = int(round(triad_cx + rx_v * triad_len))
                    ey = int(round(triad_cy - ry_v * triad_len))
                    painter.setPen(QtGui.QPen(col, 2.3))
                    painter.drawLine(triad_cx, triad_cy, ex, ey)
                    painter.setBrush(col)
                    painter.setPen(QtCore.Qt.PenStyle.NoPen)
                    painter.drawEllipse(QtCore.QPoint(ex, ey), 3, 3)
                    lx = int(round(triad_cx + rx_v * (triad_len + 8.0))) - 4
                    ly = int(round(triad_cy - ry_v * (triad_len + 8.0))) + 4
                    painter.setPen(col)
                    painter.drawText(lx, ly, ax_lbl)

            # 6. Render Calibrated 3D Scale Bar Overlay (PyMOL / ChimeraX style)
            if getattr(self, "show_scale_bar", False):
                vis_ang_h = max(1.0, self.zoom * 0.41421356 * 2.0)
                px_per_ang = float(max(1, h)) / vis_ang_h
                bar_ang = 1.0
                if px_per_ang * 1.0 < 45.0:
                    bar_ang = 2.0
                if px_per_ang * 2.0 < 45.0:
                    bar_ang = 5.0
                elif px_per_ang * 1.0 > 180.0:
                    bar_ang = 0.5
                bar_px = int(round(bar_ang * px_per_ang))
                bx2 = w - 16
                bx1 = bx2 - bar_px
                by = h - 18
                sb_rect = QtCore.QRect(bx1 - 8, by - 18, bar_px + 16, 26)
                painter.setBrush(QtGui.QColor(248, 250, 252, 225) if is_white_vp else QtGui.QColor(15, 23, 42, 215))
                painter.setPen(QtGui.QPen(QtGui.QColor("#cbd5e1" if is_white_vp else "#475569"), 1.0))
                painter.drawRoundedRect(sb_rect, 4, 4)

                fg_col = QtGui.QColor("#0f172a" if is_white_vp else "#f8fafc")
                painter.setPen(QtGui.QPen(fg_col, 2.2))
                painter.drawLine(bx1, by, bx2, by)
                painter.drawLine(bx1, by - 4, bx1, by + 4)
                painter.drawLine(bx2, by - 4, bx2, by + 4)
                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                lbl_sb = f"{bar_ang:g} Å"
                sb_fm = painter.fontMetrics()
                sb_tw = sb_fm.horizontalAdvance(lbl_sb)
                painter.drawText(bx1 + (bar_px - sb_tw) // 2, by - 5, lbl_sb)

            # Live Molecular Formula & Mass HUD Pill (Top-Right Viewport)
            if len(self.symbols) > 0:
                from .elements import get_formula_and_weight
                formula, mw = get_formula_and_weight(self.symbols)
                pill_text = f"✨ {formula}  •  {mw:.2f} g/mol"

                painter.setFont(QtGui.QFont("DejaVu Sans", 8, QtGui.QFont.Weight.Bold))
                p_fm = painter.fontMetrics()
                p_tw = p_fm.horizontalAdvance(pill_text)
                p_th = p_fm.height()

                pill_pad_x = 10
                pill_pad_y = 5
                pill_w = p_tw + pill_pad_x * 2
                pill_h = p_th + pill_pad_y * 2
                pill_x = w - pill_w - 12
                pill_y = 12

                pill_rect = QtCore.QRect(pill_x, pill_y, pill_w, pill_h)
                is_white_vp = (self.bg_color_mode == "white" and not getattr(self, "transparent_bg", False))
                painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255, 215) if is_white_vp else QtGui.QColor(15, 23, 42, 210)))
                painter.setPen(QtGui.QPen(QtGui.QColor("#0284c7" if is_white_vp else "#38bdf8"), 1.2))
                painter.drawRoundedRect(pill_rect, 6, 6)

                painter.setPen(QtGui.QColor("#0f172a" if is_white_vp else "#f8fafc"))
                painter.drawText(pill_x + pill_pad_x, pill_y + pill_pad_y + p_fm.ascent(), pill_text)

            # Live Distance / Angle / Dihedral Measurement HUD Pill (Bottom-Left Viewport)
            hud_meas = getattr(self, "_hud_measurement_text", "")
            if hud_meas and len(getattr(self, "multi_selected", [])) in (2, 3, 4):
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                m_fm = painter.fontMetrics()
                m_tw = m_fm.horizontalAdvance(hud_meas)
                m_th = m_fm.height()
                m_rect = QtCore.QRect(12, max(12, h - m_th - 22), m_tw + 20, m_th + 10)
                is_white_vp = (self.bg_color_mode == "white" and not getattr(self, "transparent_bg", False))
                painter.setBrush(QtGui.QColor(255, 255, 255, 225) if is_white_vp else QtGui.QColor(15, 23, 42, 220))
                painter.setPen(QtGui.QPen(QtGui.QColor("#0284c7" if is_white_vp else "#38bdf8"), 1.3))
                painter.drawRoundedRect(m_rect, 6, 6)
                painter.setPen(QtGui.QColor("#0f172a" if is_white_vp else "#f8fafc"))
                painter.drawText(m_rect.x() + 10, m_rect.y() + 5 + m_fm.ascent(), hud_meas)

            if getattr(self, "_drag_hover", False):
                self._draw_drop_overlay(painter, w, h)

            painter.end()

    def _draw_drop_overlay(self, painter: QtGui.QPainter, w: int, h: int) -> None:
        """Draw a PyMOL/Chimera-style drop-target highlight when dragging files over the 3D viewport."""
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        pen = QtGui.QPen(QtGui.QColor("#38bdf8"), 2.5, QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtGui.QColor(56, 189, 248, 22))
        painter.drawRoundedRect(QtCore.QRect(8, 8, max(10, w - 16), max(10, h - 16)), 10, 10)

        msg = "📂 Drop molecular / cube / orbital file(s) to open"
        painter.setFont(QtGui.QFont("DejaVu Sans", 10, QtGui.QFont.Weight.Bold))
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(msg)
        th = fm.height()
        bw = tw + 28
        bh = th + 16
        bx = (w - bw) // 2
        by = max(18, h - bh - 24)
        painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 1.5))
        painter.setBrush(QtGui.QColor(15, 23, 42, 230))
        painter.drawRoundedRect(QtCore.QRect(bx, by, bw, bh), 8, 8)
        painter.setPen(QtGui.QColor("#f8fafc"))
        painter.drawText(bx + 14, by + 8 + fm.ascent(), msg)
        painter.restore()

    # Mouse & Keyboard interaction
