"""Modern OpenGL 3D Molecule Viewport using PyQt6 QtOpenGLWidgets and modern shaders."""

import ctypes
import math
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
    glDeleteShader, glDrawElements, glDrawElementsInstanced, glEnable, glEnableVertexAttribArray,
    glGenBuffers, glGenVertexArrays, glGetProgramInfoLog, glGetProgramiv,
    glGetShaderInfoLog, glGetShaderiv, glGetUniformLocation, glLinkProgram,
    glShaderSource, glUniform1i, glUniform1f, glUniform3f, glUniformMatrix3fv,
    glUniformMatrix4fv, glUseProgram, glVertexAttribPointer, glVertexAttribDivisor, glViewport,
    glDepthFunc, glDepthMask, glDisable, GL_LESS, GL_TRUE,
    GL_CULL_FACE, GL_SCISSOR_TEST, GL_STENCIL_TEST,
    glPolygonMode, GL_FRONT_AND_BACK, GL_LINE, GL_FILL,
    glCullFace, GL_FRONT, GL_BACK,
)

from .aromatic import find_delocalized_bonds
from .elements import get_color, get_cov_radius, get_vdw_radius
from .viewer_meshes import (
    create_sphere_mesh, create_cylinder_mesh, create_cone_mesh,
    VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC,
    SURFACE_VERTEX_SHADER, SURFACE_FRAGMENT_SHADER,
)
from .viewer_interaction import ViewerInteractionMixin


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
        self.sphere_scale: float = 0.35
        self.cylinder_radius: float = 0.08
        self.style_mode: str = "Ball & Stick" # Ball & Stick, Spacefill, Wireframe

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

        self.cyl_vao = 0
        self.cyl_vbo = 0
        self.cyl_ebo = 0
        self.cyl_indices_count = 0

        self.cone_vao = 0
        self.cone_vbo = 0
        self.cone_ebo = 0
        self.cone_indices_count = 0

        # Vibrational mode animation state
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

    def compute_bonds(self):
        """Auto detect covalent bonds based on pairwise covalent radii, strictly honoring user_bonds."""
        from .forcefield import detect_connectivity
        self.bonds = detect_connectivity(self.symbols, self.coords, user_bonds=self.user_bonds,
                                         strict=bool(getattr(self, 'strict_bonds', False)))

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

    def auto_center(self):
        """Reset camera pan and zoom to perfectly frame and center the molecule in the canvas."""
        self.fit_camera()
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

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_MULTISAMPLE)
        glEnable(GL_BLEND)
        glBlendFuncSeparate(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO, GL_ONE)
        glClearColor(0.12, 0.13, 0.16, 1.0) # Modern dark slate background

        # Compile shaders
        self.shader_program = self._build_shader(VERTEX_SHADER_SRC, FRAGMENT_SHADER_SRC)
        self.surface_program = self._build_shader(SURFACE_VERTEX_SHADER, SURFACE_FRAGMENT_SHADER)

        # Build sphere geometry
        s_verts, s_inds = create_sphere_mesh(stacks=48, sectors=64)
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
            return

        render_coords = self.get_render_coords()

        glUseProgram(self.shader_program)

        proj_mat = self._get_projection_matrix()
        view_mat = self._get_view_matrix()

        # Set uniforms
        loc_proj = glGetUniformLocation(self.shader_program, "u_proj")
        loc_view = glGetUniformLocation(self.shader_program, "u_view")
        loc_model = glGetUniformLocation(self.shader_program, "u_model")
        loc_norm = glGetUniformLocation(self.shader_program, "u_normal_mat")
        loc_color = glGetUniformLocation(self.shader_program, "u_color")
        loc_light = glGetUniformLocation(self.shader_program, "u_light_pos")
        loc_view_pos = glGetUniformLocation(self.shader_program, "u_view_pos")
        loc_hl = glGetUniformLocation(self.shader_program, "u_is_highlighted")
        loc_use_inst = glGetUniformLocation(self.shader_program, "u_use_instancing")

        c_x, c_y, c_z = 0.0, 0.0, 0.0
        if len(self.coords) > 0:
            c = np.mean(self.coords, axis=0)
            c_x, c_y, c_z = float(c[0]), float(c[1]), float(c[2])
        if hasattr(self, "rot_quat") and not self.rot_quat.isNull():
            inv_q = self.rot_quat.inverted()
            cw = inv_q.rotatedVector(QtGui.QVector3D(0.0, 0.0, float(self.zoom)))
            cam_world_pos = (cw.x() + c_x, cw.y() + c_y, cw.z() + c_z)
        else:
            cam_world_pos = (c_x, c_y, self.zoom + c_z)

        glUniformMatrix4fv(loc_proj, 1, GL_FALSE, proj_mat.data())
        glUniformMatrix4fv(loc_view, 1, GL_FALSE, view_mat.data())
        glUniform3f(loc_light, cam_world_pos[0] + 15.0, cam_world_pos[1] + 25.0, cam_world_pos[2] + 20.0)
        glUniform3f(loc_view_pos, cam_world_pos[0], cam_world_pos[1], cam_world_pos[2])
        glUniform1i(loc_use_inst, 0)

        # 1. Render Bonds (Cylinders)
        if self.style_mode != "Spacefill":
            glBindVertexArray(self.cyl_vao)
            # Reference vector perpendicular to view line of sight
            right_cam, up_cam = self._get_camera_axes()
            deloc = self._delocalized_bonds()
            for i1, i2 in self.bonds:
                p1 = render_coords[i1]
                p2 = render_coords[i2]
                v = p2 - p1
                length = np.linalg.norm(v)
                if length < 1e-4:
                    continue
                v_norm = v / length

                # Rotation to align z-axis (0, 0, 1) with v_norm
                z_axis = np.array([0.0, 0.0, 1.0])
                axis = np.cross(z_axis, v_norm)
                axis_len = np.linalg.norm(axis)
                angle_deg = 0.0
                if axis_len > 1e-6:
                    axis = axis / axis_len
                    angle_deg = math.degrees(math.acos(np.clip(np.dot(z_axis, v_norm), -1.0, 1.0)))
                elif np.dot(z_axis, v_norm) < 0:
                    axis = np.array([1.0, 0.0, 0.0])
                    angle_deg = 180.0
                else:
                    axis = np.array([0.0, 1.0, 0.0])
                    angle_deg = 0.0

                # Determine bond order: 1, 2, or 3
                b_key = (min(i1, i2), max(i1, i2))
                order = self.bond_orders.get(b_key, 1)

                # Determine cylinder offsets perpendicular to bond vector
                # Perpendicular direction in screen/camera plane
                perp = np.cross(v_norm, up_cam)
                if np.linalg.norm(perp) < 1e-3:
                    perp = np.cross(v_norm, right_cam)
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 1e-4:
                    perp = perp / perp_norm
                else:
                    perp = np.array([0.0, 1.0, 0.0])

                # Each offset is (displacement, length_scale)
                if b_key in deloc:
                    # Aromatic / resonance bond (order 1.5): solid line + shorter parallel line
                    inward = deloc[b_key]
                    side = inward if inward is not None else perp
                    # Rings: the second line sits inside the ring; otherwise beside the bond
                    offsets = [(np.zeros(3), 1.0), (side * 0.30, 0.6)]
                    cyl_rad = self.cylinder_radius * 0.8
                elif order == 1:
                    offsets = [(np.zeros(3), 1.0)]
                    cyl_rad = self.cylinder_radius
                elif order == 2:
                    shift = 0.22
                    offsets = [(perp * shift, 1.0), (-perp * shift, 1.0)]
                    cyl_rad = self.cylinder_radius * 0.85
                else:  # order >= 3
                    shift = 0.30
                    offsets = [(np.zeros(3), 1.0), (perp * shift, 1.0), (-perp * shift, 1.0)]
                    cyl_rad = self.cylinder_radius * 0.75

                # Apply representation style modifiers
                if self.style_mode == "Licorice / Tube":
                    cyl_rad *= 1.35
                elif self.style_mode == "Wireframe":
                    cyl_rad *= 0.35

                half_len = 0.5 * length
                raw_c1 = get_color(self.symbols[i1])
                raw_c2 = get_color(self.symbols[i2])
                # Ensure bond halves have crisp visibility against dark viewport backgrounds
                # (especially carbon bonds which are otherwise very dark)
                def _bond_contrast(c):
                    # If dark grey/black (like dark carbon), brighten to a clear metallic slate / medium-light grey
                    lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
                    if lum < 0.55 and max(c) - min(c) < 0.12:  # achromatic dark
                        return (max(c[0], 0.58), max(c[1], 0.58), max(c[2], 0.60))
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
                    half_len = 0.5 * length * len_scale

                    # First half (p1 -> mid)
                    m1 = QtGui.QMatrix4x4()
                    m1.translate(float(p1_off[0]), float(p1_off[1]), float(p1_off[2]))
                    if angle_deg != 0:
                        m1.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m1.scale(cyl_rad, cyl_rad, float(half_len))

                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m1.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m1.normalMatrix().data())
                    if is_bond_selected:
                        glUniform3f(loc_color, 1.0, 0.84, 0.0) # Vibrant gold
                        glUniform1i(loc_hl, 1)
                    else:
                        glUniform3f(loc_color, c1[0], c1[1], c1[2])
                        glUniform1i(loc_hl, 0)
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

                    # Second half (mid -> p2)
                    m2 = QtGui.QMatrix4x4()
                    m2.translate(float(mid_off[0]), float(mid_off[1]), float(mid_off[2]))
                    if angle_deg != 0:
                        m2.rotate(angle_deg, float(axis[0]), float(axis[1]), float(axis[2]))
                    m2.scale(cyl_rad, cyl_rad, float(half_len))

                    glUniformMatrix4fv(loc_model, 1, GL_FALSE, m2.data())
                    glUniformMatrix3fv(loc_norm, 1, GL_FALSE, m2.normalMatrix().data())
                    if is_bond_selected:
                        glUniform3f(loc_color, 1.0, 0.84, 0.0) # Vibrant gold
                        glUniform1i(loc_hl, 1)
                    else:
                        glUniform3f(loc_color, c2[0], c2[1], c2[2])
                        glUniform1i(loc_hl, 0)
                    glDrawElements(GL_TRIANGLES, self.cyl_indices_count, GL_UNSIGNED_INT, None)

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

        # 2. Render Atoms (Spheres) via GPU Instancing (single draw call for all spheres)
        charges_arr = self._charges_and_dipole()[0] if getattr(self, 'show_charges', False) else None
        clean_export = getattr(self, "export_clean_mode", False)
        n_atoms = len(self.symbols)

        if n_atoms > 0:
            # Build instance data array: 8 float32 per atom [pos.xyz, rad, col.rgb, hl]
            inst_data = np.empty((n_atoms, 8), dtype=np.float32)
            for i, (sym, pos) in enumerate(zip(self.symbols, render_coords)):
                cov_r = get_cov_radius(sym)
                if self.style_mode == "Spacefill":
                    rad = get_vdw_radius(sym) * 0.75
                elif self.style_mode == "Licorice / Tube":
                    rad = self.cylinder_radius * 1.05
                elif self.style_mode == "Wireframe":
                    rad = 0.05
                else:
                    rad = max(0.2, cov_r * self.sphere_scale)

                c = self._charge_to_rgb(charges_arr[i]) if (charges_arr is not None and i < len(charges_arr)) else get_color(sym)
                hl = 0
                if not clean_export:
                    if i in self.highlighted_feature_atoms:
                        role = self.highlighted_feature_atoms[i]
                        if role == "target":
                            hl = 4  # Vibrant Gold
                        elif role == "bond":
                            hl = 5  # Vibrant Cyan
                        elif role == "angle":
                            hl = 6  # Bright Purple/Magenta
                        elif role == "dihedral":
                            hl = 7  # Bright Orange
                    elif i in self.picked_references:
                        hl = 3
                    elif i == self.selected_atom_idx:
                        hl = 1
                    elif i in self.multi_selected:
                        hl = 2

                inst_data[i, 0:3] = pos
                inst_data[i, 3] = rad
                inst_data[i, 4:7] = c
                inst_data[i, 7] = float(hl)

            glBindVertexArray(self.sphere_vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.sphere_inst_vbo)
            glBufferData(GL_ARRAY_BUFFER, inst_data.nbytes, inst_data, GL_DYNAMIC_DRAW)
            glUniform1i(loc_use_inst, 1)

            glDrawElementsInstanced(GL_TRIANGLES, self.sphere_indices_count, GL_UNSIGNED_INT, None, n_atoms)
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
            if hasattr(self, "vibration_displacements") and self.vibration_displacements is not None:
                # Transition dipole derivative approx: sum(q_i * dr_i)
                charges_arr = self._charges_and_dipole()[0] if hasattr(self, "_charges_and_dipole") else None
                if charges_arr is not None and len(charges_arr) == len(render_coords):
                    q = charges_arr
                else:
                    q = np.array([1.0 if s != "H" else 0.5 for s in self.symbols])
                disp = self.vibration_displacements
                # Oscillating displacement factor
                cos_phase = math.cos(getattr(self, "vibration_phase", 0.0))
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

        glBindVertexArray(0)
        glUseProgram(0)

        # 3c. Render 3D Volumetric Isosurfaces (Molecular Orbitals / ESP / Density)
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
                        except Exception:
                            pass
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
        has_selection = (self.selected_atom_idx is not None) or bool(self.multi_selected) or (self.selected_bond is not None)
        surface_extrema = []
        for s in active_surfaces:
            if getattr(s, "extrema_points", None):
                surface_extrema.extend(s.extrema_points)

        if (
            has_selection
            or bool(active_surfaces)
            or self.show_labels
            or self.show_indices
            or self.show_charges
            or self.show_dipole
            or self.vibrating
            or self.partial_reorder_mode
            or self.reordering_active
            or self.picking_active
            or surface_extrema
        ):
            painter = QtGui.QPainter(self)
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            fm = painter.fontMetrics()

            mvp = proj_mat * view_mat
            w = self.width()
            h = self.height()

            # Render glowing high-contrast selection rings around all selected atoms
            if has_selection and len(render_coords) == len(self.symbols):
                for i in range(len(self.symbols)):
                    is_prim = (i == self.selected_atom_idx)
                    is_multi = (i in self.multi_selected)
                    if not (is_prim or is_multi):
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
                    else:
                        # Secondary: vibrant cyan ring
                        painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 55), 5))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 6), int(r_px + 6))
                        painter.setPen(QtGui.QPen(QtGui.QColor(186, 230, 253), 2.0))
                        painter.drawEllipse(QtCore.QPoint(sx, sy), int(r_px + 4), int(r_px + 4))

            if self.show_labels or self.show_indices or self.show_charges:
                for i, (sym, pos) in enumerate(zip(self.symbols, render_coords)):
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

                    # Role tag when feature box is clicked in Z-Matrix
                    if i in self.highlighted_feature_atoms:
                        role_name = self.highlighted_feature_atoms[i].capitalize()
                        label_text = f"[{role_name}] {label_text}"

                    # Text pill styling: high-contrast background and crisp border
                    bg_color = QtGui.QColor(15, 23, 42, 215) # Dark slate semi-transparent
                    border_color = QtGui.QColor("#38bdf8")
                    text_color = QtGui.QColor("#ffffff")

                    if self.partial_reorder_mode and i in self.partial_reorder_picks:
                        step_idx = self.partial_reorder_picks.index(i)
                        role = ["Target", "Bond", "Angle", "Dihedral"][step_idx] if step_idx < 4 else f"#{step_idx+1}"
                        label_text = f"[{role}] {sym}{i+1}"
                        text_color = QtGui.QColor("#fef08a") # Pale bright yellow
                        border_color = QtGui.QColor("#f59e0b")
                        bg_color = QtGui.QColor(69, 26, 3, 230)
                    elif i in self.highlighted_feature_atoms:
                        role = self.highlighted_feature_atoms[i]
                        if role == "target":
                            text_color = QtGui.QColor("#fef08a") # Gold/yellow
                            border_color = QtGui.QColor("#eab308")
                            bg_color = QtGui.QColor(50, 40, 5, 235)
                        elif role == "bond":
                            text_color = QtGui.QColor("#a5f3fc") # Cyan
                            border_color = QtGui.QColor("#06b6d4")
                            bg_color = QtGui.QColor(8, 47, 73, 235)
                        elif role == "angle":
                            text_color = QtGui.QColor("#f5d0fe") # Purple/magenta
                            border_color = QtGui.QColor("#c026d3")
                            bg_color = QtGui.QColor(59, 7, 100, 235)
                        elif role == "dihedral":
                            text_color = QtGui.QColor("#fed7aa") # Orange
                            border_color = QtGui.QColor("#f97316")
                            bg_color = QtGui.QColor(67, 20, 7, 235)
                    elif self.reordering_active and i in self.reorder_list:
                        text_color = QtGui.QColor("#86efac")
                        border_color = QtGui.QColor("#22c55e")
                    elif i in self.picked_references:
                        text_color = QtGui.QColor("#86efac")
                        border_color = QtGui.QColor("#22c55e")
                    elif i == self.selected_atom_idx:
                        text_color = QtGui.QColor("#fef08a")
                        border_color = QtGui.QColor("#fbbf24")
                    elif i in self.multi_selected:
                        text_color = QtGui.QColor("#a5f3fc")
                        border_color = QtGui.QColor("#38bdf8")
                    else:
                        text_color = QtGui.QColor("#f1f5f9")
                        border_color = QtGui.QColor(100, 116, 139, 180)

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

            # Render GaussView-style 3D floating measurement overlays (Distance, Angle, Dihedral)
            if self.show_measurements and len(self.multi_selected) >= 2 and len(render_coords) == len(self.symbols):
                k_sel = len(self.multi_selected)
                sel_valid = all(0 <= idx < len(render_coords) for idx in self.multi_selected)
                if sel_valid:
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
                        painter.setBrush(QtGui.QColor(15, 23, 42, bg_alpha))
                        painter.setPen(QtGui.QPen(QtGui.QColor(border_col_hex), 1.5))
                        painter.drawRoundedRect(b_rect, 5, 5)
                        painter.setPen(QtGui.QColor("#ffffff"))
                        painter.drawText(bx + 6, by + th - 2, text)

                    if k_sel == 2:
                        a, b = self.multi_selected[0], self.multi_selected[1]
                        pa, pb = render_coords[a], render_coords[b]
                        dist = float(np.linalg.norm(pa - pb))
                        s_a = _proj_pos(pa)
                        s_b = _proj_pos(pb)
                        if s_a and s_b:
                            painter.setPen(QtGui.QPen(QtGui.QColor(56, 189, 248, 200), 2.0, QtCore.Qt.PenStyle.DashLine))
                            painter.drawLine(s_a[0], s_a[1], s_b[0], s_b[1])
                            s_mid = _proj_pos((pa + pb) * 0.5)
                            if s_mid:
                                _draw_badge(s_mid[0], s_mid[1], f"{dist:.3f} Å", "#38bdf8")

                    elif k_sel == 3:
                        a, b, c = self.multi_selected[0], self.multi_selected[1], self.multi_selected[2]
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
                            _draw_badge(s_badge[0], s_badge[1], f"θ = {deg:.1f}°", "#c026d3")

                    elif k_sel >= 4:
                        a, b, c, d = self.multi_selected[:4]
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
                painter.setBrush(QtGui.QColor(15, 23, 42, 220))
                painter.setPen(QtGui.QPen(QtGui.QColor("#ec4899"), 1.2))
                painter.drawRoundedRect(d_rect, 5, 5)
                painter.setPen(QtGui.QColor("#fdf2f8"))
                painter.drawText(d_rect, QtCore.Qt.AlignmentFlag.AlignCenter, dip_txt)

            # Top-left HUD badge: Active Element Brush indicator
            brush_sym = getattr(self, "brush_element", "C") or "C"
            brush_text = f"Brush: {brush_sym} (Keys: C, H, O, N, S, P, F, Cl, Br, I)"
            painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
            b_fm = painter.fontMetrics()
            b_tw = b_fm.horizontalAdvance(brush_text)
            b_rect = QtCore.QRect(10, 8, b_tw + 18, 24)
            painter.setBrush(QtGui.QColor(15, 23, 42, 215)) # Dark slate semi-transparent
            painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 1))
            painter.drawRoundedRect(b_rect, 5, 5)
            # Dot in element color
            brush_col = get_color(brush_sym)
            painter.setBrush(QtGui.QColor(int(brush_col[0]*255), int(brush_col[1]*255), int(brush_col[2]*255)))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(18, 15, 10, 10)
            painter.setPen(QtGui.QColor("#f1f5f9"))
            painter.drawText(34, 24, brush_text)

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

            # Top-left HUD badge: Publication Molecule Watermark / Figure Overlay
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
                wrect = QtCore.QRect(12, 10, wtw + 14, 22)
                painter.setBrush(QtGui.QColor(15, 23, 42, 190))
                painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1.0))
                painter.drawRoundedRect(wrect, 4, 4)
                painter.setPen(QtGui.QColor("#e2e8f0"))
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
                painter.setBrush(QtGui.QColor(15, 23, 42, 230))
                painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1.5))
                painter.drawRoundedRect(badge_rect, 5, 5)
                painter.setPen(QtGui.QColor("#fef08a"))
                painter.drawText(badge_rect, QtCore.Qt.AlignmentFlag.AlignCenter, sel_str)

            # Top-right HUD badge: Clear 3D Orbital / Surface Pill (when active_isosurfaces is non-empty)
            if active_surfaces:
                surf_txt = "✖ Clear 3D Surface (Esc)"
                painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                s_fm = painter.fontMetrics()
                s_tw = s_fm.horizontalAdvance(surf_txt)
                y_surf = 38 if ((self.show_dipole or self.show_charges) and len(self.symbols) > 0) else 8
                s_rect = QtCore.QRect(w - s_tw - 28, y_surf, s_tw + 18, 26)
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
                    cbar_w = 140
                    cbar_h = 10
                    cbar_x = w - cbar_w - 28
                    cbar_y = y_surf + 32
                    box_rect = QtCore.QRect(cbar_x - 6, cbar_y - 14, cbar_w + 12, cbar_h + 30)
                    painter.setBrush(QtGui.QColor(15, 23, 42, 225))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#475569"), 1.0))
                    painter.drawRoundedRect(box_rect, 5, 5)

                    # Draw gradient ramp
                    grad = QtGui.QLinearGradient(cbar_x, cbar_y, cbar_x + cbar_w, cbar_y)
                    # Sample gradient matching the colormap or standard Red -> White -> Blue
                    grad.setColorAt(0.0, QtGui.QColor("#ef4444"))   # Negative (Red)
                    grad.setColorAt(0.5, QtGui.QColor("#f8fafc"))   # Neutral 0.0 (White)
                    grad.setColorAt(1.0, QtGui.QColor("#3b82f6"))   # Positive (Blue)
                    painter.setBrush(QtGui.QBrush(grad))
                    painter.setPen(QtGui.QPen(QtGui.QColor("#64748b"), 1.0))
                    painter.drawRect(cbar_x, cbar_y, cbar_w, cbar_h)

                    # Labels: -V_lim, 0, +V_lim
                    painter.setFont(QtGui.QFont("DejaVu Sans", 7, QtGui.QFont.Weight.Bold))
                    painter.setPen(QtGui.QColor("#f87171"))
                    painter.drawText(cbar_x, cbar_y + cbar_h + 12, f"{e_min:+.3f}")
                    painter.setPen(QtGui.QColor("#cbd5e1"))
                    painter.drawText(cbar_x + cbar_w // 2 - 4, cbar_y + cbar_h + 12, "0")
                    painter.setPen(QtGui.QColor("#60a5fa"))
                    painter.drawText(cbar_x + cbar_w - 38, cbar_y + cbar_h + 12, f"{e_max:+.3f}")
                    painter.setPen(QtGui.QColor("#94a3b8"))
                    painter.drawText(cbar_x, cbar_y - 3, "ESP Heatmap (a.u.)")
            else:
                self.clear_surface_rect = None

            # 4. Interactive 3D Measurement Overlay (HUD Distance line / Angle arc directly in 3D viewport)
            if has_selection and 2 <= len(self.multi_selected) <= 4 and len(render_coords) > 0:
                n_rc = len(render_coords)
                sel_sub = [i for i in self.multi_selected if 0 <= i < n_rc]
                if len(sel_sub) == 2:
                    i1, i2 = sel_sub
                    p4_1 = mvp * QtGui.QVector4D(float(render_coords[i1][0]), float(render_coords[i1][1]), float(render_coords[i1][2]), 1.0)
                    p4_2 = mvp * QtGui.QVector4D(float(render_coords[i2][0]), float(render_coords[i2][1]), float(render_coords[i2][2]), 1.0)
                    if p4_1.w() > 0.05 and p4_2.w() > 0.05:
                        ndc1 = p4_1 / p4_1.w()
                        ndc2 = p4_2 / p4_2.w()
                        sx1, sy1 = int((ndc1.x() * 0.5 + 0.5) * w), int((-ndc1.y() * 0.5 + 0.5) * h)
                        sx2, sy2 = int((ndc2.x() * 0.5 + 0.5) * w), int((-ndc2.y() * 0.5 + 0.5) * h)
                        # Draw measurement line
                        painter.setPen(QtGui.QPen(QtGui.QColor("#f59e0b"), 2, QtCore.Qt.PenStyle.DashLine))
                        painter.drawLine(sx1, sy1, sx2, sy2)
                        # Midpoint distance label
                        mx, my = (sx1 + sx2) // 2, (sy1 + sy2) // 2
                        dist = float(np.linalg.norm(render_coords[i1] - render_coords[i2]))
                        d_str = f"{dist:.3f} Å"
                        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                        dfm = painter.fontMetrics()
                        dtw = dfm.horizontalAdvance(d_str)
                        drect = QtCore.QRect(mx - dtw // 2 - 6, my - 12, dtw + 12, 22)
                        painter.setBrush(QtGui.QColor(15, 23, 42, 220))
                        painter.setPen(QtGui.QPen(QtGui.QColor("#fbbf24"), 1.2))
                        painter.drawRoundedRect(drect, 4, 4)
                        painter.setPen(QtGui.QColor("#fef08a"))
                        painter.drawText(drect, QtCore.Qt.AlignmentFlag.AlignCenter, d_str)
                elif len(sel_sub) == 3:
                    i1, i2, i3 = sel_sub
                    p4_2 = mvp * QtGui.QVector4D(float(render_coords[i2][0]), float(render_coords[i2][1]), float(render_coords[i2][2]), 1.0)
                    if p4_2.w() > 0.05:
                        ndc2 = p4_2 / p4_2.w()
                        sx2, sy2 = int((ndc2.x() * 0.5 + 0.5) * w), int((-ndc2.y() * 0.5 + 0.5) * h)
                        v1 = render_coords[i1] - render_coords[i2]
                        v2 = render_coords[i3] - render_coords[i2]
                        cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
                        ang_deg = math.degrees(math.acos(np.clip(cos_th, -1.0, 1.0)))
                        ang_str = f"∠ {ang_deg:.1f}°"
                        painter.setFont(QtGui.QFont("DejaVu Sans", 9, QtGui.QFont.Weight.Bold))
                        dfm = painter.fontMetrics()
                        dtw = dfm.horizontalAdvance(ang_str)
                        drect = QtCore.QRect(sx2 - dtw // 2 - 6, sy2 - 32, dtw + 12, 22)
                        painter.setBrush(QtGui.QColor(15, 23, 42, 220))
                        painter.setPen(QtGui.QPen(QtGui.QColor("#38bdf8"), 1.2))
                        painter.drawRoundedRect(drect, 4, 4)
                        painter.setPen(QtGui.QColor("#e0f2fe"))
                        painter.drawText(drect, QtCore.Qt.AlignmentFlag.AlignCenter, ang_str)

            # 5. Render 3D Translation & Manipulation Gizmo (X=Red, Y=Green, Z=Blue) on selection center
            if (
                getattr(self, "show_manipulator_gizmo", False)
                and has_selection
                and len(render_coords) > 0
                and not self.atom_add_drag_active
            ):
                valid_sel = [i for i in self.multi_selected if 0 <= i < len(render_coords)]
                if not valid_sel and self.selected_atom_idx is not None and 0 <= self.selected_atom_idx < len(render_coords):
                    valid_sel = [self.selected_atom_idx]
                if valid_sel:
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

            painter.end()

    # Mouse & Keyboard interaction
