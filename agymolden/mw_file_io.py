"""Mixin modules extracted from MainWindow for clean separation of concerns."""

import os
import math
import copy
import logging
from typing import List, Optional, Dict, Tuple
import numpy as np

from PyQt6 import QtCore, QtGui, QtWidgets

from .workers import run_in_background
from .elements import ELEMENT_DATA, normalize_symbol, estimate_bond_length
from .zmatrix import (
    ZAtom,
    zmatrix_to_cartesian,
    cartesian_to_zmatrix,
    parse_zmatrix_text,
    format_zmatrix_text,
    parse_xyz,
    write_xyz,
    parse_pdb,
    write_pdb,
    parse_mol_sdf,
    write_mol_sdf,
    parse_cif,
    calculate_bond_distance,
    calculate_angle,
    calculate_dihedral,
    format_zmatrix_numeric,
    format_zmatrix_variables,
    apply_variables,
    sync_variables_from_atoms,
    assign_symmetric_variables,
    clear_variables,
    _split_var,
)
from .fragments import FRAGMENTS, SAMPLE_MOLECULES
from .viewer import MoleculeViewer
from .table_view import ZMatrixTableWidget
from .periodic_table import PeriodicTableDialog
from .topology import (
    norm_bond, build_adj, remove_atoms, fix_valence_explicit, bfs_reorder,
    rotation_between, fragment_outward_axis, form_single_bond, BondFormationError,
)
from .forcefield import (
    DEFAULT_VALENCES,
    add_hydrogens_to_valence,
    adjust_hydrogens_to_valence,
    optimize_geometry_force_field,
    detect_connectivity,
    detect_bond_orders,
    get_cov_radius,
    _calculate_hydrogen_directions,
)
from .vibrations import VibrationalAnalysis, compute_force_field_vibrations
from .symmetry import (
    find_symmetry_operations,
    PointGroupResult,
    align_molecule_to_symmetry,
    enforce_point_group_symmetry,
    calculate_symmetry_representations,
)
from .dialog_qm_input import QMCalculationDialog
from .brorsen_builder import BrorsenInputBuilderDialog
from .qm_log_parser import parse_qm_logfile, can_parse_with_cclib
from .dialog_service import dialog_service

log = logging.getLogger(__name__)


class FileIOMixin:
    def _settings(self) -> QtCore.QSettings:
        return QtCore.QSettings("AgyMolden", "AgyMolden")

    def _last_dir(self) -> str:
        val = self._settings().value("last_dir", "")
        if val and os.path.isdir(str(val)) and not str(val).startswith("/tmp"):
            return str(val)
        return os.path.abspath(".")

    def _remember_file(self, path: str):
        abs_path = os.path.abspath(path)
        self.current_file_path = abs_path
        s = self._settings()
        s.setValue("last_dir", os.path.dirname(abs_path))
        recent = s.value("recent_files", [])
        if not isinstance(recent, list):
            recent = []
        recent = [abs_path] + [p for p in recent if p != abs_path]
        s.setValue("recent_files", recent[:10])
        self._update_recent_files_menu()

    def _update_recent_files_menu(self):
        menu = getattr(self, "menu_recent", None)
        if menu is None:
            return
        menu.clear()
        recent = self._settings().value("recent_files", [])
        if not isinstance(recent, list):
            recent = []
        existing = [p for p in recent if isinstance(p, str) and os.path.isfile(p)]
        if not existing:
            act = menu.addAction("(No recent files)")
            act.setEnabled(False)
            return
        for p in existing[:10]:
            act = menu.addAction(os.path.basename(p))
            act.setToolTip(p)
            act.triggered.connect(lambda checked=False, path=p: self.open_path(path))

    def _on_new_molecule(self):
        if self.zatoms:
            self._push_undo_state()
        self.variables = {}
        self.cross_bonds = set()
        self.strict_topology = False
        self.set_molecule_from_zatoms([])
        self.statusBar().showMessage("Cleared molecule (empty workspace, ready for Add Line)", 3000)

    def _on_open_universal_file(self):
        """Universal file opener supporting QM logs, XYZ, Molden, ORCA (.molden.input, .gbw), Cube, and Z-Matrix files sorted newest-first."""
        dlg = QtWidgets.QFileDialog(
            self,
            "Open File (All Supported Molecular & QM Formats)",
            self._last_dir(),
            "All Supported Files (*.out *.log *.xyz *.pdb *.sdf *.mol *.sd *.cif *.molden.input *.molden *.molf *.gbw *.fchk *.fch *.cube *.gjf *.com *.lowdin *.zmat *.dat *.inp *.in *.py *.txt *.smi *.smiles *.inchi);;"
            "QM Output Logs & Checkpoints (*.out *.log *.output *.fchk *.fch);;"
            "Molden & ORCA Wavefunctions (*.molden.input *.molden *.molf *.gbw);;"
            "Molecular Structures & Crystals (*.xyz *.pdb *.sdf *.mol *.sd *.cif);;"
            "QM Input Decks (*.inp *.in *.gjf *.com *.lowdin *.py);;"
            "Gaussian Volumetric Grids (*.cube);;"
            "Z-Matrix & Input Files (*.zmat *.dat *.inp *.txt);;"
            "All Files (*)",
        )
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog, True)
        for proxy in dlg.findChildren(QtCore.QSortFilterProxyModel):
            proxy.sort(3, QtCore.Qt.SortOrder.DescendingOrder)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            selected = dlg.selectedFiles()
            if selected:
                self.open_path(selected[0])

    def _on_open_zmat(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Z-Matrix File", self._last_dir(), "Z-Matrix Files (*.zmat *.dat *.inp *.txt);;All Files (*)"
        )
        if path:
            self._load_zmat_path(path)

    def _load_zmat_path(self, path: str):
        try:
            with open(path, "r") as f:
                content = f.read()
            zatoms, vars_dict = parse_zmatrix_text(content)
            self._push_undo_state()
            self.cross_bonds = set()
            self.strict_topology = False
            self.variables = vars_dict
            self.set_molecule_from_zatoms(zatoms)
            self._remember_file(path)
            self.statusBar().showMessage(f"Loaded {os.path.basename(path)}", 4000)
        except (OSError, IOError, ValueError) as e:
            dialog_service.error(self, "Error Opening File", str(e), exc=e)

    def open_path(self, path: str):
        """Open a molecule file, choosing the loader from its extension."""
        low = path.lower()
        if low.endswith(".xyz"):
            self._load_xyz_path(path)
        elif low.endswith(".pdb"):
            self._load_pdb_path(path)
        elif low.endswith((".sdf", ".mol", ".sd")):
            self._load_sdf_path(path)
        elif low.endswith(".cif"):
            self._load_cif_path(path)
        elif low.endswith((".log", ".out", ".fchk", ".fch")):
            self._load_qm_log_path(path)
        elif low.endswith((".smi", ".smiles")):
            self._load_smiles_path(path)
        elif low.endswith(".inchi"):
            self._load_inchi_path(path)
        elif low.endswith(".cube"):
            self._load_cube_path(path)
        elif low.endswith((".molden", ".molden.input", ".molf", ".gbw")):
            self._load_molden_path(path)
        elif low.endswith((".gjf", ".com")):
            self._load_gaussian_input_path(path)
        elif low.endswith(".lowdin"):
            self._load_openlowdin_input_path(path)
        elif low.endswith(".py"):
            self._load_pyscf_script_path(path)
        elif low.endswith((".inp", ".in")):
            self._load_brorsen_inp_path(path)
        else:
            self._load_zmat_path(path)

    def _load_smiles_path(self, path: str):
        try:
            with open(path, "r") as f:
                content = f.read().strip().split()[0]
            self.load_from_smiles_or_inchi(content, name=os.path.basename(path))
            self._remember_file(path)
        except (OSError, IOError, ValueError) as e:
            dialog_service.error(self, "Error Opening SMILES", str(e), exc=e)

    def _load_inchi_path(self, path: str):
        try:
            with open(path, "r") as f:
                content = f.read().strip()
            self.load_from_smiles_or_inchi(content, name=os.path.basename(path))
            self._remember_file(path)
        except (OSError, IOError, ValueError) as e:
            dialog_service.error(self, "Error Opening InChI", str(e), exc=e)

    def load_from_smiles_or_inchi(self, text: str, name: str = "Structure"):
        """Generate 3D Cartesian + Z-matrix coordinates from a SMILES or InChI string."""
        from .chem_identifiers import parse_smiles_or_inchi_to_3d
        symbols, coords, bonds, bo = parse_smiles_or_inchi_to_3d(text)
        self._push_undo_state()
        self.variables = {}
        zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
        self.cross_bonds = {norm_bond(u, v) for u, v in bonds if norm_bond(u, v) not in tree}
        self.strict_topology = True
        self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
        self.viewer.bond_orders = bo
        self.viewer.update()
        self.statusBar().showMessage(f"Generated 3D structure for {name} ({len(symbols)} atoms).", 5000)

    def _on_prompt_smiles_or_inchi(self):
        """Prompt user for a chemical name, SMILES, or InChI and generate 3D structure."""
        text, ok = QtWidgets.QInputDialog.getText(
            self,
            "Build 3D Molecule from Chemical Name / SMILES / InChI",
            "Enter Chemical Name (e.g. ferrocene, aspirin, caffeine, bipyridine),\n"
            "SMILES (e.g. c1ccccc1, CC(=O)O), or InChI (Ctrl+I):",
            QtWidgets.QLineEdit.EchoMode.Normal,
            "",
        )
        if ok and text.strip():
            try:
                self.load_from_smiles_or_inchi(text.strip(), name=text.strip()[:30])
            except Exception as e:
                dialog_service.error(self, "Chemical Name / Identifier Error", str(e), exc=e)

    def _on_open_esp_2d_dialog(self):
        """Open 2D Electrostatic Potential (ESP) Contour Slicer Dialog."""
        from .dialog_esp_2d import ESP2DContourDialog

        molden_path = getattr(self, "_active_molden_path", None)
        if not molden_path:
            last_res = getattr(self, "_last_calculation_results", {})
            if last_res.get("molden_file") and os.path.isfile(last_res["molden_file"]):
                molden_path = last_res["molden_file"]
                self._active_molden_path = molden_path
        if not molden_path and hasattr(self, "_volumetric_dlg") and self._volumetric_dlg:
            molden_path = (
                getattr(self._volumetric_dlg, "current_molden_path", None)
                or getattr(self._volumetric_dlg, "_active_molden_path", None)
            )
        if (
            not molden_path
            and hasattr(self, "current_file_path")
            and self.current_file_path
            and self.current_file_path.endswith((".molden", ".molden.input"))
        ):
            molden_path = self.current_file_path

        if not molden_path or not os.path.isfile(molden_path):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select Molden Wavefunction File for 2D ESP",
                self._last_dir(),
                "Molden Files (*.molden *.molden.input);;All Files (*)",
            )
            if not path:
                return
            molden_path = path

        try:
            dlg = ESP2DContourDialog(self, molden_path)
            dlg.exec()
        except Exception as e:
            log.exception("Error launching 2D ESP Contour Dialog")
            QtWidgets.QMessageBox.critical(self, "2D ESP Error", f"Failed to compute 2D ESP slice:\n{e}")

    def _on_open_multiwfn_dialog(self):
        """Open the Multiwfn Electronic Structure & Wavefunction Analysis Dialog."""
        from .dialog_multiwfn import MultiwfnAnalysisDialog
        molden_path = getattr(self, "_active_molden_path", None)
        if not molden_path:
            last_res = getattr(self, "_last_calculation_results", {})
            if last_res.get("molden_file") and os.path.isfile(last_res["molden_file"]):
                molden_path = last_res["molden_file"]
                self._active_molden_path = molden_path
        if not molden_path and hasattr(self, "_volumetric_dlg") and self._volumetric_dlg:
            molden_path = self._volumetric_dlg.current_molden_path
        if not molden_path:
            # Check last loaded qm log for matching molden
            last_log = getattr(self, "_last_loaded_qm_log_path", None)
            if last_log and os.path.isfile(last_log):
                stem = os.path.splitext(last_log)[0]
                for ext in (".molden.input", ".molden", "_electrons.molden"):
                    if os.path.isfile(stem + ext):
                        molden_path = stem + ext
                        break

        dlg = MultiwfnAnalysisDialog(self, molden_path=molden_path, viewer=self.viewer)
        dlg.exec()

    def _on_open_volumetric_dialog(self):
        """Open Volumetric Grid & Isosurface Controller Dialog."""
        from .dialog_volumetric import VolumetricSurfaceDialog
        dlg = getattr(self, "_volumetric_dlg", None)
        if dlg is None:
            dlg = VolumetricSurfaceDialog(self, viewer=self.viewer)
            self._volumetric_dlg = dlg

            def on_surface_updated(meshes, opacity):
                self.viewer.active_isosurfaces = meshes
                self.viewer.surface_opacity = opacity
                self.viewer.update()

            def on_load_mol(syms, crds):
                self._push_undo_state()
                bonds = detect_connectivity(syms, crds)
                bo = detect_bond_orders(syms, crds, bonds)
                zatoms = cartesian_to_zmatrix(syms, crds, user_bonds=bonds, strict_bonds=True)
                self.set_molecule_from_zatoms(zatoms, exact_coords=crds)
                self.viewer.bond_orders = bo
                self.viewer.update()

            dlg.surface_updated.connect(on_surface_updated)
            dlg.load_molecule_requested.connect(on_load_mol)

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_open_preferences(self):
        """Open user Preferences configuration dialog."""
        from .dialog_preferences import PreferencesDialog
        dlg = PreferencesDialog(self)
        dlg.exec()

    def _load_cube_path(self, path: str):
        if hasattr(self, "set_interaction_mode"):
            self.set_interaction_mode("inspect")
        self._on_open_volumetric_dialog()
        if hasattr(self, "_volumetric_dlg"):
            self._volumetric_dlg.load_cube_path(path)
            self._remember_file(path)

    def _load_molden_path(self, path: str):
        if hasattr(self, "set_interaction_mode"):
            self.set_interaction_mode("inspect")
        self._on_open_volumetric_dialog()
        if hasattr(self, "_volumetric_dlg"):
            self._volumetric_dlg.load_molden_path(path)
            self._remember_file(path)
            actual_path = self._volumetric_dlg.current_molden_path or path
            if os.path.isfile(actual_path):
                try:
                    self.load_orbitals_from_molden(actual_path, auto_render=False)
                except Exception:
                    log.debug("Ignored exception loading orbitals into MO tab", exc_info=True)
                try:
                    from .volumetric import parse_molden_file
                    m_data = parse_molden_file(actual_path)
                    if m_data:
                        if getattr(m_data, "vibrations", None) is not None and m_data.vibrations.n_modes > 0:
                            self._apply_vibrations((m_data.coords, m_data.vibrations))
                        orbitals_list = getattr(m_data, "orbitals", None) or getattr(m_data, "mos", None)
                        if hasattr(self, "_update_qm_summary_banner") and orbitals_list:
                            occ_mos = [mo for mo in orbitals_list if getattr(mo, "occupancy", getattr(mo, "occup", 0.0)) > 0.1]
                            virt_mos = [mo for mo in orbitals_list if getattr(mo, "occupancy", getattr(mo, "occup", 0.0)) <= 0.1]
                            homo_ev = occ_mos[-1].energy * 27.211386245988 if occ_mos else None
                            lumo_ev = virt_mos[0].energy * 27.211386245988 if virt_mos else None
                            from .qm_log_parser import QMLogParseResult
                            m_res = QMLogParseResult(
                                program_name="Molden",
                                symbols=m_data.symbols,
                                trajectory_coords=[m_data.coords],
                                trajectory_energies=[],
                                vibrational_analysis=getattr(m_data, "vibrations", None),
                                homo_ev=homo_ev,
                                lumo_ev=lumo_ev,
                            )
                            self._update_qm_summary_banner(m_res)
                except Exception:
                    log.debug("Ignored exception loading vibrations from Molden", exc_info=True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.open_path(url.toLocalFile())
                break

    def _on_open_qm_dialog(self, auto_enable_scan: bool = False, target_code: Optional[str] = None):
        """Open the multi-package QM calculation setup dialog (Gaussian, ORCA, PySCF, openLOWDIN, xTB)."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self, "QM Calculation Setup", "Please build or load a molecule before setting up a QM calculation."
            )
            return

        symbols = [a.symbol for a in self.zatoms]
        frozen = sorted(list(getattr(self, "frozen_atoms", set())))
        sel = [i for i in getattr(self.viewer, "multi_selected", []) if 0 <= i < len(symbols)]
        dlg = QMCalculationDialog(
            self,
            symbols=symbols,
            coords=self.coords,
            zatoms=self.zatoms,
            variables=self.variables,
            frozen_atoms=frozen,
            selected_atoms=sel,
            auto_enable_scan=bool(auto_enable_scan),
        )
        if target_code:
            for i in range(dlg.combo_qm_code.count()):
                if target_code.lower() in dlg.combo_qm_code.itemText(i).lower():
                    dlg.combo_qm_code.setCurrentIndex(i)
                    break
        dlg.exec()
        return dlg

    def _on_open_brorsen_dialog(self):
        """Open the newBRORSEN Input Builder dialog (Positrons, NEO, and CCSD(T))."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self, "newBRORSEN Input Builder", "Please build or load a molecule before setting up a newBRORSEN calculation."
            )
            return

        symbols = [a.symbol for a in self.zatoms]
        dlg = BrorsenInputBuilderDialog(
            self,
            symbols=symbols,
            coords=self.coords,
            zatoms=self.zatoms,
            variables=self.variables,
        )
        dlg.exec()

    def _apply_imported_cartesian_molecule(self, syms: List[str], crds: np.ndarray, path: str, label: str = "Input"):
        """Helper to commit imported Cartesian coordinates into the active viewer and Z-matrix."""
        if not syms or len(crds) != len(syms):
            return
        self._push_undo_state()
        self.variables = {}
        self.cross_bonds = set()
        self.strict_topology = False
        bonds = detect_connectivity(syms, crds)
        bo = detect_bond_orders(syms, crds, bonds)
        zatoms = cartesian_to_zmatrix(syms, crds, user_bonds=bonds, strict_bonds=True)
        self.set_molecule_from_zatoms(zatoms, exact_coords=crds)
        self.viewer.bond_orders = bo
        self.viewer.update()
        self._remember_file(path)
        self.statusBar().showMessage(f"Loaded {label} geometry ({len(syms)} atoms) from {os.path.basename(path)}", 4500)

    def _load_gaussian_input_path(self, path: str):
        """Parse a Gaussian (.gjf / .com) input deck and load its geometry (Cartesian or Z-matrix)."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # Gaussian deck structure:
            # [%link0 lines]
            # # route section (may span lines until blank line)
            # <blank line>
            # Title section
            # <blank line>
            # Charge Multiplicity
            # Coordinates block (Cartesian or Z-matrix)
            lines = content.splitlines()
            idx = 0
            # Skip Link 0 (%) and comments
            while idx < len(lines) and (not lines[idx].strip() or lines[idx].strip().startswith("%")):
                idx += 1
            # Skip Route (#) block until blank line
            if idx < len(lines) and lines[idx].strip().startswith("#"):
                while idx < len(lines) and lines[idx].strip():
                    idx += 1
            # Skip blank lines before Title
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            # Skip Title block until blank line
            while idx < len(lines) and lines[idx].strip():
                idx += 1
            # Skip blank lines before Charge/Multiplicity
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            # Skip Charge & Multiplicity line (e.g. "0 1")
            if idx < len(lines):
                cm_toks = lines[idx].split()
                if len(cm_toks) >= 2 and all(t.lstrip("+-").isdigit() for t in cm_toks[:2]):
                    idx += 1

            geom_lines = []
            while idx < len(lines):
                s = lines[idx].strip()
                if not s:
                    # Check if followed by Variables: block for Z-matrix
                    rem = "\n".join(lines[idx + 1 :])
                    if "variables:" in rem.lower() or any("=" in l for l in lines[idx + 1 : idx + 15]):
                        geom_lines.append("")
                        geom_lines.extend(lines[idx + 1 :])
                    break
                geom_lines.append(s)
                idx += 1

            # Determine if Cartesian (4 tokens per line: Symbol X Y Z) or Z-matrix
            first_geom = [l for l in geom_lines if l.strip()]
            is_cart = False
            if first_geom:
                p0 = first_geom[0].split()
                if len(p0) == 4:
                    try:
                        float(p0[1]); float(p0[2]); float(p0[3])
                        is_cart = True
                    except ValueError:
                        is_cart = False

            if is_cart:
                syms, crds = [], []
                for l in first_geom:
                    p = l.split()
                    if len(p) >= 4 and normalize_symbol(p[0]) in ELEMENT_DATA:
                        try:
                            syms.append(normalize_symbol(p[0]))
                            crds.append([float(p[1]), float(p[2]), float(p[3])])
                        except ValueError:
                            break
                if syms:
                    self._apply_imported_cartesian_molecule(syms, np.array(crds, dtype=np.float64), path, label="Gaussian")
                    return

            # Fallback to Z-Matrix parser
            zatoms, vars_dict = parse_zmatrix_text("\n".join(geom_lines))
            if zatoms:
                self._push_undo_state()
                self.cross_bonds = set()
                self.strict_topology = False
                self.variables = vars_dict
                self.set_molecule_from_zatoms(zatoms)
                self._remember_file(path)
                self.statusBar().showMessage(f"Loaded Gaussian Z-Matrix ({len(zatoms)} atoms) from {os.path.basename(path)}", 4500)
        except Exception as e:
            dialog_service.error(self, "Error Opening Gaussian Input", str(e), exc=e)

    def _load_openlowdin_input_path(self, path: str):
        """Parse an openLOWDIN (.lowdin) input file and load its molecular geometry."""
        try:
            import re
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            dirac_matches = re.findall(
                r'^\s*([A-Z][a-z]?)\s+dirac\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
                text,
                re.MULTILINE,
            )
            if not dirac_matches:
                dirac_matches = re.findall(
                    r'^\s*e-\(([A-Za-z]+)\)\s+\S+\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)\s+([-+]?\d+\.\d+)',
                    text,
                    re.MULTILINE,
                )
            if dirac_matches:
                syms = [normalize_symbol(m[0]) for m in dirac_matches]
                crds = np.array([[float(m[1]), float(m[2]), float(m[3])] for m in dirac_matches], dtype=np.float64)
                self._apply_imported_cartesian_molecule(syms, crds, path, label="openLOWDIN")
            else:
                raise ValueError(f"No GEOMETRY coordinates found in openLOWDIN file: {os.path.basename(path)}")
        except Exception as e:
            dialog_service.error(self, "Error Opening openLOWDIN File", str(e), exc=e)

    def _load_pyscf_script_path(self, path: str):
        """Extract molecular geometry from a PySCF Python script (.py) and load it into the viewer."""
        try:
            import re
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            m_at = re.search(r'mol\.atom\s*=\s*(?:"""(.*?)"""|\'\'\'(.*?)\'\'\')', text, re.DOTALL)
            if not m_at:
                raise ValueError(f"Could not find 'mol.atom = \"\"\"...\"\"\"' geometry block in {os.path.basename(path)}")
            blk = m_at.group(1) or m_at.group(2) or ""
            syms, crds = [], []
            for l in blk.replace(";", "\n").replace("'", "").replace('"', "").splitlines():
                p = l.split()
                if len(p) >= 4 and normalize_symbol(p[0]) in ELEMENT_DATA:
                    try:
                        syms.append(normalize_symbol(p[0]))
                        crds.append([float(p[1]), float(p[2]), float(p[3])])
                    except ValueError:
                        pass
            if not syms:
                raise ValueError(f"No valid atomic coordinates found in {os.path.basename(path)}")
            self._apply_imported_cartesian_molecule(syms, np.array(crds, dtype=np.float64), path, label="PySCF")
        except Exception as e:
            dialog_service.error(self, "Error Opening PySCF Script", str(e), exc=e)

    def _load_brorsen_inp_path(self, path: str):
        """Directly parse and load a newBRORSEN, ORCA, or Q-Chem input file (*.inp / *.in) into AGYMolden."""
        try:
            import re
            from .brorsen_builder.generator import BrorsenInputGenerator
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            # 1. Check if this is a Q-Chem input file ($molecule ... $end)
            if "$molecule" in content.lower() and "$rem" in content.lower():
                m_mol = re.search(r'\$molecule\s*\n\s*[-+]?\d+\s+\d+\s*\n(.*?)\$end', content, re.DOTALL | re.IGNORECASE)
                if m_mol:
                    syms, crds = [], []
                    for l in m_mol.group(1).strip().splitlines():
                        p = l.split()
                        if len(p) >= 4 and normalize_symbol(p[0]) in ELEMENT_DATA:
                            syms.append(normalize_symbol(p[0]))
                            crds.append([float(p[1]), float(p[2]), float(p[3])])
                    if syms:
                        self._apply_imported_cartesian_molecule(syms, np.array(crds, dtype=np.float64), path, label="Q-Chem")
                        return

            # 2. Distinguish newBRORSEN vs ORCA .inp files:
            # newBRORSEN bang lines contain sector keywords (ELECTRONIC, POSITRON, PROTON) or %jobname / %pos_basis
            is_brorsen = bool(
                re.search(r'^\s*!.*\b(?:ELECTRONIC|POSITRON|PROTON|SP_ENERGY)\b', content, re.MULTILINE | re.IGNORECASE)
                or re.search(r'^\s*%(?:jobname|n_pos|pos_basis|nuc_basis|next)\b', content, re.MULTILINE | re.IGNORECASE)
            )

            gen = BrorsenInputGenerator()
            stages, geom_info = gen.load_from_text(content)

            # Check if there is geometry in the input file, update the canvas
            if geom_info and geom_info.get("lines"):
                lines = geom_info["lines"]
                fmt = geom_info.get("format", "xyz")
                if fmt == "zmat":
                    from .zmatrix import parse_zmatrix_text
                    zatoms, vars_dict = parse_zmatrix_text("\n".join(lines))
                    if zatoms:
                        self._push_undo_state()
                        self.cross_bonds = set()
                        self.strict_topology = False
                        self.variables = vars_dict
                        self.set_molecule_from_zatoms(zatoms)
                else:
                    from .zmatrix import parse_xyz, cartesian_to_zmatrix
                    xyz_txt = f"{len(lines)}\nfrom inp\n" + "\n".join(lines)
                    syms, crds = parse_xyz(xyz_txt, is_path=False)
                    if syms and len(crds) == len(syms):
                        self._push_undo_state()
                        self.variables = {}
                        self.cross_bonds = set()
                        self.strict_topology = False
                        bonds = detect_connectivity(syms, crds)
                        bo = detect_bond_orders(syms, crds, bonds)
                        zatoms = cartesian_to_zmatrix(syms, crds, user_bonds=bonds, strict_bonds=True)
                        self.set_molecule_from_zatoms(zatoms, exact_coords=crds)
                        self.viewer.bond_orders = bo
                        self.viewer.update()

            self._remember_file(path)
            if is_brorsen:
                # Launch the newBRORSEN input builder dialog
                self._on_open_brorsen_input_file_direct(path)
            else:
                self.statusBar().showMessage(f"Loaded ORCA input geometry from {os.path.basename(path)}", 4500)
        except Exception as e:
            dialog_service.error(self, "Error Opening Input File", str(e), exc=e)

    def _on_open_brorsen_input_file_direct(self, path: str):
        symbols = [a.symbol for a in self.zatoms] if self.zatoms else []
        coords = self.coords if hasattr(self, "coords") else None
        dlg = BrorsenInputBuilderDialog(
            self,
            symbols=symbols,
            coords=coords,
            zatoms=self.zatoms,
            variables=self.variables,
        )
        dlg.load_input_file(path)
        if dlg.zatoms and len(dlg.zatoms) > 0:
            self._push_undo_state()
            self.set_molecule_from_zatoms(dlg.zatoms, exact_coords=dlg.coords)
            bonds = detect_connectivity(dlg.symbols, dlg.coords)
            self.viewer.bond_orders = detect_bond_orders(dlg.symbols, dlg.coords, bonds)
            self.viewer.update()
        self._remember_file(path)
        dlg.exec()

    def _on_open_brorsen_input_file(self):
        """Open a newBRORSEN *.inp input file into AGYMolden and the Input Builder."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open newBRORSEN Input File",
            self._last_dir(),
            "newBRORSEN Input Files (*.inp *.in *.txt);;All Files (*)",
        )
        if not path:
            return

        symbols = [a.symbol for a in self.zatoms] if self.zatoms else []
        coords = self.coords if hasattr(self, "coords") else None
        dlg = BrorsenInputBuilderDialog(
            self,
            symbols=symbols,
            coords=coords,
            zatoms=self.zatoms,
            variables=self.variables,
        )
        dlg.load_input_file(path)

        # If molecule was updated from the input file, update AGYMolden main window as well
        if dlg.zatoms and len(dlg.zatoms) > 0:
            self._push_undo_state()
            self.set_molecule_from_zatoms(dlg.zatoms, exact_coords=dlg.coords)
            bonds = detect_connectivity(dlg.symbols, dlg.coords)
            self.viewer.bond_orders = detect_bond_orders(dlg.symbols, dlg.coords, bonds)
            self.viewer.update()

        self._remember_file(path)
        dlg.exec()

    def _on_import_qm_log(self):
        """Import quantum chemistry output file (.log, .out) from Gaussian, ORCA, Q-Chem, or xTB."""
        if not can_parse_with_cclib():
            QtWidgets.QMessageBox.warning(
                self, "cclib Required", "cclib is required to parse QM output files. Please install cclib."
            )
            return

        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import QM Log / Output File",
            self._last_dir(),
            "QM Output Files (*.log *.out *.output);;All Files (*)",
        )
        if path:
            self._load_qm_log_path(path)

    def _load_qm_log_path(self, path: str):
        try:
            res = parse_qm_logfile(path)
            self._push_undo_state()
            self.variables = {}

            # Store optimization trajectory
            self._last_qm_res = res
            self.trajectory_coords = res.trajectory_coords
            self.trajectory_energies = res.trajectory_energies
            self.trajectory_symbols = res.symbols

            final_coords = res.trajectory_coords[-1]
            symbols = res.symbols

            detected_bonds = detect_connectivity(symbols, final_coords, strict=False)
            detected_bo = detect_bond_orders(symbols, final_coords, detected_bonds)

            zatoms = cartesian_to_zmatrix(symbols, final_coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True

            self.set_molecule_from_zatoms(zatoms, exact_coords=final_coords)
            self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
            if res.charges is not None and len(res.charges) == len(symbols):
                self.viewer.custom_charges = res.charges
                self.viewer.show_charges = True
                if hasattr(self, "chk_charges"):
                    self.chk_charges.blockSignals(True)
                    self.chk_charges.setChecked(True)
                    self.chk_charges.blockSignals(False)
            self.viewer.update()

            # Apply nuclear forces / gradients if present in the log file
            if getattr(res, "nuclear_forces", None) is not None:
                self.viewer.nuclear_forces = res.nuclear_forces
                if hasattr(self, "chk_forces"):
                    self.chk_forces.blockSignals(True)
                    self.chk_forces.setChecked(True)
                    self.chk_forces.blockSignals(False)
                self.viewer.show_forces = True

            # Enable Trajectory Player if multiple steps exist
            n_steps = len(res.trajectory_coords)
            if hasattr(self, "grp_trajectory"):
                if n_steps > 1:
                    is_irc = getattr(res, "is_irc", False)
                    init_step = n_steps - 1
                    if is_irc:
                        self.grp_trajectory.setTitle("⚛️ Intrinsic Reaction Coordinate (IRC Pathway Player)")
                        if hasattr(self, "combo_traj_mode"):
                            # Bounce mode is the gold standard for inspecting IRC reaction pathways
                            idx_bounce = self.combo_traj_mode.findData("bounce")
                            if idx_bounce >= 0:
                                self.combo_traj_mode.setCurrentIndex(idx_bounce)
                        if hasattr(self, "btn_irc_sweeper"):
                            self.btn_irc_sweeper.setStyleSheet(
                                "background-color: #6366f1; color: #ffffff; padding: 2px 8px; font-weight: bold; border: 1.5px solid #a5b4fc;"
                            )
                        if getattr(res, "irc_pathway", None) and hasattr(res.irc_pathway, "ts_index"):
                            init_step = res.irc_pathway.ts_index
                    else:
                        self.grp_trajectory.setTitle("Trajectory Player (Optimization Steps / MD)")
                        if hasattr(self, "btn_irc_sweeper"):
                            self.btn_irc_sweeper.setStyleSheet(
                                "background-color: #6366f1; color: #ffffff; padding: 2px 8px; font-weight: bold;"
                            )
                    self.grp_trajectory.setVisible(True)
                    self.slider_trajectory.blockSignals(True)
                    self.slider_trajectory.setRange(0, n_steps - 1)
                    self.slider_trajectory.setValue(init_step)
                    self.slider_trajectory.blockSignals(False)
                    self.lbl_trajectory_step.setText(f"Step {init_step + 1}/{n_steps}")
                    if res.trajectory_energies and len(res.trajectory_energies) == n_steps:
                        e_val = res.trajectory_energies[init_step]
                        self.lbl_trajectory_energy.setText(f"E = {e_val:.6f} eV")
                    else:
                        self.lbl_trajectory_energy.setText("")
                else:
                    self.grp_trajectory.setVisible(False)

            # Apply vibrational modes if present in the log file
            if res.vibrational_analysis and res.vibrational_analysis.n_modes > 0:
                self._apply_vibrations((final_coords, res.vibrational_analysis))

            # Apply UV-Vis transitions if present in the log file
            if hasattr(res, "et_energies") and res.et_energies and hasattr(self, "uvvis_widget"):
                self.uvvis_widget.set_uv_vis_data(res.et_energies, res.et_oscs)
                if hasattr(self, "tabs_spectra"):
                    self.tabs_spectra.setCurrentIndex(1)

            # Apply NMR chemical shieldings if present in the log file
            if getattr(res, "nmr_shieldings", None) and hasattr(self, "nmr_widget"):
                self.nmr_widget.set_nmr_data(res.nmr_shieldings)
                if hasattr(self, "tabs_spectra") and not (res.vibrational_analysis and res.vibrational_analysis.n_modes > 0):
                    self.tabs_spectra.setCurrentIndex(2)

            # Apply Molecular Orbitals & Energy Diagram directly from the QM Log if present
            if getattr(res, "orbitals", None) and hasattr(self, "set_orbitals_from_data"):
                from .volumetric import MoldenFileData
                m_dummy = MoldenFileData(
                    title=f"{res.program_name} log orbitals",
                    symbols=symbols,
                    coords=final_coords,
                    orbitals=res.orbitals,
                )
                self.set_orbitals_from_data(m_dummy, homo_idx=res.homo_idx, source_label=os.path.basename(path))
            else:
                # If log itself did not have MOs (e.g. newBRORSEN, Gaussian without pop=full),
                # automatically check for companion .molden in the same folder and populate MO table & ladder
                q_dir = os.path.dirname(path)
                q_base = os.path.splitext(os.path.basename(path))[0]
                for cand_ext in ("_hf_electrons.molden", "_electrons.molden", ".molden", ".molden.input", ".e-.molden", ".E-.molden", ".E-ALPHA.molden", ".molf"):
                    cand_m = os.path.join(q_dir, q_base + cand_ext)
                    if os.path.isfile(cand_m):
                        try:
                            self.load_orbitals_from_molden(cand_m, auto_render=False)
                            break
                        except Exception:
                            log.debug("Auto-loading companion molden failed", exc_info=True)

            # Switch to Molecular Orbitals tab if MOs were loaded
            if hasattr(self, "_active_molden_data") and self._active_molden_data:
                idx = self.tabs.indexOf(getattr(self, "tab_orbitals", None))
                if idx >= 0:
                    self.tabs.setCurrentIndex(idx)

            self._last_loaded_qm_log_path = path
            self._remember_file(path)
            if hasattr(self, "set_interaction_mode"):
                self.set_interaction_mode("inspect")
            if hasattr(self, "_update_qm_summary_banner"):
                self._update_qm_summary_banner(res)
            self.statusBar().showMessage(
                f"Loaded {res.program_name} log: {len(symbols)} atoms, {n_steps} step(s)"
                + (f", {res.vibrational_analysis.n_modes} frequency modes" if res.vibrational_analysis else ""),
                6000,
            )
            # Check for newBRORSEN / QM companion files (electron/positron densities, molden, scans)
            # Only prompt interactively if not triggered from the live Job Monitor auto-loader
            if not getattr(self, "_suppress_companion_prompt", False):
                self._check_and_prompt_companion_files(path)
        except Exception as e:
            dialog_service.error(self, "Error Reading QM Output", str(e), exc=e)

    def _on_view_qm_output_file(self):
        """Open the QM Output & Log Viewer Dialog."""
        path = getattr(self, "_last_loaded_qm_log_path", None)
        # Check active calculation runner if present
        if not path or not os.path.isfile(path):
            job_mon = getattr(self, "_job_monitor_dlg", None)
            if job_mon and getattr(job_mon, "runner", None):
                runner = job_mon.runner
                if getattr(runner, "log_file", None) and os.path.isfile(runner.log_file):
                    path = runner.log_file
                elif getattr(runner, "input_file", None):
                    base_no_ext = os.path.splitext(runner.input_file)[0]
                    for d in filter(None, [getattr(runner, "work_dir", None), os.path.dirname(runner.input_file)]):
                        for ext in (".out", ".log"):
                            cand = os.path.join(d, os.path.basename(base_no_ext) + ext)
                            if os.path.isfile(cand):
                                path = cand
                                break
                        if path:
                            break

        if not path or not os.path.isfile(path):
            # Prompt user to choose a .out or .log file to view
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Select QM Output / Log File to View",
                self._last_dir(),
                "QM Output Files (*.out *.log *.txt);;All Files (*)",
            )
            if not path:
                return

        self._last_loaded_qm_log_path = path
        from .dialog_output_viewer import QMOutputViewerDialog
        dlg = QMOutputViewerDialog(self, file_path=path)
        dlg.exec()

    def _check_and_prompt_companion_files(self, base_file_path: str):
        """
        Scan directory for matching newBRORSEN / QM companion files:
        - Electron density cube: *_hf_electron_density.cube or *.cube
        - Molden wavefunctions: *_hf_electrons.molden, *_positron.molden
        - PES scan output: *.scan
        If found, offer the user to load/overlay them automatically.
        """
        directory = os.path.dirname(base_file_path)
        base_name = os.path.splitext(os.path.basename(base_file_path))[0]
        # Clean prefix: remove trailing _checkpoint, _final, etc.
        prefix = base_name
        for suffix in ("_checkpoint", "_final", "_summary", "_profile"):
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]

        companions = {}
        # Candidate patterns
        cand_cube_elec = os.path.join(directory, f"{prefix}_hf_electron_density.cube")
        cand_cube_pos = os.path.join(directory, f"{prefix}_positron_density.cube")
        cand_molden_elec = os.path.join(directory, f"{prefix}_hf_electrons.molden")
        cand_molden_orca = os.path.join(directory, f"{prefix}.molden.input")
        cand_molden_std = os.path.join(directory, f"{prefix}.molden")
        cand_molden_pos = os.path.join(directory, f"{prefix}_positron.molden")
        cand_molden_pos_hf = os.path.join(directory, f"{prefix}_hf_positrons.molden")
        cand_molden_pos_pl = os.path.join(directory, f"{prefix}_positrons.molden")
        cand_scan = os.path.join(directory, f"{prefix}.scan")

        # Also scan [Artifact Generated] lines inside .out/.log for multi-stage (%next) Molden files
        artifact_moldens = []
        if base_file_path.lower().endswith((".out", ".log")) and os.path.isfile(base_file_path):
            try:
                import re
                with open(base_file_path, "r", encoding="utf-8", errors="ignore") as f_log:
                    log_txt = f_log.read()
                for m_art in re.finditer(r"\[Artifact Generated\]\s+([^\s]+\.molden(?:\.input)?)", log_txt):
                    art_file = m_art.group(1).strip()
                    art_full = art_file if os.path.isabs(art_file) else os.path.join(directory, art_file)
                    if os.path.isfile(art_full) and art_full not in artifact_moldens:
                        artifact_moldens.append(art_full)
            except Exception:
                log.debug("Ignored non-fatal exception", exc_info=True)
        if artifact_moldens:
            self._related_molden_files = list(artifact_moldens)
            elec_arts = [p for p in artifact_moldens if "positron" not in os.path.basename(p).lower()]
            pos_arts = [p for p in artifact_moldens if "positron" in os.path.basename(p).lower()]
            if elec_arts:
                cand_molden_elec = elec_arts[-1]
            if pos_arts:
                cand_molden_pos = pos_arts[-1]

        if os.path.isfile(cand_cube_elec):
            companions["Electron Density Cube"] = cand_cube_elec
        if os.path.isfile(cand_cube_pos):
            companions["Positron Density Cube"] = cand_cube_pos
        if os.path.isfile(cand_molden_elec):
            companions["Electronic Wavefunction (Molden)"] = cand_molden_elec
        elif os.path.isfile(cand_molden_orca):
            companions["ORCA Wavefunction (.molden.input)"] = cand_molden_orca
        elif os.path.isfile(cand_molden_std):
            companions["Electronic Wavefunction (.molden)"] = cand_molden_std
        if os.path.isfile(cand_molden_pos):
            companions["Positron Wavefunction (Molden)"] = cand_molden_pos
        elif os.path.isfile(cand_molden_pos_hf):
            companions["Positron Wavefunction (Molden)"] = cand_molden_pos_hf
        elif os.path.isfile(cand_molden_pos_pl):
            companions["Positron Wavefunction (Molden)"] = cand_molden_pos_pl
        if os.path.isfile(cand_scan):
            companions["PES Scan Curve (.scan)"] = cand_scan

        if not companions:
            return

        comp_list_text = "\n".join(f" • {k}: {os.path.basename(v)}" for k, v in companions.items())
        msg = (
            f"Detected {len(companions)} companion file(s) for this job:\n\n"
            f"{comp_list_text}\n\n"
            f"Would you like to load/overlay them automatically?"
        )
        reply = QtWidgets.QMessageBox.question(
            self,
            "newBRORSEN Companion Files Found",
            msg,
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if "Positron Wavefunction (Molden)" in companions:
                self._companion_positron_molden_path = companions["Positron Wavefunction (Molden)"]
            # Load electron cube or molden
            if "Electron Density Cube" in companions:
                self._load_cube_path(companions["Electron Density Cube"])
            elif "Electronic Wavefunction (Molden)" in companions:
                self._load_molden_path(companions["Electronic Wavefunction (Molden)"])
            elif "ORCA Wavefunction (.molden.input)" in companions:
                self._load_molden_path(companions["ORCA Wavefunction (.molden.input)"])
            elif "Electronic Wavefunction (.molden)" in companions:
                self._load_molden_path(companions["Electronic Wavefunction (.molden)"])
            elif "Positron Wavefunction (Molden)" in companions:
                self._load_molden_path(companions["Positron Wavefunction (Molden)"])

            # If PES Scan exists, trigger PES Scan Dialog
            if "PES Scan Curve (.scan)" in companions and hasattr(self, "_on_open_pes_scan_dialog"):
                self._on_open_pes_scan_dialog(companions["PES Scan Curve (.scan)"])

    def _on_trajectory_slider_changed(self, step: int):
        if not hasattr(self, "trajectory_coords") or not self.trajectory_coords:
            return
        if not (0 <= step < len(self.trajectory_coords)):
            return

        coords = self.trajectory_coords[step]
        n_steps = len(self.trajectory_coords)
        if hasattr(self, "lbl_trajectory_step"):
            self.lbl_trajectory_step.setText(f"Step {step + 1}/{n_steps}")

        if hasattr(self, "lbl_trajectory_energy") and hasattr(self, "trajectory_energies"):
            if self.trajectory_energies and step < len(self.trajectory_energies):
                e_val = self.trajectory_energies[step]
                e_ha = e_val / 27.211386245988
                e_diff_kcal = (e_val - self.trajectory_energies[0]) * 23.0605  # eV to kcal/mol
                self.lbl_trajectory_energy.setText(f"E = {e_ha:.6f} Ha (ΔE: {e_diff_kcal:+.2f} kcal/mol)")

        self.coords = np.array(coords, dtype=np.float64, copy=True)
        self.viewer.set_molecule(
            symbols=[a.symbol for a in self.zatoms],
            coords=self.coords,
            user_bonds=self._current_explicit_bonds(),
        )
        self.viewer.update()

        # Sync active PES dialog if open
        if hasattr(self, "_pes_dlg") and self._pes_dlg and self._pes_dlg.isVisible():
            self._pes_dlg.set_active_step(step)

    def _on_open_pes_plot(self):
        """Open interactive PES Scan & Morse Fit Curve Viewer."""
        from .dialog_pes_scan import PESScanDialog
        dlg = getattr(self, "_pes_dlg", None)
        if dlg is None:
            dlg = PESScanDialog(self, viewer=self.viewer, trajectory_coords=getattr(self, "trajectory_coords", []))
            self._pes_dlg = dlg

            def on_point_selected(step):
                if hasattr(self, "slider_trajectory"):
                    self.slider_trajectory.setValue(step)

            dlg.point_selected.connect(on_point_selected)

        # Pre-populate with current trajectory energies if available
        if hasattr(self, "trajectory_energies") and self.trajectory_energies:
            e_ha = [e / 27.211386245988 for e in self.trajectory_energies]
            qm_res = getattr(self, "_last_qm_res", None)
            is_irc = getattr(qm_res, "is_irc", False)
            irc_pts = getattr(qm_res, "irc_path_points", None)
            max_f = getattr(qm_res, "trajectory_max_forces", None)
            rms_f = getattr(qm_res, "trajectory_rms_forces", None)
            if is_irc and irc_pts and len(irc_pts) == len(e_ha):
                coords_pts = irc_pts
                c_lbl = "Reaction Coordinate s (amu½·Bohr)"
            else:
                coords_pts = [float(i + 1) for i in range(len(e_ha))]
                c_lbl = "Reaction / Optimization Step"
            dlg.load_scan_data(
                coords_pts,
                e_ha,
                coord_label=c_lbl,
                trajectory_coords=getattr(self, "trajectory_coords", None),
                max_forces=max_f,
                rms_forces=rms_f,
            )
            if hasattr(self, "slider_trajectory"):
                dlg.set_active_step(self.slider_trajectory.value())

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _on_traj_first_step(self):
        """Jump immediately to initial trajectory geometry (Step 1)."""
        if hasattr(self, "slider_trajectory"):
            self.slider_trajectory.setValue(0)

    def _on_traj_last_step(self):
        """Jump immediately to final optimized geometry."""
        if hasattr(self, "slider_trajectory"):
            self.slider_trajectory.setValue(self.slider_trajectory.maximum())

    def _on_adopt_trajectory_frame(self):
        """Adopt the currently displayed trajectory frame into the editable Z-Matrix table (with Undo)."""
        if not hasattr(self, "trajectory_coords") or not self.trajectory_coords:
            return
        step = self.slider_trajectory.value() if hasattr(self, "slider_trajectory") else 0
        if not (0 <= step < len(self.trajectory_coords)):
            return
        frame_coords = np.array(self.trajectory_coords[step], dtype=np.float64, copy=True)
        symbols = [a.symbol for a in self.zatoms]
        if len(symbols) != len(frame_coords):
            return
        self._push_undo_state()
        detected_bonds = detect_connectivity(symbols, frame_coords, strict=False)
        detected_bo = detect_bond_orders(symbols, frame_coords, detected_bonds)
        zatoms = cartesian_to_zmatrix(
            symbols, frame_coords, existing_zatoms=self.zatoms, user_bonds=detected_bonds, strict_bonds=True
        )
        tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
        self.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
        self.strict_topology = True
        self.set_molecule_from_zatoms(zatoms, exact_coords=frame_coords)
        self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
        self.viewer.update()
        self.statusBar().showMessage(
            f"📌 Adopted trajectory frame {step + 1}/{len(self.trajectory_coords)} into active Z-Matrix (Ctrl+Z to undo).",
            4500,
        )

    def _on_traj_prev_step(self):
        if hasattr(self, "slider_trajectory"):
            val = self.slider_trajectory.value()
            if val > 0:
                self.slider_trajectory.setValue(val - 1)

    def _on_traj_next_step(self):
        if hasattr(self, "slider_trajectory"):
            val = self.slider_trajectory.value()
            if val < self.slider_trajectory.maximum():
                self.slider_trajectory.setValue(val + 1)

    def _on_traj_speed_changed(self, idx: int):
        """Adjust animation timer interval based on user selected playback speed."""
        if hasattr(self, "combo_traj_speed"):
            interval = self.combo_traj_speed.currentData() or 120
            if not hasattr(self, "_traj_timer"):
                self._traj_timer = QtCore.QTimer(self)
                self._traj_timer.timeout.connect(self._on_traj_timer_tick)
                self._traj_direction = 1
            self._traj_timer.setInterval(int(interval))

    def _on_traj_toggle_play(self):
        """Toggle animated playback of the trajectory (optimization / IRC pathway)."""
        if not hasattr(self, "_traj_timer"):
            self._traj_timer = QtCore.QTimer(self)
            interval = self.combo_traj_speed.currentData() if hasattr(self, "combo_traj_speed") else 120
            self._traj_timer.setInterval(int(interval or 120))
            self._traj_timer.timeout.connect(self._on_traj_timer_tick)
            self._traj_direction = 1

        if self._traj_timer.isActive():
            self._traj_timer.stop()
            if hasattr(self, "btn_traj_play"):
                self.btn_traj_play.setText("▶ Play")
        else:
            self._traj_direction = 1
            # If already at end, restart from beginning
            if hasattr(self, "slider_trajectory") and self.slider_trajectory.value() >= self.slider_trajectory.maximum():
                self.slider_trajectory.setValue(0)
            self._traj_timer.start()
            if hasattr(self, "btn_traj_play"):
                self.btn_traj_play.setText("⏸ Pause")

    def _on_traj_timer_tick(self):
        if not hasattr(self, "slider_trajectory"):
            return
        val = self.slider_trajectory.value()
        max_val = self.slider_trajectory.maximum()
        if max_val <= 0:
            return

        mode_text = self.combo_traj_mode.currentText() if hasattr(self, "combo_traj_mode") else "Loop"
        direction = getattr(self, "_traj_direction", 1)

        if "Bounce" in mode_text:
            next_val = val + direction
            if next_val >= max_val:
                self._traj_direction = -1
                next_val = max_val
            elif next_val <= 0:
                self._traj_direction = 1
                next_val = 0
            self.slider_trajectory.setValue(next_val)
        elif "Once" in mode_text:
            if val >= max_val:
                self._on_traj_toggle_play()
            else:
                self.slider_trajectory.setValue(val + 1)
        else:  # Loop
            if val >= max_val:
                self.slider_trajectory.setValue(0)
            else:
                self.slider_trajectory.setValue(val + 1)

    def _on_export_trajectory_movie(self):
        """Export optimization or MD trajectory steps as animated GIF or MP4/WebM movie."""
        if not hasattr(self, "trajectory_coords") or len(self.trajectory_coords) < 2:
            QtWidgets.QMessageBox.information(
                self, "Export Trajectory Movie", "Please load a calculation with multiple trajectory steps first."
            )
            return

        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Trajectory Movie",
            "trajectory_movie.gif",
            "Animated GIF (*.gif);;MP4 Video (*.mp4);;WebM Video (*.webm);;All Files (*)"
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if not ext:
            ext = ".gif"
            path += ext

        # Pause active playback during export
        was_playing = getattr(self, "_traj_timer", None) is not None and self._traj_timer.isActive()
        if was_playing:
            self._traj_timer.stop()

        current_step = self.slider_trajectory.value()
        progress = QtWidgets.QProgressDialog("Rendering trajectory movie frames...", "Cancel", 0, len(self.trajectory_coords), self)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)

        frames_rgb = []
        try:
            for idx in range(len(self.trajectory_coords)):
                progress.setValue(idx)
                if progress.wasCanceled():
                    return

                self.slider_trajectory.setValue(idx)
                self.viewer.repaint()
                QtWidgets.QApplication.processEvents()

                if self.viewer.isValid() and self.viewer.context() is not None:
                    qimg = self.viewer.grabFramebuffer().convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
                else:
                    w = max(400, self.viewer.width())
                    h = max(300, self.viewer.height())
                    qimg = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGBA8888)
                    qimg.fill(QtGui.QColor(30, 33, 41))

                w, h = qimg.width(), qimg.height()
                ptr = qimg.bits()
                ptr.setsize(h * w * 4)
                arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4)).copy()
                frames_rgb.append(arr[:, :, :3])

            progress.setLabelText("Encoding movie file...")
            progress.setValue(len(self.trajectory_coords))

            if ext in (".mp4", ".webm"):
                try:
                    import imageio
                    writer = imageio.get_writer(path, fps=10)
                    for f in frames_rgb:
                        writer.append_data(f)
                    writer.close()
                except Exception as vid_err:
                    # Fallback to Pillow GIF if video codec/FFmpeg is unavailable
                    gif_path = os.path.splitext(path)[0] + ".gif"
                    from PIL import Image
                    imgs = [Image.fromarray(f) for f in frames_rgb]
                    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)
                    QtWidgets.QMessageBox.warning(
                        self, "Video Codec Notice",
                        f"Direct MP4/WebM encoding encountered: {vid_err}\n"
                        f"Successfully fell back to high-resolution GIF at:\n{gif_path}"
                    )
                    return
            else:
                from PIL import Image
                imgs = [Image.fromarray(f) for f in frames_rgb]
                imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

            self.statusBar().showMessage(f"Exported trajectory movie to {os.path.basename(path)}", 5000)
            QtWidgets.QMessageBox.information(self, "Export Complete", f"Trajectory movie exported successfully to:\n{path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", f"Failed to export trajectory movie:\n{e}")
        finally:
            progress.close()
            self.slider_trajectory.setValue(current_step)
            if was_playing:
                self._traj_timer.start()

    def _on_import_xyz(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Import XYZ File", self._last_dir(), "XYZ Molecular Files (*.xyz);;All Files (*)"
        )
        if path:
            self._load_xyz_path(path)

    def _load_xyz_path(self, path: str):
        try:
            symbols, coords = parse_xyz(path)
            if len(symbols) == 0:
                QtWidgets.QMessageBox.warning(self, "Empty File", "No atoms found in XYZ file.")
                return
            self._push_undo_state()
            self.variables = {}

            # Detect full covalent connectivity and multiple bond orders directly from exact XYZ coordinates
            detected_bonds = detect_connectivity(symbols, coords, strict=False)
            detected_bo = detect_bond_orders(symbols, coords, detected_bonds)

            # Build connectivity-aware Z-matrix while keeping exact XYZ coordinates & ring closures
            zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True

            # Preserve exact Cartesian coordinates and orientation from the XYZ file!
            self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
            self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
            self.viewer.update()
            self._remember_file(path)

            # Check if this is a multi-frame XYZ or CREST conformer ensemble
            try:
                from .zmatrix import parse_multi_xyz
                from .crest_explorer import extract_energy_from_comment
                _, multi_frames = parse_multi_xyz(path, is_path=True)
                if len(multi_frames) > 1:
                    traj_energies = []
                    for c_comm, _ in multi_frames:
                        p_e = extract_energy_from_comment(c_comm)
                        traj_energies.append(p_e[0] if p_e is not None else 0.0)
                    self.trajectory_coords = [f[1] for f in multi_frames]
                    self.trajectory_symbols = symbols
                    self.trajectory_energies = traj_energies
                    n_frames = len(multi_frames)
                    is_crest = "crest" in os.path.basename(path).lower()
                    if hasattr(self, "grp_trajectory"):
                        title_lbl = "Conformer Ensemble" if is_crest else "Trajectory Player"
                        self.grp_trajectory.setTitle(f"{title_lbl} ({n_frames} Frames)")
                        self.grp_trajectory.setVisible(True)
                        self.slider_trajectory.blockSignals(True)
                        self.slider_trajectory.setRange(0, n_frames - 1)
                        self.slider_trajectory.setValue(0)
                        self.slider_trajectory.blockSignals(False)
                        self.lbl_trajectory_step.setText(f"1/{n_frames}")
                    if is_crest:
                        self.statusBar().showMessage(
                            f"Imported CREST Conformer Ensemble ({n_frames} conformers): {os.path.basename(path)}. Press Ctrl+Shift+E for Ensemble Explorer!",
                            8000,
                        )
                        return
            except Exception as e:
                log.debug("Multi-XYZ trajectory/CREST detection error: %s", e)

            self.statusBar().showMessage(
                f"Imported XYZ ({len(symbols)} atoms, {len(detected_bonds)} bonds) preserving exact 3D coordinates: {os.path.basename(path)}",
                5000
            )
        except Exception as e:
            dialog_service.error(self, "Error Importing XYZ", str(e), exc=e)

    def _load_pdb_path(self, path: str):
        try:
            symbols, coords, bonds = parse_pdb(path)
            if len(symbols) == 0:
                dialog_service.warning(self, "Empty File", "No atoms found in PDB file.")
                return
            self._push_undo_state()
            self.variables = {}
            if not bonds:
                bonds = detect_connectivity(symbols, coords, strict=False)
            detected_bo = detect_bond_orders(symbols, coords, bonds)
            zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True
            self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
            self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}
            self.viewer.update()
            self._remember_file(path)
            self.statusBar().showMessage(f"Loaded PDB {os.path.basename(path)} ({len(symbols)} atoms).", 4000)
        except Exception as e:
            dialog_service.error(self, "Error Opening PDB", str(e), exc=e)

    def _load_sdf_path(self, path: str):
        try:
            symbols, coords, bonds, bo_dict = parse_mol_sdf(path)
            if len(symbols) == 0:
                dialog_service.warning(self, "Empty File", "No atoms found in SDF / MOL file.")
                return
            self._push_undo_state()
            self.variables = {}
            if not bonds:
                bonds = detect_connectivity(symbols, coords, strict=False)
                bo_dict = detect_bond_orders(symbols, coords, bonds)
            zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True
            self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
            self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in bo_dict.items() if o != 1}
            self.viewer.update()
            self._remember_file(path)
            self.statusBar().showMessage(f"Loaded SDF/MOL {os.path.basename(path)} ({len(symbols)} atoms).", 4000)
        except Exception as e:
            dialog_service.error(self, "Error Opening SDF/MOL", str(e), exc=e)

    def _load_cif_path(self, path: str):
        try:
            symbols, coords, cell = parse_cif(path)
            if len(symbols) == 0:
                dialog_service.warning(self, "Empty File", "No atomic sites found in CIF file.")
                return
            self._push_undo_state()
            self.variables = {}
            detected_bonds = detect_connectivity(symbols, coords, strict=False)
            detected_bo = detect_bond_orders(symbols, coords, detected_bonds)

            zatoms = cartesian_to_zmatrix(symbols, coords, user_bonds=detected_bonds, strict_bonds=True)
            tree = {norm_bond(a.ref1, i) for i, a in enumerate(zatoms) if a.ref1 is not None}
            self.cross_bonds = {norm_bond(u, v) for u, v in detected_bonds if norm_bond(u, v) not in tree}
            self.strict_topology = True

            self.set_molecule_from_zatoms(zatoms, exact_coords=coords)
            self.viewer.bond_orders = {norm_bond(u, v): o for (u, v), o in detected_bo.items() if o != 1}

            from .pbc import UnitCell
            self.unit_cell = UnitCell.from_dict(cell)
            self.viewer.unit_cell = self.unit_cell
            self.viewer.show_unit_cell = True

            self.viewer.update()
            self._remember_file(path)

            cell_str = f"a={cell.get('a', 0):.2f} b={cell.get('b', 0):.2f} c={cell.get('c', 0):.2f} Å"
            self.statusBar().showMessage(f"Loaded CIF crystal structure {os.path.basename(path)} ({len(symbols)} sites, {cell_str}).", 5000)
        except Exception as e:
            dialog_service.error(self, "Error Opening CIF", str(e), exc=e)

    def _on_save_zmat(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Z-Matrix File", "molecule.zmat", "Z-Matrix (*.zmat);;Gaussian Input (*.gjf *.inp);;All Files (*)"
        )
        if path:
            try:
                txt = format_zmatrix_text(self.zatoms, self.variables)
                with open(path, "w") as f:
                    f.write(txt)
                self._remember_file(path)
                self.statusBar().showMessage(f"Saved Z-matrix to {os.path.basename(path)}", 4000)
            except (OSError, IOError) as e:
                dialog_service.error(self, "Error Saving File", str(e), exc=e)

    def _on_save_zmat_format(self, fmt: str):
        default = "molecule_num.zmat" if fmt == "numeric" else "molecule_var.zmat"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Z-Matrix File", default, "Z-Matrix (*.zmat);;Gaussian Input (*.gjf *.inp);;All Files (*)"
        )
        if not path:
            return
        try:
            if self.variables:
                self.variables = sync_variables_from_atoms(self.zatoms, self.variables)
            if fmt == "numeric":
                txt = format_zmatrix_numeric(self.zatoms)
            else:
                txt = format_zmatrix_variables(self.zatoms, self.variables)
            with open(path, "w") as f:
                f.write(txt)
            self._remember_file(path)
            self.statusBar().showMessage(f"Saved Z-matrix ({fmt}) to {os.path.basename(path)}", 4000)
        except (OSError, IOError) as e:
            dialog_service.error(self, "Error Saving File", str(e), exc=e)

    def _on_assign_symmetric_bond_variables(self):
        if len(self.zatoms) < 2:
            return
        self._push_undo_state()
        self.variables = assign_symmetric_variables(
            self.zatoms, include_angles=False, include_dihedrals=False
        )
        self.update_geometry_live()
        self.table_widget.load_zatoms(self.zatoms)
        n_bonds = max(0, len(self.zatoms) - 1)
        self.statusBar().showMessage(
            f"Assigned {len(self.variables)} symmetric bond variable(s) for {n_bonds} bonds (angles & dihedrals kept fixed).", 6000)

    def _on_assign_symmetric_variables(self):
        if len(self.zatoms) < 2:
            return
        self._push_undo_state()
        self.variables = assign_symmetric_variables(self.zatoms)
        self.update_geometry_live()
        self.table_widget.load_zatoms(self.zatoms)
        n_coords = sum(min(i, 3) for i in range(len(self.zatoms)))
        self.statusBar().showMessage(
            f"Assigned {len(self.variables)} symmetric variables for {n_coords} internal coordinates.", 6000)

    def _on_assign_individual_variables(self):
        if len(self.zatoms) < 2:
            return
        self._push_undo_state()
        clear_variables(self.zatoms)
        zatoms, vars_dict = parse_zmatrix_text(format_zmatrix_variables(self.zatoms, {}))
        for a, b in zip(self.zatoms, zatoms):
            a.dist_var, a.angle_var, a.dihedral_var = b.dist_var, b.angle_var, b.dihedral_var
        self.variables = vars_dict
        self.table_widget.load_zatoms(self.zatoms)
        self._sync_raw_text()
        self.statusBar().showMessage(f"Assigned {len(vars_dict)} individual variables.", 4000)

    def _on_clear_variables(self):
        self._push_undo_state()
        clear_variables(self.zatoms)
        self.variables = {}
        self.table_widget.load_zatoms(self.zatoms)
        self._sync_raw_text()
        self.statusBar().showMessage("Variables cleared; Z-matrix uses numbers.", 4000)

    def _on_export_xyz(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Cartesian XYZ", "molecule.xyz", "XYZ Molecular Files (*.xyz);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.zatoms]
                write_xyz(path, symbols, self.coords)
                self.statusBar().showMessage(f"Exported Cartesian XYZ to {os.path.basename(path)}", 4000)
            except (OSError, IOError) as e:
                dialog_service.error(self, "Error Exporting XYZ", str(e), exc=e)

    def _on_export_pdb(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Protein Data Bank (PDB)", "molecule.pdb", "PDB Molecular Files (*.pdb);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.zatoms]
                bonds = list(self._current_explicit_bonds())
                write_pdb(path, symbols, self.coords, bonds=bonds)
                self.statusBar().showMessage(f"Exported PDB structure to {os.path.basename(path)}", 4000)
            except (OSError, IOError) as e:
                dialog_service.error(self, "Error Exporting PDB", str(e), exc=e)

    def _on_export_sdf(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Structure-Data File (SDF/MOL)", "molecule.sdf", "SDF Files (*.sdf);;MOL Files (*.mol);;All Files (*)"
        )
        if path:
            try:
                symbols = [a.symbol for a in self.zatoms]
                bonds = list(self._current_explicit_bonds())
                bo = getattr(self.viewer, "bond_orders", {})
                write_mol_sdf(path, symbols, self.coords, bonds=bonds, bond_orders=bo)
                self.statusBar().showMessage(f"Exported SDF/MOL structure to {os.path.basename(path)}", 4000)
            except (OSError, IOError) as e:
                dialog_service.error(self, "Error Exporting SDF/MOL", str(e), exc=e)

    def _on_export_html(self):
        """Export current molecule and calculation results to standalone interactive 3D WebGL HTML."""
        from .html_export import generate_interactive_html
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Interactive 3D HTML", "molecule.html", "HTML Webpage (*.html);;All Files (*)"
        )
        if not path:
            return

        try:
            symbols = [a.symbol for a in self.zatoms]
            coords = self.coords
            bonds = self._current_explicit_bonds()

            # Gather summary details if available
            qm_summary = {}
            res = getattr(self, "_last_qm_res", None) or getattr(self, "_last_qm_result", None)
            if res:
                if getattr(res, "program_name", ""):
                    qm_summary["program"] = res.program_name
                if getattr(res, "trajectory_energies", None) and res.trajectory_energies:
                    qm_summary["energy_ha"] = res.trajectory_energies[-1] / 27.211386245988
                if getattr(res, "dipole_debye", None) is not None:
                    qm_summary["dipole_debye"] = res.dipole_debye
                if getattr(res, "homo_ev", None) is not None:
                    qm_summary["homo_ev"] = res.homo_ev
                if getattr(res, "lumo_ev", None) is not None:
                    qm_summary["lumo_ev"] = res.lumo_ev
                if getattr(res, "vibrational_analysis", None) and hasattr(res.vibrational_analysis, "n_modes"):
                    qm_summary["frequencies_count"] = res.vibrational_analysis.n_modes

            traj = getattr(self, "trajectory_coords", None)
            energies = getattr(self, "trajectory_energies", None)

            title = f"{symbols[0] if symbols else 'Molecule'} System"
            if hasattr(self, "title_override") and self.title_override:
                title = self.title_override

            html_content = generate_interactive_html(
                symbols=symbols,
                coords=coords,
                title=title,
                bonds=bonds,
                trajectory=traj,
                trajectory_energies=energies,
                qm_summary=qm_summary if qm_summary else None,
            )

            with open(path, "w", encoding="utf-8") as f:
                f.write(html_content)

            self.statusBar().showMessage(f"Exported interactive 3D HTML to {os.path.basename(path)}", 4000)
            QtWidgets.QMessageBox.information(
                self,
                "HTML Export Complete",
                f"Interactive 3D HTML successfully generated:\n{path}\n\n"
                "This file is completely self-contained and can be opened in any web browser, "
                "tablet, or phone without installing Python."
            )
        except Exception as e:
            dialog_service.error(self, "Error Exporting HTML", str(e), exc=e)

    def _on_export_reproducibility_bundle(self):
        """Export complete Supporting Information (SI) reproducibility bundle (.zip)."""
        if len(self.zatoms) == 0:
            QtWidgets.QMessageBox.information(
                self, "Export SI Bundle", "Please build or load a molecule before exporting a reproducibility bundle."
            )
            return

        from .reproducibility import create_reproducibility_bundle

        default_name = "reproducibility_bundle.zip"
        loaded_path = getattr(self, "current_file_path", None)
        if loaded_path:
            stem = os.path.splitext(os.path.basename(loaded_path))[0]
            default_name = f"{stem}_SI_bundle.zip"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export SI Reproducibility Bundle",
            os.path.join(self._last_dir(), default_name),
            "ZIP Archives (*.zip);;All Files (*)",
        )
        if not path:
            return

        try:
            symbols = [a.symbol for a in self.zatoms]
            coords = self.coords
            res = getattr(self, "_last_qm_res", None) or getattr(self, "_last_qm_result", None)

            input_path = None
            log_path = None
            if loaded_path and os.path.isfile(loaded_path):
                ext = os.path.splitext(loaded_path)[1].lower()
                if ext in (".com", ".gjf", ".inp", ".in", ".py", ".lowdin"):
                    input_path = loaded_path
                elif ext in (".log", ".out", ".fchk", ".fch", ".molden"):
                    log_path = loaded_path
                    stem = os.path.splitext(loaded_path)[0]
                    for in_ext in (".com", ".gjf", ".inp", ".in"):
                        cand = stem + in_ext
                        if os.path.isfile(cand):
                            input_path = cand
                            break

            title = f"{symbols[0] if symbols else 'Molecule'} System"
            if hasattr(self, "title_override") and self.title_override:
                title = self.title_override
            elif loaded_path:
                title = os.path.splitext(os.path.basename(loaded_path))[0]

            out_zip = create_reproducibility_bundle(
                output_archive_path=path,
                symbols=symbols,
                coords=coords,
                log_result=res,
                log_file_path=log_path,
                input_file_path=input_path,
                job_title=title,
            )
            self.statusBar().showMessage(f"Exported SI Reproducibility Bundle to {os.path.basename(out_zip)}", 5000)
            QtWidgets.QMessageBox.information(
                self,
                "SI Bundle Exported",
                f"Supporting Information Bundle successfully exported to:\n{out_zip}\n\nIncludes:\n"
                f"• GEOMETRY.xyz (Cartesian coordinates)\n"
                f"• GEOMETRY.zmat (Z-matrix coordinates)\n"
                f"• CALCULATION_SUMMARY.json (Thermochemical & orbital properties)\n"
                f"• COMPUTATIONAL_DETAILS.md (Publication-ready SI methods & tables)\n"
                + (f"• Raw output log: {os.path.basename(log_path)}\n" if log_path else "")
                + (f"• Raw input file: {os.path.basename(input_path)}\n" if input_path else "")
            )
        except Exception as e:
            dialog_service.error(self, "Export Error", f"Failed to export reproducibility bundle:\n{e}", exc=e)

    def export_image(self, path: str, transparent: bool = False, bg_mode: Optional[str] = None, scale: int = 2, clean_only: bool = True, dpi: int = 300) -> QtGui.QImage:
        """Render and save a PNG image of the 3D molecule with customizable background (dark, white, or transparent) and embedded DPI metadata."""
        prev_trans = getattr(self.viewer, "transparent_bg", False)
        prev_clean = getattr(self.viewer, "export_clean_mode", False)
        prev_bg_mode = getattr(self.viewer, "bg_color_mode", "dark")
        try:
            self.viewer.transparent_bg = bool(transparent)
            self.viewer.export_clean_mode = bool(clean_only)
            if bg_mode in ("dark", "white"):
                self.viewer.bg_color_mode = bg_mode
            self.viewer.update()
            QtWidgets.QApplication.processEvents()
            if self.viewer.isValid() and self.viewer.context() is not None:
                img = self.viewer.grabFramebuffer()
            else:
                w = max(400, self.viewer.width())
                h = max(300, self.viewer.height())
                img = QtGui.QImage(w, h, QtGui.QImage.Format.Format_ARGB32)
                if transparent:
                    img.fill(QtCore.Qt.GlobalColor.transparent)
                elif bg_mode == "white" or (not transparent and prev_bg_mode == "white"):
                    img.fill(QtCore.Qt.GlobalColor.white)
                else:
                    img.fill(QtGui.QColor(30, 33, 41))
        finally:
            self.viewer.transparent_bg = prev_trans
            self.viewer.export_clean_mode = prev_clean
            self.viewer.bg_color_mode = prev_bg_mode
            self.viewer.update()

        if scale > 1 and not img.isNull():
            img = img.scaled(
                img.width() * scale,
                img.height() * scale,
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        if not img.isNull() and dpi > 0:
            dpm = int(dpi / 0.0254)
            img.setDotsPerMeterX(dpm)
            img.setDotsPerMeterY(dpm)

        if path:
            img.save(path, "PNG")
        return img

    def copy_image_to_clipboard(self, transparent: bool = True, scale: int = 2, clean_only: bool = True):
        """Quickly render and copy high-resolution viewport image to system clipboard (GaussView/Chemcraft style)."""
        try:
            bg_mode = "dark" if not transparent else "dark"
            img = self.export_image(
                path="",
                transparent=transparent,
                bg_mode=bg_mode,
                scale=scale,
                clean_only=clean_only,
            )
            QtWidgets.QApplication.clipboard().setImage(img)
            self.statusBar().showMessage(
                f"Copied {img.width()}x{img.height()} 3D image to clipboard ({'transparent' if transparent else 'solid'}). Ready to paste!",
                4000,
            )
        except Exception as e:
            log.exception("Unhandled error copying viewport image to clipboard")
            QtWidgets.QMessageBox.warning(self, "Copy Image Failed", str(e))

    def _on_export_image(self):
        """Export publication-quality image with custom resolution and background presets."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Export Publication Image (High-DPI PNG)")
        dlg.resize(460, 310)
        layout = QtWidgets.QVBoxLayout(dlg)

        form = QtWidgets.QFormLayout()
        form.setSpacing(12)

        # Journal Presets
        combo_preset = QtWidgets.QComboBox()
        combo_preset.addItem("Custom Scaling / Resolution", None)
        combo_preset.addItem("ACS Single Column (3.25 in / 8.25 cm @ 300 DPI - JACS/JOC)", {"scale": 2, "dpi": 300, "bg": "transparent"})
        combo_preset.addItem("ACS Double Column (7.0 in / 17.8 cm @ 300 DPI)", {"scale": 4, "dpi": 300, "bg": "transparent"})
        combo_preset.addItem("RSC Single Column (8.3 cm @ 300 DPI - Chem. Sci./PCCP)", {"scale": 2, "dpi": 300, "bg": "transparent"})
        combo_preset.addItem("Nature / Springer (89 mm @ 300 DPI)", {"scale": 2, "dpi": 300, "bg": "white"})
        combo_preset.addItem("TOC / Graphical Abstract Square (300 DPI)", {"scale": 3, "dpi": 300, "bg": "transparent"})
        combo_preset.addItem("Presentation Slide Widescreen (16:9 @ 300 DPI)", {"scale": 3, "dpi": 300, "bg": "transparent"})
        form.addRow("<b>Journal Preset:</b>", combo_preset)

        # Resolution preset
        combo_res = QtWidgets.QComboBox()
        combo_res.addItem("Standard (1x screen resolution, ~96 DPI)", 1)
        combo_res.addItem("High Quality (2x, ~200 DPI - Presentations)", 2)
        combo_res.addItem("Publication Print (4x, ~400 DPI - Journal figures)", 4)
        combo_res.addItem("Ultra-High Resolution (8x, ~800 DPI)", 8)
        combo_res.setCurrentIndex(1)  # Default 2x
        form.addRow("<b>Resolution:</b>", combo_res)

        # DPI metadata
        combo_dpi = QtWidgets.QComboBox()
        combo_dpi.addItem("300 DPI (Standard Journal Submission)", 300)
        combo_dpi.addItem("600 DPI (High-Density Print / Line Art)", 600)
        combo_dpi.addItem("150 DPI (Web & Screen)", 150)
        combo_dpi.addItem("72 DPI (Standard Web)", 72)
        form.addRow("<b>Physical DPI:</b>", combo_dpi)

        # Background mode
        combo_bg = QtWidgets.QComboBox()
        combo_bg.addItem("Transparent Background (Recommended for papers/slides)", "transparent")
        combo_bg.addItem("Pure White Background (Print)", "white")
        combo_bg.addItem("Dark Background (Current Viewport Theme)", "dark")
        form.addRow("<b>Background:</b>", combo_bg)

        # Clean mode
        chk_clean = QtWidgets.QCheckBox("Hide selection rings, labels, and coordinate axes")
        chk_clean.setChecked(True)
        form.addRow("Clean Graphics:", chk_clean)

        def _on_preset_selected(idx):
            data = combo_preset.itemData(idx)
            if data:
                scale_val = data.get("scale", 2)
                dpi_val = data.get("dpi", 300)
                bg_val = data.get("bg", "transparent")
                for i in range(combo_res.count()):
                    if combo_res.itemData(i) == scale_val:
                        combo_res.setCurrentIndex(i)
                        break
                for i in range(combo_dpi.count()):
                    if combo_dpi.itemData(i) == dpi_val:
                        combo_dpi.setCurrentIndex(i)
                        break
                for i in range(combo_bg.count()):
                    if combo_bg.itemData(i) == bg_val:
                        combo_bg.setCurrentIndex(i)
                        break

        combo_preset.currentIndexChanged.connect(_on_preset_selected)

        layout.addLayout(form)
        layout.addSpacing(10)

        # Action Buttons
        btn_box = QtWidgets.QHBoxLayout()
        btn_box.addStretch()
        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_cancel)

        btn_save = QtWidgets.QPushButton("💾 Save Image...")
        btn_save.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 6px 14px;")
        btn_save.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_save)

        layout.addLayout(btn_box)

        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return

        scale_factor = int(combo_res.currentData())
        dpi_val = int(combo_dpi.currentData())
        bg_choice = combo_bg.currentData()
        clean_only = chk_clean.isChecked()

        default_path = os.path.join(self._last_dir(), "molecule.png") if self._last_dir() else "molecule.png"
        path, sel_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image File",
            default_path,
            "PNG Image (*.png);;Scalable Vector Graphics (*.svg);;All Files (*)",
        )
        if not path:
            return

        if path.lower().endswith(".svg") or "svg" in sel_filter.lower():
            if not path.lower().endswith(".svg"):
                path += ".svg"
            try:
                from .svg_exporter import export_molecule_svg
                symbols = [a.symbol for a in self.zatoms]
                coords = np.array(self.coords, copy=True)
                bonds = self._current_explicit_bonds() if hasattr(self, "_current_explicit_bonds") else getattr(self, "bonds", [])
                bg = "transparent" if bg_choice == "transparent" else ("#ffffff" if bg_choice == "white" else "#111827")
                svg_data = export_molecule_svg(
                    symbols,
                    coords,
                    bonds=bonds,
                    background=bg,
                    dark_mode=(bg_choice == "dark"),
                )
                with open(path, "w", encoding="utf-8") as f:
                    f.write(svg_data)
                self.statusBar().showMessage(f"Exported publication vector SVG to {os.path.basename(path)}", 4000)
                return
            except Exception as e:
                dialog_service.error(self, "Export SVG Error", f"Failed to export SVG vector file:\n{e}", exc=e)
                return

        if not path.lower().endswith(".png"):
            path += ".png"

        transparent = (bg_choice == "transparent")
        bg_mode = "white" if bg_choice == "white" else "dark"
        try:
            img = self.export_image(
                path,
                transparent=transparent,
                bg_mode=bg_mode,
                scale=scale_factor,
                clean_only=clean_only,
                dpi=dpi_val,
            )

            self._remember_file(path)
            self.statusBar().showMessage(
                f"Exported {img.width()}x{img.height()} ({bg_choice}, {scale_factor}x) PNG to {os.path.basename(path)}", 5000
            )
        except Exception as e:
            dialog_service.error(self, "Error Exporting Image", str(e), exc=e)

    def _on_show_shortcuts(self):
        """Display an interactive dialog showing keyboard and mouse shortcuts with search filter."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("AgyMolden - Keyboard & Mouse Shortcuts")
        dlg.resize(750, 560)
        layout = QtWidgets.QVBoxLayout(dlg)

        header = QtWidgets.QLabel("<h3>⌨️ Keyboard & Mouse Controls</h3><p style='color: #94a3b8;'>Search or browse shortcuts below:</p>")
        layout.addWidget(header)

        filter_edit = QtWidgets.QLineEdit()
        filter_edit.setPlaceholderText("🔍 Type to filter shortcuts (e.g. bond, select, rotate, delete)...")
        layout.addWidget(filter_edit)

        table = QtWidgets.QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Key / Action", "Description", "Context / Category"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        layout.addWidget(table)

        shortcuts_data = [
            ("Left-Click & Drag (Canvas)", "Orbit / rotate 3D camera", "3D Navigation"),
            ("Right-Click & Drag", "Pan camera in screen plane", "3D Navigation"),
            ("Scroll Wheel / Middle Drag", "Zoom in / out", "3D Navigation"),
            ("Space", "Reset camera view & auto-center molecule", "3D Navigation"),
            ("F5", "Toggle Perspective vs Orthographic projection", "3D Navigation"),
            ("Tab", "Toggle ✏️ Build/Draw mode vs 🔍 Inspect/Measure mode", "Modes"),
            ("Left-Drag from Atom (Build)", "Sprout new bond / atom or close ring to target atom", "Building"),
            ("Double-Click Canvas / Atom", "Paste current element or fragment", "Building"),
            ("Double-Click Bond", "Cycle bond order (Single -> Double -> Triple -> Aromatic)", "Building"),
            ("1 / 2 / 3 (with bond selected)", "Set bond order to Single, Double, or Triple", "Building"),
            ("C, H, O, N, S, P, F, B, I, X", "Select element brush (or mutate selected atom in Build mode)", "Building"),
            ("Shift + C, Shift + S, Shift + B", "Select Cl (Chlorine), Si (Silicon), or Br (Bromine)", "Building"),
            ("Delete / Backspace", "Delete all currently selected atoms / bonds", "Editing"),
            ("Shift + Click / Ctrl + Click", "Multi-select atoms to measure Distance, Angle, or Dihedral", "Selection"),
            ("Shift + Drag (Canvas)", "Rubber-band box selection", "Selection"),
            ("Ctrl + Left-Drag Atom(s)", "Translate selected atom(s) in camera plane", "Manipulation"),
            ("Alt + Left-Drag (>= 2 atoms)", "Rotate selected subgroup in 3D around its centroid", "Manipulation"),
            ("Alt + Left-Drag on Bond", "Interactive 3D dihedral / rotamer torsion twist", "Manipulation"),
            ("Ctrl+Z / Ctrl+Y", "Undo / Redo last geometry modification", "History"),
            ("Ctrl+O", "Universal file opener (.out, .xyz, .molden, .cube, .zmat, .inp)", "File I/O"),
            ("Ctrl+S", "Save Z-Matrix (.zmat)", "File I/O"),
            ("Ctrl+C / Ctrl+V", "Copy selected atoms / Paste molecule into 3D space", "Clipboard"),
            ("Ctrl+K", "Flush clipboard memory", "Clipboard"),
            ("Ctrl+M", "Quick Optimize Geometry (Force Field / xTB)", "Tools"),
            ("Ctrl+W", "Open Molecular Cycle Welder & Peri-Annulation Studio", "Building"),
            ("Ctrl+Alt+P", "1-Click 3D Electrostatic Potential (ESP) Mapped Density Surface", "Surfaces"),
            ("Ctrl+Alt+O", "Molecular Orbitals & Volumetric Isosurfaces Studio", "Surfaces"),
            ("Ctrl+E", "Export high-resolution PNG image", "Export"),
            ("F1", "Open this Keyboard & Mouse Shortcuts dialog", "Help"),
            ("F2", "View and search active QM Calculation Output Log", "Help"),
            ("F3", "View internal Diagnostic Log (agymolden.log)", "Help"),
            ("Shift+F1", "Open complete User Manual (USER_MANUAL.md)", "Help"),
            ("Esc", "Clear active 3D orbital/isosurface or cancel picking", "General"),
        ]

        def _populate(text=""):
            q = text.strip().lower()
            table.setRowCount(0)
            for key, desc, cat in shortcuts_data:
                if not q or (q in key.lower() or q in desc.lower() or q in cat.lower()):
                    row = table.rowCount()
                    table.insertRow(row)
                    item_key = QtWidgets.QTableWidgetItem(key)
                    item_key.setFont(QtGui.QFont("Monospace", 9, QtGui.QFont.Weight.Bold))
                    table.setItem(row, 0, item_key)
                    table.setItem(row, 1, QtWidgets.QTableWidgetItem(desc))
                    table.setItem(row, 2, QtWidgets.QTableWidgetItem(cat))

        _populate()
        filter_edit.textChanged.connect(_populate)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(dlg.reject)
        layout.addWidget(btn_box)
        dlg.exec()

    def _on_view_diagnostic_log(self):
        """Open a dedicated dialog displaying the internal session log file (agymolden.log)."""
        import os
        from . import _log_file_path

        log_path = _log_file_path()
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("AgyMolden - Diagnostic Log")
        dlg.resize(800, 520)
        layout = QtWidgets.QVBoxLayout(dlg)

        header_lbl = QtWidgets.QLabel(f"<b>Session Log File:</b> <code>{log_path}</code>")
        header_lbl.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(header_lbl)

        txt_edit = QtWidgets.QPlainTextEdit()
        txt_edit.setReadOnly(True)
        txt_edit.setFont(QtGui.QFont("Monospace", 9))
        layout.addWidget(txt_edit)

        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    txt_edit.setPlainText(f.read())
                txt_edit.verticalScrollBar().setValue(txt_edit.verticalScrollBar().maximum())
            except Exception as e:
                txt_edit.setPlainText(f"Error reading log file: {e}")
        else:
            txt_edit.setPlainText(f"Log file does not exist yet at:\n{log_path}\n(Logs will appear here once warnings or errors occur).")

        btn_row = QtWidgets.QHBoxLayout()
        btn_refresh = QtWidgets.QPushButton("🔄 Refresh")
        def _refresh():
            if os.path.exists(log_path):
                with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                    txt_edit.setPlainText(f.read())
                txt_edit.verticalScrollBar().setValue(txt_edit.verticalScrollBar().maximum())
        btn_refresh.clicked.connect(_refresh)
        btn_row.addWidget(btn_refresh)

        btn_copy = QtWidgets.QPushButton("📋 Copy Full Log")
        def _copy():
            QtWidgets.QApplication.clipboard().setText(txt_edit.toPlainText())
            QtWidgets.QMessageBox.information(dlg, "Copied", "Log contents copied to clipboard!")
        btn_copy.clicked.connect(_copy)
        btn_row.addWidget(btn_copy)

        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        dlg.exec()

    def _on_open_example_gallery(self):
        """Open the Example Molecule & Calculation Gallery."""
        from .dialog_example_gallery import ExampleGalleryDialog
        dlg = ExampleGalleryDialog(self)
        dlg.exec()

    def _on_open_onboarding_tour(self):
        """Open the interactive quick tour and onboarding guide."""
        from .dialog_onboarding import OnboardingTourDialog
        dlg = OnboardingTourDialog(self)
        dlg.exec()

    def _check_first_run_tour(self):
        """Prompt onboarding tour on first launch if not previously dismissed or disabled."""
        import os
        if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AGYMOLDEN_HEADLESS"):
            return
        settings = QtCore.QSettings("AgyMolden", "Preferences")
        if settings.value("show_onboarding_on_startup", True, type=bool) and not settings.value("has_seen_onboarding_tour", False, type=bool):
            self._on_open_onboarding_tour()

    def _on_open_manual(self):
        """Open USER_MANUAL.md with the system's default viewer."""
        from .resources import find_resource
        path = find_resource("USER_MANUAL.md")
        if path is None:
            QtWidgets.QMessageBox.information(self, "User Manual", "USER_MANUAL.md was not found.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(path))

    def _on_about(self):
        QtWidgets.QMessageBox.about(
            self,
            "About AgyMolden",
            "<h3>AgyMolden</h3>"
            "<p>A modern Linux molecular builder and Z-Matrix editor following the spirit and workflow of <b>Molden</b>.</p>"
            "<p><b>Features:</b></p>"
            "<ul>"
            "<li>Starts with a clean empty workspace</li>"
            "<li><b>Add Line:</b> Molden periodic table element selection</li>"
            "<li><b>Interactive 3D Bind/Angle/Dihedral specification:</b> sequentially click atoms in the viewport to define connectivity</li>"
            "<li><b>Ctrl + Right-Click & Drag:</b> Quick-add connected atom of the last added element</li>"
            "<li>Interactive 3D OpenGL Viewport (Ball & Stick / Spacefill)</li>"
            "<li>Bidirectional editing: 3D pick atom highlights ZMAT line and vice-versa</li>"
            "<li>Live sliders for Bond Length, Bond Angle, and Dihedrals</li>"
            "<li>Shift+Click geometric measurements (Distances, Angles, Dihedrals)</li>"
            "<li>Molden / Gaussian Z-matrix and XYZ Cartesian file import & export</li>"
            "</ul>"
        )

