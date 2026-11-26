"""Menu bar builder mixin for MainWindow (extracted from mw_ui.py for maintainability)."""

from PyQt6 import QtCore, QtGui, QtWidgets


class MenusMixin:
    def _create_menus(self):
        menubar = self.menuBar()

        # File Menu
        menu_file = menubar.addMenu("&File")

        act_new = QtGui.QAction("&New Molecule", self)
        act_new.setShortcut(QtGui.QKeySequence.StandardKey.New)
        act_new.triggered.connect(self._on_new_molecule)
        menu_file.addAction(act_new)

        menu_file.addSeparator()

        act_open_any = QtGui.QAction("📂 &Open File (All Formats: QM Log, XYZ, Molden, ZMAT)...", self)
        act_open_any.setShortcut("Ctrl+O")
        act_open_any.setToolTip("Universal file opener sorted newest-first (.out, .log, .xyz, .molden.input, .molden, .gbw, .cube, .zmat, .inp)")
        act_open_any.triggered.connect(self._on_open_universal_file)
        menu_file.addAction(act_open_any)

        self.menu_recent = menu_file.addMenu("Open &Recent")
        self._update_recent_files_menu()

        menu_open_fmt = menu_file.addMenu("📁 Open &Specific Format / Builder...")

        act_open_zmat = QtGui.QAction("Open &Z-Matrix File...", self)
        act_open_zmat.triggered.connect(self._on_open_zmat)
        menu_open_fmt.addAction(act_open_zmat)

        act_open_xyz = QtGui.QAction("Import &XYZ File...", self)
        act_open_xyz.triggered.connect(self._on_import_xyz)
        menu_open_fmt.addAction(act_open_xyz)

        act_paste_xyz_file = QtGui.QAction("📥 Paste &Cartesian Coordinates from Clipboard...", self)
        act_paste_xyz_file.setToolTip("Parse and load Cartesian XYZ coordinates from clipboard directly into the 3D viewer (Ctrl+Shift+V)")
        act_paste_xyz_file.triggered.connect(self._on_paste_as_new_molecule)
        menu_open_fmt.addAction(act_paste_xyz_file)

        act_open_inp = QtGui.QAction("Open newBRORSEN &Input File (*.inp)...", self)
        act_open_inp.setShortcut("Ctrl+I")
        act_open_inp.setToolTip("Open a newBRORSEN input file (.inp) into AGYMolden and the Input Builder")
        act_open_inp.triggered.connect(self._on_open_brorsen_input_file)
        menu_open_fmt.addAction(act_open_inp)

        menu_open_fmt.addSeparator()

        act_open_smiles = QtGui.QAction("Build from &Name / SMILES / InChI...", self)
        act_open_smiles.setShortcut("Ctrl+Shift+B")
        act_open_smiles.setToolTip("Type a chemical name (e.g. aspirin, caffeine), SMILES (e.g. c1ccccc1), or InChI to build a 3D structure automatically (Ctrl+Shift+B)")
        act_open_smiles.triggered.connect(self._on_prompt_smiles_or_inchi)
        menu_open_fmt.addAction(act_open_smiles)

        act_2d_sketcher = QtGui.QAction("✏️ 2D &Molecular Bond-Line Sketcher (ChemDraw / Marvin Style)...", self)
        act_2d_sketcher.setShortcut("Ctrl+Shift+K")
        act_2d_sketcher.setToolTip("Sketch 2D chemical bond-line formulas (rings, heteroatoms, multiple bonds) and convert automatically to 3D (Ctrl+Shift+K)")
        act_2d_sketcher.triggered.connect(self._on_open_2d_sketcher)
        menu_open_fmt.addAction(act_2d_sketcher)

        act_gallery = QtGui.QAction("📚 Chemistry &Sample & Benchmark Gallery...", self)
        act_gallery.setToolTip("Open curated benchmark structures (Ferrocene, C60, Diels-Alder TS, Water Prism) in 1 click")
        act_gallery.triggered.connect(self._on_open_sample_gallery)
        menu_open_fmt.addAction(act_gallery)

        menu_open_fmt.addSeparator()

        act_open_qm_log = QtGui.QAction("Import &QM Log / Output File (newBRORSEN, Gaussian, ORCA)...", self)
        act_open_qm_log.setToolTip("Open output file from newBRORSEN (.out), Gaussian, ORCA, Q-Chem, or xTB to view structures, optimization steps and frequencies")
        act_open_qm_log.triggered.connect(self._on_import_qm_log)
        menu_open_fmt.addAction(act_open_qm_log)

        act_open_cube = QtGui.QAction("Open Volumetric &Grid / Orbitals (.cube, .molden)...", self)
        act_open_cube.setToolTip("Open Gaussian .cube file or Molden file to extract and view 3D Molecular Orbitals (HOMO/LUMO) and ESP isosurfaces")
        act_open_cube.triggered.connect(self._on_open_volumetric_dialog)
        menu_open_fmt.addAction(act_open_cube)

        act_orb_gallery = QtGui.QAction("📸 All Molecular Orbitals Snapshot &Gallery...", self)
        act_orb_gallery.setToolTip("Batch capture all or selected molecular orbitals into a contact sheet collage or folder of images (like Gabedit)")
        act_orb_gallery.triggered.connect(self._on_open_orbital_snapshot_gallery)
        menu_open_fmt.addAction(act_orb_gallery)

        act_view_out = QtGui.QAction("📄 View &Calculation Output Log (*.out, *.log)...", self)
        act_view_out.setToolTip("Inspect full output log with clickable section indexing and search (F2)")
        act_view_out.setShortcut("F2")
        act_view_out.triggered.connect(self._on_view_qm_output_file)
        menu_file.addAction(act_view_out)

        menu_file.addSeparator()

        act_save_proj = QtGui.QAction("Save Project Bundle (.agyproj)...", self)
        act_save_proj.setToolTip("Save full molecular model, Z-matrix, calculation history, and view settings into a portable project bundle")
        act_save_proj.triggered.connect(self._on_save_project_bundle)
        menu_file.addAction(act_save_proj)

        act_save_zmat = QtGui.QAction("Save Z-Matrix (&ZMAT)...", self)
        act_save_zmat.setShortcut("Ctrl+S")
        act_save_zmat.triggered.connect(self._on_save_zmat)
        menu_file.addAction(act_save_zmat)

        menu_save_zmat_fmt = menu_file.addMenu("💾 Save &Z-Matrix Format...")

        act_save_num = QtGui.QAction("Save Z-Matrix with &Numbers...", self)
        act_save_num.setToolTip("Every bond, angle and dihedral written as a number")
        act_save_num.triggered.connect(lambda: self._on_save_zmat_format("numeric"))
        menu_save_zmat_fmt.addAction(act_save_num)

        act_save_var = QtGui.QAction("Save Z-Matrix with &Variables...", self)
        act_save_var.setToolTip("Every coordinate written as a variable, with a Variables: block of values (shared/symmetric variables are preserved)")
        act_save_var.triggered.connect(lambda: self._on_save_zmat_format("variables"))
        menu_save_zmat_fmt.addAction(act_save_var)

        act_save_xyz = QtGui.QAction("Export Coordinates (&XYZ)...", self)
        act_save_xyz.triggered.connect(self._on_export_xyz)
        menu_file.addAction(act_save_xyz)

        menu_export = menu_file.addMenu("📤 &Export As...")

        act_save_pdb = QtGui.QAction("Export Protein Data Bank (&PDB)...", self)
        act_save_pdb.triggered.connect(self._on_export_pdb)
        menu_export.addAction(act_save_pdb)

        act_save_sdf = QtGui.QAction("Export Structure-Data File (&SDF / MOL)...", self)
        act_save_sdf.triggered.connect(self._on_export_sdf)
        menu_export.addAction(act_save_sdf)

        act_save_mol2 = QtGui.QAction("Export Tripos (&Mol2)...", self)
        act_save_mol2.setToolTip("Export molecular coordinates and bonds to Tripos Mol2 format")
        act_save_mol2.triggered.connect(self._on_export_mol2)
        menu_export.addAction(act_save_mol2)

        act_save_cif = QtGui.QAction("Export Crystallographic File (&CIF)...", self)
        act_save_cif.setToolTip("Export molecular coordinates and unit cell parameters to standard CIF syntax")
        act_save_cif.triggered.connect(self._on_export_cif)
        menu_export.addAction(act_save_cif)

        menu_export.addSeparator()

        act_export_svg = QtGui.QAction("📐 Export &Scalable Vector Graphics (SVG)...", self)
        act_export_svg.setToolTip("Export infinite-resolution 3D vector graphics for Illustrator, Inkscape, or LaTeX")
        act_export_svg.triggered.connect(self._on_export_svg)
        menu_export.addAction(act_export_svg)

        act_export_html = QtGui.QAction("🌐 Export Interactive 3D &HTML...", self)
        act_export_html.setShortcut("Ctrl+Shift+H")
        act_export_html.setToolTip("Export molecule, properties, and trajectory to a standalone, zero-install 3D HTML webpage")
        act_export_html.triggered.connect(self._on_export_html)
        menu_export.addAction(act_export_html)

        act_export_bundle = QtGui.QAction("📦 Export Reproducibility Bundle (&ZIP)...", self)
        act_export_bundle.setShortcut("Ctrl+Shift+S")
        act_export_bundle.setToolTip("Export complete Supporting Information (SI) package: XYZ, ZMAT, Summary JSON, Methods Markdown, and calculation logs (Ctrl+Shift+S)")
        act_export_bundle.triggered.connect(self._on_export_reproducibility_bundle)
        menu_export.addAction(act_export_bundle)

        menu_export.addSeparator()
        act_open_agemacs = QtGui.QAction("🚀 Open / Export to &agemacs...", self)
        act_open_agemacs.setToolTip("Open the current molecule or file in agemacs (Alt+Shift+A)")
        act_open_agemacs.triggered.connect(self._on_open_in_agemacs)
        menu_export.addAction(act_open_agemacs)

        menu_agemacs_file = menu_file.addMenu("🚀 Send / Export to &Agemacs")
        act_ag_open = QtGui.QAction("📄 Open Active File / Geometry in Agemacs", self)
        act_ag_open.setShortcut("Alt+Shift+A")
        act_ag_open.setToolTip("Open the current file or exported geometry in Agemacs (Alt+Shift+A)")
        act_ag_open.triggered.connect(self._on_open_in_agemacs)
        menu_agemacs_file.addAction(act_ag_open)
        menu_agemacs_file.addAction("📐 Export Cartesian XYZ to Agemacs", self._on_send_xyz_to_agemacs)
        menu_agemacs_file.addAction("📋 Export Z-Matrix to Agemacs", self._on_send_zmat_to_agemacs)
        menu_agemacs_file.addAction("📊 Open Active QM Log / Molden in Agemacs", self._on_send_qm_output_to_agemacs)
        menu_agemacs_file.addSeparator()
        menu_agemacs_file.addAction("📂 Browse & Open Any File in Agemacs...", self._on_send_any_file_to_agemacs)

        menu_file.addSeparator()

        act_export_png = QtGui.QAction("Export &Image (PNG)...", self)
        act_export_png.setShortcuts([
            QtGui.QKeySequence("Ctrl+E"),
            QtGui.QKeySequence("Ctrl+P"),
        ])
        act_export_png.setToolTip("Save a high-resolution PNG image of the 3D viewport (Ctrl+E)")
        act_export_png.triggered.connect(self._on_export_image)
        menu_export.addAction(act_export_png)

        act_clip_img = QtGui.QAction("📷 Copy Viewport Image to Clipboard", self)
        act_clip_img.setShortcut("Ctrl+Shift+C")
        act_clip_img.setToolTip("Copy publication-quality 3D image with transparent background directly to system clipboard (Ctrl+Shift+C)")
        act_clip_img.triggered.connect(lambda: self.copy_image_to_clipboard(transparent=True, scale=2, clean_only=True))
        menu_export.addAction(act_clip_img)

        act_prefs = QtGui.QAction("⚙️ &Preferences...", self)
        act_prefs.setShortcut("Ctrl+,")
        act_prefs.setToolTip("Configure default colors, background mode, hardware resources, and symmetry defaults")
        act_prefs.triggered.connect(self._on_open_preferences)
        menu_file.addAction(act_prefs)

        menu_file.addSeparator()

        act_exit = QtGui.QAction("E&xit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        menu_file.addAction(act_exit)

        # Edit Menu
        menu_edit = menubar.addMenu("&Edit")

        self.act_undo = QtGui.QAction("↶ &Undo", self)
        self.act_undo.setShortcut(QtGui.QKeySequence("Ctrl+Z"))
        self.act_undo.setToolTip("Undo last action (Ctrl+Z)")
        self.act_undo.setEnabled(False)
        self.act_undo.triggered.connect(self.undo)
        menu_edit.addAction(self.act_undo)

        self.act_redo = QtGui.QAction("↷ &Redo", self)
        self.act_redo.setShortcuts([
            QtGui.QKeySequence("Ctrl+Y"),
            QtGui.QKeySequence("Ctrl+Shift+Z")
        ])
        self.act_redo.setToolTip("Redo last undone action (Ctrl+Y)")
        self.act_redo.setEnabled(False)
        self.act_redo.triggered.connect(self.redo)
        menu_edit.addAction(self.act_redo)

        menu_edit.addSeparator()

        self.act_copy_mol = QtGui.QAction("📋 &Copy Molecule / Selection", self)
        self.act_copy_mol.setShortcut(QtGui.QKeySequence("Ctrl+C"))
        self.act_copy_mol.setToolTip("Copy selected atoms (or entire molecule if none selected) to clipboard / memory (Ctrl+C)")
        self.act_copy_mol.triggered.connect(self._copy_selected_fragment)
        menu_edit.addAction(self.act_copy_mol)

        self.act_paste_mol = QtGui.QAction("📋 &Paste Molecule / Fragment", self)
        self.act_paste_mol.setShortcut(QtGui.QKeySequence("Ctrl+V"))
        self.act_paste_mol.setToolTip("Paste copied fragment, Cartesian XYZ coordinates, or SMILES/InChI into the 3D viewport (Ctrl+V)")
        self.act_paste_mol.triggered.connect(self._on_paste)
        menu_edit.addAction(self.act_paste_mol)

        act_paste_xyz = QtGui.QAction("📥 Paste &XYZ Coordinates as Molecule", self)
        act_paste_xyz.setShortcut(QtGui.QKeySequence("Ctrl+Shift+V"))
        act_paste_xyz.setToolTip("Directly load XYZ coordinates from clipboard as the active molecule (replaces or resets canvas) (Ctrl+Shift+V)")
        act_paste_xyz.triggered.connect(self._on_paste_as_new_molecule)
        menu_edit.addAction(act_paste_xyz)

        menu_edit.addSeparator()

        # Copy Identifiers submenu
        menu_copy = menu_edit.addMenu("📋 Copy &Identifier...")
        act_cp_xyz = QtGui.QAction("Copy as &Cartesian XYZ (with header)", self)
        act_cp_xyz.setShortcut("Ctrl+Alt+C")
        act_cp_xyz.setToolTip("Copy molecule in standard Cartesian XYZ format to clipboard (Ctrl+Alt+C)")
        act_cp_xyz.triggered.connect(self._on_copy_as_xyz)
        menu_copy.addAction(act_cp_xyz)

        act_cp_block = QtGui.QAction("Copy &Coordinates Block (No Header — ORCA/Gaussian)", self)
        act_cp_block.setShortcut("Ctrl+Alt+X")
        act_cp_block.setToolTip("Copy headerless 'Element X Y Z' coordinates block ready to paste directly into ORCA or Gaussian input files (Ctrl+Alt+X)")
        act_cp_block.triggered.connect(self._on_copy_coords_block)
        menu_copy.addAction(act_cp_block)

        act_cp_smiles = QtGui.QAction("Copy as &SMILES", self)
        act_cp_smiles.setToolTip("Generate and copy canonical/isomeric SMILES string to clipboard")
        act_cp_smiles.triggered.connect(self._on_copy_as_smiles)
        menu_copy.addAction(act_cp_smiles)

        act_cp_inchi = QtGui.QAction("Copy as &InChI", self)
        act_cp_inchi.setToolTip("Generate and copy IUPAC InChI string to clipboard")
        act_cp_inchi.triggered.connect(self._on_copy_as_inchi)
        menu_copy.addAction(act_cp_inchi)

        act_cp_inchikey = QtGui.QAction("Copy as InChI&Key", self)
        act_cp_inchikey.setToolTip("Generate and copy 27-character InChIKey to clipboard")
        act_cp_inchikey.triggered.connect(self._on_copy_as_inchikey)
        menu_copy.addAction(act_cp_inchikey)

        menu_copy.addSeparator()

        act_cp_si_acs = QtGui.QAction("📑 Copy &Supporting Information (SI — ACS/RSC Block)", self)
        act_cp_si_acs.setShortcut("Ctrl+Alt+S")
        act_cp_si_acs.setToolTip("Copy publication-ready Supporting Information block (Formula, Point Group, Energies, N_imag, Coordinates) in ACS/RSC format (Ctrl+Alt+S)")
        act_cp_si_acs.triggered.connect(lambda: self._on_copy_supporting_info("acs"))
        menu_copy.addAction(act_cp_si_acs)

        act_cp_si_ltx = QtGui.QAction("📑 Copy Supporting Information (&LaTeX Table)", self)
        act_cp_si_ltx.setToolTip("Copy publication-ready Supporting Information table in LaTeX format")
        act_cp_si_ltx.triggered.connect(lambda: self._on_copy_supporting_info("latex"))
        menu_copy.addAction(act_cp_si_ltx)

        act_cp_si_md = QtGui.QAction("📑 Copy Supporting Information (&Markdown Block)", self)
        act_cp_si_md.setToolTip("Copy publication-ready Supporting Information entry in Markdown format")
        act_cp_si_md.triggered.connect(lambda: self._on_copy_supporting_info("markdown"))
        menu_copy.addAction(act_cp_si_md)

        menu_edit.addSeparator()

        menu_edit_struct = menu_edit.addMenu("🧱 &Atoms && Z-Matrix Structure...")

        act_add_line = QtGui.QAction("&Add Atom Line", self)
        act_add_line.triggered.connect(self._on_add_line)
        menu_edit_struct.addAction(act_add_line)

        act_del = QtGui.QAction("&Delete Selected Atom", self)
        act_del.triggered.connect(self._on_delete_selected_atom)
        menu_edit_struct.addAction(act_del)

        act_reorder = QtGui.QAction("&Reorder Z-Matrix (Full)", self)
        act_reorder.setShortcut("Ctrl+Shift+R")
        act_reorder.setToolTip("Reorder entire Z-Matrix sequentially (Molden ZMEORD) (Ctrl+Shift+R)")
        act_reorder.triggered.connect(self._on_reorder_zmat_clicked)
        menu_edit_struct.addAction(act_reorder)

        act_adj_b = QtGui.QAction("Adjust &Bond (Alt+B)", self)
        act_adj_b.setShortcut("Alt+B")
        act_adj_b.setToolTip("Click 2 atoms to redefine bond partner of the 1st atom (Alt+B)")
        act_adj_b.triggered.connect(lambda: self._start_partial_reorder("bond"))
        menu_edit_struct.addAction(act_adj_b)

        act_adj_a = QtGui.QAction("Adjust &Angle (Alt+A)", self)
        act_adj_a.setShortcut("Alt+A")
        act_adj_a.setToolTip("Click 3 atoms to redefine bond & angle partners of the 1st atom (Alt+A)")
        act_adj_a.triggered.connect(lambda: self._start_partial_reorder("angle"))
        menu_edit_struct.addAction(act_adj_a)

        act_adj_d = QtGui.QAction("Adjust &Dihedral (Alt+D)", self)
        act_adj_d.setShortcut("Alt+D")
        act_adj_d.setToolTip("Click 4 atoms to redefine bond, angle & dihedral partners of the 1st atom (Alt+D)")
        act_adj_d.triggered.connect(lambda: self._start_partial_reorder("dihedral"))
        menu_edit_struct.addAction(act_adj_d)

        menu_edit_struct.addSeparator()

        act_add_h = QtGui.QAction("&Complete Valence with Hydrogens", self)
        act_add_h.setShortcut("Ctrl+H")
        act_add_h.triggered.connect(self._on_add_hydrogens)
        menu_edit_struct.addAction(act_add_h)

        act_cycle_welder = QtGui.QAction("💍 &Weld / Fuse Molecular Rings...", self)
        act_cycle_welder.setShortcut("Ctrl+Shift+W")
        act_cycle_welder.setToolTip("Fuse aromatic or saturated rings onto selected bonds, or create bicyclic bridgeheads (Ctrl+Shift+W)")
        act_cycle_welder.triggered.connect(self._on_open_cycle_welder)
        menu_edit_struct.addAction(act_cycle_welder)

        menu_origin = menu_edit.addMenu("🎯 &Translate to Origin (0, 0, 0)...")
        act_orig_com = QtGui.QAction("Center of Mass to Origin", self)
        act_orig_com.setToolTip("Translate molecule so its center-of-mass is at Cartesian (0, 0, 0)")
        act_orig_com.triggered.connect(lambda: self._center_molecule_to_origin("com"))
        menu_origin.addAction(act_orig_com)

        act_orig_anchor = QtGui.QAction("Selected Atom to Origin", self)
        act_orig_anchor.setToolTip("Translate molecule so the currently selected atom is at (0, 0, 0)")
        act_orig_anchor.triggered.connect(lambda: self._center_molecule_to_origin("anchor"))
        menu_origin.addAction(act_orig_anchor)

        act_orient_sym = QtGui.QAction("📐 Orient to &Standard Symmetry Orientation (All Groups)", self)
        act_orient_sym.setShortcut(QtGui.QKeySequence("Alt+S"))
        act_orient_sym.setToolTip(
            "Rigidly rotate and center molecule to standard IUPAC / Schoenflies orientation (Alt+S).\n"
            "Aligns principal axis (highest Cn) along Z, mirror plane in XZ/XY, and centers COM at (0,0,0)."
        )
        act_orient_sym.triggered.connect(self._on_align_symmetry)
        menu_edit.addAction(act_orient_sym)

        menu_invert = menu_edit.addMenu("🔄 &Chirality & Mirror Inversion...")
        act_inv_all = QtGui.QAction("Invert All Coordinates (Enantiomer, -r)", self)
        act_inv_all.setToolTip("Invert all atomic coordinates through the center of inversion (x,y,z ➔ -x,-y,-z)")
        act_inv_all.triggered.connect(lambda: self._invert_molecule_coordinates("inversion"))
        menu_invert.addAction(act_inv_all)

        act_mir_xy = QtGui.QAction("Mirror Across XY Plane (z ➔ -z)", self)
        act_mir_xy.triggered.connect(lambda: self._invert_molecule_coordinates("xy"))
        menu_invert.addAction(act_mir_xy)

        act_mir_xz = QtGui.QAction("Mirror Across XZ Plane (y ➔ -y)", self)
        act_mir_xz.triggered.connect(lambda: self._invert_molecule_coordinates("xz"))
        menu_invert.addAction(act_mir_xz)

        act_mir_yz = QtGui.QAction("Mirror Across YZ Plane (x ➔ -x)", self)
        act_mir_yz.triggered.connect(lambda: self._invert_molecule_coordinates("yz"))
        menu_invert.addAction(act_mir_yz)

        menu_edit.addSeparator()

        menu_edit_model = menu_edit.addMenu("⚡ &Geometry && Modeling...")

        act_opt = QtGui.QAction("&Optimize Geometry (Force Field)", self)
        act_opt.setShortcuts([
            QtGui.QKeySequence("Ctrl+M"),
            QtGui.QKeySequence("Ctrl+Shift+O"),
        ])
        act_opt.setToolTip("Optimize molecular geometry using force field / xTB (Ctrl+M or Ctrl+Shift+O)")
        act_opt.triggered.connect(self._on_optimize_geometry)
        menu_edit_model.addAction(act_opt)

        self.act_tweezers = QtGui.QAction("🧲 Interactive Molecular Tweezers (Live UFF Drag)", self)
        self.act_tweezers.setCheckable(True)
        self.act_tweezers.setChecked(False)
        self.act_tweezers.setToolTip("When checked, simply Left-Click and drag any atom in the 3D viewport to dynamically relax connected atoms in real-time via UFF force field")
        self.act_tweezers.toggled.connect(self._on_toggle_tweezers)
        if hasattr(self, "btn_ribbon_tweezers"):
            self.act_tweezers.toggled.connect(self.btn_ribbon_tweezers.setChecked)
            self.btn_ribbon_tweezers.toggled.connect(self.act_tweezers.setChecked)
        menu_edit_model.addAction(self.act_tweezers)

        act_conf = QtGui.QAction("🔄 &Conformational Search (Find Lowest-Energy Rotamer)...", self)
        act_conf.setToolTip("Sample rotatable bonds, minimize candidate rotamers, and rank conformers by energy")
        act_conf.triggered.connect(self._on_conformer_search)
        menu_edit_model.addAction(act_conf)

        act_vib = QtGui.QAction("🎵 &Vibrational Frequencies (Normal Modes)", self)
        act_vib.setToolTip("Compute vibrational frequencies and normal modes simulation")
        act_vib.triggered.connect(self._on_calculate_vibrations)
        menu_edit_model.addAction(act_vib)

        act_sym = QtGui.QAction("🔮 &Detect Point Group Symmetry", self)
        act_sym.setToolTip("Analyze molecular point group symmetry (Schoenflies notation)")
        act_sym.triggered.connect(lambda _checked=False: self._on_detect_symmetry(switch_tab=True))
        menu_edit_model.addAction(act_sym)

        act_flush_mem = QtGui.QAction("Flush &Memory Clipboard", self)
        act_flush_mem.setShortcut(QtGui.QKeySequence("Ctrl+Shift+X"))
        act_flush_mem.setToolTip("Flush atom and fragment clipboard memory (Ctrl+Shift+X). Double-clicking canvas will prompt Periodic Table / Fragment picker.")
        act_flush_mem.triggered.connect(self._on_flush_memory)
        menu_edit.addAction(act_flush_mem)

        # Variables Menu
        menu_vars = menubar.addMenu("&Variables")

        act_sym_bonds = QtGui.QAction("🔗 Assign &Symmetric Bond Variables Only (Keep Angles/Dihedrals Fixed)", self)
        act_sym_bonds.setToolTip("Equivalent bond distances (e.g. C-C and C-H in benzene) share variables (R1, R2) "
                                 "while angles and dihedrals remain fixed numerical constants")
        act_sym_bonds.triggered.connect(self._on_assign_symmetric_bond_variables)
        menu_vars.addAction(act_sym_bonds)

        act_sym_vars = QtGui.QAction("🔗 Assign &Symmetric Variables (All: Bonds, Angles, Dihedrals)", self)
        act_sym_vars.setToolTip("Equivalent coordinates (e.g. all C-C and all C-H bonds in benzene) share one "
                                "variable, reducing the number of optimization parameters")
        act_sym_vars.triggered.connect(self._on_assign_symmetric_variables)
        menu_vars.addAction(act_sym_vars)

        act_all_vars = QtGui.QAction("🔤 Assign &Individual Variables", self)
        act_all_vars.setToolTip("Assign an individual variable name (R1..Rn, A1..An, D1..Dn) to every internal coordinate")
        act_all_vars.triggered.connect(self._on_assign_individual_variables)
        menu_vars.addAction(act_all_vars)

        menu_vars.addSeparator()

        act_var_save = QtGui.QAction("💾 Save Z-Matrix with &Variables...", self)
        act_var_save.setToolTip("Save Z-Matrix with symbolic variable names and a 'Variables:' block of values")
        act_var_save.triggered.connect(lambda: self._on_save_zmat_format("variables"))
        menu_vars.addAction(act_var_save)

        act_clear_vars = QtGui.QAction("🔢 &Clear Variables (Use Numbers Only)", self)
        act_clear_vars.setToolTip("Clear all symbolic variables and restore explicit numeric values")
        act_clear_vars.triggered.connect(self._on_clear_variables)
        menu_vars.addAction(act_clear_vars)

        # Tools Menu
        menu_tools = menubar.addMenu("&Tools")

        act_hub = QtGui.QAction("⚡ &Chemistry Workbench Hub & Workflow Dashboard...", self)
        act_hub.setShortcut("Ctrl+W")
        act_hub.setToolTip("Open unified visual dashboard of all 40+ specialized tools organized by workflow (Ctrl+W)")
        act_hub.triggered.connect(self._on_open_workbench_hub)
        menu_tools.addAction(act_hub)

        menu_tools.addSeparator()

        act_unit_conv = QtGui.QAction("⚡ Quantum &Unit Converter (Energy, Length, Dipole, Polarizability)...", self)
        act_unit_conv.setShortcut("Ctrl+U")
        act_unit_conv.setToolTip("Instant physical constant conversions: Hartree ↔ eV ↔ kcal/mol ↔ kJ/mol ↔ cm⁻¹ ↔ K; Bohr ↔ Å; Debye ↔ a.u.; Polarizability; Field & Pressure (Ctrl+U)")
        act_unit_conv.triggered.connect(self._open_unit_converter)
        menu_tools.addAction(act_unit_conv)

        act_summary_card = QtGui.QAction("📊 &Calculation Summary Card...", self)
        act_summary_card.setShortcut("F4")
        act_summary_card.setToolTip("Open publication-ready 1-page summary card of current QM calculation results (F4)")
        act_summary_card.triggered.connect(self._on_open_summary_card)
        menu_tools.addAction(act_summary_card)

        menu_tools.addSeparator()

        menu_tools_qm = menu_tools.addMenu("🚀 &Quantum Chemistry && Jobs...")
        menu_tools_struct = menu_tools.addMenu("🧬 &Structure, Geometry && Sterics...")
        menu_tools_thermo = menu_tools.addMenu("🌡️ &Thermochemistry && Kinetics...")
        menu_tools_elec = menu_tools.addMenu("⚛️ &Electronic Structure && Spectroscopy...")
        menu_tools_pub = menu_tools.addMenu("📄 &Publication && Benchmarking...")

        act_quick_orca = QtGui.QAction("🚀 Quick Run &ORCA Calculation...", self)
        act_quick_orca.setShortcut("Ctrl+R")
        act_quick_orca.setToolTip("Quick 1-Click setup and execution of ORCA calculations for coursework")
        act_quick_orca.triggered.connect(self._on_quick_run_orca)
        menu_tools_qm.addAction(act_quick_orca)

        act_generate_qm_input = QtGui.QAction("⚡ Generate &QM Calculation Input...", self)
        act_generate_qm_input.setShortcut("Ctrl+G")
        act_generate_qm_input.setToolTip("Set up and generate input files for Gaussian, ORCA, PySCF, openLOWDIN, or xTB")
        act_generate_qm_input.triggered.connect(self._on_open_qm_dialog)
        menu_tools_qm.addAction(act_generate_qm_input)

        act_generate_brorsen_input = QtGui.QAction("✨ new&BRORSEN Input Builder (Positrons, NEO)...", self)
        act_generate_brorsen_input.setShortcut("Ctrl+B")
        act_generate_brorsen_input.setToolTip("Set up multi-stage input files for newBRORSEN (Positrons, NEO, and CCSD(T))")
        act_generate_brorsen_input.triggered.connect(self._on_open_brorsen_dialog)
        menu_tools_qm.addAction(act_generate_brorsen_input)

        menu_tools_qm.addSeparator()

        act_job_monitor = QtGui.QAction("🖥️ Calculation &Job Monitor...", self)
        act_job_monitor.setShortcut("Ctrl+J")
        act_job_monitor.setToolTip("View live calculation progress, streaming output logs, and real-time 3D updates")
        act_job_monitor.triggered.connect(self._on_open_job_monitor)
        menu_tools_qm.addAction(act_job_monitor)

        act_workflow_dag = QtGui.QAction("🔀 Visual &Workflow DAG Pipeline Editor (Multi-Stage QM)...", self)
        act_workflow_dag.setShortcut("Ctrl+Shift+D")
        act_workflow_dag.setToolTip("Design and execute chained multi-stage QM workflows (Opt → Freq → TD-DFT / TS → IRC / SP) on an interactive visual DAG canvas (Ctrl+Shift+D)")
        act_workflow_dag.triggered.connect(self._on_open_workflow_dag)
        menu_tools_qm.addAction(act_workflow_dag)

        act_remote_transfer = QtGui.QAction("📁 Remote &File Transfer (SSH / Cluster)...", self)
        act_remote_transfer.setToolTip("Browse and transfer input and output files between local machine and remote cluster")
        act_remote_transfer.triggered.connect(self._on_open_remote_file_manager)
        menu_tools_qm.addAction(act_remote_transfer)

        act_qm_engines = QtGui.QAction("🔧 Configure &QM Engines & Routes...", self)
        act_qm_engines.setToolTip("Detect installed QM packages (ORCA, PySCF, Gaussian, openLOWDIN, xTB, Q-Chem) and configure custom executable routes")
        act_qm_engines.triggered.connect(self._on_configure_qm_engines)
        menu_tools_qm.addAction(act_qm_engines)

        act_dihed_scan = QtGui.QAction("🔄 1D &Dihedral Rotor Scan / PES Drive...", self)
        act_dihed_scan.setToolTip("Step selected 4-atom dihedral through 360° with real-time forcefield relaxation (conformer search)")
        act_dihed_scan.triggered.connect(self._on_open_dihedral_scan_dialog)
        menu_tools_struct.addAction(act_dihed_scan)

        act_clash = QtGui.QAction("⚠️ Highlight &Steric Clashes...", self)
        act_clash.setToolTip("Scan molecular geometry and highlight severe non-bonded steric overlaps (< 70% Σr_vdw)")
        act_clash.triggered.connect(self._check_and_highlight_steric_clashes)
        menu_tools_struct.addAction(act_clash)

        act_steric_desc = QtGui.QAction("🛡️ Catalysis &Steric && Coordination Descriptors (%V_bur, Sterimol, τ₄/τ₅)...", self)
        act_steric_desc.setToolTip("Compute SambVca / SEQCROW Percent Buried Volume (%V_bur), Verloop Sterimol (L, B1, B5), Tolman Cone Angle, and Coordination Geometry Indices (τ₄, τ₄', τ₅, Σ)")
        act_steric_desc.triggered.connect(self._on_open_steric_descriptors_dialog)
        menu_tools_struct.addAction(act_steric_desc)

        act_ring_homa = QtGui.QAction("💍 &Ring Aromaticity (HOMA Index) && Planarity Inspector...", self)
        act_ring_homa.setToolTip("Compute Kruszewski–Krygowski HOMA aromaticity index, Hückel π-electron count, and ring planarity RMSD for all rings")
        act_ring_homa.triggered.connect(self._on_open_ring_aromaticity_dialog)
        menu_tools_struct.addAction(act_ring_homa)

        act_iso_kie = QtGui.QAction("⚗️ &Isotopologue Vibrational Shift && KIE Calculator...", self)
        act_iso_kie.setToolTip("Substitute isotopes (H→D, ¹²C→¹³C, ¹⁶O→¹⁸O) and compute frequency shifts Δν, ΔZPE, and Kinetic Isotope Effects (k_H/k_D)")
        act_iso_kie.triggered.connect(self._on_open_isotopologue_dialog)
        menu_tools_thermo.addAction(act_iso_kie)

        act_pbc = QtGui.QAction("📦 &Periodic Boundary Conditions (PBC) && Supercell Generator...", self)
        act_pbc.setToolTip("View/edit crystallographic unit cell parameters (a, b, c, α, β, γ), fractional coordinates, and replicate Nx×Ny×Nz supercells")
        act_pbc.triggered.connect(self._on_open_pbc_dialog)
        menu_tools_struct.addAction(act_pbc)

        act_quick_pyscf = QtGui.QAction("⚡ Quick In-Memory DFT / HF (PySCF)...", self)
        act_quick_pyscf.setToolTip("Solve B3LYP / def2-SVP directly in RAM within seconds (no external binaries or files required)")
        act_quick_pyscf.triggered.connect(lambda: self._on_quick_in_memory_pyscf_dft())
        menu_tools_qm.addAction(act_quick_pyscf)

        act_overlay = QtGui.QAction("🔬 &Compare & Align Structures (RMSD Overlay)...", self)
        act_overlay.setShortcut("Ctrl+Shift+A")
        act_overlay.setToolTip("Superimpose two structures with Kabsch SVD alignment and calculate all-atom/heavy-atom RMSD")
        act_overlay.triggered.connect(self._on_open_structure_overlay)
        menu_tools_struct.addAction(act_overlay)

        act_fmo_align = QtGui.QAction("⚡ &Frontier Orbital (FMO) Alignment & Reactivity Matcher...", self)
        act_fmo_align.setToolTip("Compare Donor-Acceptor HOMO/LUMO levels side-by-side with chemical hardness (η) and electrophilicity (ω)")
        act_fmo_align.triggered.connect(self._on_open_fmo_alignment)
        menu_tools_elec.addAction(act_fmo_align)

        act_cdft = QtGui.QAction("⚛️ &Conceptual DFT (CDFT) & Fukui Reactivity Workbench...", self)
        act_cdft.setShortcut("Ctrl+Shift+F")
        act_cdft.setToolTip("Evaluate global CDFT indices (μ, η, ω, N_Nu) and atomic Fukui functions (f+, f-, Δf) with 3D color mapping (Ctrl+Shift+F)")
        act_cdft.triggered.connect(self._on_open_cdft_workbench)
        menu_tools_elec.addAction(act_cdft)

        act_benchmark = QtGui.QAction("⚖️ Cross-Engine &Benchmark & Result Comparison...", self)
        act_benchmark.setShortcut("Ctrl+Shift+M")
        act_benchmark.setToolTip("Compare multiple calculation logs side-by-side across Gaussian, ORCA, PySCF, and xTB (energetics, ΔE, frontier gap, and Kabsch SVD RMSD matrix) (Ctrl+Shift+M)")
        act_benchmark.triggered.connect(self._on_open_benchmark_dialog)
        menu_tools_pub.addAction(act_benchmark)

        act_thermo = QtGui.QAction("⚗️ Reaction &Thermochemistry && Eyring Kinetics (ΔG‡, ΔGrxn, Keq)...", self)
        act_thermo.setShortcut("Alt+Shift+R")
        act_thermo.setToolTip("Compute reaction free energies (ΔGrxn), activation barriers (ΔG‡), equilibrium constant (Keq), and Eyring rate constants k(T) from QM logs (Alt+Shift+R)")
        act_thermo.triggered.connect(self._on_open_reaction_thermo_dialog)
        menu_tools_thermo.addAction(act_thermo)

        act_solvation = QtGui.QAction("💧 &Continuum Solvation Free Energy (SMD / PCM / CPCM)...", self)
        act_solvation.setToolTip("Estimate solvent effects, dielectric response (ΔG_el), and cavitation/dispersion (ΔG_cds) across 11 standard solvents")
        act_solvation.triggered.connect(self._on_open_solvation_dialog)
        menu_tools_thermo.addAction(act_solvation)

        act_sterics = QtGui.QAction("🛡️ &Catalysis Sterics: Buried Volume (%V_bur) & Sterimol (L, B1, B5)...", self)
        act_sterics.setToolTip("Calculate ligand buried volume (%V_bur), SambVca-style 2D topographic steric altitude maps, and Verloop Sterimol descriptors (L, B1, B5)")
        act_sterics.triggered.connect(self._on_open_sterics_dialog)
        menu_tools_struct.addAction(act_sterics)

        act_marcus = QtGui.QAction("⚡ &Marcus Electron Transfer & Photoredox Kinetics...", self)
        act_marcus.setToolTip("Evaluate inner/outer-sphere reorganization energies (λ_in, λ_out), Marcus activation barrier (ΔG‡), non-adiabatic rate k_ET, and diabatic PES parabolas")
        act_marcus.triggered.connect(self._on_open_marcus_dialog)
        menu_tools_thermo.addAction(act_marcus)

        act_spin = QtGui.QAction("⚛️ Open-Shell &Spin Diagnostics & Yamaguchi Projection (<S²>, J)...", self)
        act_spin.setToolTip("Diagnose <S²> spin contamination, compute Yamaguchi approximate spin-projected energies, and Heisenberg exchange coupling (J)")
        act_spin.triggered.connect(self._on_open_spin_diagnostics)
        menu_tools_elec.addAction(act_spin)

        act_redox = QtGui.QAction("⚡ Electrochemical &Redox Potentials & Born-Haber Cycle (E° vs SHE/SCE)...", self)
        act_redox.setToolTip("Calculate standard reduction/oxidation potentials (E° vs. SHE, SCE, Ag/AgCl, Fc/Fc⁺) via thermodynamic Born-Haber solvation cycles")
        act_redox.triggered.connect(self._on_open_redox_dialog)
        menu_tools_thermo.addAction(act_redox)

        act_mecp = QtGui.QAction("🔀 &Minimum Energy Crossing Point (MECP) & Spin-Inversion Explorer...", self)
        act_mecp.setToolTip("Interpolate crossing seams (ΔE → 0) between two spin states in two-state reactivity and export ORCA SurfCross MECP search decks")
        act_mecp.triggered.connect(self._on_open_mecp_dialog)
        menu_tools_thermo.addAction(act_mecp)

        act_eda = QtGui.QAction("🧩 &Energy Decomposition Analysis (EDA / SAPT / ALMO)...", self)
        act_eda.setToolTip("Decompose interaction energies into Electrostatics, Pauli Repulsion, Orbital/Charge-Transfer, and Dispersion terms from Q-Chem ALMO-EDA or Psi4 SAPT")
        act_eda.triggered.connect(self._on_open_eda_dialog)
        menu_tools_elec.addAction(act_eda)

        act_pes_2d = QtGui.QAction("🗺️ &2D Relaxed PES Scan Grid Builder (ORCA / Gaussian)...", self)
        act_pes_2d.setToolTip("Build two-dimensional coordinate scan grids (distances, angles, dihedrals) and generate ORCA and Gaussian ModRedundant decks")
        act_pes_2d.triggered.connect(self._on_open_pes_2d_builder)
        menu_tools_struct.addAction(act_pes_2d)

        act_huckel = QtGui.QAction("🎓 &Hückel Molecular Orbital (HMO) Theory Teaching Workbench...", self)
        act_huckel.setShortcut("Ctrl+Alt+H")
        act_huckel.setToolTip("Explore Hückel secular determinants |H - E·I| = 0, Frost-Musulin circles, Hund's rule Aufbau electron filling, Coulson bond orders, and LCAO orbital lobes (Ctrl+Alt+H)")
        act_huckel.triggered.connect(self._on_open_huckel_teaching_dialog)
        menu_tools_elec.addAction(act_huckel)

        act_nmr = QtGui.QAction("🧲 &NMR Chemical Shift Referencing & 1D Spectrum Plotter...", self)
        act_nmr.setToolTip("Convert raw QM isotropic shieldings (σ_iso) to referenced chemical shifts (δ ppm for 1H, 13C, 19F, 31P) and synthesize 1D spectra")
        act_nmr.triggered.connect(self._on_open_nmr_dialog)
        menu_tools_elec.addAction(act_nmr)

        act_microkinetics = QtGui.QAction("📈 &Microkinetics & Catalytic Cycle Simulator (TOF, [C] vs. t)...", self)
        act_microkinetics.setToolTip("Simulate multi-step catalytic cycles and reaction networks via numerical ODE rate equation integration with TOF and TON")
        act_microkinetics.triggered.connect(self._on_open_microkinetics_dialog)
        menu_tools_thermo.addAction(act_microkinetics)

        act_ibo = QtGui.QAction("🧬 &Intrinsic Bond Orbitals (IBO / IAO) & Lewis Localization...", self)
        act_ibo.setToolTip("Localize canonical molecular orbitals into Lewis chemical bonds (σ, π), lone pairs, and core pairs via PySCF IAO/IBO")
        act_ibo.triggered.connect(self._on_open_ibo_dialog)
        menu_tools_elec.addAction(act_ibo)

        act_qtaim = QtGui.QAction("⚛️ &QTAIM Topology & Bond Critical Points (3, -1)...", self)
        act_qtaim.setToolTip("Locate (3, -1) Bond Critical Points, ρ_BCP, and ∇²ρ_BCP from Molden wavefunctions via Bader AIM theory")
        act_qtaim.triggered.connect(self._on_open_qtaim_dialog)
        menu_tools_elec.addAction(act_qtaim)

        act_quasiharmonic = QtGui.QAction("🌡️ &Quasi-Harmonic Thermochemistry (Grimme mRRHO & Truhlar)...", self)
        act_quasiharmonic.setToolTip("Evaluate Grimme mRRHO and Truhlar quasi-harmonic vibrational entropy corrections for low-frequency modes")
        act_quasiharmonic.triggered.connect(self._on_open_quasiharmonic_dialog)
        menu_tools_thermo.addAction(act_quasiharmonic)

        act_si_gen = QtGui.QAction("📄 &1-Click Supporting Information (SI) & Methods Synthesizer...", self)
        act_si_gen.setToolTip("Generate publication-ready LaTeX SI documents, Markdown/Word tables, and formatted Cartesian XYZ appendices")
        act_si_gen.triggered.connect(self._on_open_si_generator_dialog)
        menu_tools_pub.addAction(act_si_gen)

        act_sasa = QtGui.QAction("💧 &Solvent-Accessible Surface Area (SASA) & Polar Area...", self)
        act_sasa.setToolTip("Calculate total SASA, polar surface area (PSA), and project solvent accessibility onto 3D atoms")
        act_sasa.triggered.connect(self._on_open_sasa_dialog)
        menu_tools_elec.addAction(act_sasa)

        act_nto = QtGui.QAction("☀️ &Natural Transition Orbitals (NTO) & Photochemistry (Hole-Particle)...", self)
        act_nto.setToolTip("SVD analysis of TD-DFT transition density matrices into dominant Hole -> Particle exciton pairs")
        act_nto.triggered.connect(self._on_open_nto_dialog)
        menu_tools_elec.addAction(act_nto)

        act_ts_builder = QtGui.QAction("🔄 &Reaction Path && Transition State (TS) Guess Interpolator (IDPP / QST2 / NEB-TS)...", self)
        act_ts_builder.setShortcut("Ctrl+Shift+T")
        act_ts_builder.setToolTip("Interpolate smooth, collision-free reaction paths between Reactant and Product geometries using IDPP and generate ORCA NEB-TS or Gaussian QST2/3 inputs (Ctrl+Shift+T)")
        act_ts_builder.triggered.connect(self._on_open_reaction_path_dialog)
        menu_tools_struct.addAction(act_ts_builder)

        act_crest_explorer = QtGui.QAction("🧬 &CREST && Conformer Ensemble Explorer (Boltzmann, RMSD, Overlays)...", self)
        act_crest_explorer.setShortcut("Ctrl+Shift+E")
        act_crest_explorer.setToolTip("Explore, rank, and analyze conformer ensembles with Boltzmann populations, Kabsch RMSDs, duplicate pruning, and batch QM generation (Ctrl+Shift+E)")
        act_crest_explorer.triggered.connect(self._on_open_crest_explorer)
        menu_tools_struct.addAction(act_crest_explorer)

        act_ramachandran = QtGui.QAction("🗺️ &2D Dihedral Conformational Landscape (Ramachandran PES)...", self)
        act_ramachandran.setToolTip("Generate 2D conformational energy contour surfaces E(φ, ψ) across two rotatable dihedral angles")
        act_ramachandran.triggered.connect(self._on_open_ramachandran_2d)
        menu_tools_struct.addAction(act_ramachandran)

        act_irc_sweeper = QtGui.QAction("⚛️ Interactive &IRC Reaction Sweeper & Energy Profile (MEP)...", self)
        act_irc_sweeper.setShortcut("Ctrl+Shift+I")
        act_irc_sweeper.setToolTip("Explore, scrub, and analyze Intrinsic Reaction Coordinate (IRC) Minimum Energy Pathways (MEP) with activation barriers, landmark jumps, and live 3D sync (Ctrl+Shift+I)")
        act_irc_sweeper.triggered.connect(self._on_open_irc_sweeper)
        menu_tools_thermo.addAction(act_irc_sweeper)

        act_esp_2d = QtGui.QAction("2D Electrostatic &Potential (ESP) Slicer...", self)
        act_esp_2d.setToolTip("Open Molden file and plot 2D ESP slice contour maps on XY, XZ, or YZ planes")
        act_esp_2d.triggered.connect(self._on_open_esp_2d_dialog)
        menu_tools_elec.addAction(act_esp_2d)

        act_nci_plot = QtGui.QAction("📉 Non-Covalent Interactions (NCI / RDG Plot)...", self)
        act_nci_plot.setToolTip("Plot 2D Reduced Density Gradient s(r) vs sign(λ₂)ρ to reveal H-bonds, dispersion, and steric clash")
        act_nci_plot.triggered.connect(self._on_open_nci_plot_dialog)
        menu_tools_elec.addAction(act_nci_plot)

        act_multiwfn = QtGui.QAction("Multiwfn &Wavefunction Suite (Bonds / Fukui / QTAIM / CM5)...", self)
        act_multiwfn.setToolTip("Run Multiwfn Mayer/Wiberg bond orders, Conceptual DFT Fukui indices, QTAIM critical points, and CM5 charges")
        act_multiwfn.triggered.connect(self._on_open_multiwfn_dialog)
        menu_tools_elec.addAction(act_multiwfn)

        act_nbo = QtGui.QAction("🧪 &NBO Donor–Acceptor Interaction Analyzer...", self)
        act_nbo.setShortcut("Ctrl+Shift+N")
        act_nbo.setToolTip("Inspect Natural Bond Orbital (NBO) second-order perturbation donor-acceptor interactions (E(2), n→σ*, hyperconjugation) (Ctrl+Shift+N)")
        act_nbo.triggered.connect(self._on_open_nbo_dialog)
        menu_tools_elec.addAction(act_nbo)

        act_nbo_net = QtGui.QAction("🕸️ &NBO E(2) Interaction Network Graph...", self)
        act_nbo_net.setToolTip("Visualize donor-acceptor interactions as a topological chord or bipartite graph with edge thicknesses proportional to E(2)")
        act_nbo_net.triggered.connect(self._on_open_nbo_network_dialog)
        menu_tools_elec.addAction(act_nbo_net)

        menu_tools.addSeparator()

        menu_tools_ext = menu_tools.addMenu("🧩 &Plugins && Add-ons...")

        act_plugins = QtGui.QAction("🐍 &Python Script & Plugin Runner...", self)
        act_plugins.setShortcut("Ctrl+Shift+Y")
        act_plugins.setToolTip("Run custom Python scripts or load user plugins from ~/.agymolden/plugins/ (Ctrl+Shift+Y)")
        act_plugins.triggered.connect(self._on_open_script_runner)
        menu_tools_ext.addAction(act_plugins)

        act_addons = QtGui.QAction("🧩 &Add-ons & Scientific Backend Manager...", self)
        act_addons.setToolTip("Inspect and 1-click install optional packages (PySCF, RDKit, ASE, TorchANI, Spglib) directly inside the app")
        act_addons.triggered.connect(self._on_open_addons_manager)
        menu_tools_ext.addAction(act_addons)

        # Surfaces & Electrostatic Potential Menu
        menu_surfaces = menubar.addMenu("&Surfaces")
        self.menu_surfaces = menu_surfaces

        act_esp_3d = QtGui.QAction("🌈 3D Electrostatic Potential (ESP) Mapped Density Surface", self)
        act_esp_3d.setShortcut("Ctrl+Alt+P")
        act_esp_3d.setToolTip("Evaluate and display 3D Electrostatic Potential (ESP) mapped onto total electron density isosurface (ρ = 0.002 a.u.) (Ctrl+Alt+P)")
        act_esp_3d.triggered.connect(self._on_quick_view_esp_map)
        menu_surfaces.addAction(act_esp_3d)

        act_esp_2d = QtGui.QAction("📈 2D Electrostatic Potential (ESP) Contour Map Slicer...", self)
        act_esp_2d.setToolTip("Compute and interactively slice 2D ESP contour isoline heatmaps across any molecular plane")
        act_esp_2d.triggered.connect(self._on_open_esp_2d_dialog)
        menu_surfaces.addAction(act_esp_2d)

        menu_surfaces.addSeparator()

        act_vol_studio = QtGui.QAction("🌐 Molecular Orbitals & Isosurfaces Studio (Cube / Molden)...", self)
        act_vol_studio.setShortcut("Ctrl+Alt+O")
        act_vol_studio.setToolTip("Open full GaussView / Avogadro style Volumetric Studio for MOs, ESP, ELF, NCI/RDG, ALIE, and Cube file arithmetic (Ctrl+Alt+O)")
        act_vol_studio.triggered.connect(self._on_open_volumetric_dialog)
        menu_surfaces.addAction(act_vol_studio)

        act_homo = QtGui.QAction("⚡ 1-Click View HOMO Orbital Surface", self)
        act_homo.setToolTip("Compute and display Highest Occupied Molecular Orbital (HOMO) 3D isosurface lobes")
        act_homo.triggered.connect(self._on_quick_view_homo)
        menu_surfaces.addAction(act_homo)

        act_lumo = QtGui.QAction("⚡ 1-Click View LUMO Orbital Surface", self)
        act_lumo.setToolTip("Compute and display Lowest Unoccupied Molecular Orbital (LUMO) 3D isosurface lobes")
        act_lumo.triggered.connect(self._on_quick_view_lumo)
        menu_surfaces.addAction(act_lumo)

        act_dual_fmo = QtGui.QAction("✨ 1-Click Dual Frontier Overlay (HOMO + LUMO Duality)", self)
        act_dual_fmo.setToolTip("Simultaneously render HOMO (Cyan/Amber) and LUMO (Violet/Emerald) lobes to inspect excitation pathways and donor-acceptor overlap")
        act_dual_fmo.triggered.connect(self._on_quick_view_homo_lumo_duality)
        menu_surfaces.addAction(act_dual_fmo)

        act_spin = QtGui.QAction("🌀 1-Click View Open-Shell Spin Density Lobes", self)
        act_spin.setToolTip("Compute and display alpha - beta open-shell spin density 3D isosurface lobes")
        act_spin.triggered.connect(self._on_quick_view_spin_density)
        menu_surfaces.addAction(act_spin)

        act_gallery = QtGui.QAction("📸 All Molecular Orbitals Snapshot Gallery...", self)
        act_gallery.setToolTip("Batch capture all or selected molecular orbitals into a contact sheet collage or folder of images")
        act_gallery.triggered.connect(self._on_open_orbital_snapshot_gallery)
        menu_surfaces.addAction(act_gallery)

        menu_surfaces.addSeparator()

        act_clear_surf = QtGui.QAction("✖ Clear 3D Isosurfaces / Orbitals", self)
        act_clear_surf.setShortcut("Esc")
        act_clear_surf.setToolTip("Remove active 3D isosurfaces, ESP maps, and orbital lobes from viewport (Esc)")
        act_clear_surf.triggered.connect(self._on_clear_orbital_surface)
        menu_surfaces.addAction(act_clear_surf)

        # View Menu
        menu_view = menubar.addMenu("&View")

        self.act_theme_dark = QtGui.QAction("🌙 &Dark Theme", self)
        self.act_theme_dark.setCheckable(True)
        self.act_theme_dark.setChecked(True)
        self.act_theme_dark.triggered.connect(lambda: self.set_color_theme("dark"))
        menu_view.addAction(self.act_theme_dark)

        self.act_theme_white = QtGui.QAction("☀️ &White Theme", self)
        self.act_theme_white.setCheckable(True)
        self.act_theme_white.setChecked(False)
        self.act_theme_white.triggered.connect(lambda: self.set_color_theme("white"))
        menu_view.addAction(self.act_theme_white)

        theme_group = QtGui.QActionGroup(self)
        theme_group.addAction(self.act_theme_dark)
        theme_group.addAction(self.act_theme_white)

        menu_view.addSeparator()

        act_toggle_measure = QtGui.QAction("📏 Show 3D &Measurement Overlays", self)
        act_toggle_measure.setCheckable(True)
        act_toggle_measure.setChecked(True)
        act_toggle_measure.setShortcut("Alt+M")
        act_toggle_measure.setToolTip("Toggle 3D floating distance, angle, and dihedral measurements over selected atoms (Alt+M)")
        def _toggle_meas(checked):
            self.viewer.show_measurements = checked
            self.viewer.update()
        act_toggle_measure.triggered.connect(_toggle_meas)
        menu_view.addAction(act_toggle_measure)

        act_pin_measure = QtGui.QAction("📌 &Pin Current Measurement to 3D View", self)
        act_pin_measure.setShortcut("Alt+P")
        act_pin_measure.setToolTip("Pins the currently measured distance/angle/dihedral permanently to the 3D model for publication figures (Alt+P)")
        def _pin_meas():
            if len(self.viewer.multi_selected) >= 2:
                sel = list(self.viewer.multi_selected)
                if not hasattr(self.viewer, "pinned_measurements"):
                    self.viewer.pinned_measurements = []
                if sel not in self.viewer.pinned_measurements:
                    self.viewer.pinned_measurements.append(sel)
                    self.statusBar().showMessage(f"Pinned measurement for atoms {sel} to 3D view.", 3500)
                else:
                    self.viewer.pinned_measurements.remove(sel)
                    self.statusBar().showMessage(f"Unpinned measurement for atoms {sel}.", 3500)
                self.viewer.update()
            else:
                self.statusBar().showMessage("Select 2, 3, or 4 atoms first to pin a measurement.", 3500)
        act_pin_measure.triggered.connect(_pin_meas)
        menu_view.addAction(act_pin_measure)

        act_clear_pinned = QtGui.QAction("🧹 Clear All &Pinned Measurements", self)
        act_clear_pinned.setToolTip("Clears all pinned distance, angle, and dihedral measurements from the 3D canvas.")
        def _clear_pinned():
            self.viewer.pinned_measurements = []
            self.viewer.update()
            self.statusBar().showMessage("Cleared all pinned measurements.", 3000)
        act_clear_pinned.triggered.connect(_clear_pinned)
        menu_view.addAction(act_clear_pinned)

        self.act_all_bond_lengths = QtGui.QAction("📐 Show All &Bond Lengths (Chemcraft Style)", self)
        self.act_all_bond_lengths.setCheckable(True)
        self.act_all_bond_lengths.setChecked(False)
        self.act_all_bond_lengths.setShortcut("Alt+Shift+B")
        self.act_all_bond_lengths.setToolTip("Toggle 3D bond length overlays (Å) on all covalent bonds (Alt+Shift+B)")
        def _toggle_all_bl(checked):
            self.viewer.show_bond_lengths = bool(checked)
            if hasattr(self, "chk_bond_lengths"):
                self.chk_bond_lengths.blockSignals(True)
                self.chk_bond_lengths.setChecked(bool(checked))
                self.chk_bond_lengths.blockSignals(False)
            self.viewer.update()
        self.act_all_bond_lengths.triggered.connect(_toggle_all_bl)
        menu_view.addAction(self.act_all_bond_lengths)

        act_pub_outlines = QtGui.QAction("🎨 Publication Silhouette &Outlines (IboView / QuteMol Style)", self)
        act_pub_outlines.setCheckable(True)
        act_pub_outlines.setChecked(False)
        act_pub_outlines.setToolTip("Render crisp ink silhouette outlines and hemisphere lighting for publication figures")
        def _toggle_pub_out(checked):
            self.viewer.show_publication_outlines = bool(checked)
            if hasattr(self, "chk_pub_outlines"):
                self.chk_pub_outlines.blockSignals(True)
                self.chk_pub_outlines.setChecked(bool(checked))
                self.chk_pub_outlines.blockSignals(False)
            self.viewer.update()
        act_pub_outlines.triggered.connect(_toggle_pub_out)
        menu_view.addAction(act_pub_outlines)

        act_toggle_nci = QtGui.QAction("🌐 Show &Non-Covalent Interactions (π-π Stacking, Halogen Bonds)", self)
        act_toggle_nci.setCheckable(True)
        act_toggle_nci.setChecked(False)
        act_toggle_nci.setShortcut("Alt+Shift+N")
        act_toggle_nci.setToolTip("Detects and renders 3D non-covalent interactions (π-π ring stacking in purple, halogen bonds in orange) (Alt+Shift+N)")
        act_toggle_nci.triggered.connect(self._on_toggle_nci)
        menu_view.addAction(act_toggle_nci)

        menu_view.addSeparator()

        # Journal Figure Presets Submenu
        menu_journal = menu_view.addMenu("🎨 &Journal Figure Presets...")
        act_pjacs = QtGui.QAction("📄 &JACS / ACS Pastel (Clean White, Ink Outlines)", self)
        act_pjacs.setShortcut("Ctrl+Alt+1")
        act_pjacs.setToolTip("Publication preset: clean white canvas, ink silhouette rim, soft diffuse shading (Ctrl+Alt+1)")
        act_pjacs.triggered.connect(lambda: self.apply_journal_preset("jacs"))
        menu_journal.addAction(act_pjacs)

        act_pnature = QtGui.QAction("✨ &Nature / Wiley Glossy (Specular Phong, White BG)", self)
        act_pnature.setShortcut("Ctrl+Alt+2")
        act_pnature.setToolTip("Publication preset: white canvas, crisp specular Phong highlights (Ctrl+Alt+2)")
        act_pnature.triggered.connect(lambda: self.apply_journal_preset("nature"))
        menu_journal.addAction(act_pnature)

        act_pangew = QtGui.QAction("💎 &Angewandte Bold (High Contrast Outlines)", self)
        act_pangew.setShortcut("Ctrl+Alt+3")
        act_pangew.setToolTip("Publication preset: white canvas with bold silhouette outlines and vivid contrast (Ctrl+Alt+3)")
        act_pangew.triggered.connect(lambda: self.apply_journal_preset("angewandte"))
        menu_journal.addAction(act_pangew)

        act_pmono = QtGui.QAction("🖨️ &Monochrome Print (Grayscale Manuscript)", self)
        act_pmono.setShortcut("Ctrl+Alt+4")
        act_pmono.setToolTip("Publication preset: white canvas, ink outlines for black & white print manuscripts (Ctrl+Alt+4)")
        act_pmono.triggered.connect(lambda: self.apply_journal_preset("monochrome"))
        menu_journal.addAction(act_pmono)

        act_pdark = QtGui.QAction("🌙 &Dark Presentation (Modern Deep Slate)", self)
        act_pdark.setShortcut("Ctrl+Alt+5")
        act_pdark.setToolTip("Presentation preset: deep slate background with vibrant emissive atom colors (Ctrl+Alt+5)")
        act_pdark.triggered.connect(lambda: self.apply_journal_preset("dark"))
        menu_journal.addAction(act_pdark)

        act_lighting_studio = QtGui.QAction("💡 &Lighting, Shaders && Figure Studio...", self)
        act_lighting_studio.setToolTip("Open interactive 3D Lighting, Shading models, and Colorblind-safe palettes studio")
        act_lighting_studio.triggered.connect(self._on_open_lighting_studio)
        menu_journal.addSeparator()
        menu_journal.addAction(act_lighting_studio)
        menu_view.addAction(act_lighting_studio)

        # Camera Projections Submenu
        menu_projections = menu_view.addMenu("📷 &Camera Projections && Alignments...")
        act_cam_xy = QtGui.QAction("Snap to &XY Plane (Top View, +Z)", self)
        act_cam_xy.setShortcut("Ctrl+1")
        act_cam_xy.setToolTip("Snap 3D camera to look directly onto the XY Cartesian plane (Ctrl+1)")
        act_cam_xy.triggered.connect(lambda: self.viewer.set_camera_plane_view("XY"))
        menu_projections.addAction(act_cam_xy)

        act_cam_xz = QtGui.QAction("Snap to &XZ Plane (Front View, +Y)", self)
        act_cam_xz.setShortcut("Ctrl+2")
        act_cam_xz.setToolTip("Snap 3D camera to look directly onto the XZ Cartesian plane (Ctrl+2)")
        act_cam_xz.triggered.connect(lambda: self.viewer.set_camera_plane_view("XZ"))
        menu_projections.addAction(act_cam_xz)

        act_cam_yz = QtGui.QAction("Snap to &YZ Plane (Side View, +X)", self)
        act_cam_yz.setShortcut("Ctrl+3")
        act_cam_yz.setToolTip("Snap 3D camera to look directly onto the YZ Cartesian plane (Ctrl+3)")
        act_cam_yz.triggered.connect(lambda: self.viewer.set_camera_plane_view("YZ"))
        menu_projections.addAction(act_cam_yz)

        menu_projections.addSeparator()

        act_cam_axes = QtGui.QAction("Align to &Principal Inertial Axes", self)
        act_cam_axes.setShortcut("Alt+I")
        act_cam_axes.setToolTip("Align camera view to look along the molecule's Principal Inertial Axes (Alt+I)")
        act_cam_axes.triggered.connect(self.viewer.align_to_principal_axes)
        menu_projections.addAction(act_cam_axes)

        act_cam_flip = QtGui.QAction("&Flip Camera 180° (Opposite Face)", self)
        act_cam_flip.setShortcut("Alt+F")
        act_cam_flip.setToolTip("Flip 3D camera 180 degrees horizontally to view opposite side (Alt+F)")
        act_cam_flip.triggered.connect(self.viewer.flip_camera_180)
        menu_projections.addAction(act_cam_flip)

        act_cam_ortho = QtGui.QAction("Toggle &Perspective / Orthographic Projection", self)
        act_cam_ortho.setShortcut("Alt+Shift+O")
        act_cam_ortho.setToolTip("Switch between perspective and orthographic (parallel) projection (Alt+Shift+O)")
        act_cam_ortho.triggered.connect(self.viewer.toggle_orthographic)
        menu_projections.addAction(act_cam_ortho)

        act_reset_cam = QtGui.QAction("&Reset Camera View", self)
        act_reset_cam.setShortcut("Space")
        act_reset_cam.triggered.connect(self.viewer.auto_center)
        menu_projections.addAction(act_reset_cam)
        menu_view.addAction(act_reset_cam)

        menu_view.addSeparator()

        # Dockable Workspace Panels Submenu
        menu_docks = menu_view.addMenu("🗂️ &Dockable Workspace Panels...")

        act_dock_job = QtGui.QAction("⚡ Calculation &Job Monitor Dock", self)
        act_dock_job.setToolTip("Dock or show the real-time QM calculation job monitor panel")
        act_dock_job.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_job_monitor().show())
        menu_docks.addAction(act_dock_job)

        act_dock_log = QtGui.QAction("📄 QM &Output Log Viewer Dock", self)
        act_dock_log.setToolTip("Dock or show the QM output log viewer and QC Doctor panel")
        act_dock_log.triggered.connect(
            lambda: getattr(self, "dock_manager", None)
            and self.dock_manager.get_or_create_output_viewer(getattr(self, "_last_loaded_qm_log_path", "") or "").show()
        )
        menu_docks.addAction(act_dock_log)

        act_dock_console = QtGui.QAction("🐍 Python &Chemistry Console Dock", self)
        act_dock_console.setToolTip("Dock or show the interactive Python scripting console (Alt+Shift+P)")
        act_dock_console.setShortcut("Alt+Shift+P")
        act_dock_console.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_console().show())
        menu_docks.addAction(act_dock_console)

        act_dock_ts = QtGui.QAction("🔄 Reaction Path & &TS Interpolator Dock", self)
        act_dock_ts.setToolTip("Dock or show the reaction path and Transition State (IDPP) guess generator")
        act_dock_ts.triggered.connect(self._on_open_reaction_path_dialog)
        menu_docks.addAction(act_dock_ts)

        act_dock_hpc = QtGui.QAction("🖥️ HPC &Cluster Queue Dock", self)
        act_dock_hpc.setToolTip("Dock or show the remote Slurm / HPC cluster job queue monitor")
        act_dock_hpc.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.get_or_create_hpc_queue().show())
        menu_docks.addAction(act_dock_hpc)

        act_dock_doctor = QtGui.QAction("🩺 QC &Doctor Diagnostics Dock", self)
        act_dock_doctor.setToolTip("Dock or show calculation failure diagnostics and crash auto-repair remedies")
        act_dock_doctor.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.open_qc_doctor())
        menu_docks.addAction(act_dock_doctor)

        menu_docks.addSeparator()

        act_float_all = QtGui.QAction("🖥️ Multi-Monitor: &Float All Panels", self)
        act_float_all.setToolTip("Detach all docks into independent floating windows for multi-monitor scientific workflows")
        act_float_all.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.float_all_docks())
        menu_docks.addAction(act_float_all)

        act_dock_all = QtGui.QAction("📌 &Snap All Panels into Main Window", self)
        act_dock_all.setToolTip("Dock all floating panels back inside the main window")
        act_dock_all.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.dock_all_panels())
        menu_docks.addAction(act_dock_all)

        act_reset_layout = QtGui.QAction("🔄 &Reset Workspace Layout", self)
        act_reset_layout.setToolTip("Reset dock panels to default hidden/docked state")
        act_reset_layout.triggered.connect(lambda: getattr(self, "dock_manager", None) and self.dock_manager.reset_layout())
        menu_docks.addAction(act_reset_layout)

        # Help Menu
        menu_help = menubar.addMenu("&Help")

        self.act_command_palette = QtGui.QAction("🚀 Quick &Command Palette...", self)
        self.act_command_palette.setShortcuts([
            QtGui.QKeySequence("Ctrl+Shift+P"),
            QtGui.QKeySequence("Ctrl+K"),
        ])
        self.act_command_palette.setToolTip("Instant searchable command palette for all 60+ AgyMolden tools and actions (Ctrl+K / Ctrl+Shift+P)")
        self.act_command_palette.triggered.connect(self._on_open_command_palette)
        menu_help.addAction(self.act_command_palette)

        act_test_setup = QtGui.QAction("🩺 &Test My Setup (ORCA / xTB)...", self)
        act_test_setup.setToolTip("Check whether ORCA and xTB are detected and ready to run")
        act_test_setup.triggered.connect(self._on_test_qm_setup)
        menu_help.addAction(act_test_setup)

        act_copy_methods = QtGui.QAction("📋 Copy &Methods / Citation Paragraph", self)
        act_copy_methods.setToolTip("Copy a ready-to-paste computational methods sentence to clipboard")
        act_copy_methods.triggered.connect(self._on_copy_methods_paragraph)
        menu_help.addAction(act_copy_methods)

        act_lab_mode = QtGui.QAction("🎓 Interactive &Lab Exercise (.agylab)...", self)
        act_lab_mode.setToolTip("Open a guided classroom lab exercise with checklist and ORCA preset")
        act_lab_mode.triggered.connect(self._on_open_lab_exercise)
        menu_help.addAction(act_lab_mode)

        act_examples = QtGui.QAction("🧪 &Example Molecule && Calculation Gallery...", self)
        act_examples.setShortcut("Ctrl+Shift+G")
        act_examples.setToolTip("Explore bundled Molden wavefunctions, QM output logs, and Z-matrix templates (Ctrl+Shift+G)")
        act_examples.triggered.connect(self._on_open_example_gallery)
        menu_help.addAction(act_examples)

        act_tour = QtGui.QAction("🎓 &Quick Tour && Onboarding Guide...", self)
        act_tour.setToolTip("Interactive 4-step guided walkthrough of AgyMolden core features")
        act_tour.triggered.connect(self._on_open_onboarding_tour)
        menu_help.addAction(act_tour)

        act_manual = QtGui.QAction("📖 &User Manual...", self)
        act_manual.setToolTip("Open the built-in AgyMolden User Manual")
        act_manual.triggered.connect(self._on_open_user_manual)
        menu_help.addAction(act_manual)

        self.act_simple_mode = QtGui.QAction("🎓 &Simple Mode (Hide Advanced Panels)", self)
        self.act_simple_mode.setCheckable(True)
        self.act_simple_mode.setToolTip("Hide specialist QM toolbars and advanced ribbon controls for a cleaner workspace")
        self.act_simple_mode.toggled.connect(self._on_toggle_simple_mode)
        menu_help.addAction(self.act_simple_mode)
        menu_help.addSeparator()

        act_about = QtGui.QAction("&About AgyMolden", self)
        act_about.triggered.connect(self._on_about)
        menu_help.addAction(act_about)

        act_shortcuts = QtGui.QAction("⌨️ &Keyboard && Mouse Shortcuts", self)
        act_shortcuts.setShortcut("F1")
        act_shortcuts.setToolTip("Show searchable cheat sheet of all keyboard and mouse shortcuts (F1 / ?)")
        act_shortcuts.triggered.connect(self._on_show_shortcuts)
        menu_help.addAction(act_shortcuts)

        act_view_log = QtGui.QAction("📋 View Diagnostic &Log (agymolden.log)...", self)
        act_view_log.setShortcut("F3")
        act_view_log.setToolTip("Inspect internal session log file for troubleshooting and error reporting (F3)")
        act_view_log.triggered.connect(self._on_view_diagnostic_log)
        menu_help.addAction(act_view_log)

        act_diag_bundle = QtGui.QAction("📦 Export Diagnostic &Bundle (.zip)...", self)
        act_diag_bundle.setToolTip("Export sanitized environment info, QM detection status, and agymolden.log as a zip for support")
        act_diag_bundle.triggered.connect(self._on_export_diagnostic_bundle)
        menu_help.addAction(act_diag_bundle)

        act_manual = QtGui.QAction("📖 Open Full User &Manual...", self)
        act_manual.setShortcut("Shift+F1")
        act_manual.triggered.connect(self._on_open_manual)
        menu_help.addAction(act_manual)

        menu_help.addSeparator()
        act_citations = QtGui.QAction("📜 &Cite Used Tools && Methods...", self)
        act_citations.setToolTip("Generate BibTeX and computational methods text acknowledging AgyMolden, UFF, PySCF, IDPP, and Spglib")
        act_citations.triggered.connect(self._on_open_citations_dialog)
        menu_help.addAction(act_citations)

        act_desktop_icon = QtGui.QAction("🖥️ Create &Desktop / App Shortcut...", self)
        act_desktop_icon.setToolTip("Create a convenient 1-click desktop icon for launching AgyMolden on your operating system.")
        act_desktop_icon.triggered.connect(self._on_create_desktop_shortcut)
        menu_help.addAction(act_desktop_icon)
