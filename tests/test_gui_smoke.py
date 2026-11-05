"""Offscreen GUI smoke tests: load a Molden file, evaluate orbitals/ESP, and move the isovalue."""

import os
import time

import pytest

pytest.importorskip("pyscf")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "examples", "h2o_orca.molden")


def _pump(app, seconds):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()


@pytest.fixture
def dialog(monkeypatch):
    import agymolden.workers as workers
    from agymolden.main_window import MainWindow
    from agymolden.dialog_volumetric import VolumetricSurfaceDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(workers, "SYNCHRONOUS", True)
    win = MainWindow()
    dlg = VolumetricSurfaceDialog(win, viewer=win.viewer)
    yield app, dlg
    dlg.close()
    dlg.deleteLater()
    win.close()
    win.deleteLater()
    app.processEvents()


def test_molden_load_orbital_esp_and_isovalue(dialog):
    app, dlg = dialog
    dlg.load_molden_path(EXAMPLE)
    assert dlg.combo_mo.count() > 0
    assert dlg.current_vol_data is not None  # HOMO evaluated on load

    dlg._on_compute_esp()
    assert "Electrostatic" in dlg.current_vol_data.title

    for v in (0.01, 0.02, 0.05, 0.1):  # rapid isovalue changes are debounced, must not hang
        dlg.spin_iso.setValue(v)
    _pump(app, 1.5)
    assert "Marching cubes error" not in dlg.lbl_stats.text()


def test_orbital_grid_cache_returns_same_shape():
    from agymolden.volumetric import evaluate_molden_orbital_grid
    a = evaluate_molden_orbital_grid(EXAMPLE, orbital_index=0, nx=24, ny=24, nz=24)
    b = evaluate_molden_orbital_grid(EXAMPLE, orbital_index=3, nx=24, ny=24, nz=24)
    assert a is not None and b is not None
    assert a.grid_data.shape == b.grid_data.shape == (24, 24, 24)


def test_grid_resolution_selector_and_orca_raman_uvvis(dialog, tmp_path):
    app, dlg = dialog
    # 1. Verify Low (30^3), Med (40^3), High (60^3) grid selector in VolumetricSurfaceDialog
    assert hasattr(dlg, "combo_grid_points")
    dlg.combo_grid_points.setCurrentIndex(0)  # Low (30^3)
    assert dlg._get_grid_pts() == 30
    dlg.load_molden_path(EXAMPLE)
    assert dlg.current_vol_data.grid_data.shape == (30, 30, 30)

    # 2. Verify ORCA Raman + UV-Vis fallback parser
    from agymolden.qm_log_parser import _parse_orca_logfile_fallback
    sample_orca = tmp_path / "water_raman_uv.out"
    sample_orca.write_text(
        """
                                 *****************
                                 * O   R   C   A *
                                 *****************
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
  O      0.000000    0.000000    0.117300
  H      0.000000    0.757200   -0.469200
  H      0.000000   -0.757200   -0.469200

FINAL SINGLE POINT ENERGY       -76.408953210

VIBRATIONAL FREQUENCIES
-----------------------
   6:      1648.50 cm**-1
   7:      3657.10 cm**-1
   8:      3755.90 cm**-1

NORMAL MODES
------------
                  6          7          8
      0       0.000000   0.000000   0.000000
      1       0.000000   0.000000   0.070000
      2       0.070000   0.050000   0.000000
      3       0.000000   0.000000   0.000000
      4      -0.430000   0.580000  -0.550000
      5      -0.560000  -0.400000  -0.430000
      6       0.000000   0.000000   0.000000
      7       0.430000  -0.580000  -0.550000
      8      -0.560000  -0.400000   0.430000

IR SPECTRUM
-----------
 Mode   freq       eps      Int      T**2         TX        TY        TZ
   6:  1648.50   0.01234   72.40   0.002700  ( 0.000000  0.000000  0.051962)
   7:  3657.10   0.00120    8.50   0.000143  ( 0.000000  0.000000  0.011958)
   8:  3755.90   0.00850   45.20   0.000740  ( 0.000000  0.027203  0.000000)

RAMAN SPECTRUM
--------------
 Mode    freq (cm**-1)   Activity   Depolarization
   6:       1648.50       12.3400      0.720000
   7:       3657.10      108.5000      0.140000
   8:       3755.90       31.2000      0.750000

ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
----------------------------------------------------------
State   Energy (cm-1)   Wavelength (nm)   fosc         T2
   1      60120.0          166.3        0.042100     0.02300
   2      78500.0          127.4        0.115000     0.04800

****ORCA TERMINATED NORMALLY****
""",
        encoding="utf-8",
    )
    res = _parse_orca_logfile_fallback(str(sample_orca))
    assert res is not None
    assert res.vibrational_analysis is not None
    assert res.vibrational_analysis.ir_intensities == [72.40, 8.50, 45.20]
    assert res.vibrational_analysis.raman_activities == [12.34, 108.5, 31.2]
    assert res.et_energies == [166.3, 127.4]
    assert res.et_oscs == [0.0421, 0.115]

    # 3. Verify MO Energy Ladder Widget populates when loading orbitals into MainWindow
    win = dlg.parent()
    win.load_orbitals_from_molden(EXAMPLE)
    assert hasattr(win, "mo_ladder_widget")
    assert len(win.mo_ladder_widget._orbitals) > 0
    assert win.mo_ladder_widget._homo_idx is not None

    # 4. Verify Brorsen pre-flight validator detects charge/multiplicity parity mismatch
    from agymolden.brorsen_builder import BrorsenInputBuilderDialog
    import numpy as np
    b_dlg = BrorsenInputBuilderDialog(win, symbols=["O", "H", "H"], coords=np.zeros((3, 3)))
    assert b_dlg._validate_stages_preflight() == []
    b_dlg.generator.stages[0].charge = 0
    b_dlg.generator.stages[0].mult = 2  # 10 electrons cannot be doublet
    warnings = b_dlg._validate_stages_preflight()
    assert len(warnings) == 1
    assert "incompatible with Multiplicity" in warnings[0]
    b_dlg.close()


def test_audit_regressions_frozen_undo_and_html_export(dialog, tmp_path, monkeypatch):
    """Verify frozen_atoms undo/redo/deletion remapping, HTML QM summary export, and surface GL cache retirement."""
    import numpy as np
    from agymolden.qm_log_parser import QMLogParseResult

    _app, dlg = dialog
    win = dlg.parent()
    win._commit_explicit_structure(["C"], np.array([[0.0, 0.0, 0.0]]), [], {}, relax_iters=10, add_h=True)
    assert len(win.zatoms) == 5

    # Freeze atom 0 and atom 4, then delete atom 1 and verify frozen atom 4 shifts to 3
    win.viewer.multi_selected = [0, 4]
    win._on_freeze_selected_atoms()
    assert win.frozen_atoms == {0, 4}

    win._delete_atom_or_orbit(1)
    assert len(win.zatoms) == 4
    assert 0 in win.frozen_atoms and 3 in win.frozen_atoms

    # Undo should restore 5 atoms and original frozen_atoms {0, 4}
    win.undo()
    assert len(win.zatoms) == 5
    assert win.frozen_atoms == {0, 4}

    # Verify _on_export_html reads _last_qm_res
    win._last_qm_res = QMLogParseResult(
        program_name="ORCA",
        symbols=["C", "H", "H", "H", "H"],
        trajectory_coords=[np.copy(win.coords)],
        trajectory_energies=[-40.5 * 27.211386245988],
        vibrational_analysis=None,
        dipole_debye=1.85,
        homo_ev=-10.2,
        lumo_ev=2.4,
    )
    out_html = tmp_path / "export_test.html"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(out_html), "HTML"))
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **kw: None)
    win._on_export_html()
    assert out_html.is_file()
    html_txt = out_html.read_text(encoding="utf-8")
    assert "ORCA" in html_txt
    assert "1.85" in html_txt

    # Verify active_isosurfaces setter retires old _gl_cache tuples
    class _DummyMesh:
        pass

    m1 = _DummyMesh()
    m1._gl_cache = (101, 102, 103, 36)
    win.viewer.active_isosurfaces = [m1]
    win.viewer.active_isosurfaces = []
    assert (101, 102, 103, 36) in win.viewer._retired_surface_gl_caches
    assert m1._gl_cache is None


def test_easy_wins_features(dialog):
    """Verify all 6 Easy Wins + zero shortcut collisions and toolbar objectName."""
    import numpy as np
    from agymolden.chem_identifiers import format_coords_block_string, parse_smiles_or_inchi_to_3d

    _app, dlg = dialog
    win = dlg.parent()

    # 1. Toolbar objectName & no shortcut collisions across menu QActions
    toolbars = win.findChildren(QtWidgets.QToolBar)
    assert any(tb.objectName() == "MainControlsToolbar" for tb in toolbars)
    shortcuts_map = {}
    for act in win.findChildren(QtWidgets.QWidget.__bases__[0]):
        if hasattr(act, "shortcut") and not act.shortcut().isEmpty():
            sc = act.shortcut().toString()
            assert sc not in shortcuts_map, f"Duplicate shortcut {sc}: {act.text()} vs {shortcuts_map[sc]}"
            shortcuts_map[sc] = act.text()

    # 2. Easy Win 1: Publication Outlines toggle
    assert hasattr(win, "chk_pub_outlines")
    win.chk_pub_outlines.setChecked(True)
    assert win.viewer.show_publication_outlines is True
    win.chk_pub_outlines.setChecked(False)
    assert win.viewer.show_publication_outlines is False

    # 3. Easy Win 2: Copy Coordinates Block (No Header for ORCA/Gaussian)
    blk = format_coords_block_string(["O", "H", "H"], np.array([[0.0, 0.0, 0.1], [0.0, 0.75, -0.4], [0.0, -0.75, -0.4]]))
    lines = blk.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("O ")
    win.load_from_smiles_or_inchi("water", name="water")
    win._on_copy_coords_block()
    clip = QtWidgets.QApplication.clipboard().text()
    assert len(clip.strip().splitlines()) == 3

    # 4. Easy Win 3: Build by Chemical Name (offline dictionary)
    syms, crds, bonds, _bo = parse_smiles_or_inchi_to_3d("aspirin")
    assert len(syms) == 21
    assert crds.shape == (21, 3)
    assert len(bonds) >= 21

    # 5. Easy Win 4: 3D Visual Indicator for Frozen Atoms synced to viewer.frozen_atoms
    win.viewer.multi_selected = [0, 1]
    win._on_freeze_selected_atoms()
    assert win.viewer.frozen_atoms == {0, 1}
    win._on_unfreeze_all_atoms()
    assert win.viewer.frozen_atoms == set()

    # 6. Easy Win 5: Adopt Trajectory Frame into editable Z-Matrix with Undo
    win.load_from_smiles_or_inchi("water", name="water")
    c0 = np.copy(win.coords)
    c1 = c0 + np.array([[0.0, 0.0, 0.25], [0.0, 0.1, 0.0], [0.0, -0.1, 0.0]])
    win.trajectory_coords = [c0, c1]
    win.slider_trajectory.setRange(0, 1)
    win.slider_trajectory.setValue(1)
    win._on_adopt_trajectory_frame()
    assert np.allclose(win.zatoms[1].dist, np.linalg.norm(c1[1] - c1[0]), atol=1e-3)
    win.undo()
    assert np.allclose(win.zatoms[1].dist, np.linalg.norm(c0[1] - c0[0]), atol=1e-3)

    # 7. Easy Win 6: Quick Command Palette action collection
    items = win._collect_command_palette_actions()
    assert len(items) >= 35
    labels = [lbl for lbl, _sc, _act in items]
    assert any("Coordinates Block" in l for l in labels)
    assert any("Publication Silhouette" in l for l in labels)


def test_phase3_improvements(dialog):
    """Verify Phase 3 features: pre-flight sanity in QuickOrcaDialog, Show All Bond Lengths,
    multi-metric PES/Convergence plotter, 1-Click Spin Density button, and GPU cylinder instancing shader."""
    import numpy as np
    from agymolden.dialog_quick_orca import QuickOrcaDialog
    from agymolden.dialog_pes_scan import PESScanDialog
    from agymolden.viewer_meshes import VERTEX_SHADER_SRC

    _app, dlg = dialog
    win = dlg.parent()

    # 1. Universal Pre-Flight Sanity Check in QuickOrcaDialog (clash & basis detection)
    from agymolden.zmatrix import ZAtom
    clashed_coords = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.96, 0.0]])
    z_list = [ZAtom("O"), ZAtom("H", ref1=0, dist=0.25), ZAtom("H", ref1=0, dist=0.96, ref2=1, angle=104.5)]
    orca_dlg = QuickOrcaDialog(win, z_list, coords=clashed_coords)
    res = orca_dlg._run_preflight_sanity()
    assert any("clash" in e.lower() for e in res["errors"])
    assert "clash" in orca_dlg.lbl_parity_warning.text().lower() or "pre-flight" in orca_dlg.lbl_parity_warning.text().lower()

    # 2. Chemcraft-style Show All Bond Lengths toggle
    assert hasattr(win, "chk_bond_lengths")
    assert hasattr(win, "act_all_bond_lengths")
    win.chk_bond_lengths.setChecked(True)
    assert win.viewer.show_bond_lengths is True
    assert win.act_all_bond_lengths.isChecked() is True
    win.act_all_bond_lengths.trigger()
    assert win.viewer.show_bond_lengths is False
    assert win.chk_bond_lengths.isChecked() is False

    # 3. Interactive Optimization Convergence & Force Plotter (multi-metric support)
    c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.00]])
    c1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    c2 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.95]])
    pes_dlg = PESScanDialog(win, viewer=win.viewer, trajectory_coords=[c0, c1, c2])
    pes_dlg.load_scan_data(
        [1.0, 2.0, 3.0],
        [-76.010, -76.025, -76.026],
        coord_label="Optimization Step",
        trajectory_coords=[c0, c1, c2],
        max_forces=[0.045, 0.012, 0.0003],
        rms_forces=[0.030, 0.008, 0.0002],
    )
    for idx in range(pes_dlg.combo_metric.count()):
        pes_dlg.combo_metric.setCurrentIndex(idx)
        assert len(pes_dlg.ax.lines) >= 1

    # 4. 1-Click Spin Density button on the Molecular Orbitals tab
    assert hasattr(win, "btn_quick_spin_dens")
    assert hasattr(win, "_on_quick_view_spin_density")

    # 5. GPU-Instanced Bond Cylinder Rendering in VERTEX_SHADER_SRC
    assert "in_instance_end" in VERTEX_SHADER_SRC
    assert "u_use_instancing == 2" in VERTEX_SHADER_SRC


def test_phase4_easy_wins(dialog):
    """Verify Phase 4 Easy Wins: Supporting Information generator, 1-Click ESP Map,
    3D Axis Triad & Scale Bar overlays, MO Atom/Shell % Breakdown, and CSV Exports."""
    import numpy as np
    from agymolden.chem_identifiers import format_supporting_information_entry
    from agymolden.volumetric import compute_molden_mo_atom_contributions
    from agymolden.dialog_pes_scan import PESScanDialog

    _app, dlg = dialog
    win = dlg.parent()

    # 1. Easy Win 1: Supporting Information (SI) generator in ACS, LaTeX, and Markdown
    symbols = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
    si_acs = format_supporting_information_entry(
        symbols, coords, title="Water", energy_ha=-76.4321,
        n_imag=1, lowest_freq_cm1=-450.2, style="acs"
    )
    assert "Water" in si_acs
    assert "-76.43210000 Eh" in si_acs
    assert "N_imag = 1" in si_acs

    si_latex = format_supporting_information_entry(
        symbols, coords, title="Water", energy_ha=-76.4321, style="latex"
    )
    assert "\\begin{table}" in si_latex
    assert "-76.43210000" in si_latex

    si_md = format_supporting_information_entry(
        symbols, coords, title="Water", energy_ha=-76.4321, style="markdown"
    )
    assert "### Water" in si_md
    assert "```xyz" in si_md

    win._on_copy_supporting_info("acs")

    # 2. Easy Win 2 & 4: 1-Click ESP Map button & MO Atom / Shell % contribution breakdown
    assert hasattr(win, "btn_quick_esp_map")
    assert hasattr(win, "_on_quick_view_esp_map")
    contrib_str = compute_molden_mo_atom_contributions(EXAMPLE, 0, top_k=3)
    assert isinstance(contrib_str, str)
    assert "%" in contrib_str

    # 3. Easy Win 3 & 5: 3D Orientation Axis Triad & Calibrated 3D Scale Bar
    assert hasattr(win, "chk_axis_triad")
    assert hasattr(win, "chk_scale_bar")
    assert win.viewer.show_axis_triad is True
    win.chk_scale_bar.setChecked(True)
    assert win.viewer.show_scale_bar is True
    win.chk_scale_bar.setChecked(False)
    assert win.viewer.show_scale_bar is False

    # 4. Easy Win 6: 1-Click CSV Data Export in PES & Spectrum Plotters
    c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.00]])
    c1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    pes_dlg = PESScanDialog(win, viewer=win.viewer, trajectory_coords=[c0, c1])
    pes_dlg.load_scan_data([1.0, 2.0], [-76.01, -76.02], max_forces=[0.02, 0.001])
    csv_pes = pes_dlg.get_csv_string()
    assert "Energy_Ha" in csv_pes
    assert "-76.01000000" in csv_pes

    csv_ir = win.spectrum_widget.get_csv_string()
    assert "Wavenumber_cm-1" in csv_ir

    win.uvvis_widget.set_uv_vis_data([250.0, 310.0], [0.12, 0.45])
    csv_uv = win.uvvis_widget.get_csv_string()
    assert "Wavelength_nm" in csv_uv


def test_phase5_frontier_features(dialog):
    """Verify Phase 5 Frontier features: Catalysis Steric Descriptors (%V_bur, Sterimol, Cone Angle),
    PySCF Orbital Localization (Pipek-Mezey / Boys), Isotropic NMR Shielding Parser & Spectrum Widget,
    and Publication Frontier Quartet (HOMO-1 to LUMO+1) preset."""
    import os
    import numpy as np
    from agymolden.coordination import compute_buried_volume, compute_sterimol_and_cone_angle
    from agymolden.volumetric import localize_molden_orbitals
    from agymolden.qm_log_parser import _extract_nmr_shieldings_from_text
    from agymolden.dialog_orbital_gallery import OrbitalSnapshotDialog

    _app, dlg = dialog
    win = dlg.parent()

    # 1. Catalysis Steric Descriptors (%V_bur + 4 quadrants, Verloop Sterimol L/B1/B5, Tolman Cone Angle)
    symbols = ["Pd", "P", "C", "C", "C", "H", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 2.28],
        [1.45, 0.0, 2.95],
        [-0.72, 1.25, 2.95],
        [-0.72, -1.25, 2.95],
        [1.50, 0.88, 3.60],
        [-1.60, 1.30, 3.60],
        [-1.60, -1.30, 3.60],
    ])
    vbur = compute_buried_volume(symbols, coords, center_idx=0, sphere_radius=3.5)
    assert 5.0 < vbur["v_bur_pct"] < 85.0
    assert abs((vbur["v_bur_pct"] + vbur["v_free_pct"]) - 100.0) < 0.1
    assert set(vbur["quadrants"].keys()) == {"NW", "NE", "SW", "SE"}

    ster = compute_sterimol_and_cone_angle(symbols, coords, primary_idx=0, attached_idx=1)
    assert ster["L"] > 3.5
    assert ster["B5"] >= ster["B1"] > 1.0
    assert 60.0 < ster["cone_angle_deg"] < 240.0

    # 2. 1-Click PySCF Orbital Localization (Pipek-Mezey & Foster-Boys)
    assert hasattr(win, "btn_localize_mos")
    loc_pm_path = localize_molden_orbitals(EXAMPLE, method="pipek-mezey")
    assert loc_pm_path is not None and os.path.isfile(loc_pm_path)
    try:
        os.remove(loc_pm_path)
    except OSError:
        pass

    # 3. Simulated Isotropic NMR Chemical Shift Parser & Spectrum Widget
    orca_nmr_sample = """
CHEMICAL SHIELDING SUMMARY (ppm)
--------------------------------
  Nucleus  Element    Isotropic     Anisotropy
  -------  -------  ------------   ------------
     0       C          130.400        45.200
     1       H           24.580         5.120
     2       H           24.580         5.120
"""
    nmr_list = _extract_nmr_shieldings_from_text(orca_nmr_sample)
    assert nmr_list is not None and len(nmr_list) == 3
    assert nmr_list[0]["symbol"] == "C" and abs(nmr_list[0]["sigma_iso"] - 130.4) < 1e-3

    assert hasattr(win, "nmr_widget")
    win.nmr_widget.set_nmr_data(nmr_list)
    nmr_csv = win.nmr_widget.get_csv_string()
    assert "Delta_Shift_ppm" in nmr_csv
    assert "130.4000" in nmr_csv

    # 4. Publication Frontier Quartet (HOMO-1 to LUMO+1) in OrbitalSnapshotDialog
    if getattr(win, "_active_molden_data", None):
        gal_dlg = OrbitalSnapshotDialog(
            win, EXAMPLE, win._active_molden_data.orbitals, homo_idx=1
        )
        assert hasattr(gal_dlg, "radio_frontier4")
        assert hasattr(gal_dlg, "btn_copy_sheet")
        gal_dlg.radio_frontier4.setChecked(True)
        sel_idx = gal_dlg._get_selected_indices()
        assert 1 <= len(sel_idx) <= 4
        gal_dlg.close()


def test_phase6_high_impact_additions(dialog):
    """Verify Phase 6 High-Impact Additions: HOMA Ring Aromaticity, Coordination Geometry Indices (tau_4/tau_5),
    PED Normal Mode Motion Assignment, Isotopologue Vibrational Shift & KIE, Boltzmann PES Populations,
    and Electronic Density of States (TDOS) Spectrum Widget."""
    import math
    import numpy as np
    from agymolden.aromatic import compute_ring_homa_and_planarity
    from agymolden.coordination import compute_coordination_geometry_indices
    from agymolden.vibrations import (
        VibrationalAnalysis,
        assign_normal_mode_motion,
        compute_isotopologue_vibrational_shifts,
    )
    from agymolden.dialog_pes_scan import PESScanDialog

    _app, dlg = dialog
    win = dlg.parent()

    # 1. HOMA Aromaticity Index & Planarity on ideal Benzene (R_CC = 1.388 A)
    r_cc = 1.388
    bz_syms = ["C"] * 6 + ["H"] * 6
    bz_coords = []
    for k in range(6):
        ang = math.radians(60.0 * k)
        bz_coords.append([r_cc * math.cos(ang), r_cc * math.sin(ang), 0.0])
    for k in range(6):
        ang = math.radians(60.0 * k)
        bz_coords.append([(r_cc + 1.08) * math.cos(ang), (r_cc + 1.08) * math.sin(ang), 0.0])
    bz_coords = np.array(bz_coords, dtype=np.float64)
    bz_bonds = [(k, (k + 1) % 6) for k in range(6)] + [(k, k + 6) for k in range(6)]
    rings = compute_ring_homa_and_planarity(bz_syms, bz_coords, bz_bonds)
    assert len(rings) == 1
    assert abs(rings[0]["homa"] - 1.0) < 1e-3
    assert rings[0]["planarity_rmsd_ang"] < 1e-4

    # 2. Coordination Geometry Indices (tau_4 for Square Planar vs Tetrahedral)
    sq_syms = ["Pt", "Cl", "Cl", "Cl", "Cl"]
    sq_coords = np.array([
        [0.0, 0.0, 0.0],
        [2.30, 0.0, 0.0],
        [-2.30, 0.0, 0.0],
        [0.0, 2.30, 0.0],
        [0.0, -2.30, 0.0],
    ])
    g_sq = compute_coordination_geometry_indices(sq_syms, sq_coords, center_idx=0, neighbor_indices=[1, 2, 3, 4])
    assert g_sq["coord_number"] == 4
    assert abs(g_sq["tau_4"] - 0.0) < 1e-2

    # 3. Automated Normal Mode PED Motion Assignment & Isotopologue / KIE Calculator
    w_syms = ["O", "H", "H"]
    w_coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])
    # Symmetric O-H stretch displacement along O-H1
    disp_stretch = np.array([[0.0, 0.0, -0.05], [0.0, 0.70, -0.50], [0.0, 0.0, 0.0]])
    lbl_stretch = assign_normal_mode_motion(w_syms, w_coords, disp_stretch, freq_cm1=3657.0)
    assert "Stretch" in lbl_stretch

    vib_an = VibrationalAnalysis(
        frequencies=[-800.0, 1640.0, 3657.0, 3756.0],
        displacements=np.array([disp_stretch, disp_stretch, disp_stretch, disp_stretch]),
        symbols=w_syms,
        base_coords=w_coords,
    )
    iso_res = compute_isotopologue_vibrational_shifts(vib_an, {1: 2.014102, 2: 2.014102})
    assert iso_res["iso_frequencies"][2] < iso_res["std_frequencies"][2]
    assert iso_res["delta_zpe_kcal"] < 0.0
    assert iso_res["kie_estimate"] > 1.0

    # 4. Boltzmann Population metric & CSV column in PESScanDialog
    c0 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.00]])
    c1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.96]])
    pes_dlg = PESScanDialog(win, viewer=win.viewer, trajectory_coords=[c0, c1])
    pes_dlg.load_scan_data([1.0, 2.0], [-76.000, -76.002])
    idx_boltz = pes_dlg.combo_metric.findData("boltzmann")
    assert idx_boltz >= 0
    pes_dlg.combo_metric.setCurrentIndex(idx_boltz)
    csv_pes = pes_dlg.get_csv_string()
    assert "Boltzmann_Pop_298K_pct" in csv_pes

    # 5. Electronic Density of States (TDOS) Spectrum Widget
    assert hasattr(win, "dos_widget")
    if getattr(win, "_active_molden_data", None):
        win.dos_widget.set_orbitals(win._active_molden_data.orbitals, homo_idx=1)
        dos_csv = win.dos_widget.get_csv_string()
        assert "Total_DOS" in dos_csv


def test_qm_usability_audit_improvements(dialog, tmp_path, monkeypatch):
    """Verify usability audit improvements across PySCF, ORCA, Gaussian, and openLOWDIN."""
    import numpy as np
    from agymolden.qm_input_generator import (
        QMCalculationParameters,
        GaussianGenerator,
        OrcaGenerator,
        PySCFGenerator,
        OpenLOWDINGenerator,
    )
    from agymolden.qm_log_parser import parse_qm_logfile

    _app, dlg = dialog
    win = dlg.parent()
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **kw: QtWidgets.QMessageBox.StandardButton.No)

    syms = ["O", "H", "H"]
    coords = np.array([[0.0, 0.0, 0.117], [0.0, 0.757, -0.469], [0.0, -0.757, -0.469]])

    # 1. Input generators support Opt+Freq, TD-DFT (UV-Vis), and NMR (GIAO)
    p_nmr = QMCalculationParameters(job_type="NMR Chemical Shifts (GIAO)", method="B3LYP", basis="def2-SVP")
    assert "nmr=giao" in GaussianGenerator().generate_input(syms, coords, p_nmr)
    assert "gfinput" in GaussianGenerator().generate_input(syms, coords, p_nmr)
    assert "NMR" in OrcaGenerator().generate_input(syms, coords, p_nmr)
    assert "from pyscf.prop import nmr" in PySCFGenerator().generate_input(syms, coords, p_nmr)

    p_td = QMCalculationParameters(job_type="TD-DFT Excited States (UV-Vis)", method="B3LYP", basis="def2-SVP")
    assert "td=(nstates=10)" in GaussianGenerator().generate_input(syms, coords, p_td)
    assert "%tddft" in OrcaGenerator().generate_input(syms, coords, p_td)
    assert "from pyscf import tdscf" in PySCFGenerator().generate_input(syms, coords, p_td)

    p_freq = QMCalculationParameters(job_type="Opt + Freq", method="B3LYP", basis="def2-SVP")
    lowdin_txt = OpenLOWDINGenerator().generate_input(syms, coords, p_freq)
    assert "frequencyAnalysis = .true." in lowdin_txt
    assert "optimizeGeometry = .true." in lowdin_txt

    pyscf_txt = PySCFGenerator().generate_input(syms, coords, p_freq)
    assert "[Artifact Generated]" in pyscf_txt
    assert "_agy_step_cb" in pyscf_txt

    # 2. Direct PySCF .out log parsing in parse_qm_logfile
    pyscf_out = tmp_path / "water_pyscf.out"
    pyscf_out.write_text(
        """
converged SCF energy = -76.0267606755
Cartesian Coordinates (Angstrom):
  O      0.000000     0.000000     0.117300
  H      0.000000     0.757200    -0.469200
  H      0.000000    -0.757200    -0.469200
---
SCF Total Energy: -76.0267606755
Excited State   1: E =   7.4500 eV  lambda =   166.42 nm  f =   0.0420
CHEMICAL SHIELDING SUMMARY (ppm)
  Nucleus  Element    Isotropic     Anisotropy
  -------  -------  ------------   ------------
      0       O          320.5000       45.1000
      1       H           31.2000        8.4000
      2       H           31.2000        8.4000
""",
        encoding="utf-8",
    )
    res_py = parse_qm_logfile(str(pyscf_out))
    assert res_py.program_name == "PySCF"
    assert len(res_py.symbols) == 3
    assert res_py.et_energies == [166.42]
    assert len(res_py.nmr_shieldings) == 3

    # 3. openLOWDIN .out log parsing in parse_qm_logfile
    lowdin_out = tmp_path / "water_lowdin.out"
    lowdin_out.write_text(
        """
BEGIN OF LOWDIN
GEOMETRY
    O     dirac      0.000000    0.000000    0.117300
    H     dirac      0.000000    0.757200   -0.469200
    H     dirac      0.000000   -0.757200   -0.469200
END GEOMETRY
TOTAL ENERGY = -76.01234567
END OF LOWDIN
""",
        encoding="utf-8",
    )
    res_low = parse_qm_logfile(str(lowdin_out))
    assert res_low.program_name == "openLOWDIN"
    assert res_low.symbols == ["O", "H", "H"]
    assert len(res_low.trajectory_energies) == 1

    # 4. File openers for .gjf, .lowdin, .py, and ORCA .inp (without opening Brorsen dialog for ORCA)
    gjf_file = tmp_path / "test_water.gjf"
    gjf_file.write_text(GaussianGenerator().generate_input(syms, coords, p_nmr), encoding="utf-8")
    win.open_path(str(gjf_file))
    assert len(win.zatoms) == 3

    lowdin_file = tmp_path / "test_water.lowdin"
    lowdin_file.write_text(lowdin_txt, encoding="utf-8")
    win.open_path(str(lowdin_file))
    assert len(win.zatoms) == 3

    py_file = tmp_path / "test_water.py"
    py_file.write_text(pyscf_txt, encoding="utf-8")
    win.open_path(str(py_file))
    assert len(win.zatoms) == 3

    orca_file = tmp_path / "test_water_orca.inp"
    orca_file.write_text(OrcaGenerator().generate_input(syms, coords, p_nmr), encoding="utf-8")
    win.open_path(str(orca_file))
    assert len(win.zatoms) == 3

    # 5. Replacing a terminal Cl atom by Phenyl (-C6H5) cleanly substitutes Cl -> C6H5
    # Start with methane (CH4), replace one H with Chlorine (-Cl) to get CH3Cl, then replace Cl with Phenyl (-C6H5) to get Toluene (C7H8, 15 atoms, 0 Cl)
    from agymolden.fragments import FRAGMENTS
    win._commit_explicit_structure(["C"], np.array([[0.0, 0.0, 0.0]]), [], {}, relax_iters=10, add_h=True)
    assert len(win.zatoms) == 5
    h_idx = next(i for i, a in enumerate(win.zatoms) if a.symbol == "H")
    win._attach_fragment_to_atom_or_orbit(h_idx, FRAGMENTS["Chlorine (-Cl)"], conn_idx=0, relax_iters=20)
    assert [a.symbol for a in win.zatoms].count("Cl") == 1
    assert len(win.zatoms) == 5

    cl_idx = next(i for i, a in enumerate(win.zatoms) if a.symbol == "Cl")
    win._attach_fragment_to_atom_or_orbit(cl_idx, FRAGMENTS["Phenyl (-C6H5)"], conn_idx=0, relax_iters=20)
    syms_after = [a.symbol for a in win.zatoms]
    assert syms_after.count("Cl") == 0, f"Expected Cl to be replaced, got {syms_after}"
    assert syms_after.count("C") == 7
    assert syms_after.count("H") == 8
    assert len(win.zatoms) == 15










