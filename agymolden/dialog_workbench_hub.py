"""
Unified Chemistry Workbench Hub & Workflow Dashboard for AgyMolden.

Provides:
- High-level categorized visual dashboard organizing all 40+ specialized tools into clear workflows:
  1. ⚗️ Reaction Mechanisms & Catalysis
  2. 🧲 Spectroscopy & Analytical
  3. 🧬 Orbitals, Wavefunction & Bonding
  4. 📄 Publishing & HPC Automation
- Search filter / live keyword search across tools, descriptions, and keyboard shortcuts
- 1-Click Launch buttons with status badges (e.g. "Built-in", "PySCF", "ORCA")
- Eliminates menu clutter and guides newcomers through complete computational chemistry workflows.
"""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Optional
from PyQt6 import QtCore, QtGui, QtWidgets


class HubToolItem(NamedTuple):
    title: str
    category: str
    description: str
    icon: str
    shortcut: str
    badge: str
    callback_name: str


HUB_TOOLS: List[HubToolItem] = [
    # 1. Reaction Mechanisms & Catalysis
    HubToolItem(
        title="Reaction Path & TS Guess (IDPP)",
        category="Reaction Mechanisms & Catalysis",
        description="Interpolate collision-free reaction paths between Reactants and Products using IDPP for ORCA NEB-TS / Gaussian QST2.",
        icon="🔄",
        shortcut="Ctrl+Shift+T",
        badge="Built-in",
        callback_name="_on_open_reaction_path_dialog",
    ),
    HubToolItem(
        title="IRC Reaction Sweeper & Energy Profile",
        category="Reaction Mechanisms & Catalysis",
        description="Interactive scrubbing along Intrinsic Reaction Coordinate pathways with barrier heights and 3D geometry sync.",
        icon="⚛️",
        shortcut="Ctrl+Shift+I",
        badge="Built-in",
        callback_name="_on_open_irc_sweeper",
    ),
    HubToolItem(
        title="Multi-Structure 3D Superposition & RMSD Overlay",
        category="Reaction Mechanisms & Catalysis",
        description="Kabsch SVD optimal rigid superposition, all-atom and heavy-atom RMSD, and ghost model comparison overlay.",
        icon="🔬",
        shortcut="Ctrl+Shift+O",
        badge="Built-in",
        callback_name="_on_open_structure_overlay",
    ),
    HubToolItem(
        title="Reaction Thermochemistry & Kinetics",
        category="Reaction Mechanisms & Catalysis",
        description="Calculate ΔG‡, ΔH‡, ΔGrxn, Keq, and Eyring reaction rate constants k(T) from QM output logs.",
        icon="⚗️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_reaction_thermo_dialog",
    ),
    HubToolItem(
        title="Microkinetics & Catalytic Cycles",
        category="Reaction Mechanisms & Catalysis",
        description="Simulate multi-step catalytic cycles via coupled non-linear ODE integration with TOF and TON metrics.",
        icon="📈",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_microkinetics_dialog",
    ),
    HubToolItem(
        title="Catalysis Sterics (%V_bur & Sterimol)",
        category="Reaction Mechanisms & Catalysis",
        description="Quantify Percent Buried Volume, SambVca 2D topographic contour maps, and Verloop Sterimol (L, B1, B5).",
        icon="🛡️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_sterics_dialog",
    ),
    HubToolItem(
        title="MECP & Spin-Inversion Explorer",
        category="Reaction Mechanisms & Catalysis",
        description="Interpolate crossing seams (ΔE → 0) in two-state reactivity and generate ORCA SurfCross decks.",
        icon="🔀",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_mecp_dialog",
    ),
    HubToolItem(
        title="Marcus Electron Transfer Theory",
        category="Reaction Mechanisms & Catalysis",
        description="Evaluate reorganization energies (λ), electronic coupling (V_ab), and electron transfer rates k_ET.",
        icon="⚡",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_marcus_dialog",
    ),
    HubToolItem(
        title="2D Relaxed PES Scan Grid Builder",
        category="Reaction Mechanisms & Catalysis",
        description="Build 2D coordinate scan grids (bonds, angles, dihedrals) and generate ORCA/Gaussian ModRedundant decks.",
        icon="🗺️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_pes_2d_builder",
    ),

    # 2. Spectroscopy & Analytical
    HubToolItem(
        title="Solvent-Accessible Surface Area (SASA & PSA)",
        category="Spectroscopy & Analytical",
        description="Shrake-Rupley numerical surface area with Fibonacci sphere sampling, polar surface area (PSA), and 3D exposure heatmap.",
        icon="🌐",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_sasa_dialog",
    ),
    HubToolItem(
        title="Natural Transition Orbitals (NTO Photochemistry)",
        category="Spectroscopy & Analytical",
        description="SVD decomposition of TD-DFT transition density matrices into intuitive Hole → Particle exciton state pairs.",
        icon="✨",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_nto_dialog",
    ),
    HubToolItem(
        title="NMR Chemical Shift Referencing & 1D Spectrum",
        category="Spectroscopy & Analytical",
        description="Convert isotropic shieldings to referenced chemical shifts (1H, 13C, 19F, 31P) with synthetic Lorentzian spectra.",
        icon="🧲",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_nmr_dialog",
    ),
    HubToolItem(
        title="Quasi-Harmonic Thermochemistry (mRRHO)",
        category="Spectroscopy & Analytical",
        description="Evaluate Grimme mRRHO and Truhlar quasi-harmonic vibrational entropy corrections for low-frequency modes.",
        icon="🌡️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_quasiharmonic_dialog",
    ),
    HubToolItem(
        title="Vibrational Modes & Normal Mode Animation",
        category="Spectroscopy & Analytical",
        description="Inspect harmonic frequencies, vectors, IR intensities, and animate transition state imaginary frequencies.",
        icon="〰️",
        shortcut="Ctrl+Shift+V",
        badge="Built-in",
        callback_name="_on_open_vibrations_tab",
    ),
    HubToolItem(
        title="Isotopologues & Isotope Shifts",
        category="Spectroscopy & Analytical",
        description="Predict kinetic isotope effects (KIE) and vibrational frequency shifts from isotopic mass substitutions.",
        icon="🧪",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_isotopologue_dialog",
    ),
    HubToolItem(
        title="Electrochemical Redox Potentials",
        category="Spectroscopy & Analytical",
        description="Compute standard reduction potentials (E° vs SHE / Fc/Fc+) via Born-Haber thermodynamic solvation cycles.",
        icon="🔋",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_redox_dialog",
    ),
    HubToolItem(
        title="Spin Contamination & Yamaguchi J-Coupling",
        category="Spectroscopy & Analytical",
        description="Evaluate <S^2> spin contamination, calculate Yamaguchi projected energies, and Heisenberg J coupling.",
        icon="🧲",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_spin_diagnostics",
    ),

    # 3. Orbitals, Wavefunction & Bonding
    HubToolItem(
        title="3D Non-Covalent Interactions (NCI / RDG)",
        category="Orbitals, Wavefunction & Bonding",
        description="Extract 3D real-space gradient isosurfaces colored by sign(λ2)ρ (Blue=H-bonds, Green=vdW, Red=Steric clash).",
        icon="📉",
        shortcut="",
        badge="PySCF",
        callback_name="_on_open_nci_3d_action",
    ),
    HubToolItem(
        title="Intrinsic Bond Orbitals (IBO/IAO) Localization",
        category="Orbitals, Wavefunction & Bonding",
        description="Transform canonical molecular orbitals into Lewis chemical bonds (σ, π), lone pairs, and core pairs.",
        icon="🧬",
        shortcut="",
        badge="PySCF",
        callback_name="_on_open_ibo_dialog",
    ),
    HubToolItem(
        title="QTAIM Topology & Bond Critical Points (3,-1)",
        category="Orbitals, Wavefunction & Bonding",
        description="Locate (3, -1) Bond Critical Points, ρ_BCP, and ∇²ρ_BCP from Molden wavefunctions via Bader AIM theory.",
        icon="⚛️",
        shortcut="",
        badge="PySCF",
        callback_name="_on_open_qtaim_dialog",
    ),
    HubToolItem(
        title="Hückel Molecular Orbital (HMO) Theory",
        category="Orbitals, Wavefunction & Bonding",
        description="Secular determinants |H - E·I| = 0, Frost-Musulin circles, Coulson bond orders, and LCAO orbital lobes.",
        icon="🎓",
        shortcut="Ctrl+Alt+H",
        badge="Built-in",
        callback_name="_on_open_huckel_teaching_dialog",
    ),
    HubToolItem(
        title="NBO Perturbation Analyzer & Network Graph",
        category="Orbitals, Wavefunction & Bonding",
        description="Natural Bond Orbital donor-acceptor interactions E(2) and topological chord interaction network graph.",
        icon="🕸️",
        shortcut="Ctrl+Shift+N",
        badge="Built-in",
        callback_name="_on_open_nbo_dialog",
    ),
    HubToolItem(
        title="2D Electrostatic Potential (ESP) Slicer",
        category="Orbitals, Wavefunction & Bonding",
        description="Interactive 2D slicing and contour plotting of Molecular Electrostatic Potential on XY, XZ, or YZ planes.",
        icon="🗺️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_esp_2d_dialog",
    ),
    HubToolItem(
        title="Energy Decomposition Analysis (EDA / SAPT)",
        category="Orbitals, Wavefunction & Bonding",
        description="Decompose interaction energies into Electrostatics, Pauli Repulsion, Orbital/Charge-Transfer, and Dispersion.",
        icon="🧩",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_eda_dialog",
    ),
    HubToolItem(
        title="Multiwfn Wavefunction Analysis Suite",
        category="Orbitals, Wavefunction & Bonding",
        description="Conceptual DFT Fukui indices, Mayer/Wiberg bond orders, QTAIM critical points, and CM5 atomic charges.",
        icon="🔬",
        shortcut="",
        badge="Optional",
        callback_name="_on_open_multiwfn_dialog",
    ),

    # 4. Publishing & HPC Automation
    HubToolItem(
        title="1-Click Publication SI & Methods Synthesizer",
        category="Publishing & HPC Automation",
        description="Automated standalone LaTeX SI document (.tex) and Markdown/Word tables with formatted Cartesian XYZ blocks.",
        icon="📄",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_si_generator_dialog",
    ),
    HubToolItem(
        title="QC Doctor Calculation Autofix & Diagnostics",
        category="Publishing & HPC Automation",
        description="Parse failed ORCA and Gaussian runs, explain error causes, and generate autofixed input decks in 1 click.",
        icon="🩺",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_qc_doctor_dialog",
    ),
    HubToolItem(
        title="HPC Batch Script Generator (SLURM / PBS)",
        category="Publishing & HPC Automation",
        description="Generate robust supercomputer submission scripts with memory, CPU core, and scratch directory controls.",
        icon="🖥️",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_hpc_dialog",
    ),
    HubToolItem(
        title="Interactive 3D HTML Presentation Export",
        category="Publishing & HPC Automation",
        description="Export interactive standalone 3D HTML models viewable in any web browser or phone with orbital surfaces.",
        icon="🌐",
        shortcut="Ctrl+Shift+H",
        badge="Built-in",
        callback_name="_on_export_html",
    ),
    HubToolItem(
        title="ChemMCP AI Assistant Server (Stdio)",
        category="Publishing & HPC Automation",
        description="Model Context Protocol JSON-RPC 2.0 server connecting external AI assistants (Claude Desktop, local LLMs) directly to AgyMolden.",
        icon="🤖",
        shortcut="",
        badge="Built-in",
        callback_name="_on_open_mcp_info",
    ),
]


class WorkbenchHubDialog(QtWidgets.QDialog):
    """Modern, visual, categorized Chemistry Workbench Hub & Workflow Dashboard."""

    def __init__(self, parent: QtWidgets.QMainWindow):
        super().__init__(parent)
        self.parent_mw = parent
        self.setWindowTitle("⚡ AgyMolden Chemistry Workbench Hub")
        self.resize(980, 680)

        self._init_ui()
        self._populate_categories()
        self._filter_tools()

    def _init_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Banner
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px; border: 1px solid #1e293b;")
        h_lay = QtWidgets.QHBoxLayout(header)
        h_icon = QtWidgets.QLabel("<span style='font-size: 32px;'>⚡</span>")
        h_lay.addWidget(h_icon)

        t_lay = QtWidgets.QVBoxLayout()
        t_lay.addWidget(QtWidgets.QLabel("<b style='color: #38bdf8; font-size: 16px;'>AgyMolden Chemistry Workbench Hub</b>"))
        t_lay.addWidget(QtWidgets.QLabel("<span style='color: #94a3b8; font-size: 12px;'>Select a workflow or search 40+ specialized tools for reaction mechanisms, spectroscopy, wavefunction bonding, and publishing.</span>"))
        h_lay.addLayout(t_lay)
        h_lay.addStretch()

        main_layout.addWidget(header)

        # Search & Filter Bar
        filter_bar = QtWidgets.QHBoxLayout()
        self.edit_search = QtWidgets.QLineEdit()
        self.edit_search.setPlaceholderText("🔍 Search tools by keyword, method, or property (e.g. 'NMR', 'TS', 'Hessian', 'Sterics', 'LaTeX')...")
        self.edit_search.setStyleSheet("background-color: #18181b; color: #f1f5f9; padding: 8px 12px; border-radius: 6px; font-size: 13px; border: 1px solid #3f3f46;")
        self.edit_search.textChanged.connect(self._filter_tools)
        filter_bar.addWidget(self.edit_search)

        self.cbo_cat_filter = QtWidgets.QComboBox()
        self.cbo_cat_filter.setStyleSheet("background-color: #18181b; color: #f1f5f9; padding: 6px 10px; border-radius: 6px; border: 1px solid #3f3f46;")
        self.cbo_cat_filter.addItem("All Categories")
        categories = sorted(list(set(t.category for t in HUB_TOOLS)))
        for c in categories:
            self.cbo_cat_filter.addItem(c)
        self.cbo_cat_filter.currentIndexChanged.connect(self._filter_tools)
        filter_bar.addWidget(self.cbo_cat_filter)

        main_layout.addLayout(filter_bar)

        # Scrollable Cards Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        self.cards_container = QtWidgets.QWidget()
        self.cards_layout = QtWidgets.QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(10)
        scroll.setWidget(self.cards_container)

        main_layout.addWidget(scroll, stretch=1)

        # Bottom Bar
        bottom_bar = QtWidgets.QHBoxLayout()
        self.lbl_count = QtWidgets.QLabel(f"Showing {len(HUB_TOOLS)} specialized tools.")
        self.lbl_count.setStyleSheet("color: #94a3b8; font-size: 12px;")
        bottom_bar.addWidget(self.lbl_count)
        bottom_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.setStyleSheet("background-color: #27272a; color: white; padding: 6px 16px; border-radius: 4px;")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)

        main_layout.addLayout(bottom_bar)

    def _populate_categories(self):
        pass

    def _filter_tools(self):
        query = self.edit_search.text().lower().strip()
        cat_sel = self.cbo_cat_filter.currentText()

        # Clear existing layout items
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        visible_count = 0

        # Group by category
        categories = ["Reaction Mechanisms & Catalysis", "Spectroscopy & Analytical", "Orbitals, Wavefunction & Bonding", "Publishing & HPC Automation"]

        for cat in categories:
            if cat_sel != "All Categories" and cat_sel != cat:
                continue

            tools_in_cat = [
                t for t in HUB_TOOLS
                if t.category == cat and (
                    not query or query in t.title.lower() or query in t.description.lower() or query in t.shortcut.lower()
                )
            ]

            if not tools_in_cat:
                continue

            # Category Header
            cat_header = QtWidgets.QLabel(f"<b>{cat}</b>")
            cat_header.setStyleSheet("color: #38bdf8; font-size: 13px; margin-top: 8px; margin-bottom: 2px;")
            self.cards_layout.addWidget(cat_header)

            for tool in tools_in_cat:
                card = self._create_tool_card(tool)
                self.cards_layout.addWidget(card)
                visible_count += 1

        self.cards_layout.addStretch()
        self.lbl_count.setText(f"Showing {visible_count} of {len(HUB_TOOLS)} tools.")

    def _create_tool_card(self, tool: HubToolItem) -> QtWidgets.QFrame:
        card = QtWidgets.QFrame()
        card.setStyleSheet(
            "QFrame { background-color: #18181b; border: 1px solid #27272a; border-radius: 6px; padding: 8px; }"
            "QFrame:hover { border: 1px solid #0284c7; background-color: #1e293b; }"
        )
        c_lay = QtWidgets.QHBoxLayout(card)
        c_lay.setContentsMargins(10, 8, 10, 8)
        c_lay.setSpacing(12)

        # Icon
        lbl_icon = QtWidgets.QLabel(f"<span style='font-size: 24px;'>{tool.icon}</span>")
        lbl_icon.setFixedWidth(36)
        c_lay.addWidget(lbl_icon)

        # Middle: Title, Description, and Badges
        mid_lay = QtWidgets.QVBoxLayout()
        mid_lay.setSpacing(2)

        title_row = QtWidgets.QHBoxLayout()
        title_lbl = QtWidgets.QLabel(f"<b style='color: #f1f5f9; font-size: 13px;'>{tool.title}</b>")
        title_row.addWidget(title_lbl)

        badge_color = "#0284c7" if tool.badge == "Built-in" else "#8b5cf6" if tool.badge == "PySCF" else "#64748b"
        badge_lbl = QtWidgets.QLabel(f"<span style='background-color: {badge_color}; color: white; font-size: 11px; font-weight: bold; border-radius: 3px; padding: 1px 6px;'>{tool.badge}</span>")
        title_row.addWidget(badge_lbl)

        if tool.shortcut:
            sc_lbl = QtWidgets.QLabel(f"<span style='color: #94a3b8; font-size: 11px; background-color: #27272a; border-radius: 3px; padding: 1px 5px;'>{tool.shortcut}</span>")
            title_row.addWidget(sc_lbl)

        title_row.addStretch()
        mid_lay.addLayout(title_row)

        desc_lbl = QtWidgets.QLabel(f"<span style='color: #94a3b8; font-size: 11px;'>{tool.description}</span>")
        desc_lbl.setWordWrap(True)
        mid_lay.addWidget(desc_lbl)

        c_lay.addLayout(mid_lay, stretch=1)

        # Right: Launch Button
        btn_launch = QtWidgets.QPushButton("Launch →")
        btn_launch.setStyleSheet(
            "QPushButton { background-color: #0284c7; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #0369a1; }"
        )
        btn_launch.clicked.connect(lambda _, cb=tool.callback_name: self._launch_tool(cb))
        c_lay.addWidget(btn_launch)

        return card

    def _launch_tool(self, callback_name: str):
        self.accept()
        if hasattr(self.parent_mw, callback_name):
            method = getattr(self.parent_mw, callback_name)
            method()
        else:
            QtWidgets.QMessageBox.warning(self.parent_mw, "Tool Launch", f"Handler {callback_name} is being initialized.")
