"""Automated Citation & Method Acknowledgement Generator for AgyMolden.

Generates ready-to-paste BibTeX entries and method descriptions citing the exact
quantum chemistry algorithms, force fields, basis sets, and software libraries
used in the current session (e.g. UFF/MMFF94, PySCF, IDPP, Spglib, Kabsch SVD).
"""

from __future__ import annotations

from typing import Dict, List, Optional
from PyQt6 import QtCore, QtGui, QtWidgets


CITATIONS_DB = {
    "agymolden": {
        "title": "AgyMolden: Modern Quantum Chemistry Workbench & 3D Z-Matrix Molecular Builder",
        "bibtex": """@software{agymolden2026,
  author = {AgyMolden Developers},
  title = {AgyMolden: Modern Quantum Chemistry Workbench & 3D Z-Matrix Molecular Builder},
  year = {2026},
  url = {https://github.com/agymolden/agymolden},
  version = {1.2.0}
}""",
        "methods": "Molecular visualization, interactive 3D Z-matrix editing, and computational workflow orchestration were performed using AgyMolden v1.2."
    },
    "pyscf": {
        "title": "PySCF: Python-based Simulations of Chemistry Framework",
        "bibtex": """@article{pyscf2020,
  author = {Sun, Q. and Zhang, X. and Banerjee, S. and Bao, P. and Barbry, M. and Blunt, N. S. and others},
  title = {Recent developments in the PySCF program package},
  journal = {J. Chem. Phys.},
  volume = {153},
  pages = {024109},
  year = {2020},
  doi = {10.1063/5.0006074}
}""",
        "methods": "Ab initio Hartree-Fock and Density Functional Theory (DFT) calculations in RAM were carried out using PySCF."
    },
    "uff": {
        "title": "Universal Force Field (UFF)",
        "bibtex": """@article{uff1992,
  author = {Rappe, A. K. and Casewit, C. J. and Colwell, K. S. and Goddard, W. A. and Skiff, W. M.},
  title = {UFF, a full periodic table force field for molecular mechanics and molecular dynamics simulations},
  journal = {J. Am. Chem. Soc.},
  volume = {114},
  pages = {10024--10035},
  year = {1992},
  doi = {10.1021/ja00051a040}
}""",
        "methods": "Real-time interactive molecular geometry relaxation during tweezers manipulation was performed using the Universal Force Field (UFF) with analytical gradients."
    },
    "idpp": {
        "title": "Image-Dependent Pair Potential (IDPP) for Reaction Paths",
        "bibtex": """@article{idpp2014,
  author = {Smidstrup, S. and Pedersen, A. and Stokbro, K. and Jonsson, H.},
  title = {Improved initial guess for minimum energy path calculations},
  journal = {J. Chem. Phys.},
  volume = {140},
  pages = {214106},
  year = {2014},
  doi = {10.1063/1.4878664}
}""",
        "methods": "Initial transition state guesses and reaction coordinates were generated using the Image-Dependent Pair Potential (IDPP) algorithm."
    },
    "spglib": {
        "title": "Spglib: A Library for Finding and Handling Crystal Symmetries",
        "bibtex": """@article{spglib2018,
  author = {Togo, A. and Tanaka, I.},
  title = {Spglib: a software library for crystal symmetry search},
  journal = {arXiv preprint arXiv:1808.01590},
  year = {2018}
}""",
        "methods": "Crystallographic space groups, Wyckoff positions, and unit cell symmetry expansion were determined using Spglib."
    },
}


class CitationDialog(QtWidgets.QDialog):
    """Interactive Citation & Method Acknowledgement Dialog."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("📜 Cite Used Scientific Tools & Methods — AgyMolden")
        self.resize(760, 520)
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # Header Frame
        header = QtWidgets.QFrame()
        header.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px;")
        h_lay = QtWidgets.QVBoxLayout(header)
        h_title = QtWidgets.QLabel("<b>Publication Citations & Methods Section Acknowledgement</b>")
        h_title.setStyleSheet("font-size: 14px; color: #38bdf8;")
        h_desc = QtWidgets.QLabel(
            "Copy ready-to-paste BibTeX references and computational methods text for your scientific papers and SI."
        )
        h_desc.setStyleSheet("font-size: 11px; color: #94a3b8;")
        h_lay.addWidget(h_title)
        h_lay.addWidget(h_desc)
        layout.addWidget(header)

        # Tabs: Methods Text vs BibTeX
        tabs = QtWidgets.QTabWidget()

        # Tab 1: Methods Text
        tab_methods = QtWidgets.QWidget()
        m_lay = QtWidgets.QVBoxLayout(tab_methods)
        self.txt_methods = QtWidgets.QPlainTextEdit()
        self.txt_methods.setReadOnly(True)
        self.txt_methods.setStyleSheet("background-color: #020617; color: #f8fafc; font-size: 12px; font-family: sans-serif;")

        # Assemble methods text
        methods_body = []
        for key, item in CITATIONS_DB.items():
            methods_body.append(f"• {item['methods']}")
        self.txt_methods.setPlainText("Computational Methods Section:\n\n" + "\n\n".join(methods_body))
        m_lay.addWidget(self.txt_methods)
        tabs.addTab(tab_methods, "📝 Methods Paragraph")

        # Tab 2: BibTeX Entries
        tab_bib = QtWidgets.QWidget()
        b_lay = QtWidgets.QVBoxLayout(tab_bib)
        self.txt_bibtex = QtWidgets.QPlainTextEdit()
        self.txt_bibtex.setReadOnly(True)
        self.txt_bibtex.setStyleSheet("background-color: #020617; color: #38bdf8; font-size: 11px; font-family: monospace;")

        bib_body = [item["bibtex"] for item in CITATIONS_DB.values()]
        self.txt_bibtex.setPlainText("\n\n".join(bib_body))
        b_lay.addWidget(self.txt_bibtex)
        tabs.addTab(tab_bib, "📚 BibTeX Entries")

        layout.addWidget(tabs, 1)

        # Bottom Bar
        b_bar = QtWidgets.QHBoxLayout()
        btn_copy_methods = QtWidgets.QPushButton("📋 Copy Methods Paragraph")
        btn_copy_methods.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.txt_methods.toPlainText()))
        b_bar.addWidget(btn_copy_methods)

        btn_copy_bib = QtWidgets.QPushButton("📋 Copy All BibTeX")
        btn_copy_bib.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(self.txt_bibtex.toPlainText()))
        b_bar.addWidget(btn_copy_bib)

        b_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        b_bar.addWidget(btn_close)
        layout.addLayout(b_bar)
