"""
Categorized Chemical Fragment Library & User Custom Fragment Persistence.

Organizes built-in and expanded fragments into GaussView/Avogadro-style categories:
  - Functional Groups
  - Rings & Aromatics
  - Solvents & Small Molecules
  - Ligands & Coordination Groups
  - Custom User Fragments (persisted in ~/.config/AgyMolden/custom_fragments.json)

Automatically registers all fragments into `agymolden.fragments.FRAGMENTS` so all
existing code and tests referencing `FRAGMENTS[name]` continue to work seamlessly.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)

from .fragments import FRAGMENTS
from .topology import norm_bond


# Additional rich fragments to expand the built-in palette (Rings, Functional Groups, Solvents, Ligands)
EXPANDED_FRAGMENTS: Dict[str, Dict] = {
    # --- Functional Groups ---
    "Isopropyl (-CH(CH3)2)": {
        "category": "Functional Groups",
        "symbols": ["C", "C", "C", "H", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.250, 0.720, 0.510),
            (-1.250, 0.720, 0.510),
            (0.000, -1.030, 0.360),
            (1.250, 1.750, 0.150),
            (2.140, 0.210, 0.150),
            (1.250, 0.720, 1.600),
            (-1.250, 1.750, 0.150),
            (-2.140, 0.210, 0.150),
            (-1.250, 0.720, 1.600),
        ],
        "connect_idx": 0,
        "default_bond": 1.54,
    },
    "tert-Butyl (-C(CH3)3)": {
        "category": "Functional Groups",
        "symbols": ["C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.440, 0.510),
            (1.247, -0.720, 0.510),
            (-1.247, -0.720, 0.510),
            (0.000, 1.440, 1.600),
            (0.890, 1.950, 0.150),
            (-0.890, 1.950, 0.150),
            (1.247, -0.720, 1.600),
            (2.140, -0.210, 0.150),
            (1.250, -1.750, 0.150),
            (-1.247, -0.720, 1.600),
            (-2.140, -0.210, 0.150),
            (-1.250, -1.750, 0.150),
        ],
        "connect_idx": 0,
        "default_bond": 1.54,
    },
    "Vinyl (-CH=CH2)": {
        "category": "Functional Groups",
        "symbols": ["C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.160, 0.670),
            (0.000, -0.940, 0.540),
            (0.000, 2.100, 0.130),
            (0.000, 1.160, 1.755),
        ],
        "bond_orders": {(0, 1): 2},
        "connect_idx": 0,
        "default_bond": 1.50,
    },
    "Ethynyl (-C≡CH)": {
        "category": "Functional Groups",
        "symbols": ["C", "C", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 0.000, 1.205),
            (0.000, 0.000, 2.265),
        ],
        "bond_orders": {(0, 1): 3},
        "connect_idx": 0,
        "default_bond": 1.46,
    },
    "Aldehyde (-CHO)": {
        "category": "Functional Groups",
        "symbols": ["C", "O", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.050, 0.610),
            (0.000, -0.950, 0.550),
        ],
        "bond_orders": {(0, 1): 2},
        "connect_idx": 0,
        "default_bond": 1.50,
    },
    "Amide (-CONH2)": {
        "category": "Functional Groups",
        "symbols": ["C", "O", "N", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.060, 0.610),
            (0.000, -1.160, 0.670),
            (0.000, -2.030, 0.160),
            (0.000, -1.160, 1.680),
        ],
        "bond_orders": {(0, 1): 2},
        "connect_idx": 0,
        "default_bond": 1.51,
    },
    "Methoxy (-OCH3)": {
        "category": "Functional Groups",
        "symbols": ["O", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.220, 0.720),
            (0.000, 2.110, 0.090),
            (0.890, 1.220, 1.350),
            (-0.890, 1.220, 1.350),
        ],
        "connect_idx": 0,
        "default_bond": 1.41,
    },
    "Sulfonyl (-SO3H)": {
        "category": "Functional Groups",
        "symbols": ["S", "O", "O", "O", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.250, 0.720, 0.450),
            (-1.250, 0.720, 0.450),
            (0.000, -1.360, 0.780),
            (0.000, -2.080, 0.140),
        ],
        "bond_orders": {(0, 1): 2, (0, 2): 2},
        "connect_idx": 0,
        "default_bond": 1.78,
    },
    "Phosphate (-PO3H2)": {
        "category": "Functional Groups",
        "symbols": ["P", "O", "O", "O", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.380, 0.520),
            (1.280, -0.740, 0.620),
            (-1.280, -0.740, 0.620),
            (2.020, -0.250, 0.250),
            (-2.020, -0.250, 0.250),
        ],
        "bond_orders": {(0, 1): 2},
        "connect_idx": 0,
        "default_bond": 1.62,
    },

    # --- Rings & Aromatics ---
    "Cyclopentyl (-C5H9)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.210, 0.880, 0.320),
            (0.750, 2.310, 0.120),
            (-0.750, 2.310, 0.120),
            (-1.210, 0.880, 0.320),
            (0.000, -0.250, -1.060),
            (2.080, 0.620, -0.280),
            (1.480, 0.720, 1.360),
            (1.180, 2.980, 0.860),
            (1.060, 2.660, -0.860),
            (-1.180, 2.980, 0.860),
            (-1.060, 2.660, -0.860),
            (-2.080, 0.620, -0.280),
            (-1.480, 0.720, 1.360),
        ],
        "connect_idx": 0,
        "default_bond": 1.53,
    },
    "Furyl (-C4H3O)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "O", "C", "C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.110, 0.810, 0.000),
            (0.690, 2.110, 0.000),
            (-0.690, 2.110, 0.000),
            (-1.140, 0.780, 0.000),
            (1.380, 2.940, 0.000),
            (-1.380, 2.940, 0.000),
            (-2.170, 0.450, 0.000),
        ],
        "bond_orders": {(0, 4): 2, (2, 3): 2},
        "connect_idx": 0,
        "default_bond": 1.47,
    },
    "Thienyl (-C4H3S)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "S", "C", "C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.380, 1.020, 0.000),
            (0.710, 2.580, 0.000),
            (-0.710, 2.580, 0.000),
            (-1.140, 1.240, 0.000),
            (1.380, 3.430, 0.000),
            (-1.380, 3.430, 0.000),
            (-2.180, 0.940, 0.000),
        ],
        "bond_orders": {(0, 4): 2, (2, 3): 2},
        "connect_idx": 0,
        "default_bond": 1.47,
    },
    "Naphthyl (-C10H7)": {
        "category": "Rings & Aromatics",
        "symbols": [
            "C", "C", "C", "C", "C", "C", "C", "C", "C", "C",
            "H", "H", "H", "H", "H", "H", "H"
        ],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.210, 0.700, 0.000),
            (1.210, 2.100, 0.000),
            (0.000, 2.800, 0.000),
            (-1.210, 2.100, 0.000),
            (-1.210, 0.700, 0.000),
            (-2.420, 2.800, 0.000),
            (-3.630, 2.100, 0.000),
            (-3.630, 0.700, 0.000),
            (-2.420, 0.000, 0.000),
            (2.150, 0.160, 0.000),
            (2.150, 2.640, 0.000),
            (0.000, 3.880, 0.000),
            (-2.420, 3.880, 0.000),
            (-4.570, 2.640, 0.000),
            (-4.570, 0.160, 0.000),
            (-2.420, -1.080, 0.000),
        ],
        "connect_idx": 0,
        "default_bond": 1.48,
    },

    # --- Solvents & Small Molecules ---
    "Ammonia (NH3)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["N", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 0.940, 0.380),
            (0.814, -0.470, 0.380),
            (-0.814, -0.470, 0.380),
        ],
        "connect_idx": 0,
        "default_bond": 2.10,
    },
    "DMSO ((CH3)2SO)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["S", "O", "C", "C", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.480, 0.350),
            (1.380, -0.780, 0.850),
            (-1.380, -0.780, 0.850),
            (1.380, -1.850, 0.650),
            (2.310, -0.340, 0.490),
            (1.290, -0.610, 1.920),
            (-1.380, -1.850, 0.650),
            (-2.310, -0.340, 0.490),
            (-1.290, -0.610, 1.920),
        ],
        "bond_orders": {(0, 1): 2},
        "connect_idx": 0,
        "default_bond": 2.20,
    },
    "Dichloromethane (CH2Cl2)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["C", "Cl", "Cl", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.450, 0.000, 1.020),
            (-1.450, 0.000, 1.020),
            (0.000, 0.890, -0.630),
            (0.000, -0.890, -0.630),
        ],
        "connect_idx": 0,
        "default_bond": 1.54,
    },

    # --- Ligands & Coordination ---
    "Carbonyl Ligand (-CO)": {
        "category": "Ligands & Coordination",
        "symbols": ["C", "O"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 0.000, 1.135),
        ],
        "bond_orders": {(0, 1): 3},
        "connect_idx": 0,
        "default_bond": 1.90,
    },
    "Phosphine (-PH3)": {
        "category": "Ligands & Coordination",
        "symbols": ["P", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.190, 0.770),
            (1.030, -0.595, 0.770),
            (-1.030, -0.595, 0.770),
        ],
        "connect_idx": 0,
        "default_bond": 2.25,
    },
    "Triphenylphosphine (-PPh3)": {
        "category": "Ligands & Coordination",
        "symbols": ["P", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H",
                    "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H",
                    "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            # Ph 1
            (1.820, 0.000, 0.350), (2.520, 1.210, 0.350), (3.920, 1.210, 0.350), (4.620, 0.000, 0.350), (3.920, -1.210, 0.350), (2.520, -1.210, 0.350),
            (1.980, 2.150, 0.350), (4.460, 2.150, 0.350), (5.700, 0.000, 0.350), (4.460, -2.150, 0.350), (1.980, -2.150, 0.350),
            # Ph 2
            (-0.910, 1.576, 0.350), (-0.210, 2.786, 0.350), (-0.910, 4.000, 0.350), (-2.310, 4.000, 0.350), (-3.010, 2.786, 0.350), (-2.310, 1.576, 0.350),
            (0.870, 2.786, 0.350), (-0.370, 4.940, 0.350), (-2.850, 4.940, 0.350), (-4.090, 2.786, 0.350), (-2.850, 0.636, 0.350),
            # Ph 3
            (-0.910, -1.576, 0.350), (-2.310, -1.576, 0.350), (-3.010, -2.786, 0.350), (-2.310, -4.000, 0.350), (-0.910, -4.000, 0.350), (-0.210, -2.786, 0.350),
            (-2.850, -0.636, 0.350), (-4.090, -2.786, 0.350), (-2.850, -4.940, 0.350), (-0.370, -4.940, 0.350), (0.870, -2.786, 0.350),
        ],
        "connect_idx": 0,
        "default_bond": 2.30,
    },
    "Bipyridine (bpy)": {
        "category": "Ligands & Coordination",
        "symbols": ["C", "N", "C", "C", "C", "C", "H", "H", "H", "H",
                    "C", "N", "C", "C", "C", "C", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000), (0.700, 1.150, 0.000), (2.030, 1.150, 0.000), (2.730, 0.000, 0.000), (2.030, -1.150, 0.000), (0.700, -1.150, 0.000),
            (2.570, 2.090, 0.000), (3.810, 0.000, 0.000), (2.570, -2.090, 0.000), (0.160, -2.090, 0.000),
            (-1.480, 0.000, 0.000), (-2.180, -1.150, 0.000), (-3.510, -1.150, 0.000), (-4.210, 0.000, 0.000), (-3.510, 1.150, 0.000), (-2.180, 1.150, 0.000),
            (-4.050, -2.090, 0.000), (-5.290, 0.000, 0.000), (-4.050, 2.090, 0.000), (-1.640, 2.090, 0.000),
        ],
        "connect_idx": 1,
        "default_bond": 2.05,
    },
    "Cyclopentadienyl (Cp)": {
        "category": "Ligands & Coordination",
        "symbols": ["C", "C", "C", "C", "C", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 1.200, 0.000),
            (1.141, 0.371, 0.000),
            (0.705, -0.971, 0.000),
            (-0.705, -0.971, 0.000),
            (-1.141, 0.371, 0.000),
            (0.000, 2.280, 0.000),
            (2.168, 0.705, 0.000),
            (1.340, -1.845, 0.000),
            (-1.340, -1.845, 0.000),
            (-2.168, 0.705, 0.000),
        ],
        "connect_idx": 0,
        "default_bond": 2.10,
    },
    "Furan-2-yl (-C4H3O)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "O", "C", "C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (-0.800, 1.090, 0.000),
            (-2.090, 0.650, 0.000),
            (-2.150, -0.730, 0.000),
            (-0.870, -1.180, 0.000),
            (-2.870, 1.390, 0.000),
            (-3.030, -1.350, 0.000),
            (-0.550, -2.210, 0.000),
        ],
        "connect_idx": 0,
        "default_bond": 1.46,
    },
    "Thiophen-2-yl (-C4H3S)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "S", "C", "C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (-0.950, 1.430, 0.000),
            (-2.450, 0.600, 0.000),
            (-2.360, -0.780, 0.000),
            (-1.080, -1.140, 0.000),
            (-3.370, 1.170, 0.000),
            (-3.220, -1.440, 0.000),
            (-0.760, -2.170, 0.000),
        ],
        "connect_idx": 0,
        "default_bond": 1.46,
    },
    "Naphthyl (-C10H7)": {
        "category": "Rings & Aromatics",
        "symbols": ["C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (-0.700, 1.210, 0.000),
            (-2.100, 1.210, 0.000),
            (-2.800, 0.000, 0.000),
            (-2.100, -1.210, 0.000),
            (-0.700, -1.210, 0.000),
            (-2.800, 2.420, 0.000),
            (-4.200, 2.420, 0.000),
            (-4.900, 1.210, 0.000),
            (-4.200, 0.000, 0.000),
            (-0.160, 2.150, 0.000),
            (-2.640, -2.150, 0.000),
            (-0.160, -2.150, 0.000),
            (-2.260, 3.360, 0.000),
            (-4.740, 3.360, 0.000),
            (-5.980, 1.210, 0.000),
            (-4.740, -0.940, 0.000),
        ],
        "connect_idx": 0,
        "default_bond": 1.48,
    },
    "Dimethyl Sulfoxide (DMSO)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["S", "O", "C", "C", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.530, 0.000),
            (1.725, -0.510, 0.000),
            (-0.863, -0.510, 1.494),
            (2.260, -0.150, 0.880),
            (2.260, -0.150, -0.880),
            (1.725, -1.600, 0.000),
            (-0.863, -1.600, 1.494),
            (-1.890, -0.150, 1.494),
            (-0.370, -0.150, 2.370),
        ],
        "connect_idx": 0,
        "default_bond": 2.20,
    },
    "Dichloromethane (DCM)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["C", "Cl", "Cl", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (1.445, 0.000, -1.022),
            (-1.445, 0.000, -1.022),
            (0.000, 0.890, 0.629),
            (0.000, -0.890, 0.629),
        ],
        "connect_idx": 0,
        "default_bond": 1.54,
    },
    "Tetrahydrofuran (THF)": {
        "category": "Solvents & Small Molecules",
        "symbols": ["O", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.350),
            (1.150, 0.000, -0.470),
            (0.710, 0.000, -1.940),
            (-0.710, 0.000, -1.940),
            (-1.150, 0.000, -0.470),
            (1.770, 0.890, -0.270),
            (1.770, -0.890, -0.270),
            (1.180, 0.890, -2.480),
            (1.180, -0.890, -2.480),
            (-1.180, 0.890, -2.480),
            (-1.180, -0.890, -2.480),
            (-1.770, 0.890, -0.270),
            (-1.770, -0.890, -0.270),
        ],
        "connect_idx": 0,
        "default_bond": 2.00,
    },
    "Acryloyl (-COCH=CH2)": {
        "category": "Functional Groups",
        "symbols": ["C", "O", "C", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.220, 0.000),
            (1.320, -0.730, 0.000),
            (2.460, -0.050, 0.000),
            (1.320, -1.810, 0.000),
            (2.460, 1.030, 0.000),
            (3.420, -0.550, 0.000),
        ],
        "bond_orders": {(0, 1): 2, (2, 3): 2},
        "connect_idx": 0,
        "default_bond": 1.50,
    },
    "Methanesulfonyl / Mesyl (-SO2CH3)": {
        "category": "Functional Groups",
        "symbols": ["S", "O", "O", "C", "H", "H", "H"],
        "coords": [
            (0.000, 0.000, 0.000),
            (0.000, 1.440, 0.000),
            (1.360, -0.480, 0.000),
            (-0.780, -0.650, 1.450),
            (-0.780, -1.740, 1.450),
            (-0.250, -0.300, 2.330),
            (-1.810, -0.300, 1.450),
        ],
        "connect_idx": 0,
        "default_bond": 1.76,
    },
}

# Assign default categories to the original FRAGMENTS entries
_ORIGINAL_CATEGORIES: Dict[str, str] = {
    "Water (H2O)": "Solvents & Small Molecules",
    "Methanol (CH3OH)": "Solvents & Small Molecules",
    "Ethanol (C2H5OH)": "Solvents & Small Molecules",
    "Chloroform (CHCl3)": "Solvents & Small Molecules",
    "Acetonitrile (CH3CN)": "Solvents & Small Molecules",
    "Acetone ((CH3)2CO)": "Solvents & Small Molecules",
    "Methyl (-CH3)": "Functional Groups",
    "Ethyl (-C2H5)": "Functional Groups",
    "Hydroxyl (-OH)": "Functional Groups",
    "Amino (-NH2)": "Functional Groups",
    "Carbonyl (=O)": "Functional Groups",
    "Carboxyl (-COOH)": "Functional Groups",
    "Phenyl (-C6H5)": "Rings & Aromatics",
    "Pyridyl (-C5H4N)": "Rings & Aromatics",
    "Nitro (-NO2)": "Functional Groups",
    "Cyano (-CN)": "Functional Groups",
    "Trifluoromethyl (-CF3)": "Functional Groups",
    "Thiol (-SH)": "Functional Groups",
    "Chlorine (-Cl)": "Halogens & Simple",
    "Bromine (-Br)": "Halogens & Simple",
    "Fluorine (-F)": "Halogens & Simple",
}

FRAGMENT_CATEGORIES: List[str] = [
    "All Fragments",
    "Functional Groups",
    "Rings & Aromatics",
    "Solvents & Small Molecules",
    "Ligands & Coordination",
    "Halogens & Simple",
    "Custom User Library",
]


def _custom_fragments_path() -> str:
    """Return path to `~/.config/AgyMolden/custom_fragments.json`."""
    cfg_dir = os.path.join(os.path.expanduser("~"), ".config", "AgyMolden")
    return os.path.join(cfg_dir, "custom_fragments.json")


def initialize_fragment_library() -> None:
    """
    Merge `EXPANDED_FRAGMENTS` and saved user custom fragments into `FRAGMENTS`
    while tagging each entry with its `category`.
    """
    for name, data in FRAGMENTS.items():
        if "category" not in data:
            data["category"] = _ORIGINAL_CATEGORIES.get(name, "Functional Groups")

    for name, data in EXPANDED_FRAGMENTS.items():
        if name not in FRAGMENTS:
            FRAGMENTS[name] = dict(data)
        elif "category" not in FRAGMENTS[name]:
            FRAGMENTS[name]["category"] = data.get("category", "Functional Groups")

    load_custom_fragments_into_registry()


def get_fragments_by_category(category: str = "All Fragments") -> List[str]:
    """Return list of fragment names matching `category`."""
    initialize_fragment_library()
    if not category or category == "All Fragments":
        return list(FRAGMENTS.keys())
    return [
        name for name, data in FRAGMENTS.items()
        if data.get("category", "Functional Groups") == category
    ]


def load_custom_fragments_into_registry(path: Optional[str] = None) -> Dict[str, Dict]:
    """Load custom fragments from JSON file and register them into `FRAGMENTS`."""
    fpath = path or _custom_fragments_path()
    if not os.path.isfile(fpath):
        return {}
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        loaded: Dict[str, Dict] = {}
        for name, item in raw.items():
            entry = {
                "category": "Custom User Library",
                "symbols": list(item["symbols"]),
                "coords": [tuple(float(x) for x in pt) for pt in item["coords"]],
                "connect_idx": int(item.get("connect_idx", 0)),
                "default_bond": float(item.get("default_bond", 1.54)),
            }
            if "bonds" in item and item["bonds"] is not None:
                entry["bonds"] = [norm_bond(int(u), int(v)) for u, v in item["bonds"]]
            if "bond_orders" in item and item["bond_orders"] is not None:
                bo_dict = {}
                for k_str, order in item["bond_orders"].items():
                    if isinstance(k_str, str) and "-" in k_str:
                        u_s, v_s = k_str.split("-", 1)
                        bo_dict[norm_bond(int(u_s), int(v_s))] = int(order)
                entry["bond_orders"] = bo_dict
            FRAGMENTS[name] = entry
            loaded[name] = entry
        return loaded
    except Exception as e:
        log.warning(f"Failed to read custom fragments from {fpath}: {e}", exc_info=True)
        return {}


def save_custom_fragment(
    name: str,
    symbols: List[str],
    coords: np.ndarray,
    bonds: Optional[List[Tuple[int, int]]] = None,
    bond_orders: Optional[Dict[Tuple[int, int], int]] = None,
    connect_idx: int = 0,
    path: Optional[str] = None,
) -> Dict:
    """
    Save a molecule or selected subgraph as a reusable fragment in the Custom User Library
    (`~/.config/AgyMolden/custom_fragments.json`) and immediately register it in `FRAGMENTS`.
    """
    clean_name = name.strip()
    if not clean_name:
        raise ValueError("Fragment name cannot be empty.")

    pts = np.asarray(coords, dtype=np.float64)
    c_idx = int(connect_idx) if 0 <= int(connect_idx) < len(symbols) else 0
    rel_pts = pts - pts[c_idx]

    entry = {
        "category": "Custom User Library",
        "symbols": list(symbols),
        "coords": [tuple(float(x) for x in row) for row in rel_pts],
        "connect_idx": c_idx,
        "default_bond": 1.54,
    }
    if bonds is not None:
        entry["bonds"] = [norm_bond(int(u), int(v)) for u, v in bonds]
    if bond_orders:
        entry["bond_orders"] = {norm_bond(int(u), int(v)): int(o) for (u, v), o in bond_orders.items()}

    FRAGMENTS[clean_name] = entry

    # Persist to JSON
    fpath = path or _custom_fragments_path()
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    existing_raw = {}
    if os.path.isfile(fpath):
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                existing_raw = json.load(fh)
        except Exception as e:
            log.warning(f"Failed to parse existing JSON in {fpath}: {e}", exc_info=True)
            existing_raw = {}

    serializable = {
        "symbols": entry["symbols"],
        "coords": [list(pt) for pt in entry["coords"]],
        "connect_idx": entry["connect_idx"],
        "default_bond": entry["default_bond"],
        "bonds": [list(b) for b in entry.get("bonds", [])],
        "bond_orders": {
            f"{u}-{v}": int(o) for (u, v), o in (entry.get("bond_orders") or {}).items()
        },
    }
    existing_raw[clean_name] = serializable
    with open(fpath, "w", encoding="utf-8") as fh:
        json.dump(existing_raw, fh, indent=2)

    return entry


# Initialize library on import so FRAGMENTS is populated immediately
initialize_fragment_library()
