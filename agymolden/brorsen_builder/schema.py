"""
Schema, option definitions, and validation rules for newBRORSEN input builder.
Extracted and modularized from input_builder.v04.08 and MANUAL.md.
Kept external and decoupled from the newBRORSEN wrapper engine.
"""

from typing import Dict, List, Tuple, Optional, Any, Set

_B = ("true", "false")

class UIOptionDef:
    """Definition of a directive or keyword for newBRORSEN input generation."""
    __slots__ = ("name", "aliases", "kind", "default", "sector", "doc", "choices")

    def __init__(self, name: str, aliases: Tuple[str, ...], kind: str, default: Any,
                 sector: Optional[str], doc: str, choices: Optional[Tuple[str, ...]] = None):
        self.name = name
        self.aliases = aliases
        self.kind = kind          # "text", "int", "float", "bool", "choice", "block", "scan", "memory"
        self.default = default
        self.sector = sector      # None (global), "POSITRON", "PROTON", "OPT", etc.
        self.doc = doc
        self.choices = choices


# Bang line options
RUN_TYPES = [
    ("SP_ENERGY", "Single Point Energy (SP / ENERGY)"),
    ("OPT", "Geometry Optimization (GEOMOPT / OPT)"),
    ("FREQ", "Harmonic Frequencies (HESSIAN / FREQ)"),
    ("GRADIENT", "Energy Gradient (GRAD / SP_GRAD)"),
    ("SCAN", "Potential Energy Surface Scan"),
    ("TS", "Transition State Search (OPTTS / TS)"),
    ("NUMFREQ", "Numerical Frequencies"),
    ("ANFREQ", "Analytical Frequencies"),
]

METHODS = [
    ("HF", "Hartree-Fock"),
    ("DFT", "Density Functional Theory (Electronic / Proton only)"),
    ("MP2", "Møller-Plesset 2nd order (DF-MP2)"),
    ("CCSD", "Coupled Cluster Singles and Doubles"),
    ("CCSD(T)", "Coupled Cluster Singles, Doubles and Perturbative Triples"),
]

SECTORS = [
    ("ELECTRONIC", "Electronic (Pure electrons)"),
    ("POSITRON", "Positronic (Electrons + Positrons)"),
    ("PROTON", "Protonic / NEO (Nuclear-Electronic Orbital)"),
]

INTEGRAL_MODES = [
    ("DF", "Density Fitting / RI"),
    ("CONV", "Conventional / Exact Integrals"),
]

COORD_UNITS = [
    ("Angs", "Ångströms (Angstrom)"),
    ("Bohrs", "Bohr (atomic units, a.u.)"),
]

OPT_ENGINES = [
    ("GEOMETRIC", "geomeTRIC (Default recommended optimizer)"),
    ("WOPT", "Built-in BFGS (wopt)"),
    ("ZOPT", "Z-matrix Coordinate Optimizer (Internal coords)"),
    ("AUTO", "Automatic engine selection"),
    ("INTRINSIC", "Intrinsic PySCF optimizer"),
    ("BFGS", "Standard SciPy BFGS"),
]

TS_MODES = [
    ("ts", "Transition State Search (Berny / geomeTRIC TS)"),
    ("irc", "Intrinsic Reaction Coordinate (IRC)"),
    ("neb", "Nudged Elastic Band (NEB)"),
]

ELECTRONIC_BASES = [
    "cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "aug-cc-pv5z",
    "def2-svp", "def2-tzvp", "def2-tzvpp", "def2-qzvp"
]

POSITRON_BASES = [
    "PSX-DZ", "PSX-TZ", "PSX-QZ", "PSX-5Z", "PSX-6Z", "MATCH"
]

PROTON_BASES = [
    "pb4f1", "pb4d", "pb5"
]

DFT_FUNCTIONALS = [
    "b3lyp", "pbe0", "pbe", "wb97x", "wb97x-d", "m06-2x", "cam-b3lyp", "scan"
]

EPC_FUNCTIONALS = [
    "17-2", "17-1", "18-1", "18-2", "none"
]

PARTICLE_MASSES = [
    "positron", "antimuon", "proton", "deuteron", "triton", "muonic_helium"
]

# Dependency graph: child_option -> (parent_option, set_of_allowed_values_or_None_for_any_set)
OPTION_DEPENDENCIES: Dict[str, Tuple[str, Optional[Set[str]]]] = {
    "irc_direction": ("ts_mode", {"irc"}),
    "neb_product": ("ts_mode", {"neb"}),
    "neb_images": ("ts_mode", {"neb"}),
    "neb_optimize_ends": ("ts_mode", {"neb"}),
    "zopt_max_evals": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_etol": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_step_bond": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_step_angle": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_tol_bond": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_tol_angle": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_gradient_every": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_final_gradient": ("opt_engine", {"zopt", "ZOPT"}),
    "zopt_restart": ("opt_engine", {"zopt", "ZOPT"}),
    "scan_fit": ("scan", None),
    "scan_print": ("scan", None),
    "cube_points": ("cube", {"true", "True", True}),
    "density_map_plane": ("density_map", None),
    "density_map_combine": ("density_map", None),
    "pos_mult": ("n_pos", None),
    "pos_spin": ("n_pos", None),
    "nuc_mass": ("quantum_nuc", None),
    "cneo_variant": ("cneo", {"true", "True", True}),
}
