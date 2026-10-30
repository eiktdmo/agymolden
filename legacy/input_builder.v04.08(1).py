#!/usr/bin/env python3
"""input_builder v04.08 -- terminal input builder (prompt_toolkit) for newBRORSEN, split out of the
engine wrapper so it starts without importing numpy/pyscf (well under a second) and can be versioned,
tested and deployed on its own.

Drives the engine script configured in Settings > Engine path (default: a newBRORSEN.vNN.NN.py file of
the matching version next to this script), reaching it only through its command-line interface:
  * `<engine> <file> --decimal-comma`        to launch a job (ui_launch)
  * `<engine> <file> --dry-run --decimal-comma`  to dry-run a copy (ui_dry_run)
  * `<engine> --parse-text-json` (stdin)     to validate text without writing files (ui_deep_check)
  * `<engine> --view <file>`                 reached from "View molden/cube"
  * `runpy.run_path(<engine>)`               to load the engine's module namespace lazily, once, for the
    Molden/cube viewer and density-map/CHELPG reporting (ui_load_heavy)
None of this needs the engine file to be the one currently running; any matching-version newBRORSEN
engine script works.

Usage:
    python input_builder.v04.08.py [file.inp]   # open the input builder
    python input_builder.v04.08.py file.molden  # open the Molden/cube viewer directly
"""

import os
import sys
import re
import copy
import time
import io
import contextlib
import tempfile
import json

VERSION = "04.08"
# Signed off for production by the user on 2026-10-18 (replaces v04.06).
RELEASED = True

# ==== v04.08 UI: option table, lexer, static checks (stdlib only; above the numpy/pyscf imports) ====

class UIOpt:
    """One %directive for the input builder: completion, help and value checks."""
    __slots__ = ("name", "aliases", "value", "default", "sector", "doc", "choices")

    def __init__(self, name, aliases, value, default, sector, doc, choices=None):
        self.name, self.aliases, self.value, self.default = name, aliases, value, default
        self.sector, self.doc, self.choices = sector, doc, choices


_B = ("true", "false")
# value kinds: text, int, float, bool, choice, block (bare %name ... end), scan (block or one line), memory
UI_OPTIONS = [
    # general and resources
    UIOpt("jobname", (), "text", "Stage_N", None, "prefix of every output file of the stage"),
    UIOpt("charge", (), "int", "0", None, "electronic charge (positrons are not counted)"),
    UIOpt("mult", ("multiplicity",), "int", "1", None, "multiplicity 2S+1"),
    UIOpt("spin", (), "int", "0", None, "2S; sets %mult = spin + 1"),
    UIOpt("coords", ("coord_units", "units"), "choice", "angstrom", None,
          "units of this stage's geometry block", ("angstrom", "bohr", "angs", "bohrs", "au")),
    UIOpt("nprocs", ("threads", "cores"), "int", "4", None, "threads used by PySCF"),
    UIOpt("memory", (), "memory", "4000", None, "PySCF memory target in MB (16GB accepted)"),
    UIOpt("df_mem_cap", (), "float", "min(32, RAM/2)", None, "GB cap for blocked DF intermediates"),
    UIOpt("verbose", (), "int", "quiet", None, "PySCF verbosity"),
    # electronic
    UIOpt("basis", ("e_basis",), "text", "cc-pvdz", None, "electron basis set",
          ("cc-pvdz", "cc-pvtz", "cc-pvqz", "aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "aug-cc-pv5z",
           "def2-svp", "def2-tzvp")),
    UIOpt("frozen", (), "text", "0", None, "frozen-core orbitals: auto or an integer", ("auto", "0")),
    UIOpt("jkfit_aux_e", (), "text", "matched to %basis", None, "JK-fit set for the mean field, or auto"),
    UIOpt("auxbasis_e", (), "text", "<basis>-ri", None, "correlation fitting set, or default"),
    UIOpt("df_aux", (), "text", "{}", None, "per-element auxbasis_e (Python dict)"),
    UIOpt("xc", (), "text", "b3lyp", None, "DFT functional", ("b3lyp", "pbe0", "pbe", "wb97x", "m06-2x")),
    UIOpt("epc", (), "choice", "17-2", None, "electron-proton correlation functional",
          ("17-1", "17-2", "18-1", "18-2", "none")),
    UIOpt("compute_eee", (), "bool", "true", None, "compute the e-e-e three-body terms", _B),
    UIOpt("compute_t2e1p", (), "bool", "true", None, "compute the T2 e-1p terms", _B),
    UIOpt("block_size_a", (), "int", "library default", None, "block size of the DF contractions"),
    # positron
    UIOpt("n_pos", (), "int", "1", "POSITRON", "number of positrons"),
    UIOpt("pos_basis", ("positron_basis",), "text", "PSX-TZ", "POSITRON", "positron basis set",
          ("PSX-DZ", "PSX-TZ", "PSX-QZ", "PSX-5Z", "PSX-6Z", "MATCH")),
    UIOpt("psx_set", (), "text", "", "POSITRON", "PSX basis family selector"),
    UIOpt("pos_atom", (), "text", "all atoms", "POSITRON", "atoms carrying positron functions (symbols or numbers)"),
    UIOpt("pos_basis_map", (), "block", "", "POSITRON", "block: element basis ... end"),
    # any non-negative integer is valid (not just a closed-shell pair's 1/3 or 0/2): %n_pos 3 %pos_mult 4
    # is the spin-orbital POSSO route (odd/higher-spin positron counts), a real, supported code path.
    UIOpt("pos_mult", (), "int", "1", "POSITRON", "positron multiplicity (1, 3, 4, ... depending on n_pos)"),
    UIOpt("pos_spin", (), "int", "0", "POSITRON", "positron 2S (0, 2, 3, ... depending on n_pos)"),
    UIOpt("mp2_partition", (), "bool", "false", "POSITRON",
          "CC stage: also a DF-MP2 row in the electronic / positronic energy partition", _B),
    UIOpt("pos_aux", (), "text", "etb", "POSITRON", "positron fitting recipe", ("etb",)),
    UIOpt("pos_mass", ("particle", "particle_mass"), "text", "positron", "POSITRON",
          "light particle: name or mass in electron masses",
          ("positron", "antimuon", "proton", "deuteron", "triton", "muonic_helium")),
    UIOpt("quantum_site", (), "int", "none", "POSITRON", "atom number (1-based) of an H made quantum"),
    # proton / NEO
    UIOpt("nuc_basis", ("proton_basis",), "text", "pb4f1", "PROTON", "nuclear basis set", ("pb4f1", "pb4d")),
    UIOpt("quantum_nuc", (), "text", "0", "PROTON", "0-based quantum nuclei, comma-separated"),
    UIOpt("nuc_mass", (), "text", "library table", "PROTON", "one mass per %quantum_nuc entry"),
    UIOpt("cneo", (), "bool", "false", "PROTON", "constrained NEO", _B),
    UIOpt("cneo_variant", (), "choice", "ccd", None, "CNEO variant", ("ccd", "ucd")),
    # optimization
    UIOpt("opt_engine", ("optimizer",), "choice", "AUTO", None, "optimizer",
          ("AUTO", "INTRINSIC", "GEOMETRIC", "WOPT", "WRAPPER", "BFGS", "ZOPT", "ZMAT")),
    UIOpt("opt_max_cycles", ("max_opt_cycles", "max_steps"), "int", "50", None, "optimization cycles"),
    UIOpt("conv_check", (), "bool", "false", None, "DF-vs-CONV check of the final, reported OPT stage", _B),
    UIOpt("constraints", (), "block", "", None, "block of constraints ... end"),
    UIOpt("pin", ("pinning",), "text", "", None, "atoms fixed in place: 1,2"),
    UIOpt("pinnings", (), "block", "", None, "block: atom axis value ... end"),
    UIOpt("equivalence", ("equiv",), "text", "", None, "equivalent bonds a1,a2,b1,b2"),
    UIOpt("equivalences", (), "block", "", None, "block: target source sx sy sz ... end"),
    UIOpt("zopt_max_evals", (), "int", "60", None, "ZOPT energy evaluations"),
    UIOpt("zopt_etol", (), "float", "1e-6", None, "ZOPT energy tolerance (Hartree)"),
    UIOpt("zopt_step_bond", (), "float", "0.02", None, "ZOPT bond step (Angstrom)"),
    UIOpt("zopt_step_angle", (), "float", "2.0", None, "ZOPT angle step (degrees)"),
    UIOpt("zopt_tol_bond", (), "float", "0.001", None, "ZOPT bond tolerance (Angstrom)"),
    UIOpt("zopt_tol_angle", (), "float", "0.1", None, "ZOPT angle tolerance (degrees)"),
    UIOpt("zopt_gradient_every", (), "int", "0", None, "gradient every N ZOPT steps (0 = off)"),
    UIOpt("zopt_final_gradient", (), "bool", "false", None, "gradient at the ZOPT result", _B),
    UIOpt("zopt_restart", (), "bool", "true", None, "restart ZOPT from its checkpoint", _B),
    # scan
    UIOpt("scan", (), "scan", "", None,
          "scan: 'bond i j | angle i j k | dihedral i j k l | <zmat variable>' + 'start end n' or "
          "'[v1 v2 ...]' on one line, or the coordinate alone then segment rows ... end"),
    UIOpt("scan_fit", ("fit",), "choice", "none", None, "fit of the scan points",
          ("morse", "harmonic", "harm", "none", "false", "off", "no")),
    UIOpt("scan_print", (), "choice", "none", None, "extra per-point file: eda = every energy component",
          ("eda", "none")),
    # frequencies, TS, IRC, NEB
    UIOpt("freq_step", (), "float", "0.005", None, "finite-difference step (bohr)"),
    UIOpt("ts_mode", ("ts_type", "ts_search"), "choice", "ts", None, "transition-state calculation",
          ("ts", "prfo", "irc", "neb", "ci_neb")),
    UIOpt("ts_hessian", (), "text", "auto", None, "auto / first / each / never / stride:N / file:path",
          ("auto", "first", "each", "never")),
    UIOpt("ts_freq", (), "bool", "true", None, "frequency check of the TS", _B),
    UIOpt("irc_direction", (), "choice", "both", None, "IRC direction", ("both", "forward", "backward")),
    UIOpt("neb_product", (), "block", "", None, "block: product geometry (xyz lines) ... end"),
    UIOpt("neb_images", (), "int", "11", None, "NEB images (5-50)"),
    UIOpt("neb_optimize_ends", (), "bool", "false", None, "optimize the NEB end points", _B),
    # output
    UIOpt("molden", (), "bool", "false", None, "write molden files (HF/MP2/CCSD; CCSD natural orbitals)", _B),
    UIOpt("cube", (), "bool", "false", None, "write density cube files", _B),
    UIOpt("cube_points", (), "int", "80", None, "cube grid points (10-400)"),
    UIOpt("chelpg", (), "bool", "false", None, "CHELPG charges", _B),
    UIOpt("density_map", (), "text", "false", None, "one line per map: electron/positron/both/esp ...",
          ("electron", "positron", "both", "esp")),
    UIOpt("density_map_plane", (), "choice", "AUTO", None, "plane of the density maps", ("AUTO", "XY", "XZ", "YZ")),
    UIOpt("density_map_combine", (), "text", "false", None,
          "one figure of maps of earlier stages: job[:electron|positron] ... [level=hf|ccsd] [scale=shared|own]"),
]
UI_KEYS = {}
for _o in UI_OPTIONS:
    UI_KEYS[_o.name] = _o
    for _a in _o.aliases:
        UI_KEYS[_a] = _o
UI_BLOCK_KEYS = {"constraints", "pinnings", "equivalences", "pos_basis_map", "neb_product"}
UI_BOOL_WORDS = {"true", "yes", "1", "on", "t", "y", "false", "no", "0", "off", "f", "n"}
UI_POSITRON_KEYS = {"pos_basis", "n_pos", "pos_atom", "pos_aux", "pos_mass", "quantum_site", "pos_spin", "pos_mult",
                    "pos_basis_map", "pos_charge"}
UI_PROTON_KEYS = {"nuc_basis", "quantum_nuc", "nuc_mass", "cneo"}

# '!' words: word -> (group, canonical, one-line doc). Same groups as _bang_group further down.
UI_BANG = {}
for _words, _grp, _canon, _doc in (
        (("SP", "SP_ENERGY", "ENERGY"), "run type", "SP_ENERGY", "single-point energy (default)"),
        (("GRADIENT", "GRAD", "SP_GRAD"), "run type", "GRADIENT", "energy gradient"),
        (("OPT", "GEOMOPT", "OPTIMIZATION"), "run type", "OPT", "geometry optimization (optimizer: %opt_engine)"),
        (("FREQ", "FREQUENCIES", "HESSIAN"), "run type", "FREQ", "frequencies (analytic Hessian where possible)"),
        (("ANFREQ",), "run type", "FREQ", "frequencies, analytic Hessian only"),
        (("NUMFREQ",), "run type", "FREQ", "frequencies, numerical Hessian"),
        (("TS", "OPTTS"), "run type", "TS", "transition state family (choose with %ts_mode)"),
        (("SCAN",), "run type", "SCAN", "potential-energy scan (needs %scan)"),
        (("ELECTRONIC", "ELEC"), "sector", "ELECTRONIC", "electrons only (default)"),
        (("POSITRON", "POS"), "sector", "POSITRON", "electrons + positron(s)"),
        (("PROTON", "NEO"), "sector", "PROTON", "quantum protons (NEO)"),
        (("CNEO",), "sector", "PROTON", "constrained NEO (PROTON with %cneo true)"),
        (("CDFT",), "sector", "PROTON", "constrained NEO-DFT (PROTON, DFT, %cneo true)"),
        (("HF",), "method", "HF", "Hartree-Fock (default)"),
        (("DFT",), "method", "DFT", "density functional theory (%xc)"),
        (("MP2",), "method", "MP2", "MP2"),
        (("CCSD",), "method", "CCSD", "CCSD"),
        (("CCSD(T)",), "method", "CCSD(T)", "CCSD(T)"),
        (("DF", "RI"), "integrals", "DF", "density-fitted integrals (default)"),
        (("CONV", "EXACT"), "integrals", "CONV", "conventional integrals"),
        (("ANGS", "ANGSTROM", "ANGSTROMS", "ANG"), "units", "ANGSTROM", "geometry in Angstrom (default)"),
        (("BOHR", "BOHRS", "AU", "A.U."), "units", "BOHR", "geometry in Bohr")):
    for _w in _words:
        UI_BANG[_w] = (_grp, _canon, _doc)
UI_REMOVED_BANG = {
    "IRC": "'! TS' with '%ts_mode irc'", "NEB": "'! TS' with '%ts_mode neb'", "CI_NEB": "'! TS' with '%ts_mode neb'",
    "SADDLE": "'! TS'", "TS_OPT": "'! TS'",
    "MORSE_SCAN": "'! SCAN' with '%scan_fit morse'", "MORSESCAN": "'! SCAN' with '%scan_fit morse'",
    "HARMONIC_SCAN": "'! SCAN' with '%scan_fit harmonic'", "HARM_SCAN": "'! SCAN' with '%scan_fit harmonic'",
    "HARMONICSCAN": "'! SCAN' with '%scan_fit harmonic'",
    "WOPT": "'! OPT' with '%opt_engine wopt'", "WRAPPER_OPT": "'! OPT' with '%opt_engine wopt'",
    "INTRINSIC_OPT": "'! OPT' with '%opt_engine geometric'", "ZOPT": "'! OPT' with '%opt_engine zopt'",
    "ZMAT_OPT": "'! OPT' with '%opt_engine zopt'"}
UI_FREQ_PAIRS = ({"OPT", "FREQ"}, {"TS", "FREQ"})
_UI_ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
                "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm "
                "Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No "
                "Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()
UI_Z = {s.upper(): i + 1 for i, s in enumerate(_UI_ELEMENTS)}


# What a new input starts from (--ui, New input), and the suggestions shown in gray while typing (Right arrow
# accepts). Ranked from the project's own inputs (results/**/*.inp, 2026-10-08): the '!' combinations and the
# values used most often.
# ---- full template (--ui, New input) ----
# A new input starts with every '!' word and every option in gray ("ghost" text). Tab on a gray '!' word makes
# it real and removes the other words of its group; Tab on a gray option line (or block) makes it real; options
# that cannot apply to the chosen run type, sector or method disappear. Saving drops what is still gray.
UI_TEMPLATE_HEAD = """\
# ==============================================================================
# Description: 
#
#   Gray text is the template: it is not part of the input until you accept it.
#   '!' line : put the cursor on a gray word and press Tab to choose it (the other
#              words of its group disappear: one run type, sector, method, ...)
#   options  : Tab on a gray line keeps it; edit its value; gray lines left are dropped on save
#   geometry : F2 starts the builder; '%next' starts a stage that continues this one
# ==============================================================================
"""
UI_TPL_BANG = ["SP", "GRADIENT", "OPT", "FREQ", "TS", "SCAN", "ELECTRONIC", "POSITRON", "PROTON", "CNEO",
               "HF", "DFT", "MP2", "CCSD", "CCSD(T)", "DF", "CONV"]       # units: %coords, not '!' words
# v04.08: %pos_basis takes atoms directly ('%pos_basis PSX-TZ H PSX-DZ'), folding in %pos_atom and
# %pos_basis_map (MANUAL 8.2 'Atoms on the basis line'); USER_GUIDE.md no longer documents either. The
# engine still accepts both, so they stay in UI_OPTIONS/UI_KEYS and still parse and validate - just not
# offered in the template or completion (ui_suggest_keys, UI_TPL_SKIP below).
UI_DEPRECATED_KEYS = {"pos_atom", "pos_basis_map"}
UI_TPL_SKIP = {"charge", "mult", "spin", "psx_set"} | UI_DEPRECATED_KEYS  # set by the geometry header, or legacy
# example values (syntax) for options whose default is not a literal value
UI_VALUE_SYNTAX = {"jobname": "job", "pin": "1,2", "equivalence": "1,2,3,4", "pos_atom": "1 2",
                   "quantum_nuc": "0", "nuc_mass": "default", "quantum_site": "2", "density_map": "electron",
                   "df_aux": "{'auxbasis_e': 'aug-cc-pvtz-ri'}", "jkfit_aux_e": "auto", "auxbasis_e": "default",
                   "scan": "2.0 4.0 9", "df_mem_cap": "32", "block_size_a": "20", "verbose": "4",
                   "frozen": "auto", "ts_hessian": "auto"}
# block options: the syntax rows shown (gray) under them
UI_BLOCK_SYNTAX = {
    "constraints": ["bond 1 2 1.10          # bond i j [A]; angle i j k [deg]; dihedral i j k l [deg]"],
    "pinnings": ["1 z 0.0                 # atom axis(x/y/z or 0/1/2) value: set, then hold"],
    "equivalences": ["2 1 1.0 1.0 -1.0       # target source sx sy sz (mirror/copy of the source atom)"],
    "pos_basis_map": [],   # no example row: a wrong element/basis guess here would silently ship as real
    "neb_product": ["H 0.0 0.0 0.0           # product geometry: same atoms, same order"],
    "scan": ["bond 1 2 1.5 3.0 9      # bond i j start end n | translate i k axis start end n | "
             "sym_zmat var start end n  (v04.08: or '%scan bond 1 2' then 'start end n' rows)"],
}
_UI_OPT_KEYS = {"opt_engine", "opt_max_cycles", "conv_check", "constraints", "pin", "pinnings", "equivalence",
                "equivalences", "zopt_max_evals", "zopt_etol", "zopt_step_bond", "zopt_step_angle", "zopt_tol_bond",
                "zopt_tol_angle", "zopt_gradient_every", "zopt_final_gradient", "zopt_restart"}
_UI_TS_KEYS = {"ts_mode", "ts_hessian", "ts_freq", "irc_direction", "neb_product", "neb_images",
               "neb_optimize_ends"}
_UI_DF_KEYS = {"jkfit_aux_e", "auxbasis_e", "df_aux", "df_mem_cap", "block_size_a", "pos_aux", "conv_check"}


def ui_value_example(o):
    """The value a template line / a suggestion shows for option o: an example for syntax-type options, else
    the wrapper's default when it is a literal value (for on/off options, the value that is not the default)."""
    if o.name in UI_VALUE_SYNTAX:
        return UI_VALUE_SYNTAX[o.name]
    d = o.default or ""
    if o.value == "bool":                 # on/off: writing the default changes nothing, so offer the other value
        return "false" if d.lower() == "true" else "true"
    if not d or d.startswith(("<", "Stage_", "library", "min(", "all ", "matched", "quiet", "none")) or " " in d:
        return (o.choices or ("",))[0]
    return d


def ui_option_lines(o):
    """Template lines of option o (a block gets its syntax rows and a commented 'end')."""
    if o.value in ("block", "scan") and o.name in UI_BLOCK_SYNTAX:
        return [f"%{o.name}"] + UI_BLOCK_SYNTAX[o.name] + [f"end{' ' * 21}# end of %{o.name}"]
    first = f"%{o.name} {ui_value_example(o)}".rstrip()
    return [f"{first:<23s} # {o.doc}"]


def ui_option_line(name, value, note):
    first = f"%{name} {value}".rstrip()
    return f"{first:<23s} # {note}"


# Options shown only once another one is chosen: child -> (parent, values of the parent that show it, or None
# for "any value once the parent is kept").
UI_TPL_DEPENDS = {
    "irc_direction": ("ts_mode", {"irc"}), "neb_product": ("ts_mode", {"neb"}),
    "neb_images": ("ts_mode", {"neb"}), "neb_optimize_ends": ("ts_mode", {"neb"}),
    "zopt_max_evals": ("opt_engine", {"zopt"}), "zopt_etol": ("opt_engine", {"zopt"}),
    "zopt_step_bond": ("opt_engine", {"zopt"}), "zopt_step_angle": ("opt_engine", {"zopt"}),
    "zopt_tol_bond": ("opt_engine", {"zopt"}), "zopt_tol_angle": ("opt_engine", {"zopt"}),
    "zopt_gradient_every": ("opt_engine", {"zopt"}), "zopt_final_gradient": ("opt_engine", {"zopt"}),
    "zopt_restart": ("opt_engine", {"zopt"}),
    "scan_fit": ("scan", None), "scan_print": ("scan", None), "cube_points": ("cube", {"true"}), "density_map_plane": ("density_map", None),
    "density_map_combine": ("density_map", None),
    # offered whenever %n_pos is kept at all, not just at exactly 2: n_pos >= 2 with any pos_mult/pos_spin
    # (odd counts included, e.g. %n_pos 3 %pos_mult 4) is a real, supported code path, not just the
    # closed-shell pair's 1/3 (or 0/2); the real validator warns if pos_mult/pos_spin is set with n_pos == 1.
    "pos_mult": ("n_pos", None), "pos_spin": ("n_pos", None), "nuc_mass": ("quantum_nuc", None),
    "cneo_variant": ("cneo", {"true"}),
}
# Basis-set-like options each sector needs to consciously set rather than silently accept the default for
# (every one has a real default, so this is a nudge, not an error, and only once a sector has actually been
# chosen): while still gray and offered, a W replaces the usual gutter character on that row (same gutter
# class as _ui_stage_rules' warnings, which flag it on the stage's "!" line once the row is no longer
# offered, e.g. an opened file that never had it).
UI_TPL_NEEDED = {"basis", "pos_basis", "nuc_basis", "quantum_nuc"}
# Options offered as one gray line per choice (Tab on one keeps it and removes the others).
UI_TPL_ALTERNATIVES = {
    "ts_mode": [("ts", "transition-state search (P-RFO)"), ("irc", "IRC from a transition state"),
                ("neb", "nudged elastic band (needs %neb_product)")],
}
# Values offered (with what they mean) when completing an option's value, like the basis sets.
UI_CHOICE_DOCS = {
    "opt_engine": [("AUTO", "the wrapper chooses (default)"), ("GEOMETRIC", "geomeTRIC, analytic gradients"),
                   ("WOPT", "the wrapper's BFGS (works with numerical gradients)"),
                   ("ZOPT", "Z-matrix, gradient-free (needs a * zmat block)")],
    "coords": [("angstrom", "geometry in Angstrom (default)"), ("bohr", "geometry in Bohr")],
    "ts_mode": [("ts", "transition-state search, P-RFO (default)"), ("irc", "IRC from a transition state"),
                ("neb", "nudged elastic band (needs %neb_product)")],
}
_UI_NORM_VALUE = {"ts_mode": {"prfo": "ts", "search": "ts", "saddle": "ts", "optts": "ts", "ci_neb": "neb",
                              "cineb": "neb"},
                  "opt_engine": {"zmat": "zopt", "geometric": "intrinsic", "wrapper": "wopt", "bfgs": "wopt"}}
UI_TPL_GEOMETRY = ["* xyz 0 1               # example geometry (H2): edit it, or F2 builds your own",
                   "H  0.000 0.000 0.000    # example atom",
                   "H  0.000 0.000 0.740    # example atom",
                   "*                       # end of the example geometry"]
_UI_WORD_OF = {"SP_ENERGY": "SP", "ANGSTROM": "ANGS"}


def _ui_norm(name, value):
    v = (str(value).split() or [""])[0].lower()
    if name in ("cube", "cneo", "molden", "chelpg"):
        return "true" if v in ("true", "yes", "1", "on", "t", "y") else "false"
    return _UI_NORM_VALUE.get(name, {}).get(v, v)


def ui_tpl_order(geom_first=True):
    """Canonical order of the template items (ids): the geometry first by default (easiest to find and
    edit without scrolling), or last (Settings > Geometry position: bottom) - the options, each child
    right after its parent, alternatives as 'name=value', either way. The wrapper's own parser does not
    care about this order within a stage (it just scans '!'/'%'/'*' lines wherever they are), so this is
    purely a display choice; a stage's '%next' line is not an item and sits wherever it is typed, always
    after everything the stage has written for real."""
    ids = ["geometry"] if geom_first else []
    for o in UI_OPTIONS:
        if o.name in UI_TPL_SKIP or o.name in UI_TPL_DEPENDS:
            continue
        ids += [f"{o.name}={v}" for v, _ in UI_TPL_ALTERNATIVES[o.name]] if o.name in UI_TPL_ALTERNATIVES \
            else [o.name]
        ids += [c for c, (p, _) in UI_TPL_DEPENDS.items() if p == o.name]
    return ids if geom_first else ids + ["geometry"]


def ui_tpl_stage_range(lines, stage_ord):
    """Rows [start, end) of the template's stage: after the stage_ord-th %next (0 = the first stage)."""
    nexts = [i for i, l in enumerate(lines) if re.match(r"^\s*%(next|newjob)\b", l, re.I)]
    start = nexts[stage_ord - 1] + 1 if 0 < stage_ord <= len(nexts) else 0
    end = next((i for i in nexts if i >= start), len(lines))
    return start, end


def ui_tpl_bang_row(lines, rng=None):
    a, b = rng or (0, len(lines))
    return next((i for i in range(a, b) if lines[i].lstrip().startswith("!")), None)


def ui_tpl_new(stage_no=1, prev=None, geom_first=True):
    """Template state of a stage. prev = {'bang': {group: canon}, 'keys': {name: value}, 'jobname': str} of
    the stage it continues (None for the first stage)."""
    words = list(UI_TPL_BANG)
    hl = set()
    if prev:
        for grp, canon in prev["bang"].items():
            for c in (prev.get("runs") or [canon]) if grp == "run type" else [canon]:
                w = _UI_WORD_OF.get(c, c)
                if w in words:
                    hl.add(w)
    return {"stage_ord": stage_no - 1, "stage_no": stage_no, "items": {}, "hl": set(), "bang": set(words),
            "bang_hl": hl, "prev": prev, "geom_first": geom_first}


def ui_tpl_chosen(lines, tpl):
    """{group: [canon]} on the template's '!' line: the words kept, and for groups with none kept, what the
    stage inherits (prev)."""
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    r = ui_tpl_bang_row(lines, rng)
    got = {}
    if r is not None:
        for w in _ui_strip_comment(lines[r]).lstrip()[1:].split():
            up = w.upper()
            if up in UI_BANG and up not in tpl["bang"]:
                grp, canon, _ = UI_BANG[up]
                got.setdefault(grp, []).append(canon)
                if up == "CNEO":
                    got.setdefault("_cneo", ["true"])
    if tpl.get("prev"):
        for grp, canon in tpl["prev"]["bang"].items():
            if grp not in got and not (grp == "run type" and canon == "SCAN"):
                got[grp] = list(tpl["prev"].get("runs") or [canon]) if grp == "run type" else [canon]
    return got


def ui_tpl_option_fits(name, chosen):
    """False when option `name` cannot apply to the run type / sector / method / integrals chosen."""
    runs = chosen.get("run type")
    sector = (chosen.get("sector") or [None])[0]
    method = (chosen.get("method") or [None])[0]
    integ = (chosen.get("integrals") or [None])[0]
    o = UI_KEYS.get(name)
    if sector and o is not None and o.sector and o.sector != sector:
        return False
    if name in ("compute_eee", "compute_t2e1p") and not (method == "CCSD(T)" and sector in (None, "POSITRON")):
        return False                    # only for CCSD(T) (positron triples terms)
    if sector and name in ("cneo_variant", "epc") and sector != "PROTON":
        return False
    if method and name == "xc" and method != "DFT":
        return False
    if method and name == "epc" and method != "DFT":
        return False
    if integ == "CONV" and name in _UI_DF_KEYS:
        return False
    if name in ("density_map", "density_map_plane", "density_map_combine", "molden", "cube", "cube_points",
                "chelpg") and (
            sector == "PROTON" or (runs and "SCAN" in runs)):
        return False
    if runs:
        if name in _UI_OPT_KEYS and "OPT" not in runs:
            return False
        if name in ("scan", "scan_fit") and "SCAN" not in runs:
            return False
        if name == "freq_step" and "FREQ" not in runs:
            return False
        if name in _UI_TS_KEYS and "TS" not in runs:
            return False
    return True


def ui_option_dep_fits(name, stage):
    """False when option `name` has a UI_TPL_DEPENDS parent that isn't set (to a matching value) on `stage`
    yet - e.g. %irc_direction before %ts_mode irc is written. True for options with no such dependency."""
    dep = UI_TPL_DEPENDS.get(name)
    if not dep or stage is None:
        return True
    parent, values = dep
    if parent not in stage.keys:
        return False
    return values is None or _ui_norm(parent, stage.keys[parent]) in values


def ui_stage_chosen(stage):
    """A UIStage's run type(s) / sector / method / integrals, shaped like `ui_tpl_chosen`'s result so
    `ui_tpl_option_fits` can gate the live '%' completion menu the same way it gates the new-stage template."""
    if stage is None:
        return {}
    return {"run type": sorted(stage.run_types) or [stage.run_type], "sector": [stage.sector],
            "method": [stage.method], "integrals": [stage.bang.get("integrals", "DF")]}


def _ui_real_options(lines, rng, ghost_rows):
    """{canonical option: value} written for real in the stage (gray rows excluded)."""
    out = {}
    for i in range(*rng):
        if i in ghost_rows:
            continue
        m = re.match(r"^\s*%\s*(\w+)\s*(?:=\s*|\s+|$)(.*)$", _ui_strip_comment(lines[i]))
        if m and m.group(1).lower() in UI_KEYS:
            out[UI_KEYS[m.group(1).lower()].name] = m.group(2).strip()
    return out


def ui_tpl_ghost_rows(lines, tpl):
    """{row: item id} of the gray item lines in the template's stage."""
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    by_line = {}
    for iid, ls in tpl["items"].items():
        for l in ls:
            by_line[l] = iid
    return {i: by_line[lines[i]] for i in range(*rng) if lines[i] in by_line}


def ui_tpl_desired(lines, tpl):
    """[(id, lines, highlighted)] the stage should show in gray, in canonical order."""
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    ghost_rows = ui_tpl_ghost_rows(lines, tpl)
    real = _ui_real_options(lines, rng, ghost_rows)
    chosen = ui_tpl_chosen(lines, tpl)
    # for hiding not-yet-fitting ghost suggestions only: a group with nothing kept (and nothing inherited)
    # defaults to what the wrapper itself defaults to (HF, ELECTRONIC, SP, DF), so a brand-new template does
    # not show every sector's/method's options at once - unlike `chosen` itself, which some callers (the
    # gutter's missing-basis-option nudge) need to see as truly empty until a word is actually kept.
    chosen_fits = {"run type": chosen.get("run type") or ["SP_ENERGY"], "sector": chosen.get("sector") or ["ELECTRONIC"],
                   "method": chosen.get("method") or ["HF"], "integrals": chosen.get("integrals") or ["DF"]}
    prev = tpl.get("prev") or {"keys": {}, "jobname": None, "bang": {}}
    vals = dict(prev["keys"])
    vals.update(real)
    if chosen.get("_cneo"):
        vals["cneo"] = "true"
    has_geom = any(lines[i].lstrip().startswith("*") for i in range(*rng) if i not in ghost_rows)
    out = []
    dismissed = tpl.get("dismissed") or set()
    for iid in ui_tpl_order(tpl.get("geom_first", True)):
        if iid in dismissed:
            continue
        if iid == "geometry":
            if tpl["stage_no"] == 1 and not has_geom:
                out.append(("geometry", list(UI_TPL_GEOMETRY), False))
            continue
        name, _, alt = iid.partition("=")
        if name in real or not ui_tpl_option_fits(name, chosen_fits):
            continue
        if name in UI_TPL_DEPENDS:
            parent, need = UI_TPL_DEPENDS[name]
            if parent not in vals or (need is not None and _ui_norm(parent, vals[parent]) not in need):
                continue
        o = UI_KEYS[name]
        before = prev["keys"].get(name)
        if alt:
            note = dict(UI_TPL_ALTERNATIVES[name])[alt]
            hl = before is not None and _ui_norm(name, before) == _ui_norm(name, alt)
            out.append((iid, [ui_option_line(name, alt, note + ("  (stage before)" if hl else ""))], hl))
        elif name == "jobname" and prev.get("jobname"):
            out.append((iid, [ui_option_line(name, f"{prev['jobname']}_{tpl['stage_no']}",
                                             "this stage's output files")], False))
        elif before not in (None, "") and o.value not in ("block", "scan"):
            out.append((iid, [ui_option_line(name, before, f"as in stage {tpl['stage_no'] - 1}")], True))
        else:
            out.append((iid, ui_option_lines(o), False))
    return out


def _ui_anchor_end(lines, i, rng):
    """Last row of the real thing that starts at row i (a block's 'end', a geometry's '*')."""
    code = _ui_strip_comment(lines[i]).strip().lower()
    if code.startswith("*"):
        return next((j for j in range(i + 1, rng[1]) if _ui_strip_comment(lines[j]).strip() in ("*", "end")), i)
    m = re.match(r"^%\s*(\w+)\s*$", code)
    if m and (m.group(1) in UI_BLOCK_KEYS or m.group(1) == "scan"):
        return next((j for j in range(i + 1, rng[1]) if _ui_strip_comment(lines[j]).strip().lower() == "end"), i)
    return i


def ui_tpl_reconcile(lines, tpl, cursor_row=0):
    """Rebuild the gray items of the template's stage: remove those that no longer fit, add the ones that now
    fit (children of a kept option, ...), each at its canonical place. Real lines are never moved. Returns
    (lines, tpl, new cursor row)."""
    lines = list(lines)
    ghost_rows = ui_tpl_ghost_rows(lines, tpl)
    desired = ui_tpl_desired(lines, tpl)
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    # 1. drop every gray item line of the stage (remember the cursor's line)
    keep = [(i, l) for i, l in enumerate(lines) if i not in ghost_rows]
    cur = next((k for k, (i, _) in enumerate(keep) if i >= cursor_row), len(keep) - 1)
    lines = [l for _, l in keep]
    tpl = dict(tpl, items={iid: ls for iid, ls, _ in desired}, hl={iid for iid, _, h in desired if h})
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    # 2. anchors: real lines of the stage that belong to an item id (option lines, geometry)
    order = ui_tpl_order(tpl.get("geom_first", True))
    pos = {iid: k for k, iid in enumerate(order)}
    anchors = {}
    for i in range(*rng):
        code = _ui_strip_comment(lines[i]).strip()
        if code.startswith("*") and code[1:2] != "":
            anchors.setdefault("geometry", i)
        m = re.match(r"^%\s*(\w+)", code)
        if m and m.group(1).lower() in UI_KEYS:
            name = UI_KEYS[m.group(1).lower()].name
            alts = [k for k in order if k.startswith(name + "=")]
            anchors.setdefault(alts[0] if alts else name, i)
    bang = ui_tpl_bang_row(lines, rng)
    base = bang if bang is not None else rng[0] - 1
    # 3. insert the desired items in canonical order, after the previous item present
    inserted = 0
    last_end = base
    for iid in order:
        if iid in anchors:
            last_end = max(last_end, _ui_anchor_end(lines, anchors[iid], (rng[0], rng[1] + inserted)))
            continue
        item = next((ls for d, ls, _ in desired if d == iid), None)
        if item is None:
            continue
        at = last_end + 1
        lines[at:at] = item
        for k, v in list(anchors.items()):
            if v >= at:
                anchors[k] = v + len(item)
        if cur >= at:
            cur += len(item)
        inserted += len(item)
        last_end = at + len(item) - 1
    return lines, tpl, max(0, min(cur, len(lines) - 1))


def ui_tpl_start(lines, after_row, stage_no, prev=None, geom_first=True):
    """Insert the template of a stage after row after_row ('-1' = top): its '!' line with every word (the
    words of the stage it continues highlighted), then its gray items. Returns (lines, tpl, bang row)."""
    tpl = ui_tpl_new(stage_no, prev, geom_first)
    lines = list(lines)
    order = UI_TPL_BANG
    lines.insert(after_row + 1, "! " + " ".join(order))
    lines, tpl, _ = ui_tpl_reconcile(lines, tpl, after_row + 1)
    return lines, tpl, after_row + 1


def ui_tpl_accept(lines, row, col, tpl):
    """Tab at (row, col): keep the gray '!' word there (its group's other words go) or the gray item on that
    row. Then rebuild the stage's gray items. Returns (lines, tpl, (row, col)) or None."""
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    if not rng[0] <= row < rng[1]:
        return None
    bang = ui_tpl_bang_row(lines, rng)
    tpl = dict(tpl, bang=set(tpl["bang"]), items=dict(tpl["items"]))
    if row == bang and tpl["bang"]:
        line = lines[row]
        spans = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\s!]+", line)]
        pick = next((sp for sp in spans if sp[0] <= col <= sp[1] and sp[2].upper() in tpl["bang"]), None) or \
            next((sp for sp in spans if sp[0] >= col and sp[2].upper() in tpl["bang"]), None)
        if pick is None:
            return None
        word = pick[2].upper()
        grp, canon = UI_BANG[word][0], UI_BANG[word][1]
        tpl["bang"].discard(word)
        keep = []
        for a, b, w in spans:
            up = w.upper()
            if up in tpl["bang"] and UI_BANG[up][0] == grp:
                if grp == "run type" and {canon, UI_BANG[up][1]} in UI_FREQ_PAIRS:
                    keep.append(w)
                    continue
                tpl["bang"].discard(up)
                continue
            keep.append(w)
        lines = list(lines)
        lines[row] = "! " + " ".join(keep)
        lines, tpl, r = ui_tpl_reconcile(lines, tpl, row)
        new = lines[r]
        p = new.find(pick[2])
        nxt = next((m.start() for m in re.finditer(r"[^\s!]+", new)
                    if m.group().upper() in tpl["bang"] and m.start() > p), p)
        return lines, tpl, (r, nxt)
    ghost_rows = ui_tpl_ghost_rows(lines, tpl)
    iid = ghost_rows.get(row)
    if iid is None:
        return None
    tpl["items"].pop(iid)                     # its lines stay, now real
    tpl["hl"] = set(tpl["hl"]) - {iid}
    lines, tpl, r = ui_tpl_reconcile(lines, tpl, row)
    gr = ui_tpl_ghost_rows(lines, tpl)
    nxt = next((i for i in sorted(gr) if i > r), r)
    return lines, tpl, (nxt, 0)


def ui_tpl_accept_highlighted(lines, tpl):
    """Keep at once every highlighted (stage-before) '!' word and option line of the stage. Returns
    (lines, tpl, (row, col)); unchanged when nothing is highlighted."""
    lines = list(lines)
    pos = (0, 0)
    for word in [w for w in UI_TPL_BANG if w in tpl["bang_hl"] and w in tpl["bang"]]:
        rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
        r = ui_tpl_bang_row(lines, rng)
        m = next((m for m in re.finditer(r"[^\s!]+", lines[r]) if m.group().upper() == word), None)
        if m is None or word not in tpl["bang"]:
            continue
        lines, tpl, pos = ui_tpl_accept(lines, r, m.start(), tpl)
    keep = set(tpl["hl"]) & set(tpl["items"])
    if keep:
        tpl = dict(tpl, items={k: v for k, v in tpl["items"].items() if k not in keep}, hl=set())
        lines, tpl, r = ui_tpl_reconcile(lines, tpl, pos[0])
        pos = (r, 0)
    return lines, tpl, pos


def ui_tpl_realize_edited(lines, tpl):
    """Items with some of their lines edited become real (all of them); items whose lines were all deleted
    are dismissed (not offered again in this stage). Returns tpl."""
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    present = set(lines[rng[0]:rng[1]])
    items, gone = {}, set()
    for iid, ls in tpl["items"].items():
        if all(l in present for l in ls):
            items[iid] = ls
        elif not any(l in present for l in ls):
            gone.add(iid)
    if len(items) == len(tpl["items"]):
        return tpl
    return dict(tpl, items=items, dismissed=set(tpl.get("dismissed") or ()) | gone)


def ui_tpl_effective(text, tpl, blank=False):
    """The input without the gray template parts. blank=True keeps the numbering (gray rows empty, gray words
    blanked), for the checks."""
    if not tpl:
        return text
    lines = text.split("\n")
    rng = ui_tpl_stage_range(lines, tpl["stage_ord"])
    ghost_rows = ui_tpl_ghost_rows(lines, tpl)
    bang = ui_tpl_bang_row(lines, rng) if tpl["bang"] else None
    out = []
    for i, l in enumerate(lines):
        if i in ghost_rows:
            if blank:
                out.append("")
            continue
        if i == bang:
            if blank:
                l = re.sub(r"[^\s!]+", lambda m: " " * len(m.group()) if m.group().upper() in tpl["bang"]
                           else m.group(), l)
            else:
                words = [w for w in l.lstrip()[1:].split() if w.upper() not in tpl["bang"]]
                if not words:
                    continue
                l = "! " + " ".join(words)
        out.append(l)
    return "\n".join(out)


def ui_full_template(geom_first=True):
    """(text, tpl) of a new input: the description, then the first stage's template."""
    head = UI_TEMPLATE_HEAD.rstrip("\n").split("\n")
    lines, tpl, _ = ui_tpl_start(head + [""], len(head) - 1, 1, geom_first=geom_first)
    return "\n".join(lines), tpl


def ui_prev_stage_info(effective_text):
    """{'bang', 'keys', 'jobname'} of the last stage of an input (for the template of the stage after it)."""
    stages = ui_stages(ui_lex(effective_text))
    if not stages:
        return None
    s = stages[-1]
    keys = {k: v for k, v in s.keys.items() if k not in ("jobname", "scan", "neb_product")}
    job = s.keys.get("jobname") or None
    if job is None:
        for st_ in reversed(stages):
            if st_.keys.get("jobname"):
                job = st_.keys["jobname"]
                break
    runs = []                                    # every run-type word the stage wrote (OPT FREQ: both)
    for l in s.lines:
        if l.kind == "bang":
            for w in _ui_strip_comment(l.text).lstrip()[1:].split():
                info = UI_BANG.get(w.upper())
                if info and info[0] == "run type" and info[1] not in runs:
                    runs.append(info[1])
    return {"bang": dict(s.bang), "keys": keys, "jobname": job, "runs": runs or None}


UI_COMMON_BANG = ["OPT CCSD(T) POSITRON DF", "SP CCSD(T) POSITRON DF", "OPT CCSD POSITRON DF",
                  "SP CCSD(T) POSITRON CONV", "SP CCSD POSITRON DF", "SP CCSD(T) ELECTRONIC DF",
                  "OPT HF POSITRON DF", "OPT MP2 POSITRON DF", "OPT CCSD(T) ELECTRONIC DF",
                  "OPT CCSD POSITRON CONV", "OPT CCSD(T) POSITRON CONV", "SP CCSD ELECTRONIC DF",
                  "OPT CCSD ELECTRONIC DF", "SP MP2 POSITRON DF", "OPT FREQ HF ELECTRONIC DF",
                  "SCAN CCSD(T) ELECTRONIC DF", "OPT HF PROTON CONV"]
UI_COMMON_KEYS = ["basis", "jobname", "memory", "frozen", "nprocs", "pos_basis", "charge", "mult", "n_pos",
                  "conv_check", "molden", "jkfit_aux_e", "auxbasis_e", "pinnings", "verbose"]
UI_COMMON_VALUES = {"basis": "aug-cc-pvtz", "nuc_basis": "pb4d", "memory": "8000", "frozen": "auto", "nprocs": "4",
                    "pos_basis": "PSX-TZ", "charge": "-2", "mult": "1", "n_pos": "1", "conv_check": "true",
                    "molden": "true", "verbose": "4"}


_UI_BASIS_CACHE = {}
UI_COMMON_E_BASIS = ["aug-cc-pvdz", "aug-cc-pvtz", "aug-cc-pvqz", "aug-cc-pv5z", "cc-pvdz", "cc-pvtz", "cc-pvqz",
                     "cc-pv5z", "def2-svp", "def2-tzvp", "def2-qzvp", "6-31g", "6-31g*", "6-311g**", "sto-3g"]
UI_PSX_BASIS = ["PSX-DZ", "PSX-TZ", "PSX-QZ", "PSX-5Z", "PSX-6Z", "MATCH"]


def _ui_pretty_basis(name):
    """PySCF's normalized alias ('augccpvtzjkfit') in the usual spelling ('aug-cc-pvtz-jkfit')."""
    s = re.sub(r"^aug(?=cc)", "aug-", name)
    s = re.sub(r"cc(?=p[cw]?v)", "cc-", s)
    s = re.sub(r"(?<=z)(ri|dkh?|pp)$", r"-\1", s)
    s = re.sub(r"(?<!-)jkfit$", "-jkfit", s)
    return re.sub(r"^def2(?=[a-z])", "def2-", s)


def ui_basis_names(kind):
    """Basis names for completion, read from the installed packages' files (no import, so --ui stays light):
    'e' electron (PySCF's basis table), 'pos' positron (PSX sets, MATCH, then any electron basis), 'nuc' nuclear
    (pyscf_neo_cc's _neo/basis/*.dat), 'aux' fitting sets (jkfit / ri)."""
    if kind in _UI_BASIS_CACHE:
        return _UI_BASIS_CACHE[kind]
    import importlib.util
    names = []
    if kind in ("e", "pos", "aux"):
        try:
            loc = importlib.util.find_spec("pyscf").submodule_search_locations[0]
            src = open(os.path.join(loc, "gto", "basis", "__init__.py"), encoding="utf-8").read()
            table = src[src.index("ALIAS = {"):]
            table = table[:table.index("\n}")]
            names = [_ui_pretty_basis(n) for n in re.findall(r"^\s*'([^']+)'\s*:", table, re.M)]
        except (AttributeError, ImportError, OSError, ValueError, TypeError):
            names = []
    if kind == "aux":
        out = [n for n in names if n.endswith(("jkfit", "-ri"))]
    elif kind == "nuc":
        try:
            loc = importlib.util.find_spec("pyscf_neo_cc").submodule_search_locations[0]
            out = sorted((f[:-4] for f in os.listdir(os.path.join(loc, "_neo", "basis")) if f.endswith(".dat")),
                         key=lambda n: (not n.startswith("pb"), n))
        except (AttributeError, ImportError, OSError, TypeError):
            out = ["pb4d", "pb4f1", "pb4f2", "pb5d", "pb5f", "pb5g", "pb6d", "pb6f", "pb6g", "pb6h"]
    else:
        common = UI_COMMON_E_BASIS if kind == "e" else UI_PSX_BASIS + UI_COMMON_E_BASIS
        out = common + [n for n in names if n not in common]
    seen = set()
    _UI_BASIS_CACHE[kind] = [n for n in out if not (n.lower() in seen or seen.add(n.lower()))]
    return _UI_BASIS_CACHE[kind]


UI_BASIS_KEYS = {"basis": "e", "pos_basis": "pos", "nuc_basis": "nuc", "jkfit_aux_e": "aux", "auxbasis_e": "aux"}


def ui_value_completable(name):
    """True if %name's value has real completions worth popping the menu for right after Tab-accepting its
    gray template line (not a block/scan, and either a basis-set family, a documented choice list, or a
    plain choices tuple)."""
    o = UI_KEYS.get(name)
    return o is not None and o.value not in ("block", "scan") and bool(o.choices or name in UI_BASIS_KEYS
                                                                        or name in UI_CHOICE_DOCS)


_UI_BASIS_FILE_CACHE = {}


def _ui_basis_path(name):
    """Installed .dat file for electron basis `name` (pretty spelling), or None if it can't be resolved
    (no import, same approach as ui_basis_names: PySCF's own ALIAS table, read as text)."""
    key = re.sub(r"[-_]", "", name.lower())
    if key in _UI_BASIS_FILE_CACHE:
        return _UI_BASIS_FILE_CACHE[key]
    path = None
    try:
        import importlib.util
        loc = importlib.util.find_spec("pyscf").submodule_search_locations[0]
        src = open(os.path.join(loc, "gto", "basis", "__init__.py"), encoding="utf-8").read()
        table = src[src.index("ALIAS = {"):]
        table = table[:table.index("\n}")]
        m = re.search(r"^\s*'%s'\s*:\s*(.+?),?\s*(?:#.*)?$" % re.escape(key), table, re.M)
        if m:
            val = m.group(1).strip()
            jm = re.match(r"join\(\s*'([^']+)'\s*,\s*'([^']+)'\s*\)", val)
            sm = re.match(r"'([^']+)'", val)
            if jm:
                path = os.path.join(loc, "gto", "basis", jm.group(1), jm.group(2))
            elif sm:
                path = os.path.join(loc, "gto", "basis", sm.group(1))
    except (AttributeError, ImportError, OSError, ValueError, TypeError):
        path = None
    _UI_BASIS_FILE_CACHE[key] = path
    return path


_UI_BASIS_SHELL_RE = re.compile(r"^([A-Z][a-z]?)\s+(?:SP|[SPDFGHI])\s*$", re.M | re.I)


def ui_basis_missing_elements(basis_name, symbols):
    """Canonical element symbols among `symbols` that `basis_name`'s installed data file defines no shell
    for; [] when the basis can't be resolved/read (nothing is flagged then) or every element is covered."""
    path = _ui_basis_path(basis_name)
    if not path:
        return []
    try:
        text = open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return []
    covered = {m.group(1) for m in _UI_BASIS_SHELL_RE.finditer(text)}
    out = []
    for s in symbols:
        el = ui_element(s)[0]
        if el and el not in covered and el not in out:
            out.append(el)
    return out


def ui_basis_name_unknown(key, value):
    """None if `value` (one basis-name token, already split out of any '%basis B1 H B2 ...' grouping by
    ui_parse_basis_groups) is not checked here, or an error message if it is %basis (the only key the
    engine itself pre-validates against PySCF's basis table, v04.07's _check_basis_names) and is not a
    name PySCF has. %pos_basis/%nuc_basis/%jkfit_aux_e/%auxbasis_e are deliberately NOT checked: they also
    accept pyscf_neo_cc's protonic pb* names (scaled to another particle's mass), the 'etb'/'default'/
    'auto' keywords, and raw even-tempered shorthand ('4s4p4d') that %basis does not, and the engine
    itself does not pre-validate them either (resolve_pos_basis falls through to returning any
    unrecognized string as-is). Checked against PySCF's own alias table (same source as _ui_basis_path,
    no pyscf import), so it catches a typo such as 'aug-cc-pvt' (missing the 'z') before the engine does
    (only once a real job starts, up to v04.06 - after any earlier stages already ran)."""
    if key != "basis":
        return None
    v = value.strip()
    if not v:
        return None
    if _ui_basis_path(v) is None:
        return f"%{key} {v} is not a basis name PySCF knows (check the spelling, e.g. a missing 'z')"
    return None


def _ui_is_element_token(t):
    """An element symbol written as a basis selector ('H', 'Li', 'n'): UI mirror of the engine's
    _is_element_token (newBRORSEN.v04.07.py), without importing pyscf's element table."""
    return t.isalpha() and 1 <= len(t) <= 2 and t.upper() in UI_Z


def ui_parse_basis_groups(val):
    """UI mirror of the engine's parse_basis_groups (newBRORSEN.v04.07.py, section 8.2 'Atoms on the
    basis line'): '%basis'/'%pos_basis' value -> (groups, error). groups is a list of (basis, [selectors])
    when `val` names atoms (e.g. 'PSX-TZ H PSX-DZ'); None with no error when `val` is a single bare basis
    name (the plain, pre-v04.07 form - callers then check/use `val` itself); None with an error message
    (no leading '%key' - callers prefix that) when the value cannot be read at all."""
    toks = val.replace("{", " ").replace("}", " ").replace(",", " ").split()
    groups = []
    for t in toks:
        if t.isdigit() or _ui_is_element_token(t):
            if not groups:
                return None, "start with a basis name, then the atoms it goes on (e.g. PSX-TZ H)"
            groups[-1][1].append(t if t.isdigit() else t.capitalize())
        else:
            groups.append((t, []))
    if not groups:
        return None, "needs a basis name"
    if sum(1 for _, sel in groups if not sel) > 1:
        return None, "at most one basis without atoms (the one for all the atoms not named)"
    if len(groups) == 1 and not groups[0][1]:
        return None, None
    return groups, None


def ui_assign_basis_groups(groups, symbols):
    """UI mirror of the engine's assign_basis_groups: per-atom basis name (or None) from `groups` for
    `symbols` (the geometry rows' first token, as written, ghost/dummy decorations and all - ui_element
    strips them the same way element_of does), and a list of human-readable problems. An atom number
    beats an element symbol, which beats the basis given without atoms."""
    n = len(symbols)
    per, rank, problems = [None] * n, [0] * n, []
    fallback = next((b for b, sel in groups if not sel), None)
    elems = [ui_element(s)[0] for s in symbols]
    for b, sel in groups:
        for t in sel:
            if t.isdigit():
                i = max(int(t), 1) - 1
                if i >= n:
                    problems.append(f"atom {t} does not exist ({n} atoms)")
                    continue
                hit, r = [i], 2
            else:
                hit, r = [i for i, e in enumerate(elems) if e and e.upper() == t.upper()], 1
                if not hit:
                    problems.append(f"no {t} atom in the geometry")
            for i in hit:
                if rank[i] == r and per[i] != b:
                    problems.append(f"atom {i + 1} ({symbols[i]}) is given both {per[i]} and {b}")
                elif r > rank[i]:
                    per[i], rank[i] = b, r
    for i in range(n):
        if per[i] is None and fallback is not None:
            per[i] = fallback
    return per, problems


_UI_WIDE_BASIS = ("def2-tzvp", "def2-qzvp")


def ui_wide_basis_suggestion(symbols):
    """The first of a couple of broad-coverage basis sets (def2-tzvp, def2-qzvp) that covers every element
    in `symbols`, or None if neither installed file does (or neither can be read)."""
    for alt in _UI_WIDE_BASIS:
        if not ui_basis_missing_elements(alt, symbols):
            return alt
    return None


def ui_auto_jobname(stage):
    """formula_method_q<charge>_m<mult>_basis from `stage`'s geometry, '!' method, charge/mult and %basis
    (e.g. 'H2O_CCSDt_q0_m1_ccpvdz'), or None while there is no geometry yet to build a formula from."""
    rows = stage.geometry_rows
    if not rows:
        return None
    kind = stage.geometry or getattr(stage, "inherited_geometry", None)
    syms = ui_zmat_symbols(rows) if kind == "zmat" else [r.split()[0] for r in rows]
    elements = [e for e in (ui_element(s)[0] for s in syms) if e]
    if not elements:
        return None
    counts = {}
    for e in elements:
        counts[e] = counts.get(e, 0) + 1
    order = (["C"] if "C" in counts else []) + (["H"] if "H" in counts else []) + \
        sorted(e for e in counts if e not in ("C", "H"))
    formula = "".join(e + (str(counts[e]) if counts[e] > 1 else "") for e in order)
    method = stage.method.replace("(T)", "t")
    charge = stage.charge if stage.charge is not None else 0
    mult = stage.mult if stage.mult is not None else 1
    basis_val = stage.keys.get("basis", "cc-pvdz")
    groups, grp_err = ui_parse_basis_groups(basis_val)
    if grp_err:
        basis_name = basis_val
    elif groups:
        # '%basis B1 H B2 ...' (v04.07): the fallback (the basis without atoms, if any covers "the rest"),
        # else every distinct basis joined - either beats a jobname with spaces/atom numbers in it.
        basis_name = next((b for b, sel in groups if not sel), None) or \
            "+".join(dict.fromkeys(b for b, _ in groups))
    else:
        basis_name = basis_val
    basis = re.sub(r"[-_]", "", basis_name).lower()
    return f"{formula}_{method}_q{charge}_m{mult}_{basis}"


def ui_suggest_text(line):
    """Gray inline suggestion for the text left of the cursor (or None): the rest of a common '!' line, the rest
    of a %key plus its value (example or default), or the value after '%key '."""
    s = line.lstrip()
    if s.startswith("!"):
        typed = s[1:].lstrip()
        if not typed:
            return (" " if s == "!" else "") + UI_COMMON_BANG[0]
        for combo in UI_COMMON_BANG:
            if combo.upper().startswith(typed.upper()) and len(combo) > len(typed):
                return combo[len(typed):]
        return None
    m = re.match(r"^%(\w*)$", s)
    if m:
        prefix = m.group(1).lower()
        if not prefix:
            return None
        for k in UI_COMMON_KEYS + [o.name for o in UI_OPTIONS]:
            if k.startswith(prefix) and k != prefix and k in UI_KEYS:
                o = UI_KEYS[k]
                if o.value == "block" or (o.value == "scan" and k in UI_BLOCK_SYNTAX and prefix != "scan"):
                    return k[len(prefix):]
                v = ui_value_example(o)
                return k[len(prefix):] + (" " + v if v else "")
        return None
    m = re.match(r"^%(\w+)\s+$", s)
    if m and m.group(1).lower() in UI_KEYS:
        v = ui_value_example(UI_KEYS[m.group(1).lower()])
        return v or None
    return None


def ui_bang_group(word):
    up = word.upper()
    if up in UI_REMOVED_BANG:
        return "run type"
    return UI_BANG[up][0] if up in UI_BANG else None


def ui_element(label):
    """'C1' / 'ghost-H' / 'X-H' / 'H+' -> (element symbol or None, is_ghost); same rule as element_of."""
    s = label.strip()
    low = s.lower()
    ghost = low.startswith("ghost") or low.startswith("x-")
    if low.startswith("ghost"):
        s = s[5:].lstrip("-_")
    elif low.startswith("x-"):
        s = s[2:]
    s = re.sub(r"[\d+*#]+", "", s)
    sym = s[:1].upper() + s[1:].lower() if s else s
    return (sym, ghost) if sym.upper() in UI_Z else (None, ghost)


class UILine:
    """One input line as the lexer sees it. kind: blank, comment, next, bang, directive, block_open, block_row,
    block_end, geom_header, geom_row, geom_end, end, unknown."""
    __slots__ = ("n", "text", "kind", "stage", "key", "value", "block")

    def __init__(self, n, text, kind, stage, key=None, value=None, block=None):
        self.n, self.text, self.kind, self.stage = n, text, kind, stage
        self.key, self.value, self.block = key, value, block


_UI_DIRECTIVE_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*|\s+|$)(.*)$')
_UI_SPLIT_RE = re.compile(r'^\s*%(next|newjob)\b', re.IGNORECASE)


def _ui_strip_comment(line):
    out, quote = [], None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == '#':
            break
        out.append(ch)
    return "".join(out)


def ui_lex(text):
    """Classify every line (0-based n) the way parse_block reads it. Stage numbers count blocks after %next;
    blocks without content keep their number but are dropped by the wrapper (see ui_stages)."""
    lines, stage, mode = [], 1, None
    for n, raw in enumerate(text.split("\n")):
        if _UI_SPLIT_RE.match(raw):
            lines.append(UILine(n, raw, "next", stage))
            stage += 1
            mode = None
            continue
        code = _ui_strip_comment(raw).strip()
        if not code:
            lines.append(UILine(n, raw, "comment" if raw.strip() else "blank", stage, block=mode))
            continue
        low = code.lower()
        if mode in ("xyz", "zmat"):
            if code == "*" or low == "end":
                lines.append(UILine(n, raw, "geom_end", stage, block=mode))
                mode = None
                continue
            if code[0] not in "%!*":
                lines.append(UILine(n, raw, "geom_row", stage, block=mode))
                continue
            mode = None
        elif mode is not None:
            if low == "end":
                lines.append(UILine(n, raw, "block_end", stage, block=mode))
                mode = None
                continue
            if code[0] not in "%!*":
                lines.append(UILine(n, raw, "block_row", stage, block=mode))
                continue
            mode = None
        if code.startswith("!"):
            lines.append(UILine(n, raw, "bang", stage))
        elif code.startswith("*"):
            parts = code[1:].split()
            kind = parts[0].lower() if parts else ""
            kind = "zmat" if kind in ("zmat", "int", "internal") else "xyz" if kind == "xyz" else None
            lines.append(UILine(n, raw, "geom_header", stage, key=kind, value=parts[1:]))
            mode = kind
        elif code.startswith("%"):
            m = _UI_DIRECTIVE_RE.match(code[1:].strip())
            key, val = (m.group(1).lower(), m.group(2).strip()) if m else (None, code[1:])
            if key and ((not val and (key in UI_BLOCK_KEYS or key == "scan"))
                        or (key == "scan" and _ui_scan_header_only(val))):
                lines.append(UILine(n, raw, "block_open", stage, key=key, value=""))
                mode = key
            else:
                lines.append(UILine(n, raw, "directive", stage, key=key, value=val))
        elif low == "end":
            lines.append(UILine(n, raw, "end", stage))
        else:
            lines.append(UILine(n, raw, "unknown", stage))
    return lines


def _ui_scan_header_only(val):
    """v04.08: '%scan bond 1 2' / 'angle 1 2 3' / 'dihedral 1 2 3 4' / '<zmat variable>' with nothing after
    the coordinate opens a block of 'start end n' segments ending with 'end'."""
    toks = val.split()
    if not toks:
        return False
    n = {"bond": 2, "distance": 2, "angle": 3, "dihedral": 4}.get(toks[0].lower())
    if n is not None:
        return len(toks) == 1 + n and all(re.fullmatch(r"-?\d+", t) for t in toks[1:])
    if toks[0].lower() == "sym_zmat":
        return len(toks) == 2
    return (len(toks) == 1 and toks[0].lower() not in ("translate", "fragment")
            and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", toks[0]) is not None)


class UIStage:
    """Effective settings of one stage for the static checks (own lines + inherited ones)."""

    def __init__(self, number):
        self.number = number
        self.lines = []
        self.bang = {}            # group -> canonical word (effective)
        self.keys = {}            # canonical key -> value (effective)
        self.own_keys = set()
        self.geometry = None      # "xyz" / "zmat" / None (inherited or missing)
        self.geometry_rows = []
        self.charge = self.mult = None
        self.inherited_from = None
        self.sector_explicit = False
        self.run_types = set()    # every run-type canon on this stage's own '! ...' line, e.g. {"OPT", "FREQ"}

    @property
    def sector(self):
        return self.bang.get("sector", "ELECTRONIC")

    @property
    def method(self):
        return self.bang.get("method", "HF")

    @property
    def run_type(self):
        return self.bang.get("run type", "SP_ENERGY")


def ui_stages(lines):
    """Group lexed lines into stages and apply the wrapper's inheritance rule: a stage without a geometry block
    continues the previous one (settings and geometry), except job name, geometry settings, %scan / %neb_product
    and the SCAN / NEB run types."""
    by_num = {}
    for ln in lines:
        by_num.setdefault(ln.stage, []).append(ln)
    stages, prev = [], None
    for num in sorted(by_num):
        own = by_num[num]
        if not any(l.kind not in ("blank", "comment", "next") for l in own):
            continue
        st = UIStage(len(stages) + 1)
        st.lines = own
        has_geom = any(l.kind == "geom_header" for l in own)
        if prev is not None and not has_geom:
            st.inherited_from = prev.number
            st.bang = dict(prev.bang)
            st.run_types = set(prev.run_types) - {"SCAN"}
            if st.bang.get("run type") == "SCAN":
                del st.bang["run type"]
            st.keys = {k: v for k, v in prev.keys.items() if k not in ("jobname", "coords", "scan", "neb_product")}
            if st.keys.get("ts_mode", "").lower() in ("neb", "ci_neb"):
                st.keys["ts_mode"] = "ts"
            st.sector_explicit = prev.sector_explicit
            st.charge, st.mult = prev.charge, prev.mult
            st.geometry_rows = list(prev.geometry_rows)
            st.inherited_geometry = prev.geometry or getattr(prev, "inherited_geometry", None)
        else:
            st.inherited_geometry = None
        for l in own:
            code = _ui_strip_comment(l.text).strip()
            if l.kind == "bang":
                own_run_types = set()
                for w in code[1:].split():
                    up = w.upper()
                    if up in UI_BANG:
                        grp, canon, _ = UI_BANG[up]
                        st.bang[grp] = canon
                        if grp == "run type":
                            own_run_types.add(canon)
                        if grp == "sector":
                            st.sector_explicit = True
                            if up == "CDFT":
                                st.bang["method"] = "DFT"
                            if up in ("CNEO", "CDFT"):
                                st.keys["cneo"] = "true"
                if own_run_types:
                    st.run_types = own_run_types
            elif l.kind in ("directive", "block_open") and l.key in UI_KEYS:
                canon = UI_KEYS[l.key].name
                st.keys[canon] = l.value
                st.own_keys.add(canon)
                if canon == "charge":
                    st.charge = _ui_int(l.value, st.charge)
                elif canon == "mult":
                    st.mult = _ui_int(l.value, st.mult)
                elif canon == "spin":
                    v = _ui_int(l.value, None)
                    st.mult = v + 1 if v is not None else st.mult
            elif l.kind == "geom_header":
                st.geometry = l.key
                st.geometry_rows = []
                if l.value:
                    st.charge = _ui_int(l.value[0], st.charge)
                if len(l.value) > 1:
                    st.mult = _ui_int(l.value[1], st.mult)
            elif l.kind == "geom_row" and l.block == "xyz":
                st.geometry_rows.append(code)
        stages.append(st)
        prev = st
    return stages


def _ui_int(val, fallback):
    try:
        return int(str(val).strip())
    except (TypeError, ValueError):
        return fallback


def _ui_choice_word(v):
    """First word of a choice value; ORCA's '%coords units bohrs end' form gives 'bohrs'."""
    toks = v.lower().split()
    return toks[1] if len(toks) > 1 and toks[0] == "units" else toks[0]


def _ui_float(tok):
    try:
        float(tok)
        return True
    except ValueError:
        return False


class UIDiag:
    __slots__ = ("line", "start", "end", "severity", "msg")

    def __init__(self, line, start, end, severity, msg):
        self.line, self.start, self.end, self.severity, self.msg = line, start, end, severity, msg

    def __repr__(self):
        return f"UIDiag({self.line}, {self.start}, {self.end}, {self.severity!r}, {self.msg!r})"


def _ui_span(text, token, after=0):
    i = text.find(token, after)
    if i < 0:
        i = text.lower().find(token.lower(), after)
    return (i, i + len(token)) if i >= 0 else (0, len(text))


def _ui_electrons(rows, charge):
    total = 0
    for r in rows:
        sym, ghost = ui_element(r.split()[0])
        if sym is None:
            return None
        total += 0 if ghost else UI_Z[sym.upper()]
    return total - (charge or 0)


def _ui_missing_next_check(lines):
    """A second '!' line or a second geometry block inside one %next/%newjob-delimited block is not a new
    stage (stages split only on %next/%newjob, same as the engine's _BLOCK_SPLIT_RE): it silently
    overwrites the first one's run-type words / geometry rows, with nothing lost showing up anywhere -
    the exact way a whole stage (e.g. a single-atom reference) can vanish from a job with no error. Catch
    it here instead of only at save time (ui_deep_check) or, worse, after the engine already ran it."""
    out = []
    by_num = {}
    for l in lines:
        by_num.setdefault(l.stage, []).append(l)
    for own in by_num.values():
        bangs = [l for l in own if l.kind == "bang"]
        geoms = [l for l in own if l.kind == "geom_header"]
        if len(bangs) > 1:
            code = _ui_strip_comment(bangs[1].text).rstrip()
            out.append(UIDiag(bangs[1].n, 0, len(code), "error",
                              f"a second '!' line without %next before it (first one: line {bangs[0].n + 1}); "
                              f"it silently overwrites the first instead of starting a new stage - add %next"))
        if len(geoms) > 1:
            code = _ui_strip_comment(geoms[1].text).rstrip()
            out.append(UIDiag(geoms[1].n, 0, len(code), "error",
                              f"a second geometry block without %next before it (first one: line "
                              f"{geoms[0].n + 1}); it silently replaces the first instead of starting a "
                              f"new stage - add %next"))
    return out


def ui_check(text):
    """Static checks (no wrapper import): [UIDiag] with 0-based line and column spans. Mirrors the wrapper's
    parse-time rules plus the stage rules the spec lists; the wrapper's validator (parse_text) stays the judge."""
    lines = ui_lex(text)
    stages = ui_stages(lines)
    diags = list(_ui_missing_next_check(lines))
    for st in stages:
        seen = {}
        for l in st.lines:
            code = _ui_strip_comment(l.text)
            if l.kind == "bang":
                pos = code.find("!") + 1
                for w in code[pos:].split():
                    start = code.find(w, pos)
                    pos = start + len(w)
                    up = w.upper()
                    grp = ui_bang_group(up)
                    if up in UI_REMOVED_BANG:
                        diags.append(UIDiag(l.n, start, pos, "error",
                                            f"'! {w}' was removed in v04.05: write {UI_REMOVED_BANG[up]}"))
                    elif grp is None:
                        diags.append(UIDiag(l.n, start, pos, "error", f"unknown '!' word '{w}'"))
                        continue
                    if grp in seen and seen[grp] != up:
                        pair = {UI_BANG.get(seen[grp], ("", ""))[1], UI_BANG.get(up, ("", ""))[1]}
                        if grp == "run type" and pair in UI_FREQ_PAIRS:
                            continue
                        diags.append(UIDiag(l.n, start, pos, "error",
                                            f"'{seen[grp]}' and '{w}' are both {grp} words: keep one"
                                            + (" (use %next for a second calculation)" if grp == "run type" else "")))
                    seen.setdefault(grp, up)
            elif l.kind in ("directive", "block_open"):
                s, e = _ui_span(code, "%" + (l.key or ""))
                if l.key is None:
                    diags.append(UIDiag(l.n, 0, len(code), "error", "directive line not understood"))
                    continue
                opt = UI_KEYS.get(l.key)
                if opt is None:
                    diags.append(UIDiag(l.n, s, e, "error", f"unknown directive '%{l.key}'"))
                    continue
                if l.kind == "block_open" or opt.value == "scan":
                    continue
                if opt.value == "block":
                    if l.value:
                        diags.append(UIDiag(l.n, s, e, "error",
                                            f"%{l.key} is a block: write '%{l.key}' alone, then one "
                                            f"'element basis' row per line, then 'end' (not a value on "
                                            f"the same line)"
                                            + (" - to select atoms instead, use %pos_atom"
                                               if l.key == "pos_basis_map" else "")))
                    continue
                v = l.value
                vs, ve = _ui_span(code, v, e) if v else (e, e)
                bad = None
                if not v and opt.value not in ("text",):
                    bad = f"%{l.key} needs a value"
                elif opt.value == "int" and _ui_int(v, None) is None:
                    bad = f"%{l.key} expects an integer"
                elif opt.value == "float" and not _ui_float(v.lower().replace("d", "e")):
                    bad = f"%{l.key} expects a number"
                elif opt.value == "bool" and v.lower() not in UI_BOOL_WORDS:
                    bad = f"%{l.key} expects true/false"
                elif opt.value == "choice" and _ui_choice_word(v) not in {c.lower() for c in opt.choices}:
                    bad = f"%{l.key} expects one of: {', '.join(opt.choices)}"
                elif opt.value == "memory" and not re.match(r"^\d+(\.\d+)?\s*(mb|gb)?$", v.lower()):
                    bad = f"%{l.key} expects MB (or e.g. 16GB)"
                if bad:
                    diags.append(UIDiag(l.n, vs, ve, "error", bad))
                elif v and opt.name == "basis":
                    groups, grp_err = ui_parse_basis_groups(v)
                    if grp_err:
                        diags.append(UIDiag(l.n, vs, ve, "error", f"%basis {v}: {grp_err}"))
                    elif groups is None:
                        unknown = ui_basis_name_unknown("basis", v)
                        if unknown:
                            diags.append(UIDiag(l.n, vs, ve, "error", unknown))
                    else:
                        for b, _ in groups:
                            unknown = ui_basis_name_unknown("basis", b)
                            if unknown:
                                diags.append(UIDiag(l.n, vs, ve, "error", unknown))
                if opt.sector and not st.sector_explicit:
                    diags.append(UIDiag(l.n, s, e, "error",
                                        f"%{opt.name} is a {opt.sector.lower()} setting: add {opt.sector} to the '!' "
                                        f"line (the default sector is ELECTRONIC)"))
                elif opt.sector and st.sector != opt.sector:
                    diags.append(UIDiag(l.n, s, e, "warning",
                                        f"%{opt.name} has no effect in the {st.sector} sector"))
            elif l.kind == "geom_header":
                if l.key is None:
                    diags.append(UIDiag(l.n, 0, len(code), "error", "geometry block: use '* xyz <charge> <mult>' "
                                                                    "or '* zmat <charge> <mult>'"))
                elif any(_ui_int(v, None) is None for v in l.value[:2]):
                    diags.append(UIDiag(l.n, 0, len(code), "error", "charge and multiplicity must be integers"))
            elif l.kind == "geom_row" and l.block == "xyz":
                toks = code.split()
                if "," in code and re.search(r"\d,\d", code):
                    diags.append(UIDiag(l.n, 0, len(code), "error",
                                        "decimal comma in coordinates: input numbers use a decimal point"))
                elif len(toks) < 4 or not all(_ui_float(t) for t in toks[1:4]):
                    diags.append(UIDiag(l.n, 0, len(code), "error", "xyz row: Element x y z"))
                elif ui_element(toks[0])[0] is None:
                    diags.append(UIDiag(l.n, 0, len(toks[0]), "error", f"unknown element '{toks[0]}'"))
            elif l.kind == "unknown":
                diags.append(UIDiag(l.n, 0, len(code.rstrip()), "warning", "line not understood (ignored)"))
        diags.extend(_ui_stage_rules(st))
    return diags


def _ui_stage_rules(st):
    out = []
    first = next((l.n for l in st.lines if l.kind == "bang"),
                 next((l.n for l in st.lines if l.kind not in ("blank", "comment", "next")), 0))

    def at_key(key):
        return next((l.n for l in st.lines if l.kind in ("directive", "block_open")
                     and l.key in UI_KEYS and UI_KEYS[l.key].name == key), first)

    if st.geometry is None and st.inherited_from is None:
        # a note while the input is being written (the geometry usually comes last); the wrapper's check
        # at save time still refuses an input without one
        out.append(UIDiag(first, 0, 0, "warning", "no geometry yet: F2 builds one, or keep the gray example"))
    rt = st.run_type
    if rt == "SCAN" and "scan" not in st.own_keys:
        out.append(UIDiag(first, 0, 0, "error", "'! SCAN' needs a %scan line or block"))
    if "scan" in st.own_keys and rt != "SCAN":
        out.append(UIDiag(at_key("scan"), 0, 0, "warning", "%scan is ignored: the run type is not SCAN"))
    if "scan_fit" in st.own_keys and rt != "SCAN":
        out.append(UIDiag(at_key("scan_fit"), 0, 0, "warning", "%scan_fit is ignored outside SCAN stages"))
    eng = st.keys.get("opt_engine", "").strip().upper()
    if rt == "OPT" and eng in ("ZOPT", "ZMAT"):
        geom = st.geometry or st.inherited_geometry
        if geom != "zmat":
            out.append(UIDiag(at_key("opt_engine"), 0, 0, "error",
                              "%opt_engine zopt needs a '* zmat' geometry block (own or from the previous stage)"))
    mode = st.keys.get("ts_mode", "").strip().lower()
    if rt == "TS" and mode in ("neb", "ci_neb") and "neb_product" not in st.own_keys:
        out.append(UIDiag(at_key("ts_mode"), 0, 0, "error", "%ts_mode neb needs a %neb_product block"))
    # Fires once a sector is explicit (the moment the template opens, before one is chosen, is exempt), OR
    # for any non-inherited stage (st.inherited_from is None): a stage with its own fresh geometry never
    # inherits settings from the previous one (only a stage that continues without a geometry does), so
    # it is exactly these silently-defaulting stages that most need the warning, even with no '!' line yet.
    if st.sector_explicit or st.inherited_from is None:
        if "basis" not in st.keys:
            out.append(UIDiag(first, 0, 0, "warning", "%basis is needed (default cc-pvdz used if not set)"))
        if st.sector == "POSITRON" and "pos_basis" not in st.keys:
            out.append(UIDiag(first, 0, 0, "warning",
                              "POSITRON needs %pos_basis (default PSX-TZ used if not set)"))
        if st.sector == "PROTON":
            if "nuc_basis" not in st.keys:
                out.append(UIDiag(first, 0, 0, "warning",
                                  "PROTON needs %nuc_basis (default pb4f1 used if not set)"))
            if "quantum_nuc" not in st.keys:
                out.append(UIDiag(first, 0, 0, "warning",
                                  "PROTON needs %quantum_nuc (default '0', the first atom, used if not set)"))
    if st.inherited_from is None and st.number > 1 and not st.bang:
        out.append(UIDiag(first, 0, 0, "warning",
                          "new geometry, no '!' line or settings here: settings are not inherited between "
                          "stages (only a stage that continues without its own geometry inherits them) - "
                          "this stage runs SP_ENERGY/ELECTRONIC/HF/cc-pvdz unless you restate what you need"))
    if rt == "GRADIENT" and st.sector == "POSITRON" and st.method == "DFT":
        out.append(UIDiag(first, 0, 0, "error", "POSITRON DFT has no gradient"))
    basis = st.keys.get("basis")
    if basis and st.geometry_rows:
        kind = st.geometry or getattr(st, "inherited_geometry", None)
        syms = ui_zmat_symbols(st.geometry_rows) if kind == "zmat" else [r.split()[0] for r in st.geometry_rows]
        groups, grp_err = ui_parse_basis_groups(basis)
        if grp_err or not groups:
            missing = [] if grp_err else ui_basis_missing_elements(basis, syms)
            if missing:
                alt = ui_wide_basis_suggestion(syms)
                suffix = f"; {alt} covers them" if alt else ""
                out.append(UIDiag(at_key("basis"), 0, 0, "warning",
                                  f"%basis {basis} has no {', '.join(missing)}{suffix}"))
        else:
            # '%basis B1 H B2 ...' (v04.07): check each basis only against the atoms it is actually
            # assigned to (assign_basis_groups), and require every atom to end up with one, same as the
            # engine's _check_basis_groups - up to v04.06 this whole form did not exist.
            per, problems = ui_assign_basis_groups(groups, syms)
            for msg in problems:
                out.append(UIDiag(at_key("basis"), 0, 0, "error", f"%basis: {msg}"))
            bare = [f"{s}{i + 1}" for i, (s, b) in enumerate(zip(syms, per)) if b is None]
            if bare:
                flat = " ".join(t for b, sel in groups for t in [b, *sel])
                out.append(UIDiag(at_key("basis"), 0, 0, "error",
                                  f"%basis: {'atoms' if len(bare) > 1 else 'atom'} {', '.join(bare)} "
                                  f"{'have' if len(bare) > 1 else 'has'} no electron basis; add a basis "
                                  f"without atoms for all the others (e.g. %basis {flat} cc-pvdz)"))
            for b in sorted({b for b in per if b is not None}):
                b_syms = [s for s, x in zip(syms, per) if x == b]
                missing = ui_basis_missing_elements(b, b_syms)
                if missing:
                    alt = ui_wide_basis_suggestion(b_syms)
                    suffix = f"; {alt} covers them" if alt else ""
                    out.append(UIDiag(at_key("basis"), 0, 0, "warning",
                                      f"%basis {b} has no {', '.join(missing)}{suffix}"))
    pos_basis = st.keys.get("pos_basis")
    # Legacy dict/raw-shell literals ('%pos_basis {'H': 'PSX-TZ'}' or '[[0,[exp,1.0]],...]') are not the
    # v04.07 grouped form - same guard the engine uses before calling parse_basis_groups.
    if pos_basis and st.geometry_rows and not (pos_basis[:1] == "[" or (pos_basis[:1] == "{" and ":" in pos_basis)):
        kind = st.geometry or getattr(st, "inherited_geometry", None)
        syms = ui_zmat_symbols(st.geometry_rows) if kind == "zmat" else [r.split()[0] for r in st.geometry_rows]
        groups, grp_err = ui_parse_basis_groups(pos_basis)
        if grp_err:
            out.append(UIDiag(at_key("pos_basis"), 0, 0, "error", f"%pos_basis {pos_basis}: {grp_err}"))
        elif groups:
            # Selector problems only (unknown element, atom past the end, two bases on one atom) - unlike
            # %basis, an atom with no positron basis is not an error (it just gets no positron functions),
            # and the basis-name tokens themselves are not checked (ui_basis_name_unknown is %basis-only).
            _, problems = ui_assign_basis_groups(groups, syms)
            for msg in problems:
                out.append(UIDiag(at_key("pos_basis"), 0, 0, "error", f"%pos_basis: {msg}"))
    if st.geometry == "xyz" and st.geometry_rows and st.mult:
        ne = _ui_electrons(st.geometry_rows, st.charge or 0)
        if ne is not None and (ne % 2) == (st.mult % 2):
            out.append(UIDiag(next((l.n for l in st.lines if l.kind == "geom_header"), first), 0, 0, "error",
                              f"{ne} electrons do not fit multiplicity {st.mult}"))
    return out


def ui_suggest_bang(line_text):
    """'!' words still allowed on this line (groups already used are hidden; FREQ stays after OPT/TS)."""
    used = {}
    for w in _ui_strip_comment(line_text).lstrip()[1:].split():
        g = ui_bang_group(w)
        if g:
            used[g] = UI_BANG.get(w.upper(), ("", w.upper()))[1]
    out = []
    for w, (g, canon, doc) in UI_BANG.items():
        if g == "units":
            continue                        # units go in %coords (the '!' words still work in old inputs)
        if g in used and not (g == "run type" and {used[g], canon} in UI_FREQ_PAIRS):
            continue
        out.append((w, g, doc))
    return out


def ui_suggest_keys(stage, show_hidden=False):
    """%keys for completion: own sector first, other sectors last. Deprecated keys (UI_DEPRECATED_KEYS:
    %pos_atom, %pos_basis_map, folded into %pos_basis in v04.08) are left out - typing one still works
    and still validates (they stay in UI_OPTIONS/UI_KEYS), just not suggested, unless `show_hidden`
    (F4 toggle) asks to see them too."""
    sector = stage.sector if stage else "ELECTRONIC"
    opts = UI_OPTIONS if show_hidden else [o for o in UI_OPTIONS if o.name not in UI_DEPRECATED_KEYS]
    first = [o for o in opts if o.sector in (None, sector)]
    last = [o for o in opts if o.sector not in (None, sector)]
    return first, last

# ==== end of UI core ====


# ==== v04.08 UI: helpers without prompt_toolkit (settings, files, launch, help, geometry text) ====

UI_CONFIG = os.path.join(os.path.expanduser("~"), ".config", "newbrorsen", "ui.toml")
UI_KEYMAPS = ("vim", "emacs", "plain")
UI_NAME_RE = re.compile(r"^[A-Za-z0-9._][A-Za-z0-9._-]*$")
UI_VIEW_EXT = (".molden", ".molden.input", ".cube", ".cub")   # kept in sync with _VIEW_EXT near run_viewer;
                                                               # duplicated here so the fast --ui path (before
                                                               # numpy/pyscf are imported) can recognize these
                                                               # files without importing the viewer itself


def ui_is_view_file(path):
    return str(path).lower().endswith(UI_VIEW_EXT)


_UI_HEAVY_CACHE = {}


def ui_load_heavy(engine_path):
    """The engine module's whole namespace (CubeGrid, load_plot_file, JobStage, density-map/report
    functions, numpy, PySCF, ...), loaded once from `engine_path` (see ui_engine_path) and cached. --ui
    needs none of this for plain text editing (D24: that is the whole reason it starts before numpy/PySCF
    are imported, in well under a second), so the Molden/cube viewer loads it lazily, the moment it is
    actually opened, rather than paying numpy/PySCF's ~1 s / ~100 MB import cost on every --ui start.
    `runpy.run_path` re-executes the engine file as a plain module (its own `if __name__ == "__main__":`
    guard does not fire), returning every name bound at its module level."""
    if _UI_HEAVY_CACHE.get("path") != engine_path:
        import runpy
        _UI_HEAVY_CACHE["ns"] = runpy.run_path(engine_path, run_name="_newbrorsen_ui_heavy")
        _UI_HEAVY_CACHE["path"] = engine_path
    return _UI_HEAVY_CACHE["ns"]


def ui_term_image_protocol():
    """'kitty', 'sixel', 'iterm2' or None: whether this terminal (including over SSH, since TERM/TERM_PROGRAM
    are the client's own and are forwarded) is one we know how to draw an inline image in.

    Checked against actual terminal support (not guessed): the Kitty graphics protocol is only claimed for
    terminals that genuinely implement it (Kitty itself, WezTerm, Ghostty); foot and Konsole do not implement
    it (foot: no; Konsole: unreliable/partial) but both support Sixel, which is used for them instead."""
    env = os.environ
    term = env.get("TERM", "")
    if (env.get("KITTY_WINDOW_ID") or term.startswith("xterm-kitty") or
            env.get("WEZTERM_EXECUTABLE") or env.get("TERM_PROGRAM") == "WezTerm" or
            env.get("GHOSTTY_RESOURCES_DIR")):
        return "kitty"
    if env.get("TERM_PROGRAM") == "iTerm.app" or env.get("LC_TERMINAL") == "iTerm2":
        return "iterm2"
    if term.startswith("foot") or env.get("KONSOLE_VERSION") or "sixel" in term:
        return "sixel"
    return None


def ui_sixel_converter():
    """The ImageMagick binary to render a PNG as Sixel (no dedicated img2sixel on this machine's envs), or
    None if neither is installed."""
    import shutil
    return shutil.which("magick") or shutil.which("img2sixel") or shutil.which("convert")


def ui_chafa_available():
    import shutil
    return shutil.which("chafa") is not None


def ui_preview_mode():
    """'kitty', 'iterm2', 'sixel', 'chafa' or None: the best inline-preview path this terminal and this
    machine's installed tools actually support, checked (not guessed) end to end. Kitty/iTerm2 need no
    external tool (the terminal itself decodes the PNG); Sixel needs ImageMagick or img2sixel; chafa (plain
    ANSI/Unicode block-character art, no special terminal protocol needed at all) needs the `chafa` binary.
    None if nothing usable is installed, in which case the preview pane is not shown at all: this is all
    optional, on top of the usual Drawing/Comparing, not something to work around for."""
    proto = ui_term_image_protocol()
    if proto in ("kitty", "iterm2"):
        return proto
    if proto == "sixel" and ui_sixel_converter():
        return "sixel"
    if ui_chafa_available():
        return "chafa"
    return None


def ui_chafa_text(png_path, cols, rows):
    """ANSI/Unicode block-character art of the PNG, sized to `cols` x `rows` terminal cells (`--format
    symbols` forces this plain-text style regardless of what chafa would auto-pick for the terminal; we
    already handle kitty/iTerm2/Sixel ourselves). None if chafa is missing or fails."""
    import shutil
    import subprocess
    chafa = shutil.which("chafa")
    if not chafa:
        return None
    try:
        res = subprocess.run([chafa, "--format", "symbols", "--size", f"{cols}x{rows}", png_path],
                             capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0 or not res.stdout:
        return None
    return res.stdout.decode("utf-8", errors="replace")


def ui_image_escape(protocol, png_path, cols, rows, pixel_width=800):
    """The raw string that draws the PNG at `png_path` at the cursor's current position (`cols` x `rows`
    terminal cells for kitty/iterm2; `pixel_width` px wide, aspect preserved, for sixel, since the terminal's
    own cell pixel size is not queried here). None if the image could not be encoded for this protocol."""
    if protocol == "sixel":
        conv = ui_sixel_converter()
        if not conv:
            return None
        import subprocess
        cmd = [conv, "-w", str(pixel_width), png_path] if conv.endswith("img2sixel") else \
              [conv, png_path, "-geometry", f"{pixel_width}x", "sixel:-"]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        if res.returncode != 0 or not res.stdout:
            return None
        return res.stdout.decode("latin-1")    # sixel data is single-byte control/data, never multibyte UTF-8
    import base64
    with open(png_path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    if protocol == "iterm2":
        return f"\x1b]1337;File=inline=1;width={cols};height={rows};preserveAspectRatio=0:{b64}\a"
    # kitty graphics protocol: chunked (<=4096 B of base64 per chunk), a=T (transmit+display), f=100 (PNG),
    # m=0/1 (last/more chunks to come), c/r: size in terminal cells.
    chunks = [b64[i:i + 4096] for i in range(0, len(b64), 4096)] or [""]
    out = []
    for i, chunk in enumerate(chunks):
        first, last = i == 0, i == len(chunks) - 1
        ctrl = (f"a=T,f=100,c={cols},r={rows}," if first else "") + f"m={0 if last else 1}"
        out.append(f"\x1b_G{ctrl};{chunk}\x1b\\")
    return "".join(out)


UI_GEOMETRY_POSITIONS = ("top", "bottom")


def ui_load_settings(path=UI_CONFIG):
    """keymap, mouse, engine, geometry, recent from a small TOML file (only these keys; anything else is
    ignored)."""
    cfg = {"keymap": "vim", "mouse": True, "engine": "", "geometry": "top", "recent": []}
    try:
        text = open(path, encoding="utf-8").read()
    except OSError:
        return cfg
    for line in text.splitlines():
        m = re.match(r'^\s*(\w+)\s*=\s*(.+?)\s*$', line)
        if not m:
            continue
        key, val = m.groups()
        try:
            if key == "keymap" and json.loads(val) in UI_KEYMAPS:
                cfg["keymap"] = json.loads(val)
            elif key == "mouse":
                cfg["mouse"] = val.lower() == "true"
            elif key == "engine" and isinstance(json.loads(val), str):
                cfg["engine"] = json.loads(val)
            elif key == "geometry" and json.loads(val) in UI_GEOMETRY_POSITIONS:
                cfg["geometry"] = json.loads(val)
            elif key == "recent":
                cfg["recent"] = [p for p in json.loads(val) if isinstance(p, str)][:5]
        except ValueError:
            pass
    return cfg


def ui_save_settings(cfg, path=UI_CONFIG):
    try:
        if os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'keymap = {json.dumps(cfg["keymap"])}\nmouse = {"true" if cfg["mouse"] else "false"}\n'
                    f'engine = {json.dumps(cfg.get("engine", ""))}\n'
                    f'geometry = {json.dumps(cfg.get("geometry", "top"))}\n'
                    f'recent = {json.dumps(cfg["recent"][:5])}\n')
    except OSError:
        pass


def ui_default_engine_guess():
    """The newBRORSEN engine script this UI drives, if it sits next to this UI script (the deploy
    convention: input_builder.vNN.NN.py beside newBRORSEN.vNN.NN.py); '' if not found there."""
    guess = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"newBRORSEN.v{VERSION}.py")
    return guess if os.path.isfile(guess) else ""


def ui_engine_path(cfg):
    """The engine script path this UI drives: the one configured in Settings if it still exists,
    else the same-directory guess, else '' (callers must handle the empty case: no engine found)."""
    configured = cfg.get("engine", "")
    if configured and os.path.isfile(configured):
        return configured
    return ui_default_engine_guess()


def ui_add_recent(cfg, path):
    path = os.path.abspath(path)
    cfg["recent"] = [path] + [p for p in cfg["recent"] if p != path][:4]


def ui_name_ok(path):
    """File names the UI saves and launches: letters, digits, '.', '_', '-', no leading '-'."""
    return bool(UI_NAME_RE.match(os.path.basename(path)))


def _ui_pid_alive(pidfile):
    try:
        pid = int(open(pidfile).read().split()[0])
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def ui_folder_clash(path):
    """Reasons why running `path` in its folder would mix two jobs (one job per folder; fixed run.log)."""
    folder, base = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    stem = os.path.splitext(base)[0]
    reasons = [f"another input: {f}" for f in sorted(os.listdir(folder)) if f.endswith(".inp") and f != base]
    for f in ("run.log", "rc.txt", stem + ".out"):
        if os.path.exists(os.path.join(folder, f)):
            reasons.append(f"existing {f}")
    if _ui_pid_alive(os.path.join(folder, "pid.txt")):
        reasons.append("a job is still running here (pid.txt)")
    return reasons


def ui_move_to_jobdir(path):
    """Move the input into ./<stem>/ next to it; refuses a folder that already holds a job. Returns the new path."""
    folder, base = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    jobdir = os.path.join(folder, os.path.splitext(base)[0])
    if os.path.isdir(jobdir) and os.listdir(jobdir):
        raise FileExistsError(f"{jobdir} already exists and is not empty")
    os.makedirs(jobdir, exist_ok=True)
    new = os.path.join(jobdir, base)
    os.replace(os.path.abspath(path), new)
    return new


def ui_launch(path, wrapper, python=None):
    """Start the engine on `path` in its own folder, detached, without a shell. `wrapper` is the engine
    script's path (see ui_engine_path). Writes pid.txt; returns the pid."""
    import subprocess
    folder, base = os.path.dirname(os.path.abspath(path)), os.path.basename(path)
    if not ui_name_ok(base):
        raise ValueError(f"file name {base!r}: use letters, digits, '.', '_' and '-' only")
    if not wrapper:
        raise RuntimeError("No engine script found. Set its path from the start screen's Settings > Engine path.")
    if _ui_pid_alive(os.path.join(folder, "pid.txt")):
        raise RuntimeError("a job is already running in this folder")
    argv = [python or sys.executable, "-W", "ignore", wrapper, base, "--decimal-comma"]
    with open(os.path.join(folder, "nohup.out"), "w") as log:
        proc = subprocess.Popen(argv, cwd=folder, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
    with open(os.path.join(folder, "pid.txt"), "w") as f:
        f.write(f"{proc.pid}\n")
    return proc.pid


def ui_dry_run(text, wrapper, python=None):
    """--dry-run on a copy in a temporary folder (the user's folder is never touched). `wrapper` is the
    engine script's path (see ui_engine_path). Returns (rc, output)."""
    if not wrapper:
        return 1, "No engine script found. Set its path from the start screen's Settings > Engine path."
    import subprocess
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "dryrun.inp")
        with open(p, "w") as f:
            f.write(text)
        r = subprocess.run([python or sys.executable, "-W", "ignore", wrapper, p,
                            "--dry-run", "--decimal-comma"], cwd=tmp, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).replace(tmp + os.sep, "")


def ui_deep_check(text, wrapper, python=None):
    """The engine's own parser and validator on `text` (subprocess, writes no files). `wrapper` is the
    engine script's path (see ui_engine_path). Returns a list of dicts."""
    if not wrapper:
        return [{"stage": None, "line": None, "severity": "error",
                 "msg": "No engine script found. Set its path from the start screen's Settings > Engine path."}]
    import subprocess
    r = subprocess.run([python or sys.executable, "-W", "ignore", wrapper,
                        "--parse-text-json"], input=text, capture_output=True, text=True)
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [{"stage": None, "line": None, "severity": "error",
                 "msg": "the engine check failed: " + (r.stderr.strip().splitlines() or ["no output"])[-1]}]


_UI_MANUAL_CACHE = {}


def ui_help_text(token, manual=None):
    """Help for '%key', '!WORD' or 'geometry' from MANUAL.md (the table row and its section heading)."""
    manual = manual or os.path.join(os.path.dirname(os.path.abspath(__file__)), "MANUAL.md")
    if manual not in _UI_MANUAL_CACHE:
        try:
            _UI_MANUAL_CACHE[manual] = open(manual, encoding="utf-8").read().splitlines()
        except OSError:
            _UI_MANUAL_CACHE[manual] = []
    lines = _UI_MANUAL_CACHE[manual]
    if not lines:
        return "MANUAL.md not found next to the wrapper."

    def section(prefix, limit=60):
        for i, l in enumerate(lines):
            if l.startswith(prefix):
                out = [l]
                for m in lines[i + 1:]:
                    if m.startswith("## "):
                        break
                    out.append(m)
                return "\n".join(out[:limit])
        return None

    if token.startswith("%"):
        key = token[1:].lower()
        opt = UI_KEYS.get(key)
        names = {key} | ({opt.name} | set(opt.aliases) if opt else set())
        heading = ""
        for l in lines:
            if l.startswith("### "):
                heading = l
            if l.startswith("| `%") and any(f"`%{n}`" in l.split("|")[1] or f"`%{n} " in l.split("|")[1]
                                             or f"`%{n}`" in l.split("|")[2] for n in names):
                cells = [c.strip() for c in l.strip().strip("|").split("|")]
                labels = ["Directive", "Aliases", "Value", "Default", "Notes"]
                body = "\n".join(f"{lab}: {c}" for lab, c in zip(labels, cells) if c)
                return f"{heading.lstrip('# ')}\n\n{body}"
        if opt:
            return f"%{opt.name}  (default {opt.default})\n\n{opt.doc}"
        return f"No MANUAL entry for {token}."
    if token.startswith("!"):
        word = token[1:].upper()
        info = UI_BANG.get(word)
        head = f"! {word}: {info[2]} ({info[0]})\n\n" if info else ""
        return head + (section("## 4.") or "")
    return section("## 6.") or "No MANUAL section 6."


def ui_count_electrons(rows, charge=0):
    """rows: [(symbol, ...)] -> electrons, or None if a symbol is unknown."""
    total = 0
    for r in rows:
        sym, ghost = ui_element(r)
        if sym is None:
            return None
        total += 0 if ghost else UI_Z[sym.upper()]
    return total - charge


def ui_suggest_mult(electrons):
    return None if electrons is None else (1 if electrons % 2 == 0 else 2)


def ui_geometry_block(kind, rows, charge, mult):
    """'* xyz charge mult' (or zmat) block text from row strings."""
    return f"* {kind} {charge} {mult}\n" + "".join(r.rstrip() + "\n" for r in rows) + "*\n"


def ui_read_xyz(path):
    """Rows 'Sym x y z' of an .xyz file (count and comment lines skipped)."""
    lines = [l for l in open(path, encoding="utf-8", errors="replace").read().splitlines()]
    if lines and lines[0].strip().isdigit():
        lines = lines[2:]
    rows = []
    for l in lines:
        t = l.split()
        if len(t) >= 4 and all(_ui_float(x) for x in t[1:4]):
            rows.append(f"{t[0]:<4s} {float(t[1]):>14.8f} {float(t[2]):>14.8f} {float(t[3]):>14.8f}")
    if not rows:
        raise ValueError(f"no 'Sym x y z' rows in {path}")
    return rows


def ui_read_zmat(path):
    """(rows, header) of a Z-matrix file: rows kept verbatim; header = its own '* zmat c m' line if it has one."""
    rows, header = [], None
    for l in open(path, encoding="utf-8", errors="replace").read().splitlines():
        s = l.strip()
        if not s:
            continue
        if s.startswith("*"):
            if header is None and len(s) > 1:
                header = s
            continue
        rows.append(l.rstrip())
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows, header


def ui_geometry_from_text(text, kind=None):
    """Geometry given as text (a file's content, or --ui "geometry"): (kind, rows, charge, mult).
    Rows may be separated by newlines or ';'. Accepted: Cartesian rows 'Sym x y z' (an .xyz file with its
    count and comment lines too), a Z-matrix ('Sym', 'Sym i r', 'Sym i r j a', ... and 'name = value'), with
    or without a '* xyz|zmat <charge> <mult>' header. charge / mult are None when not given (an .xyz comment
    line like '0 1' or 'charge=-1 mult=2', or the header)."""
    if "\n" not in text and ";" in text:
        text = text.replace(";", "\n")
    lines = [l.rstrip() for l in text.splitlines()]
    q = m = None
    head = next((i for i, l in enumerate(lines) if re.match(r"^\s*\*\s*(xyz|zmat|int|internal)\b", l, re.I)), None)
    if head is not None:
        nums = re.findall(r"-?\d+", _ui_strip_comment(lines[head]))
        q = int(nums[0]) if nums else None
        m = int(nums[1]) if len(nums) > 1 else None
        if kind is None and re.search(r"zmat|int", lines[head], re.I):
            kind = "zmat"
        body = []
        for l in lines[head + 1:]:
            if l.strip() in ("*", "end"):
                break
            body.append(l)
    else:
        body = lines
        if body and body[0].strip().isdigit():               # .xyz: count line, then a comment line
            comment = body[1] if len(body) > 1 else ""
            mq = re.search(r"charge\s*[=:]\s*(-?\d+)", comment, re.I)
            mm = re.search(r"(?:mult(?:iplicity)?|spin)\s*[=:]\s*(\d+)", comment, re.I)
            q = int(mq.group(1)) if mq else None
            m = int(mm.group(1)) if mm else None
            if not (mq or mm) and re.match(r"^\s*-?\d+\s+\d+\s*$", comment):
                q, m = (int(x) for x in comment.split())
            body = body[2:]
    body = [l for l in body if l.strip() and l.strip() not in ("*", "end")]
    cart = [l for l in body if len(l.split()) >= 4 and all(_ui_float(x) for x in l.split()[1:4])]
    if kind is None:
        kind = "xyz" if body and len(cart) == len(body) else "zmat"
    if kind == "xyz":
        rows = [f"{t[0]:<4s} {float(t[1]):>14.8f} {float(t[2]):>14.8f} {float(t[3]):>14.8f}"
                for t in (l.split() for l in cart)]
    else:
        rows = [l.strip() for l in body]
    if not rows:
        raise ValueError("no geometry rows found")
    return kind, rows, q, m


def ui_geometry_from_file(path):
    """Geometry of an .xyz or .zmat file for a new input: see ui_geometry_from_text."""
    text = open(path, encoding="utf-8", errors="replace").read()
    kind = "zmat" if path.lower().endswith((".zmat", ".zmt", ".gzmat")) else None
    return ui_geometry_from_text(text, kind)


def ui_looks_like_geometry(arg):
    """--ui "<geometry text>" rather than a file name."""
    return ("\n" in arg or ";" in arg or arg.lstrip().startswith("*")) and not os.path.exists(arg)


def ui_zmat_symbols(rows):
    out = []
    for r in rows:
        t = r.replace(",", " ").split()
        if not t or "=" in r or t[0].lower().rstrip(":") in ("variables", "constants"):
            continue
        if re.match(r"^(x|xx|du|dum|dummy)\d*$", t[0].lower()):
            continue
        out.append(t[0])
    return out


# Undocumented (not in MANUAL.md/USER_GUIDE.md on purpose): Settings > "About" shows ordinary credits;
# Left/Right there also cycles through three small dedications (art framed like a painting + a verse) past
# the ordinary page. The ASCII art is pre-rendered offline from three paintings: ImageMagick for crop and
# decode, numpy for the math (Floyd-Steinberg dithering onto a 10-level ramp for character density; a
# per-cell nearest match against prompt_toolkit's own ANSI_COLORS_TO_RGB, output/vt100.py, for color - the
# only palette this app renders, color_depth=ColorDepth.DEPTH_4_BIT) - kept here as static (style, text)
# data, no image library needed at runtime (no-new-python-libraries-matplotlib-exception).

UI_EASTER_CHRIST = (
    (('ansiyellow', '+=++'), ('', '--:='), ('ansiyellow', '+*##%%#@%%%%%####%###%%%@%%%%%*='), ('', '---'), ('ansicyan', '+'), ('', '+#*'), ('ansiyellow', '*')),
    (('ansiyellow', '='), ('', '+=:.'), ('ansired', ':'), ('ansiyellow', '*##%%%#%%%##@'), ('ansibrightyellow', '@@'), ('ansiyellow', '%%%%@%#%%%@%@%@%%'), ('ansibrightyellow', '%'), ('ansiyellow', '%#+'), ('', '=.==*#')),
    (('', '+'), ('ansiyellow', '='), ('ansired', '=-'), ('ansiyellow', '+####%%#%%'), ('ansibrightyellow', '@'), ('ansiyellow', '%*+*'), ('', '==---==='), ('ansiyellow', '+##%'), ('ansibrightyellow', '@%'), ('ansiyellow', '@%@%@'), ('ansibrightyellow', '@%'), ('ansiyellow', '%'), ('ansibrightyellow', '%'), ('ansiyellow', '#='), ('', '..=+')),
    (('ansiblue', ':'), ('', '.'), ('ansiyellow', '+###%%%#%%%*-'), ('', ':.    ...... ...'), ('ansired', '-'), ('ansiyellow', '*%@%%'), ('ansibrightyellow', '@%'), ('ansiyellow', '%%%%%*'), ('', '. :')),
    (('', ':'), ('ansiyellow', '*##%#%#%'), ('ansibrightyellow', '%'), ('ansiyellow', '%='), ('', '. . . .. .::::..    .:'), ('ansiyellow', '*%%%'), ('ansibrightyellow', '@'), ('ansiyellow', '%%%#%%#'), ('', ':.')),
    (('ansiyellow', '*##%##'), ('ansibrightyellow', '%'), ('ansiyellow', '%#*'), ('', '. ......   . .....   . . :'), ('ansiyellow', '#%%%%%%%#%%=')),
    (('ansiyellow', '#*#%#%%%*'), ('', '  . ... .:'), ('ansired', '-'), ('ansiyellow', '**'), ('ansired', '-'), ('', '--:..  .  .. :'), ('ansiyellow', '#%%%%%%%%#*')),
    (('ansiyellow', '*######*'), ('', '. . .. :'), ('ansiyellow', '*%'), ('ansibrightyellow', '@@'), ('', '@@'), ('ansibrightyellow', '@'), ('ansiyellow', '@#*'), ('ansired', '='), ('', '..      . .'), ('ansiyellow', '#%%%#%###*')),
    (('ansiyellow', '#####%#'), ('ansired', '-'), ('', '   . .'), ('ansiyellow', '+'), ('ansibrightyellow', '@@@@'), ('', '@'), ('ansibrightyellow', '@@@@'), ('ansiyellow', '@@#*'), ('ansired', '-'), ('', ':.       :'), ('ansiyellow', '%####%#%#')),
    (('ansiyellow', '*####%*'), ('', '   . :'), ('ansiyellow', '#'), ('ansibrightyellow', '@'), ('ansiyellow', '%#'), ('', '#%'), ('ansibrightyellow', '@@@@'), ('ansiyellow', '@%*++*'), ('ansired', '='), ('', '...      '), ('ansiyellow', '+%%%####%')),
    (('ansiyellow', '####%#+'), ('', '     '), ('ansiyellow', '##+'), ('ansired', '==-='), ('ansiyellow', '*@'), ('ansibrightyellow', '%'), ('ansiyellow', '+'), ('ansired', '--=='), ('ansiyellow', '++='), ('ansired', '-'), ('', '.       '), ('ansired', '-'), ('ansiyellow', '%##**#%%')),
    (('ansiyellow', '######='), ('', ' . ..'), ('ansiyellow', '#+=+'), ('ansired', ':'), ('ansiyellow', '++#'), ('ansibrightyellow', '@'), ('ansiyellow', '#++='), ('ansired', ':'), ('', ':'), ('ansired', ':'), ('ansiyellow', '+#+'), ('', ':.      :'), ('ansiyellow', '%#######')),
    (('ansiyellow', '#####%='), ('', '    '), ('ansired', ':'), ('ansibrightyellow', '@'), ('', '@@'), ('ansiyellow', '%#%%@'), ('ansibrightyellow', '@'), ('ansiyellow', '*'), ('ansired', '='), ('ansiyellow', '%%**#%'), ('ansibrightyellow', '%'), ('ansiyellow', '+'), ('', ':...    :'), ('ansiyellow', '#*##%###')),
    (('ansiyellow', '##%#%%*'), ('', '.  .'), ('ansired', ':'), ('ansiyellow', '#'), ('ansibrightyellow', '@@@@@@@@'), ('ansiyellow', '#'), ('ansired', '-'), ('ansiyellow', '#@@%%%*'), ('ansired', '--='), ('', ':     '), ('ansired', '='), ('ansiyellow', '##*#%%%#')),
    (('ansiyellow', '###%###'), ('ansired', ':'), ('', '  '), ('ansired', '-'), ('ansiyellow', '*'), ('ansired', '-'), ('ansiyellow', '*@'), ('ansibrightyellow', '@@@@'), ('ansiyellow', '#%#'), ('ansired', '='), ('ansiyellow', '+'), ('ansibrightyellow', '@'), ('ansiyellow', '@%#'), ('ansired', '+:'), ('', '.'), ('ansired', '--'), ('', '..    '), ('ansiyellow', '*###%%%##')),
    (('ansiyellow', '+######='), ('', '   .:'), ('ansired', ':'), ('ansiyellow', '+%'), ('ansibrightyellow', '@'), ('ansiyellow', '#+'), ('ansired', '-=:'), ('', ':'), ('ansired', '-'), ('ansiyellow', '+%#'), ('ansired', '=:'), ('', '.'), ('ansired', ':'), ('', '..     '), ('ansired', '-'), ('ansiyellow', '%%%%#%*#*')),
    (('ansiyellow', '+**%%%##'), ('ansired', ':'), ('', '   '), ('ansired', ':'), ('', '.'), ('ansiyellow', '=#*'), ('ansired', '-='), ('ansiyellow', '+*'), ('ansired', '=-::'), ('ansiyellow', '+%'), ('ansired', '-'), ('', '..'), ('ansired', '.'), ('', ':     .'), ('ansiyellow', '*%##%%#%#'), ('', '.')),
    (('', ':'), ('ansiyellow', '+##%%%##'), ('ansired', '-'), ('', '  '), ('ansired', '::-'), ('ansiyellow', '+'), ('ansired', '='), ('ansiyellow', '+*+'), ('ansired', '-:-'), ('ansiyellow', '++'), ('ansired', '-'), ('ansiyellow', '+'), ('ansired', '-'), ('', ' .'), ('ansired', ':'), ('', '.     '), ('ansiyellow', '+##%%##%#'), ('', '::')),
    (('ansired', '-'), ('', '-'), ('ansiyellow', '***##%##+'), ('', ':.'), ('ansired', '::'), ('', '.'), ('ansired', '--'), ('ansiyellow', '+**+++'), ('ansired', '--:'), ('', '..... .   '), ('ansiyellow', '###%##'), ('ansibrightyellow', '@'), ('ansiyellow', '*'), ('', ':-'), ('ansiyellow', '=')),
    (('ansiyellow', '++'), ('', '=::'), ('ansiyellow', '+#*####'), ('ansired', '::'), ('ansiyellow', '+'), ('', '..:'), ('ansired', ':--=--'), ('', ':.. .'), ('ansired', ':'), ('', '...    '), ('ansired', '-'), ('ansiyellow', '%%###*'), ('', '---'), ('ansiyellow', '+')),
    (('ansiyellow', '=#'), ('', '==:'), ('ansiyellow', '+###+'), ('ansired', '-'), ('', '. .'), ('ansiyellow', '%#'), ('ansired', ':'), ('', '. .....  .'), ('ansired', ':::'), ('', '. .. .  '), ('ansiyellow', '+='), ('ansired', '--'), ('', ':-='), ('ansiyellow', '-=+')),
    (('ansiyellow', '*=++'), ('ansired', '=-'), ('ansiyellow', '=='), ('ansired', '-'), ('', '    '), ('ansired', '-'), ('ansiyellow', '%%%*'), ('ansired', '-'), ('', '.     .'), ('ansired', '::-:'), ('', '..  ....  .-='), ('ansiyellow', '=+*#*')),
    (('ansiyellow', '*'), ('ansired', '-'), ('ansiyellow', '=+='), ('ansired', '-:'), ('', '.  .'), ('ansired', '-'), ('ansiyellow', '%%#%@@#*='), ('ansired', '=--::::-:'), ('', '. ...    .. ....:'), ('ansired', ':')),
    (('ansiyellow', '+==+'), ('ansired', '-'), ('', '....'), ('ansired', '-'), ('ansiyellow', '#%%%##%%%%%***'), ('ansired', '-----'), ('', '... . . . . .  ... ')),
    (('ansiyellow', '='), ('ansired', '--'), ('', ':.. .'), ('ansiyellow', '+#@%%%%#*#*#*++*='), ('ansired', '-=:'), ('', '. .. .   . ........ ')),
    (('', ':.. . . ..'), ('ansired', ':'), ('ansiyellow', '+*%%#*++*+'), ('ansired', '='), ('ansiyellow', '=++='), ('', '.  . ..  .   . . .    ')),
    (('', '......       :'), ('ansiyellow', '+*#%##*##*+'), ('', '. .  ..                ')),
    (('', '.. .'), ('ansired', '.:'), ('', '   .. .  .----'), ('ansired', '=--'), ('', ':    .          . ..  .  ')),
    (('', ' ....'), ('ansired', ':'), ('', '.       .        . ..  .          .       ')),
    (('', '    '), ('ansired', '::='), ('', '.        . .  .  . .    ....:'), ('ansired', '::---=-=='), ('ansiyellow', '=+=')),
    (('', '.. .'), ('ansired', '.---'), ('', '   .        .      .'), ('ansired', ':='), ('ansiyellow', '='), ('ansired', '=-'), ('ansiyellow', '+'), ('ansired', '=-'), ('ansiyellow', '='), ('ansired', '=='), ('ansiyellow', '+++='), ('ansired', '=='), ('ansiyellow', '=='), ('ansired', '-')),
    (('', '  ..'), ('ansired', ':--=:'), ('', '.   .'), ('ansired', ':.'), ('', '.   .    .'), ('ansired', '-'), ('ansiyellow', '**#*=+'), ('ansired', ':'), ('', '..'), ('ansired', '-:'), ('', '.'), ('ansired', '='), ('ansiyellow', '='), ('ansired', '-'), ('', ':.'), ('ansired', '-==-')),
    (('', '   .'), ('ansired', ':-=--:'), ('', '.'), ('ansired', '--::.'), ('', '   .   :'), ('ansired', '-'), ('ansiyellow', '*#*#*+=='), ('ansired', ':'), ('', '.'), ('ansired', '==='), ('ansiyellow', '=+='), ('ansired', '=--'), ('ansiyellow', '='), ('ansired', '='), ('ansiyellow', '==')),
    (('', '   .'), ('ansired', '.----=--:.'), ('', '... .   .'), ('ansired', '='), ('ansiyellow', '**+*#*==+'), ('', '.'), ('ansired', '.==='), ('ansiyellow', '+=='), ('ansired', '--=='), ('ansiyellow', '='), ('ansired', '=-')),
    (('', '  ..'), ('ansired', ':::-:--::'), ('', '..  .    .'), ('ansired', '='), ('ansiyellow', '*++*#*'), ('ansired', '='), ('ansiyellow', '=='), ('', '.'), ('ansired', ':=='), ('ansiyellow', '=*+'), ('ansired', '=-'), ('', ' '), ('ansired', ':-'), ('ansiyellow', '***')),
    (('', '  . '), ('ansired', ':'), ('ansiyellow', '='), ('ansired', '==-----:-:'), ('', '    . '), ('ansired', ':'), ('ansiyellow', '=*==**+'), ('ansired', '==='), ('', '.'), ('ansired', ':-'), ('ansiyellow', '+*+*+'), ('ansired', ':'), ('', '.'), ('ansired', ':+'), ('ansiyellow', '+'), ('ansired', '='), ('ansiyellow', '=')),
    (('', '    '), ('ansired', '='), ('ansiyellow', '*+'), ('ansired', '==----:-:'), ('', '   .  '), ('ansired', ':'), ('ansiyellow', '=++=++='), ('ansired', '---=-==---='), ('', '. .'), ('ansired', ':'), ('', '.'), ('ansired', ':=')),
    (('', '    '), ('ansiyellow', '+*'), ('ansired', '=--:-:::-:'), ('', '   .  '), ('ansired', '-'), ('ansiyellow', '=+=+*#+'), ('ansired', '='), ('ansiyellow', '=='), ('ansired', ':---=:::'), ('', ' ..'), ('ansired', '-.:=')),
    (('', ' .  '), ('ansired', ':=-::.:==-::'), ('', '      '), ('ansired', '-'), ('ansiyellow', '==+=*#+'), ('ansired', '='), ('ansiyellow', '='), ('ansired', '='), ('', '.'), ('ansired', '-=--==='), ('', ':.'), ('ansired', '='), ('ansiyellow', '+*+=')),
    (('', '     '), ('ansired', '-'), ('ansiyellow', '**'), ('ansired', '-::==-:.'), ('', '      '), ('ansired', '-='), ('ansiyellow', '=+=*#+'), ('ansired', '=-'), ('ansiyellow', '=='), ('ansired', '=='), ('ansiyellow', '=+*'), ('ansired', '=:'), ('', '..'), ('ansired', '-'), ('ansiyellow', '**++')),
    (('', '  .  .'), ('ansiyellow', '+*'), ('ansired', '=::--:.'), ('', ' .    .'), ('ansired', ':='), ('ansiyellow', '=+=*+='), ('ansired', '==---='), ('ansiyellow', '++*'), ('ansired', '=:'), ('', ' .'), ('ansired', '--=--')),
    (('', ' . .  .'), ('ansired', '--::::'), ('', '        .'), ('ansired', '-'), ('ansiyellow', '=++=+'), ('ansired', '==-='), ('', '..'), ('ansired', '-=='), ('ansiyellow', '=+'), ('ansired', '--'), ('', '..'), ('ansired', '----'), ('ansiyellow', '=')),
    (('', '    .  .'), ('ansired', ':::'), ('', '. . .  . .'), ('ansired', '--==-==-=-'), ('ansiyellow', '='), ('', '.'), ('ansired', ':'), ('ansiyellow', '+=+'), ('ansired', '='), ('ansiyellow', '='), ('ansired', '=-'), ('', '.'), ('ansired', ':---==')),
)


UI_EASTER_MICHAEL = (
    (('ansired', '+='), ('', '.--:.::..:.  ................:.:::::::::..    ')),
    (('ansired', '=='), ('', ':....:::.:.   ............:.:.:..:::::::.     ')),
    (('', '.+'), ('ansired', '++*='), ('', '-...:... ................:.:...:....      ')),
    (('', ' :='), ('ansired', '-*'), ('ansiyellow', '#%*'), ('ansired', '+-'), ('', '::. .........:-'), ('ansired', '-===-'), ('', ':::.::.::... .    ')),
    (('', '  :-.:'), ('ansired', '=*'), ('ansiyellow', '##'), ('ansired', '+='), ('', '-:+='), ('ansired', '+::-'), ('', ':...:'), ('ansired', '-=+'), ('ansiyellow', '**'), ('ansired', '+'), ('', '-.:..:..:..     .')),
    (('', '   -:  ..:'), ('ansired', '--'), ('', '===*#'), ('ansiyellow', '#'), ('ansired', '+-=:'), ('', '. ..'), ('ansired', '-='), ('ansiyellow', '*%#*'), ('ansired', '-'), ('', ':::::... ..:::.')),
    (('', '.   -:     --:-:---:.'), ('ansired', ':'), ('', '. :'), ('ansired', ':+*+'), ('ansiyellow', '#'), ('ansired', '+=-'), ('', ':::::.. .--::..')),
    (('', '.    -: . .. ..     ..:::'), ('ansired', ':+++-'), ('', ':-:::.. ..--:.:...')),
    (('', ':..   -:          .:::::.:'), ('ansired', ':=-==-'), ('', '::.::-'), ('ansired', '-='), ('', '::...   ')),
    (('', '.:::.. -.  .      .: ..::.:+'), ('ansiyellow', '#'), ('', '#%'), ('ansiyellow', '%#'), ('', '++##'), ('ansired', '+'), ('', '-:.... .  ')),
    (('', ':::'), ('ansired', '::'), ('', '.  -.      .  '), ('ansired', '-'), ('', '-..--'), ('ansiblue', '+*'), ('', '+-+**#**++=....      ')),
    (('', '::'), ('ansired', '::-:'), ('', '   -.      . :'), ('ansimagenta', '%'), ('ansiblue', '+=**+-'), ('', '. :==:-=+='), ('ansired', '='), ('', ': .       ')),
    (('', ':'), ('ansired', '::--::::.'), ('', '-.   .. ..:'), ('ansimagenta', '#+'), ('ansiblue', '-=+'), ('', ':   .::'), ('ansired', '-++--=-'), ('', '..      ')),
    (('', '.:'), ('ansired', ':::------=:'), ('', ' . ....'), ('ansiblue', '-'), ('', ':'), ('ansimagenta', '*#*'), ('ansiblue', '='), ('', ': ..'), ('ansired', '.::=--:--==-'), ('', '.     ')),
    (('', '..'), ('ansired', ':-:-:-:--=-'), ('', '...  ..:'), ('ansiblue', '-:'), ('', '='), ('ansimagenta', '*'), ('ansired', '='), ('', '..'), ('ansired', '.:::-:---=---:'), ('', ':... .')),
    (('', '.:'), ('ansired', '.-:::---::'), ('', '. ....:'), ('ansiblue', '-=+'), ('ansicyan', '*#'), ('ansimagenta', '#+'), ('ansired', '-'), ('', '.'), ('ansired', ':==:------::'), ('', '..      ')),
    (('', '.'), ('ansired', '.:--:--:::'), ('', '. ...'), ('ansiblue', '-'), ('', '.'), ('ansiblue', '-'), ('ansicyan', '*'), ('ansimagenta', '#'), ('ansicyan', '#%'), ('ansimagenta', '@%'), ('ansicyan', '%'), ('ansimagenta', '+'), ('ansired', '+:=+.:.:'), ('', '.   ...-'), ('ansired', '+'), ('', ':   ')),
    (('', '.'), ('ansired', ':-------:'), ('', ' .::..::.'), ('ansiblue', '=+'), ('ansicyan', '#'), ('ansimagenta', '*%'), ('', '%@'), ('ansimagenta', '%'), ('ansiyellow', '@'), ('ansired', '=='), ('', ':..        ..'), ('ansired', '-'), ('ansiyellow', '*'), ('ansired', '+'), ('', '-:')),
    (('', ' '), ('ansired', '--=---==+*-'), ('', ':--:....'), ('ansiblue', '-+'), ('ansimagenta', '#'), ('', '#@'), ('ansimagenta', '@'), ('ansiyellow', '@'), ('', '%%#+==----:.     .'), ('ansired', '-+*')),
    (('', '.'), ('ansired', '---=-+'), ('ansibrightred', '#++##'), ('ansired', '='), ('', '-'), ('ansired', '='), ('', '=-:::-=+#'), ('ansiyellow', '%@'), ('', '#%@@@#**++**'), ('ansiblue', '='), ('', '. . .    ')),
    (('', ' '), ('ansired', ':=---'), ('ansibrightred', '**+*#'), ('ansiyellow', '%%'), ('ansired', '*'), ('', '===:::-.'), ('ansired', '=='), ('ansiyellow', '%'), ('ansimagenta', '%'), ('', '#%%*%'), ('ansiyellow', '@'), ('', '%##'), ('ansimagenta', '*'), ('ansiblue', '++'), ('', ':  ..::   ')),
    (('', '  '), ('ansired', ':---:-'), ('ansibrightred', '===*'), ('ansiyellow', '@@#'), ('ansired', '-='), ('', '-.=:..'), ('ansired', '-'), ('ansiyellow', '@'), ('ansimagenta', '*'), ('', '-###%'), ('ansiyellow', '%@%'), ('', '+'), ('ansiblue', '+'), ('', ':.  .:::.   ')),
    (('', '   ..'), ('ansired', ':='), ('ansibrightred', '='), ('ansired', '=-==='), ('ansibrightred', '#'), ('ansiyellow', '@%'), ('ansired', '='), ('', '*=..=..'), ('ansired', '+'), ('ansiyellow', '%'), ('ansired', '='), ('', '-'), ('ansicyan', '#'), ('', '#'), ('ansiyellow', '#@@%'), ('', ':  .:'), ('ansired', '-='), ('', '-      ')),
    (('', '    .'), ('ansired', '+'), ('ansibrightred', '+=='), ('ansired', '-'), ('ansibrightred', '==='), ('ansired', '-'), ('ansibrightred', '*#'), ('ansired', '+'), ('', '-.-=. :.:'), ('ansired', '+'), ('', ':'), ('ansiblue', '-'), ('ansired', '-='), ('ansiyellow', '*#%%'), ('ansired', '-'), ('', '-'), ('ansired', '=+'), ('', '-        ')),
    (('', '  :'), ('ansired', '='), ('ansibrightred', '*#++++++**+'), ('ansired', '='), ('', '...-:  -: .::.:'), ('ansired', '-=+'), ('ansiyellow', '%%'), ('ansired', '='), ('', '           ')),
    (('', ' .'), ('ansibrightred', '#'), ('ansiyellow', '%'), ('ansibrightred', '%#**+*##*'), ('ansired', '+:'), ('', '..:::.......     .'), ('ansired', ':='), ('ansiyellow', '*@*'), ('', ':         ')),
    (('', '  '), ('ansired', '-'), ('ansibrightred', '####*+++='), ('ansired', '='), ('', '. :'), ('ansired', ':--:'), ('', '.   .         .'), ('ansired', '='), ('ansiyellow', '*@%'), ('', ':        ')),
    (('', '. '), ('ansired', '='), ('ansibrightred', '#%@#%*+'), ('ansired', '=::'), ('', '   '), ('ansired', ':----'), ('', '              '), ('ansired', '-+*'), ('ansiyellow', '#%'), ('', '-       ')),
    (('', ' .'), ('ansired', '-*+===:'), ('', '.      :'), ('ansired', '+=='), ('', ':      . .:.    :'), ('ansired', '+**'), ('ansiyellow', '*'), ('', '-.     ')),
    (('', '         . .    :'), ('ansired', '=+=:'), ('', '         .:     .-'), ('ansiblue', '-'), ('ansired', '+*'), ('', '     .')),
    (('', '      .         :'), ('ansired', '==+=-'), ('', '                .'), ('ansired', '-='), ('ansiyellow', '%'), ('ansired', '='), ('', '.'), ('ansired', '---'), ('', '.')),
    (('', '  ..             '), ('ansired', ':=+=-'), ('', ':.         .:'), ('ansired', ':-++=-+'), ('ansiyellow', '%'), ('ansired', '*'), ('ansiyellow', '*'), ('ansired', '*++')),
    (('', '..... ..    .    .'), ('ansired', ':-=-:'), ('', '...      ...'), ('ansired', '--:==*:-+'), ('ansiyellow', '*'), ('ansired', '*'), ('ansiyellow', '*#')),
    (('', '....   ..:. ..     ::'), ('ansired', '='), ('', '-:    ...:'), ('ansired', ':-==+=:::'), ('', ':   .'), ('ansired', '+-')),
    (('', ' ..       ::....    .'), ('ansired', '-*-'), ('', ' ......'), ('ansired', '-+'), ('ansiyellow', '#%#'), ('ansired', '*=:'), ('', '.  ..:'), ('ansired', '=+-')),
    (('', '           ....      .'), ('ansired', '='), ('ansiyellow', '#'), ('', '.  ...'), ('ansired', ':.+'), ('ansiyellow', '%'), ('ansired', '*+-'), ('', '..   :.....')),
    (('', '  .       ...         '), ('ansired', '='), ('ansiyellow', '%'), ('ansired', '+'), ('', '    ..:'), ('ansired', ':--:'), ('', ':.     .    ')),
    (('', '  ..       .::.      .'), ('ansired', '+'), ('ansiyellow', '%'), ('ansired', '+'), ('', '   .'), ('ansired', ':'), ('', '.'), ('ansired', '::'), ('', '.   .          ')),
)


UI_EASTER_MARY = (
    (('ansired', '---='), ('', '='), ('ansired', '='), ('ansiyellow', '+'), ('', '+'), ('ansiyellow', '#'), ('ansired', '=:::'), ('', '.'), ('ansired', ':'), ('', '::'), ('ansired', '-'), ('', '=='), ('ansired', '+'), ('ansiyellow', '='), ('', '='), ('ansired', '+'), ('', '....'), ('ansired', '-:'), ('', '.'), ('ansired', ':='), ('ansiyellow', '+++++='), ('ansired', '=::.:-::=')),
    (('', '..:'), ('ansired', ':'), ('', '.'), ('ansired', '='), ('ansiyellow', '**'), ('ansired', '=:.::'), ('', ':'), ('ansired', ':-'), ('', '='), ('ansiyellow', '*'), ('', '**'), ('ansiyellow', '%%'), ('', '#'), ('ansiyellow', '*='), ('ansired', '-:'), ('', '.  .'), ('ansired', '.:'), ('ansiyellow', '=++'), ('ansired', '-::::::-===-')),
    (('', ' . :.'), ('ansired', ':==--='), ('ansiyellow', '+*******#%%#**++'), ('ansired', '=--:..'), ('', ':  .'), ('ansired', '.:::--::-=+')),
    (('', ':.'), ('ansired', '---'), ('ansiyellow', '**+******+***##%@#*+*+++'), ('ansired', '=.:.'), ('', '.     .'), ('ansired', '.::..'), ('', '  '), ('ansired', ':')),
    (('ansiyellow', '*'), ('ansired', ':='), ('ansiyellow', '*++***+*******#*#%%***++*+'), ('ansired', '='), ('', '  .         '), ('ansired', '.:-:::')),
    (('ansiyellow', '++'), ('ansired', ':'), ('ansiyellow', '+++*#********##%%@%##*#*+*'), ('ansired', '::'), ('', '.         . ..'), ('ansired', ':::')),
    (('ansiyellow', '+#='), ('ansired', ':'), ('ansiyellow', '++***#++**###*'), ('ansired', '-'), ('', ':'), ('ansired', ':::::-'), ('ansiyellow', '*##='), ('ansired', '--'), ('', '           ..'), ('ansired', ':'), ('', '..')),
    (('ansiyellow', '+*#'), ('ansired', '-'), ('', '.'), ('ansiyellow', '****+***###'), ('ansired', '-:'), ('', '.'), ('ansired', '.-'), ('ansiyellow', '+'), ('ansired', '-:'), ('', '   '), ('ansired', '-'), ('ansiyellow', '#*'), ('ansired', '--.'), ('', '             .:')),
    (('ansired', '='), ('ansiyellow', '+**'), ('', ':'), ('ansired', '.'), ('ansiyellow', '*#*++****'), ('ansired', '-'), ('', '..'), ('ansired', '='), ('ansiyellow', '%@@@#'), ('ansired', '='), ('', '. .'), ('ansired', ':'), ('ansiyellow', '#+'), ('ansired', ':-'), ('', '               ')),
    (('', ' .'), ('ansired', '-'), ('ansiyellow', '++'), ('', ':'), ('ansired', ':'), ('ansiyellow', '*%**++**'), ('', '. .'), ('ansiyellow', '***#+'), ('ansired', '==-'), ('', '.  '), ('ansired', '='), ('ansiyellow', '#'), ('ansired', '-.-'), ('', '  .           ')),
    (('', '    .'), ('ansired', ':'), ('', '.'), ('ansired', '--'), ('ansiyellow', '+*****'), ('ansired', ':'), ('', ' '), ('ansired', '-'), ('ansiyellow', '*'), ('ansired', '='), ('ansiyellow', '+%++'), ('ansired', '=-:'), ('', '.  '), ('ansiyellow', '*+'), ('', '.'), ('ansired', '.-'), ('', '.            ')),
    (('', '     . .. '), ('ansired', '='), ('ansiyellow', '*+*#'), ('ansired', '='), ('', ' .'), ('ansiyellow', '*#@#'), ('ansired', '='), ('ansiyellow', '+#'), ('ansired', '=:'), ('', '. '), ('ansired', '-'), ('ansiyellow', '**'), ('ansired', '-'), ('', '. '), ('ansired', ':'), ('', '            ')),
    (('', '        ..'), ('ansired', '-'), ('ansiyellow', '*****'), ('ansired', '-'), ('', ' '), ('ansired', ':'), ('ansiyellow', '***'), ('ansired', '====.'), ('', ' :'), ('ansiyellow', '#**+'), ('ansired', '+:'), ('', '.       . .. ')),
    (('', '       . '), ('ansired', ':'), ('', ':'), ('ansired', ':-='), ('ansiyellow', '+*'), ('', '#'), ('ansiyellow', '='), ('ansired', ':'), ('ansiyellow', '++++'), ('ansired', '==='), ('', ' :'), ('ansiyellow', '+@#=+++'), ('ansired', '=--'), ('', '.  . .. ..')),
    (('', '         .'), ('ansired', ':'), ('', '  ..'), ('ansiyellow', '*@#@@%+'), ('ansired', '=-=-'), ('', ' '), ('ansired', '-'), ('ansiyellow', '#@#'), ('ansired', '='), ('ansiyellow', '+'), ('ansired', '===+-'), ('', '.. .  .   ')),
    (('', '          .:'), ('ansiblue', '::-'), ('', '='), ('ansiyellow', '#@%@@%##*'), ('ansired', '='), ('', '.'), ('ansired', '='), ('ansiyellow', '%'), ('', '@'), ('ansiyellow', '#+'), ('ansired', '=-=='), ('', '.    .      ')),
    (('', '        .::'), ('ansiblue', '::---'), ('', '-'), ('ansiyellow', '+#@'), ('ansibrightyellow', '@'), ('ansiyellow', '@%#*'), ('ansired', '+:'), ('ansiyellow', '+@'), ('', '@'), ('ansiyellow', '%=*'), ('', ' ..            ')),
    (('', '       ..:'), ('ansiblue', ':'), ('', '.:.:'), ('ansiblue', '::'), ('', '.'), ('ansired', '-'), ('ansiyellow', '#%@%##+'), ('ansired', '--='), ('ansiyellow', '@@#'), ('', '#'), ('ansired', ':'), ('', '              ')),
    (('', '     ..'), ('ansiblue', '::::'), ('', '..:.'), ('ansiblue', ':-:'), ('', '.'), ('ansired', '=='), ('ansiyellow', '+####+'), ('ansired', '=.='), ('ansiyellow', '%#*'), ('ansired', '='), ('', '  .           ')),
    (('', '    .'), ('ansiblue', ':::'), ('', '.'), ('ansiblue', ':::'), ('', ':'), ('ansiblue', '::'), ('', '.'), ('ansiblue', ':-'), ('', '-'), ('ansiyellow', '+++#*##*'), ('ansired', '=-::'), ('ansiyellow', '***'), ('', '- .           ')),
    (('', '  .'), ('ansiblue', ':'), ('', '. ....::.-:-'), ('ansiyellow', '%'), ('ansibrightyellow', '@@@'), ('ansiyellow', '@%%##+++'), ('ansired', ':--:+'), ('ansiyellow', '#*'), ('', '. ..         ')),
    (('', '  .......:'), ('ansired', '--+'), ('ansimagenta', '*'), ('ansired', '+=='), ('ansiyellow', '%#'), ('ansired', '=-'), ('ansiyellow', '===++='), ('ansired', '=-'), ('', ': . '), ('ansired', '+'), ('ansiyellow', '+'), ('', '.   .        ')),
    (('', '...:'), ('ansiblue', ':--'), ('', ':.:'), ('ansired', '=+='), ('ansimagenta', '*+'), ('ansired', '='), ('', '..        ..     ..  .    ..    ')),
    (('ansiblue', '-:'), ('', ':'), ('ansiblue', ':--:'), ('', '.'), ('ansiblue', '::'), ('ansired', '-'), ('ansimagenta', '*'), ('ansired', ':'), ('ansimagenta', '*'), ('ansired', '=+'), ('', '.        .            .    '), ('ansired', '-'), ('', '.   ')),
    (('ansiblue', ':-:-:-'), ('', '. .::'), ('ansired', '=-:=='), ('', '.   ....  '), ('ansired', '::'), ('', '.'), ('ansired', ':'), ('', '.       .    '), ('ansired', ':'), ('', '    ')),
    (('ansiblue', ':-::::'), ('', '. .'), ('ansiblue', ':'), ('', '. '), ('ansired', '=:=:'), ('', '      ...  '), ('ansired', ':.:...'), ('', '     .   .'), ('ansired', ':'), ('', ' .. ')),
    (('', ':'), ('ansiblue', '::'), ('', ':'), ('ansiblue', ':'), ('', '.  .'), ('ansiblue', '-'), ('', ' .'), ('ansired', '+=-'), ('', '.       ..'), ('ansiblue', ':'), ('', '  '), ('ansired', '.'), ('', ' '), ('ansired', '..'), ('', '     . .  '), ('ansired', '::'), ('', ' .'), ('ansired', '::')),
    (('', '..'), ('ansiblue', ':'), ('', '.    :. '), ('ansired', '--=-'), ('', '.       .  ..   .      ..  '), ('ansired', '.-'), ('', '  '), ('ansired', '==')),
    (('', ' ..    .: ....       ..    .  . .          .'), ('ansired', ':'), ('', '.'), ('ansired', '=-')),
    (('', '..    :.          ..     .. .  ............  .'), ('ansired', ':=')),
    (('', '.    .      .....:..:.... .  ...........:.  .. '), ('ansired', ':')),
    (('', '   .   ..:'), ('ansiblue', ':-::::::::--:--:-::'), ('', ':... .   .. . . ...')),
    (('', '     :'), ('ansiblue', '::::-----:::::::-:--::---'), ('', ' .'), ('ansiblue', ':'), ('', '   .'), ('ansiblue', ':------:'), ('', ':.')),
)

UI_EASTER_PAGES = (
    (None, None, None, None),
    (UI_EASTER_MARY, "To the Blessed Virgin Mary, Seat of Wisdom",
     ("Behold, I am the handmaid of the Lord;", "let it be to me according to your word."), "Luke 1:38"),
    (UI_EASTER_CHRIST, None,
     ("For by him all things were created...", "visible and invisible...",
      "and in him all things hold together."), "Colossians 1:16-17"),
    (UI_EASTER_MICHAEL, "To Saint Michael the Archangel",
     ("And there was war in heaven: Michael and his angels", "fought against the dragon..."),
     "Revelation 12:7"),
)


def ui_easter_page_lines(i):
    """Styled lines for Settings > About page i: each line is a tuple of (style, text) fragments (style
    "" = the terminal's own default foreground). Page 0 is the ordinary credits; 1-3 are the hidden
    dedications (art framed like a painting, colored per-cell). A pure function (no prompt_toolkit import),
    so it is testable on its own; ui_easter_plain_text(i) gives the plain-text form for that."""
    i = i % len(UI_EASTER_PAGES)
    if i == 0:
        return tuple((("", t),) for t in (
            "newBRORSEN input builder", "",
            "To Claude (the AI that helped build this).", "",
            "To diruiz <diruiz@unal.edu.co>", "To areyesv <areyesv@unal.edu.co>", "",
            "And to all the QCC students who helped develop and test it."))
    art, title, verse, cite = UI_EASTER_PAGES[i]
    w = sum(len(t) for _, t in art[0])
    border = (("", "+" + "-" * (w + 2) + "+"),)
    lines = [border]
    lines += [(("", "| "),) + row + (("", " |"),) for row in art]
    lines.append(border)
    lines.append((("", ""),))
    if title:
        lines.append((("", title),))
        lines.append((("", ""),))
    quoted = [f'"{verse[0]}' if n == 0 else v for n, v in enumerate(verse)]
    quoted[-1] = quoted[-1] + '"'
    lines += [(("", v),) for v in quoted]
    lines.append((("", ""),))
    lines.append((("", f"-- {cite}"),))
    return tuple(lines)


def ui_easter_plain_text(i):
    """Plain text of ui_easter_page_lines(i), one string per line - for tests and width checks."""
    return ["".join(t for _, t in line) for line in ui_easter_page_lines(i)]

# ==== v04.08 UI: the prompt_toolkit application (imported only under --ui) ====

def _ui_make_app(path=None, cfg=None, cfg_path=UI_CONFIG, input=None, output=None):
    """Build the input-builder Application. Returns (app, state); state is a dict the tests can inspect."""
    import asyncio
    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.completion import Completer, Completion, PathCompleter
    from prompt_toolkit.document import Document
    from prompt_toolkit.enums import EditingMode
    from prompt_toolkit.filters import Condition, has_completions, has_focus, vi_navigation_mode
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.key_binding.bindings.focus import focus_next, focus_previous
    from prompt_toolkit.key_binding.vi_state import InputMode
    from prompt_toolkit.layout import (ConditionalContainer, DynamicContainer, Float, FloatContainer, HSplit,
                                       HorizontalAlign,
                                       Layout, VSplit, Window, WindowAlign)
    from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
    from prompt_toolkit.layout.dimension import D
    from prompt_toolkit.layout.menus import CompletionsMenu
    from prompt_toolkit.layout.processors import Processor, Transformation
    from prompt_toolkit.lexers import Lexer
    from prompt_toolkit.output.color_depth import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Box, Button, Frame, Label, SearchToolbar, TextArea
    from prompt_toolkit.search import start_search

    cfg = cfg if cfg is not None else ui_load_settings(cfg_path)
    st = {"screen": "start", "path": None, "saved_text": "", "diags": [], "by_line": {}, "lexed": [],
          "stages": [], "message": "", "help": None, "cmd": False, "floats_focus": [], "viewer": None,
          "view": None, "show_hidden": False}

    # ---------- editor buffer, lexer, completer ----------
    def effective(blank=False, text=None):
        """The editor text without the gray parts: the template (ui_tpl_effective) and gray syntax rows."""
        text = editor.text if text is None else text
        text = ui_tpl_effective(text, st.get("tpl"), blank)
        syn = st.get("syntax_rows") or set()
        if syn:
            lines = text.split("\n")
            text = "\n".join(("" if l in syn else l) for l in lines) if blank else \
                "\n".join(l for l in lines if l not in syn)
        return text

    def analyse(text):
        if st.get("tpl") or st.get("syntax_rows"):
            text = effective(True, text)
        st["lexed"] = ui_lex(text)
        st["stages"] = ui_stages(st["lexed"])
        diags = ui_check(text)
        row = st.get("paste_row")
        if row is not None:
            # mid-paste: the placeholder '* xyz' header does not yet know what is being pasted in (xyz
            # rows or a Z-matrix - finish_paste figures that out from F2 on), so its rows are not real xyz
            # rows yet either way and should not be flagged as invalid ones while the user is still pasting
            lines = text.split("\n")
            end = next((r for r in range(row + 1, len(lines)) if lines[r].strip() in ("*", "end")), len(lines))
            diags = [d for d in diags if not (row < d.line < end)]
        st["diags"] = diags
        by = {}
        for d in st["diags"]:
            by.setdefault(d.line, []).append(d)
        st["by_line"] = by

    def stage_at(row):
        if row >= len(st["lexed"]):
            return st["stages"][-1] if st["stages"] else None
        num = st["lexed"][row].stage
        for s in st["stages"]:
            if s.lines and s.lines[0].stage == num:
                return s
        return None

    class InputLexer(Lexer):
        def invalidation_hash(self):
            # prompt_toolkit reuses the coloring while the text is unchanged; keeping a gray line with Tab
            # changes only the template state, so that state is part of the key
            tpl = st.get("tpl")
            return (id(tpl), tuple(sorted(tpl["items"])) if tpl else (), frozenset(tpl["bang"]) if tpl else (),
                    frozenset(tpl["hl"]) if tpl else (), frozenset(st.get("syntax_rows") or ()),
                    id(st.get("by_line")))

        def lex_document(self, document):
            lexed, by = st["lexed"], st["by_line"]
            tpl = st.get("tpl")
            dl = document.lines
            grows = ui_tpl_ghost_rows(dl, tpl) if tpl else {}
            brow = ui_tpl_bang_row(dl, ui_tpl_stage_range(dl, tpl["stage_ord"])) if tpl and tpl["bang"] else None
            syn = st.get("syntax_rows") or set()

            def get_line(i):
                text = document.lines[i]
                if not text:
                    return []
                if i in grows:
                    return [("class:ghost.hl" if grows[i] in tpl["hl"] else "class:ghost", text)]
                if text in syn:
                    return [("class:ghost", text)]
                if i == brow:
                    out = []
                    for m in re.finditer(r"\s+|[^\s]+", text):
                        w = m.group()
                        up = w.upper()
                        out.append(("class:ghost.hl" if up in tpl["bang"] and up in tpl["bang_hl"] else
                                    "class:ghost" if up in tpl["bang"] else
                                    "class:bang" if w.strip() and w != "!" else "", w))
                    return out
                styles = [""] * len(text)
                code_len = len(_ui_strip_comment(text))
                for k in range(code_len, len(text)):
                    styles[k] = "class:comment"
                kind = lexed[i].kind if i < len(lexed) else ""
                code = text[:code_len]
                if kind == "bang":
                    for m in re.finditer(r"\S+", code):
                        if m.group() != "!":
                            g = ui_bang_group(m.group().lstrip("!"))
                            for k in range(m.start(), m.end()):
                                styles[k] = "class:bang"
                elif kind in ("directive", "block_open"):
                    m = re.match(r"\s*%\s*\w+", code)
                    if m:
                        for k in range(m.end()):
                            styles[k] = "class:key"
                        for k in range(m.end(), code_len):
                            styles[k] = "class:value"
                elif kind in ("geom_header", "geom_end", "block_end", "next"):
                    for k in range(code_len):
                        styles[k] = "class:struct"
                elif kind == "geom_row":
                    m = re.match(r"\s*\S+", code)
                    if m:
                        for k in range(m.end()):
                            styles[k] = "class:element"
                for d in by.get(i, []):
                    if d.end > d.start:
                        cls = " class:error" if d.severity == "error" else " class:warning"
                        for k in range(max(0, d.start), min(len(text), d.end)):
                            styles[k] += cls
                out, cur, buf = [], None, ""
                for ch, sty in zip(text, styles):
                    if sty != cur and buf:
                        out.append((cur, buf))
                        buf = ""
                    cur = sty
                    buf += ch
                if buf:
                    out.append((cur, buf))
                return out
            return get_line

    class InputCompleter(Completer):
        def get_completions(self, document, complete_event):
            if st.get("suppress_complete"):
                return                       # right after %next: no popup until the user types again
            line = document.current_line_before_cursor
            stripped = line.lstrip()
            word = re.search(r"[^\s=]*$", line).group()
            if stripped.startswith("!"):
                prefix = word.lstrip("!").upper()
                rest = line[:len(line) - len(word)]
                shown = set()
                for w, g, doc in ui_suggest_bang(rest):
                    if not w.startswith(prefix) or (not prefix and (g, doc) in shown):
                        continue                        # empty prefix: one word per meaning, not every alias
                    shown.add((g, doc))
                    yield Completion(w, start_position=-len(word.lstrip("!")), display_meta=f"{g}: {doc}")
                return
            if stripped.startswith("%"):
                m = re.match(r"^\s*%(\w*)$", line)
                if m:                                   # completing the key
                    prefix = m.group(1).lower()
                    stage = stage_at(document.cursor_position_row)
                    first, last = ui_suggest_keys(stage, show_hidden=st.get("show_hidden", False))
                    chosen = ui_stage_chosen(stage)
                    if "next".startswith(prefix) and stage is not None and (stage.geometry or stage.inherited_from):
                        yield Completion("next", start_position=-len(prefix), display_meta="start a new stage")
                    for o in first + last:
                        if not ui_tpl_option_fits(o.name, chosen) or not ui_option_dep_fits(o.name, stage):
                            continue        # not applicable to this stage's run type / sector / method yet
                        hidden = o.name in UI_DEPRECATED_KEYS
                        for n in (o.name,) + tuple(o.aliases):
                            if n.startswith(prefix) and (n == o.name or prefix):
                                meta = f"{o.default} (default) - {o.doc}" if o.default else o.doc
                                yield Completion(n, start_position=-len(prefix),
                                                 display_meta=f"[deprecated] {meta}" if hidden else meta,
                                                 style="class:dim" if (hidden or o not in first) else "")
                    return
                m = re.match(r"^\s*%(\w+)\s*(?:=\s*|\s+)(\S*)$", line)
                if m and m.group(1).lower() in UI_KEYS:  # completing the value
                    o = UI_KEYS[m.group(1).lower()]
                    # Tab (or our own Tab-accept, right after keeping the gray line): show every choice,
                    # not just the ones starting with what's already there - the value typed so far is
                    # usually the default itself, which would otherwise be the one exact (and useless,
                    # "does nothing") match, hiding the menu instead of letting the user browse alternatives.
                    show_all = complete_event is not None and complete_event.completion_requested
                    if o.name in UI_BASIS_KEYS:          # basis sets that PySCF / pyscf_neo_cc ship
                        typed = "" if show_all else m.group(2).lower().replace("-", "").replace("_", "")
                        n = 0
                        for c in ui_basis_names(UI_BASIS_KEYS[o.name]):
                            if c.lower().replace("-", "").replace("_", "").startswith(typed):
                                yield Completion(c, start_position=-len(m.group(2)),
                                                 display_meta="default" if c.lower() == o.default.lower() else "")
                                n += 1
                                if n >= 60:
                                    break
                        return
                    if o.name in UI_CHOICE_DOCS:
                        for c, doc in UI_CHOICE_DOCS[o.name]:
                            if show_all or c.lower().startswith(m.group(2).lower()):
                                yield Completion(c, start_position=-len(m.group(2)), display_meta=doc)
                        return
                    for c in o.choices or ():
                        if show_all or c.lower().startswith(m.group(2).lower()):
                            yield Completion(c, start_position=-len(m.group(2)),
                                             display_meta="default" if c.lower() == o.default.lower() else "")
                return
            row = document.cursor_position_row
            if row < len(st["lexed"]) and st["lexed"][row].kind == "geom_row" and " " not in stripped and stripped:
                for s in _UI_ELEMENTS:
                    if s.lower().startswith(stripped.lower()):
                        yield Completion(s, start_position=-len(stripped))
            elif (row < len(st["lexed"]) and st["lexed"][row].kind == "block_row"
                  and st["lexed"][row].block == "pos_basis_map" and stripped):
                if " " not in stripped:                   # first token: the element
                    for s in _UI_ELEMENTS:
                        if s.lower().startswith(stripped.lower()):
                            yield Completion(s, start_position=-len(stripped))
                else:                                      # second token: a positron basis set
                    typed = word.lower().replace("-", "").replace("_", "")
                    n = 0
                    for c in ui_basis_names("pos"):
                        if c.lower().replace("-", "").replace("_", "").startswith(typed):
                            yield Completion(c, start_position=-len(word))
                            n += 1
                            if n >= 60:
                                break

    def suggestion_for(document):
        """Gray text for the cursor position (common '!' lines, %keys and values, '%next' after a finished
        stage), or None. Computed on the spot, so what Tab / Right inserts is always what is shown."""
        if document.current_line_after_cursor.strip():
            return None
        if get_app().vi_state.input_mode == InputMode.NAVIGATION and cfg["keymap"] == "vim" \
                and document.current_line.strip():
            return None                     # normal mode: the cursor sits on a character, not after the text
        before = document.current_line_before_cursor.strip()
        if before in ("", "%", "%n", "%ne", "%nex") and next_is_due(document):
            return "%next"[len(before):] or None
        if not before:
            return None
        return ui_suggest_text(document.current_line_before_cursor)

    class GhostText(Processor):
        """Draws the suggestion after the cursor on the cursor's line (prompt_toolkit's own AutoSuggest only
        draws on the last line of the document)."""
        def apply_transformation(self, ti):
            doc = ti.document
            if ti.lineno == doc.cursor_position_row and get_app().layout.has_focus(editor):
                text = suggestion_for(doc)
                if text:
                    return Transformation(ti.fragments + [("class:auto-suggestion", text)])
            return Transformation(ti.fragments)

    def next_is_due(document):
        """A '%next' suggestion fits here: the stage above has its '!' line and a finished geometry block (or
        continues the previous stage), and the next non-empty line (if any) is not already a '%next'."""
        row = document.cursor_position_row
        lines = effective(True, document.text).split("\n") if (st.get("tpl") or st.get("syntax_rows")) \
            else document.lines
        later = [l.strip().lower() for l in lines[row + 1:] if l.strip()]
        if later and later[0].startswith(("%next", "%newjob")):
            return False
        above = next((l.strip().lower() for l in reversed(lines[:row]) if l.strip()), "")
        if above.startswith(("%next", "%newjob")):
            return False
        lexed = ui_lex("\n".join(lines[:row]))
        stages = ui_stages(lexed)
        if not stages:
            return False
        cur = stages[-1]
        has_bang = any(l.kind == "bang" for l in cur.lines)
        closed = any(l.kind == "geom_end" for l in cur.lines) or (cur.inherited_from is not None and has_bang)
        last = next((l for l in reversed(lexed) if l.kind not in ("blank", "comment")), None)
        excluded = ("geom_row", "block_row", "block_open", "geom_header")
        if cfg.get("geometry", "top") == "top":
            # the geometry sits right after the '!' line here: just having finished it is not "done with
            # this stage" (the options usually come after it), unlike the bottom layout, where the
            # geometry is the last thing written and finishing it really does mean the stage is complete
            excluded += ("geom_end",)
        return bool(has_bang and closed and last is not None and last.kind not in excluded)

    def gutter(i, w):
        if any(d.severity == "error" for d in st["by_line"].get(i, [])):
            return [("class:gutter.error", "E")]
        if st["by_line"].get(i):
            return [("class:gutter.warning", "W")]
        tpl = st.get("tpl")
        # only once a sector has actually been chosen (! POSITRON/ELECTRONIC/PROTON kept, or inherited by a
        # continuation stage) -- not from the moment the template opens, before anything is picked
        if (tpl and ui_tpl_ghost_rows(editor.document.lines, tpl).get(i) in UI_TPL_NEEDED and
                ui_tpl_chosen(editor.document.lines, tpl).get("sector")):
            return [("class:gutter.warning", "W")]
        return [("", " ")]

    search_bar = SearchToolbar(ignore_case=True)
    editor = TextArea(multiline=True, scrollbar=True, line_numbers=True, wrap_lines=False, focus_on_click=True,
                      lexer=InputLexer(), completer=InputCompleter(), complete_while_typing=True,
                      input_processors=[GhostText()], search_field=search_bar, get_line_prefix=gutter)

    # TextArea wraps its lexer in a DynamicLexer whose cache key ignores ours; give the control the lexer
    # itself, so a line kept with Tab (text unchanged, template state changed) is redrawn in its new colors
    editor.control.lexer = InputLexer()

    def snapshot():
        return (copy.deepcopy(st.get("tpl")), frozenset(st.get("syntax_rows") or ()))

    def on_change(_buf):
        if not st.get("applying"):
            st["suppress_complete"] = False   # a real keystroke: the %next popup-suppression is over
            snap = st.setdefault("snap", {}).get(editor.text)
            if snap is not None:                  # undo / redo back to a template state: restore it
                st["tpl"], st["syntax_rows"] = copy.deepcopy(snap[0]), set(snap[1])
            elif st.get("tpl"):
                st["tpl"] = ui_tpl_realize_edited(editor.document.lines, st["tpl"])
                want = {iid for iid, _, _ in ui_tpl_desired(editor.document.lines, st["tpl"])}
                if want != set(st["tpl"]["items"]) and not st.get("rebuild_pending"):
                    st["rebuild_pending"] = True          # e.g. '%opt_engine zopt' typed: its options appear
                    asyncio.get_event_loop().call_soon(rebuild_template)
        analyse(editor.text)
        if st.get("auto_jobname") is not None and not st.get("jobname_pending"):
            st["jobname_pending"] = True
            asyncio.get_event_loop().call_soon(maybe_auto_jobname)

    def maybe_auto_jobname():
        """Keeps stage 1's %jobname at formula_method_q_m_basis (ui_auto_jobname) as geometry/method/
        charge/mult/basis become known, as long as the line still holds the value this function itself
        last wrote (or the untouched 'Stage_1' default) - a name the user typed by hand is left alone."""
        st["jobname_pending"] = False
        auto = st.get("auto_jobname")
        if auto is None:
            return
        lines = editor.document.lines
        start, end = ui_tpl_stage_range(lines, 0)
        row = next((i for i in range(start, end) if re.match(r"^\s*%jobname\b", lines[i], re.I)), None)
        if row is None:
            return
        m = re.match(r"^\s*%jobname\s*=?\s*(\S*)", lines[row], re.I)
        if (m.group(1) if m else "") != auto:
            st["auto_jobname"] = None             # the user changed it: stop managing it from now on
            return
        stages = st.get("stages") or []
        if not stages:
            return
        name = ui_auto_jobname(stages[0])
        if not name or name == auto:
            return
        doc = editor.document
        cur_row, cur_col = doc.cursor_position_row, doc.cursor_position_col
        new_lines = list(lines)
        new_lines[row] = ui_option_line("jobname", name, UI_KEYS["jobname"].doc)
        if cur_row == row:
            cur_col = min(cur_col, len(new_lines[row]))
        st["auto_jobname"] = name
        pos = sum(len(l) + 1 for l in new_lines[:cur_row]) + cur_col
        st["applying"] = True
        try:
            editor.buffer.set_document(Document("\n".join(new_lines), pos), bypass_readonly=True)
        finally:
            st["applying"] = False
        analyse(editor.text)

    def rebuild_template():
        st["rebuild_pending"] = False
        tpl = st.get("tpl")
        if not tpl:
            return
        doc = editor.document
        col = doc.cursor_position_col
        lines, tpl, row = ui_tpl_reconcile(doc.lines, tpl, doc.cursor_position_row)
        if lines == doc.lines:
            st["tpl"] = tpl
            return
        remember_state()
        st["tpl"] = tpl
        apply_lines(lines, row, col)

    def apply_lines(lines, row, col=0):
        """Replace the editor text by a template operation, as one undo step that also restores the
        template state."""
        buf = editor.buffer
        snaps = st.setdefault("snap", {})
        if len(snaps) > 400:
            snaps.clear()
        text = "\n".join(lines)
        row = max(0, min(row, len(lines) - 1))
        pos = sum(len(l) + 1 for l in lines[:row]) + min(col, len(lines[row]))
        st["applying"] = True
        try:
            buf.cancel_completion()
            buf.save_to_undo_stack()
            buf.set_document(Document(text, pos), bypass_readonly=True)
        finally:
            st["applying"] = False
        snaps[text] = snapshot()
        analyse(text)
        get_app().invalidate()

    def remember_state():
        st.setdefault("snap", {})[editor.text] = snapshot()
    editor.buffer.on_text_changed += on_change

    cmdline = TextArea(height=1, multiline=False, prompt=":", style="class:cmdline")

    class UIPathCompleter(Completer):
        """File and folder names: folders end in '/', so completing one goes straight into it; files are
        limited to `exts`; hidden entries only when the name typed starts with '.'."""
        def __init__(self, exts=None):
            self.exts = tuple(exts) if exts else None

        def get_completions(self, document, complete_event):
            text = os.path.expanduser(document.text_before_cursor)
            folder, prefix = os.path.split(text)
            try:
                names = sorted(os.listdir(folder or "."))
            except OSError:
                return
            hits = [n for n in names if n.startswith(prefix)] or \
                [n for n in names if n.lower().startswith(prefix.lower())]
            for n in hits:
                if n.startswith(".") and not prefix.startswith("."):
                    continue
                isdir = os.path.isdir(os.path.join(folder or ".", n))
                if not isdir and self.exts and not n.lower().endswith(self.exts):
                    continue
                full = n + ("/" if isdir else "")
                yield Completion(full, start_position=-len(prefix), display=full)

    def field_suggestion(ta):
        """Rest of the first completion (file or folder name, element) of a one-line field, or None."""
        comp = ta.completer
        doc = ta.document
        if comp is None or not doc.text.strip() or not doc.is_cursor_at_the_end:
            return None
        from prompt_toolkit.completion import CompleteEvent
        try:
            first = next(iter(comp.get_completions(doc, CompleteEvent())), None)
        except Exception:
            return None
        if first is None:
            return None
        replaced = doc.text_before_cursor[len(doc.text_before_cursor) + first.start_position:]
        if not first.text.lower().startswith(replaced.lower()) or len(first.text) <= len(replaced):
            return None
        return first.text[len(replaced):]

    class FieldGhost(Processor):
        """Gray rest of the first completion after the cursor of a one-line field (Tab / Right takes it)."""
        def __init__(self, get_field):
            self.get_field = get_field

        def apply_transformation(self, ti):
            f = self.get_field()
            text = field_suggestion(f) if f is not None and get_app().layout.has_focus(f) else None
            return Transformation(ti.fragments + ([("class:auto-suggestion", text)] if text else []))

    def field_keys(get_field):
        fkb = KeyBindings()
        has = Condition(lambda: bool(field_suggestion(get_field())))

        @fkb.add("tab", filter=has)
        @fkb.add("right", filter=has)
        def _(e):
            f = get_field()
            text = field_suggestion(f)
            f.buffer.cancel_completion()
            f.buffer.insert_text(text)
        return fkb

    # ---------- dialogs ----------
    root = FloatContainer(content=Window(), floats=[])

    async def show(dialog_body, title, buttons, width=None, box=None):
        """Modal dialog; buttons = [(label, value)]; Esc returns None. box["set"](v) closes it from a field."""
        fut = asyncio.get_running_loop().create_future()
        if box is not None:
            box["set"] = lambda v: fut.done() or fut.set_result(v)

        def make(v):
            return lambda: fut.done() or fut.set_result(v)
        btns = [Button(text=lab, handler=make(val), width=max(len(lab) + 4, 8)) for lab, val in buttons]
        kb = KeyBindings()
        rows = [dialog_body]
        if btns:
            rows += [Window(height=1), VSplit(btns, padding=1, align=HorizontalAlign.CENTER)]
        # the frame itself is modal and carries the keys: bindings of containers around a modal one are ignored
        dlg = Frame(Box(HSplit(rows), padding_left=1, padding_right=1, padding_top=0, padding_bottom=0),
                    title=title, width=width, modal=True, key_bindings=kb, style="class:dialog.body")
        st["dialog"] = title

        @kb.add("escape", eager=True)
        def _(e):
            fut.done() or fut.set_result(None)

        not_typing = Condition(lambda: not isinstance(get_app().layout.current_control, BufferControl))
        kb.add("down")(focus_next)                 # arrows move between fields and buttons; Enter presses
        kb.add("up")(focus_previous)
        kb.add("tab")(focus_next)
        kb.add("s-tab")(focus_previous)
        kb.add("right", filter=not_typing)(focus_next)
        kb.add("left", filter=not_typing)(focus_previous)
        fl = Float(content=dlg)
        app = get_app()
        prev = app.layout.current_window
        prev_mode = app.vi_state.input_mode
        app.vi_state.input_mode = InputMode.INSERT        # text fields in dialogs take typing in vim mode
        root.floats.insert(max(0, len(root.floats) - 1), fl)      # below the completion menu
        app.layout.focus(dlg)
        app.invalidate()
        try:
            return await fut
        finally:
            st["dialog"] = None
            app.vi_state.input_mode = prev_mode
            root.floats.remove(fl)
            try:
                app.layout.focus(prev)
            except ValueError:
                pass
            app.invalidate()

    async def message(title, text, buttons=(("OK", True),)):
        return await show(Label(text), title, list(buttons), width=D(preferred=70, max=100))

    async def pick(title, items, text=None, width=None):
        """Vertical list of choices: arrows move, Enter picks, Esc cancels. items = [(label, value)]."""
        box = {}
        btns = []
        for lab, val in items:
            btns.append(Button(text=lab, handler=(lambda v=val: box["set"](v)), width=width or 44,
                               left_symbol=" ", right_symbol=" "))
        body = HSplit(([Label(text), Window(height=1)] if text else []) + btns)
        return await show(body, title, [], width=D(preferred=(width or 44) + 6), box=box)

    async def ask_text(title, label, default="", completer=None):
        ref = {}
        field = TextArea(text=default, multiline=False, completer=completer, complete_while_typing=True,
                         input_processors=[FieldGhost(lambda: ref.get("f"))])
        ref["f"] = field
        field.buffer.cursor_position = len(default)
        box = {}

        def accept(_b):
            box["set"](True)
            return True
        field.accept_handler = accept
        v = await show(HSplit([Label(label), HSplit([field], key_bindings=field_keys(lambda: field))]), title,
                       [("OK", True), ("Cancel", False)], width=D(preferred=70, max=100), box=box)
        return field.text.strip() if v else None

    def run_bg(coro):
        get_app().create_background_task(coro)

    # ---------- file operations ----------
    def load(p):
        text = open(p, encoding="utf-8", errors="replace").read()
        if not text.endswith("\n"):
            text += "\n"                     # an empty last line, where '%next' is offered
        st["tpl"], st["syntax_rows"], st["snap"] = None, set(), {}
        editor.buffer.set_document(Document(text, len(text)), bypass_readonly=True)
        st["path"], st["saved_text"] = os.path.abspath(p), text
        st["from_disk"], st["saved_once"] = True, False
        ui_add_recent(cfg, p)
        ui_save_settings(cfg, cfg_path)
        open_editor()
        remember_state()
        st["message"] = "Opened. At the end, Tab takes '%next' and shows the template of a new stage."

    def keep_essentials(lines, tpl):
        """%basis, %memory, %nprocs, %verbose pre-kept as real lines (their template default), not gray
        suggestions: every run needs them. Returns (lines, tpl)."""
        for iid in ("basis", "memory", "nprocs", "verbose"):
            gr = ui_tpl_ghost_rows(lines, tpl)
            row = next((r for r, i in gr.items() if i == iid), None)
            if row is not None:
                lines, tpl, _ = ui_tpl_accept(lines, row, 0, tpl)
        return lines, tpl

    def seed_jobname(lines, tpl):
        """%jobname pre-kept real too, but as 'Stage_1' (the wrapper's actual default for an omitted
        %jobname), not the gray row's syntax example ('job'); maybe_auto_jobname then keeps it in sync
        with the geometry/method/charge/mult/basis once those are known. Returns (lines, tpl)."""
        gr = ui_tpl_ghost_rows(lines, tpl)
        row = next((r for r, i in gr.items() if i == "jobname"), None)
        if row is None:
            return lines, tpl
        lines = list(lines)
        lines[row] = ui_option_line("jobname", "Stage_1", UI_KEYS["jobname"].doc)
        tpl = dict(tpl, items={k: v for k, v in tpl["items"].items() if k != "jobname"})
        lines, tpl, _ = ui_tpl_reconcile(lines, tpl, row)
        return lines, tpl

    def new_input():
        text, tpl = ui_full_template(geom_first=cfg.get("geometry", "top") == "top")
        lines, tpl = keep_essentials(text.split("\n"), tpl)
        lines, tpl = seed_jobname(lines, tpl)
        text = "\n".join(lines)
        st["tpl"], st["syntax_rows"], st["snap"] = tpl, set(), {}
        row = next(i for i, l in enumerate(lines) if l.startswith("# Description:"))
        editor.buffer.set_document(Document(text, sum(len(l) + 1 for l in lines[:row]) + len(lines[row])),
                                   bypass_readonly=True)
        st["path"], st["paste_row"], st["from_disk"], st["saved_once"] = None, None, False, False
        st["auto_jobname"] = "Stage_1"
        open_editor()
        st["saved_text"] = effective()
        remember_state()
        st["message"] = "Gray = template. Tab on a gray '!' word or line keeps it; the rest is dropped on save."

    def new_from_geometry(p, text=None):
        """New input whose geometry comes from an .xyz / .zmat file (or text given with --ui): the template as
        usual, with this geometry (real) where the gray example would be. Charge and multiplicity are asked
        when not given."""
        kind, rows, q, m = ui_geometry_from_text(text) if text is not None else ui_geometry_from_file(p)
        new_input()
        tpl = st["tpl"]
        lines = list(editor.document.lines)
        grows = sorted(r for r, iid in ui_tpl_ghost_rows(lines, tpl).items() if iid == "geometry")
        at = grows[0] if grows else len(lines)
        lines = [l for i, l in enumerate(lines) if i not in set(grows)]
        block = [f"* {kind} {0 if q is None else q} {1 if m is None else m}"] + rows + ["*"]
        lines[at:at] = block
        st["tpl"] = dict(tpl, items={k: v for k, v in tpl["items"].items() if k != "geometry"})
        lines, st["tpl"], _ = ui_tpl_reconcile(lines, st["tpl"], 0)
        header = next(i for i, l in enumerate(lines) if l == block[0])
        desc = next(i for i, l in enumerate(lines) if l.startswith("# Description:"))
        text = "\n".join(lines)
        st["applying"] = True
        try:
            editor.buffer.set_document(Document(text, sum(len(l) + 1 for l in lines[:desc]) + len(lines[desc])),
                                       bypass_readonly=True)
        finally:
            st["applying"] = False
        editor.buffer.reset(document=editor.buffer.document)          # no undo back to the empty template
        st["snap"] = {}
        analyse(editor.text)
        st["saved_text"] = effective()
        remember_state()
        syms = [r.split()[0] for r in rows] if kind == "xyz" else ui_zmat_symbols(rows)
        src = "the command line" if text is not None else os.path.basename(p)
        st["message"] = f"Geometry from {src}: {len(syms)} atoms ({kind}). Choose the options above it."
        if q is None or m is None:
            run_bg(finish_header(header, kind, syms))

    def tpl_tab():
        """Tab on the template: keep the gray '!' word or item under the cursor. True if done."""
        doc = editor.document
        row = doc.cursor_position_row
        line = doc.current_line
        syn = st.get("syntax_rows") or set()
        if line in syn:
            remember_state()
            st["syntax_rows"] = set(syn) - {line}
            apply_lines(doc.lines, min(row + 1, len(doc.lines) - 1))
            return True
        tpl = st.get("tpl")
        if not tpl:
            return False
        remember_state()
        r = ui_tpl_accept(doc.lines, row, doc.cursor_position_col, tpl)
        if r is None:
            return False
        lines, st["tpl"], (nr, nc) = r
        apply_lines(lines, nr, nc)
        return True

    def finalize_rows(lines, row):
        """The lines without the gray template parts, and where row ends up."""
        tpl = st.get("tpl")
        syn = st.get("syntax_rows") or set()
        gr = ui_tpl_ghost_rows(lines, tpl) if tpl else {}
        brow = ui_tpl_bang_row(lines, ui_tpl_stage_range(lines, tpl["stage_ord"])) if tpl and tpl["bang"] else None
        out, new_row = [], 0
        for i, l in enumerate(lines):
            if i == row:
                new_row = len(out)
            if i in gr or l in syn:
                continue
            if i == brow:
                words = [w for w in l.lstrip()[1:].split() if w.upper() not in tpl["bang"]]
                if not words:
                    continue
                l = "! " + " ".join(words)
            out.append(l)
        return out, min(new_row, max(0, len(out) - 1))

    def tpl_finish():
        """Drop every gray part from the editor (done on save)."""
        if st.get("tpl") or st.get("syntax_rows"):
            lines, row = finalize_rows(editor.document.lines, editor.document.cursor_position_row)
            st["tpl"], st["syntax_rows"] = None, set()
            apply_lines(lines, row)

    def tpl_stop():
        """F3, or vim ':done': stop offering this stage's template right now, same as saving would, without
        saving. For when the user is done choosing and does not want to keep seeing suggestions while editing
        something else (a later stage, going back to fix a value, ...)."""
        if st.get("tpl") or st.get("syntax_rows"):
            tpl_finish()
            st["message"] = "Template stopped: nothing more offered for this stage."
        else:
            st["message"] = "No template is active for this stage."

    def start_next_stage():
        """'%next' was just written on the cursor line: the stage before keeps only what was chosen, and the
        template of the new stage appears below (its '!' words and options of the stage before highlighted:
        they are inherited anyway)."""
        doc = editor.document
        remember_state()
        lines, row = finalize_rows(doc.lines, doc.cursor_position_row)
        st["tpl"], st["syntax_rows"] = None, set()
        stage_no = 1 + sum(1 for l in lines[:row + 1] if re.match(r"^\s*%(next|newjob)\b", l, re.I))
        prev = ui_prev_stage_info("\n".join(lines[:row]))
        lines, tpl, brow = ui_tpl_start(lines, row, stage_no, prev, geom_first=cfg.get("geometry", "top") == "top")
        st["tpl"] = tpl
        st["suppress_complete"] = True        # don't pop the '!' completions menu right after %next
        apply_lines(lines, brow, 2)
        editor.buffer.cancel_completion()
        st["message"] = (f"Stage {stage_no}: cyan = what stage {stage_no - 1} used (inherited if left as is). "
                         f"Shift-Tab (or Tab Tab) keeps them all; Tab keeps one; choose another word to change it.")

    def open_editor():
        st["screen"] = "editor"
        analyse(editor.text)
        apply_keymap()
        get_app().layout.focus(editor)
        if cfg["keymap"] == "vim":
            get_app().vi_state.input_mode = InputMode.NAVIGATION     # vim starts in normal mode

    def default_name():
        m = re.search(r"^\s*%jobname\s*=?\s*(\S+)", editor.text, re.M | re.I)
        return (m.group(1) if m else "job") + ".inp"

    async def save_flow(name=None, then_quit=False):
        if name is None and st["path"] and st.get("from_disk") and not st.get("saved_once"):
            base = os.path.basename(st["path"])
            how = await pick("Save", [(f"Replace {base}", "replace"),
                                      ("Save as a new file in the same folder", "new"), ("Cancel", None)],
                             width=max(44, len(base) + 12))
            if how is None:
                return False
            if how == "new":
                stem, ext = os.path.splitext(base)
                new = await ask_text("Save as", f"New name in {os.path.dirname(st['path'])}:",
                                     f"{stem}_2{ext or '.inp'}", UIPathCompleter((".inp", ".txt")))
                if not new:
                    return False
                name = os.path.join(os.path.dirname(st["path"]), os.path.basename(new))
        target = name or st["path"]
        if not target:
            target = await ask_text("Save input", "File name (saved in the current folder):",
                                    default_name(), UIPathCompleter((".inp", ".txt")))
            if not target:
                return False
        if not target.endswith((".inp", ".txt")):
            target += ".inp"
        if not ui_name_ok(target):
            await message("Save", "Use letters, digits, '.', '_' and '-' in the file name (no leading '-').")
            return False
        st["message"] = "Checking the input with the wrapper..."
        get_app().invalidate()
        text = effective()
        diags = await asyncio.get_running_loop().run_in_executor(None, ui_deep_check, text, ui_engine_path(cfg))
        errs = [d for d in diags if d["severity"] == "error"]
        st["message"] = ""
        if diags:
            listing = "\n".join(f"{'ERROR' if d['severity'] == 'error' else 'warn '} "
                                f"{'line ' + str(d['line']) if d['line'] else 'stage ' + str(d['stage'])}: "
                                f"{d['msg']}" for d in diags[:15])
            if errs:
                ok = await show(Label(listing), "The wrapper reports errors",
                                [("Save anyway", True), ("Back to editor", False)], width=D(preferred=90))
                if not ok:
                    return False
            else:
                await message("Warnings", listing)
        target = os.path.abspath(target)
        if target != st["path"] and os.path.exists(target):
            if not await show(Label(f"{target} exists. Overwrite it?"), "Save",
                              [("Overwrite", True), ("Cancel", False)]):
                return False
        with open(target, "w", encoding="utf-8") as f:
            f.write(text)
        tpl_finish()                            # the gray template lines are gone from the editor too
        st["path"], st["saved_text"], st["saved_once"] = target, text, True
        ui_add_recent(cfg, target)
        ui_save_settings(cfg, cfg_path)
        st["message"] = f"Saved {target}"
        if then_quit:
            get_app().exit()
            return True
        await run_prompt()
        return True

    async def run_prompt():
        choice = await show(Label(f"Saved {os.path.basename(st['path'])}. Run it?"), "Run",
                            [("Dry-run", "dry"), ("Run in background", "run"), ("Not now", None)])
        if choice == "dry":
            st["message"] = "Dry-run..."
            get_app().invalidate()
            rc, out = await asyncio.get_running_loop().run_in_executor(None, ui_dry_run, effective(),
                                                                       ui_engine_path(cfg))
            st["message"] = ""
            show_viewer(f"Dry-run (exit code {rc})", text=out)
        elif choice == "run":
            await launch_flow()

    async def launch_flow():
        p = st["path"]
        reasons = ui_folder_clash(p)
        if reasons:
            jobdir = os.path.join(os.path.dirname(p), os.path.splitext(os.path.basename(p))[0])
            ok = await show(Label("This folder already holds another job:\n  " + "\n  ".join(reasons) +
                                  f"\n\nThe wrapper writes a fixed run.log, so each job needs its own folder.\n"
                                  f"Move the input to {jobdir}/ and run it there?"), "One job per folder",
                            [("Move and run", True), ("Cancel", False)], width=D(preferred=90))
            if not ok:
                return
            try:
                p = ui_move_to_jobdir(p)
            except OSError as err:
                await message("Cannot move", str(err))
                return
            cfg["recent"] = [r for r in cfg["recent"] if r != st["path"]]
            st["path"] = p
            ui_add_recent(cfg, p)
            ui_save_settings(cfg, cfg_path)
        try:
            pid = ui_launch(p, ui_engine_path(cfg))
        except (OSError, ValueError, RuntimeError) as err:
            await message("Not started", str(err))
            return
        out = os.path.splitext(p)[0] + ".out"
        nxt = await show(Label(f"Started PID {pid} in {os.path.dirname(p)}\nLog: {out}\n"
                               f"Follow it with: tail -f {out}"), "Running",
                         [("View .out", "view"), ("Keep editing", "edit"), ("New input", "new"),
                          ("Start screen", "start")], width=D(preferred=90))
        if nxt == "view":
            show_viewer(os.path.basename(out), follow=out)
        elif nxt == "new":
            new_input()
        elif nxt == "start":
            go_start()

    # ---------- viewer (dry-run output, live .out) ----------
    viewer = TextArea(read_only=True, scrollbar=True, wrap_lines=False, focus_on_click=True)
    viewer_title = {"text": ""}

    def show_viewer(title, text=None, follow=None):
        viewer_title["text"] = title
        st["viewer"] = follow
        viewer.buffer.set_document(Document(text or "", 0), bypass_readonly=True)
        st["return_to"] = st["screen"]
        st["screen"] = "viewer"
        if follow:
            refresh_viewer(force_end=True)
        get_app().layout.focus(viewer)
        get_app().invalidate()

    def refresh_viewer(force_end=False):
        p = st["viewer"]
        if not p or st["screen"] != "viewer":
            return
        try:
            text = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            text = f"(waiting for {p})"
        if text != viewer.text:
            at_end = force_end or viewer.buffer.cursor_position >= len(viewer.text) - 1
            viewer.buffer.set_document(Document(text, len(text) if at_end else
                                                min(viewer.buffer.cursor_position, len(text))),
                                       bypass_readonly=True)

    async def follow_loop():
        while True:
            await asyncio.sleep(1.0)
            if st["screen"] == "viewer" and st["viewer"]:
                refresh_viewer()
                get_app().invalidate()

    # ---------- geometry: builder in the bottom line, inside the editor ----------
    def bar_suggestion():
        return field_suggestion(bar)

    class BarGhost(Processor):
        def apply_transformation(self, ti):
            text = bar_suggestion() if get_app().layout.has_focus(bar) else None
            return Transformation(ti.fragments + ([("class:auto-suggestion", text)] if text else []))

    bar = TextArea(height=1, multiline=False, style="class:cmdline", complete_while_typing=True,
                   prompt=lambda: [("class:bar.prompt", st.get("bar_prompt", ""))], input_processors=[BarGhost()])
    bar_state = {"fut": None}
    choice_state = {"items": [], "i": 0, "fut": None, "prompt": ""}

    def choice_text():
        out = [("class:bar.prompt", choice_state["prompt"] + " ")]
        for k, (lab, _) in enumerate(choice_state["items"]):
            out.append(("class:bar.current" if k == choice_state["i"] else "", f" {lab} "))
            out.append(("", " "))
        out.append(("class:hint", "  (arrows, Enter; Esc cancels)"))
        return out
    choice_kb = KeyBindings()

    @choice_kb.add("left")
    @choice_kb.add("up")
    def _(e):
        choice_state["i"] = (choice_state["i"] - 1) % len(choice_state["items"])

    @choice_kb.add("right")
    @choice_kb.add("down")
    @choice_kb.add("tab")
    def _(e):
        choice_state["i"] = (choice_state["i"] + 1) % len(choice_state["items"])

    @choice_kb.add("enter")
    def _(e):
        f = choice_state["fut"]
        f.done() or f.set_result(choice_state["items"][choice_state["i"]][1])

    @choice_kb.add("escape", eager=True)
    def _(e):
        f = choice_state["fut"]
        f.done() or f.set_result(None)
    choice_win = Window(FormattedTextControl(choice_text, focusable=True, key_bindings=choice_kb), height=1)

    bar_kb = KeyBindings()

    @bar_kb.add("tab", filter=Condition(lambda: bool(bar_suggestion())))
    @bar_kb.add("right", filter=Condition(lambda: bool(bar_suggestion())))
    def _(e):
        text = bar_suggestion()
        bar.buffer.cancel_completion()
        bar.buffer.insert_text(text)

    @bar_kb.add("escape", eager=True)
    def _(e):
        f = bar_state["fut"]
        f is not None and not f.done() and f.set_result(None)

    def bar_accept(_b):
        f = bar_state["fut"]
        f is not None and not f.done() and f.set_result(bar.text)
        return False
    bar.accept_handler = bar_accept

    async def _bar_session(focus_target, setup):
        app = get_app()
        prev_mode = app.vi_state.input_mode
        fut = asyncio.get_running_loop().create_future()
        setup(fut)
        app.vi_state.input_mode = InputMode.INSERT
        app.layout.focus(focus_target)
        app.invalidate()
        try:
            return await fut
        finally:
            st["bar"] = None
            app.vi_state.input_mode = prev_mode
            app.layout.focus(editor)
            app.invalidate()

    async def choose(prompt, items):
        def setup(fut):
            choice_state.update(items=items, i=0, fut=fut, prompt=prompt)
            st["bar"] = "choice"
        return await _bar_session(choice_win, setup)

    async def ask(prompt, default="", completer=None):
        def setup(fut):
            st["bar_prompt"], st["bar"] = prompt, "text"
            bar.completer = completer
            bar.buffer.set_document(Document(default, len(default)))
            bar_state["fut"] = fut
        r = await _bar_session(bar, setup)
        return None if r is None else r.strip()

    class ElementCompleter(Completer):
        def get_completions(self, document, complete_event):
            w = document.text_before_cursor.strip()
            for e in _UI_ELEMENTS:
                if w and e.lower().startswith(w.lower()):
                    yield Completion(e, start_position=-len(w))

    def lines_now():
        return editor.text.split("\n")

    def set_lines(lines, cursor_row):
        text = "\n".join(lines)
        pos = len("\n".join(lines[:cursor_row + 1]))
        editor.buffer.set_document(Document(text, min(pos, len(text))))

    def geometry_row():
        """Where a built geometry goes, wherever F2 was pressed: the stage's geometry block after its options.
        The gray example (or a geometry already there) is taken out first; one undo step brings it back.
        Without one, after the stage's last line."""
        doc = editor.document
        lines, row = list(doc.lines), doc.cursor_position_row
        nexts = [i for i, l in enumerate(lines) if re.match(r"^\s*%(next|newjob)\b", l, re.I)]
        start = max([i + 1 for i in nexts if i < row] or [0])
        end = min([i for i in nexts if i >= row] or [len(lines)])
        tpl = st.get("tpl")
        grows = ui_tpl_ghost_rows(lines, tpl) if tpl else {}
        drop = {r for r, iid in grows.items() if iid == "geometry" and start <= r < end}
        if not drop:
            head = next((i for i in range(start, end) if i not in grows
                         and re.match(r"^\*\s*(xyz|zmat|int|internal)\b", _ui_strip_comment(lines[i]).strip(), re.I)),
                        None)
            if head is not None:
                stop = head
                for j in range(head + 1, end):
                    code = _ui_strip_comment(lines[j]).strip()
                    if code in ("*", "end"):
                        stop = j
                        break
                    if code[:1] in ("%", "!"):
                        break
                    stop = j
                drop = set(range(head, stop + 1))
        if drop:
            remember_state()
            if tpl and "geometry" in tpl["items"]:
                st["tpl"] = dict(tpl, items={k: v for k, v in tpl["items"].items() if k != "geometry"})
            apply_lines([l for i, l in enumerate(lines) if i not in drop], min(drop))
            return min(drop)
        last = max([i for i in range(start, end) if lines[i].strip()] or [start - 1])
        return last + 1

    def insertion_row():
        doc = editor.document
        return doc.cursor_position_row if not doc.current_line.strip() else doc.cursor_position_row + 1

    def stage_charge():
        stage = stage_at(editor.document.cursor_position_row)
        return stage.charge if stage is not None and stage.charge is not None else 0

    async def finish_header(header_row, kind, symbols):
        """Ask charge and multiplicity (suggested from the electron count) and write them in the header."""
        q = await ask("Charge: ", str(stage_charge()))
        q = _ui_int(q, 0) if q is not None else 0
        ne = ui_count_electrons(symbols, q)
        sm = ui_suggest_mult(ne) or 1
        m = await ask(f"Multiplicity ({ne} electrons; suggested {sm}): " if ne is not None
                      else "Multiplicity: ", str(sm))
        m = _ui_int(m, sm) if m is not None else sm
        lines = lines_now()
        lines[header_row] = f"* {kind} {q} {m}"
        end = next((r for r in range(header_row + 1, len(lines)) if lines[r].strip() in ("*", "end")), len(lines) - 1)
        if end + 1 >= len(lines) or lines[end + 1].strip():
            lines.insert(end + 1, "")                       # an empty line after the block, for '%next'
        editor.buffer.set_document(Document("\n".join(lines), sum(len(l) + 1 for l in lines[:end + 1])))
        st["message"] = f"Geometry: {len(symbols)} atoms, charge {q}, multiplicity {m}. Tab/Right takes '%next'."
        leave_in_normal()

    def leave_in_normal():
        if cfg["keymap"] == "vim":
            get_app().vi_state.input_mode = InputMode.NAVIGATION

    async def geometry_flow():
        if st.get("paste_row") is not None:
            await finish_paste()
            return
        stage = stage_at(editor.document.cursor_position_row)
        items = [("Type atoms", "type"), ("Paste xyz rows or a Z-matrix", "paste"),
                 ("Load .xyz / .zmat file", "file")]
        if stage is not None and stage.number > 1:
            items.append((f"Continue stage {stage.number - 1}", "continue"))
        kind = await choose("Geometry:", items)
        if kind is None:
            return
        if kind == "continue":
            st["message"] = "This stage continues the previous one: no geometry block needed."
            return
        row = geometry_row()
        if kind == "type":
            await type_atoms(row)
        elif kind == "paste":
            lines = lines_now()
            lines[row:row] = ["* xyz 0 1", "", "*"]
            set_lines(lines, row + 1)
            editor.buffer.cursor_position = len("\n".join(lines[:row + 1])) + 1
            st["paste_row"] = row
            if cfg["keymap"] == "vim":
                get_app().vi_state.input_mode = InputMode.INSERT        # ready to paste
            st["message"] = ("Paste xyz rows ('Element x y z') or a Z-matrix on the empty line (count and "
                              "comment lines are dropped), then F2.")
        else:
            await load_geometry_file(row)

    async def type_atoms(row):
        n = await ask("How many atoms? ", "")
        if n is None:
            return
        n = _ui_int(n, 0)
        if n < 1:
            st["message"] = "Geometry: enter a number of atoms (1 or more)."
            return
        lines = lines_now()
        lines[row:row] = ["* xyz 0 1", "*"]
        set_lines(lines, row)
        symbols = []
        for i in range(1, n + 1):
            while True:
                lab = await ask(f"Atom {i}/{n} label (e.g. Cl, C1, ghost-H): ", "", ElementCompleter())
                if lab is None:
                    break
                if ui_element(lab)[0] is not None:
                    break
                st["message"] = f"'{lab}' is not an element label."
            if lab is None:
                break
            while True:
                xyz = await ask(f"Atom {i}/{n} {lab}  x y z (Angstrom): ", "0.0 0.0 0.0" if i == 1 else "")
                if xyz is None:
                    break
                toks = xyz.replace(",", " ").split() if xyz.count(",") >= 2 and "." in xyz else xyz.split()
                if len(toks) == 3 and all(_ui_float(t) for t in toks):
                    break
                st["message"] = "Write three numbers with a decimal point, e.g. 0.0 0.0 1.42"
            if xyz is None:
                break
            x, y, z = (float(t) for t in toks)
            lines = lines_now()
            lines.insert(row + 1 + len(symbols), f"{lab:<4s} {x:>14.8f} {y:>14.8f} {z:>14.8f}")
            symbols.append(lab)
            set_lines(lines, row + len(symbols))
            st["message"] = ""
        if not symbols:
            lines = lines_now()
            del lines[row:row + 2]
            set_lines(lines, max(0, row - 1))
            st["message"] = "Geometry cancelled."
            return
        await finish_header(row, "xyz", symbols)

    async def finish_paste():
        lines = lines_now()
        row = st.pop("paste_row")
        st["paste_row"] = None
        if row >= len(lines) or not lines[row].strip().lower().startswith("* xyz"):
            row = next((r for r in range(editor.document.cursor_position_row, -1, -1)
                        if lines[r].strip().lower().startswith("* xyz")), None)
            if row is None:
                st["message"] = "No pasted-geometry placeholder block above the cursor."
                return
        end = next((r for r in range(row + 1, len(lines)) if lines[r].strip() in ("*", "end")), len(lines))
        pasted = "\n".join(lines[row + 1:end])
        try:
            kind, rows, q, m = ui_geometry_from_text(pasted) if pasted.strip() else (None, [], None, None)
        except ValueError:
            kind, rows, q, m = None, [], None, None
        if end == len(lines):
            lines.append("*")
        if not rows:
            lines[row + 1:end] = []
            set_lines(lines, row)
            st["message"] = "No geometry rows (xyz 'Element x y z', or a Z-matrix) found in the pasted text."
            return
        lines[row:end] = [f"* {kind} {0 if q is None else q} {1 if m is None else m}"] + rows
        set_lines(lines, row)
        syms = [r.split()[0] for r in rows] if kind == "xyz" else ui_zmat_symbols(rows)
        if q is None or m is None:
            await finish_header(row, kind, syms)
        else:
            st["message"] = f"Geometry: {len(syms)} atoms ({kind}), charge {q}, multiplicity {m}."

    async def load_geometry_file(row):
        p = await ask("File (.xyz or .zmat): ", "", UIPathCompleter((".xyz", ".zmat")))
        if not p:
            return
        p = os.path.expanduser(p)
        try:
            if p.lower().endswith(".xyz"):
                rows, header, kind = ui_read_xyz(p), None, "xyz"
                syms = [r.split()[0] for r in rows]
            else:
                rows, header = ui_read_zmat(p)
                kind, syms = "zmat", ui_zmat_symbols(rows)
        except (OSError, ValueError) as err:
            st["message"] = f"Cannot read {p}: {err}"
            return
        lines = lines_now()
        lines[row:row] = [header or f"* {kind} 0 1"] + rows + ["*"]
        set_lines(lines, row)
        if header is None:
            await finish_header(row, kind, syms)
        else:
            st["message"] = f"Loaded {len(rows)} rows from {os.path.basename(p)} (its own header kept)."

    # ---------- help pane ----------
    help_area = TextArea(read_only=True, scrollbar=True, wrap_lines=True, focus_on_click=True)

    def token_under_cursor():
        doc = editor.document
        line = doc.current_line
        code = _ui_strip_comment(line).strip()
        if code.startswith("%"):
            m = re.match(r"%\s*(\w+)", code)
            return "%" + m.group(1) if m else "%"
        if code.startswith("!"):
            col = doc.cursor_position_col
            for m in re.finditer(r"[^\s!]+", line):
                if m.start() <= col <= m.end():
                    return "!" + m.group()
            return "!"
        return "geometry"

    def toggle_help():
        if st["help"]:
            st["help"] = None
            get_app().layout.focus(editor)
        else:
            st["help"] = token_under_cursor()
            help_area.buffer.set_document(Document(ui_help_text(st["help"]), 0), bypass_readonly=True)
            get_app().layout.focus(help_area)

    # ---------- keymaps ----------
    def apply_keymap():
        app = get_app()
        app.editing_mode = EditingMode.VI if cfg["keymap"] == "vim" else EditingMode.EMACS

    def dirty():
        return st["screen"] == "editor" and effective() != st["saved_text"]

    async def quit_flow(force=False):
        if dirty() and not force:
            v = await show(Label("The input has unsaved changes."), "Quit",
                           [("Save", "save"), ("Discard", "discard"), ("Cancel", None)])
            if v == "save":
                await save_flow(then_quit=True)
                return
            if v != "discard":
                return
        get_app().exit()

    def go_start():
        st["screen"] = "start"
        st["help"] = None
        get_app().layout.focus(start_buttons[0])

    async def leave_editor():
        if dirty():
            v = await show(Label("The input has unsaved changes."), "Back to the start screen",
                           [("Save", "save"), ("Discard", "discard"), ("Cancel", None)])
            if v == "save" and not await save_flow():
                return
            if v is None:
                return
        go_start()

    kb = KeyBindings()
    in_editor = Condition(lambda: st["screen"] == "editor") & has_focus(editor)
    vim_on = Condition(lambda: cfg["keymap"] == "vim")
    emacs_on = Condition(lambda: cfg["keymap"] == "emacs")
    plain_on = Condition(lambda: cfg["keymap"] == "plain")

    suggestion_here = Condition(lambda: bool(suggestion_for(editor.document)))

    def on_template():
        doc = editor.document
        lines, row = doc.lines, doc.cursor_position_row
        if doc.current_line in (st.get("syntax_rows") or set()):
            return True
        tpl = st.get("tpl")
        if not tpl:
            return False
        if row in ui_tpl_ghost_rows(lines, tpl):
            return True
        return bool(tpl["bang"]) and row == ui_tpl_bang_row(lines, ui_tpl_stage_range(lines, tpl["stage_ord"]))
    template_here = Condition(on_template)
    next_written = Condition(lambda: editor.document.current_line.strip().lower() == "%next")

    def accept_highlighted():
        tpl = st.get("tpl")
        if not tpl or not ((tpl["bang_hl"] & tpl["bang"]) or (tpl["hl"] & set(tpl["items"]))):
            st["message"] = "Nothing highlighted to keep here (highlights show the stage before, after %next)."
            return
        remember_state()
        lines, st["tpl"], (r, c) = ui_tpl_accept_highlighted(editor.document.lines, tpl)
        apply_lines(lines, r, c)
        st["message"] = "Kept everything the stage before used; change what differs."

    @kb.add("s-tab", filter=in_editor & Condition(lambda: bool(st.get("tpl"))))
    def _(e):
        accept_highlighted()

    @kb.add("tab", filter=in_editor & template_here)
    def _(e):
        editor.buffer.cancel_completion()
        now = time.time()
        double = now - st.get("last_tab", 0.0) < 0.4 and st.get("pre_tab") is not None
        st["last_tab"] = now
        if double and st.get("tpl") is not None:
            # Tab Tab: take back what the first Tab did, then keep all the highlighted words and lines
            text, tpl, syn, pos = st.pop("pre_tab")
            if tpl and (tpl["bang_hl"] or tpl["hl"]):
                st["applying"] = True
                try:
                    editor.buffer.set_document(Document(text, pos), bypass_readonly=True)
                finally:
                    st["applying"] = False
                st["tpl"], st["syntax_rows"] = tpl, set(syn)
                analyse(text)
                accept_highlighted()
                return
        tpl0 = st.get("tpl")
        doc0 = editor.document
        iid0 = ui_tpl_ghost_rows(doc0.lines, tpl0).get(doc0.cursor_position_row) if tpl0 else None
        name0 = iid0.partition("=")[0] if iid0 else None
        st["pre_tab"] = ((editor.text, copy.deepcopy(tpl0), frozenset(st.get("syntax_rows") or ()),
                          editor.buffer.cursor_position) if tpl0 and (tpl0["bang_hl"] or tpl0["hl"]) else None)
        if not tpl_tab():
            text = suggestion_for(editor.document)
            if text:
                editor.buffer.insert_text(text)
        elif name0 and ui_value_completable(name0):
            # the gray line just became real (its default value already filled in): show the real
            # completions menu for it right away instead of making the user retype the value from scratch
            row = next((i for i, l in enumerate(editor.document.lines)
                        if re.match(rf"^\s*%{re.escape(name0)}\b", l, re.I)), None)
            if row is not None:
                line = editor.document.lines[row]
                col = len(_ui_strip_comment(line).rstrip())
                editor.buffer.cursor_position = sum(len(l) + 1 for l in editor.document.lines[:row]) + col
                st["suppress_complete"] = False   # override any leftover suppression: this menu is wanted
                editor.buffer.start_completion()

    @kb.add("tab", filter=in_editor & next_written & ~suggestion_here & ~template_here)
    @kb.add("enter", filter=in_editor & next_written & ~suggestion_here & ~template_here)
    def _(e):
        start_next_stage()

    @kb.add("enter", filter=in_editor & has_completions)
    def _(e):
        """Enter while the real completions menu is open: pick the highlighted entry (or just close the
        menu if none is highlighted) instead of also inserting a newline; and don't let accepting it
        immediately reopen the same menu (same mechanism as the %next transition)."""
        buf = editor.buffer
        comp = buf.complete_state.current_completion
        if comp is not None:
            buf.apply_completion(comp)
        else:
            buf.cancel_completion()
        st["suppress_complete"] = True   # set after: on_change (fired by apply_completion) would clear it

    @kb.add("right", filter=in_editor & suggestion_here & ~template_here)  # accept the gray suggestion
    @kb.add("tab", filter=in_editor & suggestion_here & ~template_here)  # (before the completion menu's Tab)
    def _(e):
        text = suggestion_for(editor.document)
        if text:
            editor.buffer.cancel_completion()
            editor.buffer.insert_text(text)
            if editor.document.current_line.strip().lower() == "%next":
                start_next_stage()
            else:
                add_block_syntax()

    def add_block_syntax():
        """After '%constraints' (or another block option) is completed: its syntax rows, in gray, below."""
        doc = editor.document
        m = re.match(r"^\s*%(\w+)\s*$", doc.current_line)
        o = UI_KEYS.get(m.group(1).lower()) if m else None
        if o is None or o.name not in UI_BLOCK_SYNTAX or o.value not in ("block", "scan"):
            return
        remember_state()
        rows = ui_option_lines(o)[1:]
        lines = list(doc.lines)
        r = doc.cursor_position_row
        tpl = st.get("tpl")
        if tpl and o.name in tpl["items"]:
            # the option is now written for real: the template's gray copy goes (same text, so it would
            # otherwise make this line gray too)
            old = set(tpl["items"][o.name])
            keep = [i for i, l in enumerate(lines) if i == r or l not in old]
            r = keep.index(r)
            lines = [lines[i] for i in keep]
            tpl = dict(tpl, items={k: v for k, v in tpl["items"].items() if k != o.name})
            st["tpl"] = tpl
        st["syntax_rows"] = set(st.get("syntax_rows") or ()) | set(rows[:-1])   # the 'end' is real
        lines = lines[:r + 1] + rows + lines[r + 1:]
        if st.get("tpl"):
            lines, st["tpl"], r = ui_tpl_reconcile(lines, st["tpl"], r)
        apply_lines(lines, r + 1)
        st["message"] = f"%{o.name}: the gray rows show the syntax; Tab on them keeps them as a start."

    @kb.add("c-s", filter=in_editor & ~emacs_on)       # emacs: C-s is search (C-x C-s saves)
    def _(e):
        run_bg(save_flow())

    @kb.add("c-r", filter=in_editor & vim_on & vi_navigation_mode,      # vim: redo (not reverse search);
            save_before=lambda e: False)                                # saving first would clear the redo stack
    def _(e):
        editor.buffer.redo()

    @kb.add("c-f", filter=in_editor & plain_on)
    def _(e):
        start_search(editor.control)

    @kb.add("c-q", filter=in_editor & ~vim_on)            # back to the start screen (Quit there exits)
    def _(e):
        run_bg(leave_editor())

    @kb.add("c-x", "c-s", filter=in_editor & emacs_on)
    def _(e):
        run_bg(save_flow())

    @kb.add("c-x", "c-c", filter=in_editor & emacs_on)
    def _(e):
        run_bg(leave_editor())

    @kb.add(":", filter=in_editor & vim_on & vi_navigation_mode)
    def _(e):
        st["cmd"] = True
        cmdline.text = ""
        e.app.layout.focus(cmdline)
        e.app.vi_state.input_mode = InputMode.INSERT

    @kb.add("f1", filter=Condition(lambda: st["screen"] == "editor"))
    def _(e):
        toggle_help()

    help_kb = KeyBindings()

    @help_kb.add("escape", eager=True)
    @help_kb.add("f1")
    @help_kb.add("q", eager=True)
    def _(e):
        toggle_help()

    @kb.add("f2", filter=in_editor)
    @kb.add("c-g", filter=in_editor)
    def _(e):
        run_bg(geometry_flow())

    @kb.add("f3", filter=in_editor)
    def _(e):
        tpl_stop()

    @kb.add("f4", filter=in_editor)
    def _(e):
        """Toggle showing deprecated/hidden %keys (e.g. %pos_atom, %pos_basis_map) in the '%' completion
        menu - they keep validating either way (UI_DEPRECATED_KEYS), this only controls what's suggested."""
        st["show_hidden"] = not st["show_hidden"]
        st["message"] = "showing hidden options too (F4 to hide)" if st["show_hidden"] else "hidden options hidden again"
        if e.app.current_buffer is editor.buffer and re.match(r"^\s*%(\w*)$", editor.buffer.document.current_line_before_cursor):
            editor.buffer.start_completion()

    viewer_kb = KeyBindings()

    @viewer_kb.add("escape", eager=True)
    @viewer_kb.add("q", eager=True)
    def _(e):
        st["screen"], st["viewer"] = st.get("return_to") or "editor", None
        e.app.layout.focus(editor if st["screen"] == "editor" else start_buttons[0])

    cmd_kb = KeyBindings()

    @cmd_kb.add("escape", eager=True)
    def _(e):
        st["cmd"] = False
        e.app.layout.focus(editor)
        e.app.vi_state.input_mode = InputMode.NAVIGATION

    def run_command(_buf):
        c = cmdline.text.strip()
        st["cmd"] = False
        get_app().layout.focus(editor)
        get_app().vi_state.input_mode = InputMode.NAVIGATION
        parts = c.split(None, 1)
        if not parts:
            return False
        head, arg = parts[0], (parts[1] if len(parts) > 1 else None)
        if head in ("w", "write"):
            run_bg(save_flow(arg))
        elif head in ("wq", "x"):
            run_bg(save_flow(arg, then_quit=True))
        elif head in ("q", "quit"):                # back to the start screen
            run_bg(leave_editor())
        elif head == "q!":
            go_start()
        elif head in ("qa", "qall"):               # leave the input builder
            run_bg(quit_flow())
        elif head == "qa!":
            get_app().exit()
        elif head in ("e", "edit") and arg:
            load(os.path.expanduser(arg))
        elif head in ("help", "h"):
            toggle_help()
        elif head in ("geometry", "geom"):
            run_bg(geometry_flow())
        elif head == "start":
            run_bg(leave_editor())
        elif head == "done":
            tpl_stop()
        else:
            st["message"] = f"Unknown command ':{c}' (w, w name, wq, q, q!, geom, help, start, done)"
        return False
    cmdline.accept_handler = run_command

    # ---------- status and hint bars ----------
    def status():
        app = get_app()
        mode = ""
        if cfg["keymap"] == "vim":
            mode = {InputMode.INSERT: " INSERT ", InputMode.NAVIGATION: " NORMAL ",
                    InputMode.REPLACE: " REPLACE "}.get(app.vi_state.input_mode, " VIM ")
        else:
            mode = f" {cfg['keymap'].upper()} "
        row = editor.document.cursor_position_row
        stage = stage_at(row)
        name = os.path.basename(st["path"]) if st["path"] else "[new input]"
        parts = [("class:status.mode", mode), ("class:status", f" {name}{' *' if dirty() else ''} ")]
        if stage is not None:
            info = f"stage {stage.number}/{len(st['stages'])}"
            parts.append(("class:status", f"| {info} "))
            if stage.inherited_from:
                basis = stage.keys.get("basis", "")
                parts.append(("class:status.inherited",
                              f"| inherited from {stage.inherited_from}: {stage.run_type} {stage.method} "
                              f"{stage.sector}{' ' + basis if basis else ''} "))
        msg = st["message"]
        here = st["by_line"].get(row)
        if here:
            d = here[0]
            msg = ("ERROR: " if d.severity == "error" else "warning: ") + d.msg
            parts.append(("class:status.error" if d.severity == "error" else "class:status.warning", " " + msg))
        elif msg:
            parts.append(("class:status", " " + msg))
        return parts

    def hints():
        km = cfg["keymap"]
        save = {"vim": "Esc :w save  :q quit", "emacs": "C-x C-s save  C-x C-c quit",
                "plain": "Ctrl-S save  Ctrl-Q quit"}[km]
        if on_template():
            return [("class:hint", " Tab: keep the gray word/line under the cursor   gray parts are dropped on save   "
                                   "F2 geometry ")]
        if st.get("paste_row") is not None:
            return [("class:hint", " Paste xyz rows or a Z-matrix, then F2 to finish the geometry ")]
        return [("class:hint", f" ! keywords  % options  Right/Tab: accept gray text  F2 geometry  "
                                f"F4 show hidden options  F1 help  {save} ")]

    status_bar = Window(FormattedTextControl(status), height=1, style="class:status")
    hint_bar = Window(FormattedTextControl(hints), height=1, style="class:hint")
    cmd_bar = ConditionalContainer(HSplit([cmdline], key_bindings=cmd_kb), filter=Condition(lambda: st["cmd"]))
    help_float = Float(content=ConditionalContainer(
        HSplit([Frame(help_area, title=lambda: f"Help: {st['help']}  (Esc/F1/q close)")], key_bindings=help_kb),
        filter=Condition(lambda: bool(st["help"]))), right=1, top=1, bottom=3,
        width=lambda: min(70, max(40, get_app().output.get_size().columns // 2)))
    bar_box = ConditionalContainer(HSplit([bar], key_bindings=bar_kb), filter=Condition(lambda: st.get("bar") == "text"))
    choice_box = ConditionalContainer(choice_win, filter=Condition(lambda: st.get("bar") == "choice"))
    editor_screen = HSplit([editor, search_bar, status_bar, cmd_bar, bar_box, choice_box, hint_bar])

    viewer_screen = HSplit([
        Window(FormattedTextControl(lambda: [("class:status.mode", f" {viewer_title['text']} "),
                                             ("class:status", "  Esc/q back" +
                                              ("  (follows new output)" if st["viewer"] else ""))]),
               height=1, style="class:status"),
        viewer], key_bindings=viewer_kb)

    # ---------- start screen, recent files, settings ----------
    def b_new():
        """New input: geometry is asked for first (type, paste, or a .xyz/.zmat file), same menu as F2's
        in-editor builder; the template (with that geometry already in place) follows. Cancelling the
        geometry menu still leaves the blank template, gray example geometry included, editable with F2
        later - it is not reached when a geometry is already given another way ("--ui file.xyz", the
        command-line geometry argument, or Open)."""
        new_input()
        run_bg(geometry_flow())

    def b_open():
        async def go():
            p = await ask_text("Open", "Input file (Tab / Right completes names and folders):", "",
                               UIPathCompleter((".inp", ".txt")))
            if p:
                try:
                    load(os.path.expanduser(p))
                except OSError as err:
                    await message("Open", str(err))
        run_bg(go())

    def b_recent():
        async def go():
            if not cfg["recent"]:
                await message("Recent files", "No recent files yet.")
                return
            p = await pick("Recent files", [(r if len(r) <= 70 else "..." + r[-67:], r) for r in cfg["recent"]],
                           width=72)
            if p:
                try:
                    load(p)
                except OSError as err:
                    await message("Open", str(err))
        run_bg(go())

    # ---------- Molden/cube viewer: native screen, numpy/PySCF loaded lazily the moment it is opened ----------
    def view_heavy():
        engine = ui_engine_path(cfg)
        if not engine:
            raise RuntimeError("No engine script found. Set its path from the start screen's "
                                "Settings > Engine path.")
        return ui_load_heavy(engine)

    def view_state():
        if st["view"] is None:
            st["view"] = {"files": [], "sel": 0, "msg": "", "last_image": None, "_image_dirty": False,
                          "_image_payload": None, "_image_fragments": None,
                          "settings": {"what": "electron", "plane": "AUTO", "colors": "", "range_electron": "",
                                       "range_positron": "", "range_esp": "", "ystep": "", "prefix": "view"}}
        return st["view"]

    def view_rows():
        v = view_state()
        rows = [("file", i) for i in range(len(v["files"]))]
        rows.append(("add", None))
        rows += [("setting", k) for k in ("what", "plane", "range_electron", "range_positron", "range_esp",
                                          "colors", "ystep", "prefix")]
        rows += [("action", a) for a in ("draw", "compare", "chelpg", "info", "close")]
        return rows

    _VIEW_LABELS = {"what": "Map", "plane": "Plane", "range_electron": "Colour range, electron",
                    "range_positron": "Colour range, positron", "range_esp": "Colour range, ESP",
                    "colors": "Colour scheme", "ystep": "Y-axis labels every (bohr)", "prefix": "Output prefix"}
    _VIEW_DEFAULTS = {"range_electron": "auto (0 to 0.65 a.u.)", "range_positron": "auto (0 to the maximum)",
                      "range_esp": "auto (-0.1 to 0.1 a.u.)", "colors": "default", "ystep": "auto"}
    _VIEW_ACTIONS = {"draw": "Draw (electron / positron / both / esp)", "compare": "Compare included files",
                     "chelpg": "CHELPG charges", "info": "Files info", "close": "Close"}

    def view_text():
        v = view_state()
        rows = view_rows()
        v["sel"] = min(v["sel"], len(rows) - 1) if rows else 0
        s = v["settings"]
        out = [("class:title", "View molden/cube"), ("", "\n\n")]
        if not v["files"]:
            out.append(("class:hint", "  (no files loaded yet)\n"))
        for i, (kind, key) in enumerate(rows):
            style = "reverse" if i == v["sel"] else ""
            if kind == "file":
                f = v["files"][key]
                box = "x" if f["include"] else " "
                kindlab = "cube" if f["is_cube"] else "molden"
                out.append((style, f"  [{box}] {f['label']:<16s}{f['role']:<9s}{kindlab:<7s}"
                                   f"{os.path.basename(f['path'])}\n"))
            elif kind == "add":
                out.append((style, "  + Add file...\n"))
            elif kind == "setting":
                value = s[key] or _VIEW_DEFAULTS.get(key, s[key])
                out.append((style, f"  {_VIEW_LABELS[key]:<26s}{value}\n"))
            elif kind == "action":
                out.append((style, f"  [{_VIEW_ACTIONS[key]}]\n"))
        out.append(("", "\n"))
        if v["msg"]:
            out.append(("class:status", "\n".join("  " + l for l in v["msg"].splitlines())))
        return out

    def view_preview_text():
        v = view_state()
        if v.get("_image_fragments"):
            return [("class:hint", "  (ANSI art preview via chafa)\n")] + v["_image_fragments"]
        if not v.get("last_image"):
            return [("class:hint", "  Draw or Compare to generate a plot.")]
        if v.get("_image_payload"):
            return [("class:hint", f"  (shown inline via {preview_mode})\n"), ("", f"  {v['last_image']}")]
        return [("class:hint", "  (opened in the system image viewer)\n"), ("", f"  {v['last_image']}")]

    def view_move(d):
        rows = view_rows()
        if rows:
            v = view_state()
            v["sel"] = (v["sel"] + d) % len(rows)

    def view_current_row():
        rows = view_rows()
        return rows[min(view_state()["sel"], len(rows) - 1)] if rows else (None, None)

    def view_cycle(key, d):
        H = view_heavy()
        opts = {"what": ["electron", "positron", "both", "esp"], "plane": ["AUTO", "XY", "XZ", "YZ"],
                "colors": [""] + list(H["_MAP_SCALES"])}[key]
        v = view_state()
        i = opts.index(v["settings"][key]) if v["settings"][key] in opts else 0
        v["settings"][key] = opts[(i + d) % len(opts)]

    async def view_pick_role(basename, default):
        v = await pick("Load as", [("Electron density", "electron"), ("Positron density", "positron"),
                                   ("Other (comparison only)", "other")],
                       text=f"{basename}: what kind of density is this file?")
        return v or default

    async def view_add_file(path, role=None):
        H = view_heavy()
        v = view_state()
        if not os.path.exists(path):
            v["msg"] = f"No such file: {path}"
            return False
        try:
            mol, data = H["load_plot_file"](path)
        except Exception as err:
            v["msg"] = f"Cannot read {path}: {type(err).__name__}: {err}"
            return False
        base = os.path.basename(path)
        if role is None:
            role = await view_pick_role(base, H["_plot_species"](path))
        label = {"electron": "Electron", "positron": "Positron"}.get(role, os.path.splitext(base)[0])
        entry = {"path": path, "label": label, "role": role, "include": True, "mol": mol, "data": data,
                 "is_cube": isinstance(data, H["CubeGrid"])}
        v["files"].append(entry)
        if v["settings"]["prefix"] == "view":
            v["settings"]["prefix"] = os.path.splitext(base)[0]
        v["sel"] = len(v["files"]) - 1
        if entry["is_cube"]:
            v["msg"] = (f"Loaded {base} ({role}): cube {data.shape[0]}x{data.shape[1]}x{data.shape[2]}, "
                       f"{len(data.atoms)} atoms, integral {data.integral():.4f}")
        else:
            n = float(H["np"].einsum("ij,ji", data, mol.intor("int1e_ovlp")))
            v["msg"] = f"Loaded {base} ({role}): {mol.natm} atoms, {mol.nao} basis functions, {n:.4f} particles"
        return True

    async def view_add_flow():
        p = await ask_text("Add file", "File (.molden or .cube; Tab / Right completes):", "",
                           UIPathCompleter(UI_VIEW_EXT))
        if not p:
            return
        path = os.path.expanduser(p)
        if not ui_is_view_file(path):
            await message("Add file", f"Not a .molden/.cube file: {path}")
            return
        await view_add_file(path)

    async def view_rename(i):
        v = view_state()
        f = v["files"][i]
        new = await ask_text("Label", "New label for this file:", f["label"])
        if new:
            f["label"] = new

    async def view_edit_setting(key):
        v = view_state()
        prompts = {"range_electron": "lower,upper in a.u. (e.g. 0,0.4), or blank for auto:",
                   "range_positron": "lower,upper in a.u., or blank for auto:",
                   "range_esp": "lower,upper in a.u., or blank for auto:",
                   "ystep": "spacing in bohr, or blank for auto:", "prefix": "output file prefix:"}
        new = await ask_text("View molden/cube", prompts[key], v["settings"][key])
        if new is not None:
            v["settings"][key] = new.replace(" ", "") if key != "prefix" else (new or v["settings"][key])

    def view_open_externally(path):
        import subprocess
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        try:
            subprocess.Popen([opener, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        except OSError:
            view_state()["msg"] += f"\n(could not open a system image viewer for {path})"

    def view_show_image(written):
        v = view_state()
        pngs = [w for w in written if w.endswith(".png")]
        png = pngs[-1] if pngs else None    # last: the combined side-by-side figure when there is one
        v["_image_payload"], v["_image_fragments"] = None, None
        if not png:
            v["last_image"] = None
            return
        v["last_image"] = os.path.abspath(png)
        if not preview_mode:    # no supported terminal protocol and no chafa installed: not shown inline,
            view_open_externally(v["last_image"])    # just opened in the system's own image viewer
            return
        wp = None
        screen = app.renderer.last_rendered_screen
        if screen:
            wp = screen.visible_windows_to_write_positions.get(view_preview_window)
        cols, rows = (wp.width, wp.height) if wp else (60, 20)
        if preview_mode == "chafa":
            text = ui_chafa_text(v["last_image"], cols, rows)
            if text is None:
                view_open_externally(v["last_image"])
                return
            from prompt_toolkit.formatted_text import ANSI
            v["_image_fragments"] = ANSI(text).__pt_formatted_text__()
            get_app().invalidate()
            return
        payload = ui_image_escape(preview_mode, v["last_image"], cols, rows)
        if payload is None:
            view_open_externally(v["last_image"])
        else:
            v["_image_payload"] = payload
            v["_image_dirty"] = True
            get_app().invalidate()

    async def view_draw():
        v = view_state()
        H = view_heavy()
        s = v["settings"]
        efile = next((f for f in reversed(v["files"]) if f["role"] == "electron"), None)
        pfile = next((f for f in reversed(v["files"]) if f["role"] == "positron"), None)
        files, species = {}, {}
        if efile:
            files["electron"], species["electron"] = efile["path"], (efile["mol"], efile["data"])
        if pfile:
            files["positron"], species["positron"] = pfile["path"], (pfile["mol"], pfile["data"])
        what = s["what"]
        missing = [sp for sp in ({"both": ["electron", "positron"], "esp": ["electron"]}.get(what, [what]))
                   if sp not in files]
        if missing:
            v["msg"] = f"The {what} map needs a loaded {' and '.join(missing)} file."
            return
        if what == "esp" and (efile is None or efile["is_cube"]):
            v["msg"] = "The ESP map needs the electron .molden file (a density cube is not enough)."
            return
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                stage = H["_view_stage"](files, s)
                if stage.errors:
                    v["msg"] = "; ".join(stage.errors)
                    return
                sp_all = dict(species)
                sp_all.setdefault("electron", species.get("positron"))
                stage._plot_source = "from " + ", ".join(os.path.basename(p) for p in files.values())
                map_view = H["map_request_view"](stage, stage.density_maps[0])
                map_view._plot_source = stage._plot_source
                written = H["report_density_maps"](map_view, "file", sp_all)
        except Exception as err:
            v["msg"] = f"{type(err).__name__}: {err}"
            return
        out = buf.getvalue().strip()
        v["msg"] = (out + "\n" if out else "") + ("Written: " + ", ".join(written) if written else "Nothing written.")
        view_show_image(written)

    async def view_compare():
        v = view_state()
        H = view_heavy()
        s = v["settings"]
        included = [f for f in v["files"] if f["include"]]
        if len(included) < 2:
            v["msg"] = "Mark at least two loaded files to compare (Enter on a file row toggles it)."
            return
        sp = s["what"] if s["what"] in ("electron", "positron") else "electron"
        stage = H["JobStage"](1)
        stage.job_name, stage.system_type = s["prefix"], "ELECTRONIC"
        stage.density_map_plane = s["plane"]
        stage.density_map_colors = s["colors"] or "rainbow_bw"
        stage.density_map_ystep = float(s["ystep"]) if s["ystep"] else None
        rng = s[f"range_{sp}"]
        stage.density_map_limits = {sp: tuple(float(x) for x in rng.split(","))} if rng else {}
        stem = f"{s['prefix']}_compare_{sp}"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                plane, offset = H["density_map_plane"](stage, included[0]["mol"])
                panels = [H["density_map_panel"](stage, sp, f["mol"], f["data"], plane, offset, label=f["label"])
                          for f in included]
                written = H["write_density_map"](stem, panels, f"{sp.capitalize()} density comparison: "
                                                 f"{s['prefix']}",
                                                 f"{plane} plane; " + ", ".join(f["label"] for f in included))
        except Exception as err:
            v["msg"] = f"{type(err).__name__}: {err}"
            return
        v["msg"] = "Written: " + ", ".join(written)
        view_show_image(written)

    async def view_action(key):
        v = view_state()
        H = view_heavy()
        if key == "close":
            st["screen"] = "start"
            get_app().layout.focus(start_buttons[0])
        elif key == "info":
            if not v["files"]:
                v["msg"] = "No files loaded."
                return
            lines = []
            for f in v["files"]:
                mol = f["mol"]
                lines.append(f"{f['label']} ({f['role']}): {f['path']}")
                for i in range(mol.natm):
                    x = mol.atom_coord(i) * H["BOHR_TO_ANGSTROM"]
                    lines.append(f"   {i + 1:3d} {mol.atom_symbol(i):<3s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}  A")
            await message("Files loaded", "\n".join(lines))
        elif key == "chelpg":
            efile = next((f for f in reversed(v["files"]) if f["role"] == "electron" and not f["is_cube"]), None)
            if not efile:
                v["msg"] = "CHELPG needs an electron .molden file."
                return
            stage = H["JobStage"](1)
            stage.job_name = v["settings"]["prefix"]
            stage.system_type = "POSITRON" if any(f["role"] == "positron" for f in v["files"]) else "ELECTRONIC"
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                H["report_chelpg"](stage, "file", {"electron": (efile["mol"], efile["data"])})
            v["msg"] = buf.getvalue().strip() or "Done."
        elif key == "draw":
            await view_draw()
        elif key == "compare":
            await view_compare()

    async def view_activate():
        kind, key = view_current_row()
        v = view_state()
        if kind == "file":
            v["files"][key]["include"] = not v["files"][key]["include"]
        elif kind == "add":
            await view_add_flow()
        elif kind == "setting" and key in ("range_electron", "range_positron", "range_esp", "ystep", "prefix"):
            await view_edit_setting(key)
        elif kind == "action":
            await view_action(key)

    async def view_open(path=None):
        view_state()
        st["screen"] = "molden"
        get_app().layout.focus(view_window)
        get_app().invalidate()
        if path:
            # --ui job.molden / the start-screen button's preload: straight in, guessed role, no prompt
            # ("Load as" is only asked from the in-screen "+ Add file" flow, where it is less jarring).
            await view_add_file(path, role=view_heavy()["_plot_species"](path))
            get_app().invalidate()

    view_kb = KeyBindings()

    @view_kb.add("down")
    def _(e):
        view_move(1)

    @view_kb.add("up")
    def _(e):
        view_move(-1)

    @view_kb.add("left")
    def _(e):
        kind, key = view_current_row()
        if kind == "setting" and key in ("what", "plane", "colors"):
            view_cycle(key, -1)

    @view_kb.add("right")
    def _(e):
        kind, key = view_current_row()
        if kind == "setting" and key in ("what", "plane", "colors"):
            view_cycle(key, +1)

    @view_kb.add("enter")
    @view_kb.add(" ")
    def _(e):
        run_bg(view_activate())

    @view_kb.add("d")
    def _(e):
        kind, key = view_current_row()
        if kind == "file":
            v = view_state()
            del v["files"][key]
            v["sel"] = max(0, v["sel"] - 1)

    @view_kb.add("r")
    def _(e):
        kind, key = view_current_row()
        if kind == "file":
            run_bg(view_rename(key))

    @view_kb.add("escape", eager=True)
    @view_kb.add("q", eager=True)
    def _(e):
        st["screen"] = "start"
        get_app().layout.focus(start_buttons[0])

    # The preview pane is purely a bonus: it needs either a terminal that understands Kitty/iTerm2 graphics,
    # or Sixel plus an installed converter (ImageMagick/img2sixel), or chafa (plain ANSI art, works anywhere).
    # With none of those, there is nothing to show there, so it is left out of the layout entirely rather
    # than kept as dead, always-empty space; Draw/Compare still work, just always opening the plot externally.
    preview_mode = ui_preview_mode()
    view_control = FormattedTextControl(view_text, key_bindings=view_kb, focusable=True, show_cursor=False)
    view_window = Window(view_control, wrap_lines=False, width=(D(preferred=56) if preview_mode else None))
    if preview_mode:
        view_preview_control = FormattedTextControl(view_preview_text, focusable=False, show_cursor=False)
        view_preview_window = Window(view_preview_control, wrap_lines=True)
        molden_screen = VSplit([view_window, Window(width=1, char="│", style="class:frame.border"),
                                view_preview_window])
    else:
        view_preview_window = None
        molden_screen = view_window

    def view_after_render(app_):
        v = st.get("view")
        if (not v or view_preview_window is None or st["screen"] != "molden" or
                not v.get("_image_payload") or not v.get("_image_dirty")):
            return
        screen = app_.renderer.last_rendered_screen
        wp = screen.visible_windows_to_write_positions.get(view_preview_window) if screen else None
        if wp is None:
            return    # the preview pane has not rendered yet (e.g. a dialog is covering it); try next render
        from prompt_toolkit.data_structures import Point
        app_.output.cursor_goto(wp.ypos + 1, wp.xpos + 1)
        app_.output.write_raw(v["_image_payload"])
        # Writing raw bytes bypasses the renderer's own cursor-position bookkeeping (Renderer._cursor_pos),
        # which it otherwise uses to compute minimal cursor moves for the *next* frame's diff; left alone,
        # that next diff is computed from a stale belief about where the cursor is and corrupts the
        # following redraw (seen as the whole pane going blank after one Draw, on a real Konsole/foot
        # session). Leaving the cursor at a position we then tell the renderer about keeps the two in sync
        # without discarding its diff cache (Renderer.reset() would, and would redraw blank cells right
        # over the image we just drew).
        app_.output.cursor_goto(wp.ypos + 1, wp.xpos + 1)
        app_.renderer._cursor_pos = Point(x=wp.xpos, y=wp.ypos)
        app_.output.flush()
        v["_image_dirty"] = False

    def b_view():
        run_bg(view_open(None))

    def engine_display():
        if cfg.get("engine"):
            return os.path.basename(cfg["engine"])
        guess = ui_default_engine_guess()
        return f"(auto) {os.path.basename(guess)}" if guess else "(not found - press Enter)"

    async def set_engine():
        p = await ask_text("Engine path", "Path to the newBRORSEN engine script this UI drives:",
                           cfg.get("engine") or ui_default_engine_guess(), PathCompleter())
        if p is None:
            return
        if p and not os.path.isfile(p):
            await message("Engine path", f"No such file: {p}")
            return
        cfg["engine"] = p
        ui_save_settings(cfg, cfg_path)
        get_app().invalidate()

    async def b_about():
        """Settings > About: Left/Right cycle ui_easter_page_lines (wraps), Enter or Esc closes. The
        content window's own key bindings take priority over the dialog's (same pattern as the Settings
        rows below), so Left/Right page here instead of moving focus."""
        state = {"i": 0}
        box = {}
        rkb = KeyBindings()

        @rkb.add("left")
        def _(e):
            state["i"] -= 1
            get_app().invalidate()

        @rkb.add("right")
        def _(e):
            state["i"] += 1
            get_app().invalidate()

        @rkb.add("enter")
        def _(e):
            box["set"](True)

        def content():
            out = []
            for line in ui_easter_page_lines(state["i"]):
                out.extend(line)
                out.append(("", "\n"))
            return out
        ctl = FormattedTextControl(content, focusable=True, key_bindings=rkb, show_cursor=False)
        body = HSplit([Window(ctl), Window(height=1), Label("Left/Right: browse   Enter: close")])
        await show(body, "About", [("Close", True)], width=D(preferred=62), box=box)

    def b_settings():
        """Settings: Up/Down move between the rows, Left/Right change the value, Enter or Esc closes
        (Enter on Engine path instead opens a text prompt)."""
        async def go():
            names = {"vim": "vim (starts in normal mode)", "emacs": "emacs", "plain": "plain (Ctrl-S, Ctrl-Q)"}
            rows = [("Keymap", lambda: names[cfg["keymap"]],
                     lambda d: cfg.__setitem__("keymap", UI_KEYMAPS[(UI_KEYMAPS.index(cfg["keymap"]) + d)
                                                                    % len(UI_KEYMAPS)])),
                    ("Mouse", lambda: "on" if cfg["mouse"] else "off",
                     lambda d: cfg.__setitem__("mouse", not cfg["mouse"])),
                    ("Geometry position", lambda: cfg.get("geometry", "top"),
                     lambda d: cfg.__setitem__("geometry", UI_GEOMETRY_POSITIONS[
                         (UI_GEOMETRY_POSITIONS.index(cfg.get("geometry", "top")) + d)
                         % len(UI_GEOMETRY_POSITIONS)])),
                    ("Engine path", engine_display, None, lambda: run_bg(set_engine())),
                    ("About", lambda: "", None, lambda: run_bg(b_about()))]
            box = {}
            wins = []
            for label, value, change, *rest in rows:
                on_enter = rest[0] if rest else (lambda: box["set"](True))
                rkb = KeyBindings()
                if change is not None:
                    @rkb.add("left")
                    def _(e, change=change):
                        change(-1)

                    @rkb.add("right")
                    def _(e, change=change):
                        change(+1)

                    @rkb.add("enter")
                    def _(e):
                        box["set"](True)
                else:
                    @rkb.add("enter")
                    def _(e, on_enter=on_enter):
                        on_enter()
                ctl = FormattedTextControl(
                    lambda label=label, value=value: [("", f"{label:<12s}"), ("class:arrow", " < "),
                                                      ("class:setting", f"{value():<28s}"), ("class:arrow", " >")],
                    focusable=True, key_bindings=rkb, show_cursor=False)
                win = Window(ctl, height=1)
                win.style = (lambda w=win: "reverse" if get_app().layout.has_focus(w) else "")
                wins.append(win)
            body = HSplit([Label("Up/Down: choose a setting   Left/Right: change it   Enter: done"
                                 " (or edit, for Engine path)"),
                           Window(height=1)] + wins)
            await show(body, "Settings", [("Done", True)], width=D(preferred=56), box=box)
            ui_save_settings(cfg, cfg_path)
        run_bg(go())

    def b_quit():
        get_app().exit()

    start_buttons = [Button("New input", handler=b_new, width=24),
                     Button("Open existing .inp", handler=b_open, width=24),
                     Button("Recent files", handler=b_recent, width=24),
                     Button("View molden/cube", handler=b_view, width=24),
                     Button("Settings", handler=b_settings, width=24),
                     Button("Quit", handler=b_quit, width=24)]
    big = Condition(lambda: get_app().output.get_size().rows >= 30)
    start_kb = KeyBindings()
    start_kb.add("down")(focus_next)
    start_kb.add("tab")(focus_next)
    start_kb.add("up")(focus_previous)
    start_kb.add("s-tab")(focus_previous)
    start_kb.add("q")(lambda e: e.app.exit())
    menu = HSplit([ConditionalContainer(Window(height=6, char=" ", style="class:logo"), filter=big),  # logo space
                   Window(FormattedTextControl([("class:title", f"newBRORSEN v{VERSION}")]), height=1,
                          align=WindowAlign.CENTER),
                   Window(FormattedTextControl([("class:subtitle", "input builder")]), height=1,
                          align=WindowAlign.CENTER),
                   Window(height=1)] + [VSplit([Window(), b, Window()]) for b in start_buttons],
                  key_bindings=start_kb)
    start_screen = HSplit([Window(), VSplit([Window(), Box(menu, width=40, padding=0), Window()]), Window()])
    too_small = Window(FormattedTextControl(lambda: "Terminal too small: newBRORSEN --ui needs 80x24."),
                       align=WindowAlign.CENTER)

    def current():
        size = get_app().output.get_size()
        if size.columns < 80 or size.rows < 24:
            return too_small
        return {"start": start_screen, "editor": editor_screen, "viewer": viewer_screen,
                "molden": molden_screen}[st["screen"]]

    root.content = DynamicContainer(current)
    root.floats.extend([help_float, Float(xcursor=True, ycursor=True,
                                          content=CompletionsMenu(max_height=12, scroll_offset=1))])

    # Colors: only the terminal's own palette. prompt_toolkit always merges its built-in UI style, which has fixed
    # colors (black menu text, white dialogs, a red focused button, ...); on a dark terminal some of them are
    # unreadable (dark-on-dark completion menu). Every such rule is reset to the terminal's default colors first;
    # focus, input fields and selections are then shown with reverse / bold / underline only.
    from prompt_toolkit.styles.defaults import default_ui_style
    rules = [(cls, "fg:default bg:default noreverse") for cls, sty in default_ui_style().style_rules
             if re.search(r"#|bg:|ansi\w", sty) and not cls.startswith("ansi")]
    rules += list({
        "bang": "ansicyan bold", "key": "ansiblue bold", "value": "", "struct": "ansimagenta",
        "element": "ansigreen", "comment": "ansibrightblack italic",
        "error": "ansired underline bold", "warning": "ansiyellow underline",
        "gutter.error": "ansired bold", "gutter.warning": "ansiyellow",
        "status": "reverse", "status.mode": "reverse bold", "status.error": "reverse ansired",
        "status.warning": "reverse ansiyellow", "status.inherited": "reverse ansibrightblack",
        "hint": "ansibrightblack", "title": "bold", "subtitle": "ansibrightblack", "dim": "ansibrightblack",
        "dialog frame.label": "bold", "frame.border": "ansibrightblack",
        "button.focused": "reverse bold", "dialog.body text-area": "underline",
        "completion-menu": "reverse", "completion-menu.completion": "reverse",
        "completion-menu.completion.current": "noreverse bold",
        "completion-menu.meta.completion": "reverse", "completion-menu.meta.completion.current": "noreverse",
        "completion-menu.multi-column-meta": "reverse", "completion-toolbar.completion.current": "reverse",
        "scrollbar.button": "reverse", "search": "reverse", "matching-bracket.cursor": "reverse",
        "validation-toolbar": "reverse", "window-too-small": "reverse", "menu-bar.selected-item": "reverse",
        "radio-selected": "bold", "line-number": "ansibrightblack", "arrow": "ansibrightblack",
        "auto-suggestion": "ansibrightblack italic", "ghost": "ansibrightblack",
        "ghost.hl": "ansicyan italic underline", "search-toolbar": "bold", "search": "reverse",
        "search.current": "reverse bold", "bar.prompt": "bold", "bar.current": "reverse bold",
    }.items())
    style = Style(rules)
    app = Application(layout=Layout(root, focused_element=start_buttons[0]), key_bindings=kb, style=style,
                      full_screen=True, mouse_support=Condition(lambda: cfg["mouse"]), color_depth=ColorDepth.DEPTH_4_BIT,
                      editing_mode=EditingMode.VI if cfg["keymap"] == "vim" else EditingMode.EMACS,
                      include_default_pygments_style=False, input=input, output=output,
                      after_render=view_after_render)
    app.ttimeoutlen = 0.1          # a lone Esc (vim) is recognized quickly; still enough for escape sequences

    def pre_run():
        app.create_background_task(follow_loop())
        if path:
            if ui_is_view_file(path):
                run_bg(view_open(path))                # --ui job.molden / job.cube: straight into the viewer
            elif ui_looks_like_geometry(path):
                new_from_geometry(None, path)          # --ui "<xyz or Z-matrix rows>"
            elif os.path.exists(path) and path.lower().endswith((".xyz", ".zmat")):
                new_from_geometry(path)                # --ui molecule.xyz: a new input with this geometry
            elif os.path.exists(path):
                load(path)
            else:
                new_input()
                st["path"] = os.path.abspath(path)
    st.update(app=app, editor=editor, cmdline=cmdline, save_flow=save_flow, launch_flow=launch_flow,
              geometry_flow=geometry_flow, new_input=new_input, load=load, cfg=cfg, pre_run=pre_run,
              sugg=suggestion_for)
    return app, st


def _ui_main(argv):
    """--ui [file.inp]: the terminal input builder. Runs before numpy/pyscf are imported."""
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        print("newBRORSEN --ui needs prompt_toolkit, which is not installed in this environment.\n"
              f"Install it with:  {sys.executable} -m pip install prompt_toolkit", file=sys.stderr)
        return 1
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("newBRORSEN --ui needs an interactive terminal.", file=sys.stderr)
        return 1
    app, st = _ui_make_app(argv[0] if argv else None)
    app.run(pre_run=st["pre_run"])
    if st.get("message", "").startswith("Saved"):
        print(st["message"])
    return 0



if __name__ == "__main__":
    sys.exit(_ui_main(sys.argv[1:]))
