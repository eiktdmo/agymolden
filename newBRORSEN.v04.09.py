#!/usr/bin/env python3
"""newBRORSEN v04.09 -- v04.08 (production, signed 2026-10-18) plus:
  * Merged Kurt Brorsen's 2026-10-08 wrapper integration (written against the 2026-09-29 v04.05 he received;
    three-way merge with that file as base): two-positron DF reference_model='scf' (the pyscf_neo_cc
    0.1.0.post20261008 default, Kurt's recommendation; 'legacy' reproduced the old offsets), %aux_conditioning,
    analytic POSHF forces and geomeTRIC POSHF optimization, POSHF SCF recovery (also in scans), native numerical /
    analytic Hessians, gradient admission checks, digest-keyed result and resume caches, density_model densities
    for ESP / CHELPG, strict (T) API.
    REQUIRES pyscf_neo_cc 0.1.0.post20261008 (env MCCC_1008_stock); it does not run on the 2026-09-23 library.
  * %pos_basis / %basis atom ranges: '%pos_basis PSX-TZ [1,5] PSX-DZ' (atoms 1 to 5, inclusive; also [1-5]).
  * Fix (2026-10-19 bug report): '%basis <file path containing a space>' was split into two basis names and
    failed validation in v04.08; a value that is an existing file is now taken whole.

v04.08 -- v04.07 (development build, renamed 04.08 on 2026-10-14) plus:
  * %pos_basis and %basis take the atoms each basis goes on and several bases on one line
    ('%pos_basis PSX-DZ H PSX-TZ {1 2}', '%basis aug-cc-pvtz H cc-pvdz'), replacing %pos_atom / %pos_basis_map
    (kept as aliases); MANUAL.md 8.2.
  * Input checks: unknown %basis names, a second %jobname / geometry block in one stage (missing %next),
    %pos_basis_map written with values, numbered map keys past the last atom, %pos_atom numbers in a
    Z-matrix with dummy atoms.

v04.07 -- v04.06 plus:
  * The --ui terminal input builder is split out into its own program, input_builder.v04.07.py (next to
    this file); it drives this engine only through its command line (--dry-run, --parse-text-json,
    --view, a plain run) and, for the Molden/cube viewer, by loading this file's module namespace with
    runpy. See input_builder.v04.07.py's own docstring for that side.
  * parse_text(text) and --parse-text-json (stdin -> JSON diagnostics; writes no files) for the UI's check.
  * The Molden/cube file viewer is reached through the input builder ("View molden/cube", or a molden/cube
    file argument) or the explicit --view flag; v04.06's bare `python wrapper.py file.molden`
    auto-detection is gone.
v04.05 -- v04.04 (the production release of 2026-09-26) plus:
  * ELECTRONIC OPT runs geomeTRIC (it always ran the wrapper's WOPT before), with the exact analytic
    gradient where PySCF has one (HF, DFT, MP2, CCSD, closed-shell CCSD(T), without density fitting for the
    correlated methods) and a numerical gradient otherwise; WOPT is the fallback and stays available with
    %opt_engine WOPT. Electronic CCSD gradients use nuc_grad_method() (PySCF 2.14 has no CCSD.Gradients, so
    up to v04.04 they were silently numerical), and a GRADIENT job no longer starts an optimization when the
    analytic gradient is missing.
  * PySCF's cc-pvdz-jkfit / aug-cc-pvdz-jkfit sets are replaced by aug-cc-pvtz-jkfit for Ga-Kr (they have
    no g functions: 480-720 mHa errors in the SCF).
  * --resume (automatic after a SLURM requeue): stages that already finished are skipped and an
    interrupted optimization restarts from its checkpoint geometry.
See MANUAL.md section 19 for v04.04 and earlier.

Runs input files written for newBRORSEN v03.17 (localfix / merged) and for
twoposBrorsen (00.44 - 00.86) and the old Brorsen-MCCCSDT-v5 wrapper, on top of
pyscf + pyscf_neo_cc. Full documentation: MANUAL.md next to this file; every
option in one commented file: template.inp.

Usage:
    python newBRORSEN.v04.08.py job.inp            # run
    python newBRORSEN.v04.08.py job.inp --dry-run  # parse + validate only
    python newBRORSEN.v04.08.py job.inp --resume   # skip the stages that already finished
    python newBRORSEN.v04.08.py --view file.molden [file2]  # Molden/cube file viewer, no calculation
"""

import os
import sys
import re
import ast
import math
import copy
import time
import gc
import inspect
import io
import contextlib
import argparse
import tempfile
import json

VERSION = "04.09"
# Not approved for production runs until the user signs off on the test
# results (tests/ next to this file). Flip to True only then. (v04.08 is the production release.)
RELEASED = False


class TeeOutput:
    """Duplicate stdout/stderr into <input>.out, flushed on every write, so
    the log survives however the process is launched or killed."""
    def __init__(self, filename, append=False):
        self.terminal_out = sys.stdout
        self.terminal_err = sys.stderr
        self.log_file = open(filename, 'a' if append else 'w', encoding='utf-8')
        if append:
            self.log_file.write("\n" + "#" * 80 + f"\n# resumed {time.strftime('%Y-%m-%d %H:%M:%S')}\n" + "#" * 80 + "\n")

    def write(self, message):
        self.terminal_out.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal_out.flush()
        self.log_file.flush()

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass


import numpy as np
import warnings
import pyscf
from pyscf import lib, gto, scf, dft, cc
from pyscf import df as pyscf_df
from pyscf.data import nist, elements as pyscf_elements

try:
    import pyscf_neo_cc
    from pyscf_neo_cc import CCSD as NEO_CCSD, DFCCSD as NEO_DFCCSD
    from pyscf_neo_cc.geomopt import optimize as neo_optimize
    HAS_NEO_CC = True
except ImportError:
    pyscf_neo_cc = None
    HAS_NEO_CC = False

try:
    from pyscf_neo_cc._neo import Mole as NEOMole, HF as NEOHF, KS as NEOKS
    from pyscf_neo_cc._neo import Gradients as NEOGradients
except ImportError:
    try:
        from pyscf import neo
        NEOMole = neo.M
        NEOHF = neo.HF
        NEOKS = neo.KS
    except ImportError:
        NEOMole = NEOHF = NEOKS = None

if isinstance(NEOMole, type):
    class ScaledNEOMole(NEOMole):
        """NEO Mole whose nuclear basis on the quantum nuclei in `nuc_basis_scale` ({atom: ratio}) has
        its exponents multiplied by the ratio. The library scales the basis only for the H+ / H* / H#
        symbols; a mass list (%nuc_mass) would leave a triton or an antimuon in the proton's basis."""
        def _get_nuclear_basis(self, atom_id):
            basis = super()._get_nuclear_basis(atom_id)
            ratio = (getattr(self, "nuc_basis_scale", None) or {}).get(atom_id)
            if not ratio or ratio == 1.0:
                return basis
            out = copy.deepcopy(basis)
            for shell in out:
                for prim in shell[1:]:
                    prim[0] *= ratio
            return out
else:
    ScaledNEOMole = None
# NEO Mole.build(mass=[...]) (pyscf_neo_cc 2026-09-15 and later); needed by %nuc_mass
try:
    NEO_TAKES_MASS = ScaledNEOMole is not None and "mass" in inspect.signature(NEOMole.build).parameters
except (TypeError, ValueError):
    NEO_TAKES_MASS = False

try:
    from pyscf_neo_cc.positron import (build_system, POSHF, single_positron_dfccsd,
                                       single_positron_ccsd,
                                       POSCCSD, POSDFCCSD, apply_df_reference)
    HAS_POSITRON = True
except ImportError:
    HAS_POSITRON = False

# Qualified stationary HF gradients and their actual-SCF metric admission.
try:
    from pyscf_neo_cc.positron.hf_grad import preflight as _native_poshf_preflight
except ImportError:
    _native_poshf_preflight = None


try:
    from pyscf_neo_cc.positron import POSSOCCSD, POSSODFCCSD
except ImportError:
    POSSOCCSD = POSSODFCCSD = None

# Constrained NEO (CNEO): the expectation value of each quantum nucleus's position is held at its
# geometry coordinate (Lagrange multiplier), so optimized geometries are nuclear positions, not the
# centres of the nuclear basis functions (which plain NEO optimizes). CDFT with xc='HF' is CNEO-HF.
try:
    from pyscf_neo_cc.cneo import CNEOCCSD, CNEODFCCSD, CNEOMP2, CNEODFMP2
    from pyscf_neo_cc.cneo.scf import (CDFT as CNEO_CDFT, _second_order_tighten as _cneo_tighten,
                                       _accept_convergence as _cneo_accept)
    HAS_CNEO = True
except ImportError:
    CNEOCCSD = CNEODFCCSD = CNEOMP2 = CNEODFMP2 = CNEO_CDFT = None
    HAS_CNEO = False

# MP2 (pyscf_neo_cc 2026-09-15+): DF-MP2 with analytic gradients for one positron (DFMP2 on a POSHF built
# with its DF recipe), a closed-shell positron pair (POSDFMP2) and NEO (DFMP2 on a mean field set up by
# cd_density_fit_unified). The electronic sector uses PySCF's MP2 / UMP2 (analytic gradients) and
# DFMP2 / DFUMP2 (energies only).
try:
    from pyscf_neo_cc import DFMP2 as NEO_DFMP2
    from pyscf_neo_cc.positron import POSDFMP2
    from pyscf_neo_cc.df import cd_density_fit_unified
except ImportError:
    NEO_DFMP2 = POSDFMP2 = cd_density_fit_unified = None

# v04.09 runs only on pyscf_neo_cc 0.1.0.post20261008 or newer (Kurt Brorsen's 2026-10-08 release): the
# two-positron reference model, POSHF recovery and forces, density_model and the auxiliary conditioning all come
# from it. Stop at once, before any input is read, rather than part way through a calculation. (The older
# library's feature checks, POSHF_TAKES_DF_AUX / NEO_CC_0923 / HAS_NEO_MP2, were removed with this.)
MIN_NEO_CC = 20261008


def _library_problem():
    try:
        import pyscf_neo_cc
    except ImportError as err:
        return f"pyscf_neo_cc is not importable ({err})"
    ver = str(getattr(pyscf_neo_cc, "__version__", "?"))
    date = ver.split("post")[-1] if "post" in ver else ""
    if not (date.isdigit() and int(date) >= MIN_NEO_CC):
        return f"pyscf_neo_cc {ver} is too old (from {pyscf_neo_cc.__file__})"
    missing = [n for n, ok in (("positron sector", HAS_POSITRON), ("DF-MP2 classes", NEO_DFMP2 is not None),
                               ("POSHF forces", _native_poshf_preflight is not None),
                               ("POSHF.recover_scf", HAS_POSITRON and hasattr(POSHF, "recover_scf")))
               if not ok]
    return f"pyscf_neo_cc {ver} lacks: {', '.join(missing)}" if missing else None


_LIB_PROBLEM = _library_problem()
if _LIB_PROBLEM:
    sys.exit(f"[ERROR] newBRORSEN v04.09 needs pyscf_neo_cc 0.1.0.post{MIN_NEO_CC} or newer: {_LIB_PROBLEM}.\n"
             f"        Use the conda env MCCC_1008_stock, or run newBRORSEN.v04.08.py (legacy line) with "
             f"MCCC_0923_stock.")

try:
    from pyscf_neo_cc.df import FixedAuxiliaryShells, PivotedAuxiliaryShells
except ImportError:
    FixedAuxiliaryShells = PivotedAuxiliaryShells = None

try:
    from pyscf_neo_cc import NumericalHessian
except ImportError:
    NumericalHessian = None

try:
    from pyscf_neo_cc.scf import converge as neo_scf_converge
except ImportError:
    neo_scf_converge = None

try:
    _DEFAULT_DF_MEM_CAP = pyscf_neo_cc.get_df_mem_cap()
except Exception:
    _DEFAULT_DF_MEM_CAP = None

# Post-OPT exact-integral (CONV) cross-check default. It OOM-killed several
# n_pos=1 OCN-containing jobs in September 2026, so it is off unless a stage
# asks for it with "%conv_check true".
RUN_CONV_CROSSCHECK = False

BOHR_TO_ANGSTROM = 0.529177210903
ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

# PSX positron basis sets, copied from openLOWDIN (lib/basis/PSX-DZ, PSX-TZ,
# PSX-QZ; same exponents for every element) so the wrapper also runs where
# openLOWDIN is not installed. Uncontracted: [l, [exponent, 1.0]].
PSX_DZ_DATA = [
    [0, [0.0189693659, 1.0]],
    [0, [0.05186863351164733, 1.0]],
    [0, [0.14182630861506998, 1.0]],
    [0, [0.3878008818346877, 1.0]],
    [0, [1.0603781866729172, 1.0]],
    [1, [0.0590955656, 1.0]],
    [1, [0.1612796747084685, 1.0]],
    [1, [0.44015372744092, 1.0]],
    [2, [0.1165448188, 1.0]],
    [2, [0.31287113568838126, 1.0]],
]
PSX_TZ_DATA = [
    [0, [0.0170647199, 1.0]],
    [0, [0.043427005943720544, 1.0]],
    [0, [0.110514843272402, 1.0]],
    [0, [0.2812427501760487, 1.0]],
    [0, [0.7157181984290043, 1.0]],
    [0, [1.8213893131176055, 1.0]],
    [1, [0.0496312946, 1.0]],
    [1, [0.12420746812256472, 1.0]],
    [1, [0.31084208586043877, 1.0]],
    [1, [0.777914595656386, 1.0]],
    [2, [0.0950648091, 1.0]],
    [2, [0.23367772929933314, 1.0]],
    [2, [0.5744005766955504, 1.0]],
    [3, [0.1595675148, 1.0]],
    [3, [0.4273064584999289, 1.0]],
]
PSX_QZ_DATA = [
    [0, [0.0160464176, 1.0]],
    [0, [0.03889166807801842, 1.0]],
    [0, [0.0942616528869819, 1.0]],
    [0, [0.22846176685354913, 1.0]],
    [0, [0.5537222965570748, 1.0]],
    [0, [1.342055547968279, 1.0]],
    [0, [3.252737166322123, 1.0]],
    [1, [0.0417201747, 1.0]],
    [1, [0.09408132465023163, 1.0]],
    [1, [0.21215864294888204, 1.0]],
    [1, [0.47842959211353325, 1.0]],
    [1, [1.0788854577330238, 1.0]],
    [2, [0.0813263191, 1.0]],
    [2, [0.18086654137639638, 1.0]],
    [2, [0.40224008846675685, 1.0]],
    [2, [0.8945661676198747, 1.0]],
    [3, [0.1176930592, 1.0]],
    [3, [0.2838834408790911, 1.0]],
    [3, [0.6847456303128573, 1.0]],
    [4, [0.1419052471, 1.0]],
    [4, [0.3846470886929989, 1.0]],
]
# PSX-5Z (8s6p5d4f3g2h, 2026-10-06): built in this project with the recipe of the PSX-DZ/TZ/QZ sets (Moncada et al.,
# Chem. Sci. 11, 44 (2020), ESI): even-tempered alpha_l * beta_l**i, (alpha_l, beta_l) minimizing the e+[X-]
# (X = H, F, Cl, Br; aug-cc-pVQZ) HF energy (s) or MP2 e-e+ correlation energy (p-h). Not part of openLOWDIN; see
# results/Tests/PSX_5Z_2026-10-06/REPORT.md.
PSX_5Z_DATA = [
    [0, [0.0152, 1.0]],
    [0, [0.035416, 1.0]],
    [0, [0.08251928, 1.0]],
    [0, [0.1922699224, 1.0]],
    [0, [0.4479889192, 1.0]],
    [0, [1.0438141817, 1.0]],
    [0, [2.4320870434, 1.0]],
    [0, [5.6667628111, 1.0]],
    [1, [0.0342243406, 1.0]],
    [1, [0.0728586288, 1.0]],
    [1, [0.1551053928, 1.0]],
    [1, [0.330196756, 1.0]],
    [1, [0.7029407276, 1.0]],
    [1, [1.4964582706, 1.0]],
    [2, [0.0644239979, 1.0]],
    [2, [0.1365774959, 1.0]],
    [2, [0.2895413666, 1.0]],
    [2, [0.6138214968, 1.0]],
    [2, [1.3012884286, 1.0]],
    [3, [0.0994752235, 1.0]],
    [3, [0.2234945159, 1.0]],
    [3, [0.5021330629, 1.0]],
    [3, [1.1281601782, 1.0]],
    [4, [0.1655623398, 1.0]],
    [4, [0.3830398749, 1.0]],
    [4, [0.8861891293, 1.0]],
    [5, [0.278701794, 1.0]],
    [5, [0.7222952029, 1.0]],
]
# PSX-6Z (9s7p6d5f4g3h2i, 2026-10-15): same recipe as PSX-5Z, extended to i functions (p-i from MP2 e-e+
# correlation on a fixed background). Not part of openLOWDIN; see results/Tests/PSX_6Z_2026-10-15/REPORT.md.
PSX_6Z_DATA = [
    [0, [0.0144212781, 1.0]],
    [0, [0.032472152, 1.0]],
    [0, [0.0731170042, 1.0]],
    [0, [0.1646363414, 1.0]],
    [0, [0.3707089101, 1.0]],
    [0, [0.8347190834, 1.0]],
    [0, [1.8795230686, 1.0]],
    [0, [4.2320908143, 1.0]],
    [0, [9.5293284554, 1.0]],
    [1, [0.031438749, 1.0]],
    [1, [0.0628250834, 1.0]],
    [1, [0.1255454252, 1.0]],
    [1, [0.2508815418, 1.0]],
    [1, [0.5013448152, 1.0]],
    [1, [1.0018537911, 1.0]],
    [1, [2.0020372969, 1.0]],
    [2, [0.0543716702, 1.0]],
    [2, [0.1038881456, 1.0]],
    [2, [0.1984994531, 1.0]],
    [2, [0.3792736182, 1.0]],
    [2, [0.7246794649, 1.0]],
    [2, [1.3846476571, 1.0]],
    [3, [0.0860150907, 1.0]],
    [3, [0.1685473093, 1.0]],
    [3, [0.3302699006, 1.0]],
    [3, [0.6471667079, 1.0]],
    [3, [1.2681287245, 1.0]],
    [4, [0.1567199463, 1.0]],
    [4, [0.3308526008, 1.0]],
    [4, [0.6984652946, 1.0]],
    [4, [1.4745350846, 1.0]],
    [5, [0.2753212475, 1.0]],
    [5, [0.6939202771, 1.0]],
    [5, [1.748958191, 1.0]],
    [6, [0.4999467407, 1.0]],
    [6, [1.1372996634, 1.0]],
]
PSX_SETS = {'PSX-DZ': PSX_DZ_DATA, 'PSX-TZ': PSX_TZ_DATA, 'PSX-QZ': PSX_QZ_DATA, 'PSX-5Z': PSX_5Z_DATA,
            'PSX-6Z': PSX_6Z_DATA}

# Particle masses (bare particles, amu). Proton, deuteron, antimuon and muonic helium are the values
# pyscf_neo_cc's NEO Mole uses for H / H+ / H* / H# (its isotope table minus the electron mass), so the
# POSITRON route (%pos_mass + %quantum_site) and the PROTON route (%nuc_mass, or the H+ / H* symbols)
# describe the same particle; triton: AME2020 3H atomic mass 3.01604928132 u minus the electron.
AMU_TO_ME = nist.ATOMIC_MASS / nist.E_MASS
_ME_AMU = 1.0 / AMU_TO_ME
PARTICLE_MASS_AMU = {
    "positron": _ME_AMU,
    "antimuon": 0.1134289259,
    "proton": 1.007825 - _ME_AMU,
    "deuteron": 2.01410177811 - _ME_AMU,
    "triton": 3.01604928132 - _ME_AMU,
    "muonic_helium": 4.002603254 - 2 * _ME_AMU + 0.1134289259,   # He-4 nucleus + mu- (charge +1)
}
PARTICLE_ALIASES = {
    "positron": "positron", "e+": "positron",
    "antimuon": "antimuon", "anti-muon": "antimuon", "mu+": "antimuon", "muon+": "antimuon",
    "proton": "proton", "p": "proton", "p+": "proton",
    "deuteron": "deuteron", "d": "deuteron", "d+": "deuteron",
    "triton": "triton", "t": "triton", "t+": "triton",
    "muonic_helium": "muonic_helium", "muonic-helium": "muonic_helium", "hemu": "muonic_helium",
}
# Particles the library cannot hold: its second particle (POSITRON sector) and its quantum nuclei
# (PROTON sector) are positive. A negative muon exists only inside muonic helium (He-4 + mu-).
NEGATIVE_PARTICLES = {"muon", "mu", "mu-", "muon-", "negative_muon", "antiproton", "p-", "electron", "e-"}
# The protonic nuclear basis sets (pb4d ... pb6h) are made for the proton; for another mass the library
# multiplies their exponents by sqrt(m / m_ref) (its H+ / H* / H# rule), and so does the wrapper.
PROTON_REF_AMU = 1.007825 - _ME_AMU


def parse_particle_mass(val, default_unit):
    """'%pos_mass' / '%nuc_mass' value -> (mass in amu, particle name or None). A name from
    PARTICLE_ALIASES, or a number in `default_unit` ('me' or 'amu') with an optional unit ('me', 'amu',
    'u', 'da'). Raises ValueError with the reason."""
    s = str(val).strip().lower()
    if s in NEGATIVE_PARTICLES:
        raise ValueError(f"'{val}' is a negatively charged particle: pyscf_neo_cc only treats positive "
                         f"particles next to the electrons (positrons, antimuons, H isotopes). For mu+ use "
                         f"'antimuon'; a negative muon exists here only inside muonic helium "
                         f"(PROTON sector, %nuc_mass muonic_helium)")
    if s in PARTICLE_ALIASES:
        name = PARTICLE_ALIASES[s]
        return PARTICLE_MASS_AMU[name], name
    m = re.match(r'^([0-9.eE+-]+)\s*(me|m_e|amu|u|da)?$', s)
    if not m:
        raise ValueError(f"expected a particle name ({', '.join(sorted(PARTICLE_MASS_AMU))}) or a mass, "
                         f"got {val!r}")
    try:
        x = float(m.group(1))
    except ValueError:
        raise ValueError(f"bad mass {val!r}")
    if x <= 0:
        raise ValueError(f"a mass must be positive, got {val!r}")
    unit = m.group(2) or default_unit
    return (x / AMU_TO_ME if unit in ("me", "m_e") else x), None


def nuclear_basis_dir():
    try:
        import pyscf_neo_cc._neo as _neo_pkg
        return os.path.join(os.path.dirname(_neo_pkg.__file__), "basis")
    except ImportError:
        return None


def nuclear_basis_file(name):
    """Path of the pyscf_neo_cc nuclear basis file for `name` (pb4d, PB5-F, pb6g ...), or None. Names are
    matched as the library does: as given, lower case, and without '-' / '_'."""
    d = nuclear_basis_dir()
    if d is None or not isinstance(name, str):
        return None
    for cand in (name, name.lower(), name.replace("-", "").replace("_", ""),
                 name.lower().replace("-", "").replace("_", "")):
        path = os.path.join(d, f"{cand}.dat")
        if cand and os.path.isfile(path):
            return path
    return None


def available_nuclear_bases():
    d = nuclear_basis_dir()
    if d is None or not os.path.isdir(d):
        return []
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".dat"))


def scale_basis(basis, ratio):
    """Copy of an uncontracted/contracted PySCF basis list with every exponent multiplied by `ratio`."""
    out = copy.deepcopy(basis)
    for shell in out:
        for prim in shell[1:]:
            if isinstance(prim, (list, tuple)):
                prim[0] *= ratio
    return out


def nuclear_basis_scale(mass_amu):
    return math.sqrt(mass_amu / PROTON_REF_AMU)


ETB_NAME = re.compile(r'^(\d+)s(\d+)p(\d+)d(?:(\d+)f)?$', re.I)


def is_nuclear_basis_name(name):
    return isinstance(name, str) and (nuclear_basis_file(name.strip()) is not None
                                      or ETB_NAME.match(name.strip()) is not None)


def load_nuclear_basis(name, mass_amu):
    """A protonic nuclear basis set scaled to a particle of `mass_amu` (None if `name` is not one): a
    pyscf_neo_cc basis file (pb4d ... pb6h, 8s8p8d8f ...) or an even-tempered 'NsNpNd' / 'NsNpNdNf' name,
    built as the library builds it for H (alpha = 2 sqrt 2, beta = sqrt 2)."""
    path = nuclear_basis_file(name)
    if path is not None:
        with open(path) as f:
            basis = gto.basis.parse(f.read())
    else:
        m = ETB_NAME.match(name.strip()) if isinstance(name, str) else None
        if m is None:
            return None
        alpha, beta = 2 * math.sqrt(2), math.sqrt(2)
        basis = gto.expand_etbs([(l, int(k), alpha, beta) for l, k in enumerate(m.groups()) if k])
    r = nuclear_basis_scale(mass_amu)
    return basis if r == 1.0 else scale_basis(basis, r)

RUN_TYPE_ALIASES = {
    "SP_ENERGY": "SP_ENERGY", "SP": "SP_ENERGY", "ENERGY": "SP_ENERGY",
    "GRADIENT": "GRADIENT", "GRAD": "GRADIENT", "SP_GRAD": "GRADIENT",
    "OPT": "OPT", "GEOMOPT": "OPT", "OPTIMIZATION": "OPT",
    "FREQ": "FREQ", "FREQUENCIES": "FREQ", "HESSIAN": "FREQ",      # v04.05
    "ANFREQ": "FREQ", "NUMFREQ": "FREQ",                          # v04.05: as ORCA's AnFreq / NumFreq
    "TS": "TS", "OPTTS": "TS",                                    # v04.05: TS / IRC / NEB, chosen with %ts_mode
    "SCAN": "SCAN",
}
# v04.05 (2026-09-30): '! TS' (or '! OptTS') covers the transition-state family; %ts_mode chooses the calculation
REMOVED_TS_WORDS = {"IRC": "irc", "NEB": "neb", "CI_NEB": "neb", "SADDLE": "ts", "TS_OPT": "ts"}
TS_MODES = {"TS": "TS", "PRFO": "TS", "SEARCH": "TS", "SADDLE": "TS", "OPTTS": "TS", "IRC": "IRC", "NEB": "NEB",
            "CI_NEB": "NEB", "CINEB": "NEB"}
# v04.05 (2026-09-30): the fit of a scan is chosen with %scan_fit, not with its own '!' word
REMOVED_SCAN_WORDS = {"MORSE_SCAN": "morse", "MORSESCAN": "morse", "HARMONIC_SCAN": "harmonic",
                      "HARM_SCAN": "harmonic", "HARMONICSCAN": "harmonic"}
SCAN_FIT_VALUES = {"MORSE": "morse", "HARMONIC": "harmonic", "HARM": "harmonic", "NONE": None, "FALSE": None,
                   "OFF": None, "NO": None}
MIN_FIT_POINTS = {"morse": 4, "harmonic": 3}
SCAN_PRINT_VALUES = {"eda", "none"}    # v04.08 %scan_print
# ! FREQ = analytic Hessian where one exists, else numerical; ! ANFREQ = analytic only; ! NUMFREQ = numerical always
FREQ_MODE = {"FREQ": "auto", "FREQUENCIES": "auto", "HESSIAN": "auto", "ANFREQ": "analytic", "NUMFREQ": "numerical"}
# v04.05: units of the coordinates in the geometry blocks ('! Bohrs' / '! Angs' or '%coords bohr|angstrom', as ORCA)
UNIT_ALIASES = {"ANGS": "ANGSTROM", "ANGSTROM": "ANGSTROM", "ANGSTROMS": "ANGSTROM", "ANG": "ANGSTROM", "A": "ANGSTROM",
                "BOHR": "BOHR", "BOHRS": "BOHR", "AU": "BOHR", "A.U.": "BOHR"}
SECTOR_ALIASES = {"POSITRON": "POSITRON", "POS": "POSITRON",
                  "PROTON": "PROTON", "NEO": "PROTON",
                  "ELECTRONIC": "ELECTRONIC", "ELEC": "ELECTRONIC"}
METHODS = {"HF", "DFT", "MP2", "CCSD", "CCSD(T)"}
ENGINE_ALIASES = {"AUTO": "AUTO", "WRAPPER": "WRAPPER", "WOPT": "WRAPPER", "BFGS": "WRAPPER",
                  "INTRINSIC": "INTRINSIC", "GEOMETRIC": "INTRINSIC", "ZOPT": "ZOPT", "ZMAT": "ZOPT"}
# v04.05: the optimizer is chosen with %opt_engine, not with its own '!' word (director's request, 2026-09-30)
REMOVED_OPT_WORDS = {"WOPT": "wopt", "WRAPPER_OPT": "wopt", "INTRINSIC_OPT": "geometric", "ZOPT": "zopt",
                     "ZMAT_OPT": "zopt"}
AXIS_INDEX = {"x": 0, "y": 1, "z": 2, "0": 0, "1": 1, "2": 2}


def is_ghost_label(sym):
    """PySCF ghost atom (basis functions, no nucleus, no electrons): ghost-H, Ghost_H, ghostH, X-H."""
    low = str(sym).strip().lower()
    return low.startswith("ghost") or low.startswith("x-")


def get_core_electrons(sym):
    if is_ghost_label(sym):
        return 0                                  # a ghost atom has no electrons, so no core
    sym_up = element_of(sym).upper()
    if sym_up in ['H', 'HE']:
        return 0
    elif sym_up in ['LI', 'BE', 'B', 'C', 'N', 'O', 'F', 'NE']:
        return 2
    elif sym_up in ['NA', 'MG', 'AL', 'SI', 'P', 'S', 'CL', 'AR']:
        return 10
    elif sym_up in ['K', 'CA', 'GA', 'GE', 'AS', 'SE', 'BR', 'KR']:
        return 18
    return 0


def resolve_frozen_core(frozen_val, atoms):
    if isinstance(frozen_val, int):
        return frozen_val
    if isinstance(frozen_val, str) and frozen_val.lower() == "auto":
        total_core = 0
        for sym, _ in atoms:
            total_core += get_core_electrons(sym)
        return total_core // 2
    return 0


def normalize_symbol(sym):
    """'li' -> 'Li', 'CL' -> 'Cl'; labels with digits or ghost prefixes are
    left alone apart from the element part's case."""
    s = str(sym).strip()
    if s.isalpha():
        return s[0].upper() + s[1:].lower()
    m = re.match(r'^([A-Za-z]{1,2})(\d+)$', s)
    if m:
        return m.group(1)[0].upper() + m.group(1)[1:].lower() + m.group(2)
    return s


def element_of(sym):
    """Pure element symbol of a geometry label ('C1' -> 'C', 'ghost-H' / 'ghost_H' / 'X-H' -> 'H')."""
    s = str(sym).strip()
    low = s.lower()
    if low.startswith('ghost'):
        s = s[5:].lstrip('-_')
    elif low.startswith('x-'):
        s = s[2:]
    s = re.sub(r'[\d+*#]+', '', s)                # digits; NEO isotope marks H+ / H* / H#
    return s[:1].upper() + s[1:].lower() if s else s


def resolve_pos_basis(spec, e_basis=None, mass_amu=None):
    """Positron basis name/data -> what build_system expects. A protonic nuclear basis name (pb4d ...
    pb6h, from pyscf_neo_cc) is loaded and scaled to the particle mass `mass_amu` (default: positron)."""
    if spec is None:
        return None
    if isinstance(spec, (list, tuple)):
        return [list(x) for x in spec]
    s = str(spec).strip()
    up = s.upper().replace("_", "-")
    if up.replace("-", "") in ("PSXDZ", "PSXTZ", "PSXQZ", "PSX5Z", "PSX6Z"):
        return PSX_SETS["PSX-" + up.replace("-", "")[3:]]
    nuc = load_nuclear_basis(s, PARTICLE_MASS_AMU["positron"] if mass_amu is None else mass_amu)
    if nuc is not None:
        return nuc
    if up in ("MATCH", "E-BASIS", "EBASIS"):
        return e_basis
    if up in ("NONE", "NULL", "OFF"):
        return None
    return s


def describe_pos_basis(spec, e_basis=None, mass_amu=None):
    data = resolve_pos_basis(spec, e_basis, mass_amu)
    if isinstance(data, list):
        counts = {}
        for shell in data:
            counts[shell[0]] = counts.get(shell[0], 0) + 1
        shells = "".join(f"{counts[l]}{'spdfghi'[l]}" for l in sorted(counts))
        extra = ""
        if is_nuclear_basis_name(spec):
            m = PARTICLE_MASS_AMU["positron"] if mass_amu is None else mass_amu
            extra = f", protonic set, exponents x {nuclear_basis_scale(m):.6f} for this mass"
        return f"{spec} = {shells} ({len(data)} shells{extra})"
    return f"{spec}" if data == spec else f"{spec} -> {data}"


class JobStage:
    def __init__(self, stage_id):
        self.stage_id = stage_id
        self.job_name = f"Stage_{stage_id}"
        self.run_type = "SP_ENERGY"
        self.run_type_explicit = False
        self.system_type = "ELECTRONIC"     # v04.05 default (was POSITRON, as in twoposBrorsen)
        self.sector_explicit = False       # POSITRON / PROTON / ELECTRONIC / CNEO / CDFT written on the '!' line
        self._keys_seen = set()            # directives written in this stage (sector check)
        self.method = "HF"                 # v04.05 default (was CCSD(T))
        self.use_df = True
        self.opt_engine = "AUTO"
        self.charge = 0
        self.multiplicity = 1
        self.elec_basis = "cc-pvdz"         # v04.05 default (was aug-cc-pvtz)
        self.elec_basis_groups = None       # v04.08: [(basis, [selectors])] from '%basis B1 H B2 {1 2} B3'
        self.frozen_core = 0
        self.cpu_cores = 4
        self.memory_mb = 4000
        self.verbose = 4
        self.verbose_explicit = False
        self.atoms = []
        self.has_explicit_xyz = False
        self.geometry_kind = None          # 'xyz' | 'zmat' | None (inherited)
        self.coord_units = "ANGSTROM"      # units of the geometry blocks as written (v04.05); stored in Angstrom
        self._unit_sources = {}
        self.zmat_rows = []
        self.zmat_vars = {}

        # optimizer controls
        self.constraints = []              # {'type','indices'(1-based),'target_value'}
        self.pinnings = []                 # whole atoms, 1-based   (%pin 1,2)
        self.equivalences = []             # bond pairs, 1-based    (%equivalence a,b,c,d)
        self.pinnings_xyz = []             # (atom0, axis, value)   (%pinnings block)
        self.equivalences_xyz = []         # (target0, source0, sx, sy, sz) (%equivalences block)
        self.opt_max_cycles = None
        self.conv_check = None
        self.molden = False            # v04.05: Molden files (HF orbitals, CC natural orbitals)
        self.cube = False              # v04.05: cube files of the HF and CC densities
        self.cube_points = 80
        self.chelpg = False            # v04.05: CHELPG charges at the final geometry (in the .out)
        self.density_maps = []         # v04.05: one map request per %density_map line (see parse_density_map)
        self.density_map_plane = "AUTO"
        self.density_map_combine = None  # v04.07: {'entries': [(job, species|None)], 'level', 'scale'} (one figure)
        self._keep_density = False     # v04.07: a later %density_map_combine needs this stage's densities
        self.freq_step = None          # v04.05 FREQ: finite-difference step (bohr); None = the route's default
        self.freq_mode = "auto"        # 'auto' (! FREQ) | 'analytic' (! ANFREQ) | 'numerical' (! NUMFREQ)
        self.ts_mode = "TS"            # v04.05 '! TS': TS (P-RFO search) | IRC | NEB (%ts_mode)
        self.ts_hessian = "auto"       # v04.05 TS / IRC: auto | first | each | never | stride:N | file:<path>
        self.ts_freq = True            # v04.05 TS: FREQ check at the transition state
        self.irc_direction = "both"    # v04.05 IRC: both | forward | backward
        self.neb_product = []          # v04.05 NEB: product geometry (%neb_product block), same atom order
        self.neb_images = 11           # v04.05 NEB: number of images (geomeTRIC's default)
        self.neb_optimize_ends = False # v04.05 NEB: optimize the two end points too (geomeTRIC's optep)
        # ZOPT (gradient-free Z-matrix optimizer)
        self.zopt = {"max_evals": 60, "etol": 1e-6, "step_bond": 0.02, "step_angle": 2.0,
                     "tol_bond": 1e-3, "tol_angle": 0.1}
        self.zopt_gradient_every = 0       # hybrid ZOPT: analytic gradient every N model steps (0 = off)
        self.zopt_final_gradient = False   # analytic gradient at the ZOPT result (convergence check)
        self.zopt_restart = True

        # positron sector
        self.pos_basis_name = "PSX-TZ"
        self.pos_basis_spec = {}           # element -> basis (from %pos_basis_map or dict literal)
        self.pos_atom_spec = None          # raw selectors; resolved against atoms at run time
        self.pos_basis_groups = None       # v04.08: [(basis, [selectors])] from '%pos_basis B1 H B2 {1 2}'
        self.n_pos = 1
        self.pos_mult = 1
        self.pos_spin = 0
        self.compute_eee = True
        self.compute_t2e1p = True
        self.mp2_partition = False      # v04.08: + a DF-MP2 row in the energy partition of a POSITRON CC stage
        self.pair_reference = "scf"     # v04.09 %pair_reference: two-positron DF reference model (scf | legacy)
        # particle mass (POSITRON sector): the "positron" may be any positive particle (antimuon,
        # proton, deuteron, triton); %quantum_site puts it in place of an H nucleus (1-based atom)
        self.pos_mass_amu = None           # None = a real positron
        self.pos_particle = None           # name when given by name
        self.quantum_site = None
        self.pos_aux = "etb"
        self.jkfit_aux_e = None            # None = the JK-fit set matched to %basis (resolved per molecule)
        self.df_aux = None
        self.auxbasis_e = None             # one-positron DF correlation aux (None = library default)
        self.aux_conditioning = None       # explicit, fixed whole-shell choices for this workflow
        self.df_mem_cap_gb = None
        self.block_size_a = None

        # scans
        self.scan = None
        self.scan_fit = None               # None (plain SCAN) | 'morse' | 'harmonic'
        self.scan_print = set()            # v04.08 %scan_print: extra per-point files ('eda')
        self.scan_range = None             # legacy (r_min, r_max, n) of the one-line %scan

        # proton sector
        self.nuc_basis = "pb4f1"
        self.quantum_nuc = [0]
        self.nuc_mass = None               # [(amu or None = library default, name), ...] per %quantum_nuc
        self.cneo = False                  # constrained NEO (%cneo true, or CNEO / CDFT on the ! line)
        self.cneo_variant = "ccd"          # correlated CNEO: ccd (constraint on the correlated density) or ucd
        self.xc = "b3lyp"
        self.epc = "17-2"

        self.zmat_from = None              # v04.05: stage whose Z-matrix a continuing stage uses (ZOPT chains)
        self.zmat_consistent = True        # its variables describe the inherited geometry (not after geomeTRIC/WOPT)
        self.inherited_from = None         # v04.05: stage whose settings this stage continues (no geometry block)
        self.warnings = []
        self.errors = []
        self._charge_sources = {}
        self._mult_sources = {}

    def warn(self, msg):
        self.warnings.append(msg)

    def error(self, msg):
        self.errors.append(msg)


def sanitize_epc(epc_str):
    """'%epc' value -> '17-1' / '17-2' / '18-1' / '18-2', None for no epc functional ('none', 'off'),
    False for an unknown value (a validation error; up to v04.04 it silently became 17-2)."""
    if not epc_str:
        return "17-2"
    s = epc_str.lower().strip().replace("epc", "").replace("_", "-")
    if s in ("none", "off", "no", "false", "0"):
        return None
    if s in ["17-1", "17-2", "18-1", "18-2"]:
        return s
    if s == "17":
        return "17-2"
    if s == "18":
        return "18-2"
    return False


# ----------------------------------------------------------------------------
# Geometry math
# ----------------------------------------------------------------------------

def compute_distance(c1, c2):
    return math.sqrt(sum((a - b)**2 for a, b in zip(c1, c2)))


def compute_angle(c1, c2, c3):
    v1 = [a - b for a, b in zip(c1, c2)]
    v2 = [c - b for c, b in zip(c3, c2)]
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a**2 for a in v1))
    mag2 = math.sqrt(sum(a**2 for a in v2))
    cos_t = max(-1.0, min(1.0, dot / (mag1 * mag2 + 1e-15)))
    return math.degrees(math.acos(cos_t))


def compute_dihedral(p1, p2, p3, p4):
    """IUPAC dihedral in degrees (same sign as geomeTRIC and Gaussian
    Z-matrices; the twopos/v03.17 version returned the negative)."""
    p1, p2, p3, p4 = (np.asarray(p, dtype=float) for p in (p1, p2, p3, p4))
    b1, b2, b3 = p2 - p1, p3 - p2, p4 - p3
    y = np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3))
    x = np.dot(np.cross(b1, b2), np.cross(b2, b3))
    return float(np.degrees(np.arctan2(y, x)))


def compute_constraint_value_and_gradient(coords, ctype, indices):
    """Value and Cartesian gradient of a bond (A) / angle (deg) / dihedral
    (deg) internal coordinate; indices are 0-based."""
    n_atoms = len(coords)
    grad = np.zeros((n_atoms, 3))
    if ctype in ("bond", "distance"):
        i, j = indices
        r_ij = coords[i] - coords[j]
        d = np.linalg.norm(r_ij)
        val = d
        if d > 1e-12:
            grad[i] = r_ij / d
            grad[j] = -r_ij / d
    elif ctype == "angle":
        i, j, k = indices
        u = coords[i] - coords[j]
        v = coords[k] - coords[j]
        lu = np.linalg.norm(u)
        lv = np.linalg.norm(v)
        cos_t = np.dot(u, v) / (lu * lv + 1e-15)
        cos_t = np.clip(cos_t, -1.0, 1.0)
        theta = np.arccos(cos_t)
        val = np.degrees(theta)
        sin_t = max(np.sqrt(1.0 - cos_t**2), 1e-8)
        g_i = (cos_t * (u / lu) - (v / lv)) / (lu * sin_t)
        g_k = (cos_t * (v / lv) - (u / lu)) / (lv * sin_t)
        g_j = -(g_i + g_k)
        grad[i] = np.degrees(g_i)
        grad[j] = np.degrees(g_j)
        grad[k] = np.degrees(g_k)
    elif ctype == "dihedral":
        # Blondel-Karplus gradient of the IUPAC dihedral (the twopos/v03.17
        # formula disagreed with finite differences).
        i, j, k, l = indices
        val = compute_dihedral(coords[i], coords[j], coords[k], coords[l])
        F = coords[i] - coords[j]
        G = coords[j] - coords[k]
        H = coords[l] - coords[k]
        A = np.cross(F, G)
        B = np.cross(H, G)
        nG, A2, B2 = np.linalg.norm(G), np.dot(A, A), np.dot(B, B)
        if A2 > 1e-12 and B2 > 1e-12 and nG > 1e-12:
            fg, hg = np.dot(F, G), np.dot(H, G)
            grad[i] = np.degrees(-nG / A2 * A)
            grad[l] = np.degrees(nG / B2 * B)
            grad[j] = np.degrees(nG / A2 * A + fg / (A2 * nG) * A - hg / (B2 * nG) * B)
            grad[k] = np.degrees(-nG / B2 * B - fg / (A2 * nG) * A + hg / (B2 * nG) * B)
    elif ctype == "bond_equal":
        # r(i,j) - r(k,l): %equivalence i,j,k,l keeps it at 0
        i, j, k, l = indices
        v1, g1 = compute_constraint_value_and_gradient(coords, "bond", [i, j])
        v2, g2 = compute_constraint_value_and_gradient(coords, "bond", [k, l])
        val, grad = v1 - v2, g1 - g2
    else:
        val = 0.0
    return val, grad


def _constraint_matrix(coords, active_constraints, mask):
    """Rows = constraint gradients; frozen coordinates (mask True) get zero
    columns, so corrections and projections never move them."""
    A = []
    for c in active_constraints:
        _, grad = compute_constraint_value_and_gradient(coords, c["type"], c["indices"])
        if mask is not None:
            grad = np.where(mask, 0.0, grad)
        A.append(grad.flatten())
    return np.array(A)


def restore_constraints(coords, active_constraints, max_iter=20, tol=1e-9, mask=None):
    """Gauss-Newton correction of coords back onto the constraint surface
    (coordinates where mask is True are never moved)."""
    n_atoms = len(coords)
    M = len(active_constraints)
    if M == 0:
        return coords
    curr_coords = coords.copy()
    for _ in range(max_iter):
        residuals = []
        for c in active_constraints:
            val, _ = compute_constraint_value_and_gradient(curr_coords, c["type"], c["indices"])
            resid = val - c["target_value"]
            if c["type"] == "dihedral":
                resid = (resid + 180.0) % 360.0 - 180.0
            residuals.append(resid)
        residuals = np.array(residuals)
        A = _constraint_matrix(curr_coords, active_constraints, mask)
        if np.max(np.abs(residuals)) < tol:
            break
        M_mat = np.dot(A, A.T) + 1e-12 * np.eye(M)
        try:
            dx_flat = -np.dot(A.T, np.linalg.solve(M_mat, residuals))
        except np.linalg.LinAlgError:
            lam, _, _, _ = np.linalg.lstsq(M_mat, residuals, rcond=1e-10)
            dx_flat = -np.dot(A.T, lam)
        curr_coords = curr_coords + dx_flat.reshape((n_atoms, 3))
    return curr_coords


def project_vector_null_space(coords, vector_flat, active_constraints, mask=None):
    """Remove from a flattened Cartesian vector every component that would
    change a constrained internal coordinate (Lagrange projection); frozen
    coordinates (mask True) are left untouched."""
    n_atoms = len(coords)
    M = len(active_constraints)
    if M == 0:
        return vector_flat.reshape((n_atoms, 3))
    A = _constraint_matrix(coords, active_constraints, mask)
    M_mat = np.dot(A, A.T) + 1e-12 * np.eye(M)
    rhs = np.dot(A, vector_flat)
    try:
        lam = np.linalg.solve(M_mat, rhs)
    except np.linalg.LinAlgError:
        lam, _, _, _ = np.linalg.lstsq(M_mat, rhs, rcond=1e-10)
    v_proj = vector_flat - np.dot(A.T, lam)
    return v_proj.reshape((n_atoms, 3))


def project_translation_rotation(coords, forces):
    """Remove rigid-body translation and rotation (equal masses)."""
    num_atoms = coords.shape[0]
    masses = np.ones(num_atoms)
    total_mass = np.sum(masses)
    center_of_mass = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

    F = forces.copy()
    F -= np.sum(F, axis=0, keepdims=True) / num_atoms

    I_tensor = np.zeros((3, 3))
    for i in range(num_atoms):
        r = coords[i] - center_of_mass
        I_tensor += np.dot(r, r) * np.eye(3) - np.outer(r, r)

    try:
        I_inv = np.linalg.inv(I_tensor)
    except np.linalg.LinAlgError:
        I_inv = np.linalg.pinv(I_tensor)

    omega = np.zeros(3)
    for i in range(num_atoms):
        r = coords[i] - center_of_mass
        torque = np.cross(r, F[i])
        omega += np.dot(I_inv, torque)

    for i in range(num_atoms):
        r = coords[i] - center_of_mass
        F[i] -= np.cross(omega, r) * masses[i]

    return F


def apply_xyz_pinnings_and_equivalences(coords, vec, stage):
    """Coordinate-level constraints from the %pinnings / %equivalences
    blocks (twopos / v5 semantics). coords: pinned values set and targets
    rebuilt from their source atoms; vec (gradient or step): pinned
    components zeroed and target components folded onto the source.
    Either argument may be None."""
    c = None if coords is None else np.array(coords, dtype=float)
    v = None if vec is None else np.array(vec, dtype=float)
    for (at, ax, val) in stage.pinnings_xyz:
        if c is not None:
            c[at, ax] = val
        if v is not None:
            v[at, ax] = 0.0
    for (t, s, sx, sy, sz) in stage.equivalences_xyz:
        signs = np.array([sx, sy, sz])
        if c is not None:
            c[t] = c[s] * signs
        if v is not None:
            v[s] = v[s] + v[t] * signs
            v[t] = 0.0
    return c, v


def zero_pinned_atoms(vec, stage):
    for p in stage.pinnings:
        vec[p - 1, :] = 0.0
    return vec


def zmatrix_to_cartesian(rows, variables=None):
    """Z-matrix -> (symbols, coords in A), dummy atoms (X, Du) removed.

    rows: list of token lists; row i = [label, (ref, r, (ref, angle, (ref, dihedral)))].
    References are 1-based row numbers or earlier row labels. Values are
    numbers, variable names, or '-name'. Atom 1 at the origin, atom 2 on +z,
    atom 3 in the xz-plane (positive x); angles and dihedrals in degrees."""
    variables = variables or {}
    labels, symbols = [], []
    coords = np.zeros((len(rows), 3))

    def value(tok):
        t = str(tok)
        try:
            return float(t)
        except ValueError:
            neg = t.startswith('-')
            name = t[1:] if neg else t
            if name not in variables:
                raise ValueError(f"Z-matrix variable '{name}' has no value "
                                 f"(define it as '{name} = <value>' inside the zmat block, "
                                 f"or scan it with %scan sym_zmat)")
            return -float(variables[name]) if neg else float(variables[name])

    def ref(tok, i):
        t = str(tok)
        if t.lstrip('-').isdigit():
            k = int(t) - 1
        elif t in labels:
            k = labels.index(t)
        else:
            raise ValueError(f"Z-matrix row {i+1}: unknown reference '{t}'")
        if not 0 <= k < i:
            raise ValueError(f"Z-matrix row {i+1}: reference '{t}' must be an earlier row")
        return k

    for i, row in enumerate(rows):
        need = {0: 1, 1: 3, 2: 5}.get(i, 7)
        if len(row) < need:
            raise ValueError(f"Z-matrix row {i+1} ({' '.join(row)}) needs {need} fields, got {len(row)}")
        labels.append(row[0])
        symbols.append(normalize_symbol(re.sub(r'\d+$', '', row[0]) or row[0]))
        if i == 0:
            coords[0] = 0.0
            continue
        a = ref(row[1], i)
        r = value(row[2])
        if i == 1:
            coords[1] = coords[a] + np.array([0.0, 0.0, r])
            continue
        b = ref(row[3], i)
        theta = np.radians(value(row[4]))
        u = coords[b] - coords[a]
        u = u / (np.linalg.norm(u) + 1e-15)
        if i == 2:
            w = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], u) * u
            if np.linalg.norm(w) < 1e-8:
                w = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], u) * u
            w = w / np.linalg.norm(w)
            coords[2] = coords[a] + r * (np.cos(theta) * u + np.sin(theta) * w)
            continue
        c = ref(row[5], i)
        phi = np.radians(value(row[6]))
        # NeRF: bond to a, angle (new, a, b) = theta, dihedral (new, a, b, c) = phi
        bc = coords[a] - coords[b]
        bc = bc / (np.linalg.norm(bc) + 1e-15)
        n = np.cross(coords[b] - coords[c], bc)
        if np.linalg.norm(n) < 1e-8:
            trial = np.array([1.0, 0.0, 0.0]) if abs(bc[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            n = np.cross(trial, bc)
        n = n / np.linalg.norm(n)
        m = np.cross(n, bc)
        d2 = np.array([-r * np.cos(theta), r * np.sin(theta) * np.cos(phi), r * np.sin(theta) * np.sin(phi)])
        coords[i] = coords[a] + d2[0] * bc + d2[1] * m + d2[2] * n
    # dummy atoms (X, X1, Du...) only serve as references: drop them from the result
    keep = [k for k, lab in enumerate(labels) if not is_dummy_label(lab)]
    return [symbols[k] for k in keep], coords[keep]


def is_dummy_label(label):
    """Z-matrix dummy atom: X, XX, Du, Dum, Dummy, optionally with digits (X1).
    'Xe' (xenon) and ghost labels such as 'X-H' are real atoms."""
    return re.sub(r'\d+$', '', str(label)).upper() in ("X", "XX", "DU", "DUM", "DUMMY")


# --decimal-comma: write numbers in the report/table files with a decimal
# comma (spreadsheets in Spanish locale). .xyz files and the .out log keep dots.
DECIMAL_COMMA = False


def num(x, spec):
    """format(x, spec), with a decimal comma when DECIMAL_COMMA is on."""
    s = format(x, spec)
    return s.replace(".", ",") if DECIMAL_COMMA else s


def get_internal_coordinates_lines(atoms, for_file=False):
    fmt = num if for_file else format
    lines = []
    natm = len(atoms)
    for i in range(natm):
        for j in range(i + 1, natm):
            sym1, c1 = atoms[i]
            sym2, c2 = atoms[j]
            r = compute_distance(c1, c2)
            lines.append(f"  Distance R({sym1}{i+1} - {sym2}{j+1}) : {fmt(r, '.6f')} A")

    for j in range(natm):
        for i in range(natm):
            for k in range(j + 1, natm):
                if i != j and i != k:
                    ang = compute_angle(atoms[j][1], atoms[i][1], atoms[k][1])
                    lines.append(f"  Angle A({atoms[j][0]}{j+1} - {atoms[i][0]}{i+1} - {atoms[k][0]}{k+1}) : {fmt(ang, '.4f')} deg")
    return lines


def print_internal_profile(atoms):
    print("\n" + "-" * 35 + " Internal Coordinates " + "-" * 35)
    for l in get_internal_coordinates_lines(atoms):
        print(l)
    print("-" * 92 + "\n")


def print_cartesian(atoms, title="Cartesian Coordinates (Angstrom):"):
    print(title)
    for sym, c in atoms:
        print(f"  {sym:<3} {c[0]:14.8f} {c[1]:14.8f} {c[2]:14.8f}")


def write_stage_artifacts(stage, final_atoms, e_df, e_conv=None, comps_df=None, comps_conv=None):
    job_prefix = stage.job_name if stage.job_name else f"Stage_{stage.stage_id}"
    xyz_filename = f"{job_prefix}_final.xyz"
    with open(xyz_filename, "w") as f:
        f.write(f"{len(final_atoms)}\n")
        f.write(f"Job: {stage.job_name} | Method: {stage.method} | Sector: {stage.system_type}\n")
        for sym, c in final_atoms:
            f.write(f"{sym:<3} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f}\n")
    print(f"[Artifact Generated] Final Geometry XYZ: {xyz_filename}")

    profile_filename = f"{job_prefix}_profile.txt"
    with open(profile_filename, "w") as f:
        f.write(f"Internal Coordinate Profile: {stage.job_name}\n")
        f.write("=" * 60 + "\n")
        for l in get_internal_coordinates_lines(final_atoms, for_file=True):
            f.write(l + "\n")
    print(f"[Artifact Generated] Internal Profile: {profile_filename}")

    summary_filename = f"{job_prefix}_summary.txt"
    with open(summary_filename, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"newBRORSEN OPTIMIZATION SUMMARY: {stage.job_name}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Wrapper Version:  newBRORSEN v{VERSION}\n")
        f.write(f"Sector:           {stage.system_type}\n")
        f.write(f"Method:           {stage.method}\n")
        f.write(f"Optimizer Engine: {engine_label(stage)}\n")
        f.write(f"Initial Basis:    {stage.elec_basis}\n")
        if stage.system_type == "POSITRON":
            f.write(f"Positron Basis:   {stage.pos_basis_name}\n")
            f.write(f"Positrons:        {stage.n_pos} (pos_spin={stage.pos_spin})\n")
            if describe_particle(stage):
                f.write(f"Particle:         {describe_particle(stage)}\n")
            f.write(f"Frozen orbitals:  {resolve_frozen_core(stage.frozen_core, final_atoms)}\n")
        elif stage.system_type == "PROTON":
            f.write(f"Nuclear Basis:    {stage.nuc_basis} | Centers: {stage.quantum_nuc}"
                    f"{' | CNEO (' + stage.cneo_variant + ')' if stage.cneo else ' | plain NEO'}\n")
            if describe_nuc_masses(stage):
                f.write(f"Nuclear Masses:   {describe_nuc_masses(stage)}\n")
            f.write(f"XC Functional:    {stage.xc} | EPC: {stage.epc}\n")
        elif stage.system_type == "ELECTRONIC" and stage.method == "DFT":
            f.write(f"XC Functional:    {stage.xc}\n")
        f.write("-" * 80 + "\n")
        if e_df is not None:
            f.write(f"Optimized Energy (DF/RI):          {num(e_df, '18.10f')} Ha\n")
        if e_conv is not None:
            f.write(f"Final Conventional Energy (Exact): {num(e_conv, '18.10f')} Ha\n")
            if e_df is not None:
                f.write(f"DF Correction Difference (Exact-DF): {num(e_conv - e_df, '18.10e')} Ha\n")
        for comps, tag in ((comps_df, "DF/RI"), (comps_conv, "Exact")):
            if comps and comps.get("partition"):
                f.write(f"  Energy partition ({tag}):\n")
                for line in partition_lines(comps["partition"], num):
                    f.write("    " + line + "\n")
            if not comps or stage.method == "HF":
                continue
            f.write(f"  Components ({tag}):\n")
            f.write(f"    HF (mean field):               {num(comps['HF'], '18.10f')} Ha\n")
            for key in ("MP2", "CCSD"):
                if comps.get(key) is not None:
                    f.write(f"    {key + ':':<31}{num(comps[key], '18.10f')} Ha\n")
            if comps.get("(T)") is not None:
                f.write(f"    (T) correction:                {num(comps['(T)'], '18.10f')} Ha\n")
                f.write(f"    CCSD(T):                       {num(comps['CCSD(T)'], '18.10f')} Ha\n")
        f.write("-" * 80 + "\n")
        f.write("Final Cartesian Coordinates (Angstrom):\n")
        for sym, c in final_atoms:
            f.write(f"  {sym:<2} {num(c[0], '14.8f')} {num(c[1], '14.8f')} {num(c[2], '14.8f')}\n")
        f.write("=" * 80 + "\n")
    print(f"[Artifact Generated] Execution Summary: {summary_filename}")


# ----------------------------------------------------------------------------
# Input parsing (both syntaxes)
# ----------------------------------------------------------------------------

_DIRECTIVE_RE = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*|\s+|$)(.*)$')
_BLOCK_SPLIT_RE = re.compile(r'^\s*%(next|newjob)\b', re.IGNORECASE)
_TRUE = {"true", "yes", "1", "on", "t", "y"}
_FALSE = {"false", "no", "0", "off", "f", "n"}


def strip_inline_comment(line):
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


def new_map_request():
    """One %density_map line: what ('electron' | 'positron' | 'both' | 'esp' | 'default'), its colour ranges as
    given, the limits per map resolved in validation, the colour scale and the vertical label spacing."""
    import types
    return types.SimpleNamespace(density_map=None, density_map_ranges=[], density_map_limits={},
                                 density_map_colors="rainbow_bw", density_map_ystep=None)


def map_request_view(stage, req):
    """The stage seen with one map request's settings (what the map functions read)."""
    view = copy.copy(stage)
    for k, v in vars(req).items():
        setattr(view, k, v)
    return view


def parse_density_map(val, stage):
    """%density_map <what> [lo,hi | auto] [lo,hi | auto] [colours] [ystep=<bohr>]
    what: electron, positron, both, esp (the electrostatic potential of the nuclei and the electrons), true
    (electron for ELECTRONIC, both for POSITRON) or false. Ranges: colour-scale limits in a.u. (with both: the
    electron range, then the positron one; auto = the default). Colours: a Multiwfn colour transition, by its number
    0-19 or its name (_MAP_SCALES); default rainbow_bw, blue_white_red for esp. ystep: spacing of the labels on the
    vertical axis, in bohr (anywhere after <what>).
    Every line adds one request, each with its own options, so a stage can have density maps and an ESP map;
    %density_map false removes the requests given before it in the stage."""
    toks = re.sub(r"\s*,\s*", ",", val.strip()).split()
    what = toks[0].lower() if toks else ""
    req = new_map_request()
    if what in ("electron", "electrons", "positron", "positrons", "both", "esp"):
        req.density_map = what.rstrip("s") if what not in ("both", "esp") else what
    elif what in _FALSE:
        stage.density_maps = []
        return
    elif what in _TRUE:
        req.density_map = "default"      # electron (ELECTRONIC) or both (POSITRON), set in validation
    else:
        stage.error(f"%density_map must start with electron, positron, both, esp, true or false, got {val!r}")
        return
    ranges, colors, ystep = _parse_map_options(toks[1:], stage, "density_map")
    req.density_map_ranges = ranges
    req.density_map_colors = colors or ("blue_white_red" if req.density_map == "esp" else "rainbow_bw")
    req.density_map_ystep = ystep
    stage.density_maps.append(req)


def parse_density_map_combine(val, stage):
    """%density_map_combine <job>[:electron|positron] <job>[:...] ... [level=hf|ccsd] [scale=shared|own] (v04.07)
    One figure with the maps of the named stages side by side, left to right in the order given, written after
    this stage (which may name itself). The species defaults to positron for a POSITRON stage and electron for an
    ELECTRONIC one (e.g. 0, 1 and 2 positrons: the electron density, then the two positron densities). level:
    ccsd (default; HF for a stage without CC density) or hf. scale: shared (default; one colour scale per
    species over the figure) or own (each panel its own default scale). false removes it."""
    toks = val.split()
    if len(toks) == 1 and toks[0].lower() in _FALSE:
        stage.density_map_combine = None
        return
    req = {"entries": [], "level": "ccsd", "scale": "shared"}
    for t in toks:
        low = t.lower()
        if "=" in low:
            k, v = low.split("=", 1)
            if k == "level" and v in ("hf", "ccsd"):
                req["level"] = v
            elif k == "scale" and v in ("shared", "own"):
                req["scale"] = v
            else:
                stage.error(f"%density_map_combine: unknown option {t!r} (level=hf|ccsd, scale=shared|own)")
            continue
        job, _, sp = t.partition(":")
        sp = sp.lower().rstrip("s") or None
        if not job or sp not in (None, "electron", "positron"):
            stage.error(f"%density_map_combine: {t!r} must be a %jobname, optionally :electron or :positron")
            continue
        req["entries"].append((job, sp))
    if len(req["entries"]) < 2:
        stage.error("%density_map_combine: name at least two stages (their %jobname), e.g. "
                    "%density_map_combine X_0pos X_1pos X_2pos")
    stage.density_map_combine = req


def _validate_map_combine(stages):
    """Every %density_map_combine names this stage or an earlier one, whose densities are then kept (the stage
    runs the final-geometry export even without %density_map / %molden / %cube)."""
    for i, s in enumerate(stages):
        req = s.density_map_combine
        if not req:
            continue
        earlier = {}
        for t in stages[:i + 1]:
            if not getattr(t, "freq_of_opt", None):
                earlier[t.job_name] = t
        resolved = []
        for job, sp in req["entries"]:
            t = earlier.get(job)
            if t is None:
                later = any(u.job_name == job for u in stages[i + 1:])
                s.error(f"%density_map_combine: no stage named {job!r} {'before' if later else 'in the input up to'} "
                        f"this one{' (it runs later: put the line in the last stage)' if later else ''}")
                continue
            sp = sp or ("positron" if t.system_type == "POSITRON" else "electron")
            if t.system_type not in ("ELECTRONIC", "POSITRON"):
                s.error(f"%density_map_combine: stage {job!r} is {t.system_type} (ELECTRONIC and POSITRON only)")
            elif sp == "positron" and t.system_type != "POSITRON":
                s.error(f"%density_map_combine: stage {job!r} is ELECTRONIC and has no positron density")
            elif t.run_type not in ("SP_ENERGY", "GRADIENT", "OPT", "ZOPT", "FREQ", "TS"):
                s.error(f"%density_map_combine: stage {job!r} is a {t.run_type} stage (no single final geometry)")
            t._keep_density = True
            resolved.append((job, sp))
        req["entries"] = resolved


def _parse_map_options(toks, stage, key):
    """The options after the first word of %density_map: colour ranges (lower,upper or auto), then
    a colour scale (Multiwfn number 0-19 or name), and ystep=<bohr> anywhere. Returns (ranges, colors, ystep)."""
    ranges, colors, ystep = [], None, None
    for t in toks:
        low = t.lower().replace("-", "_")
        if low.startswith("ystep="):
            try:
                ystep = float(low.split("=", 1)[1])
                if not ystep > 0:
                    raise ValueError
            except ValueError:
                ystep = None
                stage.error(f"%{key}: ystep is the spacing of the vertical-axis labels, a positive number of "
                            f"bohr (e.g. ystep=2), got {t!r}")
            continue
        if colors is not None and (low == "auto" or "," in t):
            stage.error(f"%{key}: put the colour scale last, after the ranges")
            continue
        if low == "auto":
            ranges.append(None)
        elif "," in t:
            try:
                lo, hi = (float(x) for x in t.split(","))
            except ValueError:
                stage.error(f"%{key}: a colour range is two numbers, lower,upper (e.g. 0,0.5), got {t!r}")
                continue
            if not hi > lo:
                stage.error(f"%{key}: the upper limit must be above the lower one, got {t!r}")
            ranges.append((lo, hi))
        elif colors is None and (low in _MAP_SCALES or (low.isdigit() and int(low) < len(_MAP_SCALES))):
            colors = list(_MAP_SCALES)[int(low)] if low.isdigit() else low
        else:
            stage.error(f"%{key}: {t!r} is neither a colour range (lower,upper or auto) nor a colour scale "
                        f"(0-19 or one of: {', '.join(_MAP_SCALES)})")
    return ranges, colors, ystep


def parse_bool(val, name, stage):
    v = str(val).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    stage.error(f"%{name} expects true/false, got {val!r}")
    return None


def parse_int(val, name, stage):
    try:
        return int(str(val).strip())
    except ValueError:
        stage.error(f"%{name} expects an integer, got {val!r}")
        return None


def parse_literal(val):
    try:
        return ast.literal_eval(val)
    except Exception:
        return val


def split_blocks(text):
    blocks, current = [], []
    for line in text.splitlines():
        if _BLOCK_SPLIT_RE.match(line):
            if current:
                blocks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return [b for b in blocks if any(strip_inline_comment(l).strip() for l in b)]


def _set_charge(stage, value, source):
    stage._charge_sources[source] = value
    stage.charge = value


def _set_mult(stage, value, source):
    stage._mult_sources[source] = value
    stage.multiplicity = value


def _bang_group(up):
    if up in RUN_TYPE_ALIASES or up in REMOVED_OPT_WORDS or up in REMOVED_SCAN_WORDS or up in REMOVED_TS_WORDS:
        return "run type"
    if up in SECTOR_ALIASES or up in ("CNEO", "CDFT"):
        return "sector"
    if up in METHODS:
        return "method"
    if up in ("DF", "RI", "CONV", "EXACT"):
        return "integrals"
    if up in UNIT_ALIASES:
        return "units"
    return None


def _parse_bang(stage, line):
    seen = stage.__dict__.setdefault("_bang_seen", {})
    for t in line[1:].split():
        up = t.upper()
        group = _bang_group(up)
        pair = {RUN_TYPE_ALIASES.get(seen.get(group, ""), ""), RUN_TYPE_ALIASES.get(up, "")} if group == "run type" else set()
        if pair in ({"OPT", "FREQ"}, {"TS", "FREQ"}):
            # ORCA's '! Opt Freq': the optimization, then frequencies at its result (a second stage, made in
            # parse_input_file). '! OptTS Freq': the TS stage's own frequency check (%ts_freq, on by default).
            freq_word = up if up in FREQ_MODE else seen[group]
            stage.then_freq = FREQ_MODE[freq_word]
            stage.run_type = RUN_TYPE_ALIASES[seen[group] if up in FREQ_MODE else up]
            stage.run_type_explicit = True
            if stage.run_type == "TS":
                stage.ts_freq, stage.then_freq = True, None
            seen[group] = stage.run_type
            continue
        if group and group in seen and seen[group] != up:
            if group == "run type":
                stage.error(f"'! {seen[group]} {t}': one run type per stage. For one calculation after the other use "
                            f"stages, e.g. '! {seen[group]}', then '%next' and '! {t}' (a stage without a geometry "
                            f"block continues the one before)")
            else:
                stage.error(f"two {group} words on the '!' line ('{seen[group]}' and '{t}'): write one")
        if group:
            seen.setdefault(group, up)
        if up in RUN_TYPE_ALIASES:
            stage.run_type = RUN_TYPE_ALIASES[up]
            stage.run_type_explicit = True
            if up in FREQ_MODE:
                stage.freq_mode = FREQ_MODE[up]
        elif up in REMOVED_TS_WORDS:
            stage.run_type, stage.run_type_explicit = "TS", True
            stage.error(f"'! {t}' was removed in v04.05: write '! TS' (or '! OptTS') and choose the calculation with "
                        f"'%ts_mode {REMOVED_TS_WORDS[up]}'")
        elif up in REMOVED_SCAN_WORDS:
            stage.run_type, stage.run_type_explicit = "SCAN", True
            stage.error(f"'! {t}' was removed in v04.05: write '! SCAN' and choose the fit with "
                        f"'%scan_fit {REMOVED_SCAN_WORDS[up]}'")
        elif up in REMOVED_OPT_WORDS:
            stage.run_type, stage.run_type_explicit = "OPT", True
            stage.error(f"'! {t}' was removed in v04.05: write '! OPT' and choose the optimizer with "
                        f"'%opt_engine {REMOVED_OPT_WORDS[up]}'")
        elif up in SECTOR_ALIASES:
            stage.system_type = SECTOR_ALIASES[up]
            stage.sector_explicit = True
        elif up == "CNEO":                 # constrained NEO: the PROTON sector with %cneo true
            stage.system_type, stage.cneo, stage.sector_explicit = "PROTON", True, True
        elif up == "CDFT":                 # constrained NEO-DFT (the library's class name)
            stage.system_type, stage.cneo, stage.method, stage.sector_explicit = "PROTON", True, "DFT", True
        elif up in UNIT_ALIASES and up != "A":
            stage.coord_units = UNIT_ALIASES[up]
            stage._unit_sources["! " + t] = stage.coord_units
        elif up in ("DF", "RI"):
            stage.use_df = True
        elif up in ("CONV", "EXACT"):
            stage.use_df = False
        elif up in METHODS:
            stage.method = up
        else:
            stage.error(f"unrecognized word '{t}' on the '!' line: check its spelling in MANUAL.md section 4 (v04.05: an "
                        f"error, so a misspelled method or sector never silently falls back to the default)")


SCAN_ATOM_COUNT = {"bond": 2, "distance": 2, "angle": 3, "dihedral": 4}


def _scan_head(stage, toks):
    """v04.08: the coordinate part of a '%scan <coordinate> ...' line: (cfg, tokens used) or (None, 0).
    bond/distance i j, angle i j k, dihedral i j k l (1-based atoms, Cartesian), or a Z-matrix variable
    by name (also 'sym_zmat name', the older spelling)."""
    head = toks[0].lower()
    if head in SCAN_ATOM_COUNT:
        n = SCAN_ATOM_COUNT[head]
        try:
            atoms = tuple(int(t) for t in toks[1:1 + n])
        except ValueError:
            atoms = ()
        if len(atoms) != n:
            stage.error(f"%scan {head}: needs {n} atom numbers after '{head}', got '{' '.join(toks)}'")
            return None, 0
        return {"mode": "bond" if head == "distance" else head, "atoms": atoms}, 1 + n
    if head == "sym_zmat" and len(toks) >= 2:
        return {"mode": "sym_zmat", "var": toks[1]}, 2
    if head not in ("translate", "fragment") and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", toks[0]):
        return {"mode": "sym_zmat", "var": toks[0]}, 1
    return None, 0


def _scan_add_values(stage, cfg, vals, segment=None):
    """Append scan points; a segment that starts where the previous one ended does not repeat that point."""
    have = cfg.setdefault("values", [])
    vals = [float(v) for v in vals]
    if have and vals and abs(vals[0] - have[-1]) < 1e-10:
        vals = vals[1:]
    have.extend(vals)
    if segment is not None:
        cfg.setdefault("segments", []).append(segment)
    if have:
        cfg["start"], cfg["end"], cfg["steps"] = have[0], have[-1], len(have)


def _scan_segment(stage, cfg, toks):
    """'start end n' (n points, both ends included) or '[v1 v2 ...]' (the listed values) into cfg."""
    text = " ".join(toks)
    if "[" in text or "]" in text:
        inner = text.strip()
        if not (inner.startswith("[") and inner.endswith("]")):
            stage.error(f"%scan: a list of values goes in one pair of brackets, got '{text}'")
            return
        try:
            vals = [float(t) for t in inner[1:-1].replace(",", " ").split()]
        except ValueError:
            stage.error(f"%scan: bad number in '{text}'")
            return
        if not vals:
            stage.error("%scan: empty list of values")
            return
        cfg["custom"] = True
        _scan_add_values(stage, cfg, vals)
        return
    nums = text.replace(",", " ").split()
    if len(nums) != 3:
        stage.error(f"%scan: expected 'start end n_points' or '[v1 v2 ...]', got '{text}'")
        return
    try:
        a, b, n = float(nums[0]), float(nums[1]), int(nums[2])
    except ValueError:
        stage.error(f"%scan: bad number in '{text}' (start end n_points; n_points an integer)")
        return
    if n < 2:
        stage.error(f"%scan: each segment needs at least 2 points, got {n} in '{text}'")
        return
    _scan_add_values(stage, cfg, np.linspace(a, b, n), segment=(a, b, n))


def _parse_scan_line(stage, val):
    """v04.08 '%scan <coordinate> ...' on the directive line. Returns 'scan' when a block of segments follows
    (nothing after the coordinate), else None."""
    toks = val.split()
    n_err = len(stage.errors)
    cfg, used = _scan_head(stage, toks)
    if cfg is None:
        if len(stage.errors) == n_err:
            _parse_scan_row(stage, toks)          # translate ... / fragment ... on one line
        return None
    if stage.scan is not None:
        stage.error("%scan: only one scan per stage (split scans into %next stages)")
        return None
    cfg["header"] = True
    stage.scan = cfg
    rest = toks[used:]
    if not rest:
        return "scan"
    _scan_segment(stage, cfg, rest)
    return None


def _parse_scan_row(stage, toks):
    head = toks[0].lower()
    try:
        if head == "sym_zmat" and len(toks) >= 5:
            cfg = {"mode": "sym_zmat", "var": toks[1], "start": float(toks[2]),
                   "end": float(toks[3]), "steps": int(toks[4])}
        elif head in ("bond", "distance") and len(toks) >= 6:
            cfg = {"mode": "bond", "atoms": (int(toks[1]), int(toks[2])),
                   "start": float(toks[3]), "end": float(toks[4]), "steps": int(toks[5])}
        elif head.lstrip('-').isdigit() and len(toks) >= 5:
            cfg = {"mode": "bond", "atoms": (int(toks[0]), int(toks[1])),
                   "start": float(toks[2]), "end": float(toks[3]), "steps": int(toks[4])}
        elif head == "translate" and len(toks) >= 6:
            cfg = {"mode": "translate", "range": (int(toks[1]), int(toks[2])),
                   "axis": toks[3].lower(), "start": float(toks[4]), "end": float(toks[5]),
                   "steps": int(toks[6]) if len(toks) >= 7 else 5}
        elif head == "fragment" and len(toks) >= 4:
            cfg = {"mode": "fragment", "start": float(toks[1]), "end": float(toks[2]),
                   "steps": int(toks[3])}
        else:
            stage.error(f"%scan: cannot read line '{' '.join(toks)}'")
            return
    except ValueError as err:
        stage.error(f"%scan: bad number in '{' '.join(toks)}' ({err})")
        return
    if stage.scan is not None:
        stage.error("%scan: only one scan per stage (split scans into %next stages)")
        return
    stage.scan = cfg


def _atom_tokens_have_zero(mode, toks):
    try:
        if mode == "constraints":
            n = {"bond": 2, "distance": 2, "angle": 3, "dihedral": 4}.get(toks[0].lower(), 0)
            return any(int(x) == 0 for x in toks[1:1 + n])
        if mode == "pinnings":
            return int(toks[0]) == 0
        if mode == "equivalences":
            return int(toks[0]) == 0 or int(toks[1]) == 0
    except (ValueError, IndexError):
        pass
    return False


def _close_block(stage, mode, start):
    """Early twoposBrorsen (00.03-00.16) numbered atoms from 0 in these
    blocks. A block that contains atom 0 cannot be 1-based, so it is read as
    0-based as a whole."""
    if mode not in ("constraints", "pinnings", "equivalences") or not stage._block_zero:
        return
    if mode == "constraints":
        for c in stage.constraints[start:]:
            c["indices"] = [i + 1 for i in c["indices"]]
    elif mode == "pinnings":
        stage.pinnings_xyz[start:] = [(a + 1, ax, v) for a, ax, v in stage.pinnings_xyz[start:]]
    else:
        stage.equivalences_xyz[start:] = [(t + 1, s + 1, *rest) for t, s, *rest in stage.equivalences_xyz[start:]]
    stage.warn(f"%{mode} block uses atom 0, so its atom numbers were read as 0-based "
               f"(early twoposBrorsen convention); write 1-based numbers in new inputs")


def _block_length(stage, mode):
    return {"constraints": len(stage.constraints), "pinnings": len(stage.pinnings_xyz),
            "equivalences": len(stage.equivalences_xyz)}.get(mode, 0)


def _parse_block_row(stage, mode, line):
    toks = line.split()
    if _atom_tokens_have_zero(mode, toks):
        stage._block_zero = True
    if mode == "constraints":
        ctype = toks[0].lower()
        n = {"bond": 2, "distance": 2, "angle": 3, "dihedral": 4}.get(ctype)
        if n is None or len(toks) < 1 + n:
            stage.error(f"%constraints: cannot read '{line}' (bond i j [value] | angle i j k [value] | dihedral i j k l [value])")
            return
        try:
            idx = [int(x) for x in toks[1:1 + n]]
            target = float(toks[1 + n]) if len(toks) > 1 + n else None
        except ValueError:
            stage.error(f"%constraints: bad number in '{line}'")
            return
        stage.constraints.append({"type": "bond" if ctype == "distance" else ctype,
                                  "indices": idx, "target_value": target})
    elif mode == "pinnings":
        if len(toks) < 3 or toks[1].lower() not in AXIS_INDEX:
            stage.error(f"%pinnings: cannot read '{line}' (atom axis value; axis = 0/1/2 or x/y/z)")
            return
        try:
            stage.pinnings_xyz.append((int(toks[0]) - 1, AXIS_INDEX[toks[1].lower()], float(toks[2])))
        except ValueError:
            stage.error(f"%pinnings: bad number in '{line}'")
    elif mode == "equivalences":
        if len(toks) < 5:
            stage.error(f"%equivalences: cannot read '{line}' (target source sx sy sz)")
            return
        try:
            stage.equivalences_xyz.append((int(toks[0]) - 1, int(toks[1]) - 1,
                                           float(toks[2]), float(toks[3]), float(toks[4])))
        except ValueError:
            stage.error(f"%equivalences: bad number in '{line}'")
    elif mode == "scan":
        if stage.scan is not None and stage.scan.get("header"):
            _scan_segment(stage, stage.scan, toks)
        else:
            _parse_scan_row(stage, toks)
    elif mode == "neb_product":
        try:
            stage.neb_product.append((normalize_symbol(toks[0]), (float(toks[1]), float(toks[2]), float(toks[3]))))
        except (ValueError, IndexError):
            stage.error(f"%neb_product: cannot read '{line}' (symbol x y z, Angstrom)")
    elif mode == "pos_basis_map":
        if len(toks) >= 2:
            stage.pos_basis_spec[toks[0]] = toks[1]
        else:
            stage.error(f"%pos_basis_map: cannot read '{line}' (element basis)")


def _parse_geometry_row(stage, kind, line):
    toks = line.split()
    if kind == "xyz":
        if len(toks) < 4:
            stage.error(f"* xyz: cannot read atom line '{line}'")
            return
        try:
            stage.atoms.append((normalize_symbol(toks[0]),
                                (float(toks[1]), float(toks[2]), float(toks[3]))))
        except ValueError:
            if any(re.fullmatch(r"-?\d+,\d+", t) for t in toks[1:4]):
                stage.error(f"* xyz: decimal comma in the coordinates of '{line}': input numbers use a decimal "
                            f"point (a geometry copied from a --decimal-comma report file needs its commas "
                            f"turned into points; the _final.xyz files always use points)")
            else:
                stage.error(f"* xyz: bad coordinate in '{line}'")
    else:
        if line.rstrip(":").lower() in ("variables", "constants"):
            return
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\S+)$', line)
        if m:
            try:
                stage.zmat_vars[m.group(1)] = float(m.group(2))
            except ValueError:
                stage.error(f"* zmat: variable '{m.group(1)}' needs a number, got {m.group(2)!r}")
            return
        stage.zmat_rows.append([x for x in re.split(r"[\s,]+", line.strip()) if x])


_LIST_KEYS = {"constraints": ("constraints",), "pin": ("pinnings",), "pinning": ("pinnings",),
              "equivalence": ("equivalences",), "equiv": ("equivalences",), "pinnings": ("pinnings_xyz",),
              "equivalences": ("equivalences_xyz",), "pos_basis_map": ("pos_basis_spec",),
              "density_map": ("density_maps",)}

BLOCK_DIRECTIVES = {"constraints", "pinnings", "equivalences", "pos_basis_map", "neb_product"}


def _parse_directive(stage, line):
    m = _DIRECTIVE_RE.match(line[1:].strip())
    if not m:
        stage.error(f"unrecognized directive line {line!r}: fix or remove it (v04.05: an error, so a misspelled "
                    f"setting is never silently skipped)")
        return None
    key, val = m.group(1).lower(), m.group(2).strip()
    # v04.05: list-type settings written in a stage that continues another REPLACE the inherited ones (the first
    # time they appear in this stage); later lines in the same stage add to them, as before
    if key == "jobname" and "jobname" in stage._keys_seen:
        stage.error(f"a second %jobname in one stage ({stage.job_name!r}, then {val!r}): is a '%next' missing "
                    f"before it? Without it the two calculations are merged into one stage and the first is lost")
    if key == "pos_basis_map" and val:
        stage.error(f"%pos_basis_map is a block: write '%pos_basis_map' alone, then one 'element basis' line per "
                    f"element (e.g. 'H PSX-TZ'), then 'end'. To choose the atoms that carry positron functions "
                    f"use %pos_atom (e.g. %pos_atom {val}); got {line!r}")
        return None
    if stage.inherited_from and key in _LIST_KEYS and key not in stage._keys_seen:
        for attr in _LIST_KEYS[key]:
            setattr(stage, attr, type(getattr(stage, attr))())
    stage._keys_seen.add(key)

    if key in BLOCK_DIRECTIVES and not val:
        return key
    if key == "scan" and not val:
        return "scan"

    if key == "jobname":
        stage.job_name = val
    elif key == "charge":
        v = parse_int(val, key, stage)
        if v is not None:
            _set_charge(stage, v, "%charge")
    elif key in ("mult", "multiplicity"):
        v = parse_int(val, key, stage)
        if v is not None:
            _set_mult(stage, v, "%mult")
    elif key == "spin":
        v = parse_int(val, key, stage)
        if v is not None:
            _set_mult(stage, v + 1, "%spin")
    elif key == "memory":
        low = val.lower().replace(" ", "")
        try:
            if low.endswith("gb"):
                stage.memory_mb = int(float(low[:-2]) * 1024)
            else:
                stage.memory_mb = int(float(low.replace("mb", "")))
        except ValueError:
            stage.error(f"%memory expects MB (e.g. 16000 or 16GB), got {val!r}")
    elif key in ("nprocs", "cores", "threads"):
        v = parse_int(val, key, stage)
        if v is not None:
            stage.cpu_cores = v
    elif key in ("basis", "e_basis"):
        groups = parse_basis_groups(val, stage, key)
        if groups is None:
            pass
        elif len(groups) == 1 and not groups[0][1]:
            stage.elec_basis, stage.elec_basis_groups = groups[0][0], None
        else:
            stage.elec_basis_groups = groups
            stage.elec_basis = next((b for b, sel in groups if not sel), groups[0][0])
    elif key == "frozen":
        if val.lower() == "auto":
            stage.frozen_core = "auto"
        else:
            v = parse_int(val, key, stage)
            if v is not None:
                stage.frozen_core = v
    elif key == "verbose":
        v = parse_int(val, key, stage)
        if v is not None:
            stage.verbose, stage.verbose_explicit = v, True
    elif key in ("coords", "coord_units", "units"):
        words = [w for w in re.split(r"[\s=]+", val) if w and w.lower() not in ("units", "end")]
        unit = UNIT_ALIASES.get(words[0].upper()) if len(words) == 1 else None
        if unit is None:
            stage.error(f"%coords sets the units of the geometry blocks: '%coords bohr' or '%coords angstrom' "
                        f"(also: bohrs, au, angs), got {val!r}")
        else:
            stage.coord_units = unit
            stage._unit_sources["%" + key] = unit
    elif key in ("ts_mode", "ts_type", "ts_search"):
        if val.upper() not in TS_MODES:
            stage.error(f"%ts_mode must be ts (transition-state search, the default), irc or neb, got {val!r}")
        else:
            stage.ts_mode = TS_MODES[val.upper()]
            stage._ts_mode_written = True
    elif key == "scan_print":
        words = {w.lower() for w in val.replace(",", " ").split()}
        bad = words - SCAN_PRINT_VALUES
        if bad or not words:
            stage.error(f"%scan_print takes {', '.join(sorted(SCAN_PRINT_VALUES))} (or none), got {val!r}")
        else:
            stage.scan_print = words - {"none"}
    elif key in ("scan_fit", "fit"):
        if val.upper() not in SCAN_FIT_VALUES:
            stage.error(f"%scan_fit must be morse, harmonic or none, got {val!r}")
        else:
            stage.scan_fit = SCAN_FIT_VALUES[val.upper()]
            stage._scan_fit_written = True
    elif key in ("opt_engine", "optimizer"):
        eng = ENGINE_ALIASES.get(val.upper())
        if eng is None:
            stage.error(f"%opt_engine must be GEOMETRIC (AUTO, INTRINSIC; the default), WOPT (WRAPPER, BFGS) or ZOPT "
                        f"(Z-matrix, energy-only), got {val!r}")
        else:
            stage.opt_engine = eng
    elif key in ("opt_max_cycles", "max_opt_cycles", "max_steps"):
        stage.opt_max_cycles = parse_int(val, key, stage)
    elif key in ("zopt_max_evals", "zopt_etol", "zopt_step_bond", "zopt_step_angle",
                 "zopt_tol_bond", "zopt_tol_angle"):
        name = key[5:]
        try:
            v = int(val) if name == "max_evals" else float(val)
        except ValueError:
            stage.error(f"%{key} expects a number, got {val!r}")
            return None
        if v <= 0:
            stage.error(f"%{key} must be positive, got {val!r}")
            return None
        stage.zopt[name] = v
    elif key == "zopt_gradient_every":
        v = parse_int(val, key, stage)
        if v is not None:
            if v < 0:
                stage.error(f"%zopt_gradient_every must be 0 (off) or a positive number of steps, got {val!r}")
            else:
                stage.zopt_gradient_every = v
    elif key == "zopt_final_gradient":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.zopt_final_gradient = b
    elif key == "zopt_restart":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.zopt_restart = b
    elif key == "conv_check":
        stage.conv_check = parse_bool(val, key, stage)
    elif key == "molden":
        stage.molden = parse_bool(val, key, stage)
    elif key == "cube":
        stage.cube = parse_bool(val, key, stage)
    elif key == "cube_points":
        try:
            stage.cube_points = int(val)
            if not 10 <= stage.cube_points <= 400:
                raise ValueError
        except ValueError:
            stage.error(f"%cube_points must be a whole number of grid points per axis (10-400), got {val!r}")
    elif key == "chelpg":
        stage.chelpg = parse_bool(val, key, stage)
    elif key == "density_map":
        parse_density_map(val, stage)
    elif key == "density_map_combine":
        parse_density_map_combine(val, stage)
    elif key == "ts_hessian":
        v = val.strip()
        stage.ts_hessian = v if v.lower().startswith("file:") else v.lower()
    elif key == "ts_freq":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.ts_freq = b
    elif key == "neb_optimize_ends":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.neb_optimize_ends = b
    elif key == "neb_images":
        n = parse_int(val, key, stage)
        if n is not None:
            stage.neb_images = n
    elif key == "irc_direction":
        stage.irc_direction = val.strip().lower()
    elif key == "freq_step":
        try:
            stage.freq_step = float(val)
            if not 1e-4 <= stage.freq_step <= 0.05:
                raise ValueError
        except ValueError:
            stage.error(f"%freq_step is the finite-difference step in bohr (1e-4 to 0.05; default 0.005 with "
                        f"gradients, 0.01 with energies), got {val!r}")
    elif key == "density_map_plane":
        v = val.strip().upper()
        if v not in ("AUTO", "XY", "XZ", "YZ"):
            stage.error(f"%density_map_plane must be AUTO, XY, XZ or YZ, got {val!r}")
        else:
            stage.density_map_plane = v
    elif key == "constraints":
        stage.error("%constraints is a block: put its lines on the next lines and close with 'end'")
    elif key in ("pin", "pinning", "pinnings"):
        try:
            stage.pinnings.extend(int(x) for x in re.split(r'[,\s]+', val.strip("[]() ")) if x)
        except ValueError:
            stage.error(f"%{key} expects atom numbers (e.g. %pin 1,2), got {val!r}")
    elif key in ("equivalence", "equiv", "equivalences"):
        try:
            nums = [int(x) for x in re.split(r'[,\s]+', val.strip("[]() ")) if x]
        except ValueError:
            nums = []
        if len(nums) == 4:
            stage.equivalences.append(((nums[0], nums[1]), (nums[2], nums[3])))
        else:
            stage.error(f"%{key} expects 4 atom numbers a1,a2,b1,b2 (bond a1-a2 equivalent to b1-b2), got {val!r}")
    elif key == "scan" and not re.fullmatch(r"[-+.\d\s,eE]+", val):
        return _parse_scan_line(stage, val)
    elif key == "scan":
        parts = val.replace(",", " ").split()
        if len(parts) != 3:
            stage.error(f"%scan needs 3 values (r_min r_max n_points) or a %scan ... end block, got {val!r}")
        else:
            try:
                rng = (float(parts[0]), float(parts[1]), int(parts[2]))
            except ValueError:
                stage.error(f"%scan: bad number in {val!r}")
                return None
            stage.scan_range = rng
            if stage.scan is not None:
                stage.error("%scan: only one scan per stage")
            else:
                stage.scan = {"mode": "fragment", "start": rng[0], "end": rng[1],
                              "steps": rng[2], "one_line": True}
    # positron sector
    elif key in ("pos_basis", "positron_basis") and not (val[:1] == "[" or (val[:1] == "{" and ":" in val)):
        groups = parse_basis_groups(val, stage, key)
        if groups is None:
            pass
        elif len(groups) == 1 and not groups[0][1]:
            stage.pos_basis_name, stage.pos_basis_groups, stage.pos_basis_spec = groups[0][0], None, {}
        else:
            stage.pos_basis_groups, stage.pos_basis_spec = groups, {}
            stage.pos_basis_name = " ".join(val.split())
    elif key in ("pos_basis", "positron_basis"):
        stage.pos_basis_groups = None
        lit = parse_literal(val) if val[:1] in "[{(" else val
        if isinstance(lit, dict):
            for k, v in lit.items():
                stage.pos_basis_spec[str(k)] = v
            stage.pos_basis_name = "per-atom"
        elif isinstance(lit, (list, tuple)) and lit and all(isinstance(x, (list, tuple)) for x in lit):
            stage.pos_basis_name = [list(x) for x in lit]
        elif isinstance(lit, (list, tuple)):
            stage.pos_atom_spec = [str(x) for x in lit]
        else:
            stage.pos_basis_name = str(lit)
    elif key == "psx_set":
        stage.warn("%psx_set was removed: PSX-DZ / PSX-TZ / PSX-QZ are always the openLOWDIN sets -- ignored.")
    elif key == "pos_atom":
        lit = parse_literal(val) if val[:1] in "[(" else None
        if isinstance(lit, (list, tuple)):
            stage.pos_atom_spec = [str(x) for x in lit]
        else:
            stage.pos_atom_spec = val.replace(",", " ").split()
    elif key == "pos_aux":
        stage.pos_aux = val
    elif key in ("pos_mass", "particle_mass", "particle"):
        try:
            stage.pos_mass_amu, stage.pos_particle = parse_particle_mass(val, "me")
        except ValueError as err:
            stage.error(f"%{key}: {err}")
        if stage.pos_particle == "positron":
            stage.pos_mass_amu = stage.pos_particle = None
    elif key == "quantum_site":
        v = parse_int(val, key, stage)
        if v is not None:
            stage.quantum_site = v
    elif key == "nuc_mass":
        masses = []
        joined = re.sub(r'(\d)\s+(me|m_e|amu|u|da)\b', r'\1\2', val.strip("[]() "), flags=re.I)
        for tok in re.split(r'[,\s]+', joined):
            if not tok:
                continue
            if tok.lower() in ("default", "-", "table"):
                masses.append((None, None))
                continue
            try:
                masses.append(parse_particle_mass(tok, "amu"))
            except ValueError as err:
                stage.error(f"%nuc_mass: {err}")
        stage.nuc_mass = masses
    elif key == "jkfit_aux_e":
        stage.jkfit_aux_e = None if val.lower() in ("auto", "matched", "default", "") else val
    elif key == "n_pos":
        v = parse_int(val, key, stage)
        if v is not None:
            stage.n_pos = v
    elif key == "pos_mult":
        v = parse_int(val, key, stage)
        if v is not None:
            stage.pos_mult, stage.pos_spin = v, v - 1
    elif key == "pos_spin":
        v = parse_int(val, key, stage)
        if v is not None:
            stage.pos_spin, stage.pos_mult = v, v + 1
    elif key == "df_aux":
        lit = parse_literal(val)
        if isinstance(lit, dict):
            stage.df_aux = lit
        elif str(val).lower() in ("default", "{}", ""):
            stage.df_aux = {}
        else:
            stage.df_aux = {"auxbasis_e": str(val)}
    elif key == "auxbasis_e":
        stage.auxbasis_e = None if val.lower() in ("default", "none", "") else val
    elif key == "aux_conditioning":
        parse_aux_conditioning(stage, val)
    elif key == "df_mem_cap":
        low = val.lower().replace(" ", "").replace("gb", "")
        try:
            stage.df_mem_cap_gb = float(low)
        except ValueError:
            stage.error(f"%df_mem_cap expects a size in GB, got {val!r}")
    elif key == "block_size_a":
        stage.block_size_a = parse_int(val, key, stage)
    elif key == "mp2_partition":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.mp2_partition = b
    elif key == "pair_reference":
        v = val.strip().lower()
        if v in ("scf", "legacy"):
            stage.pair_reference = v
        else:
            stage.error(f"%pair_reference takes scf or legacy, got {val!r}")
    elif key == "compute_eee":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.compute_eee = b
    elif key == "compute_t2e1p":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.compute_t2e1p = b
    # proton sector
    elif key in ("nuc_basis", "proton_basis"):
        stage.nuc_basis = val
    elif key == "quantum_nuc":
        try:
            stage.quantum_nuc = [int(x) for x in re.split(r'[,\s]+', val.strip("[]() ")) if x]
        except ValueError:
            stage.error(f"%quantum_nuc expects atom indices, got {val!r}")
    elif key == "xc":
        stage.xc = val
    elif key == "epc":
        epc = sanitize_epc(val)
        if epc is False:
            stage.error(f"%epc must be 17-1, 17-2, 18-1, 18-2 (or 17, 18) or none, got {val!r}")
        else:
            stage.epc = epc
    elif key == "cneo":
        b = parse_bool(val, key, stage)
        if b is not None:
            stage.cneo = b
    elif key == "cneo_variant":
        v = val.strip().lower()
        if v not in ("ucd", "ccd"):
            stage.error(f"%cneo_variant must be ccd or ucd, got {val!r}")
        else:
            stage.cneo_variant = v
    else:
        stage.error(f"unrecognized directive '%{key}' (line: {line!r}): check its spelling in MANUAL.md section 5, "
                    f"or remove it (v04.05: an error; up to v04.04 it was ignored with a warning, so a misspelled "
                    f"setting such as '%conv_chek' was silently skipped)")
    return None


def _looks_cartesian(rows):
    def num(t):
        try:
            float(t)
            return True
        except ValueError:
            return False
    return len(rows) >= 2 and all(len(r) == 4 and all(num(t) for t in r[1:]) for r in rows) \
        and not rows[1][1].lstrip('-').isdigit()


def is_ri_set_name(auxbasis):
    """Same rule as pyscf_neo_cc (df/auxbasis.py, unchanged since 2026-09-23): a fitting-set
    name ending in 'ri' once dashes/underscores are removed (aug-cc-pvtz-ri,
    cc-pvdz-ri, def2-tzvp-ri) is an RI (correlation-fitting) set."""
    return isinstance(auxbasis, str) and auxbasis.lower().replace('-', '').replace('_', '').endswith('ri')


DEFAULT_JKFIT = "def2-universal-jkfit"

# v04.05: PySCF prints this whenever a named fitting set has no entry for an element (the cc-pVXZ RI / JK-fit
# families have none for Li, Be, Na, Mg, K). pyscf_neo_cc then completes the set per element (an even-tempered set),
# so the line repeated at every energy evaluation and looked like an error. The stage summary names the sets really
# used (single_positron_correlation_label) and validation warns once.
warnings.filterwarnings("ignore", message="Basis may be available in basis-set-exchange")


def single_positron_correlation_label(stage, syms):
    """(label, missing elements) of the correlation fitting set of a one-positron DF stage, as pyscf_neo_cc
    resolves it: %auxbasis_e, else '<basis>-ri' ('aug-cc-pvtz-ri' for a per-element %basis); 'etb' is an
    even-tempered set; an element the named set does not have gets an even-tempered set built from its orbital
    basis (df.aug_etb, beta 2.0)."""
    name = stage.auxbasis_e
    if isinstance(name, str) and name.lower() == "etb":
        return "even-tempered (etb)", []
    if name is not None and not isinstance(name, str):
        return str(name), []
    default = name is None
    if default:
        base = (stage.elec_basis.lower() if isinstance(stage.elec_basis, str) and not stage.elec_basis_groups
                else None)                     # several bases: the library's dict-basis default, aug-cc-pvtz-ri
        name = "aug-cc-pvtz-ri" if base is None else (base if base.endswith("-ri") else f"{base}-ri")
    elements = []
    for sym in syms:
        el = element_of(sym)
        if el not in elements:
            elements.append(el)
    missing = [el for el in elements if not _basis_has(name, el)]
    label = f"{name} (library default, <basis>-ri)" if default else name
    if missing:
        label += (f"; {', '.join(missing)}: even-tempered set from the orbital basis (PySCF has no {name} for "
                  f"{', '.join(missing)})")
    return label, missing


def _basis_has(name, symbol):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            gto.basis.load(name, symbol)
            return True
        except Exception:
            return False


def matched_fitting_set(elec_basis, symbols, suffix, fallback):
    """'<electron basis><suffix>' (e.g. aug-cc-pvtz-jkfit, aug-cc-pvtz-ri) for every element PySCF has
    it for, `fallback` for the others: a name when one choice covers all elements, else a per-element
    dict. Measured 2026-09-23 (SCF error vs exact integrals, aug-cc-pVTZ): aug-cc-pvtz-jkfit <= 0.09 mHa
    for H-, CN-, F-, Cl-, Br- and PsH, and 0.015 / 0.03 mHa for HBr2- + e+ / Br2 2- + e+; def2-universal-jkfit
    0.25-0.8 mHa on Cl/Br systems and 2.4 mHa on H-."""
    elements = sorted({element_of(s) for s in symbols if not is_dummy_label(s)})

    def fb(e):                                 # fallback may itself be per-element
        return fallback.get(e, DEFAULT_JKFIT) if isinstance(fallback, dict) else fallback
    if not isinstance(elec_basis, str):
        return fallback
    name = f"{elec_basis}{suffix}"
    have = {e: _basis_has(name, e) for e in elements}
    if have and all(have.values()):
        return name
    out = {e: (name if ok else fb(e)) for e, ok in have.items()}
    vals = set(map(str, out.values()))
    return next(iter(out.values())) if len(vals) == 1 else out


BROKEN_DZ_JKFIT = ("cc-pvdz-jkfit", "aug-cc-pvdz-jkfit")
BROKEN_DZ_JKFIT_Z = range(31, 37)            # Ga-Kr
DZ_JKFIT_REPLACEMENT = "aug-cc-pvtz-jkfit"


def fix_broken_dz_jkfit(choice, symbols):
    """PySCF's cc-pvdz-jkfit / aug-cc-pvdz-jkfit entries for Ga-Kr have no g functions and cannot fit the
    filled 3d shell (RHF, fitted minus exact, cc-pVDZ / aug-cc-pVDZ: Ga+ 482, Ge2+ 534, As3+ 585, Se2- 627,
    Br- 674, Kr 719 mHa; aug-cc-pvtz-jkfit for Br-: 0.002 / 0.004 mHa). Replace them by aug-cc-pvtz-jkfit for
    those elements (v04.05; v04.04 only warned). Returns (choice, [replaced elements]); a name stays a name
    when no element is affected, else an element-keyed dict."""
    elements = sorted({element_of(s) for s in symbols if not is_dummy_label(s)})
    mapping = _element_map(choice, elements)
    fixed = [e for e in elements if pyscf_elements.charge(e) in BROKEN_DZ_JKFIT_Z
             and str(mapping.get(e) or "").lower() in BROKEN_DZ_JKFIT]
    if not fixed:
        return choice, []
    for e in fixed:
        mapping[e] = DZ_JKFIT_REPLACEMENT
    names = set(map(str, mapping.values()))
    return (next(iter(mapping.values())) if len(names) == 1 else mapping), fixed


def jkfit_set(stage, symbols):
    """Mean-field (JK) fitting set of this stage for these atoms (broken DZ JK-fit sets replaced for Ga-Kr)."""
    if stage.jkfit_aux_e is not None:
        choice = stage.jkfit_aux_e
    else:
        choice = matched_fit(stage, symbols, "-jkfit", DEFAULT_JKFIT)
    return fix_broken_dz_jkfit(choice, symbols)[0]


def electronic_correlation_set(stage, symbols):
    """Correlation fitting set of an ELECTRONIC DF-CC stage: %auxbasis_e, else the RI set matched to
    %basis, else (no RI set in PySCF, e.g. 6-31g) the mean-field set, as before v04.03."""
    if stage.auxbasis_e:
        return stage.auxbasis_e
    return matched_fit(stage, symbols, "-ri", jkfit_set(stage, symbols))


def _resolve_fitting_roles(stage):
    """An RI set has no tight core functions, so in a mean-field role a
    Coulomb-metric fit puts the energy far too low (Br2 + e+ at aug-cc-pVTZ:
    4.9 Ha below the exact value) and every CC energy, gradient and geometry
    inherits it; pyscf_neo_cc (since 2026-09-23) refuses it in the SCF.

    %jkfit_aux_e (mean field): an RI set there is read as the one-positron
    correlation set it was meant to be (%auxbasis_e, unless given), and the
    mean field gets the JK-fit set matched to %basis.

    %df_aux auxbasis_e (two positrons): pair_df_aux preserves explicit inputs
    and defaults to matched RI. The native reference_model='scf' keeps the
    actual SCF reference independent of that correlation fit. More than two
    positrons retain their historical coupled-reference fitting policy."""
    ri = stage.jkfit_aux_e
    if is_ri_set_name(ri):
        stage.jkfit_aux_e = None                  # matched JK-fit set, resolved per molecule
        moved = ""
        if stage.auxbasis_e is None and stage.n_pos == 1:
            stage.auxbasis_e = ri
            moved = " and used as %auxbasis_e (one-positron correlation, where it belongs)"
        stage.warn(f"%jkfit_aux_e {ri} is an RI (correlation-fitting) set; in the mean-field role it misfits the "
                   f"core density (measured: 2.8 mHa per Cl, 2.4 Ha per Br). Mean field: the JK-fit set matched "
                   f"to %basis; {ri} moved out of the mean field{moved}. Set %jkfit_aux_e to a JK-fit set (e.g. "
                   f"aug-cc-pvtz-jkfit) to silence this warning.")


def _coords_in_bohr(stage):
    """'! Bohrs' / '%coords bohr': the geometry blocks (and %neb_product) were written in Bohr. Everything is kept
    in Angstrom inside the wrapper, so they are converted here: Cartesian coordinates, and in a Z-matrix the bond
    lengths (numbers and bond variables); angles and dihedrals do not change."""
    f = BOHR_TO_ANGSTROM
    stage.neb_product = [(a, tuple(x * f for x in c)) for a, c in stage.neb_product]
    if stage.geometry_kind == "xyz":
        stage.atoms = [(a, tuple(x * f for x in c)) for a, c in stage.atoms]
        return
    if stage.geometry_kind != "zmat" or _looks_cartesian(stage.zmat_rows):
        return
    bond_vars = set()
    for r in stage.zmat_rows:
        if len(r) >= 3:
            try:
                r[2] = repr(float(r[2]) * f)
            except ValueError:
                bond_vars.add(r[2].lstrip("-"))
    for nm in bond_vars:
        if nm in stage.zmat_vars:
            stage.zmat_vars[nm] *= f


def _finish_geometry(stage):
    if stage.coord_units == "BOHR":
        if stage.has_explicit_xyz or stage.neb_product:
            _coords_in_bohr(stage)
        else:
            stage.warn("Bohr units requested but this stage has no geometry block: the inherited geometry is used "
                       "as it is (units only apply to coordinates written in the stage)")
        if stage.scan is not None or stage.constraints:
            stage.warn("Bohr units apply to the geometry blocks only: %scan and %constraints values are still read "
                       "in Angstrom")
    if stage.geometry_kind == "zmat" and _looks_cartesian(stage.zmat_rows):
        stage.warn("'* zmat' block contains Cartesian coordinates (symbol x y z); read as '* xyz'")
        stage.geometry_kind = "xyz"
        stage.atoms = [(normalize_symbol(r[0]), (float(r[1]), float(r[2]), float(r[3])))
                       for r in stage.zmat_rows]
        stage.zmat_rows = []
        return
    if stage.geometry_kind == "zmat":
        variables = dict(stage.zmat_vars)
        if stage.scan and stage.scan["mode"] == "sym_zmat" and "start" in stage.scan:
            variables.setdefault(stage.scan["var"], stage.scan["start"])
        try:
            syms, coords = zmatrix_to_cartesian(stage.zmat_rows, variables)
            stage.atoms = [(s, tuple(c)) for s, c in zip(syms, coords)]
        except ValueError as err:
            stage.error(str(err))


# v04.05: settings a stage without a geometry block does NOT take from the stage before (everything else is taken)
_NOT_INHERITED_RUN_TYPES = ("SCAN", "NEB")     # they need their own %scan / %neb_product block


def _block_has_geometry(lines):
    for raw in lines:
        parts = strip_inline_comment(raw).strip()[1:].split() if strip_inline_comment(raw).strip().startswith("*") else []
        if parts and parts[0].lower() in ("xyz", "zmat", "int", "internal"):
            return True
    return False


def _inherit_stage(prev, stage_id):
    """v04.05: a stage without a geometry block continues the calculation of the stage before: it starts from
    all of that stage's settings (method, sector, integrals, basis sets, charge, multiplicity, positron and
    proton options, resources, analyses, ...), and whatever it writes itself replaces them ('! MP2' replaces
    CCSD(T), '%basis cc-pvtz' the basis). Not taken over: the job name (every stage names its own files), the
    geometry-block settings (units, Z-matrix), a %scan or %neb_product block, and the SCAN / NEB run types that
    need them (such a stage becomes SP_ENERGY unless it says otherwise). A stage with a geometry block is a new
    calculation and starts from the defaults."""
    st = copy.deepcopy(prev)
    fresh = JobStage(stage_id)
    for attr in ("stage_id", "job_name", "warnings", "errors", "_charge_sources", "_mult_sources", "_unit_sources",
                 "_keys_seen", "zmat_from", "zmat_consistent",
                 "coord_units", "atoms", "has_explicit_xyz", "geometry_kind", "zmat_rows", "zmat_vars", "scan",
                 "scan_range", "neb_product", "density_map_combine", "_keep_density"):
        setattr(st, attr, copy.deepcopy(getattr(fresh, attr)))
    for attr in ("block_text", "_block_zero", "_scan_fit_written", "_bang_seen", "then_freq", "freq_of_opt",
                 "_ts_mode_written"):
        if hasattr(st, attr):
            delattr(st, attr)
    # the Z-matrix goes on too (rows and variables), so a continuing ZOPT stage (e.g. ZOPT CCSD after ZOPT HF)
    # optimizes the same variables, starting from the previous stage's values (updated at run time)
    if prev.geometry_kind == "zmat" and prev.zmat_rows and not _looks_cartesian(prev.zmat_rows):
        st.zmat_rows, st.zmat_vars = copy.deepcopy(prev.zmat_rows), copy.deepcopy(prev.zmat_vars)
        st.zmat_from = prev.stage_id
        st.zmat_consistent = prev.run_type in ("ZOPT", "SP_ENERGY", "GRADIENT", "FREQ")
    if st.run_type == "ZOPT":                  # the Z-matrix optimizer is OPT + %opt_engine zopt
        st.run_type, st.opt_engine = "OPT", "ZOPT"
    if st.run_type in ("IRC", "NEB"):          # internally distinct; in the input both are '! TS' + %ts_mode
        st.run_type = "TS"
    if st.ts_mode == "NEB":                    # a NEB needs its own %neb_product: the next stage searches the TS
        st.ts_mode = "TS"
    if st.run_type in _NOT_INHERITED_RUN_TYPES:
        st.run_type = "SP_ENERGY"
    st.run_type_explicit = False               # a %scan block written here may still turn the stage into a SCAN
    st.inherited_from = prev.stage_id
    return st


def parse_block(lines, stage_id, prev=None):
    stage = _inherit_stage(prev, stage_id) if prev is not None and not _block_has_geometry(lines) \
        else JobStage(stage_id)
    stage._block_zero = False
    block_start = 0
    mode = None            # geometry kind or block directive being read
    for raw in lines:
        line = strip_inline_comment(raw).strip()
        if not line:
            continue
        low = line.lower()

        if mode in ("xyz", "zmat"):
            if line == "*" or low == "end":
                mode = None
                continue
            if line[0] in "%!*":
                mode = None
            else:
                _parse_geometry_row(stage, mode, line)
                continue
        elif mode is not None:
            if low == "end":
                _close_block(stage, mode, block_start)
                mode = None
                continue
            if line[0] in "%!*":
                stage.warn(f"%{mode} block was not closed with 'end' before {line!r}")
                _close_block(stage, mode, block_start)
                mode = None
            else:
                _parse_block_row(stage, mode, line)
                continue

        if line.startswith("!"):
            _parse_bang(stage, line)
        elif line.startswith("*"):
            parts = line[1:].split()
            kind = parts[0].lower() if parts else ""
            if kind not in ("xyz", "zmat", "int", "internal"):
                stage.error(f"Unknown geometry block {line!r} (use '* xyz <charge> <mult>' or '* zmat <charge> <mult>')")
                continue
            kind = "zmat" if kind in ("zmat", "int", "internal") else "xyz"
            if stage.has_explicit_xyz:
                stage.error(f"a second geometry block in one stage ({line!r}): is a '%next' missing before this "
                            f"calculation? Without it the second geometry replaces the first and the first "
                            f"calculation is lost")
            try:
                if len(parts) >= 2:
                    _set_charge(stage, int(parts[1]), "* " + kind)
                if len(parts) >= 3:
                    _set_mult(stage, int(parts[2]), "* " + kind)
            except ValueError:
                stage.error(f"Geometry header {line!r}: charge and multiplicity must be integers")
            stage.atoms, stage.zmat_rows = [], []
            stage.has_explicit_xyz = True
            stage.geometry_kind = kind
            mode = kind
        elif line.startswith("%"):
            mode = _parse_directive(stage, line)
            if mode is not None:
                stage._block_zero = False
                block_start = _block_length(stage, mode)
        elif low == "end":
            continue
        else:
            stage.warn(f"Line not understood and ignored: {line!r}")

    if mode in BLOCK_DIRECTIVES or mode == "scan":
        stage.warn(f"%{mode} block not closed with 'end' at the end of the stage")
        _close_block(stage, mode, block_start)
    _finish_geometry(stage)
    _resolve_fitting_roles(stage)

    for label, sources in (("charge", stage._charge_sources), ("multiplicity", stage._mult_sources),
                           ("coordinate units", stage._unit_sources)):
        if len(set(sources.values())) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in sources.items())
            stage.warn(f"Conflicting {label} values ({detail}); the one written last is used: "
                       f"{ {'charge': stage.charge, 'multiplicity': stage.multiplicity}.get(label, stage.coord_units)}")

    if stage.scan is not None:
        if stage.scan.get("one_line"):
            if stage.run_type != "SCAN":
                stage.warn("One-line '%scan r_min r_max n' only runs with '! SCAN' -- ignored for this "
                           f"{stage.run_type} stage.")
                stage.scan = None
        elif stage.run_type in ("OPT", "GRADIENT", "FREQ", "TS", "IRC", "NEB") and stage.run_type_explicit:
            stage.warn(f"%scan block ignored: the '!' line asks for {stage.run_type}.")
            stage.scan = None
        else:
            stage.run_type = "SCAN"
    if getattr(stage, "_scan_fit_written", False) and stage.scan_fit and stage.run_type != "SCAN":
        stage.warn(f"%scan_fit only acts in SCAN stages -- ignored for this {stage.run_type} stage")
    if stage.run_type == "SCAN" and stage.scan is None:
        stage.error("SCAN run type needs a scan definition: '%scan r_min r_max n' or a %scan ... end block")
    # v04.05: no run type means a single point (SP_ENERGY), silently; %opt_engine zopt turns OPT into the Z-matrix
    # optimizer (internally the ZOPT run type)
    if stage.opt_engine == "ZOPT":
        if stage.run_type == "OPT":
            stage.run_type = "ZOPT"
            stage.opt_engine = "AUTO"
        else:
            stage.warn(f"%opt_engine zopt only acts in OPT stages -- ignored for this {stage.run_type} stage")
            stage.opt_engine = "AUTO"
    # v04.05: '! TS' + %ts_mode irc / neb run the IRC / NEB (internally their own run types). A '! TS' / '! OptTS'
    # written in this stage without a %ts_mode line means the TS search, whatever an earlier stage used.
    if (stage.run_type == "TS" and getattr(stage, "_bang_seen", {}).get("run type") in ("TS", "OPTTS")
            and not getattr(stage, "_ts_mode_written", False)):
        stage.ts_mode = "TS"
    if stage.run_type == "TS" and stage.ts_mode in ("IRC", "NEB"):
        stage.run_type = stage.ts_mode
    elif getattr(stage, "_ts_mode_written", False) and stage.ts_mode != "TS" and stage.run_type != "TS":
        stage.warn(f"%ts_mode only acts in '! TS' stages -- ignored for this {stage.run_type} stage")
    _check_sector_words(stage)
    return stage


POSITRON_KEYS = {"pos_basis", "n_pos", "pos_atom", "pos_aux", "pos_mass", "quantum_site", "pos_spin", "pos_mult",
                 "pos_basis_map", "pos_charge", "mp2_partition",
                 "pair_reference"}
PROTON_KEYS = {"nuc_basis", "quantum_nuc", "nuc_mass", "cneo"}


def _check_sector_words(stage):
    """v04.05: the sector defaults to ELECTRONIC (it was POSITRON). An input that sets positron or proton
    directives but names no sector on the '!' line was written for the old default: refuse it with the fix,
    instead of silently running it as an electronic calculation."""
    if stage.sector_explicit:
        return
    pos = sorted(k for k in stage._keys_seen if k in POSITRON_KEYS)
    nuc = sorted(k for k in stage._keys_seen if k in PROTON_KEYS)
    if pos:
        stage.error(f"this stage sets positron directives ({', '.join('%' + k for k in pos)}) but has no sector on "
                    f"the '!' line; since v04.05 the default is ELECTRONIC: add POSITRON to the '!' line")
    elif nuc:
        stage.error(f"this stage sets quantum-proton directives ({', '.join('%' + k for k in nuc)}) but has no sector "
                    f"on the '!' line; since v04.05 the default is ELECTRONIC: add PROTON (or CNEO) to the '!' line")


def parse_input_file(filepath):
    if not os.path.exists(filepath):
        base = os.path.splitext(filepath)[0]
        for ext in (".inp", ".txt"):
            if os.path.exists(base + ext):
                filepath = base + ext
                break
        else:
            raise FileNotFoundError(f"Input file '{filepath}' not found.")
    with open(filepath, "r") as f:
        text = f.read()
    return parse_input_text(text, filepath)


def parse_input_text(text, source="<text>"):
    stages = []
    last_atoms = None
    idx = 0
    for block in split_blocks(text):
        idx += 1
        stage = parse_block(block, idx, stages[-1] if stages else None)
        stage.block_text = "\n".join(block)
        then_freq = getattr(stage, "then_freq", None)
        if not stage.has_explicit_xyz:
            if last_atoms:
                stage.atoms = copy.deepcopy(last_atoms)
                stage.geometry_kind = "zmat" if (stage.zmat_from and stage.run_type == "ZOPT"
                                                 and stage.zmat_consistent) else None
                print(f"[Block {idx}] No geometry block: continuing stage {stage.inherited_from} "
                      f"(its geometry result and all its settings, except those written here).")
            else:
                stage.error("no geometry defined and no previous stage to inherit from")
        else:
            last_atoms = copy.deepcopy(stage.atoms)
        stages.append(stage)
        if then_freq:
            # '! OPT FREQ' (as in ORCA): a FREQ stage right after, with the same settings and job name, at the
            # optimized geometry. Its files (<job>_freq.txt, <job>_freq.molden) do not clash with the OPT ones.
            idx += 1
            fq = _inherit_stage(stage, idx)
            fq.run_type, fq.run_type_explicit, fq.freq_mode = "FREQ", True, then_freq
            fq.job_name = stage.job_name
            fq.opt_engine = "AUTO"
            fq.atoms = copy.deepcopy(stage.atoms)
            fq.block_text = stage.block_text + "\n# (frequencies of '! OPT FREQ')"
            fq.freq_of_opt = stage.stage_id
            stage.then_freq = None
            stages.append(fq)
            print(f"[Block {stage.stage_id}] '! OPT FREQ': stage {idx} computes the frequencies at the optimized "
                  f"geometry.")
    if not stages:
        raise ValueError(f"No calculation blocks found in {source}")
    _warn_repeated_job_names(stages)
    return stages


def _warn_repeated_job_names(stages):
    """Two stages with the same %jobname write the same output files: the later one overwrites the earlier one's
    results. Not an error (it may be wanted), but a clear warning. The FREQ stage of '! OPT FREQ' shares its
    OPT stage's name on purpose (its files are <job>_freq.*, which do not clash)."""
    first = {}
    for s in stages:
        if getattr(s, "freq_of_opt", None):
            continue
        if s.job_name in first:
            s.warn(f"%jobname {s.job_name} is also the job name of stage {first[s.job_name]}: this stage's output "
                   f"files ({s.job_name}_final.xyz, {s.job_name}_summary.txt, ...) will OVERWRITE that stage's "
                   f"results. Give this stage its own %jobname to keep both.")
        else:
            first[s.job_name] = s.stage_id


# ----------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------

def _check_basis_groups(stage):
    """v04.08 '%basis' / '%pos_basis' with atoms: every named atom and element exists, no atom gets two bases
    from equal-rank selectors, and (electrons) every atom ends up with a basis."""
    if not stage.atoms:
        return
    for key, groups in (("basis", stage.elec_basis_groups), ("pos_basis", stage.pos_basis_groups)):
        if not groups:
            continue
        per, problems = assign_basis_groups(groups, stage.atoms)
        for msg in problems:
            stage.error(f"%{key}: {msg}")
        if key == "basis":
            bare = [f"{a}{i + 1}" for i, ((a, _), b) in enumerate(zip(stage.atoms, per)) if b is None]
            if bare:
                stage.error(f"%basis: {'atoms' if len(bare) > 1 else 'atom'} {', '.join(bare)} "
                            f"{'have' if len(bare) > 1 else 'has'} no electron basis; add a basis without atoms "
                            f"for all the others (e.g. %basis {' '.join(t for b, sel in groups for t in [b, *sel])} "
                            f"cc-pvdz)")
            if stage.use_df:
                stage.warn("%basis with several basis sets: the fitting sets are matched element by element "
                           "(an element with two bases uses its most common one's); check 'Fitting sets' below")
        elif stage.system_type != "POSITRON":
            stage.warn("%pos_basis only acts in POSITRON stages -- ignored")
        if key == "pos_basis" and stage.pos_basis_spec:
            stage.error("%pos_basis with atoms and %pos_basis_map in one stage: use one of them")


def _check_basis_names(stage):
    """v04.07: %basis must be a basis PySCF has for every element of the geometry (up to v04.06 a wrong name, e.g.
    aug-cc-pvt, passed validation and stopped the run at the first SCF, after the stages before it had run)."""
    real = [(s, c) for s, c in stage.atoms if not is_dummy_label(s)]
    if stage.elec_basis_groups:
        per, _ = assign_basis_groups(stage.elec_basis_groups, stage.atoms)
        for b in {b for b in per if b is not None}:
            _check_one_basis_name(stage, b, {element_of(sym) for (sym, _), x in zip(stage.atoms, per) if x == b})
    else:
        _check_one_basis_name(stage, stage.elec_basis, {element_of(sym) for sym, _ in real})


def _check_one_basis_name(stage, name, elements):
    if not isinstance(name, str) or not name.strip() or not elements:
        return
    from pyscf.gto import basis as _basis
    missing, unknown = [], False
    for el in sorted(elements):
        try:
            if not _basis.load(name, el):
                missing.append(el)
        except Exception as err:
            if "Unknown basis" in str(err):
                unknown = True
                break
            missing.append(el)
    if unknown:
        stage.error(f"%basis {name}: PySCF has no basis set of this name (a typo? e.g. aug-cc-pvtz, cc-pvdz, def2-tzvp)")
    elif missing:
        stage.error(f"%basis {name} has no functions for {', '.join(missing)}")


def _pos_map_element(stage, key):
    """Element of a %pos_basis_map key: the symbol itself, or the element of atom <key> (1-based)."""
    k = str(key)
    if k.isdigit():
        i = int(k) - 1
        return element_of(stage.atoms[i][0]) if 0 <= i < len(stage.atoms) else None
    return element_of(k)


def _is_element_token(t):
    """An element symbol written as a basis selector ('H', 'Li', 'n')."""
    if not (t.isalpha() and 1 <= len(t) <= 2):
        return False
    try:
        return pyscf_elements.charge(t.capitalize()) > 0
    except Exception:
        return False


def _basis_line_tokens(text):
    """Words of a %basis / %pos_basis line as (kind, text): a quoted string, or the longest run of words that names
    an existing file, is one ('file', path) token (paths may contain spaces); every other word is split further at
    braces and commas into ('word', ...) tokens."""
    words = re.findall(r'"[^"]*"|\'[^\']*\'|\S+', text)
    out, i = [], 0
    while i < len(words):
        w = words[i]
        if len(w) >= 2 and w[0] in "\"'" and w[-1] == w[0]:
            out.append(("file", w[1:-1]))
            i += 1
            continue
        for j in range(len(words), i, -1):
            cand = " ".join(words[i:j])
            if os.path.isfile(os.path.expanduser(cand)):
                out.append(("file", cand))
                i = j
                break
        else:
            out.extend(("word", t) for t in w.replace("{", " ").replace("}", " ").replace(",", " ").split())
            i += 1
    return out


def parse_basis_groups(val, stage, key):
    """v04.08: '%basis' / '%pos_basis' value -> [(basis, [selectors])].
      B                  B on every atom
      B N O  /  B {N O}  B on every N and O atom
      B 1 2  /  B {1 2}  B on atoms 1 and 2 (1-based, real atoms; 0 also means atom 1, as in %pos_atom)
      B1 H B2 N          B1 on the H atoms, B2 on the N atoms
      B1 H B2            B1 on the H atoms, B2 (no atoms after it) on all the others
      B [1,5] / B [1-5]  B on atoms 1 to 5, inclusive (v04.09)
    A token is an atom selector when it is a whole number or an element symbol, otherwise a basis name; braces
    and commas only group. Returns None (with an error) when the value cannot be read.
    v04.09: a basis file path may contain spaces (whole value, a run of words naming an existing file, or a
    quoted path); v04.08 split it into several basis names."""
    if os.path.isfile(os.path.expanduser(val.strip())):
        return [(val.strip(), [])]
    def _range(m):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo > hi:
            bad.append(m.group(0))
            return " "
        return " " + " ".join(str(i) for i in range(lo, hi + 1)) + " "
    bad = []
    val_r = re.sub(r"\[\s*(\d+)\s*[,-]\s*(\d+)\s*\]", _range, val)
    if bad:
        stage.error(f"%{key} {val}: atom range {bad[0]} runs backwards (write the lower atom first, e.g. [1,5])")
        return None
    if "[" in val_r or "]" in val_r:
        stage.error(f"%{key} {val}: an atom range is two atom numbers in brackets, e.g. [1,5] or [1-5]")
        return None
    toks = _basis_line_tokens(val_r)
    groups = []
    for kind, t in toks:
        if kind == "word" and (t.isdigit() or _is_element_token(t)):
            if not groups:
                stage.error(f"%{key} {val}: start with a basis name, then the atoms it goes on (e.g. %{key} "
                            f"PSX-TZ H)")
                return None
            groups[-1][1].append(t if t.isdigit() else t.capitalize())
        else:
            groups.append((t, []))
    if not groups:
        stage.error(f"%{key} needs a basis name")
        return None
    if sum(1 for _, sel in groups if not sel) > 1:
        stage.error(f"%{key} {val}: at most one basis without atoms (the one for all the atoms not named)")
        return None
    return groups


def assign_basis_groups(groups, atoms):
    """Per-atom basis (None = none) from parse_basis_groups groups, and a list of problems. An atom number beats
    an element symbol, which beats the basis given without atoms."""
    n = len(atoms)
    per, rank, problems = [None] * n, [0] * n, []
    fallback = next((b for b, sel in groups if not sel), None)
    elems = [element_of(a) for a, _ in atoms]
    for b, sel in groups:
        for t in sel:
            if t.isdigit():
                i = max(int(t), 1) - 1
                if i >= n:
                    problems.append(f"atom {t} does not exist ({n} atoms)")
                    continue
                hit, r = [i], 2
            else:
                hit, r = [i for i, e in enumerate(elems) if e == t], 1
                if not hit:
                    problems.append(f"no {t} atom in the geometry")
            for i in hit:
                if rank[i] == r and per[i] != b:
                    problems.append(f"atom {i + 1} ({atoms[i][0]}) is given both {per[i]} and {b}")
                elif r > rank[i]:
                    per[i], rank[i] = b, r
    for i in range(n):
        if per[i] is None and fallback is not None:
            per[i] = fallback
    return per, problems


def electron_basis_layout(stage, atoms):
    """(basis argument for gto.M / build_system, {atom index: label}) of the electrons. One basis: the name.
    Several (%basis B1 H B2 ...): a dict with each element's most common basis under its symbol and, for an atom
    that differs from it (an atom number), its own key '<symbol>@<n>' (the atom is relabelled to match)."""
    if not stage.elec_basis_groups:
        return stage.elec_basis, {}
    per, _ = assign_basis_groups(stage.elec_basis_groups, atoms)
    numbered = {max(int(t), 1) - 1 for _, sel in stage.elec_basis_groups for t in sel if t.isdigit()}
    by_el = {}
    for i, ((sym, _), b) in enumerate(zip(atoms, per)):
        by_el.setdefault(element_of(sym), []).append((i in numbered, b))
    # the element key holds the basis its atoms get by element or by default; atoms given by number keep their
    # own '@n' labels (ties broken by first appearance, so the layout never depends on set order)
    main = {}
    for e, lst in by_el.items():
        pool = [b for num, b in lst if not num] or [b for _, b in lst]
        main[e] = max(dict.fromkeys(pool), key=pool.count)
    basis, labels = dict(main), {}
    for i, ((sym, _), b) in enumerate(zip(atoms, per)):
        if b != main[element_of(sym)]:
            labels[i] = f"{sym}@{i + 1}"
            basis[labels[i]] = b
        elif is_ghost_label(sym):
            # 2026-10-20 fix: a ghost keyed only by its element gets its basis by PySCF's implicit fallback and has no
            # mol._basis entry, so the even-tempered DF fallback (aug_etb) raised KeyError 'GHOSTH'
            basis.setdefault(sym, b)
    return basis, labels


def electron_mol_input(stage, atom_str):
    """(atom string, basis) for building the electron Mole: atoms with their own basis get their '@n' label."""
    basis, labels = electron_basis_layout(stage, stage.atoms)
    if labels:
        parts = [p.strip() for p in atom_str.split(";")]
        for i, lab in labels.items():
            if i < len(parts):
                parts[i] = lab + " " + parts[i].split(None, 1)[1]
        atom_str = "; ".join(parts)
    return atom_str, basis


def element_fit_basis(stage, symbols):
    """The electron basis name that picks the fitting sets: %basis, or with several bases a per-element dict of
    each element's (most common) basis."""
    if not stage.elec_basis_groups:
        return stage.elec_basis
    basis, _ = electron_basis_layout(stage, stage.atoms)
    return {e: basis.get(e, stage.elec_basis) for e in {element_of(s) for s in symbols if not is_dummy_label(s)}}


def matched_fit(stage, symbols, suffix, fallback):
    """matched_fitting_set for this stage's electron basis, element by element when %basis has several."""
    eb = element_fit_basis(stage, symbols)
    if isinstance(eb, str):
        return matched_fitting_set(eb, symbols, suffix, fallback)
    out = {}
    for e, b in eb.items():
        fb = fallback.get(e, DEFAULT_JKFIT) if isinstance(fallback, dict) else fallback
        out.update(_element_map(matched_fitting_set(b, [e], suffix, fb), [e]))
    vals = set(map(str, out.values()))
    return next(iter(out.values())) if len(vals) == 1 else out


def basis_groups_label(groups, atoms):
    """'B1 on H2, H4; B2 on Li1, Li3' for the pre-validation summary ('' when groups is None)."""
    if not groups:
        return ""
    per, _ = assign_basis_groups(groups, atoms)
    on = {}
    for i, b in enumerate(per):
        if b is not None:
            on.setdefault(b, []).append(f"{atoms[i][0]}{i + 1}")
    return "; ".join(f"{b} on {', '.join(v)}" for b, v in on.items()) or "none"


def resolve_pos_atom(pos_atom_spec, atoms):
    """%pos_atom selectors -> sorted 0-based atom indices. Selectors are
    element symbols (every atom of that element) or 1-based atom numbers
    ('0' also means atom 1, as in twoposBrorsen). None -> None (all atoms)."""
    if not pos_atom_spec:
        return None
    indices = set()
    for tok in pos_atom_spec:
        t = str(tok).strip()
        if t.lstrip('-').isdigit():
            k = int(t)
            k = 0 if k == 0 else k - 1
            if 0 <= k < len(atoms):
                indices.add(k)
            continue
        want = element_of(t)
        for i, (sym, _) in enumerate(atoms):
            if sym.lower() == t.lower() or element_of(sym) == want:
                indices.add(i)
    return sorted(indices)


MIN_ATOM_DISTANCE = 0.5     # Angstrom; closer atoms mean a broken geometry (e.g. a Z-matrix dihedral)


def _check_electron_count(stage):
    """v04.05: the charge and multiplicity must fit the number of electrons (up to v04.04 PySCF stopped at run
    time, after the stages before had run). A stage without a geometry block inherits only the geometry, so a
    charged molecule needs %charge (and %mult) in every stage; the message says so."""
    try:
        z = sum(0 if is_ghost_label(sym) else gto.charge(element_of(sym)) for sym, _ in stage.atoms)
    except Exception:
        return                                            # an unknown symbol is reported elsewhere
    nelec = z - stage.charge
    if stage.multiplicity < 1:
        stage.error(f"multiplicity {stage.multiplicity} is not a spin state (it is 2S+1 >= 1: 1 singlet, "
                    f"2 doublet, ...; these atoms with charge {stage.charge} have {nelec} electrons)")
        return
    unpaired = stage.multiplicity - 1
    if nelec < 0 or unpaired > nelec or (nelec - unpaired) % 2:
        hint = ("" if stage.has_explicit_xyz else
                " This stage has no geometry block: it continues the stage before, with its geometry, charge "
                "and multiplicity unless it sets them.")
        stage.error(f"charge {stage.charge} and multiplicity {stage.multiplicity} do not fit these atoms "
                    f"({nelec} electrons: an {'odd' if nelec % 2 else 'even'} count needs an "
                    f"{'even' if nelec % 2 else 'odd'} multiplicity).{hint}")
        return
    _check_frozen_core(stage, nelec, unpaired)


def _check_frozen_core(stage, nelec, unpaired):
    """%frozen must leave at least one occupied electron orbital to correlate in each spin that has electrons:
    freezing them all made PySCF's DF-UCCSD stop with an h5py 'chunk dimensions must be positive' error
    (Na2+ with %frozen auto: 10 core orbitals per spin, 10 beta electrons)."""
    if stage.method not in ("MP2", "CCSD", "CCSD(T)"):
        return
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    if frozen <= 0:
        return
    n_b = (nelec - unpaired) // 2
    n_a = n_b + unpaired
    what = "%frozen auto (the core orbitals)" if isinstance(stage.frozen_core, str) else f"%frozen {frozen}"
    if frozen < min(n_a, n_b if n_b else n_a):
        return
    keep = max(min(n_a, n_b if n_b else n_a) - 1, 0)
    msg = (f"{what} freezes {frozen} orbitals per spin, but there are {n_a} alpha and {n_b} beta electrons "
           f"(multiplicity {stage.multiplicity}): no occupied {'beta' if n_b < n_a else 'electron'} orbital is "
           f"left to correlate, which crashes PySCF's DF-UCCSD (h5py 'chunk dimensions must be positive'). "
           f"Use %frozen {keep} or less.")
    if stage.system_type == "ELECTRONIC" or frozen > n_b:
        stage.error(msg)
    else:
        stage.warn(msg + " Kept for this sector, where the electron-positron/nuclear correlation remains.")


def validate_stage(stage):
    conditioning_problem = _conditioning_problem(stage)
    if conditioning_problem:
        stage.error(conditioning_problem)
    n = len(stage.atoms)
    if n >= 2:
        xyz = np.array([a[1] for a in stage.atoms], float)
        dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
        dist[np.diag_indices(n)] = np.inf
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i, j] < MIN_ATOM_DISTANCE:
            hint = (" -- check the Z-matrix angles and dihedrals (a ring needs dihedral 0 between ring atoms)"
                    if stage.geometry_kind == "zmat" else "")
            stage.error(f"atoms {min(i, j) + 1} ({stage.atoms[min(i, j)][0]}) and {max(i, j) + 1} "
                        f"({stage.atoms[max(i, j)][0]}) are {dist[i, j]:.3f} A apart{hint}")
    if n:
        _check_electron_count(stage)
    if n and stage.system_type == "POSITRON" and stage.n_pos == 1 and stage.use_df:
        label, missing = single_positron_correlation_label(stage, [a for a, _ in stage.atoms])
        if missing:
            name = label.split(";")[0].split(" (")[0]
            stage.warn(f"correlation fitting set: PySCF has no {name} for {', '.join(missing)}; pyscf_neo_cc uses an "
                       f"even-tempered set built from the orbital basis for {'it' if len(missing) == 1 else 'them'} "
                       f"(a named set with those elements, e.g. %auxbasis_e def2-qzvpp-ri, avoids the substitution)")
    if (stage.system_type == "POSITRON" and stage.method == "HF" and stage.run_type == "OPT"
            and stage.opt_engine in ("AUTO", "INTRINSIC")
            and not analytic_gradient_available(stage)[0]):
        asked = " (geomeTRIC was asked for)" if stage.opt_engine == "INTRINSIC" else ""
        stage.opt_engine = "WRAPPER"
        stage.warn(f"POSITRON HF has no analytic gradient in pyscf_neo_cc (only CCSD, CCSD(T) and MP2 have one): "
                   f"this OPT runs WOPT with numerical gradients{asked}, 6 energies per atom per step")
    if (stage.system_type == "POSITRON" and stage.method == "HF" and stage.run_type == "OPT"
            and stage.opt_engine in ("AUTO", "INTRINSIC")
            and any(is_ghost_label(sym) for sym, _ in stage.atoms)):
        stage.opt_engine = "WRAPPER"
        stage.warn("geomeTRIC does not accept ghost atom labels: this POSITRON HF OPT uses "
                   "WOPT with analytic gradients and preserves the ghost centres")
    if (stage.system_type == "ELECTRONIC" and stage.run_type == "OPT"
            and stage.opt_engine in ("AUTO", "INTRINSIC")):
        ok, why = analytic_gradient_available(stage)
        if not ok:
            stage.warn(f"ELECTRONIC {stage.method} OPT runs geomeTRIC with numerical gradients ({why}): "
                       f"6 energies per atom per step")
    _check_basis_names(stage)
    _check_basis_groups(stage)
    if stage.system_type == "POSITRON":
        for k in stage.pos_basis_spec:
            if str(k).isdigit() and not 1 <= int(k) <= n:
                stage.error(f"%pos_basis_map: atom {k} does not exist ({n} atoms)")
        problem = _positron_triples_problem(stage)
        if problem:
            stage.error(problem)
        sel = resolve_pos_atom(stage.pos_atom_spec, stage.atoms)
        if stage.pos_atom_spec and not sel:
            stage.error(f"%pos_atom {stage.pos_atom_spec} matches no atom")
        for tok in (stage.pos_atom_spec or []):
            if str(tok).isdigit() and int(tok) > n:
                stage.error(f"%pos_atom: atom {tok} does not exist ({n} atoms)")
        dummies = [k + 1 for k, r in enumerate(stage.zmat_rows or []) if r and is_dummy_label(r[0])]
        if stage.geometry_kind == "zmat" and dummies and any(str(t).isdigit() for t in (stage.pos_atom_spec or [])):
            sel_txt = ", ".join(f"{stage.atoms[i][0]}{i + 1}" for i in (sel or []))
            stage.warn(f"%pos_atom {' '.join(map(str, stage.pos_atom_spec))}: atom numbers count the real atoms "
                       f"only, not the Z-matrix rows, and this Z-matrix has dummy atoms (row{'s' * (len(dummies) > 1)} "
                       f"{', '.join(map(str, dummies))}); the selection is {sel_txt}. Check it, or select by element "
                       f"(e.g. %pos_atom H)")
        if stage.n_pos == 1 and stage.pos_spin:
            stage.warn("%pos_mult/%pos_spin ignored: a single positron has no pair spin")
        if stage.n_pos >= 2 and stage.pos_spin and POSSOCCSD is None:
            stage.error("%pos_mult > 1 needs POSSOCCSD, which this pyscf_neo_cc does not provide")
        if stage.df_aux is not None and (stage.n_pos == 1 or not stage.use_df):
            stage.warn("%df_aux only affects two-positron (n_pos >= 2) DF calculations -- ignored here")
        given = (stage.df_aux or {}).get("auxbasis_e") if isinstance(stage.df_aux, dict) else None
        heavy = sorted({element_of(a) for a, _ in stage.atoms
                        if not is_dummy_label(a) and pyscf_elements.charge(element_of(a)) > PAIR_LIGHT_Z_MAX})
        if stage.n_pos > 2 and stage.use_df and is_ri_set_name(given) and heavy:
            stage.warn(f"%n_pos {stage.n_pos} retains the legacy coupled reference and fitting policy: "
                       f"%df_aux auxbasis_e {given} is kept for H-Ne and replaced by the JK-fit set "
                       f"for {', '.join(heavy)}. The independent reference-model migration covers pairs only.")
    if stage.system_type == "POSITRON" and stage.method in ("CCSD", "CCSD(T)"):
        available, why = analytic_gradient_available(stage)
        if not available and stage.run_type == "OPT":
            if stage.opt_engine == "INTRINSIC":
                stage.error(f"Native positron OPT requires an analytic gradient: {why}. "
                            "Use '%opt_engine wopt' for numerical gradients (6 energies per atom per step).")
            else:
                stage.opt_engine = "WRAPPER"
                stage.warn(f"POSITRON {stage.method} OPT uses WOPT with numerical gradients "
                           f"({why}): 6 energies per atom per step")
        elif not available and stage.run_type == "GRADIENT":
            stage.warn(f"POSITRON {stage.method} uses numerical gradients ({why}): "
                       "6 energies per atom")
    _validate_particles(stage)
    _validate_cneo(stage)
    _validate_wavefunction_export(stage)
    _validate_freq(stage)
    _validate_ts(stage)
    _validate_neb(stage)
    if stage.use_df and stage.atoms:
        _warn_broken_dz_jkfit(stage)
    if stage.method == "MP2":
        _validate_mp2(stage)
    for c in stage.constraints:
        if any(not 1 <= i <= n for i in c["indices"]):
            stage.error(f"%constraints {c['type']} {c['indices']}: atom numbers must be 1..{n}")
    for p in stage.pinnings:
        if not 1 <= p <= n:
            stage.error(f"%pin {p}: atom numbers must be 1..{n}")
    for (a, b), (c, d) in stage.equivalences:
        if any(not 1 <= i <= n for i in (a, b, c, d)):
            stage.error(f"%equivalence {a},{b},{c},{d}: atom numbers must be 1..{n}")
    for (at, ax, _) in stage.pinnings_xyz:
        if not 0 <= at < n:
            stage.error(f"%pinnings: atom {at+1} does not exist ({n} atoms)")
    for (t, s, *_rest) in stage.equivalences_xyz:
        if not (0 <= t < n and 0 <= s < n):
            stage.error(f"%equivalences: atoms {t+1}/{s+1} must be 1..{n}")
    if (stage.pinnings or stage.equivalences or stage.pinnings_xyz or stage.equivalences_xyz
            or stage.constraints) and stage.run_type != "OPT":
        if stage.run_type == "ZOPT":
            stage.warn("ZOPT ignores %constraints / %pin / %pinnings / %equivalence(s): it only changes the "
                       "named Z-matrix variables. Hold a value fixed by writing it as a number; make two "
                       "values equal by giving them the same variable name.")
        else:
            stage.warn("Constraints / pinnings / equivalences only act during OPT -- ignored for this stage")
    if stage.run_type != "ZOPT" and (stage.zopt_gradient_every or stage.zopt_final_gradient):
        stage.warn("%zopt_gradient_every / %zopt_final_gradient only act with %opt_engine zopt -- ignored")
    if stage.run_type == "ZOPT":
        if stage.geometry_kind != "zmat" and stage.zmat_from and not stage.zmat_consistent:
            stage.error(f"ZOPT (%opt_engine zopt) continues stage {stage.zmat_from}, which moved the atoms with another "
                        f"optimizer: its Z-matrix variables no longer describe the geometry. Give this stage its own "
                        f"'* zmat' block (e.g. from that stage's results), or use %opt_engine zopt in that stage too")
        elif stage.geometry_kind != "zmat":
            stage.error("ZOPT (%opt_engine zopt) needs a '* zmat' geometry block in the same stage (it optimizes the "
                        "named Z-matrix variables)")
        else:
            try:
                kinds = zmat_variable_kinds(stage.zmat_rows)
            except ValueError as err:
                stage.error(str(err))
                kinds = {}
            if not kinds and stage.zmat_rows:
                kinds = _auto_zmat_bond_variables(stage)
                if not kinds:
                    stage.error("ZOPT: the Z-matrix has no named variables and no bond length to optimize -- "
                                "every value is a number, so there is nothing to optimize")
            for nm in kinds:
                if nm not in stage.zmat_vars:
                    stage.error(f"ZOPT: variable '{nm}' has no value (add '{nm} = <value>' to the block)")
            if len(kinds) > 12:
                stage.warn(f"ZOPT with {len(kinds)} variables needs many energies "
                           f"(>= {(len(kinds) + 1) * (len(kinds) + 2) // 2} before the first model step); "
                           "consider '! OPT' or tying variables together")
            need = (len(kinds) + 1) * (len(kinds) + 2) // 2
            if kinds and stage.zopt["max_evals"] < need + 1:
                stage.error(f"%zopt_max_evals {stage.zopt['max_evals']} is too small for {len(kinds)} "
                            f"variables (needs more than {need})")
        if stage.scan:
            stage.warn("%scan is ignored by ZOPT")
        if stage.zopt_gradient_every or stage.zopt_final_gradient:
            ok_g, why = analytic_gradient_available(stage)
            written = stage._keys_seen & {"zopt_gradient_every", "zopt_final_gradient"}
            if not ok_g and stage.inherited_from and not written:
                # carried over from a stage whose method has an analytic gradient (e.g. HF, then DF-CCSD):
                # this stage runs plain, energy-only ZOPT instead of refusing the input
                stage.zopt_gradient_every, stage.zopt_final_gradient = 0, False
                stage.warn(f"%zopt_gradient_every / %zopt_final_gradient (carried over from stage "
                           f"{stage.inherited_from}) are switched off here: {why}. This stage runs energy-only ZOPT")
            elif not ok_g:
                stage.error(f"%zopt_gradient_every / %zopt_final_gradient need an analytic gradient: {why}. A "
                            f"numerical one costs 6 energies per atom, more than ZOPT's energies")
            nvar = len(zmat_variable_kinds(stage.zmat_rows)) if stage.geometry_kind == "zmat" else 0
            if stage.zopt_gradient_every and 0 < nvar < 3:
                stage.warn(f"%zopt_gradient_every with {nvar} variable(s): gradients do not pay below 3 "
                           f"variables (measured: 9-38 % more work with 2); plain ZOPT is cheaper")
    if stage.run_type == "OPT" and stage.opt_engine in ("AUTO", "INTRINSIC") \
            and (stage.equivalences or stage.equivalences_xyz):
        stage.warn("The geomeTRIC engine does not enforce equivalences during the optimization "
                   "(the start geometry is symmetrized); use '%opt_engine wopt' to enforce them every step")
    sc = stage.scan
    if sc:
        steps = sc.get("steps", 0)
        if steps < 2:
            stage.error(f"%scan needs at least 2 points, got {steps}")
        need = MIN_FIT_POINTS.get(stage.scan_fit)
        if need and steps < need:
            stage.error(f"a {stage.scan_fit} fit needs at least {need} scan points, got {steps}")
        if sc["mode"] == "bond":
            a, b = sc["atoms"]
            if not (1 <= a <= n and 1 <= b <= n and a != b):
                stage.error(f"%scan bond {a} {b}: atom numbers must be two different atoms in 1..{n}")
        elif sc["mode"] in ("angle", "dihedral"):
            at = sc["atoms"]
            if not all(1 <= i <= n for i in at) or len(set(at)) != len(at):
                stage.error(f"%scan {sc['mode']} {' '.join(map(str, at))}: atom numbers must be "
                            f"{len(at)} different atoms in 1..{n}")
            elif stage.geometry_kind == "zmat":
                stage.error(f"%scan {sc['mode']} needs a Cartesian geometry; with a Z-matrix scan its variable "
                            f"by name (%scan <variable> start end n)")
            if sc["mode"] == "angle" and any(not 0.0 < v <= 180.0 for v in scan_values(sc)):
                stage.error("%scan angle: values must be in (0, 180] degrees")
        elif sc["mode"] == "translate":
            i0, i1 = sc["range"]
            if not (1 <= i0 <= i1 <= n):
                stage.error(f"%scan translate {i0} {i1}: need 1 <= first <= last <= {n}")
            if sc["axis"] not in ("x", "y", "z"):
                stage.error(f"%scan translate: axis must be x, y or z, got {sc['axis']!r}")
        elif sc["mode"] == "sym_zmat":
            if stage.geometry_kind != "zmat":
                stage.error("%scan sym_zmat needs a '* zmat' geometry in the same stage")
            elif not any(sc["var"] in (t.lstrip('-') for t in row[1:]) for row in stage.zmat_rows):
                stage.error(f"%scan: variable '{sc['var']}' is not used in the Z-matrix")
        elif sc["mode"] == "fragment" and n < 2:
            stage.error("%scan (fragment scan) needs at least 2 atoms")
    if stage.run_type == "GRADIENT" and stage.system_type == "POSITRON" and stage.method == "DFT":
        stage.error("DFT is not available for the POSITRON sector")


def _validate_cneo(stage):
    """Constrained NEO (%cneo, or CNEO / CDFT on the ! line)."""
    if not stage.cneo:
        if stage.cneo_variant != "ccd":
            stage.warn("%cneo_variant only acts with %cneo true -- ignored")
        return
    if stage.system_type != "PROTON":
        stage.warn(f"%cneo only acts in the PROTON sector (quantum nuclei) -- ignored in {stage.system_type}")
        stage.cneo = False
        return
    if not HAS_CNEO:
        stage.error("%cneo needs pyscf_neo_cc.cneo, which this pyscf_neo_cc does not provide")
    if not stage.quantum_nuc:
        stage.error("%cneo needs at least one quantum nucleus (%quantum_nuc)")
    if stage.method in ("HF", "DFT") and stage.cneo_variant != "ccd":
        stage.warn(f"%cneo_variant {stage.cneo_variant} only affects correlated CNEO (MP2, CCSD, CCSD(T)); "
                   f"CNEO-{stage.method} has one form -- ignored")
    if stage.method in ("CCSD", "CCSD(T)", "MP2") and stage.cneo_variant == "ucd" and stage.run_type in ("OPT", "ZOPT"):
        stage.warn("CNEO ucd keeps the constraint only on the HF reference; its optimized geometries fall back "
                   "towards plain NEO's (measured, HF molecule: CNEO-CCSD(T) 0.8957 A with ucd, 0.9677 A with "
                   "ccd, CNEO-MP2 0.966 A). ccd is the default for that reason")


def _validate_mp2(stage):
    """What each sector's MP2 route supports (see build_mp2)."""
    sector = stage.system_type
    if sector == "PROTON" and stage.cneo:
        return                           # CNEO-MP2 exists with and without DF (checked in _validate_cneo)
    if sector in ("POSITRON", "PROTON"):
        if not stage.use_df:
            stage.error(f"MP2 in the {sector} sector exists only with density fitting in pyscf_neo_cc: write DF "
                        f"(there is no conventional positron or NEO MP2)")
        if stage.conv_check:
            stage.warn(f"%conv_check skipped for MP2 in the {sector} sector: there is no conventional "
                       f"(exact-integral) positron / NEO MP2 to compare with")
    if sector == "POSITRON" and stage.n_pos >= 2:
        if stage.pos_spin:
            stage.error("MP2 with two positrons is the closed-shell pair only (POSDFMP2): %pos_mult 3 / "
                        "%pos_spin is not available")
        if stage.n_pos % 2:
            stage.error(f"MP2 with {stage.n_pos} positrons: POSDFMP2 needs an even (closed-shell) number")
    if sector == "ELECTRONIC" and stage.use_df and stage.run_type in ("OPT", "GRADIENT"):
        stage.warn("electronic DF-MP2 has no analytic gradient in PySCF: numerical gradients (6 energies per "
                   "atom per step). CONV gives analytic MP2 gradients")


HEAVY_PARTICLE_ME = 10.0      # above this mass (electron masses) a positron/electron basis is far too diffuse


def _is_nuclear_basis_spec(stage):
    names = list(stage.pos_basis_spec.values()) if stage.pos_basis_spec else [stage.pos_basis_name]
    return all(is_nuclear_basis_name(v) for v in names)


def _validate_particles(stage):
    """Particle masses: %pos_mass / %quantum_site (POSITRON) and %nuc_mass (PROTON)."""
    if stage.pair_reference != "scf":
        if not (stage.system_type == "POSITRON" and stage.n_pos == 2 and stage.use_df):
            stage.warn("%pair_reference only acts on two-positron DF stages -- ignored")
        else:
            stage.warn("%pair_reference legacy: the 2026-09-23 library's pair reference, kept only to reproduce old "
                       "numbers (Kurt Brorsen: it carries the reference inconsistency behind the large DF offsets). "
                       "For Cl / Br and heavier, v04.08 also used JK-fit correlation sets where v04.09 uses matched "
                       "RI: to reproduce a v04.08 number exactly, also give that %df_aux auxbasis_e")
    n = len(stage.atoms)
    labels = [sym for sym, _ in stage.atoms]
    iso = sorted({lab for lab in labels if re.search(r'[+*]', str(lab))})
    quantum = set(stage.quantum_nuc) if stage.system_type == "PROTON" else set()
    classical_iso = [lab for i, lab in enumerate(labels) if lab in iso and i not in quantum]
    if classical_iso:
        stage.warn(f"isotope symbols {sorted(set(classical_iso))} act as plain H here: the H+ (deuteron) / "
                   f"H* (antimuon) marks only change quantum nuclei of the PROTON sector. In the POSITRON "
                   f"sector use %pos_mass with %quantum_site.")
    if stage.system_type != "POSITRON":
        if stage.pos_mass_amu is not None or stage.quantum_site is not None:
            stage.warn("%pos_mass / %quantum_site only act in the POSITRON sector -- ignored "
                       "(PROTON sector: %nuc_mass)")
    else:
        m_me = pos_mass_me(stage)
        site = stage.quantum_site
        if site is not None:
            if not 1 <= site <= n:
                stage.error(f"%quantum_site {site}: atom numbers must be 1..{n}")
            elif element_of(labels[site - 1]) != "H":
                stage.error(f"%quantum_site {site} is {labels[site - 1]}: the particle has charge +1, so it "
                            f"can only replace a hydrogen nucleus")
            if stage.n_pos != 1:
                stage.error("%quantum_site needs %n_pos 1 (one particle replaces one H nucleus)")
        nuclear = _is_nuclear_basis_spec(stage)
        if m_me > HEAVY_PARTICLE_ME:
            if not nuclear:
                stage.warn(f"particle mass {m_me:.4g} m_e with the positron basis "
                           f"{stage.pos_basis_name if not stage.pos_basis_spec else 'map'}: a positron or "
                           f"electron basis is far too diffuse for a heavy particle. Use a protonic nuclear set "
                           f"(%pos_basis pb4d ... pb6h), which the wrapper scales to the mass.")
            if site is None:
                stage.warn(f"a particle of {m_me:.4g} m_e stays where its basis functions are: without "
                           f"%quantum_site it is described only by functions centred on the atoms in "
                           f"%pos_atom. To put it in place of an H nucleus use %quantum_site <H atom>.")
        elif nuclear:
            stage.warn(f"{stage.pos_basis_name} is a protonic nuclear basis; for a particle of {m_me:.4g} m_e "
                       f"its exponents are scaled by {nuclear_basis_scale(m_me / AMU_TO_ME):.4f}. Positron "
                       f"bases (PSX-DZ/TZ/QZ) are the tested choice for a positron.")
        if stage.pos_mass_amu is not None and stage.pos_aux != "etb" and stage.use_df:
            stage.warn(f"%pos_aux {stage.pos_aux} with a particle of {m_me:.4g} m_e: the fitting set must "
                       f"follow the particle basis; 'etb' (built from it) is the safe choice.")
    if stage.nuc_mass is not None:
        if stage.system_type != "PROTON":
            stage.warn("%nuc_mass only acts in the PROTON sector -- ignored (POSITRON sector: %pos_mass)")
            return
        if not NEO_TAKES_MASS:
            stage.error("%nuc_mass needs a pyscf_neo_cc whose NEO Mole takes a mass list (2026-09-15 or "
                        "later); this one does not. The H+ (deuteron) and H* (antimuon) symbols, or the "
                        "POSITRON sector with %pos_mass and %quantum_site, work on it.")
        if len(stage.nuc_mass) != len(stage.quantum_nuc):
            stage.error(f"%nuc_mass has {len(stage.nuc_mass)} value(s) but %quantum_nuc lists "
                        f"{len(stage.quantum_nuc)} nucleus/nuclei: give one mass per quantum nucleus, in the "
                        f"order of %quantum_nuc ('default' keeps the library's value)")
            return
        for idx, (m, _name) in zip(stage.quantum_nuc, stage.nuc_mass):
            if not 0 <= idx < n:
                stage.error(f"%quantum_nuc {idx} does not exist (0-based, {n} atoms)")
                continue
            if m is None:
                continue
            lab = labels[idx]
            if element_of(lab) != "H":
                stage.error(f"%nuc_mass for atom {idx} ({lab}): masses can be set for hydrogen-like "
                            f"(charge +1) quantum nuclei only")
            elif re.search(r'[+*]', str(lab)):
                stage.error(f"%nuc_mass for atom {idx} ({lab}): the {lab} symbol already fixes the mass; "
                            f"write plain H with %nuc_mass, or keep {lab} without it")
    if stage.system_type == "PROTON":
        for idx in stage.quantum_nuc:
            if not 0 <= idx < n:
                if stage.nuc_mass is None:
                    stage.error(f"%quantum_nuc {idx} does not exist (0-based, {n} atoms)")
        h_quantum = any(0 <= i < n and element_of(labels[i]) == "H" for i in stage.quantum_nuc)
        nb = str(stage.nuc_basis)
        if h_quantum and nuclear_basis_dir() and not (nuclear_basis_file(nb)
                                                      or re.search(r'\d+s\d+p\d+d', nb, re.I)):
            stage.error(f"%nuc_basis {nb}: not a pyscf_neo_cc nuclear basis. Available: "
                        f"{', '.join(available_nuclear_bases())}, or an even-tempered 'NsNpNd' / "
                        f"'NsNpNdNf' name (manual section 7.4)")


def describe_particle(stage):
    """One line describing the POSITRON-sector particle (None for a real positron without a site)."""
    if stage.pos_mass_amu is None and stage.quantum_site is None:
        return None
    m_amu = PARTICLE_MASS_AMU["positron"] if stage.pos_mass_amu is None else stage.pos_mass_amu
    name = stage.pos_particle or ("positron" if stage.pos_mass_amu is None else "custom mass")
    txt = f"{name}, mass {m_amu * AMU_TO_ME:.6f} m_e = {m_amu:.10f} amu, charge +1"
    if stage.quantum_site is not None:
        lab = stage.atoms[stage.quantum_site - 1][0] if 0 < stage.quantum_site <= len(stage.atoms) else "?"
        txt += f"; replaces the nucleus of atom {stage.quantum_site} ({lab})"
    return txt


def describe_nuc_masses(stage):
    if not stage.nuc_mass:
        return None
    parts = []
    for idx, (m, name) in zip(stage.quantum_nuc, stage.nuc_mass):
        if m is None:
            parts.append(f"atom {idx}: library default")
        else:
            parts.append(f"atom {idx}: {name or 'custom'} {m:.10f} amu (basis x {nuclear_basis_scale(m):.6f})")
    return "; ".join(parts)


def _warn_broken_dz_jkfit(stage):
    """Say where a broken DZ JK-fit set (Ga-Kr, see fix_broken_dz_jkfit) was asked for or would have been
    the default, and that aug-cc-pvtz-jkfit is used there instead."""
    symbols = [a for a, _ in stage.atoms if not is_dummy_label(a)]
    asked = []
    mean = (stage.jkfit_aux_e if stage.jkfit_aux_e is not None
            else matched_fit(stage, symbols, "-jkfit", DEFAULT_JKFIT))
    asked += [f"{e} (mean field, {_element_map(mean, [e])[e]})" for e in fix_broken_dz_jkfit(mean, symbols)[1]]
    given = (stage.df_aux or {}).get("auxbasis_e") if isinstance(stage.df_aux, dict) else None
    if given and stage.system_type == "POSITRON" and stage.n_pos > 2:
        asked += [f"{e} (pair set, {_element_map(given, [e])[e]})" for e in fix_broken_dz_jkfit(given, symbols)[1]]
    if asked:
        stage.warn(f"PySCF's DZ JK-fit sets have no g functions for Ga-Kr and cannot fit the filled 3d shell "
                   f"(Br- RHF 674 mHa off): {DZ_JKFIT_REPLACEMENT} is used instead for {', '.join(asked)}.")
    if given and stage.system_type == "POSITRON" and stage.n_pos == 2:
        affected = fix_broken_dz_jkfit(given, symbols)[1]
        if affected:
            stage.warn(f"The explicit pair correlation recipe {given} is retained for {', '.join(affected)}. "
                       "Its DZ JK-fit data omit g functions; check auxiliary-basis accuracy separately. "
                       f"The pair correlation method uses reference_model='{_pair_reference_model(stage)}'.")


def _positron_gradient_problem(stage, use_df=None):
    """Describe structural derivative restrictions before constructing a method."""
    if stage.system_type != "POSITRON":
        return None
    fitted = stage.use_df if use_df is None else bool(use_df)
    if stage.method == "MP2":
        if not fitted:
            return "positron MP2 requires DF; no conventional MP2 route is available"
        if stage.n_pos >= 2 and (stage.pos_spin or stage.n_pos % 2):
            return "positron MP2 requires an even closed-shell positron population"
    if stage.method in ("CCSD", "CCSD(T)") and stage.n_pos >= 2 and not fitted:
        if resolve_frozen_core(stage.frozen_core, stage.atoms):
            return ("canonical positron-pair frozen-core gradients are unavailable; "
                    "use DF, frozen 0, or numerical energy differences")
        if stage.method == "CCSD(T)" and (stage.compute_eee or stage.compute_t2e1p):
            return ("positron-pair triples gradients require DF; "
                    "use DF or numerical energy differences")
    return None


def analytic_gradient_available(stage):
    """(True, "") when the stage method has an analytic nuclear gradient in this wrapper, else
    (False, reason). Manual section 10."""
    m, sec = stage.method, stage.system_type
    if sec == "POSITRON":
        problem = _positron_gradient_problem(stage)
        if problem:
            return False, problem
        if m in ("CCSD", "CCSD(T)", "MP2") or (m == "HF" and _native_poshf_preflight is not None):
            return True, ""
        return False, "this pyscf_neo_cc has no POSHF (positron HF) gradient"
    if sec == "ELECTRONIC":
        if m == "MP2" and stage.use_df:
            return False, "PySCF has no DF-MP2 gradient (use CONV)"
        if m in ("CCSD", "CCSD(T)") and stage.use_df:
            return False, ("PySCF has no gradient of the density-fitted CC energy (its CC gradient on DF "
                           "amplitudes is 3-5e-5 Ha/Bohr off at aug-cc-pVTZ; use CONV for analytic gradients)")
        if m == "CCSD(T)" and stage.multiplicity != 1:
            return False, "PySCF has no open-shell CCSD(T) gradient"
        return True, ""
    if sec == "PROTON":
        if stage.cneo or m in ("HF", "DFT", "MP2") or not stage.use_df:
            return True, ""
        return False, "the DF NEO-CC gradient needs the library's DF-NEO mean-field setup (not in the wrapper yet)"
    return False, f"unknown sector {sec}"


def engine_label(stage):
    """The optimizer an OPT stage really runs, as printed in the stage summary and the summary file."""
    if stage.system_type == "ELECTRONIC":
        if stage.opt_engine == "WRAPPER":
            return "WRAPPER"
        ok, _ = analytic_gradient_available(stage)
        return "geomeTRIC (analytic gradients)" if ok else "geomeTRIC (numerical gradients)"
    if stage.system_type == "POSITRON" and stage.method == "HF":
        if analytic_gradient_available(stage)[0]:
            return ("WRAPPER (analytic gradients)" if stage.opt_engine == "WRAPPER"
                    else "geomeTRIC (analytic gradients)")
        return "WRAPPER (numerical gradients: no analytic POSHF gradient)"
    if stage.system_type == "POSITRON" and stage.opt_engine == "WRAPPER":
        available, why = analytic_gradient_available(stage)
        if not available:
            return f"WRAPPER (numerical gradients: {why})"
    return stage.opt_engine


class InputValidationError(ValueError):
    pass


def _block_line_spans(text):
    """1-based line numbers of each block that split_blocks keeps: [[n, ...], ...] (same order)."""
    spans, current = [], []
    for n, line in enumerate(text.splitlines(), 1):
        if _BLOCK_SPLIT_RE.match(line):
            if current:
                spans.append(current)
            current = []
        else:
            current.append((n, line))
    if current:
        spans.append(current)
    return [b for b in spans if any(strip_inline_comment(l).strip() for _, l in b)]


_MSG_TOKEN_RE = re.compile(r"(%[A-Za-z_]+|'[^']+'|\b[A-Z][A-Z0-9_()]{1,}\b)")


def _locate_message(msg, block):
    """Best-effort line of a stage message: the first line of the block holding a token the message quotes."""
    for tok in _MSG_TOKEN_RE.findall(msg):
        tok = tok.strip("'").lower()
        if len(tok) < 2:
            continue
        for n, line in block:
            code = strip_inline_comment(line).lower()
            if code.strip() and re.search(r"(?<![\w%])" + re.escape(tok) + r"(?![\w])", code):
                return n
    return None


def parse_text(text):
    """Parse and validate input text without running anything or writing files.
    Returns [{"stage", "line", "severity", "msg"}]; line is 1-based or None (stage-level message)."""
    diags = []
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            stages = parse_input_text(text)
            try:
                run_pre_validation(stages)
            except InputValidationError:
                pass
    except Exception as err:          # parse failures that are not stage messages
        return [{"stage": None, "line": None, "severity": "error", "msg": f"{type(err).__name__}: {err}"}]
    spans = _block_line_spans(text)
    for s in stages:
        block = spans[s.stage_id - 1] if 0 < s.stage_id <= len(spans) else []
        for sev, msgs in (("error", s.errors), ("warning", s.warnings)):
            for m in msgs:
                diags.append({"stage": s.stage_id, "line": _locate_message(m, block),
                              "severity": sev, "msg": m})
    return diags


class PreflightError(RuntimeError):
    """A run-time check that stops a stage before its expensive part (exit code 3)."""


_FINAL_ANALYSIS_KEYS = ("molden", "cube", "chelpg", "density_maps")


def wants_final_analysis(stage):
    """%molden, %cube, %chelpg or %density_map: work done at the stage's final geometry."""
    return any(getattr(stage, k) for k in _FINAL_ANALYSIS_KEYS)


def _validate_wavefunction_export(stage):
    maps_made = {}
    for req in stage.density_maps:
        if req.density_map == "default":
            req.density_map = "both" if stage.system_type == "POSITRON" else "electron"
        if req.density_map in ("positron", "both") and stage.system_type == "ELECTRONIC":
            stage.error(f"%density_map {req.density_map}: an ELECTRONIC stage has no positron density "
                        f"(use %density_map electron)")
        species = ["electron", "positron"] if req.density_map == "both" else [req.density_map]    # or ['esp']
        if len(req.density_map_ranges) > len(species):
            stage.error(f"%density_map {req.density_map}: {len(req.density_map_ranges)} colour ranges given, at "
                        f"most {len(species)} ({' then '.join(species)})")
        elif req.density_map == "both" and len(req.density_map_ranges) == 1:
            stage.error("%density_map both: give two colour ranges, electrons then positrons (auto keeps the "
                        "default), e.g. %density_map both 0,0.5 auto")
        else:
            req.density_map_limits = {sp: r for sp, r in zip(species, req.density_map_ranges) if r}
        for sp in species:               # the same map twice would overwrite its files
            if sp in maps_made:
                stage.error(f"%density_map: the {sp} map is requested twice (by '{maps_made[sp]}' and "
                            f"'{req.density_map}'); use one line per map, e.g. %density_map both ... and "
                            f"%density_map esp ...")
            maps_made.setdefault(sp, req.density_map)
    if not wants_final_analysis(stage):
        return
    which = " / ".join(f"%{k.rstrip('s') if k == 'density_maps' else k}" for k in _FINAL_ANALYSIS_KEYS
                       if getattr(stage, k))
    if stage.system_type not in ("ELECTRONIC", "POSITRON"):
        stage.error(f"{which}: the ELECTRONIC and POSITRON sectors only (not {stage.system_type})")
    if stage.run_type not in ("SP_ENERGY", "GRADIENT", "OPT", "ZOPT", "FREQ", "TS"):
        stage.error(f"{which}: SP_ENERGY, GRADIENT, OPT (any %opt_engine), FREQ and TS stages only (a {stage.run_type} stage has no "
                    f"single final geometry)")
    if stage.method == "MP2":
        stage.warn(f"{which}: MP2 stages write the HF level only (no MP2 density)")
    elif stage.method in ("CCSD", "CCSD(T)"):
        if stage.system_type == "POSITRON" and stage.n_pos >= 2 and stage.pos_spin:
            stage.warn(f"{which}: triplet positron pairs write the HF level only")
        stage.warn(f"{which}: an extra single point with the Lambda equations at the final geometry "
                   f"(the CC density is the unrelaxed CCSD one-particle density{'; (T) is not included' if stage.method == 'CCSD(T)' else ''})")


def positron_gradient_needed(stage):
    """Whether this correlated positron workflow will request nuclear derivatives."""
    if stage.system_type != "POSITRON" or stage.method not in ("CCSD", "CCSD(T)", "MP2"):
        return False
    if not analytic_gradient_available(stage)[0]:
        return False
    if stage.run_type == "FREQ":
        return freq_route(stage) != "energy"
    if stage.run_type == "ZOPT":
        return bool(stage.zopt_gradient_every or stage.zopt_final_gradient)
    return stage.run_type in ("OPT", "GRADIENT", "TS", "IRC", "NEB")


def positron_gradient_preflight(stage, method, use_df=None, gradient_required=None):
    """Admit the actual method and retain its fits before correlation work starts.

    Frozen orbitals must already be resolved. Energy-only cross-checks and density
    exports do not request derivatives; force callers may require them explicitly.
    The native requirement persists through later solves and scanner geometries.
    """
    if gradient_required is None:
        primary_route = use_df is None or bool(use_df) == bool(stage.use_df)
        gradient_required = primary_route and positron_gradient_needed(stage)
    if not gradient_required:
        return method
    problem = _positron_gradient_problem(stage, use_df)
    if problem:
        raise PreflightError(f"Analytic positron gradient unavailable: {problem}")
    preflight = getattr(method, "preflight", None)
    if not callable(preflight):
        raise PreflightError("Positron gradients require a pyscf_neo_cc version with "
                             "method.preflight(); update the library before this workflow.")
    try:
        preflight()
    except (ValueError, RuntimeError, NotImplementedError) as err:
        raise PreflightError(f"Positron gradient admission failed: {err}") from err
    print("  [Pre-flight] positron gradient: native method and fitting metrics admitted.")
    return method


def proton_gradient_preflight(stage, method, use_df=None, gradient_required=None):
    """Check the actual NEO/CNEO method before a declared derivative workflow.

    Preserve plain-NEO energy-only correlation and its reference Hamiltonian.
    Frozen orbitals must already be resolved. Direct force callers pass an
    explicit requirement even when their input stage describes a single point.
    """
    if gradient_required is None:
        primary_route = use_df is None or bool(use_df) == bool(stage.use_df)
        if stage.run_type == "FREQ":
            derivative_route = freq_route(stage) != "energy"
        elif stage.run_type == "ZOPT":
            derivative_route = bool(stage.zopt_gradient_every or stage.zopt_final_gradient)
        else:
            derivative_route = stage.run_type in ("OPT", "GRADIENT", "TS", "IRC", "NEB")
        gradient_required = primary_route and derivative_route
    if not gradient_required:
        return method
    preflight = getattr(method, "preflight", None)
    if not callable(preflight):
        raise PreflightError("NEO/CNEO gradients require a pyscf_neo_cc version with "
                             "method.preflight(); update the library before this workflow.")
    try:
        preflight()
    except (ValueError, RuntimeError, NotImplementedError) as err:
        raise PreflightError(f"NEO/CNEO gradient admission failed: {err}") from err
    print("  [Pre-flight] NEO/CNEO gradient: native method and fitting metrics admitted.")
    return method


def run_pre_validation(stages):
    print("=" * 80)
    print(f"newBRORSEN v{VERSION} MULTI-STAGE CHAINED JOB ENGINE (STAGE COUNT: {len(stages)})")
    print("[Input Pre-Validation Pass] Checking all stages for correctness...")
    print("=" * 80)
    all_errors = []
    _validate_map_combine(stages)
    for s in stages:
        validate_stage(s)
        resolved_frozen = resolve_frozen_core(s.frozen_core, s.atoms)
        print(f"### [Stage {s.stage_id}]: {s.job_name}")
        engine = (engine_label(s) if s.run_type == "OPT" else
                  "ZOPT (Z-matrix, energy-only; %opt_engine zopt)" if s.run_type == "ZOPT" else "-")
        mode = "OPT" if s.run_type == "ZOPT" else s.run_type
        print(f"  Mode:           {mode} (Engine: {engine}) | Method: {s.method} | Sector: {s.system_type}")
        if s.run_type in ("TS", "IRC") and s.system_type in ("ELECTRONIC", "POSITRON") and s.atoms:
            print(f"  Hessian:        {ts_hessian_label(s)}")
        if s.run_type == "FREQ" and s.system_type in ("ELECTRONIC", "POSITRON") and s.atoms:
            if s.freq_mode == "analytic" and not analytic_hessian_available(s):
                print("  Hessian:        ANFREQ: no analytic Hessian for this method (see the error below)")
            else:
                print(f"  Hessian:        {freq_route_label(s)}")
        print(f"  Integrals:      {'Density Fitted (RI)' if s.use_df else 'Conventional (Exact)'}")
        if (getattr(s, "aux_conditioning", None) is not None
                and callable(getattr(s.aux_conditioning, "report", None))):
            print(f"  Auxiliary fit:  {_conditioning_settings(s.aux_conditioning)}; "
                  "fixed choices are recorded and retained across geometries")
        print(f"  Electronic:     Charge={s.charge}, Mult={s.multiplicity}, Basis={basis_groups_label(s.elec_basis_groups, s.atoms) or s.elec_basis}, Frozen={resolved_frozen}")
        if s.use_df and s.atoms:
            syms = [a for a, _ in s.atoms]
            jk = jkfit_set(s, syms)
            if s.system_type == "ELECTRONIC":
                corr = electronic_correlation_set(s, syms)
            elif s.system_type == "POSITRON" and s.n_pos == 1:
                corr = single_positron_correlation_label(s, syms)[0]
            elif s.n_pos == 1:
                corr = s.auxbasis_e or "library default (<basis>-ri)"
            else:
                corr = pair_df_aux(s)["auxbasis_e"]
            if s.jkfit_aux_e is not None:
                how = " (from %jkfit_aux_e)"
            elif jk == DEFAULT_JKFIT:
                how = f" (default: PySCF has no {s.elec_basis}-jkfit)"
            else:
                how = " (matched to %basis)"
            print(f"  Fitting sets:   mean field (JK) = {jk}{how}; correlation = {corr}")
        if s.scan:
            sc = s.scan
            if sc["mode"] == "fragment":
                desc = "fragment gap (largest sequential gap split)"
            elif sc["mode"] == "bond":
                desc = f"bond {sc['atoms'][0]}-{sc['atoms'][1]} (other atoms fixed)"
            elif sc["mode"] == "translate":
                desc = f"translate atoms {sc['range'][0]}-{sc['range'][1]} along {sc['axis']} (displacement)"
            elif sc["mode"] in ("angle", "dihedral"):
                desc = f"{sc['mode']} {'-'.join(map(str, sc['atoms']))} (only atom {sc['atoms'][-1]} moves)"
            else:
                desc = f"Z-matrix variable '{sc['var']}'"
            fit = {None: "no fit", "morse": "Morse fit", "harmonic": "harmonic fit"}[s.scan_fit]
            print(f"  Scan:           {desc}: {sc['start']:.4f} -> {sc['end']:.4f}, {sc['steps']} points; {fit}")
            if len(sc.get("segments", [])) > 1 or sc.get("custom"):
                print("                  points: " + " ".join(f"{v:.4f}" for v in scan_values(sc)))
        if s.system_type == "POSITRON":
            pb = ("per-atom map" if s.pos_basis_spec else basis_groups_label(s.pos_basis_groups, s.atoms)
                  or describe_pos_basis(s.pos_basis_name, s.elec_basis, s.pos_mass_amu))
            print(f"  Positron:       Basis={pb}, Positrons={s.n_pos}, Aux={s.pos_aux}")
            if s.n_pos == 2 and s.use_df:
                print(f"  Pair reference: {s.pair_reference} (%pair_reference; reference_model of the pair DF classes)")
            part = describe_particle(s)
            if part:
                print(f"  Particle:       {part}")
            if s.pos_basis_spec:
                for k, v in s.pos_basis_spec.items():
                    print(f"  Pos Basis Map:  {k}: {describe_pos_basis(v, s.elec_basis, s.pos_mass_amu)}")
            sel = resolve_pos_atom(s.pos_atom_spec, s.atoms) if s.pos_atom_spec else ([
                i for i, b in enumerate(assign_basis_groups(s.pos_basis_groups, s.atoms)[0]) if b is not None]
                if s.pos_basis_groups else None) if s.pos_basis_groups else (
                sorted(i for i, (sym, _) in enumerate(s.atoms)
                       if element_of(sym) in {_pos_map_element(s, k) for k in s.pos_basis_spec}) if s.pos_basis_spec else None)
            if sel is not None:
                print(f"  Pos Atoms:      " + ", ".join(f"{s.atoms[i][0]}{i+1}" for i in sel))
            if s.n_pos >= 2:
                spin = "singlet pair" if not s.pos_spin else f"pos_spin={s.pos_spin} (spin-orbital POSSO CCSD)"
                dfa = pair_df_aux(s)
                print(f"  Pair / DF aux:  {spin}; df_aux={dfa if s.use_df else 'n/a (CONV)'}")
            print(f"  Triples:        compute_eee={s.compute_eee}, compute_t2e1p={s.compute_t2e1p}"
                  + (f", block_size_A={s.block_size_a}" if s.block_size_a else ""))
        elif s.system_type == "PROTON":
            cn = (f", CNEO ({'variant ' + s.cneo_variant if s.method in ('MP2', 'CCSD', 'CCSD(T)') else 'constrained'})"
                  if s.cneo else ", plain NEO")
            print(f"  Quantum Nuc:    Basis={s.nuc_basis}, Centers={s.quantum_nuc}{cn}")
            nm = describe_nuc_masses(s)
            if nm:
                print(f"  Nuclear Masses: {nm}")
            if s.method == "DFT":
                print(f"  DFT Models:     XC={s.xc}, EPC={s.epc}")
        elif s.method == "DFT":
            print(f"  Electronic DFT: XC={s.xc}")
        cap = f", DF cap={s.df_mem_cap_gb:g} GB" if s.df_mem_cap_gb else ""
        print(f"  Resources:      Threads={s.cpu_cores}, MaxMemory={s.memory_mb} MB{cap}")
        print(f"  Atoms:          {len(s.atoms)} | Constraints: {len(s.constraints)} | "
              f"Pinned atoms: {len(s.pinnings)} | Pinned coords: {len(s.pinnings_xyz)} | "
              f"Equivalences: {len(s.equivalences) + len(s.equivalences_xyz)}")
        if getattr(s, "freq_of_opt", None):
            print(f"  Settings:       the frequencies of stage {s.freq_of_opt}'s '! OPT FREQ', at its optimized geometry")
        elif s.inherited_from:
            print(f"  Settings:       continued from stage {s.inherited_from} (no geometry block), with this stage's "
                  f"own lines on top")
        print(f"  Geometry:       {s.geometry_kind or 'inherited from the previous stage'}"
              + (" (written in Bohr, converted to Angstrom)" if s.coord_units == "BOHR" and s.has_explicit_xyz else ""))
        for w in s.warnings:
            print(f"  [Warning] {w}")
        for e in s.errors:
            print(f"  [ERROR] {e}")
            all_errors.append(f"Stage {s.stage_id} ({s.job_name}): {e}")
        print("-" * 80)
    if all_errors:
        print("[Input Pre-Validation Pass] FAILED:")
        for e in all_errors:
            print(f"  - {e}")
        raise InputValidationError("Input validation failed:\n  " + "\n  ".join(all_errors))
    print("[Input Pre-Validation Pass] Validation successful. Proceeding with execution.\n")


# ----------------------------------------------------------------------------
# Building the calculations
# ----------------------------------------------------------------------------

def atoms_to_string(atoms):
    return "; ".join([f"{sym} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}" for sym, c in atoms])


def _positron_basis_and_atoms(stage):
    pos_atom = resolve_pos_atom(stage.pos_atom_spec, stage.atoms)
    if stage.pos_basis_groups:
        return _positron_basis_from_groups(stage, pos_atom)
    if stage.pos_basis_spec:
        pos_b = {}
        present = {element_of(sym) for sym, _ in stage.atoms}
        for k, v in stage.pos_basis_spec.items():
            key = str(k)
            if key.isdigit():
                idx = int(key) - 1
                if not 0 <= idx < len(stage.atoms):
                    raise ValueError(f"positron basis map: atom {key} does not exist")
                key = element_of(stage.atoms[idx][0])
            else:
                key = element_of(key)
            pos_b[key] = resolve_pos_basis(v, stage.elec_basis, stage.pos_mass_amu)
        if pos_atom is None:
            pos_atom = sorted(i for i, (sym, _) in enumerate(stage.atoms) if element_of(sym) in pos_b)
        missing = sorted({element_of(stage.atoms[i][0]) for i in pos_atom} - set(pos_b))
        if missing:
            raise ValueError(f"positron basis map has no entry for {missing}; add them or restrict %pos_atom")
        return pos_b, pos_atom
    return resolve_pos_basis(stage.pos_basis_name, stage.elec_basis, stage.pos_mass_amu), pos_atom


def _positron_basis_from_groups(stage, pos_atom):
    """v04.08 '%pos_basis B1 H B2 {1 2} ...': (pos_basis for build_system, positron atoms). Atoms without a
    basis get no positron functions; %pos_atom, when also given, picks among the atoms that have one. An element
    whose positron atoms all share a basis is keyed by its symbol, otherwise each of its atoms by the library's
    per-centre label <element><k> (k = position in the positron-atom list)."""
    atoms = stage.atoms
    per, problems = assign_basis_groups(stage.pos_basis_groups, atoms)
    if problems:
        raise ValueError("%pos_basis: " + "; ".join(problems))
    have = [i for i, b in enumerate(per) if b is not None]
    if pos_atom is not None:
        lacking = [i + 1 for i in pos_atom if per[i] is None]
        if lacking:
            raise ValueError(f"%pos_atom selects atoms {lacking}, which %pos_basis gives no basis")
        have = sorted(pos_atom)
    eb, _ = electron_basis_layout(stage, atoms)
    e_name = (lambda i: eb if isinstance(eb, str)
              else eb.get(f"{atoms[i][0]}@{i + 1}", eb.get(element_of(atoms[i][0]))))
    by_el = {}
    for i in have:
        by_el.setdefault(element_of(atoms[i][0]), set()).add(per[i])
    pos_b = {}
    for k, i in enumerate(have):
        el = element_of(atoms[i][0])
        data = resolve_pos_basis(per[i], e_name(i), stage.pos_mass_amu)
        pos_b[el if len(by_el[el]) == 1 else f"{el}{k}"] = data
    return pos_b, have


def pos_mass_me(stage):
    """The stage's particle mass in electron masses (the unit of build_system's pos_mass)."""
    return 1.0 if stage.pos_mass_amu is None else stage.pos_mass_amu * AMU_TO_ME


def parse_aux_conditioning(stage, value):
    """Parse an explicit policy name or native constructor settings dictionary."""
    if value.strip().lower() in ("none", "off", "false"):
        stage.aux_conditioning = None
        return
    literal = parse_literal(value)
    options = dict(literal) if isinstance(literal, dict) else {}
    name = str(options.pop("policy", value if not options else "pivoted")).lower()
    cls = {"fixed": FixedAuxiliaryShells, "pivoted": PivotedAuxiliaryShells}.get(name)
    if cls is None or not callable(getattr(cls, "from_report", None)):
        stage.error("%aux_conditioning requires none, fixed, pivoted, or a settings dictionary "
                    "with policy='fixed'/'pivoted', and the native report-restoration API")
        return
    try:
        stage.aux_conditioning = cls(**options)
    except (TypeError, ValueError) as err:
        stage.error(f"%aux_conditioning: {err}")


def _conditioning_problem(stage):
    """Restrict the opt-in policy to the qualified positron DF workflows."""
    policy = getattr(stage, "aux_conditioning", None)
    if policy is None:
        return None
    if (type(policy) not in (FixedAuxiliaryShells, PivotedAuxiliaryShells)
            or not callable(getattr(type(policy), "from_report", None))):
        return "%aux_conditioning needs a qualified native fixed/pivoted policy with from_report"
    if (stage.system_type != "POSITRON" or not stage.use_df or stage.n_pos not in (1, 2)
            or stage.method not in ("HF", "MP2", "CCSD", "CCSD(T)")):
        return ("%aux_conditioning is supported for POSITRON DF HF/MP2/CCSD/CCSD(T), "
                "one or two positrons; existing method/spin restrictions still apply")
    if stage.method == "HF" and stage.n_pos == 2 and stage.pos_spin:
        return ("%aux_conditioning has no fitted space for high-spin HF: this wrapper uses "
                "an exact mean field. Disable conditioning; high-spin DF CC still admits it.")
    return None


def _conditioning_settings(policy):
    """Return declared settings independently of the selected shell state."""
    return {key: value for key, value in policy.report().items() if key != "selections"}


def _conditioning_model_key(stage):
    """Identify a fixed fitted model across geometries and solver-accuracy changes."""
    surface = _surface_definition(stage)
    surface["aux_conditioning"] = _conditioning_settings(stage.aux_conditioning)
    surface.pop("frequency_precision", None)
    return _cache_digest(surface)


def _conditioning_declaration_signature(stage):
    """Bind a persistent selection to source, input and declared start geometry."""
    settings = {name: value for name, value in vars(stage).items()
                if not name.startswith("_") and name not in ("warnings", "errors")}
    declaration = _conditioning_settings(stage.aux_conditioning)
    settings["aux_conditioning"] = declaration
    surface = _surface_definition(stage, atoms_to_string(stage.atoms))
    surface["aux_conditioning"] = declaration
    return _cache_digest("conditioned-stage-v1", _restart_source_identity(), settings, surface)


def _prepare_conditioning(stage):
    """Restore fixed choices before any persistent result or geometry is reused."""
    import json
    policy = getattr(stage, "aux_conditioning", None)
    if policy is None:
        return
    problem = _conditioning_problem(stage)
    if problem:
        raise PreflightError(problem)
    model = _conditioning_model_key(stage)
    context = getattr(stage, "_conditioning_context", None)
    if context is not None:
        if context["policy"] is not policy or context["model"] != model:
            raise PreflightError("A conditioned workflow changed its policy/model. Use a fresh "
                                 "stage and job name; validate continuity before changing fits.")
        return
    signature = _conditioning_declaration_signature(stage)
    path = f"{_stage_prefix(stage)}_stage{stage.stage_id}_aux_conditioning.json"
    artifacts = [_stage_record_path(stage, "done")]
    if stage.run_type == "OPT":
        artifacts.append(f"{_stage_prefix(stage)}_checkpoint.xyz")
    if stage.run_type in ("FREQ", "TS", "IRC"):
        artifacts.append(f"{_stage_prefix(stage)}_freq_points.json")
        import glob
        artifacts.extend(glob.glob(
            glob.escape(f"{_stage_prefix(stage)}_stage{stage.stage_id}_freq_native_") + "*.json"))
    if stage.run_type == "ZOPT":
        artifacts.append(f"{_stage_prefix(stage)}_zopt_history.json")
    existing_results = any(os.path.exists(item) for item in artifacts)
    new_chained_stage = (getattr(stage, "_conditioning_inherited", False)
        and not any(os.path.exists(_stage_record_path(stage, kind)) for kind in ("started", "done")))
    saved_digest = None
    if os.path.exists(path):
        try:
            with open(path) as handle:
                saved = json.load(handle)
            if (saved["format"] != "fixed-auxiliary-shells-v1"
                    or saved["signature"] != signature or saved["model"] != model
                    or saved["report_digest"] != _cache_digest(saved["report"])):
                raise ValueError("source, input, model or report identity differs")
            restored = type(policy).from_report(saved["report"])
            if _conditioning_settings(restored) != _conditioning_settings(policy):
                raise ValueError("policy settings differ")
            if (policy.report()["selections"] and policy.report() != restored.report()):
                raise ValueError("existing in-memory choices differ")
            if existing_results and not restored.report()["selections"]:
                raise ValueError("no selected shells were saved for existing results")
        except (OSError, ValueError, TypeError, KeyError) as err:
            raise PreflightError(f"Cannot restore fixed auxiliary choices from {path}: {err}. "
                                 "Use a fresh job name for a new fitting workflow.") from err
        policy = stage.aux_conditioning = restored
        saved_digest = saved["report_digest"]
        print(f"  [Auxiliary fit] Restored fixed shell choices from {path}")
    elif existing_results and not new_chained_stage:
        raise PreflightError(f"Missing fixed auxiliary selection record {path} for existing "
                             "results/checkpoint. Use a fresh job name; do not reselect on resume.")
    stage._conditioning_context = dict(policy=policy, model=model, signature=signature,
                                       path=path, saved_digest=saved_digest)
    _save_conditioning(stage)


def _save_conditioning(stage):
    """Persist newly admitted choices atomically; unchanged reports do no disk work."""
    import json
    if getattr(stage, "aux_conditioning", None) is None:
        return
    _prepare_conditioning(stage)
    context = getattr(stage, "_conditioning_context", None)
    if context is None:
        return
    report = stage.aux_conditioning.report()
    digest = _cache_digest(report)
    if context["saved_digest"] == digest:
        return
    record = dict(format="fixed-auxiliary-shells-v1", signature=context["signature"],
                  model=context["model"], report=report, report_digest=digest)
    path = context["path"]
    with open(path + ".tmp", "w") as handle:
        json.dump(record, handle, indent=1, allow_nan=False)
    os.replace(path + ".tmp", path)
    context["saved_digest"] = digest
    if report["selections"]:
        kept = sum(item["retained_functions"] for item in report["selections"].values())
        original = sum(item["input_functions"] for item in report["selections"].values())
        print(f"  [Auxiliary fit] Fixed {kept}/{original} functions across "
              f"{len(report['selections'])} recorded topologies; saved {path}")


def _conditioning_continuation(stage):
    """Capture independent choices for a matching stage inheriting an optimized geometry."""
    policy = getattr(stage, "aux_conditioning", None)
    return None if policy is None else (_conditioning_model_key(stage), policy.report())


def _inherit_conditioning(stage, previous):
    """Carry selected shells into a matching chained calculation without sharing mutable state."""
    policy = getattr(stage, "aux_conditioning", None)
    if policy is not None and previous is not None:
        key, report = previous
        if key == _conditioning_model_key(stage):
            stage.aux_conditioning = type(policy).from_report(report)
            stage._conditioning_inherited = True


def safe_build_system(stage, atom_str):
    pos_b, pos_atom = _positron_basis_and_atoms(stage)
    atom_str, ebasis = electron_mol_input(stage, atom_str)
    kwargs = dict(basis=ebasis, pos_basis=pos_b, charge=stage.charge,
                  spin=stage.multiplicity - 1, n_pos=stage.n_pos, pos_atom=pos_atom)
    if stage.pos_spin and stage.n_pos >= 2:
        kwargs["pos_spin"] = stage.pos_spin
    if stage.pos_mass_amu is not None:
        kwargs["pos_mass"] = pos_mass_me(stage)
    if stage.quantum_site is not None:
        # the particle replaces the nucleus of this H atom (library "proton-limit" mode): the nuclear
        # charge is removed, the electron basis stays, the particle basis sits there only
        kwargs["quantum_site"] = stage.quantum_site - 1
    system = build_system(atom=atom_str, **kwargs)
    # POSHF's initial-guess RHF inherits mol_e.max_memory (default 4000 MB);
    # too little makes PySCF fall back to a direct-Vee path that segfaults.
    system.mol_e.max_memory = stage.memory_mb
    return system


def _df_recipe(stage):
    """The DF recipe the positron CC of this stage uses: one positron, the
    single_positron_dfccsd keywords (auxbasis_e only when given, so the
    library default '<basis>-ri' applies otherwise); two or more, the
    POSDFCCSD df_aux dict with its correlation set resolved (pair_df_aux)."""
    if stage.n_pos == 1:
        recipe = {"pos_aux": stage.pos_aux}
        if stage.auxbasis_e:
            recipe["auxbasis_e"] = stage.auxbasis_e
        return recipe
    return pair_df_aux(stage)


PAIR_LIGHT_Z_MAX = 10          # Preserve the historical recipe only for more than two positrons.


def _element_map(choice, elements):
    """A fitting-set choice (name or element-keyed dict) as {element: name}."""
    return {e: (choice.get(e) if isinstance(choice, dict) else choice) for e in elements}


def pair_correlation_set(stage, symbols):
    """Choose matched RI for a two-positron pair, with JK fallback where unavailable.

    The pair's SCF reference is independent of this correlation fit. Systems
    with more than two positrons retain their prior coupled-reference recipe;
    the pair-model migration does not qualify a new many-positron Hamiltonian.
    """
    if stage.n_pos == 2:
        return matched_fitting_set(stage.elec_basis, symbols, "-ri", jkfit_set(stage, symbols))
    elements = sorted({element_of(sym) for sym in symbols if not is_dummy_label(sym)})
    light = [e for e in elements if pyscf_elements.charge(e) <= PAIR_LIGHT_Z_MAX]
    heavy = [e for e in elements if pyscf_elements.charge(e) > PAIR_LIGHT_Z_MAX]
    mapping = {}
    if light:
        mapping.update(_element_map(matched_fit(stage, light, "-ri",
                                                        jkfit_set(stage, light)), light))
    if heavy:
        mapping.update(_element_map(jkfit_set(stage, heavy), heavy))
    names = set(mapping.values())
    return names.pop() if len(names) == 1 else mapping


def _pair_reference_model(stage):
    """reference_model for the pair DF classes (POSDFCCSD, POSSODFCCSD, POSDFMP2). Two positrons: %pair_reference,
    default 'scf' (pyscf_neo_cc 0.1.0.post20261008: the actual SCF reference energy and Fock operator); 'legacy'
    reproduces the 2026-09-23 library's numbers (v04.08 and older) for comparison. Other counts: 'legacy', as in
    the library (it keeps the historical operator there)."""
    return getattr(stage, "pair_reference", "scf") if stage.n_pos == 2 else "legacy"


def pair_df_aux(stage):
    """Preserve explicit two-positron recipes and resolve only an omitted basis.

    Scalar names, per-element maps and user basis data remain as supplied for
    a pair. Gradient admission checks actual metrics; auxiliary accuracy is a
    separate question. Larger positron counts retain the historical resolver.
    """
    recipe = dict(stage.df_aux) if stage.df_aux is not None else {}
    symbols = [sym for sym, _ in stage.atoms if not is_dummy_label(sym)]
    if stage.n_pos == 2:
        if not recipe.get("auxbasis_e"):
            recipe["auxbasis_e"] = pair_correlation_set(stage, symbols)
        return recipe
    if recipe.get("auxbasis_e"):
        recipe["auxbasis_e"] = fix_broken_dz_jkfit(recipe["auxbasis_e"], symbols)[0]
    given = recipe.get("auxbasis_e")
    if not given:
        recipe["auxbasis_e"] = pair_correlation_set(stage, symbols)
    elif is_ri_set_name(given):
        heavy = sorted({element_of(sym) for sym in symbols
                        if pyscf_elements.charge(element_of(sym)) > PAIR_LIGHT_Z_MAX})
        if heavy:
            elements = sorted({element_of(sym) for sym in symbols})
            mapping = {e: given for e in elements}
            mapping.update(_element_map(jkfit_set(stage, heavy), heavy))
            recipe["auxbasis_e"] = mapping
    return recipe


def make_poshf(system, stage, use_df=None):
    """POSHF for this stage. JK fitting (%jkfit_aux_e) only on the DF route.
    CONV calculations (a CONV stage, or the %conv_check cross-check) use an
    exact mean field: pyscf_neo_cc (since 2026-09-23) refuses canonical (exact-
    integral) gradients over a JK-fitted reference, and CONV should be exact
    throughout. The spin-orbital (pos_spin != 0) CC classes also require an
    exact-integral reference."""
    effective_use_df = getattr(stage, 'use_df', True) if use_df is None else use_df
    _prepare_conditioning(stage)
    policy = getattr(stage, "aux_conditioning", None) if effective_use_df else None
    options = {"aux_conditioning": policy} if policy is not None else {}
    if not effective_use_df or (stage.n_pos >= 2 and stage.pos_spin):
        return POSHF(system, recover_on_scan=True, **options)
    symbols = [system.mol_e.atom_pure_symbol(i) for i in range(system.mol_e.natm)]
    stage._jk_resolved = jk = jkfit_set(stage, symbols)
    # the CC's DF recipe is fitted before the first SCF: one SCF, and no
    # exact four-index e-p tensor (43 GB for a six-centre positron basis)
    return POSHF(system, jkfit_aux_e=jk, df_aux=_df_recipe(stage), recover_on_scan=True, **options)


def _ensure_poshf_converged(mf, label="POSHF"):
    """Recover a failed POSHF iterate once, or stop before reporting its energy."""
    if getattr(mf, "converged", False):
        return mf
    recover = getattr(mf, "recover_scf", None)
    if not callable(recover):
        raise PreflightError(
            f"{label} did not converge; this library has no POSHF.recover_scf. "
            "Use the qualified POSHF recovery implementation before continuing.")
    print(f"  [SCF] {label} did not converge; recovering the current orbitals "
          "with the same operators and convergence tolerances...")
    try:
        recover()
    except (NotImplementedError, RuntimeError, ValueError) as err:
        raise PreflightError(f"{label} recovery failed: {err}") from err
    report = getattr(mf, "scf_recovery", None) or {}
    if not getattr(mf, "converged", False) or not report.get("accepted", False):
        raise PreflightError(f"{label} recovery did not admit a converged stationary reference.")
    print(f"  [SCF] {label} converged after orbital recovery: E = {mf.e_tot:.12f} Ha")
    return mf


def poshf_gradient_preflight(mf):
    """Admit only this HF reference's derivative metrics before its SCF solve."""
    if _native_poshf_preflight is None:
        raise PreflightError("POSHF analytic gradients require the qualified HF derivative implementation.")
    try:
        _native_poshf_preflight(mf)
    except (ValueError, RuntimeError, NotImplementedError) as err:
        raise PreflightError(f"POSHF gradient admission failed: {err}") from err
    return mf


def optimizer_reference_accuracy(stage, mf):
    """Solve a native positron optimizer's first reference at its scanner accuracy."""
    if (stage.system_type == "POSITRON" and stage.run_type == "OPT"
            and stage.opt_engine != "WRAPPER" and stage.method in ("MP2", "CCSD", "CCSD(T)")):
        mf.conv_tol = min(mf.conv_tol, 1e-13)
        mf.conv_tol_grad = min(mf.conv_tol_grad, 1e-12 if stage.n_pos == 1 else 1e-11)
        mf.max_cycle = max(mf.max_cycle, 500)
    return mf


def run_poshf(mf, label="POSHF"):
    """Run ordinary POSHF once and recover its failed iterate before returning."""
    mf.kernel()
    return _ensure_poshf_converged(mf, label)


def build_positron_cc(mf, stage, use_df=None, *, gradient_required=None):
    cc_obj = _freq_tighten(_build_positron_cc(mf, stage, use_df), stage)
    cc_obj.frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    cc_obj = positron_gradient_preflight(stage, cc_obj, use_df, gradient_required)
    _save_conditioning(stage)
    return cc_obj


def _build_positron_cc(mf, stage, use_df=None):
    """Positron CC object for this stage. n_pos == 1: single-positron DF or
    canonical CCSD. n_pos >= 2 singlet pair: POSDFCCSD on the DF reference
    (df_aux must be a dict -- None would be the library's exact-RI validation
    mode, not DF) or POSCCSD. n_pos >= 2 with pos_spin != 0: spin-orbital
    POSSODFCCSD / POSSOCCSD."""
    effective_use_df = getattr(stage, 'use_df', True) if use_df is None else use_df
    if stage.n_pos == 1:
        if effective_use_df:
            return single_positron_dfccsd(mf, jkfit_aux_e=getattr(stage, "_jk_resolved", stage.jkfit_aux_e),
                                          **_df_recipe(stage))
        return single_positron_ccsd(mf)
    df_aux = pair_df_aux(stage)
    if stage.pos_spin:
        frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
        if effective_use_df:
            return POSSODFCCSD(mf, frozen=frozen, df_aux=df_aux, reference_model=_pair_reference_model(stage))
        return POSSOCCSD(mf, frozen=frozen)
    if effective_use_df:
        if getattr(mf, "df_aux", None) is None:      # a mean field built without the recipe
            apply_df_reference(mf, df_aux=dict(df_aux))
        return POSDFCCSD(mf, df_aux=dict(df_aux), reference_model=_pair_reference_model(stage))
    return _near_miss_retry(POSCCSD(mf))


TCC_RETRY_TOL = 1e-8        # max |residual| accepted on the retry (the library's default is 1e-9)
TCC_RETRY_CYCLES = 600


def _near_miss_retry(cc_obj):
    """Two-positron CONV CCSD (POSCCSD): its solver stops at max |residual| < 1e-9 and raises after 300 cycles,
    discarding the amplitudes. On OCN-type dianions + 2 e+ it stalled at |r| = 3-4e-9 (a near miss: the energy
    error of such a residual is far below 1e-10 Ha), and the %conv_check energy was lost. v04.05: on that error,
    solve once more to max |residual| < 1e-8 with up to 600 cycles, and say so in the log."""
    orig = cc_obj.kernel

    def kernel(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        except RuntimeError as err:
            if "tcc-factored not converged" not in str(err):
                raise
            tol, cyc = cc_obj.conv_tol, cc_obj.max_cycle
            print(f"  [Notice] two-positron CONV CCSD stopped short ({err}; limit {tol:.0e} in {cyc} cycles); "
                  f"solving again to |r| < {TCC_RETRY_TOL:.0e} with up to {TCC_RETRY_CYCLES} cycles (the energy "
                  f"error of such a residual is far below 1e-10 Ha).")
            cc_obj.conv_tol, cc_obj.max_cycle = max(tol, TCC_RETRY_TOL), max(cyc, TCC_RETRY_CYCLES)
            out = orig(*args, **kwargs)
            print(f"  [Notice] two-positron CONV CCSD converged on the retry (|r| < {cc_obj.conv_tol:.0e}).")
            return out
    cc_obj.kernel = kernel
    return cc_obj


def safe_compute_triples(cc_obj, compute_eee=True, compute_t2e1p=True, block_size_A=None):
    """Evaluate one admitted (T) correction without dropping channel requests or retrying failures."""
    if not (compute_eee or compute_t2e1p):
        return 0.0
    parameters = inspect.signature(cc_obj.ccsd_t).parameters
    accepted = set(parameters)
    mixed = next((key for key in ("compute_t2e1p", "compute_eep") if key in accepted), None)
    if not compute_eee and "compute_eee" not in accepted:
        raise NotImplementedError("This triples energy does not support a mixed-only correction.")
    if not compute_t2e1p and mixed is None:
        raise NotImplementedError("This triples energy does not support an eee-only correction.")
    kwargs = {}
    if "compute_eee" in accepted:
        kwargs["compute_eee"] = compute_eee
    if mixed is not None:
        kwargs[mixed] = compute_t2e1p
    forwards = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    if block_size_A is not None and ("block_size_A" in accepted or forwards):
        kwargs["block_size_A"] = block_size_A
    res = cc_obj.ccsd_t(**kwargs)
    # v04.08: keep the pure-electron triples for the electronic / positronic energy partition
    bd = res if isinstance(res, dict) else getattr(res, "breakdown", None)
    if isinstance(bd, dict) and bd.get("eee") is not None:
        cc_obj._wrapper_t_eee = float(bd["eee"])
        cc_obj._wrapper_t_terms = {k: float(bd[src]) for k, src in
                                   (("eee", "eee"), ("eep", "eep"), ("eep", "t2e1p"), ("epp", "epp"), ("ppp", "ppp"))
                                   if bd.get(src) is not None}
    if isinstance(res, dict):
        if 'et' not in res and 'e_t' not in res:
            raise ValueError("The triples kernel returned no et/e_t energy.")
        return float(res['et'] if 'et' in res else res['e_t'])
    elif isinstance(res, (tuple, list)):
        return float(res[0])
    elif hasattr(res, 'e_t'):
        return float(res.e_t)
    elif hasattr(res, 'et'):
        return float(res.et)
    elif res is None and hasattr(cc_obj, 'e_t'):
        return float(cc_obj.e_t)
    return float(res)


def _positron_triples_problem(stage):
    """Return a diagnostic for channel choices unsupported by the positron energy kernel."""
    if stage.system_type != "POSITRON" or stage.method != "CCSD(T)":
        return None
    if stage.n_pos == 1 and stage.compute_eee and not stage.compute_t2e1p:
        return ("Single-positron CCSD(T) supports full or mixed-only triples, not eee-only: "
                "the mixed energy cannot be disabled. Enable %compute_t2e1p.")
    if stage.n_pos >= 2 and stage.compute_eee != stage.compute_t2e1p:
        return ("Positron-pair CCSD(T) supports the full correction only; its gradient flags "
                "do not select separate energy channels. Set both triples flags true, "
                "or both false for CCSD.")
    return None


def _triples(cc_obj, stage):
    problem = _positron_triples_problem(stage)
    if problem:
        raise PreflightError(problem)
    return safe_compute_triples(cc_obj, compute_eee=stage.compute_eee,
                                compute_t2e1p=stage.compute_t2e1p,
                                block_size_A=stage.block_size_a)


def positron_cc_gradient(stage, cc_obj):
    """Differentiate the declared surface, preserving the combined-spin solver order."""
    problem = _positron_gradient_problem(stage) or _positron_triples_problem(stage)
    if problem:
        raise PreflightError(problem)
    with_t = stage.method == "CCSD(T)" and (stage.compute_eee or stage.compute_t2e1p)
    options = dict(compute_eee=stage.compute_eee,
                   compute_t2e1p=stage.compute_t2e1p) if with_t else {}
    # Preserve the combined-spin warm start before constructing its gradient.
    # That factory also declares gradient_required on the correlation object.
    if with_t and stage.n_pos >= 2 and stage.pos_spin:
        cc_obj.solve_lambda()
    grad = cc_obj.Gradients(**options)
    if not with_t:
        cc_obj.solve_lambda()
    return grad.kernel()


def _keep_input_label(ref, sym):
    """The input label when the optimizer's pure symbol would lose its meaning: a ghost (the label
    style the user wrote) or a NEO isotope mark (H+ deuteron, H* antimuon: 'H' would turn it into a
    proton in the final energy and in the next stage)."""
    if element_of(ref) != element_of(sym):
        return sym                                # order changed; trust the optimizer
    if is_ghost_label(ref) or re.search(r'[+*]', str(ref)):
        return ref
    return sym


def extract_optimized_atoms(opt_result, reference_atoms):
    new_atoms = []
    if hasattr(opt_result, "atom_coords") and hasattr(opt_result, "atom_pure_symbol"):
        coords = opt_result.atom_coords() * BOHR_TO_ANGSTROM
        same = len(coords) == len(reference_atoms)
        for i in range(len(coords)):
            sym = opt_result.atom_pure_symbol(i)
            if same:
                sym = _keep_input_label(reference_atoms[i][0], sym)
            new_atoms.append((sym, tuple(coords[i])))
        return new_atoms
    elif hasattr(opt_result, "_atom"):
        same = len(opt_result._atom) == len(reference_atoms)
        for i, (sym, (x, y, z)) in enumerate(opt_result._atom):
            if same:
                sym = _keep_input_label(reference_atoms[i][0], sym)
            new_atoms.append((sym, (x * BOHR_TO_ANGSTROM, y * BOHR_TO_ANGSTROM, z * BOHR_TO_ANGSTROM)))
        return new_atoms
    return reference_atoms


def _electronic_mf(stage, atom_str, use_df):
    atom_str, ebasis = electron_mol_input(stage, atom_str)
    mol = gto.M(atom=atom_str, basis=ebasis, charge=stage.charge,
                spin=stage.multiplicity - 1, verbose=0)
    open_shell = stage.multiplicity != 1
    if stage.method == "DFT":
        mf = (dft.UKS if open_shell else dft.RKS)(mol, xc=stage.xc)
    else:
        mf = (scf.UHF if open_shell else scf.RHF)(mol)
    if use_df:
        symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        mf = mf.density_fit(auxbasis=jkfit_set(stage, symbols))
    return mf, open_shell


def _electronic_cc(mf, stage):
    """CCSD on an electronic mean field. On a density-fitted mean field the correlation is fitted with
    the correlation set (matched RI set) rather than the mean-field JK set, where the CC class allows it
    (closed-shell DF-RCCSD; stock PySCF has no analytic DF-CCSD gradient, so gradients of DF stages are
    numerical and stay consistent with these energies)."""
    cc_obj = cc.CCSD(mf)
    cc_obj.frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    if getattr(mf, "with_df", None) is not None and getattr(cc_obj, "with_df", None) is not None:
        mol = mf.mol
        corr = electronic_correlation_set(stage, [mol.atom_pure_symbol(i) for i in range(mol.natm)])
        cc_obj.with_df = pyscf_df.DF(mol, auxbasis=corr)
    return _freq_tighten(cc_obj, stage)


def _neo_mol(stage, atom_str, verbose=None):
    """The NEO Mole of a PROTON stage. With %nuc_mass the quantum nuclei get those masses (a per-atom
    mass list; classical atoms keep the library's table value) and their nuclear basis is scaled by
    sqrt(m / m_proton), the library's own rule for the H+ / H* / H# symbols."""
    atom_str, ebasis = electron_mol_input(stage, atom_str)
    kw = dict(atom=atom_str, basis=ebasis, nuc_basis=stage.nuc_basis,
              quantum_nuc=stage.quantum_nuc, charge=stage.charge, spin=stage.multiplicity - 1)
    if verbose is not None:
        kw["verbose"] = verbose
    if not stage.nuc_mass or all(m is None for m, _ in stage.nuc_mass):
        mol = NEOMole()
        mol.build(**kw)
        return mol
    probe = NEOMole()
    probe.build(**dict(kw, verbose=0))
    masses = [float(x) for x in probe.mass]
    scale = {}
    for idx, (m, _name) in zip(stage.quantum_nuc, stage.nuc_mass):
        if m is not None:
            # v04.09 fix: the library already scales an H+ / H* / H# basis by sqrt(m_isotope / m_p); scaling it
            # again relative to the proton applied that factor twice. Apply only the remaining sqrt(m / m_isotope).
            lib_scaled = bool(re.search(r"[+*#]", str(stage.atoms[idx][0])))
            ratio = nuclear_basis_scale(m)
            if lib_scaled:
                ratio /= nuclear_basis_scale(masses[idx])
            masses[idx] = m
            scale[idx] = ratio
    del probe
    mol = ScaledNEOMole()
    mol.nuc_basis_scale = scale
    mol.build(mass=masses, **kw)
    return mol


def proton_mean_field(stage, atom_str, use_df=None, verbose=0):
    """The converged PROTON-sector mean field at `atom_str`.
    Plain NEO: NEO-HF, or NEO-KS with %xc and %epc (exact integrals, as before v04.04).
    CNEO (%cneo true): the library's CDFT with xc='HF' (CNEO-HF) or %xc + %epc (CNEO-DFT), converged like
    pyscf_neo_cc.cneo.cneo_hf (DIIS to 1e-7, then the ARH second-order solver to 1e-12 for HF). A DF
    correlated stage (MP2, CCSD, CCSD(T)) gets the DF-JK reference the CNEO DF gradient needs
    (cd_density_fit_unified: mean field = the matched JK-fit set, correlation = %auxbasis_e or the
    matched RI set). The molecule is the wrapper's (%nuc_mass, isotope symbols, %quantum_nuc)."""
    use_df = stage.use_df if use_df is None else use_df
    mol = _neo_mol(stage, atom_str, verbose=0)
    dft_kw = {"xc": stage.xc, **({"epc": stage.epc} if stage.epc else {})}
    if stage.cneo:
        mf = CNEO_CDFT(mol, **(dft_kw if stage.method == "DFT" else {"xc": "HF"}))
        if use_df and stage.method in ("MP2", "CCSD", "CCSD(T)"):
            symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
            mf = cd_density_fit_unified(mf, jkfit_aux_e=jkfit_set(stage, symbols),
                                        auxbasis_e=electronic_correlation_set(stage, symbols))
        mf.max_memory = stage.memory_mb
        mf.verbose = verbose
        mf.conv_tol = 1e-13 if stage.method != "DFT" else 1e-11
        mf.max_cycle = 400
        if stage.method == "DFT":
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("CNEO-DFT did not converge")
        else:
            mf.conv_tol_grad = 1e-7                       # DIIS warm-up, then ARH (cneo_hf's recipe)
            mf.scf()
            mf = _cneo_accept(_cneo_tighten(mf, 1e-12, verbose), "CNEO-HF", 1e-8)
        return mf
    if stage.method == "DFT":
        mf = NEOKS(mol, **dft_kw)
    else:
        mf = NEOHF(mol)
    mf.max_memory = stage.memory_mb
    mf.verbose = verbose
    mf.conv_tol = 1e-11
    mf.kernel()
    if not mf.converged:
        print(f"  [WARNING] NEO mean field is NOT converged (E = {mf.e_tot:.12f} Ha). Results of this stage are unreliable.")
    return mf


def proton_correlated(stage, mf, use_df=None, *, gradient_required=None):
    """(CC object, CCSD energy, (T) or None) of a PROTON CCSD / CCSD(T) stage on mean field `mf`.
    CNEO: CNEOCCSD / CNEODFCCSD with %cneo_variant (default ccd: the constraint on the correlated
    density; ucd keeps it only on the HF reference, and its optimized geometries fall back towards plain
    NEO's, measured HF: CNEO-CCSD(T) 0.9677 A with ccd, 0.8957 A with ucd). CCSD(T) with ccd roots the
    multiplier on the CCSD(T) density (constraint_level='ccsd_t'), which its analytic gradient needs.
    %frozen applies (it was ignored by the PROTON single points up to v04.04)."""
    use_df = stage.use_df if use_df is None else use_df
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    if stage.cneo:
        kw = {"constraint_level": "ccsd_t"} if (stage.method == "CCSD(T)" and stage.cneo_variant == "ccd") else {}
        cc_obj = (CNEODFCCSD if use_df else CNEOCCSD)(mf, frozen=frozen, variant=stage.cneo_variant, **kw)
    else:
        cc_obj = (NEO_DFCCSD if use_df else NEO_CCSD)(mf)
        cc_obj.frozen = frozen
    proton_gradient_preflight(stage, cc_obj, use_df, gradient_required)
    cc_obj.verbose = 0
    cc_obj.kernel()
    e_t = None
    if stage.method == "CCSD(T)":
        if stage.cneo:
            cc_obj.ccsd_t()
            e_t = float(cc_obj.e_t)
        else:
            e_t = _triples(cc_obj, stage)
    return cc_obj, float(cc_obj.e_tot), e_t


def proton_gradient(stage, mf, cc_obj=None):
    """Analytic PROTON gradient (Hartree/Bohr): the library's NEO gradient for the mean field
    (pyscf_neo_cc._neo.Gradients; NEO-HF / NEO-KS have no .Gradients() method, which is why up to v04.04
    these fell back to numerical WOPT), the CC gradient (Lambda first) otherwise, (T) included."""
    if cc_obj is None:
        return np.asarray(NEOGradients(mf).kernel())
    cc_obj.solve_lambda()
    if stage.cneo:
        kw = dict(compute_eee=stage.compute_eee, compute_t2e1p=stage.compute_t2e1p) \
            if stage.method == "CCSD(T)" else {}
        return np.asarray(cc_obj.Gradients(**kw).kernel())
    grad_obj = cc_obj.Gradients()
    if stage.method == "CCSD(T)":
        grad_obj.compute_eee = stage.compute_eee
        grad_obj.compute_t2e1p = stage.compute_t2e1p
    return np.asarray(grad_obj.kernel())


def electronic_ccsdt_gradient(cc_obj):
    """Analytic closed-shell CCSD(T) gradient (Hartree/Bohr). PySCF's
    grad.ccsd_t needs the CCSD(T) lambda amplitudes; without them it
    silently uses the CCSD lambdas and is off by ~1e-5 Ha/Bohr."""
    from pyscf.cc import ccsd_t_lambda
    from pyscf.grad import ccsd_t as ccsd_t_grad
    eris = cc_obj.ao2mo()
    conv, l1, l2 = ccsd_t_lambda.kernel(cc_obj, eris, cc_obj.t1, cc_obj.t2, verbose=0)
    if not conv:
        raise RuntimeError("CCSD(T) lambda equations did not converge")
    return ccsd_t_grad.Gradients(cc_obj).kernel(cc_obj.t1, cc_obj.t2, l1, l2, eris=eris)


def electronic_cc_gradient(stage, cc_obj, open_shell):
    """Analytic ELECTRONIC CCSD / CCSD(T) gradient (Hartree/Bohr), or None where PySCF has no exact one:
    density-fitted CC (PySCF's CC gradient formula on DF amplitudes is not the derivative of the DF energy:
    measured 5e-6 Ha/Bohr for H2O/6-31G, 2.6e-5 (CCSD) and 5.2e-5 (CCSD(T)) for CN-/aug-cc-pVTZ) and
    open-shell CCSD(T). PySCF 2.14's CC classes have nuc_grad_method() but no Gradients(); up to v04.04 the
    wrapper called Gradients(), so its electronic CCSD gradients were silently numerical (v04.05)."""
    if stage.use_df:
        return None
    if stage.method == "CCSD(T)":
        return None if open_shell else np.asarray(electronic_ccsdt_gradient(cc_obj))
    return np.asarray(cc_obj.nuc_grad_method().kernel())


def build_mp2(stage, atom_str, use_df=None, *, gradient_required=None, solve=True):
    """Build this stage's mean field and MP2 object; solve=False defers only the MP2 kernel
    to a native optimizer. The mean field is always converged. ELECTRONIC: PySCF MP2 / UMP2 on
    an exact mean field, or DFMP2 / DFUMP2 on the JK-fitted one with the correlation fitted by the RI set
    matched to %basis (as the electronic DF-CCSD). POSITRON (DF only): one positron, pyscf_neo_cc DFMP2
    on a POSHF built with the stage's DF recipe; two, POSDFMP2 (closed-shell pair) on the %df_aux recipe.
    PROTON (DF only): DFMP2 on a NEO-HF set up by cd_density_fit_unified (mean field: the matched JK-fit
    set; correlation: %auxbasis_e or the matched RI set), the route the library's DF gradient needs.
    %frozen applies in every sector."""
    use_df = stage.use_df if use_df is None else use_df
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    verbose = stage.verbose if stage.verbose_explicit else 0
    if stage.system_type == "ELECTRONIC":
        from pyscf import mp as pyscf_mp
        mf, open_shell = _electronic_mf(stage, atom_str, use_df)
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        mf.verbose = verbose
        mf.kernel()
        if use_df:
            from pyscf.mp.dfmp2 import DFMP2 as _DFMP2
            from pyscf.mp.dfump2 import DFUMP2 as _DFUMP2
            mp = (_DFUMP2 if open_shell else _DFMP2)(mf, frozen=frozen)
            mol = mf.mol
            corr = electronic_correlation_set(stage, [mol.atom_pure_symbol(i) for i in range(mol.natm)])
            mp.with_df = pyscf_df.DF(mol, auxbasis=corr)
        else:
            mp = (pyscf_mp.UMP2 if open_shell else pyscf_mp.MP2)(mf, frozen=frozen)
    elif stage.system_type == "POSITRON":
        system = safe_build_system(stage, atom_str)
        mf = make_poshf(system, stage, use_df=True)
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        mf.verbose = verbose
        optimizer_reference_accuracy(stage, mf)
        run_poshf(mf)
        if stage.n_pos == 1:
            mp = NEO_DFMP2(mf, frozen=frozen)
        else:
            mp = POSDFMP2(mf, frozen=frozen, df_aux=pair_df_aux(stage), reference_model=_pair_reference_model(stage))
    elif stage.system_type == "PROTON" and stage.cneo:
        mf = proton_mean_field(stage, atom_str, use_df, verbose=verbose)
        mp = (CNEODFMP2 if use_df else CNEOMP2)(mf, frozen=frozen, variant=stage.cneo_variant)
    elif stage.system_type == "PROTON":
        mol = _neo_mol(stage, atom_str, verbose=0)
        symbols = [mol.atom_pure_symbol(i) for i in range(mol.natm)]
        jk = jkfit_set(stage, symbols)
        corr = electronic_correlation_set(stage, symbols)
        mf = cd_density_fit_unified(NEOHF(mol), auxbasis_e=corr, jkfit_aux_e=jk, n_recipe="pavosevic")
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        mf.verbose = verbose
        mf.kernel()
        if not mf.converged:
            print(f"  [WARNING] NEO-HF is NOT converged (E = {mf.e_tot:.12f} Ha). Results of this stage are unreliable.")
        mp = NEO_DFMP2(mf, frozen=frozen)
    else:
        raise ValueError(f"Unknown sector {stage.system_type}")
    if stage.system_type == "POSITRON":
        positron_gradient_preflight(stage, mp, use_df, gradient_required)
        _save_conditioning(stage)
    elif stage.system_type == "PROTON":
        proton_gradient_preflight(stage, mp, use_df, gradient_required)
    mp.verbose = verbose
    if solve:
        mp.kernel()
    return mf, mp


def mp2_gradient(stage, mp, use_df=None):
    """Analytic MP2 gradient (Hartree/Bohr), or None where no route has one (PySCF's electronic DF-MP2)."""
    use_df = stage.use_df if use_df is None else use_df
    if stage.system_type == "ELECTRONIC" and use_df:
        return None
    return np.asarray(mp.Gradients().kernel())


def numerical_gradient(stage, atoms, h=0.0015):
    """Central-difference gradient (Hartree/Bohr) of the stage energy at `atoms` (step h in A)."""
    coords = np.array([c for _, c in atoms], dtype=float)
    labels = [a for a, _ in atoms]
    g = np.zeros_like(coords)
    for i in range(len(coords)):
        for j in range(3):
            e = []
            for sign in (1, -1):
                c = coords.copy()
                c[i, j] += sign * h
                e.append(run_single_point_calculation(stage, list(zip(labels, map(tuple, c))),
                                                      use_df=stage.use_df))
            g[i, j] = (e[0] - e[1]) / (2 * h) * BOHR_TO_ANGSTROM
    return g


_COMPONENTS = {}          # latest and lowest-energy components per DF/CONV route (at most four records)
_RESTART_SOURCE_IDENTITY = None
_RESTART_SOURCE_STATS = {}


def _file_digest(path):
    """Hash a source/basis file in bounded blocks, without retaining its contents."""
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache_value(value):
    """Encode small declared settings; reject unidentified executable/custom objects."""
    import json
    from types import SimpleNamespace
    if isinstance(value, np.generic):
        return _cache_value(value.item())
    if isinstance(value, np.ndarray):
        return {"array": value.tolist(), "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, dict):
        pairs = [[_cache_value(k), _cache_value(v)] for k, v in value.items()]
        return {"mapping": sorted(pairs, key=lambda row: json.dumps(row[0], sort_keys=True))}
    if isinstance(value, (tuple, list)):
        return [_cache_value(x) for x in value]
    if isinstance(value, (set, frozenset)):          # v04.09 merge: e.g. %scan_print (a set since v04.08)
        return {"set": sorted((_cache_value(x) for x in value), key=lambda row: json.dumps(row, sort_keys=True))}
    if isinstance(value, SimpleNamespace):
        return {"namespace": _cache_value(vars(value))}
    if hasattr(value, "_df_config_identity") and callable(getattr(value, "report", None)):
        return {"auxiliary_policy": _cache_value(value.report())}
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"No cache identity for {type(value).__module__}.{type(value).__name__}")


def _cache_digest(*values):
    """Hash portable settings without rounding coordinates or hashing fitted arrays."""
    import hashlib
    import json
    text = json.dumps(_cache_value(values), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _restart_source_identity():
    """Identify wrapper/dependency source and basis data once for on-disk reuse.

    In-memory results stay within this loaded wrapper. Only persistent records
    need this filesystem scan; ordinary energy/gradient calls do not perform it.
    Source installations must remain fixed while a wrapper process is running.
    """
    global _RESTART_SOURCE_IDENTITY
    if _RESTART_SOURCE_IDENTITY is None:
        import hashlib
        import platform
        import scipy
        from pathlib import Path
        wall, cpu = time.perf_counter(), time.process_time()
        roots = {"pyscf": Path(pyscf.__file__).parent, "numpy": Path(np.__file__).parent,
                 "scipy": Path(scipy.__file__).parent}
        if pyscf_neo_cc is not None:
            roots["pyscf_neo_cc"] = Path(pyscf_neo_cc.__file__).parent
        sources = {"wrapper": _file_digest(__file__)}
        count, size = 1, os.path.getsize(__file__)
        suffixes = {".py", ".so", ".dylib", ".dat", ".gbs", ".basis", ".json",
                    ".txt", ".npy", ".npz", ".pkl"}
        for label, root in sorted(roots.items()):
            digest = hashlib.sha256()
            for directory, dirs, names in os.walk(root):
                dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x != "__pycache__")
                for name in sorted(names):
                    path = Path(directory) / name
                    if path.suffix in suffixes:
                        digest.update(str(path.relative_to(root)).encode())
                        digest.update(_file_digest(path).encode())
                        count, size = count + 1, size + path.stat().st_size
            sources[label] = digest.hexdigest()
        _RESTART_SOURCE_IDENTITY = _cache_digest(
            "wrapper-restart-v1", sources, tuple(sys.version_info[:3]), sys.flags.optimize,
            (platform.system(), platform.release(), platform.version(), platform.machine()),
            pyscf.__version__, np.__version__, scipy.__version__)
        _RESTART_SOURCE_STATS.update(files=count, streamed_bytes=size,
            wall_seconds=time.perf_counter()-wall, cpu_seconds=time.process_time()-cpu)
    return _RESTART_SOURCE_IDENTITY


def _loaded_source_identity():
    """Separate in-process results after wrapper/factory reload or operator replacement."""
    names = ("POSHF", "POSCCSD", "POSSOCCSD", "POSDFCCSD", "POSSODFCCSD", "POSDFMP2",
             "NEO_CCSD", "NEO_DFCCSD", "NEO_DFMP2", "NEOHF", "NEOKS", "CNEO_CDFT",
             "CNEOCCSD", "CNEODFCCSD", "CNEOMP2", "CNEODFMP2", "single_positron_ccsd",
             "single_positron_dfccsd", "_electronic_mf", "_electronic_cc", "proton_mean_field",
             "energy_and_forces", "run_single_point_components")
    return (id(_surface_definition), tuple((name, id(globals().get(name)),
        id(getattr(globals().get(name), "kernel", None))) for name in names))


def _surface_definition(stage, atom_str=None, use_df=None):
    """Describe the declared physical surface and numerical settings, excluding solved state."""
    fields = ("method", "system_type", "elec_basis", "charge", "multiplicity", "frozen_core",
              "n_pos", "pos_spin", "pos_basis_name", "pos_basis_spec", "pos_atom_spec", "pos_aux",
              "jkfit_aux_e", "df_aux", "auxbasis_e", "pos_mass_amu", "quantum_site", "nuc_basis",
              "quantum_nuc", "nuc_mass", "cneo", "cneo_variant", "xc", "epc", "compute_eee",
              "compute_t2e1p", "block_size_a", "df_mem_cap_gb", "memory_mb", "cpu_cores",
              "aux_conditioning", "elec_basis_groups", "pos_basis_groups", "pair_reference")
    settings = {name: getattr(stage, name, None) for name in fields}
    settings["use_df"] = getattr(stage, "use_df", None) if use_df is None else bool(use_df)
    settings["frequency_precision"] = getattr(stage, "run_type", None) in ("FREQ", "TS", "IRC")
    # A user basis file can change without a change to its declared path.
    files = {}

    def basis_files(value):
        if isinstance(value, dict):
            for child in value.values():
                basis_files(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                basis_files(child)
        elif isinstance(value, str):
            try:
                if os.path.isfile(value):
                    files[value] = _file_digest(value)
            except OSError:
                pass

    for name in ("elec_basis", "pos_basis_name", "pos_basis_spec", "jkfit_aux_e", "df_aux",
                 "auxbasis_e", "nuc_basis", "elec_basis_groups", "pos_basis_groups"):
        basis_files(settings[name])
    settings["basis_files"] = files
    settings["thread_environment"] = {name: os.environ.get(name) for name in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
    if atom_str is not None:
        settings["geometry"] = _geometry_key(atom_str)
    return settings


def clear_result_caches():
    """Discard in-memory results after custom runtime operator/solver overrides."""
    _COMPONENTS.clear()
    _HESSIANS.clear()
    _FREQ_DETAILS.clear()


def _geometry_key(atom_str):
    """Retain the coordinates actually supplied to the calculation, without extra rounding."""
    key = []
    for part in atom_str.split(";"):
        t = part.split()
        if len(t) >= 4:
            key.append((t[0], *(float(x) for x in t[1:4])))
    return tuple(key)


def _record_components(stage, atom_str, use_df, hf, mp2=None, ccsd=None, t=None, partition=None):
    """Remember the energy components of one calculation (v04.05), so the final-geometry report can print
    HF, MP2 or CCSD, (T) and the total even when the optimizer only kept the total. `partition`
    (v04.08, POSITRON): the electronic / positronic split from positron_energy_partition()."""
    _save_conditioning(stage)
    total = ccsd + (t or 0.0) if ccsd is not None else (mp2 if mp2 is not None else hf)
    key = _cache_digest(_loaded_source_identity(), _surface_definition(stage, atom_str, use_df))
    components = {
        "HF": hf, "MP2": mp2, "CCSD": ccsd, "(T)": t,
        "CCSD(T)": (ccsd + t) if (ccsd is not None and t is not None) else None, "total": total,
        "partition": partition}
    # Retain the best point as well as the latest trial: WOPT/ZOPT may reject
    # their last trial and return a previously visited lower-energy geometry.
    owner = (id(stage), _cache_digest(_loaded_source_identity(), _surface_definition(stage, use_df=use_df)))
    previous = _COMPONENTS.get(bool(use_df))
    best = previous["best"] if previous is not None and previous["owner"] == owner else None
    point = (key, components)
    if best is None or components["total"] < best[1]["total"]:
        best = point
    _COMPONENTS[bool(use_df)] = {"owner": owner, "latest": point, "best": best}


def _ee_ovov(cc_obj, nfz, no, nv):
    """Electron (ia|jb) over the active occupied / virtual MOs, from the CC object's own integrals (exact
    or DF, so the split adds up to its E_corr). The tcc classes keep active-only integrals under a frozen
    core, the NEO classes all MOs."""
    def sl(n):
        off = nfz if n == nfz + no + nv else 0
        return slice(off, off + no), slice(off + no, off + no + nv)
    ints = getattr(cc_obj, "_integrals", None) or {}
    L = ints.get("L_ee_pure_mo")
    if L is None:
        L = (getattr(cc_obj, "_L", None) or {}).get("L_ee")
    if L is not None:
        o, v = sl(L.shape[1])
        return np.einsum("Kia,Kjb->iajb", L[:, o, v], L[:, o, v], optimize=True)
    g = ints.get("eri_ee_chem")
    if g is None:
        g = cc_obj._eris["g"]
    o, v = sl(g.shape[0])
    return g[o, v, o, v]


def positron_energy_partition(mf, cc_obj=None, e_t=None, mp=None):
    """Electronic / positronic split of a POSITRON energy (v04.08). Electronic: every term built from
    electron and nuclear integrals only (nuclear repulsion, T_e, V_ne, V_ee; the electron-only CCSD
    correlation; the eee triples). Positronic: every term with at least one positron integral (T_p, V_np,
    V_pp, V_ep; the rest of E_corr, i.e. the e-e+ and e+-e+ correlation and the positron-field part of the
    singles; the eep/epp/ppp triples). Both parts are evaluated in the orbitals of the full complex, so the
    electronic part is not the energy of the bare molecule or anion. Closed-shell electrons only."""
    sy = mf.sys
    c_e = mf.mo_coeff["e"]
    dm_e, dm_p = mf.make_dm_e(c_e), np.asarray(mf.make_dm_p(mf.mo_coeff["n0"]))
    dm_pt = dm_p[0] + dm_p[1] if dm_p.ndim == 3 else dm_p
    veff_e, veff_p = mf.veff_e(dm_e), np.asarray(mf.veff_p(dm_p))

    def tr(a, b):
        return float(np.einsum("ij,ji->", a, b))
    t_e = tr(sy.mol_e.intor_symmetric("int1e_kin"), dm_e)
    t_p = tr(sy.mol_p.intor_symmetric("int1e_kin"), dm_pt)
    h_p = tr(mf.h_p, dm_pt)
    v_pp = 0.5 * float(np.einsum("sPQ,sQP->", veff_p, dm_p)) if dm_p.ndim == 3 else 0.5 * tr(veff_p, dm_p)
    p = {"E_nuc": float(sy.mol_e.energy_nuc()) + float(getattr(mf, "_e_field_nuc", 0.0)),
         "T_e": t_e, "V_ne": tr(mf.h_e, dm_e) - t_e, "V_ee": 0.5 * tr(veff_e, dm_e),
         "T_p": t_p, "V_np": h_p - t_p, "V_pp": v_pp, "V_ep": tr(mf.v_e_from_p(dm_pt), dm_e)}
    p["HF_e"] = p["E_nuc"] + p["T_e"] + p["V_ne"] + p["V_ee"]
    p["HF_p"] = p["T_p"] + p["V_np"] + p["V_pp"] + p["V_ep"]
    if cc_obj is not None:
        # E_corr(e) = 2 f(ee)_ia t_ia + sum [2(ia|jb) - (ib|ja)] (t_ijab + t_ia t_jb), f(ee) = h_e + J - K/2
        # of the electrons alone; E_corr(p) = E_corr - E_corr(e)
        if hasattr(cc_obj, "t1e"):                      # NEO CCSD (one positron): t1[i,a], t2[i,j,a,b]
            t1, t2 = cc_obj.t1e, cc_obj.t2ee
        else:                                           # tcc layout (positron pair): t1[a,i], t2[a,i,b,j]
            t1, t2 = cc_obj.amps["t1"].T, cc_obj.amps["t2"].transpose(1, 3, 0, 2)
        nocc = sy.nelectron // 2
        no, nv = t1.shape
        nfz = nocc - no
        co, cv = c_e[:, nfz:nocc], c_e[:, nocc:nocc + nv]
        ovov = _ee_ovov(cc_obj, nfz, no, nv)
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        e_ee = (2.0 * float(np.einsum("ia,ia->", co.T @ (mf.h_e + veff_e) @ cv, t1))
                + float(np.einsum("iajb,ijab->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), tau, optimize=True)))
        p["CCSD_e"], p["CCSD_p"] = e_ee, float(cc_obj.e_corr) - e_ee
        if e_t is not None:
            eee = getattr(cc_obj, "_wrapper_t_eee", None)
            if eee is None:
                p["note"] = "this (T) route gives no eee breakdown; (T) is not split and not in the Total row"
            else:
                p["T3_e"], p["T3_p"] = eee, float(e_t) - eee
                p["T3_terms"] = dict(getattr(cc_obj, "_wrapper_t_terms", {}))
    if mp is not None:
        e_ee = getattr(mp, "e_corr_ee", None)
        if e_ee is not None:
            p["MP2_e"], p["MP2_p"] = float(e_ee), float(mp.e_corr) - float(e_ee)
            p["MP2_tot"] = float(mp.e_tot)
    return p


def _partition_mp2(stage, mf):
    """%mp2_partition (v04.08): the positron DF-MP2 of a CC stage, on the CC's own converged POSHF (no new
    SCF; the stage's frozen core). pyscf_neo_cc has positron MP2 only with density fitting, so on a CONV
    mean field the MP2 correlation uses fitted integrals while the CCSD uses exact ones."""
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    if stage.n_pos == 1:
        mp = NEO_DFMP2(mf, frozen=frozen)
    else:
        mp = POSDFMP2(mf, frozen=frozen, df_aux=pair_df_aux(stage),
                      reference_model=_pair_reference_model(stage))
    mp.verbose = 0
    mp.kernel()
    return mp


def safe_positron_partition(stage, mf, cc_obj=None, e_t=None, mp=None):
    """positron_energy_partition() for a POSITRON stage, or None (other sectors, the open-shell positron
    pair, or any failure: the partition is a report, it never stops a run)."""
    if stage.system_type != "POSITRON" or getattr(stage, "pos_spin", 0):
        return None
    try:
        p = positron_energy_partition(mf, cc_obj, e_t, mp)
    except Exception as err:
        print(f"  [Notice] Electronic / positronic energy partition unavailable ({type(err).__name__}: {err})")
        return None
    if mp is None and cc_obj is not None and getattr(stage, "mp2_partition", False):
        try:     # after the HF / CC partition: nothing below may change what it read from the mean field
            mp = _partition_mp2(stage, mf)
            p.update({k: v for k, v in positron_energy_partition(mf, mp=mp).items() if k.startswith("MP2")})
        except Exception as err:
            print(f"  [Notice] %mp2_partition: positron DF-MP2 failed ({type(err).__name__}: {err})")
    return p


def partition_lines(p, num=lambda x, spec: format(x, spec)):
    """Text lines of an electronic / positronic partition: decimal point for run.log, the module's num()
    (decimal comma with --decimal-comma) for the summary files."""
    if not p:
        return []
    rows = [("HF (mean field)", p["HF_e"], p["HF_p"])]
    mp2_side = "MP2_e" in p and "CCSD_e" in p     # %mp2_partition: MP2 shown beside CCSD(T), not summed
    if "MP2_e" in p and not mp2_side:
        rows.append(("MP2 correlation", p["MP2_e"], p["MP2_p"]))
    if "CCSD_e" in p:
        rows.append(("CCSD correlation", p["CCSD_e"], p["CCSD_p"]))
    if "T3_e" in p:
        rows.append(("(T) correction", p["T3_e"], p["T3_p"]))
    if len(rows) > 1:
        rows.append(("Total", sum(r[1] for r in rows), sum(r[2] for r in rows)))
    out = ["Electronic / positronic energy partition (positronic = every term with a positron integral):",
           f"  {'':<18}{'electronic':>20}{'positronic':>20}{'sum':>20}"]
    for name, a, b in rows:
        out.append(f"  {name:<18}{num(a, '20.10f')}{num(b, '20.10f')}{num(a + b, '20.10f')}")
    if mp2_side:
        out.append(f"  {'MP2 correlation':<18}{num(p['MP2_e'], '20.10f')}{num(p['MP2_p'], '20.10f')}"
                   f"{num(p['MP2_e'] + p['MP2_p'], '20.10f')}   (%mp2_partition, DF; not in Total)")
        out.append(f"  {'MP2 total':<18}{num(p['MP2_e'] + p['HF_e'], '20.10f')}"
                   f"{num(p['MP2_p'] + p['HF_p'], '20.10f')}{num(p['MP2_tot'], '20.10f')}")
    if p.get("T3_terms"):
        out.append("  (T) terms: " + "  ".join(f"{k} {num(v, '.10f')}" for k, v in p["T3_terms"].items()))
    out.append("  HF terms, electronic: "
               f"E_nuc {num(p['E_nuc'], '.10f')}  T_e {num(p['T_e'], '.10f')}  "
               f"V_ne {num(p['V_ne'], '.10f')}  V_ee {num(p['V_ee'], '.10f')}")
    out.append("  HF terms, positronic: "
               f"T_p {num(p['T_p'], '.10f')}  V_np {num(p['V_np'], '.10f')}  "
               f"V_pp {num(p['V_pp'], '.10f')}  V_ep {num(p['V_ep'], '.10f')}")
    if p.get("note"):
        out.append(f"  ({p['note']})")
    return out


def cc_mp2_energy(cc_obj, mf):
    """MP2 total energy that PySCF's electronic CCSD / UCCSD computed for its first-guess amplitudes (same
    orbitals, frozen core and integrals, DF or exact, as the CC), or None when the CC object has none
    (pyscf_neo_cc's positron and NEO classes do not keep it)."""
    emp2 = getattr(cc_obj, "emp2", None)
    return None if emp2 is None else float(mf.e_tot) + float(emp2)


def lookup_components(atoms, use_df, stage=None):
    """Return independent components only for the same declared surface and exact input geometry.

    The optional stage=None namespace supports explicit untyped records only;
    it never retrieves a typed stage's calculation by geometry alone.
    """
    key = _cache_digest(_loaded_source_identity(),
                        _surface_definition(stage, atoms_to_string(atoms), use_df))
    saved = _COMPONENTS.get(bool(use_df))
    if saved is not None:
        for name in ("latest", "best"):
            point = saved[name]
            if point[0] == key:
                return dict(point[1])
    return None


def report_components(stage, atoms, use_df):
    """Print the energy components at the final geometry (v04.05; up to v04.04 only the total was printed,
    so a CCSD(T) optimization never showed its CCSD energy). Returns the components or None."""
    comps = lookup_components(atoms, use_df, stage=stage)
    label = "DF/RI" if use_df else "CONV/Exact"
    if comps is None:
        if stage.method != "HF":
            print(f"  [Energy Components] {label}: not available (energy reused from an earlier run)")
        return None
    if stage.method == "HF":
        for line in partition_lines(comps.get("partition")):
            print(f"  [{label}] " + line if line.startswith("Electronic") else "      " + line)
        return comps
    print(f"  [Energy Components] {label}, final geometry:")
    print(f"      Mean field (HF)      : {comps['HF']:.12f} Ha")
    if comps["MP2"] is not None:
        print(f"      MP2                  : {comps['MP2']:.12f} Ha  (E_corr = {comps['MP2'] - comps['HF']:.12f})")
    if comps["CCSD"] is not None:
        print(f"      CCSD                 : {comps['CCSD']:.12f} Ha  (E_corr = {comps['CCSD'] - comps['HF']:.12f})")
    if comps["(T)"] is not None:
        print(f"      (T) correction       : {comps['(T)']:.12f} Ha")
        print(f"      CCSD(T)              : {comps['CCSD(T)']:.12f} Ha")
    for line in partition_lines(comps.get("partition")):
        print("      " + line)
    return comps


def run_single_point_components(stage, atoms, use_df=True):
    """Energies at `atoms`: {'HF', 'MP2', 'CCSD', 'CCSD(T)', 'total'} (missing = None); remembered for the
    final-geometry report."""
    out = _single_point_components(stage, atoms, use_df)
    t = (out['CCSD(T)'] - out['CCSD']) if (out['CCSD(T)'] is not None and out['CCSD'] is not None) else None
    _record_components(stage, atoms_to_string(atoms), use_df, out['HF'], mp2=out['MP2'], ccsd=out['CCSD'], t=t,
                       partition=out.get('partition'))
    return out


def _single_point_components(stage, atoms, use_df=True):
    atom_str = atoms_to_string(atoms)
    out = {"HF": None, "MP2": None, "CCSD": None, "CCSD(T)": None, "total": None}
    if stage.method == "MP2":
        mf, mp = build_mp2(stage, atom_str, use_df)
        out["HF"] = float(mf.e_tot)
        out["MP2"] = out["total"] = float(mp.e_tot)
        out["partition"] = safe_positron_partition(stage, mf, mp=mp)
        _eda_hf_terms(stage, mf, out)
        return out

    if stage.system_type == "POSITRON":
        system = safe_build_system(stage, atom_str)
        mf = make_poshf(system, stage, use_df=use_df)
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        mf.verbose = stage.verbose if stage.verbose_explicit else 0
        run_poshf(mf)
        out["HF"] = out["total"] = float(mf.e_tot)
        cc_obj = e_t = None
        if stage.method in ["CCSD", "CCSD(T)"]:
            cc_obj = build_positron_cc(mf, stage, use_df=use_df)
            cc_obj.verbose = stage.verbose if stage.verbose_explicit else 0
            cc_obj.kernel()
            out["CCSD"] = out["total"] = float(cc_obj.e_tot)
            if stage.method == "CCSD(T)":
                e_t = _triples(cc_obj, stage)
                out["CCSD(T)"] = out["total"] = out["CCSD"] + e_t
        out["partition"] = safe_positron_partition(stage, mf, cc_obj, e_t)
        if cc_obj is not None and "eda" in getattr(stage, "scan_print", ()):
            if out["partition"] and out["partition"].get("MP2_tot") is not None:
                out["MP2"] = out["partition"]["MP2_tot"]
            elif not stage.pos_spin:
                try:      # %scan_print eda: the DF-MP2 of the same POSHF (pyscf_neo_cc's CC keeps none)
                    mp = _partition_mp2(stage, mf)
                    out["MP2"] = float(mp.e_tot)
                    if out["partition"]:
                        out["partition"].update({k: v for k, v in positron_energy_partition(mf, mp=mp).items()
                                                 if k.startswith("MP2")})
                except Exception as err:
                    print(f"  [Notice] %scan_print eda: positron DF-MP2 failed ({type(err).__name__}: {err})")
        return out

    if stage.system_type == "PROTON":
        mf = proton_mean_field(stage, atom_str, use_df)
        out["HF"] = out["total"] = float(mf.e_tot)
        if stage.method in ["CCSD", "CCSD(T)"]:
            _, e_cc, e_t = proton_correlated(stage, mf, use_df)
            out["CCSD"] = out["total"] = e_cc
            if e_t is not None:
                out["CCSD(T)"] = out["total"] = e_cc + e_t
        return out

    if stage.system_type == "ELECTRONIC":
        mf, _ = _electronic_mf(stage, atom_str, use_df)
        mf.verbose = 0
        mf.kernel()
        out["HF"] = out["total"] = float(mf.e_tot)
        _eda_hf_terms(stage, mf, out)
        if stage.method in ["CCSD", "CCSD(T)"]:
            cc_obj = _electronic_cc(mf, stage)
            cc_obj.verbose = 0
            cc_obj.kernel()
            out["MP2"] = cc_mp2_energy(cc_obj, mf)
            out["CCSD"] = out["total"] = float(cc_obj.e_tot)
            if stage.method == "CCSD(T)":
                out["CCSD(T)"] = out["total"] = out["CCSD"] + _triples(cc_obj, stage)
        return out
    raise ValueError(f"Unknown sector {stage.system_type}")


# ---------------------------------------------------------------------------------------------------------------
# v04.05: FREQ -- harmonic frequencies from the nuclear Hessian at the stage geometry
# ---------------------------------------------------------------------------------------------------------------
FREQ_STEP_GRADIENT_BOHR = 0.005     # central differences of analytic gradients
FREQ_STEP_ENERGY_BOHR = 0.01        # second differences of energies (no analytic gradient)


def _positron_analytic_hessian_available(stage):
    """Recognize the admitted conventional CCSD surface without constructing integrals."""
    if (stage.system_type != "POSITRON" or stage.method != "CCSD" or stage.use_df
            or stage.n_pos != 1 or stage.multiplicity != 1 or stage.pos_spin
            or stage.quantum_site is not None):
        return False
    centres = resolve_pos_atom(stage.pos_atom_spec, stage.atoms)
    if centres is None:
        centres = list(range(len(stage.atoms)))
        if stage.pos_basis_spec:
            elements = set()
            for key in stage.pos_basis_spec:
                key = str(key)
                if key.isdigit():
                    index = int(key)-1
                    if not 0 <= index < len(stage.atoms):
                        return False
                    key = stage.atoms[index][0]
                elements.add(element_of(key))
            centres = [i for i, (symbol, _) in enumerate(stage.atoms)
                       if element_of(symbol) in elements]
    return len(centres) == 1 and callable(getattr(NEO_CCSD, "Hessian", None))


def analytic_hessian_available(stage):
    """Admit electronic HF/DFT or conventional single-carrier one-positron CCSD."""
    return ((stage.system_type == "ELECTRONIC" and stage.method in ("HF", "DFT"))
            or _positron_analytic_hessian_available(stage))


def freq_route(stage):
    """How a FREQ stage gets its Hessian: 'analytic' (an admitted public Hessian), 'gradient' (central
    differences of the analytic gradient) or 'energy' (second differences of energies). ! FREQ takes the
    analytic Hessian when there is one; ! NUMFREQ always a numerical one; ! ANFREQ only the analytic one."""
    mode = getattr(stage, "freq_mode", "auto")
    if mode == "analytic" or (mode == "auto" and analytic_hessian_available(stage)):
        return "analytic"
    return "gradient" if analytic_gradient_available(stage)[0] else "energy"


def freq_route_label(stage, route=None):
    route = route or freq_route(stage)
    asked = {"analytic": " (ANFREQ)", "numerical": " (NUMFREQ)"}.get(getattr(stage, "freq_mode", "auto"), "")
    if route == "analytic":
        backend = "pyscf_neo_cc single-positron CCSD" if stage.system_type == "POSITRON" else "PySCF"
        return f"analytic ({backend})" + asked
    h = stage.freq_step or (FREQ_STEP_GRADIENT_BOHR if route == "gradient" else FREQ_STEP_ENERGY_BOHR)
    n = 3 * len(stage.atoms)
    runs = 2 * n + 1 if route == "gradient" else 1 + 2 * n + 2 * n * (n - 1)
    what = "analytic-gradient differences" if route == "gradient" else "energy second differences"
    backend = "; public NumericalHessian" if _uses_native_frequency(stage) else ""
    if stage.system_type == "POSITRON" and stage.block_size_a is not None:
        backend = "; wrapper stencil preserves explicit triples block size"
    return f"{what}, step {h:g} bohr, {runs} {'gradient' if route == 'gradient' else 'energy'} runs{asked}{backend}"


def _validate_freq(stage):
    if stage.run_type != "FREQ":
        if stage.freq_step is not None:
            stage.warn("%freq_step only acts in FREQ stages -- ignored")
        return
    if stage.system_type not in ("ELECTRONIC", "POSITRON"):
        stage.error(f"FREQ: the ELECTRONIC and POSITRON sectors only (not {stage.system_type}): moving a quantum "
                    f"nucleus' basis centre is not a vibration")
    if any(is_ghost_label(sym) for sym, _ in stage.atoms):
        stage.error("FREQ: ghost atoms have no mass; remove them from a FREQ stage")
    if len(stage.atoms) < 2:
        stage.error("FREQ: a single atom has no vibrations")
    if stage.constraints or stage.pinnings or stage.pinnings_xyz or stage.equivalences or stage.equivalences_xyz:
        stage.warn("FREQ: constraints, pins and equivalences are ignored (every atom moves)")
    if stage.freq_mode == "analytic" and not analytic_hessian_available(stage):
        stage.error(f"ANFREQ: no analytic Hessian for {stage.method} in the {stage.system_type} sector with "
                    "these settings. Supported: electronic HF/DFT and conventional restricted CCSD with "
                    "one positron on one classical basis centre. DF, pair, multicentre, triples and "
                    "quantum-site analytic positron Hessians are unavailable; use NUMFREQ.")
        return
    route = freq_route(stage) if stage.system_type in ("ELECTRONIC", "POSITRON") else None
    if route == "energy":
        why = analytic_gradient_available(stage)[1]
        stage.warn(f"FREQ: no analytic gradient ({why}), so the Hessian comes from second differences of "
                   f"energies ({freq_route_label(stage)}); converge the geometry with the same method first")
    elif route == "gradient" and len(stage.atoms) > 2:
        stage.warn(f"FREQ: {freq_route_label(stage)}")


def _freq_tighten(obj, stage):
    """Tighter CC convergence for FREQ, TS and IRC stages (Hessians by finite differences and saddle-point
    searches need converged energies and gradients)."""
    if getattr(stage, "run_type", None) in ("FREQ", "TS", "IRC") and obj is not None:
        for k, v in (("conv_tol", 1e-10), ("conv_tol_normt", 1e-8)):
            if hasattr(obj, k):
                setattr(obj, k, v)
    return obj


def _freq_cache(stage, route, h):
    """Finished displacements of this FREQ stage (SLURM / --resume / --stage-records only), so a requeued job
    does not repeat them: <job>_freq_points.json, keyed by the stage signature, route and step."""
    import json
    _prepare_conditioning(stage)
    path = f"{_stage_prefix(stage)}_freq_points.json"
    key = {"signature": stage_signature(stage), "route": route, "step_bohr": h} if STAGE_RECORDS else None
    points = {}
    if STAGE_RECORDS and os.path.exists(path):
        try:
            with open(path) as f:
                rec = json.load(f)
            if all(rec.get(k) == v for k, v in key.items()):
                points = rec.get("points", {})
                if points:
                    print(f"[Freq] Reusing {len(points)} finished displacements from {path}")
        except (OSError, ValueError):
            points = {}

    def save():
        if not STAGE_RECORDS:
            return
        _save_conditioning(stage)
        current_key = (dict(key, signature=stage_signature(stage))
                       if getattr(stage, "aux_conditioning", None) is not None else key)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(current_key, points=points), f)
        os.replace(tmp, path)
    return points, save


_HESSIANS = {}            # only the latest identified Hessian, reusable by a matching later stage
_FREQ_DETAILS = {}        # only its small native raw matrix and diagnostics; no scanner owners


def _uses_native_frequency(stage):
    """Use the public driver where its options express the complete declared calculation."""
    return (stage.system_type == "POSITRON" and stage.n_pos in (1, 2)
            and stage.method in ("HF", "MP2", "CCSD", "CCSD(T)")
            and stage.block_size_a is None)


def _numerical_positron_method(stage, route):
    """Prepare public native owners and fitting choices without a seed SCF or CC solve."""
    problem = _positron_triples_problem(stage)
    if problem:
        raise PreflightError(problem)
    if NumericalHessian is None:
        raise PreflightError("Positron frequencies require the public NumericalHessian API.")
    mf = make_poshf(safe_build_system(stage, atoms_to_string(stage.atoms)), stage)
    mf.max_memory, mf.verbose = stage.memory_mb, 0
    # Configure the actual reference before constructing the public single-particle
    # classes. Their convenience factories expect an already converged mean field.
    if mf.df_aux is not None:
        mf.apply_df_aux()
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms)
    if stage.method == "HF":
        poshf_gradient_preflight(mf)
        method = mf
    elif stage.n_pos == 1:
        cls = NEO_DFMP2 if stage.method == "MP2" else NEO_DFCCSD if stage.use_df else NEO_CCSD
        method = cls(mf, frozen=frozen)
    elif stage.method == "MP2":
        method = POSDFMP2(mf, frozen=frozen, df_aux=pair_df_aux(stage), reference_model=_pair_reference_model(stage))
    else:
        method = _build_positron_cc(mf, stage)
        method.frozen = frozen
    if method is not mf:
        _freq_tighten(method, stage)
        positron_gradient_preflight(stage, method, gradient_required=route == "gradient")
        method.verbose = 0
    _save_conditioning(stage)
    return method


def _native_frequency_checkpoint(stage, route, step):
    """Name a source/input-bound native checkpoint without overwriting earlier geometries."""
    if not STAGE_RECORDS:
        return None
    identity = _cache_digest("native-frequency-v1", stage_signature(stage), route, step,
                             freq_masses(stage), True)
    return f"{_stage_prefix(stage)}_stage{stage.stage_id}_freq_native_{identity}.json"


def _native_positron_frequency(stage, route, step):
    """Run one public native stencil, keeping only its small result and raw diagnostics."""
    method = _numerical_positron_method(stage, route)
    checkpoint = _native_frequency_checkpoint(stage, route, step)
    options = {}
    if stage.method == "CCSD(T)" and (stage.compute_eee or stage.compute_t2e1p):
        options = (dict(compute_eee=stage.compute_eee, compute_t2e1p=stage.compute_t2e1p)
                   if route == "gradient" else dict(compute_t=True))
    driver = NumericalHessian(method, step=step, scheme=route, warm_start=True,
        masses=freq_masses(stage), chkfile=checkpoint,
        restart=checkpoint is not None and os.path.exists(checkpoint),
        **{route + "_options": options})
    estimate = driver.estimate()
    print(f"  [Freq] Public NumericalHessian: {estimate}; step {step:g} bohr")
    if checkpoint:
        print(f"  [Freq] Native checkpoint: {checkpoint}; restart rechecks the reference")
    hessian = np.asarray(driver.kernel())
    if not driver.converged:
        raise RuntimeError("Public NumericalHessian did not converge")
    details = dict(raw_hessian=np.array(driver.raw_hessian, copy=True),
                   diagnostics=copy.deepcopy(driver.diagnostics), estimate=estimate,
                   evaluations=driver.evaluation_count, checkpoint=checkpoint)
    _FREQ_DETAILS.clear()
    _FREQ_DETAILS[_hessian_key(stage, freq_route(stage))] = details
    diagnostics = driver.diagnostics
    print(f"  [Freq] {driver.evaluation_count} evaluations; raw antisymmetry "
          f"{diagnostics['antisymmetry_max']:.2e}, translation "
          f"{diagnostics['translation_max']:.2e}, rotation covariance "
          f"{diagnostics['rotation_covariance_max']:.2e} Ha/bohr^2")
    return (hessian.transpose(0, 2, 1, 3).copy(), float(driver.reference_energy),
            np.array(driver.reference_gradient, copy=True), route)


def freq_hessian_details(stage):
    """Return owned native diagnostics and raw (atom, axis, atom, axis) matrix, if retained."""
    return copy.deepcopy(_FREQ_DETAILS.get(_hessian_key(stage, freq_route(stage))))


def _hessian_key(stage, route):
    """Everything that defines the surface: a Hessian is reused only when all of it matches."""
    steps = (stage.freq_step or FREQ_STEP_GRADIENT_BOHR, stage.freq_step or FREQ_STEP_ENERGY_BOHR)
    return _cache_digest(_loaded_source_identity(), id(_freq_hessian),
        _surface_definition(stage, atoms_to_string(stage.atoms)), route, steps)


def freq_hessian(stage):
    """(Hessian (natm, natm, 3, 3) in Ha/bohr^2, reference energy, gradient at the reference (Ha/bohr) or None,
    route used). A Hessian already computed in this job at the same geometry with the same settings (e.g. by the
    TS stage's frequency check, for the IRC stage after it) is reused."""
    _prepare_conditioning(stage)
    requested_route = freq_route(stage)
    key = _hessian_key(stage, requested_route)
    if key in _HESSIANS:
        H, e0, g0, route, where = _HESSIANS[key]
        how = {"analytic": "analytic", "gradient": "numerical (from analytic gradients)",
               "energy": "numerical (from energies)"}[route]
        print(f"  [Hessian] Reusing the {how} Hessian computed at this geometry in {where}")
        return H.copy(), e0, None if g0 is None else g0.copy(), route
    _FREQ_DETAILS.clear()
    H, e0, g0, route = _freq_hessian(stage)
    # First admission may populate a fixed auxiliary selection; record its final identity.
    key = _hessian_key(stage, requested_route)
    _HESSIANS.clear()
    _HESSIANS[key] = (H.copy(), e0, None if g0 is None else np.array(g0, copy=True), route,
                      f"stage {stage.stage_id} ('{stage.job_name}')")
    return H, e0, g0, route


def _analytic_positron_frequency(stage):
    """Run one admitted public CCSD Hessian and release its numerical owners on return."""
    if not _positron_analytic_hessian_available(stage):
        raise PreflightError("Analytic positron frequencies require the public conventional "
                             "single-carrier one-positron CCSD Hessian; use NUMFREQ.")
    mf = make_poshf(safe_build_system(stage, atoms_to_string(stage.atoms)), stage)
    mf.max_memory, mf.verbose = stage.memory_mb, 0
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-12, 500
    run_poshf(mf)
    cc = build_positron_cc(mf, stage, gradient_required=True)
    cc.conv_tol = cc.lambda_conv_tol = 1e-12
    cc.conv_tol_normt = 1e-11
    cc.max_cycle = cc.lambda_max_cycle = 1000
    checkpoint = None
    if STAGE_RECORDS:
        identity = _cache_digest("native-analytic-frequency-v1", stage_signature(stage),
                                 freq_masses(stage), "full", True)
        checkpoint = f"{_stage_prefix(stage)}_freq_analytic_{identity}.npz"
    # The public constructor checks the actual reference (including field and
    # integral representation) before the first CC or Lambda iteration.
    driver = cc.Hessian(mode="full", chkfile=checkpoint, low_mem=True)
    cc.kernel()
    if not cc.converged:
        raise PreflightError("Analytic positron Hessian: CCSD did not converge.")
    cc.solve_lambda()
    if not cc.lambda_converged:
        raise PreflightError("Analytic positron Hessian: Lambda did not converge.")
    print(f"  [Freq] Public analytic single-positron CCSD Hessian; one SCF/CC/Lambda solve, "
          f"{3*len(stage.atoms)} response columns.")
    if checkpoint:
        print(f"  [Freq] Analytic column checkpoint: {checkpoint}; restart reconverges the reference and CC/Lambda.")
    H = np.asarray(driver.kernel())
    g = np.asarray(cc.Gradients().kernel())
    if (H.shape != (len(stage.atoms), 3, len(stage.atoms), 3)
            or g.shape != (len(stage.atoms), 3)
            or not np.all(np.isfinite(H)) or not np.all(np.isfinite(g))):
        raise PreflightError("Analytic positron Hessian returned an invalid Cartesian result.")
    return H.transpose(0, 2, 1, 3).copy(), float(cc.e_tot), g.copy(), "analytic"


def _freq_hessian(stage):
    labels = [a for a, _ in stage.atoms]
    x0 = np.array([c for _, c in stage.atoms], float) / BOHR_TO_ANGSTROM          # bohr
    natm, n = len(labels), 3 * len(labels)
    route = freq_route(stage)

    def atoms_at(x):
        return list(zip(labels, map(tuple, x * BOHR_TO_ANGSTROM)))

    if route == "analytic":
        if stage.system_type == "POSITRON":
            return _analytic_positron_frequency(stage)
        try:
            mf, _ = _electronic_mf(stage, atoms_to_string(stage.atoms), stage.use_df)
            mf.max_memory, mf.conv_tol, mf.verbose = stage.memory_mb, 1e-11, 0
            mf.kernel()
            hess = np.asarray(mf.Hessian().kernel())
            g = np.asarray(mf.nuc_grad_method().kernel())
            return hess, float(mf.e_tot), g, "analytic"
        except Exception as err:
            if getattr(stage, "freq_mode", "auto") == "analytic":      # ANFREQ asked for the analytic one only
                raise
            print(f"  [Notice] PySCF's analytic Hessian failed ({type(err).__name__}: {err}); using gradient "
                  f"differences")
            route = "gradient"
    h = stage.freq_step or (FREQ_STEP_GRADIENT_BOHR if route == "gradient" else FREQ_STEP_ENERGY_BOHR)
    if _uses_native_frequency(stage):
        return _native_positron_frequency(stage, route, h)
    points, save = _freq_cache(stage, route, h)
    t0 = time.time()

    if route == "gradient":
        def grad_at(key, x):
            if key not in points:
                e, g = energy_and_forces(stage, atoms_to_string(atoms_at(x)))
                if g is None:
                    raise RuntimeError("the analytic gradient failed at a displaced geometry")
                points[key] = [float(e), (np.asarray(g) * BOHR_TO_ANGSTROM).ravel().tolist()]    # Ha/bohr
                save()
            return points[key][0], np.array(points[key][1])
        e0, g0 = grad_at("0", x0)
        H = np.zeros((n, n))
        for a in range(n):
            xp, xm = x0.ravel().copy(), x0.ravel().copy()
            xp[a] += h
            xm[a] -= h
            _, gp = grad_at(f"{a}+", xp.reshape(natm, 3))
            _, gm = grad_at(f"{a}-", xm.reshape(natm, 3))
            H[a] = (gp - gm) / (2 * h)
            print(f"  [Freq] coordinate {a + 1}/{n} done ({time.time() - t0:.0f} s)")
        asym = float(np.abs(H - H.T).max())
        H = 0.5 * (H + H.T)
        print(f"  [Freq] Hessian from {2 * n + 1} gradients, step {h:g} bohr; largest asymmetry before "
              f"symmetrizing {asym:.2e} Ha/bohr^2 (numerical noise)")
    else:
        def energy_at(key, x):
            if key not in points:
                points[key] = float(run_single_point_calculation(stage, atoms_at(x), use_df=stage.use_df))
                save()
            return points[key]

        def shifted(moves):
            x = x0.ravel().copy()
            for a, s in moves:
                x[a] += s * h
            return x.reshape(natm, 3)
        e0 = energy_at("0", x0)
        H = np.zeros((n, n))
        g0 = np.zeros(n)
        ep, em = {}, {}
        for a in range(n):
            ep[a] = energy_at(f"{a}+", shifted([(a, 1)]))
            em[a] = energy_at(f"{a}-", shifted([(a, -1)]))
            H[a, a] = (ep[a] - 2 * e0 + em[a]) / h ** 2
            g0[a] = (ep[a] - em[a]) / (2 * h)
        for a in range(n):
            for b in range(a + 1, n):
                vals = [energy_at(f"{a}{sa}{b}{sb}", shifted([(a, 1 if sa == "+" else -1), (b, 1 if sb == "+" else -1)]))
                        for sa, sb in (("+", "+"), ("+", "-"), ("-", "+"), ("-", "-"))]
                H[a, b] = H[b, a] = (vals[0] - vals[1] - vals[2] + vals[3]) / (4 * h ** 2)
            print(f"  [Freq] coordinate {a + 1}/{n} done ({time.time() - t0:.0f} s)")
        print(f"  [Freq] Hessian from {1 + 2 * n + 2 * n * (n - 1)} energies, step {h:g} bohr")
        g0 = g0.reshape(natm, 3)
    return H.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3), float(e0), np.asarray(g0).reshape(natm, 3), route


def freq_masses(stage):
    """Atomic masses (amu) for the frequencies: the most abundant isotope of each element, as Gaussian and
    ORCA do."""
    from pyscf.data import elements
    return np.array([elements.COMMON_ISOTOPE_MASSES[gto.charge(element_of(sym))] for sym, _ in stage.atoms])


_LAST_FREQ = {}           # frequencies of the last run_freq (the TS check reads them)


def _harmonic_analysis(mol, hess, masses):
    """Preserve PySCF analysis and recover a missing linear-rotor mode.

    Some PySCF versions classify a tiny negative zero-inertia eigenvalue as
    a nonlinear rotor. Only retry when one mode is missing and the coordinates
    have rank one at floating-point precision. The retry uses a rotated copy;
    the calculation geometry, Hessian and dependency functions stay unchanged.
    """
    from pyscf.hessian import thermo
    result = thermo.harmonic_analysis(mol, hess, imaginary_freq=False, mass=masses)
    natm = mol.natm
    if natm < 2 or len(result["freq_wavenumber"]) != 3 * natm - 6:
        return result

    coords = mol.atom_coords()
    coords -= coords[0].copy()
    _, singular, axes = np.linalg.svd(coords, full_matrices=False)
    tolerance = np.finfo(float).eps * max(coords.shape) * singular[0]
    if singular[0] == 0 or (natm > 2 and singular[1] > tolerance):
        return result
    axis = axes[0]
    transverse = np.cross(axis, np.eye(3)[np.argmin(abs(axis))])
    transverse /= np.linalg.norm(transverse)
    rotation = np.array([transverse, np.cross(axis, transverse), axis])
    centered = coords - np.asarray(masses) @ coords / np.sum(masses)
    aligned = np.zeros_like(coords)
    aligned[:, 2] = centered @ axis
    rotated_hess = np.einsum("ax,ijxy,by->ijab", rotation, hess, rotation,
                             optimize=["einsum_path", (0, 1), (0, 1)])
    rotated_mol = mol.copy().set_geom_(aligned, unit="Bohr")
    result = thermo.harmonic_analysis(rotated_mol, rotated_hess,
                                     imaginary_freq=False, mass=masses)
    if len(result["freq_wavenumber"]) != 3 * natm - 5:
        raise RuntimeError("Frequency analysis failed to retain all linear-molecule modes")
    result["norm_mode"] = np.asarray(result["norm_mode"]) @ rotation
    print("[Freq] Recovered linear-molecule modes in an axis-aligned analysis frame")
    return result


def run_freq(stage):
    """! FREQ: harmonic frequencies, zero-point energy and normal modes at the stage geometry (use an optimized
    geometry: an OPT stage before it passes its result on)."""
    from pyscf.data import nist
    job = stage.job_name or f"Stage_{stage.stage_id}"
    print(f"[Freq] Hessian: {freq_route_label(stage)}")
    t0 = time.time()
    hess, e0, g0, route = freq_hessian(stage)
    gmax = float(np.abs(g0).max()) if g0 is not None else float("nan")
    print(f"[Freq] Reference energy: {e0:.12f} Ha; largest gradient component {gmax:.2e} Ha/bohr")
    if gmax > 1e-3:
        print("[Freq] WARNING: this is not a stationary point (gradient > 1e-3 Ha/bohr): the frequencies are not "
              "meaningful. Optimize the geometry with the same method and basis first.")
    labels = [a for a, _ in stage.atoms]
    masses = freq_masses(stage)
    mol = gto.M(atom=atoms_to_string([(element_of(a), c) for a, c in stage.atoms]), basis="sto-3g",
                spin=sum(gto.charge(element_of(a)) for a in labels) % 2, verbose=0)
    res = _harmonic_analysis(mol, hess, masses)
    freq = np.asarray(res["freq_wavenumber"]).real
    _LAST_FREQ["freq"] = freq.tolist()
    red = np.asarray(res["reduced_mass"])
    kf = np.asarray(res["force_const_dyne"])        # PySCF's "force_const_dyne" is in mdyn/A
    modes = np.asarray(res["norm_mode"])                    # (nmode, natm, 3), Cartesian displacements
    zpe = 0.5 * float(freq[freq > 0].sum()) / nist.HARTREE2WAVENUMBER
    n_imag = int((freq < 0).sum())
    print("\n" + "-" * 80)
    print(f"[Freq] Harmonic frequencies ({stage.method}, {stage.system_type}, "
          f"{'DF' if stage.use_df else 'CONV'}; {len(freq)} modes; masses: most abundant isotopes)")
    print(f"  {'Mode':>4} {'Frequency (cm^-1)':>18} {'Reduced mass (amu)':>19} {'Force const. (mdyn/A)':>22}")
    for i, (f, m, k) in enumerate(zip(freq, red, kf)):
        tag = "  imaginary" if f < 0 else ""
        print(f"  {i + 1:>4} {f:>18.2f} {m:>19.4f} {k:>22.4f}{tag}")
    print(f"  Zero-point energy: {zpe:.8f} Ha = {zpe * 2625.49964:.4f} kJ/mol "
          f"= {zpe * 627.509474:.4f} kcal/mol")
    print(f"  Energy + ZPE: {e0 + zpe:.12f} Ha")
    if n_imag:
        print(f"  [Freq] {n_imag} imaginary frequenc{'y' if n_imag == 1 else 'ies'} (printed negative): a "
              f"saddle point, or a geometry that is not fully optimized")
    print("-" * 80)
    # <job>_freq.txt (report file: decimal comma with --decimal-comma) and <job>_freq.molden (normal modes)
    with open(f"{job}_freq.txt", "w") as f:
        f.write(f"# newBRORSEN v{VERSION} FREQ: {stage.method} {stage.system_type} "
                f"{'DF' if stage.use_df else 'CONV'}, basis {stage.elec_basis}"
                f"{', positron basis ' + str(stage.pos_basis_name) if stage.system_type == 'POSITRON' else ''}\n")
        f.write(f"# Hessian: {freq_route_label(stage, route)}\n")
        f.write("# Masses: most abundant isotopes (amu): "
                + ", ".join(f"{element_of(a)} {num(m, '.6f')}" for a, m in zip(labels, masses)) + "\n")
        f.write(f"Reference energy (Ha)\t{num(e0, '.12f')}\n")
        f.write(f"Largest gradient component (Ha/bohr)\t{num(gmax, '.3e')}\n")
        f.write(f"Zero-point energy (Ha)\t{num(zpe, '.8f')}\n")
        f.write(f"Zero-point energy (kcal/mol)\t{num(zpe * 627.509474, '.4f')}\n")
        f.write(f"Energy + ZPE (Ha)\t{num(e0 + zpe, '.12f')}\n")
        f.write(f"Imaginary frequencies\t{n_imag}\n\n")
        f.write("Mode\tFrequency_cm-1\tReduced_mass_amu\tForce_constant_mdyn/A\n")
        for i, (fr, m, k) in enumerate(zip(freq, red, kf)):
            f.write(f"{i + 1}\t{num(fr, '.2f')}\t{num(m, '.4f')}\t{num(k, '.4f')}\n")
        f.write("\n# Normal modes: Cartesian displacements per atom (x, y, z)\n")
        for i, mode in enumerate(modes):
            f.write(f"Mode {i + 1}\t{num(freq[i], '.2f')} cm-1\n")
            for a, d in zip(labels, mode):
                f.write(f"{a}\t" + "\t".join(num(v, '.6f') for v in d) + "\n")
        f.write("\n# Cartesian Hessian (Ha/bohr^2), rows and columns atom-major (atom 1 x, y, z, atom 2 ...)\n")
        hm = hess.transpose(0, 2, 1, 3).reshape(3 * len(labels), -1)
        for row in hm:
            f.write("\t".join(num(v, '.8e') for v in row) + "\n")
    with open(f"{job}_freq.molden", "w") as f:            # Molden vibration format: Jmol, Avogadro, Molden
        f.write("[Molden Format]\n[FREQ]\n")
        for fr in freq:
            f.write(f"{fr:12.4f}\n")
        f.write("[FR-COORD]\n")
        for a, c in stage.atoms:
            x = np.asarray(c) / BOHR_TO_ANGSTROM
            f.write(f"{element_of(a):>3} {x[0]:14.8f} {x[1]:14.8f} {x[2]:14.8f}\n")
        f.write("[FR-NORM-COORD]\n")
        for i, mode in enumerate(modes):
            f.write(f"vibration {i + 1}\n")
            for d in mode / max(np.abs(mode).max(), 1e-12):
                f.write(f"{d[0]:14.8f} {d[1]:14.8f} {d[2]:14.8f}\n")
    for name in (f"{job}_freq.txt", f"{job}_freq.molden"):
        print(f"[Artifact Generated] {name}")
    print(f"  [Freq] {time.time() - t0:.1f} s")
    return e0


# ---------------------------------------------------------------------------------------------------------------
# v04.05: TS (transition-state search) and IRC with geomeTRIC, on the wrapper's own energies and gradients
# ---------------------------------------------------------------------------------------------------------------
TS_HESSIAN_CHOICES = ("auto", "first", "each", "never")      # plus "stride:N" and "file:<path>"


def _validate_ts(stage):
    if stage.run_type not in ("TS", "IRC"):
        for k, default in (("ts_hessian", "auto"), ("irc_direction", "both")):
            if getattr(stage, k) != default:
                stage.warn(f"%{k} only acts in TS and IRC stages -- ignored")
        return
    word = stage.run_type
    if stage.system_type not in ("ELECTRONIC", "POSITRON"):
        stage.error(f"{word}: the ELECTRONIC and POSITRON sectors only (not {stage.system_type})")
    if any(is_ghost_label(sym) for sym, _ in stage.atoms):
        stage.error(f"{word}: ghost atoms have no mass; remove them")
    if len(stage.atoms) < 3:
        stage.error(f"{word}: a transition state needs at least 3 atoms (a diatomic has only the bond length)")
    h = stage.ts_hessian
    if not (h in TS_HESSIAN_CHOICES or re.fullmatch(r"stride:\d+", h) or h.startswith("file:")):
        stage.error(f"%ts_hessian must be auto, first, each, never, stride:<N> or file:<path>, got {h!r}")
    elif h.startswith("file:") and not os.path.exists(h[5:]):
        stage.error(f"%ts_hessian {h}: no such file")
    elif h == "never" and word == "IRC":
        stage.error("IRC needs a Hessian at the transition state: %ts_hessian never is not allowed")
    if stage.irc_direction not in ("both", "forward", "backward"):
        stage.error(f"%irc_direction must be both, forward or backward, got {stage.irc_direction!r}")
    if stage.system_type in ("ELECTRONIC", "POSITRON") and not analytic_gradient_available(stage)[0]:
        stage.warn(f"{word}: no analytic gradient ({analytic_gradient_available(stage)[1]}): every step uses "
                   f"numerical gradients (6N energies each); slow except for small systems")


def ts_hessian_label(stage):
    h = stage.ts_hessian
    if h == "auto":
        return (f"computed by the wrapper at the start ({freq_route_label(stage)}), then updated by geomeTRIC "
                f"every step")
    return {"first": "geomeTRIC's finite differences at the first step, then updated",
            "each": "geomeTRIC's finite differences at every step (robust, expensive)",
            "never": "geomeTRIC's guess, updated every step (cheapest, least reliable)"}.get(
        h, f"geomeTRIC's finite differences every {h[7:]} steps" if h.startswith("stride:")
        else f"read from {h[5:]}, then updated")


def _geometric_saddle(stage, irc=False):
    """Run geomeTRIC's transition-state search (P-RFO, transition=True) or its IRC (irc=True) on the stage's
    surface, with the wrapper's energy and gradient. Returns (converged, final atoms, work folder)."""
    import geometric
    from geometric.errors import GeomOptNotConvergedError
    from pyscf.geomopt import geometric_solver
    job = stage.job_name or f"Stage_{stage.stage_id}"
    mol = gto.M(atom=atoms_to_string([(element_of(a), c) for a, c in stage.atoms]), basis="sto-3g",
                charge=stage.charge, spin=stage.multiplicity - 1, verbose=0)      # geometry carrier for geomeTRIC
    scanner = _WrapperGradScanner(stage, mol)
    engine = geometric_solver.PySCFEngine(scanner)
    engine.mol = scanner.mol.copy()
    engine.assert_convergence = True
    if irc:          # geomeTRIC's IRC reads the fragment list, which a custom engine's molecule does not have yet
        engine.M.build_topology()
    work = tempfile.mkdtemp(prefix=f".{job}_geometric_", dir=".")
    kwargs = {"prefix": os.path.join(work, job), "maxiter": stage.opt_max_cycles or (100 if not irc else 300),
              "transition": not irc, "irc": irc}
    if irc:
        kwargs["irc_direction"] = stage.irc_direction
    if not os.path.exists(os.path.join(os.path.dirname(geometric.optimize.__file__), "log.ini")):
        kwargs["logIni"] = os.path.join(os.path.dirname(geometric_solver.__file__), "log.ini")
    h = stage.ts_hessian
    if h == "auto":
        H, e0, g0, route = freq_hessian(stage)
        path = f"{job}_initial_hessian.txt"
        np.savetxt(path, H.transpose(0, 2, 1, 3).reshape(3 * len(stage.atoms), -1))
        how = {"analytic": "analytic", "gradient": "from analytic gradients", "energy": "from energies"}[route]
        print(f"[{'IRC' if irc else 'TS'}] Initial Hessian ({how}) saved in {path} (Ha/bohr^2, numpy text: "
              f"reusable with %ts_hessian file:{path})")
        kwargs["hessian"] = f"file:{os.path.abspath(path)}"
    else:
        kwargs["hessian"] = h if not h.startswith("file:") else f"file:{os.path.abspath(h[5:])}"
    constraints_file = write_geometric_constraints(stage)
    if constraints_file:
        kwargs["constraints"] = constraints_file
    converged = True
    try:
        geometric.optimize.run_optimizer(customengine=engine, input=os.path.join(work, job), **kwargs)
    except GeomOptNotConvergedError:
        converged = False
    except Exception:
        print(f"  [Notice] geomeTRIC stopped; its files are kept in {work} for inspection")
        raise
    finally:
        if constraints_file and os.path.exists(constraints_file):
            os.remove(constraints_file)
    return converged, extract_optimized_atoms(engine.mol, stage.atoms), work, scanner.cycle


def _keep_geometric_files(work, job, pairs):
    """Copy geomeTRIC's outputs we keep into the job folder, then remove its work folder."""
    import glob, shutil
    kept = []
    for pattern, name in pairs:
        found = sorted(glob.glob(os.path.join(work, pattern)))
        if found:
            shutil.copy(found[-1], name)
            kept.append(name)
    shutil.rmtree(work, ignore_errors=True)
    return kept


def run_ts(stage):
    """! TS: transition-state (first-order saddle point) search with geomeTRIC's P-RFO. Default: the wrapper
    computes the Hessian once at the start (analytic where PySCF has one, else from analytic gradients, as FREQ),
    geomeTRIC updates it every step; one gradient per step. Then FREQ at the result checks that it has exactly one
    imaginary frequency (%ts_freq false skips it)."""
    job = stage.job_name or f"Stage_{stage.stage_id}"
    print(f"\n[TS] Transition-state search: geomeTRIC P-RFO (transition=True) on the {stage.system_type} "
          f"{stage.method} surface")
    print(f"[TS] Hessian: {ts_hessian_label(stage)}")
    t_cpu, t_wall = time.process_time(), time.time()
    converged, ts_atoms, work, ncalls = _geometric_saddle(stage)
    kept = _keep_geometric_files(work, job, [("*_optim.xyz", f"{job}_ts_path.xyz"), ("*.log", f"{job}_geometric.log")])
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    print(f"  [geomeTRIC TS Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), Wall {wall:.1f} s ({wall/3600:.3f} h), "
          f"{ncalls} gradient evaluations in the search")
    if not converged:
        print(f"  [WARNING] the transition-state search did not converge in {stage.opt_max_cycles or 100} steps; the "
              f"last geometry is reported (raise %opt_max_cycles, or start from a better guess, e.g. a NEB or a scan)")
    else:
        print("[TS] Transition-state search converged.")
    for name in kept:
        print(f"[Artifact Generated] {name}")
    _report_final_energies(stage, ts_atoms)
    if stage.ts_freq:
        view = copy.copy(stage)
        view.atoms = ts_atoms
        view.run_type = "FREQ"
        view.freq_mode = "auto"
        view.job_name = f"{job}_ts"
        print(f"\n[TS] Frequency check at the transition state ({freq_route_label(view)})")
        run_freq(view)
        freqs = _LAST_FREQ.get("freq", [])
        n_imag = int(sum(1 for f in freqs if f < 0))
        if n_imag == 1:
            print(f"[TS] Exactly one imaginary frequency ({min(freqs):.2f} cm^-1): a first-order saddle point. Its "
                  f"normal mode (mode 1 in {view.job_name}_freq.molden) is the reaction coordinate; follow it with an "
                  f"IRC stage ('! TS' with '%ts_mode irc') to see which minima it connects.")
        else:
            print(f"[TS] WARNING: {n_imag} imaginary frequencies: {'a minimum, not a transition state' if n_imag == 0 else 'a higher-order saddle point'}. "
                  f"Start from a better guess or use %ts_hessian each.")
    return ts_atoms


def run_irc(stage):
    """! IRC: geomeTRIC's intrinsic reaction coordinate from the stage geometry (a transition state, e.g. from a
    TS stage before it), downhill in both directions (%irc_direction). The path goes to <job>_irc.xyz with the
    energies; the stage passes its input geometry on."""
    job = stage.job_name or f"Stage_{stage.stage_id}"
    print(f"\n[IRC] Intrinsic reaction coordinate: geomeTRIC (irc=True, direction {stage.irc_direction}) on the "
          f"{stage.system_type} {stage.method} surface")
    print(f"[IRC] Hessian at the start: {ts_hessian_label(stage)}")
    t_cpu, t_wall = time.process_time(), time.time()
    converged, last_atoms, work, ncalls = _geometric_saddle(stage, irc=True)
    kept = _keep_geometric_files(work, job, [("*_irc.xyz", f"{job}_irc.xyz"), ("*.log", f"{job}_geometric.log")])
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    print(f"  [geomeTRIC IRC Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), Wall {wall:.1f} s ({wall/3600:.3f} h), "
          f"{ncalls} gradient evaluations")
    if not converged:
        print("  [WARNING] the IRC did not reach both minima within the step limit (raise %opt_max_cycles)")
    path = f"{job}_irc.xyz"
    if os.path.exists(path):
        energies = [float(m.group(1)) for m in re.finditer(r"[Ee]nergy\s+(-?\d+\.\d+)", open(path).read())]
        if energies:
            e_ts = max(energies)
            print(f"[IRC] {len(energies)} points along the path; energies relative to the highest point "
                  f"({e_ts:.10f} Ha): first end {(energies[0] - e_ts) * 627.509474:.3f} kcal/mol, last end "
                  f"{(energies[-1] - e_ts) * 627.509474:.3f} kcal/mol")
            with open(f"{job}_irc_energies.txt", "w") as f:
                f.write("Point\tEnergy_Ha\tRelative_kcal/mol\n")
                for i, e in enumerate(energies):
                    f.write(f"{i + 1}\t{num(e, '.10f')}\t{num((e - e_ts) * 627.509474, '.4f')}\n")
            kept.append(f"{job}_irc_energies.txt")
    for name in kept:
        print(f"[Artifact Generated] {name}")
    return None


def _validate_neb(stage):
    if stage.run_type != "NEB":
        if stage.neb_product:
            stage.warn("%neb_product only acts with '! TS' and %ts_mode neb -- ignored")
        return
    if stage.system_type not in ("ELECTRONIC", "POSITRON"):
        stage.error(f"NEB: the ELECTRONIC and POSITRON sectors only (not {stage.system_type})")
    if not stage.neb_product:
        stage.error("NEB needs the product geometry: a %neb_product block (xyz lines, same atoms in the same order "
                    "as the stage geometry, which is the reactant) closed with 'end'")
        return
    if [element_of(a) for a, _ in stage.neb_product] != [element_of(a) for a, _ in stage.atoms]:
        stage.error("NEB: %neb_product must list the same atoms in the same order as the reactant geometry")
    if not 5 <= stage.neb_images <= 50:
        stage.error(f"%neb_images must be 5-50 (geomeTRIC's default 11), got {stage.neb_images}")
    if any(is_ghost_label(sym) for sym, _ in stage.atoms):
        stage.error("NEB: ghost atoms are not supported")
    if not analytic_gradient_available(stage)[0]:
        stage.warn(f"NEB: no analytic gradient ({analytic_gradient_available(stage)[1]}): every image and step uses "
                   f"numerical gradients; very slow except for tiny systems")
    stage.warn(f"NEB: {stage.neb_images} images, one gradient per moving image and step (typically 20-60 steps); "
               f"it gives a guess for a TS stage, not the transition state itself; the end points should be minima "
               f"(optimize them first, or %neb_optimize_ends true)")


def _kabsch_align(mobile, ref):
    """mobile (n, 3) rotated and translated onto ref (n, 3) (least squares, proper rotation)."""
    a, b = mobile - mobile.mean(0), ref - ref.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, d]) @ vt
    return a @ rot + ref.mean(0)


def run_neb(stage):
    """! NEB: geomeTRIC's climbing-image nudged elastic band between the stage geometry (reactant) and
    %neb_product, on the stage's surface with the wrapper's gradients. The start path is a linear interpolation
    after aligning the product onto the reactant. The highest image is passed on as the TS guess (put a TS stage
    after it)."""
    import glob, shutil
    import geometric
    from geometric.molecule import Molecule
    from geometric.params import NEBParams
    from geometric.neb import ElasticBand, OptimizeChain
    from pyscf.geomopt import geometric_solver
    job = stage.job_name or f"Stage_{stage.stage_id}"
    labels = [a for a, _ in stage.atoms]
    r = np.array([c for _, c in stage.atoms], float)
    p = _kabsch_align(np.array([c for _, c in stage.neb_product], float), r)
    n = stage.neb_images
    frames = [r + (p - r) * t for t in np.linspace(0.0, 1.0, n)]
    print(f"\n[NEB] Climbing-image nudged elastic band (geomeTRIC), {n} images, reactant -> product "
          f"(product aligned onto the reactant, RMSD {np.sqrt(((p - r) ** 2).sum(1).mean()):.3f} A)")
    M = Molecule()
    M.elem = [element_of(a) for a in labels]
    M.xyzs = [f.copy() for f in frames]
    M.comms = [f"image {i + 1}" for i in range(n)]
    mol = gto.M(atom=atoms_to_string([(element_of(a), c) for a, c in stage.atoms]), basis="sto-3g",
                charge=stage.charge, spin=stage.multiplicity - 1, verbose=0)
    scanner = _WrapperGradScanner(stage, mol)
    engine = geometric_solver.PySCFEngine(scanner)
    engine.mol = scanner.mol.copy()
    work = tempfile.mkdtemp(prefix=f".{job}_neb_", dir=".")
    params = NEBParams(images=n, neb_maxcyc=stage.opt_max_cycles or 100, prefix=os.path.join(work, job),
                       optep=stage.neb_optimize_ends)
    t_cpu, t_wall = time.process_time(), time.time()
    try:
        chain = ElasticBand(M, engine=engine, tmpdir=work, params=params, plain=0)
        chain, cycles = OptimizeChain(chain, engine, params)
    except Exception:
        print(f"  [Notice] geomeTRIC's NEB stopped; its files are kept in {work} for inspection")
        raise
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    energies = np.array([s.energy for s in chain.Structures])
    converged = chain.maxg < params.maxg and chain.avgg < params.avgg
    k = 1 + int(np.argmax(energies[1:-1]))       # the highest interior image (the end points are minima)
    ts_guess = [(lab, tuple(float(x) for x in c))
                for lab, c in zip(labels, chain.Structures[k].cartesians.reshape(-1, 3) * BOHR_TO_ANGSTROM)]
    print(f"  [geomeTRIC NEB Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), Wall {wall:.1f} s ({wall/3600:.3f} h), "
          f"{cycles} cycles, {scanner.cycle} gradient evaluations")
    if not converged:
        print(f"  [WARNING] the NEB did not converge in {params.maxcyc} cycles (max gradient {chain.maxg:.4f}, "
              f"average {chain.avgg:.4f}; thresholds {params.maxg}, {params.avgg}); the highest image is still passed "
              f"on as a TS guess (raise %opt_max_cycles for a better one)")
    else:
        print(f"[NEB] Converged in {cycles} cycles.")
    e0 = energies[0]
    print(f"  {'Image':>5} {'Energy (Ha)':>18} {'Relative to image 1 (kcal/mol)':>32}")
    for i, e in enumerate(energies):
        print(f"  {i + 1:>5} {e:>18.10f} {(e - e0) * 627.509474:>32.3f}{'   <- highest: TS guess' if i == k else ''}")
    ends = "optimized end points" if stage.neb_optimize_ends else "end points as given, not optimized"
    print(f"[NEB] Barrier estimate from image 1: {(energies[k] - e0) * 627.509474:.3f} kcal/mol; from the last "
          f"image: {(energies[k] - energies[-1]) * 627.509474:.3f} kcal/mol ({ends})")
    if min(energies[0], energies[-1]) > energies[k] - 1e-6:
        print("[NEB] WARNING: an end point is not below the highest interior image: the end points are probably not "
              "minima. Optimize them first (OPT stages, or %neb_optimize_ends true).")
    last = sorted(glob.glob(os.path.join(work, "chain_*.xyz")))
    kept = []
    if last:
        shutil.copy(last[-1], f"{job}_neb_path.xyz")
        kept.append(f"{job}_neb_path.xyz")
    with open(f"{job}_neb_energies.txt", "w") as f:
        f.write("Image\tEnergy_Ha\tRelative_kcal/mol\n")
        for i, e in enumerate(energies):
            f.write(f"{i + 1}\t{num(e, '.10f')}\t{num((e - e0) * 627.509474, '.4f')}\n")
    kept.append(f"{job}_neb_energies.txt")
    shutil.rmtree(work, ignore_errors=True)
    for name in kept:
        print(f"[Artifact Generated] {name}")
    with open(f"{job}_ts_guess.xyz", "w") as f:
        f.write(f"{len(ts_guess)}\nNEB highest image {k + 1}/{n}, E = {energies[k]:.10f} Ha\n")
        for sym, c in ts_guess:
            f.write(f"{sym:<3} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f}\n")
    print(f"[Artifact Generated] {job}_ts_guess.xyz")
    return ts_guess


def run_single_point_calculation(stage, atoms, use_df=True):
    return run_single_point_components(stage, atoms, use_df=use_df)["total"]


# ----------------------------------------------------------------------------
# Scans
# ----------------------------------------------------------------------------

def _fragment_split(stage):
    syms = [s for s, _ in stage.atoms]
    coords0 = np.array([c for _, c in stage.atoms], dtype=float)
    seq = [float(np.linalg.norm(coords0[i + 1] - coords0[i])) for i in range(len(syms) - 1)]
    k = int(np.argmax(seq))
    c1, c2 = coords0[k], coords0[k + 1]
    axis = c2 - c1
    r0 = float(np.linalg.norm(axis))
    axis = axis / r0 if r0 > 1e-8 else np.array([0.0, 0.0, 1.0])
    return syms, coords0, k, axis, r0, seq[k]


def scan_coordinate(stage):
    """(name, unit) of the scanned coordinate; unit 'A' or 'deg'."""
    sc = stage.scan
    if sc["mode"] == "fragment":
        return "inter-fragment distance", "A"
    if sc["mode"] == "bond":
        return f"R({sc['atoms'][0]}-{sc['atoms'][1]})", "A"
    if sc["mode"] == "translate":
        return f"displacement of atoms {sc['range'][0]}-{sc['range'][1]} along {sc['axis']}", "A"
    if sc["mode"] == "angle":
        return "angle({})".format("-".join(str(i) for i in sc["atoms"])), "deg"
    if sc["mode"] == "dihedral":
        return "dihedral({})".format("-".join(str(i) for i in sc["atoms"])), "deg"
    for row in stage.zmat_rows:
        for pos, tok in enumerate(row[1:], 1):
            if tok.lstrip('-') == sc["var"]:
                return f"Z-matrix variable {sc['var']}", ("A" if pos == 2 else "deg")
    return f"Z-matrix variable {sc['var']}", "A"


def scan_geometry_at(stage, value):
    """Atoms at one value of the scanned coordinate."""
    sc = stage.scan
    syms = [s for s, _ in stage.atoms]
    coords0 = np.array([c for _, c in stage.atoms], dtype=float)
    if sc["mode"] == "fragment":
        syms, coords0, k, axis, r0, _ = _fragment_split(stage)
        delta = (value - r0) / 2.0
        shift = np.array([-axis * delta] * (k + 1) + [axis * delta] * (len(syms) - k - 1))
        c = coords0 + shift
    elif sc["mode"] == "bond":
        a, b = sc["atoms"][0] - 1, sc["atoms"][1] - 1
        v = coords0[b] - coords0[a]
        u = v / np.linalg.norm(v)
        mid = 0.5 * (coords0[a] + coords0[b])
        c = coords0.copy()
        c[a] = mid - 0.5 * value * u
        c[b] = mid + 0.5 * value * u
    elif sc["mode"] == "translate":
        i0, i1 = sc["range"][0] - 1, sc["range"][1] - 1
        c = coords0.copy()
        c[i0:i1 + 1, AXIS_INDEX[sc["axis"]]] += value
    elif sc["mode"] in ("angle", "dihedral"):
        c = _scan_rotate(coords0, sc["mode"], [i - 1 for i in sc["atoms"]], float(value))
    else:
        variables = dict(stage.zmat_vars)
        variables[sc["var"]] = float(value)
        syms, c = zmatrix_to_cartesian(stage.zmat_rows, variables)
    return [(s, tuple(x)) for s, x in zip(syms, c)]


def _rotate_about(points, origin, axis, theta_deg):
    """Rodrigues rotation of points (n, 3) by theta about the axis through origin."""
    k = axis / np.linalg.norm(axis)
    t = math.radians(theta_deg)
    p = points - origin
    return (origin + p * math.cos(t) + np.cross(k, p) * math.sin(t)
            + np.outer(p @ k, k) * (1 - math.cos(t)))


def _scan_rotate(coords0, mode, idx, value):
    """v04.08 Cartesian angle / dihedral scan: only the last atom of the definition moves (as a bond scan
    moves only its two atoms). angle i j k: atom k turns about j in the i-j-k plane; dihedral i j k l:
    atom l turns about the j-k axis. To move a whole group, scan a Z-matrix variable instead."""
    c = coords0.copy()
    if mode == "angle":
        a, b, k = idx
        u, v = c[a] - c[b], c[k] - c[b]
        n = np.cross(u, v)
        if np.linalg.norm(n) < 1e-8:              # linear start: any perpendicular plane
            n = np.cross(u, [1.0, 0.0, 0.0])
            if np.linalg.norm(n) < 1e-8:
                n = np.cross(u, [0.0, 1.0, 0.0])
        delta = value - compute_angle(c[a], c[b], c[k])
        c[k] = _rotate_about(c[k:k + 1], c[b], n, delta)[0]
        return c
    a, b, k, d = idx
    delta = value - compute_dihedral(c[a], c[b], c[k], c[d])
    delta = (delta + 180.0) % 360.0 - 180.0
    for sign in (1.0, -1.0):                      # the sign that reaches the target (convention-safe)
        trial = c.copy()
        trial[d] = _rotate_about(c[d:d + 1], c[k], c[k] - c[b], sign * delta)[0]
        got = compute_dihedral(trial[a], trial[b], trial[k], trial[d])
        if abs((got - value + 180.0) % 360.0 - 180.0) < 1e-6:
            return trial
    return trial


def scan_values(sc):
    """The scanned coordinate values: the explicit list (v04.08 segments / [v1 v2 ...]) or start..end."""
    if sc.get("values"):
        return list(sc["values"])
    return list(np.linspace(sc["start"], sc["end"], sc["steps"]))


def scan_geometries(stage):
    """[(coordinate value, atoms)] for the stage's scan definition."""
    return [(v, scan_geometry_at(stage, v)) for v in scan_values(stage.scan)]


# 1 Ha/A^2 in N/m; amu in kg; c in cm/s
_HA_PER_A2_TO_N_PER_M = 4.3597447222071e-18 / 1e-20
_AMU_KG = 1.66053906660e-27
_C_CM_S = 2.99792458e10
_HA_TO_CM1 = 219474.6313632


def atomic_mass(sym):
    from pyscf.data import elements
    return float(elements.COMMON_ISOTOPE_MASSES[elements.charge(element_of(sym))])


def scan_reduced_mass(stage):
    """(reduced mass in amu, description) for a 1-D frequency along the scanned
    coordinate, or (None, reason) when it is not a simple two-body stretch."""
    sc = stage.scan
    syms = [s for s, _ in stage.atoms]
    if sc["mode"] == "fragment":
        _, _, k, _, _, _ = _fragment_split(stage)
        ma = sum(atomic_mass(s) for s in syms[:k + 1])
        mb = sum(atomic_mass(s) for s in syms[k + 1:])
        return ma * mb / (ma + mb), f"fragments {'-'.join(syms[:k + 1])} / {'-'.join(syms[k + 1:])}"
    if sc["mode"] == "bond":
        a, b = sc["atoms"][0] - 1, sc["atoms"][1] - 1
        ma, mb = atomic_mass(syms[a]), atomic_mass(syms[b])
        extra = "" if len(syms) == 2 else " (other atoms held fixed)"
        return ma * mb / (ma + mb), f"atoms {syms[a]}{a+1} / {syms[b]}{b+1}{extra}"
    if sc["mode"] == "translate":
        i0, i1 = sc["range"][0] - 1, sc["range"][1] - 1
        moving = syms[i0:i1 + 1]
        rest = syms[:i0] + syms[i1 + 1:]
        if not rest:
            return None, "the whole molecule moves"
        ma, mb = sum(atomic_mass(s) for s in moving), sum(atomic_mass(s) for s in rest)
        return ma * mb / (ma + mb), f"moving group {'-'.join(moving)} / rest {'-'.join(rest)}"
    if sc["mode"] in ("angle", "dihedral"):
        return None, "the coordinate is an angle"
    return None, "a Z-matrix variable can move several atoms"


def fit_morse(radii, energies):
    def morse(r, De, a, re, E0):
        return De * (1.0 - np.exp(-a * (r - re)))**2 + E0

    i_min = int(np.argmin(energies))
    De0 = max(energies.max() - energies[i_min], 1e-4)
    from scipy.optimize import curve_fit
    # Seeds for a (v04.08): the fixed 2.0 used before, plus the width implied by the
    # curvature of the 3 lowest points (k = 2 De a^2). The fit with the smallest residual
    # wins; with few points in the well both seeds can still land on a wrong re, which
    # morse_fit_warnings() reports.
    seeds = [2.0]
    a_curv = _local_morse_width(radii, energies, De0)
    if a_curv is not None:
        seeds.insert(0, a_curv)
    best, last_err = None, None
    for a0 in seeds:
        try:
            p, _ = curve_fit(morse, radii, energies, p0=[De0, a0, radii[i_min], energies[i_min]], maxfev=20000)
        except RuntimeError as e:
            last_err = e
            continue
        ssr = float(np.sum((energies - morse(radii, *p))**2))
        if best is None or ssr < best[0]:
            best = (ssr, p)
    if best is None:
        raise last_err
    popt = best[1]
    residuals = energies - morse(radii, *popt)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((energies - energies.mean())**2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    De_fit, a_fit, re_fit, E0_fit = (float(x) for x in popt)
    return (De_fit, a_fit, re_fit, E0_fit, r_squared), residuals


def _three_point_minimum(radii, energies):
    """Parabola through the lowest point and its two neighbours: (x_min, curvature),
    or None at the edge of the scan or without upward curvature."""
    order = np.argsort(radii)
    r, e = np.asarray(radii, float)[order], np.asarray(energies, float)[order]
    i = int(np.argmin(e))
    if i == 0 or i == len(r) - 1:
        return None
    c2, c1, _ = np.polyfit(r[i-1:i+2], e[i-1:i+2], 2)
    if c2 <= 0:
        return None
    return float(-c1 / (2 * c2)), float(2 * c2)


def _local_morse_width(radii, energies, De):
    """Morse a from the local curvature at the minimum (k = 2 De a^2), or None."""
    loc = _three_point_minimum(radii, energies)
    if loc is None or De <= 0:
        return None
    return float(np.sqrt(loc[1] / (2 * De)))


def morse_fit_warnings(radii, energies, fit_result):
    """Reliability checks for a Morse fit (v04.08). A fit can converge to a wrong
    re/a with a good-looking R^2 when few points sample the well. Returns a list
    of warning strings (empty when all checks pass)."""
    De, a, re, E0, r2 = fit_result
    radii = np.asarray(radii, float)
    energies = np.asarray(energies, float)
    out = []
    if not np.isnan(r2) and r2 < 0.995:
        out.append(f"R^2 = {r2:.6f} is below 0.995; the Morse form does not describe these points well")
    # Points inside the well: within the classical turning points at De/2 above E0,
    # i.e. |1 - exp(-a (r - re))| < 1/sqrt(2).
    if a > 0:
        lo = re - np.log(1 + 1 / np.sqrt(2)) / a
        hi = re - np.log(1 - 1 / np.sqrt(2)) / a
        n_well = int(np.sum((radii >= lo) & (radii <= hi)))
        if n_well < 3:
            out.append(f"only {n_well} scan point(s) fall in the well (E < E0 + De/2, "
                       f"{lo:.3f}-{hi:.3f}); re and a are poorly determined. Add points near the minimum")
    return out


def fit_harmonic(x, energies):
    """E(x) = E0 + k/2 (x - xe)^2 by least squares (quadratic polynomial).
    Returns (k, xe, E0, R^2), residuals; raises ValueError without a minimum."""
    x = np.asarray(x, float)
    xm = x.mean()
    a, b, c = np.polyfit(x - xm, energies, 2)
    if a <= 0:
        raise ValueError(f"the points curve downward (curvature {2*a:.3e}); there is no minimum to fit")
    xe = xm - b / (2 * a)
    E0 = c - b * b / (4 * a)
    residuals = energies - np.polyval([a, b, c], x - xm)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((energies - energies.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return (float(2 * a), float(xe), float(E0), r2), residuals


def cubic_curvature_shift(x, energies, k, xe):
    """Relative change of the force constant when a cubic term is added to
    the fit (curvature of the cubic at xe). None with fewer than 4 points."""
    if len(x) < 4:
        return None
    x = np.asarray(x, float)
    xm = x.mean()
    c3 = np.polyfit(x - xm, energies, 3)
    k3 = float(np.polyval(np.polyder(c3, 2), xe - xm))
    return abs(k3 - k) / abs(k)


def harmonic_frequency_cm1(k_ha_per_a2, mu_amu):
    return float(np.sqrt(k_ha_per_a2 * _HA_PER_A2_TO_N_PER_M / (mu_amu * _AMU_KG)) / (2 * np.pi * _C_CM_S))


def morse_frequencies_cm1(De, a_per_a, mu_amu):
    """(omega_e, omega_e x_e) in cm-1 of a Morse potential."""
    k = 2.0 * De * a_per_a**2                      # Ha/A^2: curvature at the minimum
    we = harmonic_frequency_cm1(k, mu_amu)
    wexe = we**2 / (4.0 * De * _HA_TO_CM1)
    return we, wexe


def _job_prefix(stage):
    return stage.job_name if stage.job_name else f"Stage_{stage.stage_id}"


def write_scan_points(stage, radii, energies):
    """<job>_scan_points.txt: coordinate and total energy; the title line names the energy (method, sector,
    basis, integrals, frozen orbitals), the column header the method."""
    fname = f"{_job_prefix(stage)}_scan_points.txt"
    name, unit = scan_coordinate(stage)
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms) if stage.method in ("MP2", "CCSD", "CCSD(T)") else None
    energy = (f"E = {stage.method} total energy ({stage.system_type}, {stage.elec_basis}, "
              f"{'DF' if stage.use_df else 'exact integrals'}" + (f", frozen {frozen}" if frozen is not None else "") + ")")
    col = f"E {stage.method} (Hartree)"
    with open(fname, "w") as f:
        if stage.scan["mode"] == "fragment":
            f.write(f"Bond-length scan: {stage.job_name} | {energy}\n")
            f.write(f"{'R (Angstrom)':>18}  {col:>22}\n")
        else:
            f.write(f"Scan of {name}: {stage.job_name} | {energy}\n")
            f.write(f"{'x (' + unit + ')':>18}  {col:>22}\n")
        for r, e in zip(radii, energies):
            f.write(f"{num(r, '18.8f')}  {num(e, '22.12f')}\n")
    print(f"[Artifact Generated] Scan Points: {fname}")


def electronic_hf_terms(mf):
    """E_nuc, T_e, V_ne, V_ee of an electronic SCF (RHF/UHF/ROHF; V_ne includes ECPs, V_ee = J - K)."""
    dm = np.asarray(mf.make_rdm1())
    dm_t = dm[0] + dm[1] if dm.ndim == 3 else dm
    mol = mf.mol
    t_e = float(np.einsum("ij,ji->", mol.intor_symmetric("int1e_kin"), dm_t))
    e_elec, e2 = mf.energy_elec()          # (total electronic, two-electron)
    return {"E_nuc": float(mf.energy_nuc()), "T_e": t_e, "V_ne": float(e_elec - e2) - t_e, "V_ee": float(e2)}


def _eda_hf_terms(stage, mf, out):
    """%scan_print eda: the HF terms of an ELECTRONIC point (POSITRON points carry them in the partition)."""
    if stage.system_type == "ELECTRONIC" and "eda" in getattr(stage, "scan_print", ()):
        try:
            out["hf_terms"] = electronic_hf_terms(mf)
        except Exception as err:
            print(f"  [Notice] %scan_print eda: HF terms unavailable ({type(err).__name__}: {err})")


def write_scan_eda(stage, rows):
    """%scan_print eda (v04.08): <job>_scan_eda.txt, one line per scan point with the coordinate and every
    energy component of the method: HF; MP2 + E_corr(MP2); for CCSD also CCSD + E_corr(CCSD); for CCSD(T)
    also (T) and CCSD(T). A component the route does not give is 'nan'."""
    fname = f"{_job_prefix(stage)}_scan_eda.txt"
    name, unit = scan_coordinate(stage)
    pos = stage.system_type == "POSITRON"

    def part(key):
        return lambda c: (c.get("partition") or {}).get(key)

    def term(key):
        return lambda c: (c.get("partition") or c.get("hf_terms") or {}).get(key)

    def diff(a, b):
        return lambda c: c[a] - c[b] if c.get(a) is not None and c.get(b) is not None else None
    # HF terms, then each method adds only its own component(s) and its total; positron stages also split
    # every component into its electronic (_e) and positronic (_p) part (no sums: they add up to the total)
    cols = [(k, term(k)) for k in ("E_nuc", "T_e", "V_ne", "V_ee")]
    if pos:
        cols += [(k, term(k)) for k in ("T_p", "V_np", "V_pp", "V_ep")]
        cols += [("HF_e", part("HF_e")), ("HF_p", part("HF_p"))]
    cols += [("E_HF", lambda c: c["HF"])]
    if stage.method in ("MP2", "CCSD", "CCSD(T)"):
        cols += [("Ecorr_MP2", diff("MP2", "HF"))]
        if pos:
            cols += [("Ecorr_MP2_e", part("MP2_e")), ("Ecorr_MP2_p", part("MP2_p"))]
        cols += [("E_MP2", lambda c: c["MP2"])]
    if stage.method in ("CCSD", "CCSD(T)"):
        cols += [("Ecorr_CCSD", diff("CCSD", "HF"))]
        if pos:
            cols += [("Ecorr_CCSD_e", part("CCSD_e")), ("Ecorr_CCSD_p", part("CCSD_p"))]
        cols += [("E_CCSD", lambda c: c["CCSD"])]
    if stage.method == "CCSD(T)":
        cols += [("E_(T)", diff("CCSD(T)", "CCSD"))]
        if pos:
            cols += [("E_(T)_e", part("T3_e")), ("E_(T)_p", part("T3_p"))]
        cols += [("E_CCSD(T)", lambda c: c["CCSD(T)"])]
    frozen = resolve_frozen_core(stage.frozen_core, stage.atoms) if stage.method in ("MP2", "CCSD", "CCSD(T)") else None
    with open(fname, "w") as f:
        f.write(f"Energy components along the scan of {name}: {stage.job_name} | {stage.method} "
                f"({stage.system_type}, {stage.elec_basis}, {'DF' if stage.use_df else 'exact integrals'}"
                + (f", frozen {frozen}" if frozen is not None else "") + "), Hartree\n")
        if pos:
            f.write("_e / _p: electronic part (electron and nuclear integrals only) / positronic part (every term "
                    "with a positron integral); HF_e = E_nuc + T_e + V_ne + V_ee, HF_p = T_p + V_np + V_pp + V_ep\n")
            if stage.method in ("CCSD", "CCSD(T)"):
                f.write("MP2 columns: positron DF-MP2 on the same POSHF (pyscf_neo_cc's CC has no MP2 of its own)\n")
        else:
            f.write("E_HF = E_nuc + T_e + V_ne + V_ee (V_ee = Coulomb - exchange)\n")
        f.write(f"{'x (' + unit + ')':>18}" + "".join(f"  {h:>22}" for h, _ in cols) + "\n")
        for x, c in rows:
            vals = [g(c) for _, g in cols]
            f.write(num(x, "18.8f") + "".join("  " + (num(v, "22.12f") if v is not None else f"{'nan':>22}")
                                               for v in vals) + "\n")
    print(f"[Artifact Generated] Scan energy components: {fname}")


def _fit_file_header(f, stage, title, radii, unit):
    f.write("=" * 80 + "\n")
    f.write(f"newBRORSEN {title} SUMMARY: {stage.job_name}\n")
    f.write("=" * 80 + "\n")
    f.write(f"Sector:           {stage.system_type}\n")
    f.write(f"Method:           {stage.method}\n")
    f.write(f"Basis:            {stage.elec_basis}\n")
    if stage.system_type == "POSITRON":
        f.write(f"Positron Basis:   {stage.pos_basis_name}\n")
        f.write(f"Positrons:        {stage.n_pos}\n")
        if describe_particle(stage):
            f.write(f"Particle:         {describe_particle(stage)}\n")
    f.write(f"Coordinate:       {scan_coordinate(stage)[0]}\n")
    f.write(f"Scan Points:      {len(radii)} from {num(radii[0], '.6f')} to {num(radii[-1], '.6f')} {unit}\n")
    f.write("-" * 80 + "\n")


def write_morse_summary(stage, radii, fit_result, extra):
    fname = f"{_job_prefix(stage)}_morse_fit_summary.txt"
    _, unit = scan_coordinate(stage)
    with open(fname, "w") as f:
        _fit_file_header(f, stage, "MORSE SCAN", radii, unit)
        if fit_result is None:
            f.write("Morse fit FAILED to converge -- see scan_points.txt for raw data.\n")
        else:
            De_fit, a_fit, re_fit, E0_fit, r_squared = fit_result
            f.write("Fit form: E(r) = De*(1 - exp(-a*(r-re)))^2 + E0\n")
            f.write(f"re (equilibrium bond length):  {num(re_fit, '.6f')} {'Angstrom' if unit == 'A' else unit}\n")
            f.write(f"De (well depth above E0):      {num(De_fit, '.10f')} Ha "
                    f"({num(De_fit*627.5094740631, '.4f')} kcal/mol)\n")
            f.write(f"a  (Morse width parameter):    {num(a_fit, '.6f')} 1/{'Angstrom' if unit == 'A' else unit}\n")
            f.write(f"E0 (energy at minimum):        {num(E0_fit, '.12f')} Ha\n")
            f.write(f"Dissociation asymptote (E0+De):{num(E0_fit + De_fit, '.12f')} Ha\n")
            f.write(f"Fit quality R^2:               {num(r_squared, '.8f')}\n")
            for line in extra:
                f.write(line + "\n")
        f.write("=" * 80 + "\n")
    print(f"[Artifact Generated] Morse Fit Summary: {fname}")


def write_harmonic_summary(stage, radii, fit_result, extra):
    fname = f"{_job_prefix(stage)}_harmonic_fit_summary.txt"
    _, unit = scan_coordinate(stage)
    with open(fname, "w") as f:
        _fit_file_header(f, stage, "HARMONIC SCAN", radii, unit)
        if fit_result is None:
            f.write("Harmonic fit FAILED -- see scan_points.txt for raw data.\n")
        else:
            k, xe, E0, r2 = fit_result
            f.write("Fit form: E(x) = E0 + k/2 (x - xe)^2\n")
            f.write(f"xe (minimum):                  {num(xe, '.6f')} {'Angstrom' if unit == 'A' else unit}\n")
            f.write(f"E0 (energy at minimum):        {num(E0, '.12f')} Ha\n")
            if unit == "A":
                f.write(f"k  (force constant):           {num(k, '.8f')} Ha/Angstrom^2 "
                        f"({num(k * _HA_PER_A2_TO_N_PER_M / 100.0, '.6f')} mdyn/Angstrom)\n")
            else:
                f.write(f"k  (force constant):           {num(k, '.10e')} Ha/deg^2 "
                        f"({num(k * (180.0 / np.pi)**2, '.8f')} Ha/rad^2)\n")
            f.write(f"Fit quality R^2:               {num(r2, '.10f')}\n")
            for line in extra:
                f.write(line + "\n")
        f.write("=" * 80 + "\n")
    print(f"[Artifact Generated] Harmonic Fit Summary: {fname}")


def write_scan_tsv(stage, rows):
    fname = f"{_job_prefix(stage)}.scan"
    label = "DF (Density Fitted)" if stage.use_df else "CONV (Exact Conventional)"

    def fmt(x):
        return num(x, ".10f") if x is not None else "nan"
    with open(fname, "w", encoding="utf-8") as f:
        f.write("Coordinate\tHF_Ha\tCCSD_Ha\tCCSD(T)_Ha\tTotal_Ha\tIntegral_Type\n")
        for val, comps in rows:
            f.write(f"{num(val, '.6f')}\t{fmt(comps['HF'])}\t{fmt(comps['CCSD'])}\t{fmt(comps['CCSD(T)'])}\t"
                    f"{fmt(comps['total'])}\t{label}\n")
    print(f"[Artifact Generated] Tab-separated scan table: {fname}")


def run_scan(stage):
    """! SCAN: single-point energies along one coordinate, no fit.
    %scan_fit morse / harmonic: the same, then a fit of this stage's
    points only. No gradients are computed.

    Geometry passed to the next stage (v04.05): the lowest-energy scanned point, for all three scan types
    (up to v04.04 a fit passed its fitted minimum), so a following stage (another scan, OPT, ...) starts
    from a computed point."""
    sc = stage.scan
    name, unit = scan_coordinate(stage)
    print("\n" + "=" * 30 + " POTENTIAL ENERGY SCAN " + "=" * 30)
    if sc["mode"] == "fragment":
        syms, coords0, k, axis, r0, gap = _fragment_split(stage)
        label = f"[{'-'.join(syms[:k + 1])}]...[{'-'.join(syms[k + 1:])}]"
        print(f"  Scanning {label} inter-fragment distance (split after atom "
              f"{k + 1} of {len(syms)}, largest gap = {gap:.4f} A in "
              f"the input geometry): {sc['steps']} points from {sc['start']:.4f} to "
              f"{sc['end']:.4f} A")
    else:
        print(f"  Scanning {name}: {sc['steps']} points from {sc['start']:.4f} to {sc['end']:.4f} {unit}")
        if sc.get("segments") and len(sc["segments"]) > 1 or sc.get("custom"):
            print("  Points: " + " ".join(f"{v:.4f}" for v in scan_values(sc)))
    print(f"  Fit: {({None: 'none (plain SCAN)', 'morse': 'Morse', 'harmonic': 'harmonic'})[stage.scan_fit]}")

    rows = []
    for i, (val, atoms_i) in enumerate(scan_geometries(stage), 1):
        if sc["mode"] != "fragment":
            print(f"\n--- Scan Point {i}/{sc['steps']} : value = {val:.6f} {unit} ---")
            print_cartesian(atoms_i)
        comps = run_single_point_components(stage, atoms_i, use_df=stage.use_df)
        rows.append((val, comps, atoms_i))
        print(f"  Point {i:2d}/{sc['steps']}: R = {val:.6f} {unit}   E = {comps['total']:.12f} Ha"
              + (f"   (HF {comps['HF']:.10f}"
                 + (f", MP2 {comps['MP2']:.10f}" if comps['MP2'] is not None else "")
                 + f", CCSD {comps['CCSD']:.10f}"
                 + (f", (T) {comps['CCSD(T)'] - comps['CCSD']:.10f}" if comps['CCSD(T)'] is not None else "")
                 + ")" if comps['CCSD'] is not None else ""))

    radii = np.array([r for r, _, _ in rows], dtype=float)
    energies = np.array([c["total"] for _, c, _ in rows], dtype=float)
    write_scan_tsv(stage, [(r, c) for r, c, _ in rows])
    write_scan_points(stage, radii, energies)
    if "eda" in stage.scan_print:
        write_scan_eda(stage, [(r, c) for r, c, _ in rows])

    print("\n" + "=" * 80)
    print(f"  SCAN SUMMARY ({stage.job_name})   coordinate ({unit})   total energy (Ha)")
    for r, e in zip(radii, energies):
        print(f"  {r:14.6f}   {e:20.12f}")
    print("=" * 80)

    i_low = int(np.argmin(energies))
    lowest = rows[i_low][2]
    if stage.scan_fit is None:
        print(f"  [Scan] No fit requested (no %scan_fit). Passing the lowest-energy point "
              f"({radii[i_low]:.6f} {unit}) to the next stage.")
        return lowest

    mu, mu_desc = scan_reduced_mass(stage) if unit == "A" else (None, "the coordinate is an angle")
    extra = []
    try:
        if stage.scan_fit == "morse":
            fit, residuals = fit_morse(radii, energies)
            x_min = fit[2]
        else:
            fit, residuals = fit_harmonic(radii, energies)
            x_min = fit[1]
    except (RuntimeError, ValueError, TypeError) as e:
        what = "Morse" if stage.scan_fit == "morse" else "Harmonic"
        print(f"\n[{what} Fit] FAILED: {e}")
        print("  Raw scan points are still reported above; try a different %scan range or more points.")
        (write_morse_summary if stage.scan_fit == "morse" else write_harmonic_summary)(stage, radii, None, [])
        print(f"  [Scan] Passing the lowest-energy point ({radii[i_low]:.6f} {unit}) to the next stage.")
        return lowest

    print("\n" + "-" * 80)
    if stage.scan_fit == "morse":
        De_fit, a_fit, re_fit, E0_fit, r_squared = fit
        print("  [Morse Fit] E(r) = De*(1 - exp(-a*(r-re)))^2 + E0")
        print(f"    re (equilibrium bond length): {re_fit:.6f} {unit}")
        print(f"    De (well depth above E0):     {De_fit:.10f} Ha "
              f"({De_fit*627.5094740631:.4f} kcal/mol)")
        print(f"    a  (Morse width parameter):   {a_fit:.6f} 1/{unit}")
        print(f"    E0 (energy at minimum):       {E0_fit:.12f} Ha")
        print(f"    Fit quality R^2:              {r_squared:.8f}")
        print(f"    Max |residual|:               {np.max(np.abs(residuals)):.3e} Ha")
        if mu is not None and De_fit > 0 and a_fit > 0:
            we, wexe = morse_frequencies_cm1(De_fit, a_fit, mu)
            extra = [f"Reduced mass:                  {num(mu, '.6f')} amu ({mu_desc})",
                     f"omega_e (harmonic frequency):  {num(we, '.2f')} cm-1",
                     f"omega_e x_e (anharmonicity):   {num(wexe, '.4f')} cm-1",
                     "(1-D estimates along the scanned coordinate)"]
            print(f"    omega_e:                      {we:.2f} cm-1   omega_e x_e: {wexe:.4f} cm-1 "
                  f"(mu = {mu:.4f} amu, {mu_desc})")
        for w in morse_fit_warnings(radii, energies, fit):
            print(f"    [Warning] Morse fit: {w}")
            extra.append("Warning: " + w)
        write_morse_summary(stage, radii, fit, extra)
    else:
        k, xe, E0, r2 = fit
        print("  [Harmonic Fit] E(x) = E0 + k/2 (x - xe)^2")
        print(f"    xe (minimum):                 {xe:.6f} {unit}")
        print(f"    E0 (energy at minimum):       {E0:.12f} Ha")
        if unit == "A":
            print(f"    k  (force constant):          {k:.8f} Ha/A^2 ({k * _HA_PER_A2_TO_N_PER_M / 100.0:.6f} mdyn/A)")
        else:
            print(f"    k  (force constant):          {k:.10e} Ha/deg^2")
        print(f"    Fit quality R^2:              {r2:.10f}")
        print(f"    Max |residual|:               {np.max(np.abs(residuals)):.3e} Ha")
        if mu is not None:
            we = harmonic_frequency_cm1(k, mu)
            extra = [f"Reduced mass:                  {num(mu, '.6f')} amu ({mu_desc})",
                     f"omega_e (harmonic frequency):  {num(we, '.2f')} cm-1",
                     "(1-D estimate along the scanned coordinate)"]
            print(f"    omega_e:                      {we:.2f} cm-1 (mu = {mu:.4f} amu, {mu_desc})")
        else:
            extra = [f"No frequency: {mu_desc}."]
        dk = cubic_curvature_shift(radii, energies, k, xe)
        if dk is not None:
            print(f"    Anharmonicity check:          k changes by {100*dk:.2f}% when a cubic term is added")
            extra.append(f"Anharmonicity check:           k changes by {num(100*dk, '.2f')}% with a cubic term")
            if dk > 0.02:
                msg = ("the points reach the anharmonic part of the curve (k changes > 2% with a cubic "
                       "term); narrow the %scan range around the minimum for a reliable k")
                print(f"    [Warning] {msg}")
                extra.append("Warning: " + msg)
        write_harmonic_summary(stage, radii, fit, extra)
    print("-" * 80)

    if not radii.min() <= x_min <= radii.max():
        print(f"  [Warning] The fitted minimum ({x_min:.6f} {unit}) is outside the scanned range (an extrapolation).")
    print(f"  [Scan] Passing the lowest-energy point ({radii[i_low]:.6f} {unit}) to the next stage "
          f"(fitted minimum: {x_min:.6f} {unit}).")
    return lowest


# ----------------------------------------------------------------------------
# ZOPT: gradient-free optimization of Z-matrix variables
# ----------------------------------------------------------------------------

def _trust_region_step(g, H, delta):
    """argmin_s g.s + s.H.s/2 subject to |s| <= delta (small dense n)."""
    lam, Q = np.linalg.eigh(H)
    gq = Q.T @ g
    if lam[0] > 1e-12:
        s = -Q @ (gq / lam)
        if np.linalg.norm(s) <= delta:
            return s
    lo = max(0.0, -lam[0]) + 1e-12
    hi = lo + np.linalg.norm(g) / delta + abs(lam).max() + 1.0

    def norm_at(mu):
        return np.linalg.norm(gq / (lam + mu))
    while norm_at(hi) > delta:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if norm_at(mid) > delta:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-12 * max(1.0, hi):
            break
    s = -Q @ (gq / (lam + hi))
    if np.linalg.norm(s) < 1e-12 * delta:           # g ~ 0 on a saddle: move along negative curvature
        s = Q[:, 0] * delta
    return s


def _quadratic_features(S):
    """[1, s_i, s_i s_j (i<=j)] for each row of S."""
    n = S.shape[1]
    cols = [np.ones(len(S))] + [S[:, i] for i in range(n)]
    for i in range(n):
        for j in range(i, n):
            cols.append(S[:, i] * S[:, j] * (0.5 if i == j else 1.0))
    return np.column_stack(cols)


def _quadratic_feature_gradients(s):
    """d/ds_k of the features of _quadratic_features at one point s: an (n, p) array."""
    n = len(s)
    rows = np.zeros((n, 1 + n + n * (n + 1) // 2))
    for k in range(n):
        rows[k, 1 + k] = 1.0
    col = 1 + n
    for i in range(n):
        for j in range(i, n):
            if i == j:
                rows[i, col] = s[i]                    # d(s_i^2 / 2)/ds_i
            else:
                rows[i, col] = s[j]
                rows[j, col] = s[i]
            col += 1
    return rows


def _fit_quadratic(U, E, center, delta, GU=None, G=None):
    """Weighted least-squares quadratic model around `center` from every
    remembered point (closer points weigh more), plus, when given, gradients
    G (dE/du, one row per point in GU): n extra equations each. Returns
    (c, g, H, rank_ok)."""
    n = U.shape[1]
    S = U - center
    d = np.linalg.norm(S, axis=1)
    w = 1.0 / (1.0 + (d / max(delta, 1e-12)) ** 2) ** 2
    A = _quadratic_features(S) * w[:, None]
    b = (E - E.min()) * w
    if GU is not None and len(GU):
        rows, rhs = [A], [b]
        for u, gvec in zip(GU, G):
            sg = u - center
            wg = 1.0 / (1.0 + (np.linalg.norm(sg) / max(delta, 1e-12)) ** 2) ** 2
            rows.append(_quadratic_feature_gradients(sg) * wg)
            rhs.append(np.asarray(gvec, float) * wg)
        A, b = np.vstack(rows), np.concatenate(rhs)
    p = A.shape[1]
    coef, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    ok = rank == p and sv[-1] > 1e-10 * sv[0]
    g = coef[1:1 + n]
    H = np.zeros((n, n))
    k = 1 + n
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = coef[k]
            k += 1
    return coef[0] + E.min(), g, H, ok


def zopt_minimize(func, x0, steps, tols, etol=1e-6, max_evals=60, lower=None, upper=None,
                  cache=None, log=print, grad_func=None, grad_every=0, grad_cache=None):
    """Minimize func(x) (energy only, no gradients) over a few variables.

    Quadratic-model trust region (in the spirit of Powell's UOBYQA): every
    energy computed -- higher or lower than the best so far -- is kept and used
    to fit E ~ c + g.s + s.H.s/2 around the best point; the next trial is the
    model minimum inside a trust radius, which grows after good steps and
    shrinks after bad ones. Variables are scaled by `steps` (the first trial
    displacement of each); convergence when a step changes the energy by less
    than `etol` and every variable by less than its `tols`, or when the trust
    radius falls below the tolerances.

    cache: dict {key: energy} of energies already known (restart); new ones
    are added to it. Returns a dict with x, e, n_evals, n_cached, history,
    converged, reason, n_grads.

    Hybrid (grad_func and grad_every > 0): grad_func(x) -> (E, dE/dx) with an
    analytic gradient, used where it replaces several energies: at the start
    and at the centre of every rebuilt local model (the - samples are dropped:
    the gradient gives those n equations), at the best point to confirm
    convergence (instead of re-sampling a small pattern), and on every
    grad_every-th model step. Each gradient adds n equations to the fit."""
    x0 = np.asarray(x0, float)
    steps = np.asarray(steps, float)
    tols = np.asarray(tols, float)
    n = len(x0)
    lower = np.full(n, -np.inf) if lower is None else np.asarray(lower, float)
    upper = np.full(n, np.inf) if upper is None else np.asarray(upper, float)
    cache = {} if cache is None else cache
    grad_cache = {} if grad_cache is None else grad_cache
    hybrid = grad_func is not None and grad_every > 0
    history = []                      # (x, E, status)
    U, E = [], []
    GU, G = [], []                    # points with a gradient, dE/du there
    counts = {"computed": 0, "cached": 0, "grads": 0}

    def key(x):
        return tuple(round(float(v), 8) for v in x)

    def clip_u(u):
        """The point the evaluation will really use: variables clipped to their bounds (e.g. angles <= 179 deg)."""
        return (np.clip(x0 + np.asarray(u, float) * steps, lower, upper) - x0) / steps

    stall = {"cached_in_a_row": 0}

    def evaluate(u, label, want_grad=False):
        x = np.clip(x0 + u * steps, lower, upper)
        u = (x - x0) / steps
        k = key(x)
        want_grad = hybrid and want_grad
        gx = None
        if k in cache and (not want_grad or k in grad_cache):
            e = cache[k]
            gx = grad_cache.get(k)
            counts["cached"] += 1
            src = "cached"
            # a point already known adds nothing; many in a row means the search is going round in circles
            # (2026-10-05: a local pattern pushed past a bound was clipped back onto its centre, forever)
            stall["cached_in_a_row"] += 1
            if stall["cached_in_a_row"] > 4 * (n + 1) * (n + 2):
                raise RuntimeError("stalled")
        else:
            if counts["computed"] >= max_evals:
                raise StopIteration
            if want_grad:
                e, gx = grad_func(x)
                e = float(e)
                if gx is not None:
                    gx = np.asarray(gx, float)
                    grad_cache[k] = gx
                    counts["grads"] += 1
            else:
                e = float(func(x))
            cache[k] = e
            counts["computed"] += 1
            src = "computed"
            stall["cached_in_a_row"] = 0
        U.append(u)
        E.append(e)
        if gx is not None:
            GU.append(u)
            G.append(np.asarray(gx, float) * steps)          # dE/du
            label = label + ", +gradient"
        history.append((x.copy(), e, label))
        log(x, e, label, src, len(history))
        return u, e

    def has_grad(u):
        return any(np.allclose(gu, u, atol=1e-9) for gu in GU)

    def add_grad(u, label):
        """Gradient at a point whose energy is already known (E recomputed with it, not re-added)."""
        if not hybrid or has_grad(u):
            return
        x = np.clip(x0 + u * steps, lower, upper)
        k = key(x)
        if k in grad_cache:
            gx = grad_cache[k]
            src = "cached"
            counts["cached"] += 1
        else:
            if counts["computed"] >= max_evals:
                raise StopIteration
            e, gx = grad_func(x)
            counts["computed"] += 1
            if gx is None:
                GU.append(u)            # remember the attempt so it is not repeated; no equations added
                G.append(None)
                return
            gx = np.asarray(gx, float)
            grad_cache[k] = gx
            counts["grads"] += 1
            src = "computed"
        GU.append(u)
        G.append(np.asarray(gx, float) * steps)
        e_known = cache.get(k, float("nan"))
        history.append((x.copy(), e_known, label + " (gradient only)"))
        log(x, e_known, label + " (gradient only)", src, len(history))

    delta_min = float(np.min(tols / steps))
    delta_check = 3.0 * delta_min            # model radius used to confirm convergence
    delta, delta_max = 1.0, 8.0
    result = {"converged": False, "reason": ""}
    try:
        u_best, e_best = evaluate(np.zeros(n), "start", want_grad=True)
        # initial pattern: +/- one step along each variable (only + when the
        # start has a gradient), then pair diagonals
        for i in range(n):
            for sign in ((1.0,) if GU else (1.0, -1.0)):
                u = np.zeros(n)
                u[i] = sign
                u, e = evaluate(u, f"pattern {'+' if sign > 0 else '-'}{i + 1}")
                if e < e_best:
                    u_best, e_best = u, e
        for i in range(n):
            for j in range(i + 1, n):
                u = np.zeros(n)
                u[i] = u[j] = 1.0
                u, e = evaluate(u, f"pattern +{i + 1}+{j + 1}")
                if e < e_best:
                    u_best, e_best = u, e

        n_model = (n + 1) * (n + 2) // 2

        def local_design(center, radius):
            """The same +/- and pair pattern as the start, scaled to `radius` (only + when the
            centre has a gradient)."""
            pts = []
            centre_grad = any(np.allclose(gu, center, atol=1e-9) and gv is not None for gu, gv in zip(GU, G))
            for i in range(n):
                for sign in ((1.0,) if centre_grad else (1.0, -1.0)):
                    u = center.copy()
                    u[i] += sign * radius
                    pts.append(u)
            for i in range(n):
                for j in range(i + 1, n):
                    u = center.copy()
                    u[i] += radius
                    u[j] += radius
                    pts.append(u)
            return pts

        anchor = None                   # centre of a local pattern being completed
        model_steps = 0
        while True:
            Ua, Ea = np.array(U), np.array(E)
            # the model only uses points close to the best one: a quadratic is a
            # good description of the energy only locally. While a local pattern
            # is being filled in, it stays centred where it started even if one
            # of its points turns out lower.
            ref = u_best if anchor is None else anchor
            dist = np.linalg.norm(Ua - ref, axis=1)
            ok = False
            GUa = np.array(GU) if GU else np.zeros((0, n))
            gdist = np.linalg.norm(GUa - ref, axis=1) if GU else np.zeros(0)
            for reach in (2.0, 3.0):        # widen once before computing new points
                near = dist <= reach * delta + 1e-9
                gnear = (gdist <= reach * delta + 1e-9) & np.array([gv is not None for gv in G], bool)
                if near.sum() + n * gnear.sum() >= n_model and near.sum() >= 1:
                    gi = [i for i in np.flatnonzero(gnear) if G[i] is not None]
                    c, g, H, ok = _fit_quadratic(Ua[near], Ea[near], u_best, delta,
                                                 GUa[gi], [G[i] for i in gi])
                    if ok:
                        break
            if not ok:
                if anchor is None:
                    anchor = u_best.copy()
                    add_grad(anchor, "gradient at the new model centre")
                    continue
                # compare the points as they will really be evaluated (clipped to the bounds): a design point
                # beyond a bound lands on (or near) a point already known and must not be asked for again
                missing = [cu for cu in (clip_u(u) for u in local_design(anchor, delta))
                           if np.min(np.linalg.norm(Ua - cu, axis=1)) > 0.25 * delta]
                if not missing:                 # pattern present but degenerate (bounds): shrink
                    anchor = None
                    delta *= 0.5
                    if delta < delta_min:
                        result.update(converged=True, reason="trust radius below the tolerances")
                        break
                    continue
                u, e = evaluate(missing[0], f"local sample (radius {delta:.3g} steps)")
                if e < e_best:
                    u_best, e_best = u, e
                continue
            anchor = None
            # Converged when this local model puts its minimum within the
            # tolerances of the best point and predicts less than etol to gain.
            lam = np.linalg.eigvalsh(H)
            if lam[0] > 0:
                newton = -np.linalg.solve(H, g)
                gain = -(g @ newton + 0.5 * newton @ H @ newton)
                if hybrid and np.all(np.abs(newton * steps) < tols) and gain < etol:
                    if not has_grad(u_best):
                        add_grad(u_best, "gradient at the best point (convergence check)")
                        continue                # refit with it, then judge with the exact gradient
                    gb = next((gv for gu, gv in zip(GU, G) if gv is not None
                               and np.allclose(gu, u_best, atol=1e-9)), None)
                    if gb is not None:
                        # judge convergence with the exact gradient at the best point, not the fitted one
                        newton = -np.linalg.solve(H, gb)
                        gain = -(gb @ newton + 0.5 * newton @ H @ newton)
                        g = gb
                if np.all(np.abs(newton * steps) < tols) and gain < etol:
                    if delta > delta_check and not hybrid:   # confirm with a model built from close points only
                        delta = delta_check
                        anchor = u_best.copy()
                        continue
                    # finish on the model's minimum itself (one more energy), not on
                    # the best sampled point, which can sit up to a tolerance away
                    if np.linalg.norm(newton) > 1e-3 * delta_check:
                        u_fin, e_fin = evaluate(u_best + newton, "model minimum (final)")
                        if e_fin < e_best:
                            u_best, e_best = u_fin, e_fin
                    result.update(converged=True, reason="model minimum within tolerance of the best point")
                    break
            s = _trust_region_step(g, H, delta)
            pred = -(g @ s + 0.5 * s @ H @ s)
            if pred <= 0.0 or np.linalg.norm(s) < 1e-12:
                delta *= 0.5
                if delta < delta_min:
                    result.update(converged=True, reason="trust radius below the tolerances")
                    break
                continue
            model_steps += 1
            u_new, e_new = evaluate(u_best + s, "model step",
                                    want_grad=bool(grad_every) and model_steps % grad_every == 0)
            rho = (e_best - e_new) / pred
            if e_new < e_best:
                u_best, e_best = u_new, e_new
            if rho > 0.75 and np.linalg.norm(s) > 0.9 * delta:
                delta = min(2.0 * delta, delta_max)
            elif rho < 0.25:
                delta = 0.5 * min(delta, np.linalg.norm(s))
            if delta < delta_min:
                result.update(converged=True, reason="trust radius below the tolerances")
                break
    except StopIteration:
        result.update(converged=False, reason=f"max energy evaluations ({max_evals}) reached")
    except RuntimeError as err:
        if str(err) != "stalled":
            raise
        result.update(converged=False, reason="stalled: no new points could be made (a variable at its bound?)")
    xb = np.clip(x0 + u_best * steps, lower, upper)
    result["at_bound"] = [i for i in range(n) if np.isfinite(upper[i]) and abs(xb[i] - upper[i]) < 1e-6
                          or np.isfinite(lower[i]) and abs(xb[i] - lower[i]) < 1e-6]
    result.update(x=np.clip(x0 + u_best * steps, lower, upper), e=e_best, n_evals=counts["computed"],
                  n_cached=counts["cached"], n_grads=counts["grads"], history=history)
    return result


def zmat_variable_kinds(rows):
    """{name: 'bond'|'angle'|'dihedral'} from the column each variable occupies."""
    kinds = {}
    for row in rows:
        for pos, tok in enumerate(row[1:], 1):
            name = tok.lstrip('-')
            try:
                float(name)
                continue
            except ValueError:
                pass
            if pos in (1, 3, 5):
                continue                                  # reference atoms
            kind = {2: "bond", 4: "angle", 6: "dihedral"}[pos]
            if kinds.get(name, kind) != kind:
                raise ValueError(f"Z-matrix variable '{name}' is used both as a {kinds[name]} and as a {kind}")
            kinds[name] = kind
    return kinds


def _auto_zmat_bond_variables(stage):
    """v04.07: a ZOPT Z-matrix written with numbers only. Bond lengths that join the same pair of elements and
    have the same value (to 1e-4 A) become one variable each (rCC, rCH; rCH_2, ... when one pair has several
    values), so equal bonds stay equal; angles and dihedrals stay fixed (equal angles are not tied: in a ring the
    ring angles and the substituent angles can share a value without being equivalent). Rewrites the rows,
    sets the values in stage.zmat_vars and warns with the assignment. Returns {name: 'bond'}."""
    rows = stage.zmat_rows
    labels = [r[0] for r in rows]

    def elem(label):
        e = re.sub(r"\d+$", "", str(label))
        return "X" if is_dummy_label(label) else e[:1].upper() + e[1:].lower()

    groups = {}                                   # (pair, value) -> [row index]
    for k, row in enumerate(rows[1:], 1):
        if len(row) < 3:
            continue
        ref = row[1]
        j = int(ref) - 1 if ref.lstrip("-").isdigit() else (labels.index(ref) if ref in labels else None)
        try:
            val = float(row[2])
        except ValueError:
            continue
        if j is None or not 0 <= j < len(rows):
            continue
        pair = "".join(sorted((elem(row[0]), elem(labels[j])), key=lambda e: (e == "H", e == "X", e)))
        key = next((g for g in groups if g[0] == pair and abs(g[1] - val) < 1e-4), (pair, val))
        groups.setdefault(key, []).append(k)
    per_pair = {}
    for pair, _ in groups:
        per_pair[pair] = per_pair.get(pair, 0) + 1
    seen, kinds, lines = {}, {}, []
    for (pair, val), ks in groups.items():
        seen[pair] = seen.get(pair, 0) + 1
        name = f"r{pair}" + (f"_{seen[pair]}" if per_pair[pair] > 1 else "")
        for k in ks:
            stage.zmat_rows[k][2] = name
        stage.zmat_vars[name] = val
        kinds[name] = "bond"
        lines.append(f"{name} = {val:g} A ({len(ks)} bond{'s' * (len(ks) != 1)}, rows {', '.join(str(k + 1) for k in ks)})")
    if kinds:
        stage.warn("ZOPT: the Z-matrix has no named variables, so equal bond lengths between the same elements "
                   "were made variables (angles and dihedrals stay fixed): " + "; ".join(lines) + ". Check that "
                   "bonds sharing a variable are really equivalent; name the variables yourself to choose.")
    return kinds


def _zopt_signature(stage):
    """Identify the fixed Z-matrix surface and installed sources, independent of the start guess."""
    return _cache_digest("zopt-v2", _restart_source_identity(),
                         _surface_definition(stage), stage.zmat_rows)


def _read_zmat_values(fname):
    vals = {}
    for ln in open(fname):
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*=\s*(\S+)\s*$", ln)
        if m:
            vals[m.group(1)] = float(m.group(2))
    return vals


def _continue_zmat(stage, prev):
    """A ZOPT stage without a geometry block starts from the Z-matrix variables the stage before ended with: its
    in-memory values after it ran, or its <job>_final.zmat when it was skipped by --resume."""
    vals = dict(prev.zmat_vars)
    f = f"{prev.job_name}_final.zmat"
    if prev.run_type == "ZOPT" and os.path.exists(f):
        try:
            vals.update(_read_zmat_values(f))
        except Exception:
            pass
    stage.zmat_vars.update(vals)
    syms, coords = zmatrix_to_cartesian(stage.zmat_rows, stage.zmat_vars)
    stage.atoms = [(sym, tuple(c)) for sym, c in zip(syms, coords)]
    print(f"[Stage {stage.stage_id}] ZOPT continues stage {prev.stage_id}'s Z-matrix: "
          + ", ".join(f"{k} = {v:.6f}" for k, v in stage.zmat_vars.items()))


def write_zmat_file(stage, values, fname):
    with open(fname, "w") as f:
        f.write(f"* zmat {stage.charge} {stage.multiplicity}\n")
        for row in stage.zmat_rows:
            f.write("  ".join(row) + "\n")
        for name, v in values.items():
            f.write(f"{name} = {v:.8f}\n")
        f.write("*\n")
    print(f"[Artifact Generated] Optimized Z-matrix: {fname}")


def zmat_atoms(stage, names, x):
    """Atoms of the stage Z-matrix with the variables `names` set to `x` (dummies removed)."""
    variables = dict(stage.zmat_vars)
    variables.update(zip(names, (float(v) for v in x)))
    syms, coords = zmatrix_to_cartesian(stage.zmat_rows, variables)
    return [(sym, tuple(c)) for sym, c in zip(syms, coords)]


def zmat_energy_and_gradient(stage, names, x, h=1e-4):
    """(E, dE/d variable) at Z-matrix values x: the Cartesian analytic gradient (Hartree/A) through
    the chain rule dE/dq = sum_atoms dE/dX . dX/dq, with dX/dq by central differences of the Z-matrix
    construction (geometry only, no energies). Hartree/A for bonds, Hartree/deg for angles and
    dihedrals. Gradient None when the analytic one fails."""
    x = np.asarray(x, float)
    e, g_cart = energy_and_forces(stage, atoms_to_string(zmat_atoms(stage, names, x)))
    if g_cart is None:
        return e, None
    g_cart = np.asarray(g_cart, float)
    gq = np.zeros(len(x))
    for k in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        dX = (np.array([c for _, c in zmat_atoms(stage, names, xp)])
              - np.array([c for _, c in zmat_atoms(stage, names, xm)])) / (2 * h)
        gq[k] = float(np.sum(g_cart * dX))
    return e, gq


def run_zopt(stage):
    """! ZOPT: minimize the energy over the named Z-matrix variables only,
    from energies alone (no gradients). A variable shared by several rows
    (e.g. CC in benzene) moves all of them together, so the symmetry written
    in the Z-matrix is kept exactly; numbers in the Z-matrix stay fixed."""
    import json
    _prepare_conditioning(stage)
    print("\n" + "=" * 30 + " ZOPT: Z-MATRIX ENERGY-ONLY OPTIMIZER " + "=" * 30)
    kinds = zmat_variable_kinds(stage.zmat_rows)
    names = [nm for nm in dict.fromkeys(t.lstrip('-') for row in stage.zmat_rows for t in row[1:])
             if nm in kinds]
    x0 = np.array([stage.zmat_vars[nm] for nm in names], float)
    z = stage.zopt
    steps = np.array([z["step_bond"] if kinds[nm] == "bond" else z["step_angle"] for nm in names])
    tols = np.array([z["tol_bond"] if kinds[nm] == "bond" else z["tol_angle"] for nm in names])
    lower = np.array([0.3 if kinds[nm] == "bond" else (1.0 if kinds[nm] == "angle" else -np.inf) for nm in names])
    upper = np.array([np.inf if kinds[nm] == "bond" else (179.0 if kinds[nm] == "angle" else np.inf) for nm in names])
    units = {"bond": "A", "angle": "deg", "dihedral": "deg"}
    print(f"  Variables ({len(names)}): " + ", ".join(
        f"{nm} = {v:.6f} {units[kinds[nm]]} ({kinds[nm]}, used {sum(t.lstrip('-') == nm for r in stage.zmat_rows for t in r[1:])}x)"
        for nm, v in zip(names, x0)))
    print(f"  Fixed values: every number written in the Z-matrix. Atoms: {len(stage.atoms)} "
          f"(Cartesian coordinates: {3 * len(stage.atoms)}; optimized variables: {len(names)})")
    print(f"  First step: {z['step_bond']} A (bonds), {z['step_angle']} deg (angles/dihedrals); "
          f"tolerances: dE < {z['etol']:.1e} Ha, {z['tol_bond']} A, {z['tol_angle']} deg; "
          f"max {z['max_evals']} energy evaluations")

    job = stage.job_name if stage.job_name else f"Stage_{stage.stage_id}"
    hist_file = f"{job}_zopt_history.json"
    signature = _zopt_signature(stage)
    cache = {}
    if stage.zopt_restart and os.path.exists(hist_file):
        try:
            saved = json.load(open(hist_file))
            if saved.get("signature") == signature and saved.get("variables") == names:
                cache = {tuple(k): v for k, v in saved["energies"]}
                print(f"  [Restart] {len(cache)} energies reused from {hist_file}.")
            else:
                print(f"  [Restart] {hist_file} belongs to a different calculation (settings or Z-matrix differ); ignored.")
        except (ValueError, KeyError, TypeError) as err:
            print(f"  [Restart] could not read {hist_file} ({err}); ignored.")

    def atoms_for(x):
        variables = dict(stage.zmat_vars)
        variables.update(zip(names, (float(v) for v in x)))
        syms, coords = zmatrix_to_cartesian(stage.zmat_rows, variables)
        return [(s, tuple(c)) for s, c in zip(syms, coords)]

    def energy(x):
        e = run_single_point_calculation(stage, atoms_for(x), use_df=stage.use_df)
        cache[tuple(round(float(v), 8) for v in x)] = e
        save_all()                      # energies and any gradients (defined below, before use)
        gc.collect()
        return e

    table = []
    best = [np.inf]
    t_wall = [time.time()]

    def log(x, e, label, src, i):
        mark = ""
        if e < best[0]:
            best[0], mark = e, "  <- best"
        vals = "  ".join(f"{nm}={v:.6f}" for nm, v in zip(names, x))
        wall = time.time() - t_wall[0]
        t_wall[0] = time.time()
        print(f"  Eval {i:3d} [{src:8s}] {vals}   E = {e:.10f} Ha   ({label}; {wall:.1f} s){mark}")
        table.append([i, src, label, *x, e])

    grad_cache = {}
    if stage.zopt_restart and os.path.exists(hist_file):
        try:
            saved = json.load(open(hist_file))
            if saved.get("signature") == signature and saved.get("variables") == names:
                grad_cache = {tuple(k): np.array(v) for k, v in saved.get("gradients", [])}
        except (ValueError, KeyError, TypeError):
            pass

    def save_all():
        _save_conditioning(stage)
        current_signature = (_zopt_signature(stage)
            if getattr(stage, "aux_conditioning", None) is not None else signature)
        with open(hist_file, "w") as f:
            json.dump({"signature": current_signature, "variables": names,
                       "energies": [[list(k), v] for k, v in cache.items()],
                       "gradients": [[list(k), list(map(float, v))] for k, v in grad_cache.items()]}, f)

    def variable_gradient(x):
        e, gq = zmat_energy_and_gradient(stage, names, x)
        cache[tuple(round(float(v), 8) for v in x)] = float(e)
        if gq is None:
            print("  [ZOPT] analytic gradient failed at this geometry; the point is kept as an energy only")
        else:
            grad_cache[tuple(round(float(v), 8) for v in x)] = gq
        save_all()
        gc.collect()
        return e, gq

    hybrid_every = int(stage.zopt_gradient_every)
    if hybrid_every:
        print(f"  Hybrid: analytic gradients at the start, at each rebuilt model centre, to confirm "
              f"convergence and every {hybrid_every} model step(s) (%zopt_gradient_every)")
    res = zopt_minimize(energy, x0, steps, tols, etol=z["etol"], max_evals=z["max_evals"],
                        lower=lower, upper=upper, cache=cache, log=log,
                        grad_func=variable_gradient if hybrid_every else None, grad_every=hybrid_every,
                        grad_cache=grad_cache)
    save_all()
    x_opt, e_opt = res["x"], res["e"]
    values = dict(zip(names, (float(v) for v in x_opt)))
    opt_atoms = atoms_for(x_opt)

    tsv = f"{job}_zopt.txt"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("Eval\tSource\tStep\t" + "\t".join(f"{nm}_{units[kinds[nm]]}" for nm in names) + "\tE_Ha\n")
        for row in table:
            f.write(f"{row[0]}\t{row[1]}\t{row[2]}\t" + "\t".join(num(v, '.8f') for v in row[3:-1])
                    + f"\t{num(row[-1], '.12f')}\n")
    print(f"[Artifact Generated] ZOPT history table: {tsv}")

    print("\n" + "-" * 80)
    status = "CONVERGED" if res["converged"] else "NOT CONVERGED"
    print(f"  [ZOPT] {status}: {res['reason']}")
    for i in res.get("at_bound", []):
        nm = names[i]
        print(f"  [WARNING] ZOPT: {nm} ended at its limit ({values[nm]:.1f} {units[kinds[nm]]}). The energy still goes "
              f"down past it, so the true minimum lies outside what this Z-matrix can describe (an angle of 180 deg "
              f"makes the next dihedral undefined). Rewrite the Z-matrix (e.g. with a dummy atom on the axis, or the "
              f"other atom as the angle's vertex) or optimize with %opt_engine geometric.")
    n_g = res.get("n_grads", 0)
    print(f"  [ZOPT] Evaluations: {res['n_evals']} computed ({res['n_evals'] - n_g} energies, {n_g} "
          f"energies with an analytic gradient), {res['n_cached']} reused")
    final_grad = None
    if stage.zopt_final_gradient:
        k_opt = tuple(round(float(v), 8) for v in x_opt)
        final_grad = grad_cache.get(k_opt)
        if final_grad is None:
            _, final_grad = variable_gradient(np.array(x_opt, float))
        if final_grad is not None:
            worst = max(abs(gv) * tl for gv, tl in zip(final_grad, tols))
            print("  [ZOPT] Analytic gradient at the result (convergence check):")
            for nm, gv, tl in zip(names, final_grad, tols):
                print(f"    dE/d{nm:<8s} = {gv:+.3e} Ha/{units[kinds[nm]]}   (energy change for a {tl:g} "
                      f"{units[kinds[nm]]} move: {abs(gv) * tl:.1e} Ha)")
            verdict = "stationary within the tolerances" if worst < z["etol"] else "NOT stationary"
            print(f"  [ZOPT] Largest energy change for a tolerance-sized move: {worst:.1e} Ha "
                  f"(etol {z['etol']:.0e}): {verdict}")
    for nm in names:
        print(f"    {nm:>10s} = {values[nm]:.6f} {units[kinds[nm]]}   (start {stage.zmat_vars[nm]:.6f})")
    print(f"    Energy     = {e_opt:.12f} Ha")
    print("-" * 80)
    if not res["converged"]:
        print("  [Warning] ZOPT stopped before converging; the geometry passed on is the lowest-energy one found. "
              "Rerun the same input to continue (energies are reused from the history file).")
    write_zmat_file(stage, values, f"{job}_final.zmat")

    stage.zmat_vars.update(values)
    e_conv = None
    if stage.use_df and _conv_check_enabled(stage):
        write_stage_artifacts(stage, opt_atoms, e_df=e_opt, e_conv=None)
        print("  [Saved] final geometry and DF energy written before the CONV check (memory-safe).")

        try:
            e_conv = run_single_point_calculation(stage, opt_atoms, use_df=False)
            print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_conv:.12f} Ha")
        except Exception as err:
            print(f"  [Notice] CONV cross-check failed ({err}); continuing with DF result only.")
    if stage.use_df:
        print(f"  [Energy Report] Density Fitting (DF/RI) Energy : {e_opt:.12f} Ha")
        comps_df = report_components(stage, opt_atoms, True)
        comps_conv = report_components(stage, opt_atoms, False) if e_conv is not None else None
        write_stage_artifacts(stage, opt_atoms, e_df=e_opt, e_conv=e_conv, comps_df=comps_df, comps_conv=comps_conv)
    else:
        print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_opt:.12f} Ha")
        comps_conv = report_components(stage, opt_atoms, False)
        write_stage_artifacts(stage, opt_atoms, e_df=None, e_conv=e_opt, comps_conv=comps_conv)
    with open(f"{job}_summary.txt", "a") as f:
        f.write(f"ZOPT: {status} ({res['reason']}); {res['n_evals']} evaluations computed "
                f"({res.get('n_grads', 0)} with analytic gradients), {res['n_cached']} reused\n")
        if final_grad is not None:
            f.write("ZOPT final gradient: " + ", ".join(
                f"dE/d{nm} = {num(gv, '+.3e')} Ha/{units[kinds[nm]]}" for nm, gv in zip(names, final_grad)) + "\n")
        for nm in names:
            f.write(f"  {nm} = {num(values[nm], '.6f')} {units[kinds[nm]]}\n")
    return opt_atoms


# ----------------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------------

def apply_start_geometry_constraints(stage):
    """Set %pinnings values and rebuild %equivalences targets in the start
    geometry (twopos/v5 semantics), reporting any atom that moves."""
    if not (stage.pinnings_xyz or stage.equivalences_xyz):
        return
    coords = np.array([c for _, c in stage.atoms], dtype=float)
    new, _ = apply_xyz_pinnings_and_equivalences(coords, None, stage)
    for i, (old_c, new_c) in enumerate(zip(coords, new)):
        if np.max(np.abs(old_c - new_c)) > 1e-8:
            print(f"  [Pinnings/Equivalences] Atom {i+1} ({stage.atoms[i][0]}) moved from "
                  f"({old_c[0]:.6f}, {old_c[1]:.6f}, {old_c[2]:.6f}) to "
                  f"({new_c[0]:.6f}, {new_c[1]:.6f}, {new_c[2]:.6f})")
    stage.atoms = [(s, tuple(c)) for (s, _), c in zip(stage.atoms, new)]


def write_geometric_constraints(stage):
    """geomeTRIC constraints file for %constraints, %pin and %pinnings, or None."""
    freeze, set_lines = [], []
    for c in stage.constraints:
        name = "distance" if c["type"] in ("bond", "distance") else c["type"]
        idx = " ".join(str(i) for i in c["indices"])
        if c["target_value"] is None:
            freeze.append(f"{name} {idx}")
        else:
            set_lines.append(f"{name} {idx} {c['target_value']}")
    for p in stage.pinnings:
        freeze.append(f"xyz {p}")
    for (at, ax, _) in stage.pinnings_xyz:
        freeze.append(f"{'xyz'[ax]} {at + 1}")
    if not freeze and not set_lines:
        return None
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as tf:
        if freeze:
            tf.write("$freeze\n" + "\n".join(freeze) + "\n")
        if set_lines:
            tf.write("$set\n" + "\n".join(set_lines) + "\n")
        return tf.name


def energy_and_forces(stage, atom_str):
    """(energy, dE/dx in Hartree/Angstrom) of the stage method at `atom_str`, gradient None when no
    analytic one exists. Fresh SCF/CC objects every call (no warm start). Used by WOPT, and by ZOPT's
    optional gradients."""
    if stage.method == "MP2":
        mf, mp = build_mp2(stage, atom_str, gradient_required=True)
        energy = float(mp.e_tot)
        _record_components(stage, atom_str, stage.use_df, float(mf.e_tot), mp2=energy,
                           partition=safe_positron_partition(stage, mf, mp=mp))
        g = mp2_gradient(stage, mp)
        return energy, (None if g is None else g * ANGSTROM_TO_BOHR)

    if stage.system_type == "PROTON":
        mf = proton_mean_field(stage, atom_str)
        cc_obj, energy = None, float(mf.e_tot)
        e_cc = e_t = None
        if stage.method in ["CCSD", "CCSD(T)"]:
            cc_obj, e_cc, e_t = proton_correlated(stage, mf, gradient_required=True)
            energy = e_cc + (e_t or 0.0)
        _record_components(stage, atom_str, stage.use_df, float(mf.e_tot), ccsd=e_cc, t=e_t)
        try:
            return energy, proton_gradient(stage, mf, cc_obj) * ANGSTROM_TO_BOHR
        except Exception as err:
            print(f"  [Notice] PROTON analytic gradient failed ({type(err).__name__}: {err}); numerical gradient")
            return energy, None

    if stage.system_type == "POSITRON":
        system = safe_build_system(stage, atom_str)
        mf = make_poshf(system, stage)
        if stage.method == "HF" and _native_poshf_preflight is not None:
            poshf_gradient_preflight(mf)
        mf.verbose = 0
        run_poshf(mf)
        if stage.method in ["CCSD", "CCSD(T)"]:
            analytic = analytic_gradient_available(stage)[0]
            cc_obj = build_positron_cc(mf, stage, gradient_required=analytic)
            cc_obj.verbose = 0
            cc_obj.kernel()
            energy = float(cc_obj.e_tot)
            e_t = _triples(cc_obj, stage) if stage.method == "CCSD(T)" else None
            energy += e_t or 0.0
            _record_components(stage, atom_str, stage.use_df, float(mf.e_tot), ccsd=float(cc_obj.e_tot), t=e_t,
                               partition=safe_positron_partition(stage, mf, cc_obj, e_t))
            if not analytic:
                return energy, None
            try:
                return energy, positron_cc_gradient(stage, cc_obj) * ANGSTROM_TO_BOHR
            except Exception:
                return energy, None
        energy = float(mf.e_tot)
        _record_components(stage, atom_str, stage.use_df, energy, partition=safe_positron_partition(stage, mf))
        try:
            return energy, mf.Gradients().kernel() * ANGSTROM_TO_BOHR
        except Exception:
            return energy, None

    if stage.system_type == "ELECTRONIC":
        mf, open_shell = _electronic_mf(stage, atom_str, stage.use_df)
        mf.verbose = 0
        mf.kernel()
        if stage.method in ["CCSD", "CCSD(T)"]:
            cc_obj = _electronic_cc(mf, stage)
            cc_obj.verbose = 0
            cc_obj.kernel()
            energy = float(cc_obj.e_tot)
            e_t = _triples(cc_obj, stage) if stage.method == "CCSD(T)" else None
            energy += e_t or 0.0
            _record_components(stage, atom_str, stage.use_df, float(mf.e_tot), mp2=cc_mp2_energy(cc_obj, mf),
                               ccsd=float(cc_obj.e_tot), t=e_t)
            try:
                g = electronic_cc_gradient(stage, cc_obj, open_shell)
            except Exception:
                g = None
            return energy, (None if g is None else g * ANGSTROM_TO_BOHR)
        energy = float(mf.e_tot)
        _record_components(stage, atom_str, stage.use_df, energy)
        try:
            return energy, mf.Gradients().kernel() * ANGSTROM_TO_BOHR
        except Exception:
            return energy, None
    return 0.0, None


def run_wrapper_gradient_optimizer(stage):
    """BFGS with backtracking line search, rigid-body projection, internal
    coordinate constraints, whole-atom pins, coordinate pinnings and both
    kinds of equivalence. Fresh SCF/CC objects at every geometry (no warm
    start), so it avoids the library's scanner bugs."""
    print("\n" + "=" * 30 + " WRAPPER GEOMETRY OPTIMIZER " + "=" * 30)
    print("Executing analytical-first relaxation driver with Pinnings & Equivalences...")
    apply_start_geometry_constraints(stage)
    coords = np.array([c for _, c in stage.atoms], dtype=float)
    symbols = [s for s, _ in stage.atoms]

    def compute_energy_and_forces(curr_coords):
        curr_atoms = [(s, tuple(pos)) for s, pos in zip(symbols, curr_coords)]
        return energy_and_forces(stage, atoms_to_string(curr_atoms))

    # Coordinates that must never move: pinned atoms, pinned coordinates and
    # equivalence targets (rebuilt from their source atoms).
    frozen = np.zeros_like(coords, dtype=bool)
    for p in stage.pinnings:
        frozen[p - 1, :] = True
    for (at, ax, _) in stage.pinnings_xyz:
        frozen[at, ax] = True
    for (t_at, *_rest) in stage.equivalences_xyz:
        frozen[t_at, :] = True
    mask = frozen if frozen.any() else None

    def constrain(vec, at_coords):
        """Project a gradient / step: rigid body, pins and equivalences,
        then the internal constraints on the free coordinates only."""
        v = project_translation_rotation(at_coords, vec)
        v = zero_pinned_atoms(v, stage)
        _, v = apply_xyz_pinnings_and_equivalences(None, v, stage)
        if active_constraints:
            v = project_vector_null_space(at_coords, v.flatten(), active_constraints, mask)
        return v

    active_constraints = []
    for c in stage.constraints:
        if len(c["indices"]) not in (2, 3, 4):
            continue
        indices0 = [i - 1 for i in c["indices"]]
        val0, _ = compute_constraint_value_and_gradient(coords, c["type"], indices0)
        target = c["target_value"] if c["target_value"] is not None else val0
        active_constraints.append({"type": c["type"], "indices": indices0, "target_value": target})
    for (a1, a2), (b1, b2) in stage.equivalences:
        active_constraints.append({"type": "bond_equal", "indices": [a1 - 1, a2 - 1, b1 - 1, b2 - 1],
                                   "target_value": 0.0})
        print(f"  [Equivalence Active] Bond {a1}-{a2} held equal to bond {b1}-{b2}.")
    if active_constraints:
        start = coords.copy()
        coords = restore_constraints(coords, active_constraints, mask=mask)
        print(f"  [Constraints Active] {len(active_constraints)} internal-coordinate constraint(s) enforced"
              + (f" (start geometry adjusted by up to {np.max(np.abs(coords - start)):.4f} A)."
                 if np.max(np.abs(coords - start)) > 1e-8 else "."))
    if stage.pinnings:
        print(f"  [Pinning Active] Atoms {stage.pinnings} fixed in space.")
    if stage.pinnings_xyz or stage.equivalences_xyz:
        print(f"  [Pinnings/Equivalences Active] {len(stage.pinnings_xyz)} pinned coordinate(s), "
              f"{len(stage.equivalences_xyz)} equivalence(s).")

    e_curr, grad_curr = compute_energy_and_forces(coords)
    if grad_curr is not None:
        grad_curr = constrain(grad_curr, coords)
    print(f"  Cycle  0 : Initial Energy = {e_curr:.12f} Ha")

    h = 0.0015
    max_cycles = stage.opt_max_cycles or 50
    natoms = len(coords)
    ndim = natoms * 3
    H_inv = np.eye(ndim)
    converged = False
    prev_coords_flat = None
    prev_grad_flat = None
    _cycle_cpu_t0 = time.process_time()
    _cycle_wall_t0 = time.time()
    cycle = 0
    while cycle < max_cycles:
        cycle += 1
        if grad_curr is not None:
            print(f"  [Engine Notice] Cycle {cycle}: Using exact analytical nuclear gradients.")
            grad = grad_curr.copy()
        else:
            print(f"  [Engine Notice] Cycle {cycle}: Analytical forces unavailable. Evaluating symmetric numerical gradients.")
            grad = np.zeros_like(coords)
            for i in range(len(coords)):
                for j in range(3):
                    c_p = coords.copy()
                    c_m = coords.copy()
                    c_p[i, j] += h
                    c_m[i, j] -= h
                    ep, _ = compute_energy_and_forces(c_p)
                    em, _ = compute_energy_and_forces(c_m)
                    grad[i, j] = (ep - em) / (2.0 * h)

        grad = constrain(grad, coords)

        grad_flat = grad.reshape(-1)
        coords_flat = coords.reshape(-1)

        if prev_coords_flat is not None:
            s = coords_flat - prev_coords_flat
            y = grad_flat - prev_grad_flat
            sy = np.dot(s, y)
            if sy > 1e-10:
                rho = 1.0 / sy
                I = np.eye(ndim)
                A = I - rho * np.outer(s, y)
                H_inv = A @ H_inv @ A.T + rho * np.outer(s, s)
            else:
                print(f"  [BFGS] Skipping Hessian update this cycle (curvature s.y = {sy:.3e} <= 0).")

        direction = -(H_inv @ grad_flat)
        if stage.pinnings or stage.pinnings_xyz or stage.equivalences_xyz:
            d = zero_pinned_atoms(direction.reshape(natoms, 3), stage)
            _, d = apply_xyz_pinnings_and_equivalences(None, d, stage)
            direction = d.flatten()
        if active_constraints:
            direction = project_vector_null_space(coords, direction, active_constraints, mask).flatten()

        alpha = 1.0
        backtrack = 0
        while True:
            step_flat = alpha * direction
            trial_coords = (coords_flat + step_flat).reshape(natoms, 3)
            if active_constraints:
                trial_coords = restore_constraints(trial_coords, active_constraints, mask=mask)
            if stage.pinnings_xyz or stage.equivalences_xyz:
                trial_coords, _ = apply_xyz_pinnings_and_equivalences(trial_coords, None, stage)
            e_next, grad_next = compute_energy_and_forces(trial_coords)
            if e_next <= e_curr or backtrack >= 8:
                break
            alpha *= 0.5
            backtrack += 1
            print(f"  [Line Search] Step raised energy (Delta-E = {e_next - e_curr:.3e} Ha); "
                  f"shrinking alpha to {alpha:.5f} (backtrack {backtrack}/8)")
        step = trial_coords - coords if (stage.pinnings_xyz or stage.equivalences_xyz) else step_flat.reshape(natoms, 3)

        delta_e = e_next - e_curr
        rms_grad = np.sqrt(np.mean(grad**2))
        max_grad = np.max(np.abs(grad))
        rms_disp = np.sqrt(np.mean(step**2))
        max_disp = np.max(np.abs(step))

        prev_coords_flat = coords_flat
        prev_grad_flat = grad_flat
        coords = trial_coords
        e_curr = e_next
        grad_curr = grad_next

        checkpoint_atoms = [(s, tuple(c)) for s, c in zip(symbols, coords)]
        with open(f"{stage.job_name}_checkpoint.xyz", "w") as f:
            f.write(f"{len(checkpoint_atoms)}\n")
            f.write(f"cycle {cycle}  E = {e_curr:.10f} Ha  RMS-Grad = {rms_grad:.3e}  Max-Grad = {max_grad:.3e}\n")
            for sym, c in checkpoint_atoms:
                f.write(f"{sym:<3} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f}\n")

        print(f"  Cycle  {cycle:2d} : Energy = {e_curr:.12f} Ha (Delta-E = {delta_e:.3e} Ha, "
              f"RMS-Grad = {rms_grad:.3e}, Max-Grad = {max_grad:.3e})")
        _now_cpu = time.process_time()
        _now_wall = time.time()
        print(f"  CPU Time (this cycle)        = {_now_cpu - _cycle_cpu_t0:.1f} s   (wall: {_now_wall - _cycle_wall_t0:.1f} s)")
        _cycle_cpu_t0 = _now_cpu
        _cycle_wall_t0 = _now_wall

        if (abs(delta_e) < 1e-6 and rms_grad < 3.0e-4 and max_grad < 4.5e-4
                and rms_disp < 1.2e-3 and max_disp < 1.8e-3):
            converged = True
            print(f"  [Wrapper Optimizer] Converged in {cycle} cycle(s).")
            break

    if not converged:
        print(f"  [Wrapper Optimizer] WARNING: did not meet all convergence criteria "
              f"within {max_cycles} cycles; using the last geometry reached.")
    print("[Wrapper Optimizer] Optimization step completed successfully.\n")
    opt_atoms = [(s, tuple(c)) for s, c in zip(symbols, coords)]

    e_conv = None
    if stage.use_df:
        if _conv_check_enabled(stage):
            write_stage_artifacts(stage, opt_atoms, e_df=e_curr, e_conv=None)
            print("  [Saved] final geometry and DF energy written before the CONV check (memory-safe).")
            print("\n[Auto-Cleanup] Executing final conventional energy check at relaxed geometry...")
            try:
                e_conv = run_single_point_calculation(stage, opt_atoms, use_df=False)
            except Exception as err:
                print(f"  [Notice] CONV cross-check failed ({err}); continuing with DF result only.")
                e_conv = None
        print("=" * 60)
        print(f"  [Energy Report] Density Fitting (DF/RI) Energy : {e_curr:.12f} Ha")
        if e_conv is not None:
            print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_conv:.12f} Ha")
        comps_df = report_components(stage, opt_atoms, True)
        comps_conv = report_components(stage, opt_atoms, False) if e_conv is not None else None
        print("=" * 60)
        write_stage_artifacts(stage, opt_atoms, e_df=e_curr, e_conv=e_conv, comps_df=comps_df, comps_conv=comps_conv)
    else:
        print("=" * 60)
        print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_curr:.12f} Ha")
        comps_conv = report_components(stage, opt_atoms, False)
        print("=" * 60)
        write_stage_artifacts(stage, opt_atoms, e_df=None, e_conv=e_curr, comps_conv=comps_conv)
    return opt_atoms


def _conv_check_enabled(stage):
    if stage.method == "MP2" and stage.system_type != "ELECTRONIC" and not stage.cneo:
        return False                 # no conventional positron / NEO MP2 in pyscf_neo_cc (validation warns)
    return RUN_CONV_CROSSCHECK if stage.conv_check is None else stage.conv_check


def _geometric_opt(stage, target_obj):
    """neo_optimize() with constraints / pins, checkpoint trajectory, cap and step limit."""
    _save_conditioning(stage)
    constraints_file = write_geometric_constraints(stage)
    opt_kwargs = {"traj_file": f"{stage.job_name}_checkpoint.xyz"}
    if constraints_file:
        opt_kwargs["constraints"] = constraints_file
    if stage.method == "CCSD(T)":
        opt_kwargs["compute_eee"] = stage.compute_eee
        opt_kwargs["compute_t2e1p"] = stage.compute_t2e1p
    if stage.df_mem_cap_gb:
        opt_kwargs["df_mem_cap"] = int(stage.df_mem_cap_gb * 1e9)
    if stage.opt_max_cycles:
        opt_kwargs["max_steps"] = stage.opt_max_cycles
    t_cpu, t_wall = time.process_time(), time.time()
    try:
        opt_mol = neo_optimize(target_obj, **opt_kwargs)
    finally:
        if constraints_file and os.path.exists(constraints_file):
            os.remove(constraints_file)
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    print(f"  [geomeTRIC Optimizer Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), "
          f"Wall {wall:.1f} s ({wall/3600:.3f} h)")
    return opt_mol


def _pyscf_geometric_opt(stage, mf):
    """geomeTRIC through PySCF's driver (method.nuc_grad_method().as_scanner()) for a mean field with an
    analytic gradient (NEO-HF, NEO-KS, CNEO-HF, CNEO-DFT); constraints, pins, trajectory, step cap as in
    _geometric_opt."""
    from pyscf.geomopt.geometric_solver import optimize as pyscf_geometric_optimize
    constraints_file = write_geometric_constraints(stage)
    kwargs = {"maxsteps": stage.opt_max_cycles or 100}
    if constraints_file:
        kwargs["constraints"] = constraints_file
    t_cpu, t_wall = time.process_time(), time.time()
    try:
        opt_mol = pyscf_geometric_optimize(mf, **kwargs)
    finally:
        if constraints_file and os.path.exists(constraints_file):
            os.remove(constraints_file)
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    print(f"  [geomeTRIC Optimizer Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), "
          f"Wall {wall:.1f} s ({wall/3600:.3f} h)")
    return opt_mol


class _WrapperGradScanner(lib.GradScanner):
    """A PySCF gradient scanner around the wrapper's own energy and gradient (energy_and_forces), so that
    PySCF's geomeTRIC driver can optimize any ELECTRONIC stage (v04.05): the exact analytic gradient where
    PySCF has one, else the central-difference gradient WOPT uses. Every cycle writes <job>_checkpoint.xyz
    (the latest geometry, with the input's atom labels), which --resume restarts from."""
    e_tot = None            # plain attributes, shadowing GradScanner's properties that read self.base
    converged = True

    def __init__(self, stage, mol):
        self.stage = stage
        self.mol = mol
        self.base = self
        self.verbose = lib.logger.NOTE
        self.stdout = sys.stdout
        self.atmlst = None
        self.cycle = 0

    def __deepcopy__(self, memo):
        """Shared, not copied: geomeTRIC's NEB deep-copies its chain (engine included), and the scanner holds the
        log's output stream, which cannot be copied. It keeps no per-image state, so sharing it is safe."""
        return self

    def __call__(self, mol):
        stage = self.stage
        coords = mol.atom_coords(unit="Angstrom")
        atoms = [(lab, tuple(float(x) for x in c)) for (lab, _), c in zip(stage.atoms, coords)]
        energy, g = energy_and_forces(stage, atoms_to_string(atoms))
        g_bohr = numerical_gradient(stage, atoms) if g is None else np.asarray(g) / ANGSTROM_TO_BOHR
        self.cycle += 1
        self.mol, self.e_tot = mol, energy
        with open(f"{stage.job_name}_checkpoint.xyz", "w") as f:
            f.write(f"{len(atoms)}\n")
            f.write(f"geomeTRIC cycle {self.cycle}  E = {energy:.10f} Ha  |grad| = {np.linalg.norm(g_bohr):.3e}\n")
            for sym, c in atoms:
                f.write(f"{sym:<3} {c[0]:15.8f} {c[1]:15.8f} {c[2]:15.8f}\n")
        return energy, g_bohr


def run_electronic_geometric(stage):
    """ELECTRONIC OPT with geomeTRIC through PySCF's driver (v04.05; up to v04.04 every electronic OPT ran
    WOPT, a rule inherited from v03.17 when geomeTRIC had only been checked with pyscf_neo_cc objects).
    Constraints, pins, step cap and final energies as in the other sectors; WOPT if geomeTRIC fails."""
    from pyscf.geomopt.geometric_solver import kernel as geometric_kernel
    analytic, why = analytic_gradient_available(stage)
    kind = "analytic gradients" if analytic else f"numerical gradients ({why})"
    print(f"\n[Geometry Optimization] geomeTRIC on the electronic {stage.method} surface, {kind}...")
    atom_str, ebasis = electron_mol_input(stage, atoms_to_string(stage.atoms))
    mol = gto.M(atom=atom_str, basis=ebasis, charge=stage.charge,
                spin=stage.multiplicity - 1, verbose=0)
    scanner = _WrapperGradScanner(stage, mol)
    constraints_file = write_geometric_constraints(stage)
    kwargs = {"maxsteps": stage.opt_max_cycles or 100}
    if constraints_file:
        kwargs["constraints"] = constraints_file
    t_cpu, t_wall = time.process_time(), time.time()
    try:
        converged, opt_mol = geometric_kernel(scanner, **kwargs)
    except Exception as err:
        print(f"[Notice] geomeTRIC on the electronic surface failed ({type(err).__name__}: {err}). "
              f"Switching to WOPT...")
        return run_wrapper_gradient_optimizer(stage)
    finally:
        if constraints_file and os.path.exists(constraints_file):
            os.remove(constraints_file)
    cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
    print(f"  [geomeTRIC Optimizer Timing] CPU {cpu:.1f} s ({cpu/3600:.3f} h), "
          f"Wall {wall:.1f} s ({wall/3600:.3f} h), {scanner.cycle} gradient evaluations")
    if not converged:
        print(f"  [WARNING] geomeTRIC did not converge in {kwargs['maxsteps']} steps; the last geometry is "
              f"reported (raise %opt_max_cycles, or restart from {stage.job_name}_checkpoint.xyz).")
    print("Optimization Complete.")
    opt_atoms = extract_optimized_atoms(opt_mol, stage.atoms)
    del scanner, opt_mol, mol
    gc.collect()
    _report_final_energies(stage, opt_atoms)
    return opt_atoms


def _report_final_energies(stage, opt_atoms):
    """Fresh single points at the optimized geometry (the CC object handed
    to neo_optimize keeps its seed-geometry energy)."""
    if stage.use_df:
        e_df_final = run_single_point_calculation(stage, opt_atoms, use_df=True)
        print(f"  [Energy Report] Density Fitting (DF/RI) Energy (from final opt step): {e_df_final:.12f} Ha")
        comps_df = report_components(stage, opt_atoms, True)
        e_conv = comps_conv = None
        write_stage_artifacts(stage, opt_atoms, e_df=e_df_final, e_conv=None, comps_df=comps_df)
        if _conv_check_enabled(stage):
            print("  [Saved] final geometry and DF energy written before the CONV check (memory-safe).")
            print("  [Energy Evaluation] Running final single-point calculation using conventional integrals (CONV) at relaxed geometry...")
            try:
                e_conv = run_single_point_calculation(stage, opt_atoms, use_df=False)
                print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_conv:.12f} Ha")
                comps_conv = report_components(stage, opt_atoms, False)
            except Exception as err:
                print(f"  [Notice] CONV cross-check failed ({err}); continuing with DF result only.")
                e_conv = None
        write_stage_artifacts(stage, opt_atoms, e_df=e_df_final, e_conv=e_conv, comps_df=comps_df,
                              comps_conv=comps_conv)
    else:
        try:
            e_conv = run_single_point_calculation(stage, opt_atoms, use_df=False)
            print(f"  [Energy Report] Conventional (CONV/Exact) Energy: {e_conv:.12f} Ha")
        except Exception as err:
            print(f"  [Notice] CONV single-point failed ({err}); no verified final energy for this stage.")
            e_conv = None
        comps_conv = report_components(stage, opt_atoms, False) if e_conv is not None else None
        write_stage_artifacts(stage, opt_atoms, e_df=None, e_conv=e_conv, comps_conv=comps_conv)


def execute_stage(stage):
    _prepare_conditioning(stage)
    print("=" * 80)
    print(f"[STAGE {stage.stage_id}] EXECUTING: {stage.job_name}")
    print("=" * 80)

    lib.num_threads(stage.cpu_cores)
    print_internal_profile(stage.atoms)

    if stage.system_type == "POSITRON" and stage.method in ("CCSD", "CCSD(T)"):
        available, why = analytic_gradient_available(stage)
        if not available and stage.run_type == "GRADIENT":
            print(f"Numerical {stage.method} gradient ({why}); 6 energies per atom.")
            gradient = numerical_gradient(stage, stage.atoms)
            print("Nuclear Gradients (Hartree/Bohr):")
            print(gradient)
            return gradient
        if not available and stage.run_type == "OPT" and stage.opt_engine != "WRAPPER":
            raise PreflightError(f"Native positron OPT requires an analytic gradient: {why}. "
                                 "Use WOPT for numerical gradients.")

    if stage.run_type == "SCAN":
        return run_scan(stage)
    if stage.run_type == "FREQ":
        return run_freq(stage)
    if stage.run_type == "TS":
        return run_ts(stage)
    if stage.run_type == "IRC":
        return run_irc(stage)
    if stage.run_type == "NEB":
        return run_neb(stage)
    if stage.run_type == "ZOPT":
        return run_zopt(stage)
    if stage.run_type == "OPT":
        apply_start_geometry_constraints(stage)
        if stage.opt_engine == "WRAPPER":
            return run_wrapper_gradient_optimizer(stage)
        if stage.system_type == "ELECTRONIC":
            return run_electronic_geometric(stage)

    atom_str = atoms_to_string(stage.atoms)

    if stage.method == "MP2":
        return _execute_mp2(stage, atom_str)

    if stage.system_type == "POSITRON":
        if not HAS_POSITRON:
            raise RuntimeError("Module pyscf_neo_cc.positron is required for POSITRON sector.")

        system = safe_build_system(stage, atom_str)
        mf = make_poshf(system, stage)
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        if stage.verbose_explicit:
            mf.verbose = stage.verbose
        if (stage.method == "HF" and stage.run_type in ("GRADIENT", "OPT")
                and _native_poshf_preflight is not None):
            poshf_gradient_preflight(mf)
        optimizer_reference_accuracy(stage, mf)
        run_poshf(mf)
        _save_conditioning(stage)
        print(f"Converged POSHF Energy: {mf.e_tot:.12f} Ha")

        if stage.method not in ["CCSD", "CCSD(T)"]:
            if stage.run_type == "GRADIENT":
                print("\n[Analytical Gradient] Computing POSHF nuclear gradients...")
                try:
                    g = mf.Gradients().kernel()
                    print("POSHF Nuclear Gradients (Hartree/Bohr):")
                except Exception as e:
                    print(f"Analytical gradient unavailable ({e}); computing a numerical gradient "
                          f"(6 POSHF energies per atom).")
                    g = numerical_gradient(stage, stage.atoms)
                    print("POSHF Numerical Nuclear Gradients (Hartree/Bohr):")
                print(g)
                return g
            if stage.run_type == "OPT":
                if not analytic_gradient_available(stage)[0]:
                    return run_wrapper_gradient_optimizer(stage)
                print("\n[Geometry Optimization] Invoking geomeTRIC with analytic POSHF gradients...")
                opt_mol = _geometric_opt(stage, mf)
                opt_atoms = extract_optimized_atoms(opt_mol, stage.atoms)
                if getattr(opt_mol, "converged", False):
                    print("POSHF optimization converged.")
                else:
                    print("  [Notice] POSHF optimization stopped before convergence; reporting the last accepted geometry.")
                del mf, system, opt_mol
                gc.collect()
                _report_final_energies(stage, opt_atoms)
                return opt_atoms
            for line in partition_lines(safe_positron_partition(stage, mf)):
                print(line)
            return float(mf.e_tot)

        cc_obj = build_positron_cc(mf, stage)
        if stage.verbose_explicit:
            cc_obj.verbose = stage.verbose
        if stage.run_type == "OPT":
            print("\n[Geometry Optimization] Invoking geomeTRIC multi-species scanner...")
            opt_mol = _geometric_opt(stage, cc_obj)
            print("Optimization Complete.")
            opt_atoms = extract_optimized_atoms(opt_mol, stage.atoms)
            # Free the seed-geometry SCF/CC objects before the final single
            # point builds new ones (two OOM kills on 2026-09-21 came from
            # both generations being alive at once).
            del cc_obj, mf, system, opt_mol
            gc.collect()
            _report_final_energies(stage, opt_atoms)
            return opt_atoms
        cc_obj.kernel()
        print(f"Converged POS-CCSD Total Energy: {cc_obj.e_tot:.12f} Ha (E_corr = {cc_obj.e_corr:.12f} Ha)")

        e_tot = float(cc_obj.e_tot)
        e_t = None
        if stage.method == "CCSD(T)":
            e_t = _triples(cc_obj, stage)
            e_tot += e_t
            print(f"CCSD(T) Triples Correction: {e_t:.12f} Ha")
            print(f"Final POS-CCSD(T) Total Energy: {e_tot:.12f} Ha")
        if stage.run_type == "SP_ENERGY":
            for line in partition_lines(safe_positron_partition(stage, mf, cc_obj, e_t)):
                print(line)

        if stage.run_type == "GRADIENT":
            print(f"\n[Analytical Gradient] Computing POS-{stage.method} gradients...")
            g = positron_cc_gradient(stage, cc_obj)
            print(f"POS-{stage.method} Nuclear Gradients (Hartree/Bohr):")
            print(g)
            return g

        return e_tot

    if stage.system_type == "PROTON":
        if NEOMole is None:
            raise RuntimeError("Module pyscf_neo_cc._neo is required for PROTON sector.")
        tag = f"CNEO ({stage.cneo_variant})" if stage.cneo else "NEO"
        mf = proton_mean_field(stage, atom_str, verbose=stage.verbose if stage.verbose_explicit else 0)
        print(f"Converged {'CNEO' if stage.cneo else 'NEO'}-MeanField Energy: {mf.e_tot:.12f} Ha")
        cc_obj = None
        e_tot = float(mf.e_tot)
        if stage.method in ["CCSD", "CCSD(T)"]:
            cc_obj, e_cc, e_t = proton_correlated(stage, mf)
            print(f"Converged {tag}-CCSD Total Energy: {e_cc:.12f} Ha")
            e_tot = e_cc
            if e_t is not None:
                e_tot += e_t
                print(f"{tag}-CCSD(T) Triples: {e_t:.12f} Ha")
                print(f"Final {tag}-CCSD(T) Total Energy: {e_tot:.12f} Ha")

        if stage.run_type == "GRADIENT":
            try:
                g = proton_gradient(stage, mf, cc_obj)
                print(f"{tag} Nuclear Gradients, analytic (Hartree/Bohr):")
            except Exception as err:
                print(f"Analytic gradient failed ({type(err).__name__}: {err}): central finite differences, "
                      f"6 energies per atom.")
                del cc_obj, mf
                gc.collect()
                g = numerical_gradient(stage, stage.atoms)
                print(f"{tag} Nuclear Gradients, numerical (Hartree/Bohr):")
            print(g)
            return g

        if stage.run_type == "OPT":
            try:
                opt_mol = _geometric_opt(stage, cc_obj) if cc_obj is not None else _pyscf_geometric_opt(stage, mf)
            except Exception as err:
                print(f"[Notice] geomeTRIC on the {tag} surface failed ({type(err).__name__}: {err}). "
                      f"Switching to WOPT...")
                return run_wrapper_gradient_optimizer(stage)
            print(f"{tag} Optimization Complete.")
            opt_atoms = extract_optimized_atoms(opt_mol, stage.atoms)
            del cc_obj, mf, opt_mol
            gc.collect()
            _report_final_energies(stage, opt_atoms)
            return opt_atoms
        return e_tot

    if stage.system_type == "ELECTRONIC":
        mf, open_shell = _electronic_mf(stage, atom_str, stage.use_df)
        mf.max_memory = stage.memory_mb
        mf.conv_tol = 1e-11
        mf.kernel()
        print(f"Converged {'UHF' if open_shell else 'RHF'} Energy: {mf.e_tot:.12f} Ha")

        if stage.method not in ["CCSD", "CCSD(T)"]:
            if stage.run_type == "GRADIENT":
                try:
                    g = mf.Gradients().kernel()
                    print("HF/DFT Nuclear Gradients (Hartree/Bohr):")
                except Exception as e:
                    print(f"Analytical gradient unavailable ({e}); computing a numerical gradient "
                          f"(6 energies per atom).")
                    g = numerical_gradient(stage, stage.atoms)
                    print("HF/DFT Numerical Nuclear Gradients (Hartree/Bohr):")
                print(g)
                return g
            return float(mf.e_tot)

        cc_obj = _electronic_cc(mf, stage)
        cc_obj.kernel()
        e_mp2 = cc_mp2_energy(cc_obj, mf)
        if e_mp2 is not None:
            print(f"Mean field (HF) Total Energy: {mf.e_tot:.12f} Ha")
            print(f"MP2 Total Energy (CC first guess): {e_mp2:.12f} Ha (E_corr = {e_mp2 - mf.e_tot:.12f} Ha)")
        print(f"Converged Electronic CCSD Total Energy: {cc_obj.e_tot:.12f} Ha (E_corr = {cc_obj.e_corr:.12f} Ha)")

        e_tot = float(cc_obj.e_tot)
        if stage.method == "CCSD(T)":
            e_t = _triples(cc_obj, stage)
            e_tot += e_t
            print(f"CCSD(T) Triples Correction: {e_t:.12f} Ha")
            print(f"Final Electronic CCSD(T) Total Energy: {e_tot:.12f} Ha")

        if stage.run_type == "GRADIENT":
            ok, why = analytic_gradient_available(stage)
            g = None
            if ok:
                try:
                    g = electronic_cc_gradient(stage, cc_obj, open_shell)
                except Exception as e:
                    why = f"{type(e).__name__}: {e}"
            if g is not None:
                print(f"Electronic {stage.method} Nuclear Gradients (Hartree/Bohr):")
            else:
                print(f"Analytical gradient unavailable ({why}); computing a numerical gradient "
                      f"(6 energies per atom).")
                g = numerical_gradient(stage, stage.atoms)
                print(f"Electronic {stage.method} Numerical Nuclear Gradients (Hartree/Bohr):")
            print(g)
            return g
        return e_tot
    raise ValueError(f"Unknown sector {stage.system_type}")


def _execute_mp2(stage, atom_str):
    """SP / GRADIENT / geomeTRIC OPT at MP2 (WOPT, ZOPT and scans go through build_mp2 directly). A
    GRADIENT job without an analytic gradient gets a numerical one; it never turns into an optimization."""
    solve_seed = not (stage.system_type == "POSITRON" and stage.run_type == "OPT")
    mf, mp = build_mp2(stage, atom_str, solve=solve_seed)
    ref = {"POSITRON": "POSHF", "PROTON": "NEO-HF (DF)", "ELECTRONIC": "SCF"}[stage.system_type]
    print(f"Converged {ref} Energy: {mf.e_tot:.12f} Ha")
    if solve_seed:
        print(f"Final {stage.system_type.capitalize()} MP2 Total Energy: {mp.e_tot:.12f} Ha "
              f"(E_corr = {float(mp.e_tot - mf.e_tot):.12f} Ha)")
        if stage.run_type == "SP_ENERGY":
            for line in partition_lines(safe_positron_partition(stage, mf, mp=mp)):
                print(line)
    e_tot = float(mp.e_tot) if solve_seed else None
    if stage.run_type == "GRADIENT":
        g = mp2_gradient(stage, mp)
        if g is None:
            print("No analytic MP2 gradient on this route (PySCF has no DF-MP2 gradient): "
                  "central finite differences, 6 energies per atom.")
            del mp, mf
            gc.collect()
            g = numerical_gradient(stage, stage.atoms)
            print("MP2 Nuclear Gradients, numerical (Hartree/Bohr):")
        else:
            print("MP2 Nuclear Gradients, analytic (Hartree/Bohr):")
        print(g)
        return g
    if stage.run_type == "OPT":
        try:
            opt_mol = _geometric_opt(stage, mp)
        except Exception as err:
            print(f"[Notice] geomeTRIC on the MP2 surface failed ({err}). Switching to WOPT...")
            return run_wrapper_gradient_optimizer(stage)
        print("MP2 Optimization Complete.")
        opt_atoms = extract_optimized_atoms(opt_mol, stage.atoms)
        del mp, mf, opt_mol
        gc.collect()
        _report_final_energies(stage, opt_atoms)
        return opt_atoms
    return e_tot


def _apply_df_mem_cap(stage):
    if pyscf_neo_cc is None or not hasattr(pyscf_neo_cc, "set_df_mem_cap"):
        return
    if stage.df_mem_cap_gb:
        pyscf_neo_cc.set_df_mem_cap(int(stage.df_mem_cap_gb * 1e9))
    elif _DEFAULT_DF_MEM_CAP is not None:
        pyscf_neo_cc.set_df_mem_cap(_DEFAULT_DF_MEM_CAP)


RESUME = False
STAGE_RECORDS = False       # write <job>_stage<N>_{started,done}.json: in a SLURM job, with --resume or --stage-records


def _stage_prefix(stage):
    return stage.job_name if stage.job_name else f"Stage_{stage.stage_id}"


_SIGNATURE_SKIP = {"warnings", "errors", "_charge_sources", "_mult_sources", "_unit_sources", "_keys_seen",
                   "inherited_from", "atoms", "has_explicit_xyz", "geometry_kind", "coord_units"}


def _canon(v):
    """A deterministic text form of a setting (objects by their attributes, never by memory address)."""
    if isinstance(v, dict):
        return "{" + ",".join(f"{k!r}:{_canon(v[k])}" for k in sorted(v, key=repr)) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_canon(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(_canon(x) for x in v)) + "}"
    if hasattr(v, "__dict__") and not isinstance(v, type):
        return type(v).__name__ + _canon(vars(v))
    return repr(v)


def stage_signature(stage):
    """Identify parsed settings, exact start geometry, source and input for disk resumption."""
    settings = {name: value for name, value in vars(stage).items()
                if not name.startswith("_") and name not in ("warnings", "errors")
                and name not in _SIGNATURE_SKIP}
    return _cache_digest("stage-v2", _restart_source_identity(), settings,
                         _surface_definition(stage, atoms_to_string(stage.atoms)))


def _stage_record_path(stage, kind):
    return f"{_stage_prefix(stage)}_stage{stage.stage_id}_{kind}.json"


def encode_stage_result(res):
    if isinstance(res, list) and res and all(isinstance(a, (tuple, list)) and len(a) == 2 for a in res):
        return {"type": "atoms", "atoms": [[str(sym), [float(x) for x in c]] for sym, c in res]}
    if isinstance(res, (float, int, np.floating)) and not isinstance(res, bool):
        return {"type": "energy", "value": float(res)}
    if isinstance(res, np.ndarray):
        return {"type": "array", "value": res.tolist()}
    return {"type": "none"}


def decode_stage_result(rec):
    rec = rec or {}
    if rec.get("type") == "atoms":
        return [(sym, tuple(c)) for sym, c in rec["atoms"]]
    if rec.get("type") == "energy":
        return rec["value"]
    if rec.get("type") == "array":
        return np.array(rec["value"])
    return None


def write_stage_record(stage, kind, signature, result=None):
    """<job>_stage<N>_started.json when a stage begins, <job>_stage<N>_done.json when it has finished (with
    its result, for chaining on --resume). Written atomically. Only in a SLURM job, with --resume or with
    --stage-records (STAGE_RECORDS): a local run leaves no JSON files."""
    import json
    _save_conditioning(stage)
    rec = {"stage": stage.stage_id, "job_name": stage.job_name, "run_type": stage.run_type,
           "version": VERSION, "signature": signature, time_key(kind): time.strftime("%Y-%m-%d %H:%M:%S")}
    if kind == "done":
        rec["result"] = encode_stage_result(result)
        if getattr(stage, "aux_conditioning", None) is not None:
            rec["aux_conditioning_digest"] = _cache_digest(stage.aux_conditioning.report())
    path = _stage_record_path(stage, kind)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, path)


def time_key(kind):
    return "finished" if kind == "done" else "started"


def read_stage_record(stage, kind, signature):
    """The record if it exists and belongs to this very stage (same signature), else None."""
    import json
    try:
        with open(_stage_record_path(stage, kind)) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return None
    if rec.get("signature") != signature:
        return None
    if kind == "done" and getattr(stage, "aux_conditioning", None) is not None:
        if rec.get("aux_conditioning_digest") != _cache_digest(stage.aux_conditioning.report()):
            raise PreflightError("Finished stage and saved auxiliary selections disagree; "
                                 "refusing cached results. Use a fresh fitting workflow.")
    return rec


def _last_xyz_frame(path):
    """Coordinates (A) of the last frame of an xyz file (one frame, or a trajectory), or None."""
    try:
        lines = open(path).read().splitlines()
    except OSError:
        return None
    frames, i = [], 0
    while i < len(lines):
        try:
            n = int(lines[i].split()[0])
        except (ValueError, IndexError):
            i += 1
            continue
        block = lines[i + 2:i + 2 + n]
        try:
            frames.append([tuple(float(x) for x in l.split()[1:4]) for l in block])
        except (ValueError, IndexError):
            break
        i += 2 + n
    return frames[-1] if frames else None


def restart_from_checkpoint(stage):
    """--resume: an optimization that started before (same input and start geometry) but did not finish
    restarts from the last geometry in <job>_checkpoint.xyz, keeping the input's atom labels."""
    path = f"{stage.job_name}_checkpoint.xyz"
    coords = _last_xyz_frame(path)
    if not coords or len(coords) != len(stage.atoms):
        return
    stage.atoms = [(sym, c) for (sym, _), c in zip(stage.atoms, coords)]
    print(f"[Resume] Stage {stage.stage_id} ('{stage.job_name}') was interrupted: the optimization restarts "
          f"from the last geometry in {path}.")


# Cube boxes (bohr beyond the outermost atoms), measured 2026-09-28 at 80^3 points with the HF densities of e+ LiH,
# e+ F- and e+ Cl2(2-): electrons 5 bohr (within 1% of the electron count; a larger box spreads the grid and
# misses the density near the nuclei: 20 bohr loses 18% for F-), positrons 15 bohr (100%; PySCF's default 3 bohr
# held only 19-52% of the positron).
CUBE_MARGIN_BOHR = {"electron": 5.0, "positron": 15.0}


def _natural_orbitals(mol, dm_ao):
    """Natural orbitals (AO coefficients, occupations, largest first) of a one-particle density matrix. The
    unrelaxed CC density is not symmetric; its symmetric part is the physical density."""
    from scipy.linalg import eigh
    dm = 0.5 * (dm_ao + dm_ao.T)
    s = mol.intor("int1e_ovlp")
    occ, coeff = eigh(s @ dm @ s, s)
    return coeff[:, ::-1], occ[::-1]


# ---------------------------------------------------------------------------------------------------------------
# v04.05: CHELPG charges and plane density maps, in-house (numpy, PySCF and the standard library; no Multiwfn,
# no plotting library). Both use the HF / CC densities that export_wavefunction builds at the final geometry.
# ---------------------------------------------------------------------------------------------------------------
CHELPG_SPACING_A = 0.3          # Breneman & Wiberg (1990), as Multiwfn
CHELPG_EXTENSION_A = 2.8
# Multiwfn's CHELPG radii (Angstrom) for H-Ar; heavier elements: UFF radii / 1.2, Multiwfn's default for them
# (checked against Multiwfn 2026.9.20: identical fitting points for K-Xe).
_CHELPG_RADII_A = {1: 1.45, 2: 1.45, 3: 1.50, 4: 1.50, 5: 1.50, 6: 1.50, 7: 1.70, 8: 1.70, 9: 1.70, 10: 1.70,
                   **{z: 2.00 for z in range(11, 19)}}


def chelpg_radius_bohr(z):
    from pyscf.data import radii
    z = int(z)
    return _CHELPG_RADII_A[z] / BOHR_TO_ANGSTROM if z in _CHELPG_RADII_A else float(radii.UFF[z]) / 1.2


def chelpg_points(mol):
    """Fitting points (bohr): a cubic grid (0.3 A) over the atoms' box extended by 2.8 A, keeping the points
    outside every atom's radius and within 2.8 A of at least one atom."""
    xyz = mol.atom_coords()
    znuc = [gto.charge(mol.atom_pure_symbol(i)) for i in range(mol.natm)]
    ext, step = CHELPG_EXTENSION_A / BOHR_TO_ANGSTROM, CHELPG_SPACING_A / BOHR_TO_ANGSTROM
    lo, hi = xyz.min(0) - ext, xyz.max(0) + ext
    axes = [lo[k] + step * np.arange(int(np.floor((hi[k] - lo[k]) / step + 1e-9)) + 1) for k in range(3)]
    rad = np.array([chelpg_radius_bohr(z) for z in znuc])
    keep = []
    for x in axes[0]:                                   # one slab at a time: small memory for large molecules
        g = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        d = np.linalg.norm(g[:, None, :] - xyz[None], axis=2)
        keep.append(g[np.all(d > rad, axis=1) & np.any(d <= ext, axis=1)])
    return np.concatenate(keep), {mol.atom_pure_symbol(i): rad[i] * BOHR_TO_ANGSTROM for i in range(mol.natm)}


def density_potential(mol, dm, pts, memory_mb=4000):
    """Potential of the charge distribution with density matrix dm, per unit charge, at pts: int rho/|r - r'|."""
    dm = 0.5 * (dm + dm.T)
    nao = mol.nao
    chunk = max(1, min(len(pts), int(min(500.0, 0.1 * memory_mb) * 1e6 / (8.0 * nao * nao))))
    v = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        ints = mol.intor("int1e_grids", grids=pts[i:i + chunk])
        v[i:i + chunk] = np.einsum("gij,ij->g", ints, dm)
    return v


def chelpg_fit(xyz, pts, v, total):
    """Least-squares atomic charges reproducing v at pts, constrained to sum to `total` (Lagrange multiplier)."""
    a = 1.0 / np.linalg.norm(pts[:, None, :] - xyz[None], axis=2)
    n = a.shape[1]
    m = np.zeros((n + 1, n + 1))
    m[:n, :n], m[:n, n], m[n, :n] = a.T @ a, 1.0, 1.0
    q = np.linalg.solve(m, np.concatenate([a.T @ v, [total]]))[:n]
    return q, float(np.sqrt(np.mean((a @ q - v) ** 2)))


def _print_chelpg(tag, stage, level, mol, q, rmse, head):
    lab = {"hf": "HF" if stage.system_type == "ELECTRONIC" else "POSHF", "ccsd": "CCSD"}.get(level, level)
    print(f"\n{tag} {lab} charges at the final geometry: {head}")
    print(f"  {'#':>3}  {'Atom':<4}{'q':>14}")
    for i in range(mol.natm):
        print(f"  {i + 1:>3}  {mol.atom_pure_symbol(i):<4}{q[i]:>14.6f}")
    print(f"  {'Sum':<9}{q.sum():>14.6f}")
    print(f"  RMSE (a.u.){rmse:>12.6f}")
    if stage.system_type == "POSITRON":
        print(f"  {tag} Fitted to the potential of the nuclei and the electrons; the positron density is not "
              f"included (charges of a positronic system are not defined with it). The charges sum to the "
              f"charge plus the number of positrons.")


def report_chelpg(stage, level, species):
    """%chelpg: CHELPG charges of one level ('hf' / 'ccsd'), printed in the .out. `species` maps 'electron' (and
    'positron') to (mol, dm). The fit uses the potential of the nuclei and the electrons only: for a positron
    system the positron density is left out (charges including it are not defined), so the charges sum to the
    charge plus the number of positrons."""
    t0 = time.time()
    mol_e, dm_e = species["electron"]
    xyz, zeff = mol_e.atom_coords(), mol_e.atom_charges().astype(float)
    pts, radii_used = chelpg_points(mol_e)
    r = np.linalg.norm(pts[:, None, :] - xyz[None], axis=2)
    v = (zeff[None] / r).sum(1) - density_potential(mol_e, dm_e, pts, stage.memory_mb)
    q, rmse = chelpg_fit(xyz, pts, v, float(zeff.sum() - mol_e.nelectron))
    radii_txt = ", ".join(f"{s} {rv:.2f}" for s, rv in radii_used.items())
    _print_chelpg("[CHELPG]", stage, level, mol_e, q, rmse,
                  f"{len(pts)} fitting points ({CHELPG_SPACING_A} A grid, {CHELPG_EXTENSION_A} A box; "
                  f"radii {radii_txt} A)")
    print(f"  [CHELPG] {time.time() - t0:.1f} s")
    return q.tolist()


# ------------------------------------------------------------------------------------------ plane density maps
_MAP_AXES = {"XY": (0, 1, 2), "XZ": (0, 2, 1), "YZ": (1, 2, 0)}   # (horizontal, vertical, normal)
_MAP_MARGIN_BOHR = {"electron": 4.5, "positron": 10.0}
_MAP_SCALES = {    # Multiwfn's colour transitions 0-19 (plane map option 19), 17 stops each, low to high,
                   # sampled from its colour bars (Multiwfn 2026.9.20)
    "rainbow_bw": "220044 2500c5 000cff 0048ff 0084ff 00c0ff 00ffff 00ffbf 00ff52 33ff00 77ff00 bbff00 fbff00 ffbf00 ff8400 ff4100 ff0000",
    "rainbow": "220044 2500c5 000cff 0048ff 0084ff 00c0ff 00ffff 00ffbf 00ff52 33ff00 77ff00 bbff00 fbff00 ffbf00 ff8400 ff4100 ff0000",
    "reversed_rainbow": "ff0000 ff4100 ff8400 ffc300 fbff00 bbff00 77ff00 2bff00 00ff58 00ffc3 00ffff 00c0ff 0084ff 0044ff 000cff 2500c5 220044",
    "white_rainbow": "fcfcff cfcfff a3a3ff 7676ff 4c4cff 1f1fff 001fff 0083ff 00e6ff 00ffb4 00ff57 0cff00 70ff00 d3ff00 ffcd00 ff6a00 ff0600",
    "spectrum": "ff00ff ae00ff 5e00ff 0d00ff 003eff 008fff 00e0ff 00ffce 00ff7d 00ff2c 1fff00 70ff00 c1ff00 ffed00 ffa100 ff5100 ff0000",
    "reversed_spectrum": "ff0000 ff5100 ffa100 fff200 c1ff00 70ff00 1fff00 00ff31 00ff82 00ffd3 00e0ff 008fff 003eff 1200ff 5e00ff ae00ff ff00ff",
    "grey": "010101 111111 212121 313131 404040 505050 606060 707070 808080 909090 9f9f9f afafaf bfbfbf cfcfcf dedede eeeeee fefefe",
    "reversed_grey": "fefefe eeeeee dedede cecece bfbfbf afafaf 9f9f9f 8f8f8f 7f7f7f 6f6f6f 606060 505050 404040 303030 212121 111111 010101",
    "blue_white_red": "0404ff 2424ff 4444ff 6464ff 8181ff a1a1ff c1c1ff e1e1ff fffdfd ffdddd ffbfbf ff9f9f ff8080 ff6060 ff4242 ff2222 ff0202",
    "red_white_blue": "ff0404 ff2424 ff4444 ff6464 ff8181 ffa1a1 ffc1c1 ffe1e1 fdfdff ddddff bfbfff 9f9fff 8080ff 6060ff 4242ff 2222ff 0202ff",
    "blue_green_red": "0004fb 0024db 0044bb 00649b 00817e 00a15e 00c13e 00e11e 02fd00 22dd00 40bf00 609f00 808000 9f6000 bd4200 dd2200 fd0200",
    "red_green_blue": "fb0400 db2400 bb4400 9b6400 7e8100 5ea100 3ec100 1ee100 00fd02 00dd22 00bf40 009f60 008080 00609f 0042bd 0022dd 0002fd",
    "white_darkred": "fffefe faeeee f5dede f0cece ecbfbf e7afaf e29f9f dd8f8f d97f7f d46f6f cf6060 ca5050 c64040 c13030 bc2121 b81111 b30101",
    "black_orange_yellow": "020000 1a0000 320000 4a0000 600000 780e00 902400 a83a00 c05000 d86600 ef7b06 ff911f ffa737 ffbd50 ffd267 ffe87f fffe97",
    "white_darkgreen": "fefffe eefaee def5de cef0ce bfecbf afe7af 9fe29f 8fdd8f 7fd97f 6fd46f 60cf60 50ca50 40c640 30c130 21bc21 11b811 01b301",
    "black_green": "000100 001100 002100 003100 004000 005000 006000 007000 008000 009000 009f00 00af00 00bf00 00cf00 00de00 00ee00 00fe00",
    "white_darkblue": "fefeff eeeefc dedef8 cecef5 bfbff2 afafef 9f9fec 8f8fe9 7f7fe5 6f6fe2 6060df 5050dc 4040d9 3030d6 2121d3 1111cf 0101cc",
    "black_blue_cyan": "000001 000016 00002a 00003e 000052 000066 00007a 00148f 002ea3 0048b8 0061cb 007bdf 0096f4 00b0ff 00c9ff 00e3ff 00fdff",
    "viridis": "440256 481a6c 472e7c 424186 3b528b 33638d 2c728e 26828e 21918c 1fa088 27ad81 3dbc74 5cc863 81d34d a8db34 d2e21b fbe723",
    "yellow_orange_black": "fffe97 ffe87f ffd267 ffbb4e ffa737 ff911f ef7b06 d76500 bf4f00 a73900 902400 780e00 600000 480000 320000 1a0000 020000",
}


def map_scale(name):
    """(n, 3) colour stops in 0-1 of a _MAP_SCALES entry."""
    return np.array([[int(h[k:k + 2], 16) / 255.0 for k in (0, 2, 4)] for h in _MAP_SCALES[name].split()])


_MAP_ELECTRON_MAX = 0.65                                           # a.u.; larger values are drawn white
_MAP_ELECTRON_CONTOURS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]


def density_map_plane(stage, mol):
    """(plane, offset): %density_map_plane, or AUTO = the Cartesian plane with the largest spread of the atoms
    (XZ for a linear molecule on z or a single atom), through the atoms' mean coordinate along its normal."""
    xyz = mol.atom_coords()
    plane = stage.density_map_plane
    if plane == "AUTO":
        span = xyz.max(0) - xyz.min(0)
        normal = min((1, 0, 2), key=lambda k: span[k] - (1e-6 if k == 1 else 0.0))
        plane = {0: "YZ", 1: "XZ", 2: "XY"}[normal]
    return plane, float(xyz[:, _MAP_AXES[plane][2]].mean())


class CubeGrid:
    """A Gaussian cube file (v04.06): origin, three axis vectors, values (bohr, a.u.); atoms (Z, x, y, z)."""

    def __init__(self, path):
        with open(path) as f:
            lines = f.read().splitlines()
        self.title = (lines[0].strip() + " " + lines[1].strip()).strip()
        head = lines[2].split()
        nat, self.origin = int(head[0]), np.array([float(x) for x in head[1:4]])
        axes = [lines[3 + k].split() for k in range(3)]
        self.shape = tuple(abs(int(a[0])) for a in axes)
        self.axes = np.array([[float(x) for x in a[1:4]] for a in axes])
        if any(int(a[0]) < 0 for a in axes):
            self.axes = self.axes * ANGSTROM_TO_BOHR        # negative counts: Angstrom units (cube convention)
            self.origin = self.origin * ANGSTROM_TO_BOHR
        self.atoms = []
        for k in range(abs(nat)):
            t = lines[6 + k].split()
            self.atoms.append((int(float(t[0])), np.array([float(x) for x in t[2:5]])))
        start = 6 + abs(nat) + (1 if nat < 0 else 0)          # an orbital cube has one more header line
        vals = np.array(" ".join(lines[start:]).split(), float)
        n = int(np.prod(self.shape))
        if vals.size < n:
            raise ValueError(f"{path}: {vals.size} values for a {self.shape} grid")
        self.values = vals[:n].reshape(self.shape)

    def values_at(self, pts):
        """Trilinear interpolation at points (bohr); 0 outside the box."""
        from scipy.ndimage import map_coordinates
        frac = np.linalg.solve(self.axes.T, (np.asarray(pts) - self.origin).T)      # grid index coordinates
        return map_coordinates(self.values, frac, order=1, mode="constant", cval=0.0)

    def integral(self):
        return float(self.values.sum() * abs(np.linalg.det(self.axes)))


def _plot_mol(atoms_z_xyz, basis=None):
    """A PySCF Mole holding the atoms of a loaded file (positions in bohr), for the map geometry and labels."""
    sym = [gto.mole._symbol(int(z)) if z > 0 else "X" for z, _ in atoms_z_xyz]
    atom = [(f"ghost-H" if s == "X" else s, tuple(float(c) for c in x)) for s, (_, x) in zip(sym, atoms_z_xyz)]
    nelec = sum(int(z) for z, _ in atoms_z_xyz)
    return gto.M(atom=atom, unit="Bohr", basis=basis or "sto-3g", spin=nelec % 2, verbose=0)


def load_plot_file(path):
    """(mol, density) of a .molden file (density = sum_i occ_i |phi_i|^2 from its orbitals and occupations; the
    natural-orbital files of %molden give the CCSD density) or of a .cube file (a CubeGrid)."""
    low = path.lower()
    if low.endswith(".cube") or low.endswith(".cub"):
        cube = CubeGrid(path)
        return _plot_mol(cube.atoms), cube
    from pyscf.tools import molden
    with contextlib.redirect_stdout(io.StringIO()):
        res = molden.load(path)
    mol, mo_coeff, mo_occ = res[0], np.asarray(res[2]), np.asarray(res[3], float)
    if mo_coeff.ndim == 3:                     # spin-unrestricted files: both spins
        dm = sum(c @ np.diag(o) @ c.T for c, o in zip(mo_coeff, mo_occ))
    else:
        dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
    return mol, dm


def _plot_species(path):
    return "positron" if "positron" in os.path.basename(path).lower() else "electron"


# ------------------------------------------------------------------------------------------ interactive viewer
_VIEW_EXT = (".molden", ".molden.input", ".cube", ".cub")


def is_view_file(path):
    return str(path).lower().endswith(_VIEW_EXT)


class _Quit(Exception):
    pass


def _ask(prompt, default=None):
    try:
        v = input(prompt).strip()
    except EOFError:
        raise _Quit
    if v.lower() in ("q", "quit", "exit"):
        raise _Quit
    return v if v else ("" if default is None else str(default))


def _view_stage(files, settings):
    """A stage-like object with one %density_map request, for the map functions."""
    st = JobStage(1)
    st.job_name = settings["prefix"]
    st.system_type = "POSITRON" if "positron" in files else "ELECTRONIC"
    st.density_map_plane = settings["plane"]
    st.density_maps = []
    line = settings["what"]
    rng = {"electron": settings["range_electron"], "positron": settings["range_positron"], "esp": settings["range_esp"]}
    if line == "both":
        line += f" {rng['electron'] or 'auto'} {rng['positron'] or 'auto'}"
    elif rng[line]:
        line += f" {rng[line]}"
    if settings["colors"]:
        line += f" {settings['colors']}"
    if settings["ystep"]:
        line += f" ystep={settings['ystep']}"
    parse_density_map(line, st)
    _validate_wavefunction_export(st)
    return st


def run_viewer(paths):
    """`python newBRORSEN.v04.06.py file.molden [other.molden]`: a Multiwfn-style menu that reads Molden / cube files
    (an electron one, a positron one, or both) and draws the wrapper's plane maps (density, ESP) with any options,
    or computes CHELPG charges, without a calculation."""
    files, species = {}, {}

    def load(path):
        if not os.path.exists(path):
            print(f"  No such file: {path}")
            return
        sp = _plot_species(path)
        if sp in files:
            a = _ask(f"  {path}: an {sp} file is already loaded ({files[sp]}). Load it as the electron or positron "
                     f"file? [{sp}] ", sp).lower().rstrip("s")
            sp = a if a in ("electron", "positron") else sp
        else:
            a = _ask(f"  {path}: electron or positron density? [{sp}] ", sp).lower().rstrip("s")
            sp = a if a in ("electron", "positron") else sp
        try:
            mol, d = load_plot_file(path)
        except Exception as err:
            print(f"  Cannot read {path}: {type(err).__name__}: {err}")
            return
        files[sp], species[sp] = path, (mol, d)
        if isinstance(d, CubeGrid):
            print(f"  Loaded {path} as the {sp} density: cube {d.shape[0]}x{d.shape[1]}x{d.shape[2]}, {len(d.atoms)} "
                  f"atoms, integral {d.integral():.4f}")
        else:
            n = float(np.einsum("ij,ji", d, mol.intor("int1e_ovlp")))
            print(f"  Loaded {path} as the {sp} density: {mol.natm} atoms, {mol.nao} basis functions, {n:.4f} "
                  f"particles")

    print(f"newBRORSEN v{VERSION} | file viewer (maps and charges from Molden / cube files, no calculation)")
    try:
        for p in paths:
            load(p)
        if not files:
            return 1
        stem = os.path.splitext(os.path.basename(next(iter(files.values()))))[0]
        settings = {"what": "both" if len(files) == 2 else next(iter(files)), "plane": "AUTO", "colors": "",
                    "range_electron": "", "range_positron": "", "range_esp": "", "ystep": "", "prefix": stem}
        while True:
            have_molden_e = "electron" in files and not isinstance(species["electron"][1], CubeGrid)
            print("\n" + " " * 20 + "------------ Main function menu ------------")
            print("  0 Draw the plane map with the current settings")
            print("  1 Load another file (electron or positron)")
            print(f"  2 Map: {settings['what']}   (electron / positron / both / esp)")
            print(f"  3 Plane: {settings['plane']}   (AUTO, XY, XZ, YZ)")
            print(f"  4 Colour range, electrons: {settings['range_electron'] or 'auto (0 to 0.65 a.u.)'}")
            print(f"  5 Colour range, positron: {settings['range_positron'] or 'auto (0 to the maximum)'}")
            print(f"  6 Colour range, ESP: {settings['range_esp'] or 'auto (-0.1 to 0.1 a.u.)'}")
            print(f"  7 Colour scheme: {settings['colors'] or 'default (rainbow_bw; blue_white_red for ESP)'}")
            print(f"  8 Labels on the vertical axis every (bohr): {settings['ystep'] or 'auto'}")
            print(f"  9 Output file prefix: {settings['prefix']}")
            print(f" 10 CHELPG charges{'' if have_molden_e else '   (needs an electron .molden file)'}")
            print(" 11 Information on the loaded files")
            print(" 12 Combined figure from several files (e.g. 0, 1 and 2 positrons side by side)")
            print("  q Quit")
            c = _ask("  Choice: ")
            if c == "0":
                need = {"both": ["electron", "positron"], "esp": ["electron"]}.get(settings["what"], [settings["what"]])
                missing = [sp for sp in need if sp not in files]
                if missing:
                    print(f"  The {settings['what']} map needs a loaded {' and '.join(missing)} file (option 1)")
                    continue
                if settings["what"] == "esp" and not have_molden_e:
                    print("  The ESP map needs the electron .molden file (a density cube is not enough)")
                    continue
                st = _view_stage(files, settings)
                if st.errors:
                    print("  " + "\n  ".join(st.errors))
                    continue
                sp_all = dict(species)
                sp_all.setdefault("electron", species.get("positron"))
                st._plot_source = "from " + ", ".join(os.path.basename(f) for f in files.values())
                view = map_request_view(st, st.density_maps[0])
                view._plot_source = st._plot_source
                for f in report_density_maps(view, "file", sp_all):
                    print(f"  Written: {f}")
            elif c == "1":
                load(_ask("  File: "))
            elif c == "2":
                v = _ask("  electron, positron, both or esp: ").lower()
                if v in ("electron", "positron", "both", "esp"):
                    settings["what"] = v
            elif c == "3":
                v = _ask("  AUTO, XY, XZ or YZ: ").upper()
                if v in ("AUTO", "XY", "XZ", "YZ"):
                    settings["plane"] = v
            elif c in ("4", "5", "6"):
                key = {"4": "range_electron", "5": "range_positron", "6": "range_esp"}[c]
                v = _ask("  lower,upper in a.u. (e.g. 0,0.4), or auto: ").replace(" ", "")
                settings[key] = "" if v.lower() in ("", "auto") else v
            elif c == "7":
                names = list(_MAP_SCALES)
                for k in range(0, len(names), 4):
                    print("    " + "   ".join(f"{k + i:2d} {names[k + i]:<20s}" for i in range(4) if k + i < len(names)))
                v = _ask("  Number or name (Enter: default): ")
                settings["colors"] = names[int(v)] if v.isdigit() and int(v) < len(names) else v
            elif c == "8":
                v = _ask("  Spacing in bohr (Enter: auto): ")
                settings["ystep"] = v
            elif c == "9":
                settings["prefix"] = _ask("  Prefix: ", settings["prefix"]) or settings["prefix"]
            elif c == "10":
                if not have_molden_e:
                    print("  CHELPG needs the electron .molden file")
                    continue
                st = JobStage(1)
                st.job_name = settings["prefix"]
                st.system_type = "POSITRON" if "positron" in files else "ELECTRONIC"
                report_chelpg(st, "file", {"electron": species["electron"]})
            elif c == "11":
                for sp, f in files.items():
                    mol, d = species[sp]
                    print(f"  {sp}: {f}")
                    for i in range(mol.natm):
                        x = mol.atom_coord(i) * BOHR_TO_ANGSTROM
                        print(f"     {i + 1:3d} {mol.atom_symbol(i):<3s} {x[0]:12.6f} {x[1]:12.6f} {x[2]:12.6f}  A")
            elif c == "12":
                print("  Files in figure order, separated by spaces; the species comes from the name (positron in it"
                      "\n  or not); add :electron or :positron to a file to choose")
                entries = []
                for tok in _ask("  Files: ").split():
                    path, _, sp = tok.partition(":")
                    sp = sp.lower().rstrip("s") if sp else _plot_species(path)
                    if sp not in ("electron", "positron") or not os.path.exists(path):
                        print(f"  Skipped {tok}: {'no such file' if not os.path.exists(path) else 'unknown species'}")
                        continue
                    try:
                        mol, d = load_plot_file(path)
                    except Exception as err:
                        print(f"  Cannot read {path}: {type(err).__name__}: {err}")
                        continue
                    entries.append((os.path.splitext(os.path.basename(path))[0], sp, mol, d, mol))
                if len(entries) < 2:
                    print("  A combined figure needs at least two readable files")
                    continue
                shared = _ask("  One colour scale per species over the figure? [Y/n] ", "y").lower() != "n"
                try:
                    ystep = float(settings["ystep"]) if settings["ystep"] else None
                except ValueError:
                    ystep = None
                for f in combined_density_map(f"{settings['prefix']}_combined_density_map", entries,
                                              settings["plane"], shared, "Densities: combined",
                                              "from " + ", ".join(e[0] for e in entries),
                                              settings["colors"] or "rainbow_bw", ystep):
                    print(f"  Written: {f}")
            else:
                print("  Unknown choice")
    except _Quit:
        print("  Bye.")
    return 0


def plane_points(mol, plane, offset, margin, n=300):
    """Grid points (bohr) of a plane map: rows from the top (largest vertical coordinate) down. Returns
    (points (nv*nh, 3), (nv, nh), extent (h0, h1, v0, v1))."""
    h, v, nrm = _MAP_AXES[plane]
    xyz = mol.atom_coords()
    lo, hi = xyz.min(0) - margin, xyz.max(0) + margin
    step = max(hi[h] - lo[h], hi[v] - lo[v]) / (n - 1)
    nh = int(round((hi[h] - lo[h]) / step)) + 1
    nv = int(round((hi[v] - lo[v]) / step)) + 1
    hh, vv = np.meshgrid(np.linspace(lo[h], hi[h], nh), np.linspace(hi[v], lo[v], nv))
    pts = np.zeros((hh.size, 3))
    pts[:, h], pts[:, v], pts[:, nrm] = hh.ravel(), vv.ravel(), offset
    return pts, (nv, nh), (lo[h], hi[h], lo[v], hi[v])


def plane_density(mol, dm, plane, offset, margin, n=300):
    """Density on a plane (bohr): rows from the top (largest vertical coordinate) down; extent (h0, h1, v0, v1).
    dm may also be a CubeGrid (v04.06 ! PLOT of a .cube file): the plane is interpolated from the grid."""
    from pyscf.dft import numint
    pts, (nv, nh), extent = plane_points(mol, plane, offset, margin, n)
    if isinstance(dm, CubeGrid):
        return dm.values_at(pts).reshape(nv, nh), extent
    dm = 0.5 * (dm + dm.T)
    rho = np.empty(len(pts))
    blk = max(100, int(1e8 / (8 * mol.nao)))
    for i in range(0, len(pts), blk):
        ao = numint.eval_ao(mol, pts[i:i + blk])
        rho[i:i + blk] = np.einsum("pi,ij,pj->p", ao, dm, ao)
    return rho.reshape(nv, nh), extent


def _map_colourize(data, vmin, vmax, colors="rainbow_bw"):
    scale = map_scale(colors)
    t = (data - vmin) / (vmax - vmin)
    s = np.clip(t, 0, 1) * (len(scale) - 1)
    i = np.minimum(s.astype(int), len(scale) - 2)
    f = (s - i)[..., None]
    rgb = scale[i] * (1 - f) + scale[i + 1] * f
    if colors == "rainbow_bw":         # Multiwfn's default: white above the scale, black below; the others clip
        rgb[t > 1] = 1.0
        rgb[t < 0] = 0.0
    return rgb


def _map_png(rgb):
    """Minimal RGB PNG (zlib + struct)."""
    import struct, zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + (np.clip(rgb[r], 0, 1) * 255).round().astype(np.uint8).tobytes() for r in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _map_ticks(lo, hi, target=6):
    raw = (hi - lo) / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9)), default=10 * mag)
    first = np.ceil(lo / step - 1e-9) * step
    return [round(float(t), 10) for t in np.arange(first, hi + step * 1e-6, step)], float(step)


def _map_fmt(x, step):
    """Tick label with as many decimals as the step needs (2.5 -> 1 decimal, 0.25 -> 2)."""
    decimals = next((d for d in range(8) if abs(step * 10 ** d - round(step * 10 ** d)) < 1e-6), 8)
    s = f"{x:.{decimals}f}"
    return "0" if float(s) == 0 else s


def _map_contours(data, level):
    """Marching squares: segments (fractional pixel coordinates) where data crosses `level`."""
    a, b, c, d = data[:-1, :-1], data[:-1, 1:], data[1:, 1:], data[1:, :-1]     # tl, tr, br, bl
    idx = (a > level) * 8 + (b > level) * 4 + (c > level) * 2 + (d > level) * 1
    segs = []

    def frac(v1, v2):
        return (level - v1) / (v2 - v1) if v2 != v1 else 0.5
    for r, q in zip(*np.nonzero((idx > 0) & (idx < 15))):
        tl, tr, br, bl = data[r, q], data[r, q + 1], data[r + 1, q + 1], data[r + 1, q]
        top, right = (q + frac(tl, tr), r), (q + 1, r + frac(tr, br))
        bottom, left = (q + frac(bl, br), r + 1), (q, r + frac(tl, bl))
        segs.extend({1: [(left, bottom)], 2: [(bottom, right)], 3: [(left, right)], 4: [(top, right)],
                     5: [(left, top), (bottom, right)], 6: [(top, bottom)], 7: [(left, top)], 8: [(left, top)],
                     9: [(top, bottom)], 10: [(left, bottom), (top, right)], 11: [(top, right)],
                     12: [(left, right)], 13: [(bottom, right)], 14: [(left, bottom)]}[idx[r, q]])
    return segs


def _map_chain(segs):
    """Join marching-squares segments into polylines (shared end points), so a dashed line keeps its pattern and
    the SVG stays small. Returns lists of points."""
    key = lambda q: (round(q[0], 6), round(q[1], 6))
    ends = {}
    for i, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append(i)
        ends.setdefault(key(b), []).append(i)
    used, lines = [False] * len(segs), []
    for i in range(len(segs)):
        if used[i]:
            continue
        used[i] = True
        line = list(segs[i])
        for forward in (True, False):
            while True:
                tip = key(line[-1] if forward else line[0])
                nxt = next((j for j in ends.get(tip, []) if not used[j]), None)
                if nxt is None:
                    break
                used[nxt] = True
                a, b = segs[nxt]
                far = b if key(a) == tip else a
                if forward:
                    line.append(far)
                else:
                    line.insert(0, far)
        lines.append(line)
    return lines


def _map_path(segs, L, T, sx, sy):
    return " ".join("M" + "L".join(f"{L + q[0] * sx:.1f},{T + q[1] * sy:.1f}" for q in line)
                    for line in _map_chain(segs))


def _map_panel(p, x0, y0, size=620):
    """SVG elements of one map panel; returns (elements, width, height). p: dict with data, extent, atoms,
    vmin, vmax, contours, unit, label, axis names."""
    import base64
    from html import escape
    data, (h0, h1, v0, v1) = p["data"], p["extent"]
    nv, nh = data.shape
    scale = size / max(h1 - h0, v1 - v0)
    wp, hp = (h1 - h0) * scale, (v1 - v0) * scale
    L, T, gap, cbw = x0 + 80, y0 + (40 if p.get("label") else 10), 30, 26
    X = lambda h: L + (h - h0) * scale
    Y = lambda v: T + (v1 - v) * scale
    vmin, vmax = p["vmin"], p["vmax"]
    img = base64.b64encode(_map_png(_map_colourize(data, vmin, vmax, p["colors"]))).decode()
    out = []
    if p.get("label"):
        out.append(f'<text x="{L + wp / 2:.1f}" y="{y0 + 24}" font-size="21" font-weight="bold" '
                   f'text-anchor="middle">{escape(p["label"])}</text>')
    out.append(f'<image x="{L:.1f}" y="{T:.1f}" width="{wp:.1f}" height="{hp:.1f}" preserveAspectRatio="none" '
               f'href="data:image/png;base64,{img}"/>')
    sx, sy = wp / (nh - 1), hp / (nv - 1)
    for lev in p.get("contours") or []:
        segs = _map_contours(data, lev)
        if segs:
            d = _map_path(segs, L, T, sx, sy)
            dash = ' stroke-dasharray="5,4"' if lev < 0 else ""
            out.append(f'<path d="{d}" stroke="black" stroke-width="0.8" fill="none"{dash}/>')
    if p.get("vdw") is not None:        # %density_map esp: the rho = 0.001 a.u. line (Multiwfn's plane-map option 15)
        segs = _map_contours(p["vdw"], 0.001)
        if segs:
            d = _map_path(segs, L, T, sx, sy)
            out.append(f'<path d="{d}" stroke="#303030" stroke-width="2.4" fill="none"/>')
    out.append(f'<rect x="{L:.1f}" y="{T:.1f}" width="{wp:.1f}" height="{hp:.1f}" fill="none" stroke="black" '
               f'stroke-width="1.5"/>')
    ticks, step = _map_ticks(h0, h1)
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" y1="{T + hp:.1f}" x2="{X(t):.1f}" y2="{T + hp + 7:.1f}" stroke="black"/>')
        out.append(f'<text x="{X(t):.1f}" y="{T + hp + 28:.1f}" font-size="17" text-anchor="middle">'
                   f'{_map_fmt(t, step)}</text>')
    step = p.get("ystep")
    if step and (v1 - v0) / step <= 60:        # %density_map ystep=<bohr>; more than 60 labels: automatic
        ticks = [round(float(t), 10) for t in np.arange(np.ceil(v0 / step - 1e-9) * step, v1 + step * 1e-6, step)]
    else:
        ticks, step = _map_ticks(v0, v1)
    for t in ticks:
        out.append(f'<line x1="{L - 7:.1f}" y1="{Y(t):.1f}" x2="{L:.1f}" y2="{Y(t):.1f}" stroke="black"/>')
        out.append(f'<text x="{L - 12:.1f}" y="{Y(t) + 6:.1f}" font-size="17" text-anchor="end">'
                   f'{_map_fmt(t, step)}</text>')
    hname, vname = p["axes"]
    out.append(f'<text x="{L + wp / 2:.1f}" y="{T + hp + 58:.1f}" font-size="17" text-anchor="middle">'
               f'{hname} (bohr)</text>')
    out.append(f'<text x="{x0 + 22}" y="{T + hp / 2:.1f}" font-size="17" text-anchor="middle" '
               f'transform="rotate(-90 {x0 + 22} {T + hp / 2:.1f})">{vname} (bohr)</text>')
    for sym, h, v in p["atoms"]:
        if h0 <= h <= h1 and v0 <= v <= v1:
            out.append(f'<text x="{X(h):.1f}" y="{Y(v) + 8:.1f}" font-size="22" font-weight="bold" fill="white" '
                       f'stroke="black" stroke-width="4" paint-order="stroke" stroke-linejoin="round" '
                       f'text-anchor="middle">{escape(sym)}</text>')
    bar = np.linspace(vmax, vmin, 256)[:, None] * np.ones((1, 4))
    bimg = base64.b64encode(_map_png(_map_colourize(bar, vmin, vmax, p["colors"]))).decode()
    cx = L + wp + gap
    out.append(f'<image x="{cx:.1f}" y="{T:.1f}" width="{cbw}" height="{hp:.1f}" preserveAspectRatio="none" '
               f'href="data:image/png;base64,{bimg}"/>')
    out.append(f'<rect x="{cx:.1f}" y="{T:.1f}" width="{cbw}" height="{hp:.1f}" fill="none" stroke="black"/>')
    ticks, step = _map_ticks(vmin, vmax)
    for t in ticks:
        y = T + (vmax - t) / (vmax - vmin) * hp
        out.append(f'<line x1="{cx + cbw:.1f}" y1="{y:.1f}" x2="{cx + cbw + 6:.1f}" y2="{y:.1f}" stroke="black"/>')
        out.append(f'<text x="{cx + cbw + 10:.1f}" y="{y + 6:.1f}" font-size="16">{_map_fmt(t, step)}</text>')
    out.append(f'<text x="{cx + cbw / 2:.1f}" y="{T - 10:.1f}" font-size="15" text-anchor="middle">'
               f'{escape(p["unit"])}</text>')
    return out, (cx + cbw + 70) - x0, (T + hp + 72) - y0


def write_density_map(stem, panels, title, subtitle):
    """<stem>.svg with the panels side by side (the colour-filled maps are embedded PNGs), and <stem>.png when an
    SVG converter (rsvg-convert, ImageMagick) is installed. Returns the files written."""
    import shutil, subprocess
    from html import escape
    top = 100
    body, x, height = [], 10, 0
    for p in panels:
        els, w, h = _map_panel(p, x, top)
        body += els
        x += w + 20
        height = max(height, h)
    width, height = x, top + height + 10
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
           f'viewBox="0 0 {width:.0f} {height:.0f}" font-family="Liberation Sans, Arial, Helvetica, sans-serif">',
           f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>',
           f'<text x="{width / 2:.0f}" y="42" font-size="25" font-weight="bold" text-anchor="middle">'
           f'{escape(title)}</text>',
           f'<text x="{width / 2:.0f}" y="76" font-size="{min(16.0, (width - 30) / (0.5 * max(len(subtitle), 1))):.1f}" '
           f'text-anchor="middle">{escape(subtitle)}</text>',
           *body, "</svg>"]
    with open(f"{stem}.svg", "w") as fh:
        fh.write("\n".join(svg))
    made = [f"{stem}.svg"]
    conv = shutil.which("rsvg-convert") or shutil.which("magick") or shutil.which("convert")
    if conv:
        cmd = ([conv, "-d", "150", "-p", "150", "-o", f"{stem}.png", f"{stem}.svg"] if conv.endswith("rsvg-convert")
               else [conv, "-density", "150", f"{stem}.svg", f"{stem}.png"])
        try:
            if subprocess.run(cmd, capture_output=True, timeout=300).returncode == 0 and os.path.exists(f"{stem}.png"):
                made.append(f"{stem}.png")
        except (OSError, subprocess.SubprocessError):
            pass
    return made


def density_map_panel(stage, sp, mol, dm, plane, offset, label=""):
    """One panel. Colour scale: the %density_map limits when given; by default the electron density on
    Multiwfn's 0-0.65 a.u. and the positron density from 0 to its maximum on the plane. Small scales (upper limit
    below 0.01 a.u.) are shown in 1e-3 a.u."""
    rho, extent = plane_density(mol, dm, plane, offset, _MAP_MARGIN_BOHR[sp], n=400)
    h, v, nrm = _MAP_AXES[plane]
    atoms = [(mol.atom_pure_symbol(i), c[h], c[v]) for i, c in enumerate(mol.atom_coords())
             if abs(c[nrm] - offset) <= 0.5]
    lim = stage.density_map_limits.get(sp)
    if lim:
        vmin, vmax = lim
    elif sp == "electron":
        vmin, vmax = 0.0, _MAP_ELECTRON_MAX
    else:
        top = max(float(rho.max()), 1e-12)
        step = _map_ticks(0.0, top)[1]
        vmin, vmax = 0.0, float(np.ceil(top / step - 1e-9) * step)
    factor, unit = (1000.0, "1e-3 a.u.") if max(abs(vmin), abs(vmax)) < 0.01 else (1.0, "a.u.")
    vmin, vmax = vmin * factor, vmax * factor
    if sp == "electron" and factor == 1.0:
        contours = [c for c in _MAP_ELECTRON_CONTOURS if vmin < c < vmax]
    else:
        ticks, step = _map_ticks(vmin, vmax)
        contours = [t for t in ticks if vmin < t < vmax and t > 0] + ([0.5 * step] if vmin < 0.5 * step < vmax else [])
    return {"data": rho * factor, "extent": extent, "atoms": atoms, "vmin": vmin, "vmax": vmax,
            "contours": contours, "unit": unit, "label": label, "axes": (plane[0], plane[1]),
            "colors": stage.density_map_colors, "ystep": stage.density_map_ystep}


def report_density_maps(stage, level, species):
    """%density_map: plane maps of one level in the job folder. electron / positron: one map; both: the two maps
    and a figure with the electron and positron maps side by side."""
    if stage.density_map == "esp":
        return report_esp_maps(stage, level, species)
    t0 = time.time()
    job = stage.job_name or f"Stage_{stage.stage_id}"
    lab = {"hf": "HF" if stage.system_type == "ELECTRONIC" else "POSHF", "ccsd": "CCSD"}.get(level, level)
    want = ["electron", "positron"] if stage.density_map == "both" else [stage.density_map]
    plane, offset = density_map_plane(stage, species["electron"][0])
    where = f"{plane} plane at {'XYZ'[_MAP_AXES[plane][2]]} = {offset:.2f} bohr"
    what = {"hf": "canonical orbitals", "ccsd": "unrelaxed CCSD density"}.get(level, getattr(stage, "_plot_source", level))
    panels, written = {}, []
    for sp in want:
        mol, dm = species[sp]
        panels[sp] = density_map_panel(stage, sp, mol, dm, plane, offset)
        written += write_density_map(f"{job}_{level}_{sp}_density_map", [panels[sp]],
                                     f"{sp.capitalize()} density: {job}, {lab}", f"{where}; {what}")
    if len(want) == 2:
        both = [dict(panels["electron"], label="Electrons"),
                dict(panels["positron"], label="Positron" if stage.n_pos == 1 else "Positrons")]
        written += write_density_map(f"{job}_{level}_electron_positron_density_map", both,
                                     f"Electron and positron densities: {job}, {lab}", f"{where}; {what}")
    for sp in want:
        p = panels[sp]
        print(f"  [Density Map] {lab} {sp} density, {where}: maximum on the plane "
              f"{float(p['data'].max()):.4g} {p['unit']} (scale {p['vmin']:g} to {p['vmax']:g} {p['unit']}"
              f"{', set in the input' if sp in stage.density_map_limits else ''}; colours {p['colors']})")
        if p.get("ystep"):
            span = p["extent"][3] - p["extent"][2]
            print(f"  [Density Map] {sp} map: vertical-axis labels every {p['ystep']:g} bohr" if span / p["ystep"] <= 60
                  else f"  [Density Map] {sp} map: ystep={p['ystep']:g} would give more than 60 labels over "
                       f"{span:.1f} bohr: automatic spacing used")
    if not any(f.endswith(".png") for f in written):
        print("  [Density Map] no SVG converter (rsvg-convert, ImageMagick) found: SVG files only (open them in a "
              "web browser or Inkscape)")
    print(f"  [Density Map] {len(written)} file{'s' * (len(written) != 1)}, {time.time() - t0:.1f} s")
    return written


_STAGE_DENSITIES = {}     # v04.07: job name -> {'levels': {level: {species: (mol, dm)}}, 'n_pos'} (export_wavefunction)


def combined_density_map(stem, entries, plane_setting, shared, title, what, colors="rainbow_bw", ystep=None):
    """One figure with several maps side by side (v04.07, %density_map_combine and the file viewer).
    entries: [(label, species, mol, dm, mol_e)], in figure order; mol_e (the electron molecule) fixes the plane.
    The plane (and its offset) is the first entry's for every panel, so the panels are the same slice; a warning
    is printed when the other entries' atoms are not where the first's are (beyond 0.1 bohr). shared: one colour
    scale per species over the figure (the largest of the panels' default upper limits)."""
    view = JobStage(1)
    view.density_map_plane, view.density_map_limits = plane_setting, {}
    view.density_map_colors, view.density_map_ystep = colors, ystep
    plane, offset = density_map_plane(view, entries[0][4])
    ref = entries[0][4]
    for label, sp, mol, dm, mol_e in entries[1:]:
        same = (mol_e.natm == ref.natm and
                [mol_e.atom_pure_symbol(i) for i in range(mol_e.natm)] == [ref.atom_pure_symbol(i) for i in range(ref.natm)])
        dev = float(np.abs(mol_e.atom_coords() - ref.atom_coords()).max()) if same else None
        if not same:
            print(f"  [Warning] combined map: {label} has other atoms than {entries[0][0]}; the plane is "
                  f"{entries[0][0]}'s")
        elif dev > 0.1:
            print(f"  [Warning] combined map: the atoms of {label} are up to {dev:.2f} bohr from those of "
                  f"{entries[0][0]}; all panels use {entries[0][0]}'s plane")
    panels = [density_map_panel(view, sp, mol, dm, plane, offset, label) for label, sp, mol, dm, _ in entries]
    if shared:
        for sp in ("electron", "positron"):
            idx = [k for k, e in enumerate(entries) if e[1] == sp]
            tops = [panels[k]["vmax"] / (1000.0 if panels[k]["unit"].startswith("1e-3") else 1.0) for k in idx]
            if len(idx) > 1 and len(set(tops)) > 1:
                view.density_map_limits[sp] = (0.0, max(tops))
                for k in idx:
                    label, _, mol, dm, _ = entries[k]
                    panels[k] = density_map_panel(view, sp, mol, dm, plane, offset, label)
    where = f"{plane} plane at {'XYZ'[_MAP_AXES[plane][2]]} = {offset:.2f} bohr"
    written = write_density_map(stem, panels, title, f"{where}; {what}")
    for (label, sp, *_), p in zip(entries, panels):
        print(f"  [Density Map] combined, {label}: {sp} density, maximum on the plane {float(p['data'].max()):.4g} "
              f"{p['unit']} (scale {p['vmin']:g} to {p['vmax']:g} {p['unit']}"
              f"{', shared' if shared and sp in view.density_map_limits else ''})")
    return written


def report_combined_density_map(stage):
    """%density_map_combine (v04.07): <job>_combined_density_map.svg/.png from the densities the named stages
    kept (_STAGE_DENSITIES). Skipped, with a warning, when a named stage left no density (failed export, or a
    stage skipped by --resume)."""
    req = stage.density_map_combine
    job = stage.job_name or f"Stage_{stage.stage_id}"
    print(f"\n[Combined Density Map] {', '.join(f'{j}:{sp}' for j, sp in req['entries'])} "
          f"(level {req['level']}, {req['scale']} scale)")
    entries, levels_used = [], []
    for j, sp in req["entries"]:
        kept = _STAGE_DENSITIES.get(j)
        if kept is None:
            print(f"  [Warning] combined map skipped: stage {j!r} left no density (its export failed or the stage "
                  f"was skipped by --resume)")
            return []
        level = req["level"] if req["level"] in kept["levels"] else "hf"
        if level != req["level"]:
            print(f"  [Notice] combined map: {j} has no CCSD density (not a CC stage); its HF density is used")
        mol, dm = kept["levels"][level][sp]
        n = kept["n_pos"]
        entries.append((f"{j}: {sp}s ({n} e+)", sp, mol, dm, kept["levels"][level]["electron"][0]))
        levels_used.append({"hf": "HF", "ccsd": "CCSD"}[level])
    what = (f"{levels_used[0]} densities" if len(set(levels_used)) == 1 else "densities at " + " / ".join(levels_used))
    t0 = time.time()
    written = combined_density_map(f"{job}_combined_density_map", entries, stage.density_map_plane,
                                   req["scale"] == "shared", f"Densities: {', '.join(j for j, _ in req['entries'])}",
                                   what)
    print(f"  [Combined Density Map] {len(written)} file{'s' * (len(written) != 1)}, {time.time() - t0:.1f} s")
    return written


_ESP_DEFAULT_RANGE = (-0.1, 0.1)     # a.u.


def report_esp_maps(stage, level, species):
    """%density_map esp: the electrostatic potential of the nuclei and the electrons, V(r) = sum_A Z_A/|r - R_A| -
    int rho_e(r')/|r - r'| dr', on the plane of %density_map_plane, with the rho_e = 0.001 a.u. line (the usual
    van der Waals surface) drawn thick. For positron systems the positron density is left out, as for %chelpg.
    200 x 200 points (Multiwfn's default), box 4.5 bohr beyond the atoms."""
    t0 = time.time()
    job = stage.job_name or f"Stage_{stage.stage_id}"
    lab = {"hf": "HF" if stage.system_type == "ELECTRONIC" else "POSHF", "ccsd": "CCSD"}.get(level, level)
    mol, dm = species["electron"]
    plane, offset = density_map_plane(stage, mol)
    pts, shape, extent = plane_points(mol, plane, offset, _MAP_MARGIN_BOHR["electron"], n=200)
    xyz, zeff = mol.atom_coords(), mol.atom_charges().astype(float)
    r = np.maximum(np.linalg.norm(pts[:, None, :] - xyz[None], axis=2), 1e-6)
    esp = ((zeff[None] / r).sum(1) - density_potential(mol, dm, pts, stage.memory_mb)).reshape(shape)
    rho, _ = plane_density(mol, dm, plane, offset, _MAP_MARGIN_BOHR["electron"], n=200)
    vmin, vmax = stage.density_map_limits.get("esp") or _ESP_DEFAULT_RANGE
    factor, unit = (1000.0, "1e-3 a.u.") if max(abs(vmin), abs(vmax)) < 0.01 else (1.0, "a.u.")
    vmin, vmax = vmin * factor, vmax * factor
    ticks, _ = _map_ticks(vmin, vmax)
    h, v, nrm = _MAP_AXES[plane]
    atoms = [(mol.atom_pure_symbol(i), c[h], c[v]) for i, c in enumerate(mol.atom_coords())
             if abs(c[nrm] - offset) <= 0.5]
    panel = {"data": esp * factor, "extent": extent, "atoms": atoms, "vmin": vmin, "vmax": vmax,
             "contours": [t for t in ticks if vmin < t < vmax and abs(t) > 1e-12], "unit": unit, "label": "",
             "axes": (plane[0], plane[1]), "colors": stage.density_map_colors, "ystep": stage.density_map_ystep,
             "vdw": rho}
    where = f"{plane} plane at {'XYZ'[nrm]} = {offset:.2f} bohr"
    what = {"hf": "canonical orbitals", "ccsd": "unrelaxed CCSD density"}.get(level, getattr(stage, "_plot_source", level))
    who = "nuclei + electrons, positron not included" if stage.system_type == "POSITRON" else "nuclei + electrons"
    written = write_density_map(f"{job}_{level}_esp_map", [panel], f"Electrostatic potential: {job}, {lab}",
                                f"{where}; {who}; {what}; thick line: rho = 0.001 a.u.")
    far = rho < 0.001                    # outside the van der Waals surface
    print(f"  [ESP Map] {lab} electrostatic potential of the {who}, {where}: on the rho = 0.001 a.u. line and beyond "
          f"{float(esp[far].min()):.4f} to {float(esp[far].max()):.4f} a.u. (scale {vmin:g} to {vmax:g} {unit}"
          f"{', set in the input' if 'esp' in stage.density_map_limits else ''}; colours {stage.density_map_colors})")
    if stage.density_map_ystep:
        print(f"  [ESP Map] vertical-axis labels every {stage.density_map_ystep:g} bohr")
    if not any(f.endswith(".png") for f in written):
        print("  [ESP Map] no SVG converter (rsvg-convert, ImageMagick) found: SVG files only")
    print(f"  [ESP Map] {len(written)} file{'s' * (len(written) != 1)}, {time.time() - t0:.1f} s")
    return written


def export_wavefunction(stage, atoms):
    """%molden / %cube (v04.05): Molden and cube files at `atoms` (the stage's final geometry), at the HF and,
    for CC stages, the CCSD level, for electrons and positrons separately (Multiwfn, Jmol, VMD, ...).

    Files (<job> = %jobname):
      <job>_hf_electrons.molden, <job>_hf_positrons.molden      canonical HF / POSHF orbitals with occupations
      <job>_ccsd_electrons_natorb.molden, <job>_ccsd_positrons_natorb.molden
                                                                natural orbitals of the CCSD density, occupations
      <job>_{hf,ccsd}_{electron,positron}_density.cube           densities on a %cube_points^3 grid
    A positron file is written like an electron one (the formats have no positrons): its density and orbitals
    are right, but a program that combines the density with the nuclear charges (ESP, QTAIM) treats the
    positron as an electron. The CC density is the unrelaxed (Lambda) CCSD one-particle density: electronic,
    PySCF's make_rdm1; positrons, the public density_model(..., 'unrelaxed') with frozen core embedded.
    The explicit fixed-orbital level excludes frozen/active occupied relaxation. This local compatibility
    copy requires the new density API. It costs one more single point plus the Lambda equations."""
    import importlib
    from pyscf.tools import molden, cubegen
    job = stage.job_name or f"Stage_{stage.stage_id}"
    atom_str = atoms_to_string(atoms)
    use_df = stage.use_df
    t0 = time.time()
    written, levels = [], {}
    which = ", ".join(f"%{k.rstrip('s') if k == 'density_maps' else k}" for k in _FINAL_ANALYSIS_KEYS
                      if getattr(stage, k)) or "densities kept for %density_map_combine"
    print(f"\n[Wavefunction Export] {which} at the final geometry ({'DF' if use_df else 'CONV'})...")

    def mo_file(mol, name, coeff, occ, ene=None):
        if np.asarray(coeff).ndim == 3:
            # spin-orbital POSSO route (odd/higher-spin %n_pos with %pos_mult): coeff/occ/ene each have an
            # extra leading spin-channel axis, shape (2, nao, nmo) / (2, nmo). molden.from_mo (-> orbital_coeff)
            # writes one spin block per call and does not accept this combined shape (PySCF's own UHF path,
            # molden.dump_scf, calls orbital_coeff per channel into one open file; mirrored here).
            with open(name, "w") as fh:
                molden.header(mol, fh)
                for s, label in enumerate(("Alpha", "Beta")):
                    molden.orbital_coeff(mol, fh, coeff[s], spin=label,
                                         ene=(ene[s] if ene is not None else None), occ=occ[s])
        else:
            molden.from_mo(mol, name, coeff, occ=occ, ene=ene)
        written.append(name)

    def ao(c, d):
        return c @ d @ c.T

    def density_from_mo(c, occ):
        """AO density from HF MO coefficients and occupations (c * occ) @ c.T; the spin-orbital POSSO
        route (odd/higher-spin %n_pos with %pos_mult, e.g. %n_pos 3 %pos_mult 4) gives c an extra leading
        spin-channel axis (shape (2, nao, nmo), occ (2, nmo)), which (c * occ) @ c.T cannot broadcast
        (nao vs 2 mismatch, v04.06 bug); both channels are summed into one combined (nao, nao) density,
        like every other density here, with np.einsum("sij,sj,skj->ik", c, occ, c)."""
        if c.ndim == 3:
            return np.einsum("sij,sj,skj->ik", c, occ, c)
        return (c * occ) @ c.T

    cc_wanted = stage.method in ("CCSD", "CCSD(T)")
    if stage.system_type == "ELECTRONIC":
        mf, open_shell = _electronic_mf(stage, atom_str, use_df)
        mf.max_memory, mf.conv_tol, mf.verbose = stage.memory_mb, 1e-11, 0
        mf.kernel()
        mol = mf.mol
        if stage.molden:
            molden.from_scf(mf, f"{job}_hf_electrons.molden")
            written.append(f"{job}_hf_electrons.molden")
        dm = mf.make_rdm1()
        levels["hf"] = {"electron": (mol, dm[0] + dm[1] if open_shell else dm)}
        if cc_wanted:
            cc_obj = _electronic_cc(mf, stage)
            cc_obj.verbose = 0
            cc_obj.kernel()
            cc_obj.solve_lambda()
            d = cc_obj.make_rdm1()
            if open_shell:
                ca, cb = mf.mo_coeff
                dm_cc = ao(ca, d[0]) + ao(cb, d[1])
            else:
                dm_cc = ao(mf.mo_coeff, d)
            levels["ccsd"] = {"electron": (mol, dm_cc)}
    else:
        system = safe_build_system(stage, atom_str)
        mf = make_poshf(system, stage)
        mf.max_memory, mf.conv_tol, mf.verbose = stage.memory_mb, 1e-11, 0
        run_poshf(mf)
        mol_e, mol_p = system.mol_e, system.mol_p
        lab = next(k for k in mf.mo_coeff if k != "e")
        if stage.molden:
            mo_file(mol_e, f"{job}_hf_electrons.molden", mf.mo_coeff["e"], mf.mo_occ["e"], mf.mo_energy["e"])
            mo_file(mol_p, f"{job}_hf_positrons.molden", mf.mo_coeff[lab], mf.mo_occ[lab], mf.mo_energy[lab])
        from pyscf_neo_cc.electrostatics import density_model
        snapshot = density_model(mf, 'hf')
        names = {'e': 'electron', 'p': 'positron'}
        levels["hf"] = {names[item.label]: (item.mol, item.dm)
                        for item in snapshot.components}
        if cc_wanted and not (stage.n_pos >= 2 and stage.pos_spin):
            cc_obj = build_positron_cc(mf, stage, gradient_required=False)
            cc_obj.verbose = 0
            cc_obj.kernel()
            cc_obj.solve_lambda()
            # Explicit fixed-orbital T+Lambda density, with the frozen core
            # embedded. The old DF private helper also added frozen/active
            # occupied response; the old conventional helper omitted frozen
            # orbitals and could not multiply the full coefficient matrix.
            snapshot = density_model(cc_obj, 'unrelaxed')
            levels["ccsd"] = {names[item.label]: (item.mol, item.dm)
                              for item in snapshot.components}

    for level, species in levels.items():
        for sp, (mol, dm) in species.items():
            n = float(np.einsum("ij,ji", 0.5 * (dm + dm.T), mol.intor("int1e_ovlp")))
            if stage.molden and level != "hf":
                coeff, occ = _natural_orbitals(mol, dm)
                name = f"{job}_{level}_{sp}s_natorb.molden"
                mo_file(mol, name, coeff, occ)
                print(f"  [Export] {level.upper()} {sp} natural orbitals: {n:.6f} particles, largest occupations "
                      f"{', '.join(f'{x:.5f}' for x in occ[:4])}")
            if stage.cube:
                name = f"{job}_{level}_{sp}_density.cube"
                pts = stage.cube_points
                margin = CUBE_MARGIN_BOHR[sp]
                rho = cubegen.density(mol, name, 0.5 * (dm + dm.T), nx=pts, ny=pts, nz=pts, margin=margin)
                box = cubegen.Cube(mol, pts, pts, pts, margin=margin).box
                held = float(np.sum(rho)) * abs(np.linalg.det(box)) / (pts - 1) ** 3
                print(f"  [Export] {level.upper()} {sp} density cube: {pts}^3 points, {margin:.0f} bohr beyond the "
                      f"atoms, holds {held:.4f} of {n:.4f} particles")
                written.append(name)
    _STAGE_DENSITIES[job] = {"levels": levels, "n_pos": stage.n_pos if stage.system_type == "POSITRON" else 0}
    analyses = [(stage, report_chelpg, "%chelpg")] if stage.chelpg else []
    analyses += [(map_request_view(stage, req), report_density_maps, f"%density_map {req.density_map}")
                 for req in stage.density_maps]
    for level, species in levels.items():
        for view, func, key in analyses:
            try:
                files = func(view, level, species)
            except Exception as err:            # one failed analysis must not stop the others
                print(f"  [Notice] {key} ({level.upper()}) failed: {type(err).__name__}: {err}")
                continue
            if func is report_density_maps:
                written += files
    for name in written:
        print(f"[Artifact Generated] {name}")
    print(f"  [Wavefunction Export] {len(written)} files, {time.time() - t0:.1f} s")
    return written


def main(argv=None):
    args_in = sys.argv[1:] if argv is None else list(argv)
    if args_in and args_in[0] == "--view":           # v04.07: explicit flag (was v04.06's bare file.molden
        sys.exit(run_viewer(args_in[1:]))            # auto-detection); reached from --ui's "View molden/cube"
    if args_in == ["--parse-text-json"]:                      # v04.07: stdin -> JSON diagnostics, no files
        diags = parse_text(sys.stdin.read())
        sys.stdout.write(json.dumps(diags) + "\n")
        sys.exit(2 if any(d["severity"] == "error" for d in diags) else 0)
    parser = argparse.ArgumentParser(description=f"newBRORSEN v{VERSION} multicomponent CC wrapper")
    parser.add_argument("input_file", help="Path to the .inp or .txt job file")
    parser.add_argument("--dry-run", action="store_true",
                        help="parse and validate the input, print the stage summary, run nothing")
    parser.add_argument("--decimal-comma", action="store_true",
                        help="write numbers in .scan/_scan_points/_morse_fit_summary/_summary/_profile "
                             "files with a decimal comma (xyz files and the .out log keep dots)")
    parser.add_argument("--resume", action="store_true",
                        help="skip the stages that already finished (their <job>_stage<N>_done.json matches the "
                             "input) and restart an interrupted optimization from its checkpoint geometry; "
                             "automatic after a SLURM requeue (SLURM_RESTART_COUNT > 0)")
    parser.add_argument("--stage-records", action="store_true",
                        help="write the stage records <job>_stage<N>_started/done.json that --resume uses, also "
                             "outside SLURM (inside a SLURM job, and with --resume, they are always written)")
    args = parser.parse_args(argv)
    global DECIMAL_COMMA, RESUME, STAGE_RECORDS
    DECIMAL_COMMA = args.decimal_comma
    restarts = os.environ.get("SLURM_RESTART_COUNT", "0").strip()
    RESUME = bool(args.resume) or (restarts not in ("", "0") and not args.dry_run)
    STAGE_RECORDS = (RESUME or bool(args.stage_records)
                     or bool(os.environ.get("SLURM_JOB_ID", "").strip()))

    out_filename = os.path.splitext(args.input_file)[0] + ".out"
    tee = TeeOutput(out_filename, append=RESUME and os.path.exists(out_filename))
    sys.stdout = tee
    sys.stderr = tee
    if RESUME and not args.resume:
        print(f"[Resume] SLURM_RESTART_COUNT={restarts}: this job was requeued; finished stages are skipped.")
    try:
        _run_main(args.input_file, dry_run=args.dry_run)
    except InputValidationError:
        print("[ERROR] Input validation failed (see the list above); nothing was computed.")
        exit_code = 2
    except PreflightError as err:
        print(f"[ERROR] Pre-flight check failed: {err}")
        exit_code = 3
    else:
        exit_code = 0
    finally:
        sys.stdout = tee.terminal_out
        sys.stderr = tee.terminal_err
        tee.close()
    if exit_code:
        sys.exit(exit_code)


def _run_main(input_file, dry_run=False):
    print(f"newBRORSEN v{VERSION} | pyscf {getattr(pyscf, '__version__', '?')} | input: {input_file}")
    if RELEASED:
        print(f"Production release v{VERSION} (pyscf_neo_cc 0.1.0.post20261008 on stock PySCF 2.14).")
    else:
        print("*" * 80)
        print("* UNRELEASED TEST BUILD: not approved for production calculations.        *")
        print("* v04.09: Kurt 2026-10-08 merge (needs pyscf_neo_cc 0.1.0.post20261008); ranges. *")
        print("* Production: v04.08                                                      *")
        print("*" * 80)
    if DECIMAL_COMMA:
        print("[Output] --decimal-comma: report/table files use a decimal comma; .xyz files and this log keep dots.")
    if STAGE_RECORDS and not dry_run:
        why = (f"SLURM job {os.environ['SLURM_JOB_ID'].strip()}" if os.environ.get("SLURM_JOB_ID", "").strip()
               else "--resume" if RESUME else "--stage-records")
        print(f"[Output] Stage records on ({why}): <job>_stage<N>_started/done.json let a requeued or resumed job "
              f"skip the finished stages.")
    stages = parse_input_file(input_file)
    run_pre_validation(stages)
    if dry_run:
        print("[Dry run] Input is valid; nothing was computed.")
        return

    last_optimized_atoms = None
    by_id = {s.stage_id: s for s in stages}
    last_optimized_conditioning = None
    for s in stages:
        if not s.has_explicit_xyz and last_optimized_atoms is not None:
            s.atoms = copy.deepcopy(last_optimized_atoms)
            _inherit_conditioning(s, last_optimized_conditioning)
        if s.run_type == "ZOPT" and s.zmat_from:
            _continue_zmat(s, by_id[s.zmat_from])

        _prepare_conditioning(s)
        signature = None
        if STAGE_RECORDS or RESUME:
            signature = (_conditioning_declaration_signature(s)
                         if s.aux_conditioning is not None else stage_signature(s))
        if RESUME:
            done = read_stage_record(s, "done", signature)
            if done is not None:
                res = decode_stage_result(done.get("result"))
                print("=" * 80)
                print(f"[Resume] Stage {s.stage_id} ('{s.job_name}') finished on {done.get('finished', '?')} "
                      f"(newBRORSEN v{done.get('version', '?')}): skipped, its result is reused.")
                if isinstance(res, list) and s.run_type in ("OPT", "SCAN", "ZOPT", "TS", "NEB"):
                    last_optimized_atoms = res
                    last_optimized_conditioning = _conditioning_continuation(s)
                continue
            if s.run_type == "OPT" and read_stage_record(s, "started", signature) is not None:
                restart_from_checkpoint(s)
        if STAGE_RECORDS:
            write_stage_record(s, "started", signature)

        _apply_df_mem_cap(s)
        t_cpu, t_wall = time.process_time(), time.time()
        res = execute_stage(s)
        if (wants_final_analysis(s) or s._keep_density) and s.run_type in ("SP_ENERGY", "GRADIENT", "OPT", "ZOPT",
                                                                          "FREQ", "TS"):
            final = res if (isinstance(res, list) and s.run_type in ("OPT", "ZOPT")) else s.atoms
            try:
                export_wavefunction(s, final)
            except Exception as err:
                print(f"  [Notice] final-geometry analysis failed ({type(err).__name__}: {err}); the stage's "
                      f"results are unaffected.")
        if s.density_map_combine:
            try:
                for name in report_combined_density_map(s):
                    print(f"[Artifact Generated] {name}")
            except Exception as err:
                print(f"  [Notice] %density_map_combine failed ({type(err).__name__}: {err}); the stage's "
                      f"results are unaffected.")
        cpu, wall = time.process_time() - t_cpu, time.time() - t_wall
        if STAGE_RECORDS:
            write_stage_record(s, "done", signature, result=res)
        print(f"  [Stage Timing] '{s.job_name}': CPU {cpu:.1f} s ({cpu/3600:.3f} h), "
              f"Wall {wall:.1f} s ({wall/3600:.3f} h)")

        if isinstance(res, list) and s.run_type in ("OPT", "SCAN", "ZOPT", "TS", "NEB"):
            last_optimized_atoms = res
            last_optimized_conditioning = _conditioning_continuation(s)
        gc.collect()


if __name__ == "__main__":
    main()
