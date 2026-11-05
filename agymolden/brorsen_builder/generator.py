"""
Stage state and input text generator for newBRORSEN calculations.
Decoupled model of a multi-stage input file.
"""

import re
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np

from .schema import RUN_TYPES, METHODS, SECTORS, INTEGRAL_MODES, COORD_UNITS


class BrorsenStage:
    """Represents one calculation stage (! line, directives, and optional geometry)."""

    def __init__(self, number: int = 1):
        self.number: int = number
        self.jobname: str = f"Stage_{number}"
        
        # Bang line components
        self.run_type: str = "SP_ENERGY"
        self.method: str = "HF"
        self.sector: str = "ELECTRONIC"
        self.integrals: str = "DF"
        self.coords_unit: str = "Angs"
        
        # Physical settings
        self.charge: int = 0
        self.mult: int = 1
        self.nprocs: int = 4
        self.memory: int = 4000
        
        # General options
        self.verbose: int = 4  # Default to verbose=4 for rich detailed SCF/CC convergence and output info
        
        # Electronic options
        self.basis: str = "cc-pvdz"
        self.frozen: str = "0"
        self.jkfit_aux_e: str = "auto"
        self.auxbasis_e: str = "default"
        self.xc: str = "b3lyp"
        
        # Positron options
        self.n_pos: int = 1
        self.pos_basis: str = "PSX-TZ"
        self.psx_set: str = ""
        self.pos_atom: str = "all"
        self.pos_mult: int = 1
        self.pos_mass: str = "positron"
        self.quantum_site: str = ""
        
        # Proton / NEO options
        self.nuc_basis: str = "pb4f1"
        self.quantum_nuc: str = ""
        self.cneo: bool = False
        self.cneo_variant: str = "ccd"
        self.epc: str = "17-2"
        
        # Optimization & TS options
        self.opt_engine: str = "GEOMETRIC"
        self.opt_max_cycles: int = 50
        self.ts_mode: str = "ts"
        self.irc_direction: str = "both"
        self.neb_product: str = ""
        self.neb_images: int = 7
        self.neb_optimize_ends: bool = False
        
        # ZOPT options
        self.zopt_max_evals: int = 60
        self.zopt_etol: float = 1e-6
        self.zopt_step_bond: float = 0.02
        self.zopt_step_angle: float = 2.0
        self.zopt_tol_bond: float = 0.001
        self.zopt_tol_angle: float = 0.1
        self.zopt_gradient_every: int = 0
        self.zopt_final_gradient: bool = False
        self.zopt_restart: bool = True
        
        # Properties / Output
        self.cube: bool = False
        self.cube_points: int = 80
        self.molden: bool = True
        self.chelpg: bool = False
        self.density_map: bool = False
        self.density_map_type: str = "both"  # "electron", "positron", "both", "esp"
        self.density_map_plane: str = "AUTO"
        
        # Scan
        self.scan_enabled: bool = False
        self.scan_type: str = "bond"
        self.scan_target: str = ""
        self.scan_start: float = 1.0
        self.scan_end: float = 2.0
        self.scan_steps: int = 10

        # Constraints, Pinnings & Equivalences
        self.pinnings: List[int] = []  # 1-based atom indices to freeze, e.g. [1, 2] -> %pin 1,2
        self.constraints: List[str] = []  # e.g. ["bond 1 2 1.40", "angle 1 2 3", "dihedral 1 2 3 4 180.0"]
        self.equivalences: List[str] = []  # e.g. ["1,2,3,4"] -> %equivalence 1,2,3,4 (bond 1-2 == bond 3-4)
        
        # Geometry override: if None, inherits from previous stage or parent molecule
        self.has_own_geometry: bool = True
        self.geometry_format: str = "xyz"  # "xyz" or "zmat"

    def copy(self, new_number: int) -> 'BrorsenStage':
        """Clone this stage with a new stage number."""
        import copy
        new_st = copy.deepcopy(self)
        new_st.number = new_number
        new_st.jobname = f"Stage_{new_number}"
        return new_st

    def build_bang_line(self) -> str:
        """Construct the '! ...' line following newBRORSEN order."""
        tokens = [self.run_type, self.method, self.sector, self.integrals, self.coords_unit]
        return "! " + " ".join(tokens)

    def generate_directives(self) -> List[str]:
        """Generate directive lines (%key value)."""
        lines = []
        if self.jobname:
            lines.append(f"%jobname {self.jobname}")
        lines.append(f"%charge {self.charge}")
        lines.append(f"%mult {self.mult}")
        lines.append(f"%nprocs {self.nprocs}")
        lines.append(f"%memory {self.memory}")
        if self.verbose != 0:
            lines.append(f"%verbose {self.verbose}")
        
        # Sector-specific directives
        if self.sector == "ELECTRONIC":
            lines.append(f"%basis {self.basis}")
            if self.frozen and self.frozen != "0":
                lines.append(f"%frozen {self.frozen}")
            if self.method == "DFT":
                lines.append(f"%xc {self.xc}")
        elif self.sector == "POSITRON":
            lines.append(f"%basis {self.basis}")
            if self.frozen and self.frozen != "0":
                lines.append(f"%frozen {self.frozen}")
            lines.append(f"%n_pos {self.n_pos}")
            lines.append(f"%pos_basis {self.pos_basis}")
            if self.psx_set:
                lines.append(f"%psx_set {self.psx_set}")
            if self.pos_atom and self.pos_atom != "all":
                lines.append(f"%pos_atom {self.pos_atom}")
            if self.n_pos >= 2:
                lines.append(f"%pos_mult {self.pos_mult}")
            if self.pos_mass and self.pos_mass != "positron":
                lines.append(f"%pos_mass {self.pos_mass}")
            if self.quantum_site:
                lines.append(f"%quantum_site {self.quantum_site}")
        elif self.sector == "PROTON":
            lines.append(f"%basis {self.basis}")
            lines.append(f"%nuc_basis {self.nuc_basis}")
            if self.quantum_nuc:
                lines.append(f"%quantum_nuc {self.quantum_nuc}")
            if self.cneo:
                lines.append("%cneo true")
                lines.append(f"%cneo_variant {self.cneo_variant}")
            if self.epc and self.epc != "none":
                lines.append(f"%epc {self.epc}")
            if self.method == "DFT":
                lines.append(f"%xc {self.xc}")

        # Run-type specific directives
        if self.run_type == "OPT":
            lines.append(f"%opt_engine {self.opt_engine.lower()}")
            lines.append(f"%opt_max_cycles {self.opt_max_cycles}")
            if self.opt_engine.upper() == "ZOPT":
                lines.append(f"%zopt_max_evals {self.zopt_max_evals}")
                lines.append(f"%zopt_etol {self.zopt_etol}")
                lines.append(f"%zopt_step_bond {self.zopt_step_bond}")
                lines.append(f"%zopt_step_angle {self.zopt_step_angle}")

            # Optimization Constraints, Pinnings & Equivalences
            if self.pinnings:
                pin_str = ",".join(str(p) for p in sorted(set(self.pinnings)))
                lines.append(f"%pin {pin_str}")

            if self.constraints:
                lines.append("%constraints")
                for c in self.constraints:
                    c_clean = c.strip()
                    if c_clean:
                        lines.append(f"  {c_clean}")
                lines.append("end")

            if self.equivalences:
                for eq in self.equivalences:
                    eq_clean = eq.strip()
                    if eq_clean:
                        lines.append(f"%equivalence {eq_clean}")
        elif self.run_type == "TS":
            lines.append(f"%ts_mode {self.ts_mode}")
            if self.ts_mode == "irc":
                lines.append(f"%irc_direction {self.irc_direction}")
            elif self.ts_mode == "neb":
                if self.neb_product:
                    lines.append(f"%neb_product {self.neb_product}")
                lines.append(f"%neb_images {self.neb_images}")
                if self.neb_optimize_ends:
                    lines.append("%neb_optimize_ends true")
        elif self.run_type == "SCAN" or self.scan_enabled:
            if self.scan_target:
                lines.append(f"%scan {self.scan_type} {self.scan_target} {self.scan_start} {self.scan_end} {self.scan_steps}")

        # Properties
        if self.molden:
            lines.append("%molden true")
        if self.cube:
            lines.append("%cube true")
            lines.append(f"%cube_points {self.cube_points}")
        if self.chelpg:
            lines.append("%chelpg true")
        if self.density_map:
            type_str = f" {self.density_map_type}" if self.density_map_type else " true"
            lines.append(f"%density_map{type_str}")
            if self.density_map_plane and self.density_map_plane != "AUTO":
                lines.append(f"%density_map_plane {self.density_map_plane}")

        return lines


class BrorsenInputGenerator:
    """Manages the full multi-stage calculation pipeline and outputs .inp text."""

    def __init__(self):
        self.stages: List[BrorsenStage] = [BrorsenStage(1)]

    def add_stage(self) -> BrorsenStage:
        new_num = len(self.stages) + 1
        last = self.stages[-1]
        st = last.copy(new_num)
        st.has_own_geometry = False  # Inherit geometry by default
        self.stages.append(st)
        return st

    def remove_stage(self, index: int) -> bool:
        if len(self.stages) <= 1 or index < 0 or index >= len(self.stages):
            return False
        self.stages.pop(index)
        for i, s in enumerate(self.stages):
            s.number = i + 1
            if not s.jobname or s.jobname.startswith("Stage_"):
                s.jobname = f"Stage_{i + 1}"
        return True

    def generate_input_text(
        self,
        symbols: List[str],
        coords: np.ndarray,
        zatoms: Optional[List[Any]] = None,
        variables: Optional[Dict[str, float]] = None,
    ) -> str:
        """Generate the complete newBRORSEN input string."""
        blocks = []
        for i, st in enumerate(self.stages):
            lines = []
            if i > 0:
                lines.append("")
                lines.append("# " + "=" * 50)
                lines.append(f"# Stage {st.number}: {st.jobname}")
                lines.append("# " + "=" * 50)
                lines.append("%next")
            
            lines.append(st.build_bang_line())
            lines.extend(st.generate_directives())

            # Geometry block
            if st.has_own_geometry or i == 0:
                lines.append("")
                if st.geometry_format == "zmat" and zatoms:
                    from agymolden.zmatrix import format_zmatrix_text
                    lines.append(f"* zmat {st.charge} {st.mult}")
                    zmat_body = format_zmatrix_text(zatoms, variables).strip()
                    if zmat_body:
                        lines.append(zmat_body)
                    lines.append("*")
                else:
                    lines.append(f"* xyz {st.charge} {st.mult}")
                    for sym, (x, y, z) in zip(symbols, coords):
                        lines.append(f"{sym:<3} {x:12.6f} {y:12.6f} {z:12.6f}")
                    lines.append("*")

            blocks.append("\n".join(lines))

        return "\n".join(blocks) + "\n"

    def load_from_text(self, text: str) -> Tuple[List[BrorsenStage], Optional[Dict[str, Any]]]:
        """
        Parse .inp file text into BrorsenStage list and extracted geometry.
        Returns (stages, geom_info).
        """
        # Split on %next, %newjob, or tolerant Latin keyboard Shift+5 mistype '5next' / '5newjob'
        raw_blocks = re.split(
            r'^\s*(?:%(?:next|newjob)\b|5(?:next|newjob)\b|#\s*Stage\s+\d+\s*:[^\n]*)',
            text,
            flags=re.MULTILINE | re.IGNORECASE,
        )
        parsed_stages: List[BrorsenStage] = []
        first_geometry = None

        run_codes = [x[0] for x in RUN_TYPES]
        method_codes = [x[0] for x in METHODS]
        sector_codes = [x[0] for x in SECTORS]
        integral_codes = [x[0] for x in INTEGRAL_MODES]
        coord_codes = [x[0] for x in COORD_UNITS]

        for idx, blk in enumerate(raw_blocks):
            lines = blk.strip().splitlines()
            if not lines:
                continue

            st = BrorsenStage(number=len(parsed_stages) + 1)
            st.jobname = f"Stage_{st.number}"

            in_geom = False
            geom_lines = []
            geom_format = "xyz"
            geom_charge = 0
            geom_mult = 1
            has_geom = False

            for line in lines:
                line_s = line.strip()
                if not line_s or line_s.startswith("#"):
                    continue

                # Geometry block delimiters
                if line_s.startswith("*"):
                    if not in_geom:
                        parts = line_s[1:].strip().split()
                        if parts:
                            geom_format = "zmat" if parts[0].lower() == "zmat" else "xyz"
                        if len(parts) > 1:
                            try:
                                geom_charge = int(parts[1])
                                st.charge = geom_charge
                            except ValueError:
                                pass
                        if len(parts) > 2:
                            try:
                                geom_mult = int(parts[2])
                                st.mult = geom_mult
                            except ValueError:
                                pass
                        in_geom = True
                        geom_lines = []
                        has_geom = True
                    else:
                        in_geom = False
                        st.has_own_geometry = True
                        st.geometry_format = geom_format
                        geom_data = {
                            "format": geom_format,
                            "charge": geom_charge,
                            "mult": geom_mult,
                            "lines": geom_lines,
                        }
                        if first_geometry is None:
                            first_geometry = geom_data
                    continue

                if in_geom:
                    geom_lines.append(line_s)
                    continue

                # Bang line
                if line_s.startswith("!"):
                    tokens = line_s[1:].strip().split()
                    for t in tokens:
                        up = t.upper()
                        if up in run_codes:
                            st.run_type = up
                        elif up in sector_codes:
                            st.sector = up
                        elif up in integral_codes:
                            st.integrals = up
                        elif up in [c.upper() for c in coord_codes]:
                            for c in coord_codes:
                                if c.upper() == up:
                                    st.coords_unit = c
                                    break
                        else:
                            for m in method_codes:
                                if m.upper() == up or m.replace("(T)", "(t)").upper() == up:
                                    st.method = m
                                    break
                    continue

                # Directives %key value
                if line_s.startswith("%"):
                    parts = line_s[1:].split(None, 1)
                    k = parts[0].lower()
                    v = parts[1].strip() if len(parts) > 1 else ""

                    if k == "jobname":
                        st.jobname = v
                    elif k == "charge":
                        try:
                            st.charge = int(v)
                        except ValueError:
                            pass
                    elif k == "mult":
                        try:
                            st.mult = int(v)
                        except ValueError:
                            pass
                    elif k == "nprocs":
                        try:
                            st.nprocs = int(v)
                        except ValueError:
                            pass
                    elif k == "memory":
                        try:
                            st.memory = int(v)
                        except ValueError:
                            pass
                    elif k == "basis":
                        st.basis = v
                    elif k == "frozen":
                        st.frozen = v
                    elif k == "xc":
                        st.xc = v
                    elif k == "n_pos":
                        try:
                            st.n_pos = int(v)
                        except ValueError:
                            pass
                    elif k == "pos_basis":
                        st.pos_basis = v
                    elif k == "psx_set":
                        st.psx_set = v
                    elif k == "pos_atom":
                        st.pos_atom = v
                    elif k == "pos_mult":
                        try:
                            st.pos_mult = int(v)
                        except ValueError:
                            pass
                    elif k == "pos_mass":
                        st.pos_mass = v
                    elif k == "quantum_site":
                        st.quantum_site = v
                    elif k == "nuc_basis":
                        st.nuc_basis = v
                    elif k == "quantum_nuc":
                        st.quantum_nuc = v
                    elif k == "cneo":
                        st.cneo = v.lower() in ("true", "1", "yes")
                    elif k == "cneo_variant":
                        st.cneo_variant = v
                    elif k == "epc":
                        st.epc = v
                    elif k == "opt_engine":
                        st.opt_engine = v.upper()
                    elif k == "opt_max_cycles":
                        try:
                            st.opt_max_cycles = int(v)
                        except ValueError:
                            pass
                    elif k == "ts_mode":
                        st.ts_mode = v.lower()
                    elif k == "irc_direction":
                        st.irc_direction = v
                    elif k == "scan":
                        sparts = v.split()
                        if len(sparts) >= 5:
                            st.scan_enabled = True
                            st.scan_type = sparts[0]
                            st.scan_target = sparts[1]
                            try:
                                st.scan_start = float(sparts[2])
                                st.scan_end = float(sparts[3])
                                st.scan_steps = int(sparts[4])
                            except ValueError:
                                pass
                    elif k == "cube":
                        st.cube = v.lower() in ("true", "1", "yes")
                    elif k == "cube_points":
                        try:
                            st.cube_points = int(v)
                        except ValueError:
                            pass
                    elif k == "molden":
                        st.molden = v.lower() in ("true", "1", "yes")
                    elif k == "chelpg":
                        st.chelpg = v.lower() in ("true", "1", "yes")
                    elif k == "density_map":
                        st.density_map = v.lower() not in ("false", "0", "no")
                        if v.lower() in ("esp", "electron", "positron", "both"):
                            st.density_map_type = v.lower()
                    elif k == "density_map_plane":
                        st.density_map_plane = v.upper()
                    elif k == "verbose":
                        try:
                            st.verbose = int(v)
                        except ValueError:
                            pass

            st.has_own_geometry = has_geom
            parsed_stages.append(st)

        if parsed_stages:
            self.stages = parsed_stages

        return self.stages, first_geometry
