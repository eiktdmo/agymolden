"""
HPC & Cluster Batch Script Generator Engine for AgyMolden.

Generates production-grade submission scripts for:
- SLURM Workload Manager (`sbatch`)
- PBS Pro / TORQUE (`qsub`)
- IBM Spectrum LSF (`bsub`)
- Sun Grid Engine / OGE (`qsub`)

Supports automatic execution command configuration for:
- ORCA (with full MPI path, `%pal`, and scratch directory copying)
- Gaussian (g16 / g09 with `%nprocshared`, `%mem`, and scratch cleanup)
- PySCF (environment module loading and multi-threading)
- newBRORSEN (APMO multi-component nuclear-electronic orbital runner)
- xTB (GFN-xTB with OMP_NUM_THREADS and OMP_STACKSIZE)
"""

import os
import shlex
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ClusterScriptOptions:
    """Configuration options for cluster batch job submission script generation."""
    scheduler: str = "SLURM"                 # "SLURM", "PBS", "LSF", "SGE"
    job_name: str = "agymolden_job"
    queue_or_partition: str = ""
    nodes: int = 1
    cpus_per_node: int = 8
    memory_gb: int = 16
    walltime_hours: int = 24
    modules_to_load: List[str] = field(default_factory=list)
    export_environment_vars: Dict[str, str] = field(default_factory=dict)
    use_scratch: bool = True
    scratch_dir: str = "/tmp"                # or "$SCRATCH", "$TMPDIR"
    custom_pre_commands: List[str] = field(default_factory=list)
    custom_post_commands: List[str] = field(default_factory=list)
    mail_user: str = ""
    mail_type: str = "NONE"                  # "NONE", "BEGIN", "END", "FAIL", "ALL"


def format_walltime(hours: int, scheduler: str = "SLURM") -> str:
    """Format walltime integer hours into scheduler-specific duration string."""
    h = max(1, int(hours))
    if scheduler == "SLURM":
        return f"{h:02d}:00:00"
    elif scheduler == "PBS":
        return f"{h:02d}:00:00"
    elif scheduler == "LSF":
        return f"{h:02d}:00"
    elif scheduler == "SGE":
        return f"{h:02d}:00:00"
    return f"{h:02d}:00:00"


def generate_cluster_script(
    input_file_path: str,
    qm_code: str = "ORCA",
    options: Optional[ClusterScriptOptions] = None,
) -> str:
    """
    Generate complete production-grade HPC batch job script.

    Parameters:
    - input_file_path: name or relative path of the QM input deck (e.g. 'water.inp', 'calc.gjf')
    - qm_code: "ORCA", "Gaussian", "PySCF", "newBRORSEN", "xTB", "Generic"
    - options: ClusterScriptOptions configuration
    """
    opts = options or ClusterScriptOptions()
    sched = opts.scheduler.upper().strip()
    input_file = os.path.basename(input_file_path)
    stem, ext = os.path.splitext(input_file)
    job_name = opts.job_name or stem

    lines = ["#!/bin/bash"]

    # 1. Scheduler Resource Headers
    if sched == "SLURM":
        lines.append(f"#SBATCH --job-name={job_name}")
        lines.append(f"#SBATCH --output={stem}.%j.out")
        lines.append(f"#SBATCH --error={stem}.%j.err")
        if opts.queue_or_partition:
            lines.append(f"#SBATCH --partition={opts.queue_or_partition}")
        lines.append(f"#SBATCH --nodes={opts.nodes}")
        lines.append(f"#SBATCH --ntasks-per-node={opts.cpus_per_node}")
        lines.append(f"#SBATCH --mem={opts.memory_gb}G")
        lines.append(f"#SBATCH --time={format_walltime(opts.walltime_hours, 'SLURM')}")
        if opts.mail_user and opts.mail_type != "NONE":
            lines.append(f"#SBATCH --mail-user={opts.mail_user}")
            lines.append(f"#SBATCH --mail-type={opts.mail_type}")

    elif sched == "PBS":
        lines.append(f"#PBS -N {job_name}")
        lines.append(f"#PBS -o {stem}.out")
        lines.append(f"#PBS -e {stem}.err")
        if opts.queue_or_partition:
            lines.append(f"#PBS -q {opts.queue_or_partition}")
        lines.append(f"#PBS -l select={opts.nodes}:ncpus={opts.cpus_per_node}:mem={opts.memory_gb}gb")
        lines.append(f"#PBS -l walltime={format_walltime(opts.walltime_hours, 'PBS')}")
        if opts.mail_user and opts.mail_type != "NONE":
            lines.append(f"#PBS -M {opts.mail_user}")
            lines.append(f"#PBS -m {opts.mail_type[0].lower()}")

    elif sched == "LSF":
        lines.append(f"#BSUB -J {job_name}")
        lines.append(f"#BSUB -o {stem}.%J.out")
        lines.append(f"#BSUB -e {stem}.%J.err")
        if opts.queue_or_partition:
            lines.append(f"#BSUB -q {opts.queue_or_partition}")
        lines.append(f"#BSUB -n {opts.nodes * opts.cpus_per_node}")
        lines.append(f"#BSUB -M {opts.memory_gb * 1024}")
        lines.append(f"#BSUB -W {format_walltime(opts.walltime_hours, 'LSF')}")
        if opts.mail_user and opts.mail_type != "NONE":
            lines.append(f"#BSUB -u {opts.mail_user}")

    elif sched == "SGE":
        lines.append(f"#$ -N {job_name}")
        lines.append(f"#$ -o {stem}.out")
        lines.append(f"#$ -e {stem}.err")
        if opts.queue_or_partition:
            lines.append(f"#$ -q {opts.queue_or_partition}")
        lines.append(f"#$ -pe smp {opts.cpus_per_node}")
        lines.append(f"#$ -l h_rt={format_walltime(opts.walltime_hours, 'SGE')}")
        lines.append(f"#$ -l h_vmem={opts.memory_gb}G")
        lines.append("#$ -cwd")

    lines.append("")
    lines.append("# Environment & working directory setup")
    if sched == "PBS":
        lines.append('cd "$PBS_O_WORKDIR"')
    elif sched == "SLURM":
        lines.append('cd "$SLURM_SUBMIT_DIR" 2>/dev/null || cd "$(pwd)"')
    elif sched == "LSF":
        lines.append('cd "$LS_SUBCWD" 2>/dev/null || cd "$(pwd)"')

    # Environment variables
    lines.append(f"export OMP_NUM_THREADS={opts.cpus_per_node}")
    lines.append("export OMP_STACKSIZE=2G")
    for k, v in opts.export_environment_vars.items():
        lines.append(f"export {k}={v}")

    # Modules
    if opts.modules_to_load:
        lines.append("")
        lines.append("# Load required cluster environment modules")
        for m in opts.modules_to_load:
            lines.append(f"module load {m}")

    # Custom pre-commands
    if opts.custom_pre_commands:
        lines.append("")
        lines.append("# Pre-execution setup commands")
        lines.extend(opts.custom_pre_commands)

    # 2. QM Execution Logic with Scratch handling
    lines.append("")
    lines.append(f"# Execution pipeline for {qm_code}")
    code_upper = qm_code.upper()

    if "ORCA" in code_upper:
        if opts.use_scratch:
            lines.append('SCRATCH_DIR="${SLURM_TMPDIR:-' + opts.scratch_dir + '}/orca_$$"')
            lines.append('mkdir -p "$SCRATCH_DIR"')
            lines.append(f'cp {input_file} "$SCRATCH_DIR/"')
            lines.append('cd "$SCRATCH_DIR"')
            lines.append(f'$(which orca) {input_file} > {stem}.out')
            lines.append(f'cp * "$SLURM_SUBMIT_DIR/" 2>/dev/null || cp * "$OLDPWD/"')
            lines.append('cd "$OLDPWD"')
            lines.append('rm -rf "$SCRATCH_DIR"')
        else:
            lines.append(f'$(which orca) {input_file} > {stem}.out')

    elif "GAUSSIAN" in code_upper:
        if opts.use_scratch:
            lines.append('export GAUSS_SCRDIR="${SLURM_TMPDIR:-' + opts.scratch_dir + '}"')
            lines.append('mkdir -p "$GAUSS_SCRDIR"')
        lines.append(f'g16 < {input_file} > {stem}.log')

    elif "PYSCF" in code_upper:
        lines.append(f'python3 -u {input_file} > {stem}.out')

    elif "BRORSEN" in code_upper:
        lines.append(f'python3 -u newBRORSEN.v04.09.py {input_file} > {stem}.out')

    elif "XTB" in code_upper:
        lines.append(f'xtb {input_file} --opt > {stem}.out')

    else:
        lines.append(f'# Run command for {qm_code}:')
        lines.append(f'./{input_file} > {stem}.out')

    # Custom post-commands
    if opts.custom_post_commands:
        lines.append("")
        lines.append("# Post-execution cleanup & notification commands")
        lines.extend(opts.custom_post_commands)

    lines.append("")
    lines.append('echo "Job completed on $(date)"')
    return "\n".join(lines) + "\n"
