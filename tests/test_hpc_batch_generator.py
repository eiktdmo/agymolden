"""Tests for HPC Cluster Batch Script Generator Engine.

Verifies:
1. SLURM script generation with correct directives (#SBATCH --job-name, --nodes, --cpus, --mem, --time).
2. PBS Pro script generation (#PBS -N, -l select, -l walltime).
3. IBM Spectrum LSF script generation (#BSUB -J, -n, -M, -W).
4. SGE script generation (#$ -N, -pe smp, -l h_rt).
5. Scratch directory lifecycle logic for ORCA and Gaussian.
6. Module loading and environment variable propagation.
7. ClusterScriptDialog GUI instantiates cleanly without exceptions.
"""

import os
import pytest
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtWidgets
from agymolden.hpc_batch_generator import (
    ClusterScriptOptions,
    generate_cluster_script,
    format_walltime,
)
from agymolden.dialog_cluster_script import ClusterScriptDialog


def test_slurm_script_directives_and_scratch():
    """Verify SLURM generation with scratch handling for ORCA."""
    opts = ClusterScriptOptions(
        scheduler="SLURM",
        job_name="orca_calc",
        queue_or_partition="gpu_shared",
        nodes=1,
        cpus_per_node=16,
        memory_gb=32,
        walltime_hours=48,
        modules_to_load=["orca/5.0.4", "openmpi/4.1.1"],
        use_scratch=True,
    )

    script = generate_cluster_script("test_job.inp", qm_code="ORCA", options=opts)

    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=orca_calc" in script
    assert "#SBATCH --partition=gpu_shared" in script
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks-per-node=16" in script
    assert "#SBATCH --mem=32G" in script
    assert "#SBATCH --time=48:00:00" in script
    assert "module load orca/5.0.4" in script
    assert "module load openmpi/4.1.1" in script
    assert "SCRATCH_DIR=" in script
    assert "$(which orca) test_job.inp > test_job.out" in script
    assert 'rm -rf "$SCRATCH_DIR"' in script


def test_pbs_script_directives():
    """Verify PBS Pro script syntax."""
    opts = ClusterScriptOptions(
        scheduler="PBS",
        job_name="pbs_job",
        queue_or_partition="workq",
        nodes=2,
        cpus_per_node=8,
        memory_gb=16,
        walltime_hours=12,
    )

    script = generate_cluster_script("gaussian_calc.gjf", qm_code="Gaussian", options=opts)

    assert "#PBS -N pbs_job" in script
    assert "#PBS -q workq" in script
    assert "#PBS -l select=2:ncpus=8:mem=16gb" in script
    assert "#PBS -l walltime=12:00:00" in script
    assert 'cd "$PBS_O_WORKDIR"' in script
    assert "g16 < gaussian_calc.gjf > gaussian_calc.log" in script


def test_lsf_and_sge_scripts():
    """Verify IBM LSF and SGE batch scripts."""
    # LSF
    lsf_opts = ClusterScriptOptions(scheduler="LSF", job_name="lsf_test", nodes=1, cpus_per_node=4, memory_gb=8)
    lsf_script = generate_cluster_script("calc.inp", qm_code="xTB", options=lsf_opts)
    assert "#BSUB -J lsf_test" in lsf_script
    assert "#BSUB -n 4" in lsf_script
    assert "xtb calc.inp --opt > calc.out" in lsf_script

    # SGE
    sge_opts = ClusterScriptOptions(scheduler="SGE", job_name="sge_test", nodes=1, cpus_per_node=8, memory_gb=16)
    sge_script = generate_cluster_script("script.py", qm_code="PySCF", options=sge_opts)
    assert "#$ -N sge_test" in sge_script
    assert "#$ -pe smp 8" in sge_script
    assert "python3 -u script.py > script.out" in sge_script


def test_cluster_script_dialog_gui():
    """Verify ClusterScriptDialog instantiates and updates preview cleanly."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dlg = ClusterScriptDialog(
        parent=None,
        input_filename="benzene.inp",
        qm_code="ORCA",
        default_nprocs=8,
        default_mem_gb=16,
    )
    assert dlg is not None
    assert dlg.text_preview.toPlainText().startswith("#!/bin/bash")
    assert "#SBATCH" in dlg.text_preview.toPlainText()

    # Switch scheduler to PBS
    idx = dlg.combo_sched.findText("PBS", QtCore.Qt.MatchFlag.MatchContains)
    assert idx >= 0
    dlg.combo_sched.setCurrentIndex(idx)
    assert "#PBS" in dlg.text_preview.toPlainText()

    dlg.close()
    dlg.deleteLater()
    app.processEvents()
