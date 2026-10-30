"""
Unit tests for Phase 2 Deliverables:
1. SLURM script generation & RemoteHostConfig SLURM options.
2. Trajectory Player: First/Last buttons, Play/Pause, Loop/Bounce/Once modes, and Speed intervals.
3. Background Volumetric Meshing: _CubeLoaderThread and Progress reporting.
"""

import os
import tempfile
import numpy as np
import pytest
from PyQt6 import QtCore, QtWidgets

from agymolden.calc_runner.remote_ssh import RemoteHostConfig, generate_slurm_script
from agymolden.volumetric import VolumetricData, generate_orbital_surfaces, extract_isosurface
from agymolden.dialog_volumetric import _CubeLoaderThread


def test_slurm_script_generation():
    """Verify SLURM sbatch script generates correct directives."""
    script = generate_slurm_script(
        job_name="ch4_opt",
        input_file="/local/path/CH4.inp",
        remote_python="/opt/conda/bin/python",
        remote_script="newBRORSEN.v04.09.py",
        extra_flags=["--resume"],
        remote_workdir="~/scratch_dir",
        cpus=8,
        mem="16GB",
        walltime="48:00:00",
        partition="chem_cluster",
        nice_prefix="nice -n 10 ",
    )

    assert "#!/bin/bash" in script
    assert "#SBATCH --job-name=ch4_opt" in script
    assert "#SBATCH --output=CH4.out" in script
    assert "#SBATCH --error=CH4.err" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=16GB" in script
    assert "#SBATCH --time=48:00:00" in script
    assert "#SBATCH --partition=chem_cluster" in script
    assert "cd scratch_dir" in script
    assert "nice -n 10 /opt/conda/bin/python -u newBRORSEN.v04.09.py --resume CH4.inp" in script


def test_remote_host_config_slurm_defaults():
    """Verify RemoteHostConfig defaults and SLURM fields."""
    cfg = RemoteHostConfig(
        hostname="cluster.unal.edu.co",
        use_slurm=True,
        slurm_partition="general",
        slurm_cpus=16,
        slurm_mem="32GB",
        slurm_time="12:00:00",
    )
    assert cfg.use_slurm is True
    assert cfg.slurm_partition == "general"
    assert cfg.slurm_cpus == 16
    assert cfg.slurm_mem == "32GB"
    assert cfg.slurm_time == "12:00:00"


def test_trajectory_player_modes_and_controls(qapp):
    """Verify Trajectory player first/last navigation, bounce logic, and speed timer."""
    from agymolden.main_window import MainWindow

    mw = MainWindow()
    mw.trajectory_coords = [
        np.zeros((3, 3)),
        np.ones((3, 3)) * 0.5,
        np.ones((3, 3)) * 1.0,
        np.ones((3, 3)) * 1.5,
    ]
    mw.trajectory_energies = [-76.0, -76.1, -76.2, -76.3]
    mw.slider_trajectory.setRange(0, 3)
    mw.slider_trajectory.setValue(0)

    # 1. Test Next and Prev
    mw._on_traj_next_step()
    assert mw.slider_trajectory.value() == 1
    mw._on_traj_prev_step()
    assert mw.slider_trajectory.value() == 0

    # 2. Test First and Last
    mw._on_traj_last_step()
    assert mw.slider_trajectory.value() == 3
    mw._on_traj_first_step()
    assert mw.slider_trajectory.value() == 0

    # 3. Test Loop Mode tick
    mw.combo_traj_mode.setCurrentText("Loop 🔁")
    mw.slider_trajectory.setValue(3)
    mw._on_traj_timer_tick()
    assert mw.slider_trajectory.value() == 0  # Wrapped to start

    # 4. Test Bounce Mode tick
    mw.combo_traj_mode.setCurrentText("Bounce ⮂")
    mw.slider_trajectory.setValue(3)
    mw._traj_direction = 1
    mw._on_traj_timer_tick()
    assert mw.slider_trajectory.value() == 3
    assert mw._traj_direction == -1
    mw._on_traj_timer_tick()
    assert mw.slider_trajectory.value() == 2

    # 5. Test Speed changes
    mw.combo_traj_speed.setCurrentIndex(2)  # 2.0x -> 60ms
    mw._on_traj_speed_changed(2)
    assert hasattr(mw, "_traj_timer")
    assert mw._traj_timer.interval() == 60

    mw.close()


def test_cube_loader_thread(qapp):
    """Verify _CubeLoaderThread correctly parses cube file in worker thread."""
    cube_data = """Test Cube
Second Line
  1  0.0 0.0 0.0
  2  1.0 0.0 0.0
  2  0.0 1.0 0.0
  2  0.0 0.0 1.0
  1  1.0 0.0 0.0 0.0
 0.1 0.2 0.3 0.4
 0.5 0.6 0.7 0.8
"""
    with tempfile.NamedTemporaryFile("w", suffix=".cube", delete=False) as f:
        f.write(cube_data)
        cpath = f.name

    try:
        thread = _CubeLoaderThread(cpath)
        captured = []
        thread.loaded.connect(lambda vol, p: captured.append((vol, p)))
        thread.run()  # Run worker payload

        assert len(captured) == 1
        vol, path = captured[0]
        assert vol.voxels_x == 2
        assert vol.voxels_y == 2
        assert vol.voxels_z == 2
        assert len(vol.symbols) == 1
    finally:
        if os.path.isfile(cpath):
            os.remove(cpath)
