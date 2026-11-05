"""Test Chained Workflow Pipeline Engine and Dialog."""

import os
import numpy as np
import pytest

from agymolden.calc_runner.pipeline import (
    ChainedWorkflowPipeline,
    WORKFLOW_TEMPLATES,
    WorkflowStageConfig,
)
from agymolden.calc_runner.dialog_pipeline import WorkflowPipelineDialog
from agymolden.qm_input_generator import QMCalculationParameters


def test_workflow_templates_structure():
    assert "Opt → Freq (Minimum Guarantee)" in WORKFLOW_TEMPLATES
    assert "Opt → TD-DFT (Absorption Spectrum)" in WORKFLOW_TEMPLATES
    assert "TS Opt → Freq (1 Imag) → IRC" in WORKFLOW_TEMPLATES

    stages = WORKFLOW_TEMPLATES["Opt → Freq (Minimum Guarantee)"]
    assert len(stages) == 2
    assert stages[0].job_type == "Opt"
    assert stages[1].job_type == "Freq"


def test_workflow_pipeline_dialog_headless(qapp, tmp_path):
    syms = ["O", "H", "H"]
    coords = np.array([
        [0.0, 0.0, 0.1173],
        [0.0, 0.7572, -0.4692],
        [0.0, -0.7572, -0.4692],
    ], dtype=np.float64)
    params = QMCalculationParameters(method="B3LYP", basis="def2-SVP", memory_gb=2, n_procs=1)

    dlg = WorkflowPipelineDialog(
        None,
        code_key="orca",
        symbols=syms,
        coords=coords,
        base_params=params,
        work_dir=str(tmp_path),
    )

    assert dlg.combo_recipe.count() >= 3
    assert dlg.list_stages.count() == 2

    # Switch recipe to TS -> Freq -> IRC
    dlg.combo_recipe.setCurrentText("TS Opt → Freq (1 Imag) → IRC")
    assert dlg.list_stages.count() == 3

    # Switch recipe to Opt -> TD-DFT
    dlg.combo_recipe.setCurrentText("Opt → TD-DFT (Absorption Spectrum)")
    assert dlg.list_stages.count() == 2

    dlg.close()
