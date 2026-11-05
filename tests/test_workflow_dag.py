import json
import os
import pytest
import numpy as np

from agymolden.dialog_workflow_dag import DAGStage, VisualWorkflowDAGDialog, DAG_TEMPLATES


def test_dag_stage_config_conversion():
    stage = DAGStage(
        name="High-Level Single Point",
        job_type="SP",
        description="DLPNO-CCSD(T) Benchmark",
        method="DLPNO-CCSD(T)",
        basis="def2-QZVPP",
        solvent="Water",
        extra_keywords="TightPNO Grid5",
    )
    cfg = stage.to_config()
    assert cfg.name == "High-Level Single Point"
    assert cfg.job_type == "SP"
    assert cfg.extra_params["method"] == "DLPNO-CCSD(T)"
    assert cfg.extra_params["basis"] == "def2-QZVPP"
    assert cfg.extra_params["solvent"] == "Water"
    assert cfg.extra_params["extra_keywords"] == "TightPNO Grid5"


def test_visual_workflow_dag_dialog_init(qapp, tmp_path):
    symbols = ["C", "H", "H", "H", "H"]
    coords = np.zeros((5, 3))
    dlg = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))

    assert len(dlg.stages) == 2  # Opt -> Freq default
    assert dlg.canvas.stages is dlg.stages
    assert dlg.txt_st_name.text() == "Geometry Optimization"
    assert dlg.combo_st_type.currentText() == "Opt"


def test_dag_add_remove_reorder_stages(qapp, tmp_path):
    symbols = ["H", "H"]
    coords = np.zeros((2, 3))
    dlg = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))

    # Add stage
    dlg._on_add_stage()
    assert len(dlg.stages) == 3
    assert dlg.selected_idx == 2
    assert dlg.stages[2].job_type == "SP"

    # Edit third stage
    dlg.txt_st_name.setText("UV-Vis Spectrum")
    dlg.combo_st_type.setCurrentText("TD-DFT")
    dlg.txt_st_method.setText("wB97X-D3")
    assert dlg.stages[2].name == "UV-Vis Spectrum"
    assert dlg.stages[2].job_type == "TD-DFT"
    assert dlg.stages[2].method == "wB97X-D3"

    # Move up
    dlg._on_move_stage_up()
    assert dlg.selected_idx == 1
    assert dlg.stages[1].name == "UV-Vis Spectrum"

    # Move down
    dlg._on_move_stage_down()
    assert dlg.selected_idx == 2
    assert dlg.stages[2].name == "UV-Vis Spectrum"

    # Remove stage
    dlg._on_remove_stage()
    assert len(dlg.stages) == 2


def test_dag_recipe_save_and_load(qapp, tmp_path):
    symbols = ["O", "H", "H"]
    coords = np.zeros((3, 3))
    dlg = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))

    # Select TS template
    dlg.combo_presets.setCurrentText("TS Opt → Freq (1 Imag) → IRC (Path)")
    assert len(dlg.stages) == 3
    assert dlg.stages[0].job_type == "TS"
    assert dlg.stages[2].job_type == "IRC"

    # Save recipe to JSON
    recipe_file = str(tmp_path / "ts_recipe.json")
    data = {
        "version": "1.0",
        "recipe_name": "TS Opt → Freq (1 Imag) → IRC (Path)",
        "engine": "ORCA",
        "stages": [
            {"name": st.name, "job_type": st.job_type, "description": st.description,
             "method": st.method, "basis": st.basis, "solvent": st.solvent, "extra_keywords": st.extra_keywords}
            for st in dlg.stages
        ],
    }
    with open(recipe_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    # Load in new dialog
    dlg2 = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))
    assert len(dlg2.stages) == 2  # default

    with open(recipe_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    dlg2.stages = [
        DAGStage(s["name"], s["job_type"], s.get("description", ""), s.get("method", ""), s.get("basis", ""))
        for s in loaded["stages"]
    ]
    dlg2.canvas.set_stages(dlg2.stages, 0)
    assert len(dlg2.stages) == 3
    assert dlg2.stages[2].job_type == "IRC"


def test_dag_execution_state_transitions(qapp, tmp_path):
    symbols = ["O", "H", "H"]
    coords = np.zeros((3, 3))
    dlg = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))

    # Simulate stage 0 started
    dlg._on_stage_started(0, "Geometry Optimization")
    assert dlg.stages[0].status == "running"
    assert "Running Stage 1" in dlg.lbl_status.text()

    # Simulate stage 0 finished with success
    dlg._on_stage_finished(0, "Geometry Optimization", True, "Stationary point found.")
    assert dlg.stages[0].status == "success"
    assert dlg.stages[0].status_msg == "Stationary point found."

    # Simulate stage 1 started & failed
    dlg._on_stage_started(1, "Harmonic Frequencies")
    assert dlg.stages[1].status == "running"
    dlg._on_stage_finished(1, "Harmonic Frequencies", False, "SCF convergence failed.")
    assert dlg.stages[1].status == "failed"

    # Pipeline finished
    dlg._on_pipeline_finished(False, "Pipeline failed at stage 2")
    assert dlg.btn_run.isEnabled()
    assert not dlg.btn_abort.isEnabled()
    assert "Pipeline failed" in dlg.lbl_status.text()


def test_dag_adopt_geometry_signal(qapp, tmp_path):
    symbols = ["O", "H", "H"]
    coords = np.zeros((3, 3))
    dlg = VisualWorkflowDAGDialog(None, code_key="orca", symbols=symbols, coords=coords, work_dir=str(tmp_path))

    # Mock pipeline with new coords
    class MockPipeline:
        current_coords = np.ones((3, 3)) * 1.5

    dlg.pipeline = MockPipeline()
    adopted = []
    dlg.pipelineAdoptRequested.connect(lambda s, c, l: adopted.append((s, c, l)))

    dlg._on_adopt_geometry()
    assert len(adopted) == 1
    assert np.allclose(adopted[0][1], 1.5)
