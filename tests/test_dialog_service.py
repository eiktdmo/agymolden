"""Tests for DialogService abstraction."""

import pytest
from agymolden.dialog_service import DialogService


def test_dialog_service_headless():
    svc = DialogService()
    svc.set_headless(True)

    svc.info(None, "Information", "Calculation completed successfully.")
    svc.warning(None, "Warning", "High force detected.")
    err_exc = ValueError("Invalid matrix dimensions")
    svc.error(None, "Error", "File not found.", exc=err_exc)

    assert len(svc.history) == 3
    assert svc.history[0]["level"] == "info"
    assert svc.history[1]["level"] == "warning"
    assert svc.history[2]["level"] == "error"
    assert svc.history[2]["exc"] is err_exc

    # Test confirm queueing
    svc.queue_confirm_response(False)
    assert svc.confirm(None, "Overwrite?", "File exists.") is False

    svc.queue_confirm_response(True)
    assert svc.confirm(None, "Overwrite?", "File exists.") is True
