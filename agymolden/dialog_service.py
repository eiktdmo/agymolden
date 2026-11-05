"""
Dialog and Notification Service for AgyMolden.

Decouples GUI dialog presentations (QMessageBox, confirmations, prompts) from core logic,
enabling headless automated testing and scriptable batch workflows without GUI blocking.
"""

import logging
from typing import Optional, List, Dict, Any
from PyQt6 import QtWidgets

log = logging.getLogger(__name__)


class DialogService:
    """Service for displaying user dialogs and notifications with headless intercept capabilities."""

    def __init__(self):
        self._headless_mode = False
        self._recorded_messages: List[Dict[str, Any]] = []
        self._mock_confirm_responses: List[bool] = []

    def set_headless(self, enabled: bool = True):
        """Enable or disable headless mode (suppresses GUI popups for testing)."""
        self._headless_mode = enabled

    def clear_history(self):
        """Clear recorded messages."""
        self._recorded_messages.clear()
        self._mock_confirm_responses.clear()

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._recorded_messages)

    def queue_confirm_response(self, response: bool):
        """Queue mock boolean response for subsequent confirm() calls in testing."""
        self._mock_confirm_responses.append(response)

    def info(self, parent: Optional[QtWidgets.QWidget], title: str, message: str) -> None:
        """Display an informational dialog."""
        self._recorded_messages.append({"level": "info", "title": title, "message": message})
        log.info("[%s] %s", title, message)
        if self._headless_mode or QtWidgets.QApplication.instance() is None:
            return
        QtWidgets.QMessageBox.information(parent, title, message)

    def warning(self, parent: Optional[QtWidgets.QWidget], title: str, message: str) -> None:
        """Display a warning dialog."""
        self._recorded_messages.append({"level": "warning", "title": title, "message": message})
        log.warning("[%s] %s", title, message)
        if self._headless_mode or QtWidgets.QApplication.instance() is None:
            return
        QtWidgets.QMessageBox.warning(parent, title, message)

    def error(self, parent: Optional[QtWidgets.QWidget], title: str, message: str) -> None:
        """Display an error/critical dialog."""
        self._recorded_messages.append({"level": "error", "title": title, "message": message})
        log.error("[%s] %s", title, message)
        if self._headless_mode or QtWidgets.QApplication.instance() is None:
            return
        QtWidgets.QMessageBox.critical(parent, title, message)

    def confirm(
        self,
        parent: Optional[QtWidgets.QWidget],
        title: str,
        message: str,
        default_yes: bool = True,
    ) -> bool:
        """Prompt user with Yes/No question and return boolean."""
        self._recorded_messages.append({"level": "confirm", "title": title, "message": message})
        if self._headless_mode or QtWidgets.QApplication.instance() is None:
            if self._mock_confirm_responses:
                return self._mock_confirm_responses.pop(0)
            return default_yes

        buttons = QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        default_btn = (
            QtWidgets.QMessageBox.StandardButton.Yes
            if default_yes
            else QtWidgets.QMessageBox.StandardButton.No
        )
        resp = QtWidgets.QMessageBox.question(parent, title, message, buttons, default_btn)
        return resp == QtWidgets.QMessageBox.StandardButton.Yes


# Global singleton instance
dialog_service = DialogService()
