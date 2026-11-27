"""
GUI Dialog for newBRORSEN calculations within AGYMolden.
Supports multi-stage pipelines, reactive options, and direct input generation.
"""

import logging
import os
from typing import List, Optional, Dict, Any
import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

log = logging.getLogger(__name__)

from agymolden.zmatrix import ZAtom
from agymolden.elements import ELEMENT_DATA

from .schema import (
    RUN_TYPES, METHODS, SECTORS, INTEGRAL_MODES, COORD_UNITS,
    OPT_ENGINES, TS_MODES, ELECTRONIC_BASES, POSITRON_BASES,
    PROTON_BASES, DFT_FUNCTIONALS, EPC_FUNCTIONALS, PARTICLE_MASSES
)
from .generator import BrorsenStage, BrorsenInputGenerator
from .dialog_basis import SmartBasisDialog


class KeymapPlainTextEdit(QtWidgets.QPlainTextEdit):
    """Text editor with Vim and Emacs keybinding support for direct editing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.keymap_mode = "plain"  # "plain", "vim", "emacs"
        self.vim_mode = "insert"    # "normal", "insert", "visual", "visual_line"
        self.mode_label: Optional[QtWidgets.QLabel] = None
        self.kill_ring: List[str] = []
        self.mark_position: Optional[int] = None
        self.emacs_prefix: Optional[str] = None  # None, "C-x", "C-c"
        self.vim_prefix: Optional[str] = None   # None, "d", "y", "c", "g"
        self.vim_yank_buffer: str = ""
        self.visual_start_pos: Optional[int] = None

    def set_keymap_mode(self, mode: str):
        self.keymap_mode = mode.lower()
        self.emacs_prefix = None
        self.mark_position = None
        self.vim_prefix = None
        self.visual_start_pos = None
        if self.keymap_mode == "vim":
            self.vim_mode = "normal"
            self.setCursorWidth(8)
        else:
            self.setCursorWidth(1)
        self._update_mode_label()

    def _update_mode_label(self):
        if self.mode_label:
            if self.keymap_mode == "vim":
                pfx = f" [{self.vim_prefix}]" if self.vim_prefix else ""
                self.mode_label.setText(f"<b>[VIM: {self.vim_mode.upper()}{pfx}]</b>")
                self.mode_label.setStyleSheet("color: #38bdf8; font-family: monospace; font-weight: bold;")
            elif self.keymap_mode == "emacs":
                pfx = f" {self.emacs_prefix}" if self.emacs_prefix else ""
                mark_str = " (Mark set)" if self.mark_position is not None else ""
                self.mode_label.setText(f"<b>[EMACS{pfx}{mark_str}]</b>")
                self.mode_label.setStyleSheet("color: #f472b6; font-family: monospace; font-weight: bold;")
            else:
                self.mode_label.setText("")

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        if self.isReadOnly():
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()
        cursor = self.textCursor()

        if self.keymap_mode == "emacs":
            ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            alt = bool(modifiers & QtCore.Qt.KeyboardModifier.AltModifier)
            shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

            # Check multi-key prefixes (e.g. C-x ...)
            if self.emacs_prefix == "C-x":
                self.emacs_prefix = None
                self._update_mode_label()
                if ctrl and key == QtCore.Qt.Key.Key_H:  # C-x h: Select entire buffer
                    self.selectAll()
                    return
                elif ctrl and key == QtCore.Qt.Key.Key_U:  # C-x u: Undo
                    self.undo()
                    return
                elif key == QtCore.Qt.Key.Key_H:  # C-x h
                    self.selectAll()
                    return
                elif key == QtCore.Qt.Key.Key_U:  # C-x u
                    self.undo()
                    return
                elif ctrl and key == QtCore.Qt.Key.Key_X:  # C-x C-x: Exchange point and mark
                    if self.mark_position is not None:
                        curr = cursor.position()
                        cursor.setPosition(self.mark_position)
                        self.setTextCursor(cursor)
                        self.mark_position = curr
                        self._update_mode_label()
                    return
                return

            # Cancel prefix or clear mark on C-g
            if ctrl and key == QtCore.Qt.Key.Key_G:  # C-g: Keyboard quit
                self.emacs_prefix = None
                self.mark_position = None
                cursor.clearSelection()
                self.setTextCursor(cursor)
                self._update_mode_label()
                return

            if ctrl and key == QtCore.Qt.Key.Key_X:  # C-x prefix
                self.emacs_prefix = "C-x"
                self._update_mode_label()
                return

            if ctrl:
                if key == QtCore.Qt.Key.Key_A:  # C-a: Beginning of line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_E:  # C-e: End of line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_F:  # C-f: Forward char
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_B:  # C-b: Backward char
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_P:  # C-p: Previous line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Up)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_N:  # C-n: Next line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Down)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_K:  # C-k: Kill to end of line (store in kill ring)
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    if not cursor.hasSelection():
                        # At end of line: kill newline
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    killed_text = cursor.selectedText()
                    if killed_text:
                        self.kill_ring.append(killed_text)
                    cursor.removeSelectedText()
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_W:  # C-w: Kill region (cut)
                    if self.mark_position is not None:
                        cursor.setPosition(self.mark_position, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        self.kill_ring.append(sel)
                        cursor.removeSelectedText()
                        self.setTextCursor(cursor)
                    self.mark_position = None
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_Y:  # C-y: Yank (paste from kill ring or clipboard)
                    if self.kill_ring:
                        cursor.insertText(self.kill_ring[-1])
                    else:
                        self.paste()
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_D:  # C-d: Delete forward char
                    cursor.deleteChar()
                    return
                elif key == QtCore.Qt.Key.Key_H:  # C-h: Backspace (delete backward char)
                    cursor.deletePreviousChar()
                    return
                elif key == QtCore.Qt.Key.Key_O:  # C-o: Open line (insert newline without moving cursor)
                    cursor.insertText("\n")
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_T:  # C-t: Transpose characters
                    pos = cursor.position()
                    if pos > 0:
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        ch2 = cursor.selectedText()
                        cursor.removeSelectedText()
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
                        cursor.insertText(ch2)
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right)
                        self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_L:  # C-l: Recenter scroll
                    self.centerCursor()
                    return
                elif key == QtCore.Qt.Key.Key_Space or (key == QtCore.Qt.Key.Key_At and shift):  # C-Space or C-@: Set mark
                    self.mark_position = cursor.position()
                    self._update_mode_label()
                    return
                elif key in (QtCore.Qt.Key.Key_Slash, QtCore.Qt.Key.Key_Underscore):  # C-/ or C-_ : Undo
                    self.undo()
                    return
                elif key == QtCore.Qt.Key.Key_V:  # C-v: Scroll down one screen (PageDown)
                    self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() + self.verticalScrollBar().pageStep()
                    )
                    return
            elif alt:
                if key == QtCore.Qt.Key.Key_F:  # M-f: Forward word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_B:  # M-b: Backward word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousWord)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_D:  # M-d: Kill word forward
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        self.kill_ring.append(sel)
                    cursor.removeSelectedText()
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_Backspace or key == QtCore.Qt.Key.Key_Delete:  # M-Backspace: Kill word backward
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        self.kill_ring.append(sel)
                    cursor.removeSelectedText()
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_W:  # M-w: Copy region to kill ring
                    if self.mark_position is not None:
                        cursor.setPosition(self.mark_position, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        self.kill_ring.append(sel)
                    self.mark_position = None
                    cursor.clearSelection()
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_V:  # M-v: Scroll up one screen (PageUp)
                    self.verticalScrollBar().setValue(
                        self.verticalScrollBar().value() - self.verticalScrollBar().pageStep()
                    )
                    return
                elif key in (QtCore.Qt.Key.Key_Less, QtCore.Qt.Key.Key_Comma):  # M-< : Beginning of buffer
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
                    self.setTextCursor(cursor)
                    return
                elif key in (QtCore.Qt.Key.Key_Greater, QtCore.Qt.Key.Key_Period):  # M-> : End of buffer
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_U:  # M-u: Uppercase word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        cursor.insertText(sel.upper())
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_L:  # M-l: Lowercase word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        cursor.insertText(sel.lower())
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_C:  # M-c: Capitalize word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    sel = cursor.selectedText()
                    if sel:
                        cursor.insertText(sel.capitalize())
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_T:  # M-t: Transpose words
                    # Move previous word and next word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousWord)
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    w1 = cursor.selectedText()
                    cursor.removeSelectedText()
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord)
                    cursor.insertText(w1)
                    self.setTextCursor(cursor)
                    return

        elif self.keymap_mode == "vim":
            ctrl = bool(modifiers & QtCore.Qt.KeyboardModifier.ControlModifier)
            shift = bool(modifiers & QtCore.Qt.KeyboardModifier.ShiftModifier)

            # Insert Mode
            if self.vim_mode == "insert":
                if key == QtCore.Qt.Key.Key_Escape or (ctrl and key == QtCore.Qt.Key.Key_BracketLeft):
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return
                super().keyPressEvent(event)
                return

            # Visual Mode (character-wise or line-wise)
            if self.vim_mode in ("visual", "visual_line"):
                if key == QtCore.Qt.Key.Key_Escape or (ctrl and key in (QtCore.Qt.Key.Key_BracketLeft, QtCore.Qt.Key.Key_C)):
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    cursor.clearSelection()
                    self.setTextCursor(cursor)
                    self.visual_start_pos = None
                    self._update_mode_label()
                    return

                # Operations on selected visual block
                if key in (QtCore.Qt.Key.Key_Y,):  # y: Yank selection
                    sel = cursor.selectedText()
                    if sel:
                        self.vim_yank_buffer = sel
                        QtWidgets.QApplication.clipboard().setText(sel.replace('\u2029', '\n'))
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    cursor.clearSelection()
                    self.setTextCursor(cursor)
                    self.visual_start_pos = None
                    self._update_mode_label()
                    return
                elif key in (QtCore.Qt.Key.Key_D, QtCore.Qt.Key.Key_X):  # d or x: Cut selection
                    sel = cursor.selectedText()
                    if sel:
                        self.vim_yank_buffer = sel
                        QtWidgets.QApplication.clipboard().setText(sel.replace('\u2029', '\n'))
                        cursor.removeSelectedText()
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    self.visual_start_pos = None
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_C:  # c: Change selection (cut and insert)
                    sel = cursor.selectedText()
                    if sel:
                        self.vim_yank_buffer = sel
                        QtWidgets.QApplication.clipboard().setText(sel.replace('\u2029', '\n'))
                        cursor.removeSelectedText()
                    self.vim_mode = "insert"
                    self.setCursorWidth(1)
                    self.visual_start_pos = None
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_U and shift:  # U: Uppercase visual
                    sel = cursor.selectedText()
                    if sel:
                        cursor.insertText(sel.upper())
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_U:  # u: Lowercase visual
                    sel = cursor.selectedText()
                    if sel:
                        cursor.insertText(sel.lower())
                    self.vim_mode = "normal"
                    self.setCursorWidth(8)
                    self._update_mode_label()
                    return

                # Visual movement navigation
                move_op = None
                if key == QtCore.Qt.Key.Key_H:
                    move_op = QtGui.QTextCursor.MoveOperation.Left
                elif key == QtCore.Qt.Key.Key_L:
                    move_op = QtGui.QTextCursor.MoveOperation.Right
                elif key == QtCore.Qt.Key.Key_J:
                    move_op = QtGui.QTextCursor.MoveOperation.Down
                elif key == QtCore.Qt.Key.Key_K:
                    move_op = QtGui.QTextCursor.MoveOperation.Up
                elif key == QtCore.Qt.Key.Key_W:
                    move_op = QtGui.QTextCursor.MoveOperation.NextWord
                elif key == QtCore.Qt.Key.Key_B:
                    move_op = QtGui.QTextCursor.MoveOperation.PreviousWord
                elif key == QtCore.Qt.Key.Key_0:
                    move_op = QtGui.QTextCursor.MoveOperation.StartOfLine
                elif key == QtCore.Qt.Key.Key_Dollar:
                    move_op = QtGui.QTextCursor.MoveOperation.EndOfLine

                if move_op is not None and self.visual_start_pos is not None:
                    cursor.movePosition(move_op)
                    curr = cursor.position()
                    cursor.setPosition(self.visual_start_pos)
                    cursor.setPosition(curr, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    self.setTextCursor(cursor)
                    return
                return

            # Normal Mode
            if self.vim_mode == "normal":
                # Escape clears prefixes
                if key == QtCore.Qt.Key.Key_Escape:
                    self.vim_prefix = None
                    self._update_mode_label()
                    return

                # Scrolling commands: Ctrl+D, Ctrl+U, Ctrl+F, Ctrl+B
                if ctrl:
                    step = self.verticalScrollBar().pageStep()
                    if key == QtCore.Qt.Key.Key_D:  # Ctrl+D: Half page down
                        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + step // 2)
                        return
                    elif key == QtCore.Qt.Key.Key_U:  # Ctrl+U: Half page up
                        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - step // 2)
                        return
                    elif key == QtCore.Qt.Key.Key_F:  # Ctrl+F: Full page down
                        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + step)
                        return
                    elif key == QtCore.Qt.Key.Key_B:  # Ctrl+B: Full page up
                        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - step)
                        return
                    elif key == QtCore.Qt.Key.Key_R:  # Ctrl+R: Redo
                        self.redo()
                        return

                # Multi-key prefixes handler: dd, dw, d$, yy, yw, y$, cc, cw, c$, gg
                if self.vim_prefix == "d":
                    self.vim_prefix = None
                    self._update_mode_label()
                    if key == QtCore.Qt.Key.Key_D:  # dd: Delete line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Down, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        sel = cursor.selectedText()
                        self.vim_yank_buffer = sel if sel else ""
                        cursor.removeSelectedText()
                        self.setTextCursor(cursor)
                        return
                    elif key == QtCore.Qt.Key.Key_W:  # dw: Delete word
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        self.vim_yank_buffer = cursor.selectedText()
                        cursor.removeSelectedText()
                        self.setTextCursor(cursor)
                        return
                    elif key == QtCore.Qt.Key.Key_Dollar:  # d$: Delete to end of line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        self.vim_yank_buffer = cursor.selectedText()
                        cursor.removeSelectedText()
                        self.setTextCursor(cursor)
                        return
                    return

                elif self.vim_prefix == "y":
                    self.vim_prefix = None
                    self._update_mode_label()
                    if key == QtCore.Qt.Key.Key_Y:  # yy: Yank line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Down, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        self.vim_yank_buffer = cursor.selectedText()
                        QtWidgets.QApplication.clipboard().setText(self.vim_yank_buffer.replace('\u2029', '\n'))
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        self.setTextCursor(cursor)
                        return
                    elif key == QtCore.Qt.Key.Key_W:  # yw: Yank word
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        self.vim_yank_buffer = cursor.selectedText()
                        QtWidgets.QApplication.clipboard().setText(self.vim_yank_buffer.replace('\u2029', '\n'))
                        cursor.clearSelection()
                        self.setTextCursor(cursor)
                        return
                    elif key == QtCore.Qt.Key.Key_Dollar:  # y$: Yank to end of line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        self.vim_yank_buffer = cursor.selectedText()
                        QtWidgets.QApplication.clipboard().setText(self.vim_yank_buffer.replace('\u2029', '\n'))
                        cursor.clearSelection()
                        self.setTextCursor(cursor)
                        return
                    return

                elif self.vim_prefix == "c":
                    self.vim_prefix = None
                    self._update_mode_label()
                    if key == QtCore.Qt.Key.Key_C:  # cc: Change line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        self.vim_mode = "insert"
                        self.setCursorWidth(1)
                        self.setTextCursor(cursor)
                        self._update_mode_label()
                        return
                    elif key == QtCore.Qt.Key.Key_W:  # cw: Change word
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord, QtGui.QTextCursor.MoveMode.KeepAnchor)
                        cursor.removeSelectedText()
                        self.vim_mode = "insert"
                        self.setCursorWidth(1)
                        self.setTextCursor(cursor)
                        self._update_mode_label()
                        return
                    return

                elif self.vim_prefix == "g":
                    self.vim_prefix = None
                    self._update_mode_label()
                    if key == QtCore.Qt.Key.Key_G:  # gg: Jump to top
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Start)
                        self.setTextCursor(cursor)
                        return
                    return

                # Setting operators / prefixes
                if key == QtCore.Qt.Key.Key_D:  # d prefix
                    self.vim_prefix = "d"
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_Y:  # y prefix
                    self.vim_prefix = "y"
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_C:  # c prefix
                    self.vim_prefix = "c"
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_G and not shift:  # g prefix (for gg)
                    self.vim_prefix = "g"
                    self._update_mode_label()
                    return

                # Visual mode trigger
                if key == QtCore.Qt.Key.Key_V:
                    if shift:  # V: Visual line
                        self.vim_mode = "visual_line"
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        self.visual_start_pos = cursor.position()
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    else:  # v: Visual character
                        self.vim_mode = "visual"
                        self.visual_start_pos = cursor.position()
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right, QtGui.QTextCursor.MoveMode.KeepAnchor)
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return

                # Insert mode switches: i, a, o, I, A, O
                if key == QtCore.Qt.Key.Key_I:
                    if shift:  # I: Insert at start of line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                    self.vim_mode = "insert"
                    self.setCursorWidth(1)
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_A:
                    if shift:  # A: Append at end of line
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine)
                    else:  # a: Append after cursor
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right)
                    self.vim_mode = "insert"
                    self.setCursorWidth(1)
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return
                elif key == QtCore.Qt.Key.Key_O:
                    if shift:  # O: Open line above
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                        cursor.insertText("\n")
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.Up)
                    else:  # o: Open line below
                        cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine)
                        cursor.insertText("\n")
                    self.vim_mode = "insert"
                    self.setCursorWidth(1)
                    self.setTextCursor(cursor)
                    self._update_mode_label()
                    return

                # Put / Paste: p (after cursor), P (before cursor)
                elif key == QtCore.Qt.Key.Key_P:
                    text_to_put = self.vim_yank_buffer or QtWidgets.QApplication.clipboard().text()
                    if text_to_put:
                        if shift:  # P: before cursor
                            cursor.insertText(text_to_put)
                        else:  # p: after cursor
                            if '\n' in text_to_put or '\u2029' in text_to_put:
                                cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine)
                                cursor.insertText("\n" + text_to_put.strip('\n\u2029'))
                            else:
                                cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right)
                                cursor.insertText(text_to_put)
                        self.setTextCursor(cursor)
                    return

                # Single char commands
                elif key == QtCore.Qt.Key.Key_X:  # x: Delete char
                    cursor.deleteChar()
                    return
                elif key == QtCore.Qt.Key.Key_U:  # u: Undo
                    self.undo()
                    return
                elif key == QtCore.Qt.Key.Key_Z and shift:  # zz: center screen
                    self.centerCursor()
                    return

                # Navigation commands
                elif key == QtCore.Qt.Key.Key_H:  # h: Left
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Left)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_L:  # l: Right
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Right)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_J:  # j: Down
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Down)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_K:  # k: Up
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.Up)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_W:  # w: Next word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextWord)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_B:  # b: Prev word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.PreviousWord)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_E:  # e: End of word
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfWord)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_0:  # 0: Beginning of line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfLine)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_Dollar:  # $: End of line
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.EndOfLine)
                    self.setTextCursor(cursor)
                    return
                elif key == QtCore.Qt.Key.Key_G and shift:  # G: End of buffer
                    cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
                    self.setTextCursor(cursor)
                    return

                return

        super().keyPressEvent(event)


class BrorsenInputBuilderDialog(QtWidgets.QDialog):
    """GaussView/ORCA-style interactive input builder dialog for newBRORSEN."""

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        symbols: List[str],
        coords: np.ndarray,
        zatoms: Optional[List[ZAtom]] = None,
        variables: Optional[Dict[str, float]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("newBRORSEN Input Builder (Positrons, NEO & CCSD(T))")
        from ..theme import fit_dialog_to_screen
        fit_dialog_to_screen(self, 920, 900, full_height=True)

        self.symbols = symbols or []
        self.coords = coords if coords is not None else np.zeros((len(self.symbols), 3))
        self.zatoms = zatoms or []
        self.variables = variables or {}

        # If zatoms not provided but symbols & coords exist, automatically derive zatoms so zmat is always available
        if not self.zatoms and len(self.symbols) > 0 and len(self.coords) == len(self.symbols):
            try:
                from agymolden.zmatrix import cartesian_to_zmatrix
                self.zatoms = cartesian_to_zmatrix(self.symbols, self.coords)
            except Exception as e:
                print(f"Notice: Could not auto-generate zatoms: {e}")

        self.generator = BrorsenInputGenerator()
        self.current_stage_idx: int = 0
        self._updating_ui: bool = False
        self._user_edited_jobname: bool = False
        self._manual_preview_edit: bool = False

        self._build_ui()
        self._load_stage_into_ui(0)
        self._update_multiplicities()
        self._refresh_preview()

    def _build_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Top Bar: Stages Management
        stage_group = QtWidgets.QGroupBox("Calculation Pipeline (Stages)")
        stage_layout = QtWidgets.QHBoxLayout(stage_group)
        stage_layout.setContentsMargins(8, 6, 8, 6)

        self.combo_stages = QtWidgets.QComboBox()
        self.combo_stages.currentIndexChanged.connect(self._on_stage_selected)
        stage_layout.addWidget(QtWidgets.QLabel("Active Stage:"), 0)
        stage_layout.addWidget(self.combo_stages, 1)

        btn_add_stage = QtWidgets.QPushButton("➕ Add Next Stage")
        btn_add_stage.setToolTip("Add a calculation stage that continues from the current one")
        btn_add_stage.clicked.connect(self._on_add_stage)
        stage_layout.addWidget(btn_add_stage)

        btn_add_pos_stage = QtWidgets.QPushButton("✨ + Positron Stage (%next)")
        btn_add_pos_stage.setStyleSheet("font-weight: bold; color: #38bdf8;")
        btn_add_pos_stage.setToolTip("Add a chained Stage 2 configured for Positron Hartree-Fock / DFT / CC calculation")
        btn_add_pos_stage.clicked.connect(self._on_add_positron_stage)
        stage_layout.addWidget(btn_add_pos_stage)

        btn_remove_stage = QtWidgets.QPushButton("➖ Remove Stage")
        btn_remove_stage.setToolTip("Remove selected calculation stage")
        btn_remove_stage.clicked.connect(self._on_remove_stage)
        stage_layout.addWidget(btn_remove_stage)

        layout.addWidget(stage_group)

        # Main horizontal splitter: Controls on left, Live Preview on right
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left Container: TabWidget with all option categories
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QtWidgets.QTabWidget()
        self._create_bang_tab()
        self._create_electronic_tab()
        self._create_positron_tab()
        self._create_proton_tab()
        self._create_opt_tab()
        self._create_scan_tab()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        left_layout.addWidget(self.tabs)

        splitter.addWidget(left_widget)

        # Right Container: Syntax Highlighted / Formatted Preview
        right_widget = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_header = QtWidgets.QHBoxLayout()
        preview_header.addWidget(QtWidgets.QLabel("<b>Live Input File Preview (*.inp):</b>"))

        self.chk_edit_preview = QtWidgets.QCheckBox("✏️ Direct Edit")
        self.chk_edit_preview.setToolTip("Unlock the preview text to type or edit directives manually")
        self.chk_edit_preview.toggled.connect(self._on_toggle_edit_preview)
        preview_header.addWidget(self.chk_edit_preview)

        # Keymap selector for direct editing: Plain, Vim, Emacs
        self.combo_keymap = QtWidgets.QComboBox()
        self.combo_keymap.addItem("⌨️ Plain", "plain")
        self.combo_keymap.addItem("⌨️ Vim", "vim")
        self.combo_keymap.addItem("⌨️ Emacs", "emacs")
        self.combo_keymap.setToolTip("Editor keybinding mode: Standard, Vim (hjkl, i, Esc, etc.), or Emacs (C-a, C-e, C-k, etc.)")
        self.combo_keymap.setEnabled(False)
        self.combo_keymap.currentIndexChanged.connect(self._on_keymap_changed)
        preview_header.addWidget(self.combo_keymap)

        self.lbl_editor_mode = QtWidgets.QLabel("")
        preview_header.addWidget(self.lbl_editor_mode)

        self.btn_reset_preview = QtWidgets.QPushButton("🔄 Reset Preview")
        self.btn_reset_preview.setToolTip("Discard manual edits and regenerate from settings")
        self.btn_reset_preview.setEnabled(False)
        self.btn_reset_preview.clicked.connect(self._on_reset_preview)
        preview_header.addWidget(self.btn_reset_preview)

        self.lbl_electron_count = QtWidgets.QLabel("Electrons: -")
        self.lbl_electron_count.setStyleSheet("color: #4a90e2; font-weight: bold;")
        preview_header.addStretch()
        preview_header.addWidget(self.lbl_electron_count)
        right_layout.addLayout(preview_header)

        self.txt_preview = KeymapPlainTextEdit()
        self.txt_preview.mode_label = self.lbl_editor_mode
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        font.setPointSize(10)
        self.txt_preview.setFont(font)
        self.txt_preview.setReadOnly(True)
        self.txt_preview.textChanged.connect(self._on_preview_text_edited)
        right_layout.addWidget(self.txt_preview)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter, 1)

        # Bottom Action Bar
        bottom_bar = QtWidgets.QHBoxLayout()

        btn_open = QtWidgets.QPushButton("📂 Open .inp File...")
        btn_open.setToolTip("Open and parse an existing newBRORSEN *.inp input file into the builder")
        btn_open.clicked.connect(self._on_open_input_file)
        bottom_bar.addWidget(btn_open)

        btn_copy = QtWidgets.QPushButton("📋 Copy to Clipboard")
        btn_copy.clicked.connect(self._on_copy_input)
        bottom_bar.addWidget(btn_copy)

        btn_save = QtWidgets.QPushButton("💾 Save Input File...")
        btn_save.setStyleSheet("font-weight: bold;")
        btn_save.clicked.connect(self._on_save_input)
        bottom_bar.addWidget(btn_save)

        btn_run = QtWidgets.QPushButton("▶ Run Calculation...")
        btn_run.setStyleSheet("background-color: #059669; color: white; font-weight: bold; padding: 4px 10px;")
        btn_run.setToolTip("Save input file and launch newBRORSEN live calculation monitor")
        btn_run.clicked.connect(self._on_run_calculation)
        bottom_bar.addWidget(btn_run)

        # Multi-Package Pipeline Stage Chaining (Option A)
        btn_chain_qm = QtWidgets.QPushButton("🔗 + Chain Stage (ORCA / PySCF / QM)...")
        btn_chain_qm.setStyleSheet("background-color: #0d9488; color: white; font-weight: bold; padding: 4px 10px;")
        btn_chain_qm.setToolTip(
            "Multi-Package Calculation Pipeline: Save this newBRORSEN input (Stage 1) and immediately\n"
            "open QM Setup to configure a companion stage in ORCA, PySCF, Gaussian, or openLOWDIN!"
        )
        btn_chain_qm.clicked.connect(self._on_chain_qm_stage)
        bottom_bar.addWidget(btn_chain_qm)

        bottom_bar.addStretch()

        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        bottom_bar.addWidget(btn_close)

        layout.addLayout(bottom_bar)
        self._update_stage_combo()
        self._load_stage_into_ui(0)
        self._update_multiplicities()
        self._update_jobname_if_auto()
        self._refresh_preview()

    def _create_bang_tab(self):
        """General & '!' line settings."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        jobname_layout = QtWidgets.QHBoxLayout()
        self.txt_jobname = QtWidgets.QLineEdit()
        self.txt_jobname.textEdited.connect(self._on_jobname_user_edited)
        self.txt_jobname.textChanged.connect(self._on_jobname_text_changed)
        btn_regen_jobname = QtWidgets.QPushButton("🔄 Auto")
        btn_regen_jobname.setToolTip("Auto-generate job name from stoichiometry, method, basis set, and run type")
        btn_regen_jobname.clicked.connect(self._on_regenerate_jobname_clicked)
        jobname_layout.addWidget(self.txt_jobname)
        jobname_layout.addWidget(btn_regen_jobname)
        form.addRow("Job Name / Prefix (%jobname):", jobname_layout)

        self.combo_run_type = QtWidgets.QComboBox()
        for code, desc in RUN_TYPES:
            self.combo_run_type.addItem(f"{code} — {desc}", code)
        self.combo_run_type.currentIndexChanged.connect(self._on_bang_changed)
        form.addRow("Run Type (!):", self.combo_run_type)

        self.combo_method = QtWidgets.QComboBox()
        for code, desc in METHODS:
            self.combo_method.addItem(f"{code} — {desc}", code)
        self.combo_method.currentIndexChanged.connect(self._on_bang_changed)
        form.addRow("Method (!):", self.combo_method)

        self.combo_sector = QtWidgets.QComboBox()
        for code, desc in SECTORS:
            self.combo_sector.addItem(f"{code} — {desc}", code)
        self.combo_sector.currentIndexChanged.connect(self._on_sector_changed)
        form.addRow("Sector (!):", self.combo_sector)

        self.combo_integrals = QtWidgets.QComboBox()
        for code, desc in INTEGRAL_MODES:
            self.combo_integrals.addItem(f"{code} — {desc}", code)
        self.combo_integrals.currentIndexChanged.connect(self._on_bang_changed)
        form.addRow("Integral Engine (!):", self.combo_integrals)

        self.combo_coords_unit = QtWidgets.QComboBox()
        for code, desc in COORD_UNITS:
            self.combo_coords_unit.addItem(f"{code} — {desc}", code)
        self.combo_coords_unit.currentIndexChanged.connect(self._on_bang_changed)
        form.addRow("Coordinate Units (!):", self.combo_coords_unit)

        self.spin_charge = QtWidgets.QSpinBox()
        self.spin_charge.setRange(-20, 20)
        self.spin_charge.setValue(0)
        self.spin_charge.valueChanged.connect(self._on_charge_changed)
        form.addRow("Total Charge (%charge):", self.spin_charge)

        self.combo_mult = QtWidgets.QComboBox()
        self.combo_mult.currentIndexChanged.connect(self._on_mult_changed)
        form.addRow("Multiplicity 2S+1 (%mult):", self.combo_mult)

        self.spin_cores = QtWidgets.QSpinBox()
        self.spin_cores.setRange(1, 128)
        self.spin_cores.setValue(1)
        self.spin_cores.valueChanged.connect(self._on_field_changed)
        form.addRow("Threads / Cores (%nprocs):", self.spin_cores)

        self.spin_memory = QtWidgets.QSpinBox()
        self.spin_memory.setRange(500, 512000)
        self.spin_memory.setSingleStep(1000)
        self.spin_memory.setValue(4000)
        self.spin_memory.setSuffix(" MB")
        self.spin_memory.valueChanged.connect(self._on_field_changed)
        form.addRow("Memory Target (%memory):", self.spin_memory)

        self.spin_verbose = QtWidgets.QSpinBox()
        self.spin_verbose.setRange(0, 9)
        self.spin_verbose.setValue(4)
        self.spin_verbose.setToolTip("PySCF Verbosity (%verbose): 4=Detailed SCF convergence/iterations & orbital reports (Recommended), 5=Debug, 0=Quiet")
        self.spin_verbose.valueChanged.connect(self._on_field_changed)
        form.addRow("Output Verbosity (%verbose):", self.spin_verbose)

        self.chk_own_geom = QtWidgets.QCheckBox("Include Geometry block in this stage")
        self.chk_own_geom.setChecked(True)
        self.chk_own_geom.toggled.connect(self._on_field_changed)
        form.addRow("", self.chk_own_geom)

        fmt_row = QtWidgets.QHBoxLayout()
        self.combo_geom_fmt = QtWidgets.QComboBox()
        self.combo_geom_fmt.addItem("Cartesian (* xyz)", "xyz")
        self.combo_geom_fmt.addItem("Z-Matrix (* zmat)", "zmat")
        self.combo_geom_fmt.currentIndexChanged.connect(self._on_geom_fmt_changed)
        fmt_row.addWidget(self.combo_geom_fmt, stretch=1)

        btn_fmt_info = QtWidgets.QToolButton()
        btn_fmt_info.setText("ℹ️")
        btn_fmt_info.setToolTip("View detailed explanation of Geometry Format options (* xyz vs * zmat)")
        btn_fmt_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #0284c7; border: none; padding: 2px 6px;")
        btn_fmt_info.clicked.connect(self._on_show_geom_format_info)
        fmt_row.addWidget(btn_fmt_info)

        form.addRow("Geometry Format:", fmt_row)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "⚡ General & !")

    def _create_electronic_tab(self):
        """Electronic Basis & DFT options."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()

        # Basis Set Row with Smart BSE Selector Button
        row_basis = QtWidgets.QHBoxLayout()
        self.combo_basis = QtWidgets.QComboBox()
        self.combo_basis.setEditable(True)
        for b in ELECTRONIC_BASES:
            self.combo_basis.addItem(b)
        self.combo_basis.currentTextChanged.connect(self._on_field_changed)
        row_basis.addWidget(self.combo_basis, stretch=1)

        btn_smart_basis = QtWidgets.QPushButton("🔍 Browse BSE (770+)...")
        btn_smart_basis.setStyleSheet("background-color: #0284c7; color: #fff; font-weight: bold; padding: 3px 8px;")
        btn_smart_basis.setToolTip("Open Smart Basis Set Exchange Selector filtered to current molecule's elements")
        btn_smart_basis.clicked.connect(self._on_browse_smart_basis)
        row_basis.addWidget(btn_smart_basis)

        btn_basis_info = QtWidgets.QToolButton()
        btn_basis_info.setText("ℹ️")
        btn_basis_info.setToolTip("Instructions & syntax for mixed / per-atom / per-element %basis options in newBRORSEN")
        btn_basis_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #0284c7; border: none; padding: 2px 6px;")
        btn_basis_info.clicked.connect(self._on_show_basis_format_info)
        row_basis.addWidget(btn_basis_info)

        form.addRow("Electron Basis (%basis):", row_basis)

        # Per-Element & Per-Atom Mixed Basis Builder Helper
        self.grp_custom_basis = QtWidgets.QGroupBox("Mixed / Per-Atom Basis Sets Helper")
        b_layout = QtWidgets.QVBoxLayout(self.grp_custom_basis)
        b_info = QtWidgets.QLabel(
            "<span style='color: #64748b; font-size: 11px;'>"
            "newBRORSEN supports different basis sets on different atoms/elements on one <code>%basis</code> line.<br>"
            "Examples: <code>aug-cc-pvtz H cc-pvdz</code> (aug-cc-pVTZ on H, cc-pVDZ on others), or "
            "<code>def2-tzvp {1 2} def2-svp</code> (def2-TZVP on atoms 1 & 2, def2-SVP on others).</span>"
        )
        b_info.setWordWrap(True)
        b_layout.addWidget(b_info)

        row_quick_mix = QtWidgets.QHBoxLayout()
        btn_mix_h = QtWidgets.QPushButton("⚡ Augment Hydrogens (aug-cc-pVTZ H cc-pVDZ)")
        btn_mix_h.setToolTip("Sets augmented basis on Hydrogens and compact correlation-consistent basis on heavy atoms")
        btn_mix_h.clicked.connect(lambda: self._apply_mixed_basis_preset("aug-cc-pvtz H cc-pvdz"))
        row_quick_mix.addWidget(btn_mix_h)

        btn_mix_heavy = QtWidgets.QPushButton("⚡ High-Level Active Site (def2-TZVP {1 2} def2-SVP)")
        btn_mix_heavy.setToolTip("Sets def2-TZVP on active site atoms 1 & 2 and def2-SVP on spectator atoms")
        btn_mix_heavy.clicked.connect(lambda: self._apply_mixed_basis_preset("def2-tzvp {1 2} def2-svp"))
        row_quick_mix.addWidget(btn_mix_heavy)
        b_layout.addLayout(row_quick_mix)

        form.addRow(self.grp_custom_basis)

        self.combo_frozen = QtWidgets.QComboBox()
        self.combo_frozen.setEditable(True)
        self.combo_frozen.addItem("0", "0")
        self.combo_frozen.addItem("auto", "auto")
        self.combo_frozen.addItem("n-2", "n-2")
        self.combo_frozen.currentTextChanged.connect(self._on_frozen_changed)
        form.addRow("Frozen Core (%frozen):", self.combo_frozen)

        self.combo_xc = QtWidgets.QComboBox()
        self.combo_xc.setEditable(True)
        for xc in DFT_FUNCTIONALS:
            self.combo_xc.addItem(xc)
        self.combo_xc.currentTextChanged.connect(self._on_field_changed)
        form.addRow("DFT Functional (%xc):", self.combo_xc)

        self.chk_molden = QtWidgets.QCheckBox("Generate Molden files (%molden: HF & CCSD orbitals)")
        self.chk_molden.setToolTip("Export Molden files at the final geometry (HF canonical orbitals and CCSD natural orbitals)")
        self.chk_molden.toggled.connect(self._on_field_changed)
        form.addRow("", self.chk_molden)

        self.chk_cube = QtWidgets.QCheckBox("Generate 3D Cube grids (%cube: density & orbitals)")
        self.chk_cube.setToolTip("Export 3D volumetric .cube files (electron and positron density grids) for viewing in AGYMolden")
        self.chk_cube.toggled.connect(self._on_field_changed)
        form.addRow("", self.chk_cube)

        self.spin_cube_points = QtWidgets.QSpinBox()
        self.spin_cube_points.setRange(20, 500)
        self.spin_cube_points.setValue(80)
        self.spin_cube_points.valueChanged.connect(self._on_field_changed)
        form.addRow("Cube Grid Points (%cube_points):", self.spin_cube_points)

        self.chk_chelpg = QtWidgets.QCheckBox("CHELPG Charges (%chelpg: printed in .out)")
        self.chk_chelpg.setToolTip("Fit electrostatic potential to atomic centers (CHELPG charges in output)")
        self.chk_chelpg.toggled.connect(self._on_field_changed)
        form.addRow("", self.chk_chelpg)

        self.chk_density_map = QtWidgets.QCheckBox("Density & ESP 2D Maps (%density_map)")
        self.chk_density_map.setToolTip("Generate high-resolution 2D plane density maps and ESP contours (SVG/PNG)")
        self.chk_density_map.toggled.connect(self._on_field_changed)
        form.addRow("", self.chk_density_map)

        row_map_opts = QtWidgets.QHBoxLayout()
        self.combo_map_type = QtWidgets.QComboBox()
        self.combo_map_type.addItem("both (Electron & Positron)", "both")
        self.combo_map_type.addItem("esp (Electrostatic Potential)", "esp")
        self.combo_map_type.addItem("electron (Electron Density)", "electron")
        self.combo_map_type.addItem("positron (Positron Density)", "positron")
        self.combo_map_type.currentIndexChanged.connect(self._on_field_changed)
        row_map_opts.addWidget(self.combo_map_type, stretch=1)

        self.combo_map_plane = QtWidgets.QComboBox()
        self.combo_map_plane.addItem("Plane: AUTO", "AUTO")
        self.combo_map_plane.addItem("Plane: XY", "XY")
        self.combo_map_plane.addItem("Plane: XZ", "XZ")
        self.combo_map_plane.addItem("Plane: YZ", "YZ")
        self.combo_map_plane.currentIndexChanged.connect(self._on_field_changed)
        row_map_opts.addWidget(self.combo_map_plane)
        form.addRow("Map Content & Plane:", row_map_opts)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "⚛️ Electronic")

    def _apply_mixed_basis_preset(self, spec: str):
        """Set a mixed basis specification into the %basis field."""
        if self.combo_basis.findText(spec) == -1:
            self.combo_basis.insertItem(0, spec)
        self.combo_basis.setCurrentText(spec)
        self._on_field_changed()

    def _on_show_basis_format_info(self):
        """Display comprehensive instructions and possibilities for %basis in newBRORSEN."""
        QtWidgets.QMessageBox.information(
            self,
            "newBRORSEN Basis Set Options & Format Instructions",
            "<h3>newBRORSEN Electronic & Positronic Basis Options</h3>"
            "<p>newBRORSEN allows setting custom basis sets per element, per atom index, or in ranges, all within a single line.</p>"
            "<hr>"
            "<h4>1. Single Global Basis:</h4>"
            "<ul>"
            "  <li><code>cc-pvdz</code> or <code>def2-tzvp</code>: Applied uniformly to all atoms in the molecule.</li>"
            "</ul>"
            "<h4>2. Per-Element Mixed Basis:</h4>"
            "<ul>"
            "  <li><code>B1 El1 B2 El2 ...</code>: Assigns basis <code>B1</code> to element <code>El1</code>, <code>B2</code> to <code>El2</code>.<br>"
            "      <b>Example:</b> <code>aug-cc-pvtz H cc-pvdz</code><br>"
            "      <i>(Places augmented triple-zeta on all Hydrogens, and cc-pVDZ fallback on all other atoms).</i></li>"
            "  <li><b>Multiple Elements:</b> <code>def2-tzvp {C N O} def2-svp</code><br>"
            "      <i>(Places def2-TZVP on C, N, and O, and def2-SVP on the rest).</i></li>"
            "</ul>"
            "<h4>3. Per-Atom Index & Range Assignment:</h4>"
            "<ul>"
            "  <li><code>B {i j k} FallbackBasis</code>: Assigns basis <code>B</code> specifically to atom numbers (1-based).<br>"
            "      <b>Example:</b> <code>def2-tzvp {1 2} def2-svp</code><br>"
            "      <i>(Places def2-TZVP on atoms 1 and 2, and def2-SVP on all other atoms).</i></li>"
            "  <li><b>Atom Ranges:</b> <code>aug-cc-pvtz [1,5] cc-pvdz</code> (or <code>[1-5]</code>)<br>"
            "      <i>(Assigns aug-cc-pVTZ to atoms 1 through 5 inclusive, and cc-pVDZ to the rest).</i></li>"
            "</ul>"
            "<h4>4. Custom External Basis Files:</h4>"
            "<ul>"
            "  <li>Specify the file path (e.g. <code>\"/path/to/my_basis.bas\"</code> or <code>./custom.gbs</code>).</li>"
            "</ul>"
            "<h4>Precedence Hierarchy:</h4>"
            "<p style='color: #64748b; font-size: 11px;'>Specific Atom Number (e.g. <code>{1 2}</code> or <code>[1,5]</code>) <b>></b> Element Symbol (e.g. <code>H</code>) <b>></b> Fallback Basis without atoms.</p>"
        )

    def _apply_mixed_pos_basis_preset(self, spec: str):
        """Set a mixed positron basis specification into the %pos_basis field."""
        if self.combo_pos_basis.findText(spec) == -1:
            self.combo_pos_basis.insertItem(0, spec)
        self.combo_pos_basis.setCurrentText(spec)
        self._on_field_changed()

    def _on_show_pos_basis_format_info(self):
        """Display comprehensive instructions and possibilities for %pos_basis in newBRORSEN."""
        QtWidgets.QMessageBox.information(
            self,
            "newBRORSEN Positron Basis Options (%pos_basis)",
            "<h3>newBRORSEN Positronic Basis Options (%pos_basis)</h3>"
            "<p>Positronic Gaussian basis sets can be assigned per-element, per-atom, or in ranges, exactly matching <code>%basis</code>.</p>"
            "<hr>"
            "<h4>1. Global Positron Basis:</h4>"
            "<ul>"
            "  <li><code>PSX-DZ</code>, <code>PSX-TZ</code>, or <code>PSX-QZ</code>: Placed on all designated positron centers.</li>"
            "</ul>"
            "<h4>2. Per-Element Positron Assignment:</h4>"
            "<ul>"
            "  <li><code>PSX-TZ H PSX-DZ</code>: Assigns <code>PSX-TZ</code> to all Hydrogens, and <code>PSX-DZ</code> to all other atoms.</li>"
            "  <li><code>PSX-TZ {H O} PSX-DZ</code>: Places <code>PSX-TZ</code> on H and O atoms, and <code>PSX-DZ</code> on the rest.</li>"
            "</ul>"
            "<h4>3. Per-Atom Index & Ranges:</h4>"
            "<ul>"
            "  <li><code>PSX-TZ {1} PSX-DZ</code>: Assigns <code>PSX-TZ</code> to atom 1 (e.g. quantum site), and <code>PSX-DZ</code> elsewhere.</li>"
            "  <li><code>PSX-TZ [1,5] PSX-DZ</code>: Assigns <code>PSX-TZ</code> on atoms 1 through 5 inclusive, and <code>PSX-DZ</code> to the rest.</li>"
            "</ul>"
            "<h4>Precedence Hierarchy:</h4>"
            "<p style='color: #64748b; font-size: 11px;'>Specific Atom Number (e.g. <code>{1}</code> or <code>[1,5]</code>) <b>></b> Element Symbol (e.g. <code>H</code>) <b>></b> Fallback Positron Basis without atoms.</p>"
        )

    def _on_browse_smart_basis(self):
        """Open Smart Basis Set Exchange Selector dialog."""
        elements = [za.symbol for za in self.zatoms if za.symbol != "X"]
        dlg = SmartBasisDialog(self, elements=elements, current_basis=self.combo_basis.currentText().strip())
        if dlg.exec():
            selected = dlg.selected_basis_name
            if selected:
                # Add to combo if not present and select it
                if self.combo_basis.findText(selected) == -1:
                    self.combo_basis.insertItem(0, selected)
                self.combo_basis.setCurrentText(selected)

    def _create_positron_tab(self):
        """Positron particle & basis options."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()

        self.spin_npos = QtWidgets.QSpinBox()
        self.spin_npos.setRange(1, 10)
        self.spin_npos.setValue(1)
        self.spin_npos.valueChanged.connect(self._on_npos_changed)
        form.addRow("Positron Count (%n_pos):", self.spin_npos)

        # Positron Basis Row with Info Button
        row_pos_basis = QtWidgets.QHBoxLayout()
        self.combo_pos_basis = QtWidgets.QComboBox()
        self.combo_pos_basis.setEditable(True)
        for pb in POSITRON_BASES:
            self.combo_pos_basis.addItem(pb)
        self.combo_pos_basis.currentTextChanged.connect(self._on_field_changed)
        row_pos_basis.addWidget(self.combo_pos_basis, stretch=1)

        btn_pos_basis_info = QtWidgets.QToolButton()
        btn_pos_basis_info.setText("ℹ️")
        btn_pos_basis_info.setToolTip("Instructions & format for mixed / per-atom %pos_basis in newBRORSEN")
        btn_pos_basis_info.setStyleSheet("font-weight: bold; font-size: 13px; color: #0284c7; border: none; padding: 2px 6px;")
        btn_pos_basis_info.clicked.connect(self._on_show_pos_basis_format_info)
        row_pos_basis.addWidget(btn_pos_basis_info)

        form.addRow("Positron Basis (%pos_basis):", row_pos_basis)

        # Quick Mixed Positron Presets Box
        grp_pos_presets = QtWidgets.QGroupBox("Mixed Positron Basis Sets Helper")
        pos_p_layout = QtWidgets.QVBoxLayout(grp_pos_presets)
        pos_p_info = QtWidgets.QLabel(
            "<span style='color: #64748b; font-size: 11px;'>"
            "Like <code>%basis</code>, <code>%pos_basis</code> accepts per-element and per-atom sets on one line.<br>"
            "Examples: <code>PSX-TZ H PSX-DZ</code> (triple-zeta on H, double-zeta on others), or "
            "<code>PSX-TZ [1,5] PSX-DZ</code> (triple-zeta on atoms 1–5, double-zeta on others).</span>"
        )
        pos_p_info.setWordWrap(True)
        pos_p_layout.addWidget(pos_p_info)

        row_pos_btns = QtWidgets.QHBoxLayout()
        btn_pos_h = QtWidgets.QPushButton("⚡ PSX-TZ on H (PSX-TZ H PSX-DZ)")
        btn_pos_h.setToolTip("Large positron basis on Hydrogens, compact set on heavy atoms")
        btn_pos_h.clicked.connect(lambda: self._apply_mixed_pos_basis_preset("PSX-TZ H PSX-DZ"))
        row_pos_btns.addWidget(btn_pos_h)

        btn_pos_site = QtWidgets.QPushButton("⚡ PSX-TZ on Target Site (PSX-TZ {1} PSX-DZ)")
        btn_pos_site.setToolTip("Triple-zeta positron basis centered on atom 1, double-zeta on spectator centers")
        btn_pos_site.clicked.connect(lambda: self._apply_mixed_pos_basis_preset("PSX-TZ {1} PSX-DZ"))
        row_pos_btns.addWidget(btn_pos_site)
        pos_p_layout.addLayout(row_pos_btns)

        form.addRow(grp_pos_presets)

        self.txt_psx_set = QtWidgets.QLineEdit()
        self.txt_psx_set.setPlaceholderText("Optional basis family selector (e.g. s, sp)")
        self.txt_psx_set.textChanged.connect(self._on_field_changed)
        form.addRow("PSX Basis Family (%psx_set):", self.txt_psx_set)

        self.txt_pos_atom = QtWidgets.QLineEdit("all")
        self.txt_pos_atom.setToolTip("Atom symbols or numbers carrying positron functions")
        self.txt_pos_atom.textChanged.connect(self._on_field_changed)
        form.addRow("Positron Centers (%pos_atom):", self.txt_pos_atom)

        self.combo_pos_mult = QtWidgets.QComboBox()
        self.combo_pos_mult.addItem("1 (Singlet pair)", 1)
        self.combo_pos_mult.addItem("3 (Triplet pair)", 3)
        self.combo_pos_mult.currentIndexChanged.connect(self._on_field_changed)
        form.addRow("Positron Mult (%pos_mult):", self.combo_pos_mult)

        self.combo_pos_mass = QtWidgets.QComboBox()
        self.combo_pos_mass.setEditable(True)
        for pm in PARTICLE_MASSES:
            self.combo_pos_mass.addItem(pm)
        self.combo_pos_mass.currentTextChanged.connect(self._on_field_changed)
        form.addRow("Particle Mass (%pos_mass):", self.combo_pos_mass)

        self.txt_quantum_site = QtWidgets.QLineEdit()
        self.txt_quantum_site.setPlaceholderText("Atom index (1-based) of H replaced by particle")
        self.txt_quantum_site.textChanged.connect(self._on_field_changed)
        form.addRow("Quantum Site (%quantum_site):", self.txt_quantum_site)

        # Positron-specific wavefunction output options
        pos_wave_box = QtWidgets.QGroupBox("Positron Wavefunction & Density Export")
        pos_wave_layout = QtWidgets.QVBoxLayout(pos_wave_box)
        self.lbl_pos_export_hint = QtWidgets.QLabel(
            "When %molden or %cube is enabled, newBRORSEN automatically exports separate\n"
            "files for both electrons and positrons at the final geometry:\n"
            " • <job>_hf_positrons.molden & <job>_ccsd_positrons_natorb.molden\n"
            " • <job>_{hf,ccsd}_positron_density.cube"
        )
        self.lbl_pos_export_hint.setStyleSheet("color: #38bdf8; font-size: 11px;")
        pos_wave_layout.addWidget(self.lbl_pos_export_hint)
        form.addRow(pos_wave_box)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "✨ Positron")

    def _create_proton_tab(self):
        """Proton / NEO & CNEO options."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()

        self.combo_nuc_basis = QtWidgets.QComboBox()
        self.combo_nuc_basis.setEditable(True)
        for nb in PROTON_BASES:
            self.combo_nuc_basis.addItem(nb)
        self.combo_nuc_basis.currentTextChanged.connect(self._on_field_changed)
        form.addRow("Nuclear Basis (%nuc_basis):", self.combo_nuc_basis)

        self.txt_quantum_nuc = QtWidgets.QLineEdit()
        self.txt_quantum_nuc.setPlaceholderText("0-based atom indices, comma-separated (e.g. 0, 1)")
        self.txt_quantum_nuc.textChanged.connect(self._on_field_changed)
        form.addRow("Quantum Nuclei (%quantum_nuc):", self.txt_quantum_nuc)

        self.chk_cneo = QtWidgets.QCheckBox("Constrained NEO (%cneo)")
        self.chk_cneo.toggled.connect(self._on_cneo_toggled)
        form.addRow("", self.chk_cneo)

        self.combo_cneo_var = QtWidgets.QComboBox()
        self.combo_cneo_var.addItem("ccd", "ccd")
        self.combo_cneo_var.addItem("ucd", "ucd")
        self.combo_cneo_var.currentIndexChanged.connect(self._on_field_changed)
        form.addRow("CNEO Variant (%cneo_variant):", self.combo_cneo_var)

        self.combo_epc = QtWidgets.QComboBox()
        for epc in EPC_FUNCTIONALS:
            self.combo_epc.addItem(epc)
        self.combo_epc.currentTextChanged.connect(self._on_field_changed)
        form.addRow("EPC Functional (%epc):", self.combo_epc)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "🔬 Proton (NEO)")

    def _create_opt_tab(self):
        """Optimization & Transition State options."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()

        self.combo_opt_engine = QtWidgets.QComboBox()
        for eng, desc in OPT_ENGINES:
            self.combo_opt_engine.addItem(f"{eng} — {desc}", eng)
        self.combo_opt_engine.currentIndexChanged.connect(self._on_opt_engine_changed)
        form.addRow("Optimizer (%opt_engine):", self.combo_opt_engine)

        self.spin_max_cycles = QtWidgets.QSpinBox()
        self.spin_max_cycles.setRange(5, 500)
        self.spin_max_cycles.setValue(50)
        self.spin_max_cycles.valueChanged.connect(self._on_field_changed)
        form.addRow("Max Cycles (%opt_max_cycles):", self.spin_max_cycles)

        self.combo_ts_mode = QtWidgets.QComboBox()
        for mode, desc in TS_MODES:
            self.combo_ts_mode.addItem(f"{mode} — {desc}", mode)
        self.combo_ts_mode.currentIndexChanged.connect(self._on_ts_mode_changed)
        form.addRow("TS Mode (%ts_mode):", self.combo_ts_mode)

        self.combo_irc_dir = QtWidgets.QComboBox()
        self.combo_irc_dir.addItem("both")
        self.combo_irc_dir.addItem("forward")
        self.combo_irc_dir.addItem("backward")
        self.combo_irc_dir.currentIndexChanged.connect(self._on_field_changed)
        form.addRow("IRC Direction (%irc_direction):", self.combo_irc_dir)

        # ZOPT box
        self.grp_zopt = QtWidgets.QGroupBox("Z-Matrix Optimizer Settings (ZOPT)")
        zlayout = QtWidgets.QFormLayout(self.grp_zopt)
        self.spin_zopt_evals = QtWidgets.QSpinBox()
        self.spin_zopt_evals.setRange(10, 1000)
        self.spin_zopt_evals.setValue(60)
        self.spin_zopt_evals.valueChanged.connect(self._on_field_changed)
        zlayout.addRow("Max Evals:", self.spin_zopt_evals)

        self.txt_zopt_etol = QtWidgets.QLineEdit("1e-6")
        self.txt_zopt_etol.textChanged.connect(self._on_field_changed)
        zlayout.addRow("Energy Tol (Ha):", self.txt_zopt_etol)

        form.addRow(self.grp_zopt)

        # Optimization Constraints, Pinnings & Equivalences
        self.grp_constraints = QtWidgets.QGroupBox("Optimization Constraints & Equivalences")
        c_layout = QtWidgets.QVBoxLayout(self.grp_constraints)
        c_form = QtWidgets.QFormLayout()

        # Frozen Atoms (%pin)
        pin_row = QtWidgets.QHBoxLayout()
        self.txt_pinnings = QtWidgets.QLineEdit()
        self.txt_pinnings.setPlaceholderText("e.g. 1, 3, 5  (1-based atom indices)")
        self.txt_pinnings.textChanged.connect(self._on_field_changed)
        pin_row.addWidget(self.txt_pinnings, stretch=1)
        btn_pin_sel = QtWidgets.QPushButton("📌 Pin Selected Atoms")
        btn_pin_sel.setToolTip("Freeze all Cartesian coordinates of currently selected 3D atoms (%pin)")
        btn_pin_sel.clicked.connect(self._on_pin_selected_atoms)
        pin_row.addWidget(btn_pin_sel)
        c_form.addRow("Freeze Atoms (%pin):", pin_row)

        # Geometric Internal Constraints (%constraints ... end)
        self.txt_constraints = QtWidgets.QPlainTextEdit()
        self.txt_constraints.setMaximumHeight(80)
        self.txt_constraints.setPlaceholderText("bond 1 2 [target_val]\nangle 1 2 3 [target_val]\ndihedral 1 2 3 4 [target_val]")
        self.txt_constraints.textChanged.connect(self._on_field_changed)
        c_form.addRow("Internal Constraints (%constraints):", self.txt_constraints)

        # Bond Equivalences (%equivalence)
        self.txt_equivalences = QtWidgets.QLineEdit()
        self.txt_equivalences.setPlaceholderText("e.g. 1,2,3,4  (bond 1-2 equivalent to bond 3-4)")
        self.txt_equivalences.textChanged.connect(self._on_field_changed)
        c_form.addRow("Bond Equivalences (%equivalence):", self.txt_equivalences)

        c_layout.addLayout(c_form)
        form.addRow(self.grp_constraints)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "🎯 Opt & TS")

    def _create_scan_tab(self):
        """Coordinate scan options."""
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        form = QtWidgets.QFormLayout()

        self.chk_scan = QtWidgets.QCheckBox("Enable Parameter Scan (%scan)")
        self.chk_scan.toggled.connect(self._on_scan_toggled)
        form.addRow("", self.chk_scan)

        self.combo_scan_type = QtWidgets.QComboBox()
        self.combo_scan_type.addItem("bond (atoms i j)", "bond")
        self.combo_scan_type.addItem("angle (atoms i j k)", "angle")
        self.combo_scan_type.addItem("dihedral (atoms i j k l)", "dihedral")
        self.combo_scan_type.addItem("zmat variable", "var")
        self.combo_scan_type.currentIndexChanged.connect(self._on_field_changed)
        form.addRow("Scan Coordinate:", self.combo_scan_type)

        self.txt_scan_target = QtWidgets.QLineEdit()
        self.txt_scan_target.setPlaceholderText("e.g. 1 2 for bond, or variable name R1")
        self.txt_scan_target.textChanged.connect(self._on_field_changed)
        form.addRow("Target (atoms / var):", self.txt_scan_target)

        self.spin_scan_start = QtWidgets.QDoubleSpinBox()
        self.spin_scan_start.setRange(-360.0, 360.0)
        self.spin_scan_start.setValue(1.0)
        self.spin_scan_start.valueChanged.connect(self._on_field_changed)
        form.addRow("Scan Start:", self.spin_scan_start)

        self.spin_scan_end = QtWidgets.QDoubleSpinBox()
        self.spin_scan_end.setRange(-360.0, 360.0)
        self.spin_scan_end.setValue(2.0)
        self.spin_scan_end.valueChanged.connect(self._on_field_changed)
        form.addRow("Scan End:", self.spin_scan_end)

        self.spin_scan_steps = QtWidgets.QSpinBox()
        self.spin_scan_steps.setRange(2, 200)
        self.spin_scan_steps.setValue(10)
        self.spin_scan_steps.valueChanged.connect(self._on_field_changed)
        form.addRow("Number of Steps:", self.spin_scan_steps)

        layout.addLayout(form)
        layout.addStretch()
        self.tabs.addTab(w, "📈 Scan")

    # ---- Stage Management ----
    def _update_stage_combo(self):
        self.combo_stages.blockSignals(True)
        self.combo_stages.clear()
        for i, st in enumerate(self.generator.stages):
            self.combo_stages.addItem(f"Stage {st.number}: {st.jobname} [{st.run_type} {st.method} {st.sector}]")
        self.combo_stages.setCurrentIndex(self.current_stage_idx)
        self.combo_stages.blockSignals(False)

    def _on_stage_selected(self, idx: int):
        if idx >= 0 and idx < len(self.generator.stages):
            self._save_ui_into_stage(self.current_stage_idx)
            self.current_stage_idx = idx
            self._load_stage_into_ui(idx)
            self._update_multiplicities()
            self._refresh_preview()

    def _on_add_stage(self):
        self._save_ui_into_stage(self.current_stage_idx)
        st = self.generator.add_stage()
        # Disambiguate jobname if it duplicates an existing stage
        existing_names = {s.jobname for s in self.generator.stages[:-1]}
        if st.jobname in existing_names:
            st.jobname = f"{st.jobname}_stage{st.number}"
        self.current_stage_idx = len(self.generator.stages) - 1
        self._update_stage_combo()
        self._load_stage_into_ui(self.current_stage_idx)
        self._update_multiplicities()
        self._update_jobname_if_auto()
        self._refresh_preview()

    def _on_add_positron_stage(self):
        """1-click convenience to append a positronic calculation continuing from Stage 1."""
        self._save_ui_into_stage(self.current_stage_idx)
        st = self.generator.add_stage()
        st.sector = "POSITRON"
        st.n_pos = 1
        st.pos_basis = "PSX-TZ"
        st.pos_mult = 1
        st.molden = True
        self.current_stage_idx = len(self.generator.stages) - 1
        self._user_edited_jobname = False
        new_jname = self._auto_construct_jobname()
        st.jobname = new_jname
        self._update_stage_combo()
        self._load_stage_into_ui(self.current_stage_idx)
        self._update_multiplicities()
        self._refresh_preview()
        # Switch to Positron tab (index 2: Bang=0, Electronic=1, Positron=2)
        if self.tabs.count() > 2:
            self.tabs.setCurrentIndex(2)

    def _on_remove_stage(self):
        if len(self.generator.stages) <= 1:
            QtWidgets.QMessageBox.information(self, "Cannot Remove", "A calculation must contain at least one stage.")
            return
        self.generator.remove_stage(self.current_stage_idx)
        self.current_stage_idx = max(0, self.current_stage_idx - 1)
        self._update_stage_combo()
        self._load_stage_into_ui(self.current_stage_idx)
        self._update_multiplicities()
        self._update_jobname_if_auto()
        self._refresh_preview()

    # ---- State Loading & Saving ----
    def _load_stage_into_ui(self, idx: int):
        self._updating_ui = True
        st = self.generator.stages[idx]

        self.txt_jobname.setText(st.jobname)

        # Bang combos
        self._set_combo_by_data(self.combo_run_type, st.run_type)
        self._set_combo_by_data(self.combo_method, st.method)
        self._set_combo_by_data(self.combo_sector, st.sector)
        self._set_combo_by_data(self.combo_integrals, st.integrals)
        self._set_combo_by_data(self.combo_coords_unit, st.coords_unit)

        self.spin_charge.setValue(st.charge)
        self.spin_cores.setValue(st.nprocs)
        self.spin_memory.setValue(st.memory)
        self.spin_verbose.setValue(getattr(st, "verbose", 4))
        self.chk_own_geom.setChecked(st.has_own_geometry)
        self._set_combo_by_data(self.combo_geom_fmt, st.geometry_format)

        # Electronic
        self.combo_basis.setEditText(st.basis)
        self.combo_frozen.setEditText(st.frozen)
        self.combo_xc.setEditText(st.xc)
        self.chk_molden.setChecked(st.molden)
        self.chk_cube.setChecked(st.cube)
        self.spin_cube_points.setValue(st.cube_points)
        self.chk_chelpg.setChecked(getattr(st, "chelpg", False))
        self.chk_density_map.setChecked(getattr(st, "density_map", False))
        self._set_combo_by_data(self.combo_map_type, getattr(st, "density_map_type", "both"))
        self._set_combo_by_data(self.combo_map_plane, getattr(st, "density_map_plane", "AUTO"))

        # Positron
        self.spin_npos.setValue(st.n_pos)
        self.combo_pos_basis.setEditText(st.pos_basis)
        self.txt_psx_set.setText(st.psx_set)
        self.txt_pos_atom.setText(st.pos_atom)
        self._set_combo_by_data(self.combo_pos_mult, st.pos_mult)
        self.combo_pos_mass.setEditText(st.pos_mass)
        self.txt_quantum_site.setText(st.quantum_site)

        # Proton
        self.combo_nuc_basis.setEditText(st.nuc_basis)
        self.txt_quantum_nuc.setText(st.quantum_nuc)
        self.chk_cneo.setChecked(st.cneo)
        self._set_combo_by_data(self.combo_cneo_var, st.cneo_variant)
        self.combo_epc.setEditText(st.epc)

        # Opt & TS
        self._set_combo_by_data(self.combo_opt_engine, st.opt_engine)
        self.spin_max_cycles.setValue(st.opt_max_cycles)
        self._set_combo_by_data(self.combo_ts_mode, st.ts_mode)
        self.combo_irc_dir.setCurrentText(st.irc_direction)
        self.spin_zopt_evals.setValue(st.zopt_max_evals)
        self.txt_zopt_etol.setText(str(st.zopt_etol))

        # Constraints, Pinnings & Equivalences
        pin_vals = getattr(st, "pinnings", []) or []
        self.txt_pinnings.setText(", ".join(str(p) for p in pin_vals) if pin_vals else "")
        c_lines = getattr(st, "constraints", []) or []
        self.txt_constraints.setPlainText("\n".join(c_lines) if c_lines else "")
        eq_vals = getattr(st, "equivalences", []) or []
        self.txt_equivalences.setText("; ".join(eq_vals) if eq_vals else "")

        # Scan
        self.chk_scan.setChecked(st.scan_enabled)
        self._set_combo_by_data(self.combo_scan_type, st.scan_type)
        self.txt_scan_target.setText(st.scan_target)
        self.spin_scan_start.setValue(st.scan_start)
        self.spin_scan_end.setValue(st.scan_end)
        self.spin_scan_steps.setValue(st.scan_steps)

        self._updating_ui = False
        self._apply_reactivity()

    def _save_ui_into_stage(self, idx: int):
        if idx < 0 or idx >= len(self.generator.stages):
            return
        st = self.generator.stages[idx]

        st.jobname = self.txt_jobname.text().strip()
        st.run_type = self.combo_run_type.currentData() or "SP_ENERGY"
        st.method = self.combo_method.currentData() or "HF"
        st.sector = self.combo_sector.currentData() or "ELECTRONIC"
        st.integrals = self.combo_integrals.currentData() or "DF"
        st.coords_unit = self.combo_coords_unit.currentData() or "Angs"

        st.charge = self.spin_charge.value()
        mult_data = self.combo_mult.currentData()
        if mult_data is not None:
            st.mult = int(mult_data)
        st.nprocs = self.spin_cores.value()
        st.memory = self.spin_memory.value()
        st.verbose = self.spin_verbose.value()
        st.has_own_geometry = self.chk_own_geom.isChecked()
        st.geometry_format = self.combo_geom_fmt.currentData() or "xyz"

        # Electronic
        st.basis = self.combo_basis.currentText().strip()
        st.frozen = self.combo_frozen.currentText().strip()
        st.xc = self.combo_xc.currentText().strip()
        st.molden = self.chk_molden.isChecked()
        st.cube = self.chk_cube.isChecked()
        st.cube_points = self.spin_cube_points.value()
        st.chelpg = self.chk_chelpg.isChecked()
        st.density_map = self.chk_density_map.isChecked()
        st.density_map_type = self.combo_map_type.currentData() or "both"
        st.density_map_plane = self.combo_map_plane.currentData() or "AUTO"

        # Positron
        st.n_pos = self.spin_npos.value()
        st.pos_basis = self.combo_pos_basis.currentText().strip()
        st.psx_set = self.txt_psx_set.text().strip()
        st.pos_atom = self.txt_pos_atom.text().strip()
        pmult = self.combo_pos_mult.currentData()
        if pmult is not None:
            st.pos_mult = int(pmult)
        st.pos_mass = self.combo_pos_mass.currentText().strip()
        st.quantum_site = self.txt_quantum_site.text().strip()

        # Proton
        st.nuc_basis = self.combo_nuc_basis.currentText().strip()
        st.quantum_nuc = self.txt_quantum_nuc.text().strip()
        st.cneo = self.chk_cneo.isChecked()
        st.cneo_variant = self.combo_cneo_var.currentData() or "ccd"
        st.epc = self.combo_epc.currentText().strip()

        # Opt & TS
        st.opt_engine = self.combo_opt_engine.currentData() or "GEOMETRIC"
        st.opt_max_cycles = self.spin_max_cycles.value()
        st.ts_mode = self.combo_ts_mode.currentData() or "ts"
        st.irc_direction = self.combo_irc_dir.currentText()
        st.zopt_max_evals = self.spin_zopt_evals.value()
        try:
            st.zopt_etol = float(self.txt_zopt_etol.text())
        except ValueError:
            pass

        # Constraints, Pinnings & Equivalences
        pin_txt = self.txt_pinnings.text().strip()
        st.pinnings = []
        if pin_txt:
            import re
            for x in re.split(r'[,\s]+', pin_txt):
                if x.isdigit():
                    st.pinnings.append(int(x))

        c_txt = self.txt_constraints.toPlainText().strip()
        st.constraints = [line.strip() for line in c_txt.splitlines() if line.strip()]

        eq_txt = self.txt_equivalences.text().strip()
        st.equivalences = [x.strip() for x in eq_txt.split(";") if x.strip()] if eq_txt else []

        # Scan
        st.scan_enabled = self.chk_scan.isChecked()
        st.scan_type = self.combo_scan_type.currentData() or "bond"
        st.scan_target = self.txt_scan_target.text().strip()
        st.scan_start = self.spin_scan_start.value()
        st.scan_end = self.spin_scan_end.value()
        st.scan_steps = self.spin_scan_steps.value()

    def _set_combo_by_data(self, combo: QtWidgets.QComboBox, data: Any):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    # ---- Dynamic Coupling & Physical Chemistry Logic ----
    def _get_stoichiometric_formula(self) -> str:
        """Construct Hill system chemical formula (e.g. H2O, CH4, or Pos_H2O if positron stage)."""
        counts: Dict[str, int] = {}
        for s in self.symbols:
            counts[s] = counts.get(s, 0) + 1
        
        # Hill system: C first, then H, then others alphabetical
        parts = []
        if "C" in counts:
            c_cnt = counts.pop("C")
            parts.append(f"C{c_cnt if c_cnt > 1 else ''}")
        if "H" in counts:
            h_cnt = counts.pop("H")
            parts.append(f"H{h_cnt if h_cnt > 1 else ''}")
        for el in sorted(counts.keys()):
            cnt = counts[el]
            parts.append(f"{el}{cnt if cnt > 1 else ''}")
        formula = "".join(parts) if parts else "Molecule"
        return formula

    def _auto_construct_jobname(self) -> str:
        """Construct the name of the run based on atoms, method, basis set, and runtype."""
        formula = self._get_stoichiometric_formula()
        st = self.generator.stages[self.current_stage_idx]
        
        run_code = self.combo_run_type.currentData() or st.run_type
        # Simplify run_code for file name
        run_map = {
            "SP_ENERGY": "SP",
            "OPT": "OPT",
            "FREQ": "FREQ",
            "GRADIENT": "GRAD",
            "SCAN": "SCAN",
            "TS": "TS",
            "NUMFREQ": "NUMFREQ",
            "ANFREQ": "ANFREQ",
        }
        run_tag = run_map.get(run_code, run_code)
        
        method_tag = self.combo_method.currentData() or st.method
        method_tag = method_tag.replace("(", "").replace(")", "")
        
        sector_tag = self.combo_sector.currentData() or st.sector
        if sector_tag == "POSITRON":
            pbasis = self.combo_pos_basis.currentText().strip() or "PSX"
            prefix = f"Pos_{formula}"
            basis_tag = f"{self.combo_basis.currentText().strip()}_{pbasis}"
        elif sector_tag == "PROTON":
            nbasis = self.combo_nuc_basis.currentText().strip() or "pb4f1"
            prefix = f"NEO_{formula}"
            basis_tag = f"{self.combo_basis.currentText().strip()}_{nbasis}"
        else:
            prefix = formula
            basis_tag = self.combo_basis.currentText().strip()
        
        # Clean up basis_tag for safe filename
        safe_basis = basis_tag.replace("/", "-").replace(" ", "")
        
        base_name = f"{prefix}_{run_tag}_{method_tag}_{safe_basis}"
        # If this name is already taken by another stage in this pipeline, differentiate by stage number
        for i, other in enumerate(self.generator.stages):
            if i != self.current_stage_idx and other.jobname == base_name:
                return f"{base_name}_s{st.number}"
        return base_name

    def _on_jobname_user_edited(self, text: str):
        self._user_edited_jobname = bool(text.strip())

    def _on_regenerate_jobname_clicked(self):
        self._user_edited_jobname = False
        new_name = self._auto_construct_jobname()
        self.txt_jobname.setText(new_name)
        self._save_ui_into_stage(self.current_stage_idx)
        self._update_stage_combo()
        self._refresh_preview()

    def _update_jobname_if_auto(self):
        if not self._user_edited_jobname and not self._updating_ui:
            new_name = self._auto_construct_jobname()
            if self.txt_jobname.text() != new_name:
                self.txt_jobname.setText(new_name)

    def _calculate_n_minus_2_frozen_orbitals(self) -> int:
        """
        Calculate n-2 frozen core orbitals for the current molecule.
        For atom in principal quantum number n (period n):
          freeze sub-shells with principal quantum number <= n-2.
        - n=1 (H, He): 0
        - n=2 (Li - Ne): 0
        - n=3 (Na - Ar): freezes n=1 shell (1s2 = 2 electrons = 1 orbital)
        - n=4 (K - Kr): freezes n<=2 shells (1s2 2s2 2p6 = 10 electrons = 5 orbitals)
        - n=5 (Rb - Xe): freezes n<=3 shells (1s2 2s2 2p6 3s2 3p6 3d10 = 28 electrons = 14 orbitals)
        - n=6 (Cs - Rn): freezes n<=4 shells (28 + 18 + 14 = 60 electrons = 30 orbitals)
        """
        frozen_electrons_map = {
            1: 0,
            2: 0,
            3: 2,   # 1s2
            4: 10,  # 1s2 2s2 2p6
            5: 28,  # 1s2 2s2 2p6 3s2 3p6 3d10
            6: 60,  # n<=4
        }
        total_frozen_orbitals = 0
        for s in self.symbols:
            d = ELEMENT_DATA.get(s, {})
            period = d.get("period", 1)
            elecs = frozen_electrons_map.get(period, 0)
            total_frozen_orbitals += elecs // 2
        return total_frozen_orbitals

    def _on_frozen_changed(self, text: str):
        if self._updating_ui:
            return
        t = text.strip().lower()
        if t in ("n-2", "n - 2", "n_2"):
            n_frozen = self._calculate_n_minus_2_frozen_orbitals()
            self._updating_ui = True
            self.combo_frozen.setEditText(str(n_frozen))
            self._updating_ui = False
            self.combo_frozen.setToolTip(f"n-2 frozen core: {n_frozen} orbital(s) frozen based on periods of atoms")
        elif t == "auto":
            self.combo_frozen.setToolTip("auto: standard chemical core (e.g. 1s for Li-Ne, 1s2s2p for Na-Ar)")
        self._on_field_changed()

    def _get_total_nuclear_charge(self) -> int:
        total = 0
        for s in self.symbols:
            d = ELEMENT_DATA.get(s, {})
            total += d.get("atomic_number", 0)
        return total

    def _update_multiplicities(self):
        charge = self.spin_charge.value()
        z_tot = self._get_total_nuclear_charge()
        electrons = z_tot - charge

        self.lbl_electron_count.setText(f"Nuclear Z: {z_tot} | Charge: {charge:+d} | Electrons: {electrons}")

        st = self.generator.stages[self.current_stage_idx]
        current_mult = st.mult

        self.combo_mult.blockSignals(True)
        self.combo_mult.clear()

        # Allowed multiplicities
        if electrons % 2 == 0:
            allowed = [1, 3, 5, 7]
            names = ["1 (Singlet)", "3 (Triplet)", "5 (Quintet)", "7 (Septet)"]
        else:
            allowed = [2, 4, 6, 8]
            names = ["2 (Doublet)", "4 (Quartet)", "6 (Sextet)", "8 (Octet)"]

        for m, name in zip(allowed, names):
            self.combo_mult.addItem(name, m)

        # Keep current if valid, else default to ground state
        if current_mult in allowed:
            self._set_combo_by_data(self.combo_mult, current_mult)
        else:
            self.combo_mult.setCurrentIndex(0)
            st.mult = allowed[0]

        self.combo_mult.blockSignals(False)

    # ---- Reactive Activation / Deactivation ----
    def _apply_reactivity(self):
        """Enable / disable options based on chosen directives."""
        sector = self.combo_sector.currentData()
        method = self.combo_method.currentData()
        run_type = self.combo_run_type.currentData()

        # Update tab visual indicators (all tabs remain clickable to let users switch sectors easily)
        self.tabs.setTabText(2, "✨ Positron (Active)" if sector == "POSITRON" else "✨ Positron")
        self.tabs.setTabText(3, "⚛️ Proton / NEO (Active)" if sector == "PROTON" else "⚛️ Proton / NEO")
        self.tabs.setTabText(4, "🎯 Opt / TS (Active)" if run_type in ("OPT", "TS") else "🎯 Opt / TS")
        self.tabs.setTabText(5, "📈 Scan (Active)" if (run_type == "SCAN" or self.chk_scan.isChecked()) else "📈 Scan")

        # Positron options
        self.combo_pos_mult.setEnabled(self.spin_npos.value() >= 2)

        # CNEO options
        self.combo_cneo_var.setEnabled(self.chk_cneo.isChecked())

        # TS options
        is_ts = run_type == "TS"
        self.combo_ts_mode.setEnabled(is_ts)
        self.combo_irc_dir.setEnabled(is_ts and self.combo_ts_mode.currentData() == "irc")

        # ZOPT options
        is_zopt = (run_type == "OPT") and (self.combo_opt_engine.currentData() == "ZOPT")
        self.grp_zopt.setEnabled(is_zopt)

        # DFT options (XC only active if method == DFT)
        self.combo_xc.setEnabled(method == "DFT")

        # Cube points only active if cube is checked
        self.spin_cube_points.setEnabled(self.chk_cube.isChecked())

        # Constraints & Equivalences (only active during OPT run_type)
        is_opt = (run_type == "OPT")
        self.grp_constraints.setEnabled(is_opt)

        # Density map options: reactively adapt options to Electronic vs Positronic sector
        has_pos = (sector == "POSITRON")
        current_map_type = self.combo_map_type.currentData()
        self.combo_map_type.blockSignals(True)
        self.combo_map_type.clear()
        if has_pos:
            self.combo_map_type.addItem("both (Electron & Positron)", "both")
            self.combo_map_type.addItem("esp (Electrostatic Potential)", "esp")
            self.combo_map_type.addItem("electron (Electron Density)", "electron")
            self.combo_map_type.addItem("positron (Positron Density)", "positron")
            target_type = current_map_type if current_map_type in ("both", "esp", "electron", "positron") else "both"
        else:
            self.combo_map_type.addItem("esp (Electrostatic Potential)", "esp")
            self.combo_map_type.addItem("electron (Electron Density)", "electron")
            target_type = current_map_type if current_map_type in ("esp", "electron") else "esp"
        self._set_combo_by_data(self.combo_map_type, target_type)
        self.combo_map_type.blockSignals(False)

        # Scan fields
        scan_active = (run_type == "SCAN") or self.chk_scan.isChecked()
        self.combo_scan_type.setEnabled(scan_active)
        self.txt_scan_target.setEnabled(scan_active)
        self.spin_scan_start.setEnabled(scan_active)
        self.spin_scan_end.setEnabled(scan_active)
        self.spin_scan_steps.setEnabled(scan_active)

    # ---- Event Handlers ----
    def _on_sector_changed(self):
        if self._updating_ui:
            return
        sector = self.combo_sector.currentData()
        if sector == "POSITRON":
            self.tabs.setCurrentIndex(2)
        elif sector == "PROTON":
            self.tabs.setCurrentIndex(3)
        self._on_bang_changed()

    def _on_tab_changed(self, idx: int):
        if self._updating_ui:
            return
        # If user clicks the Positron tab, automatically switch Sector to POSITRON if not already
        if idx == 2 and self.combo_sector.currentData() != "POSITRON":
            self._set_combo_by_data(self.combo_sector, "POSITRON")
            self._on_bang_changed()
        # If user clicks the Proton tab, automatically switch Sector to PROTON if not already
        elif idx == 3 and self.combo_sector.currentData() != "PROTON":
            self._set_combo_by_data(self.combo_sector, "PROTON")
            self._on_bang_changed()
        # If user clicks the Electronic tab, switch Sector to ELECTRONIC if it was Positron or Proton
        elif idx == 1 and self.combo_sector.currentData() != "ELECTRONIC":
            self._set_combo_by_data(self.combo_sector, "ELECTRONIC")
            self._on_bang_changed()

    def _on_bang_changed(self):
        if self._updating_ui:
            return
        self._apply_reactivity()
        self._update_jobname_if_auto()
        self._save_ui_into_stage(self.current_stage_idx)
        self._update_stage_combo()
        self._refresh_preview()

    def _on_charge_changed(self):
        if self._updating_ui:
            return
        self._update_multiplicities()
        self._on_field_changed()

    def _on_mult_changed(self):
        if self._updating_ui:
            return
        self._on_field_changed()

    def _on_pin_selected_atoms(self):
        """Read selected atom indices from parent main window viewer and add to %pin."""
        parent = self.parent()
        viewer = getattr(parent, "viewer", None)
        if not viewer:
            QtWidgets.QMessageBox.information(self, "Pin Atoms", "No active 3D viewer found.")
            return

        selected_indices = list(getattr(viewer, "multi_selected", set()) or set())
        if getattr(viewer, "selected_atom_idx", None) is not None and viewer.selected_atom_idx not in selected_indices:
            selected_indices.append(viewer.selected_atom_idx)

        if not selected_indices:
            QtWidgets.QMessageBox.information(
                self,
                "No Atoms Selected",
                "Please select one or more atoms in the 3D viewer (by clicking or Shift+dragging) first."
            )
            return

        # 1-based indexing for newBRORSEN
        pin_atoms = sorted(set(i + 1 for i in selected_indices))
        curr_txt = self.txt_pinnings.text().strip()
        existing = [int(x) for x in curr_txt.replace(",", " ").split() if x.isdigit()]
        combined = sorted(set(existing + pin_atoms))
        self.txt_pinnings.setText(", ".join(str(x) for x in combined))
        self._on_field_changed()

    def _on_show_geom_format_info(self):
        """Show explanatory dialog detailing newBRORSEN geometry format options."""
        QtWidgets.QMessageBox.information(
            self,
            "Geometry Format Options (* xyz vs * zmat)",
            "<h3>newBRORSEN Geometry Format Options</h3>"
            "<p><b>1. Cartesian Coordinates (<code>* xyz</code>):</b></p>"
            "<ul>"
            "  <li>Standard 3D Cartesian coordinates: <code>Symbol  X  Y  Z</code>.</li>"
            "  <li>Units set by the 5th token in the bang line: <b>Angs</b> (Ångströms, standard) or <b>Bohr</b> (Atomic units, 1 a.u. ≈ 0.529177 Å).</li>"
            "  <li>Recommended for standard molecular systems, periodic clusters, geomeTRIC optimizations, and transition states.</li>"
            "</ul>"
            "<p><b>2. Z-Matrix Coordinates (<code>* zmat</code>):</b></p>"
            "<ul>"
            "  <li>Internal coordinates using bond distances (Å), valence angles (°), and dihedral angles (°).</li>"
            "  <li>Supports symbolic parameter variables (e.g. <code>r_CH = 1.09</code>) placed in the variables block.</li>"
            "  <li>Enables <b>ZOPT</b> (gradient-free direct Nelder-Mead optimization on internal coordinates).</li>"
            "</ul>"
            "<p style='color: #64748b;'><i>Tip: AGYMolden automatically generates exact Z-matrices from Cartesian 3D coordinates on the fly.</i></p>"
        )

    def _on_npos_changed(self):
        if self._updating_ui:
            return
        self.combo_pos_mult.setEnabled(self.spin_npos.value() >= 2)
        self._on_field_changed()

    def _on_cneo_toggled(self, checked: bool):
        self.combo_cneo_var.setEnabled(checked)
        self._on_field_changed()

    def _on_opt_engine_changed(self):
        if self._updating_ui:
            return
        self._apply_reactivity()
        self._on_field_changed()

    def _on_ts_mode_changed(self):
        if self._updating_ui:
            return
        self._apply_reactivity()
        self._on_field_changed()

    def _on_scan_toggled(self, checked: bool):
        if self._updating_ui:
            return
        self._apply_reactivity()
        self._on_field_changed()

    def _on_geom_fmt_changed(self):
        if self._updating_ui:
            return
        fmt = self.combo_geom_fmt.currentData()
        # If user chooses zmat but zatoms list is empty, build zatoms from Cartesian coordinates
        if fmt == "zmat" and not self.zatoms and len(self.symbols) > 0:
            try:
                from agymolden.zmatrix import cartesian_to_zmatrix
                self.zatoms = cartesian_to_zmatrix(self.symbols, self.coords)
            except Exception as e:
                print(f"Notice: Failed to build zatoms on format change: {e}")
        self._on_field_changed()

    def _on_jobname_text_changed(self, text: str):
        if self._updating_ui:
            return
        self._save_ui_into_stage(self.current_stage_idx)
        self._update_stage_combo()
        self._refresh_preview()

    def _on_field_changed(self):
        if self._updating_ui:
            return
        self._update_jobname_if_auto()
        self._save_ui_into_stage(self.current_stage_idx)
        self._refresh_preview()

    def _on_toggle_edit_preview(self, checked: bool):
        self.txt_preview.setReadOnly(not checked)
        self.combo_keymap.setEnabled(checked)
        self.btn_reset_preview.setEnabled(checked or self._manual_preview_edit)
        if checked:
            is_white = self.parent() and getattr(self.parent(), "viewer", None) and getattr(self.parent().viewer, "bg_color_mode", "") == "white"
            if is_white:
                self.txt_preview.setStyleSheet("background-color: #f1f5f9; color: #047857; border: 1px solid #10b981;")
            else:
                self.txt_preview.setStyleSheet("background-color: #1a1a24; color: #6ee7b7; border: 1px solid #10b981;")
            self.txt_preview.set_keymap_mode(self.combo_keymap.currentData() or "plain")
        else:
            self.txt_preview.setStyleSheet("")
            self.txt_preview.set_keymap_mode("plain")
            self.lbl_editor_mode.setText("")

    def _on_keymap_changed(self):
        if self.chk_edit_preview.isChecked():
            mode = self.combo_keymap.currentData() or "plain"
            self.txt_preview.set_keymap_mode(mode)

    def _on_preview_text_edited(self):
        if self.chk_edit_preview.isChecked():
            self._manual_preview_edit = True
            self.btn_reset_preview.setEnabled(True)

    def _on_reset_preview(self):
        self._manual_preview_edit = False
        self.chk_edit_preview.setChecked(False)
        self.btn_reset_preview.setEnabled(False)
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setStyleSheet("")
        self._refresh_preview()

    def _refresh_preview(self):
        if self._manual_preview_edit:
            return  # Do not overwrite user's manual edits
        text = self.generator.generate_input_text(
            symbols=self.symbols,
            coords=self.coords,
            zatoms=self.zatoms,
            variables=self.variables
        )
        self.txt_preview.setPlainText(text)

    def _on_open_input_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open newBRORSEN Input File",
            "",
            "Input Files (*.inp *.in *.txt);;All Files (*)"
        )
        if not path:
            return
        self.load_input_file(path)

    def load_input_file(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Reading File", f"Could not read {path}:\n{e}")
            return

        stages, geom_info = self.generator.load_from_text(content)
        if not stages:
            QtWidgets.QMessageBox.warning(self, "Parse Warning", f"No calculation stages could be parsed from:\n{path}")
            return

        # If file contains geometry, load it
        if geom_info and geom_info.get("lines"):
            lines = geom_info["lines"]
            fmt = geom_info.get("format", "xyz")
            if fmt == "zmat":
                from agymolden.zmatrix import parse_zmatrix_text, zmatrix_to_cartesian
                try:
                    zatoms, vars_dict = parse_zmatrix_text("\n".join(lines))
                    if zatoms:
                        self.zatoms = zatoms
                        self.variables = vars_dict
                        self.symbols = [a.symbol for a in zatoms]
                        self.coords = zmatrix_to_cartesian(zatoms)
                except Exception as ex:
                    print(f"Warning parsing zmat from inp: {ex}")
            else:
                from agymolden.zmatrix import parse_xyz, cartesian_to_zmatrix
                try:
                    xyz_txt = f"{len(lines)}\nfrom inp\n" + "\n".join(lines)
                    syms, crds = parse_xyz(xyz_txt, is_path=False)
                    if syms and len(crds) == len(syms):
                        self.symbols = syms
                        self.coords = crds
                        try:
                            self.zatoms = cartesian_to_zmatrix(syms, crds)
                        except Exception:
                            self.zatoms = []
                except Exception as ex:
                    print(f"Warning parsing xyz from inp: {ex}")

        self.current_stage_idx = 0
        self._user_edited_jobname = True  # Preserve imported job name
        self._update_stage_combo()
        self._load_stage_into_ui(0)
        self._update_multiplicities()
        self._refresh_preview()
        QtWidgets.QMessageBox.information(
            self,
            "Input Loaded",
            f"Successfully loaded {len(stages)} stage(s) from:\n{path}"
        )

    def _on_copy_input(self):
        text = self.txt_preview.toPlainText()
        QtWidgets.QApplication.clipboard().setText(text)
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), "Input copied to clipboard!")

    def _validate_stages_preflight(self) -> List[str]:
        """Validate stage settings (charge/multiplicity parity, positron/nuclear basis, geometry) and return warnings."""
        warnings: List[str] = []
        if not self.symbols:
            warnings.append("No atoms are present in the molecular structure.")
            return warnings

        # Check geometry clashes via chemistry_checker (when non-degenerate 3D coordinates are provided)
        try:
            if self.coords is not None and len(self.coords) >= 2 and float(np.max(np.abs(self.coords))) > 1e-5:
                from ..chemistry_checker import check_calculation_sanity
                first_st = self.generator.stages[0] if self.generator.stages else None
                geom_check = check_calculation_sanity(
                    symbols=self.symbols,
                    coords=self.coords,
                    charge=getattr(first_st, "charge", 0),
                    multiplicity=getattr(first_st, "mult", 1),
                    basis_set=getattr(first_st, "basis", None),
                    qm_package="newBRORSEN",
                )
                for err in geom_check.get("errors", []):
                    if "Atomic clash" in err:
                        warnings.append(err)
                for wrn in geom_check.get("warnings", []):
                    if "short interatomic" in wrn or "Heavy element" in wrn:
                        warnings.append(wrn)
        except Exception as e:
            log.debug("Sanity geometry check error in newBRORSEN dialog: %s", e)

        z_tot = self._get_total_nuclear_charge()
        for idx, st in enumerate(self.generator.stages):
            stage_label = f"Stage {idx + 1} ({st.jobname})"
            electrons = z_tot - st.charge
            if electrons <= 0:
                warnings.append(f"{stage_label}: Total electron count is {electrons} (Z={z_tot}, charge={st.charge:+d}).")
            else:
                expected_even_mult = (electrons % 2 == 1)
                is_even_mult = (st.mult % 2 == 0)
                if expected_even_mult != is_even_mult:
                    warnings.append(
                        f"{stage_label}: Charge ({st.charge:+d}) gives {electrons} electrons, which is incompatible with Multiplicity = {st.mult}."
                    )
            if st.sector == "POSITRON":
                if getattr(st, "npos", 1) < 1:
                    warnings.append(f"{stage_label}: Sector is POSITRON, but number of positrons (npos) < 1.")
                if not getattr(st, "pos_basis", "").strip():
                    warnings.append(f"{stage_label}: Sector is POSITRON, but no Positron Basis Set is specified.")
            elif st.sector == "PROTON":
                if not getattr(st, "nuc_basis", "").strip():
                    warnings.append(f"{stage_label}: Sector is PROTON (NEO), but no Nuclear Basis Set is specified.")
        return warnings

    def _on_save_input(self) -> Optional[str]:
        warnings = self._validate_stages_preflight()
        if warnings:
            res = QtWidgets.QMessageBox.warning(
                self,
                "Input Validation Notice",
                "The following potential issues were detected in your input:\n\n• "
                + "\n• ".join(warnings)
                + "\n\nDo you want to proceed anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if res != QtWidgets.QMessageBox.StandardButton.Yes:
                return None

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save newBRORSEN Input File",
            f"{self.generator.stages[0].jobname}.inp",
            "Input Files (*.inp);;All Files (*)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self.txt_preview.toPlainText())
                QtWidgets.QMessageBox.information(self, "Saved", f"Input file saved successfully to:\n{path}")
                return path
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error Saving File", str(e))
        return None

    def _on_run_calculation(self):
        """Save input and launch the live job monitor."""
        warnings = self._validate_stages_preflight()
        if warnings:
            res = QtWidgets.QMessageBox.warning(
                self,
                "Input Validation Notice",
                "The following potential issues were detected in your input:\n\n• "
                + "\n• ".join(warnings)
                + "\n\nDo you want to launch the calculation anyway?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            )
            if res != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # 1. Ask where to save or save to default working directory
        jobname = self.generator.stages[0].jobname if self.generator.stages else "calculation"
        default_dir = os.getcwd()
        default_path = os.path.join(default_dir, f"{jobname}.inp")

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Input File & Run Calculation",
            default_path,
            "Input Files (*.inp);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.txt_preview.toPlainText())
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error Saving File", f"Could not write input file:\n{e}")
            return

        # 2. Launch Job Monitor via parent main window or standalone
        main_win = self.parent()
        if hasattr(main_win, "launch_calculation_job"):
            self.accept()
            main_win.launch_calculation_job(path, runner_type="brorsen")
        else:
            from agymolden.calc_runner import JobMonitorDialog, BrorsenLaunchDialog
            launch_dlg = BrorsenLaunchDialog(self, path)
            if launch_dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
                return
            py_bin, wrapper_path, extra_flags, live_3d, remote_host = launch_dlg.get_launch_configuration()
            if remote_host is not None:
                from agymolden.calc_runner.remote_ssh import RemoteSshRunner
                runner = RemoteSshRunner(self)
                runner.configure_remote(host_cfg=remote_host, remote_script=wrapper_path, extra_flags=extra_flags)
            else:
                from agymolden.calc_runner import BrorsenRunner
                runner = BrorsenRunner(self)
                runner.configure(python_bin=py_bin, wrapper_path=wrapper_path, extra_flags=extra_flags)
            monitor = JobMonitorDialog(self, runner=runner)
            monitor.chk_live_3d.setChecked(live_3d)
            monitor.run_calculation(path, runner_type="brorsen")

    def _on_chain_qm_stage(self):
        """Save this newBRORSEN input and chain into QM Calculation Setup (ORCA/PySCF/Gaussian)."""
        saved_path = self._on_save_input()
        if not saved_path:
            return
        main_win = self.parent()
        self.accept()
        if main_win and hasattr(main_win, "_on_open_qm_dialog"):
            main_win._on_open_qm_dialog(target_code="orca")

