"""Read-only log output box used by the application to display messages."""

from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget
from PySide6.QtGui import QFont
from PySide6.QtCore import QTimer


class StatusLogWidget(QWidget):
    """Simple wrapper around ``QTextEdit`` that defaults to monospaced, read-only."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._text_edit = QTextEdit()
        self._text_edit.setReadOnly(True)
        self._text_edit.setFont(QFont("Monospace", 9))

        # Spinner state
        self._spinner_active = False
        self._spinner_message = ""
        self._spinner_frame = 0
        self._spinner_chars = ["|", "/", "-", "\\"]
        self._spinner_timer = QTimer(self)
        self._spinner_timer.timeout.connect(self._update_spinner)

        layout = QVBoxLayout()
        layout.addWidget(self._text_edit)
        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def append(self, text: str) -> None:  # noqa: D401 (docstring style)
        """Append *text* to the log display."""
        # If spinner is active, remove the spinner line first
        if self._spinner_active:
            self._remove_spinner_line()
        self._text_edit.append(text)
        # Re-add spinner if it was active
        if self._spinner_active:
            self._add_spinner_line()

    def clear(self) -> None:
        """Clear the log widget."""
        self.stop_spinner()
        self._text_edit.clear()

    def widget(self) -> QTextEdit:  # pragma: no cover
        """Return the underlying ``QTextEdit`` instance."""
        return self._text_edit
    
    def start_spinner(self, message: str = "Processing") -> None:
        """Start a text-based spinner animation with the given message."""
        if self._spinner_active:
            self.stop_spinner()
        self._spinner_active = True
        self._spinner_message = message
        self._spinner_frame = 0
        self._add_spinner_line()
        self._spinner_timer.start(100)  # Update every 100ms
    
    def stop_spinner(self) -> None:
        """Stop the spinner animation."""
        if self._spinner_active:
            self._spinner_active = False
            self._spinner_timer.stop()
            self._remove_spinner_line()
    
    def _add_spinner_line(self) -> None:
        """Add the spinner line at the end of the log."""
        spinner_char = self._spinner_chars[self._spinner_frame % len(self._spinner_chars)]
        spinner_text = f"{self._spinner_message} {spinner_char}"
        
        # Use cursor to add spinner line without extra newline from append()
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        
        # Check if we need to add a newline before the spinner
        # Only add newline if document is not empty and last character is not a newline
        text = self._text_edit.toPlainText()
        if text and not text.endswith('\n'):
            cursor.insertText('\n')
        elif not text:
            # Empty document, no newline needed
            pass
        # If text ends with newline, cursor is already positioned correctly
        
        # Insert the spinner text (without trailing newline - QTextEdit handles that)
        cursor.insertText(spinner_text)
        
        # Move cursor to end and scroll
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
    
    def _update_spinner_line(self) -> None:
        """Update the spinner line in place without removing/re-adding."""
        spinner_char = self._spinner_chars[self._spinner_frame % len(self._spinner_chars)]
        spinner_text = f"{self._spinner_message} {spinner_char}"
        
        cursor = self._text_edit.textCursor()
        # Move to the end of the document
        cursor.movePosition(cursor.MoveOperation.End)
        
        # Move to the start of the last line (the spinner line)
        cursor.movePosition(cursor.MoveOperation.StartOfLine)
        # Select to the end of the line (not including the newline)
        cursor.movePosition(cursor.MoveOperation.EndOfLine, cursor.MoveMode.KeepAnchor)
        
        # Replace the selected text (the spinner line content) with the new spinner text
        cursor.insertText(spinner_text)
        
        # Move cursor back to end and scroll
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
    
    def _remove_spinner_line(self) -> None:
        """Remove the last line (spinner) from the log."""
        text = self._text_edit.toPlainText()
        if not text.strip():
            # Empty log, nothing to remove
            return
        
        # Use cursor to remove the last line, including the newline before it
        cursor = self._text_edit.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        
        # Move to start of last line (the spinner line)
        cursor.movePosition(cursor.MoveOperation.StartOfLine)
        
        # If not at the start of document, move left to include the newline before this line
        start_pos = cursor.position()
        if start_pos > 0:
            cursor.movePosition(cursor.MoveOperation.Left)
        
        # Select from this position (newline before spinner, or start of doc) to the end
        cursor.movePosition(cursor.MoveOperation.End, cursor.MoveMode.KeepAnchor)
        
        # Remove the selected text
        cursor.removeSelectedText()
        
        # Set cursor back to end
        cursor.movePosition(cursor.MoveOperation.End)
        self._text_edit.setTextCursor(cursor)
    
    def _update_spinner(self) -> None:
        """Update the spinner animation frame."""
        if self._spinner_active:
            self._spinner_frame += 1
            self._update_spinner_line()

 