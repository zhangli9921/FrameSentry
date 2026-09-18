"""Application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    """Launch the FrameSentry GUI."""
    from PySide6.QtWidgets import QApplication

    from framesentry.core.logging_setup import log_environment_banner, setup_logging
    from framesentry.ui.main_window import MainWindow

    setup_logging()
    log_environment_banner()
    app = QApplication(sys.argv)
    app.setApplicationName("FrameSentry")
    app.setOrganizationName("FrameSentry")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
