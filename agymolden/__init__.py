"""Package initialization and CLI entrypoint for AgyMolden."""

import argparse
import logging
import logging.handlers
import os
import sys
import traceback

__version__ = "1.0.1"

_REQUIRED = {
    "PyQt6": "PyQt6",
    "OpenGL": "PyOpenGL",
    "numpy": "numpy",
    "scipy": "scipy",
}


def _check_dependencies() -> None:
    """Exit with a readable message if a required package is missing."""
    import importlib

    missing = []
    for module, pip_name in _REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(pip_name)
    if missing:
        sys.stderr.write(
            "AgyMolden cannot start: missing Python package(s): "
            + ", ".join(missing)
            + "\nInstall them with:\n    pip install "
            + " ".join(missing)
            + "\n(or run: pip install -r requirements.txt)\n"
        )
        sys.exit(1)


def _log_file_path() -> str:
    base = os.environ.get("XDG_STATE_HOME") or os.path.join(os.path.expanduser("~"), ".local", "state")
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA", base)
    elif sys.platform == "darwin":
        base = os.path.join(os.path.expanduser("~"), "Library", "Logs")
    return os.path.join(base, "agymolden", "agymolden.log")


def _setup_logging(debug: bool) -> None:
    """Log to a rotating file (and to the console with --debug)."""
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        path = _log_file_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(path, maxBytes=500_000, backupCount=2, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        pass  # logging must never prevent the program from starting
    if debug:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)


def _install_excepthook(QtWidgets) -> None:
    """Show unexpected errors in a dialog (the GUI exe has no console)."""
    _in_hook = False

    def hook(exc_type, exc, tb):
        nonlocal _in_hook
        if _in_hook:
            # Prevent recursive repaint / message box loops
            return
        _in_hook = True
        try:
            text = "".join(traceback.format_exception(exc_type, exc, tb))
            logging.getLogger("agymolden").error("Unexpected error:\n%s", text)
            sys.stderr.write(text)
            if QtWidgets.QApplication.instance() is not None:
                box = QtWidgets.QMessageBox()
                box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                box.setWindowTitle("AgyMolden - Unexpected error")
                box.setText(f"{exc_type.__name__}: {exc}")
                box.setDetailedText(text)
                box.exec()
        finally:
            _in_hook = False

    sys.excepthook = hook


def main(argv=None):
    raw_args = sys.argv[1:] if argv is None else argv
    if raw_args and raw_args[0] in ("parse", "gen", "check", "convert", "bundle"):
        from .cli import build_cli_parser, cmd_parse, cmd_gen, cmd_check, cmd_convert, cmd_bundle
        cli_p = build_cli_parser()
        cli_args = cli_p.parse_args(raw_args)
        if cli_args.subcommand == "parse":
            sys.exit(cmd_parse(cli_args))
        elif cli_args.subcommand == "gen":
            sys.exit(cmd_gen(cli_args))
        elif cli_args.subcommand == "check":
            sys.exit(cmd_check(cli_args))
        elif cli_args.subcommand == "convert":
            sys.exit(cmd_convert(cli_args))
        elif cli_args.subcommand == "bundle":
            sys.exit(cmd_bundle(cli_args))

    parser = argparse.ArgumentParser(
        prog="agymolden", description="AgyMolden - Z-Matrix molecular builder & quantum chemistry viewer"
    )
    parser.add_argument("file", nargs="?", help="molecule file to open (.zmat, .inp, .gjf, .xyz, .cube, .molden, .out, .log)")
    parser.add_argument("--render-image", dest="render_image", help="render publication-quality PNG offscreen and exit without opening GUI")
    parser.add_argument("--width", type=int, default=1920, help="image render width in pixels (default: 1920)")
    parser.add_argument("--height", type=int, default=1080, help="image render height in pixels (default: 1080)")
    parser.add_argument("--background", dest="background", choices=["dark", "white", "transparent"], default="dark", help="render image background color: dark, white, or transparent (default: dark)")
    parser.add_argument("--transparent", action="store_true", help="render image with transparent background (shorthand for --background transparent)")
    parser.add_argument("--debug", action="store_true", help="verbose logging to the console")
    parser.add_argument("--version", action="version", version=f"AgyMolden {__version__}")
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    _setup_logging(args.debug)
    _check_dependencies()

    from PyQt6 import QtWidgets, QtGui, QtCore
    from .main_window import MainWindow

    # Enable shared OpenGL contexts before creating QApplication to prevent context recreation flicker
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

    # Request a modern OpenGL core context (4.1 on macOS, 3.3 on Linux/Windows)
    # and a high-quality surface (24-bit depth + 4x MSAA + double buffering)
    # with 0-bit alpha for the window framebuffer. If the window framebuffer has an alpha channel,
    # semi-transparent objects (such as orbital lobes) blend alpha into the desktop compositor,
    # causing background windows / desktop wallpaper to bleed through the molecule!
    fmt = QtGui.QSurfaceFormat()
    if sys.platform == "darwin":
        fmt.setVersion(4, 1)
    else:
        fmt.setVersion(3, 3)
    fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setAlphaBufferSize(0)
    fmt.setSamples(4)
    fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setSwapInterval(1)
    QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    app = QtWidgets.QApplication([sys.argv[0]] + qt_args)
    app.setApplicationName("AgyMolden")
    app.setOrganizationName("AgyMolden")
    app.setApplicationVersion(__version__)
    _install_excepthook(QtWidgets)

    # Standard application font, using system font cleanly across platforms
    font = QtGui.QFont(app.font())
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()

    # Headless / Batch rendering mode
    if args.render_image:
        if not args.file:
            sys.stderr.write("Error: --render-image requires an input file (e.g., agymolden mol.xyz --render-image out.png)\n")
            sys.exit(1)
        window.resize(args.width, args.height)
        window.show()
        window.open_path(args.file)
        QtWidgets.QApplication.processEvents()
        is_transparent = args.transparent or (args.background == "transparent")
        bg_mode = "white" if args.background == "white" else "dark"
        window.export_image(args.render_image, transparent=is_transparent, bg_mode=bg_mode, scale=1, clean_only=True)
        bg_desc = "transparent" if is_transparent else bg_mode
        print(f"Successfully rendered {args.width}x{args.height} image ({bg_desc} background) to: {args.render_image}")
        sys.exit(0)

    window.show()
    if args.file:
        window.open_path(args.file)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
