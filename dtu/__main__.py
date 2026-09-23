"""``python -m dtu``: GUI without arguments, CLI otherwise."""
import multiprocessing
import sys


def main() -> int:
    multiprocessing.freeze_support()  # subtitle upscaling uses worker processes
    if len(sys.argv) > 1:
        from .cli import main as cli_main
        return cli_main()
    try:
        from .gui import main as gui_main
    except ImportError as e:  # tkinter missing (e.g. minimal Linux Python)
        sys.stderr.write(f"GUI를 시작할 수 없습니다 ({e}). 'python -m dtu --help'로 CLI를 사용하세요.\n")
        return 1
    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
