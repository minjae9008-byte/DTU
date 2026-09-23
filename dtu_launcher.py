"""Entry script for PyInstaller builds (see .github/workflows/windows-build.yml)."""
import sys

from dtu.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
