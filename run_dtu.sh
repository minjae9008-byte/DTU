#!/bin/sh
# DTU launcher for macOS / Linux (GUI without arguments, CLI with arguments).
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY -c "import numpy" 2>/dev/null || $PY -m pip install --user -r requirements.txt
exec $PY -m dtu "$@"
