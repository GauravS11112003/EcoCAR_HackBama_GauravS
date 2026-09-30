#!/usr/bin/env bash
# One-time setup on Linux/macOS (client only on macOS): creates .venv, installs the harness and
# the CARLA client matching the server, then runs a self-check.
#   CARLA_ROOT=~/CARLA_0.9.16 ./scripts/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
CARLA_VERSION="${CARLA_VERSION:-0.9.16}"

PY=""
for v in 3.12 3.11 3.10; do
  if command -v "python$v" >/dev/null 2>&1; then PY="python$v"; break; fi
done
[ -n "$PY" ] || { echo "Python 3.10-3.12 required"; exit 1; }
echo "Using $PY"

[ -d .venv ] || "$PY" -m venv .venv
VPY=.venv/bin/python
"$VPY" -m pip install --upgrade pip -q
"$VPY" -m pip install -e ".[dev]" -q

if ! "$VPY" -m pip install "carla==$CARLA_VERSION" -q 2>/dev/null; then
  TAG="cp$("$VPY" -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
  WHL=$(ls "${CARLA_ROOT:-$HOME/CARLA_$CARLA_VERSION}"/PythonAPI/carla/dist/carla-*-"$TAG"-*.whl 2>/dev/null | head -n1 || true)
  [ -n "$WHL" ] || { echo "No carla==$CARLA_VERSION wheel for $TAG (set CARLA_ROOT)"; exit 1; }
  "$VPY" -m pip install "$WHL" -q
fi

"$VPY" -m autocross check || true
echo
echo "Setup complete. Activate with:  source .venv/bin/activate"
