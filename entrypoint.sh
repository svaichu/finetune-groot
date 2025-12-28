#!/bin/bash

set -euo pipefail

MODE="${1:-}"
if [ $# -gt 0 ]; then
  shift
fi


python -c "from opencv_fixer import AutoFix; AutoFix()"



if [ "$MODE" = "job" ]; then
  echo "Running job mode"
  python main.py "$@"
  exit 0
fi

if [ "$MODE" = "dev" ]; then
  echo "Running dev mode"
  exec /bin/bash
fi

if [ -z "$MODE" ]; then
  exec /bin/bash
fi

exec "$MODE" "$@"
