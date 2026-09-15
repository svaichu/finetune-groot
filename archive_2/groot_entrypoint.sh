#!/bin/bash

echo "Starting entrypoint script..."
set -eo pipefail
shopt -s expand_aliases

cd /workspace/c3po/source/c3po_utils
# source ${HOME}/.bashrc && \
pip install -e . --quiet

cd /workspace
git clone https://github.com/svaichu/rldata.git
cd rldata
pip install -e . --quiet

python -c "from opencv_fixer import AutoFix; AutoFix()"
cd /workspace/c3po

echo "All dependencies installed successfully."

# if deploy arg is passed, run this file
if [ "${1:-}" = "job" ]; then
    python3 scripts/vla/play-groot.py
fi

if [ "$1" = "dev" ]; then
    echo "Running in dev mode"
    cd /workspace/c3po # BUG no working when running in detached mode
    /bin/bash
fi

# if [ "$1" = "job" ]; then
#     echo "Running in job mode"
#     cd /workspace/c3po # BUG no working when running in detached mode
#     python3 scripts/vla/play-groot.py
#     echo "Job completed successfully."
# fi

