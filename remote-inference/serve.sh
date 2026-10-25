#!/usr/bin/env bash
# Cloud server side of remote GR00T inference for ElRobot (see plan.md).
# Starts gr00t/eval/run_gr00t_server.py against a checkpoint and prints
# tunnel instructions so the local client (running Station) can reach it.

set -euo pipefail

GR00T_REPO="${GR00T_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/vendor/Isaac-GR00T}"

MODEL_PATH="nvidia/GR00T-N1.7-3B"
EMBODIMENT_TAG="new_embodiment"
DEVICE="cuda"
HOST="0.0.0.0"
PORT="5555"
EXTRA_ARGS=()

usage() {
    cat <<'EOF'
Usage: bash remote-inference/serve.sh \
  --model-path <hf-repo-or-local-dir> \
  [--embodiment-tag <tag>] \
  [--device <cuda|cpu>] \
  [--host <addr>] \
  [--port <port>] \
  [-- <extra run_gr00t_server.py args>...]

--model-path     nvidia/GR00T-N1.7-3B (downloaded from HF) or a local
                 fine-tuned checkpoint directory for ElRobot's embodiment.
--embodiment-tag must match the checkpoint's training embodiment tag.

Expose --port to the local client with an SSH tunnel, run FROM the local
client:
  ssh -N -L <port>:localhost:<port> <user>@<this-box-address>
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --embodiment-tag)
            EMBODIMENT_TAG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model-path is required (HF repo id, e.g. nvidia/GR00T-N1.7-3B, or a local checkpoint dir)." >&2
    usage >&2
    exit 1
fi

echo "# On the local client: ssh -N -L ${PORT}:localhost:${PORT} <user>@<this-box-address>" >&2
echo "# Then point PolicyClient at host=localhost, port=${PORT}." >&2

cd "$GR00T_REPO"
exec python gr00t/eval/run_gr00t_server.py \
    --model-path "$MODEL_PATH" \
    --embodiment-tag "$EMBODIMENT_TAG" \
    --device "$DEVICE" \
    --host "$HOST" \
    --port "$PORT" \
    "${EXTRA_ARGS[@]}"
