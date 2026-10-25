#!/usr/bin/env bash
# VM-level setup: clones every repo this VM's /workspace is expected to hold, side by side.
# Idempotent -- re-running fetches/switches existing checkouts instead of re-cloning.
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"

# name | url | ref (branch or tag) | submodules (yes/no)
REPOS=(
    "ASPIRE|https://github.com/NVlabs/ASPIRE.git|main|yes"
    "finetune-groot|https://github.com/svaichu/finetune-groot.git|main|no"
    "inference-client|https://github.com/svaichu/inference-client.git|main|no"
    "norma-core|https://github.com/norma-core/norma-core.git|main|no"
    "robotdataset|https://github.com/svaichu/robotdataset.git|feature/model-adapters|no"
)

for entry in "${REPOS[@]}"; do
    IFS='|' read -r name url ref submodules <<< "$entry"
    dir="$WORKSPACE/$name"
    echo "== $name =="
    if [ -d "$dir/.git" ]; then
        echo "already present at $dir, fetching and switching to $ref"
        git -C "$dir" fetch origin "$ref"
        git -C "$dir" switch "$ref" 2>/dev/null || git -C "$dir" checkout "$ref"
    else
        git clone --branch "$ref" "$url" "$dir"
    fi
    if [ "$submodules" = "yes" ]; then
        git -C "$dir" submodule update --init --recursive
    fi
done

# finetune-groot's own install step: vendors Isaac-GR00T and sets up the
# sibling robotdataset checkout it needs (see scripts/install.sh).
"$WORKSPACE/finetune-groot/scripts/install.sh"

echo "== done =="
for entry in "${REPOS[@]}"; do
    IFS='|' read -r name _ ref _ <<< "$entry"
    dir="$WORKSPACE/$name"
    echo "$name: $(git -C "$dir" rev-parse --short HEAD) ($ref)"
done
