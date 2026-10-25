#!/usr/bin/env bash
# Sets up the two repos this project needs, per README.md's "Layout" section:
#   vendor/Isaac-GR00T  <- vendored at tag n1.7-release, with its submodules
#   ../robotdataset     <- sibling checkout, branch feature/model-adapters
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_DIR="$REPO_ROOT/vendor/Isaac-GR00T"
ROBOTDATASET_DIR="$(cd "$REPO_ROOT/.." && pwd)/robotdataset"

GR00T_URL="https://github.com/NVIDIA/Isaac-GR00T.git"
GR00T_REF="n1.7-release"
ROBOTDATASET_URL="https://github.com/svaichu/robotdataset.git"
ROBOTDATASET_BRANCH="feature/model-adapters"

echo "== vendor/Isaac-GR00T =="
if [ -d "$VENDOR_DIR/.git" ]; then
    echo "already present at $VENDOR_DIR, fetching and checking out $GR00T_REF"
    git -C "$VENDOR_DIR" fetch origin "$GR00T_REF"
    git -C "$VENDOR_DIR" checkout "$GR00T_REF"
else
    mkdir -p "$(dirname "$VENDOR_DIR")"
    git clone --branch "$GR00T_REF" "$GR00T_URL" "$VENDOR_DIR"
fi
git -C "$VENDOR_DIR" submodule update --init --recursive

echo "== ../robotdataset =="
if [ -d "$ROBOTDATASET_DIR/.git" ]; then
    echo "already present at $ROBOTDATASET_DIR, fetching and switching to $ROBOTDATASET_BRANCH"
    git -C "$ROBOTDATASET_DIR" fetch origin "$ROBOTDATASET_BRANCH"
    git -C "$ROBOTDATASET_DIR" switch "$ROBOTDATASET_BRANCH"
else
    git clone --branch "$ROBOTDATASET_BRANCH" "$ROBOTDATASET_URL" "$ROBOTDATASET_DIR"
fi

echo "== done =="
echo "vendor/Isaac-GR00T: $(git -C "$VENDOR_DIR" rev-parse --short HEAD) ($GR00T_REF)"
echo "../robotdataset:    $(git -C "$ROBOTDATASET_DIR" rev-parse --short HEAD) ($ROBOTDATASET_BRANCH)"
