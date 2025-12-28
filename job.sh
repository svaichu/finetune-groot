#!/bin/sh

set -e

IMAGE_NAME="${IMAGE_NAME:-gr00t-dev:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-finetune-groot}"
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ENTRYPOINT_PATH="/docker-entrypoint-init.d/finetune-groot/entrypoint.sh"

MODE="${1:-}"
if [ $# -gt 0 ]; then
  shift
fi

if [ "$MODE" = "" ] || [ "$MODE" = "job" ]; then
  docker build "$SCRIPT_DIR" --tag "$IMAGE_NAME"
  docker run --gpus all -d --rm --name "${CONTAINER_NAME}-job" \
    -e "WANDB_API_KEY=$WANDB_API_KEY" \
    -e "WANDB_USERNAME=$WANDB_USERNAME" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$SCRIPT_DIR":/docker-entrypoint-init.d/finetune-groot \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -i "$ENTRYPOINT_PATH" job "$@"
  exit 0
fi

if [ "$MODE" = "ijob" ]; then
  docker build "$SCRIPT_DIR" --tag "$IMAGE_NAME"
  docker run --gpus all -it --rm --name "${CONTAINER_NAME}-ijob" \
    -e "WANDB_API_KEY=$WANDB_API_KEY" \
    -e "WANDB_USERNAME=$WANDB_USERNAME" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$SCRIPT_DIR":/docker-entrypoint-init.d/finetune-groot \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -i "$ENTRYPOINT_PATH" job "$@"
  exit 0
fi

if [ "$MODE" = "dev" ]; then
  docker build "$SCRIPT_DIR" --tag "$IMAGE_NAME"
  docker run --gpus all -it --rm --name "${CONTAINER_NAME}-dev" \
    -e "WANDB_API_KEY=$WANDB_API_KEY" \
    -e "WANDB_USERNAME=$WANDB_USERNAME" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$SCRIPT_DIR":/docker-entrypoint-init.d/finetune-groot \
    -v "$SCRIPT_DIR":/workspace/finetune-groot \
    --entrypoint /bin/bash \
    "$IMAGE_NAME" -i "$ENTRYPOINT_PATH" dev "$@"
  exit 0
fi

if [ "$MODE" = "shell" ]; then
  # docker build "$SCRIPT_DIR" --tag "$IMAGE_NAME"
  docker run --gpus all -it --rm --name "${CONTAINER_NAME}-shell" \
    -e "WANDB_API_KEY=$WANDB_API_KEY" \
    -e "WANDB_USERNAME=$WANDB_USERNAME" \
    -e "HF_TOKEN=$HF_TOKEN" \
    -v "$SCRIPT_DIR":/workspace/finetune-groot \
    --entrypoint /bin/bash \
    "$IMAGE_NAME"
  exit 0
fi

if [ "$MODE" = "build" ]; then
  docker build "$SCRIPT_DIR" --tag "$IMAGE_NAME"
  exit 0
fi

if [ "$MODE" = "stop" ]; then
  docker stop "${CONTAINER_NAME}-job" "${CONTAINER_NAME}-ijob" \
    "${CONTAINER_NAME}-dev" "${CONTAINER_NAME}-shell"
  exit 0
fi

echo "Usage: ./job.sh [job|ijob|dev|shell|build|stop] [args...]"
exit 1
