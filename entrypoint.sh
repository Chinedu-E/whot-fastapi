#!/bin/sh
set -eu

CHECKPOINT_PATH="${RL_CHECKPOINT_PATH:-/app/rl/checkpoints/production_model.zip}"
CHECKPOINT_DIR="$(dirname "$CHECKPOINT_PATH")"

mkdir -p "$CHECKPOINT_DIR"

if [ ! -f "$CHECKPOINT_PATH" ]; then
  if [ -n "${RL_CHECKPOINT_URL:-}" ]; then
    echo "Downloading RL checkpoint from RL_CHECKPOINT_URL -> $CHECKPOINT_PATH"
    curl -fsSL "$RL_CHECKPOINT_URL" -o "$CHECKPOINT_PATH"
    echo "Download complete ($(wc -c < "$CHECKPOINT_PATH") bytes)"
  else
    echo "WARNING: No checkpoint at $CHECKPOINT_PATH and RL_CHECKPOINT_URL unset; Human-like CPU will fall back to normal"
  fi
else
  echo "Using existing RL checkpoint at $CHECKPOINT_PATH"
fi

exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
