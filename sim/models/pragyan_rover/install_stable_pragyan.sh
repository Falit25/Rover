#!/usr/bin/env bash
set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/Desktop/Project/SIH/sim/models/pragyan_rover"
mkdir -p "$DEST"
cp "$SRC_DIR/model.sdf" "$DEST/model.sdf"
cp "$SRC_DIR/model.config" "$DEST/model.config"
echo "Installed stable Pragyan model to $DEST"
