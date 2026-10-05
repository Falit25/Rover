#!/usr/bin/env bash
set -e
PROJECT="$HOME/Desktop/Project/SIH"
MODEL_SRC="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$PROJECT/sim/models/pragyan_rover"
WORLD="$PROJECT/sim/worlds/lunar_terrain.world"
mkdir -p "$MODEL_DIR"
cp "$MODEL_SRC/model.sdf" "$MODEL_DIR/model.sdf"
cp "$MODEL_SRC/model.config" "$MODEL_DIR/model.config"

export GAZEBO_MODEL_PATH="${GAZEBO_MODEL_PATH:-}:$PROJECT/sim/models"
export GAZEBO_PLUGIN_PATH="${GAZEBO_PLUGIN_PATH:-}:$PROJECT/sim/models/thermal_camera_plugin/build"

if ! grep -q '<uri>model://pragyan_rover</uri>' "$WORLD"; then
  cp "$WORLD" "$WORLD.before_pragyan.bak"
  sed -i '/^[[:space:]]*<\/world>[[:space:]]*$/i\    <include>\n      <uri>model://pragyan_rover</uri>\n      <pose>0 0 1.0 0 0 0</pose>\n    </include>' "$WORLD"
  echo "Added Pragyan rover include to $WORLD"
else
  echo "Pragyan rover include already exists in $WORLD"
fi

echo "Starting Gazebo..."
echo "GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"
echo "GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
gazebo --verbose "$WORLD"
