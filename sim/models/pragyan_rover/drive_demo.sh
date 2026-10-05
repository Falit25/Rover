#!/usr/bin/env bash
set -e
source /opt/ros/humble/setup.bash
stop(){ ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}, angular: {z: 0.0}}" >/dev/null 2>&1 || true; }
trap stop EXIT

echo "Forward"
timeout 12s ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.06}, angular: {z: 0.0}}" || true
sleep 1
echo "Gentle left"
timeout 5s ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.05}, angular: {z: 0.10}}" || true
sleep 1
echo "Forward"
timeout 10s ros2 topic pub --rate 10 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.06}, angular: {z: 0.0}}" || true
sleep 1
stop
echo "Demo complete."
