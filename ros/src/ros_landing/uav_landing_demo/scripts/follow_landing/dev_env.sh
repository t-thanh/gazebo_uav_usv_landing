#!/bin/bash
# Portable env for the packaged take-off-and-land demo (AR perception, no YOLO/conda).
NODES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${NODES}/../../../../../.." && pwd)"
source /opt/ros/${ROS_DISTRO:-noetic}/setup.bash
source "${REPO_ROOT}/ros/devel/setup.bash"
export GAZEBO_PLUGIN_PATH="${REPO_ROOT}/ros/devel/lib:${GAZEBO_PLUGIN_PATH:-}"
export GAZEBO_RESOURCE_PATH="${REPO_ROOT}/ros/src/usv_sim/otter_gazebo:${GAZEBO_RESOURCE_PATH:-}"
export UAV_NAME=uav1 UAV_NUMBER=1 RUN_TYPE=simulation UAV_TYPE=x500 WORLD_NAME=simulation
export SENSORS="garmin_down" ODOMETRY_TYPE="${ODOMETRY_TYPE:-gps}"
echo "[demo_env] ROS + workspace sourced (REPO_ROOT=${REPO_ROOT})"
