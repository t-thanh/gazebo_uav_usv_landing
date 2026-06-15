#!/bin/bash
# Container entrypoint: source ROS + the built workspace, then exec the command.
set -e

source /opt/ros/${ROS_DISTRO:-noetic}/setup.bash
source /opt/gazebo_uav_usv_landing/ros/devel/setup.bash
[ -f /usr/share/gazebo/setup.sh ] && source /usr/share/gazebo/setup.sh

exec "$@"
