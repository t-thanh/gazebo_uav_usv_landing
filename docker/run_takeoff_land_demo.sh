#!/bin/bash
# Docker entry for the take-off-and-land demo. Selectable:  USV_SPEED · WAVE · GUI.
set -e
source /opt/ros/${ROS_DISTRO:-noetic}/setup.bash
source /opt/gazebo_uav_usv_landing/ros/devel/setup.bash
exec /opt/gazebo_uav_usv_landing/ros/src/ros_landing/uav_landing_demo/scripts/follow_landing/run_takeoff_land_demo.sh
