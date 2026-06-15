#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_demo.sh — orchestrates the human-watched gimbal-tracking demo.
#
#   1. Launch Gazebo: static Otter USV + MRS X500 + standalone gimbal.
#   2. Wait until the UAV (PX4 SITL) is up.
#   3. Launch the arm/takeoff + climb + gimbal-nadir + YOLO + IBVS tracker
#      sequence, with the tracking altitude set to 20 m.
#
# The two stages are sequenced (rather than one combined launch) because the
# arm/takeoff timeline uses fixed sleeps relative to its own start time and must
# begin only after the drone has spawned.
#
# Environment overrides:
#   GUI=true|false        show the Gazebo GUI (default true; set false for headless)
#   INIT_ALTITUDE=20.0    UAV tracking altitude in metres (default 20.0)
#   DEVICE=0              YOLO CUDA device index (default 0)
# ─────────────────────────────────────────────────────────────────────────────
set -e

GUI="${GUI:-true}"
INIT_ALTITUDE="${INIT_ALTITUDE:-20.0}"
DEVICE="${DEVICE:-0}"

source /opt/ros/${ROS_DISTRO:-noetic}/setup.bash
source /opt/gazebo_uav_usv_landing/ros/devel/setup.bash

# MRS stack environment (normally set by the MRS shell setup). core.launch
# fails hard if these are unset.
export UAV_NAME=uav1
export UAV_NUMBER=1
export RUN_TYPE=simulation
export UAV_TYPE=x500
export WORLD_NAME=simulation
export SENSORS="garmin_down"
export ODOMETRY_TYPE="${ODOMETRY_TYPE:-gps}"

echo "[demo] Starting roscore..."
roscore &
ROSCORE_PID=$!
sleep 3

cleanup() {
  echo "[demo] Shutting down..."
  kill ${TRACKER_PID} ${GAZEBO_PID} ${ROSCORE_PID} 2>/dev/null || true
  pkill -f gzserver 2>/dev/null || true
  pkill -f gzclient 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[demo] Launching Gazebo (USV + UAV + gimbal), GUI=${GUI}..."
roslaunch otter_gazebo usv_uav_gimbal.launch gui:="${GUI}" &
GAZEBO_PID=$!

echo "[demo] Waiting for the UAV (PX4 SITL) to come up..."
DEADLINE=$((SECONDS + 180))
until rostopic list 2>/dev/null | grep -q "/uav1/mavros/state"; do
  if [ ${SECONDS} -ge ${DEADLINE} ]; then
    echo "[demo] ERROR: UAV did not appear within 180 s." >&2
    exit 1
  fi
  sleep 2
done
echo "[demo] UAV detected. Letting the EKF settle (10 s)..."
sleep 10

echo "[demo] Launching takeoff + climb to ${INIT_ALTITUDE} m + gimbal nadir + tracker..."
roslaunch gimbal_usv_tracker start_uav_gimbal_tracker.launch \
    init_altitude:="${INIT_ALTITUDE}" device:="${DEVICE}" &
TRACKER_PID=$!

echo "[demo] Demo running. Watch Gazebo and:"
echo "       rostopic echo /uav1/gimbal/tracker/tracking_status"
echo "       rqt_image_view /uav1/usv_detection/image"
echo "[demo] Press Ctrl-C to stop."
wait ${TRACKER_PID}
