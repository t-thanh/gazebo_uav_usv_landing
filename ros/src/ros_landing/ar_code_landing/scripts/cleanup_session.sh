#!/bin/bash
# cleanup_session.sh
# ──────────────────
# Replaces mrs_simulation/kill_previous_session.sh.
# Kills the previous gzserver (and related processes), waits for them to
# fully die, then removes the flag file that is blocking the new gzserver
# from starting.  This prevents the boost::lock_error crash that occurs when
# the old gzserver is still holding Gazebo shared-memory resources while the
# new one tries to initialise.
#
# Flag file protocol:
#   The launch file creates  /tmp/landing_sim_cleanup_flag  BEFORE starting
#   this script (via `touch` in wait_and_launch.sh's pre-exec block).
#   This script removes it when cleanup is complete, unblocking gzserver.

FLAG=/tmp/landing_sim_cleanup_flag

# ── 1. Ensure the flag exists so gzserver blocks even if we're fast ──────────
touch "$FLAG"

# ── 2. Kill previous session processes ───────────────────────────────────────
echo "[cleanup] Sending SIGTERM to gzserver, gzclient, px4, mavros …"
killall -SIGTERM gzserver gzclient px4 mavros mavros_node 2>/dev/null || true
sleep 0.4

# Force-kill anything still alive
killall -SIGKILL gzserver gzclient 2>/dev/null || true

# ── 3. Wait for gzserver to fully exit (up to 20 s) ──────────────────────────
WAITED=0
while pgrep -x gzserver > /dev/null 2>&1; do
    sleep 0.2
    WAITED=$((WAITED + 1))
    if [ $WAITED -ge 100 ]; then
        echo "[cleanup] WARNING: gzserver still alive after 20 s — continuing anyway."
        break
    fi
done

# Extra pause so the OS can release Gazebo shared-memory segments
sleep 0.5

# ── 4. Remove flag — gzserver is now allowed to start ────────────────────────
rm -f "$FLAG"
echo "[cleanup] Done. Gazebo may now start."
exit 0
