#!/bin/bash
# wait_and_launch.sh FLAG_FILE CMD [ARGS...]
# ──────────────────────────────────────────
# Used as launch-prefix for gzserver/gzclient so they only start AFTER
# cleanup_session.sh has finished (i.e. the flag file has been removed).
#
# Usage in launch file:
#   launch-prefix="$(find ar_code_landing)/scripts/wait_and_launch.sh
#                   /tmp/landing_sim_cleanup_flag"

FLAG="$1"
shift

# Touch the flag now so cleanup_session sees it even if cleanup races ahead.
touch "$FLAG" 2>/dev/null || true

# Wait until cleanup removes the flag (or it never existed → start immediately)
while [ -f "$FLAG" ]; do
    sleep 0.1
done

exec "$@"
