#!/bin/bash
# Take-off & land on the SAME moving Otter USV, with the campaign-winning per-phase controllers.
# Only knobs:  USV_SPEED (0.0..2.0 m/s, default 1.0) · WAVE (none|small|medium|large, default small) · GUI.
# Pinned: APPROACH=PD+lookahead · FOLLOW/LAND=MPC near-field (q_pos 1.5, Cell A) · FOLLOW 9→floor 4,
#         cone 0.70·alt · AR perception + UWB near-field · directed gimbal search + climb-to-regain.
set -u
NODES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USV_SPEED="${USV_SPEED:-1.0}"; WAVE="${WAVE:-small}"; GUI="${GUI:-true}"
echo "═══ UAV → USV take-off & land  |  USV_SPEED=${USV_SPEED} m/s  WAVE=${WAVE}  GUI=${GUI} ═══"
GUI="${GUI}" CONTROLLER=handover CONTROL_LAW=mpc MPC_QPOS=1.5 \
APPROACH_LAW=pd USE_LOOKAHEAD=false VIS_TRAJ=false GIMBAL_MODE=ff MODEA_FIX=true \
ENABLE_THERMAL=false ENABLE_BLOB=false ENABLE_YOLO=false THERMAL_SEARCH=false LOSS_RECOVERY_CLIMB=true \
FOLLOW_ALTITUDE=9.0 FOLLOW_FLOOR_ALT=4.0 DESCEND_CONE_DEG=35.0 FOLLOW_DESCENT=true \
KP_XY=0.45 KD_XY=0.90 IMU_DAMP_K=1.2 LAND=true KILL_AT=LANDED \
USV_SPEED="${USV_SPEED}" WAVE="${WAVE}" \
  bash "${NODES}/run_follow_landing.sh"
