#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# tmp/run_local_usv_follow.sh
#
# GPS-FREE closed-loop USV-following demo.  The UAV arms, climbs, then flies over
# and follows the moving Otter USV at a defined altitude — staying directly above
# it — using ONLY onboard sensors (gimbal camera + IMU + rangefinder).  No GPS of
# UAV or USV, no UAV↔USV comms.
#
# Pipeline (all GPS-free):
#   YOLO detector → ray-cast relative-pose estimator (tmp/usv_relpos_estimator)
#   → CTRV UKF tracker (tmp/usv_motion_tracker) → Bézier predictor
#   (tmp/usv_bezier_predictor) → attitude follow controller (tmp/usv_follow_controller)
#   Gimbal kept on the USV by a pure-pixel IBVS node (tmp/usv_gimbal_ibvs).
#   Ground truth is logged for scoring only (tmp/usv_follow_logger).
#
# The USV is held at the origin (usv_gate) until the UAV reaches FOLLOW, then it
# is released to move so the following behaviour can be evaluated.
#
# Env overrides:  GUI=true|false  CLIMB_ALTITUDE=10  FOLLOW_ALTITUDE=8  DEVICE=0
#   USV_SPEED=0.0|0.5|1.0|1.5   WAVE=none|small|medium|large   WAVE_ANGLE=<deg>
#   CSV=<path>  (default tmp/eva_results/follow_<ts>_<wave>_v<speed>.csv)
#   KILL_AT=<STATE>  APPROACH-phase mode: exit when the controller reaches <STATE>
#                    (e.g. FOLLOW); APPROACH_TIMEOUT=<s> caps a deadlocked run (default 120).
#
# Analyse with:  python3 tmp/analyze_usv_follow.py <csv>
# ─────────────────────────────────────────────────────────────────────────────
set -e
NODES="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # the follow_landing node dir
REPO_ROOT="$(cd "${NODES}/../../../../../.." && pwd)"           # repo root (above ros/)

GUI="${GUI:-true}"
CLIMB_ALTITUDE="${CLIMB_ALTITUDE:-10.0}"
FOLLOW_ALTITUDE="${FOLLOW_ALTITUDE:-8.0}"
DEVICE="${DEVICE:-0}"
# Verification descent (Task 25): start FOLLOWing high, then ramp altitude down
# through the AR detection bands.  FOLLOW_DESCENT=true enables it.
FOLLOW_DESCENT="${FOLLOW_DESCENT:-false}"
FOLLOW_FLOOR_ALT="${FOLLOW_FLOOR_ALT:-6.0}"
FOLLOW_DESCENT_RATE="${FOLLOW_DESCENT_RATE:-0.5}"
USV_SPEED="${USV_SPEED:-0.5}"
WAVE="${WAVE:-small}"
WAVE_ANGLE="${WAVE_ANGLE:-0}"
USV_GATE="${USV_GATE:-system_ready}"
CSV="${CSV:-/tmp/uav_demo_$(date +%Y%m%d_%H%M%S)_${WAVE}_v${USV_SPEED}.csv}"
RELPOS_CSV="${CSV%.csv}_relpos.csv"

mkdir -p "$(dirname "${CSV}")"
source "${NODES}/dev_env.sh"

echo "[follow] roscore..."
roscore &
ROSCORE_PID=$!
sleep 3

cleanup() {
  echo "[follow] shutting down..."
  kill ${LOG_PID} ${CTRL_PID} ${IBVS_PID} ${PRED_PID} ${UKF_PID} ${SEL_PID} ${AR_PID} ${EST_PID} \
       ${BLOB_PID} ${DET_PID} ${BRIDGE_PID} ${GAZEBO_PID} ${RELEASE_PID} ${ROSCORE_PID} \
       ${THERMDET_PID:-} ${THERMSRCH_PID:-} 2>/dev/null || true
  pkill -f usv_thermal_detector_node.py 2>/dev/null || true
  pkill -f usv_thermal_search_node.py 2>/dev/null || true
  pkill -f usv_blob_relpos_node.py 2>/dev/null || true
  pkill -f usv_follow_controller_node.py 2>/dev/null || true
  pkill -f usv_uwb_follow_controller_node.py 2>/dev/null || true
  pkill -f usv_visibility_traj_node.py 2>/dev/null || true
  pkill -f usv_visibility_traj2_node.py 2>/dev/null || true
  pkill -f usv_uwb_sim_node.py 2>/dev/null || true
  pkill -f usv_uwb_ekf_node.py 2>/dev/null || true
  pkill -f usv_motion_tracker_node.py 2>/dev/null || true
  pkill -f usv_bezier_predictor_node.py 2>/dev/null || true
  pkill -f usv_perception_selector_node.py 2>/dev/null || true
  pkill -f usv_ar_relpos_estimator_node.py 2>/dev/null || true
  pkill -f usv_relpos_estimator_node.py 2>/dev/null || true
  pkill -f usv_gimbal_ibvs_node.py 2>/dev/null || true
  pkill -f usv_gimbal_ff_node.py 2>/dev/null || true
  pkill -f usv_follow_logger.py 2>/dev/null || true
  pkill -f gzserver 2>/dev/null || true
  pkill -f gzclient 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[follow] Gazebo (USV+UAV+gimbal) GUI=${GUI}  USV speed=${USV_SPEED} wave=${WAVE}@${WAVE_ANGLE}"
roslaunch otter_gazebo usv_uav_gimbal.launch gui:="${GUI}" \
    usv_speed:="${USV_SPEED}" wave:="${WAVE}" wave_angle:="${WAVE_ANGLE}" \
    usv_gate:="${USV_GATE}" uav_z:="${UAV_Z:-1.4}" \
    uav_x:="${UAV_X:-0.0}" uav_y:="${UAV_Y:-0.0}" &
GAZEBO_PID=$!

echo "[follow] waiting for /uav1/mavros/state ..."
DEADLINE=$((SECONDS + 180))
until rostopic list 2>/dev/null | grep -q "/uav1/mavros/state"; do
  if [ ${SECONDS} -ge ${DEADLINE} ]; then echo "[follow] ERROR: UAV not up" >&2; exit 1; fi
  sleep 2
done
echo "[follow] UAV up. EKF settle 10 s ..."
sleep 10

echo "[follow] ocean_rangefinder bridge ..."
roslaunch ocean_rangefinder bridge.launch uav_name:=uav1 &
BRIDGE_PID=$!

# Mode-B: open the PX4 arming + takeoff gate for the heaving deck (RC/failsafe/auto-disarm/
# IMU-consistency).  Runs to completion BEFORE the controller arms (blocking).  OPEN_ARMING=false
# to skip.  Local-only param sets via mavros — no firmware rebuild, no repo/Docker change.
if [ "${OPEN_ARMING:-true}" = "true" ]; then
  echo "[follow] opening PX4 arming/takeoff gate (SITL params) ..."
  python3 "${NODES}/px4_open_arming.py" __name:=px4_open_arming _ns:=uav1 || \
    echo "[follow] WARN: open_arming returned non-zero (some params may not be set)"
fi

# YOLO is UNRELIABLE at altitude (19–71 m RMS error at 8–12 m — the horizon-garbage that
# drove the UAV off-target) and its detector is the heavy CPU node that starved the AR
# pipeline (logger 18→9 Hz).  Default OFF: AR-only perception (OUTER reliable @~10 m,
# INNER close-range backup) + UWB <5 m.  ENABLE_YOLO=true restores the YOLO baseline.
ENABLE_YOLO="${ENABLE_YOLO:-false}"
if [ "${ENABLE_YOLO}" = "true" ]; then
  echo "[follow] YOLO detector ..."
  roslaunch otter_usv_detector detector.launch device:="${DEVICE}" \
      image_topic:=/uav1/overhead_cam/image_raw \
      detection_topic:=/uav1/usv_detection/result \
      viz_topic:=/uav1/usv_detection/image &
  DET_PID=$!
  sleep 8
  echo "[follow] YOLO ray-cast producer → /uav1/usv_relpos/yolo ..."
  python3 "${NODES}/usv_relpos_estimator_node.py" \
      __name:=usv_relpos_estimator _ns:=uav1 _csv_path:="${RELPOS_CSV}" &
  EST_PID=$!
fi

# THERMAL (Hadron 640R sim) is now the DEFAULT decode-free backup source (replaces the unreliable
# blob/yolo) in the unified perception — lowest priority, below UWB + AR.  Blob therefore default OFF.
ENABLE_THERMAL="${ENABLE_THERMAL:-true}"
ENABLE_BLOB="${ENABLE_BLOB:-false}"
echo "[follow] AR (OUTER>INNER) + BLOB producers → selector (YOLO=${ENABLE_YOLO} BLOB=${ENABLE_BLOB}) ..."
# GPS-free nested-AR PnP producer → /uav1/usv_relpos/{ar_inner,ar_outer}
python3 "${NODES}/usv_ar_relpos_estimator_node.py" \
    __name:=usv_ar_relpos_estimator _ns:=uav1 &
AR_PID=$!
# P3: robust B/W-pad blob producer → /uav1/usv_relpos/blob (coarse fallback, replaces YOLO;
# keeps a fix so the gimbal stays pointed when AR decoding drops)
if [ "${ENABLE_BLOB}" = "true" ]; then
  python3 "${NODES}/usv_blob_relpos_node.py" \
      __name:=usv_blob_relpos _ns:=uav1 &
  BLOB_PID=$!
fi
# selector → owns /uav1/usv_relpos/estimate.  enable_yolo/enable_blob gate those sources.
python3 "${NODES}/usv_perception_selector_node.py" \
    __name:=usv_perception_selector _ns:=uav1 \
    _promote_frames:="${PROMOTE_FRAMES:-8}" \
    _quality_factor:="${QUALITY_FACTOR:-2.0}" \
    _enable_yolo:="${ENABLE_YOLO}" _enable_blob:="${ENABLE_BLOB}" \
    _enable_thermal:="${ENABLE_THERMAL}" \
    _force_source:="${FORCE_SOURCE:-}" &
SEL_PID=$!

# THERMAL (Hadron 640R sim) producer + fast rotate-search — decode-free warm-pad detector, the
# long-range/search/backup source (far takeoff, regain on loss).  Lowest selector priority.
if [ "${ENABLE_THERMAL}" = "true" ]; then
  echo "[follow] thermal detector (default unified-perception backup source) ..."
  python3 "${NODES}/usv_thermal_detector_node.py" \
      __name:=usv_thermal_detector _ns:=uav1 _hot_thr:="${THERMAL_HOT_THR:-220}" \
      _max_incidence_deg:="${THERMAL_MAX_INCID:-60}" &
  THERMDET_PID=$!
  # The rotate-SEARCH (fixed-pitch yaw sweep + climb) is opt-in: it drives the gimbal off-nadir and
  # would disrupt the normal takeoff-on-USV nadir acquisition, so it is ONLY for far-takeoff /
  # full-loss recovery.  THERMAL_SEARCH=true to enable (e.g. the far-takeoff demo).
  if [ "${THERMAL_SEARCH:-false}" = "true" ]; then
    echo "[follow] thermal rotate-search (far/loss recovery) ..."
    python3 "${NODES}/usv_thermal_search_node.py" \
        __name:=usv_thermal_search _ns:=uav1 \
        _search_pitch_deg:="${THERMAL_SEARCH_PITCH:-20}" \
        _search_start_alt:="${THERMAL_SEARCH_START_ALT:-9.5}" \
        _search_active:="${THERMAL_SEARCH_ACTIVE:-true}" &
    THERMSRCH_PID=$!
  fi
fi

echo "[follow] UKF tracker + Bézier predictor ..."
python3 "${NODES}/usv_motion_tracker_node.py" \
    __name:=usv_motion_tracker _ns:=uav1 &
UKF_PID=$!
python3 "${NODES}/usv_bezier_predictor_node.py" \
    __name:=usv_bezier_predictor _ns:=uav1 &
PRED_PID=$!

# CONTROLLER=v3 (vision-only baseline) | handover (vision far → UWB near) | uwb (pure UWB)
CONTROLLER="${CONTROLLER:-v3}"
# NO_UWB=true skips the UWB sensor entirely → the handover controller flies on the GIMBAL CAMERA
# ONLY (vision drives horizontal all the way to touchdown).  Used to test UWB-free landing with
# the MPC / MPPI+L1 control laws (which now run on the vision state, same sign convention).
if [ "${CONTROLLER}" != "v3" ] && [ "${NO_UWB:-false}" != "true" ]; then
  echo "[follow] simulated UWB sensor (<${UWB_RANGE:-5} m, ${UWB_ACC:-0.10} m) + UWB/IMU EKF ..."
  python3 "${NODES}/usv_uwb_sim_node.py" __name:=usv_uwb_sim _ns:=uav1 \
      _max_range_m:="${UWB_RANGE:-5.0}" _accuracy_m:="${UWB_ACC:-0.10}" &
  UWBSIM_PID=$!
  # USV IMU for the EKF's frame de-rotation + FF: use the REAL Otter IMU (/imu/data, 15 Hz)
  # instead of the UWB-sim's /p3d-synthesized IMU (2 Hz) — the 2 Hz attitude can't track wave
  # roll/pitch (~0.3–0.5 Hz), so the IMU-stabilized frame needs the high-rate real IMU.
  python3 "${NODES}/usv_uwb_ekf_node.py" __name:=usv_uwb_ekf _ns:=uav1 \
      _use_uav_imu:="${UWB_USE_UAV_IMU:-true}" _sigma_a:="${UWB_SIGMA_A:-2.0}" \
      _usv_imu_topic:="${USV_IMU_TOPIC:-/imu/data}" &
  UWBEKF_PID=$!
fi

# GIMBAL_MODE=ff (DEFAULT: AR-fed model-based feed-forward pointing — keeps the USV in frame ~97%)
#            | ibvs (legacy reactive pixel IBVS — YOLO-only, DEAD when ENABLE_YOLO=false).
# NB: ibvs subscribes only to /usv_detection/result (YOLO); with YOLO off it gets no input and the
# gimbal never tracks (22% in-frame → the apparent "Mode-A" far-field losses).  ff points via the
# AR/blob-fed UKF track (/usv_track/filtered), so it tracks GPS-free without YOLO.
if [ "${GIMBAL_MODE:-ff}" = "ff" ]; then
  echo "[follow] P2 feed-forward gimbal pointing ..."
  python3 "${NODES}/usv_gimbal_ff_node.py" \
      __name:=usv_gimbal_ff _ns:=uav1 \
      _rate_hz:="${GIMBAL_RATE:-30}" \
      _use_lookahead:="${GIMBAL_LOOKAHEAD:-false}" \
      _inertial_setpoint:="${GIMBAL_INERTIAL:-false}" \
      _lock_nadir_in_follow:="${GIMBAL_NADIR_LOCK:-false}" &
  IBVS_PID=$!
else
  echo "[follow] pure-pixel gimbal IBVS (GPS-free centring) ..."
  python3 "${NODES}/usv_gimbal_ibvs_node.py" \
      __name:=usv_gimbal_ibvs _ns:=uav1 _rate_hz:="${GIMBAL_RATE:-30}" &
  IBVS_PID=$!
fi

echo "[follow] closed-loop CSV logger → ${CSV}"
python3 "${NODES}/usv_follow_logger.py" \
    __name:=usv_follow_logger _ns:=uav1 _csv_path:="${CSV}" &
LOG_PID=$!

# Release the USV to move once the UAV has FINISHED THE CLIMB and is ready to approach
# (state reaches APPROACH).  Because the UAV takes off FROM the USV and holds station over
# the deck during the climb (UWB low / vision high, gimbal nadir), it reaches climb altitude
# directly above the now-stationary USV with a warm lock — so when the USV starts moving the
# UAV follows it from a good initial geometry (no cold far-field acquisition).
# USV_RELEASE_STATE selects WHICH controller state releases the USV to move.
# Default APPROACH|FOLLOW|ALIGN = release as soon as the climb is done (proven
# eval behaviour).  For a DEMO, set USV_RELEASE_STATE='FOLLOW|ALIGN' so the USV
# stays put until the UAV has a TIGHT FOLLOW lock — the UAV then converges over a
# STATIONARY boat (no chase, no low-speed approach limit-cycle / overshoot) before
# the boat starts moving.  Cleaner and far more reliable at low USV speed.
(
  REL="${USV_RELEASE_STATE:-APPROACH|FOLLOW|ALIGN}"
  echo "[release] waiting for state matching '${REL}' to release USV ..."
  for i in $(seq 1 300); do
    ST=$(rostopic echo -n1 /usv_follow_controller/state 2>/dev/null | grep -oE "${REL}" | head -1 || true)
    if [ -n "${ST}" ]; then
      echo "[release] state=${ST} → releasing USV (system_ready=true)"
      rostopic pub -1 /uav1/gimbal/tracker/system_ready std_msgs/Bool "data: true" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
) &
RELEASE_PID=$!

echo "[follow] CONTROLLER=${CONTROLLER}  arm + climb to ${CLIMB_ALTITUDE} m → follow at ${FOLLOW_ALTITUDE} m"
echo "         descent=${FOLLOW_DESCENT} floor=${FOLLOW_FLOOR_ALT} m rate=${FOLLOW_DESCENT_RATE} m/s ..."
if [ "${CONTROLLER}" = "v3" ]; then
  # vision-only baseline (BASELINE_v3) — the proven CTRV-UKF + Bézier + PD controller
  python3 "${NODES}/usv_follow_controller_node.py" \
      __name:=usv_follow_controller _ns:=uav1 \
      _climb_alt_m:="${CLIMB_ALTITUDE}" _follow_alt_m:="${FOLLOW_ALTITUDE}" \
      _follow_descent:="${FOLLOW_DESCENT}" \
      _follow_floor_alt_m:="${FOLLOW_FLOOR_ALT}" \
      _follow_descent_rate_ms:="${FOLLOW_DESCENT_RATE}" \
      _use_lookahead:="${USE_LOOKAHEAD:-false}" \
      _kp_xy:="${KP_XY:-0.25}" _kd_xy:="${KD_XY:-0.90}" \
      _ki_xy:="${KI_XY:-0.0}" _max_int_xy:="${MAX_INT_XY:-2.0}" \
      _use_adrc:="${USE_ADRC:-false}" _adrc_obs_bw:="${ADRC_OBS_BW:-2.0}" \
      _land_enable:="${LAND:-false}" _align_alt_m:="${ALIGN_ALT:-1.8}" \
      _commit_rate_ms:="${COMMIT_RATE:-0.8}" _commit_radius_m:="${COMMIT_RADIUS:-0.4}" \
      _commit_vmax_ms:="${COMMIT_VMAX:-0.30}" \
      _ev_alpha:="${EV_ALPHA:-0.30}" \
      _tilt_slew_deg:="${TILT_SLEW:-30}" _max_tilt_deg:="${MAX_TILT:-12}" &
else
  # PD+FF on the UWB+IMU EKF.  CONTROLLER=handover → vision far-field + UWB near-field;
  # CONTROLLER=uwb → pure UWB (needs a close start so it begins in range).  Runs under
  # the name usv_follow_controller so the USV-release watcher + logger work unchanged.
  HO=$([ "${CONTROLLER}" = "handover" ] && echo true || echo false)
  # CONTROL_LAW=pdff (PD + USV-accel FF) | mpc (constrained QP) — the ablation switch
  python3 "${NODES}/usv_uwb_follow_controller_node.py" \
      __name:=usv_follow_controller _ns:=uav1 \
      _control_law:="${CONTROL_LAW:-pdff}" \
      _takeoff_thrust:="${TAKEOFF_THRUST:-0.80}" _takeoff_ramp_s:="${TAKEOFF_RAMP_S:-1.5}" \
      _deck_clear_alt_m:="${DECK_CLEAR_ALT:-2.5}" _climb_retries:="${CLIMB_RETRIES:-2}" \
      _reboot_settle_s:="${REBOOT_SETTLE_S:-14.0}" _reboot_predelay_s:="${REBOOT_PREDELAY_S:-7.0}" \
      _force_reboot_once:="${FORCE_REBOOT_ONCE:-false}" \
      _mpc_N:="${MPC_N:-15}" _mpc_qpos:="${MPC_QPOS:-1.0}" _mpc_qvel:="${MPC_QVEL:-0.3}" \
      _mpc_ru:="${MPC_RU:-0.05}" _mpc_qNpos:="${MPC_QNPOS:-5.0}" \
      _vis_max_tilt_rad:="${VIS_MAX_TILT:-0.12}" \
      _vis_mpc_qpos:="${VIS_MPC_QPOS:-0.35}" _vis_mpc_qvel:="${VIS_MPC_QVEL:-0.9}" \
      _vis_mpc_ru:="${VIS_MPC_RU:-0.4}" _vis_mpc_qNpos:="${VIS_MPC_QNPOS:-2.0}" \
      _vis_mpc_qNvel:="${VIS_MPC_QNVEL:-1.5}" _vis_mpc_N:="${VIS_MPC_N:-15}" \
      _vis_mppi_qpos:="${VIS_MPPI_QPOS:-0.7}" _vis_mppi_qvel:="${VIS_MPPI_QVEL:-1.4}" \
      _vis_mppi_ru:="${VIS_MPPI_RU:-0.4}" _vis_mppi_ki:="${VIS_MPPI_KI:-0.12}" \
      _vis_mppi_sigma:="${VIS_MPPI_SIGMA:-1.0}" \
      _vis_ki:="${VIS_KI:-0.20}" _vis_int_enable_m:="${VIS_INT_ENABLE:-6.0}" \
      _vis_max_int:="${VIS_MAX_INT:-3.0}" \
      _use_vision_handover:="${HO}" _handover_streak_n:="${HANDOVER_STREAK_N:-5}" \
      _handover_immediate_oor:="${HANDOVER_IMMEDIATE_OOR:-true}" \
      _handover_smooth_s:="${HANDOVER_SMOOTH_S:-0.4}" \
      _vis_coast_timeout_s:="${VIS_COAST_S:-3.0}" _track_lost_timeout_s:="${TRACK_LOST_S:-4.0}" \
      _vision_only:="${NO_UWB:-false}" _approach_law:="${APPROACH_LAW:-pd}" \
      _use_lookahead:="${USE_LOOKAHEAD:-false}" \
      _appr_mpc_dt:="${APPR_MPC_DT:-0.1}" _appr_mpc_N:="${APPR_MPC_N:-25}" \
      _appr_mpc_qpos:="${APPR_MPC_QPOS:-0.5}" _appr_mpc_qvel:="${APPR_MPC_QVEL:-1.8}" \
      _appr_mpc_ru:="${APPR_MPC_RU:-0.2}" _appr_mpc_qNpos:="${APPR_MPC_QNPOS:-2.0}" \
      _appr_mpc_qNvel:="${APPR_MPC_QNVEL:-2.0}" \
      _appr_mppi_N:="${APPR_MPPI_N:-25}" _appr_mppi_sigma:="${APPR_MPPI_SIGMA:-1.5}" \
      _appr_mppi_lambda:="${APPR_MPPI_LAMBDA:-0.05}" _appr_mppi_qpos:="${APPR_MPPI_QPOS:-0.6}" \
      _appr_mppi_qvel:="${APPR_MPPI_QVEL:-1.8}" _appr_mppi_ru:="${APPR_MPPI_RU:-0.2}" \
      _appr_mppi_ki:="${APPR_MPPI_KI:-0.1}" \
      _signal_split:="${SIGNAL_SPLIT:-false}" _calibrate_vusv:="${CALIBRATE_VUSV:-false}" \
      _imu_damp_k:="${IMU_DAMP_K:-1.2}" \
      _climb_alt_m:="${CLIMB_ALTITUDE}" _follow_alt_m:="${FOLLOW_ALTITUDE}" \
      _follow_descent:="${FOLLOW_DESCENT}" \
      _follow_floor_alt_m:="${FOLLOW_FLOOR_ALT}" \
      _follow_descent_rate_ms:="${FOLLOW_DESCENT_RATE}" \
      _kp_xy:="${KP_XY:-0.25}" _kd_xy:="${KD_XY:-0.90}" _kff_xy:="${KFF_XY:-1.0}" \
      _nf_kp_xy:="${NF_KP_XY:-${KP_XY:-0.25}}" _nf_kd_xy:="${NF_KD_XY:-${KD_XY:-0.90}}" \
      _nf_kff_xy:="${NF_KFF_XY:-${KFF_XY:-1.0}}" \
      _ki_xy:="${KI_XY:-0.0}" _max_int_xy:="${MAX_INT_XY:-1.5}" \
      _land_enable:="${LAND:-false}" _align_alt_m:="${ALIGN_ALT:-1.8}" \
      _commit_rate_ms:="${COMMIT_RATE:-0.8}" _commit_radius_m:="${COMMIT_RADIUS:-0.4}" \
      _commit_vmax_ms:="${COMMIT_VMAX:-0.30}" _ev_alpha:="${EV_ALPHA:-0.30}" \
      _commit_mode:="${COMMIT_MODE:-predictive}" _commit_hold_s:="${COMMIT_HOLD:-0.4}" \
      _commit_pred_radius_m:="${COMMIT_PRED_R:-0.9}" _commit_alt_band_m:="${COMMIT_ALT_BAND:-0.8}" \
      _commit_vmax_hard_ms:="${COMMIT_VMAX_HARD:-1.5}" \
      _tilt_slew_deg:="${TILT_SLEW:-30}" _max_tilt_deg:="${MAX_TILT:-12}" \
      _modeA_fix:="${MODEA_FIX:-true}" _search_alt_m:="${SEARCH_ALT:-9.0}" \
      _descend_cone_deg:="${DESCEND_CONE_DEG:-35.0}" _loss_recovery_climb:="${LOSS_RECOVERY_CLIMB:-true}" \
      _search_climb_rate_ms:="${SEARCH_CLIMB:-1.0}" _search_scan_enable:="${SEARCH_SCAN:-true}" \
      _divergence_max_m:="${DIVERGENCE_MAX:-30.0}" _divergence_abort_s:="${DIVERGENCE_ABORT:-20.0}" \
      _search_abort_s:="${SEARCH_ABORT:-120.0}" \
      _inject_dropout_at_s:="${INJECT_DROPOUT_AT:--1.0}" \
      _inject_dropout_dur_s:="${INJECT_DROPOUT_DUR:-6.0}" \
      _vis_traj:="${VIS_TRAJ:-false}" _vt_in_approach:="${VT_IN_APPROACH:-false}" &
fi
CTRL_PID=$!

# Visibility-constrained descent generator (fixed-nadir cone) when VIS_TRAJ=true.
# Phase 2 (default): jerk-limited smooth descent + velocity FF, engaged across ALL phases →
# single-controller architecture.  VIS_TRAJ_PHASE=1 selects the old Phase-1 step node.
if [ "${VIS_TRAJ:-false}" = "true" ]; then
  if [ "${VIS_TRAJ_PHASE:-2}" = "1" ]; then
    echo "[follow] visibility descent Phase-1 (rate-limited, FOLLOW/ALIGN) ..."
    python3 "${NODES}/usv_visibility_traj_node.py" __name:=usv_visibility_traj _ns:=uav1 \
        _vfov_deg:="${VT_VFOV:-52.6}" _gimbal_allow_deg:="${VT_GIMBAL_ALLOW:-0}" \
        _safety_margin:="${VT_MARGIN:-0.2}" _descend_rate_ms:="${VT_DESCEND:-0.6}" \
        _engage_states:="${VT_ENGAGE:-FOLLOW,ALIGN}" &
  else
    echo "[follow] visibility descent Phase-2 (jerk-limited + rate FF, all phases) ..."
    python3 "${NODES}/usv_visibility_traj2_node.py" __name:=usv_visibility_traj2 _ns:=uav1 \
        _vfov_deg:="${VT_VFOV:-52.6}" _gimbal_allow_deg:="${VT_GIMBAL_ALLOW:-0}" \
        _safety_margin:="${VT_MARGIN:-0.2}" _descend_rate_ms:="${VT_DESCEND:-0.6}" \
        _accel_max_ms2:="${VT_ACCEL:-0.5}" _settle_tau_s:="${VT_TAU:-1.5}" \
        _engage_states:="${VT_ENGAGE:-FOLLOW,ALIGN}" &
  fi
  VISTRAJ_PID=$!
fi

echo "[follow] running. Ctrl-C to stop, then:"
echo "        python3 tmp/analyze_usv_follow.py '${CSV}'"

# ── Climb-phase (Mode-B) monitor — runs in BOTH CLIMB_ONLY and full-mission modes ────────────
# Determine whether the deck takeoff succeeded (reached GIMBAL_NADIR or later) or aborted, and
# write CLIMB_OK / ABORT / TIMEOUT to RESULT_FILE.  A supervisor (tmp/run_with_relaunch.sh) reads
# this and RELAUNCHES the whole sim on a Mode-B ABORT — the only reliable way to get a fresh FCU,
# since an in-sim PX4 reboot/respawn is blocked by gzserver lockstep (see memory:px4-lockstep-fcu-reset).
RESULT_FILE="${CSV%.csv}_climb_result.txt"
CLIMB_DEADLINE=$((SECONDS + ${CLIMB_MONITOR_TIMEOUT:-${CLIMB_ONLY_TIMEOUT:-220}}))
OUTCOME="TIMEOUT"
echo "[follow] climb-phase monitor (timeout $((CLIMB_DEADLINE - SECONDS))s) ..."
while [ ${SECONDS} -lt ${CLIMB_DEADLINE} ]; do
  ST=$(rostopic echo -n1 /usv_follow_controller/state 2>/dev/null \
       | grep -oE 'PRE_ARMED|ARMING|REBOOT|CLIMBING|GIMBAL_NADIR|SEARCH|APPROACH|FOLLOW|ALIGN|COMMIT|TOUCHDOWN|LANDED|ABORT' \
       | head -1 || true)
  case "${ST}" in
    GIMBAL_NADIR|SEARCH|APPROACH|FOLLOW|ALIGN|COMMIT|TOUCHDOWN|LANDED) OUTCOME="CLIMB_OK"; break;;
    ABORT) OUTCOME="ABORT"; break;;
  esac
  sleep 1
done
echo "${OUTCOME}" > "${RESULT_FILE}"
echo "[follow] climb result=${OUTCOME} → ${RESULT_FILE}"

# CLIMB_ONLY: stop after the climb phase (cleanup trap tears everything down).
if [ "${CLIMB_ONLY:-false}" = "true" ]; then
  exit 0
fi
# Full mission: only fly on if the takeoff succeeded; otherwise exit so the supervisor relaunches
# (a Mode-B ABORT needs a fresh FCU = full relaunch).
if [ "${OUTCOME}" != "CLIMB_OK" ]; then
  echo "[follow] takeoff did not succeed (${OUTCOME}) — exiting for relaunch"
  exit 0
fi

# KILL_AT=<STATE>: APPROACH-phase benchmark mode.  Once the climb succeeds, watch the controller
# state and EXIT the moment it reaches KILL_AT (e.g. FOLLOW) — the run is scored ONLY over the
# APPROACH window (SEARCH-acquire → APPROACH → KILL_AT), no need to fly on to landing.  This keeps
# each approach-tuning run short so many seeds are affordable.  A max-time cap (APPROACH_TIMEOUT,
# default 120 s) terminates a non-promoting (deadlocked) run.  The USV is released at APPROACH entry
# by the default USV_RELEASE_STATE, so the boat is MOVING throughout the approach (the realistic,
# limit-cycle-prone regime).  Outcome → <csv>_phase_result.txt as "OUTCOME promote_dt_s":
#   PROMOTED  reached KILL_AT  |  DEADLOCK  cap hit, never promoted  |  ABORT  controller aborted.
# The authoritative APPROACH-window metrics come from the CSV (analyze_phase_sweep.py slices
# fstate==APPROACH); this file is the harness-level outcome + a best-effort promote time.
if [ -n "${KILL_AT:-}" ]; then
  PHASE_RESULT_FILE="${CSV%.csv}_phase_result.txt"
  PDEADLINE=$((SECONDS + ${APPROACH_TIMEOUT:-120}))
  POUT="DEADLOCK"; APPROACH_T0=""; PROMOTE_DT=""
  echo "[follow] KILL_AT=${KILL_AT} — APPROACH-window run (cap $((PDEADLINE - SECONDS))s) ..."
  while [ ${SECONDS} -lt ${PDEADLINE} ]; do
    ST=$(rostopic echo -n1 /usv_follow_controller/state 2>/dev/null \
         | grep -oE 'PRE_ARMED|ARMING|REBOOT|CLIMBING|GIMBAL_NADIR|SEARCH|APPROACH|FOLLOW|ALIGN|COMMIT|TOUCHDOWN|LANDED|ABORT' \
         | head -1 || true)
    if [ "${ST}" = "APPROACH" ] && [ -z "${APPROACH_T0}" ]; then APPROACH_T0=${SECONDS}; fi
    case "${ST}" in
      "${KILL_AT}")
        POUT="PROMOTED"
        [ -n "${APPROACH_T0}" ] && PROMOTE_DT=$((SECONDS - APPROACH_T0))
        break;;
      ABORT) POUT="ABORT"; break;;
    esac
    sleep 1
  done
  echo "${POUT} ${PROMOTE_DT}" > "${PHASE_RESULT_FILE}"
  echo "[follow] phase result=${POUT} promote_dt=${PROMOTE_DT:-NA}s → ${PHASE_RESULT_FILE}"
  exit 0
fi

# MISSION_TIME bounds a full-mission run (e.g. the Mode-A benchmark): follow for a fixed window
# after takeoff, then exit so the harness can score the CSV.  Unset → fly until Ctrl-C.
# DIVERGE_ABORT_M (default 50): abort early once the TRUE UAV-over-USV offset (gt_horiz in the CSV)
# blows past this — tracking has failed, no point waiting out the window.
if [ -n "${MISSION_TIME:-}" ]; then
  DEADLINE=$((SECONDS + MISSION_TIME))
  DAB="${DIVERGE_ABORT_M:-50}"
  echo "[follow] mission window: ${MISSION_TIME}s (abort early if gt_horiz > ${DAB} m) ..."
  while [ ${SECONDS} -lt ${DEADLINE} ]; do
    sleep 3
    GH=$(tail -3 "${CSV}" 2>/dev/null | awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="gt_horiz")c=i} END{if(c)print $c}' )
    # header may not be in tail; fall back to a python one-liner for robustness
    GH=$(python3 - "${CSV}" 2>/dev/null <<'PY'
import csv,sys
try:
    rows=list(csv.DictReader(open(sys.argv[1])))
    for r in reversed(rows):
        v=r.get('gt_horiz')
        if v not in (None,''): print(abs(float(v))); break
except Exception: pass
PY
)
    if [ -n "${GH}" ] && awk "BEGIN{exit !(${GH} > ${DAB})}"; then
      echo "[follow] DIVERGED: gt_horiz=${GH} m > ${DAB} m → abort early"
      break
    fi
  done
  echo "[follow] mission window done → exit"
  exit 0
fi

wait ${CTRL_PID}
