#!/usr/bin/env python3
"""
usv_uwb_follow_controller_node.py  (tmp / dev-only)
─────────────────────────────────────────────────────
PD + feed-forward follow/land controller on the UWB+IMU EKF relative state — the
FIRST ablation arm vs the v3 vision-only baseline and vs MPC (tmp/PLAN_uwb_mpc.md).

Same flight stack / state machine / attitude+thrust machinery as
`tmp/usv_follow_controller_node.py` (v3).  Only the STATE SOURCE and the CONTROL LAW
change:
  • State  : /<ns>/uwb_ekf/rel_odom  (UAV pose r + rel-velocity ṙ in the USV frame)
             /<ns>/uwb_ekf/usv_accel_ff (USV kinematic accel a_usv, the FF term)
             /<ns>/uwb/in_range  (UWB fix available — only <5 m)
  • Law    : a_cmd = Kp·(−r) + Kd·(−ṙ) + Kff·a_usv          [USV/relative frame]
             a_body = Rz(−yaw_rel)·a_cmd  →  pitch = a_x/g, roll = −a_y/g
    The EKF velocity ṙ is used DIRECTLY for damping (clean — no noisy offset
    differentiation, the thing the vision baseline lacked), and Kff·a_usv lets the
    UAV anticipate the USV's acceleration (the GPS-free analogue of the GPS+MPC ref).

Operating envelope: UWB works only within 5 m range, so this is the NEAR-FIELD /
landing controller (low follow altitude).  Horizontal control engages once the UWB
is in range; otherwise the loop holds level (vision would bootstrap the far field).

Run:  python3 tmp/usv_uwb_follow_controller_node.py _ns:=uav1
"""
import math
import os
import sys
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from usv_uwb_mpc import build_axis_mpc, mpc_horizontal   # MPC ablation arm
from usv_mppi_l1 import build_mppi_l1, mppi_l1_horizontal  # MPPI+L1 ablation arm

from std_msgs.msg import Float64, String, Bool
from sensor_msgs.msg import Range, Imu, JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Vector3, Quaternion, PoseStamped

from mavros_msgs.msg import State as MavState, AttitudeTarget, ParamValue
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import CommandLong, CommandLongRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import ParamSet, ParamSetRequest

_G = 9.806


def horizontal_pd_ff(r_xy, rdot_xy, ff_xy, yaw_rel, kp, kd, kff,
                     int_xy=None, ki=0.0, deadband=0.0):
    """PD + USV-accel feed-forward in the USV/relative frame, returned in the UAV
    body-yaw frame (ready for tilt).  r_xy = UAV pos in USV frame (drive → 0);
    rdot_xy = relative velocity; ff_xy = a_usv (kinematic).  yaw_rel = ψ_uav − ψ_usv.
    Pure function (no ROS) so the control law is unit-testable."""
    e_pos = -np.asarray(r_xy, float)
    e_vel = -np.asarray(rdot_xy, float)
    a = kd * e_vel + kff * np.asarray(ff_xy, float)
    if int_xy is not None:
        a = a + ki * np.asarray(int_xy, float)
    if np.linalg.norm(r_xy) >= deadband:
        a = a + kp * e_pos
    a_b = Rot.from_euler('z', -yaw_rel).apply(np.append(a, 0.0))[:2]
    return a_b


class S:
    PRE_ARMED    = 'PRE_ARMED'
    ARMING       = 'ARMING'
    REBOOT       = 'REBOOT'
    CLIMBING     = 'CLIMBING'
    GIMBAL_NADIR = 'GIMBAL_NADIR'
    SEARCH       = 'SEARCH'
    APPROACH     = 'APPROACH'
    FOLLOW       = 'FOLLOW'
    ALIGN        = 'ALIGN'
    COMMIT       = 'COMMIT'
    TOUCHDOWN    = 'TOUCHDOWN'
    LANDED       = 'LANDED'
    ABORT        = 'ABORT'


class UwbFollowController:
    def __init__(self):
        rospy.init_node('usv_uwb_follow_controller')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns

        # ── control gains (PD + FF) ───────────────────────────────────────────
        self._ctrl_hz = float(rospy.get_param('~ctrl_rate_hz', 50.0))
        self._sm_hz   = float(rospy.get_param('~sm_rate_hz', 10.0))
        self._kp_xy   = float(rospy.get_param('~kp_xy', 0.25))
        self._kd_xy   = float(rospy.get_param('~kd_xy', 0.90))
        self._kff_xy  = float(rospy.get_param('~kff_xy', 1.0))   # USV-accel feed-forward
        self._ki_xy   = float(rospy.get_param('~ki_xy', 0.0))
        # NEAR-FIELD (UWB) PD gains — DECOUPLED from the approach PD (kp_xy/kd_xy) so the near-field
        # pdff law can be tuned for the <5 m landing without disturbing the locked approach front-end.
        # Default to the approach gains (no behaviour change unless ~nf_* set).
        self._nf_kp_xy  = float(rospy.get_param('~nf_kp_xy', self._kp_xy))
        self._nf_kd_xy  = float(rospy.get_param('~nf_kd_xy', self._kd_xy))
        self._nf_kff_xy = float(rospy.get_param('~nf_kff_xy', self._kff_xy))
        self._max_int_xy = float(rospy.get_param('~max_int_xy', 1.5))
        self._int_enable = float(rospy.get_param('~int_enable_m', 1.5))
        self._brake_decay = float(rospy.get_param('~brake_decay', 0.97))
        self._max_tilt  = math.radians(float(rospy.get_param('~max_tilt_deg', 12.0)))
        self._tilt_slew = math.radians(float(rospy.get_param('~tilt_slew_deg', 30.0)))
        # ── horizontal control law: pdff (PD+FF) | mpc (QP) | mppi (MPPI+L1) ──
        self._law = str(rospy.get_param('~control_law', 'pdff')).lower()
        self._mpc = self._mpc_vis = None
        self._mppi = self._mppi_vis = None
        # TWO tunings per law.  UWB (low-lag, <5 m) → aggressive.  VISION (gimbal camera, ~30 Hz +
        # UKF lag) → the aggressive UWB tuning OSCILLATES and loses the gimbal, so the vision branch
        # uses a SEPARATE gentler / more-damped instance: lower position weight, higher velocity
        # weight + control penalty, and a lower tilt cap (~7° vs ~12°) so the airframe stays calm
        # enough for the gimbal to keep the USV centred.
        vis_tilt = float(rospy.get_param('~vis_max_tilt_rad', 0.12))   # ≈7°
        # vision integral (closes the moving-target lag the gentle MPC/MPPI otherwise holds)
        self._vis_ki         = float(rospy.get_param('~vis_ki', 0.20))
        self._vis_int_enable = float(rospy.get_param('~vis_int_enable_m', 6.0))
        self._vis_max_int    = float(rospy.get_param('~vis_max_int', 3.0))
        # The gentle/damped vision tunings (_mpc_vis/_mppi_vis, ~7° tilt) exist ONLY for the
        # UWB-FREE (vision-only) experiments, where the aggressive UWB tuning oscillates the gimbal.
        # In the normal UWB-handover ablation the far field must use the AGGRESSIVE PD+FF (run17):
        # the gentle controller caps tilt so low it cannot catch a moving USV (~7 m lag at 1 m/s) and
        # loses the target at the marker hand-off.  So the gentle branch is gated behind this flag.
        self._vision_only    = bool(rospy.get_param('~vision_only', False))
        if self._law == 'mpc':
            self._mpc = build_axis_mpc(                                 # UWB tuning (near-field)
                dt=float(rospy.get_param('~mpc_dt', 0.1)),
                N=int(rospy.get_param('~mpc_N', 15)),
                q_pos=float(rospy.get_param('~mpc_qpos', 1.0)),
                q_vel=float(rospy.get_param('~mpc_qvel', 0.3)),
                r_u=float(rospy.get_param('~mpc_ru', 0.05)),
                qN_pos=float(rospy.get_param('~mpc_qNpos', 5.0)),
                qN_vel=float(rospy.get_param('~mpc_qNvel', 1.0)),
                max_tilt_rad=self._max_tilt)
            self._mpc_vis = build_axis_mpc(                             # VISION tuning (gentle/damped)
                dt=float(rospy.get_param('~vis_mpc_dt', 0.1)),
                N=int(rospy.get_param('~vis_mpc_N', 15)),
                q_pos=float(rospy.get_param('~vis_mpc_qpos', 0.35)),
                q_vel=float(rospy.get_param('~vis_mpc_qvel', 0.9)),
                r_u=float(rospy.get_param('~vis_mpc_ru', 0.4)),
                qN_pos=float(rospy.get_param('~vis_mpc_qNpos', 2.0)),
                qN_vel=float(rospy.get_param('~vis_mpc_qNvel', 1.5)),
                max_tilt_rad=vis_tilt)
        elif self._law == 'mppi':
            self._mppi = build_mppi_l1(                                 # UWB tuning (near-field)
                dt=float(rospy.get_param('~mppi_dt', 0.1)),
                N=int(rospy.get_param('~mppi_N', 20)),
                K=int(rospy.get_param('~mppi_K', 512)),
                lam=float(rospy.get_param('~mppi_lambda', 0.05)),
                sigma=float(rospy.get_param('~mppi_sigma', 2.0)),
                q_pos=float(rospy.get_param('~mppi_qpos', 2.0)),
                q_vel=float(rospy.get_param('~mppi_qvel', 0.6)),
                r_u=float(rospy.get_param('~mppi_ru', 0.05)),
                K_i=float(rospy.get_param('~mppi_ki', 0.3)),
                d_clamp=float(rospy.get_param('~mppi_dclamp', 2.0)),
                max_tilt_rad=self._max_tilt)
            self._mppi_vis = build_mppi_l1(                            # VISION tuning (gentle/damped)
                dt=float(rospy.get_param('~vis_mppi_dt', 0.1)),
                N=int(rospy.get_param('~vis_mppi_N', 20)),
                K=int(rospy.get_param('~vis_mppi_K', 512)),
                lam=float(rospy.get_param('~vis_mppi_lambda', 0.05)),
                sigma=float(rospy.get_param('~vis_mppi_sigma', 1.0)),
                q_pos=float(rospy.get_param('~vis_mppi_qpos', 0.7)),
                q_vel=float(rospy.get_param('~vis_mppi_qvel', 1.4)),
                r_u=float(rospy.get_param('~vis_mppi_ru', 0.4)),
                K_i=float(rospy.get_param('~vis_mppi_ki', 0.12)),
                d_clamp=float(rospy.get_param('~vis_mppi_dclamp', 2.0)),
                max_tilt_rad=vis_tilt)
        # ── PER-PHASE control ablation: APPROACH (far-field, pre-UWB) controller selector ──────
        # The handover far-field law has historically been the PD (run17 default).  To choose the
        # best APPROACH controller independently of the near-field/landing arm, ~approach_law can
        # route the far-field branch through pd (default) | mpc | mppi, built here as a DEDICATED
        # instance (own tuning, longer horizon ok) so it works whatever CONTROL_LAW the arm uses.
        self._appr_kind = str(rospy.get_param('~approach_law', 'pd')).lower()
        self._appr_ctrl = None
        # APPROACH wants HEAVY damping to avoid overshooting a slow/wave-swayed target (the PD's
        # kd/kp≈4.8 is what keeps it stable).  So the approach MPC/MPPI default to a high velocity-
        # penalty ratio (q_vel/q_pos≈3.6, matching the PD) with FULL 12° tilt authority (NOT the
        # gentle 7° vision-only cap — that cap was the regression).  Longer horizon (N=25) too.
        if self._appr_kind == 'mpc':
            self._appr_ctrl = build_axis_mpc(
                dt=float(rospy.get_param('~appr_mpc_dt', 0.1)),
                N=int(rospy.get_param('~appr_mpc_N', 25)),
                q_pos=float(rospy.get_param('~appr_mpc_qpos', 0.5)),
                q_vel=float(rospy.get_param('~appr_mpc_qvel', 1.8)),
                r_u=float(rospy.get_param('~appr_mpc_ru', 0.2)),
                qN_pos=float(rospy.get_param('~appr_mpc_qNpos', 2.0)),
                qN_vel=float(rospy.get_param('~appr_mpc_qNvel', 2.0)),
                max_tilt_rad=self._max_tilt)
        elif self._appr_kind == 'mppi':
            self._appr_ctrl = build_mppi_l1(
                dt=float(rospy.get_param('~appr_mppi_dt', 0.1)),
                N=int(rospy.get_param('~appr_mppi_N', 25)),
                K=int(rospy.get_param('~appr_mppi_K', 512)),
                lam=float(rospy.get_param('~appr_mppi_lambda', 0.05)),
                sigma=float(rospy.get_param('~appr_mppi_sigma', 1.5)),
                q_pos=float(rospy.get_param('~appr_mppi_qpos', 0.6)),
                q_vel=float(rospy.get_param('~appr_mppi_qvel', 1.8)),
                r_u=float(rospy.get_param('~appr_mppi_ru', 0.2)),
                K_i=float(rospy.get_param('~appr_mppi_ki', 0.1)),
                d_clamp=float(rospy.get_param('~appr_mppi_dclamp', 2.0)),
                max_tilt_rad=self._max_tilt)
        # Provenance: log the resolved APPROACH-law tuning so every run's log self-documents the
        # gains being swept (use_lookahead is set later in __init__ — log it from the param here).
        if self._appr_kind == 'mpc':
            rospy.loginfo("[uwb_follow] APPROACH law=mpc  appr_mpc[N=%d qpos=%.2f qvel=%.2f ru=%.2f "
                          "qN=%.1f/%.1f]  use_lookahead=%s",
                          int(rospy.get_param('~appr_mpc_N', 25)),
                          float(rospy.get_param('~appr_mpc_qpos', 0.5)),
                          float(rospy.get_param('~appr_mpc_qvel', 1.8)),
                          float(rospy.get_param('~appr_mpc_ru', 0.2)),
                          float(rospy.get_param('~appr_mpc_qNpos', 2.0)),
                          float(rospy.get_param('~appr_mpc_qNvel', 2.0)),
                          bool(rospy.get_param('~use_lookahead', False)))
        else:
            rospy.loginfo("[uwb_follow] APPROACH law=%s  use_lookahead=%s",
                          self._appr_kind, bool(rospy.get_param('~use_lookahead', False)))
        self._deadband  = float(rospy.get_param('~deadband_m', 0.25))
        self._hover     = float(rospy.get_param('~hover_thrust', 0.58))
        self._kp_z      = float(rospy.get_param('~kp_z', 0.35))
        self._ki_z      = float(rospy.get_param('~ki_z', 0.02))
        self._kd_z      = float(rospy.get_param('~kd_z', 0.20))
        self._max_dthr  = float(rospy.get_param('~max_thrust_delta', 0.25))
        self._ekf_to    = float(rospy.get_param('~ekf_timeout_s', 0.4))
        self._warm_n    = int(rospy.get_param('~ekf_warm_n', 10))
        # ── Mode-B heave-robust takeoff ───────────────────────────────────────
        # On a heaving deck (large waves) the rangefinder reads the deck moving
        # up/down under the beam, so the altitude error is garbage and the PD
        # never commits to a clean thrust → the UAV rides the deck for the whole
        # climb timeout then ABORTs.  While below deck_clear_alt we ignore the
        # rangefinder and apply an OPEN-LOOP thrust ramp (hover→takeoff_thrust
        # over ramp_s) with its own higher ceiling so it can actually break free;
        # only above deck_clear_alt (sustained) do we hand back to the altitude
        # PD where the rangefinder is clean of deck contact.
        self._takeoff_thr   = float(rospy.get_param('~takeoff_thrust', 0.80))
        self._takeoff_ramp  = float(rospy.get_param('~takeoff_ramp_s', 1.5))
        self._deck_clear    = float(rospy.get_param('~deck_clear_alt_m', 2.5))
        self._deck_clear_n  = int(rospy.get_param('~deck_clear_n', 5))
        self._climb_retries = int(rospy.get_param('~climb_retries', 3))  # reboot+re-arm cycles
        self._reboot_settle = float(rospy.get_param('~reboot_settle_s', 14.0))  # EKF re-settle after FCU reboot
        self._reboot_predelay = float(rospy.get_param('~reboot_predelay_s', 7.0))  # disarm→land-detect settle before reboot
        self._reboot_sent   = False
        self._force_reboot_once = bool(rospy.get_param('~force_reboot_once', False))  # deterministic mechanism check
        self._forced_reboot_done = False
        self._reboot_logf = str(rospy.get_param('~reboot_log', '/tmp/uwb_reboot.log'))
        self._clear_streak  = 0      # consecutive ticks above deck_clear_alt
        self._climb_try     = 0      # climb attempts used
        self._climb_t0      = None   # wall-time the current climb attempt began
        self._reboot_t0     = None   # wall-time the FCU reboot was commanded

        # ── vision→UWB handover ───────────────────────────────────────────────
        # When true, the VISION pipeline (/usv_track/*) flies the far field (climb,
        # approach, high follow); as the UAV descends into UWB range (<5 m) the
        # horizontal law HANDS OVER to PD+FF on the UWB EKF state, which then owns
        # the near-field precision follow + landing.  One attitude publisher, one
        # SM; only the horizontal source switches (with streak hysteresis).
        self._handover    = bool(rospy.get_param('~use_vision_handover', False))
        self._ho_streak_n = int(rospy.get_param('~handover_streak_n', 5))
        # demote with a TIMEOUT, not instantly: once latched onto UWB, coast through
        # brief boundary blips and only fall back to vision after a sustained loss —
        # this stops the boundary flip-flop that kept the handover from stabilising.
        self._ho_drop_s   = float(rospy.get_param('~handover_drop_s', 2.0))
        # CLIMB hand-off: when UWB goes genuinely OUT OF RANGE (above ~5 m) during the ascent and
        # vision/AR is already fresh, switch to vision IMMEDIATELY instead of coasting ho_drop_s —
        # the 2 s coast is only meant for near-field wave-heave blips, not the one-way climb crossing.
        self._ho_immediate = bool(rospy.get_param('~handover_immediate_oor', True))
        # jump-smoothing across a UWB↔vision handover (two estimators differ ~0.1m vs ~0.3-0.6m):
        # ramp the horizontal command from the pre-switch value to the new source over this window.
        self._ho_smooth_tau = float(rospy.get_param('~handover_smooth_s', 0.4))
        self._ev_alpha    = float(rospy.get_param('~ev_alpha', 0.30))   # vision offset-deriv LPF
        self._vis_to      = float(rospy.get_param('~visible_timeout_s', 0.8))
        # Far-field acquisition robustness: when vision briefly drops (the AR/gimbal loses the
        # USV during climb/approach/early-FOLLOW), don't go limp and let a slow/creeping USV
        # drift away — COAST toward the last filtered (UKF predict-only) position + predicted
        # velocity until it re-acquires.  vis_coast_to ≈ the UKF predict window; kp_coast is a
        # gentler position gain (it's a prediction).  Safe: with the speed-ramp the far field
        # always faces a SLOW USV, so this can't perturb the proven high-speed near-field case.
        # vis-coast / track-lost windows.  (Stage-1 tried 5/6 s + coast-on-Bézier-prediction to bridge
        # dropouts but it REGRESSED — the raw lookahead flings at 2.0 m/s, like the approach fling, and
        # the longer window gave it more time to diverge.  Reverted to the 3/4 s baseline; see
        # findings_followland_stage1.md.  Kept as params for a future CAPPED-coast / brake-on-loss try.)
        self._vis_coast_to = float(rospy.get_param('~vis_coast_timeout_s', 3.0))
        self._track_lost_to = float(rospy.get_param('~track_lost_timeout_s', 4.0))  # guard → SEARCH
        self._kp_coast     = float(rospy.get_param('~kp_coast_frac', 0.7))
        # FOLLOW verification/handover descent: lower the setpoint from follow_alt to
        # follow_floor so the UAV crosses into UWB range (triggers the handover).
        self._follow_descent      = bool(rospy.get_param('~follow_descent', False))
        self._follow_floor_alt    = float(rospy.get_param('~follow_floor_alt_m', 4.0))
        self._follow_descent_rate = float(rospy.get_param('~follow_descent_rate_ms', 0.4))
        self._follow_descent_settle = float(rospy.get_param('~follow_descent_settle_s', 4.0))
        # 0.C hand-off hold + descent ratchet (guards the vis_traj cone descent only):
        #  • hold:    don't descend BELOW band_top until the inner marker is solidly acquired
        #             (N consecutive ar_inner fixes) — stops blind descent through the 6-7 m
        #             outer→inner hand-off gap.
        #  • ratchet: the cone re-lifts the UAV whenever the horizontal offset spikes (the low-
        #             speed limit-cycle), which blocks landing (alt yo-yos 6↔11 m, never sinks).
        #             Rate-limit and CAP how far a climb may rise above the lowest altitude
        #             reached, so a transient spike can't yank the UAV back up; descents pass
        #             through immediately.  The brief perception loss is bridged by the UKF.
        self._handoff_band_top = float(rospy.get_param('~handoff_band_top_m', 7.0))
        self._handoff_band     = float(rospy.get_param('~handoff_band_m', 2.0))   # band depth below top
        self._handoff_inner_n  = int(rospy.get_param('~handoff_inner_n', 5))
        self._handoff_cross_rate = float(rospy.get_param('~handoff_cross_rate_ms', 0.25))
        self._vt_climb_rate    = float(rospy.get_param('~vt_climb_rate_ms', 0.3))
        self._vt_climb_cap     = float(rospy.get_param('~vt_climb_cap_m', 2.0))
        # CONE-gated descent: only sink while the USV is within the AR detection cone
        # (incidence < cone_deg from nadir).  Measured AR cutoff is ~40° (both markers
        # die >40°, solid <35°); 30° gives margin.  r_allowed = tan(cone)·alt — a CONE
        # that tightens with altitude, funnelling the UAV into the centred region where
        # UWB (alt<5 ∧ r<7 cylinder) then takes over, instead of descending offset into
        # the blind hole where outer-AR has clipped and UWB is still out of range.
        # descend cone: USV must be inside a fixed half-angle cone (cone_r = tan(deg)·alt) to descend.
        # Default raised 30°→35° (tan≈0.70) for a more generous gate — descend more readily, less HOLD.
        self._descend_cone = math.tan(math.radians(
            float(rospy.get_param('~descend_cone_deg', 35.0))))
        self._descend_cone_min = float(rospy.get_param('~descend_cone_min_r', 0.8))
        # loss-recovery body climb: when perception is lost (UWB absent), climb to widen the FOV and
        # re-acquire.  False = gimbal-only search (no body climb).  Either way the climb is gated to fire
        # only when UWB is NOT fresh (UWB owns the body when present).  A/B knob for the recovery method.
        self._loss_recovery_climb = bool(rospy.get_param('~loss_recovery_climb', True))
        # P1 — climb-to-regain: if perception is lost during FOLLOW, do NOT keep descending
        # blind (that's the spiral that loses the target); instead CLIMB to re-acquire (AR
        # works better higher + re-enters the FOV).  Only descend when a fix is fresh AND
        # the target is inside the cone.
        self._regain_to = float(rospy.get_param('~regain_timeout_s', 0.6))
        self._regain_climb = float(rospy.get_param('~regain_climb_rate_ms', 0.8))
        self._regain_since = None
        # APPROACH now mirrors FOLLOW's climb-to-regain (no passive relapse to SEARCH on a
        # brief loss while the USV is moving); only give up to SEARCH after a long sustained
        # loss.  This + the hardened blob closes the APPROACH-descent FARFIELD escape.
        self._appr_lost_abort = float(rospy.get_param('~appr_lost_abort_s', 4.0))

        # ── Mode-A (far-field acquisition collapse) fixes ─────────────────────
        # #1 stable-lock descent gate: don't descend to follow_alt on a FIRST marginal fix —
        #    wait until the lock is WARM (>=warm_n consecutive fresh fixes).  Premature descent
        #    narrows the FOV and drops the moving USV → relapse to SEARCH.
        # #2 SEARCH FOV-recovery: on loss, CLIMB to widen the nadir footprint + cone-scan the
        #    gimbal to re-acquire (SEARCH used to just HOLD → dead-end → divergence).
        # #3 divergence guard: bound the chase — if the estimated offset blows up, or SEARCH
        #    never re-acquires, ABORT cleanly instead of flying off to hundreds of metres.
        # _modeA_fix is the master switch (False = legacy behaviour, for the A/B benchmark).
        self._modeA_fix      = bool(rospy.get_param('~modeA_fix', True))
        self._search_alt     = float(rospy.get_param('~search_alt_m', 9.0))  # FOV-recovery ceiling
        self._search_climb   = float(rospy.get_param('~search_climb_rate_ms', 1.0))
        # UNIFIED THERMAL re-acquisition (APPROACH + FOLLOW share this S.SEARCH block): replace the
        # slow blind FOV-recovery dither with a FAST expanding spiral around the last-seen bearing.
        # The decode-free thermal detector (ENABLE_THERMAL → /usv_relpos/thermal → selector → UKF →
        # vis_fresh) already supplies `acquired`, so the sweep re-acquires WITHOUT waiting for an AR
        # decode — it just has to POINT at the USV fast/wide enough, which the slow legacy dither
        # (0.30 Hz / 0.18 rad/s / 0.8 rad) did not (detector-only run stayed at 58%).  The body-safety
        # half of climb-to-regain (no blind descent) + the divergence/abort guard are KEPT below — a
        # gimbal sweep cannot substitute for those.  thermal_search_recovery=false → legacy slow dither
        # (for A/B).  Far-takeoff cold start (no last-seen anchor) still uses the standalone search node.
        self._thermal_search_recovery = bool(rospy.get_param('~thermal_search_recovery', True))
        _fast = self._thermal_search_recovery
        self._search_scan    = bool(rospy.get_param('~search_scan_enable', True))
        self._search_pitch   = float(rospy.get_param('~search_scan_pitch_rad', math.pi / 2.0 - 0.35))
        self._search_yaw_amp = float(rospy.get_param('~search_scan_yaw_rad', 1.8 if _fast else 0.8))
        self._search_yaw_hz  = float(rospy.get_param('~search_scan_hz', 0.8 if _fast else 0.30))
        self._search_dither_rate = float(rospy.get_param('~search_dither_rate', 0.6 if _fast else 0.18))
        self._div_max        = float(rospy.get_param('~divergence_max_m', 30.0))
        self._div_abort_s    = float(rospy.get_param('~divergence_abort_s', 20.0))
        self._search_abort_s = float(rospy.get_param('~search_abort_s', 30.0))  # bound an unrecoverable search
        self._div_since      = None
        # Test hook: inject a deterministic perception BLACKOUT starting inject_dropout_at_s
        # after first reaching FOLLOW, lasting inject_dropout_dur_s — reproduces the far-field
        # loss (the Mode-A trigger) so baseline-vs-fixed recovery can be benchmarked.  Off by
        # default (at<0).  Anchored to FOLLOW entry so it always lands while following.
        self._inj_drop_at  = float(rospy.get_param('~inject_dropout_at_s', -1.0))
        self._inj_drop_dur = float(rospy.get_param('~inject_dropout_dur_s', 6.0))
        self._follow_t0    = None
        self._inj_blackout = False
        self._inj_blk_prev = False

        # ── mission params (NEAR-FIELD: UWB only <5 m → low altitudes) ─────────
        self._climb_alt  = float(rospy.get_param('~climb_alt_m', 4.0))
        self._follow_alt = float(rospy.get_param('~follow_alt_m', 3.5))
        self._appr_radius  = float(rospy.get_param('~approach_radius_m', 3.0))
        self._appr_alt_tol = float(rospy.get_param('~approach_alt_tol_m', 0.7))
        self._appr_set   = float(rospy.get_param('~approach_settle_s', 2.0))
        # 0.B anti-deadlock escape: when APPROACH drags on past this timeout but we ARE
        # centred-in-cone with a fresh (if flaky) fix held for ~settle_soft, promote without
        # the strict WARM latch.  Stops the indefinite APPROACH limbo seen at 0.3 m/s where the
        # 6-7 m AR hand-off band never sustains WARM at the promotion altitude.
        self._appr_promote_to = float(rospy.get_param('~approach_promote_timeout_s', 25.0))
        self._appr_set_soft   = float(rospy.get_param('~approach_settle_soft_s', 3.0))
        self._alt_down   = float(rospy.get_param('~alt_down_rate_ms', 1.0))
        self._alt_up     = float(rospy.get_param('~alt_up_rate_ms', 1.5))
        # ── final-commit landing (LAND=true) — same gates as v3 ───────────────
        self._land_enable   = bool(rospy.get_param('~land_enable', False))
        self._follow_hold_s = float(rospy.get_param('~follow_hold_s', 6.0))
        self._align_alt     = float(rospy.get_param('~align_alt_m', 1.8))
        self._align_rate    = float(rospy.get_param('~align_rate_ms', 0.4))
        self._align_radius  = float(rospy.get_param('~align_radius_m', 0.8))
        self._commit_radius = float(rospy.get_param('~commit_radius_m', 0.4))
        self._commit_vmax   = float(rospy.get_param('~commit_vmax_ms', 0.30))
        self._commit_hold_s = float(rospy.get_param('~commit_hold_s', 0.6))
        self._commit_rate   = float(rospy.get_param('~commit_rate_ms', 0.8))
        # PREDICTIVE commit (default): commit when the predicted touchdown point
        # r + ṙ·t_descent lands within the deck safe radius, with a ONE-SIDED altitude
        # check (commit from anywhere below align_alt+band, so a descent stall can't
        # block it) + a loose sanity speed cap.  Robust to the high-speed residual
        # oscillation that made the strict (centred AND slow, held 0.6 s) gate never
        # latch.  ~commit_mode=strict restores the old gate.
        self._commit_mode   = str(rospy.get_param('~commit_mode', 'predictive')).lower()
        self._commit_pred_r = float(rospy.get_param('~commit_pred_radius_m', 0.9))
        # Commit-gate fix #1: the predictive miss r+ṙ·t_desc extrapolates the relative
        # velocity over a ~2 s horizon, so at high USV speed the zero-mean ṙ RIPPLE (the
        # UAV chasing a fast deck, never perfectly velocity-matched) swings the prediction
        # past the radius and the gate never latches — even with r tight (~0.17 m).  Use a
        # short EMA of ṙ in the GATE ONLY (alpha≈0.2 ≈ 0.5 s @10 Hz): the ripple averages
        # out, the honest mean (≈velocity-matched) shows through.  Condition-adaptive — at
        # low speed there is no ripple so it is a near-no-op (no threshold/rate change).
        self._commit_vel_lpf = float(rospy.get_param('~commit_vel_lpf', 0.2))
        self._rdot_gate = np.zeros(2)
        self._commit_alt_band = float(rospy.get_param('~commit_alt_band_m', 0.8))
        self._commit_vmax_hard = float(rospy.get_param('~commit_vmax_hard_ms', 1.5))
        self._touchdown_alt = float(rospy.get_param('~touchdown_alt_m', 0.35))
        self._noreturn_alt  = float(rospy.get_param('~noreturn_alt_m', 1.5))
        self._abort_radius  = float(rospy.get_param('~land_abort_radius_m', 2.5))
        # COMMIT lock-retention: how long the committed descent may coast on the EKF's
        # predict-only state through a UWB blip before it bounces back to FOLLOW.  The
        # EKF runs CV-predict at 50 Hz so the relative state stays usable for a beat; a
        # short blip mid-descent should NOT throw away the commit.
        self._commit_lost_to = float(rospy.get_param('~commit_lost_timeout_s', 2.5))
        self._land_ok_since = None
        self._t_prearm   = float(rospy.get_param('~prearm_timeout_s', 10.0))
        self._t_arming   = float(rospy.get_param('~arming_timeout_s', 30.0))
        self._t_climb    = float(rospy.get_param('~climb_timeout_s', 60.0))
        # No-liftoff watchdog (Mode B): a fresh arm lifts past deck_clear in ~5 s; if an
        # attempt is still on the deck after t_liftoff, it's a stochastic post-arm pinning
        # (uncorrelated with deck heave/tilt — measured).  Re-ramping the SAME armed state
        # does nothing, so DISARM + RE-ARM for an independent (~80%) fresh attempt.
        self._t_liftoff  = float(rospy.get_param('~liftoff_timeout_s', 7.0))
        self._t_nadir    = float(rospy.get_param('~nadir_timeout_s', 15.0))
        self._t_search   = float(rospy.get_param('~search_timeout_s', 90.0))
        self._track_to   = float(rospy.get_param('~track_timeout_s', 1.5))

        # ── runtime state ─────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._state = S.PRE_ARMED
        self._state_t = rospy.Time.now().to_sec()
        self._garmin = None
        self._imu_yaw = 0.0
        self._tilt = 0.0
        self._imu_ready = False
        self._gpitch = 0.0
        self._gyaw = 0.0
        self._gimbal_seen_yaw = 0.0      # gimbal yaw/pitch the instant the USV was last SEEN —
        self._gimbal_seen_pitch = math.pi / 2.0  # the directed-search anchor (USV reappears near here)
        # DIRECTED-search state captured at the last sighting: dead-reckon the USV forward from here
        # along its last RELATIVE velocity (observable track twist) and point the gimbal at the
        # PREDICTED bearing — search where it actually went, not a symmetric blind spiral.
        self._seen_off = None            # world-aligned UAV→USV offset at last sighting
        self._seen_vrel = np.zeros(2)    # relative offset velocity at last sighting (= vis_ev)
        self._seen_yaw0 = 0.0            # UAV heading at last sighting
        self._seen_alt0 = -1.0           # altitude at last sighting
        self._seen_t = 0.0               # time of last sighting
        self._mav_armed = False
        self._mav_mode = ''
        self._prev_armed = False
        # EKF relative state (USV frame)
        self._r = None              # [rx, ry] UAV pos in USV frame
        self._rdot = np.zeros(2)    # [vx, vy] relative velocity
        self._rz = None             # relative altitude (available; SM uses garmin)
        self._yaw_rel = 0.0         # ψ_uav − ψ_usv
        self._ff = np.zeros(2)      # a_usv (USV frame)
        self._in_range = False
        self._last_ekf_t = None
        self._last_inrange_t = None
        self._n_ekf = 0
        self._uwb_active = False     # handover latch: UWB owns horizontal
        self._inrange_streak = 0
        self._uwb_lost_since = None   # for the demote timeout
        # ── Signal split: IMU UAV velocity (damping) + clean USV velocity estimate (feedforward) ──
        self._v_uav = np.zeros(2)        # UAV horizontal velocity, leaky-integrated IMU specific force
        self._imu_prev_t = None
        self._vuav_tau = float(rospy.get_param('~vuav_tau_s', 1.2))      # leak time-constant (s)
        # v_usv_est = LPF(v_rel + v_uav): the UKF twist is the RELATIVE velocity (v_usv − v_uav);
        # add back the IMU UAV velocity to recover the USV's ABSOLUTE velocity, then low-pass HARD
        # (the USV is ~constant-velocity) → a clean target-velocity FF uncontaminated by UAV swing.
        self._v_usv_est = np.zeros(2)
        self._vusv_tau = float(rospy.get_param('~vusv_tau_s', 2.5))      # LPF time-constant (s)
        self._signal_split = bool(rospy.get_param('~signal_split', False))
        # IMU-AC damping (SHIPPED as the approach controller): subtract k·v_uav_imu (the UAV's OWN
        # low-lag oscillatory velocity) from the far-field command to brake the overshoot limit-cycle.
        # Needs no absolute v_usv (the IMU velocity's DC is unobservable anyway) — only the in-phase
        # AC swing, which it does capture.  Ablation winner: 100% reach-FOLLOW at 0.3 AND 2.0 m/s
        # (vs 67% for plain PD) + lowest follow jitter.  Default ON (k=1.2).
        self._imu_damp_k = float(rospy.get_param('~imu_damp_k', 1.2))
        # ── CALIBRATE-by-hold (the robust USV-velocity estimator) ──────────────────────────────
        # GPS-free, v_usv is only cleanly observable when the UAV is STILL (v_uav≈0 ⇒ offset-rate =
        # v_usv).  So at the start of APPROACH the UAV HOLDS LEVEL (no chase) while the released USV
        # drifts, and we LINEAR-FIT the offset over a sliding window → slope = v_usv.  RAMP-AWARE:
        # freeze only when consecutive fits agree (velocity has stopped ramping).  Offset-capped so
        # the USV can't drift out of frame during the hold.
        self._calibrate_vusv = bool(rospy.get_param('~calibrate_vusv', False))
        self._cal_active = False
        self._cal_done = False
        self._cal_buf = []                       # (t, off_x, off_y) sliding window for the fit
        self._cal_fit_hist = []                  # (t, fit) history for PLATEAU detection
        self._cal_win = float(rospy.get_param('~cal_window_s', 2.0))     # fit window (s)
        self._cal_min_s = float(rospy.get_param('~cal_min_s', 4.0))      # min hold before accepting
        self._cal_max_s = float(rospy.get_param('~cal_max_s', 16.0))     # hold timeout
        self._cal_tol = float(rospy.get_param('~cal_steady_tol', 0.08))  # |Δfit| plateau threshold m/s
        self._cal_plateau_s = float(rospy.get_param('~cal_plateau_s', 2.5))  # fit must be flat THIS long
        self._cal_max_off = float(rospy.get_param('~cal_max_off_m', 11.0))  # bail if USV drifts past
        self._cal_t0 = None
        # vision far-field state (world-ENU offset, from /usv_track/filtered)
        self._vis_off = None
        self._vis_ev = np.zeros(2)
        self._vis_prev_off = None
        self._vis_prev_t = None
        self._last_vis_track_t = None
        self._vis_visible = False
        self._last_vis_seen_t = None
        self._n_vis = 0
        # USE_LOOKAHEAD (AR-based): aim horizontal at the Bézier-predicted USV position
        # (/usv_track/lookahead ← UKF track ← AR) instead of the current offset — the lead gives the
        # far-field laws the catch-up they otherwise lack.  Gate + scoring keep the TRUE offset.
        self._use_lookahead = bool(rospy.get_param('~use_lookahead', False))
        # APPROACH is HARD-LOCKED to the benchmark-winning front-end (PD + lookahead): the lead is
        # always on during APPROACH regardless of ~use_lookahead, which now scopes FOLLOW/ALIGN only.
        # Rationale: the APPROACH-law benchmark chose PD+lookahead; without the lead the high-speed
        # catch-up lags, the USV leaves the gimbal FOV, and APPROACH relapses to SEARCH (the false
        # "approach deadlocks" seen when a FOLLOW sweep cell happened to set use_lookahead=false).
        self._appr_use_lookahead = bool(rospy.get_param('~approach_use_lookahead', True))
        self._look_off = None
        self._last_look_t = None
        self._look_to = float(rospy.get_param('~lookahead_timeout_s', 0.5))
        # SPEED-GATE (M3 fix): the lead helps the high-speed catch-up but at low USV speed the
        # near-stationary heading is noisy and the Bézier extrapolation can fling the UAV out
        # (measured: 16 m @0.3).  Cap the lead vs the current offset (reject bad extrapolations) and
        # ramp it by the estimated USV speed: 0 below v_lo → no lead, full above v_hi.
        self._look_vlo = float(rospy.get_param('~lookahead_v_lo', 0.5))
        self._look_vhi = float(rospy.get_param('~lookahead_v_hi', 1.5))
        self._look_max = float(rospy.get_param('~lookahead_max_m', 3.0))
        self._prev_roll = 0.0
        self._prev_pitch = 0.0
        self._desired_alt = 0.5
        self._target_alt = 0.5
        self._down_rate = self._alt_down
        self._int_z = 0.0
        self._int_xy = np.zeros(2)
        self._a_b_last = None             # last published horizontal accel (handover jump-smoothing)
        self._a_b_anchor = np.zeros(2)    # command captured at the last UWB↔vision switch
        self._ho_sw_t0 = -1e9             # time of the last handover switch
        self._prev_ez = 0.0
        self._last_ctrl_t = None
        self._params_set = False
        self._appr_ok_since = None
        self._appr_soft_since = None      # 0.B soft-promote dwell timer
        self._src = ''                    # 0.A: selected perception source (/usv_relpos/source)
        self._src_t = 0.0
        self._inner_streak = 0            # 0.C: consecutive ar_inner fixes (hand-off hold gate)
        self._vt_floor = 1e9              # 0.C: lowest vis-traj altitude reached (descent ratchet)
        self._e_horiz = 999.0
        self._arm_srv = self._mode_srv = self._param_srv = self._cmd_srv = None

        # ── pubs ──
        self._pub_att = rospy.Publisher('/%s/mavros/setpoint_raw/attitude' % ns,
                                        AttitudeTarget, queue_size=5)
        self._pub_pitch = rospy.Publisher('/%s/gimbal/position/pitch/command' % ns,
                                          Float64, queue_size=1)
        self._pub_yaw = rospy.Publisher('/%s/gimbal/position/yaw/command' % ns,
                                        Float64, queue_size=1)
        self._pub_state = rospy.Publisher('~state', String, queue_size=5, latch=True)
        self._pub_stat = rospy.Publisher('~status', String, queue_size=5)
        self._pub_herr = rospy.Publisher('~horizontal_error', Vector3Stamped, queue_size=5)
        self._pub_aerr = rospy.Publisher('~altitude_error', Float64, queue_size=5)
        # signal-split debug: estimated USV velocity (x,y) + speed in z, and IMU UAV velocity
        self._pub_vusv = rospy.Publisher('/%s/usv_vel_est' % ns, Vector3Stamped, queue_size=5)
        self._pub_vuav = rospy.Publisher('/%s/uav_vel_imu' % ns, Vector3Stamped, queue_size=5)

        # ── subs ──
        rospy.Subscriber('/%s/mavros/state' % ns, MavState, self._mav_cb, queue_size=5)
        rospy.Subscriber('/%s/mavros/imu/data' % ns, Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber('/%s/garmin/range' % ns, Range, self._range_cb, queue_size=5)
        rospy.Subscriber('/%s/gimbal/joint_states' % ns, JointState, self._js_cb, queue_size=5)
        rospy.Subscriber('/%s/uwb_ekf/rel_odom' % ns, Odometry, self._ekf_cb, queue_size=10)
        rospy.Subscriber('/%s/uwb_ekf/usv_accel_ff' % ns, Vector3Stamped, self._ff_cb, queue_size=10)
        rospy.Subscriber('/%s/uwb/in_range' % ns, Bool, self._inrange_cb, queue_size=5)
        # vision far-field source (only used when use_vision_handover=true)
        rospy.Subscriber('/%s/usv_track/filtered' % ns, Odometry, self._vis_track_cb, queue_size=10)
        # AR Bézier lead (predicted USV offset) — used as the horizontal aim when ~use_lookahead.
        rospy.Subscriber('/%s/usv_track/lookahead' % ns, PoseStamped, self._look_cb, queue_size=10)
        # Phase-1 visibility-constrained descent: when ~vis_traj, the FOLLOW/ALIGN descent altitude
        # is driven by tmp/usv_visibility_traj_node (keeps the USV inside the fixed-nadir cone).
        self._vis_traj = bool(rospy.get_param('~vis_traj', False))
        # APPROACH-phase study: let the vis-traj setpoint own the altitude in APPROACH too (default
        # off — engaging it in APPROACH historically chased the offset oscillation → stuck-high; the
        # per-law approach tuning re-tests whether that holds).  vis_traj2 must also publish there
        # (VT_ENGAGE=APPROACH,FOLLOW,ALIGN).
        self._vt_in_approach = bool(rospy.get_param('~vt_in_approach', False))
        self._vt_alt = None
        self._vt_alt_t = 0.0
        self._vt_rate = 0.0          # Phase-2 trajectory vertical-velocity FF (m/s, <0 descending)
        self._vt_rate_t = 0.0
        self._kvff_z = float(rospy.get_param('~kvff_z', 0.08))   # alt-rate → thrust feed-forward gain
        rospy.Subscriber('/%s/usv_traj/desired_alt' % ns, Float64, self._vt_alt_cb, queue_size=5)
        rospy.Subscriber('/%s/usv_traj/desired_alt_rate' % ns, Float64, self._vt_rate_cb, queue_size=5)
        rospy.Subscriber('/%s/usv_track/visible' % ns, Bool, self._vis_visible_cb, queue_size=10)
        # 0.A: which perception source the selector is on (ar_outer/ar_inner/blob/yolo) — lets us
        # see WHERE in the descent the outer→inner hand-off collapses WARM.
        rospy.Subscriber('/%s/usv_relpos/source' % ns, String, self._src_cb, queue_size=10)

        rospy.Timer(rospy.Duration(1.0 / self._ctrl_hz), self._control_cb)
        rospy.Timer(rospy.Duration(1.0 / self._sm_hz), self._sm_cb)
        rospy.loginfo("[uwb_follow] up — ns=%s law=%s handover=%s climb=%.1fm follow=%.1fm "
                      "Kp=%.2f Kd=%.2f Kff=%.2f", ns, self._law, self._handover,
                      self._climb_alt, self._follow_alt, self._kp_xy, self._kd_xy, self._kff_xy)

    # ── callbacks ──
    def _mav_cb(self, m):
        with self._lock:
            self._mav_armed = bool(m.armed); self._mav_mode = str(m.mode)

    def _imu_cb(self, m):
        q = m.orientation
        R = Rot.from_quat([q.x, q.y, q.z, q.w])
        yaw = R.as_euler('xyz')[2]
        down = R.apply([0, 0, -1.0])
        tilt = math.acos(max(-1.0, min(1.0, -down[2])))
        # Low-lag UAV horizontal velocity by LEAKY integration of the IMU specific force.
        # a_world = R·f_body; gravity is purely vertical so the horizontal kinematic accel is just
        # (R·f_body)[:2].  Leaky integrate (decay τ) → bounds drift/bias while preserving the
        # ~1 Hz overshoot-velocity content we need for DAMPING (signal split: damp the UAV's OWN
        # motion, not the laggy perception-derived relative velocity).
        a_body = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        a_xy = R.apply(a_body)[:2]
        now = rospy.Time.now().to_sec()
        with self._lock:
            self._imu_yaw = yaw; self._tilt = tilt; self._imu_ready = True
            if self._imu_prev_t is not None:
                dtt = now - self._imu_prev_t
                if 0.0 < dtt < 0.1:
                    self._v_uav = (self._v_uav + a_xy * dtt) * math.exp(-dtt / self._vuav_tau)
            self._imu_prev_t = now

    def _range_cb(self, m):
        with self._lock:
            self._garmin = float(m.range)

    def _src_cb(self, m):
        self._src = str(m.data)
        self._src_t = rospy.Time.now().to_sec()
        # 0.C: streak of consecutive inner-marker fixes — the hand-off hold releases the descent
        # below the band only once this is solid (inner reliably decoded, not a flaky single frame).
        if self._src == 'ar_inner':
            self._inner_streak += 1
        else:
            self._inner_streak = 0

    def _vt_alt_cb(self, m):
        self._vt_alt = float(m.data)
        self._vt_alt_t = rospy.Time.now().to_sec()

    def _vt_rate_cb(self, m):
        self._vt_rate = float(m.data)
        self._vt_rate_t = rospy.Time.now().to_sec()

    def _js_cb(self, m):
        lut = dict(zip(m.name, m.position))
        with self._lock:
            self._gpitch = lut.get('%s_gimbal_pitch_joint' % self._ns, self._gpitch)
            self._gyaw = lut.get('%s_gimbal_yaw_joint' % self._ns, self._gyaw)

    def _ekf_cb(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        yaw_rel = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._r = np.array([p.x, p.y])
            self._rz = float(p.z)
            self._rdot = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y])
            self._yaw_rel = float(yaw_rel)
            self._last_ekf_t = rospy.Time.now().to_sec()
            self._n_ekf += 1

    def _ff_cb(self, m):
        with self._lock:
            self._ff = np.array([m.vector.x, m.vector.y])

    def _vis_track_cb(self, m):
        off = np.array([m.pose.pose.position.x, m.pose.pose.position.y])  # world-ENU offset
        # The UKF twist is the CLEAN filtered rate-of-change of the offset (relative
        # velocity) — use it for the damping term instead of noisy self-differencing of
        # an intermittent/outlier-prone offset.  This is the velocity signal that lets
        # the UAV MATCH a moving USV (so it stays near nadir → ray-cast stays accurate),
        # instead of lagging it into the shallow-angle garbage regime.
        tw = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y])
        now = rospy.Time.now().to_sec()
        with self._lock:
            if np.all(np.isfinite(tw)) and np.linalg.norm(tw) < 5.0:
                de = np.clip(tw, -2.5, 2.5)
                self._vis_ev = (1.0 - self._ev_alpha) * self._vis_ev + self._ev_alpha * de
            elif self._vis_prev_t is not None:
                dtt = now - self._vis_prev_t
                if 0.0 < dtt < 0.5:
                    de = np.clip((off - self._vis_prev_off) / dtt, -2.5, 2.5)
                    self._vis_ev = (1.0 - self._ev_alpha) * self._vis_ev + self._ev_alpha * de
            self._vis_prev_off = off
            self._vis_prev_t = now
            self._vis_off = off
            self._last_vis_track_t = now
            self._n_vis += 1

    def _look_cb(self, m):
        # AR-based Bézier look-ahead point (predicted relative offset, world-ENU) — same frame as
        # _vis_off so it is directly substitutable as the horizontal position aim.
        with self._lock:
            self._look_off = np.array([m.pose.position.x, m.pose.position.y])
            self._last_look_t = rospy.Time.now().to_sec()

    def _vis_visible_cb(self, m):
        with self._lock:
            self._vis_visible = bool(m.data)
            if m.data:
                self._last_vis_seen_t = rospy.Time.now().to_sec()

    def _inrange_cb(self, m):
        with self._lock:
            self._in_range = bool(m.data)
            if m.data:
                self._last_inrange_t = rospy.Time.now().to_sec()

    # ── MAVROS service helpers ──
    def _arm(self, val):
        srv = '/%s/mavros/cmd/arming' % self._ns
        if self._arm_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._arm_srv = rospy.ServiceProxy(srv, CommandBool)
            except rospy.ROSException:
                return False
        try:
            return bool(self._arm_srv(CommandBoolRequest(value=val)).success)
        except rospy.ServiceException:
            return False

    def _force_disarm(self):
        """Force-disarm via MAV_CMD_COMPONENT_ARM_DISARM (400) with the PX4 force magic
        (param2=21196).  Needed because a plain disarm is VETOED ("Disarming denied! Not
        landed") when the deck-heave false-triggers PX4's takeoff detector — the veto is
        what blocked the no-liftoff watchdog from ever getting a fresh arm."""
        srv = '/%s/mavros/cmd/command' % self._ns
        if self._cmd_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._cmd_srv = rospy.ServiceProxy(srv, CommandLong)
            except rospy.ROSException:
                return False
        try:
            req = CommandLongRequest()
            req.command = 400          # MAV_CMD_COMPONENT_ARM_DISARM
            req.param1 = 0.0           # 0 = disarm
            req.param2 = 21196.0       # force (bypass land/flying safety veto)
            return bool(self._cmd_srv(req).success)
        except rospy.ServiceException:
            return False

    def _reboot_fcu(self):
        """Reboot the (SITL) autopilot via MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN (246, param1=1).

        NB (proven): PX4 SITL DENIES this (MAV_RESULT_DENIED) on the heaving deck and an in-sim
        FCU reset is impossible anyway under gzserver lockstep — so REBOOT degrades to a plain
        force-disarm + re-arm.  The real Mode-B fix is relaunch-on-ABORT at the harness level
        (tmp/run_with_relaunch.sh).  See memory:px4-lockstep-fcu-reset.
        Mode B is a per-sim-instance PX4/EKF boot artifact: a bad accel-bias/attitude init at
        FCU boot permanently pins the climb for that instance, so NO arm-level retry escapes it
        (force-disarm+re-arm shares the same EKF).  A full FCU reboot forces a FRESH boot — new
        EKF init — which is the only thing that can clear it without restarting Gazebo."""
        srv = '/%s/mavros/cmd/command' % self._ns
        if self._cmd_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._cmd_srv = rospy.ServiceProxy(srv, CommandLong)
            except rospy.ROSException:
                return False
        try:
            req = CommandLongRequest()
            req.command = 246          # MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN
            req.param1 = 1.0           # 1 = reboot autopilot
            resp = self._cmd_srv(req)
            self._reboot_log("reboot cmd246 ack: success=%s result=%s"
                             % (getattr(resp, 'success', '?'), getattr(resp, 'result', '?')))
            return bool(resp.success)
        except rospy.ServiceException as e:
            self._reboot_log("reboot cmd246 ServiceException: %s" % e)
            return False

    def _reboot_log(self, msg):
        """Append a line to a sidecar file that survives the harness SIGKILL (the node's own
        stdout is block-buffered to a pipe and lost on kill)."""
        try:
            t = rospy.Time.now().to_sec()
        except Exception:
            t = -1.0
        line = "[%.2f] %s\n" % (t, msg)
        rospy.logwarn("[uwb_follow] %s", msg)
        try:
            with open(self._reboot_logf, 'a') as f:
                f.write(line); f.flush()
        except Exception:
            pass

    def _trigger_reboot(self, now, reason):
        """No-liftoff watchdog: disarm, then park in REBOOT.  The reboot command is NOT sent
        here — PX4 DENIES a reboot while its land-detector still thinks the vehicle is flying
        (the deck-heave false-takeoff).  REBOOT first waits reboot_predelay_s after disarm for
        the land-detector to settle to 'landed', THEN sends the reboot."""
        self._climb_try += 1
        # disarm and confirm armed→False (force-disarm bypasses the "Not landed" veto)
        self._force_disarm()
        t_end = rospy.Time.now().to_sec() + 2.0
        while rospy.Time.now().to_sec() < t_end and not rospy.is_shutdown():
            with self._lock:
                if not self._mav_armed:
                    break
            rospy.sleep(0.05)
            self._force_disarm()
        with self._lock:
            still_armed, cur_mode = self._mav_armed, self._mav_mode
        self._reboot_log("%s — disarmed (still_armed=%s mode=%s) — REBOOT FCU %d/%d, "
                         "settle %.0fs before reboot"
                         % (reason, still_armed, cur_mode, self._climb_try,
                            self._climb_retries, self._reboot_predelay))
        self._reboot_t0 = now
        self._reboot_sent = False          # REBOOT state sends it after the pre-delay
        self._params_set = False           # re-apply safety params on the fresh boot
        self._prev_armed = False
        self._transition(S.REBOOT)

    def _set_mode(self, mode):
        srv = '/%s/mavros/set_mode' % self._ns
        if self._mode_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._mode_srv = rospy.ServiceProxy(srv, SetMode)
            except rospy.ROSException:
                return False
        try:
            return bool(self._mode_srv(SetModeRequest(custom_mode=mode)).mode_sent)
        except rospy.ServiceException:
            return False

    def _set_param(self, name, val, integer=False):
        srv = '/%s/mavros/param/set' % self._ns
        if self._param_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._param_srv = rospy.ServiceProxy(srv, ParamSet)
            except rospy.ROSException:
                return False
        try:
            req = ParamSetRequest()
            req.param_id = name
            req.value = ParamValue(integer=int(val), real=0.0) if integer \
                else ParamValue(integer=0, real=float(val))
            return bool(self._param_srv(req).success)
        except rospy.ServiceException:
            return False

    # ── helpers ──
    def _transition(self, new):
        rospy.loginfo("[uwb_follow] %s → %s", self._state, new)
        if new != S.APPROACH:
            with self._lock:
                self._int_xy = np.zeros(2)
        else:
            # 0.B: re-arm the promotion altitude on (re)entering APPROACH.  A prior FOLLOW
            # descent (or vis-traj cone) may have driven _desired_alt below follow_alt; the
            # APPROACH descent gate only clamps DOWN, so without this a demoted low UAV could
            # never climb back to the wide-cone promotion altitude → permanent APPROACH limbo
            # (the "stuck-low" deadlock).  Only raises; first-entry climb_alt is unaffected.
            self._desired_alt = max(self._desired_alt, self._follow_alt)
            self._appr_soft_since = None
        if new == S.FOLLOW:
            # 0.C: arm the descent ratchet at follow_alt when the descent phase begins; it then
            # only ratchets DOWN (persisting into ALIGN) so the cone can't re-lift past the cap.
            self._vt_floor = self._follow_alt
        self._state = new
        self._state_t = rospy.Time.now().to_sec()
        self._pub_state.publish(String(data=new))

    def _age(self):
        return rospy.Time.now().to_sec() - self._state_t

    def _ramp_alt(self):
        d, t = self._desired_alt, self._target_alt
        if abs(d - t) < 0.005:
            return
        dt = 1.0 / self._sm_hz
        rate = self._down_rate if d < t else self._alt_up
        step = min(abs(d - t), rate * dt)
        self._target_alt = t + math.copysign(step, d - t)

    # ── 50 Hz control: the ONLY AttitudeTarget publisher ──
    def _control_cb(self, _e):
        with self._lock:
            state = self._state
            garmin = self._garmin
            yaw = self._imu_yaw
            tilt = self._tilt
            imu_ready = self._imu_ready
            target_alt = self._target_alt
            r = None if self._r is None else self._r.copy()
            rdot = self._rdot.copy()
            ff = self._ff.copy()
            yaw_rel = self._yaw_rel
            in_range = self._in_range
            last_ekf_t = self._last_ekf_t
            uwb_active = self._uwb_active
            streak = self._inrange_streak
            uwb_lost_since = self._uwb_lost_since
            vis_off = None if self._vis_off is None else self._vis_off.copy()
            vis_ev = self._vis_ev.copy()
            look_off = None if self._look_off is None else self._look_off.copy()
            last_look_t = self._last_look_t
            v_uav = self._v_uav.copy()
            last_vis_track_t = self._last_vis_track_t
            last_vis_seen_t = self._last_vis_seen_t
            prev_roll, prev_pitch = self._prev_roll, self._prev_pitch
            int_z, prev_ez, last_t = self._int_z, self._prev_ez, self._last_ctrl_t
            int_xy = self._int_xy.copy()

        if not imu_ready:
            return
        uwb_active_prev = uwb_active        # pre-handover value (for the jump-smoothing switch detect)
        now = rospy.Time.now().to_sec()
        dt = (now - last_t) if (last_t is not None and 0 < now - last_t < 0.5) \
            else (1.0 / self._ctrl_hz)
        # Signal split: recover the USV's ABSOLUTE velocity = relative (UKF twist) + UAV (IMU),
        # then hard low-pass (USV ≈ constant velocity) → clean FF.  Updated/ published every tick.
        if not (self._calibrate_vusv and self._cal_done):
            v_usv_inst = vis_ev + v_uav                # continuous fallback estimate (pre-calibration)
            a_lpf = dt / max(dt, self._vusv_tau)
            self._v_usv_est = (1.0 - a_lpf) * self._v_usv_est + a_lpf * v_usv_inst
        v_usv_est = self._v_usv_est                    # frozen calibrated value once cal_done
        self._pub_vusv.publish(Vector3Stamped(vector=Vector3(
            x=float(v_usv_est[0]), y=float(v_usv_est[1]), z=float(np.linalg.norm(v_usv_est)))))
        self._pub_vuav.publish(Vector3Stamped(vector=Vector3(
            x=float(v_uav[0]), y=float(v_uav[1]), z=float(np.linalg.norm(v_uav)))))

        # ── altitude PID (Garmin + IMU) ──
        # deck-clear tracking: count sustained ticks with a clean above-deck reading
        if garmin is not None and garmin * math.cos(tilt) > self._deck_clear:
            self._clear_streak += 1
        else:
            self._clear_streak = 0
        deck_cleared = self._clear_streak >= self._deck_clear_n
        in_takeoff = (state == S.CLIMBING and not deck_cleared)

        if state in (S.TOUCHDOWN, S.LANDED):
            thrust, e_z = 0.0, 0.0
        elif in_takeoff:
            # Mode-B: open-loop thrust ramp, rangefinder ignored (deck heave) —
            # ramp hover→takeoff_thr over takeoff_ramp_s, hold until deck cleared.
            t0 = self._climb_t0 if self._climb_t0 is not None else now
            frac = min(1.0, max(0.0, (now - t0) / max(self._takeoff_ramp, 1e-3)))
            thrust = self._hover + frac * (self._takeoff_thr - self._hover)
            thrust = float(np.clip(thrust, 0.05, 1.0))
            e_z = 0.0
        elif garmin is None:
            thrust, e_z = self._hover, 0.0
        else:
            alt = garmin * math.cos(tilt)
            e_z = target_alt - alt
            int_z = float(np.clip(int_z + e_z * dt, -0.5, 0.5))
            d_ez = (e_z - prev_ez) / dt
            thrust = self._hover + self._kp_z * e_z + self._ki_z * int_z + self._kd_z * d_ez
            # Phase-2: vertical velocity feed-forward from the visibility trajectory's descent rate
            # (track the profile's velocity, not just its position → smoother, less-laggy descent).
            if self._vis_traj and (now - self._vt_rate_t) < 0.5:
                thrust += self._kvff_z * self._vt_rate
            thrust = float(np.clip(thrust, self._hover - self._max_dthr,
                                   self._hover + self._max_dthr))
            thrust = float(np.clip(thrust, 0.05, 1.0))

        # ── source selector (vision far → UWB near): latch + DROP-timeout demote ──
        ekf_fresh = (last_ekf_t is not None and (now - last_ekf_t) < self._ekf_to
                     and in_range)
        track_recent = (last_ekf_t is not None and (now - last_ekf_t) < 1.0)
        if self._inj_blackout:                 # test hook: go fully blind during the blackout
            ekf_fresh = track_recent = False
            vis_off = None
        vis_fresh = (self._handover and vis_off is not None
                     and last_vis_seen_t is not None and (now - last_vis_seen_t) < self._vis_to
                     and last_vis_track_t is not None and (now - last_vis_track_t) < 1.0)
        if ekf_fresh:
            streak += 1
            uwb_lost_since = None
        else:
            streak = 0
            if uwb_lost_since is None:
                uwb_lost_since = now
        # UWB is the TOP of the perception priority sequence (most accurate, ≤10 cm):
        # own horizontal in EVERY active flight phase whenever in range.  The UAV takes
        # off FROM the USV, so UWB is in range (0–5 m) at lift-off → it holds station
        # over the deck during the low climb (no open-loop drift, the FARFIELD_LOST root
        # cause); above 5 m UWB drops out and vision (gimbal already nadir) takes over.
        promotable = state in (S.CLIMBING, S.GIMBAL_NADIR, S.SEARCH, S.APPROACH,
                               S.FOLLOW, S.ALIGN, S.COMMIT)
        if not uwb_active and promotable and streak >= self._ho_streak_n:
            uwb_active = True
            rospy.loginfo("[uwb_follow] *** HANDOVER → UWB (|r|≈%.2f m) ***",
                          float(np.linalg.norm(r)) if r is not None else -1.0)
        elif uwb_active and self._ho_immediate and not in_range and vis_fresh \
                and state in (S.CLIMBING, S.GIMBAL_NADIR, S.SEARCH, S.APPROACH, S.FOLLOW):
            # NB: ALIGN/COMMIT are EXCLUDED — near the deck UWB stays authoritative through a brief
            # in_range flicker (the ho_drop_s coast below debounces it); handing back to AR there lets
            # band-churn thrash the descent (the FOLLOW↔ALIGN border limit-cycle).
            # UWB genuinely out of range (not a brief in-range blip) + AR already fresh → hand to
            # vision NOW, in EVERY phase except the committed final descent (COMMIT keeps its own
            # blind-coast).  Never ride ho_drop_s on stale UWB while a fresh AR fix is available.
            # (in_range=false = real OOR; an in-range stale fix still coasts ho_drop_s as anti-chatter.)
            uwb_active = False
            rospy.loginfo("[uwb_follow] UWB out-of-range + vision fresh → VISION (immediate, %s)", state)
        elif uwb_active and uwb_lost_since is not None \
                and (now - uwb_lost_since) > self._ho_drop_s:
            uwb_active = False
            rospy.logwarn("[uwb_follow] UWB lost >%.1fs → fall back to vision", self._ho_drop_s)
        # while latched, UWB OWNS horizontal — coast through brief blips, never jitter to vision
        coasting = uwb_active and not ekf_fresh

        # ── horizontal control ────────────────────────────────────────────────
        a_b = None
        e_horiz = 0.0
        active = state in (S.CLIMBING, S.GIMBAL_NADIR, S.SEARCH, S.APPROACH, S.FOLLOW,
                           S.ALIGN, S.COMMIT)
        # #4 Mode-A: when perception is genuinely LOST (SEARCH recovery, or an injected blackout),
        # HOLD level (no horizontal accel) instead of chasing the stale UKF prediction + feedforward.
        # That chase is what flies the UAV AWAY during a loss (measured: UAV ran 171 m while the USV
        # moved 34 m), leaving the area so the climb+scan recovery can never re-acquire.  Holding
        # lets drag brake the UAV ~in place so the widened FOV can re-find the USV.
        blind_hold = self._modeA_fix and (state == S.SEARCH or self._inj_blackout)
        cal_hold = (state == S.APPROACH and self._cal_active)   # hold level while calibrating v_usv
        if active and (blind_hold or cal_hold):
            a_b = np.zeros(2)
            e_horiz = self._e_horiz
        elif active and uwb_active and r is not None:
            # NEAR field: UWB owns horizontal.  Fresh → MPC/PD+FF; brief blip → coast/brake.
            e_horiz = float(np.linalg.norm(r))
            if state == S.COMMIT and coasting:
                a_b = np.zeros(2)                       # blind committed descent → coast level
            elif coasting:
                rdot = rdot * self._brake_decay         # brake drift on a brief UWB blip
                with self._lock:
                    self._rdot = rdot
                a_b = horizontal_pd_ff(r, rdot, ff, yaw_rel, 0.0, self._nf_kd_xy,
                                       self._nf_kff_xy, int_xy=int_xy, ki=self._ki_xy, deadband=1e9)
            else:
                if e_horiz < self._int_enable:
                    int_xy = np.clip(int_xy + (-r) * dt, -self._max_int_xy, self._max_int_xy)
                if self._mpc is not None:
                    a_b = mpc_horizontal(r, rdot, ff, yaw_rel, self._mpc)
                elif self._mppi is not None:
                    a_b = mppi_l1_horizontal(r, rdot, ff, yaw_rel, self._mppi)
                else:
                    a_b = horizontal_pd_ff(r, rdot, ff, yaw_rel, self._nf_kp_xy, self._nf_kd_xy,
                                           self._nf_kff_xy, int_xy=int_xy, ki=self._ki_xy,
                                           deadband=self._deadband)
        elif active and self._handover and vis_fresh and vis_off is not None:
            # FAR field: vision drives horizontal (world-ENU offset).  Same sign convention as the
            # UWB path — r = UAV-in-target frame = −vis_off, rdot = −vis_ev — so the MPC/MPPI control
            # laws run UNCHANGED on the gimbal-camera state (UWB-free landing); their Rz(−yaw_rel)
            # output rotation with yaw_rel=yaw matches the v3 PD's Rz(−imu_yaw).  Default = PD+FF.
            e_horiz = float(np.linalg.norm(vis_off))
            # USE_LOOKAHEAD: aim the POSITION term at the AR Bézier lead (predicted USV position) so
            # the far-field law gets catch-up feed-forward; e_horiz above (true current offset) still
            # drives the promotion gate + scoring.  Falls back to the live offset if the lead stales.
            # APPROACH always uses the lead (winner); FOLLOW/ALIGN/etc. honour the cell's ~use_lookahead.
            la_on = self._appr_use_lookahead if state == S.APPROACH else self._use_lookahead
            look_fresh = (la_on and look_off is not None
                          and last_look_t is not None and (now - last_look_t) < self._look_to)
            tgt_off = vis_off
            if look_fresh:
                lead = look_off - vis_off                 # predicted displacement over the horizon
                ln = float(np.linalg.norm(lead))
                if ln > self._look_max:                   # reject bad low-speed extrapolations
                    lead = lead * (self._look_max / ln)
                spd = float(np.linalg.norm(v_usv_est))    # USV speed estimate (UKF rel + IMU, LPF)
                g = float(np.clip((spd - self._look_vlo)
                                  / max(self._look_vhi - self._look_vlo, 1e-3), 0.0, 1.0))
                tgt_off = vis_off + g * lead              # g=0 near-stationary → no lead (safe)
            if self._vision_only and (self._mpc_vis is not None or self._mppi_vis is not None):
                # Integral action closes the steady-state lag to the MOVING USV: a gentle (gimbal-
                # safe) MPC/MPPI otherwise holds a ~v·τ position offset (~4 m at 1 m/s) and never
                # tightens enough to commit on the deck.  Enable the integral OUT to vis_int_enable
                # (the lag is bigger than the normal 1.5 m window) with its own gain/clamp.
                # Only integrate in the steady FOLLOW/landing phase — integrating during the
                # APPROACH transient winds up and destabilises (overshoots, loses the gimbal).
                if state in (S.FOLLOW, S.ALIGN, S.COMMIT) and e_horiz < self._vis_int_enable:
                    int_xy = np.clip(int_xy + tgt_off * dt, -self._vis_max_int, self._vis_max_int)
                if self._mpc_vis is not None:
                    a_b = mpc_horizontal(-tgt_off, -vis_ev, np.zeros(2), yaw, self._mpc_vis)
                else:
                    a_b = mppi_l1_horizontal(-tgt_off, -vis_ev, np.zeros(2), yaw, self._mppi_vis)
                a_b = a_b + Rot.from_euler('z', -yaw).apply(
                    np.append(self._vis_ki * int_xy, 0.0))[:2]
            elif self._appr_kind == 'mpc' and self._appr_ctrl is not None:
                # APPROACH-law ablation: receding-horizon MPC on the gimbal-camera far-field state.
                a_b = mpc_horizontal(-tgt_off, -vis_ev, np.zeros(2), yaw, self._appr_ctrl)
                # AC inertial damping — same term + param as the PD path (prevents the cone overshoot
                # the relative-velocity weight q_vel can miss).  a_b is body-frame, so rotate v_uav
                # the same way the PD path does (Rz(−yaw)); _v_uav is the leaky-integrated (AC-only)
                # IMU velocity, so it never fights the useful steady motion.
                a_b = a_b - Rot.from_euler('z', -yaw).apply(
                    np.append(self._imu_damp_k * v_uav, 0.0))[:2]
            elif self._appr_kind == 'mppi' and self._appr_ctrl is not None:
                a_b = mppi_l1_horizontal(-tgt_off, -vis_ev, np.zeros(2), yaw, self._appr_ctrl)
                a_b = a_b - Rot.from_euler('z', -yaw).apply(    # AC inertial damping (as MPC/PD)
                    np.append(self._imu_damp_k * v_uav, 0.0))[:2]
            else:
                # PD+FF far-field (run17 default) — the regression-fixed handover path.
                if e_horiz < self._int_enable:
                    int_xy = np.clip(int_xy + tgt_off * dt, -self._max_int_xy, self._max_int_xy)
                a_world = self._kp_xy * tgt_off + self._kd_xy * vis_ev + self._ki_xy * int_xy
                a_world = a_world - self._imu_damp_k * v_uav     # IMU-AC damping (0 = off)
                a_b = Rot.from_euler('z', -yaw).apply(np.append(a_world, 0.0))[:2]
        elif active and self._handover and vis_off is not None and last_vis_track_t is not None \
                and (now - last_vis_track_t) < self._vis_coast_to:
            # FAR-field COAST (Stage-0 baseline — the WINNER): coast toward the static last offset +
            # velocity.  Coast-on-Bézier-prediction was tried both raw (Stage 1, 46%) and capped/speed-
            # gated (Stage 1B, 50%) and BOTH lost to this simple static coast (63%) — see
            # findings_followland_stage1.md.  Controller loss-handling caps ~63%; the AR dropout itself
            # is the binding constraint (→ thermal-perception backup is the real fix).
            e_horiz = float(np.linalg.norm(vis_off))
            a_world = self._kp_coast * self._kp_xy * vis_off + self._kd_xy * vis_ev
            a_b = Rot.from_euler('z', -yaw).apply(np.append(a_world, 0.0))[:2]
        elif active and not self._handover and track_recent and r is not None:
            # pure-UWB ablation: brief-dropout brake / COMMIT coast (no vision fallback)
            e_horiz = float(np.linalg.norm(r))
            if state == S.COMMIT:
                a_b = np.zeros(2)               # blind committed descent → coast level
            else:
                rdot = rdot * self._brake_decay
                with self._lock:
                    self._rdot = rdot
                a_b = horizontal_pd_ff(r, rdot, ff, yaw_rel, 0.0, self._kd_xy,
                                       self._kff_xy, int_xy=int_xy, ki=self._ki_xy,
                                       deadband=1e9)

        # ── handover jump-smoothing ──────────────────────────────────────────────
        # On a UWB↔vision switch the horizontal command can STEP (the two estimators differ).  Ramp
        # a_b from the command at the switch instant to the new source's command over ho_smooth_tau,
        # so the attitude doesn't jerk.  (Distinct from tilt_slew below, which rate-limits all motion;
        # this anchors specifically to the pre-switch command for continuity through the handover.)
        if uwb_active != uwb_active_prev and self._a_b_last is not None:
            self._ho_sw_t0 = now
            self._a_b_anchor = self._a_b_last.copy()
        if a_b is not None and (now - self._ho_sw_t0) < self._ho_smooth_tau:
            w = float(np.clip((now - self._ho_sw_t0) / max(self._ho_smooth_tau, 1e-3), 0.0, 1.0))
            a_b = w * a_b + (1.0 - w) * self._a_b_anchor
        if a_b is not None:
            self._a_b_last = a_b.copy()

        dmax = self._tilt_slew * dt
        if a_b is not None:
            pitch_raw = float(np.clip(a_b[0] / _G, -self._max_tilt, self._max_tilt))
            roll_raw = float(np.clip(-a_b[1] / _G, -self._max_tilt, self._max_tilt))
            pitch_cmd = prev_pitch + float(np.clip(pitch_raw - prev_pitch, -dmax, dmax))
            roll_cmd = prev_roll + float(np.clip(roll_raw - prev_roll, -dmax, dmax))
        else:
            pitch_cmd = prev_pitch - float(np.clip(prev_pitch, -dmax, dmax))
            roll_cmd = prev_roll - float(np.clip(prev_roll, -dmax, dmax))

        q = Rot.from_euler('xyz', [roll_cmd, pitch_cmd, yaw]).as_quat()
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.header.frame_id = 'world'
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE
                         | AttitudeTarget.IGNORE_PITCH_RATE
                         | AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                     z=float(q[2]), w=float(q[3]))
        att.thrust = thrust
        self._pub_att.publish(att)

        v3 = Vector3Stamped()
        v3.header.stamp = att.header.stamp
        v3.vector = Vector3(x=0.0, y=0.0, z=e_horiz)
        self._pub_herr.publish(v3)
        self._pub_aerr.publish(Float64(data=e_z))
        self._pub_stat.publish(String(data=('UWB:' if uwb_active else 'VIS:') + state))

        with self._lock:
            self._int_z, self._prev_ez, self._last_ctrl_t = int_z, e_z, now
            self._int_xy = int_xy
            self._e_horiz = e_horiz
            self._uwb_active = uwb_active
            self._inrange_streak = streak
            self._uwb_lost_since = uwb_lost_since
            self._prev_roll, self._prev_pitch = roll_cmd, pitch_cmd

    def _descent_step(self, e_horiz, alt_now):
        """Centering-scaled descent (user's idea): sink FAST when the USV is near the cone
        axis (right below the UAV, gimbal ~nadir), slow as it goes off-centre, and HOLD
        outside the cone.  Returns the per-tick desired-altitude decrement [m].
            cone_r = tan(cone)·alt ;  frac = 1 − e/cone_r ∈ (0,1] ;  step = rate·frac/sm_hz
        Generalises the binary cone gate into a smooth rate → reaches the UWB cylinder
        sooner when lined up, never descends blind/offset into the detection hole."""
        cone_r = max(self._descend_cone_min, self._descend_cone * max(alt_now, 0.0))
        frac = 1.0 - e_horiz / cone_r
        if frac <= 0.0:
            return 0.0
        return self._follow_descent_rate * frac / self._sm_hz

    @staticmethod
    def _fit_vel(buf):
        """Least-squares slope of offset(t) over the window → constant velocity [vx,vy].
        Valid only when the UAV is ~stationary (offset rate = USV velocity).  None if too few pts."""
        if len(buf) < 6:
            return None
        t = np.array([s[0] for s in buf]); t = t - t[0]
        if t[-1] - t[0] < 0.8:
            return None
        A = np.vstack([t, np.ones_like(t)]).T
        vx = np.linalg.lstsq(A, np.array([s[1] for s in buf]), rcond=None)[0][0]
        vy = np.linalg.lstsq(A, np.array([s[2] for s in buf]), rcond=None)[0][0]
        return np.array([vx, vy])

    # ── 10 Hz state machine (same structure as v3) ──
    def _sm_cb(self, _e):
        with self._lock:
            state = self._state
            garmin = self._garmin
            gpitch = self._gpitch
            gyaw = self._gyaw
            armed = self._mav_armed
            mode = self._mav_mode
            in_range = self._in_range
            last_ekf_t = self._last_ekf_t
            last_inrange_t = self._last_inrange_t
            uwb_active = self._uwb_active
            last_vis_track_t = self._last_vis_track_t
            last_vis_seen_t = self._last_vis_seen_t
            n_vis = self._n_vis
            e_horiz = self._e_horiz
            r_sm = None if self._r is None else self._r.copy()      # UWB rel pos (USV frame)
            rdot_sm = self._rdot.copy()                            # UWB rel velocity
            ev_mag = float(np.linalg.norm(self._rdot))
            n_ekf = self._n_ekf
            vis_off_sm = None if self._vis_off is None else self._vis_off.copy()
            vis_ev_sm = self._vis_ev.copy()       # relative offset velocity (track twist = OBSERVABLE)
            imu_yaw_sm = self._imu_yaw            # UAV heading (for the directed-search bearing)
        # EMA of the relative velocity for the commit-gate prediction only (fix #1)
        a = self._commit_vel_lpf
        self._rdot_gate = (1.0 - a) * self._rdot_gate + a * rdot_sm
        rdot_gate = self._rdot_gate.copy()
        now = rospy.Time.now().to_sec()
        age = self._age()
        self._ramp_alt()
        uwb_warm = (n_ekf >= self._warm_n and last_inrange_t is not None
                    and (now - last_inrange_t) < self._ekf_to and in_range)
        uwb_fresh = (last_ekf_t is not None and (now - last_ekf_t) < self._track_to
                     and in_range)
        # vision far-field acquisition (only in handover mode)
        vis_fresh = (self._handover and last_vis_seen_t is not None
                     and (now - last_vis_seen_t) < self._vis_to
                     and last_vis_track_t is not None
                     and (now - last_vis_track_t) < self._track_to)
        vis_warm = vis_fresh and n_vis >= self._warm_n
        # acquired = far-field (vision, handover) OR near-field (UWB) has a fix
        acquired = uwb_fresh or vis_fresh
        warm = uwb_warm or vis_warm
        alt_now = garmin if garmin is not None else -1.0
        # Test hook: a deterministic perception blackout makes the controller go blind (Mode-A
        # trigger), forcing the loss-recovery path so baseline vs fixed can be compared.
        self._inj_blackout = self._blackout_active(now)
        if self._inj_blackout != self._inj_blk_prev:
            tf = (now - self._follow_t0) if self._follow_t0 is not None else -1.0
            self._reboot_log("BLACKOUT %s  state=%s  t_since_follow=%.1f  inj_at=%.1f dur=%.1f"
                             % ("START" if self._inj_blackout else "END", state, tf,
                                self._inj_drop_at, self._inj_drop_dur))
        self._inj_blk_prev = self._inj_blackout
        if self._inj_blackout:
            acquired = False
            warm = False
            vis_fresh = uwb_fresh = False
        # While the USV is SEEN, remember where the gimbal is pointing — that's the directed-search
        # anchor on a loss (the USV reappears near where it left the frame, not at a blind sweep).
        if acquired and not self._inj_blackout:
            self._gimbal_seen_yaw = gyaw
            self._gimbal_seen_pitch = gpitch
            if vis_off_sm is not None:
                self._seen_off = vis_off_sm.copy()
                self._seen_vrel = vis_ev_sm.copy()
                self._seen_yaw0 = imu_yaw_sm
                self._seen_alt0 = alt_now
                self._seen_t = now

        if state == S.PRE_ARMED:
            if not self._params_set and age >= 0.5:
                # COM_DISARM_PRFLT is a FLOAT param (seconds) — setting it as an integer
                # triggers "param types mismatch" + mavros resend/timeout spam and the
                # all() never completes.  Use float.  (px4_open_arming.py also sets it.)
                ok = all([self._set_param('COM_DISARM_PRFLT', -1.0, False),
                          self._set_param('NAV_RCL_ACT', 0, True),
                          self._set_param('COM_OBL_ACT', 1, True)])
                if ok:
                    self._params_set = True
                    rospy.loginfo("[uwb_follow] PX4 safety params set")
            if age >= self._t_prearm:
                self._transition(S.ABORT)
            elif garmin is not None and age >= 3.0:
                self._climb_try = 0          # reset attempt counter on FIRST arm only
                self._transition(S.ARMING)

        elif state == S.ARMING:
            disarm_edge = self._prev_armed and not armed
            self._prev_armed = armed
            if armed and mode == 'OFFBOARD':
                self._desired_alt = self._climb_alt
                self._climb_t0 = now            # start (or restart) open-loop takeoff ramp
                self._clear_streak = 0          # NB: keep _climb_try across re-arm cycles
                self._transition(S.CLIMBING)
            else:
                periodic = int(age * self._sm_hz) % int(2.0 * self._sm_hz) == 0
                if periodic or disarm_edge:
                    if mode != 'OFFBOARD':
                        self._set_mode('OFFBOARD')
                    if not armed:
                        self._arm(True)
                    t_poll = now + 0.2
                    while rospy.Time.now().to_sec() < t_poll and not rospy.is_shutdown():
                        rospy.sleep(0.01)
                        with self._lock:
                            armed, mode = self._mav_armed, self._mav_mode
                        if armed and mode == 'OFFBOARD':
                            break
                if age > self._t_arming:
                    self._transition(S.ABORT)

        elif state == S.REBOOT:
            # Phase 1 (0 .. predelay): keep the vehicle disarmed and let PX4's land-detector
            # settle to 'landed' so the reboot is no longer DENIED.  Then send the reboot.
            # Phase 2 (predelay .. predelay+settle): if the reboot was accepted the FCU is
            # re-booting (fresh EKF); wait, re-apply safety params, then re-arm.
            if not self._reboot_sent:
                if armed:
                    self._force_disarm()           # hold disarmed while the detector settles
                if age >= self._reboot_predelay:
                    with self._lock:
                        m0 = self._mav_mode
                    ok = self._reboot_fcu()
                    if not ok:
                        # still denied → drop OFFBOARD and retry once
                        self._set_mode('AUTO.LOITER')
                        rospy.sleep(0.3)
                        ok = self._reboot_fcu()
                    self._reboot_log("reboot sent after %.0fs predelay (mode=%s) → %s"
                                     % (self._reboot_predelay, m0,
                                        "ACCEPTED" if ok else "DENIED"))
                    self._reboot_sent = True
                    self._transition(S.REBOOT)      # reset the state clock for the settle phase
            else:
                if not self._params_set and age >= self._reboot_settle * 0.6:
                    ok = all([self._set_param('COM_DISARM_PRFLT', -1.0, False),
                              self._set_param('COM_DISARM_LAND', -1.0, False),
                              self._set_param('NAV_RCL_ACT', 0, True),
                              self._set_param('COM_OBL_ACT', 1, True)])
                    if ok:
                        self._params_set = True
                        rospy.loginfo("[uwb_follow] post-reboot safety params re-applied")
                if age >= self._reboot_settle:
                    rospy.logwarn("[uwb_follow] reboot settle done → re-arming (try %d/%d)",
                                  self._climb_try, self._climb_retries)
                    self._prev_armed = False
                    self._transition(S.ARMING)

        elif state == S.CLIMBING:
            t_attempt = (now - self._climb_t0) if self._climb_t0 is not None else age
            stuck = (self._clear_streak == 0 and t_attempt > self._t_liftoff)
            # one-shot deterministic mechanism check: force ONE reboot on the first attempt
            force_now = (self._force_reboot_once and not self._forced_reboot_done
                         and self._climb_try == 0 and t_attempt > 2.0)
            if force_now:
                self._forced_reboot_done = True
                self._trigger_reboot(now, "FORCED (mechanism check)")
            elif garmin is not None and garmin >= self._climb_alt - 0.5:
                self._transition(S.GIMBAL_NADIR)
            elif stuck or t_attempt > self._t_climb:
                # No-liftoff watchdog: armed but never cleared the deck within t_liftoff.
                # PROVEN per-sim-instance EKF boot artifact — NO arm-level retry escapes it
                # (force-disarm+re-arm shares the same corrupt EKF init, stayed 8/10).  The
                # only fix without restarting Gazebo is a full FCU REBOOT → fresh EKF init.
                if self._climb_try < self._climb_retries:
                    self._trigger_reboot(now, "no liftoff in %.1fs (garmin=%.1f, clear_streak=%d)"
                                         % (t_attempt, garmin or -1, self._clear_streak))
                else:
                    rospy.logerr("[uwb_follow] no liftoff after %d reboots → ABORT",
                                 self._climb_retries)
                    self._transition(S.ABORT)
            else:
                rospy.loginfo_throttle(3.0, "[uwb_follow] CLIMBING garmin=%.1f/%.1f "
                                       "clear_streak=%d try=%d",
                                       garmin or -1, self._climb_alt,
                                       self._clear_streak, self._climb_try)

        elif state == S.GIMBAL_NADIR:
            self._pub_pitch.publish(Float64(data=math.pi / 2.0))
            self._pub_yaw.publish(Float64(data=0.0))
            # proceed once the gimbal is near nadir OR UWB already has a fix
            if gpitch >= 1.2 or acquired:
                rospy.loginfo("[uwb_follow] nadir/acquired (gpitch=%.0f° uwb=%s) → SEARCH",
                              math.degrees(gpitch), acquired)
                self._transition(S.SEARCH)
            elif age > self._t_nadir:
                # Don't kill the flight if the gimbal was slow to hit exact nadir — the
                # FF/IBVS node points the gimbal continuously regardless; nadir is just an
                # init. Fall through to SEARCH (which waits for acquisition, never aborts).
                rospy.logwarn("[uwb_follow] nadir timeout (gpitch=%.0f°) — proceed to SEARCH anyway",
                              math.degrees(gpitch))
                self._transition(S.SEARCH)

        elif state == S.SEARCH:
            if acquired:
                rospy.loginfo("[uwb_follow] acquired — APPROACH (stable-lock descent gate)")
                self._appr_ok_since = None
                if not self._modeA_fix:
                    self._desired_alt = self._follow_alt      # legacy: descend immediately
                # else #1: hold current (high) altitude; APPROACH descends once the lock is WARM
                self._transition(S.APPROACH)
            elif self._modeA_fix:
                # GIMBAL-LED recovery: keep the UAV STABLE (blind-hold owns horizontal — no dramatic
                # attitude chasing) and SEARCH with the high-rate GIMBAL in an EXPANDING dither around
                # where the USV was LAST SEEN (it reappears near where it left the frame, not at a
                # blind sweep).  A gentle altitude climb widens the FOV as a slow backup.  IBVS is
                # silent without a detection, so we own the gimbal here; it reclaims it on re-acquire.
                # Climb to widen the FOV for re-acquisition (load-bearing) — but ONLY while UWB is not
                # freshly owning the body: the instant UWB re-engages, stop climbing (UWB flies).  The
                # gimbal directed search (below) runs in parallel.
                if self._loss_recovery_climb and not uwb_fresh:
                    self._desired_alt = min(self._search_alt,
                                            self._desired_alt + self._search_climb / self._sm_hz)
                if self._search_scan:
                    # DIRECTED anchor: dead-reckon the USV from the last sighting along its last
                    # RELATIVE velocity (observable track twist — NOT the GPS-unobservable absolute
                    # USV velocity) and point the gimbal at that PREDICTED bearing.  Differential vs
                    # the known-good last-seen gimbal angle, so it cancels the absolute-frame/optical
                    # convention AND auto-corrects the UAV heading change (incl. wave-tilt) that lost
                    # the USV.  The expanding dither then covers prediction error: tight first (re-
                    # acquire fast when the predict is good), widening only if the USV maneuvered.
                    yaw_anchor, pitch_anchor = self._gimbal_seen_yaw, self._gimbal_seen_pitch
                    if self._thermal_search_recovery and self._seen_off is not None:
                        tl = max(0.0, now - self._seen_t)
                        off_pred = self._seen_off + self._seen_vrel * tl
                        b0 = math.atan2(self._seen_off[1], self._seen_off[0])
                        b1 = math.atan2(off_pred[1], off_pred[0])
                        d0 = max(0.3, float(np.linalg.norm(self._seen_off)))
                        d1 = max(0.3, float(np.linalg.norm(off_pred)))
                        H0 = self._seen_alt0 if self._seen_alt0 > 0.0 else max(alt_now, 0.5)
                        H1 = alt_now if alt_now > 0.0 else H0
                        dbear = math.atan2(math.sin(b1 - b0), math.cos(b1 - b0))   # predicted Δbearing
                        duav = math.atan2(math.sin(imu_yaw_sm - self._seen_yaw0),  # UAV heading change
                                          math.cos(imu_yaw_sm - self._seen_yaw0))
                        yaw_anchor = self._gimbal_seen_yaw + dbear - duav
                        pitch_anchor = self._gimbal_seen_pitch + (math.atan2(H1, d1) - math.atan2(H0, d0))
                    rad = min(self._search_yaw_amp, self._search_dither_rate * age)
                    ph = 2.0 * math.pi * self._search_yaw_hz * age
                    g_yaw = yaw_anchor + rad * math.cos(ph)
                    g_pitch = float(np.clip(pitch_anchor + rad * math.sin(ph),
                                            0.7, math.pi / 2.0 + 0.05))
                    self._pub_yaw.publish(Float64(data=g_yaw))
                    self._pub_pitch.publish(Float64(data=g_pitch))
                if age > self._search_abort_s:
                    rospy.logerr("[uwb_follow] SEARCH %.0fs, no re-acquire → ABORT (divergence)", age)
                    self._transition(S.ABORT)
                else:
                    rospy.loginfo_throttle(3.0, "[uwb_follow] SEARCH recover: climb→%.1fm scan age=%.0fs",
                                           self._desired_alt, age)
            elif age > self._t_search:
                rospy.logwarn("[uwb_follow] search timeout — holding (stay SEARCH)")
                self._state_t = now
            else:
                rospy.loginfo_throttle(5.0, "[uwb_follow] SEARCH (%.0fs) in_range=%s",
                                       age, in_range)

        elif state == S.APPROACH:
            # ── CALIBRATE-by-hold: measure v_usv BEFORE chasing (UAV holds level via cal_hold) ──
            if self._calibrate_vusv and not self._cal_done:
                if not self._cal_active:
                    self._cal_active = True; self._cal_t0 = now
                    self._cal_buf = []; self._cal_fit_hist = []
                    rospy.loginfo("[uwb_follow] CALIBRATE v_usv — holding level, fitting offset…")
                if vis_off_sm is not None and vis_fresh:
                    self._cal_buf.append((now, float(vis_off_sm[0]), float(vis_off_sm[1])))
                self._cal_buf = [s for s in self._cal_buf if now - s[0] <= self._cal_win]
                held = now - self._cal_t0
                off_mag = float(np.linalg.norm(vis_off_sm)) if vis_off_sm is not None else 0.0
                fit = self._fit_vel(self._cal_buf)
                done = False
                if fit is not None:
                    self._cal_fit_hist.append((now, fit))
                    self._cal_fit_hist = [(t, f) for (t, f) in self._cal_fit_hist
                                          if now - t <= self._cal_plateau_s + 0.5]
                    # PLATEAU: the fit must be flat vs the fit from ~plateau_s AGO (a slow ramp is NOT
                    # flat over that span — only the post-ramp constant velocity is).  Ramp-aware.
                    old = [(t, f) for (t, f) in self._cal_fit_hist if now - t >= self._cal_plateau_s]
                    if held >= self._cal_min_s and old \
                            and np.linalg.norm(fit - old[0][1]) < self._cal_tol:
                        done = True
                if held >= self._cal_max_s or off_mag >= self._cal_max_off:
                    done = True                                  # timeout / about to lose USV → take it
                if done and fit is not None:
                    self._v_usv_est = fit; self._cal_done = True; self._cal_active = False
                    rospy.loginfo("[uwb_follow] CALIBRATE done: v_usv=%.2f m/s [%.2f,%.2f] "
                                  "(held %.1fs |off|=%.1fm)", float(np.linalg.norm(fit)),
                                  fit[0], fit[1], held, off_mag)
                else:
                    rospy.loginfo_throttle(2.0, "[uwb_follow] CALIBRATE… held=%.1fs |off|=%.1fm "
                                           "v_fit=%.2f", held, off_mag,
                                           float(np.linalg.norm(fit)) if fit is not None else -1.0)
                    return                                       # keep holding; gimbal still tracks
            # #1 stable-lock descent gate: close the horizontal gap at the (high) search
            # altitude and only begin descending to follow_alt once the lock is stable.
            # 0.B: the strict WARM latch (N-consecutive fresh fixes) rarely sustains through the
            # 6-7 m AR hand-off band, so the UAV could sit at climb_alt forever ("stuck-high").
            # A fresh fix held for ~settle is enough to BEGIN the descent (descending TO follow_alt
            # is always safe; crossing BELOW the band is gated separately in Stage 0.C).
            if self._modeA_fix and self._desired_alt > self._follow_alt + 0.01:
                if warm or (acquired and self._age() > self._appr_set):
                    self._desired_alt = self._follow_alt
            # 0.B: reject only being too HIGH — descending past follow_alt with a good fix is
            # progress, not a deadlock.  The old |alt-follow|<tol locked the UAV out once it had
            # sunk below follow_alt (the descent gate only clamps DOWN, so it couldn't climb back).
            at_alt = (alt_now <= self._follow_alt + self._appr_alt_tol)
            # 0.B: cone-aware horizontal gate — only promote when the USV is actually inside the
            # camera cone at THIS altitude.  At follow_alt the cone radius exceeds appr_radius, so
            # this is a no-op at the nominal promotion altitude (no regression); it only tightens
            # when promoting from BELOW follow_alt (the relaxed at_alt case), where observability
            # is the real risk.
            cone_r = max(self._descend_cone_min, self._descend_cone * max(alt_now, 0.0))
            e_ok = e_horiz < min(self._appr_radius, cone_r)
            gate = (e_ok and at_alt and warm)
            if gate:
                if self._appr_ok_since is None:
                    self._appr_ok_since = now
                elif now - self._appr_ok_since >= self._appr_set:
                    rospy.loginfo("[uwb_follow] gate met (|e|=%.2f m, alt=%.1f m, warm) "
                                  "— FOLLOW @ %.1f m", e_horiz, alt_now, self._follow_alt)
                    self._desired_alt = self._follow_alt
                    self._transition(S.FOLLOW)
                    return
            else:
                self._appr_ok_since = None
            # 0.B: anti-deadlock soft-promote — if APPROACH has dragged past the promote timeout
            # but we ARE centred-in-cone with a fresh (if flaky) fix held for settle_soft, promote
            # WITHOUT the strict WARM latch.  Kills the indefinite APPROACH limbo at 0.3 m/s.
            soft = (e_ok and at_alt and acquired)
            if soft and self._age() > self._appr_promote_to:
                if self._appr_soft_since is None:
                    self._appr_soft_since = now
                elif now - self._appr_soft_since >= self._appr_set_soft:
                    rospy.logwarn("[uwb_follow] APPROACH soft-promote (|e|=%.2f m, alt=%.1f m, "
                                  "acq, age=%.0fs, src=%s) — FOLLOW @ %.1f m", e_horiz, alt_now,
                                  self._age(), self._src, self._follow_alt)
                    self._desired_alt = self._follow_alt
                    self._transition(S.FOLLOW)
                    return
            else:
                self._appr_soft_since = None
            if not gate:
                # 0.A: enriched trace — _desired_alt, acquired, and the perception source make the
                # stuck-high vs stuck-low deadlock (and WHERE WARM collapses) visible in the log.
                rospy.loginfo_throttle(3.0, "[uwb_follow] APPROACH |e|=%.2f/%.1f m  alt=%.1f/%.1f "
                                       "desA=%.1f  warm=%s acq=%s src=%s age=%.0fs", e_horiz,
                                       self._appr_radius, alt_now, self._follow_alt,
                                       self._desired_alt, warm, acquired, self._src, self._age())
            self._guard_track_lost(now)
            if self._modeA_fix:
                self._divergence_guard(now, e_horiz)

        elif state == S.FOLLOW:
            if self._follow_t0 is None:
                self._follow_t0 = now          # anchor the test-hook blackout window
            self._guard_track_lost(now)
            if self._modeA_fix:
                self._divergence_guard(now, e_horiz)
            # Handover descent: lower the setpoint from follow_alt → follow_floor so
            # the UAV crosses into UWB range (<5 m) and the horizontal law hands over.
            # P1 (NUANCED): on perception loss, CLIMB to regain — widening the FOV is load-bearing for
            # re-acquisition (the gimbal sweep alone, at fixed altitude, can't re-find a drifted USV).
            # But the climb is gated by `not acquired`, which already implies `not uwb_fresh` — so it
            # can NEVER fire while UWB freshly owns the body (honors "UWB flies, gimbal/climb only when
            # vision-only & lost").  The gimbal directed search runs in parallel.
            if not acquired:
                if self._regain_since is None:
                    self._regain_since = now
                if self._loss_recovery_climb and (now - self._regain_since) > self._regain_to:
                    self._desired_alt = min(self._follow_alt,
                                            self._desired_alt + self._regain_climb / self._sm_hz)
                    rospy.logwarn_throttle(1.0, "[uwb_follow] perception LOST in FOLLOW (no UWB) — "
                                           "CLIMB to regain (alt→%.1f)", self._desired_alt)
            else:
                self._regain_since = None
                if self._follow_descent and self._age() > self._follow_descent_settle \
                        and self._desired_alt > self._follow_floor_alt + 0.01:
                    # binary cone gate (run-11): sink at a fixed rate only while the USV is
                    # inside the AR detection cone; else hold and re-centre.
                    cone_r = max(self._descend_cone_min, self._descend_cone * max(alt_now, 0.0))
                    if e_horiz < cone_r:
                        step = self._follow_descent_rate / self._sm_hz
                        self._desired_alt = max(self._follow_floor_alt, self._desired_alt - step)
                    else:
                        rospy.loginfo_throttle(2.0, "[uwb_follow] descent HOLD: |e|=%.2f > cone "
                                               "r=%.2f @ alt=%.1f (re-centring)", e_horiz, cone_r, alt_now)
            # Begin landing only once the precise UWB owns horizontal (in handover
            # mode); in pure-UWB mode uwb_active is already the only source.
            ready = uwb_active if self._handover else True
            if self._land_enable and ready and self._age() > self._follow_hold_s \
                    and e_horiz < self._align_radius:
                rospy.loginfo("[uwb_follow] FOLLOW stable on UWB (|e|=%.2f) — ALIGN (→ %.1f m)",
                              e_horiz, self._align_alt)
                self._desired_alt = self._align_alt
                self._down_rate = self._align_rate
                self._land_ok_since = None
                self._transition(S.ALIGN)
                return
            rospy.loginfo_throttle(2.0, "[uwb_follow] FOLLOW |e_xy|=%.2f m  alt=%.1f→%.1f m "
                                   "src=%s", e_horiz, self._target_alt, self._desired_alt,
                                   'UWB' if uwb_active else 'VIS')

        elif state == S.ALIGN:
            if self._commit_mode == 'strict' or r_sm is None:
                # legacy gate: centred AND slow AND at-altitude, held continuously
                at_alt = abs(alt_now - self._align_alt) < 0.5
                gate_ok = (e_horiz < self._commit_radius and at_alt
                           and ev_mag < self._commit_vmax)
                why = "|e|=%.2f v=%.2f alt=%.1f" % (e_horiz, ev_mag, alt_now)
            else:
                # PREDICTIVE gate: where does the UAV land relative to the deck if it
                # coasts (velocity-matched) through the blind descent? Commit if that
                # predicted point is inside the deck safe radius.  One-sided altitude
                # (commit from anywhere below align_alt+band → descent stall can't block).
                t_desc = max(0.4, (alt_now - self._touchdown_alt) / max(0.2, self._commit_rate))
                pred_miss = float(np.linalg.norm(r_sm + rdot_gate * t_desc))  # EMA ṙ (fix #1)
                low_enough = alt_now < self._align_alt + self._commit_alt_band
                gate_ok = (pred_miss < self._commit_pred_r and low_enough
                           and ev_mag < self._commit_vmax_hard)
                why = "pred=%.2f<%.2f |e|=%.2f v=%.2f alt=%.1f" % (
                    pred_miss, self._commit_pred_r, e_horiz, ev_mag, alt_now)
            if gate_ok:
                if self._land_ok_since is None:
                    self._land_ok_since = now
                elif now - self._land_ok_since >= self._commit_hold_s:
                    rospy.loginfo("[uwb_follow] COMMIT gate met (%s)", why)
                    self._desired_alt = -1.0
                    self._down_rate = self._commit_rate
                    self._transition(S.COMMIT)
                    return
            else:
                self._land_ok_since = None
            # Abandon ALIGN only on a SUSTAINED UWB loss (uwb_active already debounces brief in_range
            # flicker via the ho_drop_s coast) — NOT on a momentary in_range blip or a vision e_horiz
            # spike from AR band-churn.  UWB owns horizontal here, so vision is not authoritative; this
            # kills the FOLLOW↔ALIGN border limit-cycle.  No upward _desired_alt reset (don't re-climb).
            if not uwb_active:
                rospy.logwarn("[uwb_follow] ALIGN lost UWB (sustained) — back to FOLLOW")
                self._down_rate = self._alt_down
                self._transition(S.FOLLOW)
                return
            rospy.loginfo_throttle(1.0, "[uwb_follow] ALIGN %s  alt=%.1f→%.1f m",
                                   why, alt_now, self._align_alt)

        elif state == S.COMMIT:
            if alt_now >= 0.0 and alt_now < self._touchdown_alt:
                rospy.loginfo("[uwb_follow] TOUCHDOWN at %.2f m — disarm", alt_now)
                self._transition(S.TOUCHDOWN)
                return
            if alt_now > self._noreturn_alt and \
                    (e_horiz > self._abort_radius or
                     (last_ekf_t is not None and now - last_ekf_t > self._commit_lost_to)):
                rospy.logwarn("[uwb_follow] COMMIT abort above no-return (|e|=%.2f) — climb", e_horiz)
                self._desired_alt = self._follow_alt
                self._down_rate = self._alt_down
                self._transition(S.FOLLOW)
                return
            rospy.loginfo_throttle(0.5, "[uwb_follow] COMMIT alt=%.2f m  |e|=%.2f m", alt_now, e_horiz)

        elif state == S.TOUCHDOWN:
            self._arm(False)
            self._transition(S.LANDED)

        elif state == S.LANDED:
            self._arm(False)
            rospy.loginfo_once("[uwb_follow] *** LANDED ***")

        elif state == S.ABORT:
            rospy.logerr_once("[uwb_follow] *** ABORT — manual recovery ***")

        # Phase-1 visibility-constrained descent: override the FOLLOW/ALIGN descent altitude with the
        # visibility-gated setpoint from usv_visibility_traj_node (descend only as the USV offset
        # closes → stays inside the fixed-nadir cone).  COMMIT→touchdown stays the committed descent.
        # SINGLE-controller architecture: the visibility trajectory owns the altitude across ALL
        # active descent phases (Phase-1 limited this to FOLLOW/ALIGN) — the discrete per-state
        # altitude logic is subsumed by one continuous cone-gated setpoint.
        vt_states = (S.APPROACH, S.FOLLOW, S.ALIGN) if self._vt_in_approach else (S.FOLLOW, S.ALIGN)
        if self._vis_traj and state in vt_states \
                and self._vt_alt is not None and (now - self._vt_alt_t) < 0.5:
            tgt = self._vt_alt
            # 0.C hand-off SLOW-CROSS: while descending through the 6-7 m band before the inner
            # marker is solid, cross slowly (don't drop fast through the outer→inner gap) so the
            # UKF bridge + blob fallback carry the estimate and inner can acquire near the bottom.
            # NB: a hard HOLD would deadlock — inner only decodes ≤~5 m, i.e. below the band — so
            # we throttle the sink rate rather than stop it.
            in_band = (self._handoff_band_top - self._handoff_band) < self._desired_alt \
                <= self._handoff_band_top
            if self._inner_streak < self._handoff_inner_n and in_band and tgt < self._desired_alt:
                tgt = max(tgt, self._desired_alt - self._handoff_cross_rate / self._sm_hz)
            # 0.C descent ratchet: descents pass straight through, but a CLIMB (cone re-lifting on
            # an offset spike) is rate-limited AND capped to floor+cap — a transient low-speed
            # offset spike can lift the UAV at most vt_climb_cap above the lowest altitude reached.
            if tgt > self._desired_alt:
                tgt = min(tgt, self._desired_alt + self._vt_climb_rate / self._sm_hz,
                          self._vt_floor + self._vt_climb_cap)
            self._desired_alt = max(tgt, self._touchdown_alt)
            self._vt_floor = min(self._vt_floor, self._desired_alt)

    def _guard_track_lost(self, now):
        """Fall back to SEARCH only when BOTH sources are lost.  In handover mode the
        far field has UWB out of range (normal) while vision tracks; near field is the
        reverse — so abort only if neither has a fresh fix."""
        with self._lock:
            last_ekf_t, in_range = self._last_ekf_t, self._in_range
            last_vt, last_vs = self._last_vis_track_t, self._last_vis_seen_t
        uwb_ok = in_range and last_ekf_t is not None and (now - last_ekf_t) < self._track_lost_to
        vis_ok = (self._handover and last_vs is not None and (now - last_vs) < self._track_lost_to
                  and last_vt is not None and (now - last_vt) < self._track_lost_to)
        if self._inj_blackout:                 # test hook: force "both lost" during the blackout
            uwb_ok = vis_ok = False
        # Only ENTER search on a fresh loss.  Re-calling _transition(S.SEARCH) every tick while
        # already searching would keep resetting _state_t, so the SEARCH `age` never grows and the
        # search_abort_s timeout can never fire → the UAV hangs in SEARCH forever (until the
        # harness timeout) instead of ABORTing cleanly.  Don't re-trigger from terminal states.
        if not uwb_ok and not vis_ok and self._state not in (S.SEARCH, S.ABORT, S.LANDED, S.REBOOT):
            rospy.logwarn("[uwb_follow] both sources lost (uwb=%s vis=%s) → SEARCH",
                          uwb_ok, vis_ok)
            self._appr_ok_since = None
            self._transition(S.SEARCH)

    def _blackout_active(self, now):
        """Test hook: True while the injected perception blackout is in effect (a window after
        first reaching FOLLOW).  Used to deterministically reproduce the Mode-A far-field loss."""
        if self._inj_drop_at < 0 or self._follow_t0 is None:
            return False
        tf = now - self._follow_t0
        return self._inj_drop_at <= tf < (self._inj_drop_at + self._inj_drop_dur)

    def _divergence_guard(self, now, e_horiz):
        """#3 (Mode-A): bound the chase.  If the estimated horizontal offset blows past
        divergence_max for divergence_abort_s, the far-field acquisition has collapsed (the UAV
        is chasing a runaway estimate) — ABORT cleanly rather than fly off.  Resets when the
        offset is back inside the threshold."""
        if e_horiz is not None and e_horiz > self._div_max:
            if self._div_since is None:
                self._div_since = now
                rospy.logwarn("[uwb_follow] DIVERGENCE |e|=%.1f > %.1f m — guarding",
                              e_horiz, self._div_max)
            elif (now - self._div_since) > self._div_abort_s:
                rospy.logerr("[uwb_follow] divergence persisted %.0fs (|e|=%.1f m) → ABORT",
                             self._div_abort_s, e_horiz)
                self._transition(S.ABORT)
        else:
            self._div_since = None

    def run(self):
        self._pub_state.publish(String(data=self._state))
        rospy.spin()


if __name__ == '__main__':
    try:
        UwbFollowController().run()
    except rospy.ROSInterruptException:
        pass
