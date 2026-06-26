#!/usr/bin/env python3
"""
usv_gimbal_ff_node.py  (tmp / dev-only)
────────────────────────────────────────
P2 — model-based FEED-FORWARD gimbal pointing (GPS-free).

Replaces the reactive pixel-IBVS gimbal (usv_gimbal_ibvs_node.py) with an
open-loop pointing law: each tick it computes the gimbal joint angles that put
the camera optical axis on the USV, from
  • the USV relative position estimate  (/usv_track/filtered — already the
    best-performing source via the unified selector; optionally the Bézier
    lookahead for a predictive lead), and
  • the UAV attitude from the IMU  (so when the UAV tilts, the gimbal compensates
    immediately rather than waiting for the target pixel to move).

Why: the dominant follow error is a CROSS-TRACK limit cycle driven by the gimbal
lag in the reactive IBVS loop (the gimbal chases the noisy detection pixel, which
jitters as the UAV moves).  Pointing the gimbal straight at the smooth filtered /
predicted USV position — with analytic attitude compensation — removes that lag.

Pointing law (inverse of the gimbal FK  R_opt = R_wb·Rz(yaw)·Rx(roll)·Ry(pitch)·R_OPT,
optical forward = R_opt·ẑ):
  d_w = normalise([dx, dy, -H])            # look direction, level (world-aligned) frame
  d_b = R_wb⁻¹ · d_w                       # in the UAV body frame
  pitch = asin(-d_b_z)   (π/2 = nadir)     # since d_b = [cosθ cosψ, cosθ sinψ, -sinθ]
  yaw   = atan2(d_b_y, d_b_x)
  roll  = 0
An optional small residual IBVS correction (off by default) can absorb model bias.

Inputs : /<ns>/usv_track/filtered (Odometry, USV offset)  [fallback /usv_relpos/estimate]
         /<ns>/usv_track/lookahead (PoseStamped)          [if ~use_lookahead]
         /<ns>/mavros/imu/data (Imu), /<ns>/garmin/range (Range)
Output : /<ns>/gimbal/position/{yaw,roll,pitch}/command

Run:  python3 tmp/usv_gimbal_ff_node.py _ns:=uav1
"""
import math
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Float64, String
from sensor_msgs.msg import Imu, Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

_HALF_PI = math.pi / 2.0


class GimbalFF:
    def __init__(self):
        rospy.init_node('usv_gimbal_ff')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns
        self._rate = float(rospy.get_param('~rate_hz', 30.0))
        self._use_look = bool(rospy.get_param('~use_lookahead', False))
        # INERTIAL setpoint mode: publish world-frame angles for the embedded
        # camera-IMU-stabilized gimbal plugin (vs body-frame joint angles for the
        # teleporter). Pitch stays in [0, π/2] (nadir=π/2); the plugin's pitch_sign
        # maps it to the joint convention (nadir=-π/2).
        self._inertial = bool(rospy.get_param('~inertial_setpoint', False))
        self._fresh_to = float(rospy.get_param('~fresh_timeout_s', 1.0))
        self._slew = float(rospy.get_param('~max_cmd_delta', 0.10))   # rad/tick limit
        self._alpha = float(rospy.get_param('~cmd_lpf', 0.5))         # command smoothing
        # Lock the gimbal at NADIR during FOLLOW (mimic a fixed-down camera): the UAV
        # is centred over the USV in FOLLOW so it stays in the nadir FOV, and this
        # removes the gimbal motion / sync / loop error to isolate its contribution.
        # APPROACH/SEARCH still track (FF) so the gimbal can acquire & centre.
        self._lock_nadir = bool(rospy.get_param('~lock_nadir_in_follow', False))
        self._fstate = ''

        self._lock = threading.Lock()
        self._R_wb = None
        self._tilt = 0.0
        self._rng = None
        self._off = None         # USV offset (dx,dy) level frame [m]
        self._look = None        # predicted offset (dx,dy)
        self._last_off_t = None
        self._cmd_yaw = 0.0
        self._cmd_pitch = _HALF_PI
        self._cmd_roll = 0.0
        self._init = False
        self._last_dw = None     # P2: last WORLD-frame look dir (for IMU-compensated hold)
        self._hold_to = float(rospy.get_param('~hold_timeout_s', 3.0))

        self._pub_yaw = rospy.Publisher('/%s/gimbal/position/yaw/command' % ns,
                                        Float64, queue_size=1)
        self._pub_pitch = rospy.Publisher('/%s/gimbal/position/pitch/command' % ns,
                                          Float64, queue_size=1)
        self._pub_roll = rospy.Publisher('/%s/gimbal/position/roll/command' % ns,
                                         Float64, queue_size=1)
        rospy.Subscriber('/%s/mavros/imu/data' % ns, Imu, self._cb_imu, queue_size=5)
        rospy.Subscriber('/%s/garmin/range' % ns, Range, self._cb_rng, queue_size=5)
        rospy.Subscriber('/%s/usv_track/filtered' % ns, Odometry, self._cb_filt, queue_size=10)
        rospy.Subscriber('/%s/usv_relpos/estimate' % ns, PoseStamped, self._cb_est, queue_size=10)
        rospy.Subscriber('/%s/usv_track/lookahead' % ns, PoseStamped, self._cb_look, queue_size=10)
        rospy.Subscriber('/usv_follow_controller/state', String, self._cb_state, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)
        rospy.loginfo("[gimbal_ff] up — ns=%s feed-forward pointing (use_lookahead=%s)",
                      ns, self._use_look)

    def _cb_state(self, m):
        with self._lock:
            self._fstate = m.data

    def _cb_imu(self, m):
        q = m.orientation
        R = Rot.from_quat([q.x, q.y, q.z, q.w])
        down_w = R.apply([0, 0, -1.0])
        tilt = math.acos(max(-1.0, min(1.0, -down_w[2])))
        with self._lock:
            self._R_wb = R
            self._tilt = tilt

    def _cb_rng(self, m):
        with self._lock:
            self._rng = float(m.range)

    def _cb_filt(self, m):
        with self._lock:
            self._off = np.array([m.pose.pose.position.x, m.pose.pose.position.y])
            self._last_off_t = rospy.Time.now().to_sec()

    def _cb_est(self, m):
        # fallback source if the UKF track is not (yet) available
        with self._lock:
            if self._off is None or self._last_off_t is None or \
                    (rospy.Time.now().to_sec() - self._last_off_t) > self._fresh_to:
                self._off = np.array([m.pose.position.x, m.pose.position.y])
                self._last_off_t = rospy.Time.now().to_sec()

    def _cb_look(self, m):
        with self._lock:
            self._look = np.array([m.pose.position.x, m.pose.position.y])

    def _tick(self, _e):
        now = rospy.Time.now().to_sec()
        with self._lock:
            R_wb, tilt, rng = self._R_wb, self._tilt, self._rng
            off = None if self._off is None else self._off.copy()
            look = None if self._look is None else self._look.copy()
            last_off_t = self._last_off_t
            cmd_yaw, cmd_pitch, cmd_roll = self._cmd_yaw, self._cmd_pitch, self._cmd_roll
            fstate = self._fstate

        # NADIR LOCK in FOLLOW: hold a fixed straight-down camera (no gimbal motion)
        if self._lock_nadir and fstate == 'FOLLOW':
            with self._lock:
                self._cmd_yaw, self._cmd_pitch, self._cmd_roll = 0.0, _HALF_PI, 0.0
            self._pub_yaw.publish(Float64(data=0.0))
            self._pub_pitch.publish(Float64(data=_HALF_PI))
            self._pub_roll.publish(Float64(data=0.0))
            return

        fresh = (off is not None and last_off_t is not None and
                 (now - last_off_t) < self._fresh_to and R_wb is not None and rng is not None)
        # P2 IMU-compensated hold: when the fix drops, keep pointing at the LAST WORLD
        # direction using the current UAV attitude (so a UAV yaw/tilt during the dropout
        # doesn't drift the gimbal off the target) — instead of freezing the body-frame
        # joint command.  Bridges brief AR losses so the target stays in the FOV for re-acquire.
        d_w = None
        if fresh:
            dxy = look if (self._use_look and look is not None) else off
            H = max(0.3, rng * math.cos(tilt))           # altitude over water
            v = np.array([dxy[0], dxy[1], -H])
            if np.linalg.norm(v) > 1e-6:
                d_w = v / np.linalg.norm(v)
                self._last_dw = d_w.copy()
        elif (self._last_dw is not None and R_wb is not None and last_off_t is not None
              and (now - last_off_t) < self._hold_to):
            d_w = self._last_dw                          # IMU-compensated hold on dropout
        if d_w is not None and R_wb is not None:
            # INERTIAL mode (embedded stabilized gimbal): output the WORLD-frame
            # pointing angles — the gimbal plugin's camera-IMU loop rejects UAV
            # attitude itself, so we must NOT pre-compensate (that's the joint-space
            # path that lagged). JOINT mode (teleporter): body-frame angles as before.
            src = d_w if self._inertial else R_wb.inv().apply(d_w)
            pitch_t = math.asin(float(np.clip(-src[2], -1.0, 1.0)))
            yaw_t = math.atan2(src[1], src[0])
            roll_t = 0.0
            if not self._init:
                cmd_yaw, cmd_pitch, cmd_roll = yaw_t, pitch_t, roll_t
                self._init = True
            else:
                # low-pass + slew-limit for smooth gimbal motion
                cmd_yaw = self._step(cmd_yaw, yaw_t)
                cmd_pitch = self._step(cmd_pitch, pitch_t)
                cmd_roll = self._step(cmd_roll, roll_t)
            cmd_pitch = float(np.clip(cmd_pitch, 0.0, _HALF_PI + 0.3))
            cmd_yaw = float(np.clip(cmd_yaw, -math.pi, math.pi))

        with self._lock:
            self._cmd_yaw, self._cmd_pitch, self._cmd_roll = cmd_yaw, cmd_pitch, cmd_roll
        # Publish only once we have EVER had a track (self._init): on a COLD start (never acquired)
        # stay silent so the thermal rotate-search can own the gimbal; once a track exists we point/
        # hold (and the thermal search yields).  Otherwise republish to hold through brief drop-outs.
        if self._init:
            self._pub_yaw.publish(Float64(data=cmd_yaw))
            self._pub_pitch.publish(Float64(data=cmd_pitch))
            self._pub_roll.publish(Float64(data=cmd_roll))

    def _step(self, cur, tgt):
        # angle-aware low-pass toward target, then slew-limit the increment
        d = math.atan2(math.sin(tgt - cur), math.cos(tgt - cur))   # wrapped error
        d = self._alpha * d
        d = float(np.clip(d, -self._slew, self._slew))
        return cur + d


if __name__ == '__main__':
    try:
        GimbalFF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
