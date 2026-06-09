#!/usr/bin/env python3
"""
traj_landing_sm_node.py
────────────────────────
GPS-free state machine for autonomous USV deck landing using trajectory-based
attitude control (traj_landing_controller_node).

Mirrors ar_landing_sm_node.py exactly in state-machine logic so the two
approaches are directly comparable.  Only the controller topic namespace
differs: /ar_landing/landing_ctrl/ → /traj_landing/ctrl/

State flow  (identical to ar_code_landing)
──────────────────────────────────────────
  PRE_ARMED          Stream hover setpoints; wait for sensors (3 s)
  ARMING             Request OFFBOARD + arm; retry until confirmed
  CLIMBING           Climb to climb_alt (10 m) via trajectory controller
  GIMBAL_NADIR       Slew gimbal to nadir (π/2)
  ARUCO_SEARCH       Wait for gimbal_aruco_tracker → TRACKING
  TRAJECTORY_ALIGN   Trajectory to hold_alt above USV deck;
                     exit: |e_xy| < threshold for settle_s
  TRAJECTORY_DESCEND Gradually lower target_alt; trajectory controller
                     continuously replans to descending waypoint;
                     exit: garmin < touch_range → AUTO.LAND
  LANDED             Wait for PX4 auto-disarm
  HOVER              ArUco-lost recovery; level hold at hold_alt
  ABORT              Fatal; manual recovery required

Services called (pure MAVROS)
──────────────────────────────
  /<ns>/mavros/cmd/arming   mavros_msgs/CommandBool
  /<ns>/mavros/set_mode     mavros_msgs/SetMode
  /<ns>/mavros/param/set    mavros_msgs/ParamSet

Subscriptions
─────────────
  /<ns>/mavros/state                         mavros_msgs/State
  /<ns>/garmin/range                         sensor_msgs/Range
  /<ns>/gimbal/joint_states                  sensor_msgs/JointState
  /<ns>/ground_truth                         nav_msgs/Odometry
  /<ns>/mavros/imu/data                      sensor_msgs/Imu
  /ar_landing/gimbal_tracker/tracking_status std_msgs/String
  /traj_landing/ctrl/tracking_error          geometry_msgs/Vector3Stamped

Publications
────────────
  ~state                               std_msgs/String  (latched)
  /traj_landing/ctrl/enable            std_msgs/Bool    (latched)
  /traj_landing/ctrl/target_altitude   std_msgs/Float64
  /<ns>/gimbal/position/pitch/command  std_msgs/Float64
  /<ns>/gimbal/position/yaw/command    std_msgs/Float64
  /<ns>/mavros/setpoint_raw/attitude   mavros_msgs/AttitudeTarget
    (PRE_ARMED, ARMING, and first 2 s of CLIMBING only — then controller takes over)
"""

import math
import threading

import numpy as np
import rospy

from std_msgs.msg      import Float64, String, Bool
from sensor_msgs.msg   import Range, JointState, Imu
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import Vector3Stamped, Quaternion

from mavros_msgs.msg   import State as MavState, AttitudeTarget, ParamValue
from mavros_msgs.srv   import (CommandBool,  CommandBoolRequest,
                               SetMode,      SetModeRequest,
                               ParamSet,     ParamSetRequest)

from scipy.spatial.transform import Rotation as Rot


# ─────────────────────────────────────────────────────────────────────────────
# States
# ─────────────────────────────────────────────────────────────────────────────

class State:
    PRE_ARMED           = 'PRE_ARMED'
    ARMING              = 'ARMING'
    CLIMBING            = 'CLIMBING'
    GIMBAL_NADIR        = 'GIMBAL_NADIR'
    ARUCO_SEARCH        = 'ARUCO_SEARCH'
    TRAJECTORY_ALIGN    = 'TRAJECTORY_ALIGN'
    TRAJECTORY_DESCEND  = 'TRAJECTORY_DESCEND'
    LANDED              = 'LANDED'
    HOVER               = 'HOVER'
    ABORT               = 'ABORT'


# ─────────────────────────────────────────────────────────────────────────────
# State machine
# ─────────────────────────────────────────────────────────────────────────────

class TrajLandingStateMachine:

    def __init__(self):
        rospy.init_node('traj_landing_sm_node', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')
        self._ns = ns

        # ── Arming / hover parameters ─────────────────────────────────────
        self._hover_thrust = rospy.get_param('~hover_thrust', 0.58)

        # ── Mission parameters ────────────────────────────────────────────
        self._climb_alt   = rospy.get_param('~climb_alt_m',          10.0)
        self._hold_alt    = rospy.get_param('~hold_alt_m',            5.0)
        self._align_thr   = rospy.get_param('~align_threshold_m',     0.6)
        self._align_set   = rospy.get_param('~align_settle_s',        3.0)
        self._des_rate    = rospy.get_param('~descent_rate_ms',       0.3)
        self._touch_rng   = rospy.get_param('~touch_range_m',         0.8)
        self._desc_lost   = rospy.get_param('~descend_lost_timeout_s', 4.0)
        self._alt_down_rate = rospy.get_param('~alt_down_rate_ms',    1.0)

        # Dynamic descent rate parameters (close approach)
        self._close_des_alt       = rospy.get_param('~close_descent_alt_m',  2.5)
        self._desc_err_scale      = rospy.get_param('~desc_err_scale_m',     0.5)
        self._desc_min_pos_factor = rospy.get_param('~desc_min_pos_factor',  0.05)
        self._desc_min_alt_factor = rospy.get_param('~desc_min_alt_factor',  0.5)
        self._rate_hz             = rospy.get_param('~rate_hz',              10.0)

        # Timeouts [s]
        self._t_prearm  = rospy.get_param('~prearm_timeout_s',   10.0)
        self._t_arming  = rospy.get_param('~arming_timeout_s',   30.0)
        self._t_climb   = rospy.get_param('~climb_timeout_s',    60.0)
        self._t_nadir   = rospy.get_param('~nadir_timeout_s',    15.0)
        self._t_search  = rospy.get_param('~search_timeout_s',   60.0)
        self._t_align   = rospy.get_param('~align_timeout_s',   120.0)

        # ── Runtime state ─────────────────────────────────────────────────
        self._lock     = threading.Lock()
        self._state    = State.PRE_ARMED
        self._state_t  = rospy.Time.now().to_sec()

        self._altitude     = 0.0
        self._garmin       = None
        self._gimbal_pitch = 0.0
        self._tracking     = 'LOST'
        self._e_horiz      = 999.0
        self._mav_armed    = False
        self._mav_mode     = ''

        self._imu_yaw   = 0.0
        self._imu_ready = False

        self._align_ok_since  = None
        self._desired_alt     = 0.5
        self._target_alt      = 0.5
        self._last_desc_t     = None
        self._last_tracking_t = None
        self._disarmed        = False
        self._prev_armed      = False
        self._params_set      = False

        # Service proxies (lazy-connected)
        self._arm_srv   = None
        self._mode_srv  = None
        self._param_srv = None

        # ── Publishers ────────────────────────────────────────────────────
        self._pub_att    = rospy.Publisher(
            f'/{ns}/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=5)
        self._pub_state  = rospy.Publisher(
            '~state', String, queue_size=5, latch=True)
        self._pub_tgtalt = rospy.Publisher(
            '/traj_landing/ctrl/target_altitude', Float64, queue_size=1)
        self._pub_enable = rospy.Publisher(
            '/traj_landing/ctrl/enable', Bool, queue_size=1, latch=True)
        self._pub_pitch  = rospy.Publisher(
            f'/{ns}/gimbal/position/pitch/command', Float64, queue_size=1)
        self._pub_yaw    = rospy.Publisher(
            f'/{ns}/gimbal/position/yaw/command', Float64, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/mavros/imu/data',
                         Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/state',
                         MavState, self._mav_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/ground_truth',
                         Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber(f'/{ns}/garmin/range',
                         Range, self._range_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',
                         JointState, self._js_cb, queue_size=5)
        rospy.Subscriber('/ar_landing/gimbal_tracker/tracking_status',
                         String, self._track_cb, queue_size=5)
        rospy.Subscriber('/traj_landing/ctrl/tracking_error',
                         Vector3Stamped, self._herr_cb, queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._spin)

        rospy.loginfo(
            f'[traj_sm] init  ns={ns}  '
            f'climb={self._climb_alt} m  hold={self._hold_alt} m  '
            f'des_rate={self._des_rate} m/s  touch={self._touch_rng} m')

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _mav_cb(self, msg: MavState):
        with self._lock:
            self._mav_armed = bool(msg.armed)
            self._mav_mode  = str(msg.mode)

    def _odom_cb(self, msg: Odometry):
        with self._lock:
            self._altitude = msg.pose.pose.position.z

    def _range_cb(self, msg: Range):
        with self._lock:
            self._garmin = float(msg.range)

    def _js_cb(self, msg: JointState):
        ns  = self._ns
        lut = dict(zip(msg.name, msg.position))
        with self._lock:
            self._gimbal_pitch = lut.get(f'{ns}_gimbal_pitch_joint', 0.0)

    def _track_cb(self, msg: String):
        status = msg.data
        with self._lock:
            self._tracking = status
            if status == 'TRACKING':
                self._last_tracking_t = rospy.Time.now().to_sec()

    def _herr_cb(self, msg: Vector3Stamped):
        with self._lock:
            self._e_horiz = float(msg.vector.z)   # |e_xy| stored in z (convention)

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        yaw = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._imu_yaw   = yaw
            self._imu_ready = True

    # ── MAVROS service helpers ────────────────────────────────────────────────

    def _call_arm(self, arm: bool) -> bool:
        srv = f'/{self._ns}/mavros/cmd/arming'
        if self._arm_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._arm_srv = rospy.ServiceProxy(srv, CommandBool)
            except rospy.ROSException:
                rospy.logerr(f'[traj_sm] arm service {srv} unavailable')
                return False
        try:
            return bool(self._arm_srv(CommandBoolRequest(value=arm)).success)
        except rospy.ServiceException as exc:
            rospy.logerr(f'[traj_sm] arm call failed: {exc}')
            return False

    def _call_set_mode(self, mode: str) -> bool:
        srv = f'/{self._ns}/mavros/set_mode'
        if self._mode_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._mode_srv = rospy.ServiceProxy(srv, SetMode)
            except rospy.ROSException:
                rospy.logerr(f'[traj_sm] set_mode service {srv} unavailable')
                return False
        try:
            return bool(self._mode_srv(
                SetModeRequest(custom_mode=mode)).mode_sent)
        except rospy.ServiceException as exc:
            rospy.logerr(f'[traj_sm] set_mode {mode} failed: {exc}')
            return False

    def _set_px4_param(self, name: str, val, integer: bool = False) -> bool:
        srv = f'/{self._ns}/mavros/param/set'
        if self._param_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._param_srv = rospy.ServiceProxy(srv, ParamSet)
            except rospy.ROSException:
                rospy.logwarn(f'[traj_sm] param/set service {srv} unavailable')
                return False
        try:
            req = ParamSetRequest()
            req.param_id = name
            req.value = (ParamValue(integer=int(val),   real=0.0)
                         if integer else
                         ParamValue(integer=0, real=float(val)))
            resp = self._param_srv(req)
            rospy.loginfo(f'[traj_sm] param {name}={val} → '
                          f'{"OK" if resp.success else "FAIL"}')
            return bool(resp.success)
        except rospy.ServiceException as exc:
            rospy.logwarn(f'[traj_sm] param/set {name} failed: {exc}')
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_hover_att(self):
        """Level hover AttitudeTarget at current IMU yaw.
        Published by SM during PRE_ARMED/ARMING so the same process that calls
        set_mode/arm is streaming setpoints — PX4 requirement for OFFBOARD."""
        with self._lock:
            yaw = self._imu_yaw
        q = Rot.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE  |
                         AttitudeTarget.IGNORE_PITCH_RATE |
                         AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]),
            z=float(q[2]), w=float(q[3]))
        att.thrust = self._hover_thrust
        self._pub_att.publish(att)

    def _transition(self, new_state: str):
        rospy.loginfo(f'[traj_sm] {self._state} → {new_state}')
        self._state   = new_state
        self._state_t = rospy.Time.now().to_sec()
        self._pub_state.publish(String(data=new_state))

    def _age(self) -> float:
        return rospy.Time.now().to_sec() - self._state_t

    def _enable_ctrl(self, on: bool):
        self._pub_enable.publish(Bool(data=on))

    def _set_tgt(self, alt: float):
        """Request a target altitude; actual commanded value tracks toward it
        at _alt_down_rate m/s downward (upward: immediate)."""
        self._desired_alt = float(alt)

    def _tick_alt_ramp(self):
        """Advance _target_alt toward _desired_alt and publish if changed.
        Same rate-limiting logic as ar_landing_sm_node."""
        desired = self._desired_alt
        target  = self._target_alt
        if abs(desired - target) < 0.005:
            return
        dt = 1.0 / self._rate_hz
        if desired < target:
            step = min(abs(desired - target), self._alt_down_rate * dt)
        else:
            step = abs(desired - target)
        new_target = target + math.copysign(step, desired - target)
        self._target_alt = new_target
        self._pub_tgtalt.publish(Float64(data=new_target))

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _spin(self, _event):
        with self._lock:
            state    = self._state
            alt      = self._altitude
            garmin   = self._garmin
            gpitch   = self._gimbal_pitch
            tracking = self._tracking
            e_horiz  = self._e_horiz
            armed    = self._mav_armed
            mode     = self._mav_mode
            last_trk = self._last_tracking_t

        now = rospy.Time.now().to_sec()
        self._tick_alt_ramp()

        # ── PRE_ARMED ─────────────────────────────────────────────────────
        if state == State.PRE_ARMED:
            self._publish_hover_att()

            if self._age() < 0.2:
                self._pub_yaw.publish(Float64(data=0.0))
                self._enable_ctrl(False)
                rospy.loginfo('[traj_sm] PRE_ARMED: setpoints flowing …')

            if not self._params_set and self._age() >= 0.5:
                ok1 = self._set_px4_param('COM_DISARM_PRFLT', -1,  integer=True)
                ok2 = self._set_px4_param('NAV_RCL_ACT',       0,  integer=True)
                ok3 = self._set_px4_param('COM_RC_IN_MODE',    1,  integer=True)
                ok4 = self._set_px4_param('COM_OBL_ACT',       1,  integer=True)
                if ok1 and ok2 and ok3 and ok4:
                    self._params_set = True
                    rospy.loginfo('[traj_sm] PX4 safety params set')

            if self._age() >= self._t_prearm:
                rospy.logerr('[traj_sm] PRE_ARMED timeout — ABORT')
                self._transition(State.ABORT)
            elif garmin is not None and self._age() >= 3.0:
                self._transition(State.ARMING)

        # ── ARMING ────────────────────────────────────────────────────────
        elif state == State.ARMING:
            self._publish_hover_att()

            disarm_edge   = self._prev_armed and not armed
            self._prev_armed = armed

            if armed and mode == 'OFFBOARD':
                rospy.loginfo('[traj_sm] Armed in OFFBOARD — begin climb')
                self._set_tgt(self._climb_alt)
                self._enable_ctrl(True)
                self._transition(State.CLIMBING)
            else:
                periodic = (int(self._age() * self._rate_hz)
                            % int(2.0 * self._rate_hz) == 0)
                if periodic or disarm_edge:
                    if disarm_edge:
                        rospy.logwarn('[traj_sm] Unexpected disarm — retrying')
                    if mode != 'OFFBOARD':
                        ok = self._call_set_mode('OFFBOARD')
                        rospy.loginfo(
                            f'[traj_sm] set_mode OFFBOARD → '
                            f'{"OK" if ok else "FAIL"} (mode={mode!r})')
                    if not armed:
                        ok = self._call_arm(True)
                        rospy.loginfo(
                            f'[traj_sm] arm → {"OK" if ok else "FAIL"} '
                            f'(mode={mode!r} armed={armed})')
                    # Poll at 100 Hz for 200 ms to catch the brief armed+OFFBOARD window
                    t_poll = rospy.Time.now().to_sec() + 0.2
                    while (rospy.Time.now().to_sec() < t_poll and
                           not rospy.is_shutdown()):
                        self._publish_hover_att()
                        rospy.sleep(0.01)
                        with self._lock:
                            armed = self._mav_armed
                            mode  = self._mav_mode
                        if armed and mode == 'OFFBOARD':
                            break
                if self._age() > self._t_arming:
                    rospy.logerr(
                        f'[traj_sm] Arming timeout ({self._age():.0f} s) — ABORT')
                    self._transition(State.ABORT)

        # ── CLIMBING ──────────────────────────────────────────────────────
        elif state == State.CLIMBING:
            # Keep SM setpoints flowing for the first 2 s so the controller
            # has time to receive the 'enable' message and begin publishing.
            if self._age() < 2.0:
                self._publish_hover_att()

            if garmin is not None and garmin >= self._climb_alt - 0.5:
                rospy.loginfo(
                    f'[traj_sm] Climb complete (garmin={garmin:.1f} m)')
                self._transition(State.GIMBAL_NADIR)
            elif self._age() > self._t_climb:
                rospy.logerr('[traj_sm] Climb timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                rospy.loginfo_throttle(
                    3.0, f'[traj_sm] CLIMBING  garmin={garmin:.1f} m  '
                         f'target={self._climb_alt:.1f} m')

        # ── GIMBAL_NADIR ──────────────────────────────────────────────────
        elif state == State.GIMBAL_NADIR:
            self._publish_hover_att()
            self._pub_pitch.publish(Float64(data=math.pi / 2.0))

            if gpitch >= 1.2:
                rospy.loginfo(
                    f'[traj_sm] Gimbal nadir ({math.degrees(gpitch):.1f}°)')
                self._transition(State.ARUCO_SEARCH)
            elif self._age() > self._t_nadir:
                rospy.logerr('[traj_sm] Gimbal nadir timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)

        # ── ARUCO_SEARCH ──────────────────────────────────────────────────
        elif state == State.ARUCO_SEARCH:
            self._publish_hover_att()

            if tracking == 'TRACKING':
                rospy.loginfo('[traj_sm] Deck TRACKING — begin TRAJECTORY_ALIGN')
                self._align_ok_since = None
                self._set_tgt(self._hold_alt)
                self._transition(State.TRAJECTORY_ALIGN)
            elif self._age() > self._t_search:
                rospy.logerr(
                    f'[traj_sm] ArUco search timeout ({self._age():.0f} s) — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                rospy.loginfo_throttle(
                    5.0, f'[traj_sm] SEARCHING '
                         f'({self._age():.0f}/{self._t_search:.0f} s)  '
                         f'tracking={tracking}')

        # ── TRAJECTORY_ALIGN ──────────────────────────────────────────────
        elif state == State.TRAJECTORY_ALIGN:
            # Trajectory controller drives UAV to hold_alt above USV.
            # SM only monitors convergence; trajectory replanning is automatic.
            if e_horiz < self._align_thr:
                if self._align_ok_since is None:
                    self._align_ok_since = now
                elif now - self._align_ok_since >= self._align_set:
                    rospy.loginfo(
                        f'[traj_sm] Aligned  |e_xy|={e_horiz:.2f} m '
                        f'< {self._align_thr:.2f} m  for {self._align_set:.1f} s')
                    self._last_desc_t = now
                    self._transition(State.TRAJECTORY_DESCEND)
                    return
            else:
                self._align_ok_since = None

            if self._age() > self._t_align:
                rospy.logerr('[traj_sm] TRAJECTORY_ALIGN timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                settled = (f'{now - self._align_ok_since:.1f} s'
                           if self._align_ok_since else 'no')
                rospy.loginfo_throttle(
                    2.0, f'[traj_sm] TRAJECTORY_ALIGN  |e_xy|={e_horiz:.2f} m  '
                         f'settled={settled}  tracking={tracking}')

        # ── TRAJECTORY_DESCEND ────────────────────────────────────────────
        elif state == State.TRAJECTORY_DESCEND:
            # Decrease target_alt over time; trajectory controller continuously
            # replans a short trajectory to the new (lower) waypoint above USV.
            # Same adaptive rate logic as ar_code_landing for close approach.
            dt_d = now - self._last_desc_t if self._last_desc_t else 0.0

            if garmin is not None and garmin < self._close_des_alt:
                if tracking == 'TRACKING':
                    pos_factor = float(np.clip(
                        1.0 - e_horiz / self._desc_err_scale,
                        self._desc_min_pos_factor, 1.0))
                else:
                    pos_factor = self._desc_min_pos_factor
                alt_factor = float(np.clip(
                    garmin / self._close_des_alt,
                    self._desc_min_alt_factor, 1.0))
                rate = self._des_rate * pos_factor * alt_factor
                rospy.loginfo_throttle(
                    2.0, f'[traj_sm] DESCEND rate={rate:.3f} m/s '
                         f'pos={pos_factor:.2f} alt={alt_factor:.2f} '
                         f'|e_xy|={e_horiz:.2f} m  garmin={garmin:.2f} m')
            else:
                rate = self._des_rate

            new_alt = max(-1.0, self._desired_alt - rate * dt_d)
            with self._lock:
                self._last_desc_t = now
            self._desired_alt = new_alt
            self._target_alt  = new_alt
            self._pub_tgtalt.publish(Float64(data=new_alt))

            # Touch-down: switch to AUTO.LAND before disabling controller
            if garmin is not None and garmin < self._touch_rng:
                rospy.loginfo(
                    f'[traj_sm] Touch-down: garmin={garmin:.2f} m '
                    f'< {self._touch_rng:.2f} m')
                ok = self._call_set_mode('AUTO.LAND')
                rospy.loginfo(
                    f'[traj_sm] AUTO.LAND → {"OK" if ok else "FAIL"}')
                self._enable_ctrl(False)
                self._transition(State.LANDED)
                return

            # ArUco lost guard
            if tracking not in ('TRACKING', 'SEARCHING'):
                lost_dur = (now - last_trk) if last_trk else self._age()
                if lost_dur > self._desc_lost:
                    rospy.logwarn(
                        f'[traj_sm] ArUco lost {lost_dur:.1f} s in DESCEND — HOVER')
                    self._set_tgt(self._hold_alt)
                    self._transition(State.HOVER)
                    return

            rospy.loginfo_throttle(
                2.0, f'[traj_sm] TRAJECTORY_DESCEND  '
                     f'tgt={new_alt:.2f} m  garmin={garmin:.2f} m  '
                     f'|e_xy|={e_horiz:.2f} m  track={tracking}')

        # ── LANDED ────────────────────────────────────────────────────────
        elif state == State.LANDED:
            if not self._disarmed:
                if not armed:
                    self._disarmed = True
                    rospy.loginfo('[traj_sm] *** DISARMED — motors off ***')
                else:
                    rospy.loginfo_throttle(
                        3.0, f'[traj_sm] LANDED waiting for PX4 disarm '
                             f'(age={self._age():.1f} s  mode={mode})')
            else:
                rospy.loginfo_once('[traj_sm] *** LANDED ***')

        # ── HOVER (ArUco-lost recovery) ───────────────────────────────────
        elif state == State.HOVER:
            rospy.logwarn_throttle(
                5.0, f'[traj_sm] HOVER — waiting for ArUco  '
                     f'garmin={garmin:.2f} m  tracking={tracking}')
            if tracking == 'TRACKING':
                rospy.loginfo('[traj_sm] ArUco reacquired — re-enter TRAJECTORY_ALIGN')
                self._align_ok_since = None
                self._set_tgt(self._hold_alt)
                self._transition(State.TRAJECTORY_ALIGN)

        # ── ABORT ─────────────────────────────────────────────────────────
        elif state == State.ABORT:
            self._enable_ctrl(False)
            rospy.logerr_once(
                '[traj_sm] *** ABORTED — disarm manually if needed ***')

    def run(self):
        self._pub_state.publish(String(data=self._state))
        rospy.spin()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        TrajLandingStateMachine().run()
    except rospy.ROSInterruptException:
        pass
