#!/usr/bin/env python3
"""
ar_landing_sm_node.py
──────────────────────
GPS-free state machine for autonomous USV deck landing.

No GPS, no MRS goto, no global position reference of any kind.
All flight is controlled via attitude + thrust using only:
  • Garmin rangefinder (altitude above surface)
  • ArUco pose in camera frame (lateral position of deck)
  • IMU / odometry (orientation + velocity)

Startup sequence handled internally — no start_uav.launch needed.
  The SM publishes AttitudeTarget at 50 Hz before arming so that
  PX4 can enter OFFBOARD mode, then arms and begins climbing.

State flow
──────────
  PRE_ARMED     Enable controller (setpoints flow → PX4 OFFBOARD ready)
                Wait for controller + sensors to be ready (3 s)
  ARMING        Request OFFBOARD mode + arm via MAVROS services
                Wait: mavros/state.armed == True  AND  mode == "OFFBOARD"
  CLIMBING      Set target_alt = climb_alt (10 m)
                Controller climbs via Garmin PID, level flight (no ArUco yet)
                Exit: garmin_range >= climb_alt − 0.5 m
  GIMBAL_NADIR  Command gimbal pitch = +π/2.  Exit: pitch >= 1.2 rad
  ARUCO_SEARCH  Wait: gimbal_aruco_tracker status == TRACKING
  ALIGN         target_alt = hold_alt (5 m), ArUco drives roll/pitch
                Exit: |e_xy| < align_threshold for align_settle_s
  DESCEND       Decrease target_alt at descent_rate m/s
                Exit: garmin_range < touch_range (near deck surface)
  LANDED        Disable controller, disarm via MAVROS
  HOVER         Recovery: level hold at hold_alt when ArUco lost in DESCEND
                Re-enter ALIGN when TRACKING resumes
  ABORT         Fatal: disable controller (manual recovery required)

Services called (pure MAVROS — no MRS required)
────────────────────────────────────────────────
  /<ns>/mavros/cmd/arming    mavros_msgs/CommandBool
  /<ns>/mavros/set_mode      mavros_msgs/SetMode

Subscribed topics
─────────────────
  /<ns>/mavros/state             mavros_msgs/State
  /<ns>/garmin/range             sensor_msgs/Range
  /<ns>/gimbal/joint_states      sensor_msgs/JointState
  /<ns>/ground_truth             nav_msgs/Odometry   (altitude reference only)
  /ar_landing/gimbal_tracker/tracking_status  std_msgs/String
  /ar_landing/landing_ctrl/horizontal_error   geometry_msgs/Vector3Stamped

Published topics
────────────────
  ~state                   std_msgs/String  (latched)
  /ar_landing/landing_ctrl/target_altitude  std_msgs/Float64 → to landing controller
  /ar_landing/landing_ctrl/enable          std_msgs/Bool    → to landing controller
  /<ns>/gimbal/position/pitch/command  std_msgs/Float64
  /<ns>/gimbal/position/yaw/command    std_msgs/Float64
"""

import math
import threading
import numpy as np
import rospy

from std_msgs.msg import Float64, String, Bool
from sensor_msgs.msg import Range, JointState, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Quaternion

from mavros_msgs.msg import State as MavState, AttitudeTarget, ParamValue
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode,    SetModeRequest
from mavros_msgs.srv import ParamSet,   ParamSetRequest

from scipy.spatial.transform import Rotation as Rot


class State:
    PRE_ARMED    = 'PRE_ARMED'
    ARMING       = 'ARMING'
    CLIMBING     = 'CLIMBING'
    GIMBAL_NADIR = 'GIMBAL_NADIR'
    ARUCO_SEARCH = 'ARUCO_SEARCH'
    ALIGN        = 'ALIGN'
    DESCEND      = 'DESCEND'
    LANDED       = 'LANDED'
    HOVER        = 'HOVER'
    ABORT        = 'ABORT'


class ArLandingStateMachine:

    def __init__(self):
        rospy.init_node('ar_landing_sm_node', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')
        self._ns = ns

        # ── Arming / hover parameters ─────────────────────────────────────────
        self._hover_thrust = rospy.get_param('~hover_thrust', 0.58)

        # ── Mission parameters ────────────────────────────────────────────────
        self._climb_alt  = rospy.get_param('~climb_alt_m',             10.0)
        self._hold_alt   = rospy.get_param('~hold_alt_m',               5.0)
        self._align_thr  = rospy.get_param('~align_threshold_m',        0.6)
        self._align_set  = rospy.get_param('~align_settle_s',           3.0)
        self._des_rate        = rospy.get_param('~descent_rate_ms',          0.3)
        self._touch_rng       = rospy.get_param('~touch_range_m',            0.4)
        self._desc_lost       = rospy.get_param('~descend_lost_timeout_s',   4.0)
        # Rate-limit how fast the target altitude can decrease (m/s).
        # Prevents the "drop near sea" oscillation when target_alt jumps from
        # climb_alt (10 m) to hold_alt (5 m) at ALIGN entry.
        self._alt_down_rate   = rospy.get_param('~alt_down_rate_ms',          1.0)
        # Dynamic descent rate below close_des_alt:
        #   rate = des_rate × pos_factor(e_horiz) × alt_factor(garmin)
        # pos_factor: 1.0 when centred → min_pos when |e_xy| >= err_scale
        # alt_factor: 1.0 at close_des_alt → min_alt at 0 m (caution near deck)
        # → fast when centred, slow when off-centre or very close
        self._close_des_alt         = rospy.get_param('~close_descent_alt_m',       2.5)
        self._desc_err_scale        = rospy.get_param('~desc_err_scale_m',           0.5)
        self._desc_min_pos_factor   = rospy.get_param('~desc_min_pos_factor',        0.05)
        self._desc_min_alt_factor   = rospy.get_param('~desc_min_alt_factor',        0.5)
        self._rate_hz    = rospy.get_param('~rate_hz',                 10.0)

        # Timeouts [s]
        self._t_prearm   = rospy.get_param('~prearm_timeout_s',        10.0)
        self._t_arming   = rospy.get_param('~arming_timeout_s',        30.0)
        self._t_climb    = rospy.get_param('~climb_timeout_s',         60.0)
        self._t_nadir    = rospy.get_param('~nadir_timeout_s',         15.0)
        self._t_search   = rospy.get_param('~search_timeout_s',        60.0)
        self._t_align    = rospy.get_param('~align_timeout_s',        120.0)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._lock   = threading.Lock()
        self._state  = State.PRE_ARMED
        self._state_t = rospy.Time.now().to_sec()

        # Sensor values
        self._altitude      = 0.0
        self._garmin        = None
        self._gimbal_pitch  = 0.0
        self._tracking      = 'LOST'
        self._e_horiz       = 999.0
        self._mav_armed     = False
        self._mav_mode      = ''

        # IMU yaw (for hover AttitudeTarget during PRE_ARMED / ARMING)
        self._imu_yaw   = 0.0
        self._imu_ready = False

        # Phase trackers
        self._align_ok_since  = None
        self._desired_alt     = 0.5      # requested setpoint (set by states)
        self._target_alt      = 0.5      # rate-limited value sent to controller
        self._last_desc_t     = None
        self._last_tracking_t = None
        self._disarmed        = False

        # Service proxies (lazy-connected)
        self._arm_srv   = None
        self._mode_srv  = None
        self._param_srv = None

        # Track last known armed state for quick retry on unexpected disarm
        self._prev_armed = False

        # PX4 param setup: retry each tick until confirmed
        self._params_set = False

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_att     = rospy.Publisher(
            f'/{ns}/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=5)
        self._pub_state   = rospy.Publisher('~state',             String,  queue_size=5, latch=True)
        self._pub_tgtalt  = rospy.Publisher('/ar_landing/landing_ctrl/target_altitude',
                                             Float64, queue_size=1)
        self._pub_enable  = rospy.Publisher('/ar_landing/landing_ctrl/enable',
                                             Bool,    queue_size=1, latch=True)
        self._pub_pitch   = rospy.Publisher(
            f'/{ns}/gimbal/position/pitch/command', Float64, queue_size=1)
        self._pub_yaw     = rospy.Publisher(
            f'/{ns}/gimbal/position/yaw/command',   Float64, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/mavros/imu/data',       Imu,            self._imu_cb,   queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/state',         MavState,       self._mav_cb,   queue_size=5)
        rospy.Subscriber(f'/{ns}/ground_truth',         Odometry,       self._odom_cb,  queue_size=1)
        rospy.Subscriber(f'/{ns}/garmin/range',         Range,          self._range_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',  JointState,     self._js_cb,    queue_size=5)
        rospy.Subscriber('/ar_landing/gimbal_tracker/tracking_status',  String,
                         self._track_cb,  queue_size=5)
        rospy.Subscriber('/ar_landing/landing_ctrl/horizontal_error',   Vector3Stamped,
                         self._herr_cb,   queue_size=5)

        # ── Main loop ─────────────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._spin)

        rospy.loginfo(
            f'[ar_landing_sm] Init — ns={ns}  '
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
            self._e_horiz = float(msg.vector.z)   # |e_xy| stored in z component

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        yaw = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._imu_yaw   = yaw
            self._imu_ready = True

    # ── MAVROS service helpers ────────────────────────────────────────────────

    def _call_arm(self, arm: bool):
        srv = f'/{self._ns}/mavros/cmd/arming'
        if self._arm_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._arm_srv = rospy.ServiceProxy(srv, CommandBool)
            except rospy.ROSException:
                rospy.logerr(f'[ar_landing_sm] arm service {srv} unavailable')
                return False
        try:
            resp = self._arm_srv(CommandBoolRequest(value=arm))
            return bool(resp.success)
        except rospy.ServiceException as e:
            rospy.logerr(f'[ar_landing_sm] arm call failed: {e}')
            return False

    def _call_set_mode(self, mode: str):
        srv = f'/{self._ns}/mavros/set_mode'
        if self._mode_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._mode_srv = rospy.ServiceProxy(srv, SetMode)
            except rospy.ROSException:
                rospy.logerr(f'[ar_landing_sm] set_mode service {srv} unavailable')
                return False
        try:
            resp = self._mode_srv(SetModeRequest(custom_mode=mode))
            return bool(resp.mode_sent)
        except rospy.ServiceException as e:
            rospy.logerr(f'[ar_landing_sm] set_mode call failed: {e}')
            return False

    def _set_px4_param(self, name: str, val, integer: bool = False) -> bool:
        """Set a PX4 parameter via /mavros/param/set.

        integer=True  → ParamValue(integer=int(val), real=0)  — for enum/int params
        integer=False → ParamValue(integer=0, real=float(val)) — for float params
        """
        srv = f'/{self._ns}/mavros/param/set'
        if self._param_srv is None:
            try:
                rospy.wait_for_service(srv, timeout=5.0)
                self._param_srv = rospy.ServiceProxy(srv, ParamSet)
            except rospy.ROSException:
                rospy.logwarn(f'[ar_landing_sm] param/set service {srv} unavailable')
                return False
        try:
            req = ParamSetRequest()
            req.param_id = name
            if integer:
                req.value = ParamValue(integer=int(val), real=0.0)
            else:
                req.value = ParamValue(integer=0, real=float(val))
            resp = self._param_srv(req)
            rospy.loginfo(
                f'[ar_landing_sm] param {name}={val} → '
                f'{"OK" if resp.success else "FAIL"}')
            return bool(resp.success)
        except rospy.ServiceException as e:
            rospy.logwarn(f'[ar_landing_sm] param/set {name} failed: {e}')
            return False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _publish_hover_att(self):
        """Publish level hover AttitudeTarget at current IMU yaw.

        Called every tick during PRE_ARMED and ARMING so that the same node
        that calls set_mode/arm is also streaming setpoints — this is the
        pattern PX4 requires to accept OFFBOARD mode reliably.
        """
        with self._lock:
            yaw = self._imu_yaw
        q = Rot.from_euler('xyz', [0.0, 0.0, yaw]).as_quat()
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE |
                         AttitudeTarget.IGNORE_PITCH_RATE |
                         AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                     z=float(q[2]), w=float(q[3]))
        att.thrust = self._hover_thrust
        self._pub_att.publish(att)

    def _transition(self, new_state):
        rospy.loginfo(f'[ar_landing_sm] {self._state} → {new_state}')
        self._state   = new_state
        self._state_t = rospy.Time.now().to_sec()
        self._pub_state.publish(String(data=new_state))

    def _age(self):
        return rospy.Time.now().to_sec() - self._state_t

    def _enable_ctrl(self, on: bool):
        self._pub_enable.publish(Bool(data=on))

    def _set_tgt(self, alt: float):
        """Request a new target altitude.  The actual commanded value tracks
        toward this at a rate-limited speed (downward only) to prevent the
        altitude overshoot/undershoot when the setpoint jumps suddenly."""
        self._desired_alt = float(alt)

    def _tick_alt_ramp(self):
        """Advance _target_alt toward _desired_alt and publish if changed.

        Upward transitions are instantaneous (let the altitude PID climb fast).
        Downward transitions are rate-limited to _alt_down_rate m/s so sudden
        target drops (climb_alt → hold_alt) produce a gentle glide rather than
        an aggressive descent that undershoots past the sea surface.
        """
        desired = self._desired_alt
        target  = self._target_alt
        if abs(desired - target) < 0.005:
            return
        dt = 1.0 / self._rate_hz
        if desired < target:
            step = min(abs(desired - target), self._alt_down_rate * dt)
        else:
            step = abs(desired - target)   # climb: no rate limit
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
            tgt_alt  = self._target_alt

        now = rospy.Time.now().to_sec()
        age = self._age()

        # Rate-limit downward target altitude changes every tick.
        self._tick_alt_ramp()

        # ── PRE_ARMED ─────────────────────────────────────────────────────────
        if state == State.PRE_ARMED:
            # Publish AttitudeTarget from this node so that the same process
            # that calls set_mode/arm is streaming setpoints.  PX4 requires
            # setpoints at >2 Hz before it will accept OFFBOARD mode.
            self._publish_hover_att()

            # Controller stays disabled until CLIMBING — SM has exclusive control
            # of AttitudeTarget during PRE_ARMED and ARMING.
            if age < 0.2:
                self._pub_yaw.publish(Float64(data=0.0))
                self._enable_ctrl(False)
                rospy.loginfo('[ar_landing_sm] PRE_ARMED: setpoints flowing, waiting for sensors …')

            # Disable PX4's preflight auto-disarm (COM_DISARM_PRFLT).
            # Default is 10 s: if the vehicle is armed on the ground and doesn't
            # take off within that window, PX4 disarms it automatically.
            # Our ARMING state can take >10 s (OFFBOARD confirmation + climb
            # setpoint propagation), so we disable this once, early.
            # Set PX4 safety params once MAVROS is ready.
            # Retry every tick (each 50 ms) until all succeed — MAVROS param/set
            # can fail with "unknown parameter" if called before the FCU has
            # finished booting.  We stop retrying as soon as all succeed.
            if not self._params_set and age >= 0.5:
                ok1 = self._set_px4_param('COM_DISARM_PRFLT', -1,  integer=True)
                ok2 = self._set_px4_param('NAV_RCL_ACT',      0,   integer=True)
                ok3 = self._set_px4_param('COM_OBL_ACT',      1,   integer=True)
                if ok1 and ok2 and ok3:
                    self._params_set = True
                    rospy.loginfo('[ar_landing_sm] PX4 safety params confirmed')

            if age >= self._t_prearm:
                rospy.logerr('[ar_landing_sm] PRE_ARMED timeout — ABORT')
                self._transition(State.ABORT)
            elif garmin is not None and age >= 3.0:
                # Sensors ready, setpoints have been flowing for ≥ 3 s
                self._transition(State.ARMING)

        # ── ARMING ────────────────────────────────────────────────────────────
        elif state == State.ARMING:
            # Publish hover setpoints every tick — same node must stream
            # setpoints AND call arm/set_mode for PX4 to stick in OFFBOARD.
            self._publish_hover_att()

            # Request OFFBOARD + arm.  Retry every 2 s until confirmed.
            # Also retry immediately when an unexpected disarm is detected
            # (e.g. PX4 kill-switch or preflight safety that we couldn't fully
            # suppress) so we don't sit out the full 2 s cadence.
            disarm_edge = self._prev_armed and not armed   # just got disarmed
            self._prev_armed = armed

            if armed and mode == 'OFFBOARD':
                rospy.loginfo(f'[ar_landing_sm] Armed in OFFBOARD — begin climb')
                self._set_tgt(self._climb_alt)
                self._enable_ctrl(True)   # hand off setpoints to landing controller
                self._transition(State.CLIMBING)
            else:
                periodic = int(age * self._rate_hz) % int(2.0 * self._rate_hz) == 0
                if periodic or disarm_edge:
                    if disarm_edge:
                        rospy.logwarn('[ar_landing_sm] Unexpected disarm — retrying immediately')
                    if mode != 'OFFBOARD':
                        ok = self._call_set_mode('OFFBOARD')
                        rospy.loginfo(
                            f'[ar_landing_sm] set_mode OFFBOARD → {"OK" if ok else "FAIL"} '
                            f'(current mode={mode!r})')
                    if not armed:
                        ok = self._call_arm(True)
                        rospy.loginfo(
                            f'[ar_landing_sm] arm → {"OK" if ok else "FAIL"} '
                            f'(mode={mode!r} armed={armed})')
                    # PX4 briefly enters armed+OFFBOARD before MRS can override it
                    # back (~50 ms window).  Poll at 100 Hz for 200 ms so we catch
                    # that window regardless of the Timer's own tick rate.
                    t_poll = rospy.Time.now().to_sec() + 0.2
                    while rospy.Time.now().to_sec() < t_poll and not rospy.is_shutdown():
                        self._publish_hover_att()   # keep setpoints flowing at 100 Hz
                        rospy.sleep(0.01)
                        with self._lock:
                            armed = self._mav_armed
                            mode  = self._mav_mode
                        if armed and mode == 'OFFBOARD':
                            break
                if age > self._t_arming:
                    rospy.logerr(
                        f'[ar_landing_sm] Arming timeout ({age:.0f} s) — ABORT')
                    self._transition(State.ABORT)

        # ── CLIMBING ──────────────────────────────────────────────────────────
        elif state == State.CLIMBING:
            # Attitude controller handles the climb: target_alt = climb_alt.
            # ArUco is not yet active (gimbal not at nadir) → controller runs
            # in level-flight / altitude-hold mode, climbing on Garmin PID.
            #
            # Keep SM setpoints flowing for the first 2 s so the controller node
            # has time to receive its 'enable' message and begin publishing.
            # After 2 s the controller is definitely running and we stop.
            if age < 2.0:
                self._publish_hover_att()
            if garmin is not None and garmin >= self._climb_alt - 0.5:
                rospy.loginfo(
                    f'[ar_landing_sm] Climb complete (garmin={garmin:.1f} m)')
                self._transition(State.GIMBAL_NADIR)
            elif age > self._t_climb:
                rospy.logerr(f'[ar_landing_sm] Climb timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                rospy.loginfo_throttle(
                    3.0, f'[ar_landing_sm] CLIMBING  garmin={garmin:.1f} m  '
                         f'target={self._climb_alt:.1f} m')

        # ── GIMBAL_NADIR ──────────────────────────────────────────────────────
        elif state == State.GIMBAL_NADIR:
            # SM keeps hover setpoints flowing until the controller is confirmed
            # active — prevents OFFBOARD from dropping during the gimbal slew.
            self._publish_hover_att()
            self._pub_pitch.publish(Float64(data=math.pi / 2.0))

            if gpitch >= 1.2:
                rospy.loginfo(
                    f'[ar_landing_sm] Gimbal nadir ({math.degrees(gpitch):.1f}°)')
                self._transition(State.ARUCO_SEARCH)
            elif age > self._t_nadir:
                rospy.logerr('[ar_landing_sm] Gimbal nadir timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)

        # ── ARUCO_SEARCH ──────────────────────────────────────────────────────
        elif state == State.ARUCO_SEARCH:
            # Keep hover setpoints as safety net — controller handles altitude
            # PID but if it stalls, the SM's 20 Hz setpoints prevent OFFBOARD loss.
            self._publish_hover_att()
            if tracking == 'TRACKING':
                rospy.loginfo('[ar_landing_sm] Deck TRACKING — begin ALIGN')
                self._align_ok_since = None
                self._set_tgt(self._hold_alt)
                self._transition(State.ALIGN)
            elif age > self._t_search:
                rospy.logerr(
                    f'[ar_landing_sm] ArUco search timeout ({age:.0f} s) — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                rospy.loginfo_throttle(
                    5.0, f'[ar_landing_sm] SEARCHING  '
                         f'({age:.0f}/{self._t_search:.0f} s)  status={tracking}')

        # ── ALIGN ─────────────────────────────────────────────────────────────
        elif state == State.ALIGN:
            # Attitude controller drives UAV laterally to deck using ArUco.
            # No goto, no GPS.  SM only monitors convergence.
            if e_horiz < self._align_thr:
                if self._align_ok_since is None:
                    self._align_ok_since = now
                elif now - self._align_ok_since >= self._align_set:
                    rospy.loginfo(
                        f'[ar_landing_sm] Aligned  |e_xy|={e_horiz:.2f} m < '
                        f'{self._align_thr:.2f} m  for {self._align_set:.1f} s')
                    self._last_desc_t = now
                    self._transition(State.DESCEND)
                    return
            else:
                self._align_ok_since = None

            if age > self._t_align:
                rospy.logerr('[ar_landing_sm] ALIGN timeout — ABORT')
                self._enable_ctrl(False)
                self._transition(State.ABORT)
            else:
                settled = f'{now - self._align_ok_since:.1f} s' if self._align_ok_since else 'no'
                rospy.loginfo_throttle(
                    2.0, f'[ar_landing_sm] ALIGN  |e_xy|={e_horiz:.2f} m  '
                         f'settled={settled}  tracking={tracking}')

        # ── DESCEND ───────────────────────────────────────────────────────────
        elif state == State.DESCEND:
            # Decrease target_alt at descent_rate.
            # Close to the deck (garmin < close_des_alt) use a much slower rate
            # so the controller has time to centre on the inner ArUco marker
            # before the drone touches down.
            dt_d = now - self._last_desc_t if self._last_desc_t else 0.0

            if garmin is not None and garmin < self._close_des_alt:
                # Position factor: 1.0 when centred, min_pos when |e_xy| >= err_scale.
                # If not TRACKING (pose stale), clamp to min so the drone crawls
                # and does not race through the last metres while the marker is lost.
                if tracking == 'TRACKING':
                    pos_factor = float(np.clip(
                        1.0 - e_horiz / self._desc_err_scale,
                        self._desc_min_pos_factor, 1.0))
                else:
                    pos_factor = self._desc_min_pos_factor

                # Altitude factor: 1.0 at close_des_alt, min_alt at deck level.
                alt_factor = float(np.clip(
                    garmin / self._close_des_alt,
                    self._desc_min_alt_factor, 1.0))

                rate = self._des_rate * pos_factor * alt_factor
                rospy.loginfo_throttle(
                    2.0, f'[ar_landing_sm] DESCEND rate={rate:.3f} m/s '
                         f'pos={pos_factor:.2f} alt={alt_factor:.2f} '
                         f'|e_xy|={e_horiz:.2f} m  garmin={garmin:.2f} m')
            else:
                rate = self._des_rate
            # Allow target to go to −1.0 m so the altitude PID always sees a
            # large negative error through the ground-effect zone (garmin < 1 m).
            # With target = 0.0 the PID error shrinks as garmin → 0, reducing
            # descent drive exactly when we need it most.  With target = −1.0 m
            # the PID hits its lower clamp and pushes constant min-thrust down.
            new_alt = max(-1.0, self._desired_alt - rate * dt_d)
            with self._lock:
                self._last_desc_t = now
            # Set desired directly (DESCEND manages its own rate)
            self._desired_alt = new_alt
            self._target_alt  = new_alt
            self._pub_tgtalt.publish(Float64(data=new_alt))

            # Touch-down: Garmin reads near deck surface.
            # Switch to AUTO.LAND BEFORE disabling the controller so PX4 is
            # already in AUTO.LAND when OFFBOARD setpoints stop.  Without this,
            # the 50 ms gap between controller-disable and AUTO.LAND causes PX4
            # to briefly enter Hold (COM_OBL_ACT=1) and climb slightly.
            if garmin is not None and garmin < self._touch_rng:
                rospy.loginfo(
                    f'[ar_landing_sm] Touch-down: garmin={garmin:.2f} m < '
                    f'{self._touch_rng:.2f} m')
                ok = self._call_set_mode('AUTO.LAND')
                rospy.loginfo(
                    f'[ar_landing_sm] AUTO.LAND → {"OK" if ok else "FAIL"}')
                self._enable_ctrl(False)
                self._transition(State.LANDED)
                return

            # ArUco lost guard
            if tracking not in ('TRACKING', 'SEARCHING'):
                lost_dur = (now - last_trk) if last_trk else age
                if lost_dur > self._desc_lost:
                    rospy.logwarn(
                        f'[ar_landing_sm] ArUco lost {lost_dur:.1f} s in DESCEND — HOVER')
                    self._set_tgt(self._hold_alt)
                    self._transition(State.HOVER)
                    return

            rospy.loginfo_throttle(
                2.0, f'[ar_landing_sm] DESCEND  '
                     f'tgt={new_alt:.2f} m  garmin={garmin:.2f} m  '
                     f'|e_xy|={e_horiz:.2f} m  track={tracking}')

        # ── LANDED ────────────────────────────────────────────────────────────
        elif state == State.LANDED:
            # Controller already disabled in DESCEND→LANDED transition.
            # PX4 refuses manual disarm until its own landing detector fires
            # ("Not landed").  Switch to AUTO.LAND so PX4 handles the final
            # drop + landing detection and triggers auto-disarm itself.
            if not self._disarmed:
                # AUTO.LAND was already set in DESCEND before this transition.
                # PX4 auto-disarms after confirming touchdown — just monitor for it.
                if not armed:
                    self._disarmed = True
                    rospy.loginfo('[ar_landing_sm] *** DISARMED — motors off ***')
                else:
                    rospy.loginfo_throttle(
                        3.0, f'[ar_landing_sm] LANDED waiting for PX4 disarm '
                             f'(age={age:.1f} s  mode={mode})')
            else:
                rospy.loginfo_once('[ar_landing_sm] *** LANDED ***')

        # ── HOVER (ArUco-lost recovery) ───────────────────────────────────────
        elif state == State.HOVER:
            # Controller stays enabled.  ArUco unavailable → level altitude hold.
            # The attitude controller will hold level at target_alt (hold_alt).
            # No goto needed — purely sensor-driven (Garmin altitude PID).
            rospy.logwarn_throttle(
                5.0, f'[ar_landing_sm] HOVER — waiting for ArUco  '
                     f'garmin={garmin:.2f} m  tracking={tracking}')
            if tracking == 'TRACKING':
                rospy.loginfo('[ar_landing_sm] ArUco reacquired — re-enter ALIGN')
                self._align_ok_since = None
                self._set_tgt(self._hold_alt)
                self._transition(State.ALIGN)

        # ── ABORT ─────────────────────────────────────────────────────────────
        elif state == State.ABORT:
            self._enable_ctrl(False)
            rospy.logerr_once(
                '[ar_landing_sm] *** ABORTED — disarm manually if needed ***')

    def run(self):
        self._pub_state.publish(String(data=self._state))
        rospy.spin()


if __name__ == '__main__':
    try:
        ArLandingStateMachine().run()
    except rospy.ROSInterruptException:
        pass
