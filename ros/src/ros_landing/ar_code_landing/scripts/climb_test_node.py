#!/usr/bin/env python3
"""
climb_test_node.py
───────────────────
Standalone CLIMB verification — replicates the SM's PRE_ARMED → ARMING →
CLIMBING path using pure MAVROS (no MRS, no GPS, no ground truth).

Horizontal stabilisation — EKF velocity from MAVROS
─────────────────────────────────────────────────────
Raw accelerometer integration (our previous attempt) failed because the
decay was applied at IMU rate (100 Hz): τ = −0.01/ln(0.97) ≈ 0.3 s, so
|v_est| decayed to ~0 before the control loop ran.

The correct approach: PX4's EKF2 already does IMU-based velocity estimation
(gyro + accelerometer + barometer, with bias calibration and proper sensor
fusion).  Its output is /mavros/local_position/odom which is available in
real life WITHOUT GPS — only barometer + IMU are required.

We subscribe to that topic for velocity feedback and apply a PD law:

    v_body  = Rz(−yaw) @ v_world         [body frame]
    pitch   = clip(−kd_vel · v_x / g,  ±max_tilt)
    roll    = clip( kd_vel · v_y / g,  ±max_tilt)

This damps horizontal drift reliably without requiring GPS or ground truth.

State flow
──────────
  PRE_ARMED   Flow AttitudeTarget so PX4 can enter OFFBOARD.
              Disable COM_DISARM_PRFLT.  Wait for Garmin + IMU.
  ARMING      Request OFFBOARD + arm every 2 s until confirmed.
  CLIMBING    Altitude PID, level flight (no velocity damping yet).
  HOLD        Altitude PID + EKF velocity damping.
  DONE/ABORT  Continue altitude + velocity damping indefinitely.
"""

import threading
import numpy as np
import rospy

from scipy.spatial.transform import Rotation as Rot

from sensor_msgs.msg import Range, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion
from mavros_msgs.msg import AttitudeTarget, State as MavState, ParamValue
from mavros_msgs.srv import (CommandBool, CommandBoolRequest,
                              SetMode,    SetModeRequest,
                              ParamSet,   ParamSetRequest)

_GRAVITY = 9.806


class State:
    PRE_ARMED = 'PRE_ARMED'
    ARMING    = 'ARMING'
    CLIMBING  = 'CLIMBING'
    HOLD      = 'HOLD'
    DONE      = 'DONE'
    ABORT     = 'ABORT'


class ClimbTest:

    def __init__(self):
        rospy.init_node('climb_test', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')
        self._ns = ns

        # ── Parameters ────────────────────────────────────────────────────────
        self._target_alt   = rospy.get_param('~target_alt_m',      10.0)
        self._hover_thrust = rospy.get_param('~hover_thrust',       0.58)
        # Altitude PID
        self._kp_z         = rospy.get_param('~kp_z',               0.35)
        self._ki_z         = rospy.get_param('~ki_z',               0.02)
        self._kd_z         = rospy.get_param('~kd_z',               0.20)
        self._max_dthrust  = rospy.get_param('~max_dthrust',        0.25)
        # Horizontal velocity damping (uses EKF velocity from mavros/local_position/odom)
        self._kd_vel       = rospy.get_param('~kd_vel',             1.0)
        self._max_tilt     = np.radians(rospy.get_param('~max_tilt_deg', 12.0))

        self._rate_hz      = rospy.get_param('~rate_hz',           20.0)
        self._prearm_s     = rospy.get_param('~prearm_s',           3.0)
        self._prearm_to    = rospy.get_param('~prearm_timeout_s',  10.0)
        self._arm_to       = rospy.get_param('~arming_timeout_s',  30.0)
        self._climb_to     = rospy.get_param('~climb_timeout_s',   90.0)
        self._hold_s       = rospy.get_param('~hold_s',            10.0)

        # ── Runtime state ─────────────────────────────────────────────────────
        self._lock       = threading.Lock()
        self._state      = State.PRE_ARMED
        self._state_t    = None

        self._garmin     = None
        self._imu_yaw    = 0.0
        self._imu_ready  = False
        self._mav_armed  = False
        self._mav_mode   = ''
        self._prev_armed = False

        # EKF velocity from /mavros/local_position/odom (world frame, m/s)
        self._ekf_vel    = np.zeros(2)
        self._ekf_ready  = False

        # Altitude PID state
        self._int_z       = 0.0
        self._prev_e_z    = 0.0
        self._last_ctrl_t = None

        self._disarm_param_set = False

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_att = rospy.Publisher(
            f'/{ns}/mavros/setpoint_raw/attitude',
            AttitudeTarget, queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/mavros/state',             MavState, self._mav_cb,   queue_size=5)
        rospy.Subscriber(f'/{ns}/garmin/range',             Range,    self._range_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/imu/data',          Imu,      self._imu_cb,   queue_size=10)
        # EKF2 velocity — available in real life without GPS (needs IMU + barometer)
        rospy.Subscriber(f'/{ns}/mavros/local_position/odom', Odometry, self._ekf_cb, queue_size=5)

        # ── Service proxies (lazy) ────────────────────────────────────────────
        self._arm_srv   = None
        self._mode_srv  = None
        self._param_srv = None

        rospy.loginfo(
            f'[climb_test] Init — ns={ns}  target={self._target_alt} m  '
            f'hover_thrust={self._hover_thrust}  kd_vel={self._kd_vel}')

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _mav_cb(self, msg: MavState):
        with self._lock:
            self._mav_armed = bool(msg.armed)
            self._mav_mode  = str(msg.mode)

    def _range_cb(self, msg: Range):
        with self._lock:
            self._garmin = float(msg.range)

    def _imu_cb(self, msg: Imu):
        q   = msg.orientation
        yaw = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._imu_yaw   = float(yaw)
            self._imu_ready = True

    def _ekf_cb(self, msg: Odometry):
        """
        PX4 EKF2 velocity output via MAVROS.  Available without GPS —
        EKF2 fuses IMU + barometer (+ vision/optical-flow when present).
        In local_position/odom the twist is in the body frame (child_frame_id
        is base_link), so rotate to world using the pose quaternion.
        """
        q   = msg.pose.pose.orientation
        rot = Rot.from_quat([q.x, q.y, q.z, q.w])
        v   = msg.twist.twist.linear          # body-frame velocity from EKF
        v_body = np.array([v.x, v.y, v.z])
        v_world = rot.apply(v_body)
        with self._lock:
            self._ekf_vel   = v_world[:2].copy()
            self._ekf_ready = True

    # ── Service helpers ───────────────────────────────────────────────────────

    def _call_arm(self, value: bool) -> bool:
        svc = f'/{self._ns}/mavros/cmd/arming'
        try:
            if self._arm_srv is None:
                rospy.wait_for_service(svc, timeout=3.0)
                self._arm_srv = rospy.ServiceProxy(svc, CommandBool)
            return bool(self._arm_srv(CommandBoolRequest(value=value)).success)
        except Exception as exc:
            rospy.logwarn(f'[climb_test] arm error: {exc}')
            return False

    def _call_set_mode(self, mode: str) -> bool:
        svc = f'/{self._ns}/mavros/set_mode'
        try:
            if self._mode_srv is None:
                rospy.wait_for_service(svc, timeout=3.0)
                self._mode_srv = rospy.ServiceProxy(svc, SetMode)
            return bool(self._mode_srv(SetModeRequest(custom_mode=mode)).mode_sent)
        except Exception as exc:
            rospy.logwarn(f'[climb_test] set_mode error: {exc}')
            return False

    def _set_px4_param(self, name: str, real_val: float) -> bool:
        svc = f'/{self._ns}/mavros/param/set'
        try:
            if self._param_srv is None:
                rospy.wait_for_service(svc, timeout=3.0)
                self._param_srv = rospy.ServiceProxy(svc, ParamSet)
            req = ParamSetRequest()
            req.param_id = name
            req.value    = ParamValue(integer=0, real=real_val)
            return bool(self._param_srv(req).success)
        except Exception as exc:
            rospy.logwarn(f'[climb_test] param/set error: {exc}')
            return False

    # ── Control ───────────────────────────────────────────────────────────────

    def _altitude_pid(self, garmin: float) -> float:
        """Altitude PID — identical to ar_landing_controller_node."""
        now = rospy.Time.now().to_sec()
        dt  = ((now - self._last_ctrl_t)
               if (self._last_ctrl_t is not None and 0 < now - self._last_ctrl_t < 0.5)
               else (1.0 / self._rate_hz))
        self._last_ctrl_t = now

        e_z = self._target_alt - garmin
        self._int_z    = float(np.clip(self._int_z + e_z * dt, -0.5, 0.5))
        d_ez           = (e_z - self._prev_e_z) / dt
        self._prev_e_z = e_z

        thrust = (self._hover_thrust
                  + self._kp_z * e_z
                  + self._ki_z * self._int_z
                  + self._kd_z * d_ez)
        thrust = float(np.clip(thrust,
                               self._hover_thrust - self._max_dthrust,
                               self._hover_thrust + self._max_dthrust))
        return float(np.clip(thrust, 0.05, 1.0))

    def _velocity_damp_angles(self, yaw: float, v_world: np.ndarray) -> tuple:
        """
        Convert world-frame EKF velocity to body-frame counter-tilt.
        Returns (roll_cmd, pitch_cmd) in radians.

        Desired deceleration:  a_cmd = -kd_vel · v_body
        Tilt to achieve it:    angle = a_cmd / g
        """
        v_body = Rot.from_euler('z', -yaw).apply(
            np.array([v_world[0], v_world[1], 0.0]))[:2]

        pitch = float(np.clip(-self._kd_vel * v_body[0] / _GRAVITY,
                               -self._max_tilt, self._max_tilt))
        roll  = float(np.clip( self._kd_vel * v_body[1] / _GRAVITY,
                               -self._max_tilt, self._max_tilt))
        return roll, pitch

    def _publish_att(self, thrust: float, roll: float = 0.0, pitch: float = 0.0):
        with self._lock:
            yaw = self._imu_yaw
        q = Rot.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        att = AttitudeTarget()
        att.header.stamp    = rospy.Time.now()
        att.header.frame_id = 'world'
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE
                       | AttitudeTarget.IGNORE_PITCH_RATE
                       | AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation = Quaternion(x=float(q[0]), y=float(q[1]),
                                     z=float(q[2]), w=float(q[3]))
        att.thrust = float(np.clip(thrust, 0.05, 1.0))
        self._pub_att.publish(att)

    # ── State machine ─────────────────────────────────────────────────────────

    def _transition(self, new_state: str):
        rospy.loginfo(f'[climb_test] {self._state} → {new_state}')
        self._state   = new_state
        self._state_t = rospy.Time.now().to_sec()

    def _age(self) -> float:
        return rospy.Time.now().to_sec() - self._state_t

    def run(self):
        rate = rospy.Rate(self._rate_hz)
        self._state_t = rospy.Time.now().to_sec()
        _last_arm_attempt = 0.0

        rospy.loginfo('[climb_test] Starting — flowing setpoints to warm up PX4 OFFBOARD …')

        while not rospy.is_shutdown():
            with self._lock:
                garmin   = self._garmin
                armed    = self._mav_armed
                mode     = self._mav_mode
                imu_ok   = self._imu_ready
                ekf_ok   = self._ekf_ready
                yaw      = self._imu_yaw
                ekf_vel  = self._ekf_vel.copy()

            age = self._age()

            # ── PRE_ARMED ─────────────────────────────────────────────────────
            if self._state == State.PRE_ARMED:
                self._publish_att(self._hover_thrust)

                if not self._disarm_param_set and age >= 0.5:
                    ok1 = self._set_px4_param('COM_DISARM_PRFLT', -1.0)
                    # NAV_RCL_ACT = 0: disable RC-loss failsafe so PX4 does not
                    # emergency-land when there is no RC transmitter (simulation
                    # or companion-only flight with OFFBOARD control).
                    ok2 = self._set_px4_param('NAV_RCL_ACT', 0.0)
                    if ok1:
                        rospy.loginfo(
                            f'[climb_test] COM_DISARM_PRFLT=-1  '
                            f'NAV_RCL_ACT=0 → {"OK" if ok2 else "FAIL (ignored)"}')
                        self._disarm_param_set = True
                    else:
                        rospy.logwarn('[climb_test] COM_DISARM_PRFLT set failed — will retry')

                if age >= self._prearm_to:
                    rospy.logerr('[climb_test] PRE_ARMED timeout — ABORT')
                    self._transition(State.ABORT)
                elif garmin is not None and imu_ok and age >= self._prearm_s:
                    rospy.loginfo(f'[climb_test] Sensors ready (garmin={garmin:.2f} m) — ARMING')
                    _last_arm_attempt = age - 999.0
                    self._transition(State.ARMING)
                elif not imu_ok:
                    rospy.loginfo_throttle(3.0, '[climb_test] PRE_ARMED: waiting for IMU …')
                elif garmin is None:
                    rospy.loginfo_throttle(3.0, '[climb_test] PRE_ARMED: waiting for Garmin …')
                else:
                    rospy.loginfo_throttle(
                        3.0, f'[climb_test] PRE_ARMED: flowing setpoints '
                             f'({age:.1f}/{self._prearm_s:.0f} s) …')

            # ── ARMING ────────────────────────────────────────────────────────
            elif self._state == State.ARMING:
                self._publish_att(self._hover_thrust)

                disarm_edge = self._prev_armed and not armed
                self._prev_armed = armed

                if armed and mode == 'OFFBOARD':
                    rospy.loginfo('[climb_test] Armed in OFFBOARD — begin CLIMB')
                    self._int_z     = 0.0
                    self._prev_e_z  = 0.0
                    self._last_ctrl_t = None
                    self._transition(State.CLIMBING)
                elif age > self._arm_to:
                    rospy.logerr(f'[climb_test] Arming timeout ({age:.0f} s) — ABORT')
                    self._transition(State.ABORT)
                else:
                    now_t    = rospy.Time.now().to_sec()
                    periodic = (now_t - _last_arm_attempt) >= 2.0
                    if periodic or disarm_edge:
                        _last_arm_attempt = now_t
                        if disarm_edge:
                            rospy.logwarn('[climb_test] Unexpected disarm — retrying')
                        if mode != 'OFFBOARD':
                            ok = self._call_set_mode('OFFBOARD')
                            rospy.loginfo(
                                f'[climb_test] set_mode OFFBOARD → {"OK" if ok else "FAIL"} '
                                f'(current mode={mode!r})')
                        if not armed:
                            ok = self._call_arm(True)
                            rospy.loginfo(
                                f'[climb_test] arm → {"OK" if ok else "FAIL"} '
                                f'(mode={mode!r} armed={armed})')

            # ── CLIMBING — altitude PID, level flight ─────────────────────────
            elif self._state == State.CLIMBING:
                thrust = (self._altitude_pid(garmin) if garmin is not None
                          else self._hover_thrust)
                self._publish_att(thrust)   # roll=0, pitch=0 during climb

                if garmin is not None:
                    rospy.loginfo_throttle(
                        3.0, f'[climb_test] CLIMBING  garmin={garmin:.1f} m  '
                             f'target={self._target_alt:.1f} m  thrust={thrust:.3f}  '
                             f'e_z={self._target_alt - garmin:+.2f} m')
                    if garmin >= self._target_alt - 0.5:
                        rospy.loginfo(
                            f'[climb_test] Climb complete (garmin={garmin:.1f} m) — HOLD')
                        self._transition(State.HOLD)
                    elif age > self._climb_to:
                        rospy.logerr(
                            f'[climb_test] Climb timeout ({age:.0f} s) at '
                            f'garmin={garmin:.1f} m — ABORT')
                        self._transition(State.ABORT)

            # ── HOLD / DONE / ABORT — altitude PID + EKF velocity damping ──────
            else:
                thrust = (self._altitude_pid(garmin) if garmin is not None
                          else self._hover_thrust)

                if imu_ok and ekf_ok:
                    roll_cmd, pitch_cmd = self._velocity_damp_angles(yaw, ekf_vel)
                else:
                    roll_cmd, pitch_cmd = 0.0, 0.0
                    if not ekf_ok:
                        rospy.logwarn_throttle(
                            5.0, '[climb_test] EKF velocity not yet available — level flight')

                self._publish_att(thrust, roll=roll_cmd, pitch=pitch_cmd)

                v_mag = float(np.linalg.norm(ekf_vel))

                if self._state == State.HOLD:
                    remaining = self._hold_s - age
                    rospy.loginfo_throttle(
                        3.0, f'[climb_test] HOLD  garmin={garmin:.1f} m  '
                             f'|v_ekf|={v_mag:.2f} m/s  thrust={thrust:.3f}  '
                             f'roll={np.degrees(roll_cmd):+.1f}°  '
                             f'pitch={np.degrees(pitch_cmd):+.1f}°  '
                             f'({remaining:.0f} s remaining)')
                    if age >= self._hold_s:
                        rospy.loginfo('[climb_test] Hold complete — DONE (holding indefinitely)')
                        self._transition(State.DONE)

                elif self._state == State.DONE:
                    rospy.loginfo_throttle(
                        5.0, f'[climb_test] DONE  garmin={garmin:.1f} m  '
                             f'|v_ekf|={v_mag:.2f} m/s  thrust={thrust:.3f}')

                elif self._state == State.ABORT:
                    rospy.logerr_throttle(
                        5.0, f'[climb_test] ABORT  garmin={garmin:.1f} m  '
                             f'|v_ekf|={v_mag:.2f} m/s  thrust={thrust:.3f}')

            rate.sleep()


def main():
    ClimbTest().run()


if __name__ == '__main__':
    main()
