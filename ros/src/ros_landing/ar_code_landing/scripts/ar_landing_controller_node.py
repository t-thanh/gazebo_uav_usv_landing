#!/usr/bin/env python3
"""
ar_landing_controller_node.py
──────────────────────────────
GPS-free attitude + thrust controller for autonomous USV deck landing.

Sensor inputs (no GPS required)
────────────────────────────────
  • Landing target  (/ar_landing/gimbal_tracker/usv_world_pose)
                    PoseStamped in world ENU frame — published by gimbal_aruco_tracker_node.
                    Automatically switches outer→inner marker at close range so the
                    controller always tracks the correct target regardless of altitude.
  • Garmin range   (/<ns>/garmin/range)  — AGL altitude [m]
  • IMU            (/<ns>/mavros/imu/data)  — current yaw
  • EKF odometry   (/<ns>/mavros/local_position/odom)  — velocity for damping

Control law
───────────
  Altitude (always active, uses only Garmin + IMU):
    e_z = target_alt − garmin_range
    thrust = hover_thrust + Kp_z·e_z + Ki_z·∫e_z·dt + Kd_z·(de_z/dt)

  Horizontal (active only when tracker world pose is fresh):
    e_xy_w    = usv_world_pos[:2] − drone_pos[:2]   (world-frame lateral error)
    e_xy_b    = Rz(−yaw) @ e_xy_w                   (rotate to body frame)
    a_body    = Kp·e_xy_b − Kd·vel_xy_body          (PD)
    pitch_cmd = clamp( a_x / g, ±max_tilt)
    roll_cmd  = clamp(−a_y / g, ±max_tilt)

  When tracker pose is stale: velocity damping only (active braking, no position term).

Output
──────
  /<ns>/mavros/setpoint_raw/attitude   mavros_msgs/AttitudeTarget  (50 Hz)
  ~horizontal_error  geometry_msgs/Vector3Stamped  (ex, ey, |e_xy|)
  ~altitude_error    std_msgs/Float64
  ~status            std_msgs/String   ACTIVE_ALIGN / ACTIVE_DESCEND / ACTIVE_HOVER / IDLE

SM interface
────────────
  ~enable          std_msgs/Bool    from SM — enables / disables publishing
  ~target_altitude std_msgs/Float64 from SM — current altitude setpoint [m]
"""

import math
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from sensor_msgs.msg import Range, Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String, Bool
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Vector3

from mavros_msgs.msg import AttitudeTarget

_GRAVITY = 9.806


class ArLandingControllerNode:

    def __init__(self):
        rospy.init_node('ar_landing_controller_node', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')
        self._ns = ns

        # ── Parameters ────────────────────────────────────────────────────────
        self._rate_hz      = rospy.get_param('~rate_hz',      50.0)
        self._kp_xy        = rospy.get_param('~kp_xy',        0.40)
        self._kd_xy        = rospy.get_param('~kd_xy',        0.20)
        self._max_tilt     = math.radians(rospy.get_param('~max_tilt_deg', 12.0))
        self._hover_thrust = rospy.get_param('~hover_thrust', 0.58)
        # Close-range boost: when garmin < close_range_alt_m, multiply kp/kd_xy
        # by this gain so the drone centres aggressively on the inner ArUco marker.
        self._close_alt    = rospy.get_param('~close_range_alt_m',   2.5)
        self._close_gain   = rospy.get_param('~close_range_xy_gain', 2.5)
        self._kp_z         = rospy.get_param('~kp_z',         0.35)
        self._ki_z         = rospy.get_param('~ki_z',         0.02)
        self._kd_z         = rospy.get_param('~kd_z',         0.20)
        self._max_dthrust  = rospy.get_param('~max_thrust_delta', 0.25)
        self._pose_timeout = rospy.get_param('~pose_timeout_s',   2.0)
        self._base_off     = np.array([
            rospy.get_param('~x_offset', 0.10),
            rospy.get_param('~y_offset', 0.00),
            rospy.get_param('~z_offset', 0.00),
        ])

        # ── Shared state ──────────────────────────────────────────────────────
        self._lock = threading.Lock()

        self._enabled     = False
        self._target_alt  = 0.5        # set by SM; starts low (pre-arm)

        self._drone_pos   = np.zeros(3)
        self._drone_vel   = np.zeros(3)
        self._drone_rot   = Rot.identity()
        self._odom_ready  = False

        self._imu_yaw     = 0.0
        self._imu_ready   = False

        self._garmin      = None       # latest Garmin range [m]
        self._last_rng_t  = None

        self._usv_world_pos = None     # latest landing target in world frame (from tracker)
        self._last_pose_t   = None

        # Altitude PID state
        self._int_z       = 0.0
        self._prev_e_z    = 0.0
        self._last_ctrl_t = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_att   = rospy.Publisher(
            f'/{ns}/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        self._pub_herr  = rospy.Publisher('horizontal_error', Vector3Stamped, queue_size=5)
        self._pub_aerr  = rospy.Publisher('altitude_error',   Float64,        queue_size=5)
        self._pub_stat  = rospy.Publisher('status',           String,         queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────────
        odom_topic = rospy.get_param('~odom_topic', f'/{ns}/ground_truth')

        # EKF velocity — available without GPS (PX4 EKF2 fuses IMU + barometer).
        # Used for horizontal velocity damping instead of ground_truth so the
        # controller works identically in simulation and on real hardware.
        # local_position/odom twist is in body frame; _ekf_cb rotates to world.
        self._ekf_vel   = np.zeros(3)
        self._ekf_ready = False

        # Subscribe to the gimbal tracker's world-frame target pose.
        # The tracker handles outer→inner marker switching internally, so the
        # controller always receives the correct landing target regardless of altitude.
        rospy.Subscriber('/ar_landing/gimbal_tracker/usv_world_pose', PoseStamped,
                         self._pose_cb,  queue_size=5)
        rospy.Subscriber(f'/{ns}/garmin/range', Range,
                         self._range_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/imu/data', Imu,
                         self._imu_cb,   queue_size=5)
        rospy.Subscriber(odom_topic, Odometry,
                         self._odom_cb,  queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/local_position/odom', Odometry,
                         self._ekf_cb,   queue_size=5)
        rospy.Subscriber('enable',          Bool,    self._enable_cb,  queue_size=1)
        rospy.Subscriber('target_altitude', Float64, self._target_cb,  queue_size=1)

        # ── Control timer ─────────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._control_cb)

        rospy.loginfo(
            f'[ar_landing_ctrl] Init — ns={ns}  odom={odom_topic}  '
            f'rate={self._rate_hz:.0f} Hz  Kp_xy={self._kp_xy}  '
            f'hover_thrust={self._hover_thrust}  Kp_z={self._kp_z}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _enable_cb(self, msg: Bool):
        with self._lock:
            was = self._enabled
            self._enabled = bool(msg.data)
            if self._enabled and not was:
                self._int_z = 0.0
                # Seed prev_e_z with the CURRENT error so the first derivative
                # term is (e_z - e_z) / dt = 0 instead of e_z / dt → avoids
                # the massive thrust spike that caused the >20 m overshoot.
                if self._garmin is not None:
                    self._prev_e_z = self._target_alt - self._garmin
                else:
                    self._prev_e_z = 0.0
                self._last_ctrl_t = None   # force dt = 1/rate on first tick
        rospy.loginfo(f'[ar_landing_ctrl] {"ENABLED" if msg.data else "DISABLED"}')

    def _target_cb(self, msg: Float64):
        with self._lock:
            self._target_alt = float(msg.data)

    def _pose_cb(self, msg: PoseStamped):
        """Receive landing target in world ENU frame from gimbal_aruco_tracker_node.
        The tracker publishes the active marker (outer or inner) as a world position,
        so no FK is needed here — the target is already in the drone's reference frame."""
        p = msg.pose.position
        with self._lock:
            self._usv_world_pos = np.array([p.x, p.y, p.z])
            self._last_pose_t   = rospy.Time.now().to_sec()

    def _range_cb(self, msg: Range):
        with self._lock:
            self._garmin    = float(msg.range)
            self._last_rng_t = rospy.Time.now().to_sec()

    def _imu_cb(self, msg: Imu):
        q   = msg.orientation
        yaw = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._imu_yaw  = yaw
            self._imu_ready = True

    def _odom_cb(self, msg: Odometry):
        p  = msg.pose.pose.position
        q  = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        with self._lock:
            self._drone_pos  = np.array([p.x, p.y, p.z])
            self._drone_vel  = np.array([lv.x, lv.y, lv.z])
            self._drone_rot  = Rot.from_quat([q.x, q.y, q.z, q.w])
            self._odom_ready = True

    def _ekf_cb(self, msg: Odometry):
        """PX4 EKF2 velocity (body frame) → world frame for damping term."""
        q  = msg.pose.pose.orientation
        lv = msg.twist.twist.linear
        rot = Rot.from_quat([q.x, q.y, q.z, q.w])
        v_world = rot.apply(np.array([lv.x, lv.y, lv.z]))
        with self._lock:
            self._ekf_vel   = v_world
            self._ekf_ready = True

    # ── Main control loop (50 Hz) ─────────────────────────────────────────────

    def _control_cb(self, _event):
        with self._lock:
            enabled       = self._enabled
            target_alt    = self._target_alt
            garmin        = self._garmin
            imu_yaw       = self._imu_yaw
            imu_ready     = self._imu_ready
            odom_ready    = self._odom_ready
            ekf_ready     = self._ekf_ready
            usv_world_pos = self._usv_world_pos.copy() if self._usv_world_pos is not None else None
            last_pose_t   = self._last_pose_t
            pos           = self._drone_pos.copy()
            vel           = self._ekf_vel.copy()
            int_z         = self._int_z
            prev_e_z      = self._prev_e_z
            last_ctrl_t   = self._last_ctrl_t

        if not enabled:
            self._pub_stat.publish(String(data='IDLE'))
            return

        if not imu_ready:
            rospy.logwarn_throttle(3.0, '[ar_landing_ctrl] Waiting for IMU …')
            return

        now = rospy.Time.now().to_sec()
        dt  = (now - last_ctrl_t) \
              if (last_ctrl_t is not None and 0 < now - last_ctrl_t < 0.5) \
              else (1.0 / self._rate_hz)

        # ── Altitude PID (always active — only requires Garmin + IMU) ─────────
        if garmin is None:
            # No Garmin yet: publish a safe zero-acceleration hover command
            # using slightly above hover thrust so the UAV doesn't fall.
            thrust = self._hover_thrust
            e_z    = 0.0
        else:
            e_z    = target_alt - garmin
            int_z  = float(np.clip(int_z + e_z * dt, -0.5, 0.5))
            d_ez   = (e_z - prev_e_z) / dt
            thrust = (self._hover_thrust
                      + self._kp_z * e_z
                      + self._ki_z * int_z
                      + self._kd_z * d_ez)
            thrust = float(np.clip(
                thrust,
                self._hover_thrust - self._max_dthrust,
                self._hover_thrust + self._max_dthrust))
            thrust = float(np.clip(thrust, 0.05, 1.0))

        # ── Horizontal PD (active only when ArUco pose is fresh) ─────────────
        pose_age = (now - last_pose_t) if last_pose_t is not None else 999.0
        e_horiz  = 0.0
        e_xy_world = np.zeros(2)

        if pose_age < self._pose_timeout and odom_ready and ekf_ready:
            # usv_world_pos is the landing target in world ENU frame,
            # published by gimbal_aruco_tracker_node (handles outer→inner switch).
            e_xy_world   = usv_world_pos[:2] - pos[:2]
            e_horiz      = float(np.linalg.norm(e_xy_world))

            # Close to the deck: boost lateral gains so the drone centres tightly
            # on the inner ArUco marker before touchdown.
            if garmin is not None and garmin < self._close_alt:
                kp_xy = self._kp_xy * self._close_gain
                kd_xy = self._kd_xy * self._close_gain
            else:
                kp_xy = self._kp_xy
                kd_xy = self._kd_xy

            # Rotate error to body frame (yaw-only)
            R_yaw_inv   = Rot.from_euler('z', -imu_yaw)
            e_xy_b      = R_yaw_inv.apply(np.append(e_xy_world, 0.0))[:2]
            vel_b       = R_yaw_inv.apply(vel)[:2]

            ax = kp_xy * e_xy_b[0] - kd_xy * vel_b[0]
            ay = kp_xy * e_xy_b[1] - kd_xy * vel_b[1]

            pitch_cmd = float(np.clip( ax / _GRAVITY, -self._max_tilt, self._max_tilt))
            roll_cmd  = float(np.clip(-ay / _GRAVITY, -self._max_tilt, self._max_tilt))
            status    = 'ACTIVE_DESCEND' if target_alt < 4.5 else 'ACTIVE_ALIGN'
        else:
            # ArUco unavailable: brake horizontal drift using EKF velocity.
            # Zero attitude would let any existing velocity carry the drone away;
            # applying the D-only term actively decelerates it back to zero.
            if ekf_ready:
                R_yaw_inv = Rot.from_euler('z', -imu_yaw)
                vel_b     = R_yaw_inv.apply(vel)[:2]
                ax = -self._kd_xy * vel_b[0]
                ay = -self._kd_xy * vel_b[1]
                pitch_cmd = float(np.clip( ax / _GRAVITY, -self._max_tilt, self._max_tilt))
                roll_cmd  = float(np.clip(-ay / _GRAVITY, -self._max_tilt, self._max_tilt))
            else:
                pitch_cmd = 0.0
                roll_cmd  = 0.0
            status    = 'ACTIVE_HOVER'

        # ── Build and publish AttitudeTarget ──────────────────────────────────
        q_cmd = Rot.from_euler('xyz', [roll_cmd, pitch_cmd, imu_yaw]).as_quat()

        att = AttitudeTarget()
        att.header.stamp    = rospy.Time.now()
        att.header.frame_id = 'world'
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE
                       | AttitudeTarget.IGNORE_PITCH_RATE
                       | AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation.x = float(q_cmd[0])
        att.orientation.y = float(q_cmd[1])
        att.orientation.z = float(q_cmd[2])
        att.orientation.w = float(q_cmd[3])
        att.thrust = thrust
        self._pub_att.publish(att)

        # ── Diagnostics ───────────────────────────────────────────────────────
        v3 = Vector3Stamped()
        v3.header.stamp    = att.header.stamp
        v3.header.frame_id = 'world'
        v3.vector = Vector3(x=float(e_xy_world[0]),
                            y=float(e_xy_world[1]),
                            z=e_horiz)
        self._pub_herr.publish(v3)
        self._pub_aerr.publish(Float64(data=e_z))
        self._pub_stat.publish(String(data=status))

        with self._lock:
            self._int_z       = int_z
            self._prev_e_z    = e_z
            self._last_ctrl_t = now

        rospy.logdebug(
            f'[ar_landing_ctrl] tgt={target_alt:.2f}m  '
            f'garmin={garmin:.2f}m  e_z={e_z:+.2f}m  '
            f'thrust={thrust:.3f}  '
            f'|e_xy|={e_horiz:.2f}m  '
            f'roll={math.degrees(roll_cmd):+.1f}°  '
            f'pitch={math.degrees(pitch_cmd):+.1f}°  '
            f'[{status}]')

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        ArLandingControllerNode().run()
    except rospy.ROSInterruptException:
        pass
