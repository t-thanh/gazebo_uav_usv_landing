#!/usr/bin/env python3
"""
gimbal_aruco_tracker_node.py
─────────────────────────────
IBVS gimbal controller driven by ArUco/AprilTag pose detections.

Drop-in replacement for gimbal_tracker_node (YOLO-based) that uses the more
stable ArUco pose estimate from aruco_pose_estimator_node instead of a neural-
network bounding-box centroid.

Architecture (mirrors gimbal_tracker_node exactly)
───────────────────────────────────────────────────
Detection callback (event-driven):
  Receives PoseStamped in camera optical frame from aruco_pose_estimator.
  Converts tvec (position in camera frame) to world frame via gimbal FK.
  Stores the world position in an EMA filter.  Does NOT command the gimbal.

Control timer (30 Hz):
  Projects the stored USV world position back into the current camera frame
  using up-to-date FK, obtaining synthetic angular errors.  Runs PID step,
  publishes gimbal yaw / pitch commands.

This decoupling means:
  • Gimbal keeps tracking when the drone translates/rotates (world position is
    always re-projected with current pose).
  • Gimbal keeps pointing at the last known position when the ArUco code
    leaves the frame or is temporarily obscured (SEARCHING state).

Subscribed topics
─────────────────
  ~pose_topic       (outer)       geometry_msgs/PoseStamped  (outer ArUco in cam frame)
  ~inner_pose_topic (inner)       geometry_msgs/PoseStamped  (inner ArUco in cam frame)
  ~odom_topic   (param)          nav_msgs/Odometry          drone pose + velocity
                  Simulation default: /<ns>/ground_truth
                  GPS-denied production: /<ns>/odometry/odom_main
                  Both are in the same frame as the MRS goto local_origin frame —
                  use odom_main in production so FK and goto coordinates are consistent.
  /<ns>/gimbal/joint_states      sensor_msgs/JointState
  /<ns>/overhead_cam/camera_info sensor_msgs/CameraInfo     (read once)

Published topics
────────────────
  /<ns>/gimbal/position/yaw/command    std_msgs/Float64
  /<ns>/gimbal/position/pitch/command  std_msgs/Float64
  ~usv_world_pose    geometry_msgs/PoseStamped  (world ENU)
  ~tracking_status   std_msgs/String   TRACKING / SEARCHING / LOST

Parameters (loaded from landing_params.yaml)
────────────────────────────────────────────
  ~ns, ~pose_topic, ~odom_topic, ~rate_hz
  ~kp_yaw/pitch, ~ki_yaw/pitch, ~kd_yaw/pitch
  ~max_integral, ~max_cmd_delta, ~error_deadband
  ~pos_filter_alpha, ~max_pos_jump_m, ~lost_timeout_s
  ~min_altitude_m, ~min_pitch_rad
  ~x/y/z_offset (gimbal base offset in body frame)
"""

import math
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, JointState
from std_msgs.msg import Float64, String
from geometry_msgs.msg import (PoseStamped, Point, Quaternion)


# ── Gimbal FK constants (must match gimbal_position_node.py exactly) ──────────
_HP    = math.pi / 2.0
_R_OPT = Rot.from_euler('z', -_HP) * Rot.from_euler('x', -_HP)

_YAW_OFFSET  = np.array([0.0,   0.0, -0.025])
_ROLL_OFFSET = np.array([0.0,   0.0, -0.030])
_PIT_OFFSET  = np.array([0.0,   0.0, -0.025])
_OPT_OFFSET  = np.array([0.025, 0.0,  0.0  ])

_YAW_LIM   = (-3.054,  3.054)
_PITCH_LIM = (-2.094,  2.094)


def _camera_fk(drone_pos, drone_rot, base_off, yaw, roll, pitch):
    """Forward kinematics: returns (p_opt, R_opt) in world frame."""
    p, R = drone_pos + drone_rot.apply(base_off), drone_rot
    p, R = p + R.apply(_YAW_OFFSET),   R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFFSET),  R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PIT_OFFSET),   R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFFSET),   R * _R_OPT
    return p, R


class GimbalArucoTrackerNode:

    def __init__(self):
        rospy.init_node('gimbal_aruco_tracker_node', anonymous=False)

        self._ns = rospy.get_param('~ns', 'uav1')
        ns = self._ns

        self._rate_hz       = rospy.get_param('~rate_hz',          30.0)
        self._pos_alpha     = rospy.get_param('~pos_filter_alpha',  0.4)
        self._max_pos_jump  = rospy.get_param('~max_pos_jump_m',    5.0)
        self._lost_timeout  = rospy.get_param('~lost_timeout_s',    3.0)
        self._min_altitude  = rospy.get_param('~min_altitude_m',    2.0)
        self._min_pitch     = rospy.get_param('~min_pitch_rad',     0.5)

        # PID gains
        self._kp_y  = rospy.get_param('~kp_yaw',        0.15)
        self._ki_y  = rospy.get_param('~ki_yaw',        0.02)
        self._kd_y  = rospy.get_param('~kd_yaw',        0.05)
        self._kp_p  = rospy.get_param('~kp_pitch',      0.15)
        self._ki_p  = rospy.get_param('~ki_pitch',      0.02)
        self._kd_p  = rospy.get_param('~kd_pitch',      0.05)
        self._max_int   = rospy.get_param('~max_integral',   0.3)
        self._max_delta = rospy.get_param('~max_cmd_delta',  0.04)
        self._deadband  = rospy.get_param('~error_deadband', 0.01)

        self._base_off = np.array([
            rospy.get_param('~x_offset', 0.10),
            rospy.get_param('~y_offset', 0.00),
            rospy.get_param('~z_offset', 0.00),
        ])

        # How long outer must be absent before falling back to inner marker [s]
        self._outer_switch_s = rospy.get_param('~outer_switch_timeout_s', 1.0)

        # Camera intrinsics (read once from CameraInfo)
        self._fx = self._fy = self._cx = self._cy = None
        self._cam_ready  = False

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._drone_pos  = np.zeros(3)
        self._drone_rot  = Rot.identity()
        self._odom_ready = False
        self._gimbal_angles = [0.0, 0.0, 0.0]   # [yaw, roll, pitch]
        self._joint_names = [
            f'{ns}_gimbal_yaw_joint',
            f'{ns}_gimbal_roll_joint',
            f'{ns}_gimbal_pitch_joint',
        ]

        # PID state
        self._cmd_yaw    = 0.0
        self._cmd_pitch  = 0.0
        self._cmd_synced = True   # True = mirror joint_states until first detection
        self._int_yaw    = 0.0
        self._int_pitch  = 0.0
        self._prev_e_yaw   = 0.0
        self._prev_e_pitch = 0.0

        # World position (EMA-smoothed)
        self._usv_world_pos   = None   # np.ndarray [x,y,z] or None
        self._last_det_t      = None   # seconds, most recent accepted detection
        self._last_outer_t    = None   # seconds, most recent outer detection
        self._last_inner_t    = None   # seconds, most recent inner detection
        self._active_marker   = None   # 'outer' | 'inner'
        self._last_ctrl_t     = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_yaw    = rospy.Publisher(
            f'/{ns}/gimbal/position/yaw/command',   Float64, queue_size=1)
        self._pub_pitch  = rospy.Publisher(
            f'/{ns}/gimbal/position/pitch/command', Float64, queue_size=1)
        self._pub_pose   = rospy.Publisher(
            'usv_world_pose',  PoseStamped, queue_size=5)
        self._pub_status = rospy.Publisher(
            'tracking_status', String,      queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────────
        pose_topic       = rospy.get_param('~pose_topic',       '/aruco_pose/outer/usv_in_cam')
        inner_pose_topic = rospy.get_param('~inner_pose_topic', '/aruco_pose/inner/usv_in_cam')
        # odom_topic: use ground_truth in simulation (identical to MRS local_origin frame).
        # For GPS-denied real operation set to /<ns>/odometry/odom_main so that the FK
        # world positions are consistent with the MRS goto local_origin frame.
        odom_topic = rospy.get_param('~odom_topic', f'/{ns}/ground_truth')

        rospy.Subscriber(pose_topic,       PoseStamped, self._outer_cb, queue_size=5)
        rospy.Subscriber(inner_pose_topic, PoseStamped, self._inner_cb, queue_size=5)

        rospy.Subscriber(f'/{ns}/overhead_cam/camera_info', CameraInfo,
                         self._info_cb,   queue_size=1)
        rospy.Subscriber(odom_topic,                        Odometry,
                         self._odom_cb,   queue_size=1)
        rospy.Subscriber(f'/{ns}/gimbal/joint_states',      JointState,
                         self._js_cb,     queue_size=1)

        # ── Timers ────────────────────────────────────────────────────────────
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._control_cb)
        rospy.Timer(rospy.Duration(0.5),                  self._status_cb)

        rospy.loginfo(
            f'[gimbal_aruco_tracker] Init — ns={ns}  pose={pose_topic}  '
            f'rate={self._rate_hz:.0f} Hz  '
            f'Kp(y/p)={self._kp_y}/{self._kp_p}')

    # ── Sensor callbacks ──────────────────────────────────────────────────────

    def _info_cb(self, msg: CameraInfo):
        if self._cam_ready:
            return
        K = msg.K
        self._fx, self._fy = float(K[0]), float(K[4])
        self._cx, self._cy = float(K[2]), float(K[5])
        self._cam_ready = True
        rospy.loginfo(
            f'[gimbal_aruco_tracker] Camera: '
            f'fx={self._fx:.1f} cx={self._cx:.1f} cy={self._cy:.1f}')

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        with self._lock:
            self._drone_pos  = np.array([p.x, p.y, p.z])
            self._drone_rot  = Rot.from_quat([q.x, q.y, q.z, q.w])
            self._odom_ready = True

    def _js_cb(self, msg: JointState):
        lut    = dict(zip(msg.name, msg.position))
        angles = [
            lut.get(self._joint_names[0], 0.0),
            lut.get(self._joint_names[1], 0.0),
            lut.get(self._joint_names[2], 0.0),
        ]
        with self._lock:
            self._gimbal_angles = angles
            if self._cmd_synced:
                self._cmd_yaw   = angles[0]
                self._cmd_pitch = angles[2]

    # ── ArUco pose callbacks ──────────────────────────────────────────────────

    def _outer_cb(self, msg: PoseStamped):
        """Outer marker detected (large, 2 m).  Always accepted when valid."""
        now = rospy.Time.now().to_sec()
        with self._lock:
            self._last_outer_t = now
        self._accept_detection(msg, 'outer')

    def _inner_cb(self, msg: PoseStamped):
        """Inner marker detected (small, 0.2 m).
        Only used when the outer marker has been absent for outer_switch_timeout_s.
        This handles the close-approach phase where the large marker leaves the frame.
        """
        now = rospy.Time.now().to_sec()
        with self._lock:
            last_outer = self._last_outer_t
        outer_age = (now - last_outer) if last_outer is not None else 999.0
        if outer_age < self._outer_switch_s:
            return   # outer is fresh — skip inner to avoid noise
        self._accept_detection(msg, 'inner')

    def _accept_detection(self, msg: PoseStamped, marker: str):
        """Convert ArUco pose in camera optical frame to world frame,
        apply EMA filter, store for the control timer to project."""
        if not (self._odom_ready and self._cam_ready):
            return

        with self._lock:
            drone_z       = self._drone_pos[2]
            pitch_ang     = self._gimbal_angles[2]
            pos           = self._drone_pos.copy()
            rot           = Rot.from_quat(self._drone_rot.as_quat())
            angles        = list(self._gimbal_angles)
            prev_pos      = self._usv_world_pos
            prev_marker   = self._active_marker

        # Altitude and gimbal-pitch gates
        if drone_z < self._min_altitude:
            rospy.logwarn_throttle(
                5.0, f'[gimbal_aruco_tracker] Gate: alt {drone_z:.1f} m < {self._min_altitude:.1f} m')
            return
        if pitch_ang < self._min_pitch:
            rospy.logwarn_throttle(
                5.0, f'[gimbal_aruco_tracker] Gate: pitch {pitch_ang:.2f} < {self._min_pitch:.2f} rad')
            return

        # ArUco position in camera optical frame [m]
        p = msg.pose.position
        tvec_cam = np.array([p.x, p.y, p.z])
        if tvec_cam[2] < 0.01:
            return   # behind camera — invalid

        # Convert to world frame: ar_world = p_opt + R_opt @ tvec_cam
        p_opt, R_opt = _camera_fk(pos, rot, self._base_off, *angles)
        ar_world = p_opt + R_opt.apply(tvec_cam)

        # Sanity: ArUco panel is ~0.75 m above water
        if ar_world[2] > 5.0 or ar_world[2] < -2.0:
            rospy.logwarn_throttle(
                3.0, f'[gimbal_aruco_tracker] Implausible ar_world z={ar_world[2]:.1f} — rejected')
            return

        # On marker switch, reset EMA to avoid blending old outer with new inner
        if prev_marker is not None and marker != prev_marker:
            rospy.loginfo(
                f'[gimbal_aruco_tracker] Switching marker: {prev_marker} → {marker}')
            prev_pos = None

        # EMA filter + jump rejection
        if prev_pos is not None:
            jump = np.linalg.norm(ar_world - prev_pos)
            if jump > self._max_pos_jump:
                rospy.logwarn_throttle(
                    3.0, f'[gimbal_aruco_tracker] Jump {jump:.1f} m — rejected')
                return
            ar_filt = self._pos_alpha * ar_world + (1.0 - self._pos_alpha) * prev_pos
        else:
            ar_filt = ar_world.copy()

        now = rospy.Time.now().to_sec()
        with self._lock:
            self._usv_world_pos  = ar_filt
            self._last_det_t     = now
            self._active_marker  = marker
            self._cmd_synced     = False
            if marker == 'inner':
                self._last_inner_t = now

        # Publish world pose
        ps = PoseStamped()
        ps.header.stamp    = msg.header.stamp
        ps.header.frame_id = 'world'
        ps.pose.position   = Point(x=float(ar_filt[0]),
                                   y=float(ar_filt[1]),
                                   z=float(ar_filt[2]))
        ps.pose.orientation = Quaternion(w=1.0)
        self._pub_pose.publish(ps)

        horiz = np.linalg.norm(ar_filt[:2] - pos[:2])
        rospy.logdebug(
            f'[gimbal_aruco_tracker] [{marker}] deck=({ar_filt[0]:.1f},{ar_filt[1]:.1f},{ar_filt[2]:.2f}) '
            f'horiz={horiz:.1f} m')

    # ── Control timer: project world pos → synthetic pixel errors → PID ───────

    def _control_cb(self, _event):
        if not (self._cam_ready and self._odom_ready):
            return

        with self._lock:
            drone_z   = self._drone_pos[2]
            pitch_ang = self._gimbal_angles[2]
            usv_pos   = self._usv_world_pos
            synced    = self._cmd_synced
            pos       = self._drone_pos.copy()
            rot       = Rot.from_quat(self._drone_rot.as_quat())
            angles    = list(self._gimbal_angles)
            cmd_yaw   = self._cmd_yaw
            cmd_pitch = self._cmd_pitch
            int_yaw   = self._int_yaw
            int_pitch = self._int_pitch
            prev_ey   = self._prev_e_yaw
            prev_ep   = self._prev_e_pitch
            last_t    = self._last_ctrl_t

        if usv_pos is None or synced:
            return
        if drone_z < self._min_altitude or pitch_ang < self._min_pitch:
            return

        now = rospy.Time.now().to_sec()
        dt  = (now - last_t) if (last_t is not None and 0 < now - last_t < 0.5) \
              else (1.0 / self._rate_hz)

        # Project deck world pos into current camera frame
        p_opt, R_opt = _camera_fk(pos, rot, self._base_off, *angles)
        v_world = usv_pos - p_opt
        v_opt   = R_opt.inv().apply(v_world)   # in camera optical frame

        if v_opt[2] <= 0.05:
            return   # deck behind camera

        # Synthetic angular errors (same as normalised pixel coordinates)
        e_yaw   = v_opt[0] / v_opt[2]   # +: deck right of centre
        e_pitch = v_opt[1] / v_opt[2]   # +: deck below centre

        # Deadband
        e_yaw   = 0.0 if abs(e_yaw)   < self._deadband else e_yaw
        e_pitch = 0.0 if abs(e_pitch) < self._deadband else e_pitch

        # PID
        int_yaw   = float(np.clip(int_yaw   + e_yaw   * dt, -self._max_int, self._max_int))
        int_pitch = float(np.clip(int_pitch + e_pitch * dt, -self._max_int, self._max_int))

        d_yaw   = (e_yaw   - prev_ey) / dt
        d_pitch = (e_pitch - prev_ep) / dt

        delta_yaw   = -(self._kp_y * e_yaw   + self._ki_y * int_yaw   + self._kd_y * d_yaw)
        delta_pitch = +(self._kp_p * e_pitch + self._ki_p * int_pitch + self._kd_p * d_pitch)

        delta_yaw   = float(np.clip(delta_yaw,   -self._max_delta, self._max_delta))
        delta_pitch = float(np.clip(delta_pitch, -self._max_delta, self._max_delta))

        cmd_yaw   = float(np.clip(cmd_yaw   + delta_yaw,   *_YAW_LIM))
        cmd_pitch = float(np.clip(cmd_pitch + delta_pitch, *_PITCH_LIM))

        self._pub_yaw.publish(Float64(data=cmd_yaw))
        self._pub_pitch.publish(Float64(data=cmd_pitch))

        with self._lock:
            self._cmd_yaw        = cmd_yaw
            self._cmd_pitch      = cmd_pitch
            self._int_yaw        = int_yaw
            self._int_pitch      = int_pitch
            self._prev_e_yaw     = e_yaw
            self._prev_e_pitch   = e_pitch
            self._last_ctrl_t    = now

    # ── Status heartbeat ──────────────────────────────────────────────────────

    def _status_cb(self, _event):
        with self._lock:
            last_t = self._last_det_t
            marker = self._active_marker

        now = rospy.Time.now().to_sec()
        if last_t is None:
            status = 'LOST'
        elif (now - last_t) > self._lost_timeout:
            status = 'LOST'
        elif (now - last_t) > 0.5:
            status = 'SEARCHING'
        else:
            status = 'TRACKING'

        self._pub_status.publish(String(data=status))
        rospy.loginfo_throttle(
            5.0, f'[gimbal_aruco_tracker] Status: {status}  marker={marker or "none"}')

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        GimbalArucoTrackerNode().run()
    except rospy.ROSInterruptException:
        pass
