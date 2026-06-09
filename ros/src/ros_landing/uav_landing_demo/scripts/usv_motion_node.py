#!/usr/bin/env python3
"""
usv_motion_node.py — USV thruster-based straight-line motion + wave disturbance.

Forward motion
──────────────
A PI speed controller reads body-frame surge velocity from ground_truth_odometry
and drives left_thrust_cmd / right_thrust_cmd (std_msgs/Float32, range [-1, 1])
to maintain the target forward speed.  A P heading controller adds differential
thrust to hold the desired yaw.

The Otter's thrust plugin (libusv_gazebo_thrust_plugin.so) maps cmd ∈ [-1, 1] to
a body force via a configurable linear / GLF / square function; here mappingType=0
(linear), maxForceFwd = 50 N, maxForceRev = -50 N per thruster.

Thrust allocation matrix T (body surge, sway, yaw) × (left, right):
  T = [[50, 50],          surge: both thrusters positive
       [ 0,  0],          sway: ignored (not actuated)
       [-L,  L]]          yaw: differential, L = 0.39 × 50 = 19.5 N·m/cmd

  u_thrust = T_pinv @ [tau_surge, 0, tau_yaw]

Sea-state disturbances
──────────────────────
Sinusoidal external wrenches are applied to otter::base_link via the
/gazebo/apply_body_wrench service at publish_rate Hz.

  F_heave(t) = heave_force_amp  × sin(2π·f_h·t + φ_h)   [N,  world +Z]
  τ_roll(t)  = roll_torque_amp  × sin(2π·f_r·t + φ_r)   [N·m, body +X ≈ world +X]
  τ_pitch(t) = pitch_torque_amp × sin(2π·f_p·t + φ_p)   [N·m, body +Y ≈ world +Y]

Force amplitude guide (tune to achieve desired deck motion amplitude):
  Otter mass ≈ 29 kg, added mass ~5–8 kg.
  ±10 cm heave @ 0.40 Hz → ω² = 6.3 → required F ≈ (m+ma)·A·ω² ≈ 20 N minimum;
  with hydrodynamic damping set heave_force_amp ≈ 100 N then reduce until
  peak heave ≈ 0.10 m.  Use /uav_otter_landing/usv_odom z to monitor.

Debug topics (when publish_debug=true)
──────────────────────────────────────
  /uav_otter_landing/usv_odom   nav_msgs/Odometry  (from ground_truth, re-stamped)
  /uav_otter_landing/thrust_cmd geometry_msgs/Vector3Stamped  (left, right, surge_err)
"""

import math
import numpy as np
import rospy

from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3Stamped, Wrench, Point, Vector3
from gazebo_msgs.srv import ApplyBodyWrench, ApplyBodyWrenchRequest
from scipy.spatial.transform import Rotation as Rot


# ── Thrust allocation (pre-computed pseudoinverse) ────────────────────────────
# T maps thruster commands (left, right) in normalised units to body wrench.
# Columns: [left_thruster, right_thruster]
# Rows:    [surge_force_N, sway_force_N, yaw_moment_Nm]
_F_MAX  = 50.0   # N per thruster at cmd=1
_L_ARM  = 0.39   # m — half thruster separation
_T = np.array([
    [_F_MAX,  _F_MAX],
    [0.0,     0.0   ],
    [-_L_ARM*_F_MAX, _L_ARM*_F_MAX],
])
_T_PINV = np.linalg.pinv(_T)   # (2 × 3) pseudoinverse, computed once at import


def _thrust_alloc(tau_surge: float, tau_yaw: float) -> tuple[float, float]:
    """Map (surge force N, yaw moment N·m) → (left_cmd, right_cmd) ∈ [-1, 1].

    T already embeds F_MAX, so T_pinv @ tau returns normalised commands directly.
    """
    tau = np.array([tau_surge, 0.0, tau_yaw])
    u   = _T_PINV @ tau   # already in [-1, 1] range
    return float(np.clip(u[0], -1.0, 1.0)), float(np.clip(u[1], -1.0, 1.0))


class UsvMotionNode:

    def __init__(self):
        rospy.init_node('usv_motion_node', anonymous=False)

        # ── Target motion ─────────────────────────────────────────────────────
        self._u_d   = rospy.get_param('~usv_speed',    1.0)   # m/s
        self._psi_d = rospy.get_param('~usv_heading',  0.0)   # rad (0 = +X)

        # ── Speed PI gains ────────────────────────────────────────────────────
        self._Kp_u   = rospy.get_param('~Kp_speed',   70.0)
        self._Ki_u   = rospy.get_param('~Ki_speed',    5.0)
        self._Kp_psi = rospy.get_param('~Kp_heading', 20.0)
        self._Kd_psi = rospy.get_param('~Kd_heading',  1.0)

        # ── Sea-state forcing ─────────────────────────────────────────────────
        self._heave_force_amp  = rospy.get_param('~heave_force_amp',   100.0)  # N
        self._heave_hz         = rospy.get_param('~heave_hz',            0.40)
        self._heave_phase      = rospy.get_param('~heave_phase',         0.0)

        self._roll_torque_amp  = rospy.get_param('~roll_torque_amp',    30.0)  # N·m
        self._roll_hz          = rospy.get_param('~roll_hz',             0.30)
        self._roll_phase       = rospy.get_param('~roll_phase',          0.7)

        self._pitch_torque_amp = rospy.get_param('~pitch_torque_amp',   60.0)  # N·m
        self._pitch_hz         = rospy.get_param('~pitch_hz',            0.50)
        self._pitch_phase      = rospy.get_param('~pitch_phase',         1.3)

        # Name of the Gazebo body to apply the wrench on
        self._body_name = rospy.get_param('~gazebo_body_name', 'otter::base_link')

        # Delay before the USV starts moving (sea-state still runs during delay).
        # Useful in demo_sequence.launch: spawn USV near the drone, wait for the
        # drone to arm + climb before releasing the USV into the camera frame.
        self._delay_s  = rospy.get_param('~usv_delay_s', 0.0)

        rate_hz        = rospy.get_param('~publish_rate',    50)
        self._debug    = rospy.get_param('~publish_debug',  True)

        self._dt    = 1.0 / rate_hz
        self._rate  = rospy.Rate(rate_hz)

        # ── State ─────────────────────────────────────────────────────────────
        self._u         = 0.0   # measured body-frame surge speed [m/s]
        self._psi       = 0.0   # measured yaw [rad]
        self._speed_int = 0.0   # integrator for speed PI
        self._e_psi_prev = 0.0  # previous heading error for derivative
        self._gt_ready  = False

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_left  = rospy.Publisher('left_thrust_cmd',  Float32, queue_size=1)
        self._pub_right = rospy.Publisher('right_thrust_cmd', Float32, queue_size=1)

        if self._debug:
            self._pub_odom   = rospy.Publisher(
                '/uav_otter_landing/usv_odom',   Odometry,        queue_size=1)
            self._pub_thrust = rospy.Publisher(
                '/uav_otter_landing/thrust_cmd', Vector3Stamped,  queue_size=1)

        # ── Subscriber ───────────────────────────────────────────────────────
        # ground_truth_enabled:=true in the URDF spawns the p3d plugin with
        # topicName="p3d" (not "ground_truth_odometry").
        rospy.Subscriber('p3d', Odometry, self._gt_cb, queue_size=5)

        # ── Wait for Gazebo wrench service ────────────────────────────────────
        rospy.loginfo('[usv_motion] Waiting for /gazebo/apply_body_wrench …')
        rospy.wait_for_service('/gazebo/apply_body_wrench', timeout=30.0)
        self._wrench_srv = rospy.ServiceProxy(
            '/gazebo/apply_body_wrench', ApplyBodyWrench, persistent=True)

        rospy.loginfo('[usv_motion] Waiting for p3d odometry …')
        while not self._gt_ready and not rospy.is_shutdown():
            rospy.sleep(0.1)

        if self._delay_s > 0.0:
            rospy.loginfo(
                f'[usv_motion] Holding position for {self._delay_s:.0f} s '
                f'(sea-state active, thrust idle) …')
            t_delay_start = rospy.Time.now().to_sec()
            rate_delay = rospy.Rate(rate_hz)
            while not rospy.is_shutdown():
                elapsed = rospy.Time.now().to_sec() - t_delay_start
                if elapsed >= self._delay_s:
                    break
                self._apply_sea_state(elapsed)
                rate_delay.sleep()

        self._t_start = rospy.Time.now().to_sec()
        rospy.loginfo(
            f'[usv_motion] Ready — target speed={self._u_d} m/s  '
            f'heading={math.degrees(self._psi_d):.1f}°')

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _gt_cb(self, msg: Odometry) -> None:
        """Extract body-frame surge speed and yaw from ground truth."""
        vx_w = msg.twist.twist.linear.x
        vy_w = msg.twist.twist.linear.y
        q = msg.pose.pose.orientation
        # Convert quaternion to yaw
        self._psi = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        # Project world velocity onto body surge axis
        self._u = vx_w * math.cos(self._psi) + vy_w * math.sin(self._psi)
        self._gt_ready = True

        if self._debug:
            self._pub_odom.publish(msg)

    # ── Control ──────────────────────────────────────────────────────────────

    def _speed_pi(self) -> float:
        """PI controller: returns desired surge force [N]."""
        e_u = self._u_d - self._u
        self._speed_int += e_u * self._dt
        self._speed_int  = float(np.clip(self._speed_int, -50.0, 50.0))
        return self._Kp_u * e_u + self._Ki_u * self._speed_int

    def _heading_pd(self) -> float:
        """PD controller: returns desired yaw moment [N·m]."""
        e_psi = self._psi_d - self._psi
        e_psi = (e_psi + math.pi) % (2 * math.pi) - math.pi
        de_psi = (e_psi - self._e_psi_prev) / self._dt
        self._e_psi_prev = e_psi
        return self._Kp_psi * e_psi + self._Kd_psi * de_psi

    # ── Wave disturbance ─────────────────────────────────────────────────────

    def _apply_sea_state(self, t: float) -> None:
        """Apply sinusoidal heave force + roll/pitch torques to the USV body."""
        fz   = self._heave_force_amp  * math.sin(
            2 * math.pi * self._heave_hz  * t + self._heave_phase)
        tx   = self._roll_torque_amp  * math.sin(
            2 * math.pi * self._roll_hz   * t + self._roll_phase)
        ty   = self._pitch_torque_amp * math.sin(
            2 * math.pi * self._pitch_hz  * t + self._pitch_phase)

        req = ApplyBodyWrenchRequest()
        req.body_name       = self._body_name
        req.reference_frame = 'world'
        req.reference_point = Point(0.0, 0.0, 0.0)
        req.wrench = Wrench(
            force  = Vector3(0.0, 0.0, fz),
            torque = Vector3(tx,  ty,  0.0),
        )
        req.start_time = rospy.Time(0)
        # Duration covers one control step so the force is applied continuously
        req.duration   = rospy.Duration(self._dt * 1.5)

        try:
            self._wrench_srv(req)
        except rospy.ServiceException as e:
            rospy.logwarn_throttle(5.0, f'[usv_motion] apply_body_wrench failed: {e}')

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        left_msg  = Float32()
        right_msg = Float32()

        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self._t_start

            # Thruster control
            tau_surge = self._speed_pi()
            tau_yaw   = self._heading_pd()
            left_cmd, right_cmd = _thrust_alloc(tau_surge, tau_yaw)

            left_msg.data  = left_cmd
            right_msg.data = right_cmd
            self._pub_left.publish(left_msg)
            self._pub_right.publish(right_msg)

            # Sea-state disturbance
            self._apply_sea_state(t)

            if self._debug:
                vs = Vector3Stamped()
                vs.header.stamp    = rospy.Time.now()
                vs.header.frame_id = 'base_link'
                vs.vector.x = left_cmd
                vs.vector.y = right_cmd
                vs.vector.z = self._u_d - self._u   # speed error
                self._pub_thrust.publish(vs)

            self._rate.sleep()


def main():
    node = UsvMotionNode()
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
