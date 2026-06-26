#!/usr/bin/env python3
"""
usv_uwb_sim_node.py  (tmp / dev-only)
──────────────────────────────────────
Simulated UWB sensor for the UAV↔USV system, to prepare the PD+FF-vs-MPC ablation
(see tmp/PLAN_uwb_mpc.md).  Emulates a real UWB+USV-IMU package that would directly
measure the UAV's pose in the USV frame; here it is synthesised from Gazebo
ground truth and corrupted to the spec'd accuracy.

Spec emulated:
  • UAV pose w.r.t. the USV (body) frame, accuracy ~10 cm
  • USV IMU (specific force, angular rate, deck attitude)
  • 10 Hz
  • only available within 5 m relative range ("starting at 5 m distance")

Ground truth used to SYNTHESISE the measurement (a real UWB measures it directly):
  /<ns>/ground_truth   (nav_msgs/Odometry)  — UAV world pose
  /p3d                 (nav_msgs/Odometry)  — USV world pose

Outputs:
  /<ns>/uwb/uav_in_usv  geometry_msgs/PoseWithCovarianceStamped
        UAV pose in the USV body frame (frame_id = '<usv>_base'); covariance diag
        = accuracy².  Published only while relative range < max_range.
  /<ns>/uwb/usv_imu     sensor_msgs/Imu
        USV IMU: linear_acceleration = specific force in deck frame (a−g),
        angular_velocity = deck rate, orientation = deck attitude.
  /<ns>/uwb/in_range    std_msgs/Bool   (UWB fix available this tick)

Run:  python3 tmp/usv_uwb_sim_node.py _ns:=uav1
"""
import math
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool

_G_WORLD = np.array([0.0, 0.0, -9.80665])   # gravity vector (world)


class UwbSim:
    def __init__(self):
        rospy.init_node('usv_uwb_sim')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns
        usv_topic = rospy.get_param('~usv_gt_topic', '/p3d')
        uav_topic = rospy.get_param('~uav_gt_topic', '/%s/ground_truth' % ns)

        self._rate = float(rospy.get_param('~rate_hz', 10.0))
        # UWB working space = CYLINDER (not a 3-D-range sphere): valid when the UAV is
        # below alt_max AND within horiz_max of the USV.  A sphere gate (range<5) failed
        # the high-speed descents — at 4 m altitude with a 4 m horizontal offset the 3-D
        # range is 5.6 m > 5 m, so UWB never engaged exactly where the AR cone clips out.
        self._max_range = float(rospy.get_param('~max_range_m', 5.0))      # legacy sphere (unused if cyl)
        self._cyl_gate  = bool(rospy.get_param('~cylinder_gate', True))
        self._alt_max   = float(rospy.get_param('~alt_max_m', 5.0))        # vertical gate
        self._horiz_max = float(rospy.get_param('~horiz_max_m', 7.0))      # radial gate
        # ~10 cm accuracy: split over 3 axes so the 3-D RMS error ≈ accuracy
        acc = float(rospy.get_param('~accuracy_m', 0.10))
        self._pos_std = acc / math.sqrt(3.0)
        self._yaw_std = float(rospy.get_param('~yaw_std_rad', 0.02))
        self._acc_std = float(rospy.get_param('~accel_noise', 0.10))   # m/s²
        self._gyr_std = float(rospy.get_param('~gyro_noise', 0.01))    # rad/s
        self._vel_lpf = float(rospy.get_param('~vel_lpf', 0.4))
        seed = int(rospy.get_param('~seed', 0))
        self._rng = np.random.default_rng(seed if seed else None)
        usv_name = str(rospy.get_param('~usv_frame', 'otter')).strip('/')
        self._usv_frame = '%s_base' % usv_name

        self._lock = threading.Lock()
        self._uav = None      # (p[3], R)
        self._usv = None      # (p[3], R, quat[4])
        self._usv_v = np.zeros(3)   # USV world velocity (filtered)
        self._usv_a = np.zeros(3)   # USV world accel (filtered)
        self._prev_p = None
        self._prev_v = None
        self._prev_t = None
        self._usv_w = np.zeros(3)   # USV angular rate (body)

        self.pub_pose = rospy.Publisher('/%s/uwb/uav_in_usv' % ns,
                                        PoseWithCovarianceStamped, queue_size=5)
        self.pub_imu = rospy.Publisher('/%s/uwb/usv_imu' % ns, Imu, queue_size=5)
        self.pub_rng = rospy.Publisher('/%s/uwb/in_range' % ns, Bool, queue_size=5)
        rospy.Subscriber(uav_topic, Odometry, self._cb_uav, queue_size=5)
        rospy.Subscriber(usv_topic, Odometry, self._cb_usv, queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)
        rospy.loginfo('[uwb_sim] up — ns=%s  acc=%.2f m  rate=%.0f Hz  range<%.1f m',
                      ns, acc, self._rate, self._max_range)

    def _cb_uav(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        with self._lock:
            self._uav = (np.array([p.x, p.y, p.z]),
                         Rot.from_quat([q.x, q.y, q.z, q.w]))

    def _cb_usv(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        R = Rot.from_quat([q.x, q.y, q.z, q.w])
        now = rospy.Time.now().to_sec()
        w = np.array([m.twist.twist.angular.x, m.twist.twist.angular.y,
                      m.twist.twist.angular.z])
        with self._lock:
            # finite-difference + LPF world velocity / acceleration of the USV
            if self._prev_p is not None and self._prev_t is not None:
                dt = now - self._prev_t
                if dt > 1e-3:
                    v = (pos - self._prev_p) / dt
                    self._usv_v = (1 - self._vel_lpf) * self._usv_v + self._vel_lpf * v
                    if self._prev_v is not None:
                        a = (self._usv_v - self._prev_v) / dt
                        self._usv_a = (1 - self._vel_lpf) * self._usv_a + self._vel_lpf * a
                    self._prev_v = self._usv_v.copy()
            self._prev_p, self._prev_t = pos, now
            self._usv = (pos, R, np.array([q.x, q.y, q.z, q.w]))
            self._usv_w = w

    def _tick(self, _e):
        with self._lock:
            uav, usv = self._uav, self._usv
            a_world, w_body, quat = self._usv_a.copy(), self._usv_w.copy(), None
            R_usv = None
            if usv is not None:
                R_usv, quat = usv[1], usv[2]
        if uav is None or usv is None or R_usv is None:
            return
        p_uav, R_uav = uav
        p_usv = usv[0]
        rng = float(np.linalg.norm(p_uav - p_usv))
        # cylinder gate: vertical (alt) AND radial (horiz) — see __init__ note
        d_alt = float(p_uav[2] - p_usv[2])
        d_horiz = float(np.linalg.norm((p_uav - p_usv)[:2]))
        if self._cyl_gate:
            in_range = (d_alt < self._alt_max) and (d_horiz < self._horiz_max)
        else:
            in_range = rng < self._max_range
        self.pub_rng.publish(Bool(data=in_range))

        stamp = rospy.Time.now()
        # ── USV IMU (always; a real USV-IMU streams regardless of UWB range) ──
        f_body = R_usv.inv().apply(a_world - _G_WORLD)            # specific force
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self._usv_frame
        imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w = \
            [float(v) for v in quat]
        nf = self._rng.normal(0, self._acc_std, 3)
        imu.linear_acceleration.x = float(f_body[0] + nf[0])
        imu.linear_acceleration.y = float(f_body[1] + nf[1])
        imu.linear_acceleration.z = float(f_body[2] + nf[2])
        ng = self._rng.normal(0, self._gyr_std, 3)
        imu.angular_velocity.x = float(w_body[0] + ng[0])
        imu.angular_velocity.y = float(w_body[1] + ng[1])
        imu.angular_velocity.z = float(w_body[2] + ng[2])
        imu.orientation_covariance[0] = (2 * self._yaw_std) ** 2
        imu.linear_acceleration_covariance[0] = self._acc_std ** 2
        imu.angular_velocity_covariance[0] = self._gyr_std ** 2
        self.pub_imu.publish(imu)

        if not in_range:
            return
        # ── UWB relative pose: UAV pose in the USV body frame, + 10 cm noise ──
        r = R_usv.inv().apply(p_uav - p_usv)                      # rel position
        R_rel = R_usv.inv() * R_uav                              # rel orientation
        r_n = r + self._rng.normal(0, self._pos_std, 3)
        q_rel = (R_rel * Rot.from_euler('z', self._rng.normal(0, self._yaw_std))).as_quat()
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self._usv_frame
        msg.pose.pose.position.x = float(r_n[0])
        msg.pose.pose.position.y = float(r_n[1])
        msg.pose.pose.position.z = float(r_n[2])
        msg.pose.pose.orientation.x, msg.pose.pose.orientation.y = float(q_rel[0]), float(q_rel[1])
        msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = float(q_rel[2]), float(q_rel[3])
        cov = [0.0] * 36
        cov[0] = cov[7] = cov[14] = self._pos_std ** 2
        cov[35] = self._yaw_std ** 2
        msg.pose.covariance = cov
        self.pub_pose.publish(msg)


if __name__ == '__main__':
    try:
        UwbSim()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
