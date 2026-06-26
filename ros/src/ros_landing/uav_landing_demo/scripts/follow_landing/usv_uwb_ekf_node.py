#!/usr/bin/env python3
"""
usv_uwb_ekf_node.py  (tmp / dev-only)
──────────────────────────────────────
UWB + IMU fusion → smooth UAV-relative-to-USV state, the shared front-end for the
PD+FF vs MPC ablation (tmp/PLAN_uwb_mpc.md).  Turns the 10 cm / 10 Hz UWB position
into a smooth relative POSITION + VELOCITY (the damping signal the GPS-free vision
baseline lacked), and passes the USV acceleration through as the controller feed-forward.

Filter: 6-state constant-velocity Kalman filter, state x = [r, ṙ] in the USV body
(relative) frame.  IMUs drive the prediction as a known relative-acceleration input;
UWB position is the measurement.  (Straight/level USV ⇒ USV frame ≈ inertial; Coriolis
from USV yaw-rate is neglected — see PLAN §3.)

Relative-accel input (gravity cancels exactly):
    a_rel_world = R_uav·f_uav − R_usv·f_usv        (f = specific force; g drops out)
    a_rel_body  = R_usvᵀ·R_uav·f_uav − f_usv       (expressed in the state frame)
Needs BOTH IMUs (else the gravity terms don't cancel) → degrade to pure CV if either
is missing.  USV feed-forward accel (USV-IMU alone): a_usv_ff = f_usv + R_usvᵀ·g.

Inputs:
  ~uwb_pose_topic   /uav1/uwb/uav_in_usv   PoseWithCovarianceStamped  (measurement)
  ~usv_imu_topic    /uav1/uwb/usv_imu      sensor_msgs/Imu            (USV f, attitude)
  ~uav_imu_topic    /uav1/mavros/imu/data  sensor_msgs/Imu            (UAV f, attitude)
  ~in_range_topic   /uav1/uwb/in_range     std_msgs/Bool

Outputs:
  /uav1/uwb_ekf/rel_odom      nav_msgs/Odometry       pose=r, twist=ṙ (USV frame)
  /uav1/uwb_ekf/usv_accel_ff  geometry_msgs/Vector3Stamped   a_usv (relative frame)

Run:  python3 tmp/usv_uwb_ekf_node.py _ns:=uav1
"""
import threading
import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from std_msgs.msg import Bool

_G = np.array([0.0, 0.0, -9.80665])
_I3 = np.eye(3)


def _quat_R(q):
    return Rot.from_quat([q.x, q.y, q.z, q.w])


def _yaw(R):
    """Yaw (heading) of a rotation, in rad."""
    return float(R.as_euler('zyx')[0])


def _to_stab(v_body, R_usv):
    """Lift a vector from the USV BODY frame into the STABILIZED frame — gravity-up,
    aligned with the USV heading (tail→head), immune to deck roll/pitch.

        v_stab = R_z(ψ)⁻¹ · R_usv · v_body          (ψ = USV yaw)

    R_usv·v_body reconstructs the world-frame relative vector; R_z(ψ)⁻¹ then keeps only
    the heading rotation, stripping roll/pitch.  Without this, a still UAV appears to
    swing by ≈ h·sin(tilt) as waves roll/pitch the deck (≈0.9 m at h=5 m, tilt=10°)."""
    if R_usv is None:
        return v_body
    return Rot.from_euler('z', -_yaw(R_usv)).apply(R_usv.apply(v_body))


class UwbEkf:
    def __init__(self):
        rospy.init_node('usv_uwb_ekf')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        uwb_t = rospy.get_param('~uwb_pose_topic', '/%s/uwb/uav_in_usv' % ns)
        usvi_t = rospy.get_param('~usv_imu_topic', '/%s/uwb/usv_imu' % ns)
        uavi_t = rospy.get_param('~uav_imu_topic', '/%s/mavros/imu/data' % ns)
        rng_t = rospy.get_param('~in_range_topic', '/%s/uwb/in_range' % ns)

        self._rate = float(rospy.get_param('~rate_hz', 50.0))
        self._sig_a = float(rospy.get_param('~sigma_a', 2.0))      # process accel std
        self._use_uav = bool(rospy.get_param('~use_uav_imu', True))
        self._uwb_timeout = float(rospy.get_param('~uwb_timeout_s', 0.5))
        self._r0var = float(rospy.get_param('~init_pos_var', 0.25))
        self._v0var = float(rospy.get_param('~init_vel_var', 1.0))
        usv_name = str(rospy.get_param('~usv_frame', 'otter')).strip('/')
        self._frame = '%s_base' % usv_name

        self._lock = threading.Lock()
        self._x = None                       # 6-vec state, None until first UWB
        self._P = np.eye(6)
        self._q_rel = np.array([0., 0., 0., 1.])     # passthrough rel orientation
        self._z = None; self._zR = None; self._z_new = False; self._last_uwb = 0.0
        self._z_Rusv = None        # USV attitude at the UWB measurement instant
        self._f_usv = None; self._R_usv = None
        self._f_uav = None; self._R_uav = None
        self._in_range = False
        self._t_prev = None

        self.pub_odom = rospy.Publisher('/%s/uwb_ekf/rel_odom' % ns, Odometry, queue_size=10)
        self.pub_ff = rospy.Publisher('/%s/uwb_ekf/usv_accel_ff' % ns,
                                      Vector3Stamped, queue_size=10)
        rospy.Subscriber(uwb_t, PoseWithCovarianceStamped, self._cb_uwb, queue_size=10)
        rospy.Subscriber(usvi_t, Imu, self._cb_usv_imu, queue_size=10)
        rospy.Subscriber(uavi_t, Imu, self._cb_uav_imu, queue_size=10)
        rospy.Subscriber(rng_t, Bool, lambda m: setattr(self, '_in_range', m.data), queue_size=5)
        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)
        rospy.loginfo('[uwb_ekf] up — ns=%s  rate=%.0f Hz  sigma_a=%.1f  use_uav_imu=%s',
                      ns, self._rate, self._sig_a, self._use_uav)

    # ── callbacks ───────────────────────────────────────────────────────────
    def _cb_uwb(self, m):
        p = m.pose.pose.position
        c = m.pose.covariance
        with self._lock:
            self._z = np.array([p.x, p.y, p.z])
            r = np.array([[c[0], c[1], c[2]], [c[6], c[7], c[8]], [c[12], c[13], c[14]]])
            self._zR = r if np.trace(r) > 1e-9 else _I3 * (0.10 / np.sqrt(3.0)) ** 2
            q = m.pose.pose.orientation
            self._q_rel = np.array([q.x, q.y, q.z, q.w])
            self._z_Rusv = self._R_usv          # snapshot deck attitude at the fix
            self._z_new = True
            self._last_uwb = rospy.Time.now().to_sec()

    def _cb_usv_imu(self, m):
        with self._lock:
            self._f_usv = np.array([m.linear_acceleration.x, m.linear_acceleration.y,
                                    m.linear_acceleration.z])
            self._R_usv = _quat_R(m.orientation)

    def _cb_uav_imu(self, m):
        with self._lock:
            self._f_uav = np.array([m.linear_acceleration.x, m.linear_acceleration.y,
                                    m.linear_acceleration.z])
            self._R_uav = _quat_R(m.orientation)

    # ── filter step ─────────────────────────────────────────────────────────
    def _tick(self, _e):
        now = rospy.Time.now().to_sec()
        with self._lock:
            if self._t_prev is None:
                self._t_prev = now
                return
            dt = max(1e-3, now - self._t_prev)
            self._t_prev = now

            if self._x is None:                       # initialise from first UWB fix
                if self._z_new and self._z is not None:
                    z0 = _to_stab(self._z, self._z_Rusv)   # body → stabilized frame
                    self._x = np.hstack([z0, np.zeros(3)])
                    self._P = np.diag([self._r0var] * 3 + [self._v0var] * 3)
                    self._z_new = False
                return

            # relative-acceleration input in the STABILIZED frame (needs both IMUs for
            # gravity cancellation): a_rel_world = R_uav·f_uav − R_usv·f_usv, then strip
            # the deck roll/pitch by keeping only the USV-heading rotation.
            a_rel = np.zeros(3)
            if (self._use_uav and self._f_uav is not None and self._R_uav is not None
                    and self._f_usv is not None and self._R_usv is not None):
                a_rel_world = self._R_uav.apply(self._f_uav) - self._R_usv.apply(self._f_usv)
                a_rel = Rot.from_euler('z', -_yaw(self._R_usv)).apply(a_rel_world)

            # predict  x = F x + B a_rel ;  P = F P Fᵀ + Q
            F = np.eye(6); F[0:3, 3:6] = dt * _I3
            B = np.vstack([0.5 * dt * dt * _I3, dt * _I3])
            self._x = F @ self._x + B @ a_rel
            qa = self._sig_a ** 2
            Q = qa * np.block([[dt**4 / 4 * _I3, dt**3 / 2 * _I3],
                               [dt**3 / 2 * _I3, dt**2 * _I3]])
            self._P = F @ self._P @ F.T + Q

            # update on fresh, in-range UWB
            if self._z_new and self._in_range and (now - self._last_uwb) < self._uwb_timeout:
                z_stab = _to_stab(self._z, self._z_Rusv)   # IMU-corrected measurement
                H = np.zeros((3, 6)); H[0:3, 0:3] = _I3
                S = H @ self._P @ H.T + self._zR
                K = self._P @ H.T @ np.linalg.inv(S)
                self._x = self._x + K @ (z_stab - H @ self._x)
                self._P = (np.eye(6) - K @ H) @ self._P
            self._z_new = False

            x, P, qrel = self._x.copy(), self._P.copy(), self._q_rel.copy()
            # publish a clean YAW-ONLY relative orientation (ψ_uav − ψ_usv) so the
            # controller's yaw_rel is heading-aligned and not contaminated by deck tilt.
            if self._R_uav is not None and self._R_usv is not None:
                qrel = Rot.from_euler('z', _yaw(self._R_uav) - _yaw(self._R_usv)).as_quat()
            ff = None
            if self._f_usv is not None and self._R_usv is not None:
                # USV kinematic accel in the stabilized frame: a_world = R_usv·f_usv + g
                ff = Rot.from_euler('z', -_yaw(self._R_usv)).apply(
                    self._R_usv.apply(self._f_usv) + _G)

        # ── publish ──
        stamp = rospy.Time.now()
        od = Odometry(); od.header.stamp = stamp; od.header.frame_id = self._frame
        od.child_frame_id = 'uav'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = x[0:3]
        od.pose.pose.orientation.x, od.pose.pose.orientation.y, \
            od.pose.pose.orientation.z, od.pose.pose.orientation.w = qrel
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = x[3:6]
        pc = [0.0] * 36; tc = [0.0] * 36
        for i in range(3):
            pc[i * 6 + i] = P[i, i]; tc[i * 6 + i] = P[3 + i, 3 + i]
        od.pose.covariance = pc; od.twist.covariance = tc
        self.pub_odom.publish(od)
        if ff is not None:
            v = Vector3Stamped(); v.header.stamp = stamp; v.header.frame_id = self._frame
            v.vector.x, v.vector.y, v.vector.z = ff
            self.pub_ff.publish(v)


if __name__ == '__main__':
    try:
        UwbEkf()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
