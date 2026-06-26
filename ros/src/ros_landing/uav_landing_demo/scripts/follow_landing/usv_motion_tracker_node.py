#!/usr/bin/env python3
"""
usv_motion_tracker_node.py  (tmp / dev-only)
─────────────────────────────────────────────
Stage C of the GPS-free USV tracking pipeline — a CTRV (constant-turn-rate &
velocity) Unscented Kalman Filter that smooths the noisy ray-cast relative-pose
measurement, estimates the USV's relative velocity + heading + turn-rate, and
bridges short detection drop-outs by predicting forward.

This is the no-GPS analogue of `uav_packages/.../Global_pose_publisher.py`:
that node transformed the relative marker pose into a *global* frame using the
UAV's GPS odometry and ran the UKF there.  We have no GPS, so we run the very
same CTRV UKF directly on the **relative** offset published by
`usv_relpos_estimator_node.py` — i.e. in the UAV-centred, gravity-aligned,
IMU-yaw level frame.  The global-transform / odometry-correction blocks of the
reference are intentionally dropped.

State vector  x = [X, Y, v, theta, omega]   (relative to the UAV, level ENU)
  X, Y    USV position offset from the UAV   [m]
  v       speed of the tracked point         [m/s]   (relative; ~USV surge in FOLLOW)
  theta   heading / velocity direction       [rad]
  omega   turn-rate                          [rad/s]
NOTE: this ordering is taken verbatim from the reference `iterate_x` — its
docstring/comment says "[X Y Yaw vel Yaw_rate]" but the equations use
[X, Y, v, theta, omega] (index 2 = speed, index 3 = heading).

Input   : /<ns>/usv_relpos/estimate   geometry_msgs/PoseStamped
            position (x,y) = relative offset; orientation yaw = USV heading
Outputs : /<ns>/usv_track/filtered    nav_msgs/Odometry
            pose (x,y)+yaw = filtered relative pose; twist.linear (x,y) = relative
            velocity vector; twist.angular.z = omega; pose covariance carries the
            X/Y variance.
          /<ns>/usv_track/visible     std_msgs/Bool  (True = measured this cycle)

Run:
    python3 tmp/usv_motion_tracker_node.py _ns:=uav1
"""
import math
import threading
from copy import deepcopy

import numpy as np
import scipy.linalg
import rospy
from scipy.spatial.transform import Rotation as Rot

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool


# ─────────────────────────────────────────────────────────────────────────────
# Generic Unscented Kalman Filter  (ported from uav_packages/.../ukf_bis.py)
# ─────────────────────────────────────────────────────────────────────────────
class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar,
                 alpha, k, beta, iterate_function):
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function
        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim

        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)
        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) \
            + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))
        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2 * (self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2 * (self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()
        self.lock = Lock = threading.Lock()

    def __get_sigmas(self):
        ret = np.zeros((self.n_sig, self.n_dim))
        tmp_mat = (self.n_dim + self.lambd) * self.p
        # symmetrise + take the real matrix square root (covar may be slightly
        # non-PD after a measurement update); guard against complex output.
        tmp_mat = 0.5 * (tmp_mat + tmp_mat.T)
        spr_mat = np.real(scipy.linalg.sqrtm(tmp_mat))
        ret[0] = self.x
        for i in range(self.n_dim):
            ret[i + 1] = self.x + spr_mat[i]
            ret[i + 1 + self.n_dim] = self.x - spr_mat[i]
        return ret.T

    def update(self, states, data, r_matrix):
        with self.lock:
            num_states = len(states)
            sigmas_split = np.split(self.sigmas, self.n_dim)
            y = np.concatenate([sigmas_split[i] for i in states])
            x_split = np.split(self.x, self.n_dim)
            y_mean = np.concatenate([x_split[i] for i in states])

            y_diff = deepcopy(y)
            x_diff = deepcopy(self.sigmas)
            for i in range(self.n_sig):
                for j in range(num_states):
                    y_diff[j][i] -= y_mean[j]
                for j in range(self.n_dim):
                    x_diff[j][i] -= self.x[j]

            p_yy = np.zeros((num_states, num_states))
            for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
                p_yy += self.covar_weights[i] * val.dot(val.T)
            p_yy += r_matrix

            p_xy = np.zeros((self.n_dim, num_states))
            for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1),
                                        np.array_split(x_diff, self.n_sig, 1))):
                p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

            k = np.dot(p_xy, np.linalg.inv(p_yy))
            self.x = self.x + np.dot(k, (data - y_mean))
            self.p = self.p - np.dot(k, np.dot(p_yy, k.T))
            self.sigmas = self.__get_sigmas()

    def predict(self, timestep, process_noise):
        with self.lock:
            sigmas_out = np.array([self.iterate(x, timestep) for x in self.sigmas.T]).T
            x_out = np.zeros(self.n_dim)
            for i in range(self.n_dim):
                x_out[i] = sum(self.mean_weights[j] * sigmas_out[i][j]
                               for j in range(self.n_sig))
            p_out = np.zeros((self.n_dim, self.n_dim))
            for i in range(self.n_sig):
                diff = np.atleast_2d(sigmas_out.T[i] - x_out)
                p_out += self.covar_weights[i] * np.dot(diff.T, diff)
            p_out += process_noise
            self.sigmas = sigmas_out
            self.x = x_out
            self.p = p_out

    def get_state(self):
        return self.x

    def get_covar(self):
        return self.p

    def reset(self, state, covar):
        with self.lock:
            self.x = state
            self.p = covar


# ─────────────────────────────────────────────────────────────────────────────
# CTRV process model + process noise  (ported from Global_pose_publisher.py)
# ─────────────────────────────────────────────────────────────────────────────
def iterate_x(x_in, timestep):
    """CTRV transition.  State = [X, Y, v, theta, omega]."""
    x = np.array(x_in, dtype=float)
    if x[4] == 0:
        x[4] = 1e-20
    ret = np.zeros(len(x))
    ret[0] = x[0] + 2 * x[2] / x[4] * np.sin(x[4] * timestep / 2) \
        * np.cos(x[3] + x[4] * timestep / 2)
    ret[1] = x[1] + 2 * x[2] / x[4] * np.sin(x[4] * timestep / 2) \
        * np.sin(x[3] + x[4] * timestep / 2)
    ret[2] = x[2]
    ret[3] = x[3] + x[4] * timestep
    ret[4] = x[4]
    return ret


def get_q(timestep, sigma_a, sigma_alpha):
    """Process-noise covariance: white accel on v (idx 2) and white angular
    accel on omega (idx 4)."""
    SIGMA = np.diag([sigma_a ** 2, sigma_alpha ** 2])
    G = np.array([[0, 0], [0, 0], [timestep, 0], [0, 0], [0, timestep]])
    return G @ SIGMA @ G.T


def ang_norm(a):
    """Wrap to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class UsvMotionTracker:
    def __init__(self):
        rospy.init_node('usv_motion_tracker')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns

        in_topic = rospy.get_param('~estimate_topic', '/%s/usv_relpos/estimate' % ns)

        # ── tuning ────────────────────────────────────────────────────────────
        self._rate_hz   = float(rospy.get_param('~rate_hz', 30.0))
        # The Otter is a slow, smooth surface vessel → low process noise so the
        # filter does not invent velocity from position jitter.
        self._sigma_a   = float(rospy.get_param('~sigma_accel', 0.6))    # m/s^2
        self._sigma_alp = float(rospy.get_param('~sigma_alpha', 0.12))   # rad/s^2
        self._var_xy    = float(rospy.get_param('~meas_var_xy', 0.16))   # (0.4 m)^2
        self._var_th    = float(rospy.get_param('~meas_var_heading', 0.08))
        self._lost_to   = float(rospy.get_param('~lost_timeout_s', 5.0))  # predict-only bridge
        # (3→5 s for far-field acquisition robustness: more time for the controller COAST
        #  branch + gimbal IMU-hold to re-acquire a briefly-lost slow USV before giving up)
        self._init_v    = float(rospy.get_param('~init_speed', 0.0))
        self._meas_fresh = float(rospy.get_param('~meas_fresh_s', 0.5))
        # robustness clamps + outlier gating
        self._max_speed = float(rospy.get_param('~max_speed', 3.0))      # m/s
        self._max_omega = float(rospy.get_param('~max_omega', 1.0))      # rad/s
        self._max_jump  = float(rospy.get_param('~max_pos_jump_m', 3.0)) # gate radius
        self._max_outl  = int(rospy.get_param('~max_outliers', 8))       # → reinit

        self._r_pos = np.diag([self._var_xy, self._var_xy])
        self._r_h   = np.array([[self._var_th]])

        # ── shared state ──────────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._meas = None          # (x, y, heading_raw, stamp)  latest measurement
        self._meas_consumed_t = -1.0
        self._ukf = None
        self._time_out = 0.0
        self._last_tick = None
        self._outliers = 0

        # ── pub/sub ───────────────────────────────────────────────────────────
        self._pub_odom = rospy.Publisher('/%s/usv_track/filtered' % ns,
                                         Odometry, queue_size=5)
        self._pub_vis  = rospy.Publisher('/%s/usv_track/visible' % ns,
                                         Bool, queue_size=5)
        rospy.Subscriber(in_topic, PoseStamped, self._cb_meas, queue_size=10)
        rospy.Timer(rospy.Duration(1.0 / self._rate_hz), self._tick)
        rospy.loginfo("[usv_track] CTRV UKF up — in=%s out=/%s/usv_track/filtered "
                      "rate=%.0fHz", in_topic, ns, self._rate_hz)

    # ── measurement callback ────────────────────────────────────────────────
    def _cb_meas(self, msg):
        q = msg.pose.orientation
        heading = Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]
        with self._lock:
            self._meas = (float(msg.pose.position.x), float(msg.pose.position.y),
                          float(heading), msg.header.stamp.to_sec())

    # ── 30 Hz filter tick ────────────────────────────────────────────────────
    def _tick(self, _evt):
        now = rospy.Time.now().to_sec()
        with self._lock:
            meas = self._meas
            consumed_t = self._meas_consumed_t
            last_tick = self._last_tick

        dt = (now - last_tick) if (last_tick is not None and 0 < now - last_tick < 0.5) \
            else (1.0 / self._rate_hz)

        fresh = (meas is not None and (now - meas[3]) < self._meas_fresh)
        new = (meas is not None and meas[3] > consumed_t)

        # ── (re)initialise the filter when uninitialised + a fresh fix exists ──
        if self._ukf is None:
            if fresh and new:
                x0 = np.array([meas[0], meas[1], self._init_v, meas[2], 0.0])
                q0 = get_q(1.0 / self._rate_hz, self._sigma_a, self._sigma_alp)
                self._ukf = UKF(5, q0, x0, 5 * np.eye(5), 0.04, 0.0, 2.0, iterate_x)
                self._ukf.reset(x0, 5 * np.eye(5))
                self._time_out = 0.0
                with self._lock:
                    self._meas_consumed_t = meas[3]
                    self._last_tick = now
            else:
                with self._lock:
                    self._last_tick = now
            return

        # ── predict ────────────────────────────────────────────────────────────
        self._ukf.predict(dt, get_q(dt, self._sigma_a, self._sigma_alp))
        self._clamp_state()
        state = self._ukf.get_state()

        visible = False
        if fresh and new:
            with self._lock:
                self._meas_consumed_t = meas[3]
            jump = math.hypot(meas[0] - state[0], meas[1] - state[1])
            if jump > self._max_jump:
                # reject outlier (sign-flip / FOV-edge garbage); predict only.
                self._outliers += 1
                self._time_out += dt
                if self._outliers > self._max_outl:
                    # persistent → genuine large move or divergence: reinit here
                    rospy.logwarn("[usv_track] %d outliers → reinit at measurement",
                                  self._outliers)
                    self._ukf = None
                    self._outliers = 0
                    with self._lock:
                        self._last_tick = now
                    self._pub_vis.publish(Bool(data=False))
                    return
            else:
                self._outliers = 0
                # resolve the OBB 180° bow/stern ambiguity + unwrap vs the filter
                h = self._align_heading(meas[2], state[3])
                self._ukf.update([0, 1], np.array([meas[0], meas[1]]), self._r_pos)
                self._ukf.update([3], np.array([h]), self._r_h)
                self._clamp_state()
                self._time_out = 0.0
                visible = True
        else:
            # predict-only bridging; give up after lost_timeout
            self._time_out += dt
            if self._time_out > self._lost_to:
                self._ukf = None
                with self._lock:
                    self._last_tick = now
                self._pub_vis.publish(Bool(data=False))
                return

        with self._lock:
            self._last_tick = now
        self._publish(self._ukf.get_state(), self._ukf.get_covar(), visible, now)

    def _clamp_state(self):
        """Keep speed (idx 2) and turn-rate (idx 4) physically bounded so the
        CTRV filter cannot blow up on a slow/near-stationary target."""
        x = self._ukf.x
        x[2] = float(np.clip(x[2], -self._max_speed, self._max_speed))
        x[4] = float(np.clip(x[4], -self._max_omega, self._max_omega))

    @staticmethod
    def _align_heading(h_meas, theta_filt):
        """Pick h_meas or h_meas+π (bow/stern ambiguity), then unwrap so it is
        within ±π of the filter heading — keeps theta continuous."""
        best, best_d = h_meas, abs(ang_norm(h_meas - theta_filt))
        alt = h_meas + math.pi
        if abs(ang_norm(alt - theta_filt)) < best_d:
            best = alt
        # unwrap onto the same branch as theta_filt
        return theta_filt + ang_norm(best - theta_filt)

    def _publish(self, x, P, visible, stamp):
        vx = x[2] * math.cos(x[3])
        vy = x[2] * math.sin(x[3])
        od = Odometry()
        od.header.stamp = rospy.Time.from_sec(stamp)
        od.header.frame_id = '%s/base_link_world_aligned' % self._ns
        od.child_frame_id = '%s/usv' % self._ns
        od.pose.pose.position.x = float(x[0])
        od.pose.pose.position.y = float(x[1])
        od.pose.pose.position.z = 0.0
        qz = Rot.from_euler('z', x[3]).as_quat()
        od.pose.pose.orientation.x = float(qz[0])
        od.pose.pose.orientation.y = float(qz[1])
        od.pose.pose.orientation.z = float(qz[2])
        od.pose.pose.orientation.w = float(qz[3])
        cov = np.zeros((6, 6))
        cov[0, 0], cov[1, 1] = float(P[0, 0]), float(P[1, 1])
        cov[5, 5] = float(P[3, 3])
        od.pose.covariance = cov.flatten().tolist()
        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)
        od.twist.twist.angular.z = float(x[4])
        self._pub_odom.publish(od)
        self._pub_vis.publish(Bool(data=visible))


if __name__ == '__main__':
    try:
        UsvMotionTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
