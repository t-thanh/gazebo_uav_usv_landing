#!/usr/bin/env python3
"""
usv_bezier_predictor_node.py  (tmp / dev-only)
───────────────────────────────────────────────
Stage D of the GPS-free USV tracking pipeline — a weighted, regularised Bézier
extrapolator that fits the USV's recent **relative** trajectory and predicts
where it will be a short horizon ahead.  Ported from
`uav_packages/prediction/bezier_predictor/scripts/Predictor.py`; the only change
is that it consumes the relative filtered pose (no GPS / global frame) and emits
a single look-ahead point for the follow controller plus a Path for RViz.

Input   : /<ns>/usv_track/filtered   nav_msgs/Odometry   (Stage C output)
Outputs : /<ns>/usv_track/lookahead  geometry_msgs/PoseStamped
            predicted relative offset at t+lookahead_s (controller aim point)
          /<ns>/usv_track/path       nav_msgs/Path   (full predicted curve, viz)

Run:
    python3 tmp/usv_bezier_predictor_node.py _ns:=uav1
"""
import itertools
import math

import numpy as np
from scipy.special import comb
from cvxopt import solvers, matrix

import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

solvers.options['show_progress'] = False


# ── Bézier algebra (ported verbatim from Predictor.py) ───────────────────────
def get_M(degree):
    Bernstein = np.zeros(degree + 1)
    Bez = np.zeros([degree + 1, degree + 1])
    for i in range(degree + 1):
        Bernstein[i] = comb(degree, i, exact=True)
    for j in range(degree + 1):
        val = Bernstein[j]
        mat = np.zeros(degree + 1)
        for k in range(degree + 1 - j):
            if (j % 2) == 0:
                mat[k] = (-1) ** ((j + 1) * (k + 1)) * val * comb(degree - j, k, exact=True)
            else:
                mat[k] = (-1) ** ((j) * (k)) * val * comb(degree - j, k, exact=True)
        Bez[j, :] = mat
    return Bez


def enumerate_combinations(n, k):
    return np.array(list(itertools.combinations(range(1, n + 1), k)))


def get_regularizer(degree):
    derive2_deg = degree - 2
    poly_mat = np.transpose(get_M(derive2_deg))
    combinaison = enumerate_combinations(derive2_deg + 1, 2)
    integral = np.zeros(len(combinaison) + derive2_deg + 1)
    for i in range(len(combinaison)):
        p1 = np.poly1d(np.flip(poly_mat[combinaison[i, 0] - 1, :]))
        p2 = np.poly1d(np.flip(poly_mat[combinaison[i, 1] - 1, :]))
        integral[i + derive2_deg + 1] = 2 * (np.polyval(np.polyint(p1 * p2), 1)
                                             - np.polyval(np.polyint(p1 * p2), 0))
    for i in range(derive2_deg + 1):
        p = np.poly1d(np.flip(poly_mat[i, :]))
        integral[i] = np.polyval(np.polyint(p * p), 1) - np.polyval(np.polyint(p * p), 0)
    Mp = np.zeros([derive2_deg + 1, derive2_deg + 1])
    for i in range(derive2_deg + 1):
        Mp[i, i] = integral[i]
    for i in range(len(combinaison)):
        Mp[combinaison[i, 0] - 1, combinaison[i, 1] - 1] = integral[i + derive2_deg + 1] / 2
        Mp[combinaison[i, 1] - 1, combinaison[i, 0] - 1] = integral[i + derive2_deg + 1] / 2
    Mp = Mp * (degree) ** 2 * (degree - 1) ** 2
    Pas = np.zeros([derive2_deg + 1, degree + 1])
    for i in range(derive2_deg + 1):
        Pas[i, i] = 1
        Pas[i, i + 1] = -2
        Pas[i, i + 2] = 1
    return np.transpose(Pas) @ Mp @ Pas


def get_velocity(P):
    Pas = np.zeros((P.shape[1] - 1, P.shape[1]))
    np.fill_diagonal(Pas, -1)
    np.fill_diagonal(Pas[:, 1:], 1)
    return (P.shape[1] - 1) * (Pas @ np.transpose(P))


def get_G(degree):
    A = np.zeros([2 * degree - 1, degree + 1])
    for i in range(degree):
        A[i, i] = -1 * degree
        A[i, i + 1] = 1 * degree
    for i in range(degree - 1):
        A[i + degree, i] = degree * (degree - 1) * 1
        A[i + degree, i + 1] = -2 * degree * (degree - 1)
        A[i + degree, i + 2] = degree * (degree - 1) * 1
    return np.concatenate((A, -A), axis=0)


def get_h(degree, v_max, a_max):
    u = np.zeros(2 * (2 * degree - 1))
    u[0:degree] = v_max
    u[degree:degree + 2] = a_max
    u[degree + 2:2 * degree + 2] = v_max
    u[2 * degree + 2:] = a_max
    return u


def get_T(time_scaled, degree):
    L = len(time_scaled)
    T = np.zeros([L, degree + 1])
    for i in range(degree + 1):
        if i < degree:
            T[:, i] = time_scaled ** (degree - i)
        else:
            T[:, degree] = np.ones(L)
    return T


def check_constraints(constraint, control_points, time_scale):
    M = get_M(len(control_points) - 1)
    px = np.poly1d((M @ control_points[:, 0]))
    py = np.poly1d((M @ control_points[:, 1]))
    sq = (px * px) + (py * py)
    constr = np.poly1d(np.array([constraint ** 2]))
    zeros = np.roots(sq / time_scale - constr)
    real = zeros.real[abs(zeros.imag) < 1e-5]
    for i in real:
        if 0 <= i <= 1:
            return False
    return True


class Queue:
    def __init__(self, max_len, dim):
        self.L = max_len
        self.Q = np.zeros([1, dim])
        self.Time = np.zeros(1)
        self.Weight = np.empty(0)
        self.Time_scaled = np.empty(0)
        self.Index = 0

    def push_element(self, element, T):
        if self.Index >= 1:
            self.Q = np.vstack([self.Q, element])
            self.Time = np.append(self.Time, T)
            if self.Q.shape[0] > self.L:
                self.Q = np.delete(self.Q, 0, axis=0)
                self.Time = np.delete(self.Time, 0)
        elif self.Index == 0:
            self.Q[0] = element
            self.Time[0] = T
            self.Index += 1

    def calculate_weight(self, factor, max_time):
        self.Weight = np.zeros(len(self.Time))
        n = max(self.Q.shape)
        non_zero = n
        for i in range(n):
            if i < n - 1:
                if (self.Time[-1] - self.Time[i]) > max_time:
                    self.Weight[i] = 0
                    non_zero = n - (i + 1)
                else:
                    self.Weight[i] = np.tanh(factor / (self.Time[-1] - self.Time[i]))
            else:
                self.Weight[-1] = 1
        return non_zero

    def scale_time(self, factor):
        self.Time_scaled = (self.Time - self.Time[0]) / (self.Time[-1] - self.Time[0]) * factor


class UsvBezierPredictor:
    def __init__(self):
        rospy.init_node('usv_bezier_predictor')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns
        self._deg = int(rospy.get_param('~degree', 5))
        self._pred_t = float(rospy.get_param('~pred_horizon_s', 2.5))   # curve future span
        self._look_t = float(rospy.get_param('~lookahead_s', 1.0))      # controller aim point
        self._min_pts = int(rospy.get_param('~min_points', 7))
        self._n_path = int(rospy.get_param('~path_points', 20))
        self._vmax = float(rospy.get_param('~v_max', 5.0))
        self._amax = float(rospy.get_param('~a_max', 3.0))

        self._M = get_M(self._deg)
        self._reg = get_regularizer(self._deg)
        self._G = get_G(self._deg)
        self._QL, self._Tw = 50, 0.01
        self._queue = Queue(50, 2)
        self._sol = None

        in_topic = rospy.get_param('~filtered_topic', '/%s/usv_track/filtered' % ns)
        self._pub_look = rospy.Publisher('/%s/usv_track/lookahead' % ns,
                                         PoseStamped, queue_size=5)
        self._pub_path = rospy.Publisher('/%s/usv_track/path' % ns, Path, queue_size=5)
        rospy.Subscriber(in_topic, Odometry, self._cb, queue_size=10)
        rospy.loginfo("[usv_bezier] up — in=%s  deg=%d  horizon=%.1fs  lookahead=%.1fs",
                      in_topic, self._deg, self._pred_t, self._look_t)

    def _cb(self, msg):
        t = msg.header.stamp.to_sec() or rospy.Time.now().to_sec()
        pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        self._queue.push_element(pos, t)
        non_zero = self._queue.calculate_weight(0.5, 3.0)
        if len(self._queue.Time) < self._min_pts or non_zero < self._min_pts:
            return

        temps_queue = self._queue.Time[-1] - self._queue.Time[0]
        if temps_queue <= 1e-3:
            return
        time_scale = self._pred_t + temps_queue
        factor = temps_queue / time_scale

        ok = self._fit(factor, constrained=False)
        if ok:
            vel = get_velocity(self._sol)
            acc = get_velocity(vel)
            if not (check_constraints(self._vmax, vel, time_scale)
                    and check_constraints(self._amax, acc, time_scale)):
                ok = self._fit(factor, constrained=True,
                               vmax=time_scale * self._vmax, amax=time_scale * self._amax)
        if not ok or self._sol is None:
            return

        self._publish(msg.header.stamp, temps_queue)

    def _fit(self, factor, constrained, vmax=0.0, amax=0.0):
        self._queue.scale_time(factor)
        T = get_T(self._queue.Time_scaled, self._deg)
        W = np.diag(self._queue.Weight)
        X, Y = self._queue.Q[:, 0], self._queue.Q[:, 1]
        TM = T @ self._M
        P = TM.T @ (W @ TM) + self._QL * self._Tw * self._reg
        qx = TM.T @ (W @ X)
        qy = TM.T @ (W @ Y)
        if constrained:
            G, h = matrix(self._G), matrix(get_h(self._deg, vmax, amax))
        else:
            G = h = None
        try:
            sx = solvers.qp(matrix(P), matrix(-qx), G, h, solver='OSQP')
            sy = solvers.qp(matrix(P), matrix(-qy), G, h, solver='OSQP')
            self._sol = np.vstack([np.squeeze(sx['x']), np.squeeze(sy['x'])])
            return True
        except Exception as exc:                       # keep old solution
            rospy.logwarn_throttle(5.0, "[usv_bezier] QP failed: %s", exc)
            return False

    def _curve(self, s_array):
        """Evaluate the fitted Bézier (control pts self._sol, 2x(deg+1)) at scaled
        times s_array ∈ [0,1]."""
        T = get_T(np.asarray(s_array, dtype=float), self._deg)
        cx = T @ self._M @ self._sol[0, :]
        cy = T @ self._M @ self._sol[1, :]
        return cx, cy

    def _publish(self, stamp, temps_queue):
        frame = '%s/base_link_world_aligned' % self._ns
        total = self._pred_t + temps_queue
        # the "now" instant sits at scaled time = temps_queue/total; future is beyond it
        s_now = temps_queue / total
        # look-ahead point
        s_look = min(1.0, (temps_queue + self._look_t) / total)
        cx, cy = self._curve([s_look])
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame
        ps.pose.position.x, ps.pose.position.y = float(cx[0]), float(cy[0])
        ps.pose.orientation.w = 1.0
        self._pub_look.publish(ps)
        # predicted future path (now → end of horizon)
        s_path = np.linspace(s_now, 1.0, self._n_path)
        px, py = self._curve(s_path)
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = frame
        for i in range(self._n_path):
            p = PoseStamped()
            p.header.frame_id = frame
            p.pose.position.x, p.pose.position.y = float(px[i]), float(py[i])
            p.pose.orientation.w = 1.0
            path.poses.append(p)
        self._pub_path.publish(path)


if __name__ == '__main__':
    try:
        UsvBezierPredictor()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
