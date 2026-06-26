#!/usr/bin/env python3
"""
usv_uwb_mpc.py  (tmp / dev-only)
──────────────────────────────────
Linear MPC horizontal control law for the UWB+IMU EKF relative state — the SECOND
ablation arm (vs PD+FF), drop-in for `horizontal_pd_ff` (tmp/PLAN_uwb_mpc.md §4).

Plant (relative double integrator, USV/relative frame, per axis — x,y decoupled):
    r̈ = u − a_usv          u = commanded UAV accel,  a_usv = measured USV accel
The USV-IMU acceleration enters the prediction as a KNOWN constant disturbance over
the horizon, so the optimiser produces the USV feed-forward implicitly (at steady
state u → a_usv) on top of the regulation — the GPS-free analogue of the GPS+MPC ref.

Condensed box-constrained QP (precomputed once):
    min_U  Σ_k qpos·r_k² + qvel·ṙ_k² (+ terminal) + ρ·u_k²   s.t.  |u_k| ≤ u_max
where u_max = g·tan(max_tilt) ⇒ the tilt limit is enforced INSIDE the optimisation
(the MPC's advantage over PD+FF, which clips post-hoc).  First input u_0 is applied.

`mpc_horizontal(r, rdot, ff, yaw_rel, mpc)` returns the body-yaw-frame horizontal
accel a_b (= Rz(−yaw_rel)·[u_x, u_y]), identical interface to horizontal_pd_ff.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

try:
    from cvxopt import matrix as _cvxmat, solvers as _cvxsolvers
    _cvxsolvers.options['show_progress'] = False
    _HAVE_CVX = True
except Exception:
    _HAVE_CVX = False


class AxisMPC:
    """Finite-horizon MPC for one axis of the relative double integrator."""

    def __init__(self, dt=0.1, N=15, q_pos=1.0, q_vel=0.3, r_u=0.05,
                 qN_pos=5.0, qN_vel=1.0, u_max=2.0):
        self.dt, self.N, self.u_max = dt, N, u_max
        A = np.array([[1.0, dt], [0.0, 1.0]])
        B = np.array([[0.5 * dt * dt], [dt]])
        # condensed prediction  X = Sx·x0 + Su·U + Sw·w   (w = a_usv, disturbance −B·w)
        Sx = np.zeros((2 * N, 2))
        Su = np.zeros((2 * N, N))
        Apow = np.eye(2)
        for k in range(N):
            Apow = Apow @ A                      # A^(k+1)
            Sx[2 * k:2 * k + 2, :] = Apow
            for j in range(k + 1):
                Su[2 * k:2 * k + 2, j:j + 1] = np.linalg.matrix_power(A, k - j) @ B
        Sw = -Su @ np.ones((N, 1))               # constant −w enters like a control offset
        Qbar = np.zeros((2 * N, 2 * N))
        for k in range(N):
            qp, qv = (qN_pos, qN_vel) if k == N - 1 else (q_pos, q_vel)
            Qbar[2 * k, 2 * k] = qp
            Qbar[2 * k + 1, 2 * k + 1] = qv
        H = 2.0 * (Su.T @ Qbar @ Su + r_u * np.eye(N))
        H = 0.5 * (H + H.T)                       # symmetrise
        self._Sx, self._Su, self._Sw, self._Qbar, self._H = Sx, Su, Sw, Qbar, H
        self._Hinv = np.linalg.inv(H)
        # box constraints  −u_max ≤ U ≤ u_max
        self._G = np.vstack([np.eye(N), -np.eye(N)])
        self._h = np.full(2 * N, u_max)

    def solve(self, x0, w):
        """x0 = [pos, vel] (relative), w = a_usv (axis).  Returns u_0 (commanded accel)."""
        c0 = self._Sx @ np.asarray(x0, float) + (self._Sw.flatten() * float(w))
        f = 2.0 * self._Su.T @ self._Qbar @ c0
        if _HAVE_CVX:
            try:
                sol = _cvxsolvers.qp(_cvxmat(self._H), _cvxmat(f),
                                     _cvxmat(self._G), _cvxmat(self._h))
                if sol['status'] == 'optimal':
                    return float(np.asarray(sol['x']).flatten()[0])
            except Exception:
                pass
        # fallback: unconstrained optimum then clip
        U = -self._Hinv @ f
        return float(np.clip(U[0], -self.u_max, self.u_max))


def build_axis_mpc(dt=0.1, N=15, q_pos=1.0, q_vel=0.3, r_u=0.05,
                   qN_pos=5.0, qN_vel=1.0, max_tilt_rad=0.209, g=9.806):
    u_max = g * np.tan(max_tilt_rad)
    return AxisMPC(dt=dt, N=N, q_pos=q_pos, q_vel=q_vel, r_u=r_u,
                   qN_pos=qN_pos, qN_vel=qN_vel, u_max=u_max)


def mpc_horizontal(r_xy, rdot_xy, ff_xy, yaw_rel, mpc):
    """Drop-in for horizontal_pd_ff: per-axis MPC → body-yaw-frame accel a_b."""
    r = np.asarray(r_xy, float); v = np.asarray(rdot_xy, float); w = np.asarray(ff_xy, float)
    u = np.array([mpc.solve([r[0], v[0]], w[0]),
                  mpc.solve([r[1], v[1]], w[1])])
    return Rot.from_euler('z', -yaw_rel).apply(np.append(u, 0.0))[:2]
