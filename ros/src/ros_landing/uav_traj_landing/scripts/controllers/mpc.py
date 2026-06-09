#!/usr/bin/env python3
"""
mpc.py — Linear MPC for trajectory tracking (pure Python, no Acados).

Model: double integrator (state = [pos, vel], input = acceleration)
    x_{k+1} = A x_k + B u_k
    A = [[I, dt·I], [0, I]],  B = [[0.5·dt²·I], [dt·I]]

Cost (condensed QP, closed-form solution):
    J = Σ (x_k - x_ref_k)^T Q (x_k - x_ref_k) + (u_k - u_ref_k)^T R (u_k - u_ref_k)
      + (x_N - x_ref_N)^T Q_N (x_N - x_ref_N)

Condensed formulation (no decision variable for states):
    X = S_x x0 + S_u U
    H = S_u^T Q_bar S_u + R_bar   (precomputed + Cholesky factorized at init)
    g = S_u^T Q_bar (S_x x0 - X_ref) - R_bar U_ref
    U* = -H⁻¹ g                   (unconstrained, solved via Cholesky triangular solve)

The first control u*_0 is extracted; the rest of the horizon is discarded
(receding-horizon principle).

No external dependencies beyond numpy/scipy.

Reference trajectory horizon is passed via context dict (set by controller node):
    context['pos_horizon']  (N, 3) — desired positions at steps 1..N
    context['vel_horizon']  (N, 3) — desired velocities at steps 1..N
    context['acc_horizon']  (N, 3) — desired accelerations for inputs 0..N-1
"""

import numpy as np
import scipy.linalg
from .base import BaseController


class LinearMPCController(BaseController):
    """
    Linear MPC trajectory tracker, condensed QP, closed-form solve.

    Precomputes the Hessian H at init time (O(N³) once).
    Each 50 Hz call does one matrix-vector product + one Cholesky solve: O(N²).

    needs_horizon = True: the controller node builds the full N-step reference
    and passes it in context.
    """

    needs_horizon = True

    def __init__(self, drone_mass: float, hover_thrust: float,
                 max_tilt_rad: float,
                 N: int = 20, dt_mpc: float = 0.05,
                 Q_pos=None, Q_vel=None,
                 Q_pos_N=None, Q_vel_N=None,
                 R_acc=None, a_max: float = 15.0):
        super().__init__(drone_mass, hover_thrust, max_tilt_rad)
        self.N      = int(N)
        self.dt     = float(dt_mpc)
        self._a_max = float(a_max)

        # Default weights (CrazyTraj-inspired; heavier on Z for altitude)
        if Q_pos   is None: Q_pos   = [50.0,  50.0,  400.0]
        if Q_vel   is None: Q_vel   = [10.0,  10.0,   10.0]
        if Q_pos_N is None: Q_pos_N = [250.0, 250.0, 2000.0]  # 5× terminal weight
        if Q_vel_N is None: Q_vel_N = [50.0,  50.0,   50.0]
        if R_acc   is None: R_acc   = [1.0,   1.0,    1.0]

        self._r_diag = np.array(R_acc, float)   # shape (3,) for fast R_bar @ U_ref

        # ── Discrete double-integrator (ZOH) ─────────────────────────────
        dt = self.dt
        I3, Z3 = np.eye(3), np.zeros((3, 3))
        A = np.block([[I3, dt * I3],
                      [Z3,        I3]])          # (6, 6)
        B = np.block([[0.5 * dt**2 * I3],
                      [dt * I3]])                # (6, 3)

        N_h, n_x, n_u = self.N, 6, 3
        self._n_x, self._n_u, self._N_h = n_x, n_u, N_h

        # ── Condensed prediction matrices S_x (6N×6), S_u (6N×3N) ───────
        S_x = np.zeros((n_x * N_h, n_x))
        S_u = np.zeros((n_x * N_h, n_u * N_h))

        A_pow = A.copy()
        for i in range(N_h):
            S_x[i*n_x:(i+1)*n_x, :] = A_pow
            A_pow = A @ A_pow

        for j in range(N_h):
            A_pow_j = np.eye(n_x)
            for i in range(j, N_h):
                S_u[i*n_x:(i+1)*n_x, j*n_u:(j+1)*n_u] = A_pow_j @ B
                A_pow_j = A @ A_pow_j

        # ── Cost matrices Q_bar (6N×6N), R_bar (3N×3N) ──────────────────
        q_diag  = np.concatenate([Q_pos, Q_vel])    # (6,)
        qN_diag = np.concatenate([Q_pos_N, Q_vel_N])
        Q_bar   = np.kron(np.eye(N_h), np.diag(q_diag))
        Q_bar[(N_h-1)*n_x:, (N_h-1)*n_x:] = np.diag(qN_diag)   # terminal override

        R_bar = np.kron(np.eye(N_h), np.diag(self._r_diag))

        # ── Hessian (precomputed + Cholesky factorized) ───────────────────
        SuTQbar = S_u.T @ Q_bar              # (3N, 6N)
        H       = SuTQbar @ S_u + R_bar      # (3N, 3N)
        self._H_cho   = scipy.linalg.cho_factor(H)
        self._SuTQbar = SuTQbar              # saved for online gradient
        self._S_x     = S_x

    def reset(self):
        pass   # stateless

    def compute(self, des_pos, des_vel, des_acc,
                cur_pos, cur_vel, quat_xyzw, des_yaw, context=None):
        N_h, n_x, n_u = self._N_h, self._n_x, self._n_u

        x0    = np.concatenate([cur_pos, cur_vel])  # (6,)
        X_ref = np.empty(n_x * N_h)
        U_ref = np.empty(n_u * N_h)

        # Fill horizon reference from context or constant setpoint
        ph = vh = ah = None
        if context is not None:
            ph = context.get('pos_horizon')
            vh = context.get('vel_horizon')
            ah = context.get('acc_horizon')

        for i in range(N_h):
            ix, iu = i * n_x, i * n_u
            X_ref[ix:ix+3] = (ph[i] if ph is not None and i < len(ph)
                               else des_pos)
            X_ref[ix+3:ix+6] = (vh[i] if vh is not None and i < len(vh)
                                  else des_vel)
            U_ref[iu:iu+3] = (ah[i] if ah is not None and i < len(ah)
                               else des_acc)

        # ── Gradient g = S_u^T Q_bar (S_x x0 - X_ref) - R_bar U_ref ─────
        xi    = self._S_x @ x0 - X_ref                        # (6N,)
        # R_bar is block-diag(r_diag); R_bar @ U_ref = tile(r_diag, N) * U_ref
        R_Uref = np.tile(self._r_diag, N_h) * U_ref           # (3N,)
        g     = self._SuTQbar @ xi - R_Uref                    # (3N,)

        # ── Solve H U* = -g ──────────────────────────────────────────────
        U_opt = scipy.linalg.cho_solve(self._H_cho, -g)        # (3N,)

        # Apply first input (receding horizon)
        a_des = np.clip(U_opt[:3], -self._a_max, self._a_max)

        thrust_vec = self.mass * (a_des + np.array([0.0, 0.0, self.G]))
        roll, pitch, thrust_norm = self.thrust_vec_to_attitude(thrust_vec, des_yaw)
        return roll, pitch, des_yaw, thrust_norm
