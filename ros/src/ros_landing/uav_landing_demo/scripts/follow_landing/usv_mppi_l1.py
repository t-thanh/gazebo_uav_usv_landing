#!/usr/bin/env python3
"""
usv_mppi_l1.py  (tmp / dev-only)
─────────────────────────────────
MPPI + L1-style integral disturbance adaptation horizontal control law for the
UWB+IMU EKF relative state — the THIRD ablation arm (vs PD+FF and linear MPC),
drop-in for `horizontal_pd_ff` / `mpc_horizontal` (tmp/PLAN_usv_frame_traj_mppi.md §4B).

Adapted from the user's Crazyflie controller
  /home/t-thanh/Garage/cf_traj_control/cf_traj_control/controllers/mppi_l1_wind_ctrl.py
reduced from the full 13-state quaternion rollout to a 2-D relative TRANSLATIONAL
model in the USV/relative frame (the attitude inner loop is the X500's; we command
horizontal accel = tilt, exactly like the MPC arm):

    plant (per call, USV/relative frame):   r̈ = u − a_usv − d̂
        r        relative position  (target = origin)
        u        commanded UAV horizontal accel        (the control we sample)
        a_usv    measured USV accel (known FF, from the UWB/IMU EKF)
        d̂        L1 integral estimate of the UNMODELLED residual (waves, drag,
                 the time-varying part of a_usv the constant-velocity USV-frame
                 assumption drops) — injected into EVERY rollout so all samples
                 are disturbance-aware (verbatim concept from the CF wrapper).

MPPI (sampling MPC):
    • sample K control sequences  U = mean + ε,  ε ~ N(0, σ²),  clipped to |u|≤u_max
      (u_max = g·tan(max_tilt) enforces the tilt limit, like the MPC box constraint)
    • analytic double-integrator rollout, cost
          J = Σ_k  q_pos·‖r_k‖² + q_vel‖ṙ_k‖² + r_u‖u_k‖²
      (+ this is a REGULATOR to the origin now; the USV-frame trajectory reference
       r_ref(t) from Stage T slots straight into the rollout cost later — §2/§4B)
    • softmax weights  w = softmax(−(J−min J)/λ),  mean ← Σ w·U,  warm-started
      (shift) across calls;  u_0 applied.

L1 integral estimator (from CF `_IntegralEstimator`):
    err_int += r · dt ;  d̂ = clip(K_i · err_int, ±clamp)            (target = origin)

`mppi_l1_horizontal(r, rdot, ff, yaw_rel, mppi)` returns the body-yaw-frame accel
a_b = Rz(−yaw_rel)·[u_x,u_y], identical interface to the other two arms.  Stateful
(warm-start mean + integral) → ONE MppiL1 instance per controller, reused each tick.
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

_G = 9.80665


class _IntegralEstimator:
    """Position-error integral → corrective accel d̂ (the L1-style adaptive term).
    Anti-windup clips the integral so |d̂| never exceeds `clamp`."""

    def __init__(self, K_i=0.3, clamp=2.0, dt=0.1):
        self._Ki, self._clamp, self._dt = K_i, clamp, dt
        self.reset()

    def reset(self):
        self._int = np.zeros(2)

    def update(self, pos_err):
        # Sign: in this relative frame the disturbance enters the plant SUBTRACTIVELY
        # (r̈ = u − a_usv − d), the opposite of the CF frame (accel = u + d).  For the
        # estimate to converge to the true disturbance d̂→d_true (steady state needs
        # d̂=d_true, see module header), the integral must use −pos_err: while d̂<d_true
        # the residual drifts r negative → −∫r grows → d̂ rises toward d_true.
        self._int += -np.asarray(pos_err, float) * self._dt
        if self._Ki > 1e-9:
            m = self._clamp / self._Ki
            self._int = np.clip(self._int, -m, m)
        return self._Ki * self._int


class MppiL1:
    """MPPI horizontal controller (2-D relative double integrator) + L1 integral
    disturbance adaptation.  Stateful: warm-start control mean + integral term."""

    def __init__(self, dt=0.1, N=20, K=512, lam=0.05, sigma=2.0,
                 q_pos=2.0, q_vel=0.6, r_u=0.05, u_max=2.0,
                 K_i=0.3, d_clamp=2.0, seed=0):
        self.dt, self.N, self.K, self.lam = dt, int(N), int(K), lam
        self.sigma, self.u_max = sigma, u_max
        self.q_pos, self.q_vel, self.r_u = q_pos, q_vel, r_u
        self._mean = np.zeros((self.N, 2))           # warm-started control mean
        self._rng = np.random.default_rng(seed if seed else None)
        self._est = _IntegralEstimator(K_i=K_i, clamp=d_clamp, dt=dt)

    def reset(self):
        self._mean[:] = 0.0
        self._est.reset()

    def solve(self, r_xy, rdot_xy, a_usv_xy):
        """r,ṙ,a_usv : 2-vectors (USV/relative frame).  Returns u_0 (commanded accel, 2-D)."""
        r0 = np.asarray(r_xy, float)
        v0 = np.asarray(rdot_xy, float)
        w = np.asarray(a_usv_xy, float)
        d_hat = self._est.update(r0)                 # L1 residual estimate
        bias = w + d_hat                             # known + adaptive disturbance

        # sample K control sequences around the warm-started mean, clip to tilt box
        eps = self._rng.normal(0.0, self.sigma, size=(self.K, self.N, 2))
        U = np.clip(self._mean[None, :, :] + eps, -self.u_max, self.u_max)

        # vectorised analytic rollout of  r̈ = u − bias
        pos = np.tile(r0, (self.K, 1))
        vel = np.tile(v0, (self.K, 1))
        cost = np.zeros(self.K)
        for k in range(self.N):
            uk = U[:, k, :]
            vel = vel + (uk - bias) * self.dt
            pos = pos + vel * self.dt
            cost += (self.q_pos * np.einsum('ij,ij->i', pos, pos)
                     + self.q_vel * np.einsum('ij,ij->i', vel, vel)
                     + self.r_u * np.einsum('ij,ij->i', uk, uk))

        # softmax weighting → new control mean
        beta = cost.min()
        wts = np.exp(-(cost - beta) / self.lam)
        wts /= (wts.sum() + 1e-9)
        new_mean = np.einsum('k,knj->nj', wts, U)
        u0 = new_mean[0].copy()
        # warm-start: shift the mean forward one step
        self._mean[:-1] = new_mean[1:]
        self._mean[-1] = new_mean[-1]
        return np.clip(u0, -self.u_max, self.u_max)


def build_mppi_l1(dt=0.1, N=20, K=512, lam=0.05, sigma=2.0,
                  q_pos=2.0, q_vel=0.6, r_u=0.05, max_tilt_rad=0.209,
                  K_i=0.3, d_clamp=2.0, seed=0):
    u_max = _G * np.tan(max_tilt_rad)
    return MppiL1(dt=dt, N=N, K=K, lam=lam, sigma=sigma, q_pos=q_pos, q_vel=q_vel,
                  r_u=r_u, u_max=u_max, K_i=K_i, d_clamp=d_clamp, seed=seed)


def mppi_l1_horizontal(r_xy, rdot_xy, ff_xy, yaw_rel, mppi):
    """Drop-in for horizontal_pd_ff / mpc_horizontal: MPPI+L1 → body-yaw-frame accel."""
    u = mppi.solve(r_xy, rdot_xy, ff_xy)
    return Rot.from_euler('z', -yaw_rel).apply(np.append(u, 0.0))[:2]


# ── offline self-test ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    g = _G
    npass = ntot = 0

    def check(name, cond):
        global npass, ntot
        ntot += 1
        npass += bool(cond)
        print(("  ok  " if cond else " FAIL ") + name)

    # 1. tilt-limit respected
    m = build_mppi_l1(max_tilt_rad=np.radians(12))
    u = m.solve([10.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    umax = g * np.tan(np.radians(12))
    check("u within tilt box (|u|<=u_max)", np.all(np.abs(u) <= umax + 1e-6))

    # 2. sign: positive offset → negative accel (drive back to origin)
    m = build_mppi_l1()
    u = m.solve([3.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    check("sign: +x offset → −x accel", u[0] < 0)
    u = m.solve([0.0, -2.0], [0.0, 0.0], [0.0, 0.0])
    check("sign: −y offset → +y accel", u[1] > 0)

    # 3. closed-loop convergence to a static target (regulator)
    m = build_mppi_l1(K_i=0.0)
    r = np.array([4.0, -3.0]); v = np.array([0.0, 0.0]); dt = 0.1
    for _ in range(200):
        u = m.solve(r, v, [0.0, 0.0])
        v = v + (u - 0.0) * dt
        r = r + v * dt
    check("converges to origin (|r|<0.3)", np.linalg.norm(r) < 0.3)

    # 4. feed-forward: tracks a CONSTANT-accel USV (a_usv) at small steady offset
    m = build_mppi_l1(K_i=0.0)
    a_usv = np.array([1.0, 0.0])
    r = np.zeros(2); v = np.zeros(2)
    for _ in range(300):
        u = m.solve(r, v, a_usv)
        v = v + (u - a_usv) * dt        # true relative dynamics
        r = r + v * dt
    check("FF tracks accel USV (|r|<0.5)", np.linalg.norm(r) < 0.5)

    # 5. L1 rejects an UNKNOWN constant disturbance (not given as a_usv)
    m = build_mppi_l1(K_i=0.5, d_clamp=3.0)
    d_true = np.array([0.8, -0.6])      # unmodelled, NOT passed as a_usv
    r = np.zeros(2); v = np.zeros(2)
    errs = []
    for i in range(600):
        u = m.solve(r, v, [0.0, 0.0])   # controller does NOT know d_true
        v = v + (u - d_true) * dt
        r = r + v * dt
        if i > 400:
            errs.append(np.linalg.norm(r))
    m0 = build_mppi_l1(K_i=0.0)          # baseline: no L1
    r2 = np.zeros(2); v2 = np.zeros(2); e2 = []
    for i in range(600):
        u = m0.solve(r2, v2, [0.0, 0.0])
        v2 = v2 + (u - d_true) * dt
        r2 = r2 + v2 * dt
        if i > 400:
            e2.append(np.linalg.norm(r2))
    check("L1 reduces steady error vs no-L1 (%.2f < %.2f)"
          % (np.mean(errs), np.mean(e2)), np.mean(errs) < np.mean(e2))

    # 6. yaw_rel rotation handled by the wrapper
    m = build_mppi_l1()
    a_b = mppi_l1_horizontal([2.0, 0.0], [0.0, 0.0], [0.0, 0.0], np.radians(90), m)
    check("wrapper returns finite 2-vec", np.all(np.isfinite(a_b)) and a_b.shape == (2,))

    # 7. compute budget (must fit the 50 Hz tick)
    m = build_mppi_l1(K=512, N=20)
    t0 = time.time()
    for _ in range(50):
        m.solve([2.0, 1.0], [0.1, 0.0], [0.0, 0.0])
    ms = (time.time() - t0) / 50 * 1e3
    check("tick < 20 ms (%.1f ms, K=512 N=20)" % ms, ms < 20.0)

    print("\nMPPI+L1 self-test: %d/%d passed" % (npass, ntot))
