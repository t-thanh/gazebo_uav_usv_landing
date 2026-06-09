#!/usr/bin/env python3
"""
smc.py — Sliding Mode Controller (Super-Twisting Algorithm).

Adapted from:
    CrazyTraj/CrazyFlow/lsy_drone_racing/lsy_drone_racing/control/
    sliding_mode_controller.py

Sliding surface:  s = lambda_s * e_pos + e_vel
Control law:      u = -k1 * |s|^0.5 * sign(s) + integral
                  d(integral)/dt = -k2 * sign(s)

Includes feedforward des_acc from the trajectory planner.
"""

import numpy as np
import rospy
from .base import BaseController


class SlidingModeController(BaseController):
    """
    Super-Twisting sliding mode trajectory tracker.

    s = λ·e_pos + e_vel            (sliding surface)
    u = -k1|s|^0.5·sign(s) + ∫    (STA: bounded, continuous)
    d∫/dt = -k2·sign(s)

    target_acc = des_acc + u            (feedforward + STA)
    thrust_vec = mass * (target_acc + [0, 0, G])
    """

    def __init__(self, drone_mass: float, hover_thrust: float,
                 max_tilt_rad: float,
                 lambda_s, k1, k2,
                 integral_limit: float = 2.0):
        super().__init__(drone_mass, hover_thrust, max_tilt_rad)
        self.lambda_s  = np.array(lambda_s, float)
        self.k1        = np.array(k1,       float)
        self.k2        = np.array(k2,       float)
        self.int_limit = float(integral_limit)
        self._integral = np.zeros(3)
        self._last_t   = None

    def reset(self):
        self._integral[:] = 0.0
        self._last_t = None

    def compute(self, des_pos, des_vel, des_acc,
                cur_pos, cur_vel, quat_xyzw, des_yaw, context=None):
        now = rospy.Time.now().to_sec()
        if self._last_t is not None and 0.0 < now - self._last_t < 0.5:
            dt = now - self._last_t
        else:
            dt = 0.02
        self._last_t = now

        e_pos = np.asarray(des_pos, float) - np.asarray(cur_pos, float)
        e_vel = np.asarray(des_vel, float) - np.asarray(cur_vel, float)
        s     = self.lambda_s * e_pos + e_vel

        self._integral += -self.k2 * np.sign(s) * dt
        self._integral  = np.clip(self._integral, -self.int_limit, self.int_limit)
        u_sta = -self.k1 * np.sqrt(np.abs(s)) * np.sign(s) + self._integral

        target_acc     = np.asarray(des_acc, float) + u_sta
        target_acc[2] += self.G

        thrust_vec = self.mass * target_acc
        roll, pitch, thrust_norm = self.thrust_vec_to_attitude(thrust_vec, des_yaw)
        return roll, pitch, des_yaw, thrust_norm
