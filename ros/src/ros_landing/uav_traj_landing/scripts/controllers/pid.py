#!/usr/bin/env python3
"""
pid.py — PID position controller for trajectory tracking.

Adapted from:
    CrazyTraj/CrazyFlow/lsy_drone_racing/lsy_drone_racing/control/
    attitude_controller.py

Differences vs SMC:
  - No feedforward acceleration (des_acc is NOT used)
  - Classical P + I + D on position/velocity error
  - Anti-windup via element-wise integral clamping
"""

import numpy as np
import rospy
from .base import BaseController


class PIDController(BaseController):
    """
    PID position controller with gravity compensation.

    target_acc = kp * e_pos + ki * i_error + kd * e_vel
    thrust_vec = mass * (target_acc + [0, 0, G])
    """

    def __init__(self, drone_mass: float, hover_thrust: float,
                 max_tilt_rad: float,
                 kp, ki, kd, ki_range=None):
        super().__init__(drone_mass, hover_thrust, max_tilt_rad)
        self.kp       = np.array(kp,   float)
        self.ki       = np.array(ki,   float)
        self.kd       = np.array(kd,   float)
        self.ki_range = np.array(
            ki_range if ki_range is not None else [2.0, 2.0, 0.4], float)

        self._i_error = np.zeros(3)
        self._last_t  = None

    def reset(self):
        self._i_error[:] = 0.0
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

        self._i_error += e_pos * dt
        self._i_error  = np.clip(self._i_error, -self.ki_range, self.ki_range)

        target_acc     = self.kp * e_pos + self.ki * self._i_error + self.kd * e_vel
        target_acc[2] += self.G

        thrust_vec = self.mass * target_acc
        roll, pitch, thrust_norm = self.thrust_vec_to_attitude(thrust_vec, des_yaw)
        return roll, pitch, des_yaw, thrust_norm
