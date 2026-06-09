#!/usr/bin/env python3
"""
base.py — abstract base for all trajectory-tracking attitude controllers.

Each subclass receives desired trajectory state + current drone state and
returns (roll, pitch, yaw, thrust_norm) for MAVROS AttitudeTarget.

Class variable:
    needs_horizon (bool): set True in MPC subclass; the controller node will
        build a full N-step reference horizon and pass it via context dict:
            context['pos_horizon']  (N, 3)
            context['vel_horizon']  (N, 3)
            context['acc_horizon']  (N, 3)
"""

import math
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial.transform import Rotation


class BaseController(ABC):

    G             = 9.806   # m/s²
    needs_horizon = False   # True for MPC

    def __init__(self, drone_mass: float, hover_thrust: float,
                 max_tilt_rad: float):
        self.mass         = float(drone_mass)
        self.hover_thrust = float(hover_thrust)
        self.max_tilt     = float(max_tilt_rad)

    @abstractmethod
    def compute(self, des_pos: np.ndarray, des_vel: np.ndarray,
                des_acc: np.ndarray,
                cur_pos: np.ndarray, cur_vel: np.ndarray,
                quat_xyzw: np.ndarray, des_yaw: float,
                context: dict = None):
        """
        Returns (roll, pitch, yaw, thrust_norm).
        thrust_norm is normalised [0, 1] for MAVROS AttitudeTarget.
        """

    def reset(self):
        """Called when the controller is re-enabled after a pause."""

    def thrust_vec_to_attitude(self, thrust_vec: np.ndarray, des_yaw: float):
        """
        Convert a 3-D thrust vector + desired yaw to (roll, pitch, thrust_norm).

        thrust_vec: mass × desired_world_acceleration [N], gravity already added.
        des_yaw: desired heading [rad].
        """
        thrust_mag = float(np.linalg.norm(thrust_vec))
        z_des = thrust_vec / (thrust_mag + 1e-8)

        x_c = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
        y_des = np.cross(z_des, x_c)
        if np.linalg.norm(y_des) < 1e-6:
            x_c   = np.array([1.0, 0.0, 0.0])
            y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des)
        x_des  = np.cross(y_des, z_des)
        R_des  = np.column_stack([x_des, y_des, z_des])

        euler = Rotation.from_matrix(R_des).as_euler('xyz')
        roll  = float(np.clip(euler[0], -self.max_tilt, self.max_tilt))
        pitch = float(np.clip(euler[1], -self.max_tilt, self.max_tilt))

        thrust_norm = float(np.clip(
            thrust_mag / (self.mass * self.G) * self.hover_thrust, 0.05, 1.0))

        return roll, pitch, thrust_norm
