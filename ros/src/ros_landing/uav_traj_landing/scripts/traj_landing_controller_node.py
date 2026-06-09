#!/usr/bin/env python3
"""
traj_landing_controller_node.py
────────────────────────────────
Trajectory-following attitude controller for GPS-denied UAV landing on USV.

Replaces ar_landing_controller_node.py from ar_code_landing.
Same sensors and inputs, same AttitudeTarget output — different control law:

  ar_code_landing  : altitude PID  +  horizontal PD  →  attitude + thrust
  THIS NODE        : trajectory planner → pluggable attitude controller
                     → attitude + thrust

Trajectory planner (default: minimum-jerk polynomial)
──────────────────────────────────────────────────────
  5th-order polynomial with boundary conditions:
    t=0 : pos=current, vel=current, acc=0
    t=T : pos=target,  vel=0,       acc=0
  Coefficients derived analytically (closed-form).
  T = max(t_min, 1.5 × distance / v_max)

  Optional: time-optimal trajectory via rpg_time_optimal (uzh-rpg/rpg_time_optimal).
  Enable with param use_time_optimal:true and rpg_time_optimal_path.
  Falls back to polynomial on any import / solver error.

Trajectory trackers (selectable via controller_type param)
──────────────────────────────────────────────────────────
  smc  — Sliding Mode Controller, Super-Twisting Algorithm (default)
         s = λ·e_pos + e_vel;  u = -k1|s|^0.5·sign(s) + ∫;  d∫/dt = -k2·sign(s)
         Includes des_acc feedforward from trajectory.

  pid  — Classical PID position controller (no feedforward)
         target_acc = kp·e_pos + ki·∫e_pos + kd·e_vel

  mpc  — Linear MPC, condensed QP (pure Python, no Acados)
         Double-integrator model, receding horizon N steps.
         Uses full trajectory horizon reference for feedforward.

Subscribed topics
─────────────────
  /ar_landing/gimbal_tracker/usv_world_pose  geometry_msgs/PoseStamped
  /<ns>/garmin/range                          sensor_msgs/Range
  /<ns>/mavros/imu/data                       sensor_msgs/Imu
  /<ns>/mavros/local_position/odom            nav_msgs/Odometry
  /<ns>/ground_truth                          nav_msgs/Odometry
  /traj_landing/ctrl/enable                   std_msgs/Bool
  /traj_landing/ctrl/target_altitude          std_msgs/Float64

Published topics (50 Hz)
────────────────────────
  /<ns>/mavros/setpoint_raw/attitude  mavros_msgs/AttitudeTarget
  /traj_landing/ctrl/status           std_msgs/String
  /traj_landing/ctrl/tracking_error   geometry_msgs/Vector3Stamped
    vector.x = e_x, vector.y = e_y,  vector.z = |e_xy|  (ar_code_landing convention)
"""

import math
import os
import sys
import tempfile
import threading

# Allow 'from controllers import ...' when run as a ROS node
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rospy

from std_msgs.msg      import Bool, Float64, String
from sensor_msgs.msg   import Range, Imu
from nav_msgs.msg      import Odometry
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Quaternion
from mavros_msgs.msg   import AttitudeTarget

from scipy.spatial.transform import Rotation

from controllers import PIDController, SlidingModeController, LinearMPCController


# ─────────────────────────────────────────────────────────────────────────────
# Minimum-jerk polynomial trajectory
# ─────────────────────────────────────────────────────────────────────────────

class MinJerkTrajectory:
    """
    5th-order polynomial trajectory (minimum jerk).

    Boundary conditions:
        t=0: pos=p0, vel=v0, acc=0
        t=T: pos=pf, vel=0,  acc=0

    Coefficients are derived analytically:
        a0 = p0,  a1 = v0,  a2 = 0
        a3 = (10·Δp − 6·v0·T) / T³
        a4 = (8·v0·T − 15·Δp) / T⁴
        a5 = (6·Δp  − 3·v0·T) / T⁵
    where Δp = pf − p0.
    """

    def __init__(self, p0: np.ndarray, v0: np.ndarray,
                 pf: np.ndarray, duration: float):
        self._mode = 'poly'
        self.p0 = np.asarray(p0, float)
        self.pf = np.asarray(pf, float)
        self.t_total = float(duration)
        T = self.t_total
        dp = self.pf - self.p0
        v0 = np.asarray(v0, float) if v0 is not None else np.zeros(3)

        # _a shape (3, 6): one row per axis, columns = coefficients a0..a5
        self._a = np.zeros((3, 6))
        for i in range(3):
            d, v = dp[i], v0[i]
            self._a[i] = [p0[i], v, 0.0,
                          (10.0*d - 6.0*v*T) / T**3,
                          (8.0*v*T - 15.0*d) / T**4,
                          (6.0*d  - 3.0*v*T) / T**5]

    @classmethod
    def from_arrays(cls, pos: np.ndarray, vel: np.ndarray,
                    acc: np.ndarray, t: np.ndarray):
        """
        Build from pre-sampled arrays returned by rpg_time_optimal.
        pos/vel/acc: (N, 3) arrays;  t: (N,) timestamps in seconds.
        """
        obj = object.__new__(cls)
        obj._mode  = 'array'
        obj.p0     = np.asarray(pos[0], float)
        obj.pf     = np.asarray(pos[-1], float)
        obj.t_total = float(t[-1])
        obj._pos   = np.asarray(pos, float)
        obj._vel   = np.asarray(vel, float)
        obj._acc   = np.asarray(acc, float)
        obj._t     = np.asarray(t, float)
        obj._a     = None
        return obj

    def evaluate(self, t: float):
        """Return (pos, vel, acc) at time t, clamped to [0, t_total]."""
        t = float(np.clip(t, 0.0, self.t_total))
        if self._mode == 'array':
            i = int(np.clip(np.searchsorted(self._t, t, side='right') - 1,
                            0, len(self._t) - 1))
            return (self._pos[i].copy(),
                    self._vel[i].copy(),
                    self._acc[i].copy())
        a = self._a
        t2, t3, t4, t5 = t**2, t**3, t**4, t**5
        pos = a[:,0] + a[:,1]*t + a[:,2]*t2 + a[:,3]*t3 + a[:,4]*t4 + a[:,5]*t5
        vel = a[:,1] + 2*a[:,2]*t + 3*a[:,3]*t2 + 4*a[:,4]*t3 + 5*a[:,5]*t4
        acc = 2*a[:,2] + 6*a[:,3]*t + 12*a[:,4]*t2 + 20*a[:,5]*t3
        return pos, vel, acc


# ─────────────────────────────────────────────────────────────────────────────
# ROS node
# ─────────────────────────────────────────────────────────────────────────────

class TrajLandingControllerNode:
    """
    50 Hz attitude controller with background trajectory planning.

    When enabled=False: publishes level hover setpoints (warm-up / idle).
    When enabled=True:  generates trajectory to current target and tracks
                        it with the selected controller (smc / pid / mpc).

    Replanning is triggered in the background (non-blocking):
      • when the USV target has moved > replan_distance_m from the trajectory
        endpoint, OR
      • when the current trajectory has completed (t > t_total + 2 s grace).
    Minimum interval between replanning calls: replan_interval_s.
    """

    def __init__(self):
        rospy.init_node('traj_landing_controller', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')
        self._ns = ns

        # ── Physical parameters ────────────────────────────────────────────
        self._hover_thrust = rospy.get_param('~hover_thrust', 0.58)
        self._drone_mass   = rospy.get_param('~drone_mass',   2.0)
        max_tilt_deg       = rospy.get_param('~max_tilt_deg', 25.0)
        max_tilt_rad       = math.radians(max_tilt_deg)

        # ── Controller selection ───────────────────────────────────────────
        ctrl_type = rospy.get_param('~controller_type', 'smc').lower()
        self._ctrl = self._build_controller(ctrl_type, max_tilt_rad)
        rospy.loginfo(f'[traj_ctrl] controller={ctrl_type}')

        # ── Trajectory parameters ──────────────────────────────────────────
        self._v_max         = rospy.get_param('~traj_v_max_ms',     3.0)
        self._t_min         = rospy.get_param('~traj_t_min_s',      2.0)
        self._replan_dist   = rospy.get_param('~replan_distance_m', 0.5)
        self._replan_ivl    = rospy.get_param('~replan_interval_s', 2.0)
        self._pose_timeout  = rospy.get_param('~pose_timeout_s',    3.0)

        # ── Optional rpg_time_optimal ──────────────────────────────────────
        self._use_time_opt  = rospy.get_param('~use_time_optimal',       False)
        self._rpg_path      = rospy.get_param('~rpg_time_optimal_path',  '')
        self._quad_yaml     = rospy.get_param('~quad_params_file',       '')

        # ── Sensor state (protected by _lock) ─────────────────────────────
        self._lock        = threading.Lock()
        self._drone_pos   = None    # (3,) world ENU
        self._drone_vel   = None    # (3,) world ENU
        self._drone_quat  = None    # (4,) [x,y,z,w]
        self._imu_yaw     = 0.0
        self._target_pos  = None    # (3,) USV world position from gimbal tracker
        self._target_t    = None    # float, wall-clock time of last target message
        self._enabled     = False
        self._target_alt  = 5.0

        # ── Trajectory state (protected by _lock) ─────────────────────────
        self._traj        = None    # MinJerkTrajectory
        self._traj_t0     = None    # float, wall time at trajectory start
        self._traj_end    = None    # (3,) endpoint of current trajectory
        self._planning    = False   # True while background thread is running
        self._last_plan_t = 0.0     # wall time of last plan start (no lock needed)

        # ── Publishers ────────────────────────────────────────────────────
        self._att_pub    = rospy.Publisher(
            f'/{ns}/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=5)
        self._status_pub = rospy.Publisher(
            '/traj_landing/ctrl/status', String, queue_size=5)
        self._err_pub    = rospy.Publisher(
            '/traj_landing/ctrl/tracking_error', Vector3Stamped, queue_size=5)

        # ── Subscribers ───────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/ground_truth',
                         Odometry, self._gt_cb, queue_size=2)
        rospy.Subscriber(f'/{ns}/mavros/local_position/odom',
                         Odometry, self._odom_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/mavros/imu/data',
                         Imu, self._imu_cb, queue_size=5)
        rospy.Subscriber(f'/{ns}/garmin/range',
                         Range, self._garmin_cb, queue_size=5)
        rospy.Subscriber('/ar_landing/gimbal_tracker/usv_world_pose',
                         PoseStamped, self._target_cb, queue_size=5)
        rospy.Subscriber('/traj_landing/ctrl/enable',
                         Bool, self._enable_cb, queue_size=2)
        rospy.Subscriber('/traj_landing/ctrl/target_altitude',
                         Float64, self._alt_cb, queue_size=2)

        rospy.Timer(rospy.Duration(0.02), self._ctrl_loop)   # 50 Hz

        rospy.loginfo(
            f'[traj_ctrl] init  ns={ns}  '
            f'mass={self._drone_mass} kg  hover={self._hover_thrust}  '
            f'time_opt={self._use_time_opt}')

    # ── Controller factory ────────────────────────────────────────────────────

    def _build_controller(self, ctrl_type: str, max_tilt_rad: float):
        mass   = self._drone_mass
        hover  = self._hover_thrust

        if ctrl_type == 'pid':
            kp       = rospy.get_param('~pid_kp',       [0.4,  0.4,  1.25])
            ki       = rospy.get_param('~pid_ki',       [0.05, 0.05, 0.05])
            kd       = rospy.get_param('~pid_kd',       [0.2,  0.2,  0.4])
            ki_range = rospy.get_param('~pid_ki_range', [2.0,  2.0,  0.4])
            return PIDController(mass, hover, max_tilt_rad,
                                 kp, ki, kd, ki_range)

        if ctrl_type == 'mpc':
            N_h     = rospy.get_param('~mpc_N',       20)
            dt_mpc  = rospy.get_param('~mpc_dt',      0.05)
            Q_pos   = rospy.get_param('~mpc_Q_pos',   [50.0,  50.0,  400.0])
            Q_vel   = rospy.get_param('~mpc_Q_vel',   [10.0,  10.0,   10.0])
            Q_pos_N = rospy.get_param('~mpc_Q_pos_N', [250.0, 250.0, 2000.0])
            Q_vel_N = rospy.get_param('~mpc_Q_vel_N', [50.0,  50.0,   50.0])
            R_acc   = rospy.get_param('~mpc_R_acc',   [1.0,   1.0,    1.0])
            a_max   = rospy.get_param('~mpc_a_max',   15.0)
            return LinearMPCController(mass, hover, max_tilt_rad,
                                       N=N_h, dt_mpc=dt_mpc,
                                       Q_pos=Q_pos, Q_vel=Q_vel,
                                       Q_pos_N=Q_pos_N, Q_vel_N=Q_vel_N,
                                       R_acc=R_acc, a_max=a_max)

        if ctrl_type != 'smc':
            rospy.logwarn(
                f'[traj_ctrl] unknown controller_type "{ctrl_type}"; using smc')

        lambda_s  = rospy.get_param('~smc_lambda',         [1.5, 1.5, 2.0])
        k1        = rospy.get_param('~smc_k1',             [1.2, 1.2, 1.5])
        k2        = rospy.get_param('~smc_k2',             [0.7, 0.7, 1.0])
        int_limit = rospy.get_param('~smc_integral_limit', 2.0)
        return SlidingModeController(mass, hover, max_tilt_rad,
                                     lambda_s, k1, k2, int_limit)

    # ── Sensor callbacks (any ROS thread) ────────────────────────────────────

    def _gt_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        with self._lock:
            self._drone_pos = np.array([p.x, p.y, p.z])

    def _odom_cb(self, msg: Odometry):
        # mavros local_position/odom: linear velocity is in body frame;
        # rotate to world frame via pose quaternion.
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear
        R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        v_world = R @ np.array([v.x, v.y, v.z])
        with self._lock:
            self._drone_vel = v_world

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        qxyzw = [q.x, q.y, q.z, q.w]
        yaw = Rotation.from_quat(qxyzw).as_euler('xyz')[2]
        with self._lock:
            self._drone_quat = np.array(qxyzw)
            self._imu_yaw    = float(yaw)

    def _garmin_cb(self, msg: Range):
        pass    # Garmin not used directly; SM monitors it for touchdown

    def _target_cb(self, msg: PoseStamped):
        p = msg.pose.position
        with self._lock:
            self._target_pos = np.array([p.x, p.y, p.z])
            self._target_t   = rospy.Time.now().to_sec()

    def _enable_cb(self, msg: Bool):
        with self._lock:
            self._enabled = bool(msg.data)
            if not msg.data:
                self._traj    = None
                self._traj_t0 = None
        if not msg.data:
            self._ctrl.reset()

    def _alt_cb(self, msg: Float64):
        with self._lock:
            self._target_alt = float(msg.data)

    # ── 50 Hz control loop (ROS Timer, main thread) ───────────────────────────

    def _ctrl_loop(self, _event):
        with self._lock:
            pos      = self._drone_pos.copy()  if self._drone_pos  is not None else None
            vel      = self._drone_vel.copy()  if self._drone_vel  is not None else None
            quat     = self._drone_quat.copy() if self._drone_quat is not None else None
            yaw      = self._imu_yaw
            tgt_pos  = self._target_pos.copy() if self._target_pos is not None else None
            tgt_t    = self._target_t
            enabled  = self._enabled
            tgt_alt  = self._target_alt
            traj     = self._traj
            traj_t0  = self._traj_t0

        if pos is None:
            self._publish_hover(yaw)
            return

        if not enabled:
            self._publish_hover(yaw)
            self._status_pub.publish(String(data='IDLE'))
            return

        now = rospy.Time.now().to_sec()

        # Determine XY waypoint: use USV position if fresh, else hover in place
        target_fresh = (tgt_t is not None and now - tgt_t < self._pose_timeout)
        if target_fresh and tgt_pos is not None:
            waypoint = np.array([tgt_pos[0], tgt_pos[1], tgt_alt])
        else:
            waypoint = np.array([pos[0], pos[1], tgt_alt])

        vel_cur = vel if vel is not None else np.zeros(3)

        # Check whether replanning is needed (may launch background thread)
        self._maybe_replan(pos, vel_cur, waypoint, now)

        # Re-read trajectory after potential replan decision
        with self._lock:
            traj    = self._traj
            traj_t0 = self._traj_t0

        # Look up desired state on the trajectory
        if traj is not None:
            t_rel = now - traj_t0
            des_pos, des_vel, des_acc = traj.evaluate(t_rel)
            status = 'TRACKING'
        else:
            des_pos = waypoint
            des_vel = np.zeros(3)
            des_acc = np.zeros(3)
            status  = 'HOLDING'

        # Build horizon context for MPC
        context = None
        if getattr(self._ctrl, 'needs_horizon', False) and traj is not None:
            N_h  = self._ctrl.N
            dt_h = self._ctrl.dt
            t_rel_now = now - traj_t0
            ph = np.empty((N_h, 3))
            vh = np.empty((N_h, 3))
            ah = np.empty((N_h, 3))
            for i in range(N_h):
                ph[i], vh[i], ah[i] = traj.evaluate(t_rel_now + (i + 1) * dt_h)
            context = {'pos_horizon': ph, 'vel_horizon': vh, 'acc_horizon': ah}

        # Compute attitude command
        if quat is not None:
            roll, pitch, yaw_cmd, thrust = self._ctrl.compute(
                des_pos, des_vel, des_acc, pos, vel_cur, quat, yaw,
                context=context)
        else:
            roll, pitch, yaw_cmd, thrust = 0.0, 0.0, yaw, self._hover_thrust

        self._publish_attitude(roll, pitch, yaw_cmd, thrust)
        self._status_pub.publish(String(data=status))

        # Horizontal error to target (SM reads this for ALIGN/DESCEND exit)
        err = pos - waypoint
        ev = Vector3Stamped()
        ev.header.stamp = rospy.Time.now()
        ev.vector.x = float(err[0])
        ev.vector.y = float(err[1])
        ev.vector.z = float(np.linalg.norm(err[:2]))   # |e_xy| in z (ar_code_landing convention)
        self._err_pub.publish(ev)

    # ── Trajectory replanning ─────────────────────────────────────────────────

    def _maybe_replan(self, pos, vel, waypoint, now):
        with self._lock:
            traj      = self._traj
            traj_t0   = self._traj_t0
            traj_end  = self._traj_end.copy() if self._traj_end is not None else None
            planning  = self._planning

        if planning:
            return

        need_replan = False
        if traj is None:
            need_replan = True
        else:
            # Target has drifted from trajectory endpoint
            if (traj_end is not None and
                    np.linalg.norm(waypoint - traj_end) > self._replan_dist):
                need_replan = True
            # Trajectory completed (include grace period for hover at endpoint)
            if now - traj_t0 > traj.t_total + 2.0:
                need_replan = True

        if need_replan and (now - self._last_plan_t) >= self._replan_ivl:
            self._last_plan_t = now
            with self._lock:
                self._planning = True
            t = threading.Thread(
                target=self._plan_and_update,
                args=(pos.copy(), vel.copy(), waypoint.copy()),
                daemon=True)
            t.start()

    def _plan_and_update(self, p0, v0, target_pos):
        try:
            if self._use_time_opt and self._rpg_path:
                traj = self._plan_time_optimal(p0, v0, target_pos)
            else:
                traj = self._plan_poly(p0, v0, target_pos)

            with self._lock:
                self._traj     = traj
                self._traj_t0  = rospy.Time.now().to_sec()
                self._traj_end = target_pos.copy()
            self._ctrl.reset()

            rospy.loginfo(
                f'[traj_ctrl] trajectory: '
                f'dist={np.linalg.norm(target_pos - p0):.2f} m  '
                f'T={traj.t_total:.1f} s  mode={traj._mode}')
        except Exception as exc:
            rospy.logerr(f'[traj_ctrl] planning failed: {exc}')
        finally:
            with self._lock:
                self._planning = False

    # ── Trajectory generators ─────────────────────────────────────────────────

    def _plan_poly(self, p0: np.ndarray, v0: np.ndarray,
                   target: np.ndarray) -> MinJerkTrajectory:
        """Minimum-jerk polynomial trajectory. Always succeeds."""
        dist = float(np.linalg.norm(target - p0))
        T = max(self._t_min, dist / self._v_max * 1.5)
        return MinJerkTrajectory(p0, v0, target, T)

    def _plan_time_optimal(self, p0: np.ndarray, v0: np.ndarray,
                           target: np.ndarray) -> MinJerkTrajectory:
        """
        Time-optimal trajectory via rpg_time_optimal (uzh-rpg/rpg_time_optimal).
        Falls back to polynomial on any failure.

        The target is specified as both an intermediate gate (with tolerance 0.5 m)
        and the end state (zero velocity), which is the standard racing-track
        formulation of a point-to-point minimum-time problem.

        Install: pip install casadi  &&  git clone uzh-rpg/rpg_time_optimal
        Set param rpg_time_optimal_path to the repo's 'src' directory.
        Set param quad_params_file to config/quad_params.yaml.
        """
        import sys as _sys
        _sys.path.insert(0, self._rpg_path)

        try:
            import yaml as _yaml
            from track import Track
            from quad import Quad
            from integrator import RungeKutta4
            from planner import Planner
            from trajectory import Trajectory as RpgTraj
        except ImportError as exc:
            rospy.logwarn_once(
                f'[traj_ctrl] rpg_time_optimal not importable ({exc}); '
                f'using polynomial')
            return self._plan_poly(p0, v0, target)

        with self._lock:
            q_xyzw = (self._drone_quat.copy()
                      if self._drone_quat is not None else np.array([0,0,0,1.]))
        q_wxyz = [float(q_xyzw[3]),
                  float(q_xyzw[0]),
                  float(q_xyzw[1]),
                  float(q_xyzw[2])]

        track_data = {
            'initial': {
                'position': p0.tolist(),
                'attitude': q_wxyz,
                'velocity': v0.tolist(),
                'omega':    [0.0, 0.0, 0.0],
            },
            # Single gate at the target position; tolerance 0.5 m
            'gates': [{'position': target.tolist(), 'radius': 0.5}],
            'end': {
                'position': target.tolist(),
                'attitude': [1, 0, 0, 0],
                'velocity': [0.0, 0.0, 0.0],
                'omega':    [0.0, 0.0, 0.0],
            },
            'ring': False,
        }

        tmp_file = None
        try:
            with tempfile.NamedTemporaryFile(
                    mode='w', suffix='.yaml', delete=False, dir='/tmp') as f:
                _yaml.dump(track_data, f)
                tmp_file = f.name

            quad  = Quad(self._quad_yaml)
            track = Track(tmp_file)
            planner = Planner(quad, track, RungeKutta4, {
                'tolerance':      0.5,
                'nodes_per_gate': 50,
                'vel_guess':      min(self._v_max, 3.0),
            })
            planner.setup()
            x_sol = planner.solve()
            rpg   = RpgTraj(x_sol, NPW=planner.NPW, wp=planner.wp)

            N     = rpg.p.shape[1]
            t_arr = np.linspace(0.0, rpg.t_total, N)
            return MinJerkTrajectory.from_arrays(
                pos=rpg.p.T,       # (N,3)
                vel=rpg.v.T,
                acc=rpg.a_lin.T,
                t=t_arr)
        except Exception as exc:
            rospy.logwarn(
                f'[traj_ctrl] rpg_time_optimal solver failed ({exc}); '
                f'polynomial fallback')
            return self._plan_poly(p0, v0, target)
        finally:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)

    # ── Output helpers ────────────────────────────────────────────────────────

    def _publish_attitude(self, roll: float, pitch: float,
                          yaw: float, thrust: float):
        q = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()  # [x,y,z,w]
        att = AttitudeTarget()
        att.header.stamp = rospy.Time.now()
        att.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE  |
                         AttitudeTarget.IGNORE_PITCH_RATE |
                         AttitudeTarget.IGNORE_YAW_RATE)
        att.orientation = Quaternion(
            x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
        att.thrust = float(np.clip(thrust, 0.05, 1.0))
        self._att_pub.publish(att)

    def _publish_hover(self, yaw: float):
        self._publish_attitude(0.0, 0.0, yaw, self._hover_thrust)

    def run(self):
        rospy.spin()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        TrajLandingControllerNode().run()
    except rospy.ROSInterruptException:
        pass
