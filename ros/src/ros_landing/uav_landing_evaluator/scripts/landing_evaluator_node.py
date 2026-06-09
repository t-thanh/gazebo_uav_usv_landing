#!/usr/bin/env python3
"""
landing_evaluator_node.py
─────────────────────────
Drop-test evaluator for UAV-on-USV landing.

For each scenario:
  1. Teleport the drop_uav model to a random position above the landing pad
     (uniform disc of radius xy_range, optional random tilt).
  2. Release (zero velocity) — UAV falls freely under gravity.
  3. Detect first contact via /landing_pad/contact (gazebo_ros_bumper).
  4. Extract the model state just BEFORE contact from a rolling buffer.
  5. Compute and log:
       • fall_time        — seconds from release to contact
       • impact_speed     — total velocity [m/s] at impact
       • kinetic_energy   — 0.5 · m · v²  [J]  (total and vertical component)
       • tilt_angle       — UAV body tilt from vertical [deg] at contact
       • contact_distance — horizontal miss distance from pad centre [m]
       • landing_score    — quality score 0–100

6. Print per-scenario result, then a full summary table, and save CSV.

Subscribed topics
─────────────────
  /gazebo/model_states          gazebo_msgs/ModelStates   (buffered)
  /landing_pad/contact          gazebo_msgs/ContactsState (bumper trigger)

Service calls
─────────────
  /gazebo/set_model_state       gazebo_msgs/SetModelState  (reposition drop_uav)
"""

import collections
import csv
import math
import os
import random
import threading

import numpy as np
import rospy

from gazebo_msgs.msg import ContactsState, ModelStates
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from geometry_msgs.msg import Twist
from scipy.spatial.transform import Rotation


# ── Column order for summary table and CSV ────────────────────────────────────
CSV_FIELDS = [
    'scenario',
    'drop_x',   'drop_y',   'drop_z',
    'drop_roll_deg', 'drop_pitch_deg',
    'fall_time_s',
    'impact_vx', 'impact_vy', 'impact_vz', 'impact_speed_ms',
    'KE_total_J', 'KE_vertical_J',
    'tilt_deg',
    'contact_dx', 'contact_dy', 'contact_dist_m',
    'landing_score',
]


class LandingEvaluatorNode:
    """Orchestrates N drop-test scenarios and records landing quality metrics."""

    # ── Init ─────────────────────────────────────────────────────────────────

    def __init__(self):
        rospy.init_node('landing_evaluator', anonymous=False)

        # Parameters
        self.n_scenarios     = rospy.get_param('~n_scenarios',       5)
        self.drop_height     = rospy.get_param('~drop_height',      15.0)
        self.xy_range        = rospy.get_param('~xy_range',          2.0)
        self.tilt_range_deg  = rospy.get_param('~tilt_range_deg',   15.0)
        pad                  = rospy.get_param('~pad_center',  [0.0, 0.0, 0.355])
        self.pad_center      = np.array(pad, float)
        self.drone_mass      = rospy.get_param('~drone_mass',        2.0)
        self.contact_timeout = rospy.get_param('~contact_timeout',  12.0)
        self.settle_time     = rospy.get_param('~settle_time',       2.0)
        self.park_alt        = rospy.get_param('~park_altitude',   100.0)
        self.buf_size        = rospy.get_param('~state_buffer_size', 1000)
        self.pre_contact_dt  = rospy.get_param('~pre_contact_window', 0.08)
        self.alpha           = rospy.get_param('~score_alpha',       0.05)
        self.beta            = rospy.get_param('~score_beta',        0.04)
        self.gamma           = rospy.get_param('~score_gamma',       0.40)
        self.csv_path        = rospy.get_param('~csv_output',
                                '/tmp/landing_eval_results.csv')

        # Internal state
        self._lock           = threading.Lock()
        self._state_buffer   = collections.deque(maxlen=self.buf_size)
        self._contact_event  = threading.Event()
        self._contact_data   = None   # (t_contact, buffer_snapshot)
        self._results        = []

        # Subscribers
        rospy.Subscriber('/gazebo/model_states', ModelStates,
                         self._model_states_cb, queue_size=10)
        rospy.Subscriber('/landing_pad/contact', ContactsState,
                         self._contact_cb, queue_size=20)

        # Service
        rospy.loginfo('[evaluator] waiting for /gazebo/set_model_state ...')
        rospy.wait_for_service('/gazebo/set_model_state', timeout=60.0)
        self._set_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        rospy.loginfo('[evaluator] ready — %d scenarios planned', self.n_scenarios)

    # ── ROS callbacks ─────────────────────────────────────────────────────────

    def _model_states_cb(self, msg: ModelStates):
        """Buffer (timestamp, pose, twist) for drop_uav at every model_states tick."""
        try:
            idx = msg.name.index('drop_uav')
        except ValueError:
            return
        t = rospy.Time.now().to_sec()
        with self._lock:
            self._state_buffer.append((t, msg.pose[idx], msg.twist[idx]))

    def _contact_cb(self, msg: ContactsState):
        """Record first contact that involves drop_uav."""
        if not msg.states or self._contact_event.is_set():
            return
        # Verify at least one contact body is part of drop_uav
        drop_uav_contact = any(
            'drop_uav' in s.collision1_name or 'drop_uav' in s.collision2_name
            for s in msg.states
        )
        if not drop_uav_contact:
            return
        t = rospy.Time.now().to_sec()
        with self._lock:
            self._contact_data = (t, list(self._state_buffer))
        self._contact_event.set()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _teleport(self, x: float, y: float, z: float,
                  roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
        """Move drop_uav to (x, y, z) with given attitude and zero velocity."""
        q = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        req = SetModelStateRequest()
        req.model_state.model_name      = 'drop_uav'
        req.model_state.pose.position.x = x
        req.model_state.pose.position.y = y
        req.model_state.pose.position.z = z
        req.model_state.pose.orientation.x = float(q[0])
        req.model_state.pose.orientation.y = float(q[1])
        req.model_state.pose.orientation.z = float(q[2])
        req.model_state.pose.orientation.w = float(q[3])
        req.model_state.twist           = Twist()  # zero velocity
        req.model_state.reference_frame = 'world'
        try:
            self._set_state(req)
        except rospy.ServiceException as e:
            rospy.logerr('[evaluator] set_model_state failed: %s', e)

    def _pre_contact_state(self, t_contact: float, buffer: list):
        """
        Return the (t, pose, twist) entry from the buffer that is closest to
        (t_contact - pre_contact_dt), i.e. the state just before impact.
        """
        target = t_contact - self.pre_contact_dt
        best, best_err = None, float('inf')
        for entry in reversed(buffer):
            t, _, _ = entry
            if t > t_contact:
                continue
            err = abs(t - target)
            if err < best_err:
                best_err, best = err, entry
        return best

    def _wait_for_drop_uav(self, timeout: float = 30.0):
        """Block until drop_uav appears in /gazebo/model_states."""
        deadline = rospy.Time.now().to_sec() + timeout
        rospy.loginfo('[evaluator] waiting for drop_uav in model_states ...')
        while not rospy.is_shutdown():
            with self._lock:
                # Check if any buffered entry exists (= model is in sim)
                if self._state_buffer:
                    rospy.loginfo('[evaluator] drop_uav found')
                    return True
            if rospy.Time.now().to_sec() > deadline:
                rospy.logerr('[evaluator] timeout: drop_uav never appeared')
                return False
            rospy.sleep(0.5)
        return False

    # ── Metrics computation ───────────────────────────────────────────────────

    def _compute_metrics(self, scenario_id: int,
                         drop_pos: np.ndarray,
                         drop_roll_deg: float, drop_pitch_deg: float,
                         t_drop: float, t_contact: float,
                         state):
        t_s, pose, twist = state

        vx, vy, vz = twist.linear.x, twist.linear.y, twist.linear.z
        v  = np.array([vx, vy, vz])
        speed = float(np.linalg.norm(v))

        m    = self.drone_mass
        KE   = 0.5 * m * speed**2
        KE_z = 0.5 * m * vz**2          # vertical kinetic energy (main impact)

        # Tilt: angle between body z-axis and world z-axis
        o = pose.orientation
        R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        z_body = R @ np.array([0.0, 0.0, 1.0])
        cos_t  = float(np.clip(np.dot(z_body, [0.0, 0.0, 1.0]), -1.0, 1.0))
        tilt   = math.degrees(math.acos(cos_t))

        # Miss distance (horizontal only — quality of lateral placement)
        dx = pose.position.x - self.pad_center[0]
        dy = pose.position.y - self.pad_center[1]
        r_miss = math.sqrt(dx**2 + dy**2)

        fall_time = t_contact - t_drop

        # Landing quality score: 100 = perfect (zero energy, zero tilt, on centre)
        score = 100.0 * (
            math.exp(-self.alpha * KE) *
            math.exp(-self.beta  * tilt) *
            math.exp(-self.gamma * r_miss)
        )

        return {
            'scenario':       scenario_id,
            'drop_x':         round(float(drop_pos[0]), 3),
            'drop_y':         round(float(drop_pos[1]), 3),
            'drop_z':         round(float(drop_pos[2]), 3),
            'drop_roll_deg':  round(drop_roll_deg, 2),
            'drop_pitch_deg': round(drop_pitch_deg, 2),
            'fall_time_s':    round(fall_time, 3),
            'impact_vx':      round(vx, 3),
            'impact_vy':      round(vy, 3),
            'impact_vz':      round(vz, 3),
            'impact_speed_ms':round(speed, 3),
            'KE_total_J':     round(KE, 3),
            'KE_vertical_J':  round(KE_z, 3),
            'tilt_deg':       round(tilt, 2),
            'contact_dx':     round(dx, 3),
            'contact_dy':     round(dy, 3),
            'contact_dist_m': round(r_miss, 3),
            'landing_score':  round(score, 1),
        }

    # ── Scenario runner ───────────────────────────────────────────────────────

    def run(self):
        """Run all N drop scenarios sequentially, then print summary."""
        # Park away while Gazebo finishes spawning everything
        rospy.sleep(3.0)
        rospy.wait_for_message('/gazebo/model_states', ModelStates, timeout=30.0)

        if not self._wait_for_drop_uav(timeout=30.0):
            rospy.logerr('[evaluator] cannot start: drop_uav not in simulation')
            return

        random.seed(rospy.Time.now().secs % 1000)

        for i in range(1, self.n_scenarios + 1):
            if rospy.is_shutdown():
                break

            # ── Random drop position (uniform disc) ─────────────────────
            r   = self.xy_range * math.sqrt(random.random())
            phi = random.uniform(0.0, 2.0 * math.pi)
            dx  = r * math.cos(phi)
            dy  = r * math.sin(phi)

            # ── Random initial tilt to simulate imperfect deployment ─────
            roll_deg  = random.uniform(-self.tilt_range_deg, self.tilt_range_deg)
            pitch_deg = random.uniform(-self.tilt_range_deg, self.tilt_range_deg)

            drop_x = self.pad_center[0] + dx
            drop_y = self.pad_center[1] + dy
            drop_z = self.pad_center[2] + self.drop_height
            drop_pos = np.array([drop_x, drop_y, drop_z])

            rospy.loginfo(
                '[evaluator] ── Scenario %d/%d ──  '
                'drop=(%.2f, %.2f, %.2f)  tilt=(%.1f°, %.1f°)',
                i, self.n_scenarios,
                drop_x, drop_y, drop_z, roll_deg, pitch_deg)

            # ── Reset contact state ──────────────────────────────────────
            self._contact_event.clear()
            self._contact_data = None

            # ── Teleport to drop position ────────────────────────────────
            self._teleport(
                drop_x, drop_y, drop_z,
                math.radians(roll_deg), math.radians(pitch_deg), 0.0)
            t_drop = rospy.Time.now().to_sec()

            # ── Wait for contact ─────────────────────────────────────────
            hit = self._contact_event.wait(timeout=self.contact_timeout)

            if not hit:
                rospy.logwarn(
                    '[evaluator] scenario %d: timeout (%.0f s) — '
                    'no contact detected; skipping',
                    i, self.contact_timeout)
                self._park()
                continue

            # ── Extract and compute metrics ──────────────────────────────
            with self._lock:
                t_contact, buf_snap = self._contact_data

            pre_state = self._pre_contact_state(t_contact, buf_snap)
            if pre_state is None:
                rospy.logwarn(
                    '[evaluator] scenario %d: no pre-contact state in buffer; '
                    'increase state_buffer_size', i)
                self._park()
                continue

            m = self._compute_metrics(
                i, drop_pos, roll_deg, pitch_deg,
                t_drop, t_contact, pre_state)
            self._results.append(m)
            self._log_scenario(m)

            # ── Settle then park before next drop ────────────────────────
            rospy.sleep(self.settle_time)
            self._park()
            rospy.sleep(0.5)

        self._print_summary()
        self._save_csv()

    def _park(self):
        """Move drop_uav far above the scene between scenarios."""
        self._teleport(self.pad_center[0], self.pad_center[1], self.park_alt)

    # ── Output ────────────────────────────────────────────────────────────────

    def _log_scenario(self, m: dict):
        rospy.loginfo(
            '[evaluator] ✓  #%-2d  '
            'fall=%.2fs  speed=%.2f m/s  KE=%.1f J  KE_z=%.1f J  '
            'tilt=%.1f°  miss=%.2f m  score=%.1f/100',
            m['scenario'],
            m['fall_time_s'],
            m['impact_speed_ms'],
            m['KE_total_J'],
            m['KE_vertical_J'],
            m['tilt_deg'],
            m['contact_dist_m'],
            m['landing_score'])

    def _print_summary(self):
        if not self._results:
            rospy.loginfo('[evaluator] no completed scenarios')
            return

        bar  = '═' * 100
        thin = '─' * 100
        hdr  = (f'{"#":>3}  {"fall(s)":>7}  {"speed":>7}  '
                f'{"KE(J)":>7}  {"KE_z(J)":>7}  '
                f'{"tilt°":>6}  {"miss(m)":>7}  {"score":>6}  '
                f'{"drop offset XY":>16}')

        rospy.loginfo('')
        rospy.loginfo(bar)
        rospy.loginfo('  UAV DROP TEST — LANDING QUALITY EVALUATION')
        rospy.loginfo('  Pad centre: (%.2f, %.2f, %.2f)   '
                      'Drone mass: %.1f kg   '
                      'Drop height: %.1f m',
                      self.pad_center[0], self.pad_center[1], self.pad_center[2],
                      self.drone_mass, self.drop_height)
        rospy.loginfo(bar)
        rospy.loginfo(hdr)
        rospy.loginfo(thin)

        for m in self._results:
            rospy.loginfo(
                '%3d  %7.2f  %7.2f  %7.2f  %7.2f  %6.1f  %7.3f  %6.1f  '
                '(%.2f, %.2f)',
                m['scenario'],
                m['fall_time_s'],
                m['impact_speed_ms'],
                m['KE_total_J'],
                m['KE_vertical_J'],
                m['tilt_deg'],
                m['contact_dist_m'],
                m['landing_score'],
                m['contact_dx'],
                m['contact_dy'])

        rospy.loginfo(thin)

        n = len(self._results)
        def avg(key): return sum(m[key] for m in self._results) / n

        rospy.loginfo(
            'AVG  %7.2f  %7.2f  %7.2f  %7.2f  %6.1f  %7.3f  %6.1f',
            avg('fall_time_s'),
            avg('impact_speed_ms'),
            avg('KE_total_J'),
            avg('KE_vertical_J'),
            avg('tilt_deg'),
            avg('contact_dist_m'),
            avg('landing_score'))

        rospy.loginfo(bar)
        rospy.loginfo('  Completed: %d / %d scenarios', n, self.n_scenarios)
        rospy.loginfo('  Results saved to: %s', self.csv_path)
        rospy.loginfo(bar)
        rospy.loginfo('')

    def _save_csv(self):
        if not self._results:
            return
        try:
            os.makedirs(os.path.dirname(self.csv_path) or '.', exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(self._results)
            rospy.loginfo('[evaluator] CSV written: %s', self.csv_path)
        except IOError as e:
            rospy.logerr('[evaluator] failed to write CSV: %s', e)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        node = LandingEvaluatorNode()
        # Run ROS callbacks in background thread so run() can block
        spinner = threading.Thread(target=rospy.spin, daemon=True)
        spinner.start()
        node.run()
    except rospy.ROSInterruptException:
        pass
