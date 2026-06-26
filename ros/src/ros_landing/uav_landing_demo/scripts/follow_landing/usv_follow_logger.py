#!/usr/bin/env python3
"""
usv_follow_logger.py  (tmp / dev-only)
───────────────────────────────────────
Closed-loop evaluation logger for the GPS-free USV follow demo.  Samples, at a
fixed rate, the raw ray-cast estimate, the CTRV-UKF filtered relative pose +
velocity, the Bézier look-ahead, the follow controller state, and the GPS
ground truth, and writes one CSV row per tick.

Ground truth (/<ns>/ground_truth, /p3d) is logged ONLY for scoring — it never
enters any estimate or the control loop.

The headline closed-loop metric is `follow_err` = horizontal distance between the
UAV and the USV (ground truth) — i.e. how well the UAV stays directly above the
USV.  Estimation accuracy is `filt_err_horiz` (UKF filtered vs GT relative offset).

Run:  python3 tmp/usv_follow_logger.py _ns:=uav1 _csv_path:=tmp/eva_results/follow.csv
"""
import csv
import math
import os
import threading

import numpy as np
import rospy
from scipy.spatial.transform import Rotation as Rot

from std_msgs.msg import Bool, String
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Vector3Stamped


def _yaw(q):
    return Rot.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')[2]


class FollowLogger:
    def __init__(self):
        rospy.init_node('usv_follow_logger')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns
        self._rate = float(rospy.get_param('~rate_hz', 20.0))
        state_topic = rospy.get_param('~state_topic', '/usv_follow_controller/state')
        herr_topic = rospy.get_param('~herr_topic',
                                     '/usv_follow_controller/horizontal_error')

        self._lock = threading.Lock()
        self._est = None       # raw ray-cast (x,y,heading)
        self._filt = None      # (x,y,vx,vy,yaw,omega)
        self._look = None      # (x,y)
        self._visible = False
        # per-source producer estimates (each (dx,dy) + receipt time) so the
        # analysis can compare inner/outer/yolo accuracy vs altitude over one run.
        self._src_fresh = float(rospy.get_param('~src_fresh_s', 0.30))
        self._src_pose = {'ar_inner': None, 'ar_outer': None, 'yolo': None}
        self._src_t = {'ar_inner': -1e9, 'ar_outer': -1e9, 'yolo': -1e9}
        self._sel_src = ''     # which source the selector chose
        self._fstate = ''
        self._ctrl_e = float('nan')
        self._uav = None       # (x,y,z)
        self._usv = None       # (x,y,z,yaw)
        self._usv_rp = None    # (roll,pitch) from the real USV IMU (/imu/data, 15 Hz)
        self._t0 = None
        # ── landing-quality snapshot state (computed at the touchdown instant) ──────
        self._uav_vel = None; self._uav_z = None       # UAV world velocity + body-z axis
        self._uav_att = None                            # (roll, pitch, tilt_deg) per-tick attitude
        self._uav_yaw = None                            # UAV heading [rad] (touchdown yaw misalignment)
        self._usv_vel = None; self._usv_z = None        # deck world velocity + normal (z)
        self._t_commit = None                           # time entering COMMIT
        self._td_done = False
        self._mass = float(rospy.get_param('~uav_mass_kg', 2.0))    # X500 ~2 kg
        self._imu_log_secs = float(rospy.get_param('~imu_log_secs', 30.0))  # cap roll/pitch log

        self.csv_path = os.path.abspath(rospy.get_param('~csv_path',
                                        'tmp/eva_results/follow.csv'))
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        self._f = open(self.csv_path, 'w', newline='')
        self._cols = [
            't', 'fstate', 'visible',
            'est_dx', 'est_dy', 'est_heading',
            'filt_dx', 'filt_dy', 'filt_vx', 'filt_vy', 'filt_speed',
            'filt_yaw', 'filt_omega',
            'look_dx', 'look_dy',
            'gt_dx', 'gt_dy', 'gt_horiz', 'gt_speed', 'gt_yaw',
            'follow_err',                      # = gt_horiz (UAV-over-USV horiz dist)
            'filt_err_dx', 'filt_err_dy', 'filt_err_horiz',
            'ctrl_e_horiz',
            'uav_x', 'uav_y', 'uav_z', 'uav_roll', 'uav_pitch', 'uav_tilt',
            'usv_x', 'usv_y', 'usv_z',
            'usv_roll', 'usv_pitch',           # deck roll/pitch [rad] (wave disturbance)
            # unified perception: selected source + per-source estimate & error vs
            # GT (blank = that source not detected this tick).  uav_z ≈ altitude.
            'sel_src',
            'in_dx', 'in_dy', 'in_err',
            'out_dx', 'out_dy', 'out_err',
            'yolo_dx', 'yolo_dy', 'yolo_err',
        ]
        self._w = csv.DictWriter(self._f, fieldnames=self._cols)
        self._w.writeheader()
        self._f.flush()
        self._n = 0
        self._prev_usv = None
        self._prev_usv_t = None
        self._gt_speed = 0.0

        rospy.Subscriber('/%s/usv_relpos/estimate' % ns, PoseStamped, self._cb_est, queue_size=10)
        rospy.Subscriber('/%s/usv_relpos/ar_inner' % ns, PoseWithCovarianceStamped,
                         lambda m: self._cb_src('ar_inner', m), queue_size=10)
        rospy.Subscriber('/%s/usv_relpos/ar_outer' % ns, PoseWithCovarianceStamped,
                         lambda m: self._cb_src('ar_outer', m), queue_size=10)
        rospy.Subscriber('/%s/usv_relpos/yolo' % ns, PoseWithCovarianceStamped,
                         lambda m: self._cb_src('yolo', m), queue_size=10)
        rospy.Subscriber('/%s/usv_relpos/source' % ns, String, self._cb_sel, queue_size=5)
        rospy.Subscriber('/%s/usv_track/filtered' % ns, Odometry, self._cb_filt, queue_size=10)
        rospy.Subscriber('/%s/usv_track/lookahead' % ns, PoseStamped, self._cb_look, queue_size=10)
        rospy.Subscriber('/%s/usv_track/visible' % ns, Bool, self._cb_vis, queue_size=10)
        rospy.Subscriber('/%s/ground_truth' % ns, Odometry, self._cb_uav, queue_size=5)
        rospy.Subscriber('/p3d', Odometry, self._cb_usv, queue_size=5)
        rospy.Subscriber(rospy.get_param('~usv_imu_topic', '/imu/data'),
                         Imu, self._cb_usv_imu, queue_size=20)
        rospy.Subscriber(state_topic, String, self._cb_state, queue_size=5)
        rospy.Subscriber(herr_topic, Vector3Stamped, self._cb_herr, queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / self._rate), self._tick)
        rospy.on_shutdown(self._close)
        rospy.loginfo("[follow_log] logging → %s", self.csv_path)

    def _cb_est(self, m):
        with self._lock:
            self._est = (m.pose.position.x, m.pose.position.y, _yaw(m.pose.orientation))

    def _cb_src(self, name, m):
        with self._lock:
            self._src_pose[name] = (m.pose.pose.position.x, m.pose.pose.position.y)
            self._src_t[name] = rospy.Time.now().to_sec()

    def _cb_sel(self, m):
        with self._lock:
            self._sel_src = m.data

    def _cb_filt(self, m):
        vx, vy = m.twist.twist.linear.x, m.twist.twist.linear.y
        with self._lock:
            self._filt = (m.pose.pose.position.x, m.pose.pose.position.y, vx, vy,
                          _yaw(m.pose.pose.orientation), m.twist.twist.angular.z)

    def _cb_look(self, m):
        with self._lock:
            self._look = (m.pose.position.x, m.pose.position.y)

    def _cb_vis(self, m):
        with self._lock:
            self._visible = bool(m.data)

    def _cb_state(self, m):
        with self._lock:
            self._fstate = m.data
            now = rospy.Time.now().to_sec()
            if m.data == 'COMMIT' and self._t_commit is None:
                self._t_commit = now
            if m.data in ('TOUCHDOWN', 'LANDED') and not self._td_done:
                self._td_done = True
                self._write_landing_metrics(now)

    def _write_landing_metrics(self, t_td):
        """Snapshot the landing-quality metrics at the touchdown instant → <csv>_metrics.csv:
          dist_center  : horizontal touchdown offset to the pad/USV centre [m]
          v_rel,v_rel_z: |UAV−deck| relative speed (3-D) and its vertical component [m/s]
          energy_J     : collision energy ½·m·|v_rel|²  (m=uav_mass_kg)
          tilt_deg     : angle between the UAV body-z and the deck normal at contact [deg]
          commit_to_td : time from entering COMMIT to TOUCHDOWN [s] (the critical descent)
          v_rel_h      : horizontal (skid) component of the relative speed [m/s]
          yaw_misalign : UAV heading vs deck heading at contact [deg]
          deck_heave_vz: deck (USV) vertical velocity at contact [m/s] (+ up into UAV, − down away)
        """
        uav, usv = self._uav, self._usv
        if uav is None or usv is None:
            return
        dist = math.hypot(uav[0] - usv[0], uav[1] - usv[1])
        vrel = vrelz = vrelh = energy = float('nan')
        if self._uav_vel is not None and self._usv_vel is not None:
            dv = self._uav_vel - self._usv_vel
            vrel = float(np.linalg.norm(dv)); vrelz = float(dv[2])
            vrelh = float(math.hypot(dv[0], dv[1]))          # horizontal skid at contact
            energy = 0.5 * self._mass * vrel * vrel
        tilt = float('nan')
        if self._uav_z is not None and self._usv_z is not None:
            c = float(np.clip(np.dot(self._uav_z, self._usv_z), -1.0, 1.0))
            tilt = math.degrees(math.acos(c))
        # yaw misalignment: UAV heading vs deck heading, wrapped to [0,180]
        yaw_mis = float('nan')
        if self._uav_yaw is not None:
            dyaw = (self._uav_yaw - usv[3] + math.pi) % (2.0 * math.pi) - math.pi
            yaw_mis = abs(math.degrees(dyaw))
        # deck-heave phase: the deck's own vertical velocity at contact (+ rising into the UAV)
        heave_vz = float(self._usv_vel[2]) if self._usv_vel is not None else float('nan')
        ctd = (t_td - self._t_commit) if self._t_commit is not None else float('nan')
        path = self.csv_path[:-4] + '_metrics.csv' if self.csv_path.endswith('.csv') \
            else self.csv_path + '_metrics.csv'
        try:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['dist_center_m', 'v_rel_ms', 'v_rel_z_ms', 'v_rel_h_ms', 'energy_J',
                            'tilt_deg', 'yaw_misalign_deg', 'deck_heave_vz_ms', 'commit_to_td_s'])
                w.writerow([round(dist, 4), round(vrel, 4), round(vrelz, 4), round(vrelh, 4),
                            round(energy, 4), round(tilt, 3), round(yaw_mis, 3),
                            round(heave_vz, 4), round(ctd, 3)])
            rospy.loginfo("[logger] LANDING METRICS: dist=%.2fm v_rel=%.2f(vz=%.2f h=%.2f) E=%.2fJ "
                          "tilt=%.1f° yaw=%.1f° heave_vz=%.2f commit→td=%.1fs", dist, vrel, vrelz,
                          vrelh, energy, tilt, yaw_mis, heave_vz, ctd)
        except Exception as e:
            rospy.logwarn("[logger] metrics write failed: %s", e)

    def _cb_herr(self, m):
        with self._lock:
            self._ctrl_e = float(m.vector.z)

    def _cb_uav(self, m):
        p = m.pose.pose.position
        v = m.twist.twist.linear
        q = m.pose.pose.orientation
        R = Rot.from_quat([q.x, q.y, q.z, q.w])
        roll, pitch, yaw = R.as_euler('xyz')
        bz = R.apply([0.0, 0.0, 1.0])
        with self._lock:
            self._uav = (p.x, p.y, p.z)
            self._uav_vel = np.array([v.x, v.y, v.z])
            self._uav_z = bz                           # UAV body-z in world (for tilt angle)
            self._uav_yaw = yaw                         # UAV heading (for touchdown yaw misalignment)
            # per-tick attitude for control-effort / attitude-profile metrics (FOLLOW + LANDING):
            # roll, pitch, and tilt = deviation of body-z from world-vertical [deg].
            self._uav_att = (roll, pitch,
                             math.degrees(math.acos(float(np.clip(bz[2], -1.0, 1.0)))))

    def _cb_usv_imu(self, m):
        # deck roll/pitch + normal from the REAL Otter IMU (/imu/data, 15 Hz) — captures the
        # wave roll/pitch that /p3d (2 Hz) aliases.  Used for the wave profile + touchdown angle.
        q = m.orientation
        R = Rot.from_quat([q.x, q.y, q.z, q.w])
        roll, pitch, _ = R.as_euler('xyz')
        with self._lock:
            self._usv_rp = (roll, pitch)
            self._usv_z = R.apply([0.0, 0.0, 1.0])     # deck normal in world

    def _cb_usv(self, m):
        p = m.pose.pose.position
        yaw = _yaw(m.pose.pose.orientation)
        now = rospy.Time.now().to_sec()
        with self._lock:
            if self._prev_usv is not None and self._prev_usv_t is not None:
                dtp = now - self._prev_usv_t
                if dtp > 1e-3:
                    self._gt_speed = math.hypot(p.x - self._prev_usv[0],
                                                p.y - self._prev_usv[1]) / dtp
            self._prev_usv = (p.x, p.y)
            self._prev_usv_t = now
            self._usv = (p.x, p.y, p.z, yaw)
            v = m.twist.twist.linear
            self._usv_vel = np.array([v.x, v.y, v.z])   # deck world velocity (heave incl.)

    def _tick(self, _e):
        now = rospy.Time.now().to_sec()
        if self._t0 is None:
            self._t0 = now
        with self._lock:
            est, filt, look = self._est, self._filt, self._look
            visible, fstate, ctrl_e = self._visible, self._fstate, self._ctrl_e
            uav, usv, gt_speed = self._uav, self._usv, self._gt_speed
            sel_src = self._sel_src
            uav_att = self._uav_att
            src_pose = {k: (v if (now - self._src_t[k]) <= self._src_fresh else None)
                        for k, v in self._src_pose.items()}
        row = {c: '' for c in self._cols}
        row['t'] = round(now - self._t0, 3)
        row['fstate'] = fstate
        row['visible'] = int(visible)
        row['ctrl_e_horiz'] = '' if math.isnan(ctrl_e) else round(ctrl_e, 4)

        if est is not None:
            row['est_dx'], row['est_dy'], row['est_heading'] = \
                round(est[0], 4), round(est[1], 4), round(est[2], 5)
        if filt is not None:
            row['filt_dx'], row['filt_dy'] = round(filt[0], 4), round(filt[1], 4)
            row['filt_vx'], row['filt_vy'] = round(filt[2], 4), round(filt[3], 4)
            row['filt_speed'] = round(math.hypot(filt[2], filt[3]), 4)
            row['filt_yaw'], row['filt_omega'] = round(filt[4], 5), round(filt[5], 5)
        if look is not None:
            row['look_dx'], row['look_dy'] = round(look[0], 4), round(look[1], 4)

        gt = None
        if uav is not None and usv is not None:
            gt = (usv[0] - uav[0], usv[1] - uav[1])
            gh = math.hypot(gt[0], gt[1])
            row['gt_dx'], row['gt_dy'], row['gt_horiz'] = \
                round(gt[0], 4), round(gt[1], 4), round(gh, 4)
            row['follow_err'] = round(gh, 4)
            row['gt_speed'], row['gt_yaw'] = round(gt_speed, 4), round(usv[3], 5)
            row['uav_x'], row['uav_y'], row['uav_z'] = [round(v, 4) for v in uav]
            if uav_att is not None:
                row['uav_roll'], row['uav_pitch'], row['uav_tilt'] = \
                    round(uav_att[0], 5), round(uav_att[1], 5), round(uav_att[2], 3)
            row['usv_x'], row['usv_y'], row['usv_z'] = [round(v, 4) for v in usv[:3]]
            # deck roll/pitch profile: log only the first imu_log_secs (default 30 s) — enough
            # to characterise the wave; the touchdown angle uses the snapshot, not this column.
            rp = self._usv_rp
            if rp is not None and (now - self._t0) <= self._imu_log_secs:
                row['usv_roll'], row['usv_pitch'] = round(rp[0], 5), round(rp[1], 5)
            if filt is not None:
                row['filt_err_dx'] = round(filt[0] - gt[0], 4)
                row['filt_err_dy'] = round(filt[1] - gt[1], 4)
                row['filt_err_horiz'] = round(math.hypot(filt[0] - gt[0],
                                                         filt[1] - gt[1]), 4)

        # unified perception: selected source + per-source estimate & error vs GT
        row['sel_src'] = sel_src
        for name, pre in (('ar_inner', 'in'), ('ar_outer', 'out'), ('yolo', 'yolo')):
            p = src_pose.get(name)
            if p is None:
                continue
            row['%s_dx' % pre], row['%s_dy' % pre] = round(p[0], 4), round(p[1], 4)
            if gt is not None:
                row['%s_err' % pre] = round(math.hypot(p[0] - gt[0], p[1] - gt[1]), 4)
        self._w.writerow(row)
        self._n += 1
        if self._n % 20 == 0:
            self._f.flush()

    def _close(self):
        try:
            self._f.flush()
            self._f.close()
            rospy.loginfo("[follow_log] wrote %d rows → %s", self._n, self.csv_path)
        except Exception:
            pass


if __name__ == '__main__':
    try:
        FollowLogger()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
