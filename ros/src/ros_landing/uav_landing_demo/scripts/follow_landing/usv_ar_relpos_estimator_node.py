#!/usr/bin/env python3
"""
usv_ar_relpos_estimator_node.py  (tmp / dev-only)
─────────────────────────────────────────────────
Task 20 — GPS-FREE relative-pose estimator for the Otter USV from the **nested AR
board** (outer AprilTag 36h11 ID=10 @ 2.0 m, inner ArUco 4x4_50 ID=1 @ 0.2 m).

This is the AR sibling of `tmp/usv_relpos_estimator_node.py` (YOLO).  It produces
the USV position *relative to the UAV* in the SAME GPS-free, gravity-aligned,
IMU-yaw, UAV-centred ENU "level" frame, so the downstream UKF → Bézier → follow
controller can consume it unchanged via the selector (Task 22).

Why GPS-free (unlike the repo's aruco_pose_estimator_node.py)
────────────────────────────────────────────────────────────
The repo node builds the camera pose from `/<ns>/ground_truth` (a GPS leak) and
publishes in the camera-optical frame.  Here we instead build the camera→world
rotation R_opt from the **UAV IMU attitude + gimbal joint FK** (drone at origin),
exactly as the YOLO estimator does, then rotate the solvePnP result into the
common level frame.  No GPS, no rangefinder — solvePnP gives the marker range
directly, so the AR sources are independent of (and usually sharper than) the
YOLO ray-cast that relies on the rangefinder/water-plane assumption.

Per-detection pipeline (per marker)
  1. detectMarkers (both dictionaries), keep ID=10 (outer) and ID=1 (inner).
  2. solvePnP IPPE_SQUARE → (rvec, tvec): marker pose in camera optical frame.
  3. R_marker_in_cam · Rz(+π/2) = R_usv_in_cam   (URDF visual rpy="0 0 π/2").
  4. R_opt, p_opt from gimbal FK (IMU + joints, drone at origin).
  5. panel_world = p_opt + R_opt·tvec                       (AR-panel centre)
     R_usv_world = R_opt · R_usv_in_cam                     (USV orient in level frame)
     usv_world   = panel_world − R_usv_world·_AR_CENTER_IN_USV   (→ USV base/waterline)
        → matches the YOLO estimator's reference point so source handovers don't jump.
  6. quality = mean reprojection error [px] → an approximate horizontal position
     variance (px error projected to the ground at the marker range), stuffed into
     the covariance so the selector can quality-gate / a later filter can fuse.

The measurement CONTRACT (Task 19), shared by all three producers
  topic  /<ns>/usv_relpos/{ar_inner, ar_outer, yolo}
  type   geometry_msgs/PoseWithCovarianceStamped
  frame  <ns>/base_link_world_aligned   (UAV-centred, gravity-aligned, IMU-yaw ENU)
  pose.position      = USV (base/waterline) relative to UAV [m]  (dx, dy, dz)
  pose.orientation   = USV heading about +Z (yaw) in the level frame
  covariance diag    = [var_xy, var_xy, var_z, 0, 0, var_yaw]   (lower = better)
  header.stamp       = source image stamp (freshness for the selector)
The selector (Task 22) applies inner>outer>YOLO with asymmetric frame-count
hysteresis and republishes the winner as PoseStamped on /<ns>/usv_relpos/estimate
(the existing downstream contract) + a std_msgs/String /<ns>/usv_relpos/source.

Run (via the follow bring-up):
  python3 tmp/usv_ar_relpos_estimator_node.py _ns:=uav1
"""
import math
import threading

import numpy as np
import rospy
import cv2
import cv_bridge
from scipy.spatial.transform import Rotation as Rot

from sensor_msgs.msg import Image, CameraInfo, Imu, JointState
from geometry_msgs.msg import PoseWithCovarianceStamped

# ── Marker geometry ───────────────────────────────────────────────────────────
_DICT_ARUCO = cv2.aruco.DICT_4X4_50
_DICT_APRIL = cv2.aruco.DICT_APRILTAG_36h11
_OUTER_ID = 10   # AprilTag 36h11, 2.0 m
_INNER_ID = 1    # ArUco 4x4_50,  0.2 m
_OUTER_SIZE_DEFAULT = 2000.0 / 2500.0 * 2.5   # 2.000 m
_INNER_SIZE_DEFAULT = 200.0 / 2500.0 * 2.5    # 0.200 m

# AR-panel visual centre in the Otter ar_code (root) link frame.
_AR_CENTER_IN_USV = np.array([-0.15, 0.0, 0.75], dtype=np.float64)

# ── Gimbal FK constants — identical chain to usv_relpos_estimator_node.py ──────
_HALF_PI = math.pi / 2.0
_R_OPT = Rot.from_euler('z', -_HALF_PI) * Rot.from_euler('x', -_HALF_PI)
_R_MARKER_TO_USV = Rot.from_euler('z', +_HALF_PI)
_YAW_OFF = np.array([0.0, 0.0, -0.025])
_ROLL_OFF = np.array([0.0, 0.0, -0.030])
_PITCH_OFF = np.array([0.0, 0.0, -0.025])
_OPT_OFF = np.array([0.025, 0.0, 0.0])


def _camera_fk(drone_pos, drone_rot, base_off, yaw, roll, pitch):
    """Gimbal FK → (p_opt, R_opt) in a UAV-centred world-aligned frame."""
    p, R = drone_pos + drone_rot.apply(base_off), drone_rot
    p, R = p + R.apply(_YAW_OFF), R * Rot.from_euler('z', yaw)
    p, R = p + R.apply(_ROLL_OFF), R * Rot.from_euler('x', roll)
    p, R = p + R.apply(_PITCH_OFF), R * Rot.from_euler('y', pitch)
    p, R = p + R.apply(_OPT_OFF), R * _R_OPT
    return p, R


def _obj_pts(half):
    """solvePnP object points (TL,TR,BR,BL) for a flat square marker."""
    return np.array([[-half, half, 0.0], [half, half, 0.0],
                     [half, -half, 0.0], [-half, -half, 0.0]], dtype=np.float64)


class UsvArRelPosEstimator:
    def __init__(self):
        rospy.init_node('usv_ar_relpos_estimator')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns

        img_topic = rospy.get_param('~image_topic', '/%s/overhead_cam/image_raw' % ns)
        cinfo_topic = rospy.get_param('~camera_info_topic',
                                      '/%s/overhead_cam/camera_info' % ns)
        imu_topic = rospy.get_param('~imu_topic', '/%s/mavros/imu/data' % ns)
        js_topic = rospy.get_param('~joint_states_topic', '/%s/gimbal/joint_states' % ns)

        self.base_off = np.array([
            rospy.get_param('~base_x_offset', 0.10),
            rospy.get_param('~base_y_offset', 0.00),
            rospy.get_param('~base_z_offset', 0.00)])
        self._sizes = {
            _OUTER_ID: float(rospy.get_param('~outer_size_m', _OUTER_SIZE_DEFAULT)),
            _INNER_ID: float(rospy.get_param('~inner_size_m', _INNER_SIZE_DEFAULT))}
        self._max_reproj = float(rospy.get_param('~max_reproj_px', 12.0))
        # resolution-limited position-noise coefficient [px]: a marker of apparent
        # size s_px localises to ~k_size/s_px of its own size; projected to the
        # ground this is k_size·range²/(fx·size_m).  Makes the BIG outer tag rank
        # far better than the small inner tag (measured: outer ~0.36 m vs inner
        # ~0.61 m @12 m), so the selector prefers outer whenever it is in-FOV.
        self._k_size = float(rospy.get_param('~k_size_px', 1.2))

        # ── detectors ─────────────────────────────────────────────────────────
        params = cv2.aruco.DetectorParameters()
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 53
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.02
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._april_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_APRIL), params)
        self._aruco_det = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(_DICT_ARUCO), params)

        # ── live sensor state ─────────────────────────────────────────────────
        self._lock = threading.Lock()
        self._bridge = cv_bridge.CvBridge()
        self.K = None
        self.D = None
        self.R_wb = None         # body→world from IMU
        self.gimbal = None       # (yaw, roll, pitch)

        # ── pub/sub ───────────────────────────────────────────────────────────
        self.pub = {
            _OUTER_ID: rospy.Publisher('/%s/usv_relpos/ar_outer' % ns,
                                       PoseWithCovarianceStamped, queue_size=5),
            _INNER_ID: rospy.Publisher('/%s/usv_relpos/ar_inner' % ns,
                                       PoseWithCovarianceStamped, queue_size=5)}
        rospy.Subscriber(cinfo_topic, CameraInfo, self._cb_cinfo, queue_size=1)
        rospy.Subscriber(imu_topic, Imu, self._cb_imu, queue_size=5)
        rospy.Subscriber(js_topic, JointState, self._cb_js, queue_size=5)
        rospy.Subscriber(img_topic, Image, self._cb_img,
                         queue_size=1, buff_size=2 ** 24)

        self._n = {_OUTER_ID: 0, _INNER_ID: 0}
        rospy.loginfo('[ar_relpos] ns=%s outer=%.2fm inner=%.2fm  img=%s',
                      ns, self._sizes[_OUTER_ID], self._sizes[_INNER_ID], img_topic)

    # ── sensor callbacks ──────────────────────────────────────────────────────
    def _cb_cinfo(self, msg):
        if msg.K[0] > 0 and self.K is None:
            with self._lock:
                self.K = np.array(msg.K, dtype=np.float64).reshape(3, 3)
                self.D = np.array(msg.D, dtype=np.float64)
            rospy.loginfo_once('[ar_relpos] intrinsics fx=%.1f', msg.K[0])

    def _cb_imu(self, msg):
        q = msg.orientation
        with self._lock:
            self.R_wb = Rot.from_quat([q.x, q.y, q.z, q.w])

    def _cb_js(self, msg):
        d = dict(zip(msg.name, msg.position))
        try:
            y = d['%s_gimbal_yaw_joint' % self._ns]
            r = d['%s_gimbal_roll_joint' % self._ns]
            p = d['%s_gimbal_pitch_joint' % self._ns]
        except KeyError:
            if len(msg.position) >= 3:
                y, r, p = msg.position[0], msg.position[1], msg.position[2]
            else:
                return
        with self._lock:
            self.gimbal = (y, r, p)

    # ── core: detect + PnP + transform to the common level frame ──────────────
    def _cb_img(self, msg):
        with self._lock:
            K, D, R_wb, gimbal = self.K, self.D, self.R_wb, self.gimbal
        if K is None or R_wb is None or gimbal is None:
            rospy.logwarn_throttle(5.0, '[ar_relpos] waiting for cam_info/imu/gimbal …')
            return
        try:
            bgr = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr('[ar_relpos] cv_bridge: %s', e)
            return

        p_opt, R_opt = _camera_fk(np.zeros(3), R_wb, self.base_off, *gimbal)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        ac, aids, _ = self._april_det.detectMarkers(gray)
        rc, rids, _ = self._aruco_det.detectMarkers(gray)
        dets = []
        if aids is not None:
            dets += [(c, int(i[0])) for c, i in zip(ac, aids) if int(i[0]) == _OUTER_ID]
        if rids is not None:
            dets += [(c, int(i[0])) for c, i in zip(rc, rids) if int(i[0]) == _INNER_ID]

        for corners, mid in dets:
            half = self._sizes[mid] / 2.0
            obj = _obj_pts(half)
            ok, rvec, tvec = cv2.solvePnP(obj, corners[0].astype(np.float64),
                                          K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            t = tvec.flatten()
            dist = float(np.linalg.norm(t))
            if dist < 0.05 or dist > 300.0:
                continue

            # reprojection error [px] → quality
            proj, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
            reproj = float(np.mean(np.linalg.norm(
                proj.reshape(-1, 2) - corners[0].astype(np.float64), axis=1)))
            if reproj > self._max_reproj:
                rospy.logdebug('[ar_relpos] id=%d reproj=%.1fpx > %.1f, skip',
                               mid, reproj, self._max_reproj)
                continue

            # USV pose in the common level frame
            R_usv_in_cam = Rot.from_matrix(cv2.Rodrigues(rvec)[0]) * _R_MARKER_TO_USV
            R_usv_world = R_opt * R_usv_in_cam
            panel_world = p_opt + R_opt.apply(t)
            usv_world = panel_world - R_usv_world.apply(_AR_CENTER_IN_USV)
            heading = float(R_usv_world.as_euler('xyz')[2])

            # quality → horizontal sigma: resolution term (grows with range,
            # shrinks with marker size → outer ≪ inner) combined with the
            # reprojection term (spikes when the marker clips the FOV edge so a
            # degraded outer no longer out-ranks inner).
            fx = float(K[0, 0])
            size_m = self._sizes[mid]
            sigma_res = self._k_size * dist * dist / (fx * size_m)
            sigma_rep = reproj * dist / fx
            sigma_xy = max(0.01, math.hypot(sigma_res, sigma_rep))   # [m]
            var_xy = sigma_xy * sigma_xy
            var_z = (2.0 * sigma_xy) ** 2
            # yaw uncertainty shrinks with apparent marker size
            marker_px = fx * size_m / max(dist, 1e-3)
            var_yaw = (max(0.01, 6.0 / max(marker_px, 1.0))) ** 2

            self._publish(mid, msg.header.stamp, usv_world, heading,
                          var_xy, var_z, var_yaw)
            self._n[mid] += 1

        if (self._n[_OUTER_ID] + self._n[_INNER_ID]) % 60 == 0 and \
                (self._n[_OUTER_ID] + self._n[_INNER_ID]) > 0:
            rospy.loginfo_throttle(2.0, '[ar_relpos] outer=%d inner=%d',
                                   self._n[_OUTER_ID], self._n[_INNER_ID])

    def _publish(self, mid, stamp, pos, heading, var_xy, var_z, var_yaw):
        m = PoseWithCovarianceStamped()
        m.header.stamp = stamp
        m.header.frame_id = '%s/base_link_world_aligned' % self._ns
        m.pose.pose.position.x = float(pos[0])
        m.pose.pose.position.y = float(pos[1])
        m.pose.pose.position.z = float(pos[2])
        q = Rot.from_euler('z', heading).as_quat()
        m.pose.pose.orientation.x = float(q[0])
        m.pose.pose.orientation.y = float(q[1])
        m.pose.pose.orientation.z = float(q[2])
        m.pose.pose.orientation.w = float(q[3])
        cov = [0.0] * 36
        cov[0] = var_xy      # xx
        cov[7] = var_xy      # yy
        cov[14] = var_z      # zz
        cov[35] = var_yaw    # yaw-yaw
        m.pose.covariance = cov
        self.pub[mid].publish(m)


if __name__ == '__main__':
    try:
        UsvArRelPosEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
