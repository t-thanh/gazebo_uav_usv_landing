#!/usr/bin/env python3
"""
demo_annotated_image_node.py — Gimbal camera HUD for moving-USV landing demo.

Subscribes to:
  /<ns>/overhead_cam/image_raw          sensor_msgs/Image
  /<ns>/overhead_cam/camera_info        sensor_msgs/CameraInfo
  /ar_landing/sm/state                  std_msgs/String      (SM state)
  /ar_landing/gimbal_tracker/tracking_status  std_msgs/String  (TRACKING/SEARCHING/LOST)
  /ar_landing/landing_ctrl/horizontal_error   geometry_msgs/Vector3Stamped
  /ar_landing/landing_ctrl/target_altitude    std_msgs/Float64
  /<ns>/garmin/range                    sensor_msgs/Range
  /uav_otter_landing/usv_odom           nav_msgs/Odometry   (USV position debug)

Publishes:
  /uav_otter_landing/demo/annotated_image   sensor_msgs/Image

The node:
  1. Detects outer AprilTag (DICT_APRILTAG_36h11 ID=10) and inner ArUco
     (DICT_4X4_50 ID=1) markers in the raw camera frame.
  2. Draws bounding boxes, corner dots, and marker IDs on the image.
  3. Draws a semi-transparent HUD panel with mission telemetry:
       SM State / Tracking / Horizontal error / Garmin altitude / Target altitude
  4. Publishes the annotated image; launch an image_view to display it.
"""

import math
import threading
import numpy as np
import rospy
import cv2
import cv_bridge

from sensor_msgs.msg     import Image, CameraInfo, Range
from std_msgs.msg        import String, Float64
from geometry_msgs.msg   import Vector3Stamped
from nav_msgs.msg        import Odometry


# ── Marker dictionaries (OpenCV 4.7+ API) ────────────────────────────────────
_DICT_APRIL = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
_DICT_ARUCO = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_DET_PARAMS = cv2.aruco.DetectorParameters()
_DET_APRIL  = cv2.aruco.ArucoDetector(_DICT_APRIL, _DET_PARAMS)
_DET_ARUCO  = cv2.aruco.ArucoDetector(_DICT_ARUCO, _DET_PARAMS)

_OUTER_ID = 10
_INNER_ID = 1

# ── State colour table (BGR) ──────────────────────────────────────────────────
_STATE_COLOR = {
    'PRE_ARMED':    (120, 120, 120),
    'ARMING':       (0,   200, 255),
    'CLIMBING':     (255, 180,   0),
    'GIMBAL_NADIR': (255, 180,   0),
    'ARUCO_SEARCH': (0,   140, 255),
    'ALIGN':        (0,   220, 100),
    'DESCEND':      (0,   255,   0),
    'LANDED':       (0,   255, 100),
    'HOVER':        (255,  80,   0),
    'ABORT':        (0,     0, 255),
}
_DEFAULT_COLOR = (180, 180, 180)

# ── Outer marker box colour (BGR) ─────────────────────────────────────────────
_OUTER_BOX_BGR = (0, 220, 0)    # green
_INNER_BOX_BGR = (0, 180, 255)  # amber


class DemoAnnotatedImageNode:

    def __init__(self):
        rospy.init_node('demo_annotated_image_node', anonymous=False)

        ns = rospy.get_param('~ns', 'uav1')

        self._bridge  = cv_bridge.CvBridge()
        self._lock    = threading.Lock()

        # ── Telemetry state ───────────────────────────────────────────────────
        self._sm_state  = 'PRE_ARMED'
        self._tracking  = 'LOST'
        self._e_horiz   = float('nan')
        self._tgt_alt   = float('nan')
        self._garmin    = float('nan')
        self._usv_x     = float('nan')

        # ── Publisher ─────────────────────────────────────────────────────────
        self._pub = rospy.Publisher(
            '/uav_otter_landing/demo/annotated_image', Image, queue_size=1)

        # ── Subscribers ───────────────────────────────────────────────────────
        rospy.Subscriber(f'/{ns}/overhead_cam/image_raw', Image,
                         self._image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/ar_landing/sm/state',              String,
                         self._sm_cb,       queue_size=5)
        rospy.Subscriber('/ar_landing/gimbal_tracker/tracking_status', String,
                         self._tracking_cb, queue_size=5)
        rospy.Subscriber('/ar_landing/landing_ctrl/horizontal_error', Vector3Stamped,
                         self._herr_cb,     queue_size=5)
        rospy.Subscriber('/ar_landing/landing_ctrl/target_altitude',  Float64,
                         self._tgtalt_cb,   queue_size=5)
        rospy.Subscriber(f'/{ns}/garmin/range',       Range,
                         self._range_cb,    queue_size=5)
        rospy.Subscriber('/uav_otter_landing/usv_odom', Odometry,
                         self._usv_cb,      queue_size=2)

        rospy.loginfo('[demo_annotated_image] Ready — publishing to '
                      '/uav_otter_landing/demo/annotated_image')

    # ── Telemetry callbacks ──────────────────────────────────────────────────

    def _sm_cb(self, msg: String):
        with self._lock:
            self._sm_state = msg.data

    def _tracking_cb(self, msg: String):
        with self._lock:
            self._tracking = msg.data

    def _herr_cb(self, msg: Vector3Stamped):
        with self._lock:
            self._e_horiz = float(msg.vector.z)   # |e_xy| stored in z by controller

    def _tgtalt_cb(self, msg: Float64):
        with self._lock:
            self._tgt_alt = msg.data

    def _range_cb(self, msg: Range):
        with self._lock:
            self._garmin = float(msg.range)

    def _usv_cb(self, msg: Odometry):
        with self._lock:
            self._usv_x = msg.pose.pose.position.x

    # ── Image callback ───────────────────────────────────────────────────────

    def _image_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f'[demo_annotated_image] Bridge error: {e}')
            return

        with self._lock:
            sm_state = self._sm_state
            tracking = self._tracking
            e_horiz  = self._e_horiz
            tgt_alt  = self._tgt_alt
            garmin   = self._garmin
            usv_x    = self._usv_x

        annotated = self._annotate(frame, sm_state, tracking, e_horiz, tgt_alt, garmin, usv_x)

        try:
            out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self._pub.publish(out_msg)
        except cv_bridge.CvBridgeError as e:
            rospy.logwarn_throttle(5.0, f'[demo_annotated_image] Encode error: {e}')

    # ── Annotation ───────────────────────────────────────────────────────────

    def _annotate(self, frame, sm_state, tracking, e_horiz, tgt_alt, garmin, usv_x):
        img   = frame.copy()
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w  = img.shape[:2]

        # ── 1. Detect and draw markers ────────────────────────────────────────
        self._draw_markers(img, gray)

        # ── 2. HUD overlay ────────────────────────────────────────────────────
        self._draw_hud(img, h, w, sm_state, tracking, e_horiz, tgt_alt, garmin, usv_x)

        return img

    # ── Marker detection & drawing ───────────────────────────────────────────

    def _draw_markers(self, img, gray):
        # Outer AprilTag 36h11
        corners_a, ids_a, _ = _DET_APRIL.detectMarkers(gray)
        if ids_a is not None:
            for corner, mid in zip(corners_a, ids_a.flatten()):
                if mid == _OUTER_ID:
                    self._draw_marker_box(img, corner, f'OUTER ID={mid}', _OUTER_BOX_BGR)

        # Inner ArUco 4×4_50
        corners_r, ids_r, _ = _DET_ARUCO.detectMarkers(gray)
        if ids_r is not None:
            for corner, mid in zip(corners_r, ids_r.flatten()):
                if mid == _INNER_ID:
                    self._draw_marker_box(img, corner, f'INNER ID={mid}', _INNER_BOX_BGR)

    @staticmethod
    def _draw_marker_box(img, corner, label, color):
        pts = corner[0].astype(int)   # (4, 2)
        # Draw filled corner circles
        for pt in pts:
            cv2.circle(img, tuple(pt), 5, color, -1)
        # Draw box
        for i in range(4):
            cv2.line(img, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 2)
        # Draw label near top-left corner
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())
        cv2.putText(img, label, (pts[0][0], pts[0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # Draw centroid cross
        cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, 20, 2)

    # ── HUD panel ────────────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(img, h, w, sm_state, tracking, e_horiz, tgt_alt, garmin, usv_x):
        # Panel geometry
        pad   = 10
        lh    = 26          # line height px
        lines = 7
        pw    = 270         # panel width
        ph    = lines * lh + 2 * pad
        x0, y0 = pad, pad   # top-left corner

        # Semi-transparent dark background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + pw, y0 + ph), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        state_color = _STATE_COLOR.get(sm_state, _DEFAULT_COLOR)

        # State banner (filled coloured bar)
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + lh + 4), state_color, -1)
        cv2.putText(img, f'STATE: {sm_state}',
                    (x0 + 6, y0 + lh - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 2, cv2.LINE_AA)

        def put(row, text, color=(230, 230, 230)):
            cv2.putText(img, text,
                        (x0 + 8, y0 + lh + 4 + row * lh + lh - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

        # Tracking status
        trk_color = (0, 220, 80) if tracking == 'TRACKING' else \
                    (0, 180, 255) if tracking == 'SEARCHING' else (60, 60, 255)
        put(0, f'TRACKING: {tracking}', trk_color)

        # Horizontal error
        if not math.isnan(e_horiz):
            err_color = (0, 220, 80) if e_horiz < 0.6 else (60, 60, 255)
            put(1, f'|e_xy|  : {e_horiz:6.2f} m', err_color)
        else:
            put(1, '|e_xy|  : --')

        # Garmin altitude
        if not math.isnan(garmin):
            put(2, f'Garmin  : {garmin:6.2f} m')
        else:
            put(2, 'Garmin  : --')

        # Target altitude
        if not math.isnan(tgt_alt):
            put(3, f'Tgt alt : {tgt_alt:6.2f} m')
        else:
            put(3, 'Tgt alt : --')

        # USV position
        if not math.isnan(usv_x):
            put(4, f'USV x   : {usv_x:6.1f} m')
        else:
            put(4, 'USV x   : --')

        # USV status: HOLDING while near start x (delay active), else moving
        if not math.isnan(usv_x) and usv_x < -3.5:
            put(5, 'USV     : HOLDING (delay)', (0, 180, 255))
        else:
            put(5, 'USV     : moving  →', (180, 180, 180))

        # Border
        cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 1)

    def spin(self):
        rospy.spin()


def main():
    node = DemoAnnotatedImageNode()
    node.spin()


if __name__ == '__main__':
    main()
