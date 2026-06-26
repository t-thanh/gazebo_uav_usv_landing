#!/usr/bin/env python3
"""
usv_perception_selector_node.py  (tmp / dev-only)
─────────────────────────────────────────────────
Task 22 — UNIFIED PERCEPTION selector.  Fuses the three GPS-free USV relative-pose
PRODUCERS into the single measurement the downstream stack already consumes, so
the UKF → Bézier → follow controller stay untouched.

Inputs  (geometry_msgs/PoseWithCovarianceStamped, shared contract — Task 19)
    /<ns>/usv_relpos/ar_inner   inner ArUco 4x4  (best at close range)
    /<ns>/usv_relpos/ar_outer   outer AprilTag    (range; whole flyable envelope)
    /<ns>/usv_relpos/yolo       YOLO ray-cast     (always-available fallback)
Outputs
    /<ns>/usv_relpos/estimate   geometry_msgs/PoseStamped   ← downstream contract
    /<ns>/usv_relpos/source     std_msgs/String             (selected source name)
    /<ns>/usv_relpos/visible    std_msgs/Bool              (any source fresh)

Selection — best-performance order OUTER > INNER > YOLO with ASYMMETRIC,
quality-aware hysteresis (grounded by measurement: the big outer tag is the more
accurate source whenever it is in-FOV and only clips out at very close range,
which is exactly where the inner tag takes over):

  • PROMOTE UP (e.g. inner→outer when the outer tag re-enters the FOV climbing
    back through ~5 m, or yolo→outer) only when the higher-priority source has been
    present for `promote_frames` recent frames (N-of-M via a freshness-gapped
    streak) AND its horizontal variance is no worse than `quality_factor`× the
    current source's.  The variance is marker-size-honest (outer ≪ inner) and
    spikes when a marker clips the FOV edge, so a DEGRADED outer near its clip
    altitude does not out-rank a clean inner — the loop keeps inner for landing.
  • DEMOTE DOWN only when the current source goes stale > `fresh_timeout`
    (not on a single dropped frame) → when the outer tag clips out at close range
    the loop falls to inner; a brief dropout does not chatter.
  • Falling DOWN in priority needs no streak gate (inner / YOLO are the reliable
    fallbacks when the better source is unavailable); only promotions are gated.

The UKF absorbs the small residual step at a handover (all sources share the
common level frame and the same USV reference point).

Run:  python3 tmp/usv_perception_selector_node.py _ns:=uav1
"""
import threading

import rospy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import String, Bool

INNER, OUTER, YOLO, BLOB, THERMAL = 'ar_inner', 'ar_outer', 'yolo', 'blob', 'thermal'
# Best-PERFORMANCE order (measured): the big OUTER AprilTag is the more accurate
# source whenever it is in-FOV (≈≥5 m, error ~0.2–0.4 m); the small INNER ArUco is
# the CLOSE-RANGE source for when the outer tag clips out of frame (≤~4–5 m with a
# realistic following offset); BLOB (B/W texture of the pad) is the ROBUST coarse
# fallback that REPLACES YOLO — it keeps a fix (and so keeps the gimbal pointed,
# breaking the perception→gimbal→perception loss spiral) when AR decoding drops.
PRIORITY = [OUTER, INNER, BLOB, YOLO]   # highest → lowest (best → fallback)


class _Src:
    __slots__ = ('pose', 'var_xy', 'last_recv', 'streak')

    def __init__(self):
        self.pose = None        # geometry_msgs/Pose
        self.var_xy = float('inf')
        self.last_recv = -1e9
        self.streak = 0


class PerceptionSelector:
    def __init__(self):
        rospy.init_node('usv_perception_selector')
        ns = str(rospy.get_param('~ns', 'uav1')).strip('/')
        self._ns = ns

        self.fresh_timeout = float(rospy.get_param('~fresh_timeout', 0.40))   # s
        self.promote_frames = int(rospy.get_param('~promote_frames', 8))
        self.quality_factor = float(rospy.get_param('~quality_factor', 2.0))
        self.rate_hz = float(rospy.get_param('~rate', 50.0))
        # YOLO ray-cast is UNRELIABLE at altitude — measured 19–71 m RMS error at 8–12 m
        # (vs outer-AR 0.24–2.0 m), the source of the horizon-garbage that drove the UAV
        # off-target.  Dropped from the fusion by default: OUTER-AR is the reliable
        # workhorse at ~10 m, INNER-AR the close-range backup, and UWB takes over <5 m.
        # ~enable_yolo:=true restores it (e.g. the unified-vs-YOLO baseline).
        self._enable_yolo = bool(rospy.get_param('~enable_yolo', False))
        # BLOB (B/W pad texture) is the robust coarse fallback below AR — replaces YOLO.
        self._enable_blob = bool(rospy.get_param('~enable_blob', True))
        # THERMAL (Hadron 640R sim): decode-free warm-pad detector — the long-range/search/backup
        # source.  LOWEST priority (below UWB near-field + both AR sources); replaces blob/yolo.
        self._enable_thermal = bool(rospy.get_param('~enable_thermal', False))
        self.PRIORITY = [OUTER, INNER]
        if self._enable_blob:
            self.PRIORITY.append(BLOB)
        if self._enable_yolo:
            self.PRIORITY.append(YOLO)
        if self._enable_thermal:
            self.PRIORITY.append(THERMAL)
        # FORCE_SOURCE (ar_inner|ar_outer|yolo) pins the selector to one source,
        # bypassing the hysteresis — used for the YOLO-only baseline run in the
        # unified-vs-YOLO following comparison (Task 25/26).  Empty = normal fusion.
        self.force_source = str(rospy.get_param('~force_source', '')).strip() or None
        if self.force_source and self.force_source not in self.PRIORITY:
            rospy.logwarn('[selector] ignoring invalid force_source=%s', self.force_source)
            self.force_source = None

        self._lock = threading.Lock()
        self._src = {s: _Src() for s in self.PRIORITY}
        self._sel = None
        self._last_log = None

        self.pub_est = rospy.Publisher('/%s/usv_relpos/estimate' % ns,
                                       PoseStamped, queue_size=5)
        self.pub_src = rospy.Publisher('/%s/usv_relpos/source' % ns,
                                       String, queue_size=2)
        self.pub_vis = rospy.Publisher('/%s/usv_relpos/visible' % ns,
                                       Bool, queue_size=2)

        rospy.Subscriber('/%s/usv_relpos/ar_inner' % ns, PoseWithCovarianceStamped,
                         lambda m: self._cb(INNER, m), queue_size=5)
        rospy.Subscriber('/%s/usv_relpos/ar_outer' % ns, PoseWithCovarianceStamped,
                         lambda m: self._cb(OUTER, m), queue_size=5)
        if self._enable_blob:
            rospy.Subscriber('/%s/usv_relpos/blob' % ns, PoseWithCovarianceStamped,
                             lambda m: self._cb(BLOB, m), queue_size=5)
        if self._enable_yolo:
            rospy.Subscriber('/%s/usv_relpos/yolo' % ns, PoseWithCovarianceStamped,
                             lambda m: self._cb(YOLO, m), queue_size=5)
        if self._enable_thermal:
            rospy.Subscriber('/%s/usv_relpos/thermal' % ns, PoseWithCovarianceStamped,
                             lambda m: self._cb(THERMAL, m), queue_size=5)

        rospy.Timer(rospy.Duration(1.0 / self.rate_hz), self._tick)
        rospy.loginfo('[selector] priority=%s  promote=%d frames  '
                      'fresh_timeout=%.2fs  quality_factor=%.1f',
                      '>'.join(self.PRIORITY), self.promote_frames, self.fresh_timeout,
                      self.quality_factor)

    # ── producer callbacks ────────────────────────────────────────────────────
    def _cb(self, name, msg):
        now = rospy.get_time()
        with self._lock:
            s = self._src[name]
            # streak counts recent frames; a gap > fresh_timeout resets it
            s.streak = s.streak + 1 if (now - s.last_recv) <= self.fresh_timeout else 1
            s.last_recv = now
            s.pose = msg.pose.pose
            s.var_xy = float(msg.pose.covariance[0]) if msg.pose.covariance[0] > 0 \
                else float('inf')

    # ── selection + republish ─────────────────────────────────────────────────
    def _tick(self, _evt):
        now = rospy.get_time()
        with self._lock:
            fresh = {s: (now - self._src[s].last_recv) <= self.fresh_timeout
                     and self._src[s].pose is not None for s in self.PRIORITY}
            elig = [s for s in self.PRIORITY if fresh[s]]
            cur = self._sel

            def gated(s):
                return self._src[s].streak >= self.promote_frames

            if self.force_source is not None:
                # pinned source: select it whenever it is fresh, else nothing
                new_sel = self.force_source if fresh[self.force_source] else None
            elif not elig:
                new_sel = None
            elif cur is None or not fresh[cur]:
                # current died → fall back: highest-priority source that passed its
                # gate; if none gated, the most-reliable eligible (lowest priority).
                g = [s for s in elig if gated(s)]
                new_sel = g[0] if g else elig[-1]
            else:
                # sticky current; consider promoting to a higher-priority source
                new_sel = cur
                cur_var = self._src[cur].var_xy
                for s in self.PRIORITY:
                    if s == cur:
                        break                       # don't look below current
                    if (fresh[s] and gated(s) and
                            self._src[s].var_xy <= cur_var * self.quality_factor):
                        new_sel = s
                        break
            self._sel = new_sel
            pose = self._src[new_sel].pose if new_sel else None
            stamp = rospy.Time.now()

        self.pub_vis.publish(Bool(data=bool(elig)))
        if new_sel is None or pose is None:
            return
        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = '%s/base_link_world_aligned' % self._ns
        ps.pose = pose
        self.pub_est.publish(ps)
        self.pub_src.publish(String(data=new_sel))
        if new_sel != self._last_log:
            rospy.loginfo('[selector] → %s', new_sel)
            self._last_log = new_sel


if __name__ == '__main__':
    try:
        PerceptionSelector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
