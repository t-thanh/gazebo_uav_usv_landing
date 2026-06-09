# ar_code_landing

GPS-denied autonomous landing of an MRS X500 UAV on the Otter USV using nested
ArUco/AprilTag markers as the only lateral position reference.

No GPS at any point.  No MRS control services.  Pure MAVROS attitude + thrust.

---

## Sensor suite

| Sensor | Topic | Used for |
|--------|-------|----------|
| Gimbal camera (overhead) | `/<ns>/overhead_cam/image_raw` | ArUco detection |
| Gimbal joint states | `/<ns>/gimbal/joint_states` | Camera FK (pose in world) |
| Garmin rangefinder | `/<ns>/garmin/range` | Altitude above surface (AGL) |
| UAV IMU (MAVROS) | `/<ns>/mavros/imu/data` | Current yaw for body-frame transform |
| Odometry | `/<ns>/ground_truth` (sim) or `/<ns>/odometry/odom_main` (real) | Drone pose + velocity |
| PX4 EKF velocity | `/<ns>/mavros/local_position/odom` | Horizontal velocity damping (no GPS needed) |

---

## Architecture

```
  ┌──────────────────────────────┐
  │   usv_uav_gimbal_landing     │  Gazebo world  (UAV at x=5)
  └────────────┬─────────────────┘
               │ overhead_cam/image_raw
               ▼
  ┌──────────────────────────────┐
  │  aruco_pose_estimator_node   │  solvePnP IPPE_SQUARE
  │  (otter_usv_detector)        │  outer: AprilTag 36h11 ID=10, 2.0 m
  │                              │  inner: ArUco 4x4_50  ID=1,  0.2 m
  └────────────┬─────────────────┘
               │ /aruco_pose/{outer,inner}/usv_in_cam  (cam optical frame)
               ▼
  ┌──────────────────────────────┐
  │  gimbal_aruco_tracker_node   │  Dual-marker IBVS gimbal PID
  │                              │  Tracks outer marker, falls back to inner
  │                              │  when outer absent > 1 s (close approach)
  │                              │  Publishes world-frame target position
  │                              │  (EMA-smoothed, marker-switch-aware)
  └────────────┬─────────────────┘
               │ /ar_landing/gimbal_tracker/usv_world_pose   (world ENU)
               │ /ar_landing/gimbal_tracker/tracking_status
               ▼
  ┌──────────────────────────────┐
  │  ar_landing_controller_node  │  Attitude + thrust, 50 Hz
  │                              │  Altitude PID: always active (Garmin)
  │                              │  Horizontal PD: world_pose error → roll/pitch
  │                              │  Close range (<2.5 m): gains ×3.0
  │                              │  Stale pose: velocity damping only (no drift)
  └────────────┬─────────────────┘
               │ /<ns>/mavros/setpoint_raw/attitude
               ▼
  ┌──────────────────────────────┐
  │   ar_landing_sm_node         │  State machine, 20 Hz
  │                              │
  │  PRE_ARMED → ARMING          │  SM arms PX4 via pure MAVROS
  │  → CLIMBING                  │  Altitude PID climbs to 10 m
  │  → GIMBAL_NADIR              │  Slew gimbal to nadir
  │  → ARUCO_SEARCH              │  Wait for TRACKING
  │  → ALIGN                     │  ArUco drives lateral correction
  │  → DESCEND                   │  Dynamic-rate descent to deck
  │  → LANDED                    │  AUTO.LAND + PX4 auto-disarm
  │  → HOVER (recovery)          │  ArUco-lost hold, re-enter ALIGN
  └──────────────────────────────┘
```

---

## State machine

| State | Entry condition | Action | Exit condition |
|-------|----------------|--------|----------------|
| **PRE_ARMED** | launch | Publish hover setpoints so PX4 accepts OFFBOARD; set PX4 safety params | sensors ready ≥ 3 s |
| **ARMING** | sensors ready | Request `OFFBOARD` + arm every 2 s (100 Hz poll window to catch 50 ms OFFBOARD window) | `armed AND mode == OFFBOARD` |
| **CLIMBING** | armed in OFFBOARD | `target_alt = 10 m`; altitude PID climbs, level flight | `garmin ≥ 9.5 m` |
| **GIMBAL_NADIR** | at altitude | publish `gimbal_pitch = +π/2` (nadir) | `gimbal_pitch ≥ 1.2 rad` |
| **ARUCO_SEARCH** | gimbal nadir | wait for tracker | `tracking_status == TRACKING` |
| **ALIGN** | deck visible | `target_alt = 5 m`; ArUco drives roll/pitch | `|e_xy| < 0.6 m` for 3 s |
| **DESCEND** | aligned | decrease `target_alt` via dynamic rate formula | `garmin < 0.4 m` |
| **LANDED** | near deck | `set_mode(AUTO.LAND)` → disable controller → wait for PX4 auto-disarm | `armed == False` |
| **HOVER** | ArUco lost in DESCEND | level altitude hold at `hold_alt`; velocity damping | `tracking == TRACKING` → re-ALIGN |
| **ABORT** | timeout / failure | disable controller | manual intervention |

### No goto, no GPS

The SM uses **only MAVROS services**:
- `/<ns>/mavros/cmd/arming` — arm / disarm
- `/<ns>/mavros/set_mode` — enter OFFBOARD / AUTO.LAND mode
- `/<ns>/mavros/param/set` — PX4 safety params at startup

All flight from take-off through touchdown is driven by the attitude controller
reacting to Garmin (altitude) and ArUco world pose (lateral position).

---

## Control law

### Horizontal — `ar_landing_controller_node`

```
usv_world_pos  ← /ar_landing/gimbal_tracker/usv_world_pose  (world ENU)

e_xy_w  = usv_world_pos[:2] − drone_pos[:2]    (world-frame lateral error)
e_xy_b  = Rz(−yaw) @ e_xy_w                   (rotate to body frame)

ax = Kp_xy * e_xy_b[0] − Kd_xy * vel_x_body
ay = Kp_xy * e_xy_b[1] − Kd_xy * vel_y_body

pitch_cmd = clamp( ax / g, ±max_tilt)
roll_cmd  = clamp(−ay / g, ±max_tilt)
```

When garmin < 2.5 m (close approach, inner marker active):
```
Kp_xy_eff = Kp_xy × close_range_xy_gain
Kd_xy_eff = Kd_xy × close_range_xy_gain
```

When tracker world pose is stale (> 2 s): velocity-only damping (no position term):
```
ax = −Kd_xy * vel_x_body    (active braking, prevents drift)
ay = −Kd_xy * vel_y_body
```

### Altitude — `ar_landing_controller_node`

```
e_z    = target_alt − garmin_range
thrust = hover_thrust + Kp_z*e_z + Ki_z*∫e_z dt + Kd_z*(de_z/dt)
thrust = clamp(thrust, hover_thrust ± max_thrust_delta, [0.05, 1.0])
```

Always active.  `target_alt` is published by the SM and rate-limited downward
at 1.0 m/s to prevent sudden target drops from causing hard dives.

### Dynamic descent rate — `ar_landing_sm_node`

Above 2.5 m:
```
rate = descent_rate_ms   (= 0.3 m/s constant)
```

Below 2.5 m (ground-effect zone, inner marker):
```
pos_factor = clip(1.0 − |e_xy| / desc_err_scale_m,  desc_min_pos_factor, 1.0)
alt_factor = clip(garmin / close_descent_alt_m,       desc_min_alt_factor, 1.0)
rate       = descent_rate_ms × pos_factor × alt_factor
```

This gives fast descent when centred, slow descent when off-centre or near deck:

| garmin | \|e_xy\| | pos_factor | alt_factor | rate |
|--------|---------|-----------|-----------|------|
| > 2.5 m | any | — | — | 0.30 m/s |
| 2.0 m | 0.00 m | 1.00 | 0.80 | 0.24 m/s |
| 2.0 m | 0.50 m | 0.05 | 0.80 | 0.012 m/s |
| 0.5 m | 0.00 m | 1.00 | 0.50 | 0.15 m/s |
| any | any (LOST) | 0.05 | × | ≤ 0.075 m/s |

Target altitude floor is **−1.0 m** (not 0.0 m) — see [tuning notes](#target-altitude-floor).

### Gimbal tracking — `gimbal_aruco_tracker_node`

```
v_opt = R_opt^{-1} @ (ar_world − p_opt)   (re-project world pos to current cam)

e_yaw   =  v_opt[0] / v_opt[2]            (normalised angular error)
e_pitch =  v_opt[1] / v_opt[2]

Δyaw    = −PID(e_yaw)      → gimbal_yaw_command
Δpitch  = +PID(e_pitch)    → gimbal_pitch_command
```

Dual-marker: outer marker (2 m) at high altitude, inner marker (0.2 m) when
outer absent > 1 s. EMA filter reset on marker switch to avoid blending
old outer position with new inner.

---

## Quick start

```bash
# Terminal 1 — Gazebo world (UAV spawned at x=5, y=0, z=1)
roslaunch ar_code_landing usv_uav_gimbal_landing.launch

# Terminal 2 — Full landing sequence (arms UAV internally via MAVROS)
roslaunch ar_code_landing ar_code_landing.launch
```

`start_uav.launch` is **not required and must not be run** — it loads the MRS
core stack which intercepts OFFBOARD mode and prevents the SM from working.

Monitor state:
```bash
rostopic echo /ar_landing/sm/state
rostopic echo /ar_landing/landing_ctrl/horizontal_error
rostopic echo /ar_landing/landing_ctrl/altitude_error
rostopic echo /ar_landing/gimbal_tracker/tracking_status
```

---

## Tuning log

All parameters are set in `launch/ar_code_landing.launch`.  This section
documents every tuning decision made during simulation testing, with the
symptom that prompted each change.

### `hover_thrust` — 0.58 → **0.54**

**Symptom:** After arming, the drone climbed aggressively before the altitude
PID could regulate.

**Reason:** The default 0.58 was above the actual hover point of the X500 in
simulation. The SM publishes `hover_thrust` as the pre-arm idle setpoint, so
any value above actual hover causes immediate climb on arming. Lowering to 0.54
(actual hover) means the drone sits level at arming and the PID takes over from
a neutral starting point.

---

### `kp_z` — 0.35 → **0.20**

**Symptom:** Drone overshot to 20 m+ on takeoff.

**Reason:** At arming the initial altitude error is large (e_z ≈ +8.5 m for
`climb_alt = 10 m`).  With `kp_z = 0.35` the P-term alone contributed
`0.35 × 8.5 = +2.98` to thrust — clamped by `max_thrust_delta` but still
produced a hard launch.  Lowering to 0.20 reduces the initial climb impulse.

**Note:** A separate derivative-spike fix (seeding `prev_e_z` with the current
error on controller enable) eliminated the overshoot; `kp_z = 0.20` prevents
any residual aggression.

---

### `kd_z` — 0.20 → **0.60**

**Symptom:** Drone climbed past target altitude, then oscillated around it.

**Reason:** The derivative term (`kd_z × de_z/dt`) brakes the climb as the
drone approaches target altitude.  At 0.20 the braking was too weak — the drone
overshot past 10 m.  Raising to 0.60 (3×) produces hard braking before the
target, arresting the climb without oscillation.

---

### `max_thrust_delta` — 0.25 → 0.12 → 0.20 → **0.30**

**`0.25 → 0.12`**: halved to reduce the maximum climb rate (~1.5 m/s instead of
4+ m/s), working alongside the `kd_z` and `kp_z` reductions to prevent
overshoot.

**`0.12 → 0.20`**: at garmin < 1 m the drone oscillated and refused to reach
the deck.  Ground effect at ~0.85 m raised the effective hover thrust to ~0.42.
With `max_thrust_delta = 0.12` the minimum thrust was `0.54 − 0.12 = 0.42` —
exactly balanced by ground effect.  Raising to 0.20 (min thrust 0.34) gave
room to push through.

**`0.20 → 0.30`**: after the target-floor fix (see below), the PID always hits
the lower clamp through the ground-effect zone.  Lowering the clamp further to
`0.54 − 0.30 = 0.24` provides the constant push-down force needed to reach
garmin < 0.4 m.

---

### Target altitude floor — 0.0 m → **−1.0 m**

**Symptom:** Drone oscillated at ~0.7 m and would not descend to the 0.4 m
touch trigger.

**Root cause:** With `target = 0.0 m`, the altitude PID error shrinks as the
drone approaches the deck (`e_z = 0 − garmin`). At garmin = 0.7 m:
```
thrust = 0.54 + 0.20 × (−0.7) = 0.40
```
Ground effect at 0.7 m makes 0.40 the hover point — the PID had no drive left
to descend further.  The lower clamp (0.34) was not even being reached.

**Fix:** Allow `target_alt` to go to −1.0 m.  With `target = −1.0 m`:
```
e_z at garmin=0.7 m  →  −1.0 − 0.7 = −1.7 m
thrust = 0.54 + 0.20 × (−1.7) = 0.20  →  clamped to lower bound (0.24)
```
The PID always hits the lower clamp through the ground-effect zone, providing a
constant push-down force.  `max_thrust_delta` now controls this directly.

---

### `kp_xy` — 0.40 → 0.50 → **0.80** (with `kd_xy` 0.20 → **0.40**)

**Symptom 1:** Drone drifted laterally during descent and was slow to correct
position errors.

**Symptom 2:** At close range (inner marker), drone tracked but did not move
aggressively enough to centre over the marker before touchdown.

**Reason:** `kp_xy` is the single master gain for all lateral phases; all other
scales (close-range boost, velocity damping) multiply from it.  Raising `kp_xy`
uniformly increases aggressiveness at every altitude while preserving relative
scale ratios.  `kd_xy` was raised proportionally to provide enough velocity
damping to prevent overshoot.

---

### `max_tilt_deg` — 12° → 15° → **25°**

**Symptom:** Drone could not move fast enough laterally to keep up with the USV
at close range.

**Reason:** The maximum roll/pitch angle directly limits lateral acceleration
(`a = g × tan(tilt)`).  Raising the limit allows larger corrections when
the position error is large.

---

### `close_range_xy_gain` — 2.5 → **3.0**

**Symptom:** At garmin < 2.5 m (inner marker phase), lateral corrections were
insufficiently aggressive to centre over the 0.2 m inner marker.

**Reason:** The inner marker subtends a much smaller angle than the outer; small
position errors produce larger fractional deviations.  The close-range
multiplier compensates by boosting `kp_xy` and `kd_xy` by 3× below 2.5 m,
matching the gain to the tighter required precision.

---

### Velocity damping when ArUco stale (code change)

**Symptom:** When the tracker went LOST or SEARCHING during descent, the drone
drifted laterally and quickly left the deck footprint.

**Root cause:** The controller was commanding `roll = pitch = 0` when pose was
stale — no force to oppose existing velocity.

**Fix:** When pose is stale, apply the D-only term using PX4 EKF velocity
(available without GPS):
```python
ax = −Kd_xy × vel_x_body
ay = −Kd_xy × vel_y_body
```
This actively decelerates horizontal drift without requiring a position fix.

---

### Derivative spike fix on controller enable (code change)

**Symptom:** Drone shot to 20 m+ immediately after arming even with conservative
`kp_z`.

**Root cause:** On the first control tick after the controller was enabled,
`prev_e_z = 0` (default), so:
```
d_ez = (e_z − 0) / dt  ≈  8.5 / 0.02  =  425 rad/s
```
The derivative term produced a massive positive thrust spike.

**Fix:** On enable, seed `prev_e_z` with the current error so the first
derivative term is zero:
```python
if self._garmin is not None:
    self._prev_e_z = self._target_alt - self._garmin
```

---

### Rate-limited target altitude — `alt_down_rate_ms` = **1.0 m/s**

**Symptom:** On the CLIMBING → ALIGN transition the target jumped from 10 m to
5 m.  The large negative error (`e_z = −5 m`) immediately drove thrust to the
minimum, causing a fast dive that came close to the sea surface.

**Fix:** The SM separates `_desired_alt` (requested) from `_target_alt`
(commanded).  Downward changes are rate-limited to `alt_down_rate_ms = 1.0 m/s`
so the controller sees a gentle glide rather than a step change.  Upward
transitions are instantaneous (no rate limit on climbs).

---

### Controller target source: raw ArUco → tracker world pose (code change)

**Symptom:** At close range (garmin < 2.5 m), the gimbal switched to the inner
marker and tracked it correctly, but the drone did not move to centre over the
inner marker — it continued drifting as if chasing the outer marker.

**Root cause:** `ar_landing_controller_node` subscribed to
`/aruco_pose/outer/usv_in_cam` (raw camera-frame tvec) and did its own gimbal
FK to compute the target world position.  When the gimbal tracker switched to
the inner marker, the controller was unaware — it kept computing from the outer
marker topic which was no longer publishing fresh detections.

**Fix:** The controller now subscribes to
`/ar_landing/gimbal_tracker/usv_world_pose` instead.  The tracker already
handles outer → inner switching, FK, and EMA smoothing, so the controller
receives the correct landing target at every altitude with no duplication of FK
logic.

---

### `min_altitude_m` (gimbal tracker) — 2.0 → **0.5**

**Symptom:** At garmin ≈ 1.1 m the tracker reported LOST even though the aruco
estimator was publishing inner detections at full frame rate.

**Root cause:** `_accept_detection` gates on `drone_z < min_altitude_m`.  The
USV deck sits ~1 m above world z = 0.  At garmin = 1.1 m, drone world-z ≈
2.1 m — right at the 2.0 m gate.  As the drone descended below ~1.5 m garmin,
world-z dropped under 2.0 m and every inner detection was silently rejected,
causing LOST exactly during the critical final approach.

**Fix:** Lowered to 0.5 m.  This still prevents tracking when the drone is on
the ground (world-z ≈ 0) but allows tracking through the entire final approach.

---

### AUTO.LAND on touchdown (code change, two parts)

**Symptom 1:** After DESCEND → LANDED, PX4 repeatedly rejected disarm with
`"Disarming denied! Not landed"` — its own landing detector had not confirmed
touchdown.

**Fix:** Instead of retrying manual disarm, switch to `AUTO.LAND` mode at
touchdown.  PX4 descends the final 0.4 m under its own control, fires the
landing detector when it senses ground contact, and auto-disarms.  The SM then
monitors `armed == False`.

**Symptom 2:** After switching to `AUTO.LAND`, the drone briefly climbed before
descending.

**Root cause:** The sequence was: (1) disable controller → (2) next SM tick →
(3) call `set_mode(AUTO.LAND)`.  During the ~50 ms gap between (1) and (3),
OFFBOARD setpoints stopped and PX4 fell back to Hold mode (`COM_OBL_ACT = 1`).
In Hold at 0.4 m, PX4 commanded a slight altitude increase.

**Fix:** Call `set_mode(AUTO.LAND)` **before** `enable_ctrl(False)` in the
same DESCEND tick that detects touchdown.  PX4 is already in AUTO.LAND when
setpoints stop, so no Hold transition occurs.

---

## Parameter reference

### ar_landing_controller_node

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hover_thrust` | 0.54 | Actual X500 hover thrust in simulation |
| `kp_xy` | 0.80 | Base lateral P gain (all phases scale from this) |
| `kd_xy` | 0.40 | Lateral velocity damping; also used for stale-pose braking |
| `max_tilt_deg` | 25.0 | Maximum roll/pitch command |
| `kp_z` | 0.20 | Altitude P gain — lowered to prevent aggressive initial climb |
| `ki_z` | 0.01 | Altitude I gain — small to avoid windup |
| `kd_z` | 0.60 | Altitude D gain — raised 3× to brake hard before target |
| `max_thrust_delta` | 0.30 | Thrust authority: range [0.24, 0.84] around hover |
| `close_range_alt_m` | 2.5 | Below this garmin, boost lateral gains |
| `close_range_xy_gain` | 3.0 | Lateral gain multiplier at close range |
| `pose_timeout_s` | 2.0 | Max age of tracker pose before switching to velocity damping |

### ar_landing_sm_node

| Parameter | Value | Notes |
|-----------|-------|-------|
| `hover_thrust` | 0.54 | Used for SM-published hover setpoints during PRE_ARMED/ARMING |
| `climb_alt_m` | 10.0 | Target climb altitude |
| `hold_alt_m` | 5.0 | ALIGN and HOVER hold altitude |
| `align_threshold_m` | 0.6 | |e_xy| threshold to start DESCEND |
| `align_settle_s` | 3.0 | Must hold within threshold for this long |
| `descent_rate_ms` | 0.3 | Base descent rate (modulated by dynamic formula below 2.5 m) |
| `touch_range_m` | 0.4 | Garmin threshold for touchdown detection |
| `alt_down_rate_ms` | 1.0 | Max rate target_alt can decrease (prevents dive on ALIGN entry) |
| `close_descent_alt_m` | 2.5 | Below this, dynamic rate formula applies |
| `desc_err_scale_m` | 0.5 | pos_factor = 0 when \|e_xy\| = this value |
| `desc_min_pos_factor` | 0.05 | Minimum position factor (when LOST or off-centre) |
| `desc_min_alt_factor` | 0.5 | Minimum altitude factor (near deck) |
| `descend_lost_timeout_s` | 4.0 | Seconds without TRACKING before HOVER recovery |

### gimbal_aruco_tracker_node

| Parameter | Value | Notes |
|-----------|-------|-------|
| `min_altitude_m` | 0.5 | Below this world-z, detections rejected (prevents ground confusion) |
| `outer_switch_timeout_s` | 1.0 | Outer must be absent this long before inner accepted |
| `lost_timeout_s` | 3.0 | Seconds after last detection before LOST status |
| `pos_filter_alpha` | 0.4 | EMA weight for new detections (higher = more responsive) |

---

## GPS-denied production: change the odometry source

```xml
<!-- In ar_code_landing.launch, change: -->
<arg name="odom_topic" default="/uav1/odometry/odom_main"/>
```

`odometry/odom_main` is the MRS fused EKF odometry in `local_origin` frame —
consistent with the GPS-denied reference frame used for any goto services.

---

## Package structure

```
ar_code_landing/
├── CMakeLists.txt
├── package.xml
├── README.md
├── config/
│   └── landing_params.yaml
├── launch/
│   ├── ar_code_landing.launch        main — estimator + tracker + controller + SM
│   ├── usv_uav_gimbal_landing.launch  Gazebo world with UAV at x=5
│   └── climb_test.launch             standalone CLIMB verification
└── scripts/
    ├── gimbal_aruco_tracker_node.py   dual-marker ArUco IBVS gimbal controller
    ├── ar_landing_controller_node.py  attitude+thrust landing controller
    ├── ar_landing_sm_node.py          state machine (handles arm + OFFBOARD)
    └── climb_test_node.py             isolated PRE_ARMED→ARMING→CLIMBING test
```

---

## Dependencies

- `otter_usv_detector` — `aruco_pose_estimator_node`
- `uav_gimbal` — `gimbal_controllers.launch`, `gimbal_position_node`
- `ocean_rangefinder` — Garmin range corrector
- MAVROS: `mavros_msgs/{AttitudeTarget,CommandBool,SetMode,ParamSet}`
- Python: `scipy`, `numpy`, `cv2`

No MRS UAV system packages required at runtime.
