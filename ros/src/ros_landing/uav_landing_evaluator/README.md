# uav_landing_evaluator

Gazebo drop-test framework for evaluating UAV landing quality on a USV deck.

Drops a free-falling rigid body (representing an X500 UAV) from random positions
above a contact-sensor pad and records, for each scenario:

| Metric | Description |
|--------|-------------|
| **fall_time** | Seconds from release to first contact |
| **impact_speed** | Total velocity at touchdown [m/s] |
| **KE_total** | Total kinetic energy at impact — `0.5 · m · v²` [J] |
| **KE_vertical** | Vertical kinetic energy component — `0.5 · m · vz²` [J] |
| **tilt_deg** | Angle of body z-axis from world vertical [°] at contact |
| **contact_dist** | Horizontal miss distance from pad centre [m] |
| **landing_score** | Composite quality score 0–100 |

Results are printed as a summary table and saved to a CSV file.

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Measurement method](#measurement-method)
   - [Contact detection](#contact-detection)
   - [Velocity at impact](#velocity-at-impact)
   - [Touchdown angle](#touchdown-angle)
   - [Landing quality score](#landing-quality-score)
4. [Gazebo models](#gazebo-models)
5. [Node reference](#node-reference)
6. [Parameter reference](#parameter-reference)
7. [Usage](#usage)
   - [Build](#build)
   - [Run default test](#run-default-test)
   - [Common overrides](#common-overrides)
   - [Interpreting the output](#interpreting-the-output)
8. [Output format](#output-format)
9. [Tuning guide](#tuning-guide)
10. [Troubleshooting](#troubleshooting)

---

## Overview

```
Gazebo world  (drop_test.world — ocean + ODE physics)
│
├── usv_deck            Static platform (2 × 1.08 × 0.35 m) representing
│                       the Otter USV hull.  Deck top at z = 0.35 m.
│
├── landing_pad_sensor  0.8 × 0.8 m red sensor pad on the deck centre.
│                       Contains a Gazebo contact sensor (gazebo_ros_bumper).
│                       Publishes → /landing_pad/contact
│
└── drop_uav            2 kg rigid body (X500 footprint).
                        No propellers, no control — pure gravity free-fall.

landing_evaluator_node
│
├── Buffers /gazebo/model_states at ~100–250 Hz (rolling 1 000-entry window)
├── Blocks until /landing_pad/contact fires (drop_uav touches sensor pad)
├── Looks 80 ms back in the buffer → pre-contact velocity + attitude
├── Computes metrics → logs to console + CSV
└── Repeats for N scenarios, then prints summary table
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Gazebo                                                          │
│                                                                  │
│  /gazebo/model_states ──────────────────────────────────────┐   │
│  (pose + twist for all models, ~100–250 Hz)                 │   │
│                                                             ▼   │
│  landing_pad_sensor                              rolling state  │
│    └─ gazebo_ros_bumper → /landing_pad/contact    buffer        │
│                                │                    │           │
│                                ▼                    │           │
│                          contact event ←────────────┘           │
│                                │                                │
│                         snapshot at t_contact                   │
│                         look back 80 ms → pre-contact state     │
│                                │                                │
│                         compute metrics                         │
│                         reposition drop_uav                     │
│                         repeat N times                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Measurement method

### Contact detection

The `landing_pad_sensor` model contains a Gazebo `contact` sensor wrapped in
the `gazebo_ros_bumper` plugin.  When any collision body touches the sensor
pad, the plugin publishes `gazebo_msgs/ContactsState` on `/landing_pad/contact`.

The evaluator filters messages to those where at least one colliding body name
contains `drop_uav`, so spurious contacts from other models are ignored.

Only the **first** contact per scenario is recorded; subsequent messages while
the body is resting on the pad are discarded.

### Velocity at impact

`/gazebo/model_states` is streamed continuously into a ring buffer of 1 000
entries (≈ 4–10 seconds at typical publish rates).  When contact is detected
at time `t_contact`, the evaluator searches the buffer for the entry closest
to `t_contact − 80 ms`.  This gives the velocity **just before** the collision
response has had time to decelerate the body, avoiding the near-zero velocity
that would be seen at any point during or after contact.

```
                 t_contact − 80 ms
                       ↓
  ─────────────────────●─────── t_contact ───── resting ──────
  buffered states      ↑ pre-contact state used for KE / tilt
```

### Touchdown angle

The tilt angle is the angle between the drone's body z-axis and the world
vertical (z-axis), computed from the body quaternion at the pre-contact state:

```
R     = rotation_matrix(q)
z_b   = R @ [0, 0, 1]        # body z-axis in world frame
tilt  = arccos(z_b · [0, 0, 1])
```

`tilt = 0°` means the drone is perfectly level.  `tilt = 90°` means it is on
its side.

### Landing quality score

```
score = 100 × exp(−α · KE_total)
             × exp(−β · tilt_deg)
             × exp(−γ · contact_dist)
```

| Symbol | Parameter | Default | Effect |
|--------|-----------|---------|--------|
| α | `score_alpha` | 0.002 /J | Penalises hard impacts |
| β | `score_beta`  | 0.04 /° | Penalises tilted landings |
| γ | `score_gamma` | 0.40 /m | Penalises off-centre landings |

**Reference scores for a free-fall from 15 m (KE ≈ 294 J):**

| Tilt | Miss distance | Score |
|------|--------------|-------|
| 0°   | 0.0 m (perfect) | 55.5 / 100 |
| 0°   | 0.5 m | 45.5 / 100 |
| 0°   | 1.0 m | 37.2 / 100 |
| 10°  | 0.0 m | 37.2 / 100 |
| 15°  | 1.0 m | 20.4 / 100 |

The maximum achievable score in a 15 m free-fall is ~55 because exp(−0.002 × 294) ≈ 0.55
— the ceiling set by the impact energy.  A controlled landing at 0.5 m/s
(KE ≈ 0.25 J) can reach 100/100.

To disable the KE penalty and score on geometry only, set `score_alpha: 0.0`.

---

## Gazebo models

### `drop_uav`

Free-falling rigid body representing an X500-class UAV.  No control, no
propellers.  Falls under gravity alone.

| Property | Value |
|----------|-------|
| Mass | 2.0 kg |
| Footprint | 0.55 × 0.55 m |
| Body height | 0.15 m |
| Contact stiffness kp | 1 × 10⁵ N/m |
| Visual | Blue-grey box with four arm indicators and motor discs |

### `landing_pad_sensor`

Thin contact-sensor pad spawned on top of the USV deck.

| Property | Value |
|----------|-------|
| Size | 0.8 × 0.8 × 0.01 m |
| Visual | Red square with white corner markers |
| Contact topic | `/landing_pad/contact` (`gazebo_msgs/ContactsState`) |
| Sensor update rate | 250 Hz |

The pad is static (`<static>true</static>`).  It does not interact physically
with the `usv_deck` model (two statics never collide in Gazebo).

### `usv_deck`

Static platform approximating the Otter USV hull geometry.

| Property | Value |
|----------|-------|
| Length | 2.0 m |
| Width | 1.08 m |
| Height | 0.35 m |
| Deck top face | z = 0.35 m (when spawned at world origin) |
| Visual | Grey hull with pontoon details |

---

## Node reference

### `landing_evaluator_node`

**File:** `scripts/landing_evaluator_node.py`

Orchestrates the drop scenarios sequentially in the main thread; ROS
callbacks run in a background spinner thread.

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/gazebo/model_states` | `gazebo_msgs/ModelStates` | Continuous stream buffered for pre-contact state lookup |
| `/landing_pad/contact` | `gazebo_msgs/ContactsState` | Contact event from sensor pad; triggers metric computation |

#### Service calls

| Service | Type | Description |
|---------|------|-------------|
| `/gazebo/set_model_state` | `gazebo_msgs/SetModelState` | Repositions `drop_uav` at the start of each scenario and parks it between runs |

#### Scenario sequence

```
For each scenario i = 1 … N:
  1. Sample random drop position (uniform disc, radius xy_range)
  2. Sample random initial tilt (uniform ±tilt_range_deg for roll and pitch)
  3. call set_model_state → move drop_uav to (x, y, z=pad_z+drop_height)
                            with zero velocity and sampled attitude
  4. Record t_drop = now()
  5. Wait on contact_event (timeout = contact_timeout s)
  6. On contact:
       a. Snapshot the state buffer at t_contact
       b. Find entry closest to t_contact − pre_contact_window
       c. Compute all metrics
       d. Append to results list, log to console
  7. Sleep settle_time s
  8. Park drop_uav at park_altitude
  9. Sleep 0.5 s (let Gazebo settle contact state)

Print summary table → save CSV
```

---

## Parameter reference

All parameters are loaded from `config/evaluator_params.yaml` and can be
overridden from the launch file or on the command line.

### Scenario control

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `n_scenarios` | `5` | — | Number of drop tests |
| `drop_height` | `15.0` | m | Release altitude above `pad_center[2]` |
| `xy_range` | `2.0` | m | Radius of uniform-disc scatter around pad centre |
| `tilt_range_deg` | `15.0` | ° | Maximum random roll/pitch at release |

### Pad geometry

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `pad_center` | `[0.0, 0.0, 0.355]` | m | World-frame centre of `landing_pad_sensor` |
| `drone_mass` | `2.0` | kg | Drone mass for KE computation (must match `drop_uav` model) |

### Timing

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `contact_timeout` | `12.0` | s | Abort scenario if no contact within this time |
| `settle_time` | `2.0` | s | Pause after landing before repositioning |
| `park_altitude` | `100.0` | m | Z height where drop_uav is parked between scenarios |

### State buffer

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `state_buffer_size` | `1000` | entries | Ring buffer depth for `/gazebo/model_states` |
| `pre_contact_window` | `0.08` | s | How far back in buffer to find pre-contact state |

### Landing quality score

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `score_alpha` | `0.002` | /J | KE penalty weight |
| `score_beta` | `0.04` | /° | Tilt penalty weight |
| `score_gamma` | `0.40` | /m | Miss-distance penalty weight |

### Output

| Parameter | Default | Description |
|-----------|---------|-------------|
| `csv_output` | `/tmp/landing_eval_results.csv` | Path for CSV results file |

---

## Usage

### Build

```bash
cd ~/Garage/uav_usv_sim
catkin build uav_landing_evaluator
source devel/setup.bash
```

### Run default test

```bash
roslaunch uav_landing_evaluator landing_evaluator.launch
```

Gazebo opens with the ocean world.  The grey USV deck appears at the origin
with the red landing pad on top.  The evaluator runs 5 drop scenarios
automatically (no further input needed) and prints results at the end.

### Common overrides

**More scenarios, taller drop, headless:**
```bash
roslaunch uav_landing_evaluator landing_evaluator.launch \
  n_scenarios:=20 drop_height:=25.0 gui:=false
```

**Tighter scatter radius (tests near-perfect placement):**
```bash
roslaunch uav_landing_evaluator landing_evaluator.launch \
  xy_range:=0.5 tilt_range_deg:=5.0
```

**Custom CSV output path:**
```bash
roslaunch uav_landing_evaluator landing_evaluator.launch \
  csv_output:=/home/user/results/run_01.csv
```

**Move the pad (e.g., simulate deck at a different position):**
```bash
roslaunch uav_landing_evaluator landing_evaluator.launch \
  pad_x:=2.0 pad_y:=1.0 pad_z:=0.355
```

### Interpreting the output

Each scenario is logged immediately after landing:

```
[evaluator] ✓  #1   fall=1.75s  speed=17.15 m/s  KE=294.2 J  KE_z=290.1 J
                     tilt=7.3°  miss=0.43 m  score=42.8/100
```

At the end, a summary table is printed:

```
════════════════════════════════════════════════════════════════════════════════════
  UAV DROP TEST — LANDING QUALITY EVALUATION
  Pad centre: (0.00, 0.00, 0.36)   Drone mass: 2.0 kg   Drop height: 15.0 m
════════════════════════════════════════════════════════════════════════════════════
  #  fall(s)    speed    KE(J)  KE_z(J)   tilt°   miss(m)   score   drop offset XY
────────────────────────────────────────────────────────────────────────────────────
  1     1.75    17.15   294.22   290.14     7.3    0.430    42.8   (0.31, -0.30)
  2     1.75    17.16   294.58   292.49    13.1    1.221    23.6   (-0.88, -0.85)
  3     1.75    17.15   294.21   289.74     2.8    0.754    36.9   (-0.53, 0.54)
  4     1.76    17.18   295.22   289.92     9.4    1.893    19.4   (1.35, -1.37)
  5     1.75    17.15   294.20   293.51     0.6    0.102    53.8   (-0.07, 0.08)
────────────────────────────────────────────────────────────────────────────────────
AVG     1.75    17.16   294.49   291.16     6.6    0.880    35.3
════════════════════════════════════════════════════════════════════════════════════
```

**What to look for:**
- `KE_total` and `KE_vertical` should be similar (most KE from vertical drop).
  A large `KE_total − KE_vertical` gap means significant lateral velocity — e.g.
  from a miss-aimed drop or horizontal initial velocity.
- `tilt_deg` reflects the random initial tilt assigned at release.  In a free
  fall with no aerodynamics, tilt at contact equals tilt at release.
- `miss (m)` is the horizontal distance of the drone's base_link from the pad
  centre at the moment of first contact.  Values up to `xy_range` are expected.
- `score` should clearly separate scenarios: smaller miss + smaller tilt = higher
  score.  If all scores are identical, check the score weight parameters.

---

## Output format

### Console log

One line per scenario immediately after landing (via `rospy.loginfo`), then
the full table after all scenarios complete.

### CSV file (`/tmp/landing_eval_results.csv`)

One row per completed scenario:

| Column | Type | Description |
|--------|------|-------------|
| `scenario` | int | Scenario number (1-indexed) |
| `drop_x` | float | Drop position X [m] |
| `drop_y` | float | Drop position Y [m] |
| `drop_z` | float | Drop position Z [m] |
| `drop_roll_deg` | float | Initial roll at release [°] |
| `drop_pitch_deg` | float | Initial pitch at release [°] |
| `fall_time_s` | float | Seconds from release to first contact |
| `impact_vx` | float | X velocity at pre-contact state [m/s] |
| `impact_vy` | float | Y velocity at pre-contact state [m/s] |
| `impact_vz` | float | Z velocity at pre-contact state [m/s] (negative = downward) |
| `impact_speed_ms` | float | Total velocity magnitude [m/s] |
| `KE_total_J` | float | Total kinetic energy [J] |
| `KE_vertical_J` | float | Vertical component of KE [J] |
| `tilt_deg` | float | Body tilt from vertical at pre-contact state [°] |
| `contact_dx` | float | X offset of base_link from pad centre at contact [m] |
| `contact_dy` | float | Y offset of base_link from pad centre at contact [m] |
| `contact_dist_m` | float | Horizontal miss distance from pad centre [m] |
| `landing_score` | float | Quality score 0–100 |

Skipped scenarios (timeout, no pre-contact state) are omitted from the CSV.

---

## Tuning guide

### Contact not detected / all scenarios timeout

1. Confirm the bumper plugin is loaded:
   ```bash
   rostopic echo /landing_pad/contact
   # Should print ContactsState messages while something is touching the pad
   ```
2. Check that `landing_pad_sensor` spawned correctly:
   ```bash
   rostopic echo /gazebo/model_states | grep landing_pad
   ```
3. Verify `pad_center` matches the spawn position used for `landing_pad_sensor`
   in the launch file.  The default is `(0, 0, 0.355)`.

### Pre-contact state always None

The state buffer is empty at the time of contact.  Increase `state_buffer_size`
or decrease `pre_contact_window`:
```yaml
state_buffer_size:  2000
pre_contact_window: 0.05
```
Also confirm `/gazebo/model_states` is publishing at a reasonable rate:
```bash
rostopic hz /gazebo/model_states
```

### All scores look the same

If `xy_range` is very small or `tilt_range_deg` is zero, all scenarios land
nearly identically and scores converge.  Increase scatter:
```bash
roslaunch uav_landing_evaluator landing_evaluator.launch \
  xy_range:=3.0 tilt_range_deg:=20.0
```

Alternatively, if all scores are near zero (KE dominates), reduce `score_alpha`
or disable it:
```yaml
score_alpha: 0.0   # score on geometry only (tilt + miss)
```

### Impact speed lower than expected

Free-fall from height H in vacuum: `v = sqrt(2 · g · H)`.
From 15 m: v ≈ 17.15 m/s.  If measured speed is significantly lower, the
`pre_contact_window` is too large (state is captured too early in the fall).
Reduce it:
```yaml
pre_contact_window: 0.03   # 30 ms before contact
```

### drop_uav misses the pad entirely

The drop_uav has a 0.55 × 0.55 m footprint and the sensor pad is 0.8 × 0.8 m.
If `xy_range` is large relative to the pad, the body can miss.  The contact
sensor covers only the 0.8 m pad; landings outside it will timeout.

Options:
- Reduce `xy_range` to keep all drops within the pad:
  ```yaml
  xy_range: 0.3   # body edge stays within pad for any offset ≤ 0.3 m
  ```
- Increase the pad size by editing `models/landing_pad_sensor/model.sdf`
  (change `<size>0.8 0.8 0.01</size>` to a larger value).

### Gazebo crashes on startup

The world references `model://ocean` which requires the `usv_gazebo_plugins`
package.  If this package is not on the Gazebo model path, remove or comment
out the ocean include in `worlds/drop_test.world`:
```xml
<!-- <include>
  <uri>model://ocean</uri>
</include> -->
```
