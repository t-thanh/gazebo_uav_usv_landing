# Dynamic USV Environment — Demo Report

**Package:** `uav_otter_landing`
**Path:** `src/uav_otter_landing/`
**Purpose:** Gazebo environment for UAV autonomous landing on a moving Otter USV with realistic sea-state disturbances.

---

## Changelog

| Date | Change |
|---|---|
| 2026-05-09 | Initial environment created — Phase 0: moving USV + safe UAV spawn |
| 2026-05-10 | **USV motion reworked**: replaced kinematic `SetWorldPose` with thruster-based control. USV now driven by PI speed + P heading controller publishing `left_thrust_cmd` / `right_thrust_cmd`. Sea-state applied via `/gazebo/apply_body_wrench`. `direct_control_plugin:=false`. |
| 2026-05-10 | **UAV spawn moved laterally**: `uav_y` default increased 8 → 15 m so the drone pad is unambiguously clear of the USV track at all times. |

---

## 1. Motivation

The existing `ar_code_landing` package lands on a **static** USV. To develop a predictive landing system (analogous to the CrazyLand MPPI approach), we need a Gazebo environment where:

1. The Otter USV moves at constant forward speed (1 m/s default)
2. The sea applies sinusoidal heave, roll, and pitch to the hull (within spec: ±10 cm, <10°)
3. The UAV takeoff pad is **not** in the USV's path
4. All motion is deterministic and parameterised so experiments are reproducible

---

## 2. Environment Overview

```
World frame (ENU, +X = East, +Z = Up)
─────────────────────────────────────────────────────────────────────
                                   Drone takeoff pad
                                        ↑
              USV motion →   ───────────────────────────────────
              y = 0          USV starts at x = -30              → x
                                   ↓
                             15 m clearance (y-axis)

                             Drone at (0, 15, 1)
                             (y-offset = 15 m from USV track)
```

**Coordinate convention:** World = `ENU`. USV moves in the `+X` direction at constant speed. The drone takeoff pad sits at `y = 15 m` (default), permanently clear of the USV track at `y = 0`.

---

## 3. Gazebo World

**World file:** `$(find usv_worlds)/worlds/usv_uav.world`
Generated at build time from `usv_uav.world.xacro` → `devel/share/usv_worlds/worlds/usv_uav.world`.

| Element | Setting |
|---|---|
| Physics engine | ODE, `max_step_size = 0.004 s` (250 Hz) |
| Real-time factor | 1× |
| Ocean model | `model://ocean` (Gazebo ocean plugin) |
| Water surface plane | Static ghost collision at `z = 0` (Garmin rangefinder sensor target) |
| Ground plane | At `z = -0.4 m` (grass mesh) |
| MRS plugins | `libMRSGazeboStaticTransformRepublisher.so` |

---

## 4. Models in the Scene

### 4.1 Otter USV

| Property | Value |
|---|---|
| Model name | `otter` |
| URDF | `$(find otter_gazebo)/urdf/otter_gazebo.urdf.xacro` |
| Control mode | **Thruster-based** (`direct_control_plugin:=false`) |
| Spawn position | `(−30, 0, 0)` — 30 m behind origin |
| Sensors enabled | IMU (`imu/data`), GPS, ground truth (`ground_truth_odometry`) |
| Deck top height | `z_deck ≈ 0.35 m` above USV base z |
| Nested ArUco pad | On deck, at `(usv_x, usv_y, 0.355)` |

**Why thruster-based?** The thruster + hydrodynamics + buoyancy pipeline is active, giving physically realistic inertia, drag, and wave-induced contact forces at landing. `usv_motion_node` closes a PI speed loop and P heading loop over the ground truth odometry, publishing to the thruster plugin topics.

**Thruster topics** (subscribed by `libusv_gazebo_thrust_plugin.so`):

| Topic | Type | Range | Note |
|---|---|---|---|
| `left_thrust_cmd` | `std_msgs/Float32` | [−1, 1] | max ±50 N per thruster (linear mapping) |
| `right_thrust_cmd` | `std_msgs/Float32` | [−1, 1] | |

**Thrust allocation** (matches `otter_control` from Aquadrone-USV-Simulator):
```
T = [[50,   50  ],   # surge contribution [N]
     [ 0,    0  ],   # sway (not actuated)
     [-19.5, 19.5]]  # yaw moment [N·m], arm = 0.39 m

[left_cmd, right_cmd] = T_pinv @ [tau_surge, 0, tau_yaw]
```
At 1 m/s steady state (drag ≈ 20 N): `left = right ≈ 0.20`.

### 4.2 X500 UAV (uav1)

| Property | Value |
|---|---|
| Model | MRS X500 |
| Spawn command | `1 x500 --enable-rangefinder --enable-ground-truth --pos 0 8 1 0` |
| Spawn position | `(0, 8, 1)` — 8 m lateral from USV track |
| Sensors | Garmin downward rangefinder, ground truth odometry |
| Controller | PX4 SITL via MAVROS |
| Namespace | `uav1` |

**Separation from USV track:** 8 m (`uav_y = 8.0` default). The USV moves along `y = 0`; the drone takeoff pad is at `y = 8`. No collision at any simulation time.

### 4.3 Gimbal (uav1_gimbal)

| Property | Value |
|---|---|
| Model | `$(find uav_gimbal)/models/gimbal_ghadron_standalone/model.sdf` |
| Spawn position | `(0, 8, 1)` — same XY as drone |
| Controller | `gimbal_controllers.launch` (ns=`uav1`) |
| Tracking | Position node reads `uav1/ground_truth/odom`, calls `set_link_state` |

The gimbal is a **standalone kinematic model** — no physical joint to the drone. The `gimbal_position_node` keeps it co-located with the drone via `set_link_state` at 30 Hz.

---

## 5. USV Motion Model

Implemented in `scripts/usv_motion_node.py`. Control loop runs at **50 Hz**.

### 5.1 Forward motion — PI speed + P heading controller

```
e_u   = u_d − u_body          (body-frame surge error)
I_u  += e_u · dt              (integrator, anti-windup ±5 N·s)
τ_surge = Kp_u · e_u + Ki_u · I_u

e_ψ  = wrap(ψ_d − ψ)         (heading error, wrapped to [-π, π])
τ_yaw   = Kp_ψ · e_ψ

[left_cmd, right_cmd] = clip(T_pinv @ [τ_surge, 0, τ_yaw], −1, 1)
```

`u_body` is the body-frame surge speed projected from the world-frame velocity
in `ground_truth_odometry`:  `u = vx·cos(ψ) + vy·sin(ψ)`.

### 5.2 Sea-state — sinusoidal body wrench

Applied via `/gazebo/apply_body_wrench` at 50 Hz.  Each call covers 1.5 × dt so
there is no gap between applications.

```
F_z(t) = heave_force_amp  × sin(2π · f_h · t + φ_h)   [N, world +Z]
τ_x(t) = roll_torque_amp  × sin(2π · f_r · t + φ_r)   [N·m, ≈ body roll]
τ_y(t) = pitch_torque_amp × sin(2π · f_p · t + φ_p)   [N·m, ≈ body pitch]
```

Force/torque amplitudes are tunable parameters.  The hydrodynamic damping in
`usv_gazebo_dynamics_plugin` damps the response, so the achieved deck motion
amplitude is typically less than what a simple mass × acceleration estimate predicts.
**Tune by monitoring `/uav_otter_landing/usv_odom` position/orientation.**

### 5.3 Default parameter values

| Parameter | YAML key | Default | Units |
|---|---|---|---|
| Target speed | `usv_speed` | **1.0** | m/s |
| Target heading | `usv_heading` | 0.0 | rad |
| Speed P gain | `Kp_speed` | 2.0 | N/(m/s) |
| Speed I gain | `Ki_speed` | 0.5 | N/(m·s) |
| Heading P gain | `Kp_heading` | 3.0 | N·m/rad |
| Heave force amp | `heave_force_amp` | **100.0** | N |
| Heave frequency | `heave_hz` | 0.40 | Hz |
| Heave phase | `heave_phase` | 0.0 | rad |
| Roll torque amp | `roll_torque_amp` | **30.0** | N·m |
| Roll frequency | `roll_hz` | 0.30 | Hz |
| Roll phase | `roll_phase` | 0.7 | rad |
| Pitch torque amp | `pitch_torque_amp` | **60.0** | N·m |
| Pitch frequency | `pitch_hz` | 0.50 | Hz |
| Pitch phase | `pitch_phase` | 1.3 | rad |
| Gazebo body name | `gazebo_body_name` | `otter::base_link` | — |
| Publish rate | `publish_rate` | 50 | Hz |
| Debug output | `publish_debug` | true | — |

### 5.4 Sea-state constraint targets

| Constraint | Spec | Forcing amp (default) | Monitor topic |
|---|---|---|---|
| Heave | ≤ ±10 cm | 100 N | `usv_odom/pose/pose/position/z` |
| Roll | < ±10° | 30 N·m | `usv_odom/pose/pose/orientation` |
| Pitch | < ±10° | 60 N·m | `usv_odom/pose/pose/orientation` |

Actual motion depends on hydrodynamic damping. Reduce amplitude parameters until
the deck stays within spec.

### 5.5 Scenario timing

| Time | Event |
|---|---|
| t = 0 s | Simulation starts; USV spawned at x = −30 m, thrusters begin |
| t ≈ 5–10 s | USV accelerates to 1 m/s (PI transient) |
| t = 30 s | USV reaches x ≈ 0 (drone intercept zone) at cruise speed |
| t = 30–60 s | Drone intercepts, follows, and lands |

---

## 6. ROS Topics

### Published by `usv_motion_node`

| Topic | Type | Rate | Description |
|---|---|---|---|
| `left_thrust_cmd` | `std_msgs/Float32` | 50 Hz | Left thruster command [−1, 1] |
| `right_thrust_cmd` | `std_msgs/Float32` | 50 Hz | Right thruster command [−1, 1] |
| `/uav_otter_landing/usv_odom` | `nav_msgs/Odometry` | 50 Hz | Ground truth re-published (debug) |
| `/uav_otter_landing/thrust_cmd` | `geometry_msgs/Vector3Stamped` | 50 Hz | Debug: x=left, y=right, z=speed_error |

### Service called by `usv_motion_node`

| Service | Type | Rate | Description |
|---|---|---|---|
| `/gazebo/apply_body_wrench` | `gazebo_msgs/ApplyBodyWrench` | 50 Hz | Sinusoidal heave/roll/pitch forcing |

### Subscribed by `usv_motion_node`

| Topic | Type | Description |
|---|---|---|
| `ground_truth_odometry` | `nav_msgs/Odometry` | Otter world pose + velocity |

### Consumed by `libusv_gazebo_thrust_plugin.so` (inside Gazebo)

| Topic | Notes |
|---|---|
| `left_thrust_cmd` | Linear map to body force; cmd=1.0 → 50 N forward |
| `right_thrust_cmd` | Same; cmd timeout = 1 s |

### Key UAV topics (from MRS / MAVROS)

| Topic | Type | Description |
|---|---|---|
| `/uav1/ground_truth` | `nav_msgs/Odometry` | Drone ground truth pose + velocity |
| `/uav1/mavros/state` | `mavros_msgs/State` | Armed / mode status |
| `/uav1/garmin/range` | `sensor_msgs/Range` | Downward rangefinder [0–40 m] |
| `/uav1/gimbal/joint_states` | `sensor_msgs/JointState` | Gimbal yaw/pitch angles |

---

## 7. Launch File

**File:** `launch/moving_usv_env.launch`

### Arguments

| Arg | Default | Description |
|---|---|---|
| `gui` | `true` | Show Gazebo GUI |
| `verbose` | `false` | Gazebo verbose mode |
| `usv_speed` | `1.0` | USV forward speed [m/s] |
| `usv_start_x` | `−30.0` | USV spawn X position [m] |
| `uav_y` | `8.0` | Drone takeoff Y offset from USV track [m] |

### Node graph

```
Gazebo (usv_uav.world, 250 Hz physics)
  ├─ otter          URDF model with direct_control_plugin
  │    └─ /usv_data ←── usv_motion_node (100 Hz)
  ├─ uav1           MRS X500 via mrs_drone_spawner
  │    ├─ PX4 SITL + MAVROS
  │    ├─ Garmin rangefinder → /uav1/garmin/range
  │    └─ Ground truth → /uav1/ground_truth
  └─ uav1_gimbal    standalone gimbal model
       └─ gimbal_controllers
            ├─ gimbal_position_node  (tracks drone via set_link_state)
            └─ gimbal_attitude_node  (yaw/pitch PID)
```

### Launch sequence

```
1. kill_previous_session   — clean up prior Gazebo/PX4/MAVROS processes
2. empty_world.launch      — start Gazebo with usv_uav.world
3. gz_logger_suppress      — reduce INFO flood from gazebo_ros_api_plugin
4. usv_spawner             — spawn Otter at (usv_start_x, 0, 0) with direct_control
5. rob_st_pub              — robot_state_publisher for Otter TF
6. mrs_drone_spawner       — MRS spawner ready
7. spawn_drone             — X500 at (0, uav_y, 1) with rangefinder + ground truth
8. spawn_gimbal            — standalone gimbal model at drone XY
9. gimbal_controllers      — gimbal position + attitude controllers
10. usv_motion             — USV kinematic motion at 100 Hz
```

---

## 8. How to Run

### 8.1 Build

```bash
cd /home/t-thanh/Garage/uav_usv_sim
catkin build uav_otter_landing
source devel/setup.bash
```

### 8.2 Launch (default settings)

```bash
roslaunch uav_otter_landing moving_usv_env.launch
```

### 8.3 Common overrides

```bash
# Faster USV, more buffer time
roslaunch uav_otter_landing moving_usv_env.launch usv_speed:=1.5 usv_start_x:=-45.0

# Headless (CI / recording)
roslaunch uav_otter_landing moving_usv_env.launch gui:=false

# Wider lateral separation
roslaunch uav_otter_landing moving_usv_env.launch uav_y:=12.0

# Calm sea (debug controller only)
# Edit config/usv_motion_params.yaml: set heave_amp: 0.0, roll_amp_deg: 0.0, pitch_amp_deg: 0.0
```

### 8.4 Override sea state at runtime (no rebuild needed)

```bash
# Edit the YAML then restart the node only:
rosnode kill /usv_motion
rosrun uav_otter_landing usv_motion_node.py
```

---

## 9. Verification Procedure

After launching, run these checks before starting controller development.

### 9.1 USV is moving forward at ~1 m/s

```bash
rqt_plot /uav_otter_landing/usv_odom/twist/twist/linear/x
```
Expected: rises from 0 to ~1 m/s within 10 s (PI transient), then holds steady.

```bash
rqt_plot /uav_otter_landing/usv_odom/pose/pose/position/x
```
Expected: increases roughly linearly; slope ≈ 1 m/s after transient.

### 9.2 USV is heaving (vertical oscillation)

```bash
rqt_plot /uav_otter_landing/usv_odom/pose/pose/position/z
```
Expected: oscillates around 0 at ~0.40 Hz. Target amplitude ≤ ±0.10 m.
If amplitude is too large, reduce `heave_force_amp` in the YAML.

### 9.3 Sea-state roll and pitch

```bash
rqt_plot /uav_otter_landing/usv_odom/pose/pose/orientation/x:y
```
Expected: quaternion x (roll proxy) oscillates at ~0.30 Hz; y (pitch proxy) at ~0.50 Hz.
Convert to degrees: `rpy = 2 × arcsin(q_x)` ≈ q_x × 114.6° for small angles.

### 9.4 Thruster commands

```bash
rqt_plot /uav_otter_landing/thrust_cmd/vector/x:y:z
```
Expected: x and y (left/right) settle near 0.20 at cruise; z (speed error) → 0.

```bash
rostopic hz /left_thrust_cmd
```
Expected: ~50.0 Hz.

### 9.5 apply_body_wrench is being called

```bash
rostopic echo /rosout | grep "apply_body_wrench\|usv_motion"
```
If the service call fails, check that the Otter model spawned with its base link
named `otter::base_link`.  Verify with:
```bash
gz model -m otter -i 2>/dev/null | grep "base_link"
```

### 9.6 Drone is NOT on the USV track

```bash
rostopic echo /uav1/ground_truth -n1 | grep "position"
```
Expected: `y ≈ 8.0` (the spawn y-offset). The USV track is at `y = 0`.

### 9.7 Gimbal following drone

```bash
rostopic echo /uav1/gimbal/joint_states -n1
```
Expected: joint positions change as Gazebo starts up; no NaN values.

### 9.8 No collision between USV and drone

At t = 30 s the USV passes through x = 0 (where the drone may be hovering).
The drone is at y = 8 m; the Otter hull half-width is ~0.54 m → 7.46 m clearance.
Monitor with:
```bash
rostopic echo /gazebo/model_states | grep -A2 "otter"
```

---

## 10. Known Limitations and Next Steps

| Item | Status | Notes |
|---|---|---|
| USV moves in straight line only | Current | Heading arg `usv_heading` available but not wired to trajectory curve |
| Angular velocity set to zero in plugin | By design | Roll/pitch appear as pose steps, not continuous rotation — angular vel for physics contacts is 0 |
| Garmin hits water plane, not USV deck | Needs fix in alt controller | `water_surface_sensor_plane` at z=0 is detected first when drone is far from USV |
| ArUco marker on Otter deck | Not yet confirmed visible | Requires gimbal tracker + ArUco detector test |

### Planned phases

- **Phase 1 — USV State Estimator:** Velocity estimation from ArUco/gimbal tracker observations via linear regression; dead-reckoning on detection loss
- **Phase 2 — Landing Condition Monitor:** Multi-gate commit detector (Δv_horiz, heave_vel, tilt, position error)
- **Phase 3 — State Machine:** INTERCEPT → FOLLOW_FAR → FOLLOW_CLOSE → COMMIT_LAND states extending `ar_landing_sm_node`
- **Phase 4 — Predictive Trajectory:** `UsvPredictiveTrajectory` fed into existing `uav_traj_landing` MPC controller

---

## 11. File Reference

```
src/uav_otter_landing/
├── CMakeLists.txt
├── package.xml
├── config/
│   └── usv_motion_params.yaml      ← all sea-state and motion parameters
├── launch/
│   └── moving_usv_env.launch       ← full environment launch
├── reports/
│   └── dynamic_env_demo.md         ← this file
└── scripts/
    └── usv_motion_node.py          ← kinematic USV driver (100 Hz)
```
