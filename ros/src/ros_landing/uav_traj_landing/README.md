# uav_traj_landing

GPS-denied autonomous landing of an X500 UAV on a moving Otter USV using
**time-optimal (or minimum-jerk polynomial) trajectory planning** and a
**pluggable attitude controller** for trajectory tracking.

Three controllers are available for comparison:

| Controller | Launch arg | Description |
|------------|------------|-------------|
| **PID** | `controller_type:=pid` | Classical PID on position/velocity error — simple baseline **(default)** |
| **SMC** | `controller_type:=smc` | Sliding Mode Controller, Super-Twisting Algorithm — des_acc feedforward |
| **MPC** | `controller_type:=mpc` | Linear MPC, condensed QP — full horizon feedforward (pure Python) |

This package is a direct successor to `ar_code_landing`.  The state machine
logic, sensor stack, and gimbal tracker are identical; only the control law
changes — replacing the altitude PID + horizontal PD with a trajectory planner
driving one of the controllers above.  This makes it straightforward to compare
all approaches under identical conditions.

---

## Table of contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Control theory](#control-theory)
   - [Trajectory generation](#trajectory-generation)
   - [PID controller](#pid-controller)
   - [SMC controller](#smc-controller-super-twisting-algorithm)
   - [MPC controller](#mpc-controller-linear-mpc)
   - [Attitude from thrust vector](#attitude-from-thrust-vector-shared)
4. [Nodes](#nodes)
   - [traj_landing_controller_node](#traj_landing_controller_node)
   - [traj_landing_sm_node](#traj_landing_sm_node)
5. [Topic reference](#topic-reference)
6. [Parameter reference](#parameter-reference)
7. [Usage](#usage)
   - [Prerequisites](#prerequisites)
   - [Running the simulation](#running-the-simulation)
   - [Selecting a controller](#selecting-a-controller)
   - [Selecting a trajectory planner](#selecting-a-trajectory-planner)
   - [Monitoring](#monitoring)
8. [Optional: time-optimal trajectories](#optional-time-optimal-trajectories)
9. [Comparison with ar_code_landing](#comparison-with-ar_code_landing)
10. [Controller comparison](#controller-comparison)
11. [Tuning guide](#tuning-guide)
12. [Troubleshooting](#troubleshooting)

---

## Overview

```
Sensors                   Pipeline                            Output
───────                   ────────                            ──────
Garmin rangefinder  ──→   Trajectory planner   ──→  ╮
IMU quaternion      ──→   (polynomial / NLP)         ├─→  AttitudeTarget
EKF velocity        ──→   Pluggable controller  ──→  ╯     (50 Hz MAVROS)
Ground truth pos    ──→     PID  |  SMC  |  MPC
ArUco world pose    ──→   (from gimbal_aruco_tracker, unchanged)
```

The trajectory planner generates a smooth reference path from the drone's
current state to a point directly above the USV deck.  The selected controller
tracks this reference and outputs roll/pitch/thrust for MAVROS AttitudeTarget.

**No GPS is used at any stage.**  Altitude feedback is from the Garmin
rangefinder; horizontal position is from the ArUco-detected USV deck pose;
velocity is from the EKF2 odometry (IMU + barometer only).

---

## Architecture

```
usv_uav_gimbal_landing.launch  (Gazebo + UAV + USV — from ar_code_landing)
│
├── otter_usv_detector          ArUco pose estimation (camera → world)
│     └─ /aruco_pose/{outer,inner}/usv_in_cam  (PoseStamped)
│
├── gimbal_aruco_tracker_node   Gimbal PID + world-frame target pose
│     ↑ subscribes to aruco_pose topics + IMU + gimbal joint states
│     └─ /ar_landing/gimbal_tracker/usv_world_pose  (PoseStamped, world ENU)
│        /ar_landing/gimbal_tracker/tracking_status  (String)
│
├── traj_landing_controller_node   ← NEW (replaces ar_landing_controller_node)
│     ↑ subscribes to usv_world_pose, garmin, IMU, odom, ground_truth
│     Trajectory planner → PID / SMC / MPC (selectable at launch)
│     └─ /<ns>/mavros/setpoint_raw/attitude  (AttitudeTarget, 50 Hz)
│        /traj_landing/ctrl/tracking_error   (Vector3Stamped)
│        /traj_landing/ctrl/status           (String)
│
└── traj_landing_sm_node           ← NEW (replaces ar_landing_sm_node)
      ↑ subscribes to mavros/state, garmin, gimbal_joint_states,
        tracking_status, tracking_error
      └─ /traj_landing/ctrl/enable          (Bool, latched)
         /traj_landing/ctrl/target_altitude (Float64)
         ~state                             (String, latched)
```

---

## Control theory

### Trajectory generation

A **5th-order polynomial (minimum-jerk)** trajectory is generated for each
segment with boundary conditions:

```
t = 0:  pos = p₀ (current),  vel = v₀ (current),  acc = 0
t = T:  pos = pf (target),   vel = 0,              acc = 0
```

Coefficients are solved analytically (closed-form, O(1)):

```
a₀ = p₀,   a₁ = v₀,   a₂ = 0
a₃ = (10·Δp − 6·v₀·T) / T³
a₄ = ( 8·v₀·T − 15·Δp) / T⁴
a₅ = ( 6·Δp  −  3·v₀·T) / T⁵     where Δp = pf − p₀
```

Duration is `T = max(t_min, 1.5 × ‖Δp‖ / v_max)`.

This guarantees:
- Smooth position, velocity, and acceleration (C² continuity)
- Zero velocity and acceleration at the endpoint
- Bounded peak velocity ≈ `v_max`

The trajectory is replanned in a **background thread** whenever the USV deck
has moved more than `replan_distance_m` from the current trajectory endpoint,
or when the trajectory has completed.  The control loop is never blocked.

An optional **time-optimal** NLP planner is also available; see
[Optional: time-optimal trajectories](#optional-time-optimal-trajectories).

---

### PID controller

**File:** `scripts/controllers/pid.py`  
**Default controller.**  Adapted from
`CrazyTraj/CrazyFlow/lsy_drone_racing/control/attitude_controller.py`.

```
e_pos = des_pos − cur_pos
e_vel = des_vel − cur_vel
i_err += e_pos · dt          (clamped to ±ki_range — anti-windup)

target_acc = kp · e_pos  +  ki · i_err  +  kd · e_vel
```

`des_acc` from the trajectory planner is **not used** (no feedforward).
This is intentional: it makes PID the conservative baseline that depends
only on feedback, the same as a classical attitude controller.

Gravity compensation is added before the thrust-vector conversion:
```
target_acc[z] += g
thrust_vec = m · target_acc
```

---

### SMC controller (Super-Twisting Algorithm)

**File:** `scripts/controllers/smc.py`  
Adapted from `CrazyTraj/.../control/sliding_mode_controller.py`.

**Sliding surface:**
```
s = λ · e_pos + e_vel          e_pos = des_pos − cur_pos
                                e_vel = des_vel − cur_vel
```

**Super-Twisting Algorithm (second-order SMC, continuous):**
```
u     = −k1 · |s|^0.5 · sign(s) + w
dw/dt = −k2 · sign(s)
```

`w` is the integral of the switching term, clamped to `±smc_integral_limit`.

**Total desired acceleration (feedforward + STA + gravity):**
```
target_acc = des_acc + u           ← feedforward from trajectory planner
target_acc[z] += g
thrust_vec = m · target_acc
```

Unlike PID, the trajectory's `des_acc` is passed through as feedforward,
reducing the control effort needed from the STA term.

---

### MPC controller (Linear MPC)

**File:** `scripts/controllers/mpc.py`  
Pure Python, no Acados, no compiled code.

**Model:** double integrator (ZOH discretisation, step `dt_mpc`):
```
x = [pos (3), vel (3)]   u = acceleration (3)

x_{k+1} = A x_k + B u_k

A = [[I, dt·I],   B = [[0.5·dt²·I],
     [0,    I]]        [   dt·I   ]]
```

**Cost (condensed QP, receding horizon N steps):**
```
J = Σ_{k=1}^{N-1} (x_k − x_ref_k)ᵀ Q (x_k − x_ref_k)
  + (x_N − x_ref_N)ᵀ Q_N (x_N − x_ref_N)
  + Σ_{k=0}^{N-1} (u_k − u_ref_k)ᵀ R (u_k − u_ref_k)
```

`x_ref_k`, `u_ref_k` are the trajectory planner's position/velocity/acceleration
sampled over the full N-step horizon (passed via `context` dict).

**Condensed form (closed-form, no iterative solver):**
```
X = S_x x₀ + S_u U

H = S_uᵀ Q̄ S_u + R̄      ← precomputed and Cholesky-factorized at init
g = S_uᵀ Q̄ (S_x x₀ − X_ref) − R̄ U_ref   ← assembled online

U* = −H⁻¹ g              ← one triangular solve per 50 Hz tick
u*₀ = U*[0:3]            ← first input applied (receding horizon)
```

The Hessian `H` is computed **once at node startup** (O(N³)).  Each 50 Hz
control tick does one matrix-vector multiply plus one Cholesky triangular
solve: O(N²), well within 20 ms even for N = 20.

---

### Attitude from thrust vector (shared)

All three controllers share the same geometric attitude conversion
(in `scripts/controllers/base.py`):

```
z_desired  = thrust_vec / ‖thrust_vec‖

x_c = [cos(yaw_hold), sin(yaw_hold), 0]ᵀ
y   = z_desired × x_c  (normalised)
x   = y × z_desired
R   = [x | y | z_desired]

[roll, pitch] = RPY(R),  clipped to ±max_tilt_deg
thrust_norm   = ‖thrust_vec‖ / (m·g) × hover_thrust
```

Yaw is held at the IMU-measured current heading.

---

## Nodes

### traj_landing_controller_node

**File:** `scripts/traj_landing_controller_node.py`  
**Rate:** 50 Hz (ROS Timer)

The core attitude controller.  Maintains two operating modes:

| `enabled` | `trajectory` | Behaviour |
|-----------|--------------|-----------|
| `false`   | —            | Publish level hover at `hover_thrust` |
| `true`    | `None`       | Hold current XY position at `target_altitude` |
| `true`    | Active       | Track trajectory with selected controller; replan in background |

On `enable → false` the controller state is reset and the current trajectory
is discarded so the next activation starts clean.

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/ar_landing/gimbal_tracker/usv_world_pose` | `geometry_msgs/PoseStamped` | USV deck position in world ENU frame |
| `/<ns>/garmin/range` | `sensor_msgs/Range` | AGL altitude from Garmin rangefinder |
| `/<ns>/mavros/imu/data` | `sensor_msgs/Imu` | Orientation quaternion + angular velocity |
| `/<ns>/mavros/local_position/odom` | `nav_msgs/Odometry` | EKF2 velocity in body frame (rotated to world internally) |
| `/<ns>/ground_truth` | `nav_msgs/Odometry` | Drone position in world ENU frame |
| `/traj_landing/ctrl/enable` | `std_msgs/Bool` | Enable/disable the controller |
| `/traj_landing/ctrl/target_altitude` | `std_msgs/Float64` | Desired altitude AGL for the trajectory endpoint [m] |

#### Publications

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/<ns>/mavros/setpoint_raw/attitude` | `mavros_msgs/AttitudeTarget` | 50 Hz | Roll/pitch/yaw quaternion + normalised collective thrust [0–1] |
| `/traj_landing/ctrl/status` | `std_msgs/String` | 50 Hz | `IDLE`, `HOLDING`, or `TRACKING` |
| `/traj_landing/ctrl/tracking_error` | `geometry_msgs/Vector3Stamped` | 50 Hz | `vector.x` = e_x [m], `vector.y` = e_y [m], `vector.z` = ‖e_xy‖ [m] |

`AttitudeTarget.type_mask` is always `IGNORE_ROLL_RATE | IGNORE_PITCH_RATE | IGNORE_YAW_RATE`.

---

### traj_landing_sm_node

**File:** `scripts/traj_landing_sm_node.py`  
**Rate:** 10 Hz (ROS Timer)

State machine that sequences arming, climbing, gimbal pointing, target
acquisition, and the landing approach.  Mirrors `ar_landing_sm_node` from
`ar_code_landing` exactly — only the controller topic namespace differs.

#### State flow

```
PRE_ARMED
  │  Stream hover setpoints; set PX4 safety params; wait ≥ 3 s + Garmin ready
  ↓
ARMING
  │  Request OFFBOARD mode + arm via MAVROS; retry every 2 s
  ↓
CLIMBING
  │  target_altitude = climb_alt (10 m); controller climbs via trajectory
  │  exit: garmin ≥ climb_alt − 0.5 m
  ↓
GIMBAL_NADIR
  │  Command gimbal pitch = π/2 (nadir); exit: pitch ≥ 1.2 rad
  ↓
ARUCO_SEARCH
  │  Wait: tracking_status == "TRACKING"
  ↓
TRAJECTORY_ALIGN
  │  target_altitude ramps to hold_alt (5 m); trajectory drives UAV above deck
  │  exit: ‖e_xy‖ < align_threshold (0.6 m) for align_settle_s (3 s)
  ↓
TRAJECTORY_DESCEND
  │  Decrease target_altitude at descent_rate (0.3 m/s); trajectory replans
  │  Adaptive rate near deck (< 2.5 m): slower when off-centre
  │  exit: garmin < touch_range (0.8 m) → set_mode(AUTO.LAND)
  ↓
LANDED

─── Recovery ─────────────────────────────────────────────────────────────────
HOVER      ArUco lost in DESCEND → hold at hold_alt; re-enter ALIGN on reacquire
ABORT      Timeout or fatal error; manual recovery required
```

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/<ns>/mavros/state` | `mavros_msgs/State` | Armed flag + flight mode |
| `/<ns>/garmin/range` | `sensor_msgs/Range` | AGL altitude |
| `/<ns>/mavros/imu/data` | `sensor_msgs/Imu` | Yaw (hover setpoints during arming) |
| `/<ns>/ground_truth` | `nav_msgs/Odometry` | World altitude reference |
| `/<ns>/gimbal/joint_states` | `sensor_msgs/JointState` | Gimbal pitch feedback |
| `/ar_landing/gimbal_tracker/tracking_status` | `std_msgs/String` | `TRACKING`, `SEARCHING`, or `LOST` |
| `/traj_landing/ctrl/tracking_error` | `geometry_msgs/Vector3Stamped` | ‖e_xy‖ from controller (`vector.z`) |

#### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `~state` | `std_msgs/String` (latched) | Current SM state name |
| `/traj_landing/ctrl/enable` | `std_msgs/Bool` (latched) | Enable/disable the controller |
| `/traj_landing/ctrl/target_altitude` | `std_msgs/Float64` | Desired altitude [m] |
| `/<ns>/gimbal/position/pitch/command` | `std_msgs/Float64` | Gimbal pitch [rad] |
| `/<ns>/gimbal/position/yaw/command` | `std_msgs/Float64` | Gimbal yaw [rad] |
| `/<ns>/mavros/setpoint_raw/attitude` | `mavros_msgs/AttitudeTarget` | Hover setpoints during PRE_ARMED / ARMING only |

---

## Topic reference

### Inter-node topics (internal to this package)

| Topic | Direction | Type | Description |
|-------|-----------|------|-------------|
| `/traj_landing/ctrl/enable` | SM → Controller | `std_msgs/Bool` | Activate trajectory tracking |
| `/traj_landing/ctrl/target_altitude` | SM → Controller | `std_msgs/Float64` | Altitude setpoint [m] |
| `/traj_landing/ctrl/tracking_error` | Controller → SM | `geometry_msgs/Vector3Stamped` | Horizontal error (z field = ‖e_xy‖) |
| `/traj_landing/ctrl/status` | Controller → (monitor) | `std_msgs/String` | `IDLE` / `HOLDING` / `TRACKING` |
| `~state` (on `traj_landing_sm`) | SM → (monitor) | `std_msgs/String` (latched) | SM state name |

### External inputs

| Topic | Provider | Type |
|-------|----------|------|
| `/ar_landing/gimbal_tracker/usv_world_pose` | `gimbal_aruco_tracker_node` | `geometry_msgs/PoseStamped` |
| `/ar_landing/gimbal_tracker/tracking_status` | `gimbal_aruco_tracker_node` | `std_msgs/String` |
| `/<ns>/garmin/range` | `ocean_rangefinder` bridge | `sensor_msgs/Range` |
| `/<ns>/mavros/imu/data` | MAVROS | `sensor_msgs/Imu` |
| `/<ns>/mavros/local_position/odom` | MAVROS | `nav_msgs/Odometry` |
| `/<ns>/ground_truth` | Gazebo ground-truth plugin | `nav_msgs/Odometry` |
| `/<ns>/mavros/state` | MAVROS | `mavros_msgs/State` |
| `/<ns>/gimbal/joint_states` | Gazebo gimbal controller | `sensor_msgs/JointState` |

### External outputs

| Topic | Consumer | Type |
|-------|----------|------|
| `/<ns>/mavros/setpoint_raw/attitude` | MAVROS → PX4 | `mavros_msgs/AttitudeTarget` |
| `/<ns>/gimbal/position/pitch/command` | Gimbal PID | `std_msgs/Float64` |
| `/<ns>/gimbal/position/yaw/command` | Gimbal PID | `std_msgs/Float64` |

---

## Parameter reference

All parameters are loaded from `config/landing_params.yaml`.

### Controller node (`traj_landing_controller`)

#### Common parameters

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `ns` | `uav1` | — | ROS namespace of the UAV |
| `drone_mass` | `2.0` | kg | Total vehicle mass |
| `hover_thrust` | `0.58` | [0–1] | Normalised thrust producing 1 g net upward force |
| `max_tilt_deg` | `25.0` | deg | Safety clamp on roll/pitch commands |
| `controller_type` | `pid` | — | Active controller: `pid`, `smc`, or `mpc` |
| `traj_v_max_ms` | `3.0` | m/s | Peak velocity in generated trajectory |
| `traj_t_min_s` | `2.0` | s | Minimum trajectory duration |
| `replan_distance_m` | `0.5` | m | Replan when target moves this far from endpoint |
| `replan_interval_s` | `2.0` | s | Minimum interval between replanning calls |
| `pose_timeout_s` | `3.0` | s | Target pose age beyond which it is considered stale |

#### PID parameters (`controller_type: pid`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `pid_kp` | `[0.4, 0.4, 1.25]` | 1/s² | Proportional gain [x, y, z] |
| `pid_ki` | `[0.05, 0.05, 0.05]` | 1/s | Integral gain |
| `pid_kd` | `[0.2, 0.2, 0.4]` | 1/s | Derivative gain |
| `pid_ki_range` | `[2.0, 2.0, 0.4]` | m/s² | Anti-windup clamp on integral term |

#### SMC parameters (`controller_type: smc`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `smc_lambda` | `[1.5, 1.5, 2.0]` | 1/s | Sliding surface bandwidth [x, y, z] |
| `smc_k1` | `[1.2, 1.2, 1.5]` | m^0.5/s² | STA proportional gain |
| `smc_k2` | `[0.7, 0.7, 1.0]` | m/s³ | STA integral gain |
| `smc_integral_limit` | `2.0` | m/s² | Anti-windup clamp on STA integral |

#### MPC parameters (`controller_type: mpc`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `mpc_N` | `20` | steps | Horizon length |
| `mpc_dt` | `0.05` | s | MPC step size (horizon = N × dt = 1 s) |
| `mpc_Q_pos` | `[50, 50, 400]` | — | Position error weight [x, y, z] |
| `mpc_Q_vel` | `[10, 10, 10]` | — | Velocity error weight |
| `mpc_Q_pos_N` | `[250, 250, 2000]` | — | Terminal position weight (5× stage weight) |
| `mpc_Q_vel_N` | `[50, 50, 50]` | — | Terminal velocity weight |
| `mpc_R_acc` | `[1, 1, 1]` | — | Acceleration input cost |
| `mpc_a_max` | `15.0` | m/s² | Acceleration clamp (safety limit) |

#### Time-optimal trajectory (optional)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `use_time_optimal` | `false` | — | Replace polynomial with rpg_time_optimal NLP |
| `rpg_time_optimal_path` | `""` | path | Absolute path to `rpg_time_optimal/src` |
| `quad_params_file` | `""` | path | Path to `quad_params.yaml` for rpg_time_optimal |

### State machine node (`traj_landing_sm`)

| Parameter | Default | Unit | Description |
|-----------|---------|------|-------------|
| `ns` | `uav1` | — | UAV namespace |
| `hover_thrust` | `0.58` | [0–1] | Hover thrust for PRE_ARMED / ARMING setpoints |
| `climb_alt_m` | `10.0` | m | Altitude to climb before ArUco search |
| `hold_alt_m` | `5.0` | m | Hover altitude during TRAJECTORY_ALIGN |
| `align_threshold_m` | `0.6` | m | ‖e_xy‖ below which ALIGN → DESCEND |
| `align_settle_s` | `3.0` | s | Must be below threshold for this long |
| `descent_rate_ms` | `0.3` | m/s | Rate of altitude decrease in DESCEND |
| `alt_down_rate_ms` | `1.0` | m/s | Rate limit for climb_alt → hold_alt ramp |
| `close_descent_alt_m` | `2.5` | m | Below this use adaptive (slow) descent |
| `desc_err_scale_m` | `0.5` | m | Horizontal error at which descent rate → min |
| `desc_min_pos_factor` | `0.05` | — | Minimum position factor for adaptive descent |
| `desc_min_alt_factor` | `0.5` | — | Minimum altitude factor for adaptive descent |
| `touch_range_m` | `0.8` | m | Garmin range that triggers AUTO.LAND |
| `descend_lost_timeout_s` | `4.0` | s | ArUco lost in DESCEND → HOVER after this |
| `rate_hz` | `10.0` | Hz | State machine loop rate |
| `prearm_timeout_s` | `10.0` | s | Max wait in PRE_ARMED |
| `arming_timeout_s` | `30.0` | s | Max wait in ARMING |
| `climb_timeout_s` | `60.0` | s | Max wait in CLIMBING |
| `nadir_timeout_s` | `15.0` | s | Max wait for gimbal nadir |
| `search_timeout_s` | `60.0` | s | Max wait for ArUco detection |
| `align_timeout_s` | `120.0` | s | Max wait in TRAJECTORY_ALIGN |

---

## Usage

### Prerequisites

Same simulation stack as `ar_code_landing`:

```bash
rosdep install --from-paths src --ignore-src --rosdistro=noetic -y
catkin build uav_traj_landing
source devel/setup.bash
```

### Running the simulation

**Terminal 1 — Gazebo + UAV + USV:**
```bash
roslaunch ar_code_landing usv_uav_gimbal_landing.launch
```
Wait for Gazebo to finish loading.

**Terminal 2 — trajectory landing (PID controller, default):**
```bash
roslaunch uav_traj_landing traj_landing.launch
```
The state machine enters `PRE_ARMED` and arms automatically once sensors are
ready (~3 s).

---

### Selecting a controller

Pass `controller_type` as a launch argument.  All other launch parameters
stay the same.

**PID (default) — classical feedback baseline:**
```bash
roslaunch uav_traj_landing traj_landing.launch controller_type:=pid
```

**SMC — Super-Twisting, feedforward acceleration:**
```bash
roslaunch uav_traj_landing traj_landing.launch controller_type:=smc
```

**Linear MPC — 1-second horizon, trajectory feedforward:**
```bash
roslaunch uav_traj_landing traj_landing.launch controller_type:=mpc
```

To change the default permanently, edit `config/landing_params.yaml`:
```yaml
traj_landing_controller:
  controller_type: smc   # or pid / mpc
```

---

### Selecting a trajectory planner

**Minimum-jerk polynomial (default, no extra dependencies):**
```bash
roslaunch uav_traj_landing traj_landing.launch
```

**Time-optimal NLP (requires CasADi + rpg_time_optimal):**
```bash
roslaunch uav_traj_landing traj_landing.launch \
  use_time_optimal:=true \
  rpg_time_optimal_path:=$HOME/rpg_time_optimal/src \
  quad_params_file:=$(rospack find uav_traj_landing)/config/quad_params.yaml
```

**Combined — time-optimal trajectory with MPC tracking:**
```bash
roslaunch uav_traj_landing traj_landing.launch \
  controller_type:=mpc \
  use_time_optimal:=true \
  rpg_time_optimal_path:=$HOME/rpg_time_optimal/src \
  quad_params_file:=$(rospack find uav_traj_landing)/config/quad_params.yaml
```

---

### Monitoring

```bash
# State machine progress
rostopic echo /traj_landing_sm/state

# Horizontal error (ALIGN → DESCEND uses vector.z = |e_xy|)
rostopic echo /traj_landing/ctrl/tracking_error

# Controller status: IDLE | HOLDING | TRACKING
rostopic echo /traj_landing/ctrl/status

# Live attitude setpoint
rostopic echo /uav1/mavros/setpoint_raw/attitude

# Node graph
rqt_graph
```

**Change UAV namespace:**
```bash
roslaunch uav_traj_landing traj_landing.launch ns:=uav2 controller_type:=smc
```

---

## Optional: time-optimal trajectories

When `use_time_optimal: true`, the controller calls
[uzh-rpg/rpg_time_optimal](https://github.com/uzh-rpg/rpg_time_optimal)
to solve a nonlinear minimum-time optimal control problem via CasADi.
This finds the physically fastest trajectory under thrust and angular rate
limits.  The polynomial planner is used as a fallback if the NLP fails.

### Install

```bash
pip3 install casadi
git clone https://github.com/uzh-rpg/rpg_time_optimal ~/rpg_time_optimal
pip3 install -r ~/rpg_time_optimal/requirements.txt
```

### Tune the quadrotor model

Edit `config/quad_params.yaml` to match your vehicle:

```yaml
mass: 2.0              # kg
arm_length: 0.15       # m
inertia: [0.025, 0.025, 0.045]  # kg·m²
TWR_max: 2.8           # max thrust-to-weight ratio
omega_max_xy: 10.0     # rad/s
omega_max_z:  3.0      # rad/s
```

### Fallback behaviour

If CasADi / rpg_time_optimal is not installed, or if the NLP fails to
converge, the controller automatically falls back to the polynomial planner.
A `WARN` is printed but operation continues uninterrupted.

---

## Comparison with ar_code_landing

| Aspect | ar_code_landing | uav_traj_landing |
|--------|-----------------|-----------------|
| **Horizontal control** | Proportional-Derivative (reactive) | Trajectory planner + pluggable controller (predictive) |
| **Altitude control** | PID on Garmin error | Trajectory planner + selected controller |
| **Moving target** | PD reacts to error; no prediction | Replans trajectory to new target every 2 s |
| **Trajectory smoothness** | Step changes in roll/pitch at each tick | Smooth C² reference; controller tracks it |
| **Sensor requirements** | Garmin + ArUco + IMU + EKF vel | Identical |
| **External dependencies** | None | Optional: casadi + rpg_time_optimal |
| **State machine** | ar_code_landing states | Identical (ALIGN/DESCEND renamed with TRAJECTORY_ prefix) |
| **Launch** | `ar_code_landing.launch` | `traj_landing.launch [controller_type:=pid\|smc\|mpc]` |

```bash
# PID baseline (ar_code_landing)
roslaunch ar_code_landing ar_code_landing.launch

# Trajectory + PID (this package, default)
roslaunch uav_traj_landing traj_landing.launch controller_type:=pid

# Trajectory + SMC
roslaunch uav_traj_landing traj_landing.launch controller_type:=smc

# Trajectory + MPC
roslaunch uav_traj_landing traj_landing.launch controller_type:=mpc
```

Only one launch should be active at a time — they all publish to the same
`/<ns>/mavros/setpoint_raw/attitude` topic.

---

## Controller comparison

| | PID | SMC | MPC |
|---|-----|-----|-----|
| **Feedforward from trajectory** | No | Yes (`des_acc`) | Yes (full N-step horizon) |
| **Disturbance rejection** | Integral term (slow) | STA integral `w` (finite-time) | Via Q/R tuning + model |
| **Integral windup protection** | `ki_range` clamp | `smc_integral_limit` clamp | None (stateless) |
| **Computational cost** | O(1) | O(1) | O(N²) per tick; H precomputed at init |
| **External dependencies** | None | None | scipy.linalg (already required) |
| **Tuning effort** | Low (3 gains per axis) | Medium (λ, k1, k2 per axis) | Medium (Q, Q_N, R weights) |
| **Closed-form / iterative** | Closed-form | Closed-form | Closed-form (condensed QP) |
| **Best for** | Baseline comparison | Robustness to disturbances | Smooth tracking with long horizon |

**Recommended comparison procedure:**

1. Run with `controller_type:=pid` and record `tracking_error` and `rqt_plot` of roll/pitch.
2. Repeat with `controller_type:=smc` — expect tighter XY convergence.
3. Repeat with `controller_type:=mpc` — expect smoother roll/pitch transients.
4. All three with `use_time_optimal:=true` for the trajectory comparison axis.

---

## Tuning guide

### PID: oscillation or overshoot

Reduce `pid_kp` or increase `pid_kd`:
```yaml
pid_kp: [0.3, 0.3, 0.9]
pid_kd: [0.3, 0.3, 0.5]
```

### PID: slow convergence or steady-state error

Increase `pid_kp`; if error persists at rest, increase `pid_ki` gently:
```yaml
pid_kp: [0.6, 0.6, 1.5]
pid_ki: [0.08, 0.08, 0.08]
```

### SMC: oscillation / chattering in hover

Reduce `smc_lambda` (softer sliding surface) or `smc_k1`:
```yaml
smc_lambda: [1.0, 1.0, 1.5]
smc_k1:     [0.8, 0.8, 1.0]
```

### SMC: slow convergence / large steady-state error

Increase `smc_lambda` or `smc_k2` (stronger integral action):
```yaml
smc_lambda: [2.0, 2.0, 3.0]
smc_k2:     [1.0, 1.0, 1.5]
```

### MPC: drone moves sluggishly

Increase position weights; reduce input cost:
```yaml
mpc_Q_pos:   [100.0, 100.0, 800.0]
mpc_R_acc:   [0.5,   0.5,   0.5]
```

### MPC: drone overshoots or rings

Increase input cost or terminal weight:
```yaml
mpc_R_acc:   [2.0, 2.0, 2.0]
mpc_Q_pos_N: [500.0, 500.0, 4000.0]
```

### MPC: extend or shorten the lookahead horizon

Adjust `mpc_N` and `mpc_dt` (horizon = N × dt):
```yaml
mpc_N:  30     # 1.5 s lookahead
mpc_dt: 0.05
```
Longer horizons improve anticipation on curved trajectories but increase
the O(N³) precomputation time at node startup (still < 1 s for N ≤ 50).

### Trajectory too aggressive (large tilts)

Reduce `traj_v_max_ms` or lower `max_tilt_deg`:
```yaml
traj_v_max_ms: 1.5
max_tilt_deg:  15.0
```

### Trajectory replanning too frequent

```yaml
replan_distance_m: 1.0
replan_interval_s: 5.0
```

### Drone does not enter OFFBOARD / arm

The SM sets `COM_RC_IN_MODE=1` (MAVLINK-only arming, no RC required).
If arming still fails:
```bash
rostopic echo /uav1/mavros/state            # confirm connected=true
rostopic echo /uav1/mavros/statustext/recv  # PX4 rejection reason
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| SM stuck in `PRE_ARMED` | Garmin not publishing | Check `ocean_rangefinder` bridge |
| SM stuck in `ARMING` | MAVROS not connected | Check Gazebo + PX4 SITL are running |
| SM stuck in `ARUCO_SEARCH` | Camera / ArUco issue | Check `otter_usv_detector` and camera topic |
| Drone drifts under `TRACKING` | Controller gains too small | Increase kp (PID), lambda (SMC), or Q_pos (MPC) |
| Drone tilts > 25° | Gains too large or mass wrong | Reduce k1/kp; verify `drone_mass` in YAML |
| MPC node slow to start | Hessian precomputation | Normal for large N; reduces to < 1 s for N ≤ 50 |
| MPC: erratic commands | Horizon outside trajectory | Check `pose_timeout_s`; MPC falls back to constant setpoint when stale |
| `rpg_time_optimal` import error | CasADi not installed | `pip3 install casadi` |
| NLP solver `WARNING: Infeasible` | Target too close | Polynomial fallback activates automatically |
| `[traj_ctrl] planning failed` | CasADi constraint violation | Check `quad_params.yaml` TWR_max is realistic |
