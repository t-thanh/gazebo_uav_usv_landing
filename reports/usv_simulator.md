# USV Simulator — Modifications Report

**Date:** 2026-05-30
**Base repository:** [jhlenes/usv_simulator](https://github.com/jhlenes/usv_simulator) (commit `eb6128b`)
**Fork:** [t-thanh/usv_simulator](https://github.com/t-thanh/usv_simulator) (commit `1f069db`)
**Submodule path:** `ros/src/usv_sim/`

---

## Overview

The upstream `usv_simulator` provides a Gazebo simulation of the Maritime Robotics Otter USV for ROS. This fork adapts it for GPS-denied UAV autonomous landing research: the Otter model is equipped with a nested ArUco landing target and new launch/world files integrate the USV scene with the MRS UAV stack.

---

## Modifications

### 1. Nested ArUco Landing Target

**File:** `otter_description/urdf/otter_base.urdf.xacro`

A fixed `ar_code` link is attached to the Otter's `base_link`, carrying a nested marker board:

- **Outer marker:** DICT_APRILTAG_36h11, ID = 10
- **Inner marker:** DICT_4X4_50, ID = 1
- **Mount position:** −0.15 m along X (towards stern), 0.7 m above the base link
- **Rotation:** 90° about Z so that the bow (+X) appears at the top when viewed from above (nadir camera orientation)
- **Scale:** 2.5 × 2.5 × 0.05 m

The nesting strategy allows the outer AprilTag to be detected at long range (high altitude), while the inner ArUco provides precise pose at close range (final approach), without changing the detection pipeline.

**Mesh asset used:** `package://otter_gazebo/urdf/Target_2/meshes/Target_2_board.dae`

### 2. Target_2 Mesh and Texture Assets

**Directory:** `otter_gazebo/urdf/Target_2/`

Custom Gazebo model assets for the nested ArUco board:

| File | Description |
|---|---|
| `meshes/Target_2_board.dae` | Board variant used on the Otter (flat, oriented for nadir view) |
| `meshes/Target_2.dae` | Base mesh |
| `meshes/Target_2_nested.dae` | Variant with inner marker inset |
| `meshes/Target_2_rotated.dae`, `Target_2_straight.dae` | Orientation variants for testing |
| `materials/textures/april_36h11-10.png` | Outer AprilTag 36h11 ID=10 texture |
| `materials/textures/4x4_ID1_200px.png` | Inner DICT_4X4_50 ID=1 texture |
| `materials/textures/april_36h11-10_2000px-4x4_ID1_200px.png` | Composite nested texture |

### 3. Updated Otter Mesh

**File:** `otter_description/meshes/otter/otter.dae`

The main Otter mesh was updated to incorporate the landing target geometry. The original mesh is archived in the dev workspace as `otter_original.dae` (not committed to the repo).

### 4. Tuned Wave Dynamics

**File:** `otter_gazebo/urdf/dynamics/otter_gazebo_dynamics_plugin.xacro`

Wave parameters were reduced from the upstream defaults to simulate a low-sea-state (calm harbour) scenario more representative of the landing conditions:

| Parameter | Upstream | Modified | Effect |
|---|---|---|---|
| `wave_amp0` | 0.09 m | 0.03 m | Reduced primary wave amplitude |
| `wave_period1` | 0.7 s | 1.5 s | Longer secondary wave period (less chop) |
| `wave_direction1` | (−0.7, 0.7) | (−0.7, 0) | Secondary wave aligned with primary direction |

### 5. UAV Integration Launch Files

**Directory:** `otter_gazebo/launch/`

| Launch file | Purpose |
|---|---|
| `usv_uav_gimbal.launch` | Main integration launch: starts Gazebo with the UAV+USV world, MRS drone spawner, and all USV sensors |
| `usv_uav.launch` | Variant without gimbal-specific arguments |
| `start_uav.launch` | Spawns the MRS X500 UAV after Gazebo is ready |

`usv_uav_gimbal.launch` exposes arguments for initial USV pose (x, y, z, R, P, Y), sensor toggles (GPS, IMU, LiDAR, ground truth), and Gazebo options (gui, verbose, physics engine).

### 6. UAV + USV World Files

**Directory:** `usv_worlds/worlds/`

| World file | Description |
|---|---|
| `usv_uav.world.xacro` | Full-speed simulation world (real-time factor 1, 250 Hz physics) with MRS static transform republisher plugin, ODE solver, and harbour scene |
| `usv_uav_fast.world.xacro` | Accelerated variant for faster-than-real-time runs (reduced GUI load) |

Both worlds include the MRS Gazebo static transform republisher plugin and are registered in `usv_worlds/CMakeLists.txt` via `xacro_add_files`.

### 7. Helper Scripts

**Directory:** `otter_gazebo/scripts/`

| Script | Purpose |
|---|---|
| `delayed_launcher.py` | Waits for Gazebo to be ready before spawning models, avoiding race conditions at startup |
| `otter_pose_extractor.py` | Reads the Otter ground-truth pose from Gazebo and republishes it as a `geometry_msgs/PoseStamped` on a configurable topic |

---

## Usage

```bash
# Launch Gazebo with Otter USV + UAV integration world
roslaunch otter_gazebo usv_uav_gimbal.launch

# Then in a separate terminal, spawn the MRS X500 UAV
roslaunch otter_gazebo start_uav.launch
```

For the full demo including gimbal, landing controller, and rangefinder, use the top-level launch from `uav_landing_demo` instead.
