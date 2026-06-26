# Docker — reproducible demo image

A ROS Noetic / Ubuntu 20.04 image that builds the full stack (Otter USV + MRS
X500 PX4 SITL + Gremsy gimbal + landing packages) and runs the
**take-off-and-land demo**:

> The X500 arms and takes off **from the Otter USV deck**, climbs, then
> autonomously **follows the moving USV** and **descends to land back on it** —
> GPS-free, using only the gimbal camera (nested AR markers), a rangefinder, the
> IMU, and a short-range UWB relative fix near the deck.

It flies the **campaign-winning per-phase controllers**:

- **Approach** (far field) → PD + lookahead.
- **Follow / landing** (near field) → MPC (`q_pos = 1.5`), descend cone `0.70·alt`,
  hold ~9 m then descend to the UWB cylinder and commit.

Two things are selectable for the demo: **USV speed** and **wave (sea state)**.

## Requirements

- **NVIDIA GPU** + recent driver (Gazebo renders the gimbal camera via the driver).
- [Docker](https://docs.docker.com/engine/install/) +
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (provides `--gpus all`).
- An X server on the host (to show the Gazebo GUI).

(The demo perception is **AR-only** — no YOLO/CUDA inference is needed at run time,
though the image still bundles the YOLO stack for the opt-in tracking demo.)

## Build

From the repository root:

```bash
docker build -t gazebo_uav_usv_landing -f docker/Dockerfile .
```

A **large, long build** (tens of minutes): clones the project + submodules, pulls
the gitman-managed PX4 firmware (~1.4 GB) and MRS stack, and compiles the catkin
workspace. It clones from GitHub at build time, so it can also build from the
remote without a local checkout:

```bash
docker build -t gazebo_uav_usv_landing \
  github.com/t-thanh/gazebo_uav_usv_landing#master:docker
```

Useful build args (`--build-arg`): `REPO_BRANCH`, `WEIGHTS_URL`,
`ULTRALYTICS_VERSION`, `TORCH_VERSION`, `TORCHVISION_VERSION`.

## Run (GUI demo)

Allow the container to use your X server, then run with the GPU and X socket:

```bash
xhost +local:root

docker run --rm -it \
  --gpus all \
  --env DISPLAY=$DISPLAY --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  --env USV_SPEED=1.0 --env WAVE=small \
  gazebo_uav_usv_landing

xhost -local:root   # revoke access afterwards
```

The default command runs `docker/run_takeoff_land_demo.sh`: it starts Gazebo,
waits for the UAV, then runs the full take-off → follow → land-on-USV flight.

### What you should see

- Gazebo with the Otter USV and the X500 lifting off the deck.
- The UAV climbing, the gimbal pointing at the USV, then the USV starting to move
  at `USV_SPEED` while the UAV follows directly above it.
- The UAV descending through the AR bands into UWB range and **touching down on
  the moving USV deck**.

### Demo options (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `USV_SPEED` | `1.0` | USV forward speed [m/s], `0.0`–`2.0` |
| `WAVE` | `small` | Sea state: `none` \| `small` \| `medium` \| `large` |
| `GUI` | `true` | Show the Gazebo GUI (`false` = headless gzserver only) |

```bash
docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e USV_SPEED=2.0 -e WAVE=medium \
  gazebo_uav_usv_landing
```

### Headless run (no GUI / CI)

Gazebo still renders the gimbal camera offscreen, which needs a display even with
`GUI=false`. Provide a virtual one with `xvfb`:

```bash
docker run --rm --gpus all --name landing_demo \
  -e GUI=false -e USV_SPEED=1.0 -e WAVE=small \
  gazebo_uav_usv_landing \
  bash -c 'Xvfb :99 -screen 0 1280x1024x24 >/dev/null 2>&1 & \
           export DISPLAY=:99; exec /opt/gazebo_uav_usv_landing/docker/run_takeoff_land_demo.sh'
```

### Opt-in: the gimbal-tracking demo

The original human-watched gimbal-tracking demo (UAV climbs to 20 m, YOLO + IBVS
lock onto a **static** USV) is still available:

```bash
docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  gazebo_uav_usv_landing \
  /opt/gazebo_uav_usv_landing/docker/run_demo.sh
```

### Drop to a shell

```bash
docker run --rm -it --gpus all \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  gazebo_uav_usv_landing bash
# the workspace is sourced; then e.g.:
USV_SPEED=1.0 WAVE=small \
  ros/src/ros_landing/uav_landing_demo/scripts/follow_landing/run_takeoff_land_demo.sh
```

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`** — NVIDIA
  Container Toolkit not installed/configured. Verify with
  `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`.
- **Blank/black Gazebo or `libGL` errors** — X access not granted
  (`xhost +local:root`) or missing `--gpus all`.
- **UAV never appears / arming fails** — the deck is heaving; the demo opens the
  PX4 arming gate automatically, but on a `large` sea state arming can take longer.
- **`PRE_ARMED → ABORT` with `Unknown parameter to set …` / `connected: False`** — PX4's
  MAVLink couldn't bind its UDP port, so mavros never connected (no params → can't arm).
  Cause: a **port collision** — do **not** add `--net host` (the demo is self-contained), and
  make sure no other PX4 SITL is running on the machine (`pkill -9 -x px4`). Run isolated as shown
  above and it connects on its own network namespace.

## Notes

- The USV **moves** at `USV_SPEED` with the chosen `WAVE` sea state (waypoint +
  wave model), and the UAV takes off from and lands back on that same moving USV.
- The flight is fully GPS-free; ground-truth is logged to `/tmp` for scoring only.
