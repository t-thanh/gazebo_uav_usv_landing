# Docker — reproducible demo image

A ROS Noetic / Ubuntu 20.04 image that builds the full stack (Otter USV + MRS
X500 PX4 SITL + Gremsy gimbal + landing/tracking packages) and runs a
**human-watched gimbal-tracking demo**:

> The UAV arms, takes off, climbs to **20 m**, points the gimbal straight down
> (nadir), and `gimbal_usv_tracker` locks onto the **static** Otter USV using
> the YOLO OBB detector.

## Requirements

- **NVIDIA GPU** + recent driver on the host.
- [Docker](https://docs.docker.com/engine/install/) +
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (provides `--gpus all`). YOLO runs on CUDA and Gazebo renders the camera via
  the NVIDIA driver.
- An X server on the host (to show the Gazebo GUI).

## Build

From the repository root:

```bash
docker build -t gazebo_uav_usv_landing -f docker/Dockerfile .
```

This is a **large, long build** (tens of minutes): it clones the project with
all submodules, pulls the gitman-managed PX4 firmware (~1.4 GB) and MRS stack,
compiles the whole catkin workspace, and downloads the trained YOLO weights.

The image is self-contained — it clones from GitHub at build time, so it can
also be built directly from the remote without a local checkout:

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
  --net host \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix:rw \
  gazebo_uav_usv_landing

xhost -local:root   # revoke access afterwards
```

The default command runs `docker/run_demo.sh`, which starts Gazebo, waits for
the UAV, then runs the climb + gimbal-nadir + tracker sequence.

### What you should see

- Gazebo with the Otter USV (static) and the X500 taking off.
- The UAV climbing to ~20 m; the gimbal pivoting to point straight down at ~14 s.
- From a second shell into the container, the tracker locking on:

```bash
docker exec -it <container> bash
source /opt/gazebo_uav_usv_landing/ros/devel/setup.bash
rostopic echo /uav1/gimbal/tracker/tracking_status     # TRACKING / SEARCHING / LOST
rostopic echo /uav1/gimbal/tracker/usv_world_pose      # EMA-smoothed USV position
rqt_image_view /uav1/usv_detection/image               # annotated YOLO detections
```

### Demo options (env vars)

| Variable | Default | Meaning |
|---|---|---|
| `GUI` | `true` | Show the Gazebo GUI (`false` = headless gzserver only) |
| `INIT_ALTITUDE` | `20.0` | UAV tracking altitude [m] |
| `DEVICE` | `0` | YOLO CUDA device index |

```bash
docker run --rm -it --gpus all --net host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e INIT_ALTITUDE=15.0 \
  gazebo_uav_usv_landing
```

### Drop to a shell instead of the demo

```bash
docker run --rm -it --gpus all --net host \
  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  gazebo_uav_usv_landing bash
```

The workspace is already sourced; run the two stages manually in separate shells:

```bash
roslaunch otter_gazebo usv_uav_gimbal.launch
# then, once the UAV is up:
roslaunch gimbal_usv_tracker start_uav_gimbal_tracker.launch init_altitude:=20.0
```

## Troubleshooting

- **`could not select device driver "" with capabilities: [[gpu]]`** — the
  NVIDIA Container Toolkit is not installed/configured. Verify with
  `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi`.
- **Blank/black Gazebo or `libGL` errors** — X access not granted
  (`xhost +local:root`) or `NVIDIA_DRIVER_CAPABILITIES` not including `graphics`
  (the image sets `all`; ensure `--gpus all`).
- **YOLO fails to load `best.pt`** — the weights were trained with a specific
  ultralytics version. If you hit a deserialization error, rebuild with
  `--build-arg ULTRALYTICS_VERSION=<matching version>`.
- **Tracker stays in `SEARCHING`** — confirm the camera streams
  (`rostopic hz /uav1/overhead_cam/image_raw`) and the gimbal reached nadir
  (`pitch ≈ 1.5708`).

## Notes

- The USV is **static** in this demo (no waypoint/thruster commands are sent).
- Weights are pulled from the `otter_usv_detector` release `weights-v1`; they are
  not committed to git.
