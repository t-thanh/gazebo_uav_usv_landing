# ─────────────────────────────────────────────────────────────────────────────
# gazebo_uav_usv_landing — reproducible ROS Noetic / Ubuntu 20.04 image
#
# Builds the full simulation stack (Otter USV + MRS X500 PX4 SITL + Gremsy
# gimbal) and the landing/tracking packages, then runs a human-watched demo:
# the UAV takes off, climbs to 20 m, points the gimbal straight down (nadir),
# and the gimbal_usv_tracker locks onto the static Otter USV using YOLO OBB.
#
# GPU is required (NVIDIA): YOLO inference runs on CUDA and Gazebo uses the
# NVIDIA driver for camera rendering. Run with `--gpus all` and the NVIDIA
# Container Toolkit installed on the host. See docker/README.md.
#
# Base: ros:noetic-desktop-full already provides Gazebo 11 + rendering libs.
# The NVIDIA driver (CUDA userspace for torch, GLX/EGL for Gazebo) is injected
# at runtime by the NVIDIA Container Toolkit via the NVIDIA_* env vars below.
# ─────────────────────────────────────────────────────────────────────────────
FROM osrf/ros:noetic-desktop-full

LABEL org.opencontainers.image.title="gazebo_uav_usv_landing" \
      org.opencontainers.image.description="GPS-denied UAV-on-USV landing demo (ROS Noetic, Gazebo, MRS X500 + Otter)" \
      org.opencontainers.image.source="https://github.com/t-thanh/gazebo_uav_usv_landing"

# Expose the host GPU to both CUDA (torch) and OpenGL (Gazebo) via the toolkit.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all \
    DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=noetic

SHELL ["/bin/bash", "-c"]

# ── Build args ───────────────────────────────────────────────────────────────
ARG REPO_URL=https://github.com/t-thanh/gazebo_uav_usv_landing.git
ARG REPO_BRANCH=master
ARG WEIGHTS_URL=https://github.com/t-thanh/otter_usv_detector/releases/download/weights-v1/best.pt
# Pin the inference stack for reproducibility. cu118 wheels are py3.8-compatible.
ARG TORCH_VERSION=2.0.1+cu118
ARG TORCHVISION_VERSION=0.15.2+cu118
ARG ULTRALYTICS_VERSION=8.4.33
# Change CACHE_BUST (e.g. --build-arg CACHE_BUST=$(date +%s)) to force a fresh
# `git clone` of the project (and everything after it) without --no-cache.
ARG CACHE_BUST=1

# ── System + build dependencies ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs wget curl gnupg lsb-release sudo \
        build-essential cmake \
        python3-pip python3-setuptools python3-catkin-tools python3-osrf-pycommon \
        python3-rosdep python3-vcstool python3-rospkg \
        libgstreamer1.0-dev \
        xvfb x11-utils mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# gitman manages the large external UAV deps (PX4 firmware, MRS stack).
RUN pip3 install --no-cache-dir gitman

# ── YOLO inference stack (conda env, mirrors the dev `yolo_training` env) ─────
# ROS Noetic ships Python 3.8, but ultralytics/torch were trained/run under
# Python 3.10. Rather than fight pip's resolver on 3.8, recreate the dev conda
# env: a Python 3.10 `yolo_training` env with a CUDA build of torch + the exact
# ultralytics version. ROS stays on its system Python 3.8; the detector node
# runs under this interpreter via run_detector.sh ($DETECTOR_PYTHON), which
# injects the ROS message/client packages onto PYTHONPATH (pure-Python, import
# fine across 3.8/3.10; cv_bridge is intentionally not used).
# NOTE: conda is intentionally NOT added to PATH — that would shadow ROS's
# system Python 3.8 (with Miniforge's base Python) for every later build step
# (PX4 deps, catkin build) and produce a broken image. The env interpreter is
# referenced only by the absolute $DETECTOR_PYTHON path at runtime.
ENV CONDA_DIR=/opt/conda
ENV CONDA_ENV=yolo_training
ENV DETECTOR_PYTHON=${CONDA_DIR}/envs/${CONDA_ENV}/bin/python

RUN wget -q -O /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash /tmp/miniforge.sh -b -p ${CONDA_DIR} \
    && rm /tmp/miniforge.sh \
    && ${CONDA_DIR}/bin/conda create -y -n ${CONDA_ENV} python=3.10 \
    && ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
        --index-url https://download.pytorch.org/whl/cu118 \
    && ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install --no-cache-dir \
        ultralytics==${ULTRALYTICS_VERSION} rospkg pyyaml \
    && ${CONDA_DIR}/envs/${CONDA_ENV}/bin/pip install --no-cache-dir "numpy<2" \
    && ${CONDA_DIR}/bin/conda clean -afy

# ── Clone the project (with all submodules) ──────────────────────────────────
WORKDIR /opt
RUN echo "cache-bust: ${CACHE_BUST}" \
    && git clone --branch ${REPO_BRANCH} --recurse-submodules ${REPO_URL} gazebo_uav_usv_landing

ENV WS=/opt/gazebo_uav_usv_landing/ros
WORKDIR ${WS}

# ── Install gitman-managed dependencies (PX4 ~1.4 GB + MRS stack) ─────────────
RUN cd src/uav_sim/simulation && gitman install --force \
    && cd ${WS}/src/uav_sim/uav_core && gitman install --force

# ── PX4 SITL build dependencies (no NuttX, no sim tools) ──────────────────────
# PX4 1.13's requirements.txt uses legacy specifiers like `matplotlib>=3.0.*`
# that modern pip rejects (`.*` is only valid with `==`/`!=`). Strip the trailing
# `.*` from inequality specifiers before running the setup script, then restore
# the file so the firmware git tree stays clean (see version tag below).
RUN PX4_DIR=${WS}/src/uav_sim/simulation/ros_packages/px4_firmware \
    && sed -i -E 's/(>=?[0-9][0-9.]*)\.\*/\1/g' "$PX4_DIR/Tools/setup/requirements.txt" \
    && "$PX4_DIR/Tools/setup/ubuntu.sh" --no-nuttx --no-sim-tool \
    && git -C "$PX4_DIR" checkout -- Tools/setup/requirements.txt \
    && pip3 install --no-cache-dir packaging toml \
    && rm -rf /var/lib/apt/lists/*

# ── PX4 version tag ───────────────────────────────────────────────────────────
# gitman checks out a bare commit (no tags), but PX4's version generator requires
# a `v<major>.<minor>.<patch>` tag. Force one onto HEAD (rev was 1.13.2-dev).
RUN git -C ${WS}/src/uav_sim/simulation/ros_packages/px4_firmware tag -f v1.13.2 HEAD

# ── geographiclib geoid dataset (required by mavros at runtime) ───────────────
# mavros raises a FATAL if the egm96-5 geoid is missing. geographiclib-tools
# (pulled by rosdep) provides geographiclib-get-geoids.
RUN apt-get update && apt-get install -y --no-install-recommends geographiclib-tools \
    && geographiclib-get-geoids egm96-5 \
    && rm -rf /var/lib/apt/lists/*

# ── Workspace adjustments ─────────────────────────────────────────────────────
# 1. otter_control needs usv_msgs (not in the fork) and is not used by the static
#    USV demo -> CATKIN_IGNORE it.
# 2. mavlink: ROS Noetic's only apt mavlink is now 2025.x, whose camera/gimbal
#    message signatures break PX4 1.13's sitl_gazebo. The uav_core gitman fetch
#    includes the matching mavlink 2022.8.8 under lib/ (CATKIN_IGNORE-d); expose
#    it to catkin so the whole stack builds against one consistent mavlink.
RUN touch ${WS}/src/usv_sim/otter_control/CATKIN_IGNORE \
    && ln -sfn ../.gitman/mavlink-gbp-release ${WS}/src/uav_sim/uav_core/ros_packages/mavlink

# ── rosdep + extra build deps ─────────────────────────────────────────────────
# rosdep aborts on a few unresolvable/typo'd keys; skip those and install the
# real packages behind them explicitly. Do NOT install ros-noetic-mavlink (see
# above — wrong version). All listed apt packages are declared deps that rosdep
# either skips or misses.
RUN apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -y -r \
        --skip-keys "qt effort_controllers mav_msgs opencv visualisation_msgs teleop_twist_keyboard" \
        || true \
    && apt-get install -y --no-install-recommends \
        ros-noetic-geographic-msgs \
        ros-noetic-tf2-sensor-msgs \
        ros-noetic-robot-localization \
        ros-noetic-velodyne-description \
        ros-noetic-effort-controllers \
        ros-noetic-teleop-twist-keyboard \
        libncurses-dev \
    && rm -rf /var/lib/apt/lists/*

# System Python scipy: Ubuntu 20.04 ships 1.3.3, but the MRS/landing nodes that
# run under system Python (gimbal_position_node, gimbal_tracker_node) call
# scipy.spatial.transform.Rotation.identity(), added in scipy 1.4.
RUN pip3 install --no-cache-dir "scipy==1.5.4"

# ── Build the catkin workspace ───────────────────────────────────────────────
# PX4 sitl_gazebo's FindMAVLink can't locate the nested workspace mavlink headers,
# so pin _MAVLINK_INCLUDE_DIR to the devel include space. Build mavlink first so
# those headers exist before sitl_gazebo configures.
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
    && catkin config --cmake-args -D_MAVLINK_INCLUDE_DIR=${WS}/devel/include \
    && catkin build mavlink --no-status -j$(nproc) \
    && catkin build --no-status -j$(nproc)

# ── Fetch trained YOLO OBB weights into the detector package ──────────────────
RUN wget -q -O \
        ${WS}/src/ros_landing/otter_usv_detector/otter_usv_detector/models/best.pt \
        ${WEIGHTS_URL} \
    && ls -lh ${WS}/src/ros_landing/otter_usv_detector/otter_usv_detector/models/best.pt

# ── Entrypoint sources ROS + workspace ───────────────────────────────────────
# Scripts come from the cloned repo, so the image builds even with no local
# build context (e.g. `docker build github.com/t-thanh/gazebo_uav_usv_landing`).
RUN chmod +x /opt/gazebo_uav_usv_landing/docker/entrypoint.sh \
             /opt/gazebo_uav_usv_landing/docker/run_demo.sh

ENTRYPOINT ["/opt/gazebo_uav_usv_landing/docker/entrypoint.sh"]
CMD ["/opt/gazebo_uav_usv_landing/docker/run_demo.sh"]
