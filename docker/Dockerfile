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
# `.*` from inequality specifiers before running the setup script.
RUN PX4_REQ=${WS}/src/uav_sim/simulation/ros_packages/px4_firmware/Tools/setup/requirements.txt \
    && sed -i -E 's/(>=?[0-9][0-9.]*)\.\*/\1/g' "$PX4_REQ" \
    && ${WS}/src/uav_sim/simulation/ros_packages/px4_firmware/Tools/setup/ubuntu.sh \
        --no-nuttx --no-sim-tool \
    && pip3 install --no-cache-dir packaging toml \
    && rm -rf /var/lib/apt/lists/*

# ── geographiclib datasets (required by mavros at runtime) ────────────────────
RUN MAVROS_GEO=$(find ${WS}/src/uav_sim/uav_core -path '*mavros/scripts/install_geographiclib_datasets.sh' | head -1) \
    && [ -n "$MAVROS_GEO" ] && bash "$MAVROS_GEO" || echo "geographiclib script not found, skipping"

# ── rosdep: resolve remaining ROS package dependencies (best-effort) ──────────
RUN apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src -y -r || true \
    && rm -rf /var/lib/apt/lists/*

# ── Build the catkin workspace ───────────────────────────────────────────────
RUN source /opt/ros/${ROS_DISTRO}/setup.bash \
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
