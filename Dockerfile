FROM ros:noetic-ros-base

LABEL org.opencontainers.image.title="ODD-SEC Core"
LABEL org.opencontainers.image.description="ROS Noetic environment for the platform-independent ODD-SEC core tools"
LABEL org.opencontainers.image.source="https://github.com/KuanDai0123/ODD-SEC"
LABEL org.opencontainers.image.licenses="MIT"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive

ENV CATKIN_WS=/catkin_ws

# Keep this image independent of a specific GPU platform. The panorama package is intentionally
# excluded because it requires a platform-specific CUDA and TensorRT toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-thread-dev \
    libeigen3-dev \
    libopencv-dev \
    python3-catkin-tools \
    ros-noetic-cv-bridge \
    ros-noetic-geometry-msgs \
    ros-noetic-image-transport \
    ros-noetic-message-filters \
    ros-noetic-message-generation \
    ros-noetic-message-runtime \
    ros-noetic-rosbag \
    ros-noetic-roscpp \
    ros-noetic-rospy \
    ros-noetic-sensor-msgs \
    ros-noetic-serial \
    ros-noetic-std-msgs \
    ros-noetic-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${CATKIN_WS}

COPY src/ src/

RUN source /opt/ros/noetic/setup.bash \
    && catkin init \
    && catkin config --extend /opt/ros/noetic \
        --cmake-args -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    && catkin build datasync event_converter photogate_reader --no-status

# Source both ROS and the compiled workspace for interactive shells as well as
# commands passed directly to `docker run`.
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    'source /opt/ros/noetic/setup.bash' \
    'source "${CATKIN_WS}/devel/setup.bash"' \
    'exec "$@"' \
    > /usr/local/bin/odd-sec-entrypoint \
    && chmod +x /usr/local/bin/odd-sec-entrypoint

ENTRYPOINT ["/usr/local/bin/odd-sec-entrypoint"]
CMD ["bash"]
