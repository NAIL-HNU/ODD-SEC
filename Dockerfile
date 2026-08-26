# 使用ROS Noetic作为基础镜像
FROM ros:noetic-ros-core

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=noetic
ENV CATKIN_WS=/catkin_ws

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    vim \
    nano \
    python3-pip \
    python3-dev \
    libopencv-dev \
    libeigen3-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# 从源码编译安装 libcaer
RUN wget https://github.com/inilabs/libcaer/archive/refs/tags/v3.3.7.tar.gz && \
    tar -xzf v3.3.7.tar.gz && \
    cd libcaer-3.3.7 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf libcaer-3.3.7 v3.3.7.tar.gz

# 安装Python依赖
RUN pip3 install --no-cache-dir \
    numpy \
    opencv-python \
    matplotlib \
    scipy \
    scikit-learn \
    pandas \
    jupyter \
    ipython

# 安装ROS依赖
RUN apt-get update && apt-get install -y \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-sensor-msgs \
    ros-noetic-std-msgs \
    ros-noetic-geometry-msgs \
    ros-noetic-visualization-msgs \
    ros-noetic-message-filters \
    ros-noetic-tf \
    ros-noetic-tf2 \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    ros-noetic-dynamic-reconfigure \
    ros-noetic-nodelet \
    ros-noetic-bond \
    ros-noetic-nodelet-topic-tools \
    ros-noetic-angles \
    ros-noetic-cmake-modules \
    ros-noetic-control-toolbox \
    ros-noetic-realtime-tools \
    ros-noetic-urdfdom-py \
    ros-noetic-kdl-conversions \
    ros-noetic-kdl-parser-py \
    ros-noetic-sophus \
    ros-noetic-tf-conversions \
    ros-noetic-eigen-conversions \
    ros-noetic-geometry \
    ros-noetic-common-msgs \
    ros-noetic-common-tutorials \
    ros-noetic-ros-tutorials \
    ros-noetic-rospy-tutorials \
    ros-noetic-urdf-tutorial \
    ros-noetic-visualization-tutorials \
    ros-noetic-actionlib-tutorials \
    ros-noetic-bond-core \
    ros-noetic-bondcpp \
    ros-noetic-bondpy \
    ros-noetic-smach \
    ros-noetic-smach-ros \
    ros-noetic-executive-smach \
    ros-noetic-smach-viewer \
    ros-noetic-qt-gui-core \
    ros-noetic-qt-gui-cpp \
    ros-noetic-qt-gui-py-common \
    ros-noetic-rqt \
    ros-noetic-rqt-common-plugins \
    ros-noetic-rqt-gui \
    ros-noetic-rqt-gui-cpp \
    ros-noetic-rqt-gui-py \
    ros-noetic-rqt-py-common \
    ros-noetic-rqt-py-console \
    ros-noetic-rqt-reconfigure \
    ros-noetic-rqt-service-caller \
    ros-noetic-rqt-shell \
    ros-noetic-rqt-srv \
    ros-noetic-rqt-top \
    ros-noetic-rqt-topic \
    ros-noetic-rqt-web \
    ros-noetic-rqt-graph \
    ros-noetic-rqt-plot \
    ros-noetic-rqt-console \
    ros-noetic-rqt-logger-level \
    ros-noetic-rqt-msg \
    ros-noetic-rqt-bag \
    ros-noetic-rqt-bag-plugins \
    ros-noetic-rqt-image-view \
    ros-noetic-rqt-multiplot \
    ros-noetic-rqt-nav-view \
    ros-noetic-rqt-pose-view \
    ros-noetic-rqt-publisher \
    ros-noetic-rqt-py-console \
    ros-noetic-rqt-robot-monitor \
    ros-noetic-rqt-robot-steering \
    ros-noetic-rqt-runtime-monitor \
    ros-noetic-rqt-rviz \
    && rm -rf /var/lib/apt/lists/*

# 创建catkin工作空间
RUN mkdir -p $CATKIN_WS/src

# 设置工作目录
WORKDIR $CATKIN_WS

# 初始化catkin工作空间
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_init_workspace src"

# 安装catkin工具
RUN pip3 install catkin-tools

# 复制源代码
COPY src/ src/

# 编译ROS包（只编译核心包，跳过需要libcaer的包）
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && catkin_make --pkg panorama --pkg datasync --pkg event_converter --pkg photogate"

# 设置环境变量
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source $CATKIN_WS/devel/setup.bash" >> ~/.bashrc

# 创建启动脚本
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/ros/noetic/setup.bash' >> /entrypoint.sh && \
    echo 'source $CATKIN_WS/devel/setup.bash' >> /entrypoint.sh && \
    echo 'exec "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

# 设置入口点
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"] 