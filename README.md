# ODD-SEC: Onboard Drone Detection with a Spinning Event Camera

ODD-SEC is an onboard drone detection system based on a spinning event camera. This repository provides ROS 1 tools for event-camera motion compensation, data conversion, image processing, camera triggering, and Docker-based deployment.

Choose one of the following installation methods according to your deployment environment:

1. Install and build ODD-SEC directly on Ubuntu 20.04 with ROS 1 Noetic.
2. Install and run ODD-SEC with Docker when an isolated deployment is preferred.

After completing either installation method, use the processing tools, launch files, and scripts provided by the repository.

## Table of Contents

- [System Requirements](#system-requirements)
- [1. Native Installation](#1-native-installation)
- [2. Docker Installation](#2-docker-installation)
- [3. Using ODD-SEC](#3-using-odd-sec)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Native Installation

The supported native development and runtime environment is:

- Ubuntu 20.04 LTS
- ROS 1 Noetic
- Python 3
- `catkin_tools`
- `rosbag`

Install and configure ROS 1 Noetic before building the workspace. Each terminal that uses ROS commands must source the ROS environment:

```bash
source /opt/ros/noetic/setup.bash
```

### Docker Installation

The Docker image contains the Ubuntu 20.04 and ROS Noetic runtime environment, so
Ubuntu 20.04 is not required on the host machine.

The host machine should provide:

- Docker Engine 20.10 or later
- Docker Compose v2 plugin
- At least 8 GB of RAM
- At least 20 GB of available disk space

A Linux host is recommended when accessing USB cameras, sensors, `/dev` devices,
host networking, or X11-based visualization. Windows and macOS hosts may be
suitable for offline data processing, but device passthrough, host networking,
and graphical display may require additional configuration.

Optional GPU acceleration requires:

- A compatible NVIDIA GPU
- A compatible NVIDIA driver
- NVIDIA Container Toolkit

The NVIDIA driver and container runtime must be compatible with the CUDA version
required by the application.

## 1. Native Installation

Use this method when ROS 1 Noetic and the required camera or sensor drivers are installed directly on the Ubuntu system.

### Clone and Build the Workspace

```bash
git clone https://github.com/ZhangHX-2514/Dataset_Toolbox.git
cd ODD-SEC
source /opt/ros/noetic/setup.bash
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

Source both ROS and the workspace again in every new terminal:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/ODD-SEC/devel/setup.bash
```

After the build succeeds, continue with [Using ODD-SEC](#3-using-odd-sec).

## 2. Docker Installation

Use Docker when you need a reproducible environment, an isolated runtime, or deployment on a machine without direct internet access.

### Option A: Build and Run Locally

From the repository root, build the Docker image using the project script:

```bash
./scripts/run_docker.sh build
```

Run the container:

```bash
./scripts/run_docker.sh run
```

Stop the container when finished:

```bash
./scripts/run_docker.sh stop
```

The equivalent manual build command is:

```bash
docker build -t odd-sec:latest .
```

### Option B: Deploy with a Docker TAR Archive

TAR deployment is intended for offline or restricted-network environments. Build and export the image on a development machine, transfer the archive to the target machine, and import it there.

#### Build and Export

```bash
./scripts/export_image.sh all
```

The equivalent manual commands are:

```bash
docker build -t odd-sec:latest .
docker save odd-sec:latest -o odd-sec.tar
```

Supported export actions:

```bash
# Build and export
./scripts/export_image.sh all

# Build only
./scripts/export_image.sh build

# Export only
./scripts/export_image.sh export

# Specify an output file name
./scripts/export_image.sh all my-custom-name.tar
```

#### Transfer the Archive

Transfer the TAR file with `scp`, a USB drive, or a network share:

```bash
scp odd-sec.tar user@target-machine:/path/to/
```

#### Import and Run on the Target Machine

From the project directory on the target machine:

```bash
# Import the image and deploy the system
./scripts/import_and_run.sh deploy

# Run the system after placing model files in model/
./scripts/import_and_run.sh run
```

Individual actions are also available:

```bash
# Import the image only
./scripts/import_and_run.sh import

# Run the system only
./scripts/import_and_run.sh run

# Specify a TAR file name
./scripts/import_and_run.sh deploy my-custom-name.tar
```

### Docker Project Layout

```text
ODD-SEC/
├── docker-compose.yml
├── .dockerignore
├── scripts/
│   ├── run_docker.sh
│   ├── export_image.sh
│   └── import_and_run.sh
├── src/
├── launch/
├── config/
├── model/
└── data/
```

Place model files in `model/`. Input and output data can be placed in `data/` according to the project configuration. Make the scripts executable before using them:

```bash
chmod +x scripts/*.sh
```

### Docker Image Naming and Versioning

Use versioned archive names to distinguish deployments:

```text
odd-sec-v1.0.tar
odd-sec-v1.1.tar
odd-sec-latest.tar
```

To compress the image for transfer:

```bash
docker save odd-sec:latest | gzip > odd-sec.tar.gz
gunzip -c odd-sec.tar.gz | docker load
```

## 3. Using ODD-SEC

The commands below apply to a native installation. For Docker, run the corresponding commands inside the running container or use the project Docker scripts to start the container first.

### 3.1 Event Motion Compensation

Run `Motion_Compensation_Event.cpp` through the corresponding ROS launch file:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch datasync Motion_Compensation_Event.launch
```

In a second terminal, play the raw event data:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosbag play -r 0.1 raw_data.bag
```

To record the processed topics, use a third terminal:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosbag record /count_image /event_new -O event_image.bag
```

### 3.2 Convert an Event Bag to DAT

Run `event_bag_to_dat.cpp` using the `event_converter` ROS package:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
rosrun event_converter event_bag_to_dat <input_bag_file> <output_dat_file> <t>
```

Arguments:

- `<input_bag_file>`: input ROS bag file
- `<output_dat_file>`: output DAT file
- `<t>`: time parameter required by the converter

### 3.3 Convert a Bag File to Images

Run `bag_to_image.py` to export image data from a ROS bag file:

```bash
python3 bag_to_image.py /path/to/your/bagfile.bag /path/to/output/folder
```

### 3.4 Convert JSON Data to NPY

Run `json_to_npy.py` to convert a folder of JSON files into a NumPy file:

```bash
python3 json_to_npy.py /path/to/json/folder /path/to/output.npy
```

### 3.5 Start the Triggered Camera

Start the camera with the Trigger launch file:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch dvs_renderer dvxplorer_mono_trigger.launch
```

## Troubleshooting

### Check the Docker Image and Container Status

```bash
docker images
docker ps -a
```

For a TAR archive, also check its size:

```bash
ls -lh odd-sec.tar
```

### TAR Import Failure

Re-export the image and verify that the target machine has enough disk space:

```bash
./scripts/export_image.sh export
docker system df
docker system prune -a
```

Review the images and containers before running `docker system prune -a`, because it may remove resources that are still needed.

### Missing Model Files

Verify that the required models are present in `model/`:

```bash
ls model/
cp /path/to/models/* model/
```

### Inspect the Docker Image

```bash
docker inspect odd-sec:latest
```

### Installation Checklist

1. Confirm that Ubuntu 20.04 and ROS 1 Noetic are installed for native use.
2. Confirm that Docker and Docker Compose are installed for Docker use.
3. Build the workspace natively or build/import the Docker image.
4. Source the required environment before running ROS commands.
5. Confirm that the required model files are in `model/`.
6. Check the container status with `docker ps -a` when using Docker.

## Notes

This document describes the ODD-SEC native ROS 1 workflow, Docker installation options, and the main data-processing commands. Refer to the shell scripts and configuration files in the repository for project-specific parameters and runtime settings.
