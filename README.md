# ODD-SEC: Onboard Drone Detection with a Spinning Event Camera

ODD-SEC is a real-time onboard drone detection system based on a spinning event camera. This repository contains the ROS 1 implementation of the system, including panorama construction and drone detection.


## Installation

ODD-SEC is developed for Ubuntu 20.04 and ROS Noetic. A native build is required for the complete detection pipeline. The provided Dockerfile offers an isolated build environment for the platform-independent ROS tools.

### Prerequisites

- Ubuntu 20.04 LTS
- ROS 1 Noetic
- Python 3
- `catkin_tools`
- OpenCV and Eigen
- CUDA and TensorRT for the `panorama` package
- libcaer for the bundled event-camera drivers

Install and configure ROS Noetic before building the workspace. The CUDA and TensorRT versions must be compatible with the target GPU platform.

Source ROS in each terminal before using ROS commands:

```bash
source /opt/ros/noetic/setup.bash
```

### Native Build

Clone the repository and build the ROS workspace:

```bash
git clone https://github.com/KuanDai0123/ODD-SEC.git
cd ODD-SEC
source /opt/ros/noetic/setup.bash
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setup.bash
```

Source ROS and the workspace in each new terminal:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/ODD-SEC/devel/setup.bash
```

After the build succeeds, continue with [Using ODD-SEC](#using-odd-sec).

### Docker Build

The Dockerfile provides an isolated ROS Noetic environment for the following platform-independent packages:

- `datasync`
- `event_converter`
- `photogate_reader`
- `dvs_msgs`

The Docker image does not include the `panorama` package, CUDA, TensorRT, camera passthrough, or GPU runtime configuration. Use the native build with a platform-specific CUDA and TensorRT environment to run the complete detection pipeline.

With Docker Engine installed, build the core image from the repository root:

```bash
docker build -t odd-sec:core .
```

Verify that the workspace was built successfully:

```bash
docker run --rm odd-sec:core rospack find datasync
```

## Using ODD-SEC

The complete ODD-SEC pipeline requires a native build. Source ROS and the workspace before running the system:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
```

### Offline Detection

Run ODD-SEC on a ROS bag:

```bash
roslaunch panorama davis.launch bag_path:=/path/to/input.bag
```

### Live Detection

Start the event-camera driver:

```bash
roslaunch dvs_renderer dvxplorer_mono_trigger.launch
```

In another terminal, source ROS and the workspace, then start ODD-SEC:

```bash
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch panorama davis_live.launch engine_file_path:=/path/to/model.engine
```

## Citation

If you find this work useful, please cite:

```bibtex
@misc{dai2026oddsec,
  title         = {ODD-SEC: Onboard Drone Detection with a Spinning Event Camera},
  author        = {Kuan Dai and Hongxin Zhang and Sheng Zhong and Yi Zhou},
  year          = {2026},
  eprint        = {2603.06265},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2603.06265}
}
```

## Contact

For any questions or inquiries, please contact us at:

- **KuanDai**: daikuan@hnu.edu.cn
- **HongxinZhang**: zhx_2514@hnu.edu.cn

Or open an issue in this repository.
