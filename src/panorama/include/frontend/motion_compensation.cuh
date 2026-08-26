#ifndef MOTION_COMPENSATION_CUH
#define MOTION_COMPENSATION_CUH

#include <stdint.h>

#include <ros/ros.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>

// CUDA 核函数声明
__global__ void motionCompensationKernel(const dvs_msgs::Event* event_buffer, int* count_image,
                                    const sensor_msgs::Imu* imu_buffer, int imu_count,
                                    int event_count, int cam_width, int cam_height,
                                    float pixel_focus_ratio, float Focus, float pixel_size,
                                    int64_t t0, int threshold);

// CUDA 加速函数接口
void run_motion_compensation_cuda(const dvs_msgs::Event* h_events, int num_events,
                                    const sensor_msgs::Imu* h_imu_buffer, int num_imu,
                                    int cam_width, int cam_height,
                                    float pixel_size, float focus,
                                    int64_t t0,
                                    int* h_count_image, int threshold);

void preprocessImuData(const sensor_msgs::Imu* imu_buffer, int imu_count, int64_t t0, 
                                    float& avg_angular_velocity_x, float& avg_angular_velocity_y, float& avg_angular_velocity_z);                                    

void preprocessEventTimestamps(const dvs_msgs::Event* h_events, int num_events, int64_t* event_timestamps);

#endif // MOTION_COMPENSATION_CUH