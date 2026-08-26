#pragma once

#include <ros/ros.h>
#include <string>
#include <vector>
#include <mutex>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>


namespace panorama {

    extern std::vector<dvs_msgs::Event> event_buffer;
    extern std::vector<sensor_msgs::Imu> local_imu;





// Options of the method
struct OptionsProcess
{
    // Type of objective function to be optimized: 0=Variance, 1=RMS, etc.
    // See focus_funcs.h
    int contrast_measure;
};

// Structure that collects the options for warping the events onto
// a histogram or image: the "image of warped events"
struct OptionsWarp
{
    // Gaussian blur (in pixels) to make the image of warped events smoother,
    // and consequently, have a smoother objective function
    double blur_sigma;

    // Share a common pose for a event batch
    int event_batch_size;

    // Event sample rate
    int event_sample_rate;
};

struct OptionsImage
{
    // Visualization
    int panorama_height;
    int panorama_width;
    std::vector<double> R_matrix;
};

// Options for data recording and visualization
struct OptionsData
{
    // Visualization
    bool show_iwe;
};

// Options of the sliding window for the back-end
struct OptionSlidingwindow
{
    // Size of the time window (sec)
    double time_window_size;

    // Stride of sliding window (sec)
    double sliding_window_stride;
};

// Options for the back-end trajectory
struct OptionTraj
{
    // Time gap between two control points (knots)
    double dt_knots;

    // Degree of trajectory: 1=Linear, 3=Cubic
    int spline_degree;
};


// Options for the global map maintained in the back-end
struct OptionPanoMap
{
    // Resolution of the panoramic IWE
    int pano_height, pano_width;

    // Initial Y-angle to control the start point of the panorama
    double Y_angle;

    // Stop updateing at some point
    int max_update_times; // dt: 0.05s, max: 255, due to CV_8UC1

    // Minimal event rate to update IG [ev/s]
    // (For the case that camera stays still)
    int backend_min_ev_rate;
};

// Options for drone detection

struct Optionsfile
{
    std::string bag_file_path;

};
struct OptionsDet
{
    std::string engine_file_path;
    float conf_thresh;
    float nms_thresh;
    bool enable_rotation; // 是否启用旋转检测
};

struct OptionsCamera
{
    float pixel_size;
    float Focus;
};

struct OptionsAA
{
    int x_patches;
    int y_patches;
    double decay_sec;
};

struct Optionsbias
{
    double gyro_bias_x;
    double gyro_bias_y;
    double gyro_bias_z;
};

// A collection of the front-end parameters
struct AngVelEstParams
{
    OptionsWarp warp_opt;
    OptionsProcess process_opt;
    OptionsData data_opt;
    OptionsImage image_opt;
    OptionsCamera camera_opt;
    OptionsAA AA_opt;
    Optionsbias imu_bias_opt;
    Optionsfile file_opt;

    // Frequency of the output ang_vel
    double dt_ang_vel;
    // The number of events used for one angular velocity estimate
    size_t num_events_per_packet;
};

// A collection of the back-end parameters
struct PoseGraphParams
{
    OptionSlidingwindow sliding_window_opt;
    OptionsWarp warp_opt;
    OptionsProcess process_opt;
    OptionTraj traj_opt;
    OptionsData data_opt;
    OptionPanoMap map_opt;
    OptionsImage image_opt;

    // Show the trajectory on the map
    bool draw_FOV;
    // Gamma correction
    double gamma;
};

struct DetectorParams
{

    OptionsWarp warp_opt;
    OptionsImage image_opt;
    OptionsDet det_opt;
    Optionsfile file_opt;

};

struct ProcessedEventData {
    ros::Time timestamp;      // 事件时间戳
    // Eigen::Matrix3d delta_rotation; // 旋转增量
    Eigen::Vector2d px_mosaic;      // 投影坐标
    float t_diff;
};

}
