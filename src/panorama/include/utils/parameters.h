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

struct OptionsProcess
{
    int contrast_measure;
};

struct OptionsWarp
{
    double blur_sigma;

    int event_batch_size;

    int event_sample_rate;
};

struct OptionsImage
{
    int panorama_height;
    int panorama_width;
    std::vector<double> R_matrix;
};

struct OptionsData
{
    bool show_iwe;
};

struct OptionSlidingwindow
{
    double time_window_size;  // seconds

    double sliding_window_stride;  // seconds
};

struct OptionTraj
{
    double dt_knots;  // seconds

    int spline_degree;
};

struct OptionPanoMap
{
    int pano_height, pano_width;

    double Y_angle;

    int max_update_times;

    int backend_min_ev_rate;  // events per second
};

struct Optionsfile
{
    std::string bag_file_path;
    std::string output_dir;

};
struct OptionsDet
{
    std::string engine_file_path;
    float conf_thresh;
    float nms_thresh;
    bool enable_rotation;
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

    double dt_ang_vel;
    size_t num_events_per_packet;
};

struct PoseGraphParams
{
    OptionSlidingwindow sliding_window_opt;
    OptionsWarp warp_opt;
    OptionsProcess process_opt;
    OptionTraj traj_opt;
    OptionsData data_opt;
    OptionPanoMap map_opt;
    OptionsImage image_opt;

    bool draw_FOV;
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
    ros::Time timestamp;
    Eigen::Vector2d px_mosaic;
    float t_diff;
};

}
