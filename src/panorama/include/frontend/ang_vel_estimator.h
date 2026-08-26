#pragma once

#include "backend/drone_detector.h"
#include "utils/parameters.h"

#include <opencv2/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_geometry/pinhole_camera_model.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <panorama/CountImage.h>
#include <tuple>

namespace panorama {

class PoseGraphOptimizer;
class DroneDetector;

class AngVelEstimator {
public:
    AngVelEstimator(ros::NodeHandle* nh);

    ~AngVelEstimator();
    

    void initialize(image_geometry::PinholeCameraModel* cam,
                                 const AngVelEstParams& val,
                                 const std::vector<cv::Point3d>& precomputed_bearing_vectors);
    void setInitialTimestamp(const ros::Time& t0) { t0_p = t0; }
    
    void setBackend(DroneDetector* ptr) { drone_detector_ = ptr; }
    bool processEvent(const dvs_msgs::Event& ev);
    std::vector<std::vector<int>> motion_compensation(const std::vector<dvs_msgs::Event>& event_buffer,const std::vector<sensor_msgs::Imu>& imu_buffer_, int threshold);
    std::vector<int> motion_compensation_cuda(const std::vector<dvs_msgs::Event>& event_buffer,const std::vector<sensor_msgs::Imu>& imu_buffer_);

    void publishCountImageData(const std::vector<std::vector<int>>& count_image,ros::Time& t0, ros::Time& stamp);
    void publishCountImageData_cuda(const std::vector<int>& count_image, ros::Time& t0, ros::Time& stamp);
    void compute_ts_splitting(const std::vector<dvs_msgs::Event>& event_queue,double decay_sec,ros::Time stamp);
    std::vector<std::vector<uint8_t>> AA_thread(
        const std::vector<dvs_msgs::Event>& event_queue,
        const ros::Time& external_sync_time,
        double decay_sec);
    std::vector<std::vector<uint8_t>> compute_ts(
        const std::vector<dvs_msgs::Event>& event_queue,
        const ros::Time& external_sync_time,
        double decay_sec);
    std::tuple<float, float, float> IMU_Average(const std::vector<sensor_msgs::Imu>& imu_buffer_);
    void publishNewImageData(const std::vector<std::vector<uint8_t>>& count_image,ros::Time& t0,ros::Time& stamp);
    void publishImage_ts(const std::vector<std::vector<uint8_t>>& count_image, ros::Time& t0);

    void show_count_image(std::vector<std::vector<int>>&count_image, int& max_count);
    void publishIMU(float x,float y,float z,uint32_t seq);
    void show_count_pano(cv::Mat& count_image, double& max_count, ros::Time timestamp) ;
    void signal();

    AngVelEstParams params;
    std::vector<dvs_msgs::Event> events_;
    

private:
    ros::NodeHandle* nh_;
    ros::NodeHandle nh;
    image_transport::ImageTransport it_;
    image_transport::Publisher pano_pub_;

    ros::Publisher pub_signal,detection_pub,imu_pub,image_pub;

    typedef long long int sll;
    double pi = M_PI;
    ros::Time t0_p;
    sll t0_p_;

    int cam_width_, cam_height_;
    Eigen::Matrix3d R;
    float count;
    cv::Mat count_pano;
    int cc;
    std::vector<double> R_matrix;
    cv::Matx33d camera_matrix_;
    Eigen::Vector2d center_;
    Eigen::Matrix3d R_eigen_;
    Eigen::Vector2d px_mosaic;

    std::vector<cv::Point3d> precomputed_bearing_vectors_;

    std::vector<sensor_msgs::Imu> imu_buffer_;
    std::vector<dvs_msgs::Event> local_events;
    std::vector<dvs_msgs::Event> segments;

    std::vector<std::vector<int>> count_image;
    std::vector<int> flat_count_image;

    float average_angular_rate_x, average_angular_rate_y,average_angular_rate_z;

    float Focus_;
    float pixel_size_; 
    float pixel_focus_ratio;

    int panorama_width_;
    int panorama_height_;

    int x_patches_;
    int y_patches_;

    double gyro_bias_x_;
    double gyro_bias_y_;
    double gyro_bias_z_;

    DroneDetector* drone_detector_;

    std::mutex mtx;
    std::mutex data_mutex_;
    

    std::chrono::high_resolution_clock::time_point T1;
    std::chrono::high_resolution_clock::time_point T2{};
    bool START;

};

}