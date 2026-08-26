#pragma once
#include <ros/ros.h>
#include <thread>
#include <fstream>
#include <vector>
#include <string>

#include <sensor_msgs/CameraInfo.h>
#include <geometry_msgs/PointStamped.h>
#include <image_geometry/pinhole_camera_model.h>

#include "frontend/ang_vel_estimator.h"
#include "backend/pose_graph_optimizer.h"
#include "backend/drone_detector.h"

#include <sensor_msgs/Image.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <panorama/CountImage.h>

namespace panorama {

class Panorama
{
public:
    Panorama(ros::NodeHandle& events_nh, ros::NodeHandle& imu_nh);  
    ~Panorama();

    // Row-major lookup table: bearing_vectors[y * image_width + x].
    std::vector<cv::Point3d> precomputed_bearing_vectors;
    image_geometry::PinholeCameraModel cam;
    sensor_msgs::CameraInfo camera_info;

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // These handles are attached to the independent queues configured in node.cpp.
    ros::NodeHandle events_nh_;
    
    ros::NodeHandle imu_nh_;
    ros::NodeHandle compensation_nh_;

    ros::Subscriber event_sub_, imu_sub_,camera_info_sub_,compensation_sub_,trigger_sub_,detection_sub_, GPS_sub_;
    ros::Time t0_p;
    ros::Time stamp ,GPS_stamp;
    bool is_t0_p_set_; 
    bool first_event_received;
    bool got_camera_info_;    
    bool first_trigger_received;

    std::vector<dvs_msgs::Event> sampled_events;

    

    void precomputeBearingVectors();
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info);
    void imuCallback(const sensor_msgs::ImuConstPtr& imu);
    void triggerCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void GPSCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void compensationCallback(const std_msgs::Header::ConstPtr& msg);
    void detectionCallback(const panorama::CountImage::ConstPtr& msg);

    std::mutex mtx;
    std::mutex data_mutex_;

    // Callbacks append to the raw buffers; processing callbacks consume snapshots below.
    std::vector<dvs_msgs::Event> m_event_buffer_raw; 
    std::vector<sensor_msgs::Imu> m_local_imu_raw;

    std::vector<dvs_msgs::Event> m_event_buffer; 
    std::vector<sensor_msgs::Imu> m_local_imu;

    // Each mutex protects a raw buffer and its transfer into a processing snapshot.
    std::mutex event_buffer_mutex_;
    std::mutex imu_buffer_mutex_;

    std::string bag_file_path;
    std::string output_path;
    std::string time_path;

    cv::Mat m_image_to_detect;

    int c=0;
    double count=0;
    
    int c_=0;
    double count_=0;
    
    uint32_t seq;
    int m_threshold;

    AngVelEstParams front_end_params_;
    DetectorParams back_end_params_;

    AngVelEstimator* ang_vel_estimator_;
    DroneDetector* drone_detector_;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_time_norm;

};

}
