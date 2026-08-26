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
// #include "utils/parameters.h"

#include <sensor_msgs/Image.h>
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
// #include <dvs_msgs/CountImage.h>
#include <panorama/CountImage.h>


namespace panorama {

class Panorama
{
public:
    Panorama(ros::NodeHandle& events_nh, ros::NodeHandle& imu_nh);  
    ~Panorama();

    std::vector<cv::Point3d> precomputed_bearing_vectors; //预计算
    image_geometry::PinholeCameraModel cam;//相机模型
    sensor_msgs::CameraInfo camera_info;



private:
    // Node handle used to subscribe to ROS topics
    ros::NodeHandle nh_;
    // Private node handle for reading parameters
    ros::NodeHandle pnh_;

    ros::NodeHandle events_nh_;  // 新增：事件专用 NodeHandle
    
    ros::NodeHandle imu_nh_;     // 新增：IMU 专用 NodeHandle
    ros::NodeHandle compensation_nh_;     // 新增：IMU 专用 NodeHandle

    ros::Subscriber event_sub_, imu_sub_,camera_info_sub_,compensation_sub_,trigger_sub_,detection_sub_, GPS_sub_;
    ros::Time t0_p;
    ros::Time stamp ,GPS_stamp;
    bool is_t0_p_set_; 
    bool first_event_received;
    bool got_camera_info_;    
    bool first_trigger_received;


    std::vector<dvs_msgs::Event> sampled_events;

    // std::vector<dvs_msgs::Event> event_buffer;

    // std::vector<dvs_msgs::Event> event_buffer_;

    // std::vector<sensor_msgs::Imu> imu_buffer;
    // std::vector<sensor_msgs::Imu> imu_buffer_;
    // moodycamel::ConcurrentQueue<sensor_msgs::Imu> imu_buffer;
    // moodycamel::ConcurrentQueue<sensor_msgs::Imu> imu_buffer_;
    
    // std::vector<sensor_msgs::Imu> local_imu;

    void precomputeBearingVectors();
    void eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg);
    void cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info);
    void imuCallback(const sensor_msgs::ImuConstPtr& imu);
    void triggerCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    // void triggerCallback(const std_msgs::Header::ConstPtr& msg);
    void GPSCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
    void compensationCallback(const std_msgs::Header::ConstPtr& msg);
    void detectionCallback(const panorama::CountImage::ConstPtr& msg);
    // void clearImuQueue(moodycamel::ConcurrentQueue<sensor_msgs::Imu>& q);

    std::mutex mtx;
    std::mutex data_mutex_;

    // Data buffer from message
    std::vector<dvs_msgs::Event> m_event_buffer_raw; 
    std::vector<sensor_msgs::Imu> m_local_imu_raw;

    // Data buffer for processing
    std::vector<dvs_msgs::Event> m_event_buffer; 
    std::vector<sensor_msgs::Imu> m_local_imu;

    // Lock
    std::mutex event_buffer_mutex_;
    std::mutex imu_buffer_mutex_;

    std::string bag_file_path;
    std::string output_path;
    std::string time_path;

    // Detection related
    cv::Mat m_image_to_detect;

    int c=0;
    double count=0;
    
    int c_=0;
    double count_=0;
    
    uint32_t seq;
    int m_threshold;

    AngVelEstParams front_end_params_;
    DetectorParams back_end_params_;

    AngVelEstimator* ang_vel_estimator_;       // Front-end
    // PoseGraphOptimizer* pose_graph_optimizer_; // Back-end
    DroneDetector* drone_detector_;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_time_norm;




};

}