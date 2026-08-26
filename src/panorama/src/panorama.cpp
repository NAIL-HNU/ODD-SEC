#include "panorama.h"
#include "utils/parameters.h"
#include <glog/logging.h>
#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/PointStamped.h>
#include <filesystem>
#include <ctime>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <chrono>
#include <sensor_msgs/NavSatFix.h>
#include <boost/filesystem.hpp>

using namespace std;  

namespace panorama {

Panorama::Panorama(ros::NodeHandle& events_nh, ros::NodeHandle& imu_nh)
    : events_nh_(events_nh),
      imu_nh_(imu_nh),
      pnh_("~"), 
      got_camera_info_(false) 
{
    const std::string events_topic = pnh_.param<std::string>("events_topic", "/dvs/events");
    const std::string camera_info_topic = pnh_.param<std::string>("camera_info_topic", "/dvs/camera_info");
    const std::string imu_topic = pnh_.param<std::string>("imu_topic", "/dvs/imu");
    const std::string compensation_topic = pnh_.param<std::string>("compensation_topic", "/timestamp_topic");
    const std::string trigger_topic = pnh_.param<std::string>("trigger_topic", "/photogate_reader/photogate_state");
    const std::string GPS_topic = pnh_.param<std::string>("GPS_topic", "/dvs/ext_trigger");
    const std::string detection_topic = pnh_.param<std::string>("detection_topic", "/detection_data");

    LOG(INFO) << "Event topic: " << events_topic;
    LOG(INFO) << "Camera info topic: " << camera_info_topic;
    LOG(INFO) << "Imu topic: " << imu_topic;

    camera_info_sub_ = imu_nh.subscribe(camera_info_topic, 0, &Panorama::cameraInfoCallback, this);
    event_sub_ = events_nh.subscribe(events_topic, 1, &Panorama::eventsCallback, this);
    imu_sub_ = imu_nh.subscribe(imu_topic, 0, &Panorama::imuCallback, this);
    compensation_sub_ = events_nh.subscribe(compensation_topic, 1, &Panorama::compensationCallback, this);
    trigger_sub_ = imu_nh.subscribe(trigger_topic, 1, &Panorama::triggerCallback, this);
    GPS_sub_ = imu_nh.subscribe(GPS_topic, 1, &Panorama::GPSCallback, this);
    detection_sub_ = events_nh.subscribe(detection_topic, 1, &Panorama::detectionCallback, this);

    got_camera_info_ = false;
    is_t0_p_set_ = false; 
    first_event_received = true;
    first_trigger_received = true;

    front_end_params_.warp_opt.event_sample_rate = pnh_.param<int>("frontend_event_sample_rate", 1);
    front_end_params_.image_opt.panorama_height = pnh_.param<int>("panorama_height", 1);
    front_end_params_.image_opt.panorama_width = pnh_.param<int>("panorama_width", 1);
    front_end_params_.camera_opt.Focus = pnh_.param<float>("Focus", 1);
    front_end_params_.camera_opt.pixel_size = pnh_.param<float>("pixel_size", 1);
    front_end_params_.AA_opt.x_patches = pnh_.param<int>("x_patches", 1);
    front_end_params_.AA_opt.y_patches = pnh_.param<int>("y_patches", 1);
    front_end_params_.AA_opt.decay_sec = pnh_.param<double>("decay_sec", 1);
    front_end_params_.imu_bias_opt.gyro_bias_x = pnh_.param<double>("gyro_bias_x", 1);
    front_end_params_.imu_bias_opt.gyro_bias_y = pnh_.param<double>("gyro_bias_y", 1);
    front_end_params_.imu_bias_opt.gyro_bias_z = pnh_.param<double>("gyro_bias_z", 1);

    
    back_end_params_.det_opt.engine_file_path = pnh_.param<std::string>("engine_file_path", "./yolox_s_fp16_cpp.engine");
    back_end_params_.det_opt.conf_thresh = pnh_.param<float>("conf_thresh", 1);
    back_end_params_.det_opt.nms_thresh = pnh_.param<float>("nms_thresh", 1);
    back_end_params_.det_opt.enable_rotation = pnh_.param<bool>("enable_rotation", false);
    back_end_params_.file_opt.bag_file_path = pnh_.param<std::string>("bag_file", "2025-08-17-18-56-42.bag");
    back_end_params_.file_opt.output_dir = pnh_.param<std::string>("output_dir", ".");
    

    m_threshold = pnh_.param<int>("threshold", 1);

    std::vector<double> default_R_matrix = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    if (!pnh_.getParam("R_matrix", front_end_params_.image_opt.R_matrix)) {
        front_end_params_.image_opt.R_matrix = default_R_matrix;
        ROS_WARN("No R_matrix provided, using identity matrix");
    } else if (front_end_params_.image_opt.R_matrix.size() != 9) {
        front_end_params_.image_opt.R_matrix = default_R_matrix;
        ROS_ERROR("Invalid R_matrix size, using identity matrix");
    }

    if (!pnh_.getParam("R_matrix", back_end_params_.image_opt.R_matrix)) {
        back_end_params_.image_opt.R_matrix = default_R_matrix;
        ROS_WARN("No R_matrix provided, using identity matrix");
    } else if (back_end_params_.image_opt.R_matrix.size() != 9) {
        back_end_params_.image_opt.R_matrix = default_R_matrix;
        ROS_ERROR("Invalid R_matrix size, using identity matrix");
    }

    bag_file_path=back_end_params_.file_opt.bag_file_path;
    boost::filesystem::path path_obj(bag_file_path);
    std::string bag_filename = path_obj.stem().string();
    std::cout << "bag_filename : " << bag_filename << std::endl;

    const boost::filesystem::path output_dir(back_end_params_.file_opt.output_dir);
    boost::system::error_code error;
    boost::filesystem::create_directories(output_dir, error);
    if (error) {
        ROS_WARN("Failed to create output directory '%s': %s",
                 output_dir.string().c_str(), error.message().c_str());
    }
    output_path = (output_dir / (bag_filename + "_panorama_output.txt")).string();
    time_path = (output_dir / (bag_filename + "_time_consume.txt")).string();
    m_time_norm = std::chrono::high_resolution_clock::now();

    back_end_params_.image_opt.panorama_height = pnh_.param<int>("panorama_height", 1);
    back_end_params_.image_opt.panorama_width = pnh_.param<int>("panorama_width", 1);

    LOG(INFO) << "frontend_event_sample_rate: " << front_end_params_.warp_opt.event_sample_rate;

    int panorama_height_ = front_end_params_.image_opt.panorama_height;
    int panorama_width_ = front_end_params_.image_opt.panorama_width;

    ang_vel_estimator_ = new AngVelEstimator(&nh_);

    drone_detector_ = new DroneDetector(&nh_);

    ang_vel_estimator_->setBackend(drone_detector_);
    drone_detector_->setFrontend(ang_vel_estimator_);
}

Panorama::~Panorama() {

    if (ang_vel_estimator_) {
        delete ang_vel_estimator_;
        ang_vel_estimator_ = nullptr;
    }
    if (drone_detector_) {
        delete drone_detector_;
        drone_detector_ = nullptr;
    }
    camera_info_sub_.shutdown();
    event_sub_.shutdown();
    imu_sub_.shutdown();
}

void Panorama::precomputeBearingVectors() {

    const int width = cam.fullResolution().width;
    const int height = cam.fullResolution().height;

    if (!cam.initialized()) {
        std::cerr << "Camera model is not initialized." << std::endl;
    }

    std::cout << "Camera matrix K:\n" << cam.fullIntrinsicMatrix() << std::endl;
    std::cout << "Distortion coefficients D: " << cam.distortionCoeffs() << std::endl;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Point2d rectified_point = cam.rectifyPoint(cv::Point2d(x, y));
            cv::Point3d bearing_vec = cam.projectPixelTo3dRay(rectified_point);
            precomputed_bearing_vectors.emplace_back(bearing_vec);
        }
    }
}

void Panorama::cameraInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& camera_info) {

    if (!got_camera_info_) {
        ROS_INFO("Loading camera information");
        cam.fromCameraInfo(camera_info);
        got_camera_info_ = true;

        ROS_INFO("Camera info got");
        camera_info_sub_.shutdown();

        precomputeBearingVectors();
        ang_vel_estimator_->initialize(&cam, front_end_params_, precomputed_bearing_vectors);

        int camera_width = cam.fullResolution().width;
        int camera_height = cam.fullResolution().height;
        drone_detector_->initialize(camera_width, camera_height,
                                          back_end_params_,
                                          &precomputed_bearing_vectors);
        
        m_image_to_detect.create(camera_height, camera_width, CV_8UC1);
        ROS_INFO("System initialized with camera info");
    }

}

void Panorama::detectionCallback(const panorama::CountImage::ConstPtr& msg) {
    auto callback_start = std::chrono::high_resolution_clock::now();

    ros::Time t0 = msg->init_time;
    ros::Time stamp = msg->header.stamp;
    const std::vector<int>& count_image = msg->count_image;
    int camera_width = cam.fullResolution().width;
    int camera_height = cam.fullResolution().height;
    const int num_channels = 10;

    if (count_image.size() != camera_width * camera_height * num_channels) {
        ROS_ERROR("Received count_image size (%lu) doesn't match expected size (%d x %d x %d)",
                 count_image.size(), camera_width, camera_height, num_channels);
        return;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> sequence_images = drone_detector_->createSequenceImages(count_image, camera_width, camera_height, num_channels);
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    cv::Mat display_img = sequence_images[0].clone();

    std::vector<Detection> dets = drone_detector_->detect(sequence_images, time_path, t0);

    auto img_start_time = std::chrono::high_resolution_clock::now();
    drone_detector_->show_count_image(display_img, dets, t0);
    auto img_end_time = std::chrono::high_resolution_clock::now();
    double countimg_ms = std::chrono::duration_cast<std::chrono::microseconds>(img_end_time - img_start_time).count() / 1000.0;
    if (!dets.empty()) {
        for (const auto& d : dets) {
            auto angle_start = std::chrono::high_resolution_clock::now();
            float centerX = d.box.x + static_cast<double>(d.box.width) / 2.0;
            float centerY = d.box.y + static_cast<double>(d.box.height) / 2.0;
            drone_detector_->AngleCalculations(centerX, centerY, t0, stamp);
            auto angle_end = std::chrono::high_resolution_clock::now();
            double angle_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(angle_end - angle_start).count() / 1000.0;

            std::ofstream outFile(time_path, std::ios::app);
            outFile << "[ " << t0 << " ] bearing vector computing " << angle_elapsed_ms << std::endl;
            outFile.close();
            
        }
    }
    auto final_time = std::chrono::high_resolution_clock::now();
    double end_ms = std::chrono::duration_cast<std::chrono::microseconds>(final_time - m_time_norm).count() / 1000.0;
    double back_end = std::chrono::duration_cast<std::chrono::microseconds>(callback_start - m_time_norm).count() / 1000.0;
    
    auto callback_end = std::chrono::high_resolution_clock::now();
    double compute_callback_ms = std::chrono::duration_cast<std::chrono::microseconds>(callback_end - callback_start).count() / 1000.0;
    std::ofstream outFile(time_path, std::ios::app);
    outFile << "[ " << t0 << " ] transform 1 " << elapsed_ms << std::endl;
    outFile << "[ " << t0 << " ] count_img time  " << countimg_ms << std::endl;
    outFile << "[ " << t0 << " ] DetectionCallback  " << compute_callback_ms << std::endl;
    outFile << "[ " << t0 << " ] back-end start  " << back_end << std::endl;
    outFile << "[ " << t0 << " ] back-end stop " << end_ms << std::endl;

    outFile.close();

}

void Panorama::triggerCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    stamp = msg->header.stamp;
    first_trigger_received = false;
}

void Panorama::GPSCallback(const geometry_msgs::PointStamped::ConstPtr& msg) {
    GPS_stamp = msg->header.stamp;
    std::ofstream outFile(output_path, std::ios::app);
    outFile << "\n-----------------------------------" << std::endl;
    outFile << "[ GT  ] Timestamp: " << GPS_stamp.toNSec() << std::endl;
    outFile.close();
}

void Panorama::compensationCallback(const std_msgs::Header::ConstPtr& msg) {
    
    ros::Time stamp_ =stamp;

    if (!got_camera_info_) {
        ROS_WARN_THROTTLE(1, "Waiting for camera info...");
        return;
    }

    if (first_event_received == false && first_trigger_received==false) {

        {
            std::lock_guard<std::mutex> lock(imu_buffer_mutex_);
            this->m_local_imu = std::move(this->m_local_imu_raw);
            this->m_local_imu_raw.clear();
        }

        if (this->m_local_imu.empty()) return;
        auto move_start = std::chrono::high_resolution_clock::now();
        {
            std::lock_guard<std::mutex> lock(event_buffer_mutex_);
            this->m_event_buffer.insert(
                this->m_event_buffer.end(),
                std::make_move_iterator(this->m_event_buffer_raw.begin()),
                std::make_move_iterator(this->m_event_buffer_raw.end()));
            this->m_event_buffer_raw.clear();
            
        }
        auto move_end = std::chrono::high_resolution_clock::now();
        ros::Time t0=this->m_event_buffer[0].ts;

        double decay_sec=front_end_params_.AA_opt.decay_sec;

        auto compute_start = std::chrono::high_resolution_clock::now();
        ang_vel_estimator_->compute_ts_splitting(m_event_buffer, decay_sec, stamp_);
        auto compute_end = std::chrono::high_resolution_clock::now();

        double move_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(move_end - move_start).count() / 1000.0;
        double compute_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(compute_end - compute_start).count() / 1000.0;

        
        this->m_event_buffer.clear();
        this->m_local_imu.clear();
        auto callback_end = std::chrono::high_resolution_clock::now();
        double front_end = std::chrono::duration_cast<std::chrono::microseconds>(callback_end - m_time_norm).count() / 1000.0;
        std::ofstream outFile(time_path, std::ios::app);
        outFile << "[ " << t0 << " ] Data Reading - pop out: " << move_elapsed_ms << std::endl;
        outFile << "[ " << t0 << " ] Image Generation: " << compute_elapsed_ms << std::endl;
        outFile << "[ " << t0 << " ] front-end time: " << front_end << std::endl;
        outFile.close();

    } else {
        first_event_received = false;
        std::lock_guard<std::mutex> lock(this->imu_buffer_mutex_);
        this->m_local_imu_raw.clear();
   
    }
}

void Panorama::eventsCallback(const dvs_msgs::EventArray::ConstPtr& msg) {
    auto callback_start = std::chrono::high_resolution_clock::now();

    std::lock_guard<std::mutex> lock(this->event_buffer_mutex_);

    auto start_time = std::chrono::high_resolution_clock::now();

    this->m_event_buffer_raw.reserve(msg->events.size() / front_end_params_.warp_opt.event_sample_rate);
    
    for (auto ev = msg->events.begin(); ev < msg->events.end();
         ev += front_end_params_.warp_opt.event_sample_rate) {
            this->m_event_buffer_raw.emplace_back(*ev);
    }
    ang_vel_estimator_->signal();
    ros::Time t0=this->m_event_buffer_raw[0].ts;

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
    double start_ms = std::chrono::duration_cast<std::chrono::microseconds>(start_time - m_time_norm).count() / 1000.0;
    auto callback_end = std::chrono::high_resolution_clock::now();
    double compute_callback_ms = std::chrono::duration_cast<std::chrono::microseconds>(callback_end - callback_start).count() / 1000.0;

    std::ofstream outFile(time_path, std::ios::app);
    outFile << "[ " << t0 << " ] StartMs: " << start_ms << std::endl;
    outFile << "[ " << t0 << " ] Data Reading - read in: " << elapsed_ms << std::endl;
    outFile << "[ " << t0 << " ] EventsCallback: " << compute_callback_ms << std::endl;

    outFile.close();

}

void Panorama::imuCallback(const sensor_msgs::ImuConstPtr& imu) {

    std::lock_guard<std::mutex> lock(this->imu_buffer_mutex_);
    
    if (first_event_received == false) {
        this->m_local_imu_raw.emplace_back(*imu);
    }
}

}
