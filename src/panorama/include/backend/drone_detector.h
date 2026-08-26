#ifndef DRONE_DETECTOR_H
#define DRONE_DETECTOR_H

#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <fstream>

#include "utils/parameters.h"
#include "frontend/ang_vel_estimator.h"
#include <vector>
#include <string>
#include <memory>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include "backend/post_process.h"
using namespace nvinfer1;

namespace panorama {

struct Detection {
    cv::Rect box;
    float score;
    int class_id;
};

class AngVelEstimator;

class DroneDetector {
public:
    DroneDetector(ros::NodeHandle* nh);
    ~DroneDetector();
    
    void initialize(int camera_width, int camera_height,
            const DetectorParams &opt,
            std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr);

    void setFrontend(AngVelEstimator* ptr) { ang_vel_estimator_ = ptr; }

    std::vector<Detection> detect(const std::vector<cv::Mat>& sequence_images, std::string time_path, ros::Time t0);
    void show_count_image(cv::Mat& count_image, const std::vector<Detection>& detections, ros::Time& stamp);
    void AngleCalculations(float x,float y,ros::Time& t0,ros::Time& stamp);
    std::vector<cv::Mat> createSequenceImages(const std::vector<int>& count_image, int camera_width, int camera_height, int num_channels);
private:
    bool loadEngine(const std::string& engine_file_path);
    bool prepareBuffers();
    void preprocessImage(const cv::Mat& image, std::vector<float>& input_buffer);
    std::vector<Detection> postprocessResults(const float* output_data, int original_image_width, int original_image_height);
    static float iou(const cv::Rect& a, const cv::Rect& b);
    static std::vector<Detection> doNMS(const std::vector<Detection>& detections, float iou_thresh);
    void sphericalToCartesian(double r, double theta_rad, double phi_rad, double& x, double& y, double& z);
    void publishSphericalMarker(ros::Publisher& pub, double r, double theta_rad, double phi_rad, const std::string& frame_id );
    void publishOriginAxes(ros::Publisher& pub, const std::string& frame_id ) ;
    void prepareSequenceInput(const std::vector<cv::Mat>& sequence_images, std::vector<float>& sequence_input_host);
    bool initializePostProcessBuffers();
    void cleanupPostProcessBuffers();

    ros::NodeHandle nh_;
    ros::Publisher m_count_image_pub_;
    ros::Publisher m_marker_pub_;

    DetectorParams m_params;
    std::string bag_file_path;
    std::string output_path;

    std::vector<cv::Point3d> m_precomputed_bearing_vectors_ptr_;
    Eigen::Matrix3d m_R;

    double pi = M_PI;

    int m_cam_width_, m_cam_height_;

    bool m_initialized = false;
    bool m_enable_rotation = true;
    std::string m_engine_file_path;

    AngVelEstimator* ang_vel_estimator_;

    nvinfer1::ICudaEngine* m_engine = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    nvinfer1::IRuntime* m_runtime = nullptr;
    cudaStream_t m_cuda_stream = nullptr;
    void* m_gpu_buffers[3] = {nullptr};
    nvinfer1::Dims m_input_dims, m_output_dims;
    int m_input_c = 3;
    size_t m_input_buffer_size_elements = 0;
    size_t m_output_buffer_size_elements = 0;
    float m_conf_thresh = 0.5f;
    float m_nms_thresh = 0.5f;
    int m_max_detections = 100;

    
    float* m_gpu_best_box = nullptr;
    float* m_gpu_best_score = nullptr;
    int* m_gpu_found_detection = nullptr;

};
}
#endif