#include "backend/drone_detector.h"
#include <fstream>
#include <algorithm> // For std::sort
#include <cmath>     // For expf


namespace panorama {
DroneDetector::DroneDetector(ros::NodeHandle* nh)
    : nh_("~"),
      m_marker_pub_ (nh_.advertise<visualization_msgs::Marker>("polar_line_3d", 10)),
      m_count_image_pub_(nh_.advertise<sensor_msgs::Image>("count_image", 10)) ,
      m_gpu_buffers{nullptr, nullptr}{

}

DroneDetector::~DroneDetector() {

    m_count_image_pub_.shutdown();
    LOG_INFO("Releasing TensorRTDetector resources...");
    if (m_cuda_stream) {
        cudaStreamDestroy(m_cuda_stream);
        m_cuda_stream = nullptr;
    }
    if (m_gpu_buffers[0]) {
        cudaFree(m_gpu_buffers[0]);
        m_gpu_buffers[0] = nullptr;
    }
    if (m_gpu_buffers[1]) {
        cudaFree(m_gpu_buffers[1]);
        m_gpu_buffers[1] = nullptr;
    }
    if (m_context) {
        m_context->destroy();
        m_context = nullptr;
    }
    if (m_engine) {
        m_engine->destroy();
        m_engine = nullptr;
    }
    if (m_runtime) {
        m_runtime->destroy();
        m_runtime = nullptr;
    }

    m_marker_pub_.shutdown();
    m_count_image_pub_.shutdown();
    LOG_INFO("TensorRTDetector resources released.");
}

void DroneDetector::initialize(int camera_width, int camera_height,
    const DetectorParams &opt,
    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr){
// Load params
    m_params = opt;

    m_cam_width_ = camera_width;
    m_cam_height_ = camera_height;

    LOG_INFO("Input W: %d, H: %d", m_cam_width_, m_cam_height_);

    m_precomputed_bearing_vectors_ptr_ = *precomputed_bearing_vectors_ptr;

    std::vector<double> R_matrix = m_params.image_opt.R_matrix;
    std::string engine_file_path = m_params.det_opt.engine_file_path;
    float conf_thresh = m_params.det_opt.conf_thresh;
    float nms_thresh = m_params.det_opt.nms_thresh;

    if (R_matrix.size() == 9) {
        Eigen::Map<const Eigen::Matrix3d> R_raw(R_matrix.data());
        m_R = R_raw.transpose();  // ROS数组是行优先，需转置为列优先
    } else {
        ROS_ERROR("Invalid rotation matrix size");
        m_R = Eigen::Matrix3d::Identity();
    }

    LOG_INFO("Initializing TensorRTDetector...");
    LOG_INFO("Engine path: %s", engine_file_path.c_str());
    LOG_INFO("Conf Thresh: %.2f, NMS Thresh: %.2f", m_conf_thresh, m_nms_thresh);

    if (!loadEngine(engine_file_path)) {
        LOG_ERROR("Failed to load TensorRT engine.");
        return;
    }

    if (!m_engine) {
        LOG_ERROR("Engine not created after loadEngine.");
        return;
    }

    m_context = m_engine->createExecutionContext();
    prepareBuffers();
    cudaStreamCreate(&m_cuda_stream);

    m_initialized = true;
    LOG_INFO("DroneDetector initialized successfully!");

// count_pano = cv::Mat::zeros(panorama_height_, panorama_width_, CV_32F); 

}


// Loading engine
bool DroneDetector::loadEngine(const std::string& engine_file_path) {
    std::ifstream engine_file(engine_file_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        LOG_ERROR("Failed to open engine file: %s", engine_file_path.c_str());
        return false;
    }

    std::streamsize size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    LOG_INFO("Attempting to load engine file: %s with size: %ld bytes.", engine_file_path.c_str(), size);

    std::vector<char> engine_data(static_cast<size_t>(size));
    engine_file.read(engine_data.data(), size);

    m_runtime = nvinfer1::createInferRuntime(m_logger);
    m_engine = m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

    return true;
}

// Prepare GPU buffer for the following propressing
bool DroneDetector::prepareBuffers() {
    if (!m_engine) return false;

    m_input_binding_index = m_engine->getBindingIndex(m_engine->getIOTensorName(0)); // 假设第一个I/O张量是输入
    m_output_binding_index = m_engine->getBindingIndex(m_engine->getIOTensorName(1)); // 假设第二个I/O张量是输出

    if (m_input_binding_index < 0 || m_output_binding_index < 0) {
        LOG_ERROR("Failed to get valid binding indices. Input: %d, Output: %d", m_input_binding_index, m_output_binding_index);

        LOG_INFO("Available bindings:");
        for (int i = 0; i < m_engine->getNbBindings(); ++i) {
            LOG_INFO("Binding %d: Name='%s', IsInput=%s", i, m_engine->getBindingName(i), m_engine->bindingIsInput(i) ? "true" : "false");
        }
        return false;
    }

    m_input_dims = m_engine->getBindingDimensions(m_input_binding_index);
    m_output_dims = m_engine->getBindingDimensions(m_output_binding_index);

    nvinfer1::Dims batch_input_dims = nvinfer1::Dims4{1, m_input_c, m_cam_height_, m_cam_width_};
    if (!m_context->setBindingDimensions(m_input_binding_index, batch_input_dims)) {
        LOG_ERROR("Failed to set binding dimensions for input.");
        return false;
    }
    m_input_dims = batch_input_dims;

    // compute size of Buffer (input and output)
    m_input_buffer_size_elements = 1; // batch size
    for (int j = 1; j < m_input_dims.nbDims; ++j) {
        m_input_buffer_size_elements *= m_input_dims.d[j];
    }

    m_output_buffer_size_elements = 1;
    for (int j = 0; j < m_output_dims.nbDims; ++j) {
        m_output_buffer_size_elements *= m_output_dims.d[j];
    }

    // Malloc CUDA memory
    cudaMalloc(&m_gpu_buffers[0], m_input_buffer_size_elements * sizeof(float));   // input buffer
    cudaMalloc(&m_gpu_buffers[1], m_output_buffer_size_elements * sizeof(float)); // output buffer

    // Binding context
    m_context->setTensorAddress(m_engine->getBindingName(m_input_binding_index), m_gpu_buffers[0]);
    m_context->setTensorAddress(m_engine->getBindingName(m_output_binding_index), m_gpu_buffers[1]);

    return true;
}

// Preprocessing images
void DroneDetector::preprocessImage(const cv::Mat& image, std::vector<float>& input_buffer) {
    cv::Mat rgb, resized;

    cv::cvtColor(image, rgb, cv::COLOR_GRAY2BGR);
    cv::resize(rgb, resized, cv::Size(m_cam_width_, m_cam_height_));
    resized.convertTo(resized, CV_32F);

    // HWC -> CHW
    std::vector<float> input(m_input_c * m_cam_height_ * m_cam_width_);
    std::vector<cv::Mat> channels(m_input_c);
    cv::split(resized, channels);

    for (int c = 0; c < m_input_c; ++c) {
        memcpy(input.data() + c*m_cam_height_*m_cam_width_, 
                channels[c].data,
                m_cam_height_*m_cam_width_*sizeof(float));
    }
}

// Main detector
std::vector<Detection> DroneDetector::detect(const cv::Mat& image) {
    if (!m_initialized || image.empty()) {
        LOG_ERROR("Detector not initialized or empty image provided.");
        return {};
    }

    std::vector<float> preprocess_input;
    preprocessImage(image, preprocess_input);

    // Copy preprocessed data to GPU
    if (cudaMemcpyAsync(m_gpu_buffers[0], preprocess_input.data(),
                        m_input_buffer_size_elements * sizeof(float),
                        cudaMemcpyHostToDevice, m_cuda_stream) != cudaSuccess) {
        LOG_ERROR("Failed to copy input data to GPU.");
        return {};
    }

    // Inference    //TODO Will it work well?
    if (!m_context->enqueueV3(m_cuda_stream)) {
        LOG_ERROR("Failed to execute inference.");
        return {};
    }

    // Copy Data back to CPU 
    std::vector<float> output_cpu_buffer(m_output_buffer_size_elements);
    if (cudaMemcpyAsync(output_cpu_buffer.data(), m_gpu_buffers[1],
                        m_output_buffer_size_elements * sizeof(float),
                        cudaMemcpyDeviceToHost, m_cuda_stream) != cudaSuccess) {
        LOG_ERROR("Failed to copy output data from GPU.");
        return {};
    }

    // Synchronize CUDA Stream
    cudaStreamSynchronize(m_cuda_stream);

    return postprocessResults(output_cpu_buffer.data(), image.cols, image.rows);
}

std::vector<Detection> DroneDetector::postprocessResults(const float* output, int original_image_width, int original_image_height) {
    std::vector<Detection> detections;

    int num_proposals = m_output_buffer_size_elements / 7;

    for (int i = 0; i < num_proposals; ++i) {
        float x0 = output[i * 7 + 0] - output[i * 7 + 2] / 2;
        float y0 = output[i * 7 + 1] - output[i * 7 + 3] / 2;
        float x1 = output[i * 7 + 0] + output[i * 7 + 2] / 2;
        float y1 = output[i * 7 + 1] + output[i * 7 + 3] / 2;

        float obj_conf = output[i * 7 + 4];
        float cls_conf = output[i * 7 + 5];

        int   cid = static_cast<int>(output[i*7 + 6] + 0.5f);
        float score = obj_conf * cls_conf;
        if (score > m_conf_thresh) {
            detections.push_back({cv::Rect(x0, y0, x1-x0, y1-y0), score, cid});
        }
    }
    auto finalDets = doNMS(detections, m_nms_thresh);

    return finalDets;
}

// Compute IoU
float DroneDetector::iou(const cv::Rect& a, const cv::Rect& b) {
    float inter_area = static_cast<float>((a & b).area());
    float union_area = static_cast<float>(a.area() + b.area() - inter_area);
    return (union_area > 0) ? (inter_area / union_area) : 0.0f;
}

// NMS
std::vector<Detection> DroneDetector::doNMS(const std::vector<Detection>& detections, float iou_thresh) {
    if (detections.empty()) {
        return {};
    }
    std::vector<Detection> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<Detection> nms_results;
    std::vector<bool> suppressed(sorted_detections.size(), false);

    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        nms_results.push_back(sorted_detections[i]);
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            // 仅对相同类别的进行NMS (如果你的模型有多类别且希望分别NMS)
            if (sorted_detections[i].class_id == sorted_detections[j].class_id) {
                 if (iou(sorted_detections[i].box, sorted_detections[j].box) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return nms_results;
}

void DroneDetector::show_count_image(cv::Mat& count_image, const std::vector<Detection>& detections) {
    // Create image
    cv::Mat image;
    cv::cvtColor(count_image, image, cv::COLOR_GRAY2BGR);


    if (!detections.empty()) {
        // Drone detected in the image scene, add rectangle to image and process 
        for (auto& d: detections) {
            cv::rectangle(image, d.box, cv::Scalar(0, 255, 0), 2);
            char buf[32];
            std::cout << "Mapping!" << std::endl;
            cv::putText(image, buf, d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        }
    }

    // Create ROS message head with timestamp
    std_msgs::Header header;
    // header.stamp = timestamp;

    // Transform OpenCV image to ROS image message
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", image).toImageMsg();

    // Publish
    m_count_image_pub_.publish(msg);

}


void DroneDetector::AngleCalculations(float x,float y,ros::Time& t0,ros::Time& stamp)
{

    const cv::Point3d bvec = m_precomputed_bearing_vectors_ptr_.at(y*m_cam_width_ + x);
    Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);

    float t_diff = double(t0.toNSec() - stamp.toNSec()) / 1000000000.0;

    double cos_term = cos(2 * pi * t_diff);
    double sin_term = sin(2 * pi * t_diff);
    
    Eigen::Matrix3d R_eigen_;
    R_eigen_ << cos_term, m_R(2, 1) * sin_term, m_R(2, 2) * sin_term,
                0, m_R(1, 1), m_R(1, 2),
                -sin_term, m_R(2, 1) * cos_term, m_R(2, 2) * cos_term;

    Eigen::Vector3d e_ray_w = R_eigen_ * e_ray_cam;

    const double phi = std::atan2(e_ray_w[0], e_ray_w[2]);
    const double theta = std::asin(e_ray_w[1] / e_ray_w.norm());

    std::cout << "phi :" << phi << std::endl;
    std::cout << "theta :" << theta << std::endl;

    double r=1;
    publishSphericalMarker(m_marker_pub_, r, theta, phi,"map");

}


// 球坐标转笛卡尔坐标
void DroneDetector::sphericalToCartesian(double r, double theta_rad, double phi_rad, 
                         double& x, double& y, double& z) {
    x = r * sin(phi_rad) * cos(theta_rad);
    y = r * sin(phi_rad) * sin(theta_rad);
    z = r * cos(phi_rad);
}

// 计算并发布Marker消息的函数
void DroneDetector::publishSphericalMarker(ros::Publisher& pub, 
                          double r, double theta_rad, double phi_rad, 
                          const std::string& frame_id ) {
    // 计算终点笛卡尔坐标
    double x_end, y_end, z_end;
    sphericalToCartesian(r, theta_rad, phi_rad, x_end, y_end, z_end);

    // 1. 创建直线Marker
    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "polar_line_3d";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.scale.x = 0.05; // 线宽
    line_marker.color.b = 1.0;  // 蓝色
    line_marker.color.a = 1.0;  // 不透明度

    // 设置起点和终点（LINE_STRIP通过points定义方向，无需orientation）
    geometry_msgs::Point start_point, end_point;
    start_point.x = 0; start_point.y = 0; start_point.z = 0;
    end_point.x = x_end; end_point.y = y_end; end_point.z = z_end;
    line_marker.points.push_back(start_point);
    line_marker.points.push_back(end_point);

    // 2. 创建文本Marker
    visualization_msgs::Marker text_marker;
    text_marker.header = line_marker.header;
    text_marker.ns = "polar_text_3d";
    text_marker.id = 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    const double rad_to_deg = 180.0 / pi;
    double theta_deg = theta_rad * rad_to_deg;
    double phi_deg = phi_rad * rad_to_deg;

    text_marker.text = "End: (" + std::to_string(x_end).substr(0, 4) + ", " + 
    std::to_string(y_end).substr(0, 4) + ", " + 
    std::to_string(z_end).substr(0, 4) + ")\n" +
    "Spherical: r=" + std::to_string(r).substr(0, 3) + 
    ", θ=" + std::to_string(theta_deg).substr(0, 2) + "°" +
    ", φ=" + std::to_string(phi_deg).substr(0, 2) + "°";

    // 关键修复：初始化文本的位姿（位置+方向）
    text_marker.pose.position = end_point;
    text_marker.pose.orientation.x = 0.0;  // 单位四元数
    text_marker.pose.orientation.y = 0.0;
    text_marker.pose.orientation.z = 0.0;
    text_marker.pose.orientation.w = 1.0;

    text_marker.scale.z = 0.2;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.a = 1.0;

    // 发布Markers
    pub.publish(line_marker);
    pub.publish(text_marker);
}
















}