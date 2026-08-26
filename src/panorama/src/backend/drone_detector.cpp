#include <iostream>
#include "backend/drone_detector.h"
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Header.h>
#include <boost/filesystem.hpp>

#include <NvInfer.h>
using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};
static Logger gLogger;

namespace panorama {

DroneDetector::DroneDetector(ros::NodeHandle* nh_ptr)
    : nh_("~"),
      m_marker_pub_ (nh_ptr->advertise<visualization_msgs::Marker>("polar_line_3d", 10)),
      m_count_image_pub_(nh_ptr->advertise<sensor_msgs::Image>("count_image", 10))
       {

}

DroneDetector::~DroneDetector() {
    m_count_image_pub_.shutdown();
    m_marker_pub_.shutdown();
    cleanupPostProcessBuffers();
    if (m_gpu_buffers[0]) cudaFree(m_gpu_buffers[0]);
    if (m_gpu_buffers[1]) cudaFree(m_gpu_buffers[1]);
    if (m_gpu_buffers[2]) cudaFree(m_gpu_buffers[2]);
}

void DroneDetector::initialize(int camera_width, int camera_height,
    const DetectorParams &opt,
    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr){
    m_initialized = false;
    m_params = opt;

    m_cam_width_ = camera_width;
    m_cam_height_ = camera_height;

    m_precomputed_bearing_vectors_ptr_ = *precomputed_bearing_vectors_ptr;

    std::vector<double> R_matrix = m_params.image_opt.R_matrix;
    std::string engine_file_path = m_params.det_opt.engine_file_path;
    m_enable_rotation = true;

    bag_file_path=m_params.file_opt.bag_file_path;
    boost::filesystem::path path_obj(bag_file_path);
    std::string bag_filename = path_obj.stem().string();
    
    output_path = (boost::filesystem::path(m_params.file_opt.output_dir) /
                   (bag_filename + "_panorama_output.txt")).string();

    if (R_matrix.size() == 9) {
        Eigen::Map<const Eigen::Matrix3d> R_raw(R_matrix.data());
        // ROS arrays are row-major; Eigen maps raw data as column-major.
        m_R = R_raw.transpose();
    } else {
        ROS_ERROR("Invalid rotation matrix size. Using identity.");
        m_R = Eigen::Matrix3d::Identity();
    }

    std::string base_engine_path = m_params.det_opt.engine_file_path;
    
    if (m_enable_rotation == true) {
        m_engine_file_path = base_engine_path;
        size_t dot_pos = m_engine_file_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
        } else {
            m_engine_file_path += "_vertical";
        }
        ROS_INFO("Vertical mode: Using vertical model: %s", m_engine_file_path.c_str());
    } else if (m_enable_rotation == false) {
        m_engine_file_path = base_engine_path;
        size_t dot_pos = m_engine_file_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            m_engine_file_path.insert(dot_pos, "_horizontal");
        } else {
            m_engine_file_path += "_horizontal";
        }
        ROS_INFO("Horizontal mode: Using horizontal model: %s", m_engine_file_path.c_str());
    } else {
        m_enable_rotation = false;
        m_engine_file_path = base_engine_path;
        ROS_INFO("Default mode: Using default model: %s", m_engine_file_path.c_str());
    }

    if (!loadEngine(m_engine_file_path)) {
        ROS_ERROR("Engine loading failed, aborting initialization.");
        return;
    }
    m_context = m_engine->createExecutionContext();
    if (!m_context) {
        ROS_ERROR("Failed to create execution context.");
        return;
    }

    if (!prepareBuffers()) {
        ROS_ERROR("TensorRT buffer preparation failed.");
        return;
    }

    if (!initializePostProcessBuffers()) {
        ROS_ERROR("Post-processing buffers initialization failed.");
        return;
    }

    m_initialized = true;
    ROS_INFO("DroneDetector initialized successfully!");
}

bool DroneDetector::loadEngine(const std::string& engine_file_path) {
    std::ifstream engine_file(engine_file_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        ROS_ERROR("Failed to open engine file: %s", engine_file_path.c_str());
        return false;
    }

    std::streamsize size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);

    ROS_INFO("Attempting to load engine file: %s with size: %ld bytes.", engine_file_path.c_str(), static_cast<long>(size));

    std::vector<char> engine_data(static_cast<size_t>(size));
    if (size == 0) {
        ROS_ERROR("Engine file is empty: %s", engine_file_path.c_str());
        engine_file.close();
        return false;
    }
    engine_file.read(engine_data.data(), size);
    engine_file.close();

    if (m_runtime) {
        delete m_runtime;
        m_runtime = nullptr;
    }
    m_runtime = nvinfer1::createInferRuntime(gLogger);
    if (!m_runtime) {
        ROS_ERROR("Failed to create TensorRT Runtime.");
        return false;
    }

    if (m_engine) {
      m_engine = nullptr;
    }
    m_engine = m_runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if (!m_engine) {
        ROS_ERROR("Failed to deserialize CUDA engine from file: %s", engine_file_path.c_str());
        return false;
    }

    return true;
}

bool DroneDetector::prepareBuffers() {
    if (!m_engine || !m_context) {
        ROS_ERROR("Engine or context is null in prepareBuffers.");
        return false;
    }
    if (m_engine->getNbIOTensors() < 3) {
        ROS_ERROR("Engine does not have the required 3 I/O tensors.");
        return false;
    }
    // The exported engine contract is image input, temporal input, then output.
    const char* input_tensor_name = m_engine->getIOTensorName(0);
    const char* sequence_tensor_name = m_engine->getIOTensorName(1);
    const char* output_tensor_name = m_engine->getIOTensorName(2);

    if (!input_tensor_name || !sequence_tensor_name || !output_tensor_name) {
        ROS_ERROR("Failed to get I/O tensor names.");
        return false;
    }
    
    m_input_dims = m_engine->getTensorShape(input_tensor_name);
    m_output_dims = m_engine->getTensorShape(output_tensor_name);

    ROS_INFO("Available I/O Tensors:");
    for (int32_t i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
        ROS_INFO("I/O Tensor %d: Name='%s', Mode=%s", i, name, (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output"));
    }

    nvinfer1::Dims batch_input_dims; 
    batch_input_dims.nbDims = 4;
    batch_input_dims.d[0] = 1;
    batch_input_dims.d[1] = m_input_c;
    
    if (m_enable_rotation == true) {
        batch_input_dims.d[2] = m_cam_width_;
        batch_input_dims.d[3] = m_cam_height_;
    } else {
        batch_input_dims.d[2] = m_cam_height_;
        batch_input_dims.d[3] = m_cam_width_;
    }
    
    if (m_engine->getTensorIOMode(input_tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
        if (!m_context->setInputShape(input_tensor_name, batch_input_dims)) {
            ROS_ERROR("Failed to set input shape for tensor: %s", input_tensor_name);
            return false;
        }
    } else {
        ROS_ERROR("Tensor %s is not an input tensor, cannot set its shape.", input_tensor_name);
        return false;
    }

    m_input_buffer_size_elements = 1;
    nvinfer1::Dims actual_input_shape = m_engine->getTensorShape(input_tensor_name);
    for (int j = 0; j < actual_input_shape.nbDims; ++j) {
        m_input_buffer_size_elements *= actual_input_shape.d[j];
    }

    m_output_buffer_size_elements = 1;
    nvinfer1::Dims actual_output_shape = m_engine->getTensorShape(output_tensor_name);
    for (int j = 0; j < actual_output_shape.nbDims; ++j) {
        m_output_buffer_size_elements *= actual_output_shape.d[j];
    }

    if (m_gpu_buffers[0]) cudaFree(m_gpu_buffers[0]);
    if (m_gpu_buffers[1]) cudaFree(m_gpu_buffers[1]);

    // Keep buffer indices aligned with the TensorRT binding order above.
    cudaMalloc(&m_gpu_buffers[0], m_input_buffer_size_elements * sizeof(float));
    cudaMalloc(&m_gpu_buffers[1], m_input_buffer_size_elements * 10 * sizeof(float));
    cudaMalloc(&m_gpu_buffers[2], m_output_buffer_size_elements * sizeof(float));

    if (!m_context->setTensorAddress(input_tensor_name, m_gpu_buffers[0])) {
        ROS_ERROR("Failed to set tensor address for input: %s", input_tensor_name);
        return false;
    }
    if (!m_context->setTensorAddress(sequence_tensor_name, m_gpu_buffers[1])) {
        ROS_ERROR("Failed to set tensor address for sequence: %s", sequence_tensor_name);
        return false;
    }
    if (!m_context->setTensorAddress(output_tensor_name, m_gpu_buffers[2])) {
        ROS_ERROR("Failed to set tensor address for output: %s", output_tensor_name);
        return false;
    }

    return true;
}

void DroneDetector::preprocessImage(const cv::Mat& image, std::vector<float>& input_buffer_host) {
    cv::Mat rgb, resized;

    if (m_cam_width_ <= 0 || m_cam_height_ <= 0) {
        ROS_ERROR("Invalid dimensions for resizing: width=%d, height=%d", m_cam_width_, m_cam_height_);
        input_buffer_host.clear();
        return;
    }

    // The image branch is a three-channel CHW float tensor. Values remain in [0, 255].
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        rgb = image.clone();
    } else {
        ROS_ERROR("Unsupported image channel count: %d", image.channels());
        input_buffer_host.clear();
        return;
    }

    cv::Size target_size;
    if (m_enable_rotation == true) {
        target_size = cv::Size(m_cam_height_, m_cam_width_);
    } else {
        target_size = cv::Size(m_cam_width_, m_cam_height_);
    }
    cv::resize(rgb, resized, target_size);
    resized.convertTo(resized, CV_32F);

    if (m_input_c <= 0) {
        ROS_ERROR("Invalid input channel count for preprocessing: %d", m_input_c);
        input_buffer_host.clear();
        return;
    }
    input_buffer_host.resize(static_cast<size_t>(m_input_c) * m_cam_height_ * m_cam_width_);
    std::vector<cv::Mat> channels(m_input_c);
    cv::split(resized, channels);

    size_t channel_size_bytes = static_cast<size_t>(m_cam_height_) * m_cam_width_ * sizeof(float);
    for (int c = 0; c < m_input_c; ++c) {
        if (channels[c].isContinuous()) {
             memcpy(input_buffer_host.data() + c * (m_cam_height_ * m_cam_width_),
                    channels[c].data,
                    channel_size_bytes);
        } else {
            for (int i = 0; i < m_cam_height_; ++i) {
                memcpy(input_buffer_host.data() + c * (m_cam_height_ * m_cam_width_) + i * m_cam_width_,
                       channels[c].ptr<float>(i),
                       m_cam_width_ * sizeof(float));
            }
        }
    }
}

void DroneDetector::prepareSequenceInput(const std::vector<cv::Mat>& sequence_images, 
    std::vector<float>& sequence_input_host) {
    if (sequence_images.empty()) {
        ROS_ERROR("Empty image sequence provided!");
        return;
    }

    const int num_frames = sequence_images.size();
    
    int frame_size;
    if (m_enable_rotation == true) {
        frame_size = m_cam_height_ * m_cam_width_;
    } else {
        frame_size = m_cam_width_ * m_cam_height_;
    }
    const int total_elements = num_frames * frame_size;

    sequence_input_host.resize(total_elements);

    // Flatten the temporal tensor as [frame][row][column].
    for (int i = 0; i < num_frames; ++i) {
        const cv::Mat& frame = sequence_images[i];

        if (frame.empty()) {
            ROS_ERROR("Invalid frame %d in sequence!", i);
            continue;
        }

        cv::Mat resized_frame;
        cv::Size target_size;
        if (m_enable_rotation == true) {
            target_size = cv::Size(m_cam_height_, m_cam_width_);
        } else {
            target_size = cv::Size(m_cam_width_, m_cam_height_);
        }
        cv::resize(frame, resized_frame, target_size);

        int target_width, target_height;
        if (m_enable_rotation == true) {
            target_width = m_cam_height_;
            target_height = m_cam_width_;
        } else {
            target_width = m_cam_width_;
            target_height = m_cam_height_;
        }
        
        for (int y = 0; y < target_height; ++y) {
            for (int x = 0; x < target_width; ++x) {
                int idx = i * frame_size + y * target_width + x;
                sequence_input_host[idx] = static_cast<float>(resized_frame.at<uchar>(y, x));
            }
        }
    }
}
std::vector<cv::Mat> DroneDetector::createSequenceImages(const std::vector<int>& count_image, int camera_width, int camera_height, int num_channels) {
    
    if (count_image.empty()) {
        ROS_WARN("createSequenceImages received an empty input vector.");
        return {};
    }

    // The input vector is channel-major, with each channel stored row-major.
    std::vector<uchar> image_data(count_image.begin(), count_image.end());

    std::vector<cv::Mat> sequence_images;
    sequence_images.reserve(num_channels);

    const size_t channel_byte_size = static_cast<size_t>(camera_height) * camera_width;

    for (int ch = 0; ch < num_channels; ++ch) {
        uchar* channel_data_ptr = image_data.data() + ch * channel_byte_size;

        cv::Mat channel_mat_header(camera_height, camera_width, CV_8UC1, channel_data_ptr);

        cv::Mat channel_mat = channel_mat_header.clone();

        // The vertical model consumes each channel after a 90-degree CCW rotation.
        if (m_enable_rotation) {
            cv::Mat rotated_channel;
            cv::rotate(channel_mat, rotated_channel, cv::ROTATE_90_COUNTERCLOCKWISE);
            sequence_images.push_back(rotated_channel);
        } else {
            sequence_images.push_back(channel_mat);
        }
    }
    
    return sequence_images;
}

std::vector<Detection> DroneDetector::detect(const std::vector<cv::Mat>& sequence_images, std::string time_path, ros::Time t0) {

    if (!m_initialized || sequence_images.empty() || sequence_images[0].empty()) {
        ROS_ERROR("m_initialized: %d, sequence_images.empty(): %d, sequence_images[0].empty(): %d", m_initialized, sequence_images.empty(), sequence_images[0].empty());
        return {};
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const cv::Mat& image = sequence_images[0];

    std::vector<float> preprocess_input_host;
    preprocessImage(image, preprocess_input_host);
    if (preprocess_input_host.empty()) {
        ROS_ERROR("Preprocessing failed.");
        return {};
    }
    std::vector<float> sequence_input_host;
    prepareSequenceInput(sequence_images, sequence_input_host);

    // H2D copies, inference, and D2H copy share one stream and execute in order.
    if (cudaMemcpyAsync(m_gpu_buffers[0], preprocess_input_host.data(),
                        m_input_buffer_size_elements * sizeof(float),
                        cudaMemcpyHostToDevice, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy input data to GPU.");
        return {};
    }
    if (cudaMemcpyAsync(m_gpu_buffers[1], sequence_input_host.data(),
                        sequence_input_host.size() * sizeof(float),
                        cudaMemcpyHostToDevice, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy sequence data to GPU!");
        return {};
    }

    const char* input_tensor_name = m_engine->getIOTensorName(0);
    const char* sequence_tensor_name = m_engine->getIOTensorName(1);
    const char* output_tensor_name = m_engine->getIOTensorName(2);
    
    if (!m_context->setTensorAddress(input_tensor_name, m_gpu_buffers[0])) {
        ROS_ERROR("Failed to set tensor address for input: %s", input_tensor_name);
        return {};
    }
    if (!m_context->setTensorAddress(sequence_tensor_name, m_gpu_buffers[1])) {
        ROS_ERROR("Failed to set tensor address for sequence: %s", sequence_tensor_name);
        return {};
    }
    if (!m_context->setTensorAddress(output_tensor_name, m_gpu_buffers[2])) {
        ROS_ERROR("Failed to set tensor address for output: %s", output_tensor_name);
        return {};
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    cudaEvent_t evt_start, evt_infer_done, evt_h2d_done;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_infer_done);
    cudaEventCreate(&evt_h2d_done);

    cudaEventRecord(evt_start, m_cuda_stream);

    auto infer_start = std::chrono::high_resolution_clock::now();
    if (!m_context->enqueueV3(m_cuda_stream)) {
        ROS_ERROR("Failed to execute inference via enqueueV3.");
        return {};
    }

    cudaEventRecord(evt_infer_done, m_cuda_stream);

    // Pinned host memory allows the output copy to remain asynchronous until sync.
    float* output_cpu_pinned = nullptr;
    size_t output_bytes = m_output_buffer_size_elements * sizeof(float);
    cudaError_t err = cudaMallocHost((void**)&output_cpu_pinned, output_bytes);
    if (err != cudaSuccess) {
        ROS_ERROR("cudaMallocHost failed: %s", cudaGetErrorString(err));
        return {};
    }

    auto post_start = std::chrono::high_resolution_clock::now();
    if (cudaMemcpyAsync(output_cpu_pinned, m_gpu_buffers[2],
                        output_bytes, cudaMemcpyDeviceToHost, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy output data from GPU.");
        cudaFreeHost(output_cpu_pinned);
        return {};
    }
    cudaEventRecord(evt_h2d_done, m_cuda_stream);

    cudaStreamSynchronize(m_cuda_stream);
    auto post_end = std::chrono::high_resolution_clock::now();
    double post_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count() / 1000.0;

    float t_infer_ms = 0.f, t_h2d_ms = 0.f;
    cudaEventElapsedTime(&t_infer_ms, evt_start, evt_infer_done);
    cudaEventElapsedTime(&t_h2d_ms, evt_infer_done, evt_h2d_done);

    ROS_INFO("GPU inference time (event): %.3f ms, D2H copy (event): %.3f ms, host-measured post_elapsed: %.3f ms",
            t_infer_ms, t_h2d_ms, post_elapsed_ms);

    std::vector<float> output_cpu_buffer(m_output_buffer_size_elements);
    memcpy(output_cpu_buffer.data(), output_cpu_pinned, output_bytes);

    cudaFreeHost(output_cpu_pinned);

    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_infer_done);
    cudaEventDestroy(evt_h2d_done);

    std::ofstream outFile(time_path, std::ios::app);
    outFile << "[ " << t0 << " ] transform 2 " << elapsed_ms << std::endl;
    outFile << "[ " << t0 << " ] Inference: " << t_infer_ms << std::endl;
    outFile << "[ " << t0 << " ] Post-process: " << post_elapsed_ms << std::endl;
    outFile.close();

    return postprocessResults(output_cpu_buffer.data(), image.cols, image.rows);

    
    
    
    
        
        

}

    
    

std::vector<Detection> DroneDetector::postprocessResults(const float* output, int original_image_width, int original_image_height) {
    std::vector<Detection> detections;

    if (m_output_dims.nbDims < 2) {
        ROS_ERROR("Output dimensions are not as expected. nbDims: %d", m_output_dims.nbDims);
        return {};
    }
    

    // Proposal layout: [cx, cy, width, height, objectness, class score, class ID].
    int num_elements_per_proposal = 7;
    if (m_output_buffer_size_elements == 0 || (m_output_buffer_size_elements % num_elements_per_proposal != 0) ) {
        ROS_ERROR("Output buffer size (%ld) is not a multiple of elements per proposal (%d).", 
                  static_cast<long>(m_output_buffer_size_elements), num_elements_per_proposal);
        return {};
    }
    int num_proposals = m_output_buffer_size_elements / num_elements_per_proposal;

    const int MIN_BOX_DIM = 5;
    const int MAX_CLASS_ID = 10;
    const float MIN_AREA_RATIO = 0.001f;
    const float MIN_ASPECT_RATIO = 0.2f;
    const float MAX_ASPECT_RATIO = 5.0f;

    // Map boxes from the possibly rotated model input back to the published image.
    float scale_x, scale_y;
    if (m_enable_rotation == true) {
        scale_x = static_cast<float>(original_image_width) / m_cam_height_;
        scale_y = static_cast<float>(original_image_height) / m_cam_width_;
    } else {
        scale_x = static_cast<float>(original_image_width) / m_cam_width_;
        scale_y = static_cast<float>(original_image_height) / m_cam_height_;
    }

    for (int i = 0; i < num_proposals; ++i) {
        float x_center = output[i * num_elements_per_proposal + 0];
        float y_center = output[i * num_elements_per_proposal + 1];
        float width    = output[i * num_elements_per_proposal + 2];
        float height   = output[i * num_elements_per_proposal + 3];

        float x0 = (x_center - width / 2.0f) * scale_x;
        float y0 = (y_center - height / 2.0f) * scale_y;
        float x1 = (x_center + width / 2.0f) * scale_x;
        float y1 = (y_center + height / 2.0f) * scale_y;

        float obj_conf = output[i * num_elements_per_proposal + 4];
        float cls_conf = output[i * num_elements_per_proposal + 5];
        int   cid = static_cast<int>(output[i * num_elements_per_proposal + 6] + 0.5f);
        float score = obj_conf * cls_conf;

        int box_w = static_cast<int>(x1 - x0);
        int box_h = static_cast<int>(y1 - y0);
        float area = box_w * box_h;
        float image_area = original_image_width * original_image_height;
        float aspect_ratio = (box_h > 0) ? static_cast<float>(box_w) / box_h : 0;
        if (score > m_conf_thresh &&
            box_w >= MIN_BOX_DIM &&
            box_h >= MIN_BOX_DIM &&
            x0 >= 0 && y0 >= 0 &&
            x1 <= original_image_width && y1 <= original_image_height &&
            area > (MIN_AREA_RATIO * image_area) &&
            aspect_ratio > MIN_ASPECT_RATIO &&
            aspect_ratio < MAX_ASPECT_RATIO) 
        {
            detections.push_back({cv::Rect(static_cast<int>(x0), static_cast<int>(y0), box_w, box_h), score, cid});
        }
    }
    auto finalDets = doNMS(detections, m_nms_thresh);
    int num_final_dets = static_cast<int>(finalDets.size());

    return finalDets;
}

float DroneDetector::iou(const cv::Rect& a, const cv::Rect& b) {
    float inter_area = static_cast<float>((a & b).area());
    float union_area = static_cast<float>(a.area() + b.area() - inter_area);
    return (union_area > 1e-6) ? (inter_area / union_area) : 0.0f;
}

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
            // Suppress overlapping boxes only within the same class.
            if (sorted_detections[i].class_id == sorted_detections[j].class_id) {
                 if (iou(sorted_detections[i].box, sorted_detections[j].box) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return nms_results;
}

void DroneDetector::show_count_image(cv::Mat& frame, const std::vector<Detection>& detections, ros::Time& stamp) {
    cv::Mat image_to_show;
        for (int y = 0; y < frame.rows; ++y) {
        for (int x = 0; x < frame.cols; ++x) {
            frame.at<uchar>(y, x) = 255 - frame.at<uchar>(y, x);
        }
    }
    if (frame.channels() == 1) {
        cv::cvtColor(frame, image_to_show, cv::COLOR_GRAY2BGR);
    } else if (frame.channels() == 3) {
        image_to_show = frame.clone();
    } else {
        ROS_ERROR("Unsupported channel count in show_count_image: %d", frame.channels());
        return;
    }

    if (!detections.empty()) {
        for (const auto& d: detections) {

            cv::rectangle(image_to_show, d.box, cv::Scalar(0, 0, 255), 4);

            char buf[64];
            float display_score = std::isfinite(d.score) ? d.score : 0.0f;
            snprintf(buf, sizeof(buf), "drone: %.2f", display_score);

            cv::Point text_pos = d.box.tl() - cv::Point(0, 10);

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(buf, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1.0, &baseline);
            cv::rectangle(image_to_show, text_pos + cv::Point(0, baseline),
                          text_pos + cv::Point(text_size.width, -text_size.height),
                          cv::Scalar(0, 0, 255), cv::FILLED);

            cv::putText(image_to_show, buf, text_pos, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
        }
    }

    std_msgs::Header header;
    header.stamp = stamp;
    try {
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "bgr8", image_to_show).toImageMsg();
        m_count_image_pub_.publish(msg);
    } catch (const cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

void DroneDetector::AngleCalculations(float u,float v,ros::Time& t0,ros::Time& stamp)
{

    int x,y;

    // Undo the image rotation before indexing the row-major bearing-vector table.
    if(m_enable_rotation){
        y=(int)u;
        x=m_cam_width_-(int)v;
    }
    else{
        y=(int)v;
        x=(int)u;
    }
    
    const cv::Point3d bvec = m_precomputed_bearing_vectors_ptr_.at(y * m_cam_width_ + x);

    Eigen::Vector3d e_ray_cam;
    if (m_enable_rotation) {
        e_ray_cam = Eigen::Vector3d(bvec.y, -bvec.x, bvec.z);
    } else {
        e_ray_cam = Eigen::Vector3d(bvec.x, bvec.y, bvec.z);
    }
    

    // Convert nanoseconds to seconds and compensate a 0.95-revolution-per-second spin.
    float t_diff = double(t0.toNSec() - stamp.toNSec()) / 1000000000.0 * 0.95;

    double cos_term = cos(2 * pi * t_diff);
    double sin_term = sin(2 * pi * t_diff);

    Eigen::Matrix3d R_eigen_;

    R_eigen_ << cos_term, m_R(2, 1) * sin_term, m_R(2, 2) * sin_term,
                0, m_R(1, 1), m_R(1, 2),
                -sin_term, m_R(2, 1) * cos_term, m_R(2, 2) * cos_term;

    Eigen::Vector3d e_ray_w = R_eigen_ * e_ray_cam;

    const double theta = std::atan2(e_ray_w[0], e_ray_w[2]);
    const double phi= std::asin(e_ray_w[1] / e_ray_w.norm());
    std::ofstream outFile(output_path, std::ios::app);
    outFile << "[infer] Timestamp: " << t0.toNSec();
    outFile << " Phi: " << phi << " Theta: " << theta << std::endl;
    outFile.close();
    double r=2;

    publishSphericalMarker(m_marker_pub_, r, theta+pi, pi/2-phi,"map");
    publishOriginAxes(m_marker_pub_, "map");

}

void DroneDetector::sphericalToCartesian(double r, double theta_rad, double phi_rad, 
    double& x, double& y, double& z) {
    // theta is azimuth about +z; phi is the polar angle measured from +z.
    x = r * sin(phi_rad) * cos(theta_rad);
    y = r * sin(phi_rad) * sin(theta_rad);
    z = r * cos(phi_rad);
}

void DroneDetector::publishSphericalMarker(ros::Publisher& pub, 
    double r, double theta_rad, double phi_rad, 
    const std::string& frame_id ) {
    double x_end, y_end, z_end;
    sphericalToCartesian(r, theta_rad, phi_rad, x_end, y_end, z_end);

    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = frame_id;
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = "polar_arrow_3d";
    arrow_marker.id = 0;
    arrow_marker.type = visualization_msgs::Marker::ARROW;
    arrow_marker.action = visualization_msgs::Marker::ADD;

    arrow_marker.scale.x = 0.05;
    arrow_marker.scale.y = 0.1;
    arrow_marker.scale.z = 0.0;
    arrow_marker.color.b = 1.0;
    arrow_marker.color.a = 1.0;

    geometry_msgs::Point start_point, end_point;
    start_point.x = 0; start_point.y = 0; start_point.z = 0;
    end_point.x = x_end; end_point.y = y_end; end_point.z = z_end;
    arrow_marker.points.push_back(start_point);
    arrow_marker.points.push_back(end_point);

    visualization_msgs::Marker text_marker;
    text_marker.header = arrow_marker.header;
    text_marker.ns = "polar_text_3d";
    text_marker.id = 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;

    const double rad_to_deg = 180.0 / pi;
    double theta_deg = theta_rad * rad_to_deg;
    double phi_deg = phi_rad * rad_to_deg;

    text_marker.text = "End: (" + std::to_string(x_end).substr(0, 4) + ", " + 
    std::to_string(y_end).substr(0, 4) + ", " + 
    std::to_string(z_end).substr(0, 4) + ")\n" +
    "Spherical: r=" + std::to_string(r).substr(0, 3) + 
    ", θ=" + std::to_string(theta_deg).substr(0, 5) + "°" +
    ", φ=" + std::to_string(phi_deg).substr(0, 4) + "°";

    text_marker.pose.position.x = 1.5;
    text_marker.pose.position.y = 1.5;
    text_marker.pose.position.z = 2;

    text_marker.pose.orientation.x = 0.0;
    text_marker.pose.orientation.y = 0.0;
    text_marker.pose.orientation.z = 0.0;
    text_marker.pose.orientation.w = 1.0;

    text_marker.scale.x = 0.0;
    text_marker.scale.y = 0.0;
    text_marker.scale.z = 0.2;

    text_marker.lifetime = ros::Duration();
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0;
    text_marker.color.a = 1.0;
    text_marker.lifetime = ros::Duration(0.5);

    pub.publish(arrow_marker);
    pub.publish(text_marker);
}

void DroneDetector::publishOriginAxes(ros::Publisher& pub, const std::string& frame_id ) {
    visualization_msgs::Marker x_axis, y_axis, z_axis;

    auto initAxisMarker = [&](visualization_msgs::Marker& marker, int id, 
                            const geometry_msgs::Vector3& color, 
                            const geometry_msgs::Point& end_point) {
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "origin_axes";
        marker.id = id;
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.03;
        marker.scale.y = 0.08;
        marker.scale.z = 0.0;
        marker.color.r = color.x;
        marker.color.g = color.y;
        marker.color.b = color.z;
        marker.color.a = 1.0;
        
        geometry_msgs::Point start_point;
        start_point.x = start_point.y = start_point.z = 0;
        marker.points.push_back(start_point);
        marker.points.push_back(end_point);
    };

    geometry_msgs::Vector3 red; red.x = 1.0; red.y = 0.0; red.z = 0.0;
    geometry_msgs::Point x_end; x_end.x = 1.0; x_end.y = 0.0; x_end.z = 0.0;
    initAxisMarker(x_axis, 0, red, x_end);

    geometry_msgs::Vector3 green; green.x = 0.0; green.y = 1.0; green.z = 0.0;
    geometry_msgs::Point y_end; y_end.x = 0.0; y_end.y = 1.0; y_end.z = 0.0;
    initAxisMarker(y_axis, 1, green, y_end);

    geometry_msgs::Vector3 blue; blue.x = 0.0; blue.y = 0.0; blue.z = 1.0;
    geometry_msgs::Point z_end; z_end.x = 0.0; z_end.y = 0.0; z_end.z = 1.0;
    initAxisMarker(z_axis, 2, blue, z_end);

    pub.publish(x_axis);
    pub.publish(y_axis);
    pub.publish(z_axis);
}

bool DroneDetector::initializePostProcessBuffers() {
    cudaError_t err;
    
    err = cudaMalloc(&m_gpu_best_box, 4 * sizeof(float));
    if (err != cudaSuccess) {
        ROS_ERROR("Failed to allocate m_gpu_best_box: %s", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&m_gpu_best_score, sizeof(float));
    if (err != cudaSuccess) {
        ROS_ERROR("Failed to allocate m_gpu_best_score: %s", cudaGetErrorString(err));
        return false;
    }
    
    err = cudaMalloc(&m_gpu_found_detection, sizeof(int));
    if (err != cudaSuccess) {
        ROS_ERROR("Failed to allocate m_gpu_found_detection: %s", cudaGetErrorString(err));
        return false;
    }
    
    cudaMemset(m_gpu_best_box, 0, 4 * sizeof(float));
    cudaMemset(m_gpu_best_score, 0, sizeof(float));
    cudaMemset(m_gpu_found_detection, 0, sizeof(int));
    
    return true;
}

void DroneDetector::cleanupPostProcessBuffers() {
    if (m_gpu_best_box) {
        cudaFree(m_gpu_best_box);
        m_gpu_best_box = nullptr;
    }
    
    if (m_gpu_best_score) {
        cudaFree(m_gpu_best_score);
        m_gpu_best_score = nullptr;
    }
    
    if (m_gpu_found_detection) {
        cudaFree(m_gpu_found_detection);
        m_gpu_found_detection = nullptr;
    }
}

}
