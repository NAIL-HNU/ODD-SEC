#include <iostream>
#include "backend/drone_detector.h"
#include <fstream>
#include <algorithm> // For std::sort
#include <cmath>     // For expf
#include <vector>    // For std::vector
#include <string>    // For std::string
#include <nvtx3/nvToolsExt.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h> // 确保包含
#include <std_msgs/Header.h>     // 确保包含
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

// 构造函数中 nh->advertise 的修改
DroneDetector::DroneDetector(ros::NodeHandle* nh_ptr) //更改参数名以避免与成员变量nh_混淆
    : nh_("~"), // 成员nh_使用私有节点句柄初始化
      m_marker_pub_ (nh_ptr->advertise<visualization_msgs::Marker>("polar_line_3d", 10)), // 使用传入的nh_ptr
      m_count_image_pub_(nh_ptr->advertise<sensor_msgs::Image>("count_image", 10)) // 使用传入的nh_ptr
       {

}

DroneDetector::~DroneDetector() {
    m_count_image_pub_.shutdown();
    m_marker_pub_.shutdown();
    cleanupPostProcessBuffers(); // 清理新的缓冲区
    // 如果 m_gpu_buffers 是在这里清理的，也要确保清理
    if (m_gpu_buffers[0]) cudaFree(m_gpu_buffers[0]);
    if (m_gpu_buffers[1]) cudaFree(m_gpu_buffers[1]);
    if (m_gpu_buffers[2]) cudaFree(m_gpu_buffers[2]);
}

void DroneDetector::initialize(int camera_width, int camera_height,
    const DetectorParams &opt,
    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr){
    m_params = opt;

    m_cam_width_ = camera_width;
    m_cam_height_ = camera_height;

    // 从 opt 中获取阈值并赋值给成员变量

    m_precomputed_bearing_vectors_ptr_ = *precomputed_bearing_vectors_ptr;

    std::vector<double> R_matrix = m_params.image_opt.R_matrix;
    std::string engine_file_path = m_params.det_opt.engine_file_path;
    // m_enable_rotation = m_params.det_opt.enable_rotation;
    m_enable_rotation = true;

    bag_file_path=m_params.file_opt.bag_file_path;
    boost::filesystem::path path_obj(bag_file_path);
    std::string bag_filename = path_obj.stem().string();
    
    // 构建输出文件路径
    output_path = "/home/zhx/codes/2_Drone_Dection/Dataset_Toolbox/src/panorama/data/" + 
                             bag_filename + "_panorama_output.txt";


    if (R_matrix.size() == 9) {
        Eigen::Map<const Eigen::Matrix3d> R_raw(R_matrix.data());
        m_R = R_raw.transpose();  // ROS数组是行优先，需转置为列优先
    } else {
        ROS_ERROR("Invalid rotation matrix size. Using identity.");
        m_R = Eigen::Matrix3d::Identity();
    }

    // 根据相机尺寸选择模型类型
    // m_enable_rotation = true: 使用竖向模型 (480x640)
    // m_enable_rotation = false: 使用横向模型 (640x480)
    std::string base_engine_path = m_params.det_opt.engine_file_path;
    
    // 竖向模式
    if (m_enable_rotation == true) {
        m_engine_file_path = base_engine_path;
        size_t dot_pos = m_engine_file_path.find_last_of('.');
        if (dot_pos != std::string::npos) {
            // m_engine_file_path.insert(dot_pos, "_vertical");
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
        // 其他尺寸，使用默认模型
        m_enable_rotation = false; // 默认使用横向模式
        m_engine_file_path = base_engine_path;
        ROS_INFO("Default mode: Using default model: %s", m_engine_file_path.c_str());
    }

    // Initializing model
    if (!loadEngine(m_engine_file_path)) {
        ROS_ERROR("Engine loading failed, aborting initialization.");
        return;
    }
    m_context = m_engine->createExecutionContext();
    if (!m_context) {
        ROS_ERROR("Failed to create execution context.");
        return;
    }

    !prepareBuffers();

    if (!initializePostProcessBuffers()) {
        ROS_ERROR("Post-processing buffers initialization failed.");
        return;
    }

    m_initialized = true;
    ROS_INFO("DroneDetector initialized successfully!");
}


// Loading engine
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
    engine_file.close(); // 关闭文件

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

// Prepare GPU buffer for the following propressing
bool DroneDetector::prepareBuffers() {
    if (!m_engine || !m_context) { // 确保 context 也已创建
        ROS_ERROR("Engine or context is null in prepareBuffers.");
        return false;
    }
    if (m_engine->getNbIOTensors() < 2) {
        ROS_ERROR("Engine does not have at least 2 I/O tensors.");
        return false;
    }
    const char* input_tensor_name = m_engine->getIOTensorName(0);
    const char* sequence_tensor_name = m_engine->getIOTensorName(1);
    const char* output_tensor_name = m_engine->getIOTensorName(2);

    if (!input_tensor_name || !output_tensor_name) {
        ROS_ERROR("Failed to get I/O tensor names.");
        return false;
    }
    
    m_input_dims = m_engine->getTensorShape(input_tensor_name);
    m_output_dims = m_engine->getTensorShape(output_tensor_name);

    // 打印所有I/O张量信息以供调试
    ROS_INFO("Available I/O Tensors:");
    for (int32_t i = 0; i < m_engine->getNbIOTensors(); ++i) {
        const char* name = m_engine->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = m_engine->getTensorIOMode(name);
        ROS_INFO("I/O Tensor %d: Name='%s', Mode=%s", i, name, (mode == nvinfer1::TensorIOMode::kINPUT ? "Input" : "Output"));
    }

    nvinfer1::Dims batch_input_dims; 
    batch_input_dims.nbDims = 4; // 例如：[N, C, H, W]
    batch_input_dims.d[0] = 1; // Batch size
    batch_input_dims.d[1] = m_input_c; // Channels
    
    // 根据模式调整输入尺寸：配置中的H和W都是按横向模式设置的
    if (m_enable_rotation == true) {
        // 竖向模式：交换H和W
        batch_input_dims.d[2] = m_cam_width_;  // Height = 原Width
        batch_input_dims.d[3] = m_cam_height_; // Width = 原Height
    } else {
        // 横向模式：保持原配置
        batch_input_dims.d[2] = m_cam_height_; // Height
        batch_input_dims.d[3] = m_cam_width_;  // Width
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

    m_input_buffer_size_elements = 1; // batch size
    nvinfer1::Dims actual_input_shape = m_engine->getTensorShape(input_tensor_name); // 获取设置后的形状
    for (int j = 0; j < actual_input_shape.nbDims; ++j) { // 通常从0开始乘所有维度
        m_input_buffer_size_elements *= actual_input_shape.d[j];
    }


    m_output_buffer_size_elements = 1;
    nvinfer1::Dims actual_output_shape = m_engine->getTensorShape(output_tensor_name);
    for (int j = 0; j < actual_output_shape.nbDims; ++j) {
        m_output_buffer_size_elements *= actual_output_shape.d[j];
    }

    // Malloc CUDA memory
    // 确保之前的 m_gpu_buffers 已被正确 cudaFree
    if (m_gpu_buffers[0]) cudaFree(m_gpu_buffers[0]);
    if (m_gpu_buffers[1]) cudaFree(m_gpu_buffers[1]);

    // Input Buffer
    cudaMalloc(&m_gpu_buffers[0], m_input_buffer_size_elements * sizeof(float));  // image
    cudaMalloc(&m_gpu_buffers[1], m_input_buffer_size_elements * 10 * sizeof(float));   // image sequence
    cudaMalloc(&m_gpu_buffers[2], m_output_buffer_size_elements * sizeof(float)); // output buffer

    // TensorRT 10.x: Binding context (setTensorAddress)
    // setTensorAddress 使用张量名称
    if (!m_context->setTensorAddress(input_tensor_name, m_gpu_buffers[0])) {
        ROS_ERROR("Failed to set tensor address for input: %s", input_tensor_name);
        return false;
    }
    if (!m_context->setTensorAddress(output_tensor_name, m_gpu_buffers[1])) {
        ROS_ERROR("Failed to set tensor address for output: %s", output_tensor_name);
        return false;
    }

    return true;
}

// Preprocessing images
void DroneDetector::preprocessImage(const cv::Mat& image, std::vector<float>& input_buffer_host) { // Renamed for clarity
    cv::Mat rgb, resized;

    // 确保 m_cam_width_ 和 m_cam_height_ > 0
    if (m_cam_width_ <= 0 || m_cam_height_ <= 0) {
        ROS_ERROR("Invalid dimensions for resizing: width=%d, height=%d", m_cam_width_, m_cam_height_);
        input_buffer_host.clear(); // Indicate error
        return;
    }

    if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2BGR);
    } else if (image.channels() == 3) {
        rgb = image.clone(); // Assuming it's BGR, TRT typically wants RGB
        // cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB); // Uncomment if your model expects RGB
    } else {
        ROS_ERROR("Unsupported image channel count: %d", image.channels());
        input_buffer_host.clear();
        return;
    }

    // 根据模式调整图像尺寸：配置中的H和W都是按横向模式设置的
    cv::Size target_size;
    if (m_enable_rotation == true) {
        // 竖向模式：交换H和W
        target_size = cv::Size(m_cam_height_, m_cam_width_);
    } else {
        // 横向模式：保持原配置
        target_size = cv::Size(m_cam_width_, m_cam_height_);
    }
    cv::resize(rgb, resized, target_size);
    resized.convertTo(resized, CV_32F); // Normalize to [0,1] if model expects it

    // HWC -> CHW
    // 确保 m_input_c (channels) > 0
    if (m_input_c <= 0) {
        ROS_ERROR("Invalid input channel count for preprocessing: %d", m_input_c);
        input_buffer_host.clear();
        return;
    }
    input_buffer_host.resize(static_cast<size_t>(m_input_c) * m_cam_height_ * m_cam_width_); // Use static_cast for size_t
    std::vector<cv::Mat> channels(m_input_c);
    cv::split(resized, channels); // resized should be 3-channel here

    size_t channel_size_bytes = static_cast<size_t>(m_cam_height_) * m_cam_width_ * sizeof(float);
    for (int c = 0; c < m_input_c; ++c) {
        if (channels[c].isContinuous()) {
             memcpy(input_buffer_host.data() + c * (m_cam_height_ * m_cam_width_),
                    channels[c].data,
                    channel_size_bytes);
        } else {
            // Handle non-continuous Mats if necessary, though split usually makes them continuous
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
    // 1. 检查输入有效性
    if (sequence_images.empty()) {
        ROS_ERROR("Empty image sequence provided!");
        return;
    }

    // 2. 计算总元素数量（假设所有图像尺寸相同）
    const int num_frames = sequence_images.size();
    
    // 根据模式计算帧大小：配置中的H和W都是按横向模式设置的
    int frame_size;
    if (m_enable_rotation == true) {
        // 竖向模式：交换H和W
        frame_size = m_cam_height_ * m_cam_width_;
    } else {
        // 横向模式：保持原配置
        frame_size = m_cam_width_ * m_cam_height_;
    }
    const int total_elements = num_frames * frame_size;

    // 3. 预分配主机内存
    sequence_input_host.resize(total_elements);

    // 4. 处理每一帧
    for (int i = 0; i < num_frames; ++i) {
        const cv::Mat& frame = sequence_images[i];

        // 4.1 基础检查
        if (frame.empty()) {
            ROS_ERROR("Invalid frame %d in sequence!", i);
            continue;
        }

        // 4.2 根据模式调整尺寸：配置中的H和W都是按横向模式设置的
        cv::Mat resized_frame;
        cv::Size target_size;
        if (m_enable_rotation == true) {
            // 竖向模式：交换H和W
            target_size = cv::Size(m_cam_height_, m_cam_width_);
        } else {
            // 横向模式：保持原配置
            target_size = cv::Size(m_cam_width_, m_cam_height_);
        }
        cv::resize(frame, resized_frame, target_size);

        // 4.3 直接将uint8像素转换为float并拷贝
        int target_width, target_height;
        if (m_enable_rotation == true) {
            // 竖向模式：交换H和W
            target_width = m_cam_height_;
            target_height = m_cam_width_;
        } else {
            // 横向模式：保持原配置
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

    // 1. 准备好包含所有通道数据的uchar向量
    std::vector<uchar> image_data(count_image.begin(), count_image.end());

    // 2. 准备一个空的vector来存放最终的10张图像
    std::vector<cv::Mat> sequence_images;
    sequence_images.reserve(num_channels);

    // 3. 计算单个通道图像所占的字节数
    const size_t channel_byte_size = static_cast<size_t>(camera_height) * camera_width;

    // 4. 循环遍历10个通道，逐个提取和处理
    for (int ch = 0; ch < num_channels; ++ch) {
        // a. 计算当前通道在长向量中的数据起始指针
        uchar* channel_data_ptr = image_data.data() + ch * channel_byte_size;

        // b. 创建一个零拷贝的Mat头，直接指向该通道的数据
        cv::Mat channel_mat_header(camera_height, camera_width, CV_8UC1, channel_data_ptr);

        // c. clone()数据，创建拥有独立内存的、干净的单通道图像
        cv::Mat channel_mat = channel_mat_header.clone();

        // d. 如果需要，对这张干净的单通道图进行旋转
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

// Main detector
std::vector<Detection> DroneDetector::detect(const std::vector<cv::Mat>& sequence_images, std::string time_path, ros::Time t0) {

    if (!m_initialized || sequence_images.empty() || sequence_images[0].empty()) {
        ROS_ERROR("m_initialized: %d, sequence_images.empty(): %d, sequence_images[0].empty(): %d", m_initialized, sequence_images.empty(), sequence_images[0].empty());
        return {};
    }

    // pop the first channel as image
    // 记录开始时间
    auto start_time = std::chrono::high_resolution_clock::now();

    const cv::Mat& image = sequence_images[0];

    std::vector<float> preprocess_input_host; // Host buffer
    preprocessImage(image, preprocess_input_host);
    if (preprocess_input_host.empty()) {
        ROS_ERROR("Preprocessing failed.");
        return {};
    }
    std::vector<float> sequence_input_host;
    prepareSequenceInput(sequence_images, sequence_input_host);

    // Copy preprocessed data to GPU
    if (cudaMemcpyAsync(m_gpu_buffers[0], preprocess_input_host.data(), // Use host buffer
                        m_input_buffer_size_elements * sizeof(float),
                        cudaMemcpyHostToDevice, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy input data to GPU.");
        return {};
    }
    if (cudaMemcpyAsync(m_gpu_buffers[1], sequence_input_host.data(),
                        sequence_input_host.size() * sizeof(float), // 使用实际序列数据大小
                        cudaMemcpyHostToDevice, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy sequence data to GPU!");
        return {};
    }

    // 设置Tensor地址
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

    // 记录结束时间并计算耗时
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;

    // --- prepare events for timing ---
    cudaEvent_t evt_start, evt_infer_done, evt_h2d_done;
    cudaEventCreate(&evt_start);
    cudaEventCreate(&evt_infer_done);
    cudaEventCreate(&evt_h2d_done);

    // Record start before enqueue
    cudaEventRecord(evt_start, m_cuda_stream);

    // Inference: enqueue on m_cuda_stream
    auto infer_start = std::chrono::high_resolution_clock::now();
    if (!m_context->enqueueV3(m_cuda_stream)) {
        ROS_ERROR("Failed to execute inference via enqueueV3.");
        return {};
    }
    // Note: enqueueV3 submitted kernels to m_cuda_stream

    // Record an event right after enqueue to mark inference end (GPU-side)
    cudaEventRecord(evt_infer_done, m_cuda_stream);

    // --- Allocate pinned host memory for output ---
    float* output_cpu_pinned = nullptr;
    size_t output_bytes = m_output_buffer_size_elements * sizeof(float);
    cudaError_t err = cudaMallocHost((void**)&output_cpu_pinned, output_bytes); // pinned
    if (err != cudaSuccess) {
        ROS_ERROR("cudaMallocHost failed: %s", cudaGetErrorString(err));
        return {};
    }

    // Launch async device->host copy on same stream
    auto post_start = std::chrono::high_resolution_clock::now();
    if (cudaMemcpyAsync(output_cpu_pinned, m_gpu_buffers[2],
                        output_bytes, cudaMemcpyDeviceToHost, m_cuda_stream) != cudaSuccess) {
        ROS_ERROR("Failed to copy output data from GPU.");
        cudaFreeHost(output_cpu_pinned);
        return {};
    }
    // mark event after copy
    cudaEventRecord(evt_h2d_done, m_cuda_stream);

    // Now synchronize the stream (wait for copy to finish) — only here we block
    cudaStreamSynchronize(m_cuda_stream);
    auto post_end = std::chrono::high_resolution_clock::now();
    double post_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count() / 1000.0;

    // Measure GPU-side times precisely
    float t_infer_ms = 0.f, t_h2d_ms = 0.f;
    cudaEventElapsedTime(&t_infer_ms, evt_start, evt_infer_done);   // GPU time for inference kernels
    cudaEventElapsedTime(&t_h2d_ms, evt_infer_done, evt_h2d_done);  // GPU time for D2H copy

    ROS_INFO("GPU inference time (event): %.3f ms, D2H copy (event): %.3f ms, host-measured post_elapsed: %.3f ms",
            t_infer_ms, t_h2d_ms, post_elapsed_ms);

    // Now, if you need a std::vector, copy from pinned memory to vector (this is a CPU memcpy, fast)
    std::vector<float> output_cpu_buffer(m_output_buffer_size_elements);
    memcpy(output_cpu_buffer.data(), output_cpu_pinned, output_bytes);

    // free pinned
    cudaFreeHost(output_cpu_pinned);

    // destroy events
    cudaEventDestroy(evt_start);
    cudaEventDestroy(evt_infer_done);
    cudaEventDestroy(evt_h2d_done);


    std::ofstream outFile(time_path, std::ios::app);
    outFile << "[ " << t0 << " ] transform 2 " << elapsed_ms << std::endl;
    outFile << "[ " << t0 << " ] Inference: " << t_infer_ms << std::endl;
    outFile << "[ " << t0 << " ] Post-process: " << post_elapsed_ms << std::endl;
    outFile.close();

    return postprocessResults(output_cpu_buffer.data(), image.cols, image.rows);

    // // 后处理 - 修改为单目标检测
    // float h_score;
    // std::vector<float> h_box(4);
    // bool h_found = false;
    
    // postprocessResultsGPU(image.cols, image.rows, h_score, h_box, h_found);
    
    // // 等待GPU完成
    // cudaStreamSynchronize(m_cuda_stream);
    
    // auto post_end = std::chrono::high_resolution_clock::now();
    // double post_elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count() / 1000.0;
    
    // std::vector<Detection> finalDetections;
    // if (h_found) {
    //     float scale_x = static_cast<float>(image.cols) / (m_enable_rotation ? m_cam_height_ : m_cam_width_);
    //     float scale_y = static_cast<float>(image.rows) / (m_enable_rotation ? m_cam_width_ : m_cam_height_);
        
    //     float x0 = h_box[0] * scale_x;
    //     float y0 = h_box[1] * scale_y;
    //     float x1 = h_box[2] * scale_x;
    //     float y1 = h_box[3] * scale_y;
        
    //     finalDetections.push_back({
    //         cv::Rect(static_cast<int>(x0), static_cast<int>(y0), 
    //                  static_cast<int>(x1 - x0), static_cast<int>(y1 - y0)),
    //         h_score,
    //         0  // 类别ID
    //     });
    // }

    // // 记录时间

    // return finalDetections;
}




// void DroneDetector::postprocessResultsGPU(
//     int original_image_width, int original_image_height,
//     float& h_score, std::vector<float>& h_box, bool& h_found
// ) {
//     // 清零找到的检测标志
//     cudaMemsetAsync(m_gpu_found_detection, 0, sizeof(int), m_cuda_stream);
    
//     // 启动单目标检测核函数
//     launch_find_single_target(
//         static_cast<float*>(m_gpu_buffers[2]),
//         m_conf_thresh,
//         m_gpu_best_box,
//         m_gpu_best_score,
//         m_gpu_found_detection,
//         m_cuda_stream
//     );
    
//     // 异步拷贝结果
//     cudaMemcpyAsync(&h_found, m_gpu_found_detection, sizeof(int), 
//                    cudaMemcpyDeviceToHost, m_cuda_stream);
//     cudaMemcpyAsync(&h_score, m_gpu_best_score, sizeof(float), 
//                    cudaMemcpyDeviceToHost, m_cuda_stream);
//     cudaMemcpyAsync(h_box.data(), m_gpu_best_box, 4 * sizeof(float), 
//                    cudaMemcpyDeviceToHost, m_cuda_stream);
// }




std::vector<Detection> DroneDetector::postprocessResults(const float* output, int original_image_width, int original_image_height) {
    std::vector<Detection> detections;

    if (m_output_dims.nbDims < 2) { // Basic check
        ROS_ERROR("Output dimensions are not as expected. nbDims: %d", m_output_dims.nbDims);
        return {};
    }
    

    int num_elements_per_proposal = 7; // As per your existing code [xc,yc,w,h,obj,cls,cid]
    if (m_output_buffer_size_elements == 0 || (m_output_buffer_size_elements % num_elements_per_proposal != 0) ) {
        ROS_ERROR("Output buffer size (%ld) is not a multiple of elements per proposal (%d).", 
                  static_cast<long>(m_output_buffer_size_elements), num_elements_per_proposal);
        return {};
    }
    int num_proposals = m_output_buffer_size_elements / num_elements_per_proposal;


    const int MIN_BOX_DIM = 5;        // 最小宽度/高度
    const int MAX_CLASS_ID = 10;      // 最大类别ID
    const float MIN_AREA_RATIO = 0.001f; // 最小面积比例
    const float MIN_ASPECT_RATIO = 0.2f; // 最小宽高比
    const float MAX_ASPECT_RATIO = 5.0f; // 最大宽高比

    // 根据模式计算缩放坐标：配置中的H和W都是按横向模式设置的
    float scale_x, scale_y;
    if (m_enable_rotation == true) {
        // 竖向模式：交换H和W
        scale_x = static_cast<float>(original_image_width) / m_cam_height_;
        scale_y = static_cast<float>(original_image_height) / m_cam_width_;
    } else {
        // 横向模式：保持原配置
        scale_x = static_cast<float>(original_image_width) / m_cam_width_;
        scale_y = static_cast<float>(original_image_height) / m_cam_height_;
    }

    for (int i = 0; i < num_proposals; ++i) {
        float x_center = output[i * num_elements_per_proposal + 0];
        float y_center = output[i * num_elements_per_proposal + 1];
        float width    = output[i * num_elements_per_proposal + 2];
        float height   = output[i * num_elements_per_proposal + 3];

        // 直接缩放坐标到原始图像尺寸
        float x0 = (x_center - width / 2.0f) * scale_x;
        float y0 = (y_center - height / 2.0f) * scale_y;
        float x1 = (x_center + width / 2.0f) * scale_x;
        float y1 = (y_center + height / 2.0f) * scale_y;

        float obj_conf = output[i * num_elements_per_proposal + 4];
        float cls_conf = output[i * num_elements_per_proposal + 5]; // Assuming single class or max class score
        int   cid = static_cast<int>(output[i * num_elements_per_proposal + 6] + 0.5f); // Add 0.5 for rounding before cast
        float score = obj_conf * cls_conf; // Or just obj_conf if cls_conf is already class-specific score

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
    // NMS is applied after collecting all potential detections
    auto finalDets = doNMS(detections, m_nms_thresh);
    // 计算finalDets的尺寸
    int num_final_dets = static_cast<int>(finalDets.size());

    return finalDets;
}



// Compute IoU
float DroneDetector::iou(const cv::Rect& a, const cv::Rect& b) {
    float inter_area = static_cast<float>((a & b).area());
    float union_area = static_cast<float>(a.area() + b.area() - inter_area);
    return (union_area > 1e-6) ? (inter_area / union_area) : 0.0f; // Avoid division by zero or very small numbers
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
            if (sorted_detections[i].class_id == sorted_detections[j].class_id) { // Ensure class_id is relevant
                 if (iou(sorted_detections[i].box, sorted_detections[j].box) > iou_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }
    return nms_results;
}

void DroneDetector::show_count_image(cv::Mat& frame, const std::vector<Detection>& detections, ros::Time& stamp) {
    // Create image
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

            // 边框变粗
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





void DroneDetector::AngleCalculations(float u,float v,ros::Time& t0,ros::Time& stamp) // Changed x,y to u,v for clarity (pixel coords)
{
    // Ensure m_cam_width_ is not zero to prevent division by zero or incorrect indexing

    // u=420;
    // v=340;

    int x,y;

    if(m_enable_rotation){
        y=(int)u;
        x=m_cam_width_-(int)v;
    }
    else{
        y=(int)v;
        x=(int)u;
    }
    
    // ROS_INFO("m_cam_width_: %d", m_cam_height_);
    const cv::Point3d bvec = m_precomputed_bearing_vectors_ptr_.at(y * m_cam_width_ + x);



    Eigen::Vector3d e_ray_cam;
    if (m_enable_rotation) {
        // 逆时针旋转90度，在XY平面中
        e_ray_cam = Eigen::Vector3d(bvec.y, -bvec.x, bvec.z);
    } else {
        e_ray_cam = Eigen::Vector3d(bvec.x, bvec.y, bvec.z);
    }
    

    float t_diff = double(t0.toNSec() - stamp.toNSec()) / 1000000000.0 * 0.95;

    // double result_x=double(bvec.x)+35/180*pi;
    // double result_y=double(bvec.y)-2 * pi * t_diff;

    double cos_term = cos(2 * pi * t_diff);
    double sin_term = sin(2 * pi * t_diff);

    Eigen::Matrix3d R_eigen_;

    R_eigen_ << cos_term, m_R(2, 1) * sin_term, m_R(2, 2) * sin_term,
                0, m_R(1, 1), m_R(1, 2),
                -sin_term, m_R(2, 1) * cos_term, m_R(2, 2) * cos_term;

    Eigen::Vector3d e_ray_w = R_eigen_ * e_ray_cam;

    const double theta = std::atan2(e_ray_w[0], e_ray_w[2]);
    const double phi= std::asin(e_ray_w[1] / e_ray_w.norm());
    // std::cout << "u、v " << u << " " << v << std::endl;
    // std::cout << "x、y、t " << x << " " << y << " " << t_diff << std::endl;
    // std::cout << "t0、tamp " << t0.toNSec() << " " << stamp.toNSec() << std::endl;
    // std::cout <<"bvec"<< bvec.x  << " " << bvec.y << " " << bvec.z<<std::endl;
    // // std::cout <<"result_x result_y"<< result_x  << " " << result_y<<std::endl;
    // std::cout << "e_ray_cam" << e_ray_cam[0] << " " << e_ray_cam[1] << " " << e_ray_cam[2] << std::endl;
    // std::cout<<"e_ray_w[0] :"<< e_ray_w[0] <<std::endl;
    // std::cout<<"e_ray_w[1] :"<< e_ray_w[1] <<std::endl;
    // std::cout<<"e_ray_w[2] :"<< e_ray_w[2] <<std::endl;

    // std::cout<<"phi :"<< phi <<std::endl;
    // std::cout<<"theta :"<< theta <<std::endl;

    // FIXME: for track alignment
    // std::ofstream outFile("/home/zhx/codes/2_Drone_Dection/Dataset_Toolbox/src/panorama/data/panorama_output.txt", std::ios::app);
    std::ofstream outFile(output_path, std::ios::app);
    outFile << "[infer] Timestamp: " << t0.toNSec();
    outFile << " Phi: " << phi << " Theta: " << theta << std::endl;
    outFile.close();
    // std::cout << "[infer] Timestamp: " << t0.toNSec() << " Phi: " << phi << " Theta: " << theta << std::endl;

    double r=2;

    publishSphericalMarker(m_marker_pub_, r, theta+pi, pi/2-phi,"map");
    publishOriginAxes(m_marker_pub_, "map");

}


// 球坐标转笛卡尔坐标
// Assumes theta_rad is elevation from XY plane, phi_rad is azimuth in XY plane
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

    // 1. 创建箭头Marker
    visualization_msgs::Marker arrow_marker;
    arrow_marker.header.frame_id = frame_id;
    arrow_marker.header.stamp = ros::Time::now();
    arrow_marker.ns = "polar_arrow_3d";
    arrow_marker.id = 0;
    arrow_marker.type = visualization_msgs::Marker::ARROW;  // 修改为ARROW类型
    arrow_marker.action = visualization_msgs::Marker::ADD;

    // 设置箭头属性
    arrow_marker.scale.x = 0.05;  // 箭杆直径
    arrow_marker.scale.y = 0.1;   // 箭头直径
    arrow_marker.scale.z = 0.0;   // 未使用
    arrow_marker.color.b = 1.0;   // 蓝色箭头
    arrow_marker.color.a = 1.0;

    // 设置起点（原点）和终点
    geometry_msgs::Point start_point, end_point;
    start_point.x = 0; start_point.y = 0; start_point.z = 0; // Origin
    end_point.x = x_end; end_point.y = y_end; end_point.z = z_end;
    arrow_marker.points.push_back(start_point);
    arrow_marker.points.push_back(end_point);

    // 2. 创建文本Marker
    visualization_msgs::Marker text_marker;
    text_marker.header = arrow_marker.header;
    text_marker.ns = "polar_text_3d";
    text_marker.id = 1; // Different ID for text
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::Marker::ADD;

    const double rad_to_deg = 180.0 / pi;
    double theta_deg = theta_rad * rad_to_deg;
    double phi_deg = phi_rad * rad_to_deg;

    // std::cout<<"[result] theta_deg :"<< theta_deg <<std::endl;
    // std::cout<<"[result] phi_deg :"<< phi_deg <<std::endl;

    text_marker.text = "End: (" + std::to_string(x_end).substr(0, 4) + ", " + 
    std::to_string(y_end).substr(0, 4) + ", " + 
    std::to_string(z_end).substr(0, 4) + ")\n" +
    "Spherical: r=" + std::to_string(r).substr(0, 3) + 
    ", θ=" + std::to_string(theta_deg).substr(0, 5) + "°" +
    ", φ=" + std::to_string(phi_deg).substr(0, 4) + "°";

    // 关键修复：初始化文本的位姿（位置+方向）
    text_marker.pose.position.x = 1.5;  // X坐标（可调整）
    text_marker.pose.position.y = 1.5;  // Y坐标（可调整）
    text_marker.pose.position.z = 2;  // Z坐标（可调整）

    text_marker.pose.orientation.x = 0.0;  // 单位四元数
    text_marker.pose.orientation.y = 0.0;
    text_marker.pose.orientation.z = 0.0;
    text_marker.pose.orientation.w = 1.0; // Identity quaternion

    text_marker.scale.x = 0.0;
    text_marker.scale.y = 0.0;
    text_marker.scale.z = 0.2;

    text_marker.lifetime = ros::Duration();
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.b = 1.0; // White text
    text_marker.color.a = 1.0;
    text_marker.lifetime = ros::Duration(0.5);

    // 发布Markers
    pub.publish(arrow_marker);
    pub.publish(text_marker);
}

void DroneDetector::publishOriginAxes(ros::Publisher& pub, const std::string& frame_id ) {
    // 创建Marker消息（三个箭头分别代表X/Y/Z轴）
    visualization_msgs::Marker x_axis, y_axis, z_axis;

    // 公共属性设置
    auto initAxisMarker = [&](visualization_msgs::Marker& marker, int id, 
                            const geometry_msgs::Vector3& color, 
                            const geometry_msgs::Point& end_point) {
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "origin_axes";
        marker.id = id;  // X轴=0, Y轴=1, Z轴=2
        marker.type = visualization_msgs::Marker::ARROW;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.03;  // 箭杆直径
        marker.scale.y = 0.08;   // 箭头直径
        marker.scale.z = 0.0;   // 未使用
        marker.color.r = color.x;
        marker.color.g = color.y;
        marker.color.b = color.z;
        marker.color.a = 1.0;
        
        // 起点为原点，终点为轴方向
        geometry_msgs::Point start_point;
        start_point.x = start_point.y = start_point.z = 0;
        marker.points.push_back(start_point);
        marker.points.push_back(end_point);
    };

    // X轴（红色，长度为1.0）
    geometry_msgs::Vector3 red; red.x = 1.0; red.y = 0.0; red.z = 0.0;
    geometry_msgs::Point x_end; x_end.x = 1.0; x_end.y = 0.0; x_end.z = 0.0;
    initAxisMarker(x_axis, 0, red, x_end);

    // Y轴（绿色，长度为1.0）
    geometry_msgs::Vector3 green; green.x = 0.0; green.y = 1.0; green.z = 0.0;
    geometry_msgs::Point y_end; y_end.x = 0.0; y_end.y = 1.0; y_end.z = 0.0;
    initAxisMarker(y_axis, 1, green, y_end);

    // Z轴（蓝色，长度为1.0）
    geometry_msgs::Vector3 blue; blue.x = 0.0; blue.y = 0.0; blue.z = 1.0;
    geometry_msgs::Point z_end; z_end.x = 0.0; z_end.y = 0.0; z_end.z = 1.0;
    initAxisMarker(z_axis, 2, blue, z_end);

    // 发布三个轴
    pub.publish(x_axis);
    pub.publish(y_axis);
    pub.publish(z_axis);
}


bool DroneDetector::initializePostProcessBuffers() {
    cudaError_t err;
    
    // 分配单目标检测所需的内存
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
    
    // 初始化这些变量
    cudaMemset(m_gpu_best_box, 0, 4 * sizeof(float));
    cudaMemset(m_gpu_best_score, 0, sizeof(float));
    cudaMemset(m_gpu_found_detection, 0, sizeof(int));
    
    return true;
}

// 清理后处理缓冲区
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
