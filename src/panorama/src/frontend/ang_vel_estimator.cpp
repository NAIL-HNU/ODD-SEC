#include <ros/ros.h>
#include <omp.h>
#include <backend/drone_detector.h>
#include <frontend/motion_compensation.cuh>
#include <frontend/ang_vel_estimator.h>
#include <utils/parameters.h>


#include <camera_info_manager/camera_info_manager.h>
#include <geometry_msgs/TwistStamped.h>
#include <sensor_msgs/Image.h>
#include <opencv2/highgui.hpp>
#include <glog/logging.h>
#include <Eigen/Dense> 
#include <chrono>
#include <iostream>
#include <tuple>


#include <vector>
#include <algorithm> // for std::max_element
#include <climits>   // for INT_MIN

using namespace std;  


namespace panorama {

AngVelEstimator::AngVelEstimator(ros::NodeHandle* nh): nh_(nh), it_(*nh)
{
    // Set publishers
    // pano_pub_ = it_.advertise("count_pano", 1);
    pub_signal = nh_->advertise<std_msgs::Header>("timestamp_topic", 0);
    detection_pub = nh_->advertise<panorama::CountImage>("detection_data", 10);
    imu_pub = nh_->advertise<sensor_msgs::Imu>("imu_publish", 2);
    // image_pub = nh_->advertise<sensor_msgs::Image>("image_ts", 10);
    // image_pub = it_.advertise("count_image", 1);
    // img_pub_ = it_.advertise("local_iwe", 1);
    // ang_vel_pub_ = nh_->advertise<geometry_msgs::TwistStamped>("/dvs/angular_velocity", 1);

}

AngVelEstimator::~AngVelEstimator()
{
    // img_pub_.shutdown();
    // ang_vel_pub_.shutdown();
    // pano_pub_.shutdown();
    pub_signal.shutdown();
    detection_pub.shutdown();
    image_pub.shutdown();
    // image_pub.shutdown();
}


void AngVelEstimator::initialize(image_geometry::PinholeCameraModel* cam,
                                 const AngVelEstParams& val,
                                 const std::vector<cv::Point3d>& precomputed_bearing_vectors)
{
    // Load camera information
    cam_width_ = cam->fullResolution().width;
    cam_height_ = cam->fullResolution().height;
    camera_matrix_ = cam->fullIntrinsicMatrix();

    // Load params
    params = val;
    // Get the pre-computed bearing vector
    precomputed_bearing_vectors_ = precomputed_bearing_vectors;

    R_matrix = params.image_opt.R_matrix;

    panorama_width_=params.image_opt.panorama_width;
    panorama_height_=params.image_opt.panorama_height;
    x_patches_=params.AA_opt.x_patches;
    y_patches_=params.AA_opt.y_patches;

    gyro_bias_x_=params.imu_bias_opt.gyro_bias_x;
    gyro_bias_y_=params.imu_bias_opt.gyro_bias_y;
    gyro_bias_z_=params.imu_bias_opt.gyro_bias_z;

    count_pano = cv::Mat::zeros(panorama_height_, panorama_width_, CV_32F); 

    count = 0.0;
    START=false;

    cc=0;

    center_ = Eigen::Vector2d(
        (double)params.image_opt.panorama_width / 2.0, 
        (double)params.image_opt.panorama_height
    );

    if (R_matrix.size() == 9) {
        Eigen::Map<const Eigen::Matrix3d> R_raw(R_matrix.data());
        R = R_raw.transpose();  // ROS数组是行优先，需转置为列优先
    } else {
        ROS_ERROR("Invalid rotation matrix size");
        R = Eigen::Matrix3d::Identity();
    }

    pixel_size_ = params.camera_opt.pixel_size ;
    Focus_ = params.camera_opt.Focus;
    pixel_focus_ratio = pixel_size_ / Focus_;
    // Initialize the event maintainance (containers and lists)
    // std::unique_lock<std::mutex> ev_lock(pose_graph_optimizer_->mutex_events);
    // events_.clear();
    // num_event_total_ = 0;
    // event_subsets_info_.clear();
    // ev_lock.unlock();


}


bool AngVelEstimator::processEvent(const dvs_msgs::Event& ev)
{
    const cv::Point3d bvec = precomputed_bearing_vectors_.at(ev.y*cam_width_ + ev.x);
    Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);

    float t_diff = double(ev.ts.toNSec() - t0_p.toNSec()) / 1000000000.0;

    double cos_term = cos(2 * pi * t_diff);
    double sin_term = sin(2 * pi * t_diff);
    

    R_eigen_ << cos_term, R(2, 1) * sin_term, R(2, 2) * sin_term,
                0, R(1, 1), R(1, 2),
                -sin_term, R(2, 1) * cos_term, R(2, 2) * cos_term;

    Eigen::Vector3d e_ray_w = R_eigen_ * e_ray_cam;
    
    // cc+=1;

    // if(cc%100000==0){cout<<t_diff<<endl;}

    // Calculate the pixel position in the panorama image
    const double phi = std::atan2(e_ray_w[0], e_ray_w[2]);
    const double theta = std::asin(e_ray_w[1] / e_ray_w.norm());

    px_mosaic = center_ + Eigen::Vector2d(-phi * params.image_opt.panorama_width / (2.0 * pi), -theta * params.image_opt.panorama_height / (pi*57.99/180));

    const int xx = px_mosaic[0],
              yy = px_mosaic[1];
    const float dx = px_mosaic[0] - xx, 
                dy = px_mosaic[1] - yy;

    if (1 <= xx && xx < panorama_width_ - 1 && 1 <= yy && yy < panorama_height_ - 1) {
        if (t_diff - count < 1.) {
            count_pano.at<float>(yy, xx) += (1.f - dx) * (1.f - dy);
            count_pano.at<float>(yy, xx + 1) += dx * (1.f - dy);
            count_pano.at<float>(yy + 1, xx) += (1.f - dx) * dy;
            count_pano.at<float>(yy + 1, xx + 1) += dx * dy;
            // return cv::Mat(); 
        } else {
            count += 1.0;
            // return true;
            double max_val = 10.0;
            cv::threshold(count_pano, count_pano, max_val, max_val, cv::THRESH_TRUNC);
            show_count_pano(count_pano, max_val, t0_p);
            count_pano = cv::Mat::zeros(panorama_height_, panorama_width_, CV_32F); // 重置为 0
        }
    }
    return false;
   
}

std::tuple<float, float, float> AngVelEstimator::IMU_Average(const std::vector<sensor_msgs::Imu>& imu_buffer_)
{
    double avg_x = 0.0, avg_y = 0.0, avg_z = 0.0;

    if (imu_buffer_.empty()) {
        return {0.0f, 0.0f, 0.0f};
    }

    for (const auto& imu : imu_buffer_) {
        avg_x += imu.angular_velocity.x;
        avg_y += imu.angular_velocity.y;
        avg_z += imu.angular_velocity.z;
    }

    double count = static_cast<double>(imu_buffer_.size());
    avg_x /= count;
    avg_y /= count;
    avg_z /= count;

    // 构建列向量 (double 类型)
    Eigen::Vector3d avg_vector(avg_x-gyro_bias_x_, avg_y-gyro_bias_y_, avg_z-gyro_bias_z_);

    // 乘以3x3矩阵
    Eigen::Vector3d transformed_vector = R* avg_vector;

    // 返回 float 类型的元组
    return std::make_tuple(
        static_cast<float>(transformed_vector.x()),
        static_cast<float>(transformed_vector.y()-2*pi),
        static_cast<float>(transformed_vector.z()));
    // return {avg_x, avg_y, avg_z};
}



void AngVelEstimator::signal(){
    std_msgs::Header msg;
    msg.stamp = ros::Time::now();  // 设置当前时间戳

    pub_signal.publish(msg);

}

std::vector<int> AngVelEstimator::motion_compensation_cuda(const std::vector<dvs_msgs::Event>& event_buffer,const std::vector<sensor_msgs::Imu>& imu_buffer_) {
    //TODO CUDA版本代码很糊
    auto ts0 = std::chrono::high_resolution_clock::now();
    count_image.clear();
    std::vector<int> flat_count_image(cam_height_ * cam_width_, 0); // 直接初始化大小和值

    if (imu_buffer_[imu_buffer_.size() - 1].header.stamp.toNSec() > event_buffer[0].ts.toNSec()) {
        int64_t t0_ns = event_buffer.front().ts.toNSec();
        int threshold = 20;

        // Launch CUDA accelerate Function
        run_motion_compensation_cuda(
            event_buffer.data(), event_buffer.size(),
            imu_buffer_.data(), imu_buffer_.size(),
            cam_width_, cam_height_,
            pixel_size_, Focus_,
            t0_ns, flat_count_image.data(), threshold);

        auto ts1 = std::chrono::high_resolution_clock::now();
    }

    return flat_count_image;
}




std::vector<std::vector<int>> AngVelEstimator::motion_compensation(const std::vector<dvs_msgs::Event>& event_buffer,const std::vector<sensor_msgs::Imu>& imu_buffer_, int threshold)
// bool AngVelEstimator::motion_compensation(const std::vector<dvs_msgs::Event>& event_buffer)
{
    auto ts0 = std::chrono::high_resolution_clock::now();
    count_image.clear();
    count_image.resize(cam_height_, std::vector<int>(cam_width_));
    // for (int i = 0; i < cam_height_; ++i) {
    //     std::fill(count_image[i].begin(), count_image[i].end(), 0);
    // }
    if(imu_buffer_[imu_buffer_.size() - 1].header.stamp.toNSec() > event_buffer[0].ts.toNSec()){

        float angular_velocity_x=0.0, angular_velocity_y=0.0,angular_velocity_z=0.0;
        
        int cnt=0;//imu counter
        for(int i=0;i<imu_buffer_.size();++i){
                if(imu_buffer_[i].header.stamp.toNSec() >= (event_buffer[0].ts.toNSec()-3000000)){
                        angular_velocity_x+=imu_buffer_[i].angular_velocity.x;
                        angular_velocity_y+=imu_buffer_[i].angular_velocity.y;
                        angular_velocity_z+=imu_buffer_[i].angular_velocity.z;
                        cnt++;
            }  
        }
        //Calculate the average imu angular rates
        average_angular_rate_x = angular_velocity_x/float(cnt);
        average_angular_rate_y = angular_velocity_y/float(cnt);
        average_angular_rate_z = angular_velocity_z/float(cnt);

        // average_angular_rate_x = 0.052686;
        // average_angular_rate_y = -5.160851;
        // average_angular_rate_z = 3.777805;

    //    auto ts1 = std::chrono::high_resolution_clock::now();
        //Motion  compensation
        
        float time_diff = 0.0;//time diff

        // auto ts2 = std::chrono::high_resolution_clock::now();

        sll t0=event_buffer[0].ts.toNSec();//the first event
        #pragma omp parallel for
        for(int i=0;i<event_buffer.size();i++){
            time_diff = double(event_buffer[i].ts.toNSec()-t0)/1000000000.0;

            //Calculate the rotation offset of the event point
            float x_angular=time_diff*average_angular_rate_x;
            float y_angular=time_diff*average_angular_rate_y;
            float z_angular=time_diff*average_angular_rate_z;

            int x=event_buffer[i].x - cam_width_/2; 
            int y=event_buffer[i].y - cam_height_/2;
            
            //Angle of initial position of event point
            float pre_x_angel = atan(y*pixel_focus_ratio);
            float pre_y_angel = atan(x*pixel_focus_ratio);
            // float pre_x_angel = fast_atan(y*pixel_focus_ratio);
            // float pre_y_angel = fast_atan(x*pixel_focus_ratio);

            //compensate
            int compen_x = (int)((x*cos(z_angular) - sin(z_angular)*y) - (x - (Focus_*tan(pre_y_angel + y_angular)/pixel_size_)) + cam_width_/2);
            int compen_y = (int)((x*sin(z_angular) + cos(z_angular)*y) - (y - (Focus_*tan(pre_x_angel - x_angular)/pixel_size_)) + cam_height_/2);
            
            //count image and time image
            if(compen_y < cam_height_ && compen_y >= 0 && compen_x < cam_width_ && compen_x >= 0){
                if(count_image[compen_y][compen_x]<threshold)count_image[compen_y][compen_x]++; 
                // std::cout << "1" << std::endl;
            }

        }

        // std::vector<int> count_image_new.resize(cam_height_*cam_width_);

        // auto ts3_5 = std::chrono::high_resolution_clock::now();

        
        auto ts4 = std::chrono::high_resolution_clock::now();


        // std::cout << "Motion compensation took "
        //           << std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts0).count() / 1000.0
        //           << " ms" << std::endl;
        // std::cout << "Event processing took " << std::chrono::duration_cast<std::chrono::microseconds>(ts3_5 - ts0).count() / 1000.0 << " ms" << std::endl;
        // std::cout << "ts2 - ts0 " << std::chrono::duration_cast<std::chrono::microseconds>(ts2 - ts0).count() / 1000.0 << " ms" << std::endl;
        // std::cout << "ts4 - ts0 " << std::chrono::duration_cast<std::chrono::microseconds>(ts4 - ts0).count() / 1000.0 << " ms" << std::endl;
    }
    return count_image;
    

}

void AngVelEstimator::compute_ts_splitting(const std::vector<dvs_msgs::Event>& event_queue,double decay_sec,ros::Time stamp){
    auto Ts0 = std::chrono::high_resolution_clock::now();

    const size_t total_events = event_queue.size();
    const size_t segment_size = total_events / 10;
    
    // 预分配内存
    std::vector<std::vector<dvs_msgs::Event>> segments;
    segments.reserve(10);
    // decay_sec = 30.0/1000.0;   
    // std::cout << "decay_sec: " << decay_sec << std::endl;
    // 设置网络的输入
    std::vector<std::vector<uint8_t>> all_results;
    all_results.reserve(10 * cam_height_);
    
    // 使用迭代器避免拷贝
    auto it = event_queue.begin();
    ros::Time t0=it->ts;
    for (size_t i = 0; i < 10; ++i) {
        auto segment_end = (i == 9) ? event_queue.end() : it + segment_size;
        segments.emplace_back(it, segment_end);
        it = segment_end;
        
        // 处理当前分段
        ros::Time external_sync_time=segments.back().back().ts;

        auto result = compute_ts(segments.back(),external_sync_time,decay_sec);
        // auto result = AA_thread(segments.back(),external_sync_time,decay_sec);
        // publishNewImageData(result, t0);
        // publishImage_ts(result, t0);
        for (const auto& row : result) {
            all_results.push_back(row);
        }
    }

    publishNewImageData(all_results, t0, stamp);
    auto Ts1 = std::chrono::high_resolution_clock::now();
    // std::cout << "compensationCallback Ts1 - Ts0 " << std::chrono::duration_cast<std::chrono::microseconds>(Ts1 - Ts0).count() / 1000.0 << " ms" << std::endl;

}


std::vector<std::vector<uint8_t>> AngVelEstimator::AA_thread(
    const std::vector<dvs_msgs::Event>& event_queue,
    const ros::Time& external_sync_time,
    double decay_sec)
{
    const double sync_sec = external_sync_time.toSec();

    // 初始化活动图（像素级），值范围 0~255
    std::vector<std::vector<uint8_t>> representation_AA(
        cam_height_,
        std::vector<uint8_t>(cam_width_, 0));

    // 初始化每个 patch 的活动值
    std::vector<double> final_activity(x_patches_ * y_patches_, 0.0);
    std::vector<double> event_activity(x_patches_ * y_patches_, 0.0);
    std::vector<double> last_event_time(x_patches_ * y_patches_, 0.0);
    std::vector<double> last_activity(x_patches_ * y_patches_, 0.0);
    std::vector<bool> flag(x_patches_ * y_patches_, true);

    int flags = 0;
    const double conv_thresh = 0.95; // 收敛阈值
    const int events_per_check = 10; // 每多少事件检查一次收敛

    // 正向遍历事件，计算每个 patch 的最终活动值
    for (const auto& e : event_queue) {
        double event_sec = e.ts.toSec();
        if (event_sec >= sync_sec) continue;

        // 确定当前事件属于哪个 patch
        int y_patch = e.y / static_cast<int>(
            std::ceil(static_cast<double>(cam_height_) / y_patches_));
        int x_patch = e.x / static_cast<int>(
            std::ceil(static_cast<double>(cam_width_) / x_patches_));
        int patch_idx = y_patch * x_patches_ + x_patch;

        if (patch_idx >= x_patches_ * y_patches_) continue;

        // 更新活动值（衰减）
        double dt = std::abs(event_sec - last_event_time[patch_idx]);
        double beta = 1.0 / (1.0 + final_activity[patch_idx] * dt);
        final_activity[patch_idx] = beta * final_activity[patch_idx] + 1.0;
        last_event_time[patch_idx] = event_sec;
    }

    // 反向遍历事件，检测收敛并生成像素级活动图
    std::fill(event_activity.begin(), event_activity.end(), 0.0);
    std::fill(last_event_time.begin(), last_event_time.end(), 0.0);
    std::vector<int> event_count(x_patches_ * y_patches_, 0);

    for (auto it = event_queue.rbegin(); it != event_queue.rend(); ++it) {
        const auto& e = *it;
        double event_sec = e.ts.toSec();
        if (event_sec >= sync_sec) continue;

        // 确定当前事件属于哪个 patch
        int y_patch = e.y / static_cast<int>(
            std::ceil(static_cast<double>(cam_height_) / y_patches_));
        int x_patch = e.x / static_cast<int>(
            std::ceil(static_cast<double>(cam_width_) / x_patches_));
        int patch_idx = y_patch * x_patches_ + x_patch;

        if (patch_idx >= x_patches_ * y_patches_) continue;
        if (!flag[patch_idx]) continue; // 已收敛，跳过

        // 更新活动值（衰减）
        double dt = std::abs(event_sec - last_event_time[patch_idx]);
        double beta = 1.0 / (1.0 + event_activity[patch_idx] * dt);
        event_activity[patch_idx] = beta * event_activity[patch_idx] + 1.0;
        last_event_time[patch_idx] = event_sec;

        // 归一化并写入像素级活动图（值域0~255）
        double normalized = std::min(1.0, event_activity[patch_idx] / final_activity[patch_idx]);
        representation_AA[e.y][e.x] = static_cast<uint8_t>(normalized * 255.0);

        event_count[patch_idx]++;
        if (event_count[patch_idx] >= events_per_check) {
            if (std::abs(event_activity[patch_idx] - final_activity[patch_idx]) < conv_thresh) {
                flag[patch_idx] = false;
                flags++;
                if (flags == x_patches_ * y_patches_) break; // 全部收敛
            }
            event_count[patch_idx] = 0;
        }
    }
    // 打印 representation_AA 的尺寸
    // ROS_INFO("representation_AA size: %lu x %lu", representation_AA.size(), representation_AA.empty() ? 0 : representation_AA[0].size());
    return representation_AA;
}  

std::vector<std::vector<uint8_t>> AngVelEstimator::compute_ts(
    const std::vector<dvs_msgs::Event>& event_queue,
    const ros::Time& external_sync_time,
    double decay_sec)
{
    const double sync_sec = external_sync_time.toSec();

    // 初始化二维时间表面和最新时间戳记录
    std::vector<std::vector<uint8_t>> representation_ts(
        cam_height_,  // 注意行列顺序
        std::vector<uint8_t>(cam_width_, 0));
    
    std::vector<std::vector<double>> latest_ts(
        cam_height_,
        std::vector<double>(cam_width_, 0.0));

    // 单次遍历记录各像素最新时间戳
    for (const auto& event : event_queue) {
        const double event_sec = event.ts.toSec();
        if (event_sec >= sync_sec) continue;
        if (event.x >= cam_width_ || event.y >= cam_height_) continue;
        
        // 注意这里应该是event.y而不是event.x
        if (event_sec > latest_ts[event.y][event.x]) {
            latest_ts[event.y][event.x] = event_sec;
        }
    }
    double value = 0.0;
    // 3. 计算时间表面值
    for (int y = 0; y < cam_height_; ++y) {
        for (int x = 0; x < cam_width_; ++x) {
            if (latest_ts[y][x] > 0) {
                double dt = sync_sec - latest_ts[y][x];
                value = std::exp(-dt / decay_sec);
                representation_ts[y][x] = static_cast<uint8_t>(value * 255.0);
            }
        }
    }
    // std::cout << "[Debug] value: " << value << " " << static_cast<int>(value * 255.0) << std::endl;

    return representation_ts;
}


void AngVelEstimator::show_count_image(std::vector<std::vector<int>>&count_image, int& max_count){
    using namespace cv;
    cv::Mat image(cam_height_,cam_width_,CV_8UC1);
    int scale = (int)(255/max_count) + 1;
    #pragma omp parallel for collapse(2) 
    for(int i = 0;i < cam_height_;++i){
            for(int j = 0; j < cam_width_;++j){
                    image.at<uchar>(i,j) = count_image[i][j]*scale;
            }
    }

    //Change to sensor_message
    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
    image_pub.publish(*msg2); 
}

void AngVelEstimator::publishCountImageData_cuda(const std::vector<int>& count_image, ros::Time& t0, ros::Time& stamp) {
    panorama::CountImage msg;

    // Set header
    msg.header.stamp = stamp;

    // Set initial time
    msg.init_time = t0;

    // Set count image data
    msg.count_image.clear();
    for (int element : count_image) {
        msg.count_image.push_back(element);
    }

    // Publish message
    detection_pub.publish(msg);
}

void AngVelEstimator::publishNewImageData(const std::vector<std::vector<uint8_t>>& count_image, 
    ros::Time& t0 ,ros::Time& stamp)
{
    panorama::CountImage msg;

    msg.header.stamp = stamp;

    // Set initial time
    msg.init_time = t0;

    // Set count image data
    msg.count_image.clear();
    msg.count_image.reserve(10 * cam_height_ * cam_width_);

    for (const auto& row : count_image) {
        // 遍历当前行的每一个元素
        for (int element : row) {
            msg.count_image.push_back(element);
        }
    }


    detection_pub.publish(msg);
}

void AngVelEstimator::publishImage_ts(const std::vector<std::vector<uint8_t>>& count_image, 
    ros::Time& t0)
{
    // 创建一个 sensor_msgs::Image 消息
    sensor_msgs::Image msg;
    
    // 设置消息头和时间戳
    msg.header.stamp = t0;
    msg.header.frame_id = "camera_frame";  // 你可以根据需要修改 frame_id
    
    // 设置图像参数
    msg.height = cam_height_;
    msg.width = cam_width_;
    msg.encoding = sensor_msgs::image_encodings::MONO8;  // 8位单通道图像
    msg.is_bigendian = false;
    msg.step = msg.width;  // 对于 MONO8 格式，步长等于宽度
    
    // 分配数据空间
    msg.data.resize(msg.height * msg.width);
    
    // 将数据复制到消息中
    size_t index = 0;
    for (const auto& row : count_image) {
        for (uint8_t element : row) {
            msg.data[index++] = element;
        }
    }
    
    // 发布消息
    image_pub.publish(msg);
}


void AngVelEstimator::publishCountImageData(const std::vector<std::vector<int>>& count_image, 
    ros::Time& t0, 
    ros::Time& stamp)
{
    panorama::CountImage msg;

    // Set header
    msg.header.stamp = stamp;

    // Set initial time
    msg.init_time = t0;

    // Set count image data
    msg.count_image.clear();
    msg.count_image.reserve(cam_height_ * cam_width_);
    //TODO
    // for (const auto& row : count_image) {
    //     std_msgs::Int32MultiArray row_msg;
    //     row_msg.data = row;
    //     msg.count_image.push_back(row_msg);
    // }


    for (const auto& row : count_image) {
        // 遍历当前行的每一个元素
        for (int element : row) {
            msg.count_image.push_back(element);
        }
    }
    // msg.count_image.clear();
    // msg.count_image.reserve(cam_height_ * cam_width_);
    // msg.count_image.insert(msg.count_image.end(), count_image.begin(), count_image.end());

    detection_pub.publish(msg);
}


void AngVelEstimator::publishIMU(float x,float y,float z,uint32_t seq) {
    sensor_msgs::Imu imu_msg;
    
    // 设置 header（时间戳和坐标系）
    imu_msg.header.stamp = ros::Time::now();
    imu_msg.header.seq = seq;

    // 设置 IMU 数据
    geometry_msgs::Vector3 angular_velocity;
    angular_velocity.x = x;
    angular_velocity.y = y;
    angular_velocity.z = z;

    imu_msg.angular_velocity = angular_velocity;


    // 发布消息
    imu_pub.publish(imu_msg);
}


void AngVelEstimator::show_count_pano(cv::Mat& count_image, double& max_count, ros::Time timestamp) {
    // 创建灰度图像
    cv::Mat image(panorama_height_, panorama_width_, CV_8UC1);

    // 计算缩放比例，将 count_image 的值映射到 [0, 255]
    int scale = (int)(255 / max_count) + 1;
    count_image.convertTo(image, CV_8UC1, scale); // 将浮点型矩阵转换为 8 位无符号整型矩阵

    // 创建带时间戳的 ROS 消息头
    std_msgs::Header header;
    header.stamp = timestamp;
    // std::cout<<image.type()<<endl;

    // 将 OpenCV 图像转换为 ROS 图像消息
    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", image).toImageMsg();

    // 清空矩阵
    count_pano.release();
    image.release();

    // 发布图像消息
    pano_pub_.publish(msg);
}

} // namespace panorama