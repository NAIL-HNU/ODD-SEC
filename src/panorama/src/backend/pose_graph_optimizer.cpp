
#include "backend/pose_graph_optimizer.h"
#include "frontend/ang_vel_estimator.h"

#include <ros/ros.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>

#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <cmath>
#include <iostream>
using namespace std;  

namespace panorama {

PoseGraphOptimizer::PoseGraphOptimizer(ros::NodeHandle* nh): nh_(nh), it_(*nh)
{
    image_pub = it_.advertise("count_image", 2);
    marker_pub = nh->advertise<visualization_msgs::Marker>("polar_line_3d", 10);
}

PoseGraphOptimizer::~PoseGraphOptimizer()
{

    marker_pub.shutdown();
    image_pub.shutdown();
}

void PoseGraphOptimizer::initialize(int camera_width, int camera_height,
                                    const PoseGraphParams &opt,
                                    std::vector<dvs_msgs::Event>* ptr,
                                    std::vector<cv::Point3d>* precomputed_bearing_vectors_ptr)
{
    params = opt;
    panorama_width_=params.image_opt.panorama_width;
    panorama_height_=params.image_opt.panorama_height;

    cam_width_ = camera_width;
    cam_height_ = camera_height;
    precomputed_bearing_vectors_ptr_ = *precomputed_bearing_vectors_ptr;

    std::vector<double> R_matrix = params.image_opt.R_matrix;

    if (R_matrix.size() == 9) {
        Eigen::Map<const Eigen::Matrix3d> R_raw(R_matrix.data());
        R = R_raw.transpose();
    } else {
        ROS_ERROR("Invalid rotation matrix size");
        R = Eigen::Matrix3d::Identity();
    }

}

void PoseGraphOptimizer::WarpPano(std::vector<std::vector<int>>& count_image,ros::Time& t0,ros::Time& stamp)
{
    int threshold=15;
    show_count_image(count_image, threshold); 

    auto [x, y] = DroneDetection(count_image);

    const cv::Point3d bvec = precomputed_bearing_vectors_ptr_.at(y*cam_width_ + x);
    Eigen::Vector3d e_ray_cam(bvec.x, bvec.y, bvec.z);

    float t_diff = double(t0.toNSec() - stamp.toNSec()) / 1000000000.0;

    double cos_term = cos(2 * pi * t_diff);
    double sin_term = sin(2 * pi * t_diff);
    

    R_eigen_ << cos_term, R(2, 1) * sin_term, R(2, 2) * sin_term,
                0, R(1, 1), R(1, 2),
                -sin_term, R(2, 1) * cos_term, R(2, 2) * cos_term;

    Eigen::Vector3d e_ray_w = R_eigen_ * e_ray_cam;

    const double phi = std::atan2(e_ray_w[0], e_ray_w[2]);
    const double theta = std::asin(e_ray_w[1] / e_ray_w.norm());

    cout<<"phi :"<< phi <<endl;
    cout<<"theta :"<< theta <<endl;

    double r=1;

    publishSphericalMarker(marker_pub, r, theta, phi,"map");

}

void PoseGraphOptimizer::sphericalToCartesian(double r, double theta_rad, double phi_rad, 
                         double& x, double& y, double& z) {
    x = r * sin(phi_rad) * cos(theta_rad);
    y = r * sin(phi_rad) * sin(theta_rad);
    z = r * cos(phi_rad);
}

void PoseGraphOptimizer::publishSphericalMarker(ros::Publisher& pub, 
                          double r, double theta_rad, double phi_rad, 
                          const std::string& frame_id ) {
    double x_end, y_end, z_end;
    sphericalToCartesian(r, theta_rad, phi_rad, x_end, y_end, z_end);

    visualization_msgs::Marker line_marker;
    line_marker.header.frame_id = frame_id;
    line_marker.header.stamp = ros::Time::now();
    line_marker.ns = "polar_line_3d";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::Marker::ADD;
    line_marker.scale.x = 0.05;
    line_marker.color.b = 1.0;
    line_marker.color.a = 1.0;

    geometry_msgs::Point start_point, end_point;
    start_point.x = 0; start_point.y = 0; start_point.z = 0;
    end_point.x = x_end; end_point.y = y_end; end_point.z = z_end;
    line_marker.points.push_back(start_point);
    line_marker.points.push_back(end_point);

    visualization_msgs::Marker text_marker;

    double theta_deg=theta_rad*pi;
    double phi_deg=phi_rad*pi;
    text_marker.header = line_marker.header;
    text_marker.ns = "polar_text_3d";
    text_marker.id = 1;
    text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    text_marker.text = "End: (" + std::to_string(x_end).substr(0, 4) + ", " + 
                      std::to_string(y_end).substr(0, 4) + ", " + 
                      std::to_string(z_end).substr(0, 4) + ")\n" +
                      "Spherical: r=" + std::to_string(r).substr(0, 3) + 
                      ", θ=" + std::to_string(theta_deg).substr(0, 2) + "°" +
                      ", φ=" + std::to_string(phi_deg).substr(0, 2) + "°";
    text_marker.pose.position = end_point;
    text_marker.scale.z = 0.2;
    text_marker.color.r = 1.0;
    text_marker.color.g = 1.0;
    text_marker.color.a = 1.0;

    pub.publish(line_marker);
    pub.publish(text_marker);
}

std::tuple<float, float> PoseGraphOptimizer::DroneDetection(std::vector<std::vector<int>>& count_image) {
    float x=40.1;
    float y=380.5;

    return std::make_tuple(x, y);
}

void PoseGraphOptimizer::show_count_image(std::vector<std::vector<int>>&count_image, int& max_count){

    static auto last_time = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::high_resolution_clock::now();

    count+=std::chrono::duration_cast<std::chrono::microseconds>(now - last_time).count() / 1000.0;
    c++;
    double avg = count / c;

    std::cout << "count_image times: " 
            << c
            << std::endl;    

    std::cout << "count_image avg interval: " 
            << avg
            << " ms" << std::endl;

    last_time = now;

    using namespace cv;
    cv::Mat image(cam_height_,cam_width_,CV_8UC1);
    int scale = (int)(255/max_count) + 1;
    #pragma omp parallel for collapse(2) 
    for(int i = 0;i < cam_height_;++i){
            for(int j = 0; j < cam_width_;++j){
                    image.at<uchar>(i,j) = count_image[i][j]*scale;
            }
    }

    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
    image_pub.publish(*msg2); 

    auto ts4 = std::chrono::high_resolution_clock::now();
    std::cout << "count_image Ts1 - Ts0 " << std::chrono::duration_cast<std::chrono::microseconds>(ts4 - now).count() / 1000.0 << " ms" << std::endl;
    
}

}