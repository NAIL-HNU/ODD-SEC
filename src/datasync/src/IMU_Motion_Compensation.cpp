#include <ros/ros.h>
#include <algorithm>
#include <cmath>
#include <cfloat>

#include <iostream>
#include <fstream>

#include <cv_bridge/cv_bridge.h> 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <image_transport/image_transport.h> 

#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Imu.h>

#include <sensor_msgs/image_encodings.h>
#include "time.h"
#include <boost/thread.hpp>     
#include <mutex> 

using namespace std;

typedef long long int sll;

std::vector<dvs_msgs::Event> event_buffer;
std::vector<sensor_msgs::Imu> imu_buffer;
std::vector<sensor_msgs::Imu> imu_buffer_;
bool first_event_received = true;

int height_;
int weight_;
float Focus_;
float pixel_size_;

// Offline ROS utility that uses a local IMU window to compensate event motion.
class EventVisualizer{
        protected:
                 ros::NodeHandle n_;
                 ros::Publisher  image_pub;
                 ros::Publisher  split_pub;

                 ros::Subscriber event_sub;
                 ros::Subscriber imu_sub;
                 

                 std::mutex mtx;
        public:
                 EventVisualizer (ros::NodeHandle n):
                        n_(n){
                                this->event_sub = this->n_.subscribe("/dvs/events", 1, &EventVisualizer::event_cb, this);
                                this->imu_sub = this->n_.subscribe("/dvs/imu", 7, &EventVisualizer::imu_cb, this);
                                this->image_pub = n_.advertise<sensor_msgs::Image> ("/count_image2", 1);
                                this->split_pub = n_.advertise<sensor_msgs::Image> ("/split_image", 1);
                        }
                 
                 void show_count_image(std::vector<std::vector<int>>&count_image,int& max_count);

                 void data_process();

                 void show_split_image(std::vector<std::vector<float>>&time_image);
                 
                 std::vector<std::vector<bool>> dynamicThresholdSegmentation(
                        const std::vector<std::vector<float>>& time_image,
                        const std::vector<std::vector<int>>& count_image,
                        float average_angular_rate,int height,int width,
                        float total_time, int trigger_pixels, float delta_t);

                 void convertDynamicMaskToMat(const std::vector<std::vector<bool>>& dynamic_mask);
                 
                 void event_cb(const dvs_msgs::EventArray::ConstPtr& msg){
                        if (first_event_received == false) {

                                mtx.lock();
                                imu_buffer_ = imu_buffer;
                                if(imu_buffer.size() != 0){
                                    imu_buffer.clear();
                                }
                                mtx.unlock();

                                if(imu_buffer_.size() == 0){
                                       return;
                                }
                    
                                for (int i = 0; i < msg->events.size(); ++i) {
                                         event_buffer.emplace_back(msg->events[i]);
                                        
                                }

                                data_process();
                        }else{
                                first_event_received = false;
                                
                                if(imu_buffer.size() != 0){
                                      imu_buffer.clear();  
                                }

                                std::cout<<"Data aligned!"<<std::endl;
                                std::cout<<"Start processing data..."<<std::endl;
                        }
                }

                 void imu_cb(const sensor_msgs::ImuConstPtr& imu){
                         if(first_event_received == false){
                                 imu_buffer.emplace_back(*imu);        
                        }
                }
}; 

void EventVisualizer::show_count_image(std::vector<std::vector<int>>&count_image, int& max_count){
        using namespace cv;
        cv::Mat image(height_,weight_,CV_8UC1);
        int scale = (int)(255/max_count) + 1;
        for(int i = 0;i < height_;++i){
                for(int j = 0; j < weight_;++j){
                        image.at<uchar>(i,j) = count_image[i][j]*scale;
                }
        }

        sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
        image_pub.publish(*msg2); 
}

void EventVisualizer::show_split_image(std::vector<std::vector<float>>&time_image){
        using namespace cv;

        cv::Mat image(height_, weight_, CV_8UC3);

        for (int i = 0; i < height_; ++i) {
                for (int j = 0; j < weight_; ++j) {
                float value = time_image[i][j];
                if (value < 0.0f) value = 0.0f;
                if (value > 1.0f) value = 1.0f;

                float hue = value * 180;
                float saturation = 1.0f;
                float brightness = 1.0f;

                cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(hue, saturation * 255, brightness * 255));
                cv::Mat bgr;
                cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

                image.at<cv::Vec3b>(i, j) = cv::Vec3b(bgr.data[0], bgr.data[1], bgr.data[2]);
                }
        }

        sensor_msgs::ImagePtr msg3 = cv_bridge::CvImage(std_msgs::Header(), "mono8", image).toImageMsg();
        image_pub.publish(*msg3); 
}

void EventVisualizer::data_process(){
         if(imu_buffer_[imu_buffer_.size() - 1].header.stamp.toNSec() > event_buffer[0].ts.toNSec()){
                ROS_INFO("1");
                 
                 float angular_velocity_x=0.0, angular_velocity_y=0.0,angular_velocity_z=0.0;
                 float average_angular_rate_x, average_angular_rate_y,average_angular_rate_z;

                 
                 int cnt=0;
                 for(int i=0;i<imu_buffer_.size();++i){
                         if(imu_buffer_[i].header.stamp.toNSec() >= (event_buffer[0].ts.toNSec()-3000000)){
                                 angular_velocity_x+=imu_buffer_[i].angular_velocity.x;
                                 angular_velocity_y+=imu_buffer_[i].angular_velocity.y;
                                 angular_velocity_z+=imu_buffer_[i].angular_velocity.z;
                                 cnt++;
                        }  
                }
                 average_angular_rate_x = angular_velocity_x/float(cnt);
                 average_angular_rate_y = angular_velocity_y/float(cnt);
                 average_angular_rate_z = angular_velocity_z/float(cnt);
                 float average_angular_rate = std::sqrt((average_angular_rate_x*average_angular_rate_x) + (average_angular_rate_y*average_angular_rate_y) + (average_angular_rate_z*average_angular_rate_z));

                 
                 sll t0=event_buffer[0].ts.toNSec();
                 float time_diff = 0.0;
                 std::vector<std::vector<int>>count_image(height_,std::vector<int>(weight_));
                 std::vector<std::vector<float>>time_image(height_,std::vector<float>(weight_));
                 std::vector<std::vector<bool>> dynamic_mask(height_, std::vector<bool>(weight_));
                 for(int i=0;i<event_buffer.size();++i){
                        time_diff = double(event_buffer[i].ts.toNSec()-t0)/1000000000.0;

                        float x_angular=time_diff*average_angular_rate_x;
                        float y_angular=time_diff*average_angular_rate_y;
                        float z_angular=time_diff*average_angular_rate_z;

                        
                        int x=event_buffer[i].x - weight_/2; 
                        int y=event_buffer[i].y - height_/2;
                        
                        float pre_x_angel = atan(y*pixel_size_/Focus_);
                        float pre_y_angel = atan(x*pixel_size_/Focus_);

                        int compen_x = (int)((x*cos(z_angular) - sin(z_angular)*y) - (x - (Focus_*tan(pre_y_angel + y_angular)/pixel_size_)) + weight_/2);
                        int compen_y = (int)((x*sin(z_angular) + cos(z_angular)*y) - (y - (Focus_*tan(pre_x_angel - x_angular)/pixel_size_)) + height_/2);
                        event_buffer[i].x = compen_x;
                        event_buffer[i].y = compen_y;
                        
                        
                        if(compen_y < height_ && compen_y >= 0 && compen_x < weight_ && compen_x >= 0){
                            if(count_image[compen_y][compen_x]<20)count_image[compen_y][compen_x]++; 
                            time_image[compen_y][compen_x] += time_diff;
                        }
                    }

                 int max_count = 0;
                 float max_time = 0.0;
                 float total_time = 0.0;
                 float average_time = 0.0;
                 int trigger_pixels = 0;

                 for(int i = 0; i<height_; ++i){
                        for(int j = 0; j < weight_; ++j){
                                if(count_image[i][j] != 0){
                                        trigger_pixels++;
                                        time_image[i][j] /= count_image[i][j];
                                        total_time+=time_image[i][j];
                                        max_count = std::max(max_count,count_image[i][j]);
                                }
                        }
                 }

                float delta_t=double(event_buffer[event_buffer.size()- 1].ts.toNSec()-t0)/1000000000.0;
                ROS_INFO_STREAM("event_buffer[event_buffer.size()].ts.toNSec(): " << event_buffer[event_buffer.size()].ts.toNSec());
                ROS_INFO_STREAM("t0: " << t0);
                ROS_INFO_STREAM("delta_t: " << delta_t);
                
                average_time=total_time/trigger_pixels;
                ROS_INFO_STREAM("average_time: " << average_time);
                 dynamic_mask=dynamicThresholdSegmentation(
                        time_image,
                        count_image,
                        average_angular_rate, height_, weight_, average_time,  trigger_pixels, delta_t);
                convertDynamicMaskToMat(dynamic_mask);

                 show_count_image(count_image, max_count);  
                
                 event_buffer.clear();
                 imu_buffer_.clear();
        }else{
                if(event_buffer.size() != 0){

                        event_buffer.clear();
                } 
                if(imu_buffer_.size() != 0){

                        imu_buffer_.clear();
                }
        }
}

std::vector<std::vector<bool>> EventVisualizer::dynamicThresholdSegmentation(
    const std::vector<std::vector<float>>& time_image,
    const std::vector<std::vector<int>>& count_image,
    float average_angular_rate,int height,int width,float total_time, int trigger_pixels, float delta_t) {

    std::vector<std::vector<float>> normalized_time_image(height, std::vector<float>(width, 0.0f));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (count_image[y][x] != 0) {
                normalized_time_image[y][x] = (total_time-time_image[y][x]  ) / delta_t;
            }
        }
    }

        float a=0.1;
        float b=0.15;
    float lambda_threshold = a * std::abs(average_angular_rate) + b;

    ROS_INFO_STREAM("average_angular_rate: " << average_angular_rate);
    ROS_INFO_STREAM("lambda_threshold: " << lambda_threshold);

    std::vector<std::vector<bool>> dynamic_mask(height, std::vector<bool>(width, false));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (normalized_time_image[y][x] > lambda_threshold) {
                dynamic_mask[y][x] = true;
            }
        }
    }

    return dynamic_mask;
}

void EventVisualizer::convertDynamicMaskToMat(const std::vector<std::vector<bool>>& dynamic_mask) {
    int height = dynamic_mask.size();
    int width = dynamic_mask[0].size();
    cv::Mat mask_mat(height, width, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (dynamic_mask[y][x]) {
                mask_mat.at<uchar>(y, x) = 255;
            }
        }
    }

        sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(std_msgs::Header(), "mono8", mask_mat).toImageMsg();
        split_pub.publish(*msg2); 

    
}

  int main(int argc, char** argv)
   {
   ros::init(argc, argv, "datasync_node");
 
   ros::NodeHandle nh;
   EventVisualizer visualizer(nh);
   ros::NodeHandle nh_priv("~");

   nh_priv.param<int>("weight_param", weight_, 346);
   nh_priv.param<int>("height_param", height_, 260);
   nh_priv.param<float>("focus", Focus_, 6550);
   nh_priv.param<float>("pixel_size",pixel_size_ , 18.5);

   ros::AsyncSpinner spinner(3);
   spinner.start();
   ros::waitForShutdown();

   return 0;
 }
