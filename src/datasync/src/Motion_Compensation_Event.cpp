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
#include <Eigen/Dense> 
#include <image_transport/image_transport.h> 
#include <dvs_msgs/Event.h>
#include <dvs_msgs/EventArray.h>
#include <sensor_msgs/Imu.h>

#include <sensor_msgs/image_encodings.h>
#include "time.h"
#include <boost/thread.hpp>     
#include <mutex> 
#include <chrono> 
#include <cmath>

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

int panorama_height_;
int panorama_weight_;
cv::Mat count_pano = cv::Mat::zeros(panorama_height_, panorama_weight_, CV_32F);

double pi = M_PI;
sll t0_p;

cv::Size imageSize(panorama_weight_, panorama_height_);

class EventVisualizer {
    protected:
        ros::NodeHandle n_;
        ros::Publisher image_pub;
        ros::Publisher event_pub;
        ros::Publisher pano_pub;

        ros::Subscriber event_sub;
        ros::Subscriber imu_sub;

        std::mutex mtx;

        Eigen::Matrix3d R;
        Eigen::Matrix3d K;

    public:
        EventVisualizer(ros::NodeHandle n) : n_(n) {
            R << 1., 0., 0.,
                 0., -0.93033817, 0.36670272,
                 0., -0.36670272, -0.93033817;

            K << 433.398607901605, 0.284227880114643, 235.267698381848,
                 0, 433.485784159950, 320.716442889263,
                 0, 0, 1;

            this->event_sub = this->n_.subscribe("/dvs/events", 1, &EventVisualizer::event_cb, this);
            this->imu_sub = this->n_.subscribe("/dvs/imu", 7, &EventVisualizer::imu_cb, this);
            this->image_pub = n_.advertise<sensor_msgs::Image>("/count_image", 1);
            this->pano_pub = n_.advertise<sensor_msgs::Image>("/count_pano", 1);
            this->event_pub = n_.advertise<dvs_msgs::EventArray>("/event_new", 1);
            
        }

        void show_count_image(std::vector<std::vector<int>>&count_image,int& max_count, ros::Time timestamp);
        void show_count_pano(cv::Mat& count_image, double& max_count, ros::Time timestamp);

        void data_process();

        void events(const std::vector<dvs_msgs::Event>& event_buffer, int size);

        void event_cb(const dvs_msgs::EventArray::ConstPtr& msg) {
            if (first_event_received == false) {

                mtx.lock();
                imu_buffer_ = imu_buffer;
                if (imu_buffer.size() != 0) {
                    imu_buffer.clear();
                }
                mtx.unlock();

                if (imu_buffer_.size() == 0) {
                    return;
                }

                for (int i = 0; i < msg->events.size(); ++i) {
                    event_buffer.emplace_back(msg->events[i]);
                }

                data_process();
            } else {
                first_event_received = false;

                if (imu_buffer.size() != 0) {
                    imu_buffer.clear();
                }

                std::cout << "Data aligned!" << std::endl;
                std::cout << "Start processing data..." << std::endl;
            }
        }

        void imu_cb(const sensor_msgs::ImuConstPtr& imu) {
            if (first_event_received == false) {
                imu_buffer.emplace_back(*imu);
            }
        }
};

void EventVisualizer::show_count_image(std::vector<std::vector<int>>&count_image, int& max_count, ros::Time timestamp){
    using namespace cv;
    cv::Mat image(height_,weight_,CV_8UC1);
    int scale = (int)(255/max_count) + 1;
    for(int i = 0;i < height_;++i){
            for(int j = 0; j < weight_;++j){
                    image.at<uchar>(i,j) = count_image[i][j]*scale;
            }
    }

    std_msgs::Header header;
    header.stamp = timestamp;

    sensor_msgs::ImagePtr msg2 = cv_bridge::CvImage(header, "mono8", image).toImageMsg();
    image_pub.publish(*msg2); 
}

void EventVisualizer::show_count_pano(cv::Mat& count_image, double& max_count, ros::Time timestamp) {
    cv::Mat image(panorama_height_, panorama_weight_, CV_8UC1);

    int scale = (int)(255 / max_count) + 1;
    count_image.convertTo(image, CV_8UC1, scale);

    std_msgs::Header header;
    header.stamp = timestamp;

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(header, "mono8", image).toImageMsg();

    pano_pub.publish(msg);
}

void EventVisualizer::events(const std::vector<dvs_msgs::Event>& event_buffer, int size) {
    dvs_msgs::EventArray event_array_msg;

    event_array_msg.header.stamp = ros::Time::now();
    event_array_msg.header.frame_id = std::to_string(size);

    event_array_msg.height = 480;
    event_array_msg.width = 640;

    event_array_msg.events = event_buffer;

    event_pub.publish(event_array_msg);
}

void EventVisualizer::data_process() {

    if (imu_buffer_[imu_buffer_.size() - 1].header.stamp.toNSec() > event_buffer[0].ts.toNSec()) {
        float angular_velocity_x = 0.0, angular_velocity_y = 0.0, angular_velocity_z = 0.0;
        float average_angular_rate_x, average_angular_rate_y, average_angular_rate_z;

        int cnt = 0;

        for (int i = 0; i < imu_buffer_.size(); ++i) {
            if (imu_buffer_[i].header.stamp.toNSec() >= (event_buffer[0].ts.toNSec() - 3000000)) {
                angular_velocity_x += imu_buffer_[i].angular_velocity.x;
                angular_velocity_y += imu_buffer_[i].angular_velocity.y;
                angular_velocity_z += imu_buffer_[i].angular_velocity.z;
                cnt++;
            }
        }
        average_angular_rate_x = angular_velocity_x / float(cnt);
        average_angular_rate_y = angular_velocity_y / float(cnt);
        average_angular_rate_z = angular_velocity_z / float(cnt);
        float average_angular_rate = std::sqrt((average_angular_rate_x * average_angular_rate_x) + (average_angular_rate_y * average_angular_rate_y) + (average_angular_rate_z * average_angular_rate_z));

        sll t0=event_buffer[0].ts.toNSec();
        float time_diff = 0.0;
        std::vector<std::vector<int>>count_image(height_,std::vector<int>(weight_));
        std::vector<std::vector<float>>time_image(height_,std::vector<float>(weight_));
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

        int size = event_buffer.size();

         int max_count = 0;
         float max_time = 0.0;
         float total_time = 0.0;
         float average_time = 0.0;
         int trigger_pixels = 0;

         for(int i = 0; i<height_; ++i){
                 for(int j = 0; j < weight_; ++j){
                         if(count_image[i][j] != 0){
                                 time_image[i][j] /= count_image[i][j];
                                 max_count = std::max(max_count,count_image[i][j]);
                         }
                 }
         }

        events(event_buffer,size);
        show_count_image(count_image, max_count,event_buffer[0].ts);

        event_buffer.clear();
        imu_buffer_.clear();
    } else {
        if (event_buffer.size() != 0) {

            event_buffer.clear();
        }
        if (imu_buffer_.size() != 0) {

            imu_buffer_.clear();
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "datasync_node");

    ros::NodeHandle nh;
    EventVisualizer visualizer(nh);
    ros::NodeHandle nh_priv("~");

    nh_priv.param<int>("weight_param", weight_, 346);
    nh_priv.param<int>("height_param", height_, 260);
    nh_priv.param<int>("panorama_weight", panorama_weight_, 1024);
    nh_priv.param<int>("panorama_height", panorama_height_, 480);
    nh_priv.param<float>("focus", Focus_, 6550);
    nh_priv.param<float>("pixel_size", pixel_size_, 18.5);

    ros::AsyncSpinner spinner(3);
    spinner.start();
    ros::waitForShutdown();

    return 0;
}
