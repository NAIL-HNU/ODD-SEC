#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <serial/serial.h>
#include <std_msgs/Header.h>
#include <deque>

class PhotogateReader {
public:
    PhotogateReader() : nh_("~"), filter_window_(5) {
        // 参数配置
        nh_.param<std::string>("port", port_, "/dev/ttyUSB0");
        nh_.param<int>("baudrate", baudrate_, 9600);
        nh_.param<int>("filter_window", filter_window_, 5);
        nh_.param<double>("high_threshold", high_threshold_, 0.7);  // 信号阈值比例

        // 初始化串口
        try {
            ser_.setPort(port_);
            ser_.setBaudrate(baudrate_);
            serial::Timeout to = serial::Timeout::simpleTimeout(1000);
            ser_.setTimeout(to);
            ser_.open();
        } catch (const std::exception& e) {
            ROS_FATAL("Serial port %s open failed: %s", port_.c_str(), e.what());
            ros::shutdown();
        }

        // 发布Topic
        pub_ = nh_.advertise<std_msgs::Header>("photogate_state", 10);
        ROS_INFO("Photogate reader initialized on %s @ %d baud", 
                port_.c_str(), baudrate_);
    }

    void run() {
        ros::Rate rate(50);  // 50Hz采样率
        std::deque<bool> state_buffer;

        while (ros::ok()) {
            if (ser_.available()) {
                // 读取原始信号
                uint8_t byte;
                ser_.read(&byte, 1);
                bool raw_state = (byte > 0);

                // 滑动窗口滤波
                state_buffer.push_back(raw_state);
                if (state_buffer.size() > filter_window_) {
                    state_buffer.pop_front();
                }

                // 计算高电平比例
                int high_count = std::count(state_buffer.begin(), state_buffer.end(), true);
                double high_ratio = static_cast<double>(high_count) / state_buffer.size();

                // 发布稳定状态
                std_msgs::Header msg;
                msg.stamp = ros::Time::now();  // 设置时间戳
                msg.frame_id = (high_ratio >= high_threshold_) ? "true" : "false";  // 使用 frame_id 表示状态
                pub_.publish(msg);

                // 调试输出
                // ROS_DEBUG_THROTTLE(1, "Raw: %d, Filtered: %d (%.1f%%)", 
                //                   raw_state, msg.frame_id.c_str(), high_ratio*100);
            }
            ros::spinOnce();
            rate.sleep();
        }
        ser_.close();
    }

private:
    ros::NodeHandle nh_;
    serial::Serial ser_;
    ros::Publisher pub_;
    int filter_window_;
    double high_threshold_;
    std::string port_;
    int baudrate_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "photogate_reader");
    PhotogateReader reader;
    reader.run();
    return 0;
}