#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <serial/serial.h>
#include <std_msgs/Header.h>
#include <deque>

class PhotogateReader {
public:
    PhotogateReader() : nh_("~"), filter_window_(5) {
        nh_.param<std::string>("port", port_, "/dev/ttyUSB0");
        nh_.param<int>("baudrate", baudrate_, 9600);
        nh_.param<int>("filter_window", filter_window_, 5);
        nh_.param<double>("high_threshold", high_threshold_, 0.7);

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

        pub_ = nh_.advertise<std_msgs::Header>("photogate_state", 10);
        ROS_INFO("Photogate reader initialized on %s @ %d baud", 
                port_.c_str(), baudrate_);
    }

    void run() {
        ros::Rate rate(50);
        std::deque<bool> state_buffer;

        while (ros::ok()) {
            if (ser_.available()) {
                uint8_t byte;
                ser_.read(&byte, 1);
                // Any non-zero byte is high; encode the window majority in Header.frame_id.
                bool raw_state = (byte > 0);

                state_buffer.push_back(raw_state);
                if (state_buffer.size() > filter_window_) {
                    state_buffer.pop_front();
                }

                int high_count = std::count(state_buffer.begin(), state_buffer.end(), true);
                double high_ratio = static_cast<double>(high_count) / state_buffer.size();

                std_msgs::Header msg;
                msg.stamp = ros::Time::now();
                msg.frame_id = (high_ratio >= high_threshold_) ? "true" : "false";
                pub_.publish(msg);

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