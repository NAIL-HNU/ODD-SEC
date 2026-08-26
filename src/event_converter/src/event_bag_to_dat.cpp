#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <dvs_msgs/EventArray.h>
#include <std_msgs/Header.h>
#include <fstream>
#include <cstdint>
#include <cstdlib>

struct EventData {
    uint32_t t;
    uint32_t raw;  // polarity[28], y[27:14], x[13:0]
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "event_bag_to_dat");
    ros::NodeHandle nh;

    if (argc < 4) {
        ROS_ERROR("Usage: rosrun your_package event_bag_to_dat <input_bag_file> <output_dat_file> <t>");
        return -1;
    }

    std::string input_bag_file = argv[1];
    std::string output_dat_file = argv[2];
    uint64_t t = std::strtoull(argv[3], nullptr, 10);

    rosbag::Bag bag;
    try {
        bag.open(input_bag_file, rosbag::bagmode::Read);
    } catch (rosbag::BagException& e) {
        ROS_ERROR("Failed to open bag file: %s", e.what());
        return -1;
    }

    std::vector<std::string> topics = {"/event_new", "/count_image"};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    std::ofstream dat_file(output_dat_file, std::ios::out | std::ios::binary);
    if (!dat_file.is_open()) {
        ROS_ERROR("Failed to open output .dat file");
        return -1;
    }

    uint64_t T0 = 0;
    bool first_count_image_received = false;

    uint64_t first_event_timestamp = 0;
    bool first_event_received = false;

    for (const rosbag::MessageInstance& msg : view) {

        if (msg.getTopic() == "/event_new") {
            dvs_msgs::EventArray::ConstPtr event_array = msg.instantiate<dvs_msgs::EventArray>();
            if (event_array != nullptr) {
                for (const auto& event : event_array->events) {
                    if (!first_event_received) {
                        first_event_timestamp = event.ts.toNSec();
                        first_event_received = true;
                        ROS_INFO("First event original timestamp: %lu", first_event_timestamp);
                        ROS_INFO("First event timestamp will be set to t: %lu", t);
                    }

                    EventData eventData;

                    uint16_t x = event.x;
                    uint16_t y = event.y;
                    uint8_t p = event.polarity ? 1 : 0;

                    eventData.t = static_cast<uint32_t>(
                        t + ((event.ts.toNSec() - first_event_timestamp) / 1000)
                    );

                    eventData.raw = (p << 28) | (y << 14) | x;

                    dat_file.write(reinterpret_cast<const char*>(&eventData), sizeof(EventData));
                }
            }
        }
    }

    dat_file.close();
    bag.close();

    ROS_INFO("Conversion completed successfully. Output saved to %s", output_dat_file.c_str());
    return 0;
}
