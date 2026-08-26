#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <dvs_msgs/EventArray.h>
#include <std_msgs/Header.h> // 用于 /count_image 消息类型
#include <fstream>
#include <cstdint>
#include <cstdlib> // for std::strtoull

struct EventData {
    uint32_t t;
    uint32_t raw; // x, y, polarity
};

int main(int argc, char** argv) {
    // 初始化 ROS 节点
    ros::init(argc, argv, "event_bag_to_dat");
    ros::NodeHandle nh;

    // 检查输入参数
    if (argc < 4) {
        ROS_ERROR("Usage: rosrun your_package event_bag_to_dat <input_bag_file> <output_dat_file> <t>");
        return -1;
    }

    // 获取输入和输出文件路径以及 t
    std::string input_bag_file = argv[1];
    std::string output_dat_file = argv[2];
    uint64_t t = std::strtoull(argv[3], nullptr, 10); // 将字符串转换为 uint64_t

    // 打开 ROS bag 文件
    rosbag::Bag bag;
    try {
        bag.open(input_bag_file, rosbag::bagmode::Read);
    } catch (rosbag::BagException& e) {
        ROS_ERROR("Failed to open bag file: %s", e.what());
        return -1;
    }

    // 设置要读取的 topics
    std::vector<std::string> topics = {"/event_new", "/count_image"};
    rosbag::View view(bag, rosbag::TopicQuery(topics));

    // 打开输出文件
    std::ofstream dat_file(output_dat_file, std::ios::out | std::ios::binary);
    if (!dat_file.is_open()) {
        ROS_ERROR("Failed to open output .dat file");
        return -1;
    }

    // 用于记录 /count_image 的第一个时间戳
    uint64_t T0 = 0;
    bool first_count_image_received = false;

    // 用于记录 /event_new 的第一个事件的时间戳
    uint64_t first_event_timestamp = 0;
    bool first_event_received = false;

    // 遍历 bag 文件中的消息
    for (const rosbag::MessageInstance& msg : view) {

        // 处理 /event_new 话题的消息
        if (msg.getTopic() == "/event_new") {
            dvs_msgs::EventArray::ConstPtr event_array = msg.instantiate<dvs_msgs::EventArray>();
            if (event_array != nullptr) {
                // 遍历所有事件
                for (const auto& event : event_array->events) {
                    // 如果是第一个事件，记录其原始时间戳
                    if (!first_event_received) {
                        first_event_timestamp = event.ts.toNSec();
                        first_event_received = true;
                        ROS_INFO("First event original timestamp: %lu", first_event_timestamp);
                        ROS_INFO("First event timestamp will be set to t: %lu", t);
                    }

                    // 将事件数据写入 .dat 文件
                    EventData eventData;

                    uint16_t x = event.x;                   // x 坐标
                    uint16_t y = event.y;                   // y 坐标
                    uint8_t p = event.polarity ? 1 : 0;     // 极性（True -> 1, False -> 0）

                    // 计算时间戳（以微秒为单位）
                    // 第一个事件的时间戳设置为 t，后续事件的时间戳以 t 为参考
                    eventData.t = static_cast<uint32_t>(
                        t + ((event.ts.toNSec() - first_event_timestamp) / 1000) // 相对时间（微秒）
                    );

                    // 将 x, y, p 打包到 raw 中
                    eventData.raw = (p << 28) | (y << 14) | x;

                    // 写入二进制数据
                    dat_file.write(reinterpret_cast<const char*>(&eventData), sizeof(EventData));
                }
            }
        }
    }

    // 关闭文件
    dat_file.close();
    bag.close();

    ROS_INFO("Conversion completed successfully. Output saved to %s", output_dat_file.c_str());
    return 0;
}