#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <glog/logging.h>
#include <gflags/gflags.h>
#include "panorama.h"


int main(int argc, char* argv[]) {
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();
    FLAGS_alsologtostderr = true;
    FLAGS_colorlogtostderr = true;

    ros::init(argc, argv, "panorama");
    ros::NodeHandle nh;

    // 创建两个独立的回调队列
    ros::CallbackQueue imu_queue;
    ros::CallbackQueue events_queue;
    // ros::CallbackQueue compensation_queue;

    // 为每个队列分配独立的 NodeHandle
    ros::NodeHandle imu_nh(nh);
    ros::NodeHandle events_nh(nh);
    // ros::NodeHandle compensation_nh(nh);
    imu_nh.setCallbackQueue(&imu_queue);
    events_nh.setCallbackQueue(&events_queue);
    // compensation_nh.setCallbackQueue(&compensation_queue);

    // 初始化 Panorama 类，并传入不同的 NodeHandle
    panorama::Panorama pano(events_nh, imu_nh);  // 需修改 Panorama 构造函数

    // 启动两个独立的 AsyncSpinner，分别处理不同队列
    ros::AsyncSpinner imu_spinner(1, &imu_queue);  // 1个线程处理IMU
    ros::AsyncSpinner events_spinner(4, &events_queue);  // 2个线程处理事件
    // ros::AsyncSpinner compensation_spinner(1, &compensation_queue);  // 2个线程处理事件

    imu_spinner.start();
    events_spinner.start();
    // compensation_spinner.start();

    ros::waitForShutdown();
    return 0;
}
