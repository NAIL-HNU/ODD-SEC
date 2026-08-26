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

    ros::CallbackQueue imu_queue;
    ros::CallbackQueue events_queue;

    ros::NodeHandle imu_nh(nh);
    ros::NodeHandle events_nh(nh);
    imu_nh.setCallbackQueue(&imu_queue);
    events_nh.setCallbackQueue(&events_queue);

    panorama::Panorama pano(events_nh, imu_nh);

    ros::AsyncSpinner imu_spinner(1, &imu_queue);
    ros::AsyncSpinner events_spinner(4, &events_queue);

    imu_spinner.start();
    events_spinner.start();

    ros::waitForShutdown();
    return 0;
}
