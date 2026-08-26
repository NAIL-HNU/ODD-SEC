#include <iostream>
#include <Eigen/Dense>
int main() {
    std::cout << "Eigen version: " 
              << EIGEN_WORLD_VERSION << "."
              << EIGEN_MAJOR_VERSION << "."
              << EIGEN_MINOR_VERSION << std::endl;
    Eigen::Matrix3f m = Eigen::Matrix3f::Identity();
    std::cout << m << std::endl;
    return 0;
}