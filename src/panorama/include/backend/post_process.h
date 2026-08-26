#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include <cuda_runtime.h>
#include <vector>
// #include "backend/drone_detector.h" // 包含Detection结构体的定义

struct GpuDetection {
    float box[4]; // x1, y1, x2, y2
    float score;
    int class_id;
};

// 使用 extern "C" 确保C++编译器能够正确链接到CUDA C编译器生成的函数
#ifdef __cplusplus
extern "C" {
#endif

    void launch_find_single_target(
        const float* raw_output, float conf_thresh,
        float* best_box, float* best_score,
        int* found_detection, cudaStream_t stream
    );

#ifdef __cplusplus
}
#endif

#endif // POST_PROCESS_H