#ifndef POST_PROCESS_H
#define POST_PROCESS_H

#include <cuda_runtime.h>
#include <vector>

struct GpuDetection {
    float box[4];  // x1, y1, x2, y2
    float score;
    int class_id;
};

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

#endif
