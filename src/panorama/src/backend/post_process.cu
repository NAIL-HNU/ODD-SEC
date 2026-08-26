#include <cuda_runtime.h>
#include "backend/post_process.h"

__global__ void find_single_target_kernel(
    const float* raw_output,
    float conf_thresh,
    float* best_box,
    float* best_score,
    int* found_detection
) {
    __shared__ float s_best_score;
    __shared__ float s_best_box[4];
    __shared__ int s_found;
    
    if (threadIdx.x == 0) {
        s_best_score = conf_thresh;
        s_found = 0;
    }
    __syncthreads();
    
    const int num_proposals = 6300;
    for (int i = threadIdx.x; i < num_proposals; i += blockDim.x) {
        const float* proposal = &raw_output[i * 6];
        
        float obj_conf = proposal[4];
        float cls_conf = proposal[5];
        float score = obj_conf * cls_conf;
        
        if (score > conf_thresh) {
            if (score > s_best_score) {
                atomicMax((int*)&s_best_score, __float_as_int(score));
                __threadfence();
                
                if (__float_as_int(s_best_score) == __float_as_int(score)) {
                    float x_center = proposal[0];
                    float y_center = proposal[1];
                    float width = proposal[2];
                    float height = proposal[3];
                    
                    s_best_box[0] = x_center - width / 2.0f;
                    s_best_box[1] = y_center - height / 2.0f;
                    s_best_box[2] = x_center + width / 2.0f;
                    s_best_box[3] = y_center + height / 2.0f;
                    
                    atomicExch(&s_found, 1);
                }
            }
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && s_found) {
        best_box[0] = s_best_box[0];
        best_box[1] = s_best_box[1];
        best_box[2] = s_best_box[2];
        best_box[3] = s_best_box[3];
        *best_score = __int_as_float(s_best_score);
        *found_detection = 1;
    } else if (threadIdx.x == 0) {
        *found_detection = 0;
    }
}

extern "C" {

    void launch_find_single_target(
        const float* raw_output, float conf_thresh,
        float* best_box, float* best_score,
        int* found_detection, cudaStream_t stream
    ) {
        find_single_target_kernel<<<1, 256, 0, stream>>>(
            raw_output, conf_thresh, best_box, best_score, found_detection
        );
    }

}