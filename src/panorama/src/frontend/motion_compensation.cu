#include <frontend/motion_compensation.cuh>



__global__ void motionCompensationKernel(const dvs_msgs::Event* event_buffer, const int64_t* event_timestamps, int* count_image,
                                        int event_count, int cam_width, int cam_height,
                                        float pixel_focus_ratio, float Focus, float pixel_size,
                                        int64_t t0, int threshold,
                                        float avg_angular_velocity_x, float avg_angular_velocity_y, float avg_angular_velocity_z) {

    // process event data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= event_count) return;

    float time_diff = float(event_timestamps[idx] - t0) / 1e9;

    // Compute rotation angular
    float x_angular = time_diff * avg_angular_velocity_x;
    float y_angular = time_diff * avg_angular_velocity_y;
    float z_angular = time_diff * avg_angular_velocity_z;

    int x = event_buffer[idx].x - cam_width / 2;
    int y = event_buffer[idx].y - cam_height / 2;

    // Compute initial angular
    float pre_x_angle = atanf(y * pixel_focus_ratio);
    float pre_y_angle = atanf(x * pixel_focus_ratio);

    // Compute coordinate after compensation
    int compen_x = int((x * __cosf(z_angular) - __sinf(z_angular) * y) -
                       (x - (Focus * __tanf(pre_y_angle + y_angular) / pixel_size)) + cam_width / 2);
    int compen_y = int((x * __sinf(z_angular) + __cosf(z_angular) * y) -
                       (y - (Focus * __tanf(pre_x_angle - x_angular) / pixel_size)) + cam_height / 2);

    // update count image
    if (compen_x < cam_height && compen_y >= 0 && compen_x < cam_width && compen_x >= 0) {
        atomicAdd(&count_image[compen_y * cam_width + compen_x], 1);
    }
}

void run_motion_compensation_cuda(const dvs_msgs::Event* h_events, int num_events,
                                const sensor_msgs::Imu* h_imu_buffer, int num_imu,
                                int cam_width, int cam_height,
                                float pixel_size, float focus,
                                int64_t t0,
                                int* h_count_image, int threshold) {
    // Allocate GPU Memory
    dvs_msgs::Event* d_events;
    int64_t* d_event_stamps;
    int* d_count_img;

    // Preprocess event timestamps
    int64_t* event_stamps = new int64_t[num_events]; // malloc memory
    preprocessEventTimestamps(h_events, num_events, event_stamps);

    cudaMalloc(&d_event_stamps, sizeof(int64_t) * num_events);
    cudaMemcpy(d_event_stamps, event_stamps, sizeof(int64_t) * num_events, cudaMemcpyHostToDevice);

    cudaMalloc(&d_events, sizeof(dvs_msgs::Event) * num_events);
    cudaMemcpy(d_events, h_events, sizeof(dvs_msgs::Event) * num_events, cudaMemcpyHostToDevice);

    cudaMalloc(&d_count_img, sizeof(int) * cam_height * cam_width);
    cudaMemset(d_count_img, 0, sizeof(int) * cam_height * cam_width);

    // Preprocess IMU data
    float avg_angular_velocity_x, avg_angular_velocity_y, avg_angular_velocity_z;
    preprocessImuData(h_imu_buffer, num_imu, t0, avg_angular_velocity_x, avg_angular_velocity_y, avg_angular_velocity_z);

    // Launch the CUDA kernel function
    int blockSize = 256;
    int gridSize = (num_events + blockSize - 1) / blockSize;
    motionCompensationKernel<<<gridSize, blockSize>>>(d_events, d_event_stamps, d_count_img, num_events, cam_width, cam_height,
                                                    pixel_size / focus, focus, pixel_size, t0, threshold, 
                                                    avg_angular_velocity_x, avg_angular_velocity_y, avg_angular_velocity_z);
    cudaDeviceSynchronize();

    // Send result back to CPU
    cudaMemcpy(h_count_image, d_count_img, sizeof(int) * cam_height * cam_width, cudaMemcpyDeviceToHost);

    // Release GPU memory
    cudaFree(d_events);
    cudaFree(d_count_img);
    cudaFree(d_event_stamps);
    delete[] event_stamps;
    event_stamps = nullptr;
    delete[] event_stamps;
}

void preprocessImuData(const sensor_msgs::Imu* imu_buffer, int imu_count, int64_t t0, 
                        float& avg_angular_velocity_x, float& avg_angular_velocity_y, float& avg_angular_velocity_z) {
    avg_angular_velocity_x = 0.0f;
    avg_angular_velocity_y = 0.0f;
    avg_angular_velocity_z = 0.0f;

    int valid_imu_count = 0;

    for (int i = 0; i < imu_count; i++){
        if (imu_buffer[i].header.stamp.toNSec() >= (t0 - 3000000)) {
            avg_angular_velocity_x += imu_buffer[i].angular_velocity.x;
            avg_angular_velocity_y += imu_buffer[i].angular_velocity.y;
            avg_angular_velocity_z += imu_buffer[i].angular_velocity.z;
            valid_imu_count++;
        }
    }

    if (valid_imu_count > 0) {
        avg_angular_velocity_x /= valid_imu_count;
        avg_angular_velocity_y /= valid_imu_count;
        avg_angular_velocity_z /= valid_imu_count;
    }
    
}

void preprocessEventTimestamps(const dvs_msgs::Event* h_events, int num_events, int64_t* event_timestamps) {
    for (int i = 0; i < num_events; ++i) {
        event_timestamps[i] = h_events[i].ts.toNSec();
    }
}