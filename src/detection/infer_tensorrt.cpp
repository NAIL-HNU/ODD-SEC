#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional> // std::function

using namespace nvinfer1;
#include <filesystem>
namespace fs = std::filesystem;

// Logger for TensorRT info/warning/errors
class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

// Detection struct
struct Detection {
    cv::Rect box;
    float score;
    int   class_id;
};

// Load engine file into a buffer
// std::vector<char> loadEngine(const std::string& file) {
//     std::ifstream in(file, std::ios::binary);
//     in.seekg(0, in.end);
//     size_t size = in.tellg(); in.seekg(0, in.beg);
//     std::vector<char> buf(size);
//     in.read(buf.data(), size);
//     return buf;
// }

std::vector<char> loadEngine(const std::string& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate); // Open at the end to get size
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open engine file: " << file << std::endl;
        return {}; // Return an empty vector or throw an exception
    }

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg); // Go back to the beginning to read

    if (size <= 0) { // Check for invalid or empty file size
         std::cerr << "Error: Engine file is empty or size is invalid: " << file << " (size: " << size << ")" << std::endl;
        return {};
    }

    // Optional: Add a sanity check for a maximum expected file size
    // const std::streamsize MAX_EXPECTED_ENGINE_SIZE = 1024 * 1024 * 500; // e.g., 500MB
    // if (size > MAX_EXPECTED_ENGINE_SIZE) {
    //     std::cerr << "Error: Engine file size (" << size << " bytes) is unexpectedly large for " << file << std::endl;
    //     return {};
    // }


    std::cout << "Attempting to load engine file: " << file << " with size: " << size << " bytes." << std::endl; // Debug output

    std::vector<char> buf(static_cast<size_t>(size)); // Explicitly cast to size_t
    if (!in.read(buf.data(), size)) {
        std::cerr << "Error: Failed to read engine file content: " << file << std::endl;
        return {}; // Return an empty vector or throw an exception
    }

    return buf;
}

// Compute IoU for NMS
float IoU(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

// Simple NMS
std::vector<Detection> doNMS(const std::vector<Detection>& dets, float iouThresh) {
    std::vector<Detection> ret;
    auto v = dets;
    std::sort(v.begin(), v.end(), [](auto& A, auto& B){ return A.score > B.score; });
    std::vector<bool> skip(v.size(), false);
    for (size_t i = 0; i < v.size(); ++i) {
        if (skip[i]) continue;
        ret.push_back(v[i]);
        for (size_t j = i+1; j < v.size(); ++j) {
            if (!skip[j] && v[i].class_id == v[j].class_id && IoU(v[i].box, v[j].box) > iouThresh)
                skip[j] = true;
        }
    }
    return ret;
}

int main(int argc, char** argv) {

    const std::string engineFile = "yolox_grayscale_v2_2_x86.engine";
    const std::string imgFolder  = "./low_data";
    const std::string outputFolder = "./output";
    const int   INPUT_W = 640, INPUT_H = 480, INPUT_C = 3;
    const int   OUTPUT_SIZE = 6300 * 7;
    const float CONF_THRESH = 0.1f, NMS_THRESH = 0.45f;

    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }

    // 遍历当前路径下的所有图片
    for (const auto& entry : fs::directory_iterator(imgFolder)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
            std::string imgFile = entry.path().string();
            std::cout << "Processing image: " << imgFile << std::endl;

            // 1. Preprocessing
            cv::Mat gray = cv::imread(imgFile, cv::IMREAD_GRAYSCALE);
            if (gray.empty()) {
                std::cerr << "Failed to read image\n"; return -1;
            }
            int origW = gray.cols, origH = gray.rows;
            cv::Mat rgb, resized;
            cv::cvtColor(gray, rgb, cv::COLOR_GRAY2BGR);
            cv::resize(rgb, resized, cv::Size(INPUT_W, INPUT_H));
            resized.convertTo(resized, CV_32F);

            // HWC -> CHW
            std::vector<float> input(INPUT_C * INPUT_H * INPUT_W);
            std::vector<cv::Mat> channels(INPUT_C);
            cv::split(resized, channels);
            for (int c = 0; c < INPUT_C; ++c) {
                memcpy(input.data() + c*INPUT_H*INPUT_W,
                    channels[c].data,
                    INPUT_H*INPUT_W*sizeof(float));
            }
            
            // 2. 加载 TensorRT 引擎
            auto engineData = loadEngine(engineFile);
            std::cout << "Engine file size: " << engineData.size() << " bytes\n";
            IRuntime* runtime = createInferRuntime(gLogger);    std::cout << "Here" << std::endl;
            ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
            if (!engine) {
                std::cerr << "Failed to deserialize CUDA engine\n";
                return -1;
            }
            IExecutionContext* context = engine->createExecutionContext();  std::cout << "Here" << std::endl;

    
            // 3. Malloc GPU cache
            void* buffers[2];
            size_t inBytes  = INPUT_C * INPUT_H * INPUT_W * sizeof(float);
            size_t outBytes = OUTPUT_SIZE * sizeof(float);
            if (cudaMalloc(&buffers[0], inBytes) != cudaSuccess || cudaMalloc(&buffers[1], outBytes) != cudaSuccess) {
                std::cerr << "Failed to allocate GPU memory\n";
                return -1;
            }

            // 4. Copy input to GPU
            cudaMemcpy(buffers[0], input.data(), inBytes, cudaMemcpyHostToDevice);

            // 5. inference
            cudaStream_t stream;
            cudaStreamCreate(&stream);

            const char* input_name  = engine->getIOTensorName(0);
            const char* output_name = engine->getIOTensorName(1);


            context->setTensorAddress(input_name, buffers[0]);
            context->setTensorAddress(output_name, buffers[1]);
            auto t0 = std::chrono::high_resolution_clock::now();

            context->enqueueV3(stream);
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
            auto t1 = std::chrono::high_resolution_clock::now();

            // 6. Copy output to CPU
            std::vector<float> output(OUTPUT_SIZE);
            cudaMemcpy(output.data(), buffers[1], outBytes, cudaMemcpyDeviceToHost);
            if (output.size() != OUTPUT_SIZE) {
                std::cerr << "Output size mismatch\n";
                return -1;
            }
            auto t2 = std::chrono::high_resolution_clock::now();

            float tInfer = std::chrono::duration<float, std::milli>(t1 - t0).count();
            float tCopy  = std::chrono::duration<float, std::milli>(t2 - t1).count();
            std::cout << "Inference time: " << tInfer << " ms, Copy time: " << tCopy << " ms\n";

            // 7. Postprocess
            std::vector<Detection> dets;
            int num = OUTPUT_SIZE / 7;
            for (int i = 0; i < num; ++i) {
                float x0 = output[i * 7 + 0] - output[i * 7 + 2] / 2;
                float y0 = output[i * 7 + 1] - output[i * 7 + 3] / 2;
                float x1 = output[i * 7 + 0] + output[i * 7 + 2] / 2;
                float y1 = output[i * 7 + 1] + output[i * 7 + 3] / 2;

                float obj_conf = output[i * 7 + 4];
                float cls_conf = output[i * 7 + 5];

                int   cid = static_cast<int>(output[i*7 + 6] + 0.5f);
                float score = obj_conf * cls_conf;
                if (score > CONF_THRESH) {
                    dets.push_back({cv::Rect(x0, y0, x1-x0, y1-y0), score, cid});
                }
            }
            auto finalDets = doNMS(dets, NMS_THRESH);

            // 8. 可视化：灰度图转 BGR，再画绿框
            cv::Mat vis; 
            cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
            for (auto& d : finalDets) {
                cv::rectangle(vis, d.box, cv::Scalar(0,255,0), 2);
                char buf[32];
                snprintf(buf, sizeof(buf), "drone: %.2f", d.score);
                cv::putText(vis, buf, d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
            }
            // 保存推理结果到输出文件夹
            std::string outFile = outputFolder + "/" + entry.path().filename().string();
            cv::imwrite(outFile, vis);
            std::cout << "Saved " << outFile << " with " << finalDets.size() << " detections\n";

            // 9. 清理
            cudaFree(buffers[0]);
            cudaFree(buffers[1]);
            delete context;
            delete engine;
            delete runtime;
        }
    }

    return 0;
}

