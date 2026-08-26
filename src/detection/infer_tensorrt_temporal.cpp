#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <functional>
#include <filesystem>

using namespace nvinfer1;
namespace fs = std::filesystem;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
} gLogger;

struct Detection {
    cv::Rect box;
    float score;
    int   class_id;
};

std::vector<char> loadEngine(const std::string& file) {
    std::ifstream in(file, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        std::cerr << "Error: Failed to open engine file: " << file << std::endl;
        return {};
    }

    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);

    if (size <= 0) {
        std::cerr << "Error: Engine file is empty or size is invalid: " << file << std::endl;
        return {};
    }

    std::vector<char> buf(static_cast<size_t>(size));
    if (!in.read(buf.data(), size)) {
        std::cerr << "Error: Failed to read engine file content: " << file << std::endl;
        return {};
    }

    return buf;
}

float IoU(const cv::Rect& a, const cv::Rect& b) {
    float inter = (a & b).area();
    float uni   = a.area() + b.area() - inter;
    return uni > 0 ? inter / uni : 0.f;
}

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
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_file> <input_folder> <output_folder>\n";
        return 1;
    }
    const std::string engineFile = argv[1];
    const std::string tenFolder = argv[2];
    const std::string outputFolder = argv[3];

    const int   INPUT_W = 480, INPUT_H = 640;
    const int   INPUT_C_SINGLE = 3;
    const int   INPUT_C_TEN = 10;
    const int   OUTPUT_SIZE = 6300 * 6;
    const float CONF_THRESH = 0.6f, NMS_THRESH = 0.45f;

    if (!fs::exists(outputFolder)) {
        fs::create_directory(outputFolder);
    }
    if (!fs::exists(tenFolder)) {
        std::cerr << "10-channel data folder not found: " << tenFolder << std::endl;
        return -1;
    }

    auto engineData = loadEngine(engineFile);
    if (engineData.empty()) {
        std::cerr << "Error: Failed to load engine file" << std::endl;
        return -1;
    }

    std::cout << "Engine file size: " << engineData.size() << " bytes\n";
    IRuntime* runtime = createInferRuntime(gLogger);
    ICudaEngine* engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine\n";
        return -1;
    }
    IExecutionContext* context = engine->createExecutionContext();

    
    // Engine contract: [1,3,H,W] image, [1,10,H,W] temporal input, then output.
    const char* input1Name = engine->getIOTensorName(0);
    const char* input2Name = engine->getIOTensorName(1);
    const char* outputName = engine->getIOTensorName(2);
    
    std::cout << "Input1 name: " << input1Name << "\n";
    std::cout << "Input2 name: " << input2Name << "\n";
    std::cout << "Output name: " << outputName << "\n";

    const size_t sizeInput1 = INPUT_C_SINGLE * INPUT_H * INPUT_W * sizeof(float);
    const size_t sizeInput2 = INPUT_C_TEN * INPUT_H * INPUT_W * sizeof(float);
    const size_t sizeOutput = OUTPUT_SIZE * sizeof(float);

    void* buffers[3];
    if (cudaMalloc(&buffers[0], sizeInput1) != cudaSuccess ||
        cudaMalloc(&buffers[1], sizeInput2) != cudaSuccess ||
        cudaMalloc(&buffers[2], sizeOutput) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory" << std::endl;
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    context->setTensorAddress(input1Name, buffers[0]);
    context->setTensorAddress(input2Name, buffers[1]);
    context->setTensorAddress(outputName, buffers[2]);

    context->setInputShape(input1Name, Dims4(1, INPUT_C_SINGLE, INPUT_H, INPUT_W));
    context->setInputShape(input2Name, Dims4(1, INPUT_C_TEN, INPUT_H, INPUT_W));

    for (const auto& entry : fs::directory_iterator(tenFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            std::string binFile = entry.path().string();
            std::cout << "Processing: " << binFile << std::endl;

            std::ifstream inFile(binFile, std::ios::binary);
            if (!inFile) {
                std::cerr << "Failed to open 10-channel data" << std::endl;
                continue;
            }

            const size_t tenDataSize = 10 * INPUT_H * INPUT_W * sizeof(float);
            inFile.seekg(0, std::ios::end);
            size_t fileSize = inFile.tellg();
            inFile.seekg(0, std::ios::beg);
            
            std::cout << "File size: " << fileSize << " bytes" << std::endl;
            std::cout << "Expected size: " << tenDataSize << " bytes" << std::endl;
            std::cout << "Expected dimensions: 10 x " << INPUT_H << " x " << INPUT_W << std::endl;
            
            if (fileSize != tenDataSize) {
                std::cerr << "Invalid 10-channel data size. Expected: " 
                          << tenDataSize << ", Got: " << fileSize << std::endl;
                continue;
            }

            std::vector<float> tenData(10 * INPUT_H * INPUT_W);
            inFile.read(reinterpret_cast<char*>(tenData.data()), fileSize);

            cv::Mat firstFrameDebug(INPUT_H, INPUT_W, CV_32F, tenData.data());
            cv::Mat firstFrameVis;
            cv::normalize(firstFrameDebug, firstFrameVis, 0, 255, cv::NORM_MINMAX, CV_8U);
            
            fs::path inputPath = entry.path();
            fs::path debugPath = fs::path(outputFolder) / (inputPath.stem().string() + "_first_frame_debug.png");
            cv::imwrite(debugPath.string(), firstFrameVis);
            std::cout << "Saved first frame debug image: " << debugPath.string() << std::endl;
            
            double minVal, maxVal;
            cv::minMaxLoc(firstFrameDebug, &minVal, &maxVal);
            std::cout << "First frame - min: " << minVal << ", max: " << maxVal << std::endl;

            std::cout << "Original data dimensions: " << INPUT_W << "x" << INPUT_H << std::endl;
            
            cv::Mat firstChannel(INPUT_H, INPUT_W, CV_32F, tenData.data());
            
            std::cout << "Channel dimensions: " << firstChannel.cols << "x" << firstChannel.rows << std::endl;

            cv::Mat bgrSingle;
            cv::cvtColor(firstChannel, bgrSingle, cv::COLOR_GRAY2BGR);

            std::vector<float> inputSingle(INPUT_C_SINGLE * INPUT_H * INPUT_W);
            std::vector<cv::Mat> channelsSingle(INPUT_C_SINGLE);
            cv::split(bgrSingle, channelsSingle);
            for (int c = 0; c < INPUT_C_SINGLE; ++c) {
                memcpy(inputSingle.data() + c * INPUT_H * INPUT_W,
                    channelsSingle[c].data,
                    INPUT_H * INPUT_W * sizeof(float));
            }

            std::vector<float> inputTen(tenData.begin(), tenData.end());

            std::cout << "Input1 data range: " << *std::min_element(inputSingle.begin(), inputSingle.end()) 
                    << " to " << *std::max_element(inputSingle.begin(), inputSingle.end()) << std::endl;
            std::cout << "Input2 data range: " << *std::min_element(inputTen.begin(), inputTen.end()) 
                    << " to " << *std::max_element(inputTen.begin(), inputTen.end()) << std::endl;

            cudaMemcpyAsync(buffers[0], inputSingle.data(), sizeInput1, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(buffers[1], inputTen.data(), sizeInput2, cudaMemcpyHostToDevice, stream);

            auto t0 = std::chrono::high_resolution_clock::now();
            bool success = context->enqueueV3(stream);
            if (!success) {
                std::cerr << "Inference failed for file: " << binFile << std::endl;
                continue;
            }
            cudaStreamSynchronize(stream);
            auto t1 = std::chrono::high_resolution_clock::now();

            std::vector<float> output(OUTPUT_SIZE);
            cudaMemcpyAsync(output.data(), buffers[2], sizeOutput, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            auto t2 = std::chrono::high_resolution_clock::now();

            std::cout << "Output data range: " << *std::min_element(output.begin(), output.end()) 
            << " to " << *std::max_element(output.begin(), output.end()) << std::endl;

            float tInfer = std::chrono::duration<float, std::milli>(t1 - t0).count();
            float tCopy  = std::chrono::duration<float, std::milli>(t2 - t1).count();
            std::cout << "Inference time: " << tInfer << " ms, Copy time: " << tCopy << " ms\n";

            if (tInfer < 0.1f) {
                std::cerr << "Warning: Inference time too short, possible failure for: " << binFile << std::endl;
                continue;
            }

            std::vector<Detection> dets;
            int num = OUTPUT_SIZE / 6;

            const int MIN_BOX_DIM = 5;
            const int MAX_CLASS_ID = 10;
            const float MIN_AREA_RATIO = 0.001f;
            const float MIN_ASPECT_RATIO = 0.2f;
            const float MAX_ASPECT_RATIO = 5.0f;

            for (int i = 0; i < num; ++i) {
                float x0 = output[i * 6 + 0] - output[i * 6 + 2] / 2;
                float y0 = output[i * 6 + 1] - output[i * 6 + 3] / 2;
                float x1 = output[i * 6 + 0] + output[i * 6 + 2] / 2;
                float y1 = output[i * 6 + 1] + output[i * 6 + 3] / 2;

                float obj_conf = output[i * 6 + 4];
                float cls_conf = output[i * 6 + 5];
                int   cid = 0; 
                float score = obj_conf * cls_conf;
                
                if (score > CONF_THRESH && score < 1.) {
                    int width = static_cast<int>(x1 - x0);
                    int height = static_cast<int>(y1 - y0);
                    
                    float area = width * height;
                    float image_area = INPUT_W * INPUT_H;
                    float aspect_ratio = (height > 0) ? static_cast<float>(width) / height : 0;

                    dets.push_back({cv::Rect(x0, y0, width, height), score, cid});
                
                }
            }
            std::cout << "dets.size(): " << dets.size() << std::endl;
            auto finalDets = doNMS(dets, NMS_THRESH);

            cv::Mat vis;
            cv::normalize(firstChannel, vis, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::cvtColor(vis, vis, cv::COLOR_GRAY2BGR);
            

            for (auto& d : finalDets) {
                cv::rectangle(vis, d.box, cv::Scalar(0,255,0), 2);
                char buf[32];
                snprintf(buf, sizeof(buf), "drone: %.2f", d.score);
                cv::putText(vis, buf, d.box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
            }
            
            fs::path outputPath = fs::path(outputFolder) / entry.path().filename();
            outputPath.replace_extension(".jpg");

            std::string outFile = outputPath.string();
            std::cout << outFile << std::endl;
            cv::imwrite(outFile, vis);
            std::cout << "Saved " << outFile << " with " << finalDets.size() << " detections\n";
            for (const auto& det : finalDets){
                std::cout << "Detection(box=["
                << det.box.x << ", " << det.box.y << ", "
                << det.box.width << ", " << det.box.height << "], "
                << "score=" << det.score << ", "
                << "class_id=" << det.class_id << ")" << std::endl;
            }

        }
    }

    cudaStreamDestroy(stream);
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    delete context;
    delete engine;
    delete runtime;

    return 0;
}
