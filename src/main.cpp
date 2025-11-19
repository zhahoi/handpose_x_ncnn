#include "yolov11.h"
#include "handpose.h"
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp>

std::mutex queueMutex;
std::condition_variable cond;
std::queue<cv::Mat> frameQueue;
bool finished = false;

std::unique_ptr<YOLOv11> yolov11(new YOLOv11());
std::unique_ptr<Handpose> handpose(new Handpose());

double fps = 0.0;

// ROI 扩展参数
constexpr float ROI_EXPAND_RATIO = 1.3f;
constexpr int MIN_ROI_SIZE = 128;
constexpr int MAX_ROI_SIZE = 320;

struct HandObject {
    cv::Rect_<float> rect;
    float prob;
    std::vector<cv::Point2f> kpts;
};

cv::Rect expandROI(const cv::Rect& rect, int img_w, int img_h) {
    float center_x = rect.x + rect.width / 2.f;
    float center_y = rect.y + rect.height / 2.f;
    int w = std::clamp(int(rect.width * ROI_EXPAND_RATIO), MIN_ROI_SIZE, MAX_ROI_SIZE);
    int h = std::clamp(int(rect.height * ROI_EXPAND_RATIO), MIN_ROI_SIZE, MAX_ROI_SIZE);
    int x = std::clamp(int(center_x - w / 2.f), 0, img_w - w);
    int y = std::clamp(int(center_y - h / 2.f), 0, img_h - h);
    return cv::Rect(x, y, w, h);
}

cv::Mat preprocessROI(const cv::Mat& roi) {
    if (roi.channels() == 1) {
        cv::Mat tmp;
        cv::cvtColor(roi, tmp, cv::COLOR_GRAY2BGR);
        return tmp;
    }
    return roi.clone();
}

void drawHandPose(cv::Mat& frame, const HandObject& handObj) {
    cv::rectangle(frame, handObj.rect, cv::Scalar(0, 255, 0), 2);
    std::string label = cv::format("Hand: %d%%", int(handObj.prob * 100));
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
    cv::rectangle(frame,
                  cv::Point(handObj.rect.x, handObj.rect.y - textSize.height - 4),
                  cv::Point(handObj.rect.x + textSize.width, handObj.rect.y),
                  cv::Scalar(0, 255, 0), -1);
    cv::putText(frame, label, cv::Point(handObj.rect.x, handObj.rect.y - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    static const int connections[20][2] = {
        {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},
        {0,9},{9,10},{10,11},{11,12},{0,13},{13,14},{14,15},{15,16},
        {0,17},{17,18},{18,19},{19,20}
    };
    static const cv::Scalar finger_colors[5] = {
        {0,0,255},{255,0,255},{0,255,255},{0,255,0},{255,0,0}
    };

    for (int k = 0; k < 21; ++k) {
        int radius = (k == 0) ? 6 : 4;
        cv::circle(frame, handObj.kpts[k], radius, cv::Scalar(0, 255, 255), -1);
        cv::circle(frame, handObj.kpts[k], radius + 1, cv::Scalar(0, 0, 0), 1);
    }
    for (int j = 0; j < 20; ++j) {
        int finger_idx = j / 4;
        cv::line(frame, handObj.kpts[connections[j][0]], handObj.kpts[connections[j][1]],
                 finger_colors[finger_idx], 2, cv::LINE_AA);
    }
}

void processFrame(cv::Mat& frame, const std::vector<Object>& detections) {
    std::vector<HandObject> handObjects;
    for (const auto& det : detections) {
        cv::Rect roi_rect = expandROI(det.rect, frame.cols, frame.rows);
        if (roi_rect.width < 64 || roi_rect.height < 64) continue;

        cv::Mat roi = preprocessROI(frame(roi_rect));
        Object_handpose handpose_obj;
        if (handpose->detect(roi, handpose_obj) != 0 || handpose_obj.kpts.size() != 21) continue;

        HandObject handObj;
        handObj.rect = det.rect;
        handObj.prob = det.prob;
        handObj.kpts.resize(21);
        for (int k = 0; k < 21; ++k) {
            handObj.kpts[k] = handpose_obj.kpts[k] + cv::Point2f(roi_rect.x, roi_rect.y);
        }
        handObjects.push_back(handObj);
    }

    for (const auto& handObj : handObjects)
        drawHandPose(frame, handObj);

    // 显示手部数量
    std::string info = cv::format("Hands: %zu", handObjects.size());
    cv::putText(frame, info, cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
}

// 图像推理函数
int processImage(const std::string& imagePath, const std::string& outputPath = "") {
    std::cout << "Processing image: " << imagePath << std::endl;
    
    // 读取图像
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Cannot read image from " << imagePath << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // YOLOv8 手部检测
    std::vector<Object> detections;
    double t0 = cv::getTickCount();
    yolov11->detect(image, detections);
    double t1 = cv::getTickCount();
    double yolo_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "YOLOv8 detection time: " << yolo_time << " ms" << std::endl;
    std::cout << "Detected " << detections.size() << " hand(s)" << std::endl;
    
    // 手部关键点检测和绘制
    t0 = cv::getTickCount();
    processFrame(image, detections);
    t1 = cv::getTickCount();
    double handpose_time = (t1 - t0) / cv::getTickFrequency() * 1000;
    
    std::cout << "Handpose inference time: " << handpose_time << " ms" << std::endl;
    std::cout << "Total inference time: " << (yolo_time + handpose_time) << " ms" << std::endl;
    
    // 添加总推理时间到图像
    std::string timeInfo = cv::format("Total: %.1f ms", yolo_time + handpose_time);
    cv::putText(image, timeInfo, cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    
    // 保存结果
    if (!outputPath.empty()) {
        if (cv::imwrite(outputPath, image)) {
            std::cout << "Output saved to: " << outputPath << std::endl;
        } else {
            std::cerr << "Error: Cannot save image to " << outputPath << std::endl;
        }
    }
    
    // 显示结果
    cv::namedWindow("Hand Detection + Pose Estimation", cv::WINDOW_NORMAL);
    cv::imshow("Hand Detection + Pose Estimation", image);
    
    std::cout << "\nPress any key to exit..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}

void videoODThread(const std::string& source, bool useCamera) {
    cv::VideoCapture cap;
    if (useCamera) {
        cap.open(0);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
    } else cap.open(source);

    if (!cap.isOpened()) {
        std::cerr << "Failed to open " << (useCamera ? "camera" : "video") << std::endl;
        finished = true; cond.notify_all(); return;
    }

    cv::Mat frame;
    std::vector<Object> detections;
    double t0 = cv::getTickCount();
    int localFrameCount = 0;

    while (cap.read(frame)) {
        detections.clear();
        yolov11->detect(frame, detections);
        processFrame(frame, detections);

        localFrameCount++;
        double t1 = cv::getTickCount();
        double elapsed = (t1 - t0) / cv::getTickFrequency();
        if (elapsed >= 1.0) { 
            fps = localFrameCount / elapsed; 
            localFrameCount = 0; 
            t0 = t1; 
        }

        // 显示FPS
        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

        {
            std::unique_lock<std::mutex> lock(queueMutex);
            frameQueue.push(frame.clone());
        }
        cond.notify_one();
    }

    finished = true;
    cond.notify_all();
}

// **判断文件类型**
bool isImageFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "jpg" || ext == "jpeg" || ext == "png" || 
            ext == "bmp" || ext == "tiff" || ext == "webp");
}

bool isVideoFile(const std::string& path) {
    std::string ext = path.substr(path.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return (ext == "mp4" || ext == "avi" || ext == "mov" || 
            ext == "mkv" || ext == "flv" || ext == "wmv");
}

void printUsage(const char* progName) {
    std::cout << "\nUsage:" << std::endl;
    std::cout << "  1. Camera mode:        " << progName << std::endl;
    std::cout << "  2. Image mode:         " << progName << " <image_path> [output_path]" << std::endl;
    std::cout << "  3. Video mode:         " << progName << " <video_path> [save_video]" << std::endl;
    std::cout << "\nExamples:" << std::endl;
    std::cout << "  " << progName << "                              # Use camera" << std::endl;
    std::cout << "  " << progName << " hand.jpg                     # Process image" << std::endl;
    std::cout << "  " << progName << " hand.jpg output.jpg          # Process and save image" << std::endl;
    std::cout << "  " << progName << " video.mp4                    # Process video" << std::endl;
    std::cout << "  " << progName << " video.mp4 1                  # Process and save video" << std::endl;
    std::cout << "\nSupported image formats: jpg, jpeg, png, bmp, tiff, webp" << std::endl;
    std::cout << "Supported video formats: mp4, avi, mov, mkv, flv, wmv" << std::endl;
    std::cout << "\nControls:" << std::endl;
    std::cout << "  ESC - Exit" << std::endl;
    std::cout << "  Any key - Next (image mode)" << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "=== Hand Detection + Pose Estimation ===" << std::endl;
    std::cout << "ROI Expand Ratio: " << ROI_EXPAND_RATIO << std::endl;
    std::cout << "Min ROI Size: " << MIN_ROI_SIZE << " pixels" << std::endl;
    std::cout << "Max ROI Size: " << MAX_ROI_SIZE << " pixels" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // **情况1：无参数 - 摄像头模式**
    if (argc == 1) {
        std::cout << "Mode: Camera" << std::endl;
        std::thread videoThread(videoODThread, "", true);
        
        cv::Mat frame;
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            cond.wait(lock, []{ return !frameQueue.empty() || finished; });
            if (!frameQueue.empty()) {
                frame = frameQueue.front(); 
                frameQueue.pop(); 
                lock.unlock();
                cv::imshow("Hand Detection + Pose Estimation", frame);
                if (cv::waitKey(1) == 27) break;
            } else if (finished) break;
        }
        
        videoThread.join();
        cv::destroyAllWindows();
        return 0;
    }

    // **情况2和3：有参数 - 判断是图像还是视频**
    std::string inputPath = argv[1];
    
    // 检查文件是否存在
    cv::Mat testRead = cv::imread(inputPath);
    cv::VideoCapture testVideo(inputPath);
    bool fileExists = (!testRead.empty() || testVideo.isOpened());
    testVideo.release();
    
    if (!fileExists) {
        std::cerr << "Error: File not found or cannot be opened: " << inputPath << std::endl;
        printUsage(argv[0]);
        return -1;
    }

    // **情况2：图像模式**
    if (isImageFile(inputPath)) {
        std::cout << "Mode: Image" << std::endl;
        std::string outputPath = (argc >= 3) ? argv[2] : "";
        return processImage(inputPath, outputPath);
    }
    
    // **情况3：视频模式**
    else if (isVideoFile(inputPath)) {
        std::cout << "Mode: Video" << std::endl;
        bool saveVideo = (argc >= 3) ? (std::stoi(argv[2]) == 1) : false;
        
        cv::VideoWriter videoWriter;
        if (saveVideo) {
            cv::VideoCapture tempCap(inputPath);
            int w = int(tempCap.get(cv::CAP_PROP_FRAME_WIDTH));
            int h = int(tempCap.get(cv::CAP_PROP_FRAME_HEIGHT));
            double vidFps = tempCap.get(cv::CAP_PROP_FPS);
            tempCap.release();
            
            std::string outputPath = "output_handpose.mp4";
            videoWriter.open(outputPath, cv::VideoWriter::fourcc('m','p','4','v'), vidFps, cv::Size(w,h));
            
            if (videoWriter.isOpened()) {
                std::cout << "Output will be saved to: " << outputPath << std::endl;
            } else {
                std::cerr << "Warning: Cannot open video writer" << std::endl;
            }
        }
        
        std::thread videoThread(videoODThread, inputPath, false);
        
        cv::Mat frame;
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex);
            cond.wait(lock, []{ return !frameQueue.empty() || finished; });
            if (!frameQueue.empty()) {
                frame = frameQueue.front(); 
                frameQueue.pop(); 
                lock.unlock();
                if (saveVideo && videoWriter.isOpened()) 
                    videoWriter.write(frame);
                cv::imshow("Hand Detection + Pose Estimation", frame);
                if (cv::waitKey(1) == 27) break;
            } else if (finished) break;
        }
        
        videoThread.join();
        if (videoWriter.isOpened()) {
            videoWriter.release();
            std::cout << "Video saved successfully!" << std::endl;
        }
        cv::destroyAllWindows();
        return 0;
    }
    
    // **未知文件类型**
    else {
        std::cerr << "Error: Unknown file type: " << inputPath << std::endl;
        printUsage(argv[0]);
        return -1;
    }
}