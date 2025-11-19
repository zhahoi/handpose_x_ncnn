#ifndef YOLOv11_H
#define YOLOv11_H
#define NOMINMAX

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <net.h>

#define YOLOv11_PARAM_PATH "/home/dell/Code/c++/handpose_detect/weights/hand_pnnx.py.ncnn.param"
#define YOLOv11_BIN_PATH "/home/dell/Code/c++/handpose_detect/weights/hand_pnnx.py.ncnn.bin"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct GridAndStride
{
    int grid0;
    int grid1;
    int stride;
};

static const char* class_names[] = {
    "hand"
};

static cv::Scalar colors[] = {
    cv::Scalar(67,  54, 244),
    cv::Scalar(30,  99, 233),
    cv::Scalar(39, 176, 156),
    cv::Scalar(58, 183, 103),
    cv::Scalar(81, 181,  63),
    cv::Scalar(150, 243,  33),
    cv::Scalar(169, 244,   3),
    cv::Scalar(188, 212,   0),
    cv::Scalar(150, 136,   0),
    cv::Scalar(175,  80,  76),
    cv::Scalar(195,  74, 139),
    cv::Scalar(220,  57, 205),
    cv::Scalar(235,  59, 255),
    cv::Scalar(193,   7, 255),
    cv::Scalar(152,   0, 255),
    cv::Scalar(87,  34, 255),
    cv::Scalar(85,  72, 121),
    cv::Scalar(158, 158, 158),
    cv::Scalar(125, 139,  96)
};

class YOLOv11
{
public:
    YOLOv11();

    ~YOLOv11();

    int detect(const cv::Mat& rgb, std::vector<Object>& objects);

    int draw(cv::Mat& rgb, const std::vector<Object>& objects);

private:
    ncnn::Net yolov11;

    const int target_size = 640;
    const float mean_vals[3] = { 103.53f, 116.28f, 123.675f };
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    const float prob_threshold = 0.35f;
    const float nms_threshold = 0.5f;
    const bool use_gpu = false;

    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif // YOLOv11_H