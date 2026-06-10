#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "usbcamera.hpp"

namespace fs = std::filesystem;

const std::string keys =
    "{help h usage ? |                   | 输出命令行参数说明 }"
    "{config-path c  | configs/demo.yaml | yaml配置文件路径 }"
    "{save-dir s     | captured_video_usb| 视频保存目录 }"
    "{fps f          | 30                | 保存视频帧率 }";

cv::Mat undistortUSBImage(const cv::Mat& img)
{
    static cv::Mat K = (cv::Mat_<double>(3,3) <<
        463.287829, 0.0, 309.605339,
        0.0, 462.545221, 223.347301,
        0.0, 0.0, 1.0);

    static cv::Mat D = (cv::Mat_<double>(1,5) <<
        -0.398661, 0.148737, -0.004289, -0.002410, 0.0);

    static bool initialized = false;
    static cv::Mat map1, map2;
    static cv::Size cached_size;
    static cv::Rect roi;

    if (img.empty()) {
        return cv::Mat();
    }

    if (!initialized || img.size() != cached_size) {
        cached_size = img.size();

        cv::Mat newK = cv::getOptimalNewCameraMatrix(
            K, D, cached_size, 1.0, cached_size, &roi
        );

        cv::initUndistortRectifyMap(
            K, D, cv::Mat(), newK, cached_size,
            CV_16SC2, map1, map2
        );

        initialized = true;
    }

    cv::Mat undistorted;
    cv::remap(img, undistorted, map1, map2, cv::INTER_LINEAR);

    if (roi.width > 0 && roi.height > 0) {
        undistorted = undistorted(roi).clone();
    }

    return undistorted;
}

int main(int argc, char* argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string config_path = cli.get<std::string>("config-path");
    const std::string save_dir = cli.get<std::string>("save-dir");
    const int fps = cli.get<int>("fps");

    fs::create_directories(save_dir);

    camera::USBCamera usbcam("video2", config_path);

    cv::VideoWriter writer;
    bool recording = false;
    int frame_count = 0;

    std::string video_path = save_dir + "/usb_record.mp4";

    std::cout << "按 s 开始录制，按 q 或 ESC 结束录制并保存" << std::endl;

    while (true) {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;

        usbcam.read(img, timestamp);

        img = undistortUSBImage(img);

        if (img.empty()) {
            continue;
        }

        cv::Mat show = img.clone();

        int key = cv::waitKey(1);

        if (key == 's' && !recording) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

            writer.open(
                video_path,
                fourcc,
                fps,
                img.size(),
                true
            );

            if (!writer.isOpened()) {
                std::cerr << "视频文件打开失败: " << video_path << std::endl;
                break;
            }

            recording = true;
            frame_count = 0;

            std::cout << "开始录制: " << video_path << std::endl;
        }

        if (recording) {
            writer.write(img);
            frame_count++;

            cv::putText(show,
                        "REC",
                        cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::imshow("USB Camera", show);

        if (key == 'q' || key == 27) {
            break;
        }
    }

    if (writer.isOpened()) {
        writer.release();
    }

    std::cout << "录制结束，共保存 " << frame_count
              << " 帧，文件: " << video_path << std::endl;

    cv::destroyAllWindows();
    return 0;
}
