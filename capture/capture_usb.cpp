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
    "{save-dir s     | captured_images_usb   | 图片保存目录 }";

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include <opencv2/opencv.hpp>

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

    if (!initialized || img.size() != cached_size)
    {
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

int main(int argc, char *argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string config_path = cli.get<std::string>("config-path");
    const std::string save_dir = cli.get<std::string>("save-dir");

    // 创建保存目录
    fs::create_directories(save_dir);

    // 创建 USB 相机对象
    camera::USBCamera usbcam("video2", config_path);

    int count = 0;

    std::cout << "按住 's' 连续保存，按 'q' 或 ESC 退出" << std::endl;

    while (true)
    {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;

        // 读取图像
        usbcam.read(img, timestamp);

        img = undistortUSBImage(img);

        if (img.empty()) {
            continue;
        }

        // 显示画面
        cv::imshow("camera", img);

        int key = cv::waitKey(1);

        // 👇 核心：按住 s 连续保存
        if (key == 's') {
            std::string filename = save_dir + "/_" + std::to_string(count) + ".jpg";

            if (cv::imwrite(filename, img)) {
                std::cout << "\r已保存 " << count + 1 << " 张" << std::flush;
                count++;
            } else {
                std::cerr << "\n保存失败: " << filename << std::endl;
            }
        }

        // 退出
        if (key == 'q' || key == 27) {
            break;
        }
    }

    std::cout << "\n结束，累计保存 " << count << " 张图像" << std::endl;

    cv::destroyAllWindows();
    return 0;
}
