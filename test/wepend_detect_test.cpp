#include <iostream>
#include <string>
#include <chrono>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "usbcamera.hpp"
#include "task/detector/wepend.hpp"

const std::string keys =
    "{help h usage ? |                   | 输出命令行参数说明 }"
    "{config-path c  | configs/demo.yaml | yaml配置文件路径 }";


int main(int argc, char *argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string config_path = cli.get<std::string>("config-path");
    // 创建 USB 相机对象
    camera::USBCamera usbcam("video2", config_path);
    detector::Wepend wepend_detector(config_path, true);

    while (true)
    {
        cv::Mat img;
        std::chrono::steady_clock::time_point timestamp;

        // 读取图像
        usbcam.read(img, timestamp);

        if (img.empty()) {
            continue;
        }
        auto wepend = wepend_detector.detect(img);

        // 显示画面
        cv::imshow("camera", img);

        int key = cv::waitKey(1);

        // 退出
        if (key == 'q' || key == 27) {
            break;
        }
    }

    cv::destroyAllWindows();
    return 0;
}
