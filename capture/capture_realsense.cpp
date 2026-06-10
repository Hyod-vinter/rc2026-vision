#include <opencv2/opencv.hpp>

#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <chrono>
#include <thread>
#include <filesystem>

#include "realsense/realsense.hpp"
#include "tools/logger.hpp"

namespace fs = std::filesystem;

const std::string keys =
    "{help h usage ? |                   | 输出命令行参数说明 }"
    "{save-dir s     | captured_images_rs | 图片保存目录 }"
    "{interval i     | 1                 | 抽帧间隔 }";

int main(int argc, char* argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);

    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string save_dir = cli.get<std::string>("save-dir");
    const int interval = cli.get<int>("interval");

    // 图片名前缀，可自行修改
    // 例如 "today"，保存结果为 today_00000001.jpg
    const std::string image_prefix = "oy603";

    fs::create_directories(save_dir);

    // 启动 RealSense
    camera::RealSense::StreamProfile prof;
    camera::RealSense rs("8086:0b3a", prof);

    while (!rs.is_ready()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << rs.get_camera_info_string() << std::endl;
    std::cout << "按住 's' 抽帧保存彩色图，按 'q' 或 ESC 退出" << std::endl;
    std::cout << "图片保存目录: " << save_dir << std::endl;
    std::cout << "图片名前缀: " << image_prefix << "_" << std::endl;

    cv::Mat color_bgr;
    cv::Mat depth_dummy;  // 仍需占位

    int frame_id = 0;
    int save_count = 0;

    while (true) {
        std::chrono::steady_clock::time_point ts;

        rs.read(color_bgr, depth_dummy, ts);

        if (color_bgr.empty()) {
            continue;
        }

        frame_id++;

        cv::imshow("RealSense Color", color_bgr);

        int key = cv::waitKey(1);

        // 按住 s 连续抽帧
        if (key == 's') {
            if (frame_id % interval == 0) {

                std::ostringstream name;

                // 文件名格式：
                // today_00000001.jpg
                name << image_prefix << "_"
                     << std::setw(8) << std::setfill('0') << save_count;

                std::string path = save_dir + "/" + name.str() + ".jpg";

                if (cv::imwrite(path, color_bgr)) {
                    std::cout << "\r已保存 " << save_count + 1
                              << " 张: " << path << std::flush;
                    save_count++;
                } else {
                    std::cerr << "\n保存失败: " << path << std::endl;
                }
            }
        }

        if (key == 'q' || key == 27) {
            break;
        }
    }

    std::cout << "\n结束，累计保存 " << save_count << " 张彩色图像" << std::endl;

    cv::destroyAllWindows();

    return 0;
}
