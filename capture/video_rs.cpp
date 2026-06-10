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
    "{save-dir s     | captured_video_rs1 | 视频保存目录 }"
    "{fps f          | 30                | 保存视频帧率 }";

int main(int argc, char* argv[])
{
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
        cli.printMessage();
        return 0;
    }

    const std::string save_dir = cli.get<std::string>("save-dir");
    const int fps = cli.get<int>("fps");

    fs::create_directories(save_dir);

    camera::RealSense::StreamProfile prof;
    camera::RealSense rs("8086:0b3a", prof);

    while (!rs.is_ready()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    std::cout << rs.get_camera_info_string() << std::endl;
    std::cout << "按 s 开始录制，按 q 或 ESC 结束录制并保存" << std::endl;

    cv::Mat color_bgr;
    cv::Mat depth_dummy;
    cv::VideoWriter writer;

    bool recording = false;
    int frame_count = 0;

    std::string video_path = save_dir + "/realsense_record6.mp4";

    while (true) {
        std::chrono::steady_clock::time_point ts;

        rs.read(color_bgr, depth_dummy, ts);

        if (color_bgr.empty()) {
            continue;
        }

        cv::Mat show = color_bgr.clone();

        int key = cv::waitKey(1);

        if (key == 's' && !recording) {
            int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');

            writer.open(
                video_path,
                fourcc,
                fps,
                color_bgr.size(),
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
            writer.write(color_bgr);
            frame_count++;

            cv::putText(show,
                        "REC",
                        cv::Point(20, 40),
                        cv::FONT_HERSHEY_SIMPLEX,
                        1.0,
                        cv::Scalar(0, 0, 255),
                        2);
        }

        cv::imshow("RealSense Color", show);

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
