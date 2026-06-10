#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <chrono>

#include "realsense/realsense.hpp"   
#include "tools/logger.hpp"

int main() {
  // 1) 启动相机
  camera::RealSense::StreamProfile prof;
  camera::RealSense rs("8086:0b3a", prof);

  while (!rs.is_ready()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  std::cout << rs.get_camera_info_string() << std::endl;

  const auto info = rs.get_camera_info(); 

  cv::Mat color_bgr, depth_u16;

  while (true) {
    std::chrono::steady_clock::time_point ts;

    rs.read(color_bgr, depth_u16, ts);
    if (color_bgr.empty() || depth_u16.empty()) continue;

    int cx = color_bgr.cols / 2;
    int cy = color_bgr.rows / 2;
    cv::Point pt (cx,cy);

    float d_m = rs.get_point_distance(pt);
    uint16_t raw = depth_u16.at<uint16_t>(cy, cx);
    std::cout << raw << std::endl;
    tools::logger()->debug("RAW:",raw);

    cv::Mat color_show = color_bgr.clone();
    cv::circle(color_show, cv::Point(cx, cy), 4, cv::Scalar(0, 255, 0), -1);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3) << d_m << " m";
    cv::putText(color_show, "Center depth: " + oss.str(),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 255, 0), 2);

    cv::imshow("Color", color_show);
    // 按 ESC 退出
    int key = cv::waitKey(1);
    if (key == 27) break;
  }

  return 0;
}
