#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>

#include <opencv2/opencv.hpp>
#include "realsense/realsense.hpp"
#include "task/detector/yolo12.hpp"
#include "task/detector/trt/yolo12_trt.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "Com/Com.hpp"
#include "usbcamera.hpp"
#include "task/detector/wepend.hpp"
#include "task/detector/yolo_ls.hpp"
#include "task/detector/trt/yolo11_lstrt.hpp"


const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path     | configs/demo.yaml | 检测模型配置文件路径 }";

void realsense_detect(
  camera::RealSense & rs,
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  while (!rs.is_ready() && !stop_flag.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  if (stop_flag.load()) {
    return;
  }

  std::cout << "[RealSense] "
            << rs.get_camera_info_string()
            << std::endl;

  detector::YOLO12 yolo_kfs(config_path, true);
  detector::yolo12_trt yolo_trt(config_path, true);
  detector::YOLOLS yolo_ls(config_path, true);
  detector::yolo11_lstrt yolo_ls_trt(config_path, true);

  cv::Mat color_bgr;
  cv::Mat depth_u16;

  Eigen::Matrix4d T;
  T <<  0,   0,   1,   0.108,
       -1,   0,   0,   0.03318,
        0,  -1,   0,  -0.048,
        0,   0,   0,   1;

  while (!stop_flag.load()) {
    std::chrono::steady_clock::time_point ts;

    rs.read(color_bgr, depth_u16, ts);

    if (color_bgr.empty() || depth_u16.empty()) {
      continue;
    }
    auto kfss_time = std::chrono::steady_clock::now();
    auto kfss = yolo_trt.detect(color_bgr);
    auto kfss_dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - kfss_time);
    std::cout << "kfss_dt: " << kfss_dt.count() << std::endl;
    auto ls_time = std::chrono::steady_clock::now();
    auto grid_opt = yolo_ls_trt.detect(color_bgr);
    auto ls_dt = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ls_time);
    std::cout << "ls_dt: " << ls_dt.count() << std::endl;

    std::vector<detector::KFS> kfs_vec(kfss.begin(), kfss.end());

    for (auto & kfs : kfs_vec) {
      if (kfs.update_xyz_in_camera(depth_u16, rs)) {
        kfs.update_xyz_in_base(T);

        // std::cout
        //   << "[KFS] "
        //   << detector::KFS_NAME[static_cast<int>(kfs.name)]
        //   << " xyz_in_base: ("
        //   << kfs.xyz_in_base.x() * 1000.0f << ", "
        //   << kfs.xyz_in_base.y() * 1000.0f << ", "
        //   << kfs.xyz_in_base.z() * 1000.0f << ") mm"
        //   << std::endl;
      } else {
        std::cout
          << "[KFS] "
          << detector::KFS_NAME[static_cast<int>(kfs.name)]
          << " depth invalid"
          << std::endl;
      }
    }

    cv::Mat show = color_bgr.clone();

    if (grid_opt.has_value()) {
      auto grid = grid_opt.value();

      grid.update_kfs_status(kfs_vec);

      const auto & cells = grid.get_cells();

      std::cout << "========== LoShuGrid ==========" << std::endl;

      for (const auto & cell : cells) {
        std::cout
          << "cell[" << cell.index << "] "
          << "row=" << cell.row
          << " col=" << cell.col
          << " has_kfs=" << cell.has_kfs;

        if (cell.has_kfs) {
          const auto & kfs = kfs_vec[cell.kfs_index];

          std::cout
            << " kfs_index=" << cell.kfs_index
            << " name="
            << detector::KFS_NAME[static_cast<int>(kfs.name)];
        }

        std::cout << std::endl;
      }
      detector::LoShuGridCell target_cell;
      const bool has_target = grid.select_target_cell(detector::loshu_RED, target_cell);
      // 如果你方是蓝色，就用 detector::loshu_BLUE

      grid.draw(
        show,
        true,   // draw_keypoints
        true,   // draw_cells
        true,   // draw_text
        has_target ? &target_cell : nullptr);  // target_cell
    } else {
      std::cout << "[YOLOLS] lo_shu_grid not found" << std::endl;
    }

    for (const auto & kfs : kfs_vec) {
      cv::rectangle(show, kfs.box, cv::Scalar(255, 0, 0), 2);
      cv::circle(show, kfs.center, 5, cv::Scalar(0, 0, 255), -1);

      cv::putText(
        show,
        detector::KFS_NAME[static_cast<int>(kfs.name)],
        kfs.center,
        cv::FONT_HERSHEY_SIMPLEX,
        0.8,
        cv::Scalar(255, 0, 0),
        2);
    }

    cv::resize(show, show, {}, 0.5, 0.5);
    cv::imshow("realsense_kfs_lo_shu_grid", show);

    int key = cv::waitKey(1);
    if (key == 'q') {
      stop_flag.store(true);
      break;
    }
  }
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);

  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  const auto config_path = cli.get<std::string>("config-path");

  tools::Exiter exiter;
  std::atomic<bool> stop_flag(false);

  camera::RealSense::StreamProfile prof;
  camera::RealSense rs("8086:0b3a", prof);

  std::thread rs_thread(
    realsense_detect,
    std::ref(rs),
    config_path,
    std::ref(stop_flag));

  while (!exiter.exit() && !stop_flag.load()) {
    int key = cv::waitKey(1);

    if (key == 'q') {
      stop_flag.store(true);
      break;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  stop_flag.store(true);

  if (rs_thread.joinable()) {
    rs_thread.join();
  }

  cv::destroyAllWindows();

  return 0;
}
