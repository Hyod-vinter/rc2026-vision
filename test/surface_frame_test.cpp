#include <chrono>
#include <iostream>
#include <list>
#include <string>

#include <opencv2/opencv.hpp>
#include "realsense/realsense.hpp"
#include "task/detector/yolo12.hpp"
#include "task/detector/KFS.hpp"

int main(int argc, char ** argv)
{
  try {
    // =========================
    // 1. 参数配置
    // =========================
    // 这里按你的工程实际路径修改
    const std::string camera_vid_pid = "Intel RealSense";
    const std::string yolo_yaml = "config/kfs.yaml";

    // 是否显示深度伪彩
    const bool show_depth = true;

    // PCA窗口半径: 2 -> 5x5
    const int pca_radius = 2;

    // 中心点取深度窗口半径
    const int depth_radius = 2;

    // 面坐标轴长度，单位米
    const float axis_len_m = 0.05f;

    // =========================
    // 2. 初始化相机和检测器
    // =========================
    camera::RealSense rs(camera_vid_pid);
    detector::YOLO12 yolo(yolo_yaml);

    std::cout << "[INFO] RealSense + YOLO12 initialized." << std::endl;

    int frame_count = 0;

    while (true) {
      // =========================
      // 3. 读图像
      // =========================
      cv::Mat color_bgr;
      cv::Mat depth_u16;
      std::chrono::steady_clock::time_point timestamp;

      // 这里按你自己的 read 接口调整
      rs.read(color_bgr, depth_u16, timestamp);

      if (color_bgr.empty()) {
        std::cerr << "[WARN] color image is empty." << std::endl;
        continue;
      }

      // 调试显示图
      cv::Mat vis = color_bgr.clone();

      // =========================
      // 4. YOLO检测
      // =========================
      std::list<detector::KFS> kfss = yolo.detect(color_bgr, frame_count);

      // =========================
      // 5. 更新3D点 + PCA面坐标系
      // =========================
      for (auto & kfs : kfss) {
        // 5.1 中心点 3D
        bool ok_xyz = kfs.update_xyz_in_camera(depth_u16, rs, depth_radius);

        // 5.2 PCA 面坐标系
        detector::SurfaceFrame frame;
        bool ok_surface = false;

        if (ok_xyz) {
          ok_surface = kfs.compute_surface_frame_pca(depth_u16, rs, frame, pca_radius);
        }

        if (ok_surface) {
          kfs.surface_frame = frame;
          kfs.surface_frame_valid = true;
        } else {
          kfs.surface_frame_valid = false;
        }

        // =========================
        // 6. 画检测框、标签信息
        // =========================
        cv::rectangle(vis, kfs.box, cv::Scalar(0, 255, 255), 2);

        const std::string cls_name =
          (kfs.class_id >= 0 && kfs.class_id < static_cast<int>(detector::KFS_NAME.size()))
          ? detector::KFS_NAME[kfs.class_id]
          : "UNKNOWN";

        const std::string work_name =
          (static_cast<int>(kfs.workname) >= 0 &&
           static_cast<int>(kfs.workname) < static_cast<int>(detector::KFS_WORKNAME.size()))
          ? detector::KFS_WORKNAME[static_cast<int>(kfs.workname)]
          : "UNKNOWN";

        std::string text = cv::format(
          "%.2f %s %s",
          kfs.confidence,
          cls_name.c_str(),
          work_name.c_str());

        cv::Point text_org(kfs.box.x, std::max(20, kfs.box.y - 8));
        cv::putText(
          vis, text, text_org,
          cv::FONT_HERSHEY_SIMPLEX, 0.55,
          cv::Scalar(0, 255, 255), 2);

        // 画中心点
        cv::circle(
          vis,
          cv::Point(static_cast<int>(std::round(kfs.center.x)),
                    static_cast<int>(std::round(kfs.center.y))),
          4, cv::Scalar(255, 255, 255), -1);

        // =========================
        // 7. 画3D状态/面坐标轴
        // =========================
        if (kfs.depth_valid) {
          std::string xyz_text = cv::format(
            "P(%.3f, %.3f, %.3f)m",
            kfs.xyz_in_camera.x(),
            kfs.xyz_in_camera.y(),
            kfs.xyz_in_camera.z());

          cv::putText(
            vis, xyz_text,
            text_org + cv::Point(0, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(255, 255, 255), 2);
        } else {
          cv::putText(
            vis, "xyz invalid",
            text_org + cv::Point(0, 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.45,
            cv::Scalar(0, 0, 255), 2);
        }

        // 画 PCA 面坐标轴
        kfs.draw_surface_axes(vis, rs, axis_len_m);
      }

      // =========================
      // 8. 显示深度图
      // =========================
      if (show_depth && !depth_u16.empty()) {
        cv::Mat depth_8u;
        cv::Mat depth_color;

        double minv = 0.0, maxv = 0.0;
        cv::minMaxLoc(depth_u16, &minv, &maxv);

        if (maxv > 0.0) {
          depth_u16.convertTo(depth_8u, CV_8U, 255.0 / maxv);
        } else {
          depth_8u = cv::Mat::zeros(depth_u16.size(), CV_8U);
        }

        cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);
        cv::imshow("depth", depth_color);
      }

      // =========================
      // 9. 显示结果
      // =========================
      cv::imshow("kfs_surface_axes", vis);

      // 键盘控制
      int key = cv::waitKey(1);
      if (key == 27 || key == 'q' || key == 'Q') {
        break;
      }

      ++frame_count;
    }

    cv::destroyAllWindows();
  } catch (const std::exception & e) {
    std::cerr << "[ERROR] exception: " << e.what() << std::endl;
    return -1;
  } catch (...) {
    std::cerr << "[ERROR] unknown exception." << std::endl;
    return -1;
  }

  return 0;
}
