#ifndef CAMERA__REALSENSE_HPP
#define CAMERA__REALSENSE_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <thread>

#include <librealsense2/rs.hpp>

#include "tools/thread_safe_queue.hpp"

namespace camera
{

class RealSense
{
public:
  struct StreamProfile
  {
    int color_width = 640;
    int color_height = 480;
    int color_fps = 30;

    int depth_width = 640;
    int depth_height = 480;
    int depth_fps = 30;

    // 是否将 depth 对齐到 color（建议 true）
    bool align_depth_to_color = true;
  };

  struct CameraInfo
  {
    std::string name;
    std::string serial;
    std::string firmware;
    std::string product_id;
    std::string product_line;
    std::string usb_type;

    float depth_scale = 0.0f;

    // color intrinsics内参
    int color_w = 0, color_h = 0;
    float color_fx = 0, color_fy = 0, color_ppx = 0, color_ppy = 0;

    // depth intrinsics内参
    int depth_w = 0, depth_h = 0;
    float depth_fx = 0, depth_fy = 0, depth_ppx = 0, depth_ppy = 0;
  };

  // vid_pid 例如 "8086:0b3a"
  explicit RealSense(const std::string & vid_pid);
   RealSense(const std::string & vid_pid, const StreamProfile & profile);

  ~RealSense();

  // depth_u16 的单位：raw depth unit，实际米 = depth_u16 * depth_scale
  void read(
    cv::Mat & color_bgr,
    cv::Mat & depth_u16,
    std::chrono::steady_clock::time_point & timestamp);

  // 获取相机信息（start 后会填充）
  CameraInfo get_camera_info() const;
  std::string get_camera_info_string() const;

  // 查询是否已采集到至少一帧（用于外部判断是否 ready）
  bool is_ready() const { return info_ready_.load(); }
  static float get_point_distance(const cv::Point& pt,const cv::Mat& depth_u16,float depth_scale);
  float get_point_distance(const cv::Point& pt);
  bool read_for(
  cv::Mat & color_bgr,
  cv::Mat & depth_u16,
  std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout);


private:
  struct FrameData
  {
    cv::Mat color_bgr;
    cv::Mat depth_u16;
    std::chrono::steady_clock::time_point timestamp;
  };

  std::string vid_pid_;
  int vid_ = -1;
  int pid_ = -1;

  StreamProfile profile_;

  // 守护线程：负责重启
  std::thread daemon_thread_;
  std::atomic<bool> daemon_quit_{false};

  // 采集线程
  std::thread capture_thread_;
  std::atomic<bool> capturing_{false};
  std::atomic<bool> capture_quit_{false};

  // 队列：只保留最新一帧（低延迟）
  tools::ThreadSafeQueue<FrameData, true> queue_;

  // RealSense objects
  std::unique_ptr<rs2::context> ctx_;
  std::unique_ptr<rs2::pipeline> pipe_;
  std::unique_ptr<rs2::config> cfg_;
  rs2::device device_;
  rs2::pipeline_profile pipe_profile_;

  // cache info
  std::atomic<bool> info_ready_{false};
  CameraInfo info_cache_;

private:
  void capture_start();
  void capture_stop();

  void parse_vid_pid(const std::string & vid_pid);
  rs2::device find_target_device(rs2::context & ctx) const;

  void fill_info_cache(const rs2::device & dev, const rs2::pipeline_profile & prof);
};

}  // namespace camera

#endif  // CAMERA__REALSENSE_HPP
