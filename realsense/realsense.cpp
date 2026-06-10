#include "realsense.hpp"

#include <cctype>
#include <sstream>
#include <thread>

#include "tools/logger.hpp"
#include <limits> 

using namespace std::chrono_literals;

namespace camera
{

static std::string get_info_safe(const rs2::device & dev, rs2_camera_info what)
{
  try {
    if (dev.supports(what)) return dev.get_info(what);
  } catch (...) {
  }
  return "";
}

RealSense::RealSense(const std::string & vid_pid)
: RealSense(vid_pid, StreamProfile{})
{
  parse_vid_pid(vid_pid_);

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("RealSense daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      // 采集正常就不动
      if (capturing_) continue;

      // 采集异常：重启
      capture_stop();
      std::this_thread::sleep_for(200ms);
      capture_start();
    }

    capture_stop();
    tools::logger()->info("RealSense daemon thread stopped.");
  }};
}


RealSense::RealSense(const std::string & vid_pid, const StreamProfile & profile)
: vid_pid_(vid_pid), profile_(profile), queue_(1)
{
  parse_vid_pid(vid_pid_);

  daemon_thread_ = std::thread{[this] {
    tools::logger()->info("RealSense daemon thread started.");

    capture_start();

    while (!daemon_quit_) {
      std::this_thread::sleep_for(100ms);

      // 采集正常就不动
      if (capturing_) continue;

      // 采集异常：重启
      capture_stop();
      std::this_thread::sleep_for(200ms);
      capture_start();
    }

    capture_stop();
    tools::logger()->info("RealSense daemon thread stopped.");
  }};
}

RealSense::~RealSense()
{
  daemon_quit_ = true;
  if (daemon_thread_.joinable()) daemon_thread_.join();
  tools::logger()->info("RealSense destructed.");
}

void RealSense::read(
  cv::Mat & color_bgr,
  cv::Mat & depth_u16,
  std::chrono::steady_clock::time_point & timestamp)
{
  
  FrameData data;
  queue_.pop(data);
  color_bgr = data.color_bgr;
  depth_u16 = data.depth_u16;
  timestamp = data.timestamp;
}

RealSense::CameraInfo RealSense::get_camera_info() const
{
  return info_cache_;
}

std::string RealSense::get_camera_info_string() const
{
  std::ostringstream oss;
  const auto & i = info_cache_;

  oss << "RealSense Camera Info\n";
  oss << "  name        : " << i.name << "\n";
  oss << "  serial      : " << i.serial << "\n";
  oss << "  firmware    : " << i.firmware << "\n";
  oss << "  product_id  : " << i.product_id << "\n";
  oss << "  product_line: " << i.product_line << "\n";
  oss << "  usb_type    : " << i.usb_type << "\n";
  oss << "  depth_scale : " << i.depth_scale << " (meters per unit)\n";

  oss << "  color intrinsics: "
      << "w=" << i.color_w << " h=" << i.color_h
      << " fx=" << i.color_fx << " fy=" << i.color_fy
      << " ppx=" << i.color_ppx << " ppy=" << i.color_ppy << "\n";

  oss << "  depth intrinsics: "
      << "w=" << i.depth_w << " h=" << i.depth_h
      << " fx=" << i.depth_fx << " fy=" << i.depth_fy
      << " ppx=" << i.depth_ppx << " ppy=" << i.depth_ppy << "\n";

  return oss.str();
}

void RealSense::parse_vid_pid(const std::string & vid_pid)
{
  auto index = vid_pid.find(':');
  if (index == std::string::npos) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    return;
  }

  auto vid_str = vid_pid.substr(0, index);
  auto pid_str = vid_pid.substr(index + 1);

  try {
    vid_ = std::stoi(vid_str, 0, 16);
    pid_ = std::stoi(pid_str, 0, 16);
  } catch (...) {
    tools::logger()->warn("Invalid vid_pid: \"{}\"", vid_pid);
    vid_ = -1;
    pid_ = -1;
  }
}

rs2::device RealSense::find_target_device(rs2::context & ctx) const
{
  auto list = ctx.query_devices();
  if (list.size() == 0) throw std::runtime_error("No RealSense devices found.");

  // 优先按 PRODUCT_ID 匹配 pid（D435i 为 0B3A）
  std::string pid_hex;
  {
    std::ostringstream oss;
    oss << std::hex;
    oss.width(4);
    oss.fill('0');
    oss << (pid_ & 0xFFFF);
    pid_hex = oss.str();
  }

  for (auto dev : list) {
    auto prod_id = get_info_safe(dev, RS2_CAMERA_INFO_PRODUCT_ID);
    if (pid_ != -1 && !prod_id.empty()) {
      std::string a = prod_id, b = pid_hex;
      for (auto & c : a) c = static_cast<char>(std::tolower(c));
      for (auto & c : b) c = static_cast<char>(std::tolower(c));
      if (a == b) return dev;
    }
  }

  // fallback：名字里包含 d435
  for (auto dev : list) {
    auto name = get_info_safe(dev, RS2_CAMERA_INFO_NAME);
    std::string n = name;
    for (auto & c : n) c = static_cast<char>(std::tolower(c));
    if (n.find("d435") != std::string::npos) return dev;
  }

  return list.front();
}

void RealSense::fill_info_cache(const rs2::device & dev, const rs2::pipeline_profile & prof)
{
  CameraInfo info;
  info.name = get_info_safe(dev, RS2_CAMERA_INFO_NAME);
  info.serial = get_info_safe(dev, RS2_CAMERA_INFO_SERIAL_NUMBER);
  info.firmware = get_info_safe(dev, RS2_CAMERA_INFO_FIRMWARE_VERSION);
  info.product_id = get_info_safe(dev, RS2_CAMERA_INFO_PRODUCT_ID);
  info.product_line = get_info_safe(dev, RS2_CAMERA_INFO_PRODUCT_LINE);
  info.usb_type = get_info_safe(dev, RS2_CAMERA_INFO_USB_TYPE_DESCRIPTOR);

  try {
    if (auto ds = dev.first<rs2::depth_sensor>()) {
      info.depth_scale = ds.get_depth_scale();
    }
  } catch (...) {
  }

  try {
    auto depth_stream = prof.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto color_stream = prof.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

    auto di = depth_stream.get_intrinsics();
    info.depth_w = di.width; info.depth_h = di.height;
    info.depth_fx = di.fx; info.depth_fy = di.fy;
    info.depth_ppx = di.ppx; info.depth_ppy = di.ppy;

    auto ci = color_stream.get_intrinsics();
    info.color_w = ci.width; info.color_h = ci.height;
    info.color_fx = ci.fx; info.color_fy = ci.fy;
    info.color_ppx = ci.ppx; info.color_ppy = ci.ppy;
  } catch (...) {
  }

  info_cache_ = info;
  info_ready_ = true;
}

float RealSense::get_point_distance(
  const cv::Point& pt,
  const cv::Mat& depth_u16,
  float depth_scale)
{
  // depth_u16 必须是 CV_16U
  if (depth_u16.empty() || depth_u16.type() != CV_16U) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  // 越界检查
  if (pt.x < 0 || pt.y < 0 || pt.x >= depth_u16.cols || pt.y >= depth_u16.rows) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  // 取 raw 深度
  uint16_t d_raw = depth_u16.at<uint16_t>(pt.y, pt.x);

  // 0 通常表示无效深度（超量程/缺失）
  if (d_raw == 0 || depth_scale <= 0.0f) {
    return std::numeric_limits<float>::quiet_NaN();
  }

  // raw * scale = 米
  return static_cast<float>(d_raw) * depth_scale;
}

float RealSense::get_point_distance(const cv::Point& pt)
{
  cv::Mat color, depth;
  std::chrono::steady_clock::time_point ts;
  read(color, depth, ts);  // 取最新帧

  return RealSense::get_point_distance(pt, depth, info_cache_.depth_scale);
}

void RealSense::capture_start()
{
  capturing_ = false;
  capture_quit_ = false;
  info_ready_ = false;

  try {
    ctx_ = std::make_unique<rs2::context>();
    device_ = find_target_device(*ctx_);

    pipe_ = std::make_unique<rs2::pipeline>(*ctx_);
    cfg_ = std::make_unique<rs2::config>();

    // 更稳：用 serial 绑定设备
    auto serial = get_info_safe(device_, RS2_CAMERA_INFO_SERIAL_NUMBER);
    if (!serial.empty()) cfg_->enable_device(serial);

    cfg_->enable_stream(RS2_STREAM_COLOR, profile_.color_width, profile_.color_height,
                        RS2_FORMAT_RGB8, profile_.color_fps);
    cfg_->enable_stream(RS2_STREAM_DEPTH, profile_.depth_width, profile_.depth_height,
                        RS2_FORMAT_Z16, profile_.depth_fps);

    pipe_profile_ = pipe_->start(*cfg_);
    fill_info_cache(device_, pipe_profile_);
    
    for (auto && sensor : device_.query_sensors()) {
      std::string name;
      try {
        if (sensor.supports(RS2_CAMERA_INFO_NAME)) {
          name = sensor.get_info(RS2_CAMERA_INFO_NAME);
      }
    } catch (...) {
    }

    if (name.find("RGB") == std::string::npos &&
        name.find("Color") == std::string::npos) {
      continue;
    }

    tools::logger()->info("Set exposure on sensor: {}", name);

    if (sensor.supports(RS2_OPTION_ENABLE_AUTO_EXPOSURE)) {
      sensor.set_option(RS2_OPTION_ENABLE_AUTO_EXPOSURE, 0.0f);
    }

    if (sensor.supports(RS2_OPTION_EXPOSURE)) {
      sensor.set_option(RS2_OPTION_EXPOSURE, 150.0f);
    }

    if (sensor.supports(RS2_OPTION_GAIN)) {
      sensor.set_option(RS2_OPTION_GAIN, 64.0f);
    }
  }
  


    capture_thread_ = std::thread{[this] {
      tools::logger()->info("RealSense capture thread started.");
      capturing_ = true;

      // 若需要：对齐 depth 到 color
      rs2::align align_to_color(RS2_STREAM_COLOR);

      while (!capture_quit_) {
        try {
          // 取一帧 frameset —— 这是“严格同帧”的根本保证
          rs2::frameset frames = pipe_->wait_for_frames(1000);

          if (profile_.align_depth_to_color) {
            frames = align_to_color.process(frames);
          }

          rs2::video_frame color = frames.get_color_frame();
          rs2::depth_frame depth = frames.get_depth_frame();

          if (!color || !depth) continue;

          auto timestamp = std::chrono::steady_clock::now();

          // color: RGB8 -> OpenCV BGR8
          cv::Mat rgb(cv::Size(color.get_width(), color.get_height()),
                      CV_8UC3, (void *)color.get_data(), cv::Mat::AUTO_STEP);
          cv::Mat bgr;
          cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);

          // depth: Z16 -> CV_16U
          cv::Mat depth_u16(cv::Size(depth.get_width(), depth.get_height()),
                            CV_16U, (void *)depth.get_data(), cv::Mat::AUTO_STEP);

          // clone：避免 SDK buffer 生命周期问题
          queue_.push({bgr.clone(), depth_u16.clone(), timestamp});
        } catch (const rs2::error & e) {
          tools::logger()->warn("RealSense rs2::error: {} ({}:{})",
                                e.what(), e.get_failed_function(), e.get_failed_args());
          break;
        } catch (const std::exception & e) {
          tools::logger()->warn("RealSense exception: {}", e.what());
          break;
        }

        std::this_thread::sleep_for(1ms);
      }

      capturing_ = false;
      tools::logger()->info("RealSense capture thread stopped.");
    }};

    tools::logger()->info("RealSense started:\n{}", get_camera_info_string());
  } catch (const rs2::error & e) {
    tools::logger()->warn("RealSense start failed: {} ({}:{})",
                          e.what(), e.get_failed_function(), e.get_failed_args());
    capturing_ = false;
  } catch (const std::exception & e) {
    tools::logger()->warn("RealSense start failed: {}", e.what());
    capturing_ = false;
  }
}

void RealSense::capture_stop()
{
  capture_quit_ = true;
  if (capture_thread_.joinable()) capture_thread_.join();

  try {
    if (pipe_) pipe_->stop();
  } catch (...) {
  }

  pipe_.reset();
  cfg_.reset();
  ctx_.reset();
}

bool RealSense::read_for(
  cv::Mat & color_bgr,
  cv::Mat & depth_u16,
  std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  FrameData data;

  if (!queue_.pop_for(data, timeout)) {
    return false;
  }

  color_bgr = data.color_bgr;
  depth_u16 = data.depth_u16;
  timestamp = data.timestamp;

  return true;
}


}  // namespace camera
