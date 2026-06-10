#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "realsense/realsense.hpp"
#include "task/detector/yolo12.hpp"
#include "task/detector/trt/yolo12_trt.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "Com/Com.hpp"
#include "usbcamera.hpp"
#include "task/detector/wepend.hpp"

const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}";


struct SharedSendState
{
  Send_data data{};

  bool has_wepend = false;

  std::chrono::steady_clock::time_point kfs_ts;
  std::chrono::steady_clock::time_point wepend_ts;
};

std::mutex g_send_mtx;
SharedSendState g_send_state;
std::mutex g_receive_mtx;
Receive_data g_receive_state;

Eigen::Matrix4d make_T_base_camera(double yaw_deg, double pitch_deg)
{
  const double yaw = yaw_deg * M_PI / 180.0;
  const double pitch = pitch_deg * M_PI / 180.0;

  // 相机在云台零位时 camera -> base 的旋转
  Eigen::Matrix3d R0;
  R0 <<  0,  0,  1,
        -1,  0,  0,
         0, -1,  0;

  // 相机/云台零位时，相机光心在 base 下的位置，单位 m
  Eigen::Vector3d t0(0.05, -0.1, 0.43);

  // 右手定则：yaw 绕 base Z 轴
  Eigen::Matrix3d R_yaw =
    Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();

  // 右手定则：pitch 绕 yaw 后的局部 Y 轴
  Eigen::Matrix3d R_pitch =
    Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()).toRotationMatrix();

  Eigen::Matrix3d R = R_yaw * R_pitch * R0;

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t0;

  return T;
}

void kfs_detect(
  camera::RealSense & rs,
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  while (!rs.is_ready() && !stop_flag.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  if (stop_flag.load()) return;

  std::cout << "[RealSense] " << rs.get_camera_info_string() << std::endl;

  detector::YOLO12 yolo(config_path, true);
  detector::yolo12_trt yolo_trt(config_path, true);
  cv::Mat color_bgr, depth_u16;
  Eigen::Matrix4d T;
    T <<  0,   0,   1,   0.05,
      -1,   0,   0,   -0.1,
        0,  -1,   0,  0.43,
        0,   0,   0,   1;
    
  // double yaw_deg = gimbal_yaw;
  // double pitch_deg = gimbal_pitch;

  // Eigen::Matrix4d T = make_T_base_camera(yaw_deg, pitch_deg);
  while (!stop_flag.load()) {
    std::chrono::steady_clock::time_point ts;
    rs.read(color_bgr, depth_u16, ts);
    if (color_bgr.empty() || depth_u16.empty()) {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.bool_getKFS = 0;
      continue;
    }
    bool found_valid_kfs = false;
    auto kfss = yolo_trt.detect(color_bgr);
    for (auto & kfs : kfss) {
      if (kfs.update_xyz_in_camera(depth_u16, rs)) {

        kfs.update_xyz_in_base(T);
        std::cout
          << "xyz_in_base: ("
          << kfs.xyz_in_base.x() * 1000.0f << ", "
          << kfs.xyz_in_base.y() * 1000.0f << ", "
          << kfs.xyz_in_base.z() * 1000.0f << ") mm"
          << std::endl;
        Receive_data receive_snapshot;
        {
          std::lock_guard<std::mutex> lock(g_receive_mtx);
          receive_snapshot = g_receive_state;
        }

        {
          std::lock_guard<std::mutex> lock(g_send_mtx);
          g_send_state.data.workl_mode = receive_snapshot.workl_mode;
          //g_send_state.data.bool_graping = receive_snapshot.bool_graping;
          //tools::logger()->debug("[Com] bool_graping: {}", static_cast<int>(receive_snapshot.bool_graping));
          g_send_state.data.xyz_in_base[0] = kfs.xyz_in_base.x() * 1000.0f;
          //tools::logger()->debug("[Com] xyz_in_base[0]: {}", static_cast<float>(g_send_state.data.xyz_in_base[0]));
          g_send_state.data.xyz_in_base[1] = kfs.xyz_in_base.y() * 1000.0f;
          g_send_state.data.xyz_in_base[2] = kfs.xyz_in_base.z() * 1000.0f;
          if(g_send_state.data.xyz_in_base[0] < 900){
            g_send_state.data.bool_getKFS = 1;
          }
          g_send_state.kfs_ts = ts;
        }
        found_valid_kfs = true;

      } else {

        std::cout
          << "[RS] name: " << detector::KFS_NAME[static_cast<int>(kfs.name)]
          << " | depth invalid"
          << std::endl;
      }
      
    }
    if (!found_valid_kfs ) {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.bool_getKFS = 0;
    }

    
  }
}

/*
void wepend_detect(
  camera::USBCamera & usbcam,
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  detector::Wepend wepend_detector(config_path, true);
  cv::Mat img;

  while (!stop_flag.load()) {
    std::chrono::steady_clock::time_point timestamp;
    usbcam.read(img, timestamp);
    if (img.empty()) {
      continue;
    }

    auto wepends = wepend_detector.detect(img);
    for (auto & wepend : wepends) {
      std::cout << "wepend_offset_x: " << wepend.offset_x << std::endl;

      {
        std::lock_guard<std::mutex> lock(g_send_mtx);
        g_send_state.data.wepend_offset_x = wepend.offset_x;
        g_send_state.has_wepend = true;
        g_send_state.wepend_ts = timestamp;
      }
    }

  }
}
*/
void send_loop(Com & com, std::atomic<bool> & stop_flag)
{
  using namespace std::chrono_literals;

  while (!stop_flag.load()) {
    Send_data local_data{};
    bool can_send = false;
    {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.i=1;
      local_data = g_send_state.data;
    }
    if(local_data.workl_mode == 0){
      can_send = true;
    }else if(local_data.workl_mode == 1){
      can_send = true;
    }else if(local_data.workl_mode == 2){
      can_send = true;
    }else{
      can_send = true;
    }

    if(can_send){
      com.send(local_data);
    }

    std::this_thread::sleep_for(2ms);  // 控制发送频率
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
  Com com(config_path);
  std::atomic<bool> stop_flag(false);

  camera::RealSense::StreamProfile prof;
  camera::RealSense rs("8086:0b3a", prof);
  //camera::USBCamera usbcam("video2", config_path);

  std::thread rs_thread(kfs_detect, std::ref(rs), config_path, std::ref(stop_flag));
  //std::thread usb_thread(wepend_detect, std::ref(usbcam), config_path, std::ref(stop_flag));

  // 独立发送线程
  std::thread send_thread(send_loop, std::ref(com), std::ref(stop_flag));

  while (!exiter.exit() && !stop_flag.load()) {
    {
      std::lock_guard<std::mutex> lock(g_receive_mtx);
      g_receive_state = com.receive();
    }
    int key = cv::waitKey(1);
    if (key == 'q') {
      stop_flag.store(true);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  stop_flag.store(true);

  if (rs_thread.joinable()) rs_thread.join();
  //if (usb_thread.joinable()) usb_thread.join();
  if (send_thread.joinable()) send_thread.join();

  cv::destroyAllWindows();
  return 0;
}
