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
#include "task/detector/trt/yolo11_lstrt.hpp"
#include "task/detector/trt/yolo12_numtrt.hpp"
#include "task/detector/trt/wependtrt.hpp"
#include "task/detector/wepend_selector.hpp"
#include "task/detector/trt/lighttrt.hpp"
#include "tools/exiter.hpp"
#include "tools/logger.hpp"
#include "tools/thread_safe_queue.hpp"
#include "Com/Com.hpp"
#include "usbcamera/usbcamera.hpp"
#include "task/detector/wepend.hpp"
#include "Com/ROS/receive_ROS.hpp"
#include "task/planner/meilin.hpp"
#include <rclcpp/rclcpp.hpp>
#include "task/planner/board.hpp"


const std::string keys =
  "{help h usage ? |                   | 输出命令行参数说明 }"
  "{config-path c  | configs/demo.yaml | yaml配置文件的路径}";


struct SharedSendState
{
  Send_data data{};
  std::chrono::steady_clock::time_point kfs_ts;
  std::chrono::steady_clock::time_point wepend_ts;
};

struct CameraFrame
{
  cv::Mat color_bgr;
  cv::Mat depth_u16;
  std::chrono::steady_clock::time_point ts;
  uint64_t seq = 0;
};
tools::ThreadSafeQueue<CameraFrame, true> g_kfs_frame_queue(1);
tools::ThreadSafeQueue<CameraFrame, true> g_num_frame_queue(1);
tools::ThreadSafeQueue<CameraFrame, true> g_wepend_frame_queue(1);
tools::ThreadSafeQueue<CameraFrame, true> g_lightt_frame_queue(1);

std::mutex g_send_mtx;
SharedSendState g_send_state;
std::mutex g_receive_mtx;
Receive_data g_receive_state;
int MODE = 0;

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
  Eigen::Vector3d t0(-0.05, 0.24, 0.41);

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

void realsense_dispatch_loop(
  camera::RealSense & rs,
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

  uint64_t seq = 0;

  while (!stop_flag.load()) {
    CameraFrame frame;

    const bool ok = rs.read_for(
      frame.color_bgr,
      frame.depth_u16,
      frame.ts,
      std::chrono::milliseconds(50));

    if (!ok) {
      continue;
    }

    if (frame.color_bgr.empty() || frame.depth_u16.empty()) {
      continue;
    }

    frame.seq = ++seq;

    CameraFrame kfs_frame;
    kfs_frame.color_bgr = frame.color_bgr.clone();
    kfs_frame.depth_u16 = frame.depth_u16.clone();
    kfs_frame.ts = frame.ts;
    kfs_frame.seq = frame.seq;

    CameraFrame num_frame;
    num_frame.color_bgr = frame.color_bgr.clone();
    num_frame.depth_u16 = frame.depth_u16.clone();
    num_frame.ts = frame.ts;
    num_frame.seq = frame.seq;

    CameraFrame wepend_frame;
    wepend_frame.color_bgr = frame.color_bgr.clone();
    wepend_frame.depth_u16 = frame.depth_u16.clone();
    wepend_frame.ts = frame.ts;
    wepend_frame.seq = frame.seq;

    CameraFrame lightt_frame;
    lightt_frame.color_bgr = frame.color_bgr.clone();
    lightt_frame.depth_u16 = frame.depth_u16.clone();
    lightt_frame.ts = frame.ts;
    lightt_frame.seq = frame.seq;

    g_lightt_frame_queue.push(std::move(lightt_frame));
    g_kfs_frame_queue.push(std::move(kfs_frame));
    g_num_frame_queue.push(std::move(num_frame));
    g_wepend_frame_queue.push(std::move(wepend_frame));
  }
}


void kfs_detect(
  camera::RealSense & rs,
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{

  detector::yolo12_trt yolo_trt(config_path, true);
  detector::yolo11_lstrt yolo_ls_trt(config_path, true);
  int yaw , pitch=0;
  while (!stop_flag.load()) {
    CameraFrame frame;

    if (!g_kfs_frame_queue.pop_for(frame, std::chrono::milliseconds(50))) {
      continue;
    }

    cv::Mat color_bgr = frame.color_bgr;
    cv::Mat depth_u16 = frame.depth_u16;
    auto ts = frame.ts;

    if (color_bgr.empty() || depth_u16.empty()) {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.bool_getKFS = 0;
      continue;
    }

    {
        std::lock_guard<std::mutex> lock(g_receive_mtx);
        yaw = g_receive_state.yaw;
        pitch = g_receive_state.pitch;
        // std::cout <<"yaw"<<yaw<<std::endl;
        // std::cout<<"pitch:"<<pitch<<std::endl;
    }
    Eigen::Matrix4d T = make_T_base_camera(yaw, pitch);

    auto kfss = yolo_trt.detect(color_bgr);
    auto lo_shu_grid = yolo_ls_trt.detect(color_bgr);
    std::vector<detector::KFS> kfs_vec(kfss.begin(), kfss.end());

    bool found_valid_work_t = false;

    int selected_kfs_index = -1;
    double selected_kfs_z_mm = std::numeric_limits<double>::max();

    for (size_t i = 0; i < kfs_vec.size(); ++i) {
      auto & kfs = kfs_vec[i];

      if (kfs.update_xyz_in_camera(depth_u16, rs)) {
        kfs.update_xyz_in_base(T);

        const auto workname = detector::KFS::name_to_workname(kfs.name);

        const double x_mm = kfs.xyz_in_base.x() * 1000.0;
        const double y_mm = kfs.xyz_in_base.y() * 1000.0;
        const double z_mm = kfs.xyz_in_base.z() * 1000.0;

        // 不是 WORK_T，不参与发送选择
        if (workname != detector::WORK_T) {
          continue;
        }

        // 距离限制不满足，不参与发送选择
        if (x_mm + 150 >= 1000) {
          continue;
        }

        // 多个满足条件的 KFS，只选择高度最低的那个
        // 这里默认 base 坐标系下 z 越小，高度越低
        if (z_mm < selected_kfs_z_mm) {
          selected_kfs_z_mm = z_mm;
          selected_kfs_index = static_cast<int>(i);
          found_valid_work_t = true;
        }

        } else {
          std::cout
            << "[RS] name: " << detector::KFS_NAME[static_cast<int>(kfs.name)]
            << " | depth invalid"
            << std::endl;
        }
      }


    if(lo_shu_grid.has_value() ){
      auto grid = lo_shu_grid.value();
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
      std::cout<< "target_cell_index: "<< target_cell.index << std::endl;
      // 如果蓝色，就用 detector::loshu_BLUE

      grid.draw(
        color_bgr,
        true,   // draw_keypoints
        true,   // draw_cells
        true,   // draw_text
        has_target ? &target_cell : nullptr);  // target_cell
      cv::imshow("total", color_bgr);
      
      } else {
        //std::cout << "[yolo_trtls] lo_shu_grid not found" << std::endl;
      }

    // 统一发送最终选中的 KFS
    if (found_valid_work_t && selected_kfs_index >= 0) {
      const auto & selected_kfs = kfs_vec[selected_kfs_index];

      Receive_data receive_snapshot;
      {
        std::lock_guard<std::mutex> lock(g_receive_mtx);
        receive_snapshot = g_receive_state;
      }

      {
        std::lock_guard<std::mutex> lock(g_send_mtx);

        //g_send_state.data.workl_mode = receive_snapshot.workl_mode;
        //g_send_state.data.bool_graping = receive_snapshot.bool_graping;

        g_send_state.data.xyz_in_base[0] = selected_kfs.xyz_in_base.x() * 1000.0f;
        g_send_state.data.xyz_in_base[1] = selected_kfs.xyz_in_base.y() * 1000.0f;
        g_send_state.data.xyz_in_base[2] = selected_kfs.xyz_in_base.z() * 1000.0f;

        g_send_state.data.bool_getKFS = 1;
        g_send_state.kfs_ts = ts;
      }

      std::cout
        << "[KFS] selected lower WORK_T: index=" << selected_kfs_index
        << " xyz_in_base=("
        << selected_kfs.xyz_in_base.x() * 1000.0f << ", "
        << selected_kfs.xyz_in_base.y() * 1000.0f << ", "
        << selected_kfs.xyz_in_base.z() * 1000.0f << ") mm"
        << std::endl;

    } else {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.bool_getKFS = 0;
    }
  }
}   


void wepend_detect(
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  detector::WependTrt wepend_detector(config_path, true);

  WependTargetSelector selector;

  int last_targetid = 0;

  while (!stop_flag.load()) {
    CameraFrame frame;

    if (!g_wepend_frame_queue.pop_for(frame, std::chrono::milliseconds(50))) {
      continue;
    }

    if (frame.color_bgr.empty()) {
      continue;
    }

    auto wepends = wepend_detector.detect(frame.color_bgr);

    int targetid = selector.update(wepends);

    if (targetid > 0 && targetid != last_targetid) {
      std::cout << "[WEPEND] targetid: " << targetid << std::endl;

      {
        std::lock_guard<std::mutex> lock(g_send_mtx);

        // TODO: 这里发送给机器人
        // send_target_id(targetid);
      }

      last_targetid = targetid;
    }
  }
}


void num_detect(
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  detector::yolo12num_trt yolo_num_trt(config_path, true);

  while (!stop_flag.load()) {
    CameraFrame frame;

    if (!g_num_frame_queue.pop_for(frame, std::chrono::milliseconds(50))) {
      continue;
    }

    if (frame.color_bgr.empty()) {
      continue;
    }

    auto nums = yolo_num_trt.detect(frame.color_bgr);

    for (auto & num : nums) {
      std::cout << "num: " << num.name_str << std::endl;
    }
  }
}

void light_detect(
  const std::string & config_path,
  std::atomic<bool> & stop_flag)
{
  detector::LightTrt light_detector(config_path, true);

  while (!stop_flag.load()) {
    CameraFrame frame;

    if (!g_lightt_frame_queue.pop_for(frame, std::chrono::milliseconds(50))) {
      continue;
    }

    if (frame.color_bgr.empty()) {
      continue;
    }

    auto lights = light_detector.detect(frame.color_bgr);
    for (auto & light : lights) {
      std::cout << "light: " << detector::LIGHT_NAME[static_cast<int>(light.name)] << std::endl;
    }
  }
}

void send_loop(Com & com, std::atomic<bool> & stop_flag)
{
  using namespace std::chrono_literals;

  while (!stop_flag.load()) {
    Send_data local_data{};
    bool can_send = false;
    {
      std::lock_guard<std::mutex> lock(g_send_mtx);
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
  rclcpp::init(argc, argv);

  const auto config_path = cli.get<std::string>("config-path");
  //const auto competition_color = cli.get<std::string>("competition-color");

  tools::Exiter exiter;
  Com com(config_path);

  auto ros_node = std::make_shared<rclcpp::Node>("lidar_test_ros");
  rclcpp::QoS qos(10);
  qos.best_effort();

  ReceiveROS recv_ros(ros_node, "/send_ros", qos);

  std::atomic<bool> stop_flag(false);

  camera::RealSense::StreamProfile prof;
  camera::RealSense rs("8086:0b3a", prof);

  std::thread rs_dispatch_thread(realsense_dispatch_loop,std::ref(rs),std::ref(stop_flag));
  std::thread kfs_thread(kfs_detect, std::ref(rs), config_path, std::ref(stop_flag));
  std::thread num_thread(num_detect, config_path, std::ref(stop_flag));
  std::thread wepend_thread(wepend_detect, config_path, std::ref(stop_flag));
  std::thread light_thread(light_detect, config_path, std::ref(stop_flag));

  const int case_id = 1957;
  const auto & defence_path = planner::Meilin::defence(case_id);
  const auto commands =
  planner::Meilin::path_to_commands(planner::Meilin::Color::red, defence_path);

  {
    std::lock_guard<std::mutex> lock(g_send_mtx);
    g_send_state.data.workl_mode = 2;
    g_send_state.data.meilin_count = 0;

    for (int i = 0; i < 9; ++i) {
      g_send_state.data.meilin_points[i] = {0, 0, 0};
    }

    for (int i = 0; i < 9; ++i) {
      const auto & cmd = commands[i];

      if (cmd.cell == 0) {
        continue;
      }

      g_send_state.data.meilin_points[i].cell =
        static_cast<uint8_t>(cmd.cell);

      g_send_state.data.meilin_points[i].horizontal_s =
        static_cast<uint8_t>(cmd.horizontal);

      g_send_state.data.meilin_points[i].vertical_s =
        static_cast<uint8_t>(cmd.vertical);

      g_send_state.data.meilin_count++;
    }
    tools::logger()->info(
    "[Meilin] meilin_count: {}",
    static_cast<int>(g_send_state.data.meilin_count));

    for (int i = 0; i < 9; ++i) {
      const auto & p = g_send_state.data.meilin_points[i];

      tools::logger()->info(
        "[Meilin] point[{}]: cell={}, horizontal_s={}, vertical_s={}",
        i,
        static_cast<int>(p.cell),
        static_cast<int>(p.horizontal_s),
        static_cast<int>(p.vertical_s));
    }
  }

  std::thread send_thread(send_loop, std::ref(com), std::ref(stop_flag));

  while (!exiter.exit() && !stop_flag.load()) {
    {
      std::lock_guard<std::mutex> lock(g_receive_mtx);
      g_receive_state = com.receive();
    }
    rclcpp::spin_some(ros_node);
    auto ros_latest = recv_ros.latest();
    {
      std::lock_guard<std::mutex> lock(g_send_mtx);
      g_send_state.data.x_map = ros_latest.x;
      g_send_state.data.y_map = ros_latest.y;
      //tools::logger()->debug("[Com] x_map: {}, y_map: {}", ros_latest.x, ros_latest.y);
      const float gx_map = g_send_state.data.x_map;
      const float gy_map = g_send_state.data.y_map;

      //tools::logger()->debug("[Com] gx_map: {:.3f}, gy_map: {:.3f}", gx_map, gy_map);

    }
    //planner::Meilin::show_case_with_mode(case_id, planner::Meilin::Mode::defence, "defence", 100, 0, 0);
    int key = cv::waitKey(1);
    if (key == 'q') {
      stop_flag.store(true);
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  stop_flag.store(true);

  if (rs_dispatch_thread.joinable()) {
  rs_dispatch_thread.join();
  }

  if (kfs_thread.joinable()) {
    kfs_thread.join();
  }

  if (num_thread.joinable()) {
    num_thread.join();
  }

  if (wepend_thread.joinable()) {
    wepend_thread.join();
  }

  if (light_thread.joinable()) {
    light_thread.join();
  }

  if (send_thread.joinable()) send_thread.join();
  rclcpp::shutdown();
  cv::destroyAllWindows();
  return 0;
}
