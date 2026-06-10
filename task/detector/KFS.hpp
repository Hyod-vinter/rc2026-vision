#pragma once
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace camera
{
class RealSense;
}

namespace detector
{

enum Colors
{
  red,
  blue
};
const std::vector<std::string> COLORS = {"red", "blue"};

enum KFSname
{
  R_R1,
  B_R1,
  T_03,
  T_04,
  T_05,
  T_06,
  T_07,
  T_08,
  T_09,
  T_10,
  T_11,
  T_12,
  T_13,
  T_14,
  T_15,
  T_16,
  T_17,
  F_18,
  F_19,
  F_20,
  F_21,
  F_22,
  F_23,
  F_24,
  F_25,
  F_26,
  F_27,
  F_28,
  F_29,
  F_30,
  F_31,
  F_32,
  UNKNOWN
};
const std::vector<std::string> KFS_NAME = {"R_R1",    "B_R1",     "T_03", "T_04",     "T_05",
                                              "T_06",    "T_07",     "T_08", "T_09",     "T_10",
                                              "T_11",    "T_12",     "T_13", "T_14",     "T_15",
                                              "T_16",    "T_17",     "F_18", "F_19",     "F_20",
                                              "F_21",    "F_22",     "F_23", "F_24",     "F_25",
                                              "F_26",    "F_27",     "F_28", "F_29",     "F_30",
                                              "F_31",    "F_32",     "UNKNOWN"};

enum KFS_workname
{
  WORK_R_R1,
  WORK_B_R1,
  WORK_T,
  WORK_F,
  WORK_UNKNOWN
};
const std::vector<std::string> KFS_WORKNAME = {
  "R_R1",
  "B_R1",
  "T",
  "F",
  "UNKNOWN"
};

struct SurfaceFrame
{
  bool valid{false};

  Eigen::Vector3d origin{Eigen::Vector3d::Zero()};
  Eigen::Vector3d x_axis{Eigen::Vector3d::Zero()};
  Eigen::Vector3d y_axis{Eigen::Vector3d::Zero()};
  Eigen::Vector3d z_axis{Eigen::Vector3d::Zero()};  // surface normal
};

struct KFS
{
  Colors color;
  KFSname name;
  KFS_workname workname; 
  cv::Point2f center;
  cv::Point2f center_norm;
  int class_id;
  double confidence;
  cv::Rect box;
  cv::Mat pattern;
  bool depth_valid{false};
  bool surface_frame_valid{false};
  // Eigen::Vector3d xyz_in_camera;  // 单位：m
  // Eigen::Vector3d xyz_in_base;   // 单位：m
  // Eigen::Vector3d ypr_in_camera;  // 单位：rad
  // Eigen::Vector3d ypr_in_base;   // 单位：rad
  Eigen::Vector3d xyz_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d xyz_in_base{Eigen::Vector3d::Zero()};

  Eigen::Vector3d ypr_in_camera{Eigen::Vector3d::Zero()};
  Eigen::Vector3d ypr_in_base{Eigen::Vector3d::Zero()};
  SurfaceFrame surface_frame;

  Eigen::Vector2d xy_in_map{0.0, 0.0};
  bool map_valid{false};

  static KFS_workname name_to_workname(KFSname name);
  void clear_depth();
  bool update_xyz_in_camera(const cv::Mat& depth_u16, const camera::RealSense& rs, int radius = 2);
   
  bool compute_surface_frame_pca(
    const cv::Mat & depth_u16,
    const camera::RealSense & rs,
    SurfaceFrame & frame,
    int radius = 2) const;

  bool project_camera_point_to_pixel(
    const Eigen::Vector3d & pt_cam,
    const camera::RealSense & rs,
    cv::Point2f & uv) const;

  void draw_surface_axes(
    cv::Mat & image,
    const camera::RealSense & rs,
    float axis_len_m = 0.05f) const;
  
  bool update_xyz_in_base(
  const Eigen::Matrix3d & R_base_camera,
  const Eigen::Vector3d & t_base_camera);

  bool update_xyz_in_base(
    const Eigen::Matrix4d & T_base_camera);

  bool update_pose_in_base(
    const Eigen::Matrix3d & R_base_camera,
    const Eigen::Vector3d & t_base_camera);

  bool update_xy_in_map(
  const Eigen::Vector3d & robot_position,
  const Eigen::Quaterniond & robot_orientation);
};

}  // namespace detector
