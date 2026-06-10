#include "KFS.hpp"
#include "realsense/realsense.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <Eigen/Eigenvalues>

namespace detector
{
namespace 
{
//功能：把图像上的一个像素点 (u, v) + 深度 z，转换成相机坐标系下的三维点 (x, y, z)
bool pixel_to_camera_3d(
  int u,
  int v,
  float z,
  const camera::RealSense::CameraInfo & info,
  Eigen::Vector3d & pt)
{
  if (!std::isfinite(z) || z <= 1e-6f) {
    return false;
  }

  const double fx = info.color_fx;
  const double fy = info.color_fy;
  const double cx = info.color_ppx;
  const double cy = info.color_ppy;

  if (fx <= 0.0 || fy <= 0.0) {
    return false;
  }

  const double x = (static_cast<double>(u) - cx) * static_cast<double>(z) / fx;
  const double y = (static_cast<double>(v) - cy) * static_cast<double>(z) / fy;

  pt = Eigen::Vector3d(x, y, static_cast<double>(z));
  return true;
}

} // namespace 

KFS_workname KFS::name_to_workname(KFSname name)
{
  switch (name) {
    case R_R1:
      return WORK_R_R1;

    case B_R1:
      return WORK_B_R1;

    case T_03:
    case T_04:
    case T_05:
    case T_06:
    case T_07:
    case T_08:
    case T_09:
    case T_10:
    case T_11:
    case T_12:
    case T_13:
    case T_14:
    case T_15:
    case T_16:
    case T_17:
      return WORK_T;

    case F_18:
    case F_19:
    case F_20:
    case F_21:
    case F_22:
    case F_23:
    case F_24:
    case F_25:
    case F_26:
    case F_27:
    case F_28:
    case F_29:
    case F_30:
    case F_31:
    case F_32:
      return WORK_F;

    default:
      return WORK_UNKNOWN;
  }
}
void KFS::clear_depth()
{
  depth_valid = false;
  surface_frame_valid = false;
  xyz_in_camera.setZero();
  xyz_in_base.setZero();
  ypr_in_camera.setZero();
  ypr_in_base.setZero();
  surface_frame = SurfaceFrame{};
}

bool KFS::update_xyz_in_camera(
  const cv::Mat& depth_u16,
  const camera::RealSense& rs,
  int radius)
{
  clear_depth();

  if (depth_u16.empty() || depth_u16.type() != CV_16U) {
    return false;
  }

  const auto info = rs.get_camera_info();
  if (info.depth_scale <= 0.0f) {
    return false;
  }

  const int u0 = static_cast<int>(std::round(center.x));
  const int v0 = static_cast<int>(std::round(center.y));

  if (u0 < 0 || v0 < 0 || u0 >= depth_u16.cols || v0 >= depth_u16.rows) {
    return false;
  }

  // 取中心附近窗口的有效深度，中值更稳
  std::vector<float> depths;
  depths.reserve((2 * radius + 1) * (2 * radius + 1));

  const int x_min = std::max(0, u0 - radius);
  const int x_max = std::min(depth_u16.cols - 1, u0 + radius);
  const int y_min = std::max(0, v0 - radius);
  const int y_max = std::min(depth_u16.rows - 1, v0 + radius);

  for (int v = y_min; v <= y_max; ++v) {
    for (int u = x_min; u <= x_max; ++u) {
      float z = camera::RealSense::get_point_distance(
        cv::Point(u, v), depth_u16, info.depth_scale);

      if (std::isfinite(z) && z > 0.01f) {
        depths.emplace_back(z);
      }
    }
  }

  if (depths.empty()) {
    return false;
  }

  std::nth_element(
    depths.begin(),
    depths.begin() + depths.size() / 2,
    depths.end());

  const float z = depths[depths.size() / 2];

  const double fx = info.color_fx;
  const double fy = info.color_fy;
  const double cx = info.color_ppx;
  const double cy = info.color_ppy;
  // std::cout << "fx: " << fx << ", fy: " << fy << ", cx: " << cx << ", cy: " << cy << std::endl;

  if (fx <= 0.0 || fy <= 0.0) {
    return false;
  }

  const double x = (static_cast<double>(u0) - cx) * z / fx;
  const double y = (static_cast<double>(v0) - cy) * z / fy;

  xyz_in_camera = Eigen::Vector3d(x, y, static_cast<double>(z));
  depth_valid = true;
  return true;
}

bool KFS::compute_surface_frame_pca(
  const cv::Mat & depth_u16,
  const camera::RealSense & rs,
  SurfaceFrame & frame,
  int radius) const
{
  frame = SurfaceFrame{};

  if (depth_u16.empty() || depth_u16.type() != CV_16U) {
    return false;
  }

  const auto info = rs.get_camera_info();
  if (info.depth_scale <= 0.0f) {
    return false;
  }

  const int u0 = static_cast<int>(std::round(center.x));
  const int v0 = static_cast<int>(std::round(center.y));

  if (u0 < 0 || v0 < 0 || u0 >= depth_u16.cols || v0 >= depth_u16.rows) {
    return false;
  }

  radius = std::max(1, radius);

  const int x_min = std::max(0, u0 - radius);
  const int x_max = std::min(depth_u16.cols - 1, u0 + radius);
  const int y_min = std::max(0, v0 - radius);
  const int y_max = std::min(depth_u16.rows - 1, v0 + radius);

  std::vector<Eigen::Vector3d> points;
  points.reserve((2 * radius + 1) * (2 * radius + 1));

  for (int v = y_min; v <= y_max; ++v) {
    for (int u = x_min; u <= x_max; ++u) {
      const float z = camera::RealSense::get_point_distance(
        cv::Point(u, v), depth_u16, info.depth_scale);

      if (!std::isfinite(z) || z <= 0.01f) {
        continue;
      }

      Eigen::Vector3d pt;
      if (!pixel_to_camera_3d(u, v, z, info, pt)) {
        continue;
      }

      points.emplace_back(pt);
    }
  }

  if (points.size() < 6) {
    return false;
  }

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const auto & p : points) {
    mean += p;
  }
  mean /= static_cast<double>(points.size());

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  for (const auto & p : points) {
    const Eigen::Vector3d d = p - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<double>(points.size());

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(cov);
  if (solver.info() != Eigen::Success) {
    return false;
  }

  const Eigen::Vector3d evals = solver.eigenvalues();

  // 经验性的平面质量检查
  const double lambda0 = evals(0);
  const double lambda1 = evals(1);
  const double lambda2 = evals(2);
  const double lambda_sum = lambda0 + lambda1 + lambda2;

  if (lambda_sum <= 1e-12 || lambda1 <= 1e-12 || lambda2 <= 1e-12) {
    return false;
  }

  // 点太不像平面时，拒绝
  if (lambda0 / lambda_sum > 0.08) {
    return false;
  }

  // 最小特征值方向 = 法向
  Eigen::Vector3d normal = solver.eigenvectors().col(0).normalized();
  Eigen::Vector3d axis_major = solver.eigenvectors().col(2).normalized();
  Eigen::Vector3d axis_minor = solver.eigenvectors().col(1).normalized();

  // 统一法向方向：尽量朝向相机
  if (normal.dot(-mean) < 0.0) {
    normal = -normal;
  }

  // 重新正交化
  axis_minor = normal.cross(axis_major).normalized();
  axis_major = axis_minor.cross(normal).normalized();

  frame.valid = true;
  frame.origin = mean;
  frame.x_axis = axis_major;
  frame.y_axis = axis_minor;
  frame.z_axis = normal;

  return true;
}

bool KFS::project_camera_point_to_pixel(
  const Eigen::Vector3d & pt_cam,
  const camera::RealSense & rs,
  cv::Point2f & uv) const
{
  const auto info = rs.get_camera_info();

  const double fx = info.color_fx;
  const double fy = info.color_fy;
  const double cx = info.color_ppx;
  const double cy = info.color_ppy;

  if (fx <= 0.0 || fy <= 0.0) {
    return false;
  }

  const double X = pt_cam.x();
  const double Y = pt_cam.y();
  const double Z = pt_cam.z();

  if (!std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z) || Z <= 1e-6) {
    return false;
  }

  const double u = fx * X / Z + cx;
  const double v = fy * Y / Z + cy;

  uv = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
  return true;
}

void KFS::draw_surface_axes(
  cv::Mat & image,
  const camera::RealSense & rs,
  float axis_len_m) const
{
  if (image.empty()) {
    return;
  }

  const cv::Point c(
    static_cast<int>(std::round(center.x)),
    static_cast<int>(std::round(center.y)));

  if (!surface_frame_valid || !surface_frame.valid) {
    cv::drawMarker(
      image, c, cv::Scalar(0, 0, 255),
      cv::MARKER_TILTED_CROSS, 20, 2);

    cv::putText(
      image, "surface invalid",
      c + cv::Point(6, -6),
      cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 255), 2);
    return;
  }

  const Eigen::Vector3d p0 = surface_frame.origin;
  const Eigen::Vector3d px = p0 + surface_frame.x_axis * axis_len_m;
  const Eigen::Vector3d py = p0 + surface_frame.y_axis * axis_len_m;
  const Eigen::Vector3d pz = p0 + surface_frame.z_axis * axis_len_m;

  cv::Point2f uv0, uvx, uvy, uvz;

  if (!project_camera_point_to_pixel(p0, rs, uv0)) {
    return;
  }

  const bool ok_x = project_camera_point_to_pixel(px, rs, uvx);
  const bool ok_y = project_camera_point_to_pixel(py, rs, uvy);
  const bool ok_z = project_camera_point_to_pixel(pz, rs, uvz);

  const cv::Point p_origin(
    static_cast<int>(std::round(uv0.x)),
    static_cast<int>(std::round(uv0.y)));

  cv::circle(image, p_origin, 4, cv::Scalar(255, 255, 255), -1);

  if (ok_x) {
    const cv::Point p_x(
      static_cast<int>(std::round(uvx.x)),
      static_cast<int>(std::round(uvx.y)));
    cv::arrowedLine(
      image, p_origin, p_x,
      cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.2);
    cv::putText(
      image, "Xs", p_x + cv::Point(4, -4),
      cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 0, 255), 2);
  }

  if (ok_y) {
    const cv::Point p_y(
      static_cast<int>(std::round(uvy.x)),
      static_cast<int>(std::round(uvy.y)));
    cv::arrowedLine(
      image, p_origin, p_y,
      cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.2);
    cv::putText(
      image, "Ys", p_y + cv::Point(4, -4),
      cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(0, 255, 0), 2);
  }

  if (ok_z) {
    const cv::Point p_z(
      static_cast<int>(std::round(uvz.x)),
      static_cast<int>(std::round(uvz.y)));
    cv::arrowedLine(
      image, p_origin, p_z,
      cv::Scalar(255, 0, 0), 2, cv::LINE_AA, 0, 0.2);
    cv::putText(
      image, "Zs", p_z + cv::Point(4, -4),
      cv::FONT_HERSHEY_SIMPLEX, 0.5,
      cv::Scalar(255, 0, 0), 2);
  }

  const std::string xyz_text = cv::format(
    "cam(%.3f, %.3f, %.3f)m",
    surface_frame.origin.x(),
    surface_frame.origin.y(),
    surface_frame.origin.z());

  cv::putText(
    image, xyz_text,
    p_origin + cv::Point(6, 18),
    cv::FONT_HERSHEY_SIMPLEX, 0.45,
    cv::Scalar(255, 255, 255), 2);
}

bool KFS::update_xyz_in_base(
  const Eigen::Matrix3d & R_base_camera,
  const Eigen::Vector3d & t_base_camera)
{
  if (!depth_valid) {
    xyz_in_base.setZero();
    return false;
  }

  // 基本合法性检查
  if (!R_base_camera.allFinite() || !t_base_camera.allFinite()) {
    xyz_in_base.setZero();
    return false;
  }

  xyz_in_base = R_base_camera * xyz_in_camera + t_base_camera;
  return true;
}

bool KFS::update_xyz_in_base(
  const Eigen::Matrix4d & T_base_camera)
{
  if (!depth_valid) {
    xyz_in_base.setZero();
    return false;
  }

  if (!T_base_camera.allFinite()) {
    xyz_in_base.setZero();
    return false;
  }

  Eigen::Vector4d p_cam_h;
  p_cam_h << xyz_in_camera, 1.0;

  const Eigen::Vector4d p_base_h = T_base_camera * p_cam_h;
  xyz_in_base = p_base_h.head<3>();

  return true;
}

static Eigen::Vector3d rotation_matrix_to_ypr(const Eigen::Matrix3d & R)
{
  // yaw-pitch-roll, ZYX
  const double yaw = std::atan2(R(1, 0), R(0, 0));
  const double pitch = std::atan2(-R(2, 0), std::sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
  const double roll = std::atan2(R(2, 1), R(2, 2));

  return Eigen::Vector3d(yaw, pitch, roll);
}

bool KFS::update_pose_in_base(
  const Eigen::Matrix3d & R_base_camera,
  const Eigen::Vector3d & t_base_camera)
{
  if (!update_xyz_in_base(R_base_camera, t_base_camera)) {
    ypr_in_base.setZero();
    return false;
  }

  if (!surface_frame_valid || !surface_frame.valid) {
    ypr_in_base.setZero();
    return true;
  }

  // surface_frame 当前是在 camera 坐标系下
  // 变到 base 坐标系
  Eigen::Matrix3d R_base_surface;
  R_base_surface.col(0) = R_base_camera * surface_frame.x_axis;
  R_base_surface.col(1) = R_base_camera * surface_frame.y_axis;
  R_base_surface.col(2) = R_base_camera * surface_frame.z_axis;

  ypr_in_base = rotation_matrix_to_ypr(R_base_surface);
  return true;
}

bool KFS::update_xy_in_map(
  const Eigen::Vector3d & robot_position,
  const Eigen::Quaterniond & robot_orientation)
{
  map_valid = false;
  xy_in_map.setZero();

  // 必须先有 base 坐标
  if (!depth_valid) {
    return false;
  }

  // 基本合法性检查
  if (!xyz_in_base.allFinite() || !robot_position.allFinite()) {
    return false;
  }

  if (!std::isfinite(robot_orientation.x()) ||
      !std::isfinite(robot_orientation.y()) ||
      !std::isfinite(robot_orientation.z()) ||
      !std::isfinite(robot_orientation.w())) {
    return false;
  }

  const double q_norm = robot_orientation.norm();
  if (q_norm <= 1e-12) {
    return false;
  }

  // 归一化四元数，避免串口数据轻微漂移
  const Eigen::Quaterniond q = robot_orientation.normalized();

  // base -> map
  const Eigen::Matrix3d R_map_base = q.toRotationMatrix();
  const Eigen::Vector3d p_map = R_map_base * xyz_in_base + robot_position;

  xy_in_map.x() = p_map.x();
  xy_in_map.y() = p_map.y();
  map_valid = true;
  return true;
}

}