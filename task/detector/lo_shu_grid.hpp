#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <string>

#include "KFS.hpp"

namespace detector
{

enum color{loshu_RED,loshu_BLUE,NONE};

enum LoShuGridKeypointName
{
  T0 = 0,
  T1,
  T2,
  T3,
  R0,
  R1,
  B0,
  B1,
  L0,
  L1,
  LO_SHU_KPT_NUM
};

const std::vector<std::string> LO_SHU_GRID_KPT_NAME = {
  "t0", "t1", "t2", "t3",
  "r0", "r1",
  "b0", "b1",
  "l0", "l1"
};

struct LoShuGridCell
{
  int row{-1};
  int col{-1};
  int index{-1};

  std::array<cv::Point2f, 4> corners;
  cv::Point2f center;

  bool has_kfs{false};
  color kfs_color{NONE};
  int kfs_index{-1};
};

struct LoShuGrid
{
  int class_id{-1};
  double confidence{0.0};
  cv::Rect box;

  std::array<cv::Point2f, LO_SHU_KPT_NUM> keypoints;
  std::array<float, LO_SHU_KPT_NUM> keypoint_scores;
  std::array<bool, LO_SHU_KPT_NUM> keypoint_valid;

  bool valid{false};

  // 4 x 4 九宫格交点
  std::array<std::array<cv::Point2f, 4>, 4> grid_points;

  // 9 个格子
  std::vector<LoShuGridCell> cells;

  LoShuGridCell target_cell;

  LoShuGrid();

  bool update(
    const cv::Rect& box_,
    double confidence_,
    int class_id_,
    const std::vector<cv::Point2f>& kpts,
    const std::vector<float>& kpt_scores = {},
    float kpt_thresh = 0.25f);

  bool build_cells();

  const std::vector<LoShuGridCell>& get_cells() const;

  int locate_point(const cv::Point2f& p) const;

  bool update_kfs_status(const std::vector<KFS>& kfs_list);

  void clear_kfs_status();

  void draw(
  cv::Mat& image,
  bool draw_keypoints = true,
  bool draw_cells = true,
  bool draw_text = true,
  const LoShuGridCell* target_cell = nullptr) const;

  bool select_target_cell(color my_color, LoShuGridCell& target_cell) const;

};

}  // namespace detector

