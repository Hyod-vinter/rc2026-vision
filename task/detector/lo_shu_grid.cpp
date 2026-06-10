#include "lo_shu_grid.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <cstdint>


namespace detector
{

namespace
{

cv::Point2f lerp(const cv::Point2f& a, const cv::Point2f& b, float t)
{
  return a * (1.0f - t) + b * t;
}

cv::Point2f polygon_center(const std::array<cv::Point2f, 4>& pts)
{
  cv::Point2f c(0.0f, 0.0f);
  for (const auto& p : pts) {
    c += p;
  }
  return c * 0.25f;
}

std::vector<cv::Point2f> to_vector(const std::array<cv::Point2f, 4>& pts)
{
  return std::vector<cv::Point2f>{pts[0], pts[1], pts[2], pts[3]};
}

}  // namespace

LoShuGrid::LoShuGrid()
{
  keypoints.fill(cv::Point2f(0.0f, 0.0f));
  keypoint_scores.fill(0.0f);
  keypoint_valid.fill(false);
}

bool LoShuGrid::update(
  const cv::Rect& box_,
  double confidence_,
  int class_id_,
  const std::vector<cv::Point2f>& kpts,
  const std::vector<float>& kpt_scores,
  float kpt_thresh)
{
  valid = false;
  cells.clear();

  if (kpts.size() < LO_SHU_KPT_NUM) {
    return false;
  }

  box = box_;
  confidence = confidence_;
  class_id = class_id_;

  for (int i = 0; i < LO_SHU_KPT_NUM; ++i) {
    keypoints[i] = kpts[i];

    if (!kpt_scores.empty() && i < static_cast<int>(kpt_scores.size())) {
      keypoint_scores[i] = kpt_scores[i];
      keypoint_valid[i] = kpt_scores[i] >= kpt_thresh;
    } else {
      keypoint_scores[i] = 1.0f;
      keypoint_valid[i] = true;
    }
  }

  return build_cells();
}

bool LoShuGrid::build_cells()
{
  // 必要点检查
  const int required[] = {T0, T1, T2, T3, R0, R1, B0, B1, L0, L1};

  for (int idx : required) {
    if (!keypoint_valid[idx]) {
      valid = false;
      return false;
    }
  }

  // 第 0 行：顶部 4 个点
  grid_points[0][0] = keypoints[T0];
  grid_points[0][1] = keypoints[T1];
  grid_points[0][2] = keypoints[T2];
  grid_points[0][3] = keypoints[T3];

  // 第 1 行：l1 到 r0
  grid_points[1][0] = keypoints[L1];
  grid_points[1][1] = lerp(keypoints[L1], keypoints[R0], 1.0f / 3.0f);
  grid_points[1][2] = lerp(keypoints[L1], keypoints[R0], 2.0f / 3.0f);
  grid_points[1][3] = keypoints[R0];

  // 第 2 行：l0 到 r1
  grid_points[2][0] = keypoints[L0];
  grid_points[2][1] = lerp(keypoints[L0], keypoints[R1], 1.0f / 3.0f);
  grid_points[2][2] = lerp(keypoints[L0], keypoints[R1], 2.0f / 3.0f);
  grid_points[2][3] = keypoints[R1];

  // 第 3 行：底部 b1 到 b0
  // 注意：你的标注里 b1 是左下，b0 是右下
  grid_points[3][0] = keypoints[B1];
  grid_points[3][1] = lerp(keypoints[B1], keypoints[B0], 1.0f / 3.0f);
  grid_points[3][2] = lerp(keypoints[B1], keypoints[B0], 2.0f / 3.0f);
  grid_points[3][3] = keypoints[B0];

  cells.clear();
  cells.reserve(9);

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      LoShuGridCell cell;
      cell.row = r;
      cell.col = c;
      cell.index = r * 3 + c;

      cell.corners = {
        grid_points[r][c],
        grid_points[r][c + 1],
        grid_points[r + 1][c + 1],
        grid_points[r + 1][c]
      };

      cell.center = polygon_center(cell.corners);

      cells.push_back(cell);
    }
  }

  valid = true;
  return true;
}

const std::vector<LoShuGridCell>& LoShuGrid::get_cells() const
{
  return cells;
}

int LoShuGrid::locate_point(const cv::Point2f& p) const
{
  if (!valid) {
    return -1;
  }

  for (const auto& cell : cells) {
    std::vector<cv::Point2f> poly = to_vector(cell.corners);

    double ret = cv::pointPolygonTest(poly, p, false);
    if (ret >= 0) {
      return cell.index;
    }
  }

  return -1;
}

void LoShuGrid::clear_kfs_status()
{
  for (auto& cell : cells) {
    cell.has_kfs = false;
    cell.kfs_color = NONE;
    cell.kfs_index = -1;
  }
}

bool LoShuGrid::update_kfs_status(const std::vector<KFS>& kfs_list)
{
  if (!valid) {
    return false;
  }

  clear_kfs_status();

  for (int i = 0; i < static_cast<int>(kfs_list.size()); ++i) {
    const int cell_id = locate_point(kfs_list[i].center);

    if (cell_id < 0 || cell_id >= static_cast<int>(cells.size())) {
      continue;
    }

    cells[cell_id].has_kfs = true;
    cells[cell_id].kfs_color = kfs_list[i].color == red ? loshu_RED : loshu_BLUE;
    cells[cell_id].kfs_index = i;
  }

  return true;
}

namespace
{

void draw_yellow_star(cv::Mat& image, const cv::Point2f& center, float radius)
{
  std::vector<cv::Point> points;
  points.reserve(10);

  const float inner_radius = radius * 0.45f;
  const float start_angle = -CV_PI / 2.0f;

  for (int i = 0; i < 10; ++i) {
    const float r = (i % 2 == 0) ? radius : inner_radius;
    const float angle = start_angle + i * CV_PI / 5.0f;

    points.emplace_back(
      static_cast<int>(std::round(center.x + r * std::cos(angle))),
      static_cast<int>(std::round(center.y + r * std::sin(angle))));
  }

  cv::fillConvexPoly(image, points, cv::Scalar(0, 255, 255));
  //cv::fillPoly(image, std::vector<std::vector<cv::Point>>{points}, cv::Scalar(0, 255, 255));
  cv::polylines(image, points, true, cv::Scalar(0, 180, 180), 1);
}

}  // namespace

void LoShuGrid::draw(
  cv::Mat& image,
  bool draw_keypoints,
  bool draw_cells,
  bool draw_text,
  const LoShuGridCell* target_cell) const
{
  if (image.empty()) {
    return;
  }

  cv::rectangle(image, box, cv::Scalar(255, 0, 255), 2);

  if (draw_cells && valid) {
    for (const auto& cell : cells) {
      std::vector<cv::Point> poly;
      for (const auto& p : cell.corners) {
        poly.emplace_back(
          static_cast<int>(std::round(p.x)),
          static_cast<int>(std::round(p.y)));
      }

      cv::Scalar border_color(0, 255, 0);
      cv::polylines(image, poly, true, border_color, 2);

      if (target_cell != nullptr && cell.index == target_cell->index) {
        draw_yellow_star(image, cell.center, 14.0f);
      }

      if (draw_text && cell.has_kfs) {
        std::string text;
        cv::Scalar text_color;

        if (cell.kfs_color == loshu_RED) {
          text = "红";
          text_color = cv::Scalar(0, 0, 255);
        } else if (cell.kfs_color == loshu_BLUE) {
          text = "蓝";
          text_color = cv::Scalar(255, 0, 0);
        } else {
          continue;
        }

        cv::putText(
          image,
          text,
          cell.center + cv::Point2f(-10.0f, 5.0f),
          cv::FONT_HERSHEY_SIMPLEX,
          0.8,
          text_color,
          2);
      }
    }
  }

  if (draw_keypoints) {
    for (int i = 0; i < LO_SHU_KPT_NUM; ++i) {
      if (!keypoint_valid[i]) {
        continue;
      }

      cv::circle(image, keypoints[i], 4, cv::Scalar(0, 255, 255), -1);

      if (draw_text) {
        cv::putText(
          image,
          LO_SHU_GRID_KPT_NAME[i],
          keypoints[i] + cv::Point2f(4.0f, -4.0f),
          cv::FONT_HERSHEY_SIMPLEX,
          0.45,
          cv::Scalar(0, 255, 255),
          1);
      }
    }
  }
}

namespace
{

constexpr int LUT_SIZE = 19683;  // 3^9

constexpr int EMPTY = 0;
constexpr int MY = 1;
constexpr int ENEMY = 2;

// 索引布局：
// 0 1 2  顶部第三排，难放，只允许胜利/阻挡
// 3 4 5  第二排，正常可放
// 6 7 8  第一排，禁止放
constexpr int WIN_LINES[5][3] = {
  {1, 4, 7},
  {0, 4, 8},
  {2, 4, 6},
  {0, 3, 6},
  {2, 5, 8}
};

constexpr bool is_top_row(int idx)
{
  return idx >= 0 && idx <= 2;
}

constexpr bool is_middle_row(int idx)
{
  return idx >= 3 && idx <= 5;
}

constexpr bool is_bottom_row(int idx)
{
  return idx >= 6 && idx <= 8;
}

constexpr bool can_place_idx(const int board[9], int idx, bool force)
{
  if (idx < 0 || idx >= 9) {
    return false;
  }

  if (board[idx] != EMPTY) {
    return false;
  }

  // 顶部最难放：只有直接胜利或阻挡时才放
  if (is_top_row(idx)) {
    return force;
  }

  // 中间排正常可放
  if (is_middle_row(idx)) {
    return true;
  }

  // 底部第一排禁止放
  if (is_bottom_row(idx)) {
    return false;
  }

  return false;
}

constexpr int find_two_one_empty(
  const int board[9],
  int target_state,
  bool force)
{
  for (int l = 0; l < 5; ++l) {
    int target_count = 0;
    int other_count = 0;
    int empty_count = 0;
    int empty_idx = -1;

    for (int k = 0; k < 3; ++k) {
      const int idx = WIN_LINES[l][k];

      if (board[idx] == EMPTY) {
        ++empty_count;
        empty_idx = idx;
      } else if (board[idx] == target_state) {
        ++target_count;
      } else {
        ++other_count;
      }
    }

    if (target_count == 2 &&
        other_count == 0 &&
        empty_count == 1 &&
        can_place_idx(board, empty_idx, force)) {
      return empty_idx;
    }
  }

  return -1;
}

constexpr int find_prepare_target(
  const int board[9],
  int target_state)
{
  for (int l = 0; l < 5; ++l) {
    int target_count = 0;
    int other_count = 0;
    int empty_count = 0;
    int empty_indices[2] = {-1, -1};

    for (int k = 0; k < 3; ++k) {
      const int idx = WIN_LINES[l][k];

      if (board[idx] == EMPTY) {
        if (empty_count < 2) {
          empty_indices[empty_count] = idx;
        }
        ++empty_count;
      } else if (board[idx] == target_state) {
        ++target_count;
      } else {
        ++other_count;
      }
    }

    // 准备阶段：只允许放中间排 3/4/5
    if (target_count == 1 &&
        other_count == 0 &&
        empty_count == 2) {
      for (int i = 0; i < 2; ++i) {
        const int idx = empty_indices[i];

        if (can_place_idx(board, idx, false)) {
          return idx;
        }
      }
    }
  }

  return -1;
}

constexpr int decide_target_idx_by_case(const int board[9])
{
  int idx = -1;

  // 1. 我方可直接胜利
  idx = find_two_one_empty(board, MY, true);
  if (idx >= 0) {
    return idx;
  }

  // 2. 阻挡对方直接胜利
  idx = find_two_one_empty(board, ENEMY, true);
  if (idx >= 0) {
    return idx;
  }

  // 3. 我方进攻准备
  idx = find_prepare_target(board, MY);
  if (idx >= 0) {
    return idx;
  }

  // 4. 对方防守准备
  idx = find_prepare_target(board, ENEMY);
  if (idx >= 0) {
    return idx;
  }

  // 5. 兜底：只放中间排，优先中间
  if (can_place_idx(board, 4, false)) {
    return 4;
  }

  if (can_place_idx(board, 3, false)) {
    return 3;
  }

  if (can_place_idx(board, 5, false)) {
    return 5;
  }

  return -1;
}

constexpr std::array<int8_t, LUT_SIZE> make_target_lut()
{
  std::array<int8_t, LUT_SIZE> lut{};

  for (int key = 0; key < LUT_SIZE; ++key) {
    int tmp = key;
    int board[9] = {};

    for (int i = 0; i < 9; ++i) {
      board[i] = tmp % 3;
      tmp /= 3;
    }

    lut[key] = static_cast<int8_t>(decide_target_idx_by_case(board));
  }

  return lut;
}

// 编译期生成完整 19683 查表
constexpr auto TARGET_LUT = make_target_lut();

}  // namespace


bool LoShuGrid::select_target_cell(
  color my_color,
  LoShuGridCell& target_cell) const
{
  if (!valid || cells.size() != 9 || my_color == NONE) {
    return false;
  }

  int key = 0;
  int base = 1;

  for (int i = 0; i < 9; ++i) {
    int state = EMPTY;

    if (cells[i].has_kfs) {
      state = cells[i].kfs_color == my_color ? MY : ENEMY;
    }

    key += state * base;
    base *= 3;
  }

  const int target_idx = TARGET_LUT[key];

  if (target_idx < 0 || target_idx >= 9) {
    return false;
  }

  target_cell = cells[target_idx];
  return true;
}


}  // namespace detector

