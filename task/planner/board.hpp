#pragma once

#include <array>
#include <string>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "task/detector/KFS.hpp"

namespace planner
{

class Board
{
public:
  static constexpr int kRows = 4;
  static constexpr int kCols = 3;
  static constexpr int kEntryRow = 0;
  static constexpr int kEntryCol = 1;
  static constexpr int kOutsideEntryRow = -1;
  static constexpr int kMaxPickedT = 2;

  enum class DecisionCmd
  {
    LEFT,
    RIGHT,
    FORWARD,
    WAIT,

    // 搜索 WORK_T 阶段：目标在相邻格时，先抓取相邻格，不能走进被 KFS 占用的格子。
    PICK_LEFT,
    PICK_RIGHT,
    PICK_FORWARD,

    // 已携带 WORK_T 去出口阶段：T/R1 作为障碍清走，不计入 picked_t。
    CLEAR_LEFT,
    CLEAR_RIGHT,
    CLEAR_FORWARD
  };

  struct RobotGridState
  {
    // row = -1, col = 1 表示机器人在棋盘外，正对入口 (0, 1)。
    int row = kOutsideEntryRow;
    int col = kEntryCol;

    // 当前身上携带的 WORK_T 数量。
    // 本策略只会携带/抓取 WORK_T，不会携带 R1/F。
    int carried_t = 0;

    // 已经完成任务抓取的 WORK_T 数量，最多抓 2 个。
    int picked_t = 0;
  };

  struct Cell
  {
    bool enabled = true;
    bool occupied = false;
    bool visible = false;

    detector::KFS_workname workname = detector::WORK_UNKNOWN;
    detector::KFSname name = detector::UNKNOWN;

    Eigen::Vector2d xy_in_map{0.0, 0.0};

    int row = -1;
    int col = -1;

    // 候选观测，用于稳定帧确认。
    bool has_candidate = false;
    detector::KFS_workname candidate_workname = detector::WORK_UNKNOWN;
    detector::KFSname candidate_name = detector::UNKNOWN;
    Eigen::Vector2d candidate_xy_in_map{0.0, 0.0};
    int candidate_count = 0;
  };

public:
  Board(
    double x_min = 1200.0,
    double x_max = 4800.0,
    double y_min = 3200.0,
    double y_max = 8000.0);

  void clear();

  bool map_to_cell(
    const Eigen::Vector2d & xy_in_map,
    int & row,
    int & col) const;

  bool observe_kfs(const detector::KFS & kfs);
  bool clear_cell(int row, int col);

  const Cell & at(int row, int col) const;
  Cell & at(int row, int col);

  double cell_width() const;
  double cell_height() const;

  bool is_placeable_cell(int row, int col) const;

  // 第一行只允许记录 R1 和 T；F 不会出现在第一行，因此不作为入口特殊情况处理。
  bool is_placeable_kfs(
    int row,
    int col,
    detector::KFS_workname workname) const;

  bool should_pick_for_search(
    int row,
    int col,
    const RobotGridState & state) const;

  bool should_clear_for_exit(int row, int col) const;

  DecisionCmd decide_next_action(const RobotGridState & state) const;

  static std::string decision_to_string(DecisionCmd cmd);

  int stable_frame_threshold() const;

  cv::Mat render(
    int cell_px = 160,
    int margin_px = 20) const;

  void show(
    const std::string & window_name = "MEILIN_BOARD",
    int cell_px = 160,
    int margin_px = 20) const;

private:
  bool is_valid_index(int row, int col) const;
  bool is_exit_cell(int row, int col) const;

  bool is_t_cell(int row, int col) const;
  bool is_r1_cell(int row, int col) const;
  bool is_empty_cell(int row, int col) const;

  DecisionCmd pick_cmd_for_delta(int d_row, int d_col) const;
  DecisionCmd clear_cmd_for_delta(int d_row, int d_col) const;

  DecisionCmd search_step_or_pick(
    const RobotGridState & state,
    int to_row,
    int to_col,
    DecisionCmd move_cmd) const;

  DecisionCmd exit_step_or_clear(
    int from_row,
    int from_col,
    int to_row,
    int to_col,
    DecisionCmd move_cmd) const;

  DecisionCmd move_towards_col_for_search(
    const RobotGridState & state,
    int target_col) const;

  DecisionCmd safe_forward_or_pick_or_avoid_for_search(
    const RobotGridState & state) const;

  DecisionCmd move_forward_or_keep_middle_for_search(
    const RobotGridState & state) const;

  DecisionCmd safe_forward_or_clear_or_avoid_for_exit(
    int row,
    int col) const;

  DecisionCmd choose_exit_from_last_middle() const;

  DecisionCmd decide_move_to_exit(int row, int col) const;

  cv::Scalar cell_color(const Cell & cell) const;
  std::string cell_text(const Cell & cell) const;

  bool commit_kfs_to_cell(int row, int col, const detector::KFS & kfs);

private:
  double x_min_;
  double x_max_;
  double y_min_;
  double y_max_;

  double cell_w_;
  double cell_h_;

  std::array<std::array<Cell, kCols>, kRows> cells_;

  int stable_frame_threshold_ = 3;
};

}  // namespace planner