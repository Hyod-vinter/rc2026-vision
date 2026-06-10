#include "board.hpp"

#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace planner
{

Board::Board(
  double x_min,
  double x_max,
  double y_min,
  double y_max)
: x_min_(x_min),
  x_max_(x_max),
  y_min_(y_min),
  y_max_(y_max)
{
  if (x_max_ <= x_min_ || y_max_ <= y_min_) {
    throw std::runtime_error("Invalid board bounds.");
  }

  cell_w_ = (x_max_ - x_min_) / static_cast<double>(kCols);
  cell_h_ = (y_max_ - y_min_) / static_cast<double>(kRows);

  clear();
}

void Board::clear()
{
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      cells_[r][c] = Cell{};
      cells_[r][c].row = r;
      cells_[r][c].col = c;
      cells_[r][c].enabled = true;
    }
  }
}

bool Board::is_valid_index(int row, int col) const
{
  return row >= 0 && row < kRows && col >= 0 && col < kCols;
}

bool Board::is_exit_cell(int row, int col) const
{
  return row == kRows - 1 && (col == 0 || col == kCols - 1);
}

bool Board::is_placeable_cell(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  return cells_[row][col].enabled;
}

bool Board::is_placeable_kfs(
  int row,
  int col,
  detector::KFS_workname workname) const
{
  if (!is_placeable_cell(row, col)) {
    return false;
  }

  if (row != 0) {
    return true;
  }

  // 第一行只允许 R1 和 T。F 不可能出现在第一行，因此入口逻辑不再单独处理 F。
  switch (workname) {
    case detector::WORK_R_R1:
    case detector::WORK_B_R1:
    case detector::WORK_T:
      return true;

    case detector::WORK_F:
    case detector::WORK_UNKNOWN:
    default:
      return false;
  }
}

bool Board::map_to_cell(
  const Eigen::Vector2d & xy_in_map,
  int & row,
  int & col) const
{
  row = -1;
  col = -1;

  if (!xy_in_map.allFinite()) {
    return false;
  }

  const double x = xy_in_map.x();
  const double y = xy_in_map.y();

  // 左闭右开，避免 x == x_max_ 或 y == y_max_ 时算出越界索引。
  if (x < x_min_ || x >= x_max_ || y < y_min_ || y >= y_max_) {
    return false;
  }

  col = static_cast<int>((x - x_min_) / cell_w_);
  row = static_cast<int>((y - y_min_) / cell_h_);

  if (!is_valid_index(row, col)) {
    row = -1;
    col = -1;
    return false;
  }

  return true;
}

bool Board::commit_kfs_to_cell(int row, int col, const detector::KFS & kfs)
{
  if (!is_placeable_kfs(row, col, kfs.workname)) {
    return false;
  }

  auto & cell = cells_[row][col];
  cell.occupied = true;
  cell.visible = true;
  cell.workname = kfs.workname;
  cell.name = kfs.name;
  cell.xy_in_map = kfs.xy_in_map;
  cell.row = row;
  cell.col = col;

  return true;
}

bool Board::observe_kfs(const detector::KFS & kfs)
{
  if (!kfs.map_valid) {
    return false;
  }

  int row = -1;
  int col = -1;
  if (!map_to_cell(kfs.xy_in_map, row, col)) {
    return false;
  }

  if (!is_placeable_kfs(row, col, kfs.workname)) {
    return false;
  }

  auto & cell = cells_[row][col];

  if (!cell.has_candidate) {
    cell.has_candidate = true;
    cell.candidate_workname = kfs.workname;
    cell.candidate_name = kfs.name;
    cell.candidate_xy_in_map = kfs.xy_in_map;
    cell.candidate_count = 1;
    return false;
  }

  if (cell.candidate_workname == kfs.workname) {
    cell.candidate_count += 1;
    cell.candidate_name = kfs.name;
    cell.candidate_xy_in_map = kfs.xy_in_map;

    if (cell.candidate_count >= stable_frame_threshold_) {
      commit_kfs_to_cell(row, col, kfs);

      cell.has_candidate = false;
      cell.candidate_workname = detector::WORK_UNKNOWN;
      cell.candidate_name = detector::UNKNOWN;
      cell.candidate_xy_in_map.setZero();
      cell.candidate_count = 0;

      return true;
    }

    return false;
  }

  cell.has_candidate = true;
  cell.candidate_workname = kfs.workname;
  cell.candidate_name = kfs.name;
  cell.candidate_xy_in_map = kfs.xy_in_map;
  cell.candidate_count = 1;

  return false;
}

bool Board::clear_cell(int row, int col)
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  const bool enabled = cells_[row][col].enabled;

  cells_[row][col] = Cell{};
  cells_[row][col].row = row;
  cells_[row][col].col = col;
  cells_[row][col].enabled = enabled;

  return true;
}

const Board::Cell & Board::at(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    throw std::out_of_range("Invalid board index.");
  }

  return cells_[row][col];
}

Board::Cell & Board::at(int row, int col)
{
  if (!is_valid_index(row, col)) {
    throw std::out_of_range("Invalid board index.");
  }

  return cells_[row][col];
}

double Board::cell_width() const
{
  return cell_w_;
}

double Board::cell_height() const
{
  return cell_h_;
}

int Board::stable_frame_threshold() const
{
  return stable_frame_threshold_;
}

bool Board::is_t_cell(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  const auto & cell = cells_[row][col];
  return cell.occupied && cell.workname == detector::WORK_T;
}

bool Board::is_r1_cell(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  const auto & cell = cells_[row][col];
  return cell.occupied &&
         (cell.workname == detector::WORK_R_R1 ||
          cell.workname == detector::WORK_B_R1);
}

bool Board::is_empty_cell(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  return !cells_[row][col].occupied;
}

bool Board::should_pick_for_search(
  int row,
  int col,
  const RobotGridState & state) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  if (state.carried_t > 0) {
    return false;
  }

  if (state.picked_t >= kMaxPickedT) {
    return false;
  }

  return is_t_cell(row, col);
}

bool Board::should_clear_for_exit(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return false;
  }

  const auto & cell = cells_[row][col];
  if (!cell.occupied) {
    return false;
  }

  return cell.workname == detector::WORK_T ||
         cell.workname == detector::WORK_R_R1 ||
         cell.workname == detector::WORK_B_R1;
}

Board::DecisionCmd Board::pick_cmd_for_delta(int d_row, int d_col) const
{
  if (d_row == 1 && d_col == 0) {
    return DecisionCmd::PICK_FORWARD;
  }

  if (d_row == 0 && d_col == -1) {
    return DecisionCmd::PICK_LEFT;
  }

  if (d_row == 0 && d_col == 1) {
    return DecisionCmd::PICK_RIGHT;
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::clear_cmd_for_delta(int d_row, int d_col) const
{
  if (d_row == 1 && d_col == 0) {
    return DecisionCmd::CLEAR_FORWARD;
  }

  if (d_row == 0 && d_col == -1) {
    return DecisionCmd::CLEAR_LEFT;
  }

  if (d_row == 0 && d_col == 1) {
    return DecisionCmd::CLEAR_RIGHT;
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::search_step_or_pick(
  const RobotGridState & state,
  int to_row,
  int to_col,
  DecisionCmd move_cmd) const
{
  const bool from_outside_entry =
    state.row == kOutsideEntryRow && state.col == kEntryCol;

  if (!is_valid_index(state.row, state.col)) {
    if (!(from_outside_entry && to_row == kEntryRow && to_col == kEntryCol)) {
      return DecisionCmd::WAIT;
    }
  }

  if (!is_valid_index(to_row, to_col)) {
    return DecisionCmd::WAIT;
  }

  const auto & target = cells_[to_row][to_col];
  if (!target.occupied) {
    return move_cmd;
  }

  if (should_pick_for_search(to_row, to_col, state)) {
    return pick_cmd_for_delta(to_row - state.row, to_col - state.col);
  }

  // 搜索阶段只抓 WORK_T；R1 / F / UNKNOWN 都不压过去，也不主动清。
  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::exit_step_or_clear(
  int from_row,
  int from_col,
  int to_row,
  int to_col,
  DecisionCmd move_cmd) const
{
  if (!is_valid_index(to_row, to_col)) {
    return DecisionCmd::WAIT;
  }

  const auto & target = cells_[to_row][to_col];
  if (!target.occupied) {
    return move_cmd;
  }

  if (should_clear_for_exit(to_row, to_col)) {
    return clear_cmd_for_delta(to_row - from_row, to_col - from_col);
  }

  // 去出口阶段：F / UNKNOWN 不清，只能绕开或等待。
  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::move_towards_col_for_search(
  const RobotGridState & state,
  int target_col) const
{
  if (!is_valid_index(state.row, state.col)) {
    return DecisionCmd::WAIT;
  }

  if (target_col < 0 || target_col >= kCols) {
    return DecisionCmd::WAIT;
  }

  if (state.col < target_col) {
    const int next_col = state.col + 1;
    const DecisionCmd cmd =
      search_step_or_pick(state, state.row, next_col, DecisionCmd::RIGHT);

    if (cmd != DecisionCmd::WAIT) {
      return cmd;
    }

    return safe_forward_or_pick_or_avoid_for_search(state);
  }

  if (state.col > target_col) {
    const int next_col = state.col - 1;
    const DecisionCmd cmd =
      search_step_or_pick(state, state.row, next_col, DecisionCmd::LEFT);

    if (cmd != DecisionCmd::WAIT) {
      return cmd;
    }

    return safe_forward_or_pick_or_avoid_for_search(state);
  }

  return safe_forward_or_pick_or_avoid_for_search(state);
}

Board::DecisionCmd Board::safe_forward_or_pick_or_avoid_for_search(
  const RobotGridState & state) const
{
  if (!is_valid_index(state.row, state.col)) {
    return DecisionCmd::WAIT;
  }

  // 只有最后一行左右出口可以继续向前离开棋盘；最后一行中间不能直接出去。
  if (state.row == kRows - 1) {
    if (is_exit_cell(state.row, state.col)) {
      return DecisionCmd::FORWARD;
    }

    return choose_exit_from_last_middle();
  }

  const int next_row = state.row + 1;

  const DecisionCmd forward_cmd =
    search_step_or_pick(state, next_row, state.col, DecisionCmd::FORWARD);
  if (forward_cmd != DecisionCmd::WAIT) {
    return forward_cmd;
  }

  if (state.col > 0) {
    const DecisionCmd left_cmd =
      search_step_or_pick(state, state.row, state.col - 1, DecisionCmd::LEFT);
    if (left_cmd != DecisionCmd::WAIT) {
      return left_cmd;
    }
  }

  if (state.col < kCols - 1) {
    const DecisionCmd right_cmd =
      search_step_or_pick(state, state.row, state.col + 1, DecisionCmd::RIGHT);
    if (right_cmd != DecisionCmd::WAIT) {
      return right_cmd;
    }
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::move_forward_or_keep_middle_for_search(
  const RobotGridState & state) const
{
  if (!is_valid_index(state.row, state.col)) {
    return DecisionCmd::WAIT;
  }

  if (state.col < kEntryCol) {
    return move_towards_col_for_search(state, kEntryCol);
  }

  if (state.col > kEntryCol) {
    return move_towards_col_for_search(state, kEntryCol);
  }

  return safe_forward_or_pick_or_avoid_for_search(state);
}

Board::DecisionCmd Board::choose_exit_from_last_middle() const
{
  const int row = kRows - 1;
  const int col = kEntryCol;

  const DecisionCmd left_exit_cmd =
    exit_step_or_clear(row, col, row, 0, DecisionCmd::LEFT);
  if (left_exit_cmd != DecisionCmd::WAIT) {
    return left_exit_cmd;
  }

  const DecisionCmd right_exit_cmd =
    exit_step_or_clear(row, col, row, kCols - 1, DecisionCmd::RIGHT);
  if (right_exit_cmd != DecisionCmd::WAIT) {
    return right_exit_cmd;
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::safe_forward_or_clear_or_avoid_for_exit(
  int row,
  int col) const
{
  if (!is_valid_index(row, col)) {
    return DecisionCmd::WAIT;
  }

  // 只有 (3,0) 和 (3,2) 能直接 FORWARD 离开棋盘。
  if (row == kRows - 1) {
    if (is_exit_cell(row, col)) {
      return DecisionCmd::FORWARD;
    }

    if (col == kEntryCol) {
      return choose_exit_from_last_middle();
    }

    return DecisionCmd::WAIT;
  }

  const int next_row = row + 1;

  const DecisionCmd forward_cmd =
    exit_step_or_clear(row, col, next_row, col, DecisionCmd::FORWARD);
  if (forward_cmd != DecisionCmd::WAIT) {
    return forward_cmd;
  }

  // 前方是 F / UNKNOWN 或不可处理障碍时，优先绕到空格。
  if (col > 0 && is_empty_cell(row, col - 1)) {
    return DecisionCmd::LEFT;
  }

  if (col < kCols - 1 && is_empty_cell(row, col + 1)) {
    return DecisionCmd::RIGHT;
  }

  // 左右没有空格时，再清走左右相邻的 T/R1。
  if (col > 0 && should_clear_for_exit(row, col - 1)) {
    return DecisionCmd::CLEAR_LEFT;
  }

  if (col < kCols - 1 && should_clear_for_exit(row, col + 1)) {
    return DecisionCmd::CLEAR_RIGHT;
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::decide_move_to_exit(int row, int col) const
{
  if (!is_valid_index(row, col)) {
    return DecisionCmd::WAIT;
  }

  if (is_exit_cell(row, col)) {
    return DecisionCmd::FORWARD;
  }

  if (row < kRows - 1) {
    return safe_forward_or_clear_or_avoid_for_exit(row, col);
  }

  if (row == kRows - 1 && col == kEntryCol) {
    return choose_exit_from_last_middle();
  }

  return DecisionCmd::WAIT;
}

Board::DecisionCmd Board::decide_next_action(const RobotGridState & state) const
{
  // 机器人在棋盘外，正对入口 (0,1)。
  if (state.row == kOutsideEntryRow && state.col == kEntryCol) {
    const auto & entry = cells_[kEntryRow][kEntryCol];

    if (!entry.occupied) {
      return DecisionCmd::FORWARD;
    }

    if (should_pick_for_search(kEntryRow, kEntryCol, state)) {
      return DecisionCmd::PICK_FORWARD;
    }

    // 第一行不会记录 F/UNKNOWN；入口被 R1 占用时等待。
    return DecisionCmd::WAIT;
  }

  if (!is_valid_index(state.row, state.col)) {
    return DecisionCmd::WAIT;
  }

  // 如果当前已经携带 T，或者已经完成最多 2 个 T 的任务抓取，进入出口模式。
  if (state.carried_t > 0 || state.picked_t >= kMaxPickedT) {
    return decide_move_to_exit(state.row, state.col);
  }

  std::vector<int> t_cols;
  t_cols.reserve(kCols);

  for (int c = 0; c < kCols; ++c) {
    if (is_t_cell(state.row, c)) {
      t_cols.push_back(c);
    }
  }

  // 当前行没有 T：沿中间列继续扫；最后一行没有 T 时从出口离开。
  if (t_cols.empty()) {
    if (state.row == kRows - 1) {
      return decide_move_to_exit(state.row, state.col);
    }

    return move_forward_or_keep_middle_for_search(state);
  }

  // 当前行只有一个 T。
  if (t_cols.size() == 1) {
    const int t_col = t_cols.front();

    if (t_col == kEntryCol) {
      return move_towards_col_for_search(state, kEntryCol);
    }

    // 前三行边上的单个 T 不特意绕过去拿，继续沿中间列扫描。
    if (state.row < kRows - 1) {
      return move_forward_or_keep_middle_for_search(state);
    }

    // 最后一行边上的 T 可以直接从相邻位置抓，抓完后就近从出口离开。
    return move_towards_col_for_search(state, t_col);
  }

  // 当前行有多个 T，选择离自己最近的那个。
  int best_col = t_cols.front();
  int best_dist = std::abs(state.col - best_col);

  for (const int c : t_cols) {
    const int dist = std::abs(state.col - c);
    if (dist < best_dist) {
      best_dist = dist;
      best_col = c;
    }
  }

  return move_towards_col_for_search(state, best_col);
}

std::string Board::decision_to_string(DecisionCmd cmd)
{
  switch (cmd) {
    case DecisionCmd::LEFT:
      return "LEFT";
    case DecisionCmd::RIGHT:
      return "RIGHT";
    case DecisionCmd::FORWARD:
      return "FORWARD";
    case DecisionCmd::WAIT:
      return "WAIT";
    case DecisionCmd::PICK_LEFT:
      return "PICK_LEFT";
    case DecisionCmd::PICK_RIGHT:
      return "PICK_RIGHT";
    case DecisionCmd::PICK_FORWARD:
      return "PICK_FORWARD";
    case DecisionCmd::CLEAR_LEFT:
      return "CLEAR_LEFT";
    case DecisionCmd::CLEAR_RIGHT:
      return "CLEAR_RIGHT";
    case DecisionCmd::CLEAR_FORWARD:
      return "CLEAR_FORWARD";
    default:
      return "UNKNOWN";
  }
}

cv::Scalar Board::cell_color(const Cell & cell) const
{
  if (!cell.occupied) {
    return cv::Scalar(0, 0, 0);
  }

  switch (cell.workname) {
    case detector::WORK_R_R1:
    case detector::WORK_B_R1:
      return cv::Scalar(0, 255, 255);

    case detector::WORK_T:
      return cv::Scalar(0, 255, 0);

    case detector::WORK_F:
      return cv::Scalar(0, 0, 255);

    default:
      return cv::Scalar(60, 60, 60);
  }
}

std::string Board::cell_text(const Cell & cell) const
{
  if (!cell.enabled) {
    return "NO PUT";
  }

  if (!cell.occupied) {
    return "EMPTY";
  }

  switch (cell.workname) {
    case detector::WORK_R_R1:
      return "WORK_R_R1";
    case detector::WORK_B_R1:
      return "WORK_B_R1";
    case detector::WORK_T:
      return "WORK_T";
    case detector::WORK_F:
      return "WORK_F";
    default:
      return "UNKNOWN";
  }
}

cv::Mat Board::render(int cell_px, int margin_px) const
{
  const int width = kCols * cell_px + 2 * margin_px;
  const int height = kRows * cell_px + 2 * margin_px;

  cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(30, 30, 30));

  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c) {
      const Cell & cell = cells_[r][c];

      const int x1 = margin_px + c * cell_px;
      const int y1 = margin_px + r * cell_px;
      const int y2 = y1 + cell_px;

      const cv::Rect rect(x1, y1, cell_px, cell_px);

      cv::rectangle(canvas, rect, cell_color(cell), cv::FILLED);
      cv::rectangle(canvas, rect, cv::Scalar(255, 255, 255), 2);

      const std::string rc_text =
        "R" + std::to_string(r) + "C" + std::to_string(c);

      cv::putText(
        canvas,
        rc_text,
        cv::Point(x1 + 8, y1 + 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.55,
        cv::Scalar(255, 255, 255),
        1);

      const std::string text = cell_text(cell);
      cv::putText(
        canvas,
        text,
        cv::Point(x1 + 8, y2 - 18),
        cv::FONT_HERSHEY_SIMPLEX,
        0.5,
        cv::Scalar(255, 255, 255),
        1);
    }
  }

  return canvas;
}

void Board::show(
  const std::string & window_name,
  int cell_px,
  int margin_px) const
{
  const cv::Mat img = render(cell_px, margin_px);
  cv::imshow(window_name, img);
  cv::waitKey(1);
}

}  // namespace planner