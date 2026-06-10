#ifndef MEILIN_HPP_
#define MEILIN_HPP_

#include <array>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

namespace planner
{

class Meilin
{
public:
  static constexpr int kRows = 4;
  static constexpr int kCols = 3;
  static constexpr int kCells = 12;
  static constexpr int kCaseCount = 2970;
  static constexpr int kMaxPathLen = 9;

  static constexpr int kEmpty = 1;
  static constexpr int kT = 2;
  static constexpr int kF = 3;

  static constexpr int kStartCell = 2;
  static constexpr int kLeftExitCell = 10;
  static constexpr int kRightExitCell = 12;

  enum class Mode
  {
    invade,
    defence
  };

  enum class Color
  {
    red,
    blue
  };

  enum Direction
  {
    kNoDirection = 0,
    kLeft = 1,
    kRight = 2,
    kFront = 3,
    kBack = 4,
    kUp = 5,
    kDown = 6
  };

  struct CaseRow
  {
    std::array<int, kCells> board{};
    int f_cell = -1;
    std::array<int, 4> t_cells{};
  };

  struct PathRow
  {
    bool reachable = false;
    int task_t_count = 0;
    int move_count = -1;
    int path_len = 0;
    std::array<int, kMaxPathLen> cells{};
  };

  struct MoveCommand
  {
    int cell = 0;
    int horizontal = kNoDirection;
    int vertical = kNoDirection;
  };

  static const std::array<CaseRow, kCaseCount> & cases();
  static const std::array<PathRow, kCaseCount> & pick1_table();
  static const std::array<PathRow, kCaseCount> & pick2_table();
  static const std::array<PathRow, kCaseCount> & pick3_table();

  static const CaseRow & case_info(int case_id);
  static const PathRow & exact_path(int case_id, int task_t_count);

  static const PathRow & invade(int case_id);
  static const PathRow & defence(int case_id);
  static const PathRow & choose(int case_id, Mode mode);

  static std::vector<int> path_cells(const PathRow & path);
  static std::array<MoveCommand, kMaxPathLen> path_to_commands(
    Color color,
    const std::array<int, kMaxPathLen> & cells);
  static std::array<MoveCommand, kMaxPathLen> path_to_commands(
    Color color,
    const PathRow & path);
  static std::string path_to_string(const PathRow & path);

  static std::array<std::array<int, kCols>, kRows> board_4x3(int case_id);
  static std::string board_to_string(int case_id);

  static cv::Mat render_case_with_path(
    int case_id,
    const PathRow & path,
    int cell_px = 120,
    int margin_px = 30);

  static cv::Mat render_case_with_mode(
    int case_id,
    Mode mode,
    int cell_px = 120,
    int margin_px = 30);

  static void show_case_with_path(
    int case_id,
    const PathRow & path,
    const std::string & window_name = "meilin",
    int cell_px = 120,
    int margin_px = 30,
    int wait_ms = 1);

  static void show_case_with_mode(
    int case_id,
    Mode mode,
    const std::string & window_name = "meilin",
    int cell_px = 120,
    int margin_px = 30,
    int wait_ms = 1);

  static bool save_case_with_path(
    int case_id,
    const PathRow & path,
    const std::string & filename,
    int cell_px = 120,
    int margin_px = 30);

  static bool save_case_with_mode(
    int case_id,
    Mode mode,
    const std::string & filename,
    int cell_px = 120,
    int margin_px = 30);

private:
  static int index_from_case_id(int case_id);
  static cv::Point cell_center(int cell_id, int cell_px, int margin_px);
};

}  // namespace planner

#endif  // MEILIN_HPP_