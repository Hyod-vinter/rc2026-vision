#include "task/planner/meilin.hpp"

#include <iostream>

using planner::Meilin;

int main()
{
  const int case_id = 2000;

  const auto & invade_path = Meilin::invade(case_id);
  const auto & defence_path = Meilin::defence(case_id);
  int path_array[planner::Meilin::kMaxPathLen];

    for (int i = 0; i < planner::Meilin::kMaxPathLen; ++i) {
    path_array[i] = invade_path.cells[i];
    }

  const auto commands = Meilin::path_to_commands(Meilin::Color::red, defence_path);

  for (const auto & cmd : commands) {
    if (cmd.cell == 0) {
      continue;
    }
    std::cout << "(" << cmd.cell << ","
              << cmd.horizontal << ","
              << cmd.vertical << ")\n";
  }

  

  
  std::cout << "case_id: " << case_id << "\n";
  std::cout << "board:\n" << Meilin::board_to_string(case_id) << "\n\n";

  std::cout << "invade: pick " << invade_path.task_t_count
            << " T, moves=" << invade_path.move_count
            << ", path=" << Meilin::path_to_string(invade_path) << "\n";

  std::cout << "defence: pick " << defence_path.task_t_count
            << " T, moves=" << defence_path.move_count
            << ", path=" << Meilin::path_to_string(defence_path) << "\n";

  // Meilin::save_case_with_mode(
  //   case_id,
  //   Meilin::Mode::invade,
  //   "case2171_invade.png");

  // Meilin::save_case_with_mode(
  //   case_id,
  //   Meilin::Mode::defence,
  //   "case2171_defence.png");

  // 如果需要窗口显示，取消下面两行注释：
   //Meilin::show_case_with_mode(case_id, Meilin::Mode::invade, "invade", 120, 30, 0);
   Meilin::show_case_with_mode(case_id, Meilin::Mode::defence, "defence", 100, 0, 0);

  return 0;
}